"""
memory_gardener.py — out-of-session corpus-refinement agent.

Architecture (2026-04-24, third rewrite):
  Uses the raw `anthropic` Python SDK with a hand-rolled tool-use loop.
  NOT claude-agent-sdk. NOT claude CLI subprocess. NOT MCP stdio.

  Why: all those layers had Windows show-stoppers this session —
    * claude-agent-sdk Python 0.1.65: STATUS_ACCESS_VIOLATION on any
      MCP tool call.
    * claude-agent-sdk TS 0.1.77: "tool_use ids must be unique" 400
      on second assistant turn (CLI bug #20631).
    * claude-agent-sdk TS 0.2.119: same STATUS_ACCESS_VIOLATION as
      Python, even at maxTurns=2.
    * `claude --print` subprocess: works but cannot guarantee hooks
      are disabled on Windows.

  The raw anthropic SDK is pure HTTPS — no subprocess, no stdio, no
  hooks, no plugin loading, no Windows-specific transport quirks.
  We call mempalace tool functions DIRECTLY via mcp_server.tool_*
  (no MCP wire protocol in between). Multi-turn tool use works
  naturally because we own the message history.

Flow (process_batch → one SDK session per flag):
  1. list_pending_flags(1) — one flag per batch keeps resolution
     mapping trivial.
  2. start_gardener_run — audit row opens.
  3. Build system + user prompt.
  4. anthropic.messages.create(tools=[...]) loop:
        while response.stop_reason == "tool_use":
            for each tool_use block: call the matching mempalace
            tool_* function; collect tool_result.
            messages.append(assistant_content); messages.append(tool_results)
            response = messages.create(messages=messages, ...)
        break when stop_reason is "end_turn" or "max_tokens".
  5. Derive resolution from the tool calls the model actually made
     (mapping tool_name + flag.kind → merged/invalidated/...).
  6. mark_flag_resolved or bump_flag_attempt.
  7. finish_gardener_run — audit row closes with counters + errors.

Kill switch: ENABLED BY DEFAULT. Set MEMPALACE_GARDENER_DISABLED=1 to opt out.
Auth: ANTHROPIC_API_KEY from palace .env (via injection_gate loader).
Session gate: MEMPALACE_GARDENER_ACTIVE=1 is set so mempalace's
_require_sid / _require_agent bypass for this process.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 1
_FINALIZE_TRIGGER_THRESHOLD = 5
_DEFAULT_GARDENER_MODEL = "claude-haiku-4-5"
_MAX_TOOL_LOOP_ITERS = 20  # caps runaway loops (edge_candidate/unlinked_entity
# often need declare-then-add, which can take 5-8
# turns when both sides of an edge need declaration)


# ═══════════════════════════════════════════════════════════════════
# Kill switch (opt-out)
#
# The gardener is ENABLED BY DEFAULT. To disable it temporarily
# (e.g. while debugging, during a sensitive session, or to pause
# background API spend), set MEMPALACE_GARDENER_DISABLED=1 in the
# env where the MCP server runs.
#
# Safety: fork-bomb cascade is structurally impossible — the
# gardener's 6-tool surface does not include finalize_intent,
# declare_intent, or wake_up, so a spawned gardener cannot trigger
# another gardener spawn.
# ═══════════════════════════════════════════════════════════════════

_DISABLED_MSG = (
    "[memory_gardener] DISABLED by MEMPALACE_GARDENER_DISABLED=1. "
    "Unset that env var (or set it to 0) to re-enable."
)


def _gardener_disabled() -> bool:
    # Truthy values: "1", "true", "yes", "on" (case-insensitive).
    val = os.environ.get("MEMPALACE_GARDENER_DISABLED", "").strip().lower()
    return val in ("1", "true", "yes", "on")


# ═══════════════════════════════════════════════════════════════════
# Tool schemas (anthropic tool-use format) + executor
# ═══════════════════════════════════════════════════════════════════

_TOOL_SCHEMAS: list[dict] = [
    {
        "name": "mempalace_kg_query",
        "description": (
            "Read an entity from the knowledge graph. Returns facts "
            "(triples) involving the entity and any stored description."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {"type": "string", "description": "Entity id."},
                "direction": {
                    "type": "string",
                    "enum": ["outgoing", "incoming", "both"],
                    "description": "Edge direction filter. Default both.",
                },
            },
            "required": ["entity"],
        },
    },
    {
        "name": "mempalace_kg_merge_entities",
        "description": (
            "Merge two entities. `source` is folded into `target` "
            "(source becomes an alias of target; all source edges move "
            "to target). Use for duplicate_pair flags."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Entity id to drop (becomes alias)."},
                "target": {"type": "string", "description": "Entity id to keep (canonical)."},
                "agent": {"type": "string", "description": "Always 'memory_gardener'."},
                "update_description": {
                    "type": "string",
                    "description": "Optional merged description for the target.",
                },
            },
            "required": ["source", "target", "agent"],
        },
    },
    {
        "name": "mempalace_kg_invalidate",
        "description": (
            "Invalidate a single triple (edge). Use for "
            "contradiction_pair / stale flags when a specific edge is "
            "no longer true. Prefer kg_delete_entity when retiring a "
            "whole memory/entity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "predicate": {"type": "string"},
                "object": {"type": "string"},
                "agent": {"type": "string", "description": "Always 'memory_gardener'."},
            },
            "required": ["subject", "predicate", "object", "agent"],
        },
    },
    {
        "name": "mempalace_propose_edge_candidate",
        "description": (
            "Route an edge suggestion to the link-author pipeline. "
            "Use this for edge_candidate AND unlinked_entity flags. "
            "DO NOT call kg_add directly — the gardener is not "
            "authorised to add edges or declare new predicates. "
            "Instead, pass the two entity ids to this tool; the "
            "link-author process (Opus-designed jury + Haiku jurors) "
            "will later decide whether to author the edge AND pick "
            "the right predicate — possibly an existing one, possibly "
            "a new one it declares through its own flow. That's the "
            "single graph-mutation gatekeeper; the gardener's job is "
            "only to seed the queue."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "from_entity": {
                    "type": "string",
                    "description": "One side of the candidate edge (canonical entity id).",
                },
                "to_entity": {
                    "type": "string",
                    "description": "Other side of the candidate edge (canonical entity id).",
                },
                "weight": {
                    "type": "number",
                    "description": (
                        "Evidence weight 0.1–1.0. Default 0.7 for "
                        "gate-flagged pairs (higher than a passive co-"
                        "occurrence observation, lower than an explicit "
                        "manual proposal)."
                    ),
                },
            },
            "required": ["from_entity", "to_entity"],
        },
    },
    {
        "name": "mempalace_kg_update_entity",
        "description": (
            "Update an entity's description / summary / importance. "
            "Use for generic_summary flags — pass a NEW summary under "
            "the 'description' field, <=280 chars."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {"type": "string"},
                "description": {
                    "type": "string",
                    "description": "New summary / description. Keep <=280 chars for summary-refinement.",
                },
                "importance": {"type": "integer", "description": "1-5."},
                "agent": {"type": "string", "description": "Always 'memory_gardener'."},
            },
            "required": ["entity", "agent"],
        },
    },
    {
        "name": "mempalace_kg_delete_entity",
        "description": (
            "Delete an entity (soft-delete). Use for orphan flags "
            "and as the normal path for retiring a whole memory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string"},
                "agent": {"type": "string", "description": "Always 'memory_gardener'."},
            },
            "required": ["entity_id", "agent"],
        },
    },
]

# Map tool name → mempalace Python function. mempalace's tool_*
# functions take the tool's arguments as kwargs and return a dict
# (usually {"success": True, ...} or {"success": False, "error": ...}).
_TOOL_DISPATCH: dict[str, Callable[..., Any]] = {}

# Set by process_batch before invoking the loop. Used by the link-
# author-route tool to stamp a unique (and debuggable) context_id on
# the candidate row it seeds.
_CURRENT_FLAG_ID: int | None = None


def _propose_edge_candidate_shim(
    from_entity: str,
    to_entity: str,
    weight: float = 0.7,
    **_ignored,
) -> dict:
    """Seed link_prediction_candidates with this pair. The link-author
    process will later run its jury, decide whether to author the edge,
    and (if so) pick the predicate. This is the gardener's ONLY edge-
    touching path — it does not call kg_add or declare predicates."""
    from . import link_author as _la
    from . import mcp_server as _mcp

    kg = _mcp._STATE.kg
    if kg is None:
        return {"success": False, "error": "KG not initialised"}

    # Context id carries the flag id for traceability and makes the
    # dedup table's unique constraint work per-flag (a gardener run
    # seeding the same pair twice within one flag → upsert counts as
    # one contribution).
    fid = _CURRENT_FLAG_ID
    context_id = f"gardener_flag_{fid}" if fid is not None else "gardener_manual"

    try:
        inserted = _la.upsert_candidate(
            kg,
            from_entity=from_entity,
            to_entity=to_entity,
            weight=float(weight),
            context_id=context_id,
        )
    except Exception as e:
        return {"success": False, "error": f"link-author seed failed: {e}"}

    if not inserted:
        # The pair was either already directly connected (no
        # candidate needed) or already seeded from this context.
        return {
            "success": True,
            "inserted": False,
            "note": "pair already directly connected OR already seeded from this context",
            "from_entity": from_entity,
            "to_entity": to_entity,
        }
    return {
        "success": True,
        "inserted": True,
        "from_entity": from_entity,
        "to_entity": to_entity,
        "weight": float(weight),
        "context_id": context_id,
        "note": "seeded link_prediction_candidates; link-author jury will decide later",
    }


def _init_tool_dispatch() -> None:
    """Lazy wire-up so import-time side-effects stay small."""
    if _TOOL_DISPATCH:
        return
    from . import mcp_server as _mcp

    _TOOL_DISPATCH["mempalace_kg_query"] = _mcp.tool_kg_query
    _TOOL_DISPATCH["mempalace_kg_merge_entities"] = _mcp.tool_kg_merge_entities
    _TOOL_DISPATCH["mempalace_kg_invalidate"] = _mcp.tool_kg_invalidate
    _TOOL_DISPATCH["mempalace_kg_update_entity"] = _mcp.tool_kg_update_entity
    _TOOL_DISPATCH["mempalace_kg_delete_entity"] = _mcp.tool_kg_delete_entity
    _TOOL_DISPATCH["mempalace_propose_edge_candidate"] = _propose_edge_candidate_shim


def _exec_tool(name: str, arguments: dict) -> dict:
    """Execute one tool call. Never raises — always returns a dict
    that's safe to stringify back to the model."""
    _init_tool_dispatch()
    handler = _TOOL_DISPATCH.get(name)
    if handler is None:
        return {"success": False, "error": f"unknown tool: {name}"}
    try:
        # Drop args the handler doesn't accept, UNLESS the handler has
        # **kwargs (our link-author shim does; filtering would strip
        # extras the caller tried to pass). Pure-keyword handlers get
        # filtering for tolerance to hallucinated extras.
        import inspect

        sig = inspect.signature(handler)
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if has_var_kw:
            clean = dict(arguments)
        else:
            accepted = set(sig.parameters.keys())
            clean = {k: v for k, v in arguments.items() if k in accepted}
        result = handler(**clean)
        if not isinstance(result, dict):
            return {"success": True, "result": result}
        return result
    except TypeError as e:
        return {"success": False, "error": f"TypeError calling {name}: {e}"}
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


# ═══════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
You are memory_gardener — a corpus-refinement agent that resolves ONE mempalace flag per invocation. You end by making exactly one mutation tool call. You do NOT declare new entities or predicates; you do NOT author edges directly; you do NOT follow any mempalace protocol (no wake_up/declare_intent/finalize_intent/diary — you don't have those tools anyway). You do NOT emit a wrap-up text message — after the mutation succeeds, just stop.

Pass agent="memory_gardener" on every mutation that accepts an agent parameter.

THE 6 TOOLS YOU HAVE — nothing else exists:
  mempalace_kg_query                    read an entity's edges + description (optional investigation)
  mempalace_kg_merge_entities           merge two entities (source folded into target)
  mempalace_kg_invalidate               invalidate one specific triple (subject, predicate, object)
  mempalace_kg_update_entity            update an entity's description/importance
  mempalace_kg_delete_entity            soft-delete an entity (invalidates all its edges)
  mempalace_propose_edge_candidate      seed a pair into the link-author queue (for ANY edge-type flag)

FLAG-KIND → ACTION (exactly one tool call per flag):

  duplicate_pair
    → mempalace_kg_merge_entities(source=<later/weaker id>, target=<canonical id>, agent="memory_gardener")
    You MAY first call kg_query to pick which is canonical (more edges = more canonical).

  contradiction_pair
    → mempalace_kg_invalidate(subject, predicate, object, agent="memory_gardener") on the stale edge.
    Call kg_query first on the subject/object to find which triple is wrong.

  stale
    → If a SPECIFIC stale edge is identified in the flag detail:
        mempalace_kg_invalidate(subject, predicate, object, agent="memory_gardener")
    → If the WHOLE memory is stale:
        mempalace_kg_delete_entity(entity_id=<memory_ids[0]>, agent="memory_gardener")

  orphan
    → mempalace_kg_delete_entity(entity_id=<memory_ids[0]>, agent="memory_gardener")
    No kg_query needed — orphan means no edges to preserve.

  generic_summary
    → mempalace_kg_query first to see current description and content.
    → Then mempalace_kg_update_entity(entity=<memory_ids[0]>, description=<new ≤280-char summary that adds SPECIFIC WHAT/WHY the current generic one lacks>, agent="memory_gardener").

  edge_candidate
    → mempalace_propose_edge_candidate(from_entity=<memory_ids[0]>, to_entity=<memory_ids[1]>, weight=0.7)
    Do NOT call kg_add. Do NOT declare any predicate. The link-author jury will pick the predicate and author the edge. Your job is just to seed the pair.

  unlinked_entity
    → mempalace_propose_edge_candidate(from_entity=<memory that mentions the entity>, to_entity=<entity name from the flag detail>, weight=0.7)
    Same rule: the link-author pipeline (not you) is the single graph-mutation gatekeeper for edges and predicates. If the target entity doesn't exist yet, the link-author still records the candidate; it won't be authored until both sides exist. Do NOT call kg_declare_entity.

WHAT YOU MUST NEVER DO:
  - Never call any tool outside the 6 listed above. They literally don't exist in your toolset.
  - Never try to declare a new predicate. That's the link-author's job, not the gardener's.
  - Never call kg_add or anything that would author an edge directly. Always route edges through propose_edge_candidate.
  - Never emit a final text summary. After the mutation tool_result comes back, stop.

ERROR RECOVERY:
  - If kg_merge_entities / kg_invalidate / kg_delete_entity / kg_update_entity returns {"success": false, "error": "Not found..."} or similar precondition mismatch: stop. The flag gets deferred automatically.
  - If propose_edge_candidate returns {"success": true, "inserted": false}: that's still a clean resolution. It means the pair was already directly connected or already seeded — nothing more to do.
  - If any tool returns a generic error you can fix (e.g., a typo in an id), retry ONCE. Beyond that, stop.
"""


def _build_user_prompt(flag: dict) -> str:
    # memory_ids may already be parsed (list) by kg.list_pending_flags
    # or still be a JSON string (direct DB row). Handle both.
    raw = flag.get("memory_ids")
    mids: list = []
    if isinstance(raw, list):
        mids = raw
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw or "[]")
            if isinstance(parsed, list):
                mids = parsed
        except Exception:
            mids = []
    lines = [
        f"FLAG #{flag.get('id')}",
        f"  kind: {flag.get('kind')}",
        f"  memory_ids: {json.dumps(mids)}",
        f"  detail: {flag.get('detail') or '(no detail)'}",
        f"  context_id: {flag.get('context_id') or '(none)'}",
    ]
    if flag.get("attempted_count"):
        lines.append(f"  prior_attempts: {flag['attempted_count']}")
    lines.append("")
    lines.append("Resolve this flag now per the system-prompt mapping.")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Anthropic tool-use loop
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ToolCallTrace:
    name: str
    arguments: dict
    tool_use_id: str = ""
    result: dict = field(default_factory=dict)
    is_error: bool = False


@dataclass
class LoopResult:
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    final_text: str = ""
    stop_reason: str = ""
    iterations: int = 0
    api_error: str = ""


def _run_anthropic_loop(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_iters: int = _MAX_TOOL_LOOP_ITERS,
) -> LoopResult:
    """Classic Anthropic tool-use loop. No SDK wrapping, no MCP wire
    protocol — tool calls dispatch to mempalace tool_* functions via
    _exec_tool directly."""
    result = LoopResult()
    try:
        import anthropic
    except ImportError as e:
        result.api_error = f"anthropic SDK not installed: {e}"
        return result

    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY from env
    messages: list[dict] = [{"role": "user", "content": user_prompt}]

    for iteration in range(max_iters):
        result.iterations = iteration + 1
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                tools=_TOOL_SCHEMAS,
                messages=messages,
            )
        except anthropic.APIError as e:
            result.api_error = f"{type(e).__name__}: {e}"
            break
        except Exception as e:
            result.api_error = f"unexpected {type(e).__name__}: {e}"
            break

        result.stop_reason = resp.stop_reason or ""
        # Collect assistant content + tool_use blocks.
        assistant_blocks: list[dict] = []
        pending_tool_uses: list[tuple[str, str, dict]] = []  # (id, name, input)
        for block in resp.content:
            # anthropic SDK types — block is either TextBlock or ToolUseBlock.
            if block.type == "text":
                result.final_text += block.text
                assistant_blocks.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                pending_tool_uses.append((block.id, block.name, dict(block.input)))
                assistant_blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": dict(block.input),
                    }
                )
        messages.append({"role": "assistant", "content": assistant_blocks})

        if resp.stop_reason != "tool_use":
            break

        # Execute each tool_use and build a tool_result user message.
        tool_results_content: list[dict] = []
        for tu_id, tu_name, tu_input in pending_tool_uses:
            exec_result = _exec_tool(tu_name, tu_input)
            is_error = bool(exec_result.get("success") is False)
            trace = ToolCallTrace(
                name=tu_name,
                arguments=tu_input,
                tool_use_id=tu_id,
                result=exec_result,
                is_error=is_error,
            )
            result.tool_calls.append(trace)
            tool_results_content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu_id,
                    "content": json.dumps(exec_result)[:4000],  # cap to avoid runaway
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": tool_results_content})

    return result


# ═══════════════════════════════════════════════════════════════════
# Resolution derivation
# ═══════════════════════════════════════════════════════════════════

_VALID_RESOLUTIONS = (
    "merged",
    "invalidated",
    "linked",
    "edge_proposed",
    "summary_rewritten",
    "pruned",
    "deferred",
    "no_action",
)

_RESOLUTION_TO_COUNTER = {
    "merged": "merges",
    "invalidated": "invalidations",
    "linked": "links_created",
    "edge_proposed": "edges_proposed",
    "summary_rewritten": "summary_rewrites",
    "pruned": "prunes",
    "deferred": "deferrals",
    "no_action": "no_action",
}

_MUTATION_TOOL_NAMES = {
    "mempalace_kg_merge_entities",
    "mempalace_kg_invalidate",
    "mempalace_kg_update_entity",
    "mempalace_kg_delete_entity",
    "mempalace_propose_edge_candidate",
}


def _derive_resolution(flag_kind: str, tool_calls: list[ToolCallTrace]) -> tuple[str, str]:
    """Return (resolution, note) for the audit row."""
    mutations = [tc for tc in tool_calls if tc.name in _MUTATION_TOOL_NAMES]
    if not mutations:
        n_reads = len(tool_calls)
        return ("deferred", f"model made no mutation ({n_reads} read call(s))")
    # Any errored mutation → deferred with the error preview.
    errored = [tc for tc in mutations if tc.is_error]
    if errored:
        last_err = errored[-1]
        err = last_err.result.get("error") or "unknown error"
        issues = last_err.result.get("issues")
        issues_str = ""
        if isinstance(issues, list) and issues:
            issues_str = f" | issues: {'; '.join(str(i) for i in issues)[:400]}"
        return ("deferred", f"mutation failed: {str(err)[:200]}{issues_str}")
    last = mutations[-1]
    args_short = json.dumps(last.arguments)[:160]
    if last.name == "mempalace_kg_merge_entities":
        return ("merged", f"merged via {args_short}")
    if last.name == "mempalace_kg_invalidate":
        return (
            ("pruned" if flag_kind == "orphan" else "invalidated"),
            f"invalidated via {args_short}",
        )
    if last.name == "mempalace_kg_delete_entity":
        return ("pruned", f"deleted via {args_short}")
    if last.name == "mempalace_kg_update_entity":
        return ("summary_rewritten", f"updated via {args_short}")
    if last.name == "mempalace_propose_edge_candidate":
        inserted = last.result.get("inserted")
        note = f"seeded link-author queue via {args_short}"
        if inserted is False:
            note += " (already connected / already seeded)"
        # unlinked_entity → 'linked' bucket; edge_candidate → 'edge_proposed'.
        return (
            ("linked" if flag_kind == "unlinked_entity" else "edge_proposed"),
            note,
        )
    return ("deferred", f"unrecognised mutation: {last.name}")


# ═══════════════════════════════════════════════════════════════════
# Env setup helpers
# ═══════════════════════════════════════════════════════════════════


def _prepare_env() -> str:
    """Load palace .env (ANTHROPIC_API_KEY), set gardener mode flag.
    Returns an error string if prerequisites missing, else ''."""
    try:
        from . import injection_gate as _ig

        _ig._ensure_palace_env_loaded()
    except Exception as e:
        log.info("palace .env loader: %s", e)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return (
            "ANTHROPIC_API_KEY not set. Required by the anthropic SDK. "
            "Put it in the palace .env (same place the gate reads it)."
        )
    # Mark this process as gardener so mempalace's _require_sid /
    # _require_agent bypass the hook-injected sid requirement.
    os.environ["MEMPALACE_GARDENER_ACTIVE"] = "1"
    return ""


# ═══════════════════════════════════════════════════════════════════
# Batch processor
# ═══════════════════════════════════════════════════════════════════


def process_batch(
    kg,
    *,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    model: str = _DEFAULT_GARDENER_MODEL,
    target_flag_id: int | None = None,
) -> dict:
    """Process one flag via the anthropic tool-use loop.

    batch_size is clamped to 1 — keeps resolution mapping trivial
    (one flag per loop → tool calls unambiguously belong to that flag).
    Returns a dict shaped like the old API: run_id, flag_ids, results,
    counters, exit_code, errors.
    """
    if _gardener_disabled():
        return {
            "run_id": None,
            "flag_ids": [],
            "results": [],
            "counters": {},
            "exit_code": 2,
            "errors": _DISABLED_MSG,
            "note": "disabled",
        }

    env_err = _prepare_env()
    if env_err:
        return {
            "run_id": None,
            "flag_ids": [],
            "results": [],
            "counters": {},
            "exit_code": 126,
            "errors": env_err,
        }

    # Fetch one flag.
    if target_flag_id is not None:
        flags = [f for f in kg.list_pending_flags(limit=50) if f.get("id") == target_flag_id]
        if not flags:
            # Not in pending — maybe attempted_count already >= 3 or
            # flag resolved. Fetch directly.
            conn = kg._conn()
            row = conn.execute(
                "SELECT id, kind, memory_ids, detail, context_id, attempted_count "
                "FROM memory_flags WHERE id=?",
                (target_flag_id,),
            ).fetchone()
            if row:
                flags = [dict(row)]
    else:
        flags = kg.list_pending_flags(limit=1)

    if not flags:
        return {
            "run_id": None,
            "flag_ids": [],
            "results": [],
            "counters": {},
            "exit_code": 0,
            "errors": "",
            "note": "no pending flags",
        }

    flag = flags[0]
    flag_id = int(flag["id"])
    flag_kind = flag["kind"]
    run_id = kg.start_gardener_run(gardener_model=model)

    # Module-global stash for the link-author shim to stamp
    # context_id=gardener_flag_<id> on any candidate it seeds.
    global _CURRENT_FLAG_ID
    _CURRENT_FLAG_ID = flag_id

    try:
        user_prompt = _build_user_prompt(flag)
        loop = _run_anthropic_loop(_SYSTEM_PROMPT, user_prompt, model)
    finally:
        _CURRENT_FLAG_ID = None

    counters: dict[str, int] = {}
    results: list[dict] = []
    errors = loop.api_error
    exit_code = 1 if loop.api_error else 0

    if loop.api_error:
        kg.bump_flag_attempt(flag_id)
    else:
        resolution, note = _derive_resolution(flag_kind, loop.tool_calls)
        if resolution == "deferred" and not loop.tool_calls:
            kg.bump_flag_attempt(flag_id)
        else:
            try:
                kg.mark_flag_resolved(flag_id, resolution, note=note)
                results.append({"flag_id": flag_id, "resolution": resolution, "note": note})
                col = _RESOLUTION_TO_COUNTER.get(resolution)
                if col:
                    counters[col] = counters.get(col, 0) + 1
            except Exception as e:
                errors = f"mark_flag_resolved failed: {e}"
                exit_code = 1
                kg.bump_flag_attempt(flag_id)

    kg.finish_gardener_run(
        run_id,
        flag_ids=[flag_id],
        counters=counters,
        subprocess_exit_code=exit_code,
        errors=errors,
    )

    return {
        "run_id": run_id,
        "flag_ids": [flag_id],
        "results": results,
        "counters": counters,
        "exit_code": exit_code,
        "errors": errors,
        "loop_iterations": loop.iterations,
        "loop_stop_reason": loop.stop_reason,
        "tool_calls": [
            {"name": tc.name, "args": tc.arguments, "is_error": tc.is_error}
            for tc in loop.tool_calls
        ],
    }


# ═══════════════════════════════════════════════════════════════════
# Finalize-trigger + CLI
# ═══════════════════════════════════════════════════════════════════


def maybe_trigger_from_finalize(kg) -> bool:
    """Called at the tail of finalize_intent. Spawns a detached
    `python -m mempalace gardener process` subprocess if pending flags
    have accumulated past the trigger threshold. Enabled by default;
    set MEMPALACE_GARDENER_DISABLED=1 in the MCP server's env to
    suppress auto-triggering (e.g. for debugging or to pause API
    spend)."""
    if _gardener_disabled():
        return False
    try:
        pending = kg.count_pending_flags()
    except Exception:
        return False
    if pending < _FINALIZE_TRIGGER_THRESHOLD:
        return False
    try:
        env = os.environ.copy()
        env["MEMPALACE_GARDENER_ACTIVE"] = "1"
        if os.name == "nt":
            DETACHED_PROCESS = 0x00000008
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            subprocess.Popen(
                [sys.executable, "-m", "mempalace", "gardener", "process"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                close_fds=True,
                env=env,
            )
        else:
            subprocess.Popen(
                [sys.executable, "-m", "mempalace", "gardener", "process"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
                env=env,
            )
        return True
    except Exception as exc:
        log.info("memory_gardener spawn failed: %s", exc)
        return False


def cli_process(args) -> int:
    """Handler for `python -m mempalace gardener process`."""
    if _gardener_disabled():
        print(_DISABLED_MSG, file=sys.stderr)
        return 2

    from . import mcp_server as _mcp

    kg = _mcp._STATE.kg
    if kg is None:
        print("[memory_gardener] KnowledgeGraph not initialised", file=sys.stderr)
        return 2

    max_batches = int(getattr(args, "max_batches", 1) or 1)
    model = getattr(args, "model", None) or _DEFAULT_GARDENER_MODEL
    flag_id = getattr(args, "flag_id", None)

    any_failed = False
    for i in range(max_batches):
        result = process_batch(kg, model=model, target_flag_id=flag_id)
        if not result.get("flag_ids"):
            print(f"[memory_gardener] no pending flags; stopping at batch {i + 1}")
            break
        if result.get("exit_code", 0) != 0:
            any_failed = True
        print(
            f"[memory_gardener] batch {i + 1}: "
            f"run_id={result['run_id']} flags={result['flag_ids']} "
            f"counters={result['counters']} exit={result['exit_code']} "
            f"iters={result.get('loop_iterations')} "
            f"stop={result.get('loop_stop_reason')}"
        )
        if result.get("errors"):
            print(f"[memory_gardener]   errors: {result['errors'][:200]}", file=sys.stderr)
        # --flag-id is single-shot
        if flag_id is not None:
            break
    return 3 if any_failed else 0
