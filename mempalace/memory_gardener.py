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
# Number of sequential single-flag batches the auto-triggered subprocess
# drains before exiting. Each batch is its own fresh Haiku tool-use loop
# (no context pollution between flags); raising this is the safe way to
# lift gardener throughput, as opposed to raising _DEFAULT_BATCH_SIZE
# which would cram multiple flags into one Haiku context.
_AUTO_TRIGGER_MAX_BATCHES = 10
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
            "(triples) involving the entity and any stored description. "
            "Use kg_search instead when you do NOT have an exact entity id."
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
        "name": "mempalace_kg_search",
        "description": (
            "Fuzzy search across memories (prose) AND entities (KG nodes) "
            "in one call. Use when (a) kg_query returned 'Not found' and "
            "you need to recover the canonical id, (b) you need grounding "
            "evidence before rewriting a generic_summary, or (c) you need "
            "to verify a target entity exists before proposing an edge. "
            "Each query is an independent perspective; keywords are exact "
            "domain terms; both 2-5 entries. The response contains a "
            "results list with id+text+source for each hit."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "context": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 5,
                            "description": "2-5 natural-language perspectives.",
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 5,
                            "description": "2-5 exact domain terms.",
                        },
                        "entities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional graph BFS seeds.",
                        },
                    },
                    "required": ["queries", "keywords"],
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 10).",
                },
            },
            "required": ["context"],
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
    {
        "name": "mempalace_synthesize_operation_template",
        "description": (
            "S3b: resolve an op_cluster_templatizable flag by minting a "
            "reusable template record that distils the cluster's pattern "
            "and writing `templatizes` edges from the new record back to "
            "every source operation it covers. `op_ids` are the entity "
            "ids from the flag's memory_ids array — copy them verbatim. "
            "Compose `title` as a short noun phrase (<=80 chars) naming "
            "the pattern; `when_to_use` as 1-2 sentences saying which "
            "context triggers it; `recipe` as the reusable pattern in "
            "prose — what the agent should do / avoid. For positive "
            "clusters (detail='positive' on the flag) leave "
            "`failure_modes` empty. For negative clusters "
            "(detail='negative') list 2-4 concrete ways the pattern "
            "goes wrong — those are what future agents read to steer "
            "clear. You MAY first call kg_query on one or two op_ids "
            "to read their args_summary / reason fields for context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "op_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The source op entity ids from the flag's memory_ids (>=2).",
                    "minItems": 2,
                },
                "title": {
                    "type": "string",
                    "description": "Short noun phrase naming the pattern (<=80 chars).",
                },
                "when_to_use": {
                    "type": "string",
                    "description": "1-2 sentences on which context triggers this recipe.",
                },
                "recipe": {
                    "type": "string",
                    "description": "The reusable pattern in prose — what to do / avoid.",
                },
                "failure_modes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "2-4 concrete ways the pattern goes wrong. "
                        "REQUIRED for negative clusters; OMIT or [] "
                        "for positive clusters."
                    ),
                },
            },
            "required": ["op_ids", "title", "when_to_use", "recipe"],
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
    touching path — it does not call kg_add or declare predicates.

    Server-side guards (defense in depth, complements the prompt-side
    rules) — addresses 2026-04-25 audit findings:
      - Both endpoints MUST be declared entities. Phantom targets like
        literature citations ("Zhao 2025") otherwise accumulate dead
        rows that the jury can never adjudicate.
      - merged_into chains are followed: if either endpoint is a
        merged tombstone, the canonical id is used instead. Saves the
        jury from chasing forwarding pointers.
    """
    from . import link_author as _la
    from . import mcp_server as _mcp

    kg = _mcp._STATE.kg
    if kg is None:
        return {"success": False, "error": "KG not initialised"}

    # ── Phantom-target guard + merged_into dereference ──
    # kg.get_entity already auto-resolves entity_aliases for merged ids,
    # so the returned row's id is canonical. We rebind the locals so the
    # candidate row uses canonical ids regardless of which alias the
    # caller passed.
    def _resolve(side: str, name: str):
        if not name or not isinstance(name, str):
            return None, {
                "success": False,
                "error": f"{side}_entity must be a non-empty string",
                "phantom": True,
            }
        ent = kg.get_entity(name)
        if not ent:
            return None, {
                "success": False,
                "error": (
                    f"{side}_entity '{name}' is not a declared entity. "
                    "Phantom targets (e.g. literature citations) clog the "
                    "link-author queue. Either kg_search to find the "
                    "canonical id, or skip seeding this pair."
                ),
                "phantom": True,
            }
        return ent["id"], None

    canonical_from, err = _resolve("from", from_entity)
    if err:
        return err
    canonical_to, err = _resolve("to", to_entity)
    if err:
        return err
    if canonical_from == canonical_to:
        return {
            "success": False,
            "error": "from_entity and to_entity resolve to the same canonical id (self-loop)",
        }
    from_entity = canonical_from
    to_entity = canonical_to

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


def _synthesize_operation_template_shim(
    op_ids: list | None = None,
    title: str = "",
    when_to_use: str = "",
    recipe: str = "",
    failure_modes: list | None = None,
    **_ignored,
) -> dict:
    """S3b: mint a template record that distils a cluster of similar
    ops + write `templatizes` edges back to each source op.

    Direct KG writes (not through kg_declare_entity) so the gardener
    bypasses caller-level conflict detection — we deliberately want
    near-duplicate template records to coexist when they describe
    genuinely distinct clusters. Dedup is handled upstream by the
    flag-table's unique index on (kind, memory_key, context_id).

    Parameters arrive from Haiku's structured tool call. The shim is
    defensive: bad input shapes return {"success": False, "error": ...}
    without raising, so the tool-use loop stays stable.
    """
    from . import mcp_server as _mcp

    kg = _mcp._STATE.kg
    if kg is None:
        return {"success": False, "error": "KG not initialised"}
    if not isinstance(op_ids, list) or len(op_ids) < 2:
        return {
            "success": False,
            "error": "op_ids must be a list of >=2 entity ids (cluster membership)",
        }
    op_ids = [str(o) for o in op_ids if o]
    if len(op_ids) < 2:
        return {"success": False, "error": "op_ids collapsed to <2 after cleaning"}

    title = str(title or "").strip()
    when_to_use = str(when_to_use or "").strip()
    recipe = str(recipe or "").strip()
    if not title or not when_to_use or not recipe:
        return {
            "success": False,
            "error": "title, when_to_use, and recipe are all required",
        }
    if not isinstance(failure_modes, list):
        failure_modes = []
    failure_modes = [str(fm).strip() for fm in failure_modes if fm]

    # Deterministic template id so re-running the gardener on the same
    # cluster lands on the same row (add_entity upserts via ON CONFLICT).
    import hashlib

    cluster_hash = hashlib.sha1("|".join(sorted(op_ids)).encode("utf-8")).hexdigest()[:12]
    template_id = f"record_memory_gardener_op_template_{cluster_hash}"

    body = [f"# {title[:80]}", "", "## When to use", when_to_use, "", "## Recipe", recipe]
    if failure_modes:
        body.extend(["", "## Failure modes"])
        body.extend(f"- {fm}" for fm in failure_modes)
    body.append("")
    body.append(f"Derived from {len(op_ids)} operation(s): " + ", ".join(op_ids))
    content = "\n".join(body)

    try:
        kg.add_entity(
            template_id,
            kind="record",
            description=content[:280] if len(content) > 280 else content,
            importance=4,
            properties={
                "title": title,
                "op_ids": op_ids,
                "content_type": "advice",
                "source": "memory_gardener_s3b",
                "full_content": content,
            },
        )
    except Exception as e:
        return {"success": False, "error": f"record write failed: {type(e).__name__}: {e}"}

    # Write `templatizes` edges. The predicate is in
    # _TRIPLE_SKIP_PREDICATES (graph-topology, like similar_to /
    # created_under) so no statement is required. We do NOT silently
    # swallow exceptions here — that pattern masked the original
    # TripleStatementRequired bug 2026-04-25 (Adrian's rule: ensure
    # this crashes, don't omit silently). If a write fails we return
    # success=False with the error string so the gardener tool-use
    # loop sees a real tool_result error and the flag gets a deferred
    # resolution it can retry.
    written = 0
    for op_id in op_ids:
        kg.add_triple(template_id, "templatizes", op_id)
        written += 1

    return {
        "success": True,
        "template_id": template_id,
        "op_ids": op_ids,
        "edges_written": written,
        "title": title,
    }


def _init_tool_dispatch() -> None:
    """Lazy wire-up so import-time side-effects stay small."""
    if _TOOL_DISPATCH:
        return
    from . import mcp_server as _mcp

    _TOOL_DISPATCH["mempalace_kg_query"] = _mcp.tool_kg_query
    _TOOL_DISPATCH["mempalace_kg_search"] = _mcp.tool_kg_search
    _TOOL_DISPATCH["mempalace_kg_merge_entities"] = _mcp.tool_kg_merge_entities
    _TOOL_DISPATCH["mempalace_kg_invalidate"] = _mcp.tool_kg_invalidate
    _TOOL_DISPATCH["mempalace_kg_update_entity"] = _mcp.tool_kg_update_entity
    _TOOL_DISPATCH["mempalace_kg_delete_entity"] = _mcp.tool_kg_delete_entity
    _TOOL_DISPATCH["mempalace_propose_edge_candidate"] = _propose_edge_candidate_shim
    _TOOL_DISPATCH["mempalace_synthesize_operation_template"] = _synthesize_operation_template_shim


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

# ═══════════════════════════════════════════════════════════════════
# Per-flag-kind system prompts (2026-04-25 refactor)
#
# History: until 2026-04-25 the gardener used ONE shared system
# prompt that listed every flag kind's action block in sequence.
# That worked but Haiku had to read ~92 lines of rules per call when
# only ~10-15 were relevant to its current flag. Adrian's design
# goal — focused per-kind prompts that don't ask Haiku to filter
# noise — was articulated repeatedly but never landed at the
# system-prompt level (the 0dd459b commit-message phrase "per-kind
# prompt" referred to record-vs-entity branching INSIDE the shared
# prompt's generic_summary block).
#
# This refactor delivers the original goal:
#   * _SHARED_PREAMBLE — universal rules every prompt carries
#     (8 tools list, recovery rule, never-do list, error priority)
#   * _TASK_<KIND>     — focused action block for one flag kind
#   * _PROMPTS_BY_KIND — kind -> shared + task
#   * _select_prompt   — dispatch helper called from process_batch
#
# Token cost per call drops ~60-70% on most kinds because Haiku no
# longer reads the other 7 kinds' rules. Failure modes that once
# read "Haiku obeyed wrong block" get harder.
# ═══════════════════════════════════════════════════════════════════


_SHARED_PREAMBLE = """\
You are memory_gardener — a corpus-refinement agent that resolves ONE mempalace flag per invocation. You end by making exactly one mutation tool call. You do NOT declare new entities or predicates; you do NOT author edges directly; you do NOT follow any mempalace protocol (no wake_up/declare_intent/finalize_intent/diary — you don't have those tools anyway). You do NOT emit a wrap-up text message — after the mutation succeeds, just stop.

Pass agent="memory_gardener" on every mutation that accepts an agent parameter.

THE 8 TOOLS YOU HAVE — nothing else exists:
  mempalace_kg_query                       read an entity's edges + description by EXACT id
  mempalace_kg_search                      fuzzy search memories+entities (use when you don't have an exact id, OR for grounding evidence, OR to verify a target exists)
  mempalace_kg_merge_entities              merge two entities (source folded into target)
  mempalace_kg_invalidate                  invalidate one specific triple (subject, predicate, object)
  mempalace_kg_update_entity               update an entity's description/importance (NOT records)
  mempalace_kg_delete_entity               soft-delete an entity (invalidates all its edges, sets status='deleted')
  mempalace_propose_edge_candidate         seed a pair into the link-author queue
  mempalace_synthesize_operation_template  mint a template record distilling an op cluster

UNIVERSAL RECOVERY RULE — when ANY tool returns "Not found in entities" or "Not found in memories":
  Do NOT defer immediately. Call kg_search with 2-3 keywords drawn from the flag's `detail` field plus the bad id. The search will surface the canonical id (or confirm the entity truly doesn't exist). Then retry the original mutation with the corrected id. Defer ONLY if kg_search returns zero hits.

WHAT YOU MUST NEVER DO:
  - Never call any tool outside the 8 listed above. Nothing else exists in your toolset.
  - Never try to declare a new predicate. That's the link-author's job.
  - Never call kg_add or author an edge directly. Always route through propose_edge_candidate.
  - Never propose an edge to a string that isn't a declared entity (no phantom targets).
  - Never invent description / recipe content not attested in retrieved evidence.
  - Never delete a record that documents a past event truthfully (those aren't "stale", they're history).
  - Never emit a final text summary. After the mutation tool_result comes back, stop.

ERROR RECOVERY (in priority order):
  1. 'Not found in entities' / 'Not found in memories' → universal recovery rule above (kg_search to find canonical, retry).
  2. propose_edge_candidate returns {"success": true, "inserted": false} → clean resolution (already connected or already seeded).
  3. Schema/typo error you can plausibly fix → retry ONCE with the corrected argument, then stop.
  4. Anything else → stop. The flag will be auto-deferred and re-attempted later.
"""


_TASK_DUPLICATE_PAIR = """\
YOUR TASK — duplicate_pair:
  The flag's memory_ids carry two entity ids stating the same thing. One is canonical; the other should fold into it.
  1. (optional) kg_query each id to confirm which is canonical (more edges = more canonical).
  2. mempalace_kg_merge_entities(source=<weaker id>, target=<canonical id>, agent="memory_gardener")
"""


_TASK_CONTRADICTION_PAIR = """\
YOUR TASK — contradiction_pair:
  The two memory_ids contradict each other on a fact. Pick the stale edge and invalidate it (do NOT delete either record entirely — they may carry other valid edges).
  1. kg_query the subject/object pair to see the actual triple text + valid_from.
  2. Pick the stale edge (older valid_from, or the one whose claim is contradicted by newer evidence).
  3. mempalace_kg_invalidate(subject, predicate, object, agent="memory_gardener")
  Surgical: prefer invalidating the one wrong triple over deleting either record.
"""


_TASK_STALE = """\
YOUR TASK — stale:
  Distinguish between (a) a SPECIFIC stale edge identified in flag.detail and (b) the WHOLE memory being genuinely obsolete. Past events that were true at the time are NOT stale — those are valid event memories; do NOT delete them.
  → If specific stale edge: mempalace_kg_invalidate(subject, predicate, object, agent="memory_gardener")
  → If whole memory genuinely obsolete: mempalace_kg_delete_entity(entity_id=<memory_ids[0]>, agent="memory_gardener")
"""


_TASK_ORPHAN = """\
YOUR TASK — orphan:
  No edges connect this memory to anything; it offers no retrieval value.
  → mempalace_kg_delete_entity(entity_id=<memory_ids[0]>, agent="memory_gardener")
  No kg_query needed — orphan means no edges to preserve.
"""


_TASK_GENERIC_SUMMARY = """\
YOUR TASK — generic_summary:
  GROUND BEFORE WRITING. Never invent a description from thin context.

  Target shape (S4 ship 060d08d, locked with Adrian 2026-04-25): every
  summary follows the WHAT + WHY + SCOPE? structure. WHAT is a noun
  phrase naming the entity; WHY is a purpose / role / claim clause
  (NOT a name-restatement); SCOPE is an optional temporal / domain
  qualifier. Total ≤280 chars. Single-noun stubs ("Adrian") and
  name-restating placeholders ("Notes on X") are rejected by
  validate_summary at write time, so a description you write here
  must include a WHY clause separated by an em-dash, semicolon, or
  a role-verb (filters / orchestrates / stores / carries / enforces).

  Examples of GOOD summaries (each line is a target shape):
    "InjectionGate — runtime gate that filters retrieved memories before injection via Haiku tool-use, emits quality flags"
    "intent.py — orchestrates declare_intent slot validation and finalize_intent feedback coverage; central glue between hooks and the gate"
    "Adrian Rivero — DSpot tech lead and project owner of mempalace; pushes back on speculation, favours load-bearing demos over plans"

  Examples of BAD summaries (rejected by validate_summary):
    "Adrian"                          ← single noun, no WHY
    "the project"                     ← name-restating, no WHY
    "Notes on Python decorators"      ← topic-only, no WHY
    "File: D:/Flowsev/mempalace/x.py" ← auto-stub from file mint

  Procedure:
  1. kg_query(entity=<memory_ids[0]>) to see kind + current description.
  2. kg_search(context={queries: ["<entity name in plain English>", "<topic from flag.detail>"], keywords: ["<2-3 exact terms>"]}, limit=5) to retrieve grounding evidence.
  3. If kg_search returns ZERO hits OR all hits are themselves stub-length descriptions: defer with reason='no grounding evidence'. Do NOT fabricate.
  4. Compose a NEW description using ONLY claims attested in the retrieved evidence. Use the "Noun — purpose-clause" shape. Better short and true than long and guessed.
  5. Apply by KIND:
      kind='record'  → MUST use the delete recipe because record content is embedded so in-place update breaks cosine retrieval:
                        a. mempalace_kg_delete_entity(entity_id=<memory_ids[0]>, agent="memory_gardener")
                        That's your one mutation. Redeclaration needs kg_declare_entity which you do NOT have, so deletion of an under-described record is the correct path — a future write will recreate it with proper grounding.
      kind!=record   → mempalace_kg_update_entity(entity=<memory_ids[0]>, description=<new <=280-char WHAT+WHY summary>, agent="memory_gardener")
                        kg_update_entity now runs validate_summary on the new description; if it rejects with "missing WHAT+WHY structure", revise to add a role-verb or em-dash separator and retry.
"""


_TASK_EDGE_CANDIDATE = """\
YOUR TASK — edge_candidate:
  VERIFY BOTH ENDPOINTS before seeding. Phantom-target seeds clog the link-author queue forever.
  1. kg_query(entity=memory_ids[0]) — confirm exists. If 'Not found', kg_search the id; if still missing, defer.
  2. kg_query(entity=memory_ids[1]) — confirm exists. Same recovery.
  3. mempalace_propose_edge_candidate(from_entity=memory_ids[0], to_entity=memory_ids[1], weight=0.7)
  NEVER call kg_add. NEVER declare a predicate. The link-author jury picks the predicate.
"""


_TASK_UNLINKED_ENTITY = """\
YOUR TASK — unlinked_entity:
  flag.detail names a concept that should be linked to memory_ids[0] but the concept may not be a declared entity yet (e.g. a literature citation like "Zhao 2025", a code symbol, or an external system).
  1. kg_search(context={queries: ["<concept name>", "<flag.detail topic>"], keywords: ["<concept name>", "<2 related terms>"]}, limit=5) to verify the concept is declared.
  2. If kg_search returns NO matching entity for the concept: defer with reason='target entity not declared'.
  3. Otherwise mempalace_propose_edge_candidate(from_entity=memory_ids[0], to_entity=<canonical id from search>, weight=0.7)
"""


_TASK_OP_CLUSTER_TEMPLATIZABLE = """\
YOUR TASK — op_cluster_templatizable:
  GROUND BEFORE WRITING. Never invent recipe content not attested in the cluster's op rows.
  1. SHOULD kg_query at least one op_id from memory_ids to read its args_summary + reason. For a positive cluster (flag.detail = "positive"), sample one of the higher-quality ones; for a negative cluster (flag.detail = "negative"), sample at least one to confirm the failure mode is real.
  2. If the queried op rows have empty / stub-length args_summary AND empty reason: defer with reason='no grounding evidence in op cluster'. Do NOT fabricate a recipe over rated-but-unannotated ops.
  3. Compose:
      title: short noun phrase naming the pattern, <=80 chars, anchored on what the queried ops actually did.
      when_to_use: 1-2 sentences on which context triggers this — drawn from the args_summary / reason fields.
      recipe: the reusable pattern in prose — what TO DO for "positive" detail, what to AVOID for "negative".
      failure_modes: REQUIRED for "negative" detail (2-4 concrete entries, each grounded in a queried op's reason). OMIT for "positive".
  4. mempalace_synthesize_operation_template(
         op_ids=<memory_ids array from the flag VERBATIM>,
         title=..., when_to_use=..., recipe=..., failure_modes=[...]
     )
  Do NOT call kg_add or kg_declare_entity — the shim handles record + templatizes edges idempotently (deterministic id from sorted op_ids; re-runs upsert).
"""


_PROMPTS_BY_KIND: dict[str, str] = {
    "duplicate_pair": _SHARED_PREAMBLE + "\n" + _TASK_DUPLICATE_PAIR,
    "contradiction_pair": _SHARED_PREAMBLE + "\n" + _TASK_CONTRADICTION_PAIR,
    "stale": _SHARED_PREAMBLE + "\n" + _TASK_STALE,
    "orphan": _SHARED_PREAMBLE + "\n" + _TASK_ORPHAN,
    "generic_summary": _SHARED_PREAMBLE + "\n" + _TASK_GENERIC_SUMMARY,
    "edge_candidate": _SHARED_PREAMBLE + "\n" + _TASK_EDGE_CANDIDATE,
    "unlinked_entity": _SHARED_PREAMBLE + "\n" + _TASK_UNLINKED_ENTITY,
    "op_cluster_templatizable": _SHARED_PREAMBLE + "\n" + _TASK_OP_CLUSTER_TEMPLATIZABLE,
}


def _select_prompt(flag_kind: str) -> str:
    """Return the focused system prompt for this flag kind.

    Each kind sees only the rules it needs — shared preamble + that
    kind's task block. Cuts ~60-70% of irrelevant token surface vs.
    the legacy single-prompt approach (2026-04-25 refactor).

    Raises KeyError on unknown kinds — fail loud, not silent.
    """
    try:
        return _PROMPTS_BY_KIND[flag_kind]
    except KeyError:
        raise KeyError(
            f"Unknown flag_kind {flag_kind!r}; valid: {sorted(_PROMPTS_BY_KIND)}"
        ) from None


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
    # S3b: minting a template record + writing templatizes edges is a
    # mutation, so it counts as a proper resolution move.
    "mempalace_synthesize_operation_template",
}


def _before_desc_from_trace(target_id: str, tool_calls: list[ToolCallTrace]) -> str:
    """Extract the pre-mutation description for `target_id` from any
    prior kg_query call in the trace. Returns a truncated snippet for
    resolution_note embedding, or '' if no such trace is present.

    Closes audit finding #6: old descriptions are overwritten in place
    with no version history; capturing them in resolution_note makes
    summary rewrites and deletes auditable + reversible.
    """
    if not target_id:
        return ""
    for tc in tool_calls:
        if tc.name != "mempalace_kg_query":
            continue
        if tc.arguments.get("entity") != target_id:
            continue
        # tool_kg_query returns a details block; older path returned a
        # flat 'description'. Handle both.
        details = tc.result.get("details") or {}
        desc = details.get("description") or details.get("content") or ""
        if not desc:
            desc = tc.result.get("description") or ""
        desc = str(desc or "").strip().replace("\n", " ")
        if desc:
            return desc[:180]
    return ""


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
        # Capture the source description as a breadcrumb so the merge
        # note preserves what was folded in.
        before = _before_desc_from_trace(last.arguments.get("source", ""), tool_calls)
        suffix = f" | before={before!r}" if before else ""
        return ("merged", f"merged via {args_short}{suffix}")
    if last.name == "mempalace_kg_invalidate":
        return (
            ("pruned" if flag_kind == "orphan" else "invalidated"),
            f"invalidated via {args_short}",
        )
    if last.name == "mempalace_kg_delete_entity":
        before = _before_desc_from_trace(last.arguments.get("entity_id", ""), tool_calls)
        suffix = f" | before={before!r}" if before else ""
        return ("pruned", f"deleted via {args_short}{suffix}")
    if last.name == "mempalace_kg_update_entity":
        before = _before_desc_from_trace(last.arguments.get("entity", ""), tool_calls)
        suffix = f" | before={before!r}" if before else ""
        return ("summary_rewritten", f"updated via {args_short}{suffix}")
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
    if last.name == "mempalace_synthesize_operation_template":
        # S3b: op_cluster_templatizable flag resolved by minting a
        # template record. The note captures the new template_id +
        # edge-write count so the audit log tells the full story.
        tid = last.result.get("template_id") or "?"
        n_edges = last.result.get("edges_written", 0)
        n_ops = len(last.result.get("op_ids") or [])
        return (
            "templatized",
            f"minted template {tid} covering {n_ops} op(s), {n_edges} edge(s) written",
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
        loop = _run_anthropic_loop(_select_prompt(flag_kind), user_prompt, model)
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
        cmd = [
            sys.executable,
            "-m",
            "mempalace",
            "gardener",
            "process",
            "--max-batches",
            str(_AUTO_TRIGGER_MAX_BATCHES),
        ]
        if os.name == "nt":
            DETACHED_PROCESS = 0x00000008
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                close_fds=True,
                env=env,
            )
        else:
            subprocess.Popen(
                cmd,
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
