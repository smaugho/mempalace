"""
memory_gardener.py -- out-of-session corpus-refinement agent.

Architecture (2026-04-24, third rewrite):
  Uses the raw `anthropic` Python SDK with a hand-rolled tool-use loop.
  NOT claude-agent-sdk. NOT claude CLI subprocess. NOT MCP stdio.

  Why: all those layers had Windows show-stoppers this session --
    * claude-agent-sdk Python 0.1.65: STATUS_ACCESS_VIOLATION on any
      MCP tool call.
    * claude-agent-sdk TS 0.1.77: "tool_use ids must be unique" 400
      on second assistant turn (CLI bug #20631).
    * claude-agent-sdk TS 0.2.119: same STATUS_ACCESS_VIOLATION as
      Python, even at maxTurns=2.
    * `claude --print` subprocess: works but cannot guarantee hooks
      are disabled on Windows.

  The raw anthropic SDK is pure HTTPS -- no subprocess, no stdio, no
  hooks, no plugin loading, no Windows-specific transport quirks.
  We call mempalace tool functions DIRECTLY via mcp_server.tool_*
  (no MCP wire protocol in between). Multi-turn tool use works
  naturally because we own the message history.

Flow (process_batch → one SDK session per flag):
  1. list_pending_flags(1) -- one flag per batch keeps resolution
     mapping trivial.
  2. start_gardener_run -- audit row opens.
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
  7. finish_gardener_run -- audit row closes with counters + errors.

Kill switch: ENABLED BY DEFAULT. Set MEMPALACE_GARDENER_DISABLED=1 to opt out.
Auth: ANTHROPIC_API_KEY from palace .env (via injection_gate loader).
Session gate: MEMPALACE_GARDENER_ACTIVE=1 is set so mempalace's
_require_sid / _require_agent bypass for this process.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import subprocess
import sys
import time
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
_AUTO_TRIGGER_MAX_BATCHES = 40
_DEFAULT_GARDENER_MODEL = "claude-haiku-4-5"

# Single-process serialization (Adrian 2026-04-29 design, kernel-flock
# upgrade same day). Stops parallel gardener spawns from racing
# list_pending_flags + kg_update_entity.
#
# PRIMARY mechanism: OS-level advisory file lock via fcntl.flock (POSIX)
# or msvcrt.locking (Windows). The kernel releases the lock automatically
# when the holding process exits for ANY reason -- normal exit, SIGTERM,
# SIGKILL, crash, OOM, power loss. No Python signal handlers, no PID
# liveness checks, no atexit dance needed for correctness. This is the
# same primitive PostgreSQL postmaster, Apache, nginx use.
#
# FALLBACK: 1-hour TTL on the lockfile mtime. If the file is older than
# _LOCKFILE_TTL_SEC AND we still cannot acquire the kernel lock, treat as
# stale (e.g. NFS mount with broken flock semantics) and forcibly clear
# before retrying once. Caps any pathological hang at 1h. Gardener spawns
# at max_batches=40 take ~12 min in the worst case, so 1h is comfortable.
#
# The file content (PID + UTC start time) is informational only -- used
# for telemetry / debugging. Correctness comes from the kernel lock.
_LOCKFILE_PATH = pathlib.Path(os.path.expanduser("~/.mempalace/hook_state/gardener.pid"))
_LOCKFILE_TTL_SEC = 3600  # 1 hour

# Module-level holder for the open lockfile descriptor. The fd MUST outlive
# the function that acquired it so the kernel keeps the advisory lock until
# process exit (or explicit _release_lock).
_LOCKFILE_FD: Any = None


def _flock_nonblocking(fd: Any) -> bool:
    """Attempt non-blocking exclusive advisory lock on fd. True if acquired."""
    try:
        if sys.platform == "win32":
            import msvcrt

            fd.seek(0)
            msvcrt.locking(fd.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except (OSError, BlockingIOError):
        return False


def _try_acquire_lock() -> bool:
    """Acquire kernel-released advisory lock on the gardener lockfile.

    Returns True if we now hold the lock and may proceed. The caller
    must NOT close the lockfile descriptor before exit; that is what
    keeps the kernel lock alive. _release_lock() handles cleanup.

    Returns False if another gardener already holds the lock.

    Filesystem failures fail open (return True without locking) rather
    than block all gardener work behind a flaky disk."""
    global _LOCKFILE_FD
    try:
        _LOCKFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return True  # fail open

    # Open in r+ if exists, else create. 'a+' creates+positions at end;
    # we'll seek as needed for locking and rewriting content.
    try:
        fd = open(_LOCKFILE_PATH, "a+")
    except OSError:
        return True  # fail open

    if _flock_nonblocking(fd):
        # Got it. Stamp our PID + UTC start time for telemetry, then
        # hold the fd open so the kernel keeps the lock alive.
        try:
            fd.seek(0)
            fd.truncate(0)
            from datetime import datetime as _dt, timezone as _tz

            fd.write(f"{os.getpid()}\n{_dt.now(_tz.utc).isoformat(timespec='seconds')}\n")
            fd.flush()
        except OSError:
            pass
        _LOCKFILE_FD = fd
        return True

    # Couldn't lock. Could be a live gardener (correct, return False),
    # OR a stale lock on a broken flock filesystem (rare). Check TTL.
    try:
        age = time.time() - _LOCKFILE_PATH.stat().st_mtime
    except OSError:
        age = 0
    fd.close()

    if age > _LOCKFILE_TTL_SEC:
        # Stale beyond cap. Force-clear and retry once. If a real live
        # gardener still holds the kernel lock somehow, the retry will
        # also fail to flock and we'll bail honestly.
        try:
            _LOCKFILE_PATH.unlink()
        except OSError:
            pass
        try:
            fd = open(_LOCKFILE_PATH, "a+")
        except OSError:
            return True  # fail open
        if _flock_nonblocking(fd):
            try:
                fd.seek(0)
                fd.truncate(0)
                from datetime import datetime as _dt, timezone as _tz

                fd.write(f"{os.getpid()}\n{_dt.now(_tz.utc).isoformat(timespec='seconds')}\n")
                fd.flush()
            except OSError:
                pass
            _LOCKFILE_FD = fd
            return True
        fd.close()

    return False


def _release_lock() -> None:
    """Close the held fd; the kernel releases the advisory lock on close.

    Idempotent. Best-effort cleanup of the file content for tidy state;
    correctness does not depend on the file being deleted -- the kernel
    has already released the lock by the time we try to unlink."""
    global _LOCKFILE_FD
    if _LOCKFILE_FD is None:
        return
    try:
        _LOCKFILE_FD.close()
    except OSError:
        pass
    _LOCKFILE_FD = None
    try:
        if _LOCKFILE_PATH.exists():
            _LOCKFILE_PATH.unlink()
    except OSError:
        pass


def _is_lock_held() -> bool:
    """Probe: does another process currently hold the gardener lock?

    Used by maybe_trigger_from_finalize to decide whether to Popen.
    Briefly tries to acquire and immediately releases; if it can acquire
    nobody else holds, so it's safe to spawn. The microsecond gap
    between this probe-release and the spawned subprocess re-acquiring
    is harmless: worst case the spawn races and one side immediately
    bails, no data corruption."""
    try:
        if not _LOCKFILE_PATH.exists():
            return False
    except OSError:
        return False
    try:
        fd = open(_LOCKFILE_PATH, "a+")
    except OSError:
        return False
    try:
        if _flock_nonblocking(fd):
            return False  # we got the lock => nobody else holds it
        return True
    finally:
        try:
            fd.close()
        except OSError:
            pass


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
# Safety: fork-bomb cascade is structurally impossible -- the
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
                        "summary": {
                            "type": "object",
                            "description": (
                                "Optional structured WHAT+WHY+SCOPE? anchor "
                                "for the search context. Same dict shape as "
                                "writes; mirrors canonical _CONTEXT_SCHEMA_READ. "
                                "Pass it when known so the search context can "
                                "be indexed for future retrieval feedback."
                            ),
                            "properties": {
                                "what": {"type": "string"},
                                "why": {"type": "string"},
                                "scope": {"type": "string"},
                            },
                            "required": ["what", "why"],
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
                "summary": {
                    "type": "object",
                    "description": (
                        "Optional structured summary {what, why, scope?} "
                        "for the merged target. Dict-only contract; strings "
                        "rejected. Mirrors kg_update_entity.summary."
                    ),
                    "properties": {
                        "what": {"type": "string"},
                        "why": {"type": "string"},
                        "scope": {"type": "string"},
                    },
                    "required": ["what", "why"],
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
            "DO NOT call kg_add directly -- the gardener is not "
            "authorised to add edges or declare new predicates. "
            "Instead, pass the two entity ids to this tool; the "
            "link-author process (Opus-designed jury + Haiku jurors) "
            "will later decide whether to author the edge AND pick "
            "the right predicate -- possibly an existing one, possibly "
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
                        "Evidence weight 0.1-1.0. Default 0.7 for "
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
            "Update an entity's summary / importance. Use for "
            "generic_summary flags on kind!=record entities. Pass `summary` "
            "as a STRUCTURED DICT {what, why, scope?} -- NOT a string. The dict "
            "is rendered to prose for the embedded text; rendered prose ≤280 "
            "chars. Strings are rejected by validate_summary on the server "
            "side. For records (kind='record'), do NOT call this tool -- use "
            "kg_delete_entity then kg_declare_entity to replace record content "
            "(the record content is embedded so in-place updates break cosine "
            "retrieval)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "Entity ID to update (kind!=record).",
                },
                "summary": {
                    "type": "object",
                    "description": (
                        "REQUIRED dict shape -- {what, why, scope?}. "
                        "what: noun phrase ≥5 chars naming the entity. "
                        "why: purpose / role / claim ≥15 chars. "
                        "scope: optional temporal/domain qualifier ≤100 chars. "
                        "Rendered prose form ≤280 chars. STRINGS ARE REJECTED. "
                        "Mirrors mcp_server canonical schema 2026-04-26."
                    ),
                    "properties": {
                        "what": {"type": "string"},
                        "why": {"type": "string"},
                        "scope": {"type": "string"},
                    },
                    "required": ["what", "why"],
                },
                "importance": {
                    "type": "integer",
                    "description": "1-5 (1=junk, 5=critical).",
                    "minimum": 1,
                    "maximum": 5,
                },
                "agent": {
                    "type": "string",
                    "description": "Always 'memory_gardener'.",
                },
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
                "entity": {"type": "string"},
                "agent": {"type": "string", "description": "Always 'memory_gardener'."},
            },
            "required": ["entity", "agent"],
        },
    },
    {
        "name": "mempalace_synthesize_operation_template",
        "description": (
            "S3b: resolve an op_cluster_templatizable flag by minting a "
            "reusable template record that distils the cluster's pattern "
            "and writing `templatizes` edges from the new record back to "
            "every source operation it covers. `op_ids` are the entity "
            "ids from the flag's memory_ids array -- copy them verbatim. "
            "Compose `title` as a short noun phrase (<=80 chars) naming "
            "the pattern; `when_to_use` as 1-2 sentences saying which "
            "context triggers it; `recipe` as the reusable pattern in "
            "prose -- what the agent should do / avoid. For positive "
            "clusters (detail='positive' on the flag) leave "
            "`failure_modes` empty. For negative clusters "
            "(detail='negative') list 2-4 concrete ways the pattern "
            "goes wrong -- those are what future agents read to steer "
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
                    "description": "The reusable pattern in prose -- what to do / avoid.",
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
    touching path -- it does not call kg_add or declare predicates.

    Server-side guards (defense in depth, complements the prompt-side
    rules) -- addresses 2026-04-25 audit findings:
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
    bypasses caller-level conflict detection -- we deliberately want
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

    # Cold-start lock 2026-05-01: every record carries a structured
    # {what, why, scope?} summary. The gardener has all the signals
    # already (title, when_to_use, op_ids) -- build the dict inline
    # rather than firing an auto_author Haiku call. This stays cheap
    # (no API call) and keeps the gardener's own "rationale" entities
    # truthful: WHAT names the template, WHY captures when-to-use,
    # SCOPE is the cluster size. Pre-cold-start the path called
    # kg.add_entity directly with no summary -- one of the 12
    # catalogued bypass surfaces.
    _gardener_what = title[:118] if len(title) > 8 else f"operation template: {title}"
    _gardener_why = when_to_use.strip()[:240]
    if len(_gardener_why) < 15:
        _gardener_why = (
            f"{_gardener_why} (memory_gardener-derived operation template "
            f"covering {len(op_ids)} clustered operations)"
        )[:240]
    _gardener_summary = {
        "what": _gardener_what,
        "why": _gardener_why,
        "scope": f"derived from {len(op_ids)} clustered operations"[:100],
    }
    try:
        from .knowledge_graph import (
            SummaryStructureRequired,
            coerce_summary_for_persist,
        )

        _gardener_summary = coerce_summary_for_persist(
            _gardener_summary, context_for_error="memory_gardener.template.summary"
        )
    except SummaryStructureRequired as exc:
        # Fall back on auto_author rather than degrading to no-summary.
        # The cold-start invariant is non-negotiable; if the structured
        # signals can't pass validate_summary we ask Haiku to author one.
        try:
            from .auto_author import AuthorRequest, auto_author_summary

            _gardener_summary = auto_author_summary(
                AuthorRequest(
                    kind="rationale",
                    anchor_text=f"Title: {title}\nWhen to use: {when_to_use}\nRecipe: {recipe}",
                    context_blocks=[
                        f"Operation cluster: {len(op_ids)} ops -- {', '.join(op_ids[:5])}",
                    ],
                )
            )
        except Exception as auto_err:
            return {
                "success": False,
                "error": (
                    f"summary validation failed and auto_author fallback "
                    f"also failed: {exc!r} / {auto_err!r}. The cold-start "
                    f"invariant requires every record to carry a summary; "
                    f"defer this template until the gardener can author one."
                ),
            }

    try:
        kg.add_entity(
            template_id,
            kind="record",
            content=content[:280] if len(content) > 280 else content,
            importance=4,
            properties={
                "title": title,
                "op_ids": op_ids,
                "content_type": "advice",
                "source": "memory_gardener_s3b",
                "full_content": content,
                "summary": _gardener_summary,
            },
        )
    except Exception as e:
        return {"success": False, "error": f"record write failed: {type(e).__name__}: {e}"}

    # Write `templatizes` edges. The predicate is in
    # _TRIPLE_SKIP_PREDICATES (graph-topology, like similar_to /
    # created_under) so no statement is required. We do NOT silently
    # swallow exceptions here -- that pattern masked the original
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
    """Execute one tool call. Never raises -- always returns a dict
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
# goal -- focused per-kind prompts that don't ask Haiku to filter
# noise -- was articulated repeatedly but never landed at the
# system-prompt level (the 0dd459b commit-message phrase "per-kind
# prompt" referred to record-vs-entity branching INSIDE the shared
# prompt's generic_summary block).
#
# This refactor delivers the original goal:
#   * _SHARED_PREAMBLE -- universal rules every prompt carries
#     (8 tools list, recovery rule, never-do list, error priority)
#   * _TASK_<KIND>     -- focused action block for one flag kind
#   * _PROMPTS_BY_KIND -- kind -> shared + task
#   * _select_prompt   -- dispatch helper called from process_batch
#
# Token cost per call drops ~60-70% on most kinds because Haiku no
# longer reads the other 7 kinds' rules. Failure modes that once
# read "Haiku obeyed wrong block" get harder.
# ═══════════════════════════════════════════════════════════════════


_SHARED_PREAMBLE = """\
You are memory_gardener -- a corpus-refinement agent that resolves ONE mempalace flag per invocation. You end by making exactly one mutation tool call. You do NOT declare new entities or predicates; you do NOT author edges directly; you do NOT follow any mempalace protocol (no wake_up/declare_intent/finalize_intent/diary -- you don't have those tools anyway). You do NOT emit a wrap-up text message -- after the mutation succeeds, just stop.

Pass agent="memory_gardener" on every mutation that accepts an agent parameter.

THE 8 TOOLS YOU HAVE -- nothing else exists:
  mempalace_kg_query                       read an entity's edges + description by EXACT id
  mempalace_kg_search                      fuzzy search memories+entities (use when you don't have an exact id, OR for grounding evidence, OR to verify a target exists)
  mempalace_kg_merge_entities              merge two entities (source folded into target)
  mempalace_kg_invalidate                  invalidate one specific triple (subject, predicate, object)
  mempalace_kg_update_entity               update an entity's description/importance (NOT records)
  mempalace_kg_delete_entity               soft-delete an entity (invalidates all its edges, sets status='deleted')
  mempalace_propose_edge_candidate         seed a pair into the link-author queue
  mempalace_synthesize_operation_template  mint a template record distilling an op cluster

UNIVERSAL RECOVERY RULE -- when ANY tool returns "Not found in entities" or "Not found in memories":
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

DEFER MEANS STOP:
  When any task block says 'defer', that means: stop without calling any mutation tool. The flag is auto-retried later. Do NOT call a 'no-op' tool, do NOT emit a final text message -- just stop.

CONTEXT_ID IS INFORMATIONAL:
  flag.context_id (when present) labels which Context surfaced the source memories. It is NEVER a target for merge/invalidate/edge-propose. Targets always come from flag.memory_ids or kg_search results.
"""


_TASK_DUPLICATE_PAIR = """\
YOUR TASK -- duplicate_pair:
  The flag's memory_ids carry two entity ids stating the same thing. Just MERGE them. Do NOT spend cycles deciding which is "more canonical" -- the merge folds both histories into one. Deterministic tiebreaker for idempotence: pick the lexicographically SMALLER id as target; the other becomes source.

  GUARDS -- defer (do nothing) instead of merging when:
    - kinds differ (record vs entity vs agent vs concept). Cross-kind merges corrupt the graph.
    - descriptions actually disagree on a substantive fact. That's a contradiction_pair, not a duplicate.
    - memory_ids[0] == memory_ids[1] (self-merge is a no-op; flag is malformed).

  GOOD example:
    memory_ids=["adrian_rivero", "adrian"], detail="same person, alias"
    Both kind=person, descriptions don't conflict.
    Tiebreaker: "adrian" < "adrian_rivero" → target="adrian", source="adrian_rivero".
    → mempalace_kg_merge_entities(source="adrian_rivero", target="adrian", agent="memory_gardener")

  DEFER example:
    memory_ids=["adrian_rivero", "adrian_rivero_diary_2026"] -- different kinds (person vs record).
    → defer; do not merge.

  CALL: mempalace_kg_merge_entities(source=<larger id>, target=<smaller id>, agent="memory_gardener")
"""


_TASK_CONTRADICTION_PAIR = """\
YOUR TASK -- contradiction_pair:
  Two memory_ids contradict each other on a fact. Invalidate the stale TRIPLE (the specific edge), not either entire memory -- both may carry other valid edges.

  PROCEDURE:
  1. kg_query both ids; identify the conflicting (subject, predicate, object) pair.
  2. Pick the stale edge: older valid_from, OR the one whose object newer evidence contradicts. If both look equally valid, defer with reason='unclear winner'.
  3. mempalace_kg_invalidate(subject, predicate, object, agent="memory_gardener")

  GOOD example:
    flag.detail: "rate_limit was 100 in 2025-03 record; 2026-01 record says 500"
    kg_query → t1: (api, has_rate_limit, "100"), valid_from=2025-03-01
               t2: (api, has_rate_limit, "500"), valid_from=2026-01-15
    t2 supersedes t1 (newer, same subject+predicate).
    → mempalace_kg_invalidate(subject="api", predicate="has_rate_limit", object="100", agent="memory_gardener")

  EDGE CASE -- dimensional, not contradictory:
    t1 = (api, has_rate_limit, "100") in env=prod;
    t2 = (api, has_rate_limit, "500") in env=staging.
    Different scopes, neither stale.
    → defer with reason='dimensional, not contradictory'.

  Surgical: prefer invalidating the one wrong triple over deleting either record.
"""


_TASK_STALE = """\
YOUR TASK -- stale:
  Distinguish (a) ONE specific stale edge identified in flag.detail from (b) the WHOLE memory being genuinely obsolete. Past events that were true at the time are NOT stale -- they are valid event memories; do NOT delete them.

  → If specific stale edge:
       mempalace_kg_invalidate(subject, predicate, object, agent="memory_gardener")
  → If whole memory genuinely obsolete:
       mempalace_kg_delete_entity(entity=<memory_ids[0]>, agent="memory_gardener")

  GOOD example (a) -- specific stale edge:
    flag.detail: "deploys_to edge points to old vercel project; project moved to fly.io 2026-02"
    → mempalace_kg_invalidate(subject="myapp", predicate="deploys_to", object="myapp_vercel", agent="memory_gardener")
    (The memory entity stays; it carries other valid edges.)

  GOOD example (b) -- whole memory obsolete:
    flag.detail: "configures rate_limit_v1 module; module was deleted 2026-03"
    kg_query → memory only references the deleted module.
    → mempalace_kg_delete_entity(entity=<memory_ids[0]>, agent="memory_gardener")

  DO NOT DELETE -- past event:
    flag.detail: "describes 2025 outage" -- outage really happened.
    → defer with reason='past event, not stale'. History is not noise.
"""


_TASK_ORPHAN = """\
YOUR TASK -- orphan:
  flag claims this memory has no edges. Before deleting, try to LINK it to something that does exist -- orphans often happen because a related entity was created but never wired up. Delete only when no related entity surfaces.

  PROCEDURE:
  1. kg_query(entity=memory_ids[0]) -- confirm it really is orphaned. ANY domain edge other than bookkeeping (created_under / surfaced / rated_useful / rated_irrelevant) means it is NOT orphan: defer with reason='not orphan; has <N> edges'.
  2. Read the entity's description (returned by kg_query in `details`). Pull 2-3 keywords from it.
  3. kg_search(context={queries: [<entity name>, <description topic>], keywords: [<2-3 keywords from description>], summary: {what: <entity name>, why: 'gardener orphan-link probe'}}, limit=5)
  4. If kg_search returns a clearly related entity (different id, same domain):
       → mempalace_propose_edge_candidate(from_entity=memory_ids[0], to_entity=<canonical id>, weight=0.7)
       The link-author jury picks the predicate. ONE proposed edge is enough.
  5. If kg_search returns ZERO real candidates (only the orphan itself, or all unrelated): the entity truly offers no retrieval value.
       → mempalace_kg_delete_entity(entity=<memory_ids[0]>, agent="memory_gardener")

  GOOD link example:
    Orphan: "user_session_cache" with description about per-request cache.
    kg_search returns "request_lifecycle" entity (related domain, no edge yet).
    → propose_edge_candidate(from_entity="user_session_cache", to_entity="request_lifecycle", weight=0.7)

  GOOD delete example:
    Orphan: stub entity "tmp_2025_03_12" with description "one-off scratch".
    kg_search returns nothing related.
    → kg_delete_entity(entity="tmp_2025_03_12", agent="memory_gardener")
"""


_TASK_GENERIC_SUMMARY = """\
YOUR TASK -- generic_summary:
  GROUND BEFORE WRITING. Never invent a description from thin context.

  Target shape (Adrian's design lock 2026-04-25): every summary is a
  STRUCTURED DICT, no regex, no auto-derive:

      {"what":  "<noun phrase naming the entity, ≥5 chars>",
       "why":   "<purpose / role / claim clause, ≥15 chars>",
       "scope": "<temporal/domain qualifier, ≤100 chars, OPTIONAL>"}

  Validation is field-level + length-bounded. Rendered prose form
  ≤280 chars. Strings on writes are rejected with a migration message;
  pass the dict.

  Field intent (each must EARN its place in the dict):
    WHAT  -- a discriminative noun phrase that names the entity. NOT
             a bare type name like 'project'/'tool', NOT a keyword
             concatenation. Must distinguish this entity from peers.
    WHY   -- a real purpose / role / claim clause. NEW information
             beyond restating WHAT. Test: replace WHAT with 'X' --
             does WHY still make sense as an explanation?
    SCOPE -- temporal/domain qualifier. Only when the entity has a
             clear scope; OMIT when universal/timeless. Don't pad.

  Examples of GOOD summaries (the target shape):
    {"what": "InjectionGate (post-retrieval relevance filter)",
     "why": "runtime gate that filters retrieved memories before "
            "injection via Haiku tool-use; emits quality flags",
     "scope": "one instance per palace process"}
    {"what": "intent.py",
     "why": "orchestrates declare_intent slot validation and "
            "finalize_intent feedback coverage; central glue "
            "between hooks and the gate"}
    {"what": "Adrian Rivero",
     "why": "DSpot tech lead and project owner of mempalace; pushes "
            "back on speculation, favours load-bearing demos over plans"}
    {"what": "data_migrations stamp table pattern",
     "why": "marks one-shot Python data migrations as applied so "
            "subsequent KG inits short-circuit O(1) instead of "
            "iterating every row",
     "scope": "mempalace internals"}

  Examples of BAD summaries (REJECT and REWRITE):
    {"what": "Adrian", "why": "the project"}
        ← stub WHY (<15 chars), restates nothing useful
    {"what": "x", "why": "stores stuff somewhere"}
        ← stub WHAT (<5 chars)
    {"what": "summary contract",
     "why": "what why scope dict",
     "scope": "dict"}
        ← KEYWORD SOUP. WHY is just three labels jammed together
          ("what", "why", "scope") not a clause. SCOPE is a single
          token with no qualifying meaning. WHAT is generic.
          Rewrite: {"what": "structured summary contract",
                    "why": "every entity write requires {what>=5,
                           why>=15, scope?<=100} dict; rendered prose
                           <=280 chars; field-level validation; no
                           auto-derive; no regex",
                    "scope": "Adrian design lock 2026-04-25"}
    {"what": "the project", "why": "is a project"}
        ← bare type name + tautological WHY. Rewrite WHAT to be a
          discriminative noun phrase; WHY must claim something.
    "InjectionGate -- runtime gate that ..."
        ← string form, retired -- always pass a dict

  Procedure:
  1. kg_query(entity=<memory_ids[0]>) to see kind + current description.
  2. kg_search(context={queries: ["<entity name in plain English>", "<topic from flag.detail>"], keywords: ["<2-3 exact terms>"], summary: {"what": "<entity>", "why": "gardener-driven retrieval to ground new description for generic_summary flag"}}, limit=5) to retrieve grounding evidence.
  3. If kg_search returns ZERO hits OR all hits are themselves stub-length descriptions: defer with reason='no grounding evidence'. Do NOT fabricate.
  4. Compose a NEW description dict using ONLY claims attested in the retrieved evidence. WHAT is the entity name; WHY is the role/purpose clause; SCOPE is an optional qualifier. Better short and true than long and guessed.
  5. Apply UNIFORMLY across all kinds (Adrian's design lock 2026-04-26):
      mempalace_kg_update_entity(
          entity=<memory_ids[0]>,
          summary={"what": ..., "why": ..., "scope": ...},
          context={...with summary dict...},
          agent="memory_gardener"
      )

      The summary dict is written to properties.summary; the entity's
      content (long-prose body) is NEVER touched by this resolution.
      That decouples the embedding cache
      (driven by content) from summary updates, so summary refinement
      cannot break cosine retrieval and there is NO reason to delete
      a record over a generic summary. Adrian's audit 2026-04-26
      identified the prior delete-record path as a data-loss bug:
      records carry the actual prose data (diary entries, learnings,
      result memos, etc.); deleting them to "fix" a vague summary
      destroys load-bearing content. The path has been retired.

      kg_update_entity runs coerce_summary_for_persist on the summary
      dict; if it rejects ('what' or 'why' too short, scope too long),
      tighten the offending field and retry. Strings are rejected
      outright.

  RETRY-ON-LENGTH (NO CAP -- keep going until the error type changes):
    If kg_update_entity returns 'rendered summary exceeds N chars (X given)':
      → DO NOT defer. The shape is correct; only the length is wrong.
      → Compose a SHORTER dict (target ≤280 chars rendered; rough budget:
        what ≤30 chars, why ≤140 chars, scope ≤80 chars or omit) and call
        kg_update_entity AGAIN. Same intent, same target, tighter prose.
      → Each retry should cut ~20% off whichever field is heaviest. Drop
        scope first, then trim why, then trim what.
      → Keep retrying as long as the error message is still
        'rendered summary exceeds N chars'. There is NO retry cap for length
        errors -- only stop when the error MESSAGE changes to something
        else (e.g. 'what too short', 'invalid type', a totally different
        rejection). At that point switch tactic per the rule below.
    If 'what' or 'why' are too SHORT (under their min): tighten by ADDING
    a few words, not by trimming. Retry once.
    Strings are NEVER acceptable -- no retry on string-form rejection;
    re-read this prompt and resend as a dict.
"""


_TASK_EDGE_CANDIDATE = """\
YOUR TASK -- edge_candidate:
  VERIFY BOTH ENDPOINTS before seeding. Phantom-target seeds clog the link-author queue forever, and self-edges (X, X) are pure noise.

  PROCEDURE:
  1. If memory_ids[0] == memory_ids[1]: defer with reason='self-edge'. Don't seed.
  2. kg_query(entity=memory_ids[0]) -- confirm exists. If 'Not found', kg_search the id; if still missing, defer.
  3. kg_query(entity=memory_ids[1]) -- confirm exists. Same recovery.
  4. mempalace_propose_edge_candidate(from_entity=memory_ids[0], to_entity=memory_ids[1], weight=0.7)
  NEVER call kg_add. NEVER declare a predicate. The link-author jury picks the predicate.

  GOOD example:
    Flag: memory_ids=["intent.py", "mempalace"], detail="intent.py belongs to mempalace project"
    Both exist (kg_query OK).
    → propose_edge_candidate(from_entity="intent.py", to_entity="mempalace", weight=0.7)

  DEFER example:
    from_entity="intent.py", to_entity="the intent system" -- second is a phrase, not a declared entity.
    → defer with reason='to_entity not declared'.
"""


_TASK_UNLINKED_ENTITY = """\
YOUR TASK -- unlinked_entity:
  flag.detail names a concept that should be linked to memory_ids[0] but the concept may not be a declared entity yet (e.g. a literature citation like "Zhao 2025", a code symbol, or an external system).

  PROCEDURE:
  1. Read flag.detail and extract the candidate concept name.
  2. kg_search(context={queries: [<concept name>, <flag.detail topic>], keywords: [<concept name>, <2 related terms>], summary: {what: <concept>, why: 'gardener unlinked-entity verify'}}, limit=5)
  3. If kg_search returns NO matching entity for the concept:
       → defer with reason='target entity not declared'.
  4. Otherwise:
       → mempalace_propose_edge_candidate(from_entity=memory_ids[0], to_entity=<canonical id from search>, weight=0.7)

  GOOD example:
    flag.detail: 'mentions "Zhao 2025" -- should link to citation entity if declared'
    kg_search → returns entity id "citation_zhao_2025"
    → propose_edge_candidate(from_entity=memory_ids[0], to_entity="citation_zhao_2025", weight=0.7)

  DEFER example:
    flag.detail: 'mentions "the new auth flow" -- concept not yet declared'
    kg_search → 0 entity hits.
    → defer with reason='target entity not declared'.
"""


_TASK_OP_CLUSTER_TEMPLATIZABLE = """\
YOUR TASK -- op_cluster_templatizable:
  GROUND BEFORE WRITING. Never invent recipe content not attested in the cluster's op rows.

  PROCEDURE:
  1. SHOULD kg_query at least one op_id from memory_ids to read its args_summary + reason. For a positive cluster (flag.detail = "positive"), sample one of the higher-quality ones; for a negative cluster (flag.detail = "negative"), sample at least one to confirm the failure mode is real.
  2. If the queried op rows have empty / stub-length args_summary AND empty reason: defer with reason='no grounding evidence in op cluster'. Do NOT fabricate a recipe over rated-but-unannotated ops.
  3. Compose:
      title:        short noun phrase naming the pattern, <=80 chars, anchored on what the queried ops actually did.
      when_to_use:  1-2 sentences on which context triggers this -- drawn from the args_summary / reason fields.
      recipe:       the reusable pattern in prose -- what TO DO for "positive" detail, what to AVOID for "negative".
      failure_modes: REQUIRED for "negative" detail (2-4 concrete entries, each grounded in a queried op's reason). OMIT for "positive".
  4. mempalace_synthesize_operation_template(
         op_ids=<memory_ids array from the flag VERBATIM>,
         title=..., when_to_use=..., recipe=..., failure_modes=[...]
     )
  Do NOT call kg_add or kg_declare_entity -- the shim handles record + templatizes edges idempotently (deterministic id from sorted op_ids; re-runs upsert).

  GOOD positive example:
    flag.detail: "positive", memory_ids=["op_edit_abc", "op_edit_def", "op_edit_ghi"]
    kg_query op_edit_abc → args_summary="Edit on test_*.py files; replaced old assert"
                            reason="Edit landed first try, test went green."
    → synthesize_operation_template(
          op_ids=memory_ids,
          title="Edit-then-rerun test assertion migration",
          when_to_use="When migrating test assertions to a new contract; the edit is local and the test is the verification step.",
          recipe="Read context around the assertion; Edit replacing old string; immediately rerun the test file. Don't batch multiple sites without re-running between them.",
          failure_modes=[]   # OMITTED for positive
      )

  GOOD negative example:
    flag.detail: "negative", memory_ids=["op_bash_xxx", "op_bash_yyy"]
    kg_query op_bash_xxx → args_summary="git push --force"
                            reason="Force-pushed over teammate's commit."
    → synthesize_operation_template(
          op_ids=memory_ids,
          title="Avoid force-push on shared branches",
          when_to_use="When tempted to use --force or --force-with-lease on a branch other people pull from.",
          recipe="If history rewrite is truly needed, branch off and PR. Otherwise use a new commit.",
          failure_modes=[
            "Force-pushed over teammate's commit on main (op_bash_xxx)",
            "Force-pushed without --force-with-lease, lost stash work (op_bash_yyy)"
          ]
      )

  DEFER example:
    Both queried ops have args_summary="" and reason="".
    → defer with reason='no grounding evidence in op cluster'. NEVER fabricate.
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


_TASK_BY_KIND: dict[str, str] = {
    "duplicate_pair": _TASK_DUPLICATE_PAIR,
    "contradiction_pair": _TASK_CONTRADICTION_PAIR,
    "stale": _TASK_STALE,
    "orphan": _TASK_ORPHAN,
    "generic_summary": _TASK_GENERIC_SUMMARY,
    "edge_candidate": _TASK_EDGE_CANDIDATE,
    "unlinked_entity": _TASK_UNLINKED_ENTITY,
    "op_cluster_templatizable": _TASK_OP_CLUSTER_TEMPLATIZABLE,
}


def _select_prompt(flag_kind: str) -> str:
    """Return the focused system prompt for this flag kind.

    Each kind sees only the rules it needs -- shared preamble + that
    kind's task block. Cuts ~60-70% of irrelevant token surface vs.
    the legacy single-prompt approach (2026-04-25 refactor).

    Raises KeyError on unknown kinds -- fail loud, not silent.
    """
    try:
        return _PROMPTS_BY_KIND[flag_kind]
    except KeyError:
        raise KeyError(
            f"Unknown flag_kind {flag_kind!r}; valid: {sorted(_PROMPTS_BY_KIND)}"
        ) from None


def _select_prompt_blocks(flag_kind: str) -> list[dict]:
    """Return the system prompt as Anthropic content-blocks with prompt-cache
    enabled on the shared preamble.

    The preamble is identical across all 8 kinds, so marking it
    cache_control:ephemeral lets Haiku read it from cache on every call after
    the first (~5min TTL). Per-kind task blocks vary and stay uncached.
    Mirrors the injection_gate caching pattern (2026-04-24).
    """
    try:
        task = _TASK_BY_KIND[flag_kind]
    except KeyError:
        raise KeyError(f"Unknown flag_kind {flag_kind!r}; valid: {sorted(_TASK_BY_KIND)}") from None
    return [
        {
            "type": "text",
            "text": _SHARED_PREAMBLE,
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": task},
    ]


# Cached tool surface -- last tool carries cache_control:ephemeral so the entire
# 8-tool schema list caches together with the shared preamble. Built once at
# module load; safe to share across calls (Anthropic SDK reads, doesn't mutate).
_TOOL_SCHEMAS_CACHED: list[dict] = list(_TOOL_SCHEMAS)
if _TOOL_SCHEMAS_CACHED:
    _TOOL_SCHEMAS_CACHED[-1] = {
        **_TOOL_SCHEMAS_CACHED[-1],
        "cache_control": {"type": "ephemeral"},
    }


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
    # Anthropic prompt-cache telemetry (2026-04-26): summed across all
    # iterations of this loop. cache_read = bytes served from cache (hit);
    # cache_creation = bytes written to cache (miss + populate).
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


def _run_anthropic_loop(
    system_blocks: list[dict] | str,
    user_prompt: str,
    model: str,
    max_iters: int = _MAX_TOOL_LOOP_ITERS,
) -> LoopResult:
    """Classic Anthropic tool-use loop. No SDK wrapping, no MCP wire
    protocol -- tool calls dispatch to mempalace tool_* functions via
    _exec_tool directly.

    system_blocks: pass a list-of-content-blocks (preferred -- enables
    prompt caching via cache_control on the shared preamble) or a plain
    string (no caching -- kept for back-compat with ad-hoc callers).
    """
    result = LoopResult()
    try:
        import anthropic
    except ImportError as e:
        result.api_error = f"anthropic SDK not installed: {e}"
        return result

    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY from env
    messages: list[dict] = [{"role": "user", "content": user_prompt}]
    # Use cached tools when system is in block form (caching path); fall
    # back to plain tool list for the string-system back-compat path.
    tools_arg = _TOOL_SCHEMAS_CACHED if isinstance(system_blocks, list) else _TOOL_SCHEMAS

    for iteration in range(max_iters):
        result.iterations = iteration + 1
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_blocks,
                tools=tools_arg,
                messages=messages,
            )
        except anthropic.APIError as e:
            result.api_error = f"{type(e).__name__}: {e}"
            break
        except Exception as e:
            result.api_error = f"unexpected {type(e).__name__}: {e}"
            break

        # Capture prompt-cache usage (Anthropic exposes both fields on usage;
        # treat missing as 0 so older SDK shapes don't crash). Sum across
        # iterations because tool-use loops can spin multiple round trips.
        usage = getattr(resp, "usage", None)
        if usage is not None:
            result.cache_creation_input_tokens += int(
                getattr(usage, "cache_creation_input_tokens", 0) or 0
            )
            result.cache_read_input_tokens += int(getattr(usage, "cache_read_input_tokens", 0) or 0)

        result.stop_reason = resp.stop_reason or ""
        # Collect assistant content + tool_use blocks.
        assistant_blocks: list[dict] = []
        pending_tool_uses: list[tuple[str, str, dict]] = []  # (id, name, input)
        for block in resp.content:
            # anthropic SDK types -- block is either TextBlock or ToolUseBlock.
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
        desc = details.get("content") or details.get("content") or ""
        if not desc:
            desc = tc.result.get("content") or ""
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

    batch_size is clamped to 1 -- keeps resolution mapping trivial
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
            # Not in pending -- maybe attempted_count already >= 3 or
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
        loop = _run_anthropic_loop(_select_prompt_blocks(flag_kind), user_prompt, model)
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
    # Single-process guard: if a live gardener subprocess already holds
    # the kernel-managed advisory lock, skip this spawn. Avoids two spawns
    # racing on the same flag and double-writing kg_update_entity
    # descriptions. Kernel-released on any process death (SIGKILL, crash,
    # etc.) -- no stale-lock false-positives. _is_lock_held does its own
    # error handling and fails open on disk trouble.
    if _is_lock_held():
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

    # Single-process serialization. Acquire the gardener lockfile on
    # entry; if another gardener subprocess is already running, exit
    # cleanly with code 0. Release on every exit path via try/finally
    # plus atexit (defence in depth against SIGTERM/uncaught exceptions).
    if not _try_acquire_lock():
        print(
            "[memory_gardener] another gardener is already running; exiting",
            file=sys.stderr,
        )
        return 0
    import atexit as _atexit

    _atexit.register(_release_lock)

    try:
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
    finally:
        _release_lock()
