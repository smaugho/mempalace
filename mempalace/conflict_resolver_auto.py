"""
conflict_resolver_auto.py -- async Haiku-driven conflict resolution.

v3.7.20 FULL CUTOVER (Adrian directive 2026-05-17): conflicts are
resolved by Haiku in the background, wherever and whenever they
appear. The mempalace_resolve_conflicts MCP tool and its main-agent
handler are GONE. There is no escape hatch -- the main agent never
sees conflicts; the bg thread owns them end-to-end.

Same pattern as v3.5.0 feedback_auto (memory + operation rating
moved to async Haiku) and v3.7.4 state_judge (judge moved to
background, results delivered on next op).

PIPELINE:
1. entity_gate / tool_mutate / knowledge_graph sites mint a conflict
   and append it to _STATE.pending_conflicts (kept for in-memory
   visibility + restart recovery via on-disk persistence).
2. The mint site immediately calls submit_conflict() here; the
   conflict dict + agent / intent_type / session_id are wrapped in a
   ConflictResolverInput and handed to a single-worker ThreadPool.
3. The worker calls Haiku with the cached system prompt + decision
   rules. Haiku returns one of: invalidate | merge | keep | skip |
   abstain with confidence 0.0-1.0.
4. _apply_resolution persists the decision via the same kg primitives
   the legacy tool_resolve_conflicts handler used (kg.invalidate,
   tool_kg_merge_entities, record_conflict_resolution). On success
   the entry is popped from _STATE.pending_conflicts and the active
   intent is persisted so the change survives a restart.
5. Every decision is logged to conflict_resolver_log.jsonl with
   applied=True|False so mempalace_bg_status can surface the audit
   trail.

NO BLOCKING. The agent's mutating tools NEVER pause for conflicts --
the conflict mint sites return success and the resolver catches up
asynchronously.

The module is feature-flagged via
MEMPALACE_CONFLICT_RESOLVER_AUTO_DISABLED=1 so tests that mock the
agent / KG can disable it cleanly (same convention as feedback_auto).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# -- env-flag plumbing -----------------------------------------------


def _disabled() -> bool:
    """True when MEMPALACE_CONFLICT_RESOLVER_AUTO_DISABLED=1.

    Tests that don't want background Haiku resolver noise set this to
    '1' via monkeypatch.setenv. Production leaves it unset.
    """
    return os.environ.get("MEMPALACE_CONFLICT_RESOLVER_AUTO_DISABLED", "").strip() == "1"


# -- data shapes -----------------------------------------------------


@dataclass
class ConflictResolverInput:
    """One Haiku-resolver job's payload."""

    conflict: dict
    agent: str = ""
    intent_type: str = ""
    session_id: str = ""


@dataclass
class ConflictResolverResult:
    """Haiku-resolver output, persisted by the caller via telemetry."""

    conflict_id: str
    recommended_action: str  # invalidate | merge | keep | skip | abstain
    reason: str
    merged_content: str = ""  # only meaningful for action='merge'
    into: str = ""  # only meaningful for action='merge'
    confidence: float = 0.0  # 0.0-1.0; <0.5 ~= abstain
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    elapsed_ms: float = 0.0
    error: str | None = None


# -- prompt + tool schema --------------------------------------------


_SYSTEM_PROMPT = (
    "You are a conflict resolver for a knowledge-graph-backed agent. "
    "The agent's mempalace_kg_declare_entity / mempalace_kg_add call "
    "just produced a CONFLICT -- a similarity-duplicate, edge "
    "contradiction, or merge candidate. Your job: read the conflict "
    "shape and recommend ONE action, with a one-sentence reason.\n\n"
    "## Actions\n\n"
    "invalidate -- The existing item is genuinely stale or wrong; "
    "mark it status='invalidated' so future retrieval skips it. Use "
    "when the new item is a clear correction of the old.\n\n"
    "merge -- The two items describe the SAME thing and should be "
    "collapsed into one canonical entity. Pick the better-id of the "
    "two as 'into' (target) and supply merged_content prose that "
    "preserves ALL unique info from both sides. Use for true "
    "duplicates where neither side alone is complete.\n\n"
    "keep -- Both items are valid and DISTINCT despite the surface "
    "similarity (e.g. two records that describe different aspects of "
    "the same person; two task entities at different phases). No kg "
    "writes; both rows stay current. Use as the default when in "
    "doubt -- 'keep' is the only safe action when you can't be sure "
    "they collapse.\n\n"
    "skip -- The NEW item is the duplicate / mistake; drop it (mark "
    "it invalidated). Use when the agent just re-declared something "
    "the existing entity already covers and there's nothing to add.\n\n"
    "abstain -- You can't tell from the conflict shape alone. There "
    "is NO main-agent escalation path in v3.7.20+ (resolve_conflicts "
    "MCP tool was retired); abstain just logs the conflict to "
    "conflict_resolver_log.jsonl with applied=False and leaves both "
    "rows current (same effect as 'keep' on the data side). Reserve "
    "for genuine ambiguity (e.g. content not loaded, schema_id "
    "mismatch, past resolutions disagree); when ambiguous you should "
    "prefer 'keep' over 'abstain' since 'keep' carries the same "
    "no-data-loss guarantee but expresses the decision affirmatively.\n\n"
    "## Decision rules\n\n"
    "1. similarity < 0.85 -> 'keep' almost always. Surface "
    "similarity doesn't imply semantic duplication.\n"
    "2. similarity >= 0.95 + conflict_type='entity_collision' or "
    "'memory_duplicate' -> 'merge' if both sides carry distinct "
    "info, otherwise 'skip'.\n"
    "3. conflict_type='edge_contradiction' -> 'invalidate' the "
    "OLD edge when the new one is a clearer / more recent claim; "
    "otherwise 'keep' (both edges can coexist if one isn't truly "
    "wrong).\n"
    "4. past_resolution exists -> follow the precedent unless the "
    "shape obviously diverged.\n"
    "5. If unsure, 'keep' beats 'merge' beats 'invalidate'. The "
    "agent retracts wrong 'keep' decisions cheaply; wrong "
    "'invalidate' loses data.\n\n"
    "Output via the resolve_conflict tool. One call per conflict. "
    "confidence is your subjective 0.0-1.0 score; below 0.5 should "
    "use action='abstain'.\n"
)


_RESOLVE_TOOL_SCHEMA = {
    "name": "resolve_conflict",
    "description": "Emit your resolution recommendation for one conflict.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["invalidate", "merge", "keep", "skip", "abstain"],
                "description": "Resolution action. See system prompt.",
            },
            "reason": {
                "type": "string",
                "description": (
                    "One sentence explaining WHY this action. Reference "
                    "the specific similarity / conflict_type / past "
                    "resolution that drove the call."
                ),
            },
            "confidence": {
                "type": "number",
                "description": (
                    "0.0-1.0 subjective confidence. < 0.5 should "
                    "almost always pair with action='abstain'."
                ),
            },
            "into": {
                "type": "string",
                "description": "Target id for action='merge'. Empty otherwise.",
            },
            "merged_content": {
                "type": "string",
                "description": (
                    "Combined description preserving ALL unique info "
                    "from both items. Required for action='merge'. Empty "
                    "otherwise."
                ),
            },
        },
        "required": ["action", "reason", "confidence"],
    },
}


def _conflict_user_prompt(conflict: dict) -> str:
    """Render one conflict for the Haiku call."""
    parts = [
        "Resolve this conflict:",
        "",
        json.dumps(conflict, indent=2, default=str),
    ]
    return "\n".join(parts)


# -- Haiku call ------------------------------------------------------


_HAIKU_MODEL = os.environ.get("MEMPALACE_HAIKU_MODEL", "claude-haiku-4-5")
_HAIKU_MAX_TOKENS = 800


def _call_haiku(batch: ConflictResolverInput) -> ConflictResolverResult:
    """Run one Haiku tool-use call for a single conflict."""
    conflict_id = (batch.conflict or {}).get("id", "?")
    t0 = time.perf_counter()
    try:
        import anthropic  # local import: keep module import-light
    except ImportError as exc:
        return ConflictResolverResult(
            conflict_id=conflict_id,
            recommended_action="abstain",
            reason="anthropic SDK not installed; falling back to abstain",
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            error=f"ImportError: {exc}",
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        try:
            from .auto_author import _ensure_env_loaded

            _ensure_env_loaded()
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        except Exception:
            pass
    if not api_key:
        return ConflictResolverResult(
            conflict_id=conflict_id,
            recommended_action="abstain",
            reason="ANTHROPIC_API_KEY missing; cannot rate this conflict",
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            error="no_api_key",
        )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=_HAIKU_MODEL,
            max_tokens=_HAIKU_MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[_RESOLVE_TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": "resolve_conflict"},
            messages=[
                {
                    "role": "user",
                    "content": _conflict_user_prompt(batch.conflict),
                }
            ],
        )
    except Exception as exc:
        return ConflictResolverResult(
            conflict_id=conflict_id,
            recommended_action="abstain",
            reason=f"Haiku call failed: {exc}",
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            error=str(exc),
        )

    # Parse the tool_use block
    block = None
    for b in getattr(resp, "content", []) or []:
        if getattr(b, "type", "") == "tool_use" and getattr(b, "name", "") == "resolve_conflict":
            block = b
            break
    if block is None:
        return ConflictResolverResult(
            conflict_id=conflict_id,
            recommended_action="abstain",
            reason="Haiku returned no resolve_conflict tool_use block",
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            error="no_tool_use",
        )

    payload = getattr(block, "input", {}) or {}
    usage = getattr(resp, "usage", None)
    return ConflictResolverResult(
        conflict_id=conflict_id,
        recommended_action=str(payload.get("action", "abstain")),
        reason=str(payload.get("reason", "")),
        merged_content=str(payload.get("merged_content", "")),
        into=str(payload.get("into", "")),
        confidence=float(payload.get("confidence", 0.0) or 0.0),
        tokens_in=int(getattr(usage, "input_tokens", 0) or 0),
        tokens_out=int(getattr(usage, "output_tokens", 0) or 0),
        cache_read_input_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        cache_creation_input_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        elapsed_ms=(time.perf_counter() - t0) * 1000,
    )


# -- telemetry -------------------------------------------------------


def _apply_resolution(
    result: ConflictResolverResult, batch: ConflictResolverInput
) -> tuple[bool, str | None]:
    """Persist Haiku's decision via kg primitives.

    Returns (applied, error). applied=True iff a kg-write fired or the
    conflict entry was removed from pending_conflicts (keep is also a
    valid no-write resolution). error carries the exception message on
    failure; the caller logs it but never raises.

    Mirrors the action dispatch of the legacy tool_resolve_conflicts
    handler (removed in v3.7.20). Only the four actionable verbs land
    here -- 'abstain' is treated as 'keep' (no kg writes) so the
    pending_conflicts entry can still be popped; otherwise the list
    would grow unbounded with un-actioned items.
    """
    conflict = batch.conflict or {}
    conflict_id = result.conflict_id
    conflict_type = conflict.get("conflict_type", "")
    existing_id = conflict.get("existing_id", "")
    new_id = conflict.get("new_id", "") or conflict.get("new_name", "")
    action = result.recommended_action

    try:
        from .mcp_server import _STATE
    except Exception as exc:
        return False, f"mcp_server import failed: {exc}"

    err: str | None = None
    try:
        if action == "invalidate":
            if conflict_type == "edge_contradiction":
                _STATE.kg.invalidate(
                    conflict.get("existing_subject", ""),
                    conflict.get("existing_predicate", ""),
                    conflict.get("existing_object", ""),
                )
            elif conflict_type in (
                "entity_collision",
                "entity_duplicate",
                "memory_duplicate",
            ):
                try:
                    conn = _STATE.kg._conn()
                    conn.execute(
                        "UPDATE entities SET status='invalidated' WHERE id=?",
                        (existing_id,),
                    )
                    conn.commit()
                except Exception:
                    pass
        elif action == "merge":
            # default into=existing unless Haiku specified otherwise
            into = result.into or existing_id
            source = new_id if into == existing_id else existing_id
            if conflict_type in (
                "entity_collision",
                "entity_duplicate",
                "memory_duplicate",
            ):
                from .mcp_server import tool_kg_merge_entities

                merge_what = result.into or existing_id or "merged"
                merge_why = result.merged_content or f"Auto-merged by Haiku: {result.reason[:200]}"
                tool_kg_merge_entities(
                    source=source,
                    target=into,
                    summary={"what": merge_what, "why": merge_why},
                    agent=batch.agent or "ga_agent",
                )
        elif action == "skip":
            # v3.7.22 fix (audit FINDING #11): skip semantics per the
            # Haiku system prompt is "drop the NEW item, mark it
            # invalidated." Pre-v3.7.22 only the edge_contradiction
            # branch honored this -- for entity_collision /
            # entity_duplicate / memory_duplicate, skip was a silent
            # no-op while the resolver still logged applied=True. The
            # dupe stayed current and accumulated; the audit trail
            # lied. Now skip mirrors invalidate's dispatch but targets
            # the NEW id (vs. invalidate which targets the existing id).
            if conflict_type == "edge_contradiction":
                try:
                    _STATE.kg.invalidate(
                        conflict.get("new_subject", ""),
                        conflict.get("new_predicate", ""),
                        conflict.get("new_object", ""),
                    )
                except Exception:
                    pass
            elif conflict_type in (
                "entity_collision",
                "entity_duplicate",
                "memory_duplicate",
            ):
                try:
                    conn = _STATE.kg._conn()
                    conn.execute(
                        "UPDATE entities SET status='invalidated' WHERE id=?",
                        (new_id,),
                    )
                    conn.commit()
                except Exception:
                    pass
        # 'keep' and 'abstain' both fall through with no kg writes.

        # Audit row -- so future sessions can see what Haiku decided.
        try:
            _STATE.kg.record_conflict_resolution(
                conflict_id=conflict_id,
                conflict_type=conflict_type,
                action=action,
                reason=result.reason[:1000],
                existing_id=existing_id,
                new_id=new_id,
                agent=batch.agent or "ga_agent_haiku_bg",
                intent_type=batch.intent_type or "",
            )
        except Exception:
            pass

        # Pop the resolved entry so pending_conflicts shrinks.
        try:
            if _STATE.pending_conflicts:
                _STATE.pending_conflicts = [
                    c
                    for c in _STATE.pending_conflicts
                    if isinstance(c, dict) and c.get("id") != conflict_id
                ]
                # Best-effort disk persist so a restart doesn't
                # resurrect the entry.
                try:
                    from . import intent as _intent

                    _intent._persist_active_intent()
                except Exception:
                    pass
        except Exception:
            pass
    except Exception as exc:
        err = str(exc)

    return (err is None), err


def _log_result(
    result: ConflictResolverResult,
    batch: ConflictResolverInput,
    applied: bool,
    apply_error: str | None,
) -> None:
    """Append the resolver decision to conflict_resolver_log.jsonl."""
    try:
        from .mcp_server import _telemetry_append_jsonl as _tel

        _tel(
            "conflict_resolver_log.jsonl",
            {
                "conflict_id": result.conflict_id,
                "agent": batch.agent,
                "intent_type": batch.intent_type,
                "session_id": batch.session_id,
                "conflict_type": (batch.conflict or {}).get("conflict_type", ""),
                "existing_id": (batch.conflict or {}).get("existing_id", ""),
                "new_id": (batch.conflict or {}).get("new_id", "")
                or (batch.conflict or {}).get("new_name", ""),
                "similarity": (batch.conflict or {}).get("similarity", None),
                "recommended_action": result.recommended_action,
                "reason": result.reason,
                "confidence": result.confidence,
                "into": result.into,
                "merged_content_len": len(result.merged_content),
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "cache_read": result.cache_read_input_tokens,
                "cache_creation": result.cache_creation_input_tokens,
                "elapsed_ms": round(result.elapsed_ms, 2),
                "haiku_error": result.error or "",
                "applied": applied,
                "apply_error": apply_error or "",
                "slice": "v3.7.20-active",
            },
        )
    except Exception:
        pass


# -- threadpool dispatcher -------------------------------------------


_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def _get_executor() -> ThreadPoolExecutor:
    """Lazy-init the singleton ThreadPoolExecutor.

    One worker is enough: conflicts are bursty but not high-volume,
    and serializing them keeps the Haiku 5-min ephemeral cache prefix
    hot (sequential calls within an intent share the same system
    block, max cache hit rate, ~10x cheaper).
    """
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mempalace-conflict-resolver"
            )
        return _executor


def _run(batch: ConflictResolverInput) -> None:
    """Worker body: call Haiku, persist via kg primitives, log telemetry.

    v3.7.20: Haiku now owns conflict resolution end-to-end. The
    pending_conflicts entry is popped on successful application; the
    conflict_resolver_log.jsonl audit row records what was decided +
    whether the apply step succeeded.
    """
    try:
        result = _call_haiku(batch)
        applied = False
        apply_error: str | None = None
        if result.error is None:
            applied, apply_error = _apply_resolution(result, batch)
        else:
            apply_error = "skipped (Haiku error)"
        _log_result(result, batch, applied, apply_error)
    except Exception as exc:  # pragma: no cover -- last-resort guard
        logger.info(
            "conflict_resolver_auto: _run crashed conflict_id=%s err=%s",
            (batch.conflict or {}).get("id", "?"),
            exc,
        )


def submit_conflict(
    conflict: dict,
    *,
    agent: str = "",
    intent_type: str = "",
    session_id: str = "",
) -> None:
    """Fire-and-forget enqueue of one conflict for background resolution.

    Returns immediately (sub-millisecond). The Haiku call, the apply
    step (kg.invalidate / tool_kg_merge_entities /
    record_conflict_resolution), and the telemetry write all happen on
    the worker thread. On successful apply the conflict entry is
    popped from _STATE.pending_conflicts and the active intent is
    persisted so the resolution survives a restart.

    v3.7.20: the manual mempalace_resolve_conflicts handler was
    removed. Haiku owns every conflict end-to-end; the agent never
    sees them. abstain decisions are logged with applied=False and
    leave both rows current (same data-side effect as 'keep').

    Called from entity_gate / tool_mutate / knowledge_graph sites
    immediately after they append a new conflict to
    _STATE.pending_conflicts.
    """
    if _disabled():
        return
    if not isinstance(conflict, dict) or not conflict.get("id"):
        return

    # FINDING #S (v3.7.33 2026-05-18, Adrian's post-v3.7.32 audit):
    # filter two classes of false-positive conflicts at the boundary
    # so Haiku is never asked to resolve them.
    #
    # CLASS 1 -- view-suffix rows. v3.7.29 added Level-3 ``{eid}__body``
    # views + Level-1 ``__identity`` + Level-4 ``__v0..__vN`` probe
    # views to the records collection. They embed the SAME content
    # as the canonical row at cosine ~1.0. col.query() returns them
    # as duplicate candidates; the upstream dedup site at
    # mcp_server.py:1356 already filters them but this is the
    # belt-and-suspenders boundary check. Skip any conflict where
    # either id carries a view suffix.
    #
    # CLASS 2 -- execution+result twin pairs. finalize_intent
    # deliberately creates a twin pair: an execution entity
    # (e.g. ``wrap_x_2026_05_18``) carrying metadata + edges, and a
    # result memory record (``record_<agent>_result_<base>``) carrying
    # the prose narrative. They are intentionally similar (~0.86-0.92
    # cosine) because the result narrates what the execution did.
    # Pre-v3.7.33 the resolver merged/skipped them, breaking the
    # twin pattern. The new_id pattern is ``record_<agent>_result_<X>``
    # whose existing_id is the bare ``<X>`` (or vice versa); detect
    # this strict suffix match and skip submission.
    import re as _re_finding_s

    _VIEW_SUFFIXES = ("__body", "__identity")
    _VIEW_INDEX_RE = _re_finding_s.compile(r"__v\d+$")
    _RESULT_PREFIX_RE = _re_finding_s.compile(r"^record_[a-z0-9_]+_result_")
    existing_id = str(conflict.get("existing_id", "") or "")
    new_id = str(conflict.get("new_id", "") or "")

    def _is_view_row(eid: str) -> bool:
        return any(eid.endswith(s) for s in _VIEW_SUFFIXES) or bool(_VIEW_INDEX_RE.search(eid))

    if _is_view_row(existing_id) or _is_view_row(new_id):
        return  # CLASS 1: view-suffix false positive

    def _twin_pair(a: str, b: str) -> bool:
        # Return True iff one side is "record_<agent>_result_<base>"
        # and the other is exactly "<base>". The twin is deliberately
        # created by finalize_intent and must not be merged.
        for left, right in ((a, b), (b, a)):
            m = _RESULT_PREFIX_RE.match(left)
            if m and left[m.end() :] == right:
                return True
        return False

    if existing_id and new_id and _twin_pair(existing_id, new_id):
        return  # CLASS 2: execution+result twin pair

    batch = ConflictResolverInput(
        conflict=conflict,
        agent=str(agent or ""),
        intent_type=str(intent_type or ""),
        session_id=str(session_id or ""),
    )
    try:
        executor = _get_executor()
        executor.submit(_run, batch)
    except Exception:
        # Never let executor failures kill the calling op.
        pass
