"""
conflict_resolver_auto.py -- async Haiku-driven conflict resolution.

v3.7.19 Slice 1 (Adrian directive 2026-05-17): the main-agent
mempalace_resolve_conflicts path used to block every mempalace
mutating tool until the agent walked the conflict list manually.
This module moves that judgment off the critical path to a
background Haiku call -- same pattern as v3.5.0 feedback_auto
(memory + operation rating) and v3.7.4 state_judge.

SLICE 1 SCOPE -- OBSERVATION ONLY:
- Every conflict minted at entity_gate / tool_mutate / knowledge_graph
  sites is submitted to this resolver in a background thread.
- Haiku reads the conflict shape (existing_id, new_id/new_name,
  similarity, conflict_type, past_resolution if any) and recommends
  one of: invalidate, merge, keep, skip -- with a one-sentence reason.
- The recommendation is appended to
  ~/.mempalace/hook_state/conflict_resolver_log.jsonl and surfaced via
  mempalace_bg_status. NO kg writes happen in this slice; the manual
  resolve_conflicts handler still owns persistence. This lets us audit
  Haiku quality on real data before flipping to active resolution.

LATER SLICES:
- v3.7.20: persist resolutions automatically; drop the blocking gate.
- v3.7.21: low-confidence escalation flag (Haiku punts -> agent sees
  pending conflict via existing resolve_conflicts path).

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
    "abstain -- You can't tell from the conflict shape alone. The "
    "main agent will be escalated this one via the existing "
    "mempalace_resolve_conflicts MCP path. Reserve for genuine "
    "ambiguity (e.g. content not loaded, schema_id mismatch, past "
    "resolutions disagree).\n\n"
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


def _log_result(result: ConflictResolverResult, batch: ConflictResolverInput) -> None:
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
                "error": result.error or "",
                "applied": False,  # v3.7.19 slice 1: observation-only
                "slice": "v3.7.19-observation-only",
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
    """Worker body: call Haiku, log telemetry. No kg writes this slice."""
    try:
        result = _call_haiku(batch)
        _log_result(result, batch)
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

    Returns immediately (sub-millisecond). The Haiku call + telemetry
    write happen on the worker thread. The pending_conflicts list and
    the manual mempalace_resolve_conflicts handler are UNCHANGED in
    this slice -- this resolver only observes and logs.

    Called from entity_gate / tool_mutate / knowledge_graph sites
    immediately after they append a new conflict to
    _STATE.pending_conflicts.
    """
    if _disabled():
        return
    if not isinstance(conflict, dict) or not conflict.get("id"):
        return
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
