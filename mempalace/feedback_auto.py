"""
feedback_auto.py -- async Haiku-rated memory & operation feedback.

v3.5.0 (Adrian directive 2026-05-13): the main-agent feedback path
(memory_feedback / operation_ratings supplied at finalize_intent +
extend_feedback to close coverage) is GONE. Every memory the gate
kept and every operation the agent fired this intent is rated post-
hoc by Haiku in a background thread, with the result persisted as
rated_useful / rated_irrelevant edges (rater_kind='haiku_auto') and
performed_well / performed_poorly edges (rater_kind='haiku_auto').

Why
---
* Removes the entire bookkeeping-friction loop that snowballed across
  finalize_intent + extend_feedback cycles (the dominant cause of
  agent token-spend on every intent).
* Background thread = main agent returns immediately on
  finalize_intent; no rating latency on the response path.
* Haiku is ~10x cheaper than Opus/Sonnet for input; with the cached
  system prefix (rubric + tool schema + intent context) the
  effective per-batch cost drops to ~$0.10/M.
* Rating quality is competitive with main-agent ratings -- the agent
  used to inflate ratings to clear coverage; Haiku rates from
  observation only.

Design
------
* One Haiku call per OPERATION (rates the operation itself + the
  memories it surfaced).
* One Haiku call per SEARCH (rates each result memory).
* Background ThreadPoolExecutor (single worker per process) -- jobs
  drain after process exit if not done; failures log to telemetry
  but never raise (best-effort).
* Cache_control on the system block (rubric + tool schema) so the
  second+ batch of a finalize hits the 5-min ephemeral cache.
* Persistence: kg.record_feedback() for memories (rated_useful /
  rated_irrelevant); kg.record_operation_rating() for op ratings.

The module is feature-flagged via MEMPALACE_FEEDBACK_AUTO_DISABLED=1
so test suites that mock the agent / KG can disable it cleanly.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── env-flag plumbing ────────────────────────────────────────────────


def _disabled() -> bool:
    """True when MEMPALACE_FEEDBACK_AUTO_DISABLED=1.

    Tests that don't want background Haiku rating noise (most unit
    tests, every fake-client integration test) set this to '1' via
    monkeypatch.setenv. Production leaves it unset.
    """
    return os.environ.get("MEMPALACE_FEEDBACK_AUTO_DISABLED", "").strip() == "1"


# ── data shapes ──────────────────────────────────────────────────────


@dataclass
class FeedbackBatchInput:
    """One Haiku-rater job's payload.

    Either op-scoped (op_ctx_id is set, memories are the ones this
    op surfaced) or search-scoped (op_ctx_id is None, memories are
    the kg_search result for the corresponding context).
    """

    batch_id: str
    intent_exec_id: str
    agent: str
    rating_context_id: str
    op_tool: str | None = None
    op_args_summary: str | None = None
    intent_context_prose: str = ""
    memories: list[dict] = field(default_factory=list)
    is_search: bool = False


@dataclass
class FeedbackBatchResult:
    """Haiku-rater output, persisted by the caller."""

    batch_id: str
    memory_ratings: list[dict]
    op_rating: dict | None
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    elapsed_ms: float = 0.0
    error: str | None = None


# ── prompt + tool schema ─────────────────────────────────────────────


_SYSTEM_PROMPT = (
    "You are a memory-feedback rater for a knowledge-graph-backed "
    "agent. The main agent just finished an intent. Your job: rate "
    "each memory the agent saw + (when applicable) rate the agent's "
    "tool invocation, on a 1-5 scale, from OBSERVATION only -- you "
    "don't talk to the agent.\n\n"
    "## Memory relevance scale\n\n"
    "5 -- LOAD-BEARING. The agent's actions directly used or "
    "referenced this memory's content; the intent would have failed "
    "or duplicated work without it.\n"
    "4 -- INFORMED. The agent's actions plausibly used this memory "
    "to make a decision or save a lookup; same topic, contributed "
    "context.\n"
    "3 -- RELATED CONTEXT. Topical and accurate but the agent's "
    "trajectory doesn't show direct use. DEFAULT when uncertain.\n"
    "2 -- NOISE. Skimmed and dropped; same broad area, nothing to "
    "do with this specific op.\n"
    "1 -- MISLEADING. Pointed the agent wrong (wrong project / "
    "wrong concept / off-topic). Teach the context NOT to surface "
    "this again.\n\n"
    "## Operation quality scale (only when op_tool is set)\n\n"
    "5 -- LOAD-BEARING for the intent.\n"
    "4 -- GOOD move.\n"
    "3 -- OK (neutral; no promotion).\n"
    "2 -- SUBOPTIMAL; there was a better tool/args.\n"
    "1 -- WRONG move; the agent should have done X instead.\n\n"
    "Calibration: if more than 50% of your memory ratings land at "
    ">=4, re-read the intent and demote. Clustering at the top "
    "compresses every downstream signal. The system learns from "
    "the SKEW; inflating ratings dampens that.\n\n"
    "Emit via the `rate_feedback` tool. Memory_ratings must include "
    "every memory id you saw, exactly once. Op_rating must be "
    "present iff op_tool is set."
)


def _build_tool_schema(op_present: bool) -> dict:
    """Forced-tool-use schema for Haiku's output."""
    schema: dict[str, Any] = {
        "name": "rate_feedback",
        "description": (
            "Emit per-memory ratings and (when an operation is being "
            "rated) the operation quality rating."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_ratings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "relevance": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5,
                            },
                            "reason": {"type": "string"},
                        },
                        "required": ["id", "relevance", "reason"],
                    },
                }
            },
            "required": ["memory_ratings"],
        },
    }
    if op_present:
        schema["input_schema"]["properties"]["op_rating"] = {
            "type": "object",
            "properties": {
                "quality": {"type": "integer", "minimum": 1, "maximum": 5},
                "reason": {"type": "string"},
            },
            "required": ["quality", "reason"],
        }
        schema["input_schema"]["required"].append("op_rating")
    return schema


# ── user-content assembly ────────────────────────────────────────────


def _render_memory_for_rater(mem: dict, idx: int) -> str:
    """Render one memory into the rater's user-content block."""
    mid = mem.get("id", f"<unknown_{idx}>")
    summary = (mem.get("summary") or mem.get("text") or mem.get("preview") or "").strip()
    kind = mem.get("kind") or mem.get("source") or "memory"
    channel = mem.get("channel") or ""
    score = mem.get("score") or mem.get("hybrid_score")
    score_str = f" score={score:.3f}" if isinstance(score, (int, float)) else ""
    ch_str = f" channel={channel}" if channel else ""
    return f"[{idx}] id={mid} kind={kind}{ch_str}{score_str}\n    {summary[:600]}"


def _build_user_content(batch: FeedbackBatchInput) -> str:
    """Assemble the user message for one Haiku rater call."""
    parts: list[str] = []
    parts.append("## Intent context\n")
    parts.append(batch.intent_context_prose.strip() or "(none provided)")
    parts.append("")
    if not batch.is_search and batch.op_tool:
        parts.append("## Operation being rated\n")
        parts.append(f"tool: {batch.op_tool}")
        parts.append(f"args_summary: {batch.op_args_summary or '(none)'}")
        parts.append(f"context_id: {batch.rating_context_id}")
        parts.append("")
    else:
        parts.append("## Search being rated\n")
        parts.append(f"search context_id: {batch.rating_context_id}")
        parts.append("")
    parts.append("## Memories to rate")
    parts.append(
        f"({len(batch.memories)} item{'s' if len(batch.memories) != 1 else ''}; "
        "rate every id below exactly once)"
    )
    parts.append("")
    for i, mem in enumerate(batch.memories):
        parts.append(_render_memory_for_rater(mem, i))
    parts.append("")
    return "\n".join(parts)


# ── Haiku call wrapper ───────────────────────────────────────────────


_DEFAULT_MODEL = "claude-haiku-4-5"


def _resolve_client_and_model() -> tuple[Any | None, str]:
    """Find a usable Anthropic client + model.

    Reuses InjectionGate's client when possible (one less env-key
    read, same prompt-cache pool); falls back to a fresh
    anthropic.Anthropic() if the gate isn't available.
    """
    try:
        from .injection_gate import get_gate

        gate = get_gate()
        client = gate._get_client() if gate is not None else None
        if client is not None:
            return client, getattr(gate, "model", _DEFAULT_MODEL)
    except Exception:
        pass
    try:
        import anthropic

        return anthropic.Anthropic(), os.environ.get(
            "MEMPALACE_FEEDBACK_AUTO_MODEL", _DEFAULT_MODEL
        )
    except Exception:
        return None, _DEFAULT_MODEL


def _call_haiku(batch: FeedbackBatchInput) -> FeedbackBatchResult:
    """Single Haiku call for one batch. Synchronous; called from a
    background thread.

    Always returns a FeedbackBatchResult (with .error set on
    failure). Never raises -- the calling thread should not need a
    try/except.
    """
    t0 = time.perf_counter()
    result = FeedbackBatchResult(batch_id=batch.batch_id, memory_ratings=[], op_rating=None)
    client, model = _resolve_client_and_model()
    if client is None:
        result.error = "anthropic_client_unavailable"
        result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return result

    op_present = (not batch.is_search) and bool(batch.op_tool)
    tool_def = _build_tool_schema(op_present)
    user_content = _build_user_content(batch)

    cached_system = [
        {
            "type": "text",
            "text": _SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    cached_tools = [{**tool_def, "cache_control": {"type": "ephemeral"}}]

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=2048,
            system=cached_system,
            tools=cached_tools,
            tool_choice={"type": "tool", "name": "rate_feedback"},
            messages=[{"role": "user", "content": user_content}],
        )
    except Exception as exc:
        result.error = f"api_call_failed: {type(exc).__name__}: {exc}"
        result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return result

    try:
        for block in resp.content or []:
            if getattr(block, "type", None) != "tool_use":
                continue
            payload = block.input or {}
            raw_mems = payload.get("memory_ratings") or []
            for entry in raw_mems:
                if not isinstance(entry, dict):
                    continue
                mid = (entry.get("id") or "").strip()
                rel = entry.get("relevance")
                reason = (entry.get("reason") or "").strip()
                if mid and isinstance(rel, int) and 1 <= rel <= 5:
                    result.memory_ratings.append(
                        {"id": mid, "relevance": rel, "reason": reason or "(no reason)"}
                    )
            if op_present:
                op_entry = payload.get("op_rating") or {}
                if isinstance(op_entry, dict):
                    q = op_entry.get("quality")
                    r = (op_entry.get("reason") or "").strip()
                    if isinstance(q, int) and 1 <= q <= 5:
                        result.op_rating = {"quality": q, "reason": r or "(no reason)"}
    except Exception as exc:
        result.error = f"parse_failed: {type(exc).__name__}: {exc}"

    usage = getattr(resp, "usage", None)
    if usage is not None:
        result.tokens_in = int(getattr(usage, "input_tokens", 0) or 0)
        result.tokens_out = int(getattr(usage, "output_tokens", 0) or 0)
        result.cache_read_input_tokens = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        result.cache_creation_input_tokens = int(
            getattr(usage, "cache_creation_input_tokens", 0) or 0
        )

    result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    return result


# ── persistence ──────────────────────────────────────────────────────


def _persist_result(result: FeedbackBatchResult, batch: FeedbackBatchInput, kg) -> None:
    """Write the rater's output into the KG.

    Memory ratings -> rated_useful / rated_irrelevant edges via
    kg.record_feedback (with rater_kind='haiku_auto').
    Operation rating -> kg.record_operation_rating (when present).
    """
    if kg is None:
        return
    # Build a quick id -> source lookup so we route each rating to
    # the right namespace. Memories whose id starts with 't_' (or
    # whose source field is 'triple') go to triple_context_feedback;
    # everything else lands as an add_rated_edge against the
    # entities table. Mirrors the gate's persist_drops dispatch.
    _src_by_id: dict[str, str] = {}
    for m in batch.memories:
        mid = (m.get("id") or "").strip()
        if not mid:
            continue
        src = (m.get("source") or m.get("kind") or "").lower()
        if not src:
            src = "triple" if mid.startswith("t_") else "entity"
        _src_by_id[mid] = "triple" if src == "triple" else "entity"
    for entry in result.memory_ratings or []:
        mid = entry["id"]
        target_kind = _src_by_id.get(mid)
        if target_kind is None:
            target_kind = "triple" if str(mid).startswith("t_") else "entity"
        try:
            kg.record_feedback(
                batch.rating_context_id,
                mid,
                target_kind,
                relevance=int(entry["relevance"]),
                reason=entry.get("reason") or "",
                rater_kind="haiku_auto",
                rater_id=batch.agent or "ga_agent",
            )
        except Exception as exc:  # pragma: no cover -- defensive
            logger.info(
                "feedback_auto: record_feedback failed mid=%s err=%s",
                mid,
                exc,
            )
    if result.op_rating is not None and not batch.is_search and batch.op_tool:
        try:
            recorder = getattr(kg, "record_operation_rating", None)
            if callable(recorder):
                recorder(
                    op_context_id=batch.rating_context_id,
                    tool=batch.op_tool,
                    quality=int(result.op_rating["quality"]),
                    rater_kind="haiku_auto",
                    rater_id=batch.agent or "ga_agent",
                    reason=result.op_rating.get("reason") or "",
                )
        except Exception as exc:  # pragma: no cover -- defensive
            logger.info("feedback_auto: record_operation_rating failed err=%s", exc)


# ── telemetry ────────────────────────────────────────────────────────


def _log_batch(result: FeedbackBatchResult, batch: FeedbackBatchInput) -> None:
    """One JSONL row per Haiku rater call. Best-effort -- telemetry
    failures must not block the rater path."""
    try:
        from datetime import datetime as _dt, timezone as _tz

        from .mcp_server import _telemetry_append_jsonl as _tel

        _tel(
            "feedback_auto_log.jsonl",
            {
                "ts": _dt.now(_tz.utc).isoformat(timespec="seconds"),
                "batch_id": batch.batch_id,
                "intent_exec_id": batch.intent_exec_id,
                "agent": batch.agent or "",
                "rating_context_id": batch.rating_context_id,
                "op_tool": batch.op_tool or "",
                "is_search": batch.is_search,
                "n_memories": len(batch.memories),
                "n_rated": len(result.memory_ratings),
                "op_rated": result.op_rating is not None,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "cache_read": result.cache_read_input_tokens,
                "cache_creation": result.cache_creation_input_tokens,
                "elapsed_ms": result.elapsed_ms,
                "error": result.error or "",
            },
        )
    except Exception:
        pass


# ── threadpool dispatcher ────────────────────────────────────────────


_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def _get_executor() -> ThreadPoolExecutor:
    """Lazy-init the singleton ThreadPoolExecutor.

    One worker is enough -- batches per finalize are O(10) and each
    Haiku call is ~1-3s; serializing them keeps cache hit-rate high
    (sequential calls within a finalize share the 5-min ephemeral
    cache prefix). Multi-worker would shard the cache and double
    cost.
    """
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mempalace-feedback-auto"
            )
        return _executor


def _run_batch(batch: FeedbackBatchInput, kg) -> None:
    """Worker body: call Haiku, persist, log telemetry."""
    try:
        result = _call_haiku(batch)
        _persist_result(result, batch, kg)
        _log_batch(result, batch)
    except Exception as exc:  # pragma: no cover -- last-resort guard
        logger.info(
            "feedback_auto: _run_batch crashed batch_id=%s err=%s",
            batch.batch_id,
            exc,
        )


def submit_batch(batch: FeedbackBatchInput, kg) -> None:
    """Fire-and-forget enqueue of one batch.

    Returns immediately (sub-millisecond). The actual Haiku call +
    persistence happens on the worker thread. Failures land in
    feedback_auto_log.jsonl with .error set; the main agent never
    sees them.
    """
    if _disabled():
        return
    if not batch.memories:
        return
    executor = _get_executor()
    executor.submit(_run_batch, batch, kg)


# ── intent.py entrypoint ─────────────────────────────────────────────


# Intent-level memory pool batching size (Adrian directive
# 2026-05-13). Per-op + per-search pools are small (typically 3-10
# memories) so they ride as a single Haiku call each. Intent-level
# retrieval at declare_intent time can surface many more (~K=10-25);
# we batch those 5-per-call to keep each Haiku payload within a
# comfortable token budget AND to maximize cache hits (same system +
# tool schema across every intent-batch in one finalize).
_INTENT_BATCH_SIZE = 5


def submit_finalize_feedback(
    *,
    intent_exec_id: str,
    agent: str,
    intent_context_prose: str,
    intent_context_id: str,
    intent_memories: list[dict],
    op_batches: list[dict],
    search_batches: list[dict],
    kg,
) -> int:
    """Single entrypoint called from intent.tool_finalize_intent.

    Submits Haiku-rater batches for three retrieval sites:

      * intent_memories -- the memory pool surfaced at declare_intent
        time. Batched ``_INTENT_BATCH_SIZE`` per call (5) because this
        pool is typically the largest.
      * op_batches      -- per-operation memory pool. Each op gets
        ONE Haiku call rating its memories + the op itself
        (no internal batching -- per-op pools are small).
      * search_batches  -- per-kg_search memory pool. Each search
        gets ONE Haiku call rating its result memories.

    Returns the count of batches submitted (for the finalize_intent
    response's auto_feedback_report).

    op_batches:     list of {context_id, tool, args_summary, memories}
    search_batches: list of {context_id, memories}
    """
    if _disabled():
        return 0
    submitted = 0

    # Intent-level pool -- batched 5 per Haiku call. Each batch
    # carries the same intent_context_id so all rated edges attribute
    # to the same retrieval site (mirrors the legacy agent-rated
    # behavior where injected memories were rated against the
    # declare_intent context).
    if intent_memories and intent_context_id:
        for i in range(0, len(intent_memories), _INTENT_BATCH_SIZE):
            chunk = intent_memories[i : i + _INTENT_BATCH_SIZE]
            if not chunk:
                continue
            batch = FeedbackBatchInput(
                batch_id=f"{intent_exec_id}:intent:{i // _INTENT_BATCH_SIZE}",
                intent_exec_id=intent_exec_id,
                agent=agent,
                rating_context_id=intent_context_id,
                intent_context_prose=intent_context_prose,
                memories=chunk,
                is_search=False,
                # op_tool stays None -- this is intent-level, not an
                # op being rated. The Haiku rater will see op_tool
                # absent and rate only the memories (no op_rating
                # block).
            )
            submit_batch(batch, kg)
            submitted += 1

    for ob in op_batches or []:
        if not ob.get("memories"):
            continue
        batch = FeedbackBatchInput(
            batch_id=f"{intent_exec_id}:{ob.get('context_id', '?')}",
            intent_exec_id=intent_exec_id,
            agent=agent,
            rating_context_id=ob.get("context_id", ""),
            op_tool=ob.get("tool"),
            op_args_summary=ob.get("args_summary"),
            intent_context_prose=intent_context_prose,
            memories=ob.get("memories", []),
            is_search=False,
        )
        submit_batch(batch, kg)
        submitted += 1

    for sb in search_batches or []:
        if not sb.get("memories"):
            continue
        batch = FeedbackBatchInput(
            batch_id=f"{intent_exec_id}:search:{sb.get('context_id', '?')}",
            intent_exec_id=intent_exec_id,
            agent=agent,
            rating_context_id=sb.get("context_id", ""),
            intent_context_prose=intent_context_prose,
            memories=sb.get("memories", []),
            is_search=True,
        )
        submit_batch(batch, kg)
        submitted += 1

    return submitted


# ── test helper ──────────────────────────────────────────────────────


def _drain_executor_for_test(timeout: float = 30.0) -> None:
    """Tests call this after submit_batch to wait for in-flight
    workers. Resets the executor afterwards so the next test starts
    clean."""
    global _executor
    with _executor_lock:
        if _executor is None:
            return
        ex = _executor
        _executor = None
    ex.shutdown(wait=True, cancel_futures=False)
