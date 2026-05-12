"""Haiku-driven length coerce for over-length summary dicts.

Adrian's directive 2026-05-12: when a summary dict's rendered prose
form (``what -- why; scope``) exceeds ``_SUMMARY_MAX_LEN`` (280 chars),
the historical behaviour is to hard-reject the write so the agent
retries with a trimmed dict. That round-trip costs 3-5 retry cycles
per length violation in practice -- a lot of agent tokens for what is
fundamentally a normalisation problem, not a correctness problem.

This module is the soft-coerce alternative. The single canonical
``coerce_summary_for_persist`` wrapper (knowledge_graph.py:498) catches
the 280-char error, hands the dict to Haiku via forced tool-use with a
"trim only, preserve meaning" prompt, validates the returned dict, and
threads it back through validation. If Haiku is unavailable (no API
key, SDK missing, budget exhausted, or the call fails) the gate
falls back to raising the original error so callers see the same
contract as before -- the coerce is a best-effort upgrade, never a
correctness compromise.

Scope (deliberate)
------------------
This module ONLY trims length. It does NOT:

* rewrite cop-out memory_feedback reasons,
* coerce slot-name mismatches,
* extend intent-class scopes,
* override the state_judge.

Adrian's direction 2026-05-12: "but only for now this one discussed
of the summary length (but apply it everywhere that a summary is
received!!)." The "everywhere" is satisfied by routing through
``coerce_summary_for_persist`` which every summary write already
goes through.

Failure semantics
-----------------
* ``haiku_coerce_summary_to_length`` returns ``None`` on any failure
  path so the caller raises the original ``SummaryStructureRequired``.
* The caller (coerce_summary_for_persist) MUST validate the returned
  dict again -- Haiku can in principle return a dict that still
  exceeds the cap, miswrites a field, or drops ``scope``. The retry
  validation is the gate.

Cache
-----
In-process LRU keyed by (what, why, scope, max_len). Cap 128. Many
length violations are recurrent shape mistakes by the same agent on
the same kind of entity (e.g. always over-detailed ``why`` clauses);
caching keeps the Haiku spend bounded across a session.

Budget
------
Separate from auto_author's budget. Default 200/process -- more
permissive than auto_author (100) because this is reactive: it only
runs when a write would otherwise fail. Once exhausted, falls back to
raising the original length error.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────

_MAX_COERCE_CALLS_PER_PROCESS = 200
_MODEL = "claude-haiku-4-5"
_MAX_OUTPUT_TOKENS = 400
_CACHE_MAX = 128

_call_counter = 0
_call_lock = threading.Lock()
_cache: "OrderedDict[tuple, dict]" = OrderedDict()
_cache_lock = threading.Lock()

# Telemetry counter -- callers can read for observability.
_stats = {
    "calls": 0,
    "cache_hits": 0,
    "haiku_invocations": 0,
    "haiku_failures": 0,
    "fallbacks_to_raise": 0,
    "successful_coerces": 0,
}
_stats_lock = threading.Lock()


def reset_budget() -> None:
    """Test hook: reset the per-process call counter + cache + stats.

    Production code must NOT call this. Tests use it between cases so
    each test sees a fresh budget / empty cache."""
    global _call_counter
    with _call_lock:
        _call_counter = 0
    with _cache_lock:
        _cache.clear()
    with _stats_lock:
        for k in _stats:
            _stats[k] = 0


def get_stats() -> dict:
    """Return a snapshot of coerce telemetry counters."""
    with _stats_lock:
        return dict(_stats)


# ── Tool schema (forced structured output) ────────────────────────────


_TRIM_TOOL_SCHEMA = {
    "name": "trim_summary",
    "description": (
        "Re-emit the structured {what, why, scope?} summary so its "
        "rendered prose form ('what -- why; scope') fits within the "
        "embedding budget. PRESERVE the identity (what), the purpose "
        "claim (why), and the scope qualifier if present. ONLY shorten "
        "wording; do NOT change facts, drop the scope when present, or "
        "replace the discriminative noun phrase with a generic one. "
        "The output MUST still pass validate_summary: what >=5 chars, "
        "why >=15 chars, scope <=100 chars."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "what": {
                "type": "string",
                "description": (
                    "Discriminative noun phrase, >=5 chars. Same "
                    "identity as the input -- only re-worded to be "
                    "shorter. Do not generalise."
                ),
            },
            "why": {
                "type": "string",
                "description": (
                    "Purpose / role / claim clause, >=15 chars. Same "
                    "claim as the input -- only shortened. Drop "
                    "redundant phrasing, keep the load-bearing verb + "
                    "object."
                ),
            },
            "scope": {
                "type": "string",
                "description": (
                    "Temporal / domain qualifier, <=100 chars. If the "
                    "input had a scope, keep one (shortened if needed). "
                    "If the input had no scope, omit this field."
                ),
            },
        },
        "required": ["what", "why"],
    },
}


# ── Errors ────────────────────────────────────────────────────────────


class SummaryCoerceError(Exception):
    """Coerce surfaced an unexpected failure path.

    Callers do NOT need to catch this -- ``haiku_coerce_summary_to_length``
    swallows it and returns ``None`` so the gate falls back to raising
    the original length error. The class exists for log clarity."""


# ── Public API ────────────────────────────────────────────────────────


def haiku_coerce_summary_to_length(
    summary,
    *,
    max_len: int,
    context_for_error: str = "summary",
):
    """Trim ``summary`` so its rendered prose form fits ``max_len`` chars.

    Returns a dict with the same shape as the input (``{what, why}`` +
    optional ``scope``) but with each field re-worded by Claude Haiku to
    fit the embedding budget while preserving meaning. Returns ``None``
    on any failure path (Anthropic SDK missing, API key missing, budget
    exhausted, Haiku call failed, Haiku returned invalid shape) so the
    caller can raise the original length error.

    Parameters
    ----------
    summary : dict
        ``{what: str, why: str, scope: str?}`` already validated by
        the structural shape check; only the rendered length is over.
        The caller MUST pass the post-fold dict so the trimmed result
        also stays ASCII.
    max_len : int
        Target rendered length budget. Caller typically passes
        ``_SUMMARY_MAX_LEN`` (280).
    context_for_error : str
        Audit / log tag -- the validator's context_for_error so logs
        link back to the failing call site.

    Returns
    -------
    dict | None
        Re-worded summary dict on success, ``None`` on failure.

    Notes
    -----
    The caller is responsible for re-validating the returned dict.
    Haiku can in principle return a dict that still over-flows; the
    retry validation in coerce_summary_for_persist is the gate.
    """
    if not isinstance(summary, dict):
        return None
    what = summary.get("what")
    why = summary.get("why")
    scope = summary.get("scope")
    if not isinstance(what, str) or not isinstance(why, str):
        return None
    scope_str = scope if isinstance(scope, str) else ""

    with _stats_lock:
        _stats["calls"] += 1

    cache_key = (what, why, scope_str, int(max_len))
    with _cache_lock:
        if cache_key in _cache:
            _cache.move_to_end(cache_key)
            hit = dict(_cache[cache_key])
            with _stats_lock:
                _stats["cache_hits"] += 1
                _stats["successful_coerces"] += 1
            logger.info(
                "summary_coerce cache_hit context=%s",
                context_for_error,
            )
            return hit

    # Budget check.
    global _call_counter
    with _call_lock:
        if _call_counter >= _MAX_COERCE_CALLS_PER_PROCESS:
            logger.warning(
                "summary_coerce budget exhausted (%d/%d); falling back to raise for context=%s",
                _call_counter,
                _MAX_COERCE_CALLS_PER_PROCESS,
                context_for_error,
            )
            with _stats_lock:
                _stats["fallbacks_to_raise"] += 1
            return None
        _call_counter += 1

    # SDK + env check.
    try:
        import anthropic  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "summary_coerce: anthropic SDK not installed; falling back to raise for context=%s",
            context_for_error,
        )
        with _stats_lock:
            _stats["fallbacks_to_raise"] += 1
        return None

    # Reuse the env-loading pattern from auto_author / link_author so
    # cold-start sessions still pick up the palace .env.
    try:
        from .auto_author import _ensure_env_loaded  # noqa: PLC0415

        _ensure_env_loaded()
    except Exception:
        # Helper is best-effort; missing env is caught next.
        pass

    if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        logger.warning(
            "summary_coerce: ANTHROPIC_API_KEY missing; falling back to raise for context=%s",
            context_for_error,
        )
        with _stats_lock:
            _stats["fallbacks_to_raise"] += 1
        return None

    try:
        client = anthropic.Anthropic()
    except Exception as exc:
        logger.warning(
            "summary_coerce: client construct failed (%s); falling back",
            exc,
        )
        with _stats_lock:
            _stats["fallbacks_to_raise"] += 1
        return None

    # Render the current too-long form for the prompt so Haiku sees the
    # exact prose it has to compress.
    cur_prose_parts = [what.strip(), " -- ", why.strip()]
    if scope_str.strip():
        cur_prose_parts.append("; ")
        cur_prose_parts.append(scope_str.strip())
    cur_prose = "".join(cur_prose_parts)
    cur_len = len(cur_prose)

    system_blocks = [
        {
            "type": "text",
            "text": (
                "You compress structured summary dicts that exceed the "
                "embedding-budget cap. Adrian's design lock 2026-04-25: "
                "what+why+scope is a tight identity anchor for retrieval, "
                "rendered as 'what -- why; scope'. Your job is to "
                "PRESERVE the identity and claim, only TRIM wording. "
                "Never replace the discriminative noun phrase with a "
                "generic one, never drop the scope when present, never "
                "change facts. Emit via the trim_summary tool only.\n\n"
                "Contract:\n"
                "  what  >= 5 chars  -- discriminative noun phrase\n"
                "  why   >= 15 chars -- purpose/role/claim clause\n"
                "  scope <= 100 chars optional -- temporal/domain qualifier\n"
                "  Rendered 'what -- why; scope' <= budget chars."
            ),
            "cache_control": {"type": "ephemeral"},
        }
    ]
    user_msg = (
        f"Current summary fields:\n"
        f"  what:  {what.strip()!r}\n"
        f"  why:   {why.strip()!r}\n"
        f"  scope: {scope_str.strip()!r}\n\n"
        f"Current rendered prose ({cur_len} chars):\n"
        f"  {cur_prose}\n\n"
        f"Target: <= {max_len} chars rendered. Trim {cur_len - max_len} "
        f"chars while preserving meaning."
    )

    try:
        resp = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_OUTPUT_TOKENS,
            system=system_blocks,
            tools=[_TRIM_TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": "trim_summary"},
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as exc:
        logger.warning(
            "summary_coerce: Haiku call failed (%s: %s) for context=%s",
            type(exc).__name__,
            exc,
            context_for_error,
        )
        with _stats_lock:
            _stats["haiku_failures"] += 1
            _stats["fallbacks_to_raise"] += 1
        return None

    with _stats_lock:
        _stats["haiku_invocations"] += 1

    # Extract tool_use payload.
    out = None
    try:
        for block in resp.content:
            if (
                getattr(block, "type", None) == "tool_use"
                and getattr(block, "name", None) == "trim_summary"
            ):
                payload = dict(getattr(block, "input", {}) or {})
                new_what = (payload.get("what") or "").strip()
                new_why = (payload.get("why") or "").strip()
                if not new_what or not new_why:
                    break
                out = {"what": new_what, "why": new_why}
                new_scope = (payload.get("scope") or "").strip()
                # Preserve scope presence: if original had one, keep one
                # (Haiku may shorten it); if original had none, omit.
                if scope_str.strip() and new_scope:
                    out["scope"] = new_scope[:100]
                break
    except Exception as exc:
        logger.warning(
            "summary_coerce: payload extract failed (%s); falling back",
            exc,
        )
        with _stats_lock:
            _stats["fallbacks_to_raise"] += 1
        return None

    if out is None:
        logger.warning(
            "summary_coerce: Haiku returned no tool_use block for context=%s; falling back",
            context_for_error,
        )
        with _stats_lock:
            _stats["fallbacks_to_raise"] += 1
        return None

    # Cache the result (caller will re-validate).
    with _cache_lock:
        _cache[cache_key] = dict(out)
        if len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)

    with _stats_lock:
        _stats["successful_coerces"] += 1

    logger.info(
        "summary_coerce success context=%s input_len=%d trimmed_what=%d "
        "trimmed_why=%d trimmed_scope=%d",
        context_for_error,
        cur_len,
        len(out["what"]),
        len(out["why"]),
        len(out.get("scope") or ""),
    )

    return out
