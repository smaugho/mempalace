"""Cheap-LLM (Claude Haiku) auto-author for system-internal summaries.

Cold-start lock 2026-05-01: every entity row in SQLite must carry a
structured ``{what, why, scope?}`` summary that passes ``validate_summary``.
For agent-initiated paths the caller provides the dict directly; for
system-internal paths (gardener auto-writes, link_author similarity
rationales, finalize_intent execution / gotcha / learning entities,
declare_user_intents) the caller usually doesn't know the WHY at
emission time -- the system is observing the agent's behaviour and
naming the entity on the fly.

This module is the single sanctioned auto-author surface for those
system-internal paths. It calls Claude Haiku via tool-use to produce a
validated dict, with a per-process budget cap and prompt-cache-friendly
system blocks. Reuses ``anthropic.Anthropic()`` initialisation
(``ANTHROPIC_API_KEY`` from env or palace ``.env``) to match
``memory_gardener`` and ``link_author``.

Failure semantics
-----------------
Pre-cold-start, summary-less entity creation slipped through 12
catalogued surfaces. The cold-start invariant -- "every entity carries
a real summary" -- requires that auto-author failures NOT silently
degrade back to no-summary writes. Callers MUST raise or surface an
error if ``auto_author_summary`` fails. The gate (``mint_entity``) will
also reject string summaries / placeholder fallbacks, so the
degradation path is closed at both ends.

Budget
------
Per-process call counter caps Haiku at ``_MAX_HAIKU_CALLS_PER_PROCESS``
(default 100). Once the cap is hit, ``BudgetExhausted`` is raised so the
caller can defer the work (queue + retry next session) rather than
let unchecked auto-author traffic blow up the bill. The cap is
intentionally low for cold-start; tune up post-rollout once Haiku spend
is observed.

Caching
-------
The system blocks include a cacheable preamble (constant across all
auto-author calls in a session) marked ``cache_control={'type':
'ephemeral'}``. The first call pays the cache-create tokens; every
subsequent call in the same session reads it from cache (90% discount
on input tokens). Effective cost per call: a few hundred user-message
tokens + ~50 output tokens via tool_use.

References
----------
* Anthropic Claude API -- prompt caching (5min ephemeral TTL).
* Anthropic Claude API -- tool_use with ``tool_choice={'type': 'tool',
  ...}`` for forced structured output.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Budget + model config ─────────────────────────────────────────────

_MAX_HAIKU_CALLS_PER_PROCESS = 100
_MODEL = "claude-haiku-4-5"
_MAX_OUTPUT_TOKENS = 400

_call_counter = 0
_call_lock = threading.Lock()


def reset_budget() -> None:
    """Test hook: reset the per-process call counter to zero.

    Production code must NOT call this; the budget exists to prevent
    runaway Haiku spend. Tests reset between cases so each test sees a
    fresh budget."""
    global _call_counter
    with _call_lock:
        _call_counter = 0


def get_call_count() -> int:
    """Return the current process-wide auto-author call count."""
    with _call_lock:
        return _call_counter


# ── Errors ────────────────────────────────────────────────────────────


class AutoAuthorError(Exception):
    """Auto-author could not produce a valid summary dict."""


class BudgetExhausted(AutoAuthorError):
    """Per-process cap reached (``_MAX_HAIKU_CALLS_PER_PROCESS``).

    The caller is expected to either queue the entity for a later pass,
    fail the operation gracefully (with a user-visible message), or
    raise to its caller. It MUST NOT silently fall back to a no-summary
    write -- that re-opens one of the 12 cold-start bypass surfaces."""


class AnthropicNotAvailable(AutoAuthorError):
    """The Anthropic SDK is missing or no API key is configured.

    Same response as ``BudgetExhausted``: caller must NOT degrade to
    no-summary writes."""


# ── Request shape ─────────────────────────────────────────────────────


@dataclass
class AuthorRequest:
    """One auto-author request to Claude Haiku.

    Attributes
    ----------
    kind : str
        The entity kind being authored. Drives the prompt template.
        Recognised values:
          - ``rationale`` (memory_gardener / link_author rationale entities)
          - ``execution`` (intent finalize_intent execution entity)
          - ``gotcha`` (intent finalize_intent gotcha entity)
          - ``learning`` (intent finalize_intent learning entity)
          - ``user_intent`` (declare_user_intents auto-derived class)
          - ``entity`` (generic fallback)
    anchor_text : str
        The prose / payload the summary is being authored ABOUT.
        Examples: a similarity-rationale paragraph, a gotcha description,
        the user's natural-language intent.
    context_blocks : list[str]
        Cacheable system-level context (parent intent type, related
        entity ids, tool name, etc.) to ground the summary. These blocks
        sit AFTER the cacheable preamble so they participate in the
        prompt cache for repeated grounding (e.g. a gardener pass that
        writes 50 rationales under the same parent intent).
    """

    kind: str
    anchor_text: str
    context_blocks: list[str] = field(default_factory=list)


# ── Tool schema (forced structured output) ────────────────────────────


_AUTHOR_TOOL_SCHEMA = {
    "name": "author_summary",
    "description": (
        "Emit a structured {what, why, scope?} summary for the entity. "
        "Must pass validate_summary: what >=8 chars (specific noun phrase, "
        "not generic), why >=15 chars (purpose/role/claim, not name "
        "restatement), scope <=100 chars optional (temporal/domain qualifier)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "what": {
                "type": "string",
                "description": (
                    "Noun phrase naming this specific entity. Must be "
                    ">=8 chars and discriminative (not 'data', 'thing', "
                    "'the project'). Example good: 'mempalace cold-start "
                    "audit 2026-05-01'. Example bad: 'audit'."
                ),
            },
            "why": {
                "type": "string",
                "description": (
                    "Purpose / role / claim clause, >=15 chars. Must "
                    "explain WHY this entity exists, not restate the "
                    "name. Example good: 'enumerates every entity-"
                    "creation surface so cold-start can route all 12 "
                    "through one gate'. Example bad: 'the audit'."
                ),
            },
            "scope": {
                "type": "string",
                "description": (
                    "Optional temporal / domain qualifier, <=100 chars. "
                    "Example: '2026-05-01 cold-start prep'."
                ),
            },
        },
        "required": ["what", "why"],
    },
}


# ── Public API ────────────────────────────────────────────────────────


def auto_author_summary(req: AuthorRequest) -> dict:
    """Author a ``{what, why, scope?}`` summary via Claude Haiku.

    Parameters
    ----------
    req : AuthorRequest
        The auto-author request. See ``AuthorRequest`` for fields.

    Returns
    -------
    dict
        ``{"what": str, "why": str, "scope": str?}`` -- guaranteed to
        pass ``coerce_summary_for_persist`` (the caller should still run
        it for ASCII-fold + length-budget normalisation).

    Raises
    ------
    BudgetExhausted
        Per-process call cap hit. Caller must queue or fail loudly --
        NOT degrade to a no-summary write.
    AnthropicNotAvailable
        SDK missing or no API key. Same caller obligation as above.
    AutoAuthorError
        The Haiku call returned a malformed response (no tool_use block,
        invalid JSON, etc.).
    """
    global _call_counter
    with _call_lock:
        if _call_counter >= _MAX_HAIKU_CALLS_PER_PROCESS:
            raise BudgetExhausted(
                f"auto_author per-process cap hit "
                f"({_MAX_HAIKU_CALLS_PER_PROCESS} calls). Defer this entity "
                f"to a later session or surface the error to the operator. "
                f"Do NOT fall back to a no-summary write -- the cold-start "
                f"invariant (every entity has a real summary) is non-"
                f"negotiable. To raise the cap, edit "
                f"_MAX_HAIKU_CALLS_PER_PROCESS in mempalace/auto_author.py."
            )
        _call_counter += 1

    try:
        import anthropic
    except ImportError as exc:
        raise AnthropicNotAvailable(
            f"anthropic SDK not installed: {exc}. Install with "
            f"`pip install anthropic` or set up the palace .env with "
            f"ANTHROPIC_API_KEY before running paths that auto-author."
        ) from exc

    # Reuse the env-loading pattern from injection_gate / link_author so
    # cold-start sessions that spawn a fresh process pick up the palace
    # .env even if the shell hasn't been re-initialised.
    _ensure_env_loaded()
    if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        raise AnthropicNotAvailable(
            "ANTHROPIC_API_KEY is not set in the environment or palace .env. "
            "auto_author cannot mint a real summary without it. Configure "
            "the key and retry; do NOT fall back to a no-summary write."
        )

    try:
        client = anthropic.Anthropic()
    except Exception as exc:
        raise AutoAuthorError(f"anthropic client construction failed: {exc}") from exc

    system_blocks = _build_system_blocks(req)
    user_msg = _build_user_message(req)

    try:
        resp = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_OUTPUT_TOKENS,
            system=system_blocks,
            tools=[_AUTHOR_TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": "author_summary"},
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as exc:
        raise AutoAuthorError(
            f"Haiku call failed for kind={req.kind!r}: {type(exc).__name__}: {exc}"
        ) from exc

    # Capture cache stats for observability (non-fatal if missing).
    try:
        usage = getattr(resp, "usage", None)
        if usage is not None:
            cc = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
            cr = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
            logger.info(
                "auto_author kind=%s cache_create=%d cache_read=%d",
                req.kind,
                cc,
                cr,
            )
    except Exception:
        pass

    # Extract the forced tool_use block. tool_choice=author_summary
    # guarantees Haiku emits exactly one such block; defend against the
    # SDK shape-shifting edge cases anyway.
    for block in resp.content:
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == "author_summary"
        ):
            payload = dict(getattr(block, "input", {}) or {})
            what = (payload.get("what") or "").strip()
            why = (payload.get("why") or "").strip()
            if not what or not why:
                raise AutoAuthorError(
                    f"Haiku tool_use payload missing what/why: payload={payload!r}"
                )
            out: dict = {"what": what, "why": why}
            scope = (payload.get("scope") or "").strip()
            if scope:
                out["scope"] = scope[:100]
            return out

    raise AutoAuthorError(
        "Haiku response contained no tool_use block; "
        f"stop_reason={getattr(resp, 'stop_reason', None)!r}"
    )


# ── Internal helpers ──────────────────────────────────────────────────


_PREAMBLE = (
    "You are the mempalace auto-author. You emit structured summary "
    "dicts {what, why, scope?} for entities created by system-internal "
    "pipelines (memory_gardener, link_author, intent.finalize_intent, "
    "declare_user_intents) which observe agent behaviour and name "
    "entities on the fly.\n\n"
    "Contract (validate_summary, Adrian's design lock 2026-04-25):\n"
    "  what:  noun phrase, >=8 chars, MUST discriminate (not 'data', "
    "'thing', 'the project'). Names THIS particular entity.\n"
    "  why:   purpose / role / claim clause, >=15 chars. Explains WHY "
    "this entity exists. NOT a name restatement.\n"
    "  scope: optional, <=100 chars. Temporal or domain qualifier.\n"
    "  Rendered prose form <=280 chars.\n\n"
    "Always emit via the author_summary tool. Never write prose freely; "
    "the calling code only reads the tool_use payload."
)


_KIND_GUIDANCE = {
    "rationale": (
        "Kind 'rationale': name a graph-mining decision the gardener / "
        "link_author made (e.g. 'similar_to edge between intent X and Y'). "
        "Use the anchor_text to derive the specific decision; 'why' "
        "explains the heuristic that produced it."
    ),
    "execution": (
        "Kind 'execution': finalize_intent execution entity for an "
        "agent-completed intent. 'what' names the intent + outcome "
        "(e.g. 'fix-injection-gate-flake'); 'why' captures the "
        "purpose and the result in one clause."
    ),
    "gotcha": (
        "Kind 'gotcha': a runtime gotcha the agent surfaced during "
        "intent execution. 'what' names the failure mode, 'why' "
        "explains the trap and the workaround."
    ),
    "learning": (
        "Kind 'learning': a lesson learned during the intent. 'what' "
        "names the lesson succinctly, 'why' explains the principle "
        "and how to apply it."
    ),
    "user_intent": (
        "Kind 'user_intent': an intent type derived from a user's "
        "natural-language request. 'what' names the intent class, "
        "'why' explains what kind of work it covers."
    ),
    "entity": (
        "Kind 'entity': fallback for system-internal entity creation. "
        "Match the discrimination floor and prose budget exactly."
    ),
}


def _build_system_blocks(req: AuthorRequest) -> list[dict]:
    """Assemble the system message as cacheable blocks.

    Block 0: the constant preamble (cache key across all auto-author
        calls in the session). cache_control=ephemeral so subsequent
        calls in the 5min window read it from cache.
    Block 1: per-kind guidance (still cacheable across multiple
        same-kind calls; e.g. a gardener pass that writes 50
        rationales).
    Block 2..N: caller-supplied context_blocks (parent intent type,
        tool name, etc.).
    """
    blocks: list[dict] = [
        {
            "type": "text",
            "text": _PREAMBLE,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    kind_text = _KIND_GUIDANCE.get(req.kind, _KIND_GUIDANCE["entity"])
    blocks.append(
        {
            "type": "text",
            "text": kind_text,
            "cache_control": {"type": "ephemeral"},
        }
    )
    for ctx in req.context_blocks:
        if not isinstance(ctx, str) or not ctx.strip():
            continue
        blocks.append({"type": "text", "text": ctx.strip()})
    return blocks


def _build_user_message(req: AuthorRequest) -> str:
    """Build the user message that grounds the auto-author call.

    The anchor_text is the only per-call dynamic input; everything else
    lives in cacheable system blocks. This shape is deliberate -- it
    keeps cache hit rate high across batched gardener passes that
    differ only in the entity payload.
    """
    return (
        f"Anchor text (what this summary is about):\n"
        f"---\n{req.anchor_text.strip()}\n---\n\n"
        f"Author the {{what, why, scope?}} summary now via the "
        f"author_summary tool. Be specific: 'what' must name THIS "
        f"particular entity, not a generic class."
    )


def _ensure_env_loaded() -> None:
    """Load the palace .env if ANTHROPIC_API_KEY isn't already set.

    Mirrors injection_gate._ensure_palace_env_loaded so this module
    behaves identically to the existing Haiku consumers.
    """
    if (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        return
    try:
        # Reuse link_author's loader so a single source of truth governs
        # how palace .env is discovered + applied.
        from mempalace.link_author import _load_env  # type: ignore[attr-defined]

        _load_env(override=True)
    except Exception:
        # Fallback: best-effort dotenv probe at the palace path. This
        # path matters only for fresh sessions started from a shell that
        # didn't pre-load the env; the gardener's runner already loads it.
        try:
            from pathlib import Path

            palace_env = Path.home() / ".mempalace" / "palace" / ".env"
            if palace_env.is_file():
                for line in palace_env.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    if k.strip() and not os.environ.get(k.strip()):
                        os.environ[k.strip()] = v.strip().strip("\"'")
        except Exception:
            pass


__all__ = [
    "AutoAuthorError",
    "BudgetExhausted",
    "AnthropicNotAvailable",
    "AuthorRequest",
    "auto_author_summary",
    "reset_budget",
    "get_call_count",
]
