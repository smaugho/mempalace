"""Intent-lifecycle mempalace tool handlers (lifecycle bucket).

This module re-exports the intent-lifecycle tool handlers from
``mempalace.mcp_server`` (which in turn delegate to ``mempalace.intent``)
so the PreToolUse carve-out hook can determine bucket membership by
reading ``__all__``. Handler bodies stay in ``mcp_server`` / ``intent`` —
moving them here would shuffle code for zero behavioural change.

The hook does NOT import this module at runtime (that would chain into the
heavy ``mcp_server`` import on every PreToolUse call). Instead, the hook
hardcodes the bucket basenames in ``hooks_cli._LIFECYCLE_BUCKET_BASENAMES``
and ``tests/test_hook_buckets.py::test_lifecycle_bucket_matches_module_all``
enforces the two stay in sync. If a handler is added or moves bucket,
update BOTH sides — the drift-sentinel test breaks loudly otherwise.

Bucket semantics: lifecycle tools manage the intent state machine itself —
declaring, extending, finalizing intents and recording feedback. They
bypass the active-intent check entirely (otherwise ``declare_intent``
itself would be deadlocked). Under user-message preemption, only the
two true tier-0 carve-outs proceed:

  - ``mempalace_declare_user_intents`` — the only path that clears the
    pending queue, so it MUST stay reachable.
  - ``mempalace_extend_feedback`` — finishes a prior incomplete finalize
    so the agent isn't trapped between an unfinished intent and a new
    user message.

Every other lifecycle call (``declare_intent``, ``finalize_intent``,
``extend_intent``, ``resolve_conflicts``, ``active_intent``, ``wake_up``)
is blocked under preemption. AskUserQuestion remains the always-allowed
clarify path. See ``hooks_cli._USER_INTENT_TIER0_BASENAMES``.
"""

from mempalace.mcp_server import (
    tool_active_intent,
    tool_declare_intent,
    tool_declare_user_intents,
    tool_extend_feedback,
    tool_extend_intent,
    tool_finalize_intent,
    tool_resolve_conflicts,
    tool_wake_up,
)

__all__ = [
    "tool_active_intent",
    "tool_declare_intent",
    "tool_declare_user_intents",
    "tool_extend_feedback",
    "tool_extend_intent",
    "tool_finalize_intent",
    "tool_resolve_conflicts",
    "tool_wake_up",
]
