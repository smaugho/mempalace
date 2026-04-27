"""Intent-lifecycle mempalace tool handlers (Phase 1 bucket skeleton).

This module re-exports the intent-lifecycle tool handlers from
``mempalace.mcp_server`` (which delegate to ``mempalace.intent``) so the
PreToolUse carve-out hook can introspect bucket membership by file without
moving handler bodies yet. Phase 2 will move handler bodies here incrementally.

Bucket semantics: lifecycle tools manage the intent state machine itself —
declaring, extending, finalizing intents and recording feedback. Tier 0
(``tool_declare_user_intents``) is ALWAYS allowed (even during pending
user-message preemption) so the agent can re-declare intents in response
to a new user request. The other lifecycle tools require an active intent.
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
