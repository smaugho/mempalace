"""Read-only mempalace tool handlers (read bucket).

This module re-exports the read-only tool handlers from ``mempalace.mcp_server``
so the PreToolUse carve-out hook can determine bucket membership by reading
``__all__``. Handler bodies stay in ``mcp_server`` — moving them here would
shuffle code for zero behavioural change.

The hook does NOT import this module at runtime (that would chain into the
heavy ``mcp_server`` import on every PreToolUse call). Instead, the hook
hardcodes the bucket basenames in ``hooks_cli._READ_BUCKET_BASENAMES`` and
``tests/test_hook_buckets.py::test_read_bucket_matches_module_all`` enforces
the two stay in sync. If a handler is added or moves bucket, update BOTH
sides — the drift-sentinel test breaks loudly otherwise.

Bucket semantics: read tools NEVER mutate state. The gate hook always allows
them outside user-message preemption; under preemption they're blocked along
with everything else except the user-intent tier-0 carve-outs
(``declare_user_intents``, ``extend_feedback``) and ``AskUserQuestion``.
"""

from mempalace.mcp_server import (
    tool_diary_read,
    tool_kg_list_declared,
    tool_kg_query,
    tool_kg_search,
    tool_kg_stats,
    tool_kg_timeline,
)

__all__ = [
    "tool_diary_read",
    "tool_kg_list_declared",
    "tool_kg_query",
    "tool_kg_search",
    "tool_kg_stats",
    "tool_kg_timeline",
]
