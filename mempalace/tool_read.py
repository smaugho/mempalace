"""Read-only mempalace tool handlers (Phase 1 bucket skeleton).

This module re-exports the read-only tool handlers from ``mempalace.mcp_server``
so the PreToolUse carve-out hook can introspect bucket membership by file
without moving handler bodies yet. Phase 2 will move the handler bodies here
incrementally.

Bucket semantics: read-only tools NEVER mutate state. They are always allowed
under any active intent (and during user-message preemption) — they only
require ``mempalace_declare_operation`` for retrieval-cue alignment.
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
