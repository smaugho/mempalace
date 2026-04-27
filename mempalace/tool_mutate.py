"""Mutating mempalace tool handlers (Phase 1 bucket skeleton).

This module re-exports the state-mutating tool handlers from
``mempalace.mcp_server`` so the PreToolUse carve-out hook can introspect
bucket membership by file without moving handler bodies yet. Phase 2 will
move the handler bodies here incrementally.

Bucket semantics: mutate tools change KG / diary state. They are gated by
the active intent's permitted_tools and BLOCKED entirely while a pending
user message is unhandled (only lifecycle tools tier 0 may run then).
"""

from mempalace.mcp_server import (
    tool_declare_operation,
    tool_diary_write,
    tool_kg_add,
    tool_kg_add_batch,
    tool_kg_declare_entity,
    tool_kg_delete_entity,
    tool_kg_invalidate,
    tool_kg_merge_entities,
    tool_kg_update_entity,
)

__all__ = [
    "tool_declare_operation",
    "tool_diary_write",
    "tool_kg_add",
    "tool_kg_add_batch",
    "tool_kg_declare_entity",
    "tool_kg_delete_entity",
    "tool_kg_invalidate",
    "tool_kg_merge_entities",
    "tool_kg_update_entity",
]
