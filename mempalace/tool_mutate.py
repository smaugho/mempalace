"""Mutating mempalace tool handlers (mutate bucket).

This module re-exports the state-mutating tool handlers from
``mempalace.mcp_server`` so the PreToolUse carve-out hook can determine
bucket membership by reading ``__all__``. Handler bodies stay in
``mcp_server`` — moving them here would shuffle code for zero behavioural
change.

The hook does NOT import this module at runtime (that would chain into the
heavy ``mcp_server`` import on every PreToolUse call). Instead, the hook
hardcodes the bucket basenames in ``hooks_cli._MUTATE_BUCKET_BASENAMES``
and ``tests/test_hook_buckets.py::test_mutate_bucket_matches_module_all``
enforces the two stay in sync. If a handler is added or moves bucket,
update BOTH sides — the drift-sentinel test breaks loudly otherwise.

Bucket semantics: mutate tools change KG / diary state. The gate hook
REQUIRES an active intent for any mutate-bucket call; without an intent
the call is denied with guidance. Under user-message preemption, mutates
are blocked entirely along with everything else except the user-intent
tier-0 carve-outs (``declare_user_intents``, ``extend_feedback``) and
``AskUserQuestion``. ``declare_operation`` lives in this bucket because
it mints a retrieval cue that has to attach to an active intent.
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
