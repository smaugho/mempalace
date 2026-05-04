"""Hardcoded state schemas for the state-protocol v1.

Adding a new state-bearing kind = code change here + cold-restart per
no_back_compat_principle. Agent-authored schemas are deferred to Phase 6;
they require whole-corpus revalidation against existing entities.

seed.py reads STATE_SCHEMAS at palace-init time and emits each entry as a
kind=state_schema entity alongside the root-class seeding.
"""

from __future__ import annotations

from typing import TypedDict


class StateSchemaDef(TypedDict, total=False):
    json_schema: dict
    slot_descriptions: dict[str, str]
    parent_schema_id: str | None
    # Phase 6 lazy-migration-at-injection (Adrian design lock 2026-05-03):
    # bump version when the schema shape changes in a way old revisions
    # need to migrate to. Add a corresponding migrate function under
    # mempalace/state_migrations/{schema_id}/v{N}_to_v{N+1}.py. The
    # migration runner walks revisions whose schema_version < current
    # version and applies the chain at injection-gate time.
    version: int
    # State-protocol v2 (Adrian 2026-05-04) Option A scope policy:
    # 'session' = per-session state (each agent session has its own
    # state stream; a session's reads see only its own writes).
    # 'global'  = canonical state shared across sessions (last-write-
    # wins; all sessions converge on one truth). Phase D wires this
    # into latest_state_for_entity reads + record_state_revision
    # writes. Until Phase D lands the field is declarative metadata.
    scope: str


# State-protocol v2 (Adrian 2026-05-04) -- todos-list shapes replace
# progress_pct + opaque current_step. Each schema carries an
# explicit `version: 2` so the Phase 6 lazy-migration runner picks
# up v1 revisions on next injection and applies the migration chain
# in mempalace/state_migrations/{schema_id}/v1_to_v2.py. `scope`
# field per Option A: 'session' = per-session, 'global' = canonical.
_TODO_ITEM_SUBSCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string"},
        "text": {"type": "string"},
        "status": {
            "type": "string",
            "enum": ["pending", "in_progress", "done", "blocked", "cancelled"],
        },
        "blocker": {"type": ["string", "null"]},
    },
    "required": ["id", "text", "status"],
}


_PROJECT_STATE: StateSchemaDef = {
    "version": 2,
    "scope": "global",
    "json_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "current_phase": {"type": "string"},
            "active_branches": {"type": "array", "items": {"type": "string"}},
            "open_todos": {"type": "array", "items": _TODO_ITEM_SUBSCHEMA},
            "recent_milestones": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "date": {"type": "string", "format": "date"},
                    },
                    "required": ["name"],
                },
            },
        },
        "required": ["current_phase"],
    },
    "slot_descriptions": {
        "current_phase": "Free-form phase label for the project right now (e.g. 'design', 'slice-a-implementation', 'staged-rollout').",
        "active_branches": "Git branches or workstreams in flight; empty list when only main is active.",
        "open_todos": "Project-level todo items, each with id/text/status/blocker. Patch a single item via `/open_todos/N/status` to update without rewriting the list.",
        "recent_milestones": "Recently-completed milestones with optional date. Cap at last ~5.",
    },
    "parent_schema_id": None,
}


_INTENT_STATE: StateSchemaDef = {
    "version": 2,
    "scope": "session",
    "json_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "todos": {"type": "array", "items": _TODO_ITEM_SUBSCHEMA},
            "active_todo_id": {"type": ["string", "null"]},
            "latest_observation": {"type": "string"},
        },
        "required": ["todos"],
    },
    "slot_descriptions": {
        "todos": "Intent-level checklist. Each item is {id, text, status (pending/in_progress/done/blocked/cancelled), blocker?}. Patch individual items via `/todos/N/status` to advance progress without rewriting the list.",
        "active_todo_id": "Id of the todo currently being worked on, or null between items.",
        "latest_observation": "Last meaningful finding or decision since the prior op declaration. One short sentence.",
    },
    "parent_schema_id": None,
}


_AGENT_STATE: StateSchemaDef = {
    "version": 2,
    "scope": "session",
    "json_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "current_focus": {"type": "string"},
            "active_intent_id": {"type": ["string", "null"]},
            "pending_followups": {"type": "array", "items": _TODO_ITEM_SUBSCHEMA},
            "recent_findings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["current_focus"],
    },
    "slot_descriptions": {
        "current_focus": "What the agent is concentrating on right now -- a topic or workstream label, not a tool call.",
        "active_intent_id": "Id of the intent the agent is currently executing, or null between intents.",
        "pending_followups": "Items the agent has noted to come back to. Each is {id, text, status, blocker?}; status='done' to retire without removing.",
        "recent_findings": "Recently-surfaced facts or decisions the agent learned this session. Cap at last ~5.",
    },
    "parent_schema_id": None,
}


_TASK_STATE: StateSchemaDef = {
    "version": 2,
    "scope": "global",
    "json_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {
                "type": "string",
                "enum": ["open", "in_progress", "blocked", "done", "cancelled"],
            },
            "subtodos": {"type": "array", "items": _TODO_ITEM_SUBSCHEMA},
            "assignee": {"type": ["string", "null"]},
            "due_date": {"type": ["string", "null"], "format": "date"},
            "blocker": {"type": ["string", "null"]},
        },
        "required": ["status"],
    },
    "slot_descriptions": {
        "status": "Lifecycle state. open=not started; in_progress=being worked; blocked=waiting on dep; done=complete; cancelled=abandoned.",
        "subtodos": "Optional sub-tasks for compound work. Each item is {id, text, status, blocker?}. Empty list when the task is atomic.",
        "assignee": "Agent or person responsible. Null when unassigned.",
        "due_date": "Target completion date in ISO format, or null when no deadline.",
        "blocker": "If status='blocked', a free-form description of what is blocking. Null otherwise.",
    },
    "parent_schema_id": None,
}


STATE_SCHEMAS: dict[str, StateSchemaDef] = {
    "project_state": _PROJECT_STATE,
    "intent_state": _INTENT_STATE,
    "agent_state": _AGENT_STATE,
    "task_state": _TASK_STATE,
}


def get_schema(kind_name: str) -> StateSchemaDef:
    """Return the StateSchemaDef for a state-bearing kind. Raises KeyError if unknown."""
    return STATE_SCHEMAS[kind_name]


def list_schemas() -> list[str]:
    """Return the registered state-schema kind names."""
    return list(STATE_SCHEMAS.keys())


def current_version(schema_id: str) -> int:
    """Return the current registered version for a state-schema kind.

    Phase 6 lazy-migration-at-injection (Adrian design lock 2026-05-03):
    each STATE_SCHEMAS entry carries a `version: int` (default 1). When
    a state revision is written, this version is stamped on the row;
    when an entity passes the InjectionGate, the apply_gate hook
    compares the row's schema_version against current_version and
    runs the migration chain if behind.

    Returns 1 when the schema entry omits the field (back-compat for
    test fixtures that build StateSchemaDef instances by hand without
    the version key). Raises KeyError if schema_id is not registered.
    """
    return int(STATE_SCHEMAS[schema_id].get("version") or 1)


_DEFAULT_BY_TYPE: dict[str, object] = {
    "string": "",
    "integer": 0,
    "number": 0,
    "boolean": False,
    "array": [],
    "object": {},
    "null": None,
}


def materialize_default(kind_name: str) -> dict:
    """Build a default slot payload for the named state schema.

    Used by the retrofit gardener handler (state-protocol v1 piece 6) when
    an instance of a state-bearing class surfaces with no recorded state.
    The returned dict satisfies the schema's required slots with
    type-appropriate empties so subsequent JSON-Schema validation passes.
    Optional slots are NOT pre-filled -- agents add them on first delta.

    Type mapping mirrors JSON Schema semantics: string="", integer/number=0,
    boolean=False, array=[], object={}, null=None. Enum slots get the first
    enumerated value (e.g. task_state.status -> "open"). Slots with a JSON
    Schema "default" field win over the type default. Nullable slots
    (type: ["string", "null"]) default to null. Raises KeyError if
    kind_name is not in STATE_SCHEMAS.
    """
    schema_def = STATE_SCHEMAS[kind_name]
    js = schema_def.get("json_schema") or {}
    props = js.get("properties") or {}
    required = set(js.get("required") or [])
    out: dict = {}
    for slot_name in required:
        slot_schema = props.get(slot_name) or {}
        if "default" in slot_schema:
            out[slot_name] = slot_schema["default"]
            continue
        slot_type = slot_schema.get("type")
        # Nullable union types (e.g. ["string", "null"]) default to None.
        if isinstance(slot_type, list):
            if "null" in slot_type:
                out[slot_name] = None
                continue
            slot_type = slot_type[0] if slot_type else "string"
        # Enum slots take the first enumerated value as the default
        # (matches "open" for task_state.status, etc.).
        enum_values = slot_schema.get("enum")
        if enum_values:
            out[slot_name] = enum_values[0]
            continue
        out[slot_name] = _DEFAULT_BY_TYPE.get(slot_type, None)
    return out
