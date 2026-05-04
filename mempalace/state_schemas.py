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


_PROJECT_STATE: StateSchemaDef = {
    "version": 1,
    "json_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "current_phase": {"type": "string"},
            "active_branches": {"type": "array", "items": {"type": "string"}},
            "blockers": {"type": "array", "items": {"type": "string"}},
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
            "open_questions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["current_phase"],
    },
    "slot_descriptions": {
        "current_phase": "Free-form phase label for the project right now (e.g. 'design', 'slice-a-implementation', 'staged-rollout').",
        "active_branches": "Git branches or workstreams in flight on this project; empty list when only main is active.",
        "blockers": "Things stopping forward progress -- missing answers, broken deps, awaiting reviews. One entry per discrete blocker.",
        "recent_milestones": "Recently-completed milestones with optional date. Cap at last ~5; older milestones consolidate into project records.",
        "open_questions": "Design or scope questions awaiting resolution. Empty list when nothing is pending.",
    },
    "parent_schema_id": None,
}


_INTENT_STATE: StateSchemaDef = {
    "version": 1,
    "json_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "current_step": {"type": "string"},
            "progress_pct": {"type": "integer", "minimum": 0, "maximum": 100},
            "blockers": {"type": "array", "items": {"type": "string"}},
            "latest_observation": {"type": "string"},
            "open_questions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["current_step"],
    },
    "slot_descriptions": {
        "current_step": "Short description of what the intent is doing right now (verb + object).",
        "progress_pct": "Estimated completion 0-100. Use coarse buckets (0/25/50/75/100); precision past 5 is noise.",
        "blockers": "Things stopping the intent from progressing. Empty when unblocked.",
        "latest_observation": "Last meaningful finding or decision since the prior op declaration. One short sentence.",
        "open_questions": "Questions the intent has not yet resolved. Empty when nothing is pending.",
    },
    "parent_schema_id": None,
}


_AGENT_STATE: StateSchemaDef = {
    "version": 1,
    "json_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "current_focus": {"type": "string"},
            "active_intent_id": {"type": ["string", "null"]},
            "recent_findings": {"type": "array", "items": {"type": "string"}},
            "pending_followups": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["current_focus"],
    },
    "slot_descriptions": {
        "current_focus": "What the agent is concentrating on right now -- a topic or workstream label, not a tool call.",
        "active_intent_id": "Id of the intent the agent is currently executing, or null between intents.",
        "recent_findings": "Recently-surfaced facts or decisions the agent learned this session. Cap at last ~5.",
        "pending_followups": "Items the agent has noted to come back to. Each entry is one sentence describing the action.",
    },
    "parent_schema_id": None,
}


_TASK_STATE: StateSchemaDef = {
    "version": 1,
    "json_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {
                "type": "string",
                "enum": ["open", "in_progress", "blocked", "done", "cancelled"],
            },
            "assignee": {"type": ["string", "null"]},
            "due_date": {"type": ["string", "null"], "format": "date"},
            "blockers": {"type": "array", "items": {"type": "string"}},
            "progress_pct": {"type": "integer", "minimum": 0, "maximum": 100},
        },
        "required": ["status"],
    },
    "slot_descriptions": {
        "status": "Lifecycle state. open=not started; in_progress=being worked; blocked=waiting on dep; done=complete; cancelled=abandoned.",
        "assignee": "Agent or person responsible. Null when unassigned.",
        "due_date": "Target completion date in ISO format, or null when no deadline.",
        "blockers": "Specific things stopping forward progress. Empty list when status != blocked.",
        "progress_pct": "Estimated completion 0-100. Coarse buckets only.",
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
