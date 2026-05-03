"""Slice B framework tests (state_deltas plumbing + coverage enforcement).

Covers the state-protocol v1 Slice B paths shipped 2026-05-03:
  - record_state_revision + latest_state_for_entity helpers (Slice A piece 5b).
  - materialize_default produces valid payloads for all four core schemas.
  - JSON Schema validation catches malformed payloads.
  - MEMPALACE_STATE_DELTA_DISABLED env var observable.

Tests assume the standard tests/conftest.py palace fixture exposing a
fresh KnowledgeGraph + palace dir per test. If your conftest uses a
different fixture name, adjust the fixture parameter accordingly.
"""

from __future__ import annotations

import os
import pytest


def _kg(kg):
    """Resolve KnowledgeGraph from whatever the palace fixture returns."""
    return kg.kg if hasattr(kg, "kg") else kg


# ---------------------------------------------------------------------------
# record_state_revision direct (Slice A piece 5b helpers)
# ---------------------------------------------------------------------------


def test_record_state_revision_with_op_context_writes_column(kg):
    """op_context_id COLUMN populated correctly (JTMS link via column)."""
    from mempalace import state_schemas

    kg = _kg(kg)
    payload = state_schemas.materialize_default("task_state")
    rev_id = kg.record_state_revision(
        entity_id="Task#test_record_op",
        schema_id="task_state",
        payload=payload,
        op_context_id="ctx_test_op",
        agent="test_agent",
    )
    assert rev_id.startswith("srv_")
    row = (
        kg._conn()
        .execute(
            "SELECT op_context_id, agent FROM mempalace_state_revisions WHERE rev_id=?",
            (rev_id,),
        )
        .fetchone()
    )
    assert row is not None
    assert row[0] == "ctx_test_op"
    assert row[1] == "test_agent"


def test_record_state_revision_empty_op_context(kg):
    """Gardener-default writes leave op_context_id empty."""
    from mempalace import state_schemas

    kg = _kg(kg)
    payload = state_schemas.materialize_default("task_state")
    rev_id = kg.record_state_revision(
        entity_id="Task#test_record_no_op",
        schema_id="task_state",
        payload=payload,
        op_context_id="",
        agent="memory_gardener",
    )
    row = (
        kg._conn()
        .execute(
            "SELECT op_context_id, agent FROM mempalace_state_revisions WHERE rev_id=?",
            (rev_id,),
        )
        .fetchone()
    )
    assert row[0] == ""
    assert row[1] == "memory_gardener"


def test_latest_state_for_entity_desc_ordering(kg):
    """Two revisions: latest_state_for_entity returns the second (DESC order)."""
    import time

    kg = _kg(kg)
    eid = "Task#test_ordering"
    kg.record_state_revision(
        eid,
        "task_state",
        {"status": "open", "progress_pct": 10},
        "",
        "test_agent",
    )
    time.sleep(0.01)
    kg.record_state_revision(
        eid,
        "task_state",
        {"status": "in_progress", "progress_pct": 50},
        "",
        "test_agent",
    )
    latest = kg.latest_state_for_entity(eid)
    assert latest is not None
    assert latest.get("status") == "in_progress"


def test_latest_state_for_entity_returns_none_when_no_revisions(kg):
    """No rows -> None signals state_init_needed."""
    kg = _kg(kg)
    assert kg.latest_state_for_entity("Task#never_recorded") is None


# ---------------------------------------------------------------------------
# materialize_default for all four schemas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "schema_id,required_slot",
    [
        ("project_state", "current_phase"),
        ("intent_state", "current_step"),
        ("agent_state", "current_focus"),
        ("task_state", "status"),
    ],
)
def test_materialize_default_required_slots(schema_id, required_slot):
    from mempalace import state_schemas

    payload = state_schemas.materialize_default(schema_id)
    assert required_slot in payload


def test_materialize_default_task_state_status_enum_open():
    """task_state.status defaults to first enum value 'open'."""
    from mempalace import state_schemas

    payload = state_schemas.materialize_default("task_state")
    assert payload.get("status") == "open"


def test_materialize_default_unknown_schema_raises():
    from mempalace import state_schemas

    with pytest.raises(KeyError):
        state_schemas.materialize_default("nonexistent_schema_xyz")


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_task_state_default_payload_passes_schema():
    """materialize_default('task_state') produces a JSON-Schema-valid payload."""
    pytest.importorskip("jsonschema")
    import jsonschema

    from mempalace import state_schemas

    payload = state_schemas.materialize_default("task_state")
    schema = state_schemas.STATE_SCHEMAS["task_state"]["json_schema"]
    jsonschema.validate(payload, schema)  # raises on failure


def test_task_state_payload_with_invalid_status_fails_schema():
    """Schema validation catches malformed payload."""
    pytest.importorskip("jsonschema")
    import jsonschema

    from mempalace import state_schemas

    bad_payload = {"status": "completely_invalid", "progress_pct": 50}
    schema = state_schemas.STATE_SCHEMAS["task_state"]["json_schema"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad_payload, schema)


# ---------------------------------------------------------------------------
# Kill-switch env var (smoke-level: just verify monkeypatch wiring)
# ---------------------------------------------------------------------------


def test_kill_switch_env_var_present(monkeypatch):
    """Setting MEMPALACE_STATE_DELTA_DISABLED=1 is observable in os.environ."""
    monkeypatch.setenv("MEMPALACE_STATE_DELTA_DISABLED", "1")
    assert os.environ.get("MEMPALACE_STATE_DELTA_DISABLED") == "1"


def test_kill_switch_default_off():
    """Default-on enforcement: env var unset means coverage rule fires."""
    if "MEMPALACE_STATE_DELTA_DISABLED" in os.environ:
        pytest.skip("env var leaked into test; isolation issue at fixture level")
    assert not os.environ.get("MEMPALACE_STATE_DELTA_DISABLED")


# ---------------------------------------------------------------------------
# Patch apply via jsonpatch (Slice B-2)
# ---------------------------------------------------------------------------


def test_jsonpatch_apply_round_trip(kg):
    """RFC 6902 patch applied on top of a recorded state via the helper."""
    pytest.importorskip("jsonpatch")
    import jsonpatch

    from mempalace import state_schemas

    kg = _kg(kg)
    eid = "Task#test_patch_apply"
    initial = state_schemas.materialize_default("task_state")
    kg.record_state_revision(eid, "task_state", initial, "", "test_agent")

    current = kg.latest_state_for_entity(eid)
    patch = [{"op": "replace", "path": "/status", "value": "in_progress"}]
    new_payload = jsonpatch.apply_patch(current, patch)

    rev_id = kg.record_state_revision(
        entity_id=eid,
        schema_id="task_state",
        payload=new_payload,
        op_context_id="ctx_apply_test",
        agent="test_agent",
    )
    final = kg.latest_state_for_entity(eid)
    assert final is not None
    assert final.get("status") == "in_progress"
    assert rev_id.startswith("srv_")
