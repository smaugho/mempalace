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


def _ensure_entity(kg, name):
    """Seed an entities row so record_state_revision's phantom-state
    guard accepts the write. Slice C-1 hardening (2026-05-03) refuses
    revisions against entity_ids that have no corresponding entities
    row -- tests that fabricate entity_ids must call this first.
    """
    from datetime import datetime as _dt

    eid = kg._entity_id(name)
    conn = kg._conn()
    conn.execute(
        "INSERT OR IGNORE INTO entities "
        "(id, name, kind, status, last_touched) "
        "VALUES (?, ?, 'entity', 'active', ?)",
        (eid, name, _dt.now().isoformat()),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# record_state_revision direct (Slice A piece 5b helpers)
# ---------------------------------------------------------------------------


def test_record_state_revision_with_op_context_writes_column(kg):
    """op_context_id COLUMN populated correctly (JTMS link via column)."""
    from mempalace import state_schemas

    kg = _kg(kg)
    _ensure_entity(kg, "Task#test_record_op")
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
    _ensure_entity(kg, "Task#test_record_no_op")
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
    _ensure_entity(kg, eid)
    kg.record_state_revision(
        eid,
        "task_state",
        {"status": "open", "subtodos": []},
        "",
        "test_agent",
    )
    time.sleep(0.01)
    kg.record_state_revision(
        eid,
        "task_state",
        {"status": "in_progress", "subtodos": []},
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
        # State-protocol v2 (Adrian 2026-05-04) required slots. v1's
        # intent_state.current_step / progress_pct were replaced by
        # intent_state.todos. Other schemas kept their primary required
        # field (current_phase / current_focus / status).
        ("project_state", "current_phase"),
        ("intent_state", "todos"),
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

    bad_payload = {"status": "completely_invalid", "subtodos": []}
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
    _ensure_entity(kg, eid)
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


# ---------------------------------------------------------------------------
# Slice C-2 schema validation hardening (2026-05-03)
# ---------------------------------------------------------------------------


def test_record_state_revision_unknown_schema_id_raises(kg):
    """schema_id not in STATE_SCHEMAS -> ValueError."""
    kg = _kg(kg)
    _ensure_entity(kg, "Task#test_unknown_schema")
    with pytest.raises(ValueError, match="not a known STATE_SCHEMAS key"):
        kg.record_state_revision(
            entity_id="Task#test_unknown_schema",
            schema_id="nonexistent_schema_xyz",
            payload={"status": "open"},
            op_context_id="",
            agent="test_agent",
        )


def test_record_state_revision_malformed_payload_raises(kg):
    """Payload that fails jsonschema -> ValueError."""
    pytest.importorskip("jsonschema")

    kg = _kg(kg)
    _ensure_entity(kg, "Task#test_bad_payload")
    with pytest.raises(ValueError, match="failed schema"):
        kg.record_state_revision(
            entity_id="Task#test_bad_payload",
            schema_id="task_state",
            payload={"status": "completely_invalid_status"},
            op_context_id="",
            agent="test_agent",
        )


def test_record_state_revision_empty_schema_id_skips_validation(kg):
    """Empty schema_id -> validation skipped (gardener-default path)."""
    kg = _kg(kg)
    _ensure_entity(kg, "Task#test_empty_schema")
    # Even a payload that would fail task_state must be accepted when
    # schema_id is empty -- this is the gardener / pre-Slice-C2 path.
    rev_id = kg.record_state_revision(
        entity_id="Task#test_empty_schema",
        schema_id="",
        payload={"arbitrary": "shape"},
        op_context_id="",
        agent="test_agent",
    )
    assert rev_id.startswith("srv_")


def test_record_state_revision_phantom_entity_raises(kg):
    """Slice C-1 phantom-state guard: entity must exist before write."""
    kg = _kg(kg)
    with pytest.raises(ValueError, match="phantom state writes are blocked"):
        kg.record_state_revision(
            entity_id="Task#never_declared_phantom",
            schema_id="",
            payload={"any": "shape"},
            op_context_id="",
            agent="test_agent",
        )


def test_record_state_revision_deleted_entity_raises(kg):
    """Slice C-1 deleted-status guard: writes refused on soft-deleted entities."""
    from datetime import datetime as _dt

    kg = _kg(kg)
    _ensure_entity(kg, "Task#test_will_be_deleted")
    eid = kg._entity_id("Task#test_will_be_deleted")
    conn = kg._conn()
    conn.execute(
        "UPDATE entities SET status='deleted', last_touched=? WHERE id=?",
        (_dt.now().isoformat(), eid),
    )
    conn.commit()
    with pytest.raises(ValueError, match="soft-deleted"):
        kg.record_state_revision(
            entity_id="Task#test_will_be_deleted",
            schema_id="",
            payload={"any": "shape"},
            op_context_id="",
            agent="test_agent",
        )


def test_latest_state_for_entity_filters_deleted(kg):
    """Slice C-1 read-time filter: deleted entities surface no state."""
    from datetime import datetime as _dt

    kg = _kg(kg)
    _ensure_entity(kg, "Task#test_read_after_delete")
    eid = kg._entity_id("Task#test_read_after_delete")
    kg.record_state_revision(
        "Task#test_read_after_delete",
        "task_state",
        {"status": "open", "subtodos": []},
        "",
        "test_agent",
    )
    # State is readable while active.
    assert kg.latest_state_for_entity("Task#test_read_after_delete") is not None
    # Soft-delete then verify state filtered out.
    conn = kg._conn()
    conn.execute(
        "UPDATE entities SET status='deleted', last_touched=? WHERE id=?",
        (_dt.now().isoformat(), eid),
    )
    conn.commit()
    assert kg.latest_state_for_entity("Task#test_read_after_delete") is None


def test_merge_entities_cascades_state_revisions(kg):
    """Slice C-1 merge cascade: source's state history follows to target id."""
    kg = _kg(kg)
    _ensure_entity(kg, "Task#merge_source")
    _ensure_entity(kg, "Task#merge_target")
    kg.record_state_revision(
        "Task#merge_source",
        "task_state",
        {"status": "open", "subtodos": []},
        "",
        "test_agent",
    )
    # Pre-merge: source has state, target empty.
    assert kg.latest_state_for_entity("Task#merge_source") is not None
    # latest_state_for_entity('Task#merge_target') BEFORE merge: target_id
    # has no rows; return None.
    assert kg.latest_state_for_entity("Task#merge_target") is None
    # Merge source into target.
    kg.merge_entities("Task#merge_source", "Task#merge_target")
    # Post-merge: target sees the source's history (alias resolution +
    # cascade UPDATE on entity_id).
    latest = kg.latest_state_for_entity("Task#merge_target")
    assert latest is not None
    assert latest.get("status") == "open"


def test_kg_delete_simulated_cascade_removes_state_revisions(kg):
    """Slice C-1 delete cascade contract: status='deleted' + DELETE
    on mempalace_state_revisions WHERE entity_id=? together remove
    all history for the entity.

    Tests the SQL contract that tool_kg_delete_entity (in
    mempalace/tool_mutate.py) relies on. We simulate the cascade
    here rather than calling tool_kg_delete_entity itself because
    the tool requires the full _STATE harness (session id, Chroma
    collections, WAL log) which the slice_b kg fixture doesn't
    provide. If the cascade SQL changes shape, this test catches
    the contract drift."""
    kg = _kg(kg)
    _ensure_entity(kg, "Task#test_delete_cascade")
    eid = kg._entity_id("Task#test_delete_cascade")
    kg.record_state_revision(
        "Task#test_delete_cascade",
        "task_state",
        {"status": "open", "subtodos": []},
        "",
        "test_agent",
    )
    # Two revisions to confirm DELETE removes history not just latest.
    kg.record_state_revision(
        "Task#test_delete_cascade",
        "task_state",
        {"status": "in_progress", "subtodos": []},
        "",
        "test_agent",
    )
    pre_count = (
        kg._conn()
        .execute(
            "SELECT COUNT(*) FROM mempalace_state_revisions WHERE entity_id=?",
            (eid,),
        )
        .fetchone()[0]
    )
    assert pre_count == 2
    # Simulate kg_delete_entity's cascade: status flip + DELETE.
    conn = kg._conn()
    conn.execute("UPDATE entities SET status='deleted' WHERE id=?", (eid,))
    conn.execute("DELETE FROM mempalace_state_revisions WHERE entity_id=?", (eid,))
    conn.commit()
    post_count = (
        kg._conn()
        .execute(
            "SELECT COUNT(*) FROM mempalace_state_revisions WHERE entity_id=?",
            (eid,),
        )
        .fetchone()[0]
    )
    assert post_count == 0
    # And read-time guard: latest_state_for_entity returns None for
    # the deleted entity even though we just verified all rows are
    # gone -- the status filter would catch it even if a revision
    # somehow leaked back in.
    assert kg.latest_state_for_entity("Task#test_delete_cascade") is None


# ---------------------------------------------------------------------------
# Phase 6 lazy-migration-at-injection (2026-05-04)
# ---------------------------------------------------------------------------


def _register_fake_migration(monkeypatch, schema_id, from_v, to_v, migrate_fn):
    """Register a fake migration module under
    mempalace.state_migrations.{schema_id}.v{from}_to_v{to} for the
    duration of a test. Uses sys.modules to bypass the filesystem
    import path so tests don't need to drop real files into the source
    tree.
    """
    import sys
    import types

    pkg_path = f"mempalace.state_migrations.{schema_id}"
    mod_path = f"{pkg_path}.v{from_v}_to_v{to_v}"
    if pkg_path not in sys.modules:
        parent = types.ModuleType(pkg_path)
        parent.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, pkg_path, parent)
    mod = types.ModuleType(mod_path)
    mod.migrate = migrate_fn
    monkeypatch.setitem(sys.modules, mod_path, mod)


def test_record_state_revision_stamps_schema_version_default_1(kg):
    """Phase 6: schema_version column populated, defaults to 1 for v1 schemas."""
    from mempalace import state_schemas

    kg = _kg(kg)
    _ensure_entity(kg, "Task#test_version_stamp")
    payload = state_schemas.materialize_default("task_state")
    rev_id = kg.record_state_revision(
        entity_id="Task#test_version_stamp",
        schema_id="task_state",
        payload=payload,
        op_context_id="",
        agent="test_agent",
    )
    row = (
        kg._conn()
        .execute(
            "SELECT schema_version FROM mempalace_state_revisions WHERE rev_id=?",
            (rev_id,),
        )
        .fetchone()
    )
    assert row is not None
    expected = state_schemas.STATE_SCHEMAS["task_state"]["version"]
    assert row[0] == expected


def test_record_state_revision_empty_schema_version_floor_1(kg):
    """Empty schema_id (gardener-default writes) gets version=1 floor."""
    kg = _kg(kg)
    _ensure_entity(kg, "Task#test_empty_schema_version")
    rev_id = kg.record_state_revision(
        entity_id="Task#test_empty_schema_version",
        schema_id="",
        payload={"any": "shape"},
        op_context_id="",
        agent="test_agent",
    )
    row = (
        kg._conn()
        .execute(
            "SELECT schema_version FROM mempalace_state_revisions WHERE rev_id=?",
            (rev_id,),
        )
        .fetchone()
    )
    assert row[0] == 1


def test_apply_pending_migrations_no_op_when_at_or_above_current():
    """Phase 6 runner is a no-op when from >= to."""
    from mempalace.state_migrations import apply_pending_migrations

    payload = {"status": "open"}
    out = apply_pending_migrations(payload, "task_state", 1, 1)
    assert out == payload
    out = apply_pending_migrations(payload, "task_state", 5, 2)
    assert out == payload


def test_apply_pending_migrations_single_step(monkeypatch):
    """Phase 6 runner walks a single migration step."""
    from mempalace.state_migrations import apply_pending_migrations

    def _migrate_v1_to_v2(payload):
        return {**payload, "due_date": payload.get("due_date") or "unset"}

    _register_fake_migration(monkeypatch, "task_state", 1, 2, _migrate_v1_to_v2)
    out = apply_pending_migrations({"status": "open"}, "task_state", 1, 2)
    assert out == {"status": "open", "due_date": "unset"}


def test_apply_pending_migrations_multi_step_chain(monkeypatch):
    """Phase 6 runner walks v1 -> v2 -> v3 in order."""
    from mempalace.state_migrations import apply_pending_migrations

    def _v1_to_v2(payload):
        return {**payload, "step1": True}

    def _v2_to_v3(payload):
        return {**payload, "step2": True}

    _register_fake_migration(monkeypatch, "task_state", 1, 2, _v1_to_v2)
    _register_fake_migration(monkeypatch, "task_state", 2, 3, _v2_to_v3)
    out = apply_pending_migrations({"status": "open"}, "task_state", 1, 3)
    assert out == {"status": "open", "step1": True, "step2": True}


def test_apply_pending_migrations_missing_module_raises():
    """Phase 6 runner raises StateMigrationError when module is absent."""
    from mempalace.state_migrations import StateMigrationError, apply_pending_migrations

    with pytest.raises(StateMigrationError, match="missing migration module"):
        apply_pending_migrations({"status": "open"}, "task_state", 1, 99)


def test_apply_pending_migrations_non_callable_migrate_raises(monkeypatch):
    """Phase 6 runner raises when module exists but migrate isn't callable."""
    import sys
    import types

    from mempalace.state_migrations import StateMigrationError, apply_pending_migrations

    pkg = types.ModuleType("mempalace.state_migrations.task_state")
    pkg.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mempalace.state_migrations.task_state", pkg)
    mod = types.ModuleType("mempalace.state_migrations.task_state.v1_to_v2")
    mod.migrate = "not_a_callable"  # noqa
    monkeypatch.setitem(sys.modules, "mempalace.state_migrations.task_state.v1_to_v2", mod)
    with pytest.raises(StateMigrationError, match="does not export a callable"):
        apply_pending_migrations({"status": "open"}, "task_state", 1, 2)


def test_apply_pending_migrations_non_dict_return_raises(monkeypatch):
    """Phase 6 runner rejects non-dict return values."""
    from mempalace.state_migrations import StateMigrationError, apply_pending_migrations

    def _bad_migrate(payload):
        return "not_a_dict"

    _register_fake_migration(monkeypatch, "task_state", 1, 2, _bad_migrate)
    with pytest.raises(StateMigrationError, match="returned str"):
        apply_pending_migrations({"status": "open"}, "task_state", 1, 2)


def test_migrate_state_for_entities_skips_entity_without_revisions(kg):
    """KG helper skips entities that have no state revisions yet."""
    kg = _kg(kg)
    _ensure_entity(kg, "Task#no_state_yet")
    out = kg.migrate_state_for_entities(["Task#no_state_yet"])
    assert "Task#no_state_yet" in out
    assert out["Task#no_state_yet"]["migrated"] is False
    assert out["Task#no_state_yet"]["version"] == 0


def test_migrate_state_for_entities_no_op_when_at_current(kg):
    """KG helper no-ops when latest revision is already at current version."""
    from mempalace import state_schemas

    kg = _kg(kg)
    _ensure_entity(kg, "Task#at_current")
    payload = state_schemas.materialize_default("task_state")
    kg.record_state_revision(
        entity_id="Task#at_current",
        schema_id="task_state",
        payload=payload,
        op_context_id="",
        agent="test_agent",
    )
    pre_count = (
        kg._conn()
        .execute(
            "SELECT COUNT(*) FROM mempalace_state_revisions WHERE entity_id=?",
            (kg._entity_id("Task#at_current"),),
        )
        .fetchone()[0]
    )
    from mempalace import state_schemas as _ss

    expected_version = _ss.STATE_SCHEMAS["task_state"]["version"]
    out = kg.migrate_state_for_entities(["Task#at_current"])
    assert out["Task#at_current"]["migrated"] is False
    assert out["Task#at_current"]["version"] == expected_version
    post_count = (
        kg._conn()
        .execute(
            "SELECT COUNT(*) FROM mempalace_state_revisions WHERE entity_id=?",
            (kg._entity_id("Task#at_current"),),
        )
        .fetchone()[0]
    )
    assert post_count == pre_count


def test_migrate_state_for_entities_runs_chain_and_writes_new_revision(kg, monkeypatch):
    """KG helper migrates a stale entity + writes a new revision."""
    from mempalace import state_schemas

    kg = _kg(kg)
    _ensure_entity(kg, "Task#needs_migration")
    # Write a revision at the CURRENT version, then monkeypatch the registry
    # to bump target one above current so the migrate path triggers.
    current = state_schemas.STATE_SCHEMAS["task_state"]["version"]
    kg.record_state_revision(
        entity_id="Task#needs_migration",
        schema_id="task_state",
        payload={"status": "open"},
        op_context_id="",
        agent="test_agent",
    )
    eid = kg._entity_id("Task#needs_migration")
    target = current + 1
    monkeypatch.setitem(state_schemas.STATE_SCHEMAS["task_state"], "version", target)

    def _migrate(payload):
        # Schema is strict additionalProperties=False; mutate an allowed field.
        return {**payload, "due_date": "2026-12-31"}

    _register_fake_migration(monkeypatch, "task_state", current, target, _migrate)
    out = kg.migrate_state_for_entities(["Task#needs_migration"])
    assert out["Task#needs_migration"]["migrated"] is True, f"out={out}"
    assert out["Task#needs_migration"]["version"] == target
    row = (
        kg._conn()
        .execute(
            "SELECT schema_version, payload FROM mempalace_state_revisions "
            "WHERE entity_id = ? ORDER BY created_at DESC LIMIT 1",
            (eid,),
        )
        .fetchone()
    )
    assert row[0] == target
    import json

    payload = json.loads(row[1])
    assert payload.get("due_date") == "2026-12-31"
    assert payload.get("status") == "open"


def test_migrate_state_for_entities_skips_deleted(kg):
    """KG helper skips soft-deleted entities even if they have revisions."""
    from datetime import datetime as _dt

    kg = _kg(kg)
    _ensure_entity(kg, "Task#deleted_with_state")
    kg.record_state_revision(
        entity_id="Task#deleted_with_state",
        schema_id="task_state",
        payload={"status": "open"},
        op_context_id="",
        agent="test_agent",
    )
    eid = kg._entity_id("Task#deleted_with_state")
    conn = kg._conn()
    conn.execute(
        "UPDATE entities SET status='deleted', last_touched=? WHERE id=?",
        (_dt.now().isoformat(), eid),
    )
    conn.commit()
    out = kg.migrate_state_for_entities(["Task#deleted_with_state"])
    assert out["Task#deleted_with_state"]["migrated"] is False
    assert out["Task#deleted_with_state"]["version"] == 0


def test_migrate_state_for_entities_empty_input(kg):
    """KG helper returns empty dict for empty input."""
    kg = _kg(kg)
    assert kg.migrate_state_for_entities([]) == {}
    assert kg.migrate_state_for_entities(None) == {}


pytestmark = pytest.mark.integration
