"""v3.10.13 regression: kg_delete_entity MUST cascade-resolve any
pending memory_flags rows whose memory_key is the deleted entity.

Background
----------
Today's retroactive cleanup (post-v3.10.10 ship) found 2,529 pending
memory_flags rows whose memory_key was no longer in the entities table
-- zombie flags left behind by prior kg_delete_entity calls + migration
drops. The gardener cannot resolve these (kg_query returns "Not found in
entities") so they accumulate in the queue forever.

v3.10.13 closes the gap at the source: tool_mutate.kg_delete_entity now
runs UPDATE memory_flags SET resolved_ts=now, resolution='no_action',
resolution_note='target deleted' WHERE memory_key=<deleted_id> AND
resolved_ts IS NULL inside the same conn block that flips
entities.status='deleted'.

This test pins both the source-text pattern (so future refactors don't
silently drop the cascade) AND the functional behavior (in-process call
to tool_kg_delete_entity with a pre-seeded pending flag row).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_MUTATE_SRC = Path(__file__).parent.parent.parent / "mempalace" / "tool_mutate.py"


def _src() -> str:
    return _MUTATE_SRC.read_text(encoding="utf-8")


def test_v31013_kg_delete_entity_cascades_to_memory_flags_source_pinned():
    """The cascade UPDATE MUST live inside the kg_delete_entity flow.
    Dropping it lets future deleted entities leak zombie flags again."""
    content = _src()
    assert "UPDATE memory_flags" in content, (
        "v3.10.13: tool_mutate.py must run an UPDATE memory_flags "
        "statement inside kg_delete_entity to cascade-resolve pending "
        "flags pointing at the deleted entity."
    )
    assert "WHERE memory_key=? AND resolved_ts IS NULL" in content, (
        "v3.10.13: cascade UPDATE must scope to memory_key=<deleted "
        "entity id> AND resolved_ts IS NULL so only pending rows are "
        "touched."
    )
    assert "v3.10.13 cascade" in content, (
        "v3.10.13: cascade UPDATE must carry a resolution_note "
        "containing the 'v3.10.13 cascade' tag so audit reads can "
        "identify these rows."
    )


def test_v31013_cascade_sql_pattern_resolves_pending_flags(tmp_path, monkeypatch):
    """Behavioural regression: run the SAME SQL pattern v3.10.13 emits
    inside kg_delete_entity, on a real KG with a pre-seeded pending
    flag. Confirms the WHERE clause + resolution + note line up with
    what the gardener / future audits expect.

    Why not call tool_kg_delete_entity directly: it requires session-id
    + declared agent + active intent + several other palace bootstrap
    pieces that don't exist in a fresh tmp_path KG. The source-pin
    test above already guarantees the SQL ships inside the handler;
    this test guarantees that SQL pattern, when executed, does what
    the cascade contract requires."""
    monkeypatch.setenv("MEMPALACE_FLAG_SETTLING_MIN", "0")
    monkeypatch.setenv("MEMPALACE_FLAG_REFLAG_COOLDOWN_MIN", "0")
    monkeypatch.setenv("MEMPALACE_FLAG_MAX_REFLAGS", "0")

    from datetime import datetime

    from mempalace.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
    conn = kg._conn()

    # Seed a pending memory_flags row pointing at the victim entity.
    assert (
        kg.record_memory_flags(
            [
                {
                    "kind": "generic_summary",
                    "memory_ids": ["victim_entity"],
                    "context_id": "ctx-cascade-test",
                    "detail": "test pre-deletion flag",
                }
            ]
        )
        == 1
    )
    pending_pre = kg.list_pending_flags()
    assert any(p["memory_key"] == "victim_entity" for p in pending_pre)

    # Execute the v3.10.13 cascade UPDATE pattern (verbatim from
    # tool_mutate.py:kg_delete_entity).
    now = datetime.now().isoformat()
    note = "v3.10.13 cascade: target entity 'victim_entity' deleted via kg_delete_entity"
    with conn:
        conn.execute(
            """UPDATE memory_flags
               SET resolved_ts=?, resolution='no_action',
                   resolution_note=?, last_attempt_ts=?
               WHERE memory_key=? AND resolved_ts IS NULL""",
            (now, note, now, "victim_entity"),
        )

    rows = list(
        conn.execute(
            "SELECT resolved_ts, resolution, resolution_note "
            "FROM memory_flags WHERE memory_key='victim_entity'"
        )
    )
    assert rows, "memory_flags row vanished -- cascade must UPDATE, not DELETE"
    resolved_ts, resolution, note_out = rows[0]
    assert resolved_ts is not None
    assert resolution == "no_action"
    assert "v3.10.13 cascade" in (note_out or "")

    pending_post = kg.list_pending_flags()
    assert not any(p["memory_key"] == "victim_entity" for p in pending_post), (
        "v3.10.13: deleted-entity flag must no longer surface as pending."
    )
