"""Integration tests for v3.2.6 vec_rowid_map rename + dual FK + cascade trigger.

Adrian directive 2026-05-12: the vec_rowid_map.entity_id column was
misleadingly named -- it stores logical vec record ids spanning four
namespaces (bare {eid}, multi-view {eid}__v{i}, context-view
{cid}_v{i}, triple_id). v3.2.6:

  * renames entity_id -> logical_id (truth in naming),
  * adds entity_id_ref + triple_id_ref FK columns with ON DELETE
    CASCADE pointing at entities(id) and triples(id) respectively,
  * adds a BEFORE DELETE trigger on vec_rowid_map that deletes the
    matching vec_palace row, so deleting an entity (or triple) now
    cascades through rowid_map and on to the virtual table.

These tests gate the rename, the FK populate on writes, the cascade
chain, and the legacy-palace migration backfill.
"""

from __future__ import annotations

import sqlite3

import pytest

from mempalace.knowledge_graph import KnowledgeGraph
from mempalace.sqlite_vec_store import SqliteVecVectorStore


pytestmark = pytest.mark.integration


def _make_palace(tmp_path):
    """Create a fresh KG + vector store sharing the same SQLite file."""
    palace = tmp_path / "palace_v326"
    palace.mkdir()
    db_path = str(palace / "knowledge_graph.sqlite3")
    kg = KnowledgeGraph(db_path=db_path)
    # Construct the vector store -- this runs the bootstrap CREATE +
    # the v326 migration helper on the same db.
    store = SqliteVecVectorStore(palace_path=str(palace))
    return kg, store, db_path


def _probe_conn(db_path: str) -> sqlite3.Connection:
    """Open a probe connection with sqlite-vec loaded so vec_palace
    is queryable. Plain ``sqlite3.connect`` cannot read the vec0
    virtual table without the extension."""
    conn = sqlite3.connect(db_path)
    try:
        conn.enable_load_extension(True)
        try:
            import sqlite_vec

            sqlite_vec.load(conn)
        finally:
            conn.enable_load_extension(False)
    except Exception:
        pass
    return conn


# ── Schema + naming -----------------------------------------------------


def test_v326_bootstrap_creates_new_schema_for_fresh_palace(tmp_path):
    """Fresh palace must have logical_id + entity_id_ref + triple_id_ref."""
    _, _, db_path = _make_palace(tmp_path)
    conn = sqlite3.connect(db_path)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(vec_rowid_map)").fetchall()}
    assert "logical_id" in cols, f"logical_id missing; got {cols!r}"
    assert "entity_id_ref" in cols, f"entity_id_ref missing; got {cols!r}"
    assert "triple_id_ref" in cols, f"triple_id_ref missing; got {cols!r}"
    assert "entity_id" not in cols, (
        f"old column name 'entity_id' should not exist post-v3.2.6; got {cols!r}"
    )


def test_v326_trigger_exists(tmp_path):
    """BEFORE DELETE trigger must exist on vec_rowid_map."""
    _, _, db_path = _make_palace(tmp_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='vec_rowid_map'"
    ).fetchone()
    assert row is not None, "expected a BEFORE DELETE trigger on vec_rowid_map"
    assert "cascade_to_vec_palace" in row[0]


# ── Write site: FK refs populated on upsert -----------------------------


def test_v326_write_populates_entity_id_ref_for_bare_entity(tmp_path):
    """Bare {eid} write -> entity_id_ref = eid, triple_id_ref NULL."""
    kg, store, db_path = _make_palace(tmp_path)
    eid = kg.add_entity("fk_v326_bare_entity", kind="entity", content="x")
    # The vector store auto-embeds via fastembed if available -- pass
    # an explicit embedding so the test is deterministic + offline.
    fake_emb = [0.1] * 384
    store.add(
        collection="mempalace_records",
        ids=[eid],
        documents=["x"],
        embeddings=[fake_emb],
    )
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT logical_id, entity_id_ref, triple_id_ref FROM vec_rowid_map "
        "WHERE collection='mempalace_records' AND logical_id=?",
        (eid,),
    ).fetchone()
    assert row is not None, "rowid_map row missing for bare entity write"
    assert row[0] == eid
    assert row[1] == eid, f"entity_id_ref must equal eid for bare row; got {row[1]!r}"
    assert row[2] is None, f"triple_id_ref must be NULL for bare row; got {row[2]!r}"


def test_v326_write_populates_entity_id_ref_for_multi_view(tmp_path):
    """{eid}__v{i} write -> entity_id_ref = eid (parent), triple_id_ref NULL."""
    kg, store, db_path = _make_palace(tmp_path)
    eid = kg.add_entity("fk_v326_multi_view_parent", kind="entity", content="parent")
    view_id = eid + "__v3"
    fake_emb = [0.1] * 384
    store.add(
        collection="mempalace_records",
        ids=[view_id],
        documents=["view doc"],
        embeddings=[fake_emb],
    )
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT entity_id_ref, triple_id_ref FROM vec_rowid_map "
        "WHERE collection='mempalace_records' AND logical_id=?",
        (view_id,),
    ).fetchone()
    assert row is not None
    assert row[0] == eid, f"multi-view entity_id_ref should strip __v3 suffix; got {row[0]!r}"
    assert row[1] is None


def test_v326_write_populates_entity_id_ref_for_context_view(tmp_path):
    """{cid}_v{i} write -> entity_id_ref = cid (parent context entity)."""
    kg, store, db_path = _make_palace(tmp_path)
    cid = kg.add_entity("fk_v326_ctx_parent", kind="context", content="ctx")
    cv_id = cid + "_v2"
    fake_emb = [0.1] * 384
    store.add(
        collection="mempalace_context_views",
        ids=[cv_id],
        documents=["context view doc"],
        embeddings=[fake_emb],
    )
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT entity_id_ref, triple_id_ref FROM vec_rowid_map "
        "WHERE collection='mempalace_context_views' AND logical_id=?",
        (cv_id,),
    ).fetchone()
    assert row is not None
    assert row[0] == cid, f"context-view entity_id_ref should strip _v2; got {row[0]!r}"
    assert row[1] is None


def test_v326_write_populates_triple_id_ref_for_triple_statement(tmp_path):
    """Triple statement write -> triple_id_ref = triple_id, entity_id_ref NULL."""
    kg, store, db_path = _make_palace(tmp_path)
    eid_a = kg.add_entity("fk_v326_triple_sub", kind="entity", content="s")
    eid_b = kg.add_entity("fk_v326_triple_obj", kind="entity", content="o")
    conn = kg._conn()
    triple_id = "t_v326_test"
    conn.execute(
        "INSERT INTO triples (id, subject, predicate, object, valid_from) VALUES (?, ?, ?, ?, ?)",
        (triple_id, eid_a, "related_to", eid_b, "2026-05-12"),
    )
    conn.commit()

    fake_emb = [0.1] * 384
    store.add(
        collection="mempalace_triples",
        ids=[triple_id],
        documents=["t_v326_test statement"],
        embeddings=[fake_emb],
    )
    map_row = (
        _probe_conn(db_path)
        .execute(
            "SELECT entity_id_ref, triple_id_ref FROM vec_rowid_map "
            "WHERE collection='mempalace_triples' AND logical_id=?",
            (triple_id,),
        )
        .fetchone()
    )
    assert map_row is not None
    assert map_row[0] is None
    assert map_row[1] == triple_id


# ── Cascade chain -------------------------------------------------------


def test_v326_delete_entity_cascades_to_rowid_map_and_vec_palace(tmp_path):
    """Deleting an entity removes its rowid_map row (FK cascade) AND its
    vec_palace row (BEFORE DELETE trigger)."""
    kg, store, db_path = _make_palace(tmp_path)
    eid = kg.add_entity("fk_v326_cascade_target", kind="entity", content="x")
    fake_emb = [0.1] * 384
    store.add(
        collection="mempalace_records",
        ids=[eid],
        documents=["x"],
        embeddings=[fake_emb],
    )

    # Pre-check.
    pre_map = (
        _probe_conn(db_path)
        .execute("SELECT COUNT(*) FROM vec_rowid_map WHERE logical_id=?", (eid,))
        .fetchone()[0]
    )
    pre_vec = (
        _probe_conn(db_path)
        .execute("SELECT COUNT(*) FROM vec_palace WHERE entity_id=?", (eid,))
        .fetchone()[0]
    )
    assert pre_map == 1 and pre_vec == 1

    # Detach the vector store's connection (it shares the file but
    # holds its own conn handle). Reconnect through the KG to get
    # PRAGMA foreign_keys=ON.
    store.close()
    conn = kg._conn()

    # Re-acquire a connection because the FK + trigger fire on a
    # connection-level FK setting; kg._conn already sets ON.
    # Verify the underlying triples don't block (this entity has no
    # incoming triples, so the FK on triples.subject/object doesn't
    # fire).
    conn.execute("DELETE FROM entities WHERE id=?", (eid,))
    conn.commit()

    # Post-check: rowid_map row gone (FK cascade) + vec_palace row
    # gone (BEFORE DELETE trigger).
    probe = _probe_conn(db_path)
    post_map = probe.execute(
        "SELECT COUNT(*) FROM vec_rowid_map WHERE logical_id=?", (eid,)
    ).fetchone()[0]
    post_vec = probe.execute(
        "SELECT COUNT(*) FROM vec_palace WHERE entity_id=?", (eid,)
    ).fetchone()[0]
    assert post_map == 0, f"FK CASCADE should drop the rowid_map row; got {post_map} remaining"
    assert post_vec == 0, (
        f"BEFORE DELETE trigger should drop the vec_palace row; got {post_vec} remaining"
    )


def test_v326_delete_triple_cascades_to_rowid_map_and_vec_palace(tmp_path):
    """Deleting a triple cascades to its mempalace_triples vec row."""
    kg, store, db_path = _make_palace(tmp_path)
    eid_a = kg.add_entity("fk_v326_trip_sub2", kind="entity", content="s")
    eid_b = kg.add_entity("fk_v326_trip_obj2", kind="entity", content="o")
    conn = kg._conn()
    triple_id = "t_v326_cascade_test"
    conn.execute(
        "INSERT INTO triples (id, subject, predicate, object, statement, valid_from) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (triple_id, eid_a, "related_to", eid_b, "stmt", "2026-05-12"),
    )
    conn.commit()

    fake_emb = [0.1] * 384
    store.add(
        collection="mempalace_triples",
        ids=[triple_id],
        documents=["stmt"],
        embeddings=[fake_emb],
    )
    store.close()

    conn = kg._conn()
    conn.execute("DELETE FROM triples WHERE id=?", (triple_id,))
    conn.commit()

    probe = _probe_conn(db_path)
    post_map = probe.execute(
        "SELECT COUNT(*) FROM vec_rowid_map WHERE logical_id=?", (triple_id,)
    ).fetchone()[0]
    post_vec = probe.execute(
        "SELECT COUNT(*) FROM vec_palace WHERE entity_id=?", (triple_id,)
    ).fetchone()[0]
    assert post_map == 0
    assert post_vec == 0


# ── _derive_fk_refs unit-like coverage ----------------------------------


def test_v326_derive_fk_refs_classification():
    """The static helper must classify all four namespaces correctly."""
    f = SqliteVecVectorStore._derive_fk_refs
    assert f("mempalace_triples", "t_abc") == (None, "t_abc")
    assert f("mempalace_records", "some_entity__v0") == ("some_entity", None)
    assert f("mempalace_records", "another_one__v15") == ("another_one", None)
    assert f("mempalace_context_views", "ctx_10000_v0") == ("ctx_10000", None)
    assert f("mempalace_context_views", "ctx_99_v42") == ("ctx_99", None)
    assert f("mempalace_records", "bare_entity_id") == ("bare_entity_id", None)
    # Edge case: prefix containing _v should still strip only the final suffix.
    assert f("mempalace_context_views", "ctx_v3_thing_v7") == ("ctx_v3_thing", None)
