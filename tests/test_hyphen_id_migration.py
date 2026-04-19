"""Unit tests for the hyphen-id → underscored-id migration (N3)."""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock


from mempalace import hyphen_id_migration
from mempalace.knowledge_graph import normalize_entity_name


# ==================== ID transformation ====================


class TestIdTransformation:
    def test_plain_hyphenated_id_normalizes(self):
        out = hyphen_id_migration.compute_target_id(
            "record_ga_agent_foo-bar-baz", normalize_entity_name
        )
        assert out == "record_ga_agent_foo_bar_baz"

    def test_already_canonical_id_unchanged(self):
        out = hyphen_id_migration.compute_target_id(
            "record_ga_agent_already_good", normalize_entity_name
        )
        assert out == "record_ga_agent_already_good"

    def test_entity_view_suffix_preserved(self):
        out = hyphen_id_migration.compute_target_id("some-entity__v3", normalize_entity_name)
        assert out == "some_entity__v3"

    def test_legacy_view_suffix_preserved(self):
        out = hyphen_id_migration.compute_target_id("some-entity::view_2", normalize_entity_name)
        assert out == "some_entity::view_2"

    def test_id_needs_migration_true_for_hyphenated(self):
        assert hyphen_id_migration.id_needs_migration("foo-bar", normalize_entity_name)

    def test_id_needs_migration_false_for_canonical(self):
        assert not hyphen_id_migration.id_needs_migration("foo_bar", normalize_entity_name)


# ==================== Chroma migration ====================


def _fake_collection(records):
    """Build a minimal mock that mimics chromadb collection .get/.delete/.upsert.

    records: list of dicts with keys id, document, metadata, embedding.
    """
    store = {r["id"]: dict(r) for r in records}

    def _get(ids=None, where=None, include=None):
        include = include or []
        if ids:
            hits = [store[i] for i in ids if i in store]
        else:
            hits = list(store.values())
        out = {"ids": [h["id"] for h in hits]}
        if "documents" in include:
            out["documents"] = [h.get("document") for h in hits]
        if "metadatas" in include:
            out["metadatas"] = [h.get("metadata") for h in hits]
        if "embeddings" in include:
            out["embeddings"] = [h.get("embedding") for h in hits]
        return out

    def _delete(ids):
        for i in ids:
            store.pop(i, None)

    def _upsert(ids, documents=None, metadatas=None, embeddings=None):
        for idx, i in enumerate(ids):
            rec = {"id": i}
            if documents is not None:
                rec["document"] = documents[idx]
            if metadatas is not None:
                rec["metadata"] = metadatas[idx]
            if embeddings is not None:
                rec["embedding"] = embeddings[idx]
            store[i] = rec

    col = MagicMock()
    col.get.side_effect = _get
    col.delete.side_effect = _delete
    col.upsert.side_effect = _upsert
    col._store = store  # test inspection
    return col


class TestChromaMigration:
    def test_migrates_hyphenated_ids(self):
        col = _fake_collection(
            [
                {
                    "id": "record_x_foo-bar",
                    "document": "D1",
                    "metadata": {"k": "v"},
                    "embedding": [0.1],
                },
                {
                    "id": "record_x_already_good",
                    "document": "D2",
                    "metadata": {},
                    "embedding": [0.2],
                },
            ]
        )
        stats = hyphen_id_migration.migrate_chroma_collection(col, normalize_entity_name)
        assert stats["scanned"] == 2
        assert stats["migrated"] == 1
        assert stats["collisions"] == 0
        assert "record_x_foo_bar" in col._store
        assert "record_x_foo-bar" not in col._store
        # Embedding preserved verbatim (no re-embed).
        assert col._store["record_x_foo_bar"]["embedding"] == [0.1]

    def test_preserves_view_suffix_and_updates_metadata_entity_id(self):
        col = _fake_collection(
            [
                {
                    "id": "my-entity__v0",
                    "document": "view 0",
                    "metadata": {"entity_id": "my-entity", "view_index": 0},
                    "embedding": [0.3],
                },
                {
                    "id": "my-entity__v1",
                    "document": "view 1",
                    "metadata": {"entity_id": "my-entity", "view_index": 1},
                    "embedding": [0.4],
                },
            ]
        )
        stats = hyphen_id_migration.migrate_chroma_collection(
            col, normalize_entity_name, update_metadata_entity_id=True
        )
        assert stats["migrated"] == 2
        assert "my_entity__v0" in col._store
        assert "my_entity__v1" in col._store
        assert col._store["my_entity__v0"]["metadata"]["entity_id"] == "my_entity"
        assert col._store["my_entity__v1"]["metadata"]["entity_id"] == "my_entity"

    def test_skips_collision(self):
        col = _fake_collection(
            [
                {"id": "foo-bar", "document": "old", "metadata": {}, "embedding": [0.1]},
                {"id": "foo_bar", "document": "new", "metadata": {}, "embedding": [0.2]},
            ]
        )
        stats = hyphen_id_migration.migrate_chroma_collection(col, normalize_entity_name)
        assert stats["collisions"] == 1
        assert stats["migrated"] == 0
        # Both remain — merge requires explicit agent action.
        assert "foo-bar" in col._store
        assert "foo_bar" in col._store

    def test_none_collection_is_noop(self):
        stats = hyphen_id_migration.migrate_chroma_collection(None, normalize_entity_name)
        assert stats == {"scanned": 0, "migrated": 0, "collisions": 0, "errors": 0}


# ==================== SQLite migration ====================


def _fresh_conn():
    """In-memory SQLite with the subset of mempalace schema relevant here."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE entities (
            id TEXT PRIMARY KEY, name TEXT, kind TEXT, merged_into TEXT
        );
        CREATE TABLE entity_aliases (
            alias TEXT PRIMARY KEY, canonical_id TEXT, merged_at TEXT
        );
        CREATE TABLE entity_keywords (
            entity_id TEXT, keyword TEXT, source TEXT, added_at TEXT,
            PRIMARY KEY (entity_id, keyword)
        );
        CREATE TABLE triples (
            id TEXT PRIMARY KEY, subject TEXT, predicate TEXT, object TEXT
        );
        CREATE TABLE conflict_resolutions (
            id INTEGER PRIMARY KEY, existing_id TEXT, new_id TEXT
        );
        CREATE TABLE edge_traversal_feedback (
            id INTEGER PRIMARY KEY, subject TEXT, predicate TEXT, object TEXT, useful BOOLEAN
        );
        CREATE TABLE keyword_feedback (
            id INTEGER PRIMARY KEY, memory_id TEXT, context_id TEXT
        );
        """
    )
    return conn


class TestSqliteMigration:
    def test_migrates_hyphenated_entities_and_cascades(self):
        conn = _fresh_conn()
        conn.execute(
            "INSERT INTO entities (id, name, kind) VALUES (?, ?, ?)",
            ("record_x_foo-bar", "foo bar", "record"),
        )
        conn.execute(
            "INSERT INTO triples (id, subject, predicate, object) VALUES (?, ?, ?, ?)",
            ("t1", "record_x_foo-bar", "is_a", "record"),
        )
        conn.execute(
            "INSERT INTO entity_keywords (entity_id, keyword) VALUES (?, ?)",
            ("record_x_foo-bar", "foo"),
        )
        conn.commit()

        stats = hyphen_id_migration.migrate_sqlite(conn, normalize_entity_name)

        assert stats["migrated_ids"] >= 1
        assert stats["collisions"] == 0
        (eid,) = conn.execute("SELECT id FROM entities").fetchone()
        assert eid == "record_x_foo_bar"
        (subj,) = conn.execute("SELECT subject FROM triples").fetchone()
        assert subj == "record_x_foo_bar"
        (kid,) = conn.execute("SELECT entity_id FROM entity_keywords").fetchone()
        assert kid == "record_x_foo_bar"

    def test_all_canonical_is_noop(self):
        conn = _fresh_conn()
        conn.execute(
            "INSERT INTO entities (id, name, kind) VALUES (?, ?, ?)",
            ("already_good_id", "x", "record"),
        )
        conn.commit()
        stats = hyphen_id_migration.migrate_sqlite(conn, normalize_entity_name)
        assert stats["candidate_ids"] == 0
        assert stats["migrated_ids"] == 0
        assert stats["rows_touched"] == 0

    def test_collision_skips_rename(self):
        conn = _fresh_conn()
        conn.execute(
            "INSERT INTO entities (id, name, kind) VALUES (?, ?, ?)",
            ("foo-bar", "hyphen", "record"),
        )
        conn.execute(
            "INSERT INTO entities (id, name, kind) VALUES (?, ?, ?)",
            ("foo_bar", "underscore", "record"),
        )
        conn.commit()
        stats = hyphen_id_migration.migrate_sqlite(conn, normalize_entity_name)
        assert stats["collisions"] == 1
        # Both rows still present.
        rows = {r[0] for r in conn.execute("SELECT id FROM entities")}
        assert rows == {"foo-bar", "foo_bar"}


# ==================== Orchestrator idempotence ====================


class TestOrchestrator:
    def test_flag_gates_second_run(self):
        state = MagicMock()
        state.hyphen_ids_migrated = False
        state.kg = None

        # First run does something (flag flips to True).
        stats1 = hyphen_id_migration.run_migration(
            state,
            chroma_record_col=None,
            chroma_entity_col=None,
            chroma_feedback_col=None,
            normalize=normalize_entity_name,
        )
        assert not stats1.get("skipped")
        assert state.hyphen_ids_migrated is True

        # Second run is a fast no-op.
        stats2 = hyphen_id_migration.run_migration(
            state,
            chroma_record_col=None,
            chroma_entity_col=None,
            chroma_feedback_col=None,
            normalize=normalize_entity_name,
        )
        assert stats2 == {"skipped": True}
