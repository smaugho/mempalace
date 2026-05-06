"""
test_unified_collection.py -- Contract for the M1 merge that collapsed
``mempalace_entities`` into ``mempalace_records``.

After M1:
  - There is ONE Chroma collection (``mempalace_records``) for all
    embedded rows. metadata.kind discriminates record vs class vs
    entity vs predicate.
  - ``_get_collection`` and ``_get_entity_collection`` both return the
    same physical collection. The entity helper remains for callsite
    compatibility and for triggering the view-schema migration.
  - On server startup, ``_migrate_entities_collection_into_records``
    absorbs any legacy ``mempalace_entities`` rows into the unified
    collection and drops the legacy collection. Idempotent.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from mempalace import mcp_server


def _fake_client(collections: dict):
    """Minimal Chroma client stub: get_collection / delete_collection /
    PersistentClient-style construction."""
    client = MagicMock()
    client.get_collection.side_effect = lambda name: (
        collections.get(name) or (_raise(ValueError(f"no collection {name}")))
    )

    def _delete(name):
        collections.pop(name, None)

    client.delete_collection.side_effect = _delete
    return client


def _raise(exc):
    raise exc


def _fake_collection(name, records=None):
    """Minimal collection stub with get/upsert/delete."""
    records = records or []
    store = {r["id"]: dict(r) for r in records}

    def _get(ids=None, where=None, include=None, limit=None, offset=None):
        include = include or []
        hits = list(store.values())
        if ids:
            hits = [store[i] for i in ids if i in store]
        out = {"ids": [h["id"] for h in hits]}
        if "documents" in include:
            out["documents"] = [h.get("document") for h in hits]
        if "metadatas" in include:
            out["metadatas"] = [h.get("metadata") for h in hits]
        if "embeddings" in include:
            out["embeddings"] = [h.get("embedding") for h in hits]
        return out

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

    def _delete_ids(ids):
        for i in ids:
            store.pop(i, None)

    col = MagicMock()
    col.name = name
    col.get.side_effect = _get
    col.upsert.side_effect = _upsert
    col.delete.side_effect = _delete_ids
    col._store = store
    return col


class TestMergeMigration:
    def test_legacy_entities_are_absorbed_into_records(self, monkeypatch):
        """Every row from mempalace_entities gets copied into
        mempalace_records with embeddings preserved, and the legacy
        collection is deleted afterward."""
        records_col = _fake_collection("mempalace_records", [])
        entities_col = _fake_collection(
            "mempalace_entities",
            [
                {
                    "id": "alpha__v0",
                    "document": "view 0 of alpha",
                    "metadata": {"entity_id": "alpha", "kind": "class"},
                    "embedding": [0.11, 0.22, 0.33],
                },
                {
                    "id": "alpha__v1",
                    "document": "view 1 of alpha",
                    "metadata": {"entity_id": "alpha", "kind": "class"},
                    "embedding": [0.12, 0.23, 0.34],
                },
                {
                    "id": "beta__v0",
                    "document": "desc of beta",
                    "metadata": {"entity_id": "beta", "kind": "entity"},
                    "embedding": [0.44, 0.55, 0.66],
                },
            ],
        )
        collections = {
            "mempalace_records": records_col,
            "mempalace_entities": entities_col,
        }
        client = _fake_client(collections)

        monkeypatch.setattr(mcp_server, "chromadb", MagicMock(PersistentClient=lambda path: client))
        monkeypatch.setattr(mcp_server, "_get_collection", lambda create=True: records_col)
        mcp_server._STATE.entity_collection_merged = False
        mcp_server._STATE.config = MagicMock(palace_path="/fake/path")

        mcp_server._migrate_entities_collection_into_records()

        # All three entity rows now live in records_col, with embeddings.
        assert set(records_col._store.keys()) == {"alpha__v0", "alpha__v1", "beta__v0"}
        assert records_col._store["alpha__v0"]["embedding"] == [0.11, 0.22, 0.33]
        # Legacy collection was deleted from the client.
        assert "mempalace_entities" not in collections
        # Flag was flipped so the migration is one-pass.
        assert mcp_server._STATE.entity_collection_merged

    def test_second_call_is_noop(self, monkeypatch):
        """The flag prevents a second call from doing anything."""
        records_col = _fake_collection("mempalace_records", [])
        entities_col = _fake_collection(
            "mempalace_entities",
            [{"id": "alpha__v0", "document": "x", "metadata": {}, "embedding": [0.0]}],
        )
        collections = {
            "mempalace_records": records_col,
            "mempalace_entities": entities_col,
        }
        client = _fake_client(collections)

        monkeypatch.setattr(mcp_server, "chromadb", MagicMock(PersistentClient=lambda path: client))
        monkeypatch.setattr(mcp_server, "_get_collection", lambda create=True: records_col)
        mcp_server._STATE.entity_collection_merged = True  # pre-flag
        mcp_server._STATE.config = MagicMock(palace_path="/fake/path")

        mcp_server._migrate_entities_collection_into_records()

        # Nothing moved -- legacy still has its row.
        assert "alpha__v0" in entities_col._store
        assert len(records_col._store) == 0

    def test_partial_copy_preserves_legacy_collection(self, monkeypatch):
        """If any upsert batch fails, the legacy collection MUST NOT be
        deleted -- otherwise rows that didn't land in the target are lost.
        """
        records_col = _fake_collection("mempalace_records", [])
        # Make upsert raise to simulate a Chroma hiccup.
        records_col.upsert.side_effect = lambda **kw: _raise(RuntimeError("Chroma disk error"))
        entities_col = _fake_collection(
            "mempalace_entities",
            [
                {
                    "id": "alpha__v0",
                    "document": "x",
                    "metadata": {"entity_id": "alpha", "kind": "class"},
                    "embedding": [0.1],
                }
            ],
        )
        collections = {
            "mempalace_records": records_col,
            "mempalace_entities": entities_col,
        }
        client = _fake_client(collections)
        monkeypatch.setattr(mcp_server, "chromadb", MagicMock(PersistentClient=lambda path: client))
        monkeypatch.setattr(mcp_server, "_get_collection", lambda create=True: records_col)
        mcp_server._STATE.entity_collection_merged = False
        mcp_server._STATE.config = MagicMock(palace_path="/fake/path")

        mcp_server._migrate_entities_collection_into_records()

        # Legacy collection preserved because upsert failed.
        assert "mempalace_entities" in collections
        # The row is still in the legacy collection (no data loss).
        assert "alpha__v0" in entities_col._store

    def test_fresh_palace_without_legacy_collection_is_noop(self, monkeypatch):
        """A palace that was never on the two-collection schema must still
        start cleanly -- get_collection("mempalace_entities") raises, which
        the migration swallows."""
        records_col = _fake_collection("mempalace_records", [])
        collections = {"mempalace_records": records_col}
        client = _fake_client(collections)

        monkeypatch.setattr(mcp_server, "chromadb", MagicMock(PersistentClient=lambda path: client))
        monkeypatch.setattr(mcp_server, "_get_collection", lambda create=True: records_col)
        mcp_server._STATE.entity_collection_merged = False
        mcp_server._STATE.config = MagicMock(palace_path="/fake/path")

        mcp_server._migrate_entities_collection_into_records()

        # No crash, flag flipped, collection is still empty, nothing added.
        assert mcp_server._STATE.entity_collection_merged
        assert len(records_col._store) == 0


class TestUnifiedAccessor:
    def test_get_entity_collection_returns_same_object_as_get_collection(self, monkeypatch):
        """After M1 the two accessors must resolve to the same physical
        collection -- otherwise half the codebase still thinks there are
        two."""
        dummy = _fake_collection("mempalace_records", [])
        monkeypatch.setattr(mcp_server, "_get_collection", lambda create=True: dummy)
        # _migrate_entity_views_schema is called by _get_entity_collection
        # on the returned collection -- stub it out.
        monkeypatch.setattr(mcp_server, "_migrate_entity_views_schema", lambda col: None)

        a = mcp_server._get_entity_collection(create=False)
        b = mcp_server._get_collection(create=False)
        assert a is b

    def test_get_entity_collection_survives_missing_helper(self, monkeypatch):
        """If _get_collection returns None (e.g., palace unavailable), the
        entity helper must NOT raise -- callers rely on None to mean
        'unavailable'."""
        monkeypatch.setattr(mcp_server, "_get_collection", lambda create=True: None)
        assert mcp_server._get_entity_collection(create=False) is None


pytestmark = pytest.mark.integration
