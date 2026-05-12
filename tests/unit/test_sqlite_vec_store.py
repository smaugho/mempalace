"""Phase 3+4 of the chromadb sealing: unit tests + parity-with-chromadb
for :class:`SqliteVecVectorStore`.

The parity tests are the safety net for swapping the default backend:
both stores receive the exact same writes, then the exact same reads,
and we assert the round-trip results match within the contract.

For KNN we don't assert byte-identical ranking against chromadb (the
two backends use different ANN implementations); we assert that the
TOP HIT is the same (i.e. each backend correctly returns the closest
match for an exact-text query) and that the result count matches. That
covers the retrieval semantics callers actually depend on while
allowing implementation-detail divergence in deeper ranks.
"""

from __future__ import annotations

import os

import pytest

from mempalace.embedder import get_default_embedder
from mempalace.sqlite_vec_store import (
    SqliteVecVectorStore,
    _compile_where,
    _pack_vec,
    _stable_rowid,
    _unpack_vec,
)
from mempalace.vector_store import RECORDS_COLLECTION


pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────────────
# Pure-function unit tests
# ─────────────────────────────────────────────────────────────────────


def test_pack_unpack_roundtrip():
    """Vectors survive struct.pack -> blob -> unpack exactly."""
    vec = [0.0, 1.0, -1.5, 0.7071]
    blob = _pack_vec(vec)
    out = _unpack_vec(blob, dim=4)
    for a, b in zip(vec, out):
        assert abs(a - b) < 1e-6


def test_stable_rowid_deterministic():
    """Same (collection, entity_id) always maps to same rowid."""
    a = _stable_rowid("records", "memory_alpha")
    b = _stable_rowid("records", "memory_alpha")
    assert a == b
    assert 0 < a < 2**63


def test_stable_rowid_distinct():
    """Different inputs produce different rowids."""
    assert _stable_rowid("records", "x") != _stable_rowid("records", "y")
    assert _stable_rowid("records", "x") != _stable_rowid("context_views", "x")


# ─────────────────────────────────────────────────────────────────────
# Where-filter compiler
# ─────────────────────────────────────────────────────────────────────


def test_where_compile_equality():
    p = _compile_where({"kind": "memory"})
    assert p({"kind": "memory"}) is True
    assert p({"kind": "operation"}) is False
    assert p({}) is False  # missing key -> not equal


def test_where_compile_eq_op():
    p = _compile_where({"kind": {"$eq": "memory"}})
    assert p({"kind": "memory"}) is True
    assert p({"kind": "operation"}) is False


def test_where_compile_ne_op():
    p = _compile_where({"kind": {"$ne": "memory"}})
    assert p({"kind": "memory"}) is False
    assert p({"kind": "operation"}) is True


def test_where_compile_and():
    p = _compile_where({"$and": [{"added_by": "agent_x"}, {"type": "diary_entry"}]})
    assert p({"added_by": "agent_x", "type": "diary_entry"}) is True
    assert p({"added_by": "agent_x", "type": "record"}) is False
    assert p({"added_by": "agent_y", "type": "diary_entry"}) is False


def test_where_compile_or():
    p = _compile_where({"$or": [{"a": 1}, {"b": 2}]})
    assert p({"a": 1}) is True
    assert p({"b": 2}) is True
    assert p({"a": 1, "b": 2}) is True
    assert p({"c": 3}) is False


def test_where_compile_comparisons():
    p_gt = _compile_where({"score": {"$gt": 0.5}})
    assert p_gt({"score": 0.7}) is True
    assert p_gt({"score": 0.3}) is False
    p_in = _compile_where({"kind": {"$in": ["memory", "record"]}})
    assert p_in({"kind": "memory"}) is True
    assert p_in({"kind": "operation"}) is False


def test_where_compile_unknown_op_raises():
    with pytest.raises(ValueError):
        _compile_where({"kind": {"$bogus": 1}})


def test_where_compile_empty_passes_all():
    p = _compile_where(None)
    assert p({}) is True
    assert p({"anything": "ok"}) is True


# ─────────────────────────────────────────────────────────────────────
# SqliteVecVectorStore end-to-end (no chromadb involved)
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def fresh_palace(tmp_path):
    """A fresh temporary palace path. tmp_path is auto-cleaned by pytest."""
    return str(tmp_path)


@pytest.fixture
def sqlite_vec_store(fresh_palace):
    """Fresh SqliteVecVectorStore. Closed after the test."""
    store = SqliteVecVectorStore(fresh_palace)
    yield store
    store.close()


def test_init_creates_db_file(sqlite_vec_store, fresh_palace):
    assert os.path.exists(os.path.join(fresh_palace, "knowledge_graph.sqlite3"))
    # The vec_palace virtual table exists.
    rows = sqlite_vec_store.conn.execute(
        "SELECT name FROM sqlite_master WHERE name='vec_palace'"
    ).fetchall()
    assert rows


def test_list_collections_includes_known_set(sqlite_vec_store):
    cols = sqlite_vec_store.list_collections()
    assert "mempalace_records" in cols
    assert "mempalace_context_views" in cols
    assert "mempalace_triples" in cols


def test_health_all_collections_empty_initially(sqlite_vec_store):
    health = sqlite_vec_store.health()
    assert isinstance(health, dict)
    assert all(not h.is_poisoned for h in health.values())


def test_is_poisoned_always_false(sqlite_vec_store):
    assert sqlite_vec_store.is_poisoned("mempalace_records") is False
    assert sqlite_vec_store.poisoned_collections() == set()


def test_add_and_count(sqlite_vec_store):
    result = sqlite_vec_store.add(
        RECORDS_COLLECTION,
        ids=["a", "b", "c"],
        documents=["alpha doc", "beta doc", "gamma doc"],
        metadatas=[{"kind": "memory"}, {"kind": "memory"}, {"kind": "record"}],
    )
    assert result.persisted is True
    assert result.rows_affected == 3
    assert sqlite_vec_store.count(RECORDS_COLLECTION) == 3
    assert sqlite_vec_store.sql_row_count(RECORDS_COLLECTION) == 3


def test_add_duplicate_fails(sqlite_vec_store):
    sqlite_vec_store.add(RECORDS_COLLECTION, ids=["a"], documents=["alpha"], metadatas=[{}])
    dup = sqlite_vec_store.add(RECORDS_COLLECTION, ids=["a"], documents=["alpha2"], metadatas=[{}])
    assert dup.persisted is False
    assert "duplicate" in dup.error


def test_upsert_replaces(sqlite_vec_store):
    sqlite_vec_store.add(
        RECORDS_COLLECTION,
        ids=["a"],
        documents=["alpha v1"],
        metadatas=[{"version": 1}],
    )
    r = sqlite_vec_store.upsert(
        RECORDS_COLLECTION,
        ids=["a"],
        documents=["alpha v2"],
        metadatas=[{"version": 2}],
    )
    assert r.persisted is True
    got = sqlite_vec_store.get(RECORDS_COLLECTION, ids=["a"])
    assert got.documents == ["alpha v2"]
    assert got.metadatas == [{"version": 2}]


def test_get_by_ids_preserves_order(sqlite_vec_store):
    sqlite_vec_store.add(
        RECORDS_COLLECTION,
        ids=["a", "b", "c"],
        documents=["doc a", "doc b", "doc c"],
        metadatas=[{}, {}, {}],
    )
    got = sqlite_vec_store.get(RECORDS_COLLECTION, ids=["c", "a"])
    assert got.ids == ["c", "a"]
    assert got.documents == ["doc c", "doc a"]


def test_get_with_where(sqlite_vec_store):
    sqlite_vec_store.add(
        RECORDS_COLLECTION,
        ids=["a", "b", "c"],
        documents=["x", "y", "z"],
        metadatas=[{"kind": "memory"}, {"kind": "memory"}, {"kind": "record"}],
    )
    got = sqlite_vec_store.get(RECORDS_COLLECTION, where={"kind": "memory"})
    assert sorted(got.ids) == ["a", "b"]


def test_query_returns_closest_hit(sqlite_vec_store):
    docs = [
        "the quick brown fox jumps over the lazy dog",
        "python is a programming language",
        "memory palaces help with mnemonic recall",
    ]
    sqlite_vec_store.add(
        RECORDS_COLLECTION,
        ids=["fox", "python", "palace"],
        documents=docs,
        metadatas=[{"i": 0}, {"i": 1}, {"i": 2}],
    )
    # Query with the EXACT text of one entry; the top-1 hit must be it.
    res = sqlite_vec_store.query(
        RECORDS_COLLECTION,
        query_texts=["python is a programming language"],
        n_results=3,
    )
    assert res.ids
    assert res.ids[0][0] == "python"


def test_delete_by_ids(sqlite_vec_store):
    sqlite_vec_store.add(
        RECORDS_COLLECTION,
        ids=["a", "b", "c"],
        documents=["x", "y", "z"],
        metadatas=[{}, {}, {}],
    )
    r = sqlite_vec_store.delete(RECORDS_COLLECTION, ids=["b"])
    assert r.persisted is True
    assert r.rows_affected == 1
    assert sqlite_vec_store.count(RECORDS_COLLECTION) == 2
    got = sqlite_vec_store.get(RECORDS_COLLECTION, ids=["a", "b", "c"])
    assert sorted(got.ids) == ["a", "c"]


def test_delete_by_where(sqlite_vec_store):
    sqlite_vec_store.add(
        RECORDS_COLLECTION,
        ids=["a", "b", "c"],
        documents=["x", "y", "z"],
        metadatas=[{"kind": "memory"}, {"kind": "memory"}, {"kind": "record"}],
    )
    r = sqlite_vec_store.delete(RECORDS_COLLECTION, where={"kind": "memory"})
    assert r.persisted is True
    assert r.rows_affected == 2
    assert sqlite_vec_store.count(RECORDS_COLLECTION) == 1


def test_delete_collection(sqlite_vec_store):
    sqlite_vec_store.add(RECORDS_COLLECTION, ids=["a"], documents=["x"], metadatas=[{}])
    r = sqlite_vec_store.delete_collection(RECORDS_COLLECTION)
    assert r.persisted is True
    assert sqlite_vec_store.count(RECORDS_COLLECTION) == 0
    assert RECORDS_COLLECTION not in sqlite_vec_store.list_collections()


def test_all_ids_returns_full_set(sqlite_vec_store):
    ids = [f"id_{i}" for i in range(25)]
    sqlite_vec_store.add(
        RECORDS_COLLECTION,
        ids=ids,
        documents=[f"doc {i}" for i in range(25)],
        metadatas=[{} for _ in range(25)],
    )
    out = sqlite_vec_store.all_ids(RECORDS_COLLECTION, batch_size=10)
    assert sorted(out) == sorted(ids)


def test_atomic_rollback_on_failed_batch(sqlite_vec_store):
    """If a write batch fails midway, no rows should have landed."""
    # Pre-insert a row so the second add() hits a duplicate.
    sqlite_vec_store.add(RECORDS_COLLECTION, ids=["existing"], documents=["x"], metadatas=[{}])
    # Now attempt a 3-row add where the middle row collides.
    r = sqlite_vec_store.add(
        RECORDS_COLLECTION,
        ids=["new_1", "existing", "new_2"],
        documents=["doc 1", "doc 2", "doc 3"],
        metadatas=[{}, {}, {}],
    )
    assert r.persisted is False
    # Only the original "existing" row should remain.
    assert sqlite_vec_store.count(RECORDS_COLLECTION) == 1
    got = sqlite_vec_store.get(RECORDS_COLLECTION, ids=["new_1", "new_2"])
    assert got.ids == []


# ─────────────────────────────────────────────────────────────────────
# Embedder-backed semantic contract (was: parity-vs-chromadb)
# ─────────────────────────────────────────────────────────────────────
#
# chromadb is retired (2026-05-12) so these tests no longer compare
# two backends -- they just confirm SqliteVecVectorStore behaves
# correctly end-to-end when fed real embeddings. Skipped when
# fastembed isn't available; the upper unit tests run without an
# embedder by passing embeddings explicitly.
_EMBEDDER = get_default_embedder()


@pytest.mark.skipif(_EMBEDDER is None, reason="fastembed not available")
def test_top_hit_for_exact_text_query(tmp_path):
    """Querying for a doc's own embedding returns that doc as the top
    hit. Deeper-rank order is implementation-defined; top-1 is the
    contract callers rely on."""
    docs = {
        "fox": "the quick brown fox jumps over the lazy dog",
        "python": "python is a programming language",
        "palace": "memory palaces help with mnemonic recall",
        "judge": "state_judge detects state changes in followed entities",
    }
    ids = list(docs.keys())
    documents = list(docs.values())
    metadatas = [{"i": i} for i in range(len(ids))]
    embeddings = _EMBEDDER(documents)

    sv_palace = str(tmp_path / "sv_palace")
    os.makedirs(sv_palace)
    sv = SqliteVecVectorStore(sv_palace)
    sv.add(
        RECORDS_COLLECTION,
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    for target in ids:
        sv_res = sv.query(
            RECORDS_COLLECTION,
            query_embeddings=[embeddings[ids.index(target)]],
            n_results=3,
        )
        assert sv_res.ids[0], f"sqlite-vec returned 0 hits for {target!r}"
        assert sv_res.ids[0][0] == target, (
            f"sqlite-vec top hit for {target!r} was {sv_res.ids[0][0]!r}, expected {target!r}"
        )

    sv.close()


@pytest.mark.skipif(_EMBEDDER is None, reason="fastembed not available")
def test_get_by_id_returns_seeded_payload(tmp_path):
    """``get(ids=[...])`` returns the document + metadata as
    written."""
    ids = ["x", "y", "z"]
    docs = ["doc x", "doc y", "doc z"]
    metas = [{"k": 1}, {"k": 2}, {"k": 3}]
    embeddings = _EMBEDDER(docs)

    # SqliteVecVectorStore does not eager-mkdir (palace lifecycle is
    # the caller's responsibility; missing palace -> degraded reads).
    # Create the dir up-front so this test exercises the populated
    # palace path.
    sv_path = str(tmp_path / "sv")
    os.makedirs(sv_path)
    sv = SqliteVecVectorStore(sv_path)
    sv.add(
        RECORDS_COLLECTION,
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=embeddings,
    )

    got = sv.get(RECORDS_COLLECTION, ids=["y"])
    assert got.ids == ["y"], "ids mismatch"
    assert got.documents == ["doc y"], "doc mismatch"
    assert got.metadatas == [{"k": 2}], "meta mismatch"

    sv.close()
