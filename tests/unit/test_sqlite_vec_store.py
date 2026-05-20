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
    _extract_entity_id_in_filter,
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


# ─────────────────────────────────────────────────────────────────────
# FINDING #B regression (v3.7.24, 2026-05-18): `get(offset=...)` was
# silently missing from SqliteVecVectorStore after the chromadb ->
# sqlite_vec swap, raising TypeError that Layer1.generate's bare
# `except Exception: pass` swallowed -- "## L1 -- No entries yet."
# rendered on every populated palace for ~6 weeks. These tests pin the
# offset kwarg + LIMIT/OFFSET semantics so the regression cannot return.
# ─────────────────────────────────────────────────────────────────────


def test_get_accepts_offset_kwarg(sqlite_vec_store):
    """``get(offset=...)`` must not raise TypeError.

    The chromadb backend accepted offset; sqlite_vec did not. Layer1
    and dedup.get_source_groups both pass ``offset=offset`` on the
    pagination loop and used to silently fall through to empty
    results when the kwarg drift landed.
    """
    sqlite_vec_store.add(
        RECORDS_COLLECTION,
        ids=["a", "b", "c"],
        documents=["doc a", "doc b", "doc c"],
        metadatas=[{}, {}, {}],
    )
    # Smoke: offset=0 returns the first row(s).
    page0 = sqlite_vec_store.get(RECORDS_COLLECTION, limit=1, offset=0)
    assert len(page0.ids) == 1
    # Smoke: offset advances the cursor.
    page1 = sqlite_vec_store.get(RECORDS_COLLECTION, limit=1, offset=1)
    assert len(page1.ids) == 1
    assert page1.ids[0] != page0.ids[0]


def test_get_offset_full_pagination_visits_all_rows(sqlite_vec_store):
    """Iterating ``offset += len(batch.ids)`` must visit every row.

    Mirrors Layer1.generate's actual loop shape. Catches off-by-one
    or duplicate-row bugs in the SQL.
    """
    ids = [f"k{i:03d}" for i in range(25)]
    sqlite_vec_store.add(
        RECORDS_COLLECTION,
        ids=ids,
        documents=[f"doc {i}" for i in range(25)],
        metadatas=[{"i": i} for i in range(25)],
    )
    visited: list[str] = []
    offset = 0
    batch_size = 10
    while True:
        page = sqlite_vec_store.get(RECORDS_COLLECTION, limit=batch_size, offset=offset)
        if not page.ids:
            break
        visited.extend(page.ids)
        offset += len(page.ids)
        if len(page.ids) < batch_size:
            break
    assert sorted(visited) == sorted(ids)
    assert len(visited) == len(set(visited)), "duplicate ids across pages"


def test_get_offset_ignored_when_ids_supplied(sqlite_vec_store):
    """ID-lookup path is point-style; offset is silently ignored."""
    sqlite_vec_store.add(
        RECORDS_COLLECTION,
        ids=["a", "b", "c"],
        documents=["doc a", "doc b", "doc c"],
        metadatas=[{}, {}, {}],
    )
    got = sqlite_vec_store.get(RECORDS_COLLECTION, ids=["a"], offset=99)
    assert got.ids == ["a"]


def test_get_offset_with_where_filter(sqlite_vec_store):
    """Offset applies pre-filter (on the rowid scan), where applies
    per-row -- matches chromadb's classic semantics."""
    sqlite_vec_store.add(
        RECORDS_COLLECTION,
        ids=["a", "b", "c", "d"],
        documents=["x", "y", "z", "w"],
        metadatas=[
            {"kind": "memory"},
            {"kind": "entity"},
            {"kind": "memory"},
            {"kind": "entity"},
        ],
    )
    # Whole-collection scan w/ offset=2 + where -> still returns rows
    # that match the predicate among the post-offset slice.
    page = sqlite_vec_store.get(RECORDS_COLLECTION, where={"kind": "memory"}, limit=10, offset=2)
    # Two memory rows total; with offset=2 we may skip one or both
    # depending on rowid order, so we only assert no crash + correct
    # filter (every returned row has kind=memory).
    for meta in page.metadatas:
        assert (meta or {}).get("kind") == "memory"


# ─────────────────────────────────────────────────────────────────────
# v3.9.3 Tier A: indexed entity_id forward lookup (keyword-channel
# Stage-2 fix). A pure entity_id equality/$in `where` filter must
# resolve via the indexed vec_rowid_map.entity_id_ref column instead of
# a whole-collection scan. These pin the detector + the indexed path's
# correctness (output identical to the scan it replaces).
# ─────────────────────────────────────────────────────────────────────


def test_extract_entity_id_filter_shapes():
    """The detector returns the eid list ONLY for pure entity_id
    equality / $in filters; None for everything else (so the general
    predicate path keeps handling those)."""
    assert _extract_entity_id_in_filter({"entity_id": "a"}) == ["a"]
    assert _extract_entity_id_in_filter({"entity_id": {"$eq": "a"}}) == ["a"]
    assert _extract_entity_id_in_filter({"entity_id": {"$in": ["a", "b"]}}) == ["a", "b"]
    # Not a pure entity_id filter -> None (fall back to scan path).
    assert _extract_entity_id_in_filter({"kind": "memory"}) is None
    assert _extract_entity_id_in_filter({"entity_id": {"$ne": "a"}}) is None
    assert _extract_entity_id_in_filter({"entity_id": "a", "kind": "memory"}) is None
    assert _extract_entity_id_in_filter({"$and": [{"entity_id": "a"}]}) is None
    assert _extract_entity_id_in_filter(None) is None
    assert _extract_entity_id_in_filter({}) is None


def _seed_entity_id_rows(store):
    """Three rows whose metadata carries entity_id (as production
    record/entity rows do). logical_id == entity_id here, so
    vec_rowid_map.entity_id_ref == entity_id -- the records-collection
    shape the keyword channel resolves against."""
    store.add(
        RECORDS_COLLECTION,
        ids=["ent_a", "ent_b", "ent_c"],
        documents=["doc a", "doc b", "doc c"],
        metadatas=[
            {"entity_id": "ent_a", "kind": "record"},
            {"entity_id": "ent_b", "kind": "record"},
            {"entity_id": "ent_c", "kind": "record"},
        ],
    )


def test_get_where_entity_id_in_returns_matches(sqlite_vec_store):
    """get(where={'entity_id': {'$in': [...]}}) returns exactly the
    matching rows via the indexed path -- this is the keyword channel's
    Stage-2 resolution call."""
    _seed_entity_id_rows(sqlite_vec_store)
    got = sqlite_vec_store.get(RECORDS_COLLECTION, where={"entity_id": {"$in": ["ent_a", "ent_c"]}})
    assert sorted(got.ids) == ["ent_a", "ent_c"]
    # Documents come back intact (forward lookup, not just ids).
    by_id = dict(zip(got.ids, got.documents))
    assert by_id["ent_a"] == "doc a"
    assert by_id["ent_c"] == "doc c"


def test_get_where_entity_id_scalar(sqlite_vec_store):
    """Scalar entity_id equality also routes through the indexed path."""
    _seed_entity_id_rows(sqlite_vec_store)
    got = sqlite_vec_store.get(RECORDS_COLLECTION, where={"entity_id": "ent_b"})
    assert got.ids == ["ent_b"]
    assert got.documents == ["doc b"]


def test_get_where_entity_id_no_match_returns_empty(sqlite_vec_store):
    """A nonexistent entity_id surfaces nothing (and doesn't error)."""
    _seed_entity_id_rows(sqlite_vec_store)
    got = sqlite_vec_store.get(RECORDS_COLLECTION, where={"entity_id": {"$in": ["nope"]}})
    assert got.ids == []


def test_get_where_non_entity_id_still_scans(sqlite_vec_store):
    """Regression: a non-entity_id where filter must STILL work via the
    general predicate path (the Tier A branch must not swallow it)."""
    _seed_entity_id_rows(sqlite_vec_store)
    got = sqlite_vec_store.get(RECORDS_COLLECTION, where={"kind": "record"})
    assert sorted(got.ids) == ["ent_a", "ent_b", "ent_c"]
