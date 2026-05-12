"""Unit tests for the VectorStore repository.

These tests exercise the gateway layer in isolation -- they construct
their own VectorStore against a tmp_path palace, verify the result-
type contracts, and confirm the poisoned-collection short-circuits
fire correctly.

The tests do NOT touch the user's production palace. They do NOT
depend on the singleton helper -- always construct VectorStore
directly so the singleton cache stays clean.
"""

from __future__ import annotations

import os

import pytest

from mempalace.vector_store import (
    ChromaVectorStore,
    CollectionHealth,
    GetResult,
    HealthInfo,
    KNOWN_COLLECTIONS,
    QueryResult,
    RECORDS_COLLECTION,
    VectorStore,
)


pytestmark = pytest.mark.unit


@pytest.fixture
def fresh_palace(tmp_path):
    """A brand-new empty palace dir. Chroma will lazily create the
    SQLite + segment dirs on first use."""
    palace = tmp_path / "vs_palace"
    palace.mkdir()
    return str(palace)


@pytest.fixture
def vs(fresh_palace):
    """A VectorStore against a fresh palace, with hnsw:sync_threshold=1
    so writes are immediately visible to count/get/query.

    Production uses sync_threshold=100 (slice 16 SIGSEGV-prevention),
    which means the first <100 writes don't sync to HNSW until the
    100th row triggers a batch flush -- col.count() returns 0 and
    col.get() returns empty until then. For unit tests that write a
    handful of rows and immediately read back, we override to 1 so
    every write is visible synchronously. Production-stress tests
    that exercise the slice-16 threshold should construct VectorStore
    with the production default explicitly."""
    # Chroma requires sync_threshold > 2 (validator in
    # chromadb.segment.impl.vector.hnsw_params). Use 3 as the minimum
    # legal value -- enough that small-row tests sync after every
    # write batch.
    return ChromaVectorStore(
        fresh_palace,
        collection_metadata={"hnsw:space": "cosine", "hnsw:sync_threshold": 3},
    )


# ─────────────────────────────────────────────────────────────────────
# Construction + health scan
# ─────────────────────────────────────────────────────────────────────


def test_construct_on_empty_palace(fresh_palace):
    vs = ChromaVectorStore(fresh_palace)
    assert os.path.abspath(vs.palace_path) == os.path.abspath(fresh_palace)
    health = vs.health()
    # Every known collection appears in the health map
    assert set(health.keys()) == set(KNOWN_COLLECTIONS)
    # Empty palace: no queue activity yet -> all UNKNOWN or EMPTY
    for info in health.values():
        assert isinstance(info, HealthInfo)
        assert info.status in (
            CollectionHealth.UNKNOWN,
            CollectionHealth.EMPTY,
            CollectionHealth.OK,
        )
        assert info.is_poisoned is False


def test_no_collections_poisoned_on_fresh_palace(fresh_palace):
    vs = ChromaVectorStore(fresh_palace)
    assert vs.poisoned_collections() == set()
    assert all(not vs.is_poisoned(c) for c in KNOWN_COLLECTIONS)


def test_health_for_unknown_collection(fresh_palace):
    vs = ChromaVectorStore(fresh_palace)
    info = vs.health("does_not_exist")
    assert isinstance(info, HealthInfo)
    assert info.status == CollectionHealth.UNKNOWN


# ─────────────────────────────────────────────────────────────────────
# Write -> read round-trip
# ─────────────────────────────────────────────────────────────────────


def test_upsert_and_query_round_trip(vs):
    res = vs.upsert(
        RECORDS_COLLECTION,
        ids=["a", "b", "c"],
        documents=["alpha doc", "beta doc", "gamma doc"],
        metadatas=[{"k": "v"}, {"k": "w"}, {"k": "x"}],
    )
    assert res.persisted is True
    assert res.rows_affected == 3
    # Query a related text -- expect non-empty hits
    qres = vs.query(RECORDS_COLLECTION, query_texts=["alpha"], n_results=3)
    assert isinstance(qres, QueryResult)
    assert qres.is_degraded is False
    assert qres.total_hits() > 0


def test_get_by_id(vs):
    vs.upsert(
        RECORDS_COLLECTION,
        ids=["x", "y"],
        documents=["foo", "bar"],
        metadatas=[{"i": 1}, {"i": 2}],
    )
    got = vs.get(RECORDS_COLLECTION, ids=["x", "y"])
    assert isinstance(got, GetResult)
    assert set(got.ids) == {"x", "y"}
    assert got.is_degraded is False


def test_count_and_sql_row_count(vs):
    vs.upsert(
        RECORDS_COLLECTION,
        ids=["a", "b", "c", "d"],
        documents=["1", "2", "3", "4"],
        metadatas=[{"_": "_"}, {"_": "_"}, {"_": "_"}, {"_": "_"}],
    )
    assert vs.count(RECORDS_COLLECTION) == 4
    assert vs.sql_row_count(RECORDS_COLLECTION) == 4


def test_delete(vs):
    vs.upsert(
        RECORDS_COLLECTION,
        ids=["k1", "k2"],
        documents=["d1", "d2"],
        metadatas=[{"_": "_"}, {"_": "_"}],
    )
    assert vs.count(RECORDS_COLLECTION) == 2
    res = vs.delete(RECORDS_COLLECTION, ids=["k1"])
    assert res.persisted is True
    assert vs.count(RECORDS_COLLECTION) == 1


def test_all_ids_paginates(vs):
    ids = [f"id{i:04d}" for i in range(120)]
    docs = [f"doc {i}" for i in range(120)]
    metas = [{"i": i} for i in range(120)]
    vs.upsert(RECORDS_COLLECTION, ids=ids, documents=docs, metadatas=metas)
    fetched = vs.all_ids(RECORDS_COLLECTION, batch_size=50)
    assert set(fetched) == set(ids)


# ─────────────────────────────────────────────────────────────────────
# Empty / missing collection paths
# ─────────────────────────────────────────────────────────────────────


def test_query_on_nonexistent_collection_returns_empty(fresh_palace):
    vs = ChromaVectorStore(fresh_palace)
    res = vs.query("definitely_not_real", query_texts=["x"], n_results=5)
    assert isinstance(res, QueryResult)
    assert res.is_empty()
    assert res.is_degraded is True
    assert "unavailable" in res.degraded_reason


def test_get_on_nonexistent_collection_returns_empty(fresh_palace):
    vs = ChromaVectorStore(fresh_palace)
    res = vs.get("nope", ids=["a"])
    assert isinstance(res, GetResult)
    assert res.ids == []
    assert res.is_degraded is True


# ─────────────────────────────────────────────────────────────────────
# Poisoned-collection short-circuit
# ─────────────────────────────────────────────────────────────────────


def _force_poison(vs: VectorStore, collection: str, lag: int = 100) -> None:
    """Overwrite the cached health entry to mark a collection as
    QUEUE_LAG without actually corrupting Chroma. Used to test the
    short-circuit logic in isolation."""
    vs._health[collection] = HealthInfo(
        name=collection,
        status=CollectionHealth.QUEUE_LAG,
        queue_max=lag * 2,
        watermark=lag,
        queue_lag=lag,
        reason=f"forced-poison test fixture (lag={lag})",
    )


def test_query_short_circuits_on_poisoned_collection(vs):
    vs.upsert(
        RECORDS_COLLECTION,
        ids=["a"],
        documents=["foo"],
        metadatas=[{"_": "_"}],
    )
    _force_poison(vs, RECORDS_COLLECTION)
    assert vs.is_poisoned(RECORDS_COLLECTION)
    res = vs.query(RECORDS_COLLECTION, query_texts=["foo"], n_results=5)
    assert res.is_degraded is True
    assert "poisoned" in res.degraded_reason
    assert res.is_empty()


def test_query_returns_one_inner_list_per_query_text_when_poisoned(vs):
    """The empty-result shape must match Chroma's: outer list with one
    inner list per query_text. Callers iterate ids[0], metadatas[0],
    etc., so a missing slot would crash them."""
    _force_poison(vs, RECORDS_COLLECTION)
    res = vs.query(RECORDS_COLLECTION, query_texts=["a", "b", "c"], n_results=2)
    assert len(res.ids) == 3
    assert all(slot == [] for slot in res.ids)
    assert len(res.documents) == 3
    assert len(res.metadatas) == 3
    assert len(res.distances) == 3


def test_upsert_short_circuits_on_poisoned_collection(vs):
    _force_poison(vs, RECORDS_COLLECTION)
    res = vs.upsert(
        RECORDS_COLLECTION,
        ids=["new"],
        documents=["x"],
        metadatas=[{"_": "_"}],
    )
    assert res.persisted is False
    assert "poisoned" in res.skipped_reason
    assert res.rows_affected == 0


def test_delete_short_circuits_on_poisoned_collection(vs):
    _force_poison(vs, RECORDS_COLLECTION)
    res = vs.delete(RECORDS_COLLECTION, ids=["x"])
    assert res.persisted is False
    assert "poisoned" in res.skipped_reason


def test_get_does_not_short_circuit_on_poisoned_collection(vs):
    """get() reads SQLite metadata segment only -- safe even on a
    poisoned palace. Must NOT short-circuit."""
    vs.upsert(
        RECORDS_COLLECTION,
        ids=["a", "b"],
        documents=["foo", "bar"],
        metadatas=[{"_": "_"}, {"_": "_"}],
    )
    _force_poison(vs, RECORDS_COLLECTION)
    res = vs.get(RECORDS_COLLECTION, ids=["a", "b"])
    assert res.is_degraded is False
    assert set(res.ids) == {"a", "b"}


def test_all_ids_works_on_poisoned_collection(vs):
    """Same reason as get(): SQLite-only, must not short-circuit."""
    ids = [f"i{n}" for n in range(20)]
    vs.upsert(
        RECORDS_COLLECTION,
        ids=ids,
        documents=["x"] * 20,
        metadatas=[{"_": "_"}] * 20,
    )
    _force_poison(vs, RECORDS_COLLECTION)
    fetched = vs.all_ids(RECORDS_COLLECTION)
    assert set(fetched) == set(ids)


# ─────────────────────────────────────────────────────────────────────
# Health refresh after rebuild
# ─────────────────────────────────────────────────────────────────────


def test_refresh_health_clears_stale_poisoning(vs):
    vs.upsert(
        RECORDS_COLLECTION,
        ids=["a"],
        documents=["x"],
        metadatas=[{"_": "_"}],
    )
    _force_poison(vs, RECORDS_COLLECTION)
    assert vs.is_poisoned(RECORDS_COLLECTION)
    # After a fresh scan against the (clean) palace, poisoning clears.
    vs.refresh_health()
    assert not vs.is_poisoned(RECORDS_COLLECTION)


# ─────────────────────────────────────────────────────────────────────
# Cache invalidation
# ─────────────────────────────────────────────────────────────────────


def test_invalidate_cache_drops_handle(vs):
    vs.upsert(RECORDS_COLLECTION, ids=["x"], documents=["d"], metadatas=[{"_": "_"}])
    assert RECORDS_COLLECTION in vs._collections
    vs.invalidate_cache(RECORDS_COLLECTION)
    assert RECORDS_COLLECTION not in vs._collections
    # Still works after invalidation -- handle re-opens lazily.
    res = vs.get(RECORDS_COLLECTION, ids=["x"])
    assert res.ids == ["x"]
