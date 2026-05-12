"""VectorStore: the sole owner of all ChromaDB access in mempalace.

Application code never imports ``chromadb`` directly -- every Chroma
operation goes through this module's :class:`VectorStore` class. This
centralises:

  * **Health management** -- per-collection queue-lag detection, bloated
    HNSW link-list detection, schema/metadata defaults. The
    queue-lag/SIGSEGV-prevention logic that was previously a global
    monkey-patch lives here as a first-class method, with a structured
    return type so call sites see "this query was degraded because the
    collection is poisoned" instead of silently empty results.
  * **Collection bootstrap** -- ``hnsw:space=cosine`` and
    ``hnsw:sync_threshold=100`` (slice 16 prevention) are applied
    consistently to every ``get_or_create_collection``.
  * **Failure modes** -- write methods return :class:`WriteResult` with
    ``persisted=False`` and a structured ``skipped_reason`` rather than
    raising or silently dropping. Read methods return :class:`QueryResult`
    or :class:`GetResult` with the same explicit-degradation contract.
  * **Telemetry hooks** -- a single observation point for every Chroma
    call (timing, sizes, failures), to be wired up later as needed.

Design rationale
----------------

The motivating failure mode was ``MCP -32000 Connection closed`` --
chromadb's ``_apply_batch`` SIGSEGVs when the embeddings_queue has
unprocessed rows from a prior crashed session. Catching this at every
Chroma call site (``col.query``, ``col.upsert``, ``col.add``, ...)
required either a global monkey-patch (action-at-a-distance, brittle
across Chroma versions) or repeated try/except blocks at every caller
(noise, easy to miss). A repository pattern collapses both into one
file: the policy lives where the collection is opened, callers see
typed return values, and adding a new Chroma method (e.g. a future
``bulk_upsert``) requires one addition here instead of N call-site
edits.

Adrian directive 2026-05-09: "isn't better to centralize the access
via a repository implementation or something?". Yes -- this module is
that centralisation.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import chromadb

logger = logging.getLogger("mempalace.vector_store")


# ─────────────────────────────────────────────────────────────────────
# Constants -- canonical collection set + default Chroma metadata
# ─────────────────────────────────────────────────────────────────────

RECORDS_COLLECTION = "mempalace_records"
CONTEXT_VIEWS_COLLECTION = "mempalace_context_views"
TRIPLES_COLLECTION = "mempalace_triples"

KNOWN_COLLECTIONS: tuple[str, ...] = (
    RECORDS_COLLECTION,
    CONTEXT_VIEWS_COLLECTION,
    TRIPLES_COLLECTION,
)

# slice 16 (commit e13c073): hnsw:sync_threshold=100 caps the
# embeddings_queue lag a crashed session can leave behind, so the next
# session's first backfill replays at most 100 rows instead of the
# 1000-row default that consistently SIGSEGVs on Windows + Python 3.13
# + Chroma 0.6.3 (record_ga_agent_chroma_hnsw_corruption_all_collections).
DEFAULT_COLLECTION_METADATA: dict[str, Any] = {
    "hnsw:space": "cosine",
    "hnsw:sync_threshold": 100,
}


# ─────────────────────────────────────────────────────────────────────
# Backend client factory -- the sole construction site for the
# underlying vector backend's client object.
#
# Phase 1 of the chromadb sealing (Adrian directive 2026-05-11): every
# ``chromadb.PersistentClient(...)`` + ``chromadb.config.Settings(...)``
# pair in the codebase routes through this helper. Future phases swap
# the body to dispatch on a backend flag (chroma / sqlite_vec / ...)
# without touching callers.
# ─────────────────────────────────────────────────────────────────────


def make_vector_client(palace_path: str) -> Any:
    """Construct a vector-backend client rooted at ``palace_path``.

    Today this returns a ``chromadb.PersistentClient`` with
    ``anonymized_telemetry=False``. The function exists so that the
    palace bootstrap, the MCP server's client cache, and VectorStore's
    lazy client all share ONE construction site -- when the sqlite-vec
    backend lands, the dispatch happens here.

    Callers that previously did:

        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=palace_path,
            settings=Settings(anonymized_telemetry=False),
        )

    now write:

        from mempalace.vector_store import make_vector_client
        client = make_vector_client(palace_path)
    """
    # Import locally so that future backends can be selected without
    # forcing every importer of vector_store to pull chromadb in
    # transitively.
    import chromadb as _chromadb  # noqa: PLC0415
    from chromadb.config import Settings as _Settings  # noqa: PLC0415

    return _chromadb.PersistentClient(
        path=palace_path,
        settings=_Settings(anonymized_telemetry=False),
    )


# ─────────────────────────────────────────────────────────────────────
# Health classification
# ─────────────────────────────────────────────────────────────────────


class CollectionHealth(str, Enum):
    """Per-collection health status, ordered roughly by severity."""

    OK = "ok"
    EMPTY = "empty"
    QUEUE_LAG = "queue_lag"  # SIGSEGV-prone backfill state
    OVERSIZED = "oversized"  # link_lists.bin abnormally large
    MISSING = "missing"
    UNKNOWN = "unknown"


@dataclass
class HealthInfo:
    """Per-collection health snapshot."""

    name: str
    status: CollectionHealth
    row_count: int = 0
    queue_max: int = 0
    watermark: int = 0
    queue_lag: int = 0
    link_lists_bytes: int = 0
    reason: str = ""

    @property
    def is_poisoned(self) -> bool:
        """A collection is poisoned when querying or writing through
        Chroma may trigger the HNSW C-level SIGSEGV. Only ``QUEUE_LAG``
        currently triggers this -- ``OVERSIZED`` is a slow-degradation
        signal that doesn't crash on first read."""
        return self.status == CollectionHealth.QUEUE_LAG


# ─────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────


@dataclass
class QueryResult:
    """Cosine-search query result. Mirrors Chroma's
    ``Collection.query`` return shape (per-query lists of inner-lists)
    but adds explicit degradation flags."""

    ids: list[list[str]]
    documents: list[list[str | None]] = field(default_factory=list)
    metadatas: list[list[dict | None]] = field(default_factory=list)
    distances: list[list[float]] = field(default_factory=list)
    is_degraded: bool = False
    degraded_reason: str = ""

    @classmethod
    def empty(cls, *, n_query_texts: int = 1, reason: str = "") -> "QueryResult":
        return cls(
            ids=[[] for _ in range(n_query_texts)],
            documents=[[] for _ in range(n_query_texts)],
            metadatas=[[] for _ in range(n_query_texts)],
            distances=[[] for _ in range(n_query_texts)],
            is_degraded=bool(reason),
            degraded_reason=reason,
        )

    def is_empty(self) -> bool:
        return all(not slot for slot in self.ids)

    def total_hits(self) -> int:
        return sum(len(slot) for slot in self.ids)


@dataclass
class GetResult:
    """``col.get(...)`` result."""

    ids: list[str]
    documents: list[str | None] = field(default_factory=list)
    metadatas: list[dict | None] = field(default_factory=list)
    embeddings: list | None = None
    is_degraded: bool = False
    degraded_reason: str = ""

    @classmethod
    def empty(cls, reason: str = "") -> "GetResult":
        return cls(ids=[], is_degraded=bool(reason), degraded_reason=reason)


@dataclass
class WriteResult:
    """``upsert`` / ``add`` / ``update`` / ``delete`` result.

    ``persisted=True`` means Chroma accepted the write; ``False`` means
    the call was skipped due to poisoning, missing collection, or an
    underlying Chroma exception (captured as ``error``). Existing
    fallback paths around Chroma writes typically check ``persisted``
    and mark the affected entity as ``_views_persisted=False`` (see
    :func:`mempalace.mcp_server.context_lookup_or_create`)."""

    persisted: bool
    rows_affected: int = 0
    skipped_reason: str = ""
    error: str = ""

    @classmethod
    def skipped(cls, reason: str) -> "WriteResult":
        return cls(persisted=False, skipped_reason=reason)

    @classmethod
    def failed(cls, error: str) -> "WriteResult":
        return cls(persisted=False, error=error)

    @classmethod
    def ok(cls, rows: int) -> "WriteResult":
        return cls(persisted=True, rows_affected=rows)


# ─────────────────────────────────────────────────────────────────────
# VectorStore -- abstract backend contract
#
# The interface every concrete backend must satisfy. Branch-by-
# abstraction (Adrian directive 2026-05-11): callers depend on this
# contract, not on a specific backend. Today's only concrete impl is
# :class:`ChromaVectorStore`; :class:`SqliteVecVectorStore` lands in
# Phase 3 of the chroma sealing.
#
# Methods are listed in the same order as the concrete class so the
# two read in parallel. Result types (QueryResult / GetResult /
# WriteResult / HealthInfo) are backend-agnostic and live above.
# ─────────────────────────────────────────────────────────────────────


class VectorStore(ABC):
    """Abstract base for every vector-store backend.

    Concrete backends (:class:`ChromaVectorStore`,
    :class:`SqliteVecVectorStore`, ...) implement these methods.
    Callers should accept this type, not a concrete class -- the
    factory :func:`get_vector_store` returns whichever backend the
    palace is configured for.

    Failure-mode contract (shared across backends):

    * Read methods (:meth:`query`, :meth:`get`, :meth:`count`) return
      a typed result with an ``is_degraded`` / ``degraded_reason``
      pair instead of raising. Empty results on a degraded backend
      are explicit, not silent.
    * Write methods (:meth:`upsert`, :meth:`add`, :meth:`update`,
      :meth:`delete`) return :class:`WriteResult` with
      ``persisted=False`` and a ``skipped_reason`` / ``error`` rather
      than raising. Callers branch on ``persisted``.
    * :meth:`health` is cached; :meth:`refresh_health` re-scans.
      :meth:`is_poisoned` is the backend-specific definition of
      "writes / queries here may corrupt or crash" -- callers SHOULD
      check it before writes that they care about persisting.
    """

    # ── lifecycle ────────────────────────────────────────────────────

    @abstractmethod
    def refresh_health(self) -> dict[str, HealthInfo]:
        """Re-scan health for every known collection. Safe to call at
        any time. Concrete backends define what they probe; the
        return shape is uniform."""

    @abstractmethod
    def health(self, collection: str | None = None) -> HealthInfo | dict[str, HealthInfo]:
        """Return cached health info. Pass ``collection=None`` for
        the full map. Does NOT re-scan."""

    @abstractmethod
    def is_poisoned(self, collection: str) -> bool:
        """True if writes / reads through this collection may
        corrupt or crash. Backend-specific definition."""

    @abstractmethod
    def poisoned_collections(self) -> set[str]:
        """Set of currently-poisoned collection names."""

    @abstractmethod
    def invalidate_cache(self, collection: str | None = None) -> None:
        """Drop the open-collection cache. Forces the next access to
        re-open. Pass ``collection=None`` to invalidate everything."""

    # ── reads ────────────────────────────────────────────────────────

    @abstractmethod
    def query(
        self,
        collection: str,
        *,
        query_texts: list[str] | None = None,
        query_embeddings: list[list[float]] | None = None,
        n_results: int = 10,
        where: dict | None = None,
        where_document: dict | None = None,
        include: list[str] | None = None,
    ) -> QueryResult:
        """Cosine KNN search against ``collection``. Returns a
        :class:`QueryResult` matching chromadb's per-query inner-list
        shape (one inner list per query text)."""

    @abstractmethod
    def get(
        self,
        collection: str,
        *,
        ids: list[str] | None = None,
        where: dict | None = None,
        limit: int | None = None,
        include: list[str] | None = None,
    ) -> GetResult:
        """ID/where-based fetch."""

    @abstractmethod
    def count(self, collection: str) -> int:
        """Approximate row count for ``collection`` (whatever the
        backend exposes cheaply -- chromadb returns
        ``Collection.count()``, sqlite-vec returns ``SELECT count(*)
        FROM vec_<collection>``)."""

    @abstractmethod
    def sql_row_count(self, collection: str) -> int:
        """Authoritative row count, computed from the backend's own
        storage layer. May be slower than :meth:`count` but is the
        ground truth for repair / migrate tools."""

    @abstractmethod
    def all_ids(self, collection: str, *, batch_size: int = 5000) -> list[str]:
        """Return every id in ``collection``, paginated internally
        by ``batch_size``."""

    # ── writes ───────────────────────────────────────────────────────

    @abstractmethod
    def upsert(
        self,
        collection: str,
        *,
        ids: list[str],
        documents: list[str] | None = None,
        metadatas: list[dict] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> WriteResult:
        """Upsert ids. Existing rows are replaced; new rows are
        inserted."""

    @abstractmethod
    def add(
        self,
        collection: str,
        *,
        ids: list[str],
        documents: list[str] | None = None,
        metadatas: list[dict] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> WriteResult:
        """Strict insert -- duplicate ids fail."""

    @abstractmethod
    def update(
        self,
        collection: str,
        *,
        ids: list[str],
        documents: list[str] | None = None,
        metadatas: list[dict] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> WriteResult:
        """Strict update -- missing ids fail."""

    @abstractmethod
    def delete(
        self,
        collection: str,
        *,
        ids: list[str] | None = None,
        where: dict | None = None,
    ) -> WriteResult:
        """Delete by id list or by where-filter."""

    # ── collection lifecycle ─────────────────────────────────────────

    @abstractmethod
    def list_collections(self) -> list[str]:
        """All collection names known to the backend."""

    @abstractmethod
    def delete_collection(self, collection: str) -> WriteResult:
        """Drop the entire collection (rows + index + metadata)."""

    @abstractmethod
    def create_collection(self, collection: str, *, metadata: dict | None = None) -> WriteResult:
        """Create an empty collection with the given metadata."""


# ─────────────────────────────────────────────────────────────────────
# ChromaVectorStore -- the chromadb-backed implementation
# ─────────────────────────────────────────────────────────────────────


class ChromaVectorStore(VectorStore):
    """Single owner of all ChromaDB access for a palace.

    Construct once per process per palace path. Methods are safe to
    call concurrently from multiple threads; the underlying
    ``PersistentClient`` is thread-safe per Chroma's documentation.

    Health is scanned at construction (read-only SQLite reads, ~100us
    per collection). Re-scan via :meth:`refresh_health` after a
    rebuild or when the queue is suspected to have advanced.
    """

    def __init__(
        self,
        palace_path: str,
        *,
        scan_on_init: bool = True,
        collection_metadata: dict | None = None,
    ):
        self.palace_path = palace_path
        # Per-instance collection metadata override. Production uses
        # DEFAULT_COLLECTION_METADATA (sync_threshold=100, slice 16
        # SIGSEGV-prevention). Tests pass {hnsw:sync_threshold:1} so
        # writes are immediately visible to count/get/query without
        # waiting for the 100-row sync batch -- otherwise small-row
        # tests would see write-then-read return zero results.
        self._metadata = collection_metadata or DEFAULT_COLLECTION_METADATA
        self._client: chromadb.PersistentClient | None = None
        self._collections: dict[str, Any] = {}
        self._health: dict[str, HealthInfo] = {}
        # Per-collection last-_open-failure exception text. Populated by
        # _open's except block so write/read paths can surface the actual
        # Chroma error in WriteResult.skipped_reason instead of the opaque
        # "collection unavailable" that hid root cause during the
        # 2026-05-09 diary-write debug.
        self._last_open_errors: dict[str, str] = {}
        if scan_on_init:
            self.refresh_health()

    # ── lifecycle ────────────────────────────────────────────────────

    @property
    def client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self._client = make_vector_client(self.palace_path)
        return self._client

    def refresh_health(self) -> dict[str, HealthInfo]:
        """Re-scan health for every known collection. Safe to call at
        any time; uses read-only SQLite (mode=ro) so it cannot
        interfere with active writes.

        Slice 17 threshold: ``POISONED_QUEUE_LAG_THRESHOLD`` (=200,
        2x the slice-16 sync_threshold) gates the QUEUE_LAG verdict.
        Lag below the threshold is normal in-memory trailing-edge
        state (HNSW only flushes every 100 inserts), not corruption.
        """
        # Late import to avoid a circular at module load.
        from mempalace.repair import (  # noqa: PLC0415
            POISONED_QUEUE_LAG_THRESHOLD,
            _queue_lag_for_collection,
        )

        new_health: dict[str, HealthInfo] = {}
        for name in KNOWN_COLLECTIONS:
            try:
                info = _queue_lag_for_collection(self.palace_path, name)
                lag = int(info.get("lag", 0))
                wm = int(info.get("watermark", 0))
                qmax = int(info.get("queue_max", 0))
                if lag > POISONED_QUEUE_LAG_THRESHOLD and wm > 0:
                    status = CollectionHealth.QUEUE_LAG
                    reason = (
                        f"embeddings_queue has {lag} unprocessed row(s) "
                        f"(queue_max={qmax}, watermark={wm}, "
                        f"threshold={POISONED_QUEUE_LAG_THRESHOLD}); a "
                        f"prior session crashed mid-write -- next backfill "
                        f"may SIGSEGV in HNSW _apply_batch."
                    )
                elif qmax == 0:
                    status = CollectionHealth.EMPTY
                    reason = "no embeddings_queue activity yet"
                else:
                    status = CollectionHealth.OK
                    reason = (
                        f"healthy (queue_lag={lag} <= "
                        f"threshold={POISONED_QUEUE_LAG_THRESHOLD}; "
                        f"normal trailing-edge sync state)"
                    )
                new_health[name] = HealthInfo(
                    name=name,
                    status=status,
                    queue_max=qmax,
                    watermark=wm,
                    queue_lag=lag,
                    reason=reason,
                )
            except Exception as exc:
                new_health[name] = HealthInfo(
                    name=name,
                    status=CollectionHealth.UNKNOWN,
                    reason=f"health probe failed: {type(exc).__name__}: {exc}",
                )
        self._health = new_health
        return dict(new_health)

    def health(self, collection: str | None = None) -> HealthInfo | dict[str, HealthInfo]:
        """Return cached health info; pass ``collection=None`` for the
        full map. Does NOT re-scan -- call :meth:`refresh_health`
        first if you need fresh data."""
        if collection is None:
            return dict(self._health)
        return self._health.get(
            collection,
            HealthInfo(name=collection, status=CollectionHealth.UNKNOWN),
        )

    def is_poisoned(self, collection: str) -> bool:
        info = self._health.get(collection)
        return bool(info and info.is_poisoned)

    def poisoned_collections(self) -> set[str]:
        return {n for n, info in self._health.items() if info.is_poisoned}

    # ── collection access (internal) ─────────────────────────────────

    def _open(self, collection: str, *, create: bool = False) -> Any | None:
        """Lazy collection handle, cached. Returns None on failure.

        Records the last failure reason in ``self._last_open_errors[collection]``
        so callers (write paths, health scans) can surface the actual
        Chroma exception in their WriteResult.skipped_reason instead of
        the opaque "collection unavailable" message that hid the root
        cause for hours during the 2026-05-09 diary-write debug.
        """
        cached = self._collections.get(collection)
        if cached is not None:
            return cached
        try:
            if create:
                col = self.client.get_or_create_collection(collection, metadata=self._metadata)
            else:
                col = self.client.get_collection(collection)
            self._collections[collection] = col
            self._last_open_errors.pop(collection, None)
            return col
        except Exception as exc:
            err_text = f"{type(exc).__name__}: {exc}"
            if create:
                # Some Chroma versions only allow create_collection on a
                # truly missing collection; fall through.
                try:
                    col = self.client.create_collection(collection, metadata=self._metadata)
                    self._collections[collection] = col
                    self._last_open_errors.pop(collection, None)
                    return col
                except Exception as exc2:
                    err_text = (
                        f"get_or_create -> {type(exc).__name__}: {exc} | "
                        f"create -> {type(exc2).__name__}: {exc2}"
                    )
            self._last_open_errors[collection] = err_text
            # Warning level (was debug) so this surfaces in normal mcp_io_log
            # without enabling debug-level for the whole process.
            logger.warning(
                "VectorStore._open(%s, create=%s) failed: %s",
                collection,
                create,
                err_text,
            )
            return None

    def invalidate_cache(self, collection: str | None = None) -> None:
        """Drop cached handles. Call after delete_collection or rebuild."""
        if collection is None:
            self._collections.clear()
        else:
            self._collections.pop(collection, None)

    # ── read paths ───────────────────────────────────────────────────

    def query(
        self,
        collection: str,
        *,
        query_texts: list[str],
        n_results: int = 10,
        where: dict | None = None,
        where_document: dict | None = None,
        include: list[str] | None = None,
    ) -> QueryResult:
        """Cosine search against ``collection``. Returns a structured
        result; on a poisoned collection or any underlying failure,
        returns an :class:`empty <QueryResult.empty>` result with
        ``is_degraded=True`` and ``degraded_reason`` set."""
        n_qt = max(1, len(query_texts) if query_texts else 1)
        if self.is_poisoned(collection):
            return QueryResult.empty(
                n_query_texts=n_qt,
                reason=f"collection {collection!r} poisoned (queue_lag); "
                f"backfill would SIGSEGV in HNSW _apply_batch",
            )
        col = self._open(collection, create=False)
        if col is None:
            return QueryResult.empty(
                n_query_texts=n_qt,
                reason=f"collection {collection!r} unavailable",
            )
        try:
            kwargs: dict[str, Any] = {
                "query_texts": query_texts,
                "n_results": n_results,
                "include": include or ["metadatas", "documents", "distances"],
            }
            if where is not None:
                kwargs["where"] = where
            if where_document is not None:
                kwargs["where_document"] = where_document
            res = col.query(**kwargs)
            return QueryResult(
                ids=list(res.get("ids") or [[]]),
                documents=list(res.get("documents") or [[]]),
                metadatas=list(res.get("metadatas") or [[]]),
                distances=list(res.get("distances") or [[]]),
            )
        except Exception as exc:
            return QueryResult.empty(
                n_query_texts=n_qt,
                reason=f"query failed: {type(exc).__name__}: {exc}",
            )

    def get(
        self,
        collection: str,
        *,
        ids: list[str] | None = None,
        where: dict | None = None,
        where_document: dict | None = None,
        limit: int | None = None,
        offset: int | None = None,
        include: list[str] | None = None,
    ) -> GetResult:
        """Direct fetch by id or filter. Reads SQLite metadata segment
        only -- safe even on poisoned collections (does NOT trigger
        HNSW load), so we do NOT short-circuit on poisoning here."""
        col = self._open(collection, create=False)
        if col is None:
            return GetResult.empty(reason=f"collection {collection!r} unavailable")
        try:
            kwargs: dict[str, Any] = {}
            if ids is not None:
                kwargs["ids"] = ids
            if where is not None:
                kwargs["where"] = where
            if where_document is not None:
                kwargs["where_document"] = where_document
            if limit is not None:
                kwargs["limit"] = limit
            if offset is not None:
                kwargs["offset"] = offset
            if include is not None:
                kwargs["include"] = include
            res = col.get(**kwargs)
            return GetResult(
                ids=list(res.get("ids") or []),
                documents=list(res.get("documents") or []),
                metadatas=list(res.get("metadatas") or []),
                embeddings=res.get("embeddings"),
            )
        except Exception as exc:
            return GetResult.empty(reason=f"get failed: {type(exc).__name__}: {exc}")

    def count(self, collection: str) -> int:
        """Total row count for ``collection``.

        Uses :meth:`sql_row_count` (SQLite ``embeddings`` table) rather
        than Chroma's ``Collection.count()``. Chroma's count walks the
        HNSW index, which trails the SQLite store by up to
        ``hnsw:sync_threshold`` rows (production = 100, slice 16) -- so
        Chroma's count can under-report by a full batch on small or
        recently-written collections, AND returns 0 entirely on
        poisoned palaces where HNSW failed to load. SQLite is the
        source of truth for "how many rows did I store"; HNSW is just
        the searchable index built on top.
        """
        return self.sql_row_count(collection)

    def sql_row_count(self, collection: str) -> int:
        """SQLite-only row count for ``collection``. Walks the
        ``embeddings_queue`` table (Chroma's write-ahead log) and
        applies the per-id ADD/UPSERT/DELETE op semantics, so the
        count reflects every accepted write -- including ones that
        haven't been processed by the metadata or vector segments
        yet. The downstream ``embeddings`` table only reflects writes
        that the SegmentManager has already drained, which lags
        behind by the sync_threshold (=100 in production); using
        that table for "how many rows do I have" under-reports on
        small collections AND on freshly-rebuilt palaces.

        Op codes (Chroma 0.6, verified empirically against the live
        embeddings_queue table -- the values differ from what their
        public types module hints):
          1 = ADD       (id created -- count it)
          2 = UPSERT    (id created or updated -- count it)
          3 = DELETE    (id removed -- subtract)
          4 = UPDATE    (id must exist -- already counted)

        Returns 0 if the palace, sqlite file, or collection is
        missing. Read-only access (``mode=ro``); cannot interfere
        with active writes.
        """
        import sqlite3 as _sqlite  # noqa: PLC0415

        sqlite_path = os.path.join(self.palace_path, "chroma.sqlite3")
        if not os.path.exists(sqlite_path):
            return 0
        try:
            conn = _sqlite.connect(f"file:{sqlite_path}?mode=ro", uri=True)
            try:
                row = conn.execute(
                    "SELECT id FROM collections WHERE name = ?", (collection,)
                ).fetchone()
                if not row:
                    return 0
                topic = f"persistent://default/default/{row[0]}"
                # Count distinct ids that have a final non-DELETE op.
                # Walk in seq order: ADD/UPSERT puts id into the live
                # set; DELETE removes it. Last write per id wins.
                live: set[str] = set()
                cur = conn.execute(
                    "SELECT id, operation FROM embeddings_queue "
                    "WHERE topic = ? ORDER BY seq_id ASC",
                    (topic,),
                )
                for ent_id, op in cur:
                    if op == 3:  # DELETE
                        live.discard(ent_id)
                    else:  # 1=ADD, 2=UPSERT, 4=UPDATE -- treat as live
                        live.add(ent_id)
                return len(live)
            finally:
                conn.close()
        except Exception:
            return 0

    def all_ids(self, collection: str, *, batch_size: int = 5000) -> list[str]:
        """Iterate every id via paginated ``col.get``. Safe on
        poisoned collections."""
        col = self._open(collection, create=False)
        if col is None:
            return []
        out: list[str] = []
        offset = 0
        while True:
            try:
                batch = col.get(limit=batch_size, offset=offset, include=[])
            except Exception:
                break
            ids = batch.get("ids") or []
            if not ids:
                break
            out.extend(ids)
            offset += len(ids)
            if offset > 10_000_000:
                break  # safety -- runaway corruption
        return out

    # ── write paths ──────────────────────────────────────────────────

    def upsert(
        self,
        collection: str,
        *,
        ids: list[str],
        documents: list[str] | None = None,
        metadatas: list[dict] | None = None,
        embeddings: list | None = None,
    ) -> WriteResult:
        """Insert-or-update by id."""
        return self._write_op(
            collection,
            "upsert",
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def add(
        self,
        collection: str,
        *,
        ids: list[str],
        documents: list[str] | None = None,
        metadatas: list[dict] | None = None,
        embeddings: list | None = None,
    ) -> WriteResult:
        """Insert new rows; raises in Chroma if any id already exists."""
        return self._write_op(
            collection,
            "add",
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def update(
        self,
        collection: str,
        *,
        ids: list[str],
        documents: list[str] | None = None,
        metadatas: list[dict] | None = None,
        embeddings: list | None = None,
    ) -> WriteResult:
        """Update existing rows by id."""
        return self._write_op(
            collection,
            "update",
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def delete(
        self,
        collection: str,
        *,
        ids: list[str] | None = None,
        where: dict | None = None,
    ) -> WriteResult:
        """Delete by id or filter. Skipped on poisoned collections to
        avoid triggering the segment load."""
        if self.is_poisoned(collection):
            return WriteResult.skipped(f"collection {collection!r} poisoned -- delete deferred")
        col = self._open(collection, create=False)
        if col is None:
            cause = self._last_open_errors.get(collection, "no underlying error captured")
            return WriteResult.skipped(f"collection {collection!r} unavailable: {cause}")
        try:
            kwargs: dict[str, Any] = {}
            if ids is not None:
                kwargs["ids"] = ids
            if where is not None:
                kwargs["where"] = where
            col.delete(**kwargs)
            return WriteResult.ok(rows=len(ids) if ids else 0)
        except Exception as exc:
            return WriteResult.failed(f"delete: {type(exc).__name__}: {exc}")

    def _write_op(
        self,
        collection: str,
        op_name: str,
        *,
        ids: list[str],
        documents: list[str] | None,
        metadatas: list[dict] | None,
        embeddings: list | None,
    ) -> WriteResult:
        if self.is_poisoned(collection):
            return WriteResult.skipped(
                f"collection {collection!r} poisoned (queue_lag); "
                f"{op_name} would SIGSEGV in HNSW _apply_batch"
            )
        col = self._open(collection, create=True)
        if col is None:
            cause = self._last_open_errors.get(collection, "no underlying error captured")
            return WriteResult.skipped(f"collection {collection!r} unavailable: {cause}")
        kwargs: dict[str, Any] = {"ids": ids}
        if documents is not None:
            kwargs["documents"] = documents
        if metadatas is not None:
            kwargs["metadatas"] = metadatas
        if embeddings is not None:
            kwargs["embeddings"] = embeddings
        try:
            getattr(col, op_name)(**kwargs)
            return WriteResult.ok(rows=len(ids))
        except Exception as exc:
            return WriteResult.failed(f"{op_name}: {type(exc).__name__}: {exc}")

    # ── admin ────────────────────────────────────────────────────────

    def list_collections(self) -> list[str]:
        try:
            return list(self.client.list_collections())
        except Exception:
            return []

    def delete_collection(self, collection: str) -> WriteResult:
        try:
            self.client.delete_collection(collection)
            self.invalidate_cache(collection)
            return WriteResult.ok(rows=1)
        except Exception as exc:
            return WriteResult.failed(f"delete_collection: {type(exc).__name__}: {exc}")

    def create_collection(self, collection: str, *, metadata: dict | None = None) -> WriteResult:
        try:
            self.client.create_collection(
                collection, metadata=metadata or DEFAULT_COLLECTION_METADATA
            )
            self.invalidate_cache(collection)
            return WriteResult.ok(rows=1)
        except Exception as exc:
            return WriteResult.failed(f"create_collection: {type(exc).__name__}: {exc}")


# ─────────────────────────────────────────────────────────────────────
# Module-level singleton helpers
# ─────────────────────────────────────────────────────────────────────


def make_persistent_client(palace_path: str) -> chromadb.PersistentClient:
    """Backwards-compatible alias for :func:`make_vector_client`.

    Kept so cli/repair/miner/migrate/admin callers that imported the
    old name don't break. New code should call ``make_vector_client``
    directly -- the backend-neutral name signals that swapping
    backends (Phase 3+) doesn't require touching the call sites.
    """
    return make_vector_client(palace_path)


_INSTANCES: dict[str, VectorStore] = {}


def _resolve_backend() -> str:
    """Pick the backend to construct.

    Today: ``chroma`` (the only one shipped). Phase 5 will flip the
    default to ``sqlite_vec`` after the new backend lands and parity-
    tests pass. The env var ``MEMPALACE_VECTOR_BACKEND`` overrides
    -- agents debugging a single palace can pin a backend without
    touching the config.
    """
    return (os.environ.get("MEMPALACE_VECTOR_BACKEND") or "chroma").strip().lower()


def get_vector_store(palace_path: str | None = None) -> VectorStore:
    """Return a process-wide cached :class:`VectorStore` for
    ``palace_path``. Multiple callers within the same process share
    one instance per palace path, so the health scan + collection
    handles are cached. Pass ``None`` to use the active palace
    (resolved via ``MempalaceConfig``).

    Backend selection (Adrian directive 2026-05-11, branch-by-
    abstraction Phase 2): the concrete class returned depends on
    :func:`_resolve_backend`. Today only ``chroma`` is wired;
    Phase 3 adds ``sqlite_vec``.

    Tests should construct the concrete backend
    (:class:`ChromaVectorStore` / :class:`SqliteVecVectorStore`)
    directly to avoid polluting the singleton cache.
    """
    if palace_path is None:
        # Late import to avoid coupling vector_store -> mcp_server.
        from mempalace.mcp_server import MempalaceConfig  # noqa: PLC0415

        palace_path = MempalaceConfig().palace_path
    palace_path = os.path.abspath(palace_path)
    inst = _INSTANCES.get(palace_path)
    if inst is None:
        backend = _resolve_backend()
        if backend in ("chroma", "chromadb"):
            inst = ChromaVectorStore(palace_path)
        else:
            raise ValueError(
                f"Unknown vector backend {backend!r}. Valid options "
                f"today: 'chroma'. Sqlite-vec lands in Phase 3."
            )
        _INSTANCES[palace_path] = inst
    return inst


def reset_singletons() -> None:
    """Drop every cached :class:`VectorStore`. Used by tests."""
    _INSTANCES.clear()
