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

# chromadb removed as a runtime dep (Adrian directive 2026-05-12).
# SqliteVecVectorStore is the sole VectorStore implementation; the
# ABC is kept so future backends can implement against the same
# surface without bringing chromadb back.

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
    """Retired (2026-05-12, chromadb removed).

    Previously constructed a ``chromadb.PersistentClient``. Callers
    should now go through :func:`get_vector_store` for a backend-
    neutral :class:`VectorStore` handle. Kept as a symbol so older
    import sites raise a clear error instead of an opaque
    AttributeError on package upgrade."""
    raise RuntimeError(
        "make_vector_client is retired (chromadb removed 2026-05-12). "
        "Use mempalace.vector_store.get_vector_store(palace_path)."
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
# ChromaVectorStore REMOVED (Adrian directive 2026-05-12)
#
# The class lived here from Phase 1 of the chromadb sealing through
# Phase 5 default-flip. With chromadb dropped as a runtime dep,
# SqliteVecVectorStore is the only concrete VectorStore. The ABC in
# this module is kept so future backends (faiss, lance, etc.) can
# implement against the same surface without bringing chromadb back.
# ─────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────
# Module-level singleton helpers
# ─────────────────────────────────────────────────────────────────────


def make_persistent_client(palace_path: str) -> Any:
    """Backwards-compatible alias for :func:`make_vector_client`.

    Retained as a name so older miner/migrate/admin import sites
    don't break in user installs that pinned the symbol. The body now
    raises -- chromadb is no longer a runtime dep and there is no
    "persistent client" concept on the sqlite_vec backend (the
    SQLite connection IS the client; use :func:`get_vector_store`).
    """
    raise RuntimeError(
        "make_persistent_client is retired (chromadb removed 2026-05-12). "
        "Use mempalace.vector_store.get_vector_store(palace_path) for the "
        "backend-neutral VectorStore handle."
    )


_INSTANCES: dict[str, VectorStore] = {}

# Recognised values for MEMPALACE_VECTOR_BACKEND. sqlite_vec is the
# only backend post-2026-05-12 (chromadb removed). The env var stays
# so future backends can be A/B-tested at the same dispatch point.
_VALID_BACKENDS = ("sqlite_vec", "sqlite-vec", "sqlitevec")


def _resolve_backend() -> str:
    """Pick the backend to construct. Only ``sqlite_vec`` is wired
    today. The env var is honoured but anything other than sqlite_vec
    aliases is rejected with a clear error -- there is no chromadb
    fallback path."""
    return (os.environ.get("MEMPALACE_VECTOR_BACKEND") or "sqlite_vec").strip().lower()


def get_vector_store(palace_path: str | None = None) -> VectorStore:
    """Return a process-wide cached :class:`VectorStore` for
    ``palace_path``. Multiple callers within the same process share
    one instance per palace path, so the health scan + collection
    handles are cached. Pass ``None`` to use the active palace
    (resolved via ``MempalaceConfig``).

    Backend selection: today only ``sqlite_vec`` is wired -- it is
    the default and the only supported value. The env-var dispatch
    site remains so adding a future backend (faiss, lance, ...)
    requires touching only this function."""
    if palace_path is None:
        # Late import to avoid coupling vector_store -> mcp_server.
        from mempalace.mcp_server import MempalaceConfig  # noqa: PLC0415

        palace_path = MempalaceConfig().palace_path
    palace_path = os.path.abspath(palace_path)
    inst = _INSTANCES.get(palace_path)
    if inst is None:
        backend = _resolve_backend()
        if backend in _VALID_BACKENDS:
            from mempalace.sqlite_vec_store import (  # noqa: PLC0415
                SqliteVecVectorStore,
            )

            inst = SqliteVecVectorStore(palace_path)
        else:
            raise ValueError(
                f"Unknown vector backend {backend!r}. Valid options: "
                f"'sqlite_vec'. Set via MEMPALACE_VECTOR_BACKEND env var. "
                f"(chromadb backend was removed 2026-05-12.)"
            )
        _INSTANCES[palace_path] = inst
    return inst


def reset_singletons() -> None:
    """Drop every cached :class:`VectorStore`. Used by tests."""
    _INSTANCES.clear()
