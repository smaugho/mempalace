"""SqliteVecVectorStore -- sqlite-vec-backed concrete :class:`VectorStore`.

Branch-by-abstraction Phase 3 (Adrian directive 2026-05-11). Drop-in
replacement for :class:`ChromaVectorStore` -- same abstract contract,
same call signatures, same result types. The failure class that
motivated this refactor (chromadb's HNSW C-extension SIGSEGVing on
queue_lag) literally cannot occur here: there is no embeddings_queue,
no backfill, no separate process / shared library managing vector
state. sqlite-vec is a SQLite extension that owns its storage as
ordinary SQLite pages.

Storage model
-------------

ONE file per palace, ``<palace_path>/knowledge_graph.sqlite3`` -- the
same file the knowledge_graph module owns. Three sqlite-vec-managed
tables live inside it:

* ``vec_palace`` -- vec0 virtual table with:

    CREATE VIRTUAL TABLE vec_palace USING vec0(
        collection TEXT PARTITION KEY,
        embedding float[384] distance_metric=cosine,
        +entity_id TEXT,
        +document TEXT,
        +metadata TEXT
    )

  The partition key scopes KNN to one collection cheaply. Aux columns
  (``+``) ride along with each row; they are returned in query results
  but not filterable during KNN (sqlite-vec restriction). Filterable
  metadata is post-filtered in Python after over-fetching.

* ``vec_rowid_map`` -- ``(collection TEXT, entity_id TEXT, rowid INTEGER,
  PRIMARY KEY (collection, entity_id))``. Maps human-facing entity_ids
  to vec0's INTEGER rowid space. Updates / deletes look up the rowid
  here; queries return entity_id from the aux column directly.

* ``vec_collections`` -- ``(name TEXT PRIMARY KEY, metadata TEXT,
  created_at INTEGER)``. Tracks which collections exist + per-
  collection metadata blob (JSON). Required because vec0's PARTITION
  KEY doesn't expose a partition-list query.

Why one file + same DB as the KG: atomic transactions span the KG's
entity writes AND vector writes. The historical "Chroma write
succeeded but KG write failed" drift class disappears.

Why one vec0 table with PARTITION KEY (instead of three vec_records /
vec_context_views / vec_triples tables): sqlite-vec partitions are
cheap and the read path is one query per collection regardless. One
table keeps the surface narrow.

Where-filter semantics
----------------------

chromadb's ``where`` DSL is rich: ``{key: value}`` equality,
``{key: {"$eq": ..., "$ne": ..., ...}}``, ``{"$and": [...]}``,
``{"$or": [...]}``. Live mempalace code uses overwhelmingly
``{key: value}`` (equality) and occasional ``{"$and": [...]}``. We
implement those plus the basic comparison operators; unknown
operators raise. Filters apply **after** the KNN fetch (over-fetch
factor 10x by default) because sqlite-vec only allows MATCH / k /
partition-key constraints in the same KNN query.

For unfiltered reads (``get`` with no where, ``list_collections``,
``count``, etc.), the rowid_map / collections tracking tables give us
ordinary fast SQL.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import struct
import threading
import time
from typing import Any, Callable

from .vector_store import (
    CollectionHealth,
    DEFAULT_COLLECTION_METADATA,
    GetResult,
    HealthInfo,
    KNOWN_COLLECTIONS,
    QueryResult,
    VectorStore,
    WriteResult,
)

logger = logging.getLogger("mempalace.sqlite_vec_store")


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

# Filename of the SQLite database that hosts the vec0 virtual table.
# Same file as KnowledgeGraph -- one substrate per palace.
_DB_FILENAME = "knowledge_graph.sqlite3"

# Embedding dimensionality. MiniLM-L6-v2 ships 384-dim L2-normalised
# cosine vectors. Hardcoded today; if/when mempalace migrates to a
# different model this becomes a per-palace config.
_DEFAULT_EMBEDDING_DIM = 384

# Internal sqlite-vec table names.
_VEC_TABLE = "vec_palace"
_ROWID_MAP_TABLE = "vec_rowid_map"
_COLLECTIONS_TABLE = "vec_collections"

# When the caller supplies a ``where`` filter, sqlite-vec can't apply
# it during KNN (aux columns aren't filterable in a vec0 MATCH query).
# We over-fetch and post-filter in Python. 10x is a sane default --
# matches chromadb's empirical recall on filtered queries at small k.
_OVERFETCH_FACTOR = 10


# ─────────────────────────────────────────────────────────────────────
# Vector packing / rowid hashing
# ─────────────────────────────────────────────────────────────────────


def _pack_vec(vec: list[float]) -> bytes:
    """Serialize a Python float list to a packed-float32 BLOB. Matches
    sqlite-vec's expected wire format for ``float[N]`` columns."""
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_vec(blob: bytes, dim: int = _DEFAULT_EMBEDDING_DIM) -> list[float]:
    """Inverse of :func:`_pack_vec`. Used only by tests + repair tools."""
    return list(struct.unpack(f"{dim}f", blob))


def _stable_rowid(collection: str, entity_id: str) -> int:
    """Deterministic positive 63-bit rowid for ``(collection, entity_id)``.

    BLAKE2b-64 truncated to 63 bits (top bit zeroed) keeps the value in
    SQLite's positive INTEGER range. Collision probability at 100k rows
    is < 1e-10. The rowid_map table provides the authoritative lookup;
    this function only seeds new inserts when no row exists yet, so
    even a hash collision would surface as an ``add`` failure rather
    than silent data corruption."""
    h = hashlib.blake2b(f"{collection}\0{entity_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") & 0x7FFFFFFFFFFFFFFF


# ─────────────────────────────────────────────────────────────────────
# Where-filter compiler
# ─────────────────────────────────────────────────────────────────────

WherePredicate = Callable[[dict[str, Any]], bool]


def _compile_where(where: dict | None) -> WherePredicate:
    """Compile a chromadb-style ``where`` dict into a Python predicate
    operating on a row's metadata dict.

    Supported forms:

    * ``{"key": value}`` -- equality
    * ``{"key": {"$eq": value}}`` -- equality
    * ``{"key": {"$ne": value}}`` -- inequality
    * ``{"key": {"$gt": value}}`` / ``$gte`` / ``$lt`` / ``$lte``
    * ``{"key": {"$in": [values]}}`` / ``$nin``
    * ``{"$and": [clauses]}`` -- all must match
    * ``{"$or": [clauses]}`` -- any must match

    Unknown operators raise ``ValueError`` -- silent ignore would
    return wrong results.
    """
    if not where:
        return lambda meta: True

    def compile_clause(clause: dict) -> WherePredicate:
        # Top-level boolean combinators
        if "$and" in clause:
            children = [compile_clause(c) for c in clause["$and"]]
            return lambda meta: all(c(meta) for c in children)
        if "$or" in clause:
            children = [compile_clause(c) for c in clause["$or"]]
            return lambda meta: any(c(meta) for c in children)

        # Otherwise a dict of {field: value-or-operator-dict} clauses.
        # Multiple keys on the same level are implicit AND.
        per_field: list[WherePredicate] = []
        for field, condition in clause.items():
            if field.startswith("$"):
                raise ValueError(f"unsupported where operator at top level: {field!r}")
            if isinstance(condition, dict):
                for op, arg in condition.items():
                    per_field.append(_field_predicate(field, op, arg))
            else:
                # Sugar: {field: value} == {field: {"$eq": value}}
                per_field.append(_field_predicate(field, "$eq", condition))
        if not per_field:
            return lambda meta: True
        if len(per_field) == 1:
            return per_field[0]
        return lambda meta: all(p(meta) for p in per_field)

    return compile_clause(where)


def _extract_entity_id_in_filter(where: dict | None) -> list | None:
    """v3.9.3 Tier A: if ``where`` selects ONLY on ``entity_id`` via
    equality or ``$in``, return the list of entity_ids; otherwise None.

    The keyword channel resolves matched entity_ids back to their
    document+metadata via ``get(where={"entity_id": {"$in": [...]}})``.
    Without this detector that call falls into the whole-collection scan
    path (load every rowid, fetch every row, filter in Python) -- O(N)
    in collection size. When this returns a non-None list, ``get`` can
    instead resolve through the indexed ``vec_rowid_map.entity_id_ref``
    column (idx_vec_rowid_map_entity_id_ref, shipped v3.2.6) -- O(matched
    eids). Returns None for any other filter shape so the general
    predicate path still handles it. See research_keyword_channel_latency
    _fts5_2026_05_20.
    """
    if not where or not isinstance(where, dict) or len(where) != 1:
        return None
    cond = where.get("entity_id")
    if cond is None and "entity_id" not in where:
        return None
    if isinstance(cond, dict):
        if len(cond) != 1:
            return None
        op, arg = next(iter(cond.items()))
        if op == "$eq":
            return [arg]
        if op == "$in" and isinstance(arg, (list, tuple)):
            return list(arg)
        return None
    # scalar-equality sugar: {"entity_id": value}
    return [cond]


def _field_predicate(field: str, op: str, arg: Any) -> WherePredicate:
    """Single-field predicate factory. Closes over the field name and
    the operator's argument so the returned callable is a hot-path
    one-liner."""
    if op == "$eq":
        return lambda meta: meta.get(field) == arg
    if op == "$ne":
        return lambda meta: meta.get(field) != arg
    if op == "$gt":
        return lambda meta: meta.get(field) is not None and meta[field] > arg
    if op == "$gte":
        return lambda meta: meta.get(field) is not None and meta[field] >= arg
    if op == "$lt":
        return lambda meta: meta.get(field) is not None and meta[field] < arg
    if op == "$lte":
        return lambda meta: meta.get(field) is not None and meta[field] <= arg
    if op == "$in":
        members = set(arg)
        return lambda meta: meta.get(field) in members
    if op == "$nin":
        members = set(arg)
        return lambda meta: meta.get(field) not in members
    raise ValueError(f"unsupported where operator {op!r} on field {field!r}")


# ─────────────────────────────────────────────────────────────────────
# SqliteVecVectorStore
# ─────────────────────────────────────────────────────────────────────


class SqliteVecVectorStore(VectorStore):
    """sqlite-vec implementation of :class:`VectorStore`.

    Construction opens / creates the palace's SQLite file, loads the
    sqlite-vec extension, and bootstraps the three tracking tables.
    The connection is held for the lifetime of the store; methods
    serialize through ``_lock`` because ``sqlite3.Connection`` is not
    thread-safe across writes.

    Health is uniformly OK: sqlite-vec storage has no segfault class.
    The :meth:`is_poisoned` contract is preserved (returns False),
    which is what callers check before attempting writes.
    """

    def __init__(
        self,
        palace_path: str,
        *,
        scan_on_init: bool = True,
        collection_metadata: dict | None = None,
        embedding_dim: int = _DEFAULT_EMBEDDING_DIM,
    ):
        self.palace_path = palace_path
        self._metadata = collection_metadata or DEFAULT_COLLECTION_METADATA
        self._embedding_dim = embedding_dim
        self._health: dict[str, HealthInfo] = {}
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        # Set if _bootstrap() can't open the DB (palace dir missing,
        # permission denied, etc.). Methods short-circuit to degraded
        # results when this is non-empty -- mirrors ChromaVectorStore's
        # ``_last_open_errors`` behavior so the searcher's "no palace"
        # branch fires uniformly across backends.
        self._bootstrap_error: str = ""
        try:
            self._bootstrap()
        except Exception as exc:
            self._bootstrap_error = f"{type(exc).__name__}: {exc}"
            logger.info(
                "SqliteVecVectorStore bootstrap failed at %s: %s",
                palace_path,
                self._bootstrap_error,
            )
        if scan_on_init and not self._bootstrap_error:
            self.refresh_health()

    # ── lifecycle ────────────────────────────────────────────────────

    def _bootstrap(self) -> None:
        """Open the SQLite connection, load sqlite-vec, ensure tables.

        We do NOT create ``self.palace_path`` here -- palace lifecycle
        is the caller's responsibility (the palace-init / migration
        path handles ``os.makedirs``). When the path doesn't exist
        sqlite3 raises ``OperationalError: unable to open database
        file``, which the searcher CLI uses as the "no palace" signal.
        """
        db_path = os.path.join(self.palace_path, _DB_FILENAME)
        # check_same_thread=False + an explicit lock lets us share one
        # connection across the (currently single-threaded but
        # ThreadPoolExecutor-using) injection_gate parallel block.
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.enable_load_extension(True)
        try:
            import sqlite_vec  # noqa: PLC0415

            sqlite_vec.load(conn)
        finally:
            conn.enable_load_extension(False)

        # WAL gives us cross-connection read concurrency while a
        # writer is in flight. KnowledgeGraph already uses WAL so
        # this is a no-op if the file is already configured.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

        # Tracking tables -- regular SQLite, not virtual.
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_COLLECTIONS_TABLE} (
                name TEXT PRIMARY KEY,
                metadata TEXT NOT NULL DEFAULT '{{}}',
                created_at INTEGER NOT NULL
            )
            """
        )
        # v3.2.6 (Adrian directive 2026-05-12): vec_rowid_map maps a
        # LOGICAL vec record id (across four namespaces -- bare {eid},
        # multi-view {eid}__v{i}, context-view {cid}_v{i}, triple_id)
        # to its physical vec0 rowid. The historical column was named
        # entity_id, which lied -- only the bare-entity case was an
        # entities.id ref. v3.2.6 renames it to logical_id and adds two
        # real FK columns (entity_id_ref -> entities, triple_id_ref ->
        # triples, both ON DELETE CASCADE) so the schema-level cascade
        # cleans rowid_map rows when their parent is deleted. A
        # BEFORE DELETE trigger on this table then cleans the matching
        # vec_palace row (vec0 virtual tables can't take FKs directly).
        #
        # The CREATE below is for FRESH palaces; existing palaces are
        # transformed by _migrate_to_v326_schema() called below.
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_ROWID_MAP_TABLE} (
                collection      TEXT NOT NULL,
                logical_id      TEXT NOT NULL,
                rowid           INTEGER NOT NULL,
                entity_id_ref   TEXT REFERENCES entities(id) ON DELETE CASCADE,
                triple_id_ref   TEXT REFERENCES triples(id)  ON DELETE CASCADE,
                PRIMARY KEY (collection, logical_id)
            )
            """
        )
        # v3.4.1 bootstrap-order fix (Adrian post-reinstall 2026-05-13):
        # the CREATE INDEX statements below reference entity_id_ref /
        # triple_id_ref columns. On pre-v3.2.6 palaces the CREATE TABLE
        # IF NOT EXISTS above is a no-op (table exists with old shape:
        # collection / entity_id / rowid), so the indexes fail with
        # "no such column: entity_id_ref", __init__ catches the
        # exception into _bootstrap_error, and the v3.2.6 migration
        # below NEVER FIRES. Result: every connection to the old
        # palace silently degrades the vector store. Fix: run the
        # migration FIRST so the table reaches the new shape before
        # any index references the new columns. The migration itself
        # creates the indexes; the post-migration CREATE INDEX IF
        # NOT EXISTS statements below are idempotent no-ops on
        # already-migrated palaces.
        self._migrate_to_v326_schema(conn)
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{_ROWID_MAP_TABLE}_rowid ON {_ROWID_MAP_TABLE} (rowid)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{_ROWID_MAP_TABLE}_collection "
            f"ON {_ROWID_MAP_TABLE} (collection)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{_ROWID_MAP_TABLE}_entity_id_ref "
            f"ON {_ROWID_MAP_TABLE} (entity_id_ref)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{_ROWID_MAP_TABLE}_triple_id_ref "
            f"ON {_ROWID_MAP_TABLE} (triple_id_ref)"
        )
        # BEFORE DELETE trigger -- fires on direct DELETE and on FK
        # CASCADE delete (SQLite docs confirm BEFORE DELETE triggers
        # fire during cascade). Removes the matching vec_palace row
        # so the virtual table doesn't accumulate orphans when
        # entities or triples are dropped.
        conn.execute(
            f"""
            CREATE TRIGGER IF NOT EXISTS trg_{_ROWID_MAP_TABLE}_cascade_to_{_VEC_TABLE}
            BEFORE DELETE ON {_ROWID_MAP_TABLE}
            BEGIN
                DELETE FROM {_VEC_TABLE} WHERE rowid = OLD.rowid;
            END
            """
        )

        # vec0 virtual table. Creating a vec0 table that already exists
        # is an error; gate with sqlite_master lookup.
        already = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (_VEC_TABLE,),
        ).fetchone()
        if not already:
            conn.execute(
                f"""
                CREATE VIRTUAL TABLE {_VEC_TABLE} USING vec0(
                    collection TEXT PARTITION KEY,
                    embedding float[{self._embedding_dim}] distance_metric=cosine,
                    +entity_id TEXT,
                    +document TEXT,
                    +metadata TEXT
                )
                """
            )

        # Pre-register the canonical mempalace collections so
        # list_collections returns the expected set even before any
        # write lands.
        now = int(time.time())
        for name in KNOWN_COLLECTIONS:
            conn.execute(
                f"INSERT OR IGNORE INTO {_COLLECTIONS_TABLE} "
                f"(name, metadata, created_at) VALUES (?, ?, ?)",
                (name, json.dumps(dict(self._metadata)), now),
            )
        conn.commit()
        self._conn = conn

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._bootstrap()
        assert self._conn is not None
        return self._conn

    def close(self) -> None:
        """Close the underlying connection. Tests use this; production
        keeps the connection for the process lifetime."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # ── health ───────────────────────────────────────────────────────

    def refresh_health(self) -> dict[str, HealthInfo]:
        """Re-scan health. sqlite-vec collections are either present
        with rows, present-but-empty, or missing -- there's no
        analogue to chroma's queue_lag poisoning. UNKNOWN occurs only
        if the SQL probe itself raises.
        """
        new_health: dict[str, HealthInfo] = {}
        with self._lock:
            conn = self.conn
            for name in KNOWN_COLLECTIONS:
                try:
                    row = conn.execute(
                        f"SELECT count(*) FROM {_ROWID_MAP_TABLE} WHERE collection = ?",
                        (name,),
                    ).fetchone()
                    row_count = int(row[0]) if row else 0
                    if row_count == 0:
                        status = CollectionHealth.EMPTY
                        reason = "no rows in collection yet"
                    else:
                        status = CollectionHealth.OK
                        reason = f"{row_count} rows"
                    new_health[name] = HealthInfo(
                        name=name,
                        status=status,
                        row_count=row_count,
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
        if collection is None:
            return dict(self._health)
        return self._health.get(
            collection,
            HealthInfo(
                name=collection,
                status=CollectionHealth.UNKNOWN,
                reason="never scanned",
            ),
        )

    def is_poisoned(self, collection: str) -> bool:
        """Always False. sqlite-vec storage has no failure class
        equivalent to chroma's HNSW queue_lag SIGSEGV. Callers that
        branch on this remain correct -- writes always proceed."""
        return False

    def poisoned_collections(self) -> set[str]:
        return set()

    def invalidate_cache(self, collection: str | None = None) -> None:
        """No persistent open-collection cache for sqlite-vec (the
        SQLite connection is the only handle). Re-scan health
        defensively so callers that invalidate after an external
        write see fresh row counts."""
        self.refresh_health()

    # ── compat shim for chroma-flavoured tests ──────────────────────

    def _open(self, name: str, create: bool = False):
        """Backwards-compat shim mirroring ChromaVectorStore._open(name,
        create=True). On the chroma backend this returns a raw
        chromadb.Collection; on sqlite_vec there's nothing analogous
        because the vec0 virtual table is single-physical-table /
        partition-keyed, and writes auto-register collection names via
        :meth:`_ensure_collection`.

        We register the collection name eagerly when ``create=True`` so
        :meth:`list_collections` reflects it, and return ``self`` (a
        VectorStore-shaped object) so older test fixtures that did
        ``vs._open(name, create=True)`` to "warm" a collection still
        function without per-test branching.

        New code SHOULD NOT call this -- use the public surface
        (:meth:`upsert`, :meth:`get`, :meth:`query`, etc.) which
        auto-creates as needed.
        """
        if create:
            with self._lock:
                conn = self.conn
                self._ensure_collection(conn, name)
                conn.commit()
        return self

    # ── helpers ──────────────────────────────────────────────────────

    def _ensure_collection(self, conn: sqlite3.Connection, name: str) -> None:
        """Register ``name`` in the collections table if absent. Idempotent."""
        conn.execute(
            f"INSERT OR IGNORE INTO {_COLLECTIONS_TABLE} "
            f"(name, metadata, created_at) VALUES (?, ?, ?)",
            (name, json.dumps(dict(self._metadata)), int(time.time())),
        )

    def _embed_documents(self, documents: list[str]) -> list[list[float]] | None:
        """Auto-embed via :mod:`mempalace.embedder`. Returns ``None`` when
        the embedder isn't available -- callers degrade rather than
        crash."""
        from .embedder import get_default_embedder  # noqa: PLC0415

        embedder = get_default_embedder()
        if embedder is None:
            return None
        try:
            return embedder(documents)
        except Exception as exc:
            logger.warning("auto-embed failed: %s", exc)
            return None

    def _resolve_rowid(
        self, conn: sqlite3.Connection, collection: str, entity_id: str
    ) -> int | None:
        """Look up rowid for ``(collection, logical_id)`` in the map.
        Returns ``None`` if not present."""
        row = conn.execute(
            f"SELECT rowid FROM {_ROWID_MAP_TABLE} WHERE collection = ? AND logical_id = ?",
            (collection, entity_id),
        ).fetchone()
        return int(row[0]) if row else None

    # ── v3.2.6 schema-migration + FK-derivation helpers ──────────────────

    @staticmethod
    def _derive_fk_refs(collection: str, logical_id: str) -> tuple[str | None, str | None]:
        """Return ``(entity_id_ref, triple_id_ref)`` for a logical vec id.

        Mapping rules (v3.2.6, Adrian directive 2026-05-12):
          * ``collection='mempalace_triples'`` -> ``triple_id_ref = logical_id``
            (triple statement rows reference the triples table).
          * logical_id matches ``<prefix>__v<digits>`` -> multi-view row;
            ``entity_id_ref = prefix`` (the parent entity in entities).
          * logical_id matches ``<prefix>_v<digits>`` -> context-view row;
            ``entity_id_ref = prefix`` (the parent context entity, which
            is a kind='context' row in entities).
          * else -> bare entity row; ``entity_id_ref = logical_id``.

        Production scan 2026-05-12 verified 0 unmatched across 56k rows
        so the dual __v / _v suffix split is sufficient. Callers must
        still tolerate the case where the derived ref doesn't match a
        live parent (the FK enforces; the helper just classifies).
        """
        import re as _re  # noqa: PLC0415

        if collection == "mempalace_triples":
            return (None, logical_id)
        m = _re.match(r"^(.+)__v\d+$", logical_id)
        if m:
            return (m.group(1), None)
        m = _re.match(r"^(.+)_v\d+$", logical_id)
        if m:
            return (m.group(1), None)
        return (logical_id, None)

    def _migrate_to_v326_schema(self, conn: sqlite3.Connection) -> None:
        """One-shot transform of vec_rowid_map from the pre-v3.2.6 shape.

        Pre-v3.2.6 the table had columns ``(collection, entity_id, rowid)``
        with no FK. v3.2.6 renames ``entity_id`` to ``logical_id`` and
        adds ``entity_id_ref`` + ``triple_id_ref`` FK columns with
        ``ON DELETE CASCADE``. SQLite can't ALTER TABLE to rename a
        column AND add FKs in one shot, so this method does the
        canonical recreate-and-swap dance.

        Idempotent via STAMP ``vec_rowid_map_v326_2026_05_12`` in the
        ``data_migrations`` table (already created by the KG yoyo
        migrations -- if it doesn't exist yet, we're on a fresh palace
        where the bootstrap CREATE already used the new schema, so the
        migration is a no-op).
        """
        STAMP = "vec_rowid_map_v326_2026_05_12"
        # If data_migrations doesn't exist (KG init hasn't run yet),
        # skip -- the bootstrap CREATE above already gave us the new
        # shape for fresh palaces.
        try:
            applied = conn.execute(
                "SELECT 1 FROM data_migrations WHERE name = ?", (STAMP,)
            ).fetchone()
        except sqlite3.OperationalError:
            return
        if applied:
            return
        # Probe current schema -- if entity_id_ref column exists we are
        # already on the new shape (fresh palace path) and just need to
        # stamp.
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info({_ROWID_MAP_TABLE})").fetchall()}
        from datetime import datetime as _datetime  # noqa: PLC0415

        if "entity_id_ref" in cols:
            conn.execute(
                "INSERT OR IGNORE INTO data_migrations(name, applied_at) VALUES (?, ?)",
                (STAMP, _datetime.now().isoformat()),
            )
            conn.commit()
            return
        # Old shape -- recreate-and-swap. Turn FK enforcement off on
        # this connection while we run the transform so the data-copy
        # phase doesn't trip on temporarily-incomplete refs.
        prev_fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        conn.execute("PRAGMA foreign_keys=OFF")
        try:
            conn.execute(
                f"""
                CREATE TABLE {_ROWID_MAP_TABLE}_new (
                    collection      TEXT NOT NULL,
                    logical_id      TEXT NOT NULL,
                    rowid           INTEGER NOT NULL,
                    entity_id_ref   TEXT REFERENCES entities(id) ON DELETE CASCADE,
                    triple_id_ref   TEXT REFERENCES triples(id)  ON DELETE CASCADE,
                    PRIMARY KEY (collection, logical_id)
                )
                """
            )
            # Backfill in Python via the helper -- handles the parse
            # rules consistently with the live write path.
            rows = conn.execute(
                f"SELECT collection, entity_id, rowid FROM {_ROWID_MAP_TABLE}"
            ).fetchall()
            for row in rows:
                col, lid, rid = row[0], row[1], int(row[2])
                ent_ref, tri_ref = self._derive_fk_refs(col, lid)
                conn.execute(
                    f"INSERT INTO {_ROWID_MAP_TABLE}_new "
                    f"(collection, logical_id, rowid, entity_id_ref, triple_id_ref) "
                    f"VALUES (?, ?, ?, ?, ?)",
                    (col, lid, rid, ent_ref, tri_ref),
                )
            # Swap tables
            conn.execute(f"DROP TABLE {_ROWID_MAP_TABLE}")
            conn.execute(f"ALTER TABLE {_ROWID_MAP_TABLE}_new RENAME TO {_ROWID_MAP_TABLE}")
            # Indexes -- mirror the bootstrap CREATE
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{_ROWID_MAP_TABLE}_rowid "
                f"ON {_ROWID_MAP_TABLE} (rowid)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{_ROWID_MAP_TABLE}_collection "
                f"ON {_ROWID_MAP_TABLE} (collection)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{_ROWID_MAP_TABLE}_entity_id_ref "
                f"ON {_ROWID_MAP_TABLE} (entity_id_ref)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{_ROWID_MAP_TABLE}_triple_id_ref "
                f"ON {_ROWID_MAP_TABLE} (triple_id_ref)"
            )
            # Trigger (the bootstrap above already creates it for fresh
            # palaces via CREATE TRIGGER IF NOT EXISTS, but the DROP
            # TABLE above also dropped the trigger -- recreate).
            conn.execute(
                f"""
                CREATE TRIGGER IF NOT EXISTS trg_{_ROWID_MAP_TABLE}_cascade_to_{_VEC_TABLE}
                BEFORE DELETE ON {_ROWID_MAP_TABLE}
                BEGIN
                    DELETE FROM {_VEC_TABLE} WHERE rowid = OLD.rowid;
                END
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO data_migrations(name, applied_at) VALUES (?, ?)",
                (STAMP, _datetime.now().isoformat()),
            )
            conn.commit()
        finally:
            # Restore FK enforcement state.
            conn.execute(f"PRAGMA foreign_keys={'ON' if prev_fk else 'OFF'}")

    # ── reads ────────────────────────────────────────────────────────

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
        if self._bootstrap_error:
            # Palace missing / file unreadable -- mirror the
            # "collection X unavailable" reason that ChromaVectorStore
            # uses so the searcher's no-palace branch matches.
            return QueryResult.empty(
                n_query_texts=len(query_texts or query_embeddings or [None]),
                reason=f"collection {collection!r} unavailable: {self._bootstrap_error}",
            )
        # where_document is unsupported (no chromadb-style document
        # full-text constraint here). Live mempalace code never passes
        # it; assert to surface accidental use.
        if where_document:
            return QueryResult.empty(
                n_query_texts=len(query_texts or query_embeddings or [None]),
                reason="where_document not supported in sqlite-vec backend",
            )

        # Resolve query vectors. If the caller passed texts, embed them.
        if not query_embeddings:
            if not query_texts:
                return QueryResult.empty(
                    n_query_texts=1, reason="no query_texts or query_embeddings"
                )
            embedded = self._embed_documents(list(query_texts))
            if embedded is None:
                return QueryResult.empty(
                    n_query_texts=len(query_texts),
                    reason="embedder unavailable for query_texts",
                )
            query_embeddings = embedded

        try:
            predicate = _compile_where(where)
        except ValueError as exc:
            return QueryResult.empty(
                n_query_texts=len(query_embeddings),
                reason=f"unsupported where filter: {exc}",
            )

        ids_outer: list[list[str]] = []
        docs_outer: list[list[str | None]] = []
        metas_outer: list[list[dict | None]] = []
        dists_outer: list[list[float]] = []

        with self._lock:
            conn = self.conn
            # Over-fetch only when post-filter is needed; otherwise
            # k = n_results gives sqlite-vec the exact request.
            k = n_results * _OVERFETCH_FACTOR if where else n_results
            for qvec in query_embeddings:
                try:
                    rows = conn.execute(
                        f"SELECT entity_id, document, metadata, distance "
                        f"FROM {_VEC_TABLE} "
                        f"WHERE collection = ? AND embedding MATCH ? AND k = ? "
                        f"ORDER BY distance",
                        (collection, _pack_vec(list(qvec)), int(k)),
                    ).fetchall()
                except Exception as exc:
                    logger.info("query failed: %s", exc)
                    ids_outer.append([])
                    docs_outer.append([])
                    metas_outer.append([])
                    dists_outer.append([])
                    continue

                ids_inner: list[str] = []
                docs_inner: list[str | None] = []
                metas_inner: list[dict | None] = []
                dists_inner: list[float] = []
                for eid, doc, meta_json, dist in rows:
                    meta = json.loads(meta_json) if meta_json else {}
                    if not predicate(meta):
                        continue
                    ids_inner.append(eid)
                    docs_inner.append(doc)
                    metas_inner.append(meta or None)
                    dists_inner.append(float(dist))
                    if len(ids_inner) >= n_results:
                        break
                ids_outer.append(ids_inner)
                docs_outer.append(docs_inner)
                metas_outer.append(metas_inner)
                dists_outer.append(dists_inner)

        return QueryResult(
            ids=ids_outer,
            documents=docs_outer,
            metadatas=metas_outer,
            distances=dists_outer,
        )

    def get(
        self,
        collection: str,
        *,
        ids: list[str] | None = None,
        where: dict | None = None,
        limit: int | None = None,
        offset: int = 0,
        include: list[str] | None = None,
    ) -> GetResult:
        if self._bootstrap_error:
            return GetResult.empty(
                reason=f"collection {collection!r} unavailable: {self._bootstrap_error}"
            )
        try:
            predicate = _compile_where(where)
        except ValueError as exc:
            return GetResult.empty(reason=f"unsupported where filter: {exc}")

        # ``offset`` is meaningful only on the whole-collection scan
        # path. The ID-list path is point-style (caller already
        # specified exactly which logical_ids to fetch) so offset is
        # silently ignored there. Bound to non-negative to match
        # chromadb's behaviour where negative offsets were rejected.
        try:
            offset_i = max(0, int(offset))
        except (TypeError, ValueError):
            offset_i = 0

        out_ids: list[str] = []
        out_docs: list[str | None] = []
        out_metas: list[dict | None] = []

        with self._lock:
            conn = self.conn
            # v3.9.3 Tier A: detect a pure entity_id equality/$in filter so
            # the keyword channel resolves via the indexed entity_id_ref
            # column instead of a whole-collection scan.
            _eid_filter = None if ids else _extract_entity_id_in_filter(where)
            if ids:
                # ID-list path: bulk SELECT from rowid_map, then fetch
                # the matching rows from vec_palace by rowid.
                placeholders = ",".join("?" * len(ids))
                rowid_rows = conn.execute(
                    f"SELECT logical_id, rowid FROM {_ROWID_MAP_TABLE} "
                    f"WHERE collection = ? AND logical_id IN ({placeholders})",
                    (collection, *ids),
                ).fetchall()
                rowid_for: dict[str, int] = {r[0]: int(r[1]) for r in rowid_rows}
                # Preserve caller's ID order.
                wanted_rowids = [rowid_for[eid] for eid in ids if eid in rowid_for]
                if wanted_rowids:
                    rid_ph = ",".join("?" * len(wanted_rowids))
                    rows = conn.execute(
                        f"SELECT entity_id, document, metadata "
                        f"FROM {_VEC_TABLE} "
                        f"WHERE collection = ? AND rowid IN ({rid_ph})",
                        (collection, *wanted_rowids),
                    ).fetchall()
                    by_eid = {r[0]: r for r in rows}
                    for eid in ids:
                        row = by_eid.get(eid)
                        if not row:
                            continue
                        meta = json.loads(row[2]) if row[2] else {}
                        if not predicate(meta):
                            continue
                        out_ids.append(eid)
                        out_docs.append(row[1])
                        out_metas.append(meta or None)
                        if limit is not None and len(out_ids) >= limit:
                            break
            elif _eid_filter is not None:
                # v3.9.3 Tier A: pure entity_id filter -> resolve rowids via
                # the indexed vec_rowid_map.entity_id_ref column instead of
                # the whole-collection scan below. O(matched eids), not
                # O(collection). predicate() is still applied per row so the
                # output is identical to the scan path (entity_id_ref agrees
                # with the metadata entity_id for entity/record/context rows;
                # the predicate filters any divergent row exactly as the scan
                # would). Empty filter list -> no rows, skip the query.
                if _eid_filter:
                    eph = ",".join("?" * len(_eid_filter))
                    rowid_rows = conn.execute(
                        f"SELECT rowid FROM {_ROWID_MAP_TABLE} "
                        f"WHERE collection = ? AND entity_id_ref IN ({eph})",
                        (collection, *_eid_filter),
                    ).fetchall()
                    if rowid_rows:
                        rids = [int(r[0]) for r in rowid_rows]
                        rid_ph = ",".join("?" * len(rids))
                        rows = conn.execute(
                            f"SELECT entity_id, document, metadata "
                            f"FROM {_VEC_TABLE} "
                            f"WHERE collection = ? AND rowid IN ({rid_ph})",
                            (collection, *rids),
                        ).fetchall()
                        for eid, doc, meta_json in rows:
                            meta = json.loads(meta_json) if meta_json else {}
                            if not predicate(meta):
                                continue
                            out_ids.append(eid)
                            out_docs.append(doc)
                            out_metas.append(meta or None)
                            if limit is not None and len(out_ids) >= limit:
                                break
            else:
                # Whole-collection path. Use rowid_map for paging
                # since vec_palace doesn't support a plain
                # ``SELECT * FROM vec_palace WHERE collection = ?``
                # cleanly (vec0 wants MATCH or rowid scan).
                #
                # Pre-filter LIMIT/OFFSET applied at the rowid scan so
                # that callers paginating across large collections
                # (Layer1.generate, dedup.get_source_groups) only load
                # the current page's rowids -- avoids the prior O(N)
                # per-page load and matches chromadb's classic
                # offset-on-raw-collection / where-applied-per-row
                # semantics. Predicate-restrictive callers may see
                # fewer than ``limit`` rows per page and converge by
                # advancing offset by ``len(batch.ids)``.
                if limit is not None:
                    rowid_rows = conn.execute(
                        f"SELECT rowid FROM {_ROWID_MAP_TABLE} "
                        f"WHERE collection = ? "
                        f"ORDER BY rowid LIMIT ? OFFSET ?",
                        (collection, int(limit), offset_i),
                    ).fetchall()
                elif offset_i:
                    rowid_rows = conn.execute(
                        f"SELECT rowid FROM {_ROWID_MAP_TABLE} "
                        f"WHERE collection = ? "
                        f"ORDER BY rowid LIMIT -1 OFFSET ?",
                        (collection, offset_i),
                    ).fetchall()
                else:
                    rowid_rows = conn.execute(
                        f"SELECT rowid FROM {_ROWID_MAP_TABLE} WHERE collection = ? ORDER BY rowid",
                        (collection,),
                    ).fetchall()
                if rowid_rows:
                    rids = [int(r[0]) for r in rowid_rows]
                    rid_ph = ",".join("?" * len(rids))
                    rows = conn.execute(
                        f"SELECT entity_id, document, metadata "
                        f"FROM {_VEC_TABLE} "
                        f"WHERE collection = ? AND rowid IN ({rid_ph})",
                        (collection, *rids),
                    ).fetchall()
                    for eid, doc, meta_json in rows:
                        meta = json.loads(meta_json) if meta_json else {}
                        if not predicate(meta):
                            continue
                        out_ids.append(eid)
                        out_docs.append(doc)
                        out_metas.append(meta or None)
                        if limit is not None and len(out_ids) >= limit:
                            break

        return GetResult(ids=out_ids, documents=out_docs, metadatas=out_metas)

    def count(self, collection: str) -> int:
        with self._lock:
            row = self.conn.execute(
                f"SELECT count(*) FROM {_ROWID_MAP_TABLE} WHERE collection = ?",
                (collection,),
            ).fetchone()
            return int(row[0]) if row else 0

    def sql_row_count(self, collection: str) -> int:
        # Identical to count() for sqlite-vec -- one storage layer,
        # one source of truth. Chroma had two (the SQLite metadata
        # store + the HNSW segment); sqlite-vec collapses them.
        return self.count(collection)

    def all_ids(self, collection: str, *, batch_size: int = 5000) -> list[str]:
        out: list[str] = []
        with self._lock:
            offset = 0
            while True:
                rows = self.conn.execute(
                    f"SELECT logical_id FROM {_ROWID_MAP_TABLE} "
                    f"WHERE collection = ? LIMIT ? OFFSET ?",
                    (collection, batch_size, offset),
                ).fetchall()
                if not rows:
                    break
                out.extend(r[0] for r in rows)
                if len(rows) < batch_size:
                    break
                offset += batch_size
        return out

    # ── writes ───────────────────────────────────────────────────────

    def _write_row(
        self,
        conn: sqlite3.Connection,
        collection: str,
        entity_id: str,
        embedding: list[float],
        document: str | None,
        metadata: dict | None,
        *,
        mode: str,
    ) -> tuple[bool, str]:
        """Internal: write a single row. ``mode`` is one of
        ``add`` (strict insert -- duplicate fails),
        ``update`` (strict update -- missing fails),
        ``upsert`` (replace-or-insert).

        Returns ``(persisted, reason)``. Reason is empty on success."""
        existing_rowid = self._resolve_rowid(conn, collection, entity_id)

        if mode == "add" and existing_rowid is not None:
            return False, f"duplicate entity_id {entity_id!r} in {collection!r}"
        if mode == "update" and existing_rowid is None:
            return False, f"missing entity_id {entity_id!r} in {collection!r}"

        rowid = existing_rowid or _stable_rowid(collection, entity_id)
        meta_json = json.dumps(metadata) if metadata else None

        # vec0 doesn't support INSERT OR REPLACE; do delete-then-insert.
        if existing_rowid is not None:
            conn.execute(f"DELETE FROM {_VEC_TABLE} WHERE rowid = ?", (rowid,))
        conn.execute(
            f"INSERT INTO {_VEC_TABLE} "
            f"(rowid, collection, embedding, entity_id, document, metadata) "
            f"VALUES (?, ?, ?, ?, ?, ?)",
            (rowid, collection, _pack_vec(embedding), entity_id, document, meta_json),
        )
        # v3.2.6: populate the two FK ref columns so DELETE on the
        # parent (entities/triples) cascades to this row, and the
        # BEFORE DELETE trigger then cleans the matching vec_palace
        # row. Mapping rules in _derive_fk_refs.
        ent_ref, tri_ref = self._derive_fk_refs(collection, entity_id)
        conn.execute(
            f"INSERT OR REPLACE INTO {_ROWID_MAP_TABLE} "
            f"(collection, logical_id, rowid, entity_id_ref, triple_id_ref) "
            f"VALUES (?, ?, ?, ?, ?)",
            (collection, entity_id, rowid, ent_ref, tri_ref),
        )
        return True, ""

    def _write_batch(
        self,
        collection: str,
        *,
        ids: list[str],
        documents: list[str] | None,
        metadatas: list[dict] | None,
        embeddings: list[list[float]] | None,
        mode: str,
    ) -> WriteResult:
        if not ids:
            return WriteResult.ok(0)
        n = len(ids)
        docs = documents or [None] * n
        metas = metadatas or [None] * n
        embs = embeddings
        if embs is None:
            # Need documents to auto-embed. None-documents in this
            # path means caller forgot both -- nothing to write.
            text_inputs = [d if d is not None else "" for d in docs]
            embs = self._embed_documents(text_inputs)
            if embs is None:
                return WriteResult.skipped("embedder unavailable and no embeddings provided")
        if len(docs) != n or len(metas) != n or len(embs) != n:
            return WriteResult.failed(
                f"length mismatch: ids={n} docs={len(docs)} metas={len(metas)} embs={len(embs)}"
            )

        rows_affected = 0
        first_error = ""
        with self._lock:
            conn = self.conn
            self._ensure_collection(conn, collection)
            try:
                with conn:  # atomic transaction
                    for eid, doc, meta, emb in zip(ids, docs, metas, embs):
                        ok, reason = self._write_row(
                            conn, collection, eid, list(emb), doc, meta, mode=mode
                        )
                        if not ok:
                            # Aborting the transaction rolls everything back.
                            first_error = reason
                            raise _BatchAbort(reason)
                        rows_affected += 1
            except _BatchAbort:
                return WriteResult.failed(first_error)
            except Exception as exc:
                return WriteResult.failed(f"{type(exc).__name__}: {exc}")

        return WriteResult.ok(rows_affected)

    def upsert(
        self,
        collection: str,
        *,
        ids: list[str],
        documents: list[str] | None = None,
        metadatas: list[dict] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> WriteResult:
        return self._write_batch(
            collection,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
            mode="upsert",
        )

    def add(
        self,
        collection: str,
        *,
        ids: list[str],
        documents: list[str] | None = None,
        metadatas: list[dict] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> WriteResult:
        return self._write_batch(
            collection,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
            mode="add",
        )

    def update(
        self,
        collection: str,
        *,
        ids: list[str],
        documents: list[str] | None = None,
        metadatas: list[dict] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> WriteResult:
        return self._write_batch(
            collection,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
            mode="update",
        )

    def delete(
        self,
        collection: str,
        *,
        ids: list[str] | None = None,
        where: dict | None = None,
    ) -> WriteResult:
        # Resolve target id set. Either an explicit list or anything
        # matching the where filter.
        try:
            predicate = _compile_where(where)
        except ValueError as exc:
            return WriteResult.failed(f"unsupported where filter: {exc}")

        with self._lock:
            conn = self.conn
            if ids:
                target_ids = list(ids)
            else:
                # Need to walk the collection to evaluate the filter
                # (sqlite-vec doesn't expose metadata via WHERE on
                # aux columns). Use the get() walk minus the limit.
                got = self.get(collection, where=where)
                if got.is_degraded:
                    return WriteResult.failed(got.degraded_reason)
                target_ids = list(got.ids)

            if not target_ids:
                return WriteResult.ok(0)

            placeholders = ",".join("?" * len(target_ids))
            rowid_rows = conn.execute(
                f"SELECT rowid FROM {_ROWID_MAP_TABLE} "
                f"WHERE collection = ? AND logical_id IN ({placeholders})",
                (collection, *target_ids),
            ).fetchall()
            if not rowid_rows:
                return WriteResult.ok(0)
            rowids = [int(r[0]) for r in rowid_rows]

            try:
                with conn:
                    # v3.2.6: deleting the rowid_map row fires the
                    # BEFORE DELETE trigger which removes the
                    # corresponding vec_palace row. Keep the explicit
                    # vec_palace DELETE for the edge case where a row
                    # exists in vec_palace but not in rowid_map (would
                    # only happen under corruption); idempotent.
                    rid_ph = ",".join("?" * len(rowids))
                    conn.execute(
                        f"DELETE FROM {_VEC_TABLE} WHERE rowid IN ({rid_ph})",
                        tuple(rowids),
                    )
                    conn.execute(
                        f"DELETE FROM {_ROWID_MAP_TABLE} "
                        f"WHERE collection = ? AND logical_id IN ({placeholders})",
                        (collection, *target_ids),
                    )
            except Exception as exc:
                return WriteResult.failed(f"{type(exc).__name__}: {exc}")

            # ``predicate`` is technically unused when ``ids`` was the
            # source path; keep a reference so the local doesn't get
            # flagged as dead by linters when ``where`` is None.
            _ = predicate
            return WriteResult.ok(len(rowids))

    # ── collection lifecycle ─────────────────────────────────────────

    def list_collections(self) -> list[str]:
        with self._lock:
            rows = self.conn.execute(
                f"SELECT name FROM {_COLLECTIONS_TABLE} ORDER BY name"
            ).fetchall()
            return [r[0] for r in rows]

    def delete_collection(self, collection: str) -> WriteResult:
        with self._lock:
            conn = self.conn
            try:
                with conn:
                    conn.execute(
                        f"DELETE FROM {_VEC_TABLE} WHERE collection = ?",
                        (collection,),
                    )
                    conn.execute(
                        f"DELETE FROM {_ROWID_MAP_TABLE} WHERE collection = ?",
                        (collection,),
                    )
                    cur = conn.execute(
                        f"DELETE FROM {_COLLECTIONS_TABLE} WHERE name = ?",
                        (collection,),
                    )
                    return WriteResult.ok(cur.rowcount)
            except Exception as exc:
                return WriteResult.failed(f"{type(exc).__name__}: {exc}")

    def create_collection(self, collection: str, *, metadata: dict | None = None) -> WriteResult:
        meta_json = json.dumps(metadata or dict(self._metadata))
        with self._lock:
            conn = self.conn
            try:
                with conn:
                    cur = conn.execute(
                        f"INSERT OR REPLACE INTO {_COLLECTIONS_TABLE} "
                        f"(name, metadata, created_at) VALUES (?, ?, ?)",
                        (collection, meta_json, int(time.time())),
                    )
                    return WriteResult.ok(cur.rowcount)
            except Exception as exc:
                return WriteResult.failed(f"{type(exc).__name__}: {exc}")


class _BatchAbort(Exception):
    """Internal sentinel for rolling back a write batch on first error."""
