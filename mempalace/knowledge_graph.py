"""
knowledge_graph.py — Temporal Entity-Relationship Graph for MemPalace
=====================================================================

Real knowledge graph with:
  - Entity nodes (people, projects, tools, concepts)
  - Typed relationship edges (daughter_of, does, loves, works_on, etc.)
  - Temporal validity (valid_from → valid_to — knows WHEN facts are true)
  - Closet references (links back to the verbatim memory)

Storage: SQLite (local, no dependencies, no subscriptions)
Query: entity-first traversal with time filtering

This is what competes with Zep's temporal knowledge graph.
Zep uses Neo4j in the cloud ($25/mo+). We use SQLite locally (free).

Usage:
    from mempalace.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph()
    kg.add_triple("Max", "child_of", "Alice", valid_from="2015-04-01")
    kg.add_triple("Max", "does", "swimming", valid_from="2025-01-01")
    kg.add_triple("Max", "loves", "chess", valid_from="2025-10-01")

    # Query: everything about Max
    kg.query_entity("Max")

    # Query: what was true about Max in January 2026?
    kg.query_entity("Max", as_of="2026-01-15")

    # Query: who is connected to Alice?
    kg.query_entity("Alice", direction="both")

    # Invalidate: Max's sports injury resolved
    kg.invalidate("Max", "has_issue", "sports_injury", ended="2026-02-15")
"""

import atexit
import hashlib
import json
import os
import re
import sqlite3
from datetime import date, datetime
from pathlib import Path

# Track all KG instances for cleanup on exit
_active_instances = []


def _cleanup_all():
    """Close all KG connections on process exit to release WAL locks."""
    for kg in _active_instances:
        try:
            kg.close()
        except Exception:
            pass


atexit.register(_cleanup_all)


# BF1: legacy default location used when KG was kept outside the palace dir.
# Pre-2026-04-18 default. The current canonical location is
# {config.palace_path}/knowledge_graph.sqlite3 (computed in __init__ when
# db_path is None). LEGACY_KG_PATH is checked at first init and migrated in
# place when the canonical file is missing or empty, so existing installs
# don't lose data on the path move.
LEGACY_KG_PATH = os.path.expanduser("~/.mempalace/knowledge_graph.sqlite3")
DEFAULT_KG_PATH = LEGACY_KG_PATH  # kept as alias for any external imports


def _resolve_default_kg_path() -> str:
    """Return the canonical KG path: inside the resolved palace directory.

    Falls back to LEGACY_KG_PATH if MempalaceConfig can't load (test setups,
    bootstrap edge cases) so the existing module-import behaviour stays safe.
    """
    try:
        from .config import MempalaceConfig

        return os.path.join(MempalaceConfig().palace_path, "knowledge_graph.sqlite3")
    except Exception:
        return LEGACY_KG_PATH


def _maybe_migrate_legacy_kg(canonical_path: str) -> None:
    """Move LEGACY_KG_PATH -> canonical_path on first init when canonical is
    missing or zero-byte and legacy has data. Idempotent and safe to call on
    every KG construction; no-op when there's nothing to migrate.
    """
    try:
        if canonical_path == LEGACY_KG_PATH:
            return  # nothing to migrate when both paths coincide
        if not os.path.exists(LEGACY_KG_PATH):
            return
        try:
            legacy_size = os.path.getsize(LEGACY_KG_PATH)
        except OSError:
            legacy_size = 0
        if legacy_size == 0:
            return
        canonical_size = os.path.getsize(canonical_path) if os.path.exists(canonical_path) else 0
        if canonical_size > 0:
            return  # canonical already has data; don't clobber it
        Path(canonical_path).parent.mkdir(parents=True, exist_ok=True)
        # Move legacy file plus any -wal/-shm sidecar files SQLite may have left.
        for suffix in ("", "-wal", "-shm"):
            src = LEGACY_KG_PATH + suffix
            dst = canonical_path + suffix
            if os.path.exists(src):
                if os.path.exists(dst):
                    os.remove(dst)  # only happens for empty/orphan dst
                os.replace(src, dst)
    except Exception:
        pass  # migration is best-effort; bad migration shouldn't crash startup


# ── Triple verbalization (research-style "triple-to-text" for retrieval) ──
# Each triple gets a natural-language sentence stored on the row + embedded
# into the mempalace_triples Chroma collection. That makes triples
# first-class search citizens alongside prose memories and entities; without
# this, a query like "who lives in Warsaw" misses an `(adrian, lives_in,
# warsaw)` triple unless prose memory text happens to match the words.

TRIPLE_COLLECTION_NAME = "mempalace_triples"

# Predicates that are pure schema glue (type membership, attribution,
# narrative back-references). Verbalizing them ("research is a inspect",
# "memory_X described_by memory_Y") just floods retrieval with low-signal
# generic statements that drown semantic content. Skip them at index time
# — structural facts are still in the SQL triples table and walkable via
# BFS, just not embedded for similarity search.
_TRIPLE_SKIP_PREDICATES = {
    "is_a",
    "described_by",
    "evidenced_by",
    "executed_by",
    "targeted",
    "has_value",
    "session_note_for",
    "derived_from",
    "mentioned_in",
    # Context-as-entity predicates. All pure graph topology — the
    # context system exposes them via kg_query, never via semantic
    # search over synthesised statements (which would add noise).
    "created_under",  # provenance from node to context
    "similar_to",  # context-to-context neighbourhood
    "surfaced",  # retrieval-event edge (context → surfaced entity)
    "rated_useful",  # positive feedback edge
    "rated_irrelevant",  # negative feedback edge
}


# NOTE: an older ``_verbalize_triple`` helper used to exist here as a naive
# "replace underscores with spaces" fallback for callers that omitted
# ``statement``. It was removed 2026-04-19 — see the TripleStatementRequired
# policy below. Autogenerated statements produced low-signal text like
# "record ga agent a relates to record ga agent b" that drowned real
# retrievals. Now callers either supply a real sentence or the non-skip
# edge is rejected at write time.


class TripleStatementRequired(ValueError):
    """Raised by ``add_triple`` when a non-structural edge is created
    without a caller-provided ``statement``.

    Skip-list predicates (``is_a``, ``described_by``, ``executed_by``, …)
    remain statement-optional because ``_index_triple_statement`` never
    embeds them anyway — they're schema glue, walkable via BFS, not
    searched by similarity. For every other predicate, the caller MUST
    supply a natural-language verbalization or the triple is refused.
    """


def _get_triple_collection(create: bool = False):
    """Return the mempalace_triples Chroma collection or None on any error.

    Lazy import + best-effort to avoid coupling the SQL layer to ChromaDB
    at construction time. Uses the live mcp_server _STATE.client_cache when
    available so we share the embedding model + persistent client.

    When `create=False` (default — used by search-side callers) we only
    return the collection if it already exists, so a search call never
    has the side effect of creating a new Chroma collection in palaces
    that have no triples yet. Write-side callers (add_triple,
    backfill_triple_statements) pass create=True.
    """
    try:
        from . import mcp_server

        client = mcp_server._get_client()
        if client is None:
            return None
        if create:
            return client.get_or_create_collection(
                TRIPLE_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
            )
        try:
            return client.get_collection(TRIPLE_COLLECTION_NAME)
        except Exception:
            return None
    except Exception:
        return None


def _index_triple_statement(kg, triple_id, sub_id, pred, obj_id, statement, confidence):
    """Upsert the verbalized statement into the triples Chroma collection.

    Best-effort: silent no-op on any failure so write-side errors never
    block the SQL insert. The SQL row remains the source of truth; the
    Chroma index is rebuildable via backfill_triple_statements().

    Structural predicates (is_a, described_by, executed_by, ...) are
    deliberately NOT embedded — see _TRIPLE_SKIP_PREDICATES. They're high-
    cardinality glue that floods search results with generic statements
    like "research is a inspect" without adding retrievable signal.
    """
    if not statement:
        return
    if pred in _TRIPLE_SKIP_PREDICATES:
        return
    col = _get_triple_collection(create=True)
    if col is None:
        return
    try:
        col.upsert(
            ids=[triple_id],
            documents=[statement],
            metadatas=[
                {
                    "triple_id": triple_id,
                    "subject": sub_id,
                    "predicate": pred,
                    "object": obj_id,
                    "confidence": float(confidence) if confidence is not None else 1.0,
                }
            ],
        )
    except Exception:
        pass


def normalize_entity_name(name: str) -> str:
    """Aggressive entity name normalization for dedup.

    Collapses: hyphens, underscores, dots, spaces, colons, slashes,
    backslashes, CamelCase boundaries, leading articles.

    Does NOT collapse: plurals, abbreviations (handled by semantic
    similarity on entity descriptions instead).

    Examples:
        "The Flowsev Repository" -> "flowsev_repository"
        "flowsev_repository"     -> "flowsev_repository"
        "FlowsevRepository"      -> "flowsev_repository"
        "D:\\Flowsev\\repo"      -> "d_flowsev_repo"
        "paperclip-server"       -> "paperclip_server"
        "paperclip_server"       -> "paperclip_server"
        "the GA agent"           -> "ga_agent"
    """
    if not isinstance(name, str) or not name.strip():
        return "unknown"
    s = name.strip()
    # Split CamelCase: "FlowsevRepo" -> "Flowsev Repo"
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    # Also split "HTTPServer" -> "HTTP Server"
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    # Lowercase
    s = s.lower()
    # Replace ALL non-alphanumeric with underscore (matches ChromaDB memory ID convention)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    # Collapse repeated underscores
    s = re.sub(r"_+", "_", s)
    # Strip leading/trailing underscores
    s = s.strip("_")
    # Strip leading articles
    for article in ("the_", "a_", "an_"):
        if s.startswith(article):
            s = s[len(article) :]
            break
    return s or "unknown"


def _normalize_predicate(predicate: str) -> str:
    """Normalize predicate strings at the storage boundary.

    Collapses hyphens, spaces, and repeated underscores. Matches how
    normalize_entity_name treats entity names, so `is-a` and `is_a` become
    the same predicate in the DB. Without this, seeded edges (`is-a`) and
    caller writes (`is_a`) were stored as distinct predicates.
    """
    if not isinstance(predicate, str):
        return ""
    s = predicate.strip().lower()
    s = re.sub(r"[-\s]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


class KnowledgeGraph:
    def __init__(self, db_path: str = None):
        # BF1: when the caller doesn't pin db_path, derive it from the live
        # palace_path so the KG always lives next to its Chroma data instead
        # of one directory up. Migrate any legacy ~/.mempalace/knowledge_graph.sqlite3
        # in place on first construction so existing installs keep their data.
        if db_path is None:
            db_path = _resolve_default_kg_path()
            _maybe_migrate_legacy_kg(db_path)
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = None
        self._init_db()
        _active_instances.append(self)

    def _init_db(self):
        """Initialize DB via yoyo-migrations + set PRAGMAs.

        Schema lives in per-file migrations under mempalace/migrations/.
        Each migration runs exactly once and is tracked by yoyo's version table.
        For legacy databases predating yoyo, set MEMPALACE_BOOTSTRAP_LEGACY=1 to
        mark all current migrations as applied without re-running them (since
        CREATE TABLE IF NOT EXISTS / ALTER on existing columns would fail).
        """
        from .migrations import apply_migrations

        # PRAGMAs first (yoyo opens its own connection briefly)
        conn = self._conn()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.commit()

        # For legacy databases that already have the schema but no yoyo marker:
        # detect and bootstrap (mark all migrations applied, run nothing).
        if self._is_legacy_unmarked_db(conn):
            self._bootstrap_yoyo_from_legacy_db()
        else:
            apply_migrations(self.db_path)

        # Seed canonical ontology on first run (no "thing" class yet)
        # Only for production palaces — test KGs are empty by design
        if not os.environ.get("MEMPALACE_SKIP_SEED"):
            self.seed_ontology()

    def _is_legacy_unmarked_db(self, conn) -> bool:
        """True if the DB has our tables but no yoyo version marker.

        Such DBs must be bootstrapped (mark migrations applied without running)
        so yoyo doesn't try to re-CREATE tables that already exist.
        """
        # If _yoyo_migration table exists, yoyo has managed this DB before — no
        # bootstrap needed.
        try:
            has_yoyo = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='_yoyo_migration'"
            ).fetchone()
            if has_yoyo:
                return False
        except sqlite3.OperationalError:
            return False
        # Otherwise: legacy only if we already have our tables
        try:
            has_entities = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='entities'"
            ).fetchone()
            return bool(has_entities)
        except sqlite3.OperationalError:
            return False

    def _bootstrap_yoyo_from_legacy_db(self) -> None:
        """Mark migrations as applied only for schema state already present.

        On a legacy DB (pre-yoyo) we inspect the actual columns/tables and
        mark migrations applied only when their effect is already in place.
        Remaining migrations then run normally to fill the gaps.
        """
        from yoyo import get_backend, read_migrations

        from .migrations import MIGRATIONS_DIR

        conn = self._conn()

        def _has_table(name: str) -> bool:
            return bool(
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (name,),
                ).fetchone()
            )

        def _has_column(table: str, col: str) -> bool:
            try:
                cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
                return col in cols
            except sqlite3.OperationalError:
                return False

        # Migration ID → predicate: does the DB already reflect this migration?
        # Each predicate returns True when the migration's effect is already in
        # place (so we should mark it applied and skip running it).
        already_applied_checks = {
            "001_initial_schema": lambda: _has_table("entities") and _has_table("triples"),
            "002_entity_metadata_columns": lambda: _has_column("entities", "kind"),
            "003_edge_traversal_feedback": lambda: _has_table("edge_traversal_feedback"),
            "004_edge_context_id": lambda: _has_column("edge_traversal_feedback", "context_id"),
            "005_keyword_feedback": lambda: _has_table("keyword_feedback"),
            "006_scoring_weight_feedback": lambda: _has_table("scoring_weight_feedback"),
            "007_context_and_keywords": lambda: _has_table("entity_keywords"),
            "008_rename_drawer_to_memory": lambda: _has_column("keyword_feedback", "memory_id"),
            "009_composite_indexes_and_provenance": lambda: _has_column("entities", "session_id"),
            "010_normalize_predicate_hyphens": lambda: not bool(
                conn.execute("SELECT 1 FROM triples WHERE predicate LIKE '%-%' LIMIT 1").fetchone()
            ),
            "011_conflict_resolutions": lambda: _has_table("conflict_resolutions"),
            "012_drop_source_closet": lambda: not _has_column("triples", "source_closet"),
            "013_triple_statement": lambda: _has_column("triples", "statement"),
            "014_context_as_entity": lambda: bool(
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='index' "
                    "AND name='idx_triples_created_under_subject' LIMIT 1"
                ).fetchone()
            ),
            "015_retire_old_feedback": lambda: _has_column("triples", "properties")
            and not _has_table("keyword_feedback"),
            "016_keyword_idf": lambda: _has_table("keyword_idf"),
        }

        backend = get_backend(f"sqlite:///{self.db_path}")
        all_migrations = read_migrations(str(MIGRATIONS_DIR))

        to_mark = []
        to_apply = []
        for m in all_migrations:
            check = already_applied_checks.get(m.id)
            if check is None:
                # Unknown migration (e.g. __init__ Python marker) — apply normally
                to_apply.append(m)
                continue
            if check():
                to_mark.append(m)
            else:
                to_apply.append(m)

        with backend.lock():
            if to_mark:
                # mark_migrations needs a MigrationList, not a bare list
                try:
                    from yoyo.migrations import MigrationList

                    backend.mark_migrations(MigrationList(to_mark))
                except ImportError:
                    backend.mark_migrations(to_mark)
            if to_apply:
                backend.apply_migrations(backend.to_apply(all_migrations))

    def _conn(self):
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA busy_timeout=10000")
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def seed_ontology(self):
        """Seed canonical classes, predicates, and intent types. Idempotent.

        Called automatically on first run (empty entities table) or on demand.
        Uses add_entity + add_triple, so normalization and schema are consistent.
        """
        conn = self._conn()
        # Check if ontology already seeded (look for root class "thing")
        thing = conn.execute("SELECT id FROM entities WHERE id = 'thing'").fetchone()
        if thing:
            return  # Already seeded

        # ── Classes (kind=class) ──
        classes = [
            ("thing", "Root class of the ontology. All other classes inherit from thing.", 5),
            (
                "system",
                "A running infrastructure component: servers, databases, containers, services",
                4,
            ),
            ("person", "A human individual", 4),
            ("agent", "An AI agent in the paperclip system (PFE, TL, Director, GA, etc.)", 4),
            ("project", "A repository, codebase, or software product", 4),
            ("file", "A specific file or path in a project", 3),
            ("rule", "A standing order, directive, or constraint authored by a human", 4),
            ("tool", "A software tool, CLI, or library", 3),
            ("process", "A workflow, procedure, or recurring operation", 3),
            ("concept", "An abstract idea, pattern, formula, or design principle", 3),
            (
                "environment",
                "A runtime environment or container that hosts processes and services",
                3,
            ),
            (
                "intent_type",
                "Class for intent types — the kind of action an agent declares before acting",
                5,
            ),
            (
                "context",
                "Class for retrieval Contexts — first-class KG nodes (kind='context') "
                "created by declare_intent / declare_operation / kg_search. "
                "Accretes via MaxSim (ColBERT-style multi-vector lookup, Khattab & "
                "Zaharia 2020) and links memories / entities / triples via "
                "created_under. See Anthropic Contextual Retrieval (2024) for the "
                "indexing-side rationale.",
                5,
            ),
        ]
        for name, desc, imp in classes:
            self.add_entity(name, kind="class", description=desc, importance=imp)
            if name != "thing":
                self.add_triple(name, "is_a", "thing")

        # ── Predicates (kind=predicate) with constraints ──
        predicates = [
            (
                "is_a",
                "Taxonomic classification: entity is_a class = instantiation, class is_a class = subtyping",
                5,
                {
                    "subject_kinds": ["entity", "class"],
                    "object_kinds": ["class", "entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "has_value",
                "Subject has a specific attribute value as object",
                4,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity", "literal"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "has_property",
                "Subject has a named property described by object",
                4,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity", "literal"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "defaults_to",
                "Subject has a default value of object",
                3,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity", "literal"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-one",
                },
            ),
            (
                "lives_at",
                "Subject is located at object (path, URL, address)",
                4,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity", "literal"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-one",
                },
            ),
            (
                "runs_in",
                "Subject operates as a process inside object runtime",
                4,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["system", "process"],
                    "object_classes": ["system", "environment"],
                    "cardinality": "many-to-one",
                },
            ),
            (
                "stored_in",
                "Subject data is persisted in object storage",
                3,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["system", "tool", "project"],
                    "object_classes": ["system", "tool", "environment"],
                    "cardinality": "many-to-one",
                },
            ),
            (
                "depends_on",
                "Subject requires object to function",
                4,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["system", "tool", "project", "process"],
                    "object_classes": ["system", "tool", "project", "process"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "requires",
                "Subject needs object as a runtime prerequisite",
                4,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["system", "tool", "project", "process"],
                    "object_classes": ["system", "tool", "project", "process"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "blocks",
                "Subject prevents object from proceeding",
                3,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "enables",
                "Subject unlocks object capability",
                3,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "must",
                "Subject is required to do/be object (positive rule)",
                4,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity", "literal"],
                    "subject_classes": [
                        "agent",
                        "system",
                        "tool",
                        "project",
                        "process",
                        "person",
                        "intent-type",
                    ],
                    "object_classes": ["rule"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "must_not",
                "Subject is forbidden from doing/being object (negative rule)",
                4,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity", "literal"],
                    "subject_classes": [
                        "agent",
                        "system",
                        "tool",
                        "project",
                        "process",
                        "person",
                        "intent-type",
                    ],
                    "object_classes": ["rule"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "forbids",
                "Subject prohibits object action",
                3,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity", "literal"],
                    "subject_classes": ["rule"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "has_gotcha",
                "Subject has a known pitfall described by object",
                4,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity", "literal"],
                    "subject_classes": ["system", "tool", "project", "process", "concept"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "warns_about",
                "Subject raises a caution about object",
                3,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity", "literal"],
                    "subject_classes": ["system", "tool", "project", "process", "concept"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "replaced_by",
                "Subject was superseded by object",
                3,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-one",
                },
            ),
            (
                "invalidated_by",
                "Subject was made obsolete by object event",
                3,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-one",
                },
            ),
            (
                "described_by",
                "Entity's canonical description lives in this memory",
                4,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "evidenced_by",
                "A rule or decision is backed by this memory's content",
                3,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "mentioned_in",
                "Entity is referenced in this memory but not its main topic",
                3,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "session_note_for",
                "A diary or session-log entry relevant to this entity",
                3,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "derived_from",
                "Entity was extracted or created from this memory's content",
                3,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "tested_by",
                "Subject is tested by the object test suite or entity",
                3,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["system", "tool", "project", "process"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "executed_by",
                "Intent execution was performed by this agent",
                4,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["agent"],
                    "cardinality": "many-to-one",
                },
            ),
            (
                "targeted",
                "Intent execution was performed on this entity (slot target)",
                4,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "resulted_in",
                "Intent execution produced this outcome memory",
                4,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "surfaced",
                "Retrieval event: a context surfaced this entity (memory / KG node) "
                "to the agent during search. Written by tool_kg_search / "
                "tool_declare_intent / tool_declare_operation after ranking. "
                "Props: {ts, rank, channel, sim_score}. Consumed by the "
                "finalize coverage validator and by Channel D retrieval "
                "(context-feedback).",
                4,
                {
                    "subject_kinds": ["context"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "rated_useful",
                "Positive feedback edge: the agent rated this surfaced entity "
                "as useful during finalize_intent. Props: {ts, relevance, "
                "reason, agent}. Consumed by Channel D and by Rocchio-style "
                "context enrichment (Rocchio 1971 / Manning/Raghavan/Schütze "
                "IR book Ch.9).",
                4,
                {
                    "subject_kinds": ["context"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "rated_irrelevant",
                "Negative feedback edge: the agent rated this surfaced entity "
                "as not relevant during finalize_intent. Props same shape as "
                "rated_useful. Channel D uses this as a demotion signal for "
                "similar future contexts.",
                3,
                {
                    "subject_kinds": ["context"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "created_under",
                "Provenance edge: a memory / entity / triple was written while this "
                "Context was active. Consumed by the retrieval Channel D "
                "(context-feedback, P2) and by the finalize coverage check.",
                4,
                {
                    "subject_kinds": ["entity", "class", "predicate", "literal", "record"],
                    "object_kinds": ["context"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-one",
                },
            ),
            (
                "similar_to",
                "Context-to-context similarity edge. Written at Context creation "
                "time when MaxSim against an existing context falls in the window "
                "[T_similar, T_reuse); prop {sim: float}. Used for 1–2-hop "
                "expansion in Channel D (P2).",
                3,
                {
                    "subject_kinds": ["context"],
                    "object_kinds": ["context"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
        ]
        for name, desc, imp, constraints in predicates:
            self.add_entity(
                name,
                kind="predicate",
                description=desc,
                importance=imp,
                properties={"constraints": constraints},
            )

        # ── Intent types (kind=class, is-a intent_type) ──
        intent_types = [
            # (name, description, importance, parent, slots, tool_permissions_or_None)
            (
                "inspect",
                "Intent type for read-only observation",
                4,
                "intent_type",
                {
                    "subject": {"classes": ["thing"], "required": True, "multiple": True},
                    "paths": {"raw": True, "required": True, "multiple": True},
                },
                [
                    {"tool": "Read", "scope": "{paths}"},
                    {"tool": "Grep", "scope": "{paths}"},
                    {"tool": "Glob", "scope": "{paths}"},
                ],
            ),
            (
                "modify",
                "Intent type for changing files",
                4,
                "intent_type",
                {
                    "files": {"classes": ["file"], "required": True, "multiple": True},
                    "paths": {"raw": True, "required": True, "multiple": True},
                },
                [
                    {"tool": "Edit", "scope": "{files}"},
                    {"tool": "Write", "scope": "{files}"},
                    {"tool": "Read", "scope": "{paths}"},
                    {"tool": "Grep", "scope": "{paths}"},
                    {"tool": "Glob", "scope": "{paths}"},
                ],
            ),
            (
                "execute",
                "Intent type for running commands and scripts",
                4,
                "intent_type",
                {
                    "target": {"classes": ["thing"], "required": True, "multiple": True},
                    "commands": {"raw": True, "required": True, "multiple": True},
                    "paths": {"raw": True, "required": True, "multiple": True},
                },
                [
                    {"tool": "Read", "scope": "{paths}"},
                    {"tool": "Grep", "scope": "{paths}"},
                    {"tool": "Glob", "scope": "{paths}"},
                    {"tool": "Bash", "scope": "{commands}"},
                ],
            ),
            (
                "communicate",
                "Intent type for external communication — sending messages, creating issues, pushing to services, fetching web content",
                4,
                "intent_type",
                {
                    "target": {"classes": ["thing"], "required": True, "multiple": True},
                    "audience": {
                        "classes": ["person", "agent"],
                        "required": False,
                        "multiple": True,
                    },
                    "paths": {"raw": True, "required": True, "multiple": True},
                },
                [
                    {"tool": "Read", "scope": "{paths}"},
                    {"tool": "Grep", "scope": "{paths}"},
                    {"tool": "Glob", "scope": "{paths}"},
                    {"tool": "Bash", "scope": "{target}"},
                    {"tool": "WebFetch", "scope": "*"},
                    {"tool": "WebSearch", "scope": "*"},
                ],
            ),
            (
                "research",
                "Intent type for researching external documentation, APIs, and web resources — read-only web access plus local code reading",
                4,
                "inspect",
                {
                    "subject": {"classes": ["thing"], "required": True, "multiple": True},
                    "paths": {"raw": True, "required": False, "multiple": True},
                },
                [
                    {"tool": "Read", "scope": "*"},
                    {"tool": "Grep", "scope": "*"},
                    {"tool": "Glob", "scope": "*"},
                    {"tool": "WebFetch", "scope": "*"},
                    {"tool": "WebSearch", "scope": "*"},
                ],
            ),
            # wrap_up_session: mandatory proof-of-done intent for the
            # never-stop rule. The Stop hook requires the LAST finalized
            # intent to be wrap_up_session(success) before it lets the
            # session stop. Must be seeded on every fresh palace or the
            # never-stop rule would wedge every install — no way to stop.
            (
                "wrap_up_session",
                "Proof-of-done intent: agent runs >=2 kg_search passes against pending-work patterns and summarises findings so the Stop hook admits a clean stop. Must be the LAST finalized intent in the session.",
                4,
                "inspect",
                {
                    "subject": {"classes": ["thing"], "required": True, "multiple": False},
                    "paths": {"raw": True, "required": True, "multiple": True},
                },
                None,  # inherits inspect's tool_permissions
            ),
            # Only generic top-level types seeded here.
            # Domain-specific children (edit_file, deploy, etc.) are declared
            # by agents via kg_declare_entity — not hardcoded in the seeder.
        ]
        for name, desc, imp, parent, slots, perms in intent_types:
            props = {"rules_profile": {"slots": slots}}
            if perms is not None:
                props["rules_profile"]["tool_permissions"] = perms
            self.add_entity(name, kind="class", description=desc, importance=imp, properties=props)
            self.add_triple(name, "is_a", parent)

    # Retired edge-feedback API (record_edge_feedback, get_edge_usefulness,
    # get_recent_rejection_reasons, get_context_ids_for_edge) deleted in
    # the cold-start cleanup — there's no legacy data to shim for.
    # Signal now flows through context --rated_useful/rated_irrelevant-->
    # memory edges written at finalize_intent.

    def get_past_conflict_resolution(
        self,
        existing_id: str,
        new_id: str,
        conflict_type: str,
    ):
        """Return the most recent past resolution for a (existing_id, new_id,
        conflict_type) triple, or None if no row exists.

        B1b: surfaces past decisions as a hint on newly-detected conflicts so
        agents don't re-derive reasoning they already captured. Matches by
        normalized entity ids on both sides plus the conflict_type (so a
        past `edge_contradiction` decision doesn't apply to a new
        `memory_duplicate` between the same ids). Ordered by created_at DESC
        so we return the freshest decision.
        """
        if not (existing_id and new_id and conflict_type):
            return None
        conn = self._conn()
        try:
            ex = self._entity_id(existing_id)
            nw = self._entity_id(new_id)
        except Exception:
            ex, nw = existing_id, new_id
        try:
            row = conn.execute(
                """SELECT action, reason, agent, intent_type, created_at
                   FROM conflict_resolutions
                   WHERE existing_id = ? AND new_id = ? AND conflict_type = ?
                   ORDER BY created_at DESC
                   LIMIT 1""",
                (ex, nw, conflict_type),
            ).fetchone()
        except Exception:
            return None
        if not row:
            return None
        return {
            "action": row[0],
            "reason": row[1],
            "agent": row[2] or "",
            "intent_type": row[3] or "",
            "when": row[4] or "",
        }

    def record_conflict_resolution(
        self,
        conflict_id: str,
        conflict_type: str,
        action: str,
        reason: str,
        existing_id: str = "",
        new_id: str = "",
        agent: str = "",
        intent_type: str = "",
        context_id: str = "",
    ):
        """Persist the agent's resolution of a conflict.

        Captures invalidate/merge/keep/skip decisions plus the mandatory
        reason, so future audits and feedback loops can learn from past
        choices instead of losing the reasoning.
        """
        conn = self._conn()
        now = datetime.now().isoformat()
        with conn:
            conn.execute(
                """INSERT INTO conflict_resolutions
                   (conflict_id, conflict_type, action, reason,
                    existing_id, new_id, agent, intent_type,
                    context_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    conflict_id,
                    conflict_type,
                    action,
                    reason,
                    existing_id,
                    new_id,
                    agent,
                    intent_type,
                    context_id,
                    now,
                ),
            )

    # ── Caller-provided keywords (stored, not auto-extracted) ──
    def add_entity_keywords(self, entity_id, keywords, source="caller"):
        """Persist caller-provided keywords for an entity.

        Replaces any existing rows with the same (entity_id, keyword) — idempotent.
        Used by kg_declare_entity (and friends) to store the Context.keywords list
        so the keyword channel can look entities up by literal term match without
        ever having to auto-extract from descriptions.
        """
        if not entity_id or not keywords:
            return 0
        cleaned = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
        if not cleaned:
            return 0
        conn = self._conn()
        rows = [(entity_id, k, source) for k in cleaned]
        conn.executemany(
            "INSERT OR REPLACE INTO entity_keywords (entity_id, keyword, source) VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()
        return len(rows)

    def get_entity_keywords(self, entity_id):
        """Return caller-provided keywords (lowercased str list) for an entity."""
        if not entity_id:
            return []
        conn = self._conn()
        rows = conn.execute(
            "SELECT keyword FROM entity_keywords WHERE entity_id=? ORDER BY added_at",
            (entity_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def entity_ids_for_keyword(self, keyword, limit=50):
        """Return entity_ids whose caller-provided keywords contain `keyword`.

        Case-insensitive exact match. Used by the keyword channel to
        surface entities by literal term hit — fast, indexed, no $contains scan.
        """
        if not keyword or not keyword.strip():
            return []
        conn = self._conn()
        rows = conn.execute(
            "SELECT DISTINCT entity_id FROM entity_keywords WHERE keyword=? LIMIT ?",
            (keyword.strip().lower(), limit),
        ).fetchall()
        return [r[0] for r in rows]

    def set_entity_creation_context(self, entity_id, context_id):
        """Record the Context.id under which an entity was created.

        The actual view vectors live in the mempalace_feedback_contexts Chroma
        collection (set by store_feedback_context). This column points at it
        so MaxSim can later weight feedback transfer by context similarity.
        """
        if not entity_id or not context_id:
            return False
        conn = self._conn()
        conn.execute(
            "UPDATE entities SET creation_context_id=? WHERE id=?",
            (context_id, entity_id),
        )
        conn.commit()
        return True

    def get_entity_creation_context(self, entity_id):
        if not entity_id:
            return ""
        conn = self._conn()
        row = conn.execute(
            "SELECT creation_context_id FROM entities WHERE id=?",
            (entity_id,),
        ).fetchone()
        return (row[0] if row else "") or ""

    # Retired keyword-suppression API (record_keyword_suppression,
    # get_keyword_suppression, reset_keyword_suppression) deleted in
    # the cold-start cleanup. BM25-IDF on keyword_idf replaces the
    # channel-level dominance signal.

    # P3: weight self-tune is RE-ENABLED. P2 cutover retired W_REL so the
    # scoring_weight_feedback table was truncated in migration 015 — the
    # learner now correlates against the four post-prune components
    # (sim, imp, decay, agent). Global weights (not per-context); see
    # docs/context_as_entity_redesign_plan.md — personal-scale palaces
    # are too sparse for LinUCB-style per-context bandits (Li et al.
    # 2010 arXiv:1003.0146; they need hundreds of observations per
    # context to converge).
    _A6_WEIGHT_SELFTUNE_ENABLED = True

    def record_scoring_feedback(self, components: dict, was_useful: bool, *, scope: str = "hybrid"):
        """Record scoring component values alongside relevance outcome.

        Two scopes:
          - scope='hybrid' (default): hybrid_score's per-memory weights
            (sim, rel, imp, decay, agent). Each row stored with component
            in that namespace.
          - scope='channel': per-channel RRF weights (cosine, graph,
            keyword, context). Components land with a ``ch_`` prefix
            so the row space stays disjoint from hybrid and
            ``compute_learned_weights(base, scope='channel')`` can
            filter by prefix.

        DISABLED by ``_A6_WEIGHT_SELFTUNE_ENABLED`` — currently a no-op
        when False. Keeping the body so flipping the flag re-enables
        data collection without touching the callers.
        """
        if not self._A6_WEIGHT_SELFTUNE_ENABLED:
            return
        conn = self._conn()
        now = datetime.now().isoformat()
        prefix = "ch_" if scope == "channel" else ""
        with conn:
            for comp, value in components.items():
                stored_name = f"{prefix}{comp}" if not comp.startswith(prefix) else comp
                conn.execute(
                    """INSERT INTO scoring_weight_feedback
                       (component, component_value, was_useful, created_at)
                       VALUES (?, ?, ?, ?)""",
                    (stored_name, float(value), was_useful, now),
                )

    def compute_learned_weights(
        self, base_weights: dict, min_samples: int = 10, *, scope: str = "hybrid"
    ):
        """Compute adjusted weights from feedback correlation.

        Works for either scope:
          - scope='hybrid': hybrid_score's per-memory weights (sim / rel
            / imp / decay / agent). Component names match base_weights
            keys exactly.
          - scope='channel': per-channel RRF weights (cosine / graph /
            keyword / context). Rows were stored with a ``ch_`` prefix
            by record_scoring_feedback; this method queries accordingly.

        Returns adjusted weights (same keys as base_weights), renormalised
        to sum to 1.0. Returns base_weights unchanged if insufficient
        feedback data or the self-tune flag is False.
        """
        if not self._A6_WEIGHT_SELFTUNE_ENABLED:
            return dict(base_weights)
        conn = self._conn()
        prefix = "ch_" if scope == "channel" else ""

        # Count rows in the relevant scope only.
        total = conn.execute(
            "SELECT COUNT(*) FROM scoring_weight_feedback WHERE component LIKE ?",
            (f"{prefix}%" if prefix else "%",),
        ).fetchone()[0]
        if total < min_samples:
            return dict(base_weights)

        adjustments = {}
        for comp in base_weights:
            stored_name = f"{prefix}{comp}" if prefix and not comp.startswith(prefix) else comp
            rows = conn.execute(
                """SELECT was_useful, AVG(component_value), COUNT(*)
                   FROM scoring_weight_feedback
                   WHERE component=?
                   GROUP BY was_useful""",
                (stored_name,),
            ).fetchall()
            avg_useful = 0.5
            avg_irrelevant = 0.5
            for row in rows:
                if row[0]:
                    avg_useful = row[1]
                else:
                    avg_irrelevant = row[1]
            correlation = avg_useful - avg_irrelevant
            adjustments[comp] = 1.0 + 0.3 * max(-1.0, min(1.0, correlation))

        adjusted = {}
        for comp, base_w in base_weights.items():
            adjusted[comp] = base_w * adjustments.get(comp, 1.0)
        total_w = sum(adjusted.values())
        if total_w > 0:
            for comp in adjusted:
                adjusted[comp] /= total_w
        return adjusted

    def close(self):
        """Close the database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _entity_id(self, name: str) -> str:
        """Normalize an entity name to a canonical ID.

        Uses aggressive normalization (hyphens, underscores, CamelCase, articles
        all collapsed). Also checks the alias table for merged entities.
        """
        normalized = normalize_entity_name(name)
        # Check if this normalized name is an alias for a merged entity
        conn = self._conn()
        alias_row = conn.execute(
            "SELECT canonical_id FROM entity_aliases WHERE alias = ?", (normalized,)
        ).fetchone()
        if alias_row:
            return alias_row["canonical_id"]
        return normalized

    def _touch_entity(self, entity_id: str):
        """Update last_touched timestamp on an entity."""
        conn = self._conn()
        now = datetime.now().isoformat()
        with conn:
            conn.execute("UPDATE entities SET last_touched = ? WHERE id = ?", (now, entity_id))

    def soft_delete_entity(self, name: str):
        """Soft-delete an entity (set status='deleted'). Also invalidates all its edges."""
        eid = self._entity_id(name)
        conn = self._conn()
        ended = date.today().isoformat()
        with conn:
            conn.execute("UPDATE entities SET status='deleted' WHERE id=?", (eid,))
            conn.execute(
                "UPDATE triples SET valid_to=? WHERE (subject=? OR object=?) AND valid_to IS NULL",
                (ended, eid, eid),
            )
        return eid

    # ── Write operations ──────────────────────────────────────────────────

    def add_entity(
        self,
        name: str,
        properties: dict = None,
        description: str = "",
        importance: int = 3,
        kind: str = "entity",
        session_id: str = None,
        intent_id: str = None,
    ):
        """Add or update an entity node.

        Args:
            kind: ontological role — 'entity' (concrete thing), 'predicate' (relationship type),
                  'class' (category/type), 'literal' (raw value). Fixed enum.
            description: precise text describing this entity.
            importance: 1-5 scale for decay-aware ranking.
            session_id: P6.7a provenance — auto-injected by callers, stored for session-scoped queries.
            intent_id: P6.7a provenance — auto-injected by callers, stored for intent-scoped queries.
        """
        eid = self._entity_id(name)
        props = json.dumps(properties or {})
        now = datetime.now().isoformat()
        conn = self._conn()
        with conn:
            # provenance columns (session_id, intent_id) are added
            # by migration 009. Use a try/except fallback for pre-migration
            # DBs where the columns don't exist yet.
            try:
                conn.execute(
                    """INSERT INTO entities (id, name, type, kind, properties, description,
                                            importance, last_touched, status, session_id, intent_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                       ON CONFLICT(id) DO UPDATE SET
                           name = excluded.name,
                           type = excluded.type,
                           kind = excluded.kind,
                           properties = excluded.properties,
                           description = CASE WHEN excluded.description != '' THEN excluded.description ELSE entities.description END,
                           importance = CASE WHEN excluded.importance != 3 THEN excluded.importance ELSE entities.importance END,
                           last_touched = excluded.last_touched,
                           session_id = COALESCE(excluded.session_id, entities.session_id),
                           intent_id = COALESCE(excluded.intent_id, entities.intent_id)
                    """,
                    (
                        eid,
                        name,
                        kind,
                        kind,
                        props,
                        description,
                        importance,
                        now,
                        session_id or "",
                        intent_id or "",
                    ),
                )
            except Exception:
                # Pre-migration fallback (columns don't exist yet)
                conn.execute(
                    """INSERT INTO entities (id, name, type, kind, properties, description, importance, last_touched, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active')
                       ON CONFLICT(id) DO UPDATE SET
                           name = excluded.name,
                           type = excluded.type,
                           kind = excluded.kind,
                           properties = excluded.properties,
                           description = CASE WHEN excluded.description != '' THEN excluded.description ELSE entities.description END,
                           importance = CASE WHEN excluded.importance != 3 THEN excluded.importance ELSE entities.importance END,
                           last_touched = excluded.last_touched
                    """,
                    (eid, name, kind, kind, props, description, importance, now),
                )
        return eid

    def merge_entities(self, source_name: str, target_name: str, update_description: str = None):
        """Merge source entity into target. All edges rewritten. Source becomes alias.

        Returns dict with counts of edges_moved, aliases_created.
        """
        source_id = normalize_entity_name(source_name)
        target_id = self._entity_id(target_name)  # resolves aliases
        if source_id == target_id:
            return {"error": "source and target resolve to the same entity"}

        conn = self._conn()
        with conn:
            # Rewrite triples: subject
            r1 = conn.execute(
                "UPDATE triples SET subject = ? WHERE subject = ?", (target_id, source_id)
            )
            # Rewrite triples: object
            r2 = conn.execute(
                "UPDATE triples SET object = ? WHERE object = ?", (target_id, source_id)
            )
            edges_moved = r1.rowcount + r2.rowcount

            # Register alias
            now = datetime.now().isoformat()
            conn.execute(
                "INSERT OR REPLACE INTO entity_aliases (alias, canonical_id, merged_at) VALUES (?, ?, ?)",
                (source_id, target_id, now),
            )

            # Soft-delete source
            conn.execute(
                "UPDATE entities SET status = 'merged', merged_into = ? WHERE id = ?",
                (target_id, source_id),
            )

            # Update target description if provided
            if update_description:
                conn.execute(
                    "UPDATE entities SET description = ?, last_touched = ? WHERE id = ?",
                    (update_description, now, target_id),
                )

            # Touch target
            conn.execute("UPDATE entities SET last_touched = ? WHERE id = ?", (now, target_id))

        return {
            "source": source_id,
            "target": target_id,
            "edges_moved": edges_moved,
            "aliases_created": 1,
        }

    def list_unverbalized_triples(self, limit: int = None) -> list:
        """Return SQL rows for semantic triples missing a ``statement``.

        Each row is (id, subject, predicate, object, confidence) — the
        raw material a human (or curation tool) needs to write a proper
        natural-language sentence for. Skip-list predicates are omitted
        because they're never embedded anyway.

        NO auto-generation happens anywhere. The previous
        ``backfill_triple_statements`` that fabricated statements from
        underscore-to-space substitution was retired 2026-04-19 — see
        the TripleStatementRequired policy in add_triple. Legacy rows
        with ``statement IS NULL`` simply remain NULL and absent from
        the mempalace_triples Chroma collection; they're still walkable
        via BFS and queryable by exact id, just not similarity-searched
        until someone writes a real statement via kg_update_triple or
        equivalent curation.
        """
        conn = self._conn()
        skip_clause = ",".join("?" for _ in _TRIPLE_SKIP_PREDICATES)
        rows = conn.execute(
            f"""SELECT id, subject, predicate, object, confidence
               FROM triples
               WHERE statement IS NULL
                 AND predicate NOT IN ({skip_clause})
               ORDER BY id
               LIMIT ?""",
            (*sorted(_TRIPLE_SKIP_PREDICATES), int(limit) if limit else 1_000_000),
        ).fetchall()
        return [
            {
                "triple_id": r["id"],
                "subject": r["subject"],
                "predicate": r["predicate"],
                "object": r["object"],
                "confidence": r["confidence"] if r["confidence"] is not None else 1.0,
            }
            for r in rows
        ]

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        valid_from: str = None,
        valid_to: str = None,
        confidence: float = 1.0,
        source_file: str = None,
        creation_context_id: str = "",
        statement: str = None,
        properties: dict = None,
    ):
        """
        Add a relationship triple: subject → predicate → object.

        Examples:
            add_triple("Max", "child_of", "Alice", valid_from="2015-04-01")
            add_triple("Max", "does", "swimming", valid_from="2025-01-01")
            add_triple("Alice", "worried_about", "Max injury", valid_from="2026-01", valid_to="2026-02")

        `statement` is the natural-language verbalization of the triple
        ("Max is a child of Alice"). Stored on the row and embedded into
        the mempalace_triples Chroma collection so the triple becomes a
        first-class search target.

        REQUIRED for every predicate OUTSIDE the skip list
        (``_TRIPLE_SKIP_PREDICATES``). For skip-list predicates (``is_a``,
        ``described_by``, ``executed_by``, ``targeted``, …) the statement
        is allowed to be None because those are schema glue that never
        gets embedded regardless — they're walkable via BFS, not searched
        by similarity.

        Rationale (2026-04-19 policy change): we used to fall back to a
        naive "replace underscores with spaces" verbalization when callers
        omitted ``statement``. That produced retrieval-poisoning text like
        ``"record ga agent a relates to record ga agent b"``. Callers now
        write a real sentence or the edge is rejected.
        """
        sub_id = self._entity_id(subject)
        obj_id = self._entity_id(obj)
        pred = _normalize_predicate(predicate)
        # Require a caller-provided statement for non-skip predicates.
        # Skip predicates stay optional — they're never embedded anyway.
        if pred not in _TRIPLE_SKIP_PREDICATES:
            if not statement or not statement.strip():
                raise TripleStatementRequired(
                    f"add_triple({subject!r}, {pred!r}, {obj!r}): predicate "
                    f"{pred!r} requires a caller-provided `statement` — a "
                    f"natural-language verbalization of the fact. "
                    f"Structural predicates (is_a, described_by, "
                    f"executed_by, targeted, has_value, "
                    f"session_note_for, derived_from, mentioned_in, "
                    f"found_useful, found_irrelevant, evidenced_by) may "
                    f"omit `statement`; every other predicate must supply "
                    f"one. Autogeneration was retired 2026-04-19 because "
                    f"naive fallbacks produced low-signal text that "
                    f"poisoned retrieval."
                )
            statement = statement.strip()

        # Auto-create entities if they don't exist
        conn = self._conn()
        with conn:
            conn.execute(
                "INSERT OR IGNORE INTO entities (id, name) VALUES (?, ?)", (sub_id, subject)
            )
            conn.execute("INSERT OR IGNORE INTO entities (id, name) VALUES (?, ?)", (obj_id, obj))

            # Check for existing identical triple
            existing = conn.execute(
                "SELECT id FROM triples WHERE subject=? AND predicate=? AND object=? AND valid_to IS NULL",
                (sub_id, pred, obj_id),
            ).fetchone()

            if existing:
                return existing["id"]  # Already exists and still valid

            triple_id = f"t_{sub_id}_{pred}_{obj_id}_{hashlib.sha256(f'{valid_from}{datetime.now().isoformat()}'.encode()).hexdigest()[:12]}"

            props_json = json.dumps(properties or {})
            conn.execute(
                """INSERT INTO triples (id, subject, predicate, object, valid_from, valid_to,
                                        confidence, source_file, creation_context_id, statement,
                                        properties)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    triple_id,
                    sub_id,
                    pred,
                    obj_id,
                    valid_from,
                    valid_to,
                    confidence,
                    source_file,
                    creation_context_id or "",
                    statement,
                    props_json,
                ),
            )
        # Touch both entities (update last_touched for decay scoring)
        self._touch_entity(sub_id)
        self._touch_entity(obj_id)
        # Embed the verbalization so kg_search and multi_channel_search can
        # surface this triple as a first-class result. Best-effort: any
        # Chroma write failure is non-fatal (the SQL row is the source of
        # truth and the backfill helper can re-embed later).
        _index_triple_statement(self, triple_id, sub_id, pred, obj_id, statement, confidence)
        return triple_id

    def invalidate(self, subject: str, predicate: str, obj: str, ended: str = None):
        """Mark a relationship as no longer valid (set valid_to date)."""
        sub_id = self._entity_id(subject)
        obj_id = self._entity_id(obj)
        pred = _normalize_predicate(predicate)
        ended = ended or date.today().isoformat()

        conn = self._conn()
        with conn:
            conn.execute(
                "UPDATE triples SET valid_to=? WHERE subject=? AND predicate=? AND object=? AND valid_to IS NULL",
                (ended, sub_id, pred, obj_id),
            )

    def get_entity(self, name: str):
        """Get entity details by name. Returns dict or None if not found."""
        eid = self._entity_id(name)
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM entities WHERE id = ? AND status = 'active'", (eid,)
        ).fetchone()
        if not row:
            return None
        # kind column may not exist in very old DBs — fall back to type
        kind = "entity"
        try:
            kind = row["kind"] or "entity"
        except (IndexError, KeyError):
            pass
        return {
            "id": row["id"],
            "name": row["name"],
            "type": row["type"],
            "kind": kind,
            "description": row["description"] or "",
            "importance": row["importance"] or 3,
            "last_touched": row["last_touched"] or "",
            "status": row["status"],
            "properties": json.loads(row["properties"]) if row["properties"] else {},
        }

    def list_entities(self, status: str = "active", kind: str = None):
        """List all entities with the given status, optionally filtered by kind.

        Args:
            status: 'active', 'merged', 'deprecated' (default 'active')
            kind: 'entity', 'predicate', 'class', 'literal' (default None = all)
        """
        conn = self._conn()
        if kind:
            rows = conn.execute(
                "SELECT * FROM entities WHERE status = ? AND kind = ? ORDER BY importance DESC, last_touched DESC",
                (status, kind),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM entities WHERE status = ? ORDER BY importance DESC, last_touched DESC",
                (status,),
            ).fetchall()
        results = []
        for row in rows:
            row_kind = "entity"
            try:
                row_kind = row["kind"] or "entity"
            except (IndexError, KeyError):
                pass
            results.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "type": row["type"],
                    "kind": row_kind,
                    "description": row["description"] or "",
                    "importance": row["importance"] or 3,
                    "last_touched": row["last_touched"] or "",
                }
            )
        return results

    def update_entity_description(self, name: str, description: str, importance: int = None):
        """Update an entity's description (and optionally importance). Returns the entity."""
        eid = self._entity_id(name)
        now = datetime.now().isoformat()
        conn = self._conn()
        with conn:
            if importance is not None:
                conn.execute(
                    "UPDATE entities SET description = ?, importance = ?, last_touched = ? WHERE id = ?",
                    (description, importance, now, eid),
                )
            else:
                conn.execute(
                    "UPDATE entities SET description = ?, last_touched = ? WHERE id = ?",
                    (description, now, eid),
                )
        return self.get_entity(name)

    def update_entity_properties(self, name: str, properties: dict):
        """Merge new properties into an entity's existing properties."""
        eid = self._entity_id(name)
        conn = self._conn()
        row = conn.execute("SELECT properties FROM entities WHERE id = ?", (eid,)).fetchone()
        if not row:
            return None
        existing = json.loads(row["properties"]) if row["properties"] else {}
        existing.update(properties)
        now = datetime.now().isoformat()
        with conn:
            conn.execute(
                "UPDATE entities SET properties = ?, last_touched = ? WHERE id = ?",
                (json.dumps(existing), now, eid),
            )
        return self.get_entity(name)

    def entity_edge_count(self, name: str) -> int:
        """Count active edges (triples) involving an entity."""
        eid = self._entity_id(name)
        conn = self._conn()
        row = conn.execute(
            "SELECT COUNT(*) as n FROM triples WHERE (subject = ? OR object = ?) AND valid_to IS NULL",
            (eid, eid),
        ).fetchone()
        return row["n"] if row else 0

    # ── Query operations ──────────────────────────────────────────────────

    def query_entity(self, name: str, as_of: str = None, direction: str = "outgoing"):
        """
        Get all relationships for an entity.

        direction: "outgoing" (entity → ?), "incoming" (? → entity), "both"
        as_of: date string — only return facts valid at that time
        """
        eid = self._entity_id(name)
        conn = self._conn()

        results = []

        if direction in ("outgoing", "both"):
            query = "SELECT t.*, e.name as obj_name FROM triples t JOIN entities e ON t.object = e.id WHERE t.subject = ?"
            params = [eid]
            if as_of:
                query += " AND (t.valid_from IS NULL OR t.valid_from <= ?) AND (t.valid_to IS NULL OR t.valid_to >= ?)"
                params.extend([as_of, as_of])
            for row in conn.execute(query, params).fetchall():
                results.append(
                    {
                        "direction": "outgoing",
                        "subject": name,
                        "predicate": row["predicate"],
                        "object": row["obj_name"],
                        "valid_from": row["valid_from"],
                        "valid_to": row["valid_to"],
                        "confidence": row["confidence"],
                        "current": row["valid_to"] is None,
                    }
                )

        if direction in ("incoming", "both"):
            query = "SELECT t.*, e.name as sub_name FROM triples t JOIN entities e ON t.subject = e.id WHERE t.object = ?"
            params = [eid]
            if as_of:
                query += " AND (t.valid_from IS NULL OR t.valid_from <= ?) AND (t.valid_to IS NULL OR t.valid_to >= ?)"
                params.extend([as_of, as_of])
            for row in conn.execute(query, params).fetchall():
                results.append(
                    {
                        "direction": "incoming",
                        "subject": row["sub_name"],
                        "predicate": row["predicate"],
                        "object": name,
                        "valid_from": row["valid_from"],
                        "valid_to": row["valid_to"],
                        "confidence": row["confidence"],
                        "current": row["valid_to"] is None,
                    }
                )

        return results

    # ── BM25-IDF keyword signals (P3 follow-up) ──
    def record_keyword_observations(self, keywords, *, recompute_idf: bool = True):
        """Bump freq for each keyword observed on a new record memory.

        Called by _add_memory_internal on record writes so the BM25-IDF
        table stays incrementally up to date. Recomputes idf for every
        keyword whose freq changed (cheap — one log per bumped row).

        IDF formula (Robertson & Jones 1976; Robertson & Zaragoza 2009
        "Foundations of BM25 and Beyond"):

            idf(t) = log((N - freq(t) + 0.5) / (freq(t) + 0.5))

        where N is the total number of record-kind memories. Rare terms
        get large positive idf; dominant terms near N approach 0 or
        negative (the keyword channel clamps at min_idf=0.5 downstream).
        """
        import math

        if not keywords:
            return
        cleaned = list({k.strip() for k in keywords if isinstance(k, str) and k.strip()})
        if not cleaned:
            return
        conn = self._conn()
        now = datetime.now().isoformat()
        try:
            with conn:
                for kw in cleaned:
                    conn.execute(
                        """INSERT INTO keyword_idf (keyword, freq, idf, last_updated_ts)
                           VALUES (?, 1, 0.0, ?)
                           ON CONFLICT(keyword) DO UPDATE SET
                             freq = freq + 1,
                             last_updated_ts = excluded.last_updated_ts""",
                        (kw, now),
                    )
                if recompute_idf:
                    n_row = conn.execute(
                        "SELECT COUNT(*) FROM entities WHERE kind='record' AND status='active'"
                    ).fetchone()
                    total_n = int((n_row[0] if n_row else 0) or 0)
                    if total_n > 0:
                        for kw in cleaned:
                            f_row = conn.execute(
                                "SELECT freq FROM keyword_idf WHERE keyword=?", (kw,)
                            ).fetchone()
                            if not f_row:
                                continue
                            f = int(f_row[0] or 0)
                            # BM25 robust IDF (log stays positive by adding 1.0
                            # inside, so even dominant terms have a floor at 0).
                            idf = math.log(max(0.0, (total_n - f + 0.5) / (f + 0.5)) + 1.0)
                            conn.execute(
                                "UPDATE keyword_idf SET idf=? WHERE keyword=?",
                                (round(idf, 6), kw),
                            )
        except sqlite3.OperationalError:
            # keyword_idf table absent (pre-migration-016 DB) — no-op.
            pass

    def get_keyword_idf(self, keywords) -> dict:
        """Return {keyword: idf} for each requested keyword (0.0 for unseen)."""
        if not keywords:
            return {}
        cleaned = list({k.strip() for k in keywords if isinstance(k, str) and k.strip()})
        if not cleaned:
            return {}
        conn = self._conn()
        result = {kw: 0.0 for kw in cleaned}
        try:
            placeholders = ",".join("?" for _ in cleaned)
            rows = conn.execute(
                f"SELECT keyword, idf FROM keyword_idf WHERE keyword IN ({placeholders})",
                cleaned,
            ).fetchall()
            for kw, idf in rows:
                try:
                    result[kw] = float(idf or 0.0)
                except (TypeError, ValueError):
                    continue
        except sqlite3.OperationalError:
            return result
        return result

    def recompute_keyword_idf_all(self):
        """Full recompute across every keyword in keyword_idf.

        O(rows). Call once after a bulk backfill, or in a maintenance
        path. For the per-write hot path, use record_keyword_observations
        which only recomputes the affected keywords.
        """
        import math

        conn = self._conn()
        try:
            n_row = conn.execute(
                "SELECT COUNT(*) FROM entities WHERE kind='record' AND status='active'"
            ).fetchone()
            total_n = int((n_row[0] if n_row else 0) or 0)
            if total_n <= 0:
                return
            rows = conn.execute("SELECT keyword, freq FROM keyword_idf").fetchall()
            updates = []
            for keyword, freq in rows:
                f = int(freq or 0)
                idf = math.log(max(0.0, (total_n - f + 0.5) / (f + 0.5)) + 1.0)
                updates.append((round(idf, 6), keyword))
            if updates:
                with conn:
                    conn.executemany("UPDATE keyword_idf SET idf=? WHERE keyword=?", updates)
        except sqlite3.OperationalError:
            return

    def triples_created_under(self, context_id: str) -> list:
        """Return triple_ids whose creation_context_id points at this context.

        Triples aren't materialised as entity rows (no kind='triple'
        entity), so a standard ``kg_query`` on a context won't return
        them via ``created_under`` edges — there are none to triples.
        This is the triples-layer analogue of the memory/entity
        ``created_under`` edge walk: "which triples were written under
        this context."
        """
        if not context_id:
            return []
        conn = self._conn()
        rows = conn.execute(
            "SELECT id FROM triples WHERE creation_context_id=? "
            "AND (valid_to IS NULL OR valid_to='')",
            (context_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_entity_degree(self, entity_id: str) -> int:
        """Total in-degree + out-degree for an entity in the current triples.

        Used by Channel B's degree-dampening: mega-hub entities (like the
        agent's own id) would otherwise flood graph-channel results with
        their many neighbours. Each seed→memory contribution is weighted
        by ``1 / log(degree + 2)``, so a degree-50 hub contributes roughly
        a quarter of what a degree-2 specialist does.

        References:
          Hogan et al. "Knowledge Graphs." arXiv:2003.02320 (2021).
          West & Leskovec. "Human wayfinding in information networks."
            WWW 2012 — inverse-log degree term is the standard dampening
            shape for random-walk over KGs.
          Bollacker et al. "Freebase." SIGMOD 2008 — same dampening for
            popular entities.
        """
        if not entity_id:
            return 0
        eid = self._entity_id(entity_id)
        conn = self._conn()
        try:
            out_degree = conn.execute(
                "SELECT COUNT(*) FROM triples WHERE subject=? "
                "AND (valid_to IS NULL OR valid_to='')",
                (eid,),
            ).fetchone()[0]
            in_degree = conn.execute(
                "SELECT COUNT(*) FROM triples WHERE object=? AND (valid_to IS NULL OR valid_to='')",
                (eid,),
            ).fetchone()[0]
        except Exception:
            return 0
        return int(out_degree or 0) + int(in_degree or 0)

    def get_similar_contexts(self, context_id: str, hops: int = 2, decay: float = 0.5) -> list:
        """BFS ``similar_to`` neighbourhood of a context, with distance decay.

        Returns ``[(neighbour_context_id, accumulated_sim), …]`` sorted by
        accumulated_sim descending. 1-hop contributes ``sim``; 2-hop
        contributes ``sim * decay * parent_sim``; 3-hop would contribute
        ``sim * decay² * parent_sim * grandparent_sim``. Early termination
        when a path's accumulated sim falls below 1e-4.

        Edge similarity is read from the ``confidence`` column (P1
        convention — see ``context_lookup_or_create`` in mcp_server.py).

        Consumed by Channel D (retrieval, P2) to expand the context
        neighbourhood around the active context. Shipping the helper in
        P1 keeps the traversal unit-testable in isolation.
        """
        if not context_id or hops < 1:
            return []
        eid = self._entity_id(context_id)
        conn = self._conn()
        visited = {eid}
        # frontier: list of (current_context_id, accumulated_sim_so_far)
        frontier = [(eid, 1.0)]
        accumulated: dict = {}
        depth_decay = 1.0
        for depth in range(hops):
            if not frontier:
                break
            depth_decay *= decay if depth > 0 else 1.0
            next_frontier = []
            for cur_id, cur_sim in frontier:
                rows = conn.execute(
                    "SELECT object, confidence FROM triples "
                    "WHERE subject=? AND predicate='similar_to' "
                    "AND (valid_to IS NULL OR valid_to = '')",
                    (cur_id,),
                ).fetchall()
                for row in rows:
                    neighbour = row["object"]
                    if neighbour in visited:
                        continue
                    edge_sim = float(row["confidence"] or 0.0)
                    if edge_sim <= 0.0:
                        continue
                    contribution = cur_sim * edge_sim * depth_decay
                    if contribution < 1e-4:
                        continue
                    # Keep max contribution if the same neighbour is reached
                    # by multiple paths at different depths.
                    prev = accumulated.get(neighbour, 0.0)
                    if contribution > prev:
                        accumulated[neighbour] = contribution
                    visited.add(neighbour)
                    next_frontier.append((neighbour, contribution))
            frontier = next_frontier

        return sorted(accumulated.items(), key=lambda kv: kv[1], reverse=True)

    def query_relationship(self, predicate: str, as_of: str = None):
        """Get all triples with a given relationship type."""
        pred = _normalize_predicate(predicate)
        conn = self._conn()
        query = """
            SELECT t.*, s.name as sub_name, o.name as obj_name
            FROM triples t
            JOIN entities s ON t.subject = s.id
            JOIN entities o ON t.object = o.id
            WHERE t.predicate = ?
        """
        params = [pred]
        if as_of:
            query += " AND (t.valid_from IS NULL OR t.valid_from <= ?) AND (t.valid_to IS NULL OR t.valid_to >= ?)"
            params.extend([as_of, as_of])

        results = []
        for row in conn.execute(query, params).fetchall():
            results.append(
                {
                    "subject": row["sub_name"],
                    "predicate": pred,
                    "object": row["obj_name"],
                    "valid_from": row["valid_from"],
                    "valid_to": row["valid_to"],
                    "current": row["valid_to"] is None,
                }
            )
        return results

    def timeline(self, entity_name: str = None):
        """Get all facts in chronological order, optionally filtered by entity."""
        conn = self._conn()
        if entity_name:
            eid = self._entity_id(entity_name)
            rows = conn.execute(
                """
                SELECT t.*, s.name as sub_name, o.name as obj_name
                FROM triples t
                JOIN entities s ON t.subject = s.id
                JOIN entities o ON t.object = o.id
                WHERE (t.subject = ? OR t.object = ?)
                ORDER BY t.valid_from ASC NULLS LAST
                LIMIT 100
            """,
                (eid, eid),
            ).fetchall()
        else:
            rows = conn.execute("""
                SELECT t.*, s.name as sub_name, o.name as obj_name
                FROM triples t
                JOIN entities s ON t.subject = s.id
                JOIN entities o ON t.object = o.id
                ORDER BY t.valid_from ASC NULLS LAST
                LIMIT 100
            """).fetchall()

        return [
            {
                "subject": r["sub_name"],
                "predicate": r["predicate"],
                "object": r["obj_name"],
                "valid_from": r["valid_from"],
                "valid_to": r["valid_to"],
                "current": r["valid_to"] is None,
            }
            for r in rows
        ]

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self):
        conn = self._conn()
        entities = conn.execute("SELECT COUNT(*) as cnt FROM entities").fetchone()["cnt"]
        triples = conn.execute("SELECT COUNT(*) as cnt FROM triples").fetchone()["cnt"]
        current = conn.execute(
            "SELECT COUNT(*) as cnt FROM triples WHERE valid_to IS NULL"
        ).fetchone()["cnt"]
        expired = triples - current
        predicates = [
            r["predicate"]
            for r in conn.execute(
                "SELECT DISTINCT predicate FROM triples ORDER BY predicate"
            ).fetchall()
        ]
        return {
            "entities": entities,
            "triples": triples,
            "current_facts": current,
            "expired_facts": expired,
            "relationship_types": predicates,
        }

    # ── Seed from known facts ─────────────────────────────────────────────

    def seed_from_entity_facts(self, entity_facts: dict):
        """
        Seed the knowledge graph from fact_checker.py ENTITY_FACTS.
        This bootstraps the graph with known ground truth.
        """
        for key, facts in entity_facts.items():
            name = facts.get("full_name", key.capitalize())
            self.add_entity(
                name,
                kind="entity",
                description=f"{name} ({facts.get('type', 'person')})",
                properties={
                    "gender": facts.get("gender", ""),
                    "birthday": facts.get("birthday", ""),
                },
            )

            # Relationships. Each add_triple supplies a statement on
            # non-skip predicates (TripleStatementRequired policy). The
            # sentences are derived from the known-fact dict at this
            # seed layer; this is still caller-written (by the
            # fact_checker author via seed_from_entity_facts) rather
            # than autogenerated at embed time.
            parent = facts.get("parent")
            if parent:
                self.add_triple(
                    name,
                    "child_of",
                    parent.capitalize(),
                    valid_from=facts.get("birthday"),
                    statement=f"{name} is the child of {parent.capitalize()}.",
                )

            partner = facts.get("partner")
            if partner:
                self.add_triple(
                    name,
                    "married_to",
                    partner.capitalize(),
                    statement=f"{name} is married to {partner.capitalize()}.",
                )

            relationship = facts.get("relationship", "")
            if relationship == "daughter":
                parent_name = facts.get("parent", "").capitalize() or name
                self.add_triple(
                    name,
                    "is_child_of",
                    parent_name,
                    valid_from=facts.get("birthday"),
                    statement=f"{name} is the child of {parent_name}.",
                )
            elif relationship == "husband":
                partner_name = facts.get("partner", name).capitalize()
                self.add_triple(
                    name,
                    "is_partner_of",
                    partner_name,
                    statement=f"{name} is the partner of {partner_name}.",
                )
            elif relationship == "brother":
                sibling_name = facts.get("sibling", name).capitalize()
                self.add_triple(
                    name,
                    "is_sibling_of",
                    sibling_name,
                    statement=f"{name} is a sibling of {sibling_name}.",
                )
            elif relationship == "dog":
                owner_name = facts.get("owner", name).capitalize()
                self.add_triple(
                    name,
                    "is_pet_of",
                    owner_name,
                    statement=f"{name} is a pet of {owner_name}.",
                )
                self.add_entity(name, "animal")

            # Interests
            for interest in facts.get("interests", []):
                interest_cap = interest.capitalize()
                self.add_triple(
                    name,
                    "loves",
                    interest_cap,
                    valid_from="2025-01-01",
                    statement=f"{name} loves {interest_cap}.",
                )
