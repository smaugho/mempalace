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


DEFAULT_KG_PATH = os.path.expanduser("~/.mempalace/knowledge_graph.sqlite3")


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
        self.db_path = db_path or DEFAULT_KG_PATH
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
                "found_useful",
                "Agent found this memory/entity useful during intent execution — contextual relevance feedback",
                4,
                {
                    "subject_kinds": ["entity", "class"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "found_irrelevant",
                "Agent found this memory/entity not relevant during intent execution — negative contextual feedback",
                3,
                {
                    "subject_kinds": ["entity", "class"],
                    "object_kinds": ["entity"],
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

    def record_edge_feedback(
        self,
        subject,
        predicate,
        obj,
        intent_type,
        useful,
        context_keywords="",
        context_id="",
    ):
        """Record whether traversing an edge was useful in a given context.

        Args:
            context_id: ID referencing stored context vectors in ChromaDB
                feedback_contexts collection. Enables contextual feedback via
                MaxSim comparison at retrieval time.
        """
        conn = self._conn()
        now = datetime.now().isoformat()
        with conn:
            conn.execute(
                """INSERT INTO edge_traversal_feedback
                   (subject, predicate, object, intent_type, useful, context_keywords,
                    context_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self._entity_id(subject),
                    _normalize_predicate(predicate),
                    self._entity_id(obj),
                    intent_type,
                    useful,
                    context_keywords,
                    context_id,
                    now,
                ),
            )

    def get_edge_usefulness(self, subject, predicate, obj, intent_type=None, context_id=None):
        """Get aggregated usefulness score for an edge. Returns float in [-1, 1].

        Positive = more useful than not, negative = more irrelevant than useful.
        If context_id provided, filters to that specific context.
        Falls back to intent_type if no context_id match, then to global.
        """
        conn = self._conn()
        sub_id = self._entity_id(subject)
        pred = _normalize_predicate(predicate)
        obj_id = self._entity_id(obj)

        # Try context_id first (most specific)
        if context_id:
            rows = conn.execute(
                """SELECT useful, COUNT(*) as cnt FROM edge_traversal_feedback
                   WHERE subject=? AND predicate=? AND object=? AND context_id=?
                   GROUP BY useful""",
                (sub_id, pred, obj_id, context_id),
            ).fetchall()
            score = self._compute_usefulness(rows)
            if score is not None:
                return score

        # Fall back to intent_type
        if intent_type:
            rows = conn.execute(
                """SELECT useful, COUNT(*) as cnt FROM edge_traversal_feedback
                   WHERE subject=? AND predicate=? AND object=? AND intent_type=?
                   GROUP BY useful""",
                (sub_id, pred, obj_id, intent_type),
            ).fetchall()
            score = self._compute_usefulness(rows)
            if score is not None:
                return score

        # Fall back to global
        rows = conn.execute(
            """SELECT useful, COUNT(*) as cnt FROM edge_traversal_feedback
               WHERE subject=? AND predicate=? AND object=?
               GROUP BY useful""",
            (sub_id, pred, obj_id),
        ).fetchall()
        return self._compute_usefulness(rows) or 0.0

    @staticmethod
    def _compute_usefulness(rows):
        """Compute usefulness from feedback rows. Returns None if no data."""
        useful_count = 0
        irrelevant_count = 0
        for row in rows:
            if row[0]:
                useful_count = row[1]
            else:
                irrelevant_count = row[1]
        total = useful_count + irrelevant_count
        if total == 0:
            return None
        return (useful_count - irrelevant_count) / total

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

    def get_context_ids_for_edge(self, subject, predicate, obj):
        """Get all context_ids associated with feedback for an edge."""
        conn = self._conn()
        sub_id = self._entity_id(subject)
        pred = _normalize_predicate(predicate)
        obj_id = self._entity_id(obj)
        rows = conn.execute(
            """SELECT DISTINCT context_id FROM edge_traversal_feedback
               WHERE subject=? AND predicate=? AND object=? AND context_id != ''""",
            (sub_id, pred, obj_id),
        ).fetchall()
        return [r[0] for r in rows]

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

    def record_keyword_suppression(self, memory_id, context_id=""):
        """Record that a keyword-only result was marked irrelevant.

        Increments suppression_count if an entry for this memory+context exists,
        otherwise creates a new entry.
        """
        conn = self._conn()
        now = datetime.now().isoformat()
        with conn:
            existing = conn.execute(
                """SELECT id, suppression_count FROM keyword_feedback
                   WHERE memory_id=? AND context_id=?""",
                (memory_id, context_id),
            ).fetchone()
            if existing:
                conn.execute(
                    """UPDATE keyword_feedback SET suppression_count=?, last_updated=?
                       WHERE id=?""",
                    (existing[1] + 1, now, existing[0]),
                )
            else:
                conn.execute(
                    """INSERT INTO keyword_feedback
                       (memory_id, context_id, suppression_count, created_at, last_updated)
                       VALUES (?, ?, 1, ?, ?)""",
                    (memory_id, context_id, now, now),
                )

    def get_keyword_suppression(self, memory_id, context_id=None):
        """Get keyword suppression score for a memory.

        Returns float in [0, 1] where 1.0 = no suppression, approaching 0 = heavily suppressed.
        Formula: 0.5 ^ suppression_count (exponential decay).

        If context_id provided, checks for contextual suppression first.
        Falls back to global (empty context_id) suppression.
        """
        conn = self._conn()

        # Try contextual suppression first
        if context_id:
            row = conn.execute(
                """SELECT suppression_count FROM keyword_feedback
                   WHERE memory_id=? AND context_id=?""",
                (memory_id, context_id),
            ).fetchone()
            if row:
                return 0.5 ** row[0]

        # Fall back to global suppression
        row = conn.execute(
            """SELECT suppression_count FROM keyword_feedback
               WHERE memory_id=? AND context_id=''""",
            (memory_id,),
        ).fetchone()
        if row:
            return 0.5 ** row[0]

        return 1.0  # No suppression

    def reset_keyword_suppression(self, memory_id, context_id=""):
        """Reset suppression for a memory (recovered via another channel)."""
        conn = self._conn()
        with conn:
            conn.execute(
                "DELETE FROM keyword_feedback WHERE memory_id=? AND context_id=?",
                (memory_id, context_id),
            )

    def record_scoring_feedback(self, components: dict, was_useful: bool):
        """Record scoring component values alongside relevance outcome.

        Args:
            components: dict of component_name -> normalized value (0-1).
                Keys: "sim", "imp", "decay", "agent", "rel"
            was_useful: True if the memory was marked useful, False if irrelevant
        """
        conn = self._conn()
        now = datetime.now().isoformat()
        with conn:
            for comp, value in components.items():
                conn.execute(
                    """INSERT INTO scoring_weight_feedback
                       (component, component_value, was_useful, created_at)
                       VALUES (?, ?, ?, ?)""",
                    (comp, float(value), was_useful, now),
                )

    def compute_learned_weights(self, base_weights: dict, min_samples: int = 10):
        """Compute adjusted weights from feedback correlation.

        For each component, compute how predictive it is of usefulness:
        - avg_value_when_useful vs avg_value_when_irrelevant
        - If a component is higher when useful → boost its weight
        - If a component is higher when irrelevant → reduce its weight

        Returns adjusted weights (same keys as base_weights), normalized to sum to 1.0.
        Returns base_weights unchanged if insufficient feedback data.
        """
        conn = self._conn()

        # Check if we have enough data
        total = conn.execute("SELECT COUNT(*) FROM scoring_weight_feedback").fetchone()[0]
        if total < min_samples:
            return dict(base_weights)

        adjustments = {}
        for comp in base_weights:
            rows = conn.execute(
                """SELECT was_useful, AVG(component_value), COUNT(*)
                   FROM scoring_weight_feedback
                   WHERE component=?
                   GROUP BY was_useful""",
                (comp,),
            ).fetchall()
            avg_useful = 0.5
            avg_irrelevant = 0.5
            for row in rows:
                if row[0]:  # useful
                    avg_useful = row[1]
                else:
                    avg_irrelevant = row[1]
            # Correlation: how much higher is this component when useful vs irrelevant
            # Range: roughly [-1, 1]
            correlation = avg_useful - avg_irrelevant
            # Damped adjustment: max ±30% change from base weight
            adjustments[comp] = 1.0 + 0.3 * max(-1.0, min(1.0, correlation))

        # Apply adjustments and normalize
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

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        valid_from: str = None,
        valid_to: str = None,
        confidence: float = 1.0,
        source_closet: str = None,
        source_file: str = None,
        creation_context_id: str = "",
    ):
        """
        Add a relationship triple: subject → predicate → object.

        Examples:
            add_triple("Max", "child_of", "Alice", valid_from="2015-04-01")
            add_triple("Max", "does", "swimming", valid_from="2025-01-01")
            add_triple("Alice", "worried_about", "Max injury", valid_from="2026-01", valid_to="2026-02")
        """
        sub_id = self._entity_id(subject)
        obj_id = self._entity_id(obj)
        pred = _normalize_predicate(predicate)

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

            conn.execute(
                """INSERT INTO triples (id, subject, predicate, object, valid_from, valid_to,
                                        confidence, source_closet, source_file, creation_context_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    triple_id,
                    sub_id,
                    pred,
                    obj_id,
                    valid_from,
                    valid_to,
                    confidence,
                    source_closet,
                    source_file,
                    creation_context_id or "",
                ),
            )
        # Touch both entities (update last_touched for decay scoring)
        self._touch_entity(sub_id)
        self._touch_entity(obj_id)
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
                        "source_closet": row["source_closet"],
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
                        "source_closet": row["source_closet"],
                        "current": row["valid_to"] is None,
                    }
                )

        return results

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

            # Relationships
            parent = facts.get("parent")
            if parent:
                self.add_triple(
                    name, "child_of", parent.capitalize(), valid_from=facts.get("birthday")
                )

            partner = facts.get("partner")
            if partner:
                self.add_triple(name, "married_to", partner.capitalize())

            relationship = facts.get("relationship", "")
            if relationship == "daughter":
                self.add_triple(
                    name,
                    "is_child_of",
                    facts.get("parent", "").capitalize() or name,
                    valid_from=facts.get("birthday"),
                )
            elif relationship == "husband":
                self.add_triple(name, "is_partner_of", facts.get("partner", name).capitalize())
            elif relationship == "brother":
                self.add_triple(name, "is_sibling_of", facts.get("sibling", name).capitalize())
            elif relationship == "dog":
                self.add_triple(name, "is_pet_of", facts.get("owner", name).capitalize())
                self.add_entity(name, "animal")

            # Interests
            for interest in facts.get("interests", []):
                self.add_triple(name, "loves", interest.capitalize(), valid_from="2025-01-01")
