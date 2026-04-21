"""Tests for the yoyo-migrations schema system.

Verifies that:
- Fresh databases get the full schema applied via migrations
- Legacy databases (pre-yoyo) are correctly bootstrapped without double-applying
- Each migration file is discoverable and applies cleanly
- The existing data is preserved during legacy bootstrap
"""

import os
import sqlite3
import tempfile
import time

import pytest


def _sleep_for_win_unlink():
    """On Windows, SQLite WAL leaves file handles open briefly."""
    if os.name == "nt":
        time.sleep(0.3)


@pytest.fixture
def fresh_db(monkeypatch):
    """Path to a non-existent DB file (ensures fresh migration run)."""
    monkeypatch.setenv("MEMPALACE_SKIP_SEED", "1")
    fd = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    path = fd.name
    fd.close()
    os.unlink(path)  # remove so KG creates fresh
    yield path
    _sleep_for_win_unlink()
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def legacy_db(monkeypatch):
    """A DB with the pre-yoyo schema already populated (no _yoyo_migration)."""
    monkeypatch.setenv("MEMPALACE_SKIP_SEED", "1")
    fd = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    path = fd.name
    fd.close()
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE entities (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, type TEXT DEFAULT 'unknown',
            properties TEXT DEFAULT '{}', description TEXT DEFAULT '',
            importance INTEGER DEFAULT 3, last_touched TEXT DEFAULT '',
            status TEXT DEFAULT 'active', merged_into TEXT DEFAULT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE triples (
            id TEXT PRIMARY KEY, subject TEXT NOT NULL, predicate TEXT NOT NULL,
            object TEXT NOT NULL, valid_from TEXT, valid_to TEXT,
            confidence REAL DEFAULT 1.0, source_closet TEXT, source_file TEXT,
            extracted_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE entity_aliases (
            alias TEXT PRIMARY KEY, canonical_id TEXT NOT NULL,
            merged_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO entities (id, name) VALUES ('legacy_entity', 'Legacy');
        INSERT INTO triples (id, subject, predicate, object)
            VALUES ('t1', 'legacy_entity', 'is_a', 'thing');
    """)
    conn.commit()
    conn.close()
    yield path
    _sleep_for_win_unlink()
    try:
        os.unlink(path)
    except OSError:
        pass


class TestFreshDatabase:
    """Fresh DBs run all migrations and end up with the full schema."""

    def test_all_tables_created(self, fresh_db):
        from mempalace.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph(fresh_db)
        conn = kg._conn()
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        # P2 cutover dropped edge_traversal_feedback + keyword_feedback
        # (migration 015). The remaining tables are the live set.
        for t in (
            "entities",
            "triples",
            "entity_aliases",
            "scoring_weight_feedback",
            "entity_keywords",
            "keyword_idf",
            "_yoyo_migration",
        ):
            assert t in tables, f"missing table {t}"
        # Retired tables MUST be absent post-migration 015.
        assert "edge_traversal_feedback" not in tables
        assert "keyword_feedback" not in tables
        kg.close()

    def test_entity_metadata_columns(self, fresh_db):
        from mempalace.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph(fresh_db)
        cols = {row[1] for row in kg._conn().execute("PRAGMA table_info(entities)").fetchall()}
        for c in ("description", "importance", "status", "merged_into", "kind"):
            assert c in cols, f"missing column {c}"
        kg.close()

    def test_triples_properties_column(self, fresh_db):
        """P2 added a generic properties TEXT column for surfaced/rated_* edges."""
        from mempalace.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph(fresh_db)
        cols = {row[1] for row in kg._conn().execute("PRAGMA table_info(triples)").fetchall()}
        assert "properties" in cols
        kg.close()


class TestLegacyBootstrap:
    """Legacy DBs (pre-yoyo) are bootstrapped: yoyo marker added, no re-runs."""

    def test_preserves_existing_data(self, legacy_db):
        from mempalace.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph(legacy_db)
        count = kg._conn().execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        assert count == 1
        kg.close()

    def test_yoyo_marker_added(self, legacy_db):
        from mempalace.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph(legacy_db)
        tables = {
            row[0]
            for row in kg._conn()
            .execute("SELECT name FROM sqlite_master WHERE type='table'")
            .fetchall()
        }
        assert "_yoyo_migration" in tables
        kg.close()

    def test_all_migrations_marked_applied(self, legacy_db):
        """Bootstrap should mark all SQL migrations as applied (no reruns)."""
        from mempalace.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph(legacy_db)
        applied = {
            row[0]
            for row in kg._conn().execute("SELECT migration_id FROM _yoyo_migration").fetchall()
        }
        # All our numbered SQL migrations should be marked (001..006)
        for expected in (
            "001_initial_schema",
            "002_entity_metadata_columns",
            "003_edge_traversal_feedback",
            "004_edge_context_id",
            "005_keyword_feedback",
            "006_scoring_weight_feedback",
        ):
            assert expected in applied, f"{expected} not marked applied"
        kg.close()

    def test_reopen_idempotent(self, legacy_db):
        """Reopening a bootstrapped DB should not re-run any migrations."""
        from mempalace.knowledge_graph import KnowledgeGraph

        kg1 = KnowledgeGraph(legacy_db)
        kg1.close()
        _sleep_for_win_unlink()
        # Second open — idempotent, no errors
        kg2 = KnowledgeGraph(legacy_db)
        assert kg2._conn().execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 1
        kg2.close()


class TestMigrationDiscovery:
    """The migrations module can be imported and discovers files correctly."""

    def test_migrations_dir_exists(self):
        from mempalace.migrations import MIGRATIONS_DIR

        assert MIGRATIONS_DIR.is_dir()

    def test_migration_files_discoverable(self):
        """All 6 numbered migrations exist as either .sql or .py files."""
        from mempalace.migrations import MIGRATIONS_DIR

        # Collect both .sql and .py migrations (002 is .py for defensive ALTERs)
        stems = {
            p.stem
            for p in MIGRATIONS_DIR.iterdir()
            if p.suffix in (".sql", ".py") and p.name != "__init__.py"
        }
        for expected in (
            "001_initial_schema",
            "002_entity_metadata_columns",
            "003_edge_traversal_feedback",
            "004_edge_context_id",
            "005_keyword_feedback",
            "006_scoring_weight_feedback",
        ):
            assert expected in stems, f"missing migration {expected}"

    def test_apply_migrations_callable(self, fresh_db):
        from mempalace.migrations import apply_migrations

        # Should not raise on a fresh DB (creates it and applies all)
        apply_migrations(fresh_db)
        conn = sqlite3.connect(fresh_db)
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        conn.close()
        assert "_yoyo_migration" in tables
        assert "entities" in tables
