"""MCP server startup smoke tests.

Guards against regressions where the MCP server fails to start on
existing databases (e.g., schema migration ordering bugs, missing
columns, import errors in the module registry).

Every Phase that adds a new SQLite column, ChromaDB collection, or
MCP tool should pass these tests on BOTH fresh databases and
databases with the old schema.
"""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest


class TestMCPStartup:
    """Verify the MCP server boots cleanly in multiple scenarios."""

    def test_fresh_database_starts(self, tmp_path):
        """MCP server starts on a brand-new (empty) palace directory."""
        palace = tmp_path / "fresh_palace"
        palace.mkdir()
        # Run the server, feed empty stdin, expect clean exit
        result = subprocess.run(
            [sys.executable, "-m", "mempalace.mcp_server", "--palace", str(palace)],
            input="",
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(Path(__file__).parent.parent),
        )
        # Server reads stdin EOF and exits cleanly
        assert "MemPalace MCP Server starting" in result.stdout or result.returncode == 0, (
            f"Server failed to start on fresh DB: stdout={result.stdout} stderr={result.stderr}"
        )

    def test_legacy_database_starts(self, tmp_path):
        """MCP server starts on a database with the OLD schema (no new columns).

        This catches ordering bugs where CREATE INDEX runs before migrations
        add the columns the index depends on.
        """
        palace = tmp_path / "legacy_palace"
        palace.mkdir()
        db_path = palace / "kg.db"

        # Simulate a pre-Phase-2 database: edge_traversal_feedback without context_id
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT DEFAULT 'unknown',
                properties TEXT DEFAULT '{}',
                description TEXT DEFAULT '',
                importance INTEGER DEFAULT 3,
                last_touched TEXT DEFAULT '',
                status TEXT DEFAULT 'active',
                merged_into TEXT DEFAULT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE triples (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                valid_from TEXT,
                valid_to TEXT,
                confidence REAL DEFAULT 1.0,
                source_closet TEXT,
                source_file TEXT,
                extracted_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE entity_aliases (
                alias TEXT PRIMARY KEY,
                canonical_id TEXT NOT NULL,
                merged_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            -- Old edge_traversal_feedback WITHOUT context_id column
            CREATE TABLE edge_traversal_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                intent_type TEXT NOT NULL,
                useful BOOLEAN NOT NULL,
                context_keywords TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        conn.close()

        # Now try to import/use knowledge_graph — triggers _init_db + migrations
        env = dict(os.environ)
        env["MEMPALACE_SKIP_SEED"] = "1"  # Don't seed ontology on legacy DB
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0, '.'); "
                "from mempalace.knowledge_graph import KnowledgeGraph; "
                f"kg = KnowledgeGraph(r'{db_path}'); "
                "print('migrated OK')",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(Path(__file__).parent.parent),
            env=env,
        )
        assert "migrated OK" in result.stdout, (
            f"Legacy DB migration failed: stdout={result.stdout} stderr={result.stderr}"
        )

        # Verify the column was added
        conn = sqlite3.connect(str(db_path))
        cols = {row[1] for row in conn.execute("PRAGMA table_info(edge_traversal_feedback)")}
        conn.close()
        assert "context_id" in cols, "context_id column not migrated"

    def test_tool_registry_complete(self):
        """Core Phase 2 tools are registered and deprecated tools are absent."""
        from mempalace import mcp_server

        required = {
            "mempalace_wake_up",
            "mempalace_declare_intent",
            "mempalace_finalize_intent",
            "mempalace_resolve_conflicts",  # P2.1
            "mempalace_kg_add",
            "mempalace_kg_declare_entity",  # also handles memories via kind='record'
            "mempalace_kg_search",  # unified memory+entity search
        }
        missing = required - set(mcp_server.TOOLS.keys())
        assert not missing, f"Missing required tools: {missing}"

        # mempalace_search → kg_search in P3.2; add_drawer → kg_declare_entity(memory) in P3.3.
        removed = {"mempalace_check_duplicate", "mempalace_search", "mempalace_add_drawer"}
        present = removed & set(mcp_server.TOOLS.keys())
        assert not present, f"Deprecated tools still present: {present}"

    def test_valid_kinds_includes_record(self):
        """kind='record' is valid. kind='record' is hard-rejected."""
        from mempalace import mcp_server

        assert "record" in mcp_server.VALID_KINDS
        assert "memory" not in mcp_server.VALID_KINDS
        # _KIND_ALIASES removed; 'memory' hard-rejects with ValueError.
        assert not hasattr(mcp_server, "_KIND_ALIASES")

    def test_jsonrpc_initialize_and_list_tools(self, tmp_path):
        """End-to-end JSON-RPC: initialize + tools/list round-trip."""
        palace = tmp_path / "rpc_palace"
        palace.mkdir()
        proc = subprocess.Popen(
            [sys.executable, "-m", "mempalace.mcp_server", "--palace", str(palace)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            encoding="utf-8",
            cwd=str(Path(__file__).parent.parent),
        )
        try:
            # initialize
            req = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1"},
                },
            }
            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            resp = json.loads(line)
            assert "result" in resp, f"initialize failed: {resp}"
            assert resp["result"]["serverInfo"]["name"] == "mempalace"

            # tools/list
            req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            resp = json.loads(line)
            assert "result" in resp, f"tools/list failed: {resp}"
            tool_names = {t["name"] for t in resp["result"]["tools"]}
            assert "mempalace_resolve_conflicts" in tool_names
            assert "mempalace_check_duplicate" not in tool_names
        finally:
            proc.stdin.close()
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


class TestPendingConflictsRecovery:
    """Regression tests for the deadlock scenario where disk state
    had pending_conflicts but MCP in-memory state was empty after restart.

    These guard against the class of bug that blocked Adrian's session on
    2026-04-16: resolve_conflicts returned 'no conflicts' because memory
    was empty, but the hook kept blocking based on the stale disk state.
    """

    def test_load_pending_conflicts_from_disk(self, tmp_path, monkeypatch):
        """_load_pending_conflicts_from_disk reads from the state file."""
        from mempalace import mcp_server

        state_dir = tmp_path / "hook_state"
        state_dir.mkdir()
        monkeypatch.setattr(mcp_server, "_INTENT_STATE_DIR", state_dir)
        monkeypatch.setattr(mcp_server._STATE, "session_id", "test-sess")

        state_file = state_dir / "active_intent_test-sess.json"
        conflicts = [
            {"id": "c1", "conflict_type": "edge_suggestion", "existing_id": "a", "new_id": "b"}
        ]
        state_file.write_text(json.dumps({"pending_conflicts": conflicts}), encoding="utf-8")

        loaded = mcp_server._load_pending_conflicts_from_disk()
        assert loaded == conflicts

    def test_resolve_conflicts_reloads_from_disk_when_memory_empty(self, tmp_path, monkeypatch):
        """resolve_conflicts loads pending_conflicts from disk if memory is None
        (simulates MCP restart scenario — disk is source of truth)."""
        from mempalace import mcp_server

        state_dir = tmp_path / "hook_state"
        state_dir.mkdir()
        monkeypatch.setattr(mcp_server, "_INTENT_STATE_DIR", state_dir)
        monkeypatch.setattr(mcp_server, "_pending_conflicts", None)
        monkeypatch.setattr(mcp_server, "_active_intent", None)
        # this test doesn't seed a KG, so _require_agent would fail the
        # lookup. Patch _STATE.kg to None → helper takes the graceful-
        # fallback path (KG unavailable) and accepts the agent name as-is.
        monkeypatch.setattr(mcp_server._STATE, "kg", None)
        monkeypatch.setattr(mcp_server._STATE, "session_id", "test-sess")
        monkeypatch.setattr(mcp_server._STATE, "pending_conflicts", None)
        monkeypatch.setattr(mcp_server._STATE, "active_intent", None)

        conflicts = [
            {
                "id": "c1",
                "conflict_type": "edge_suggestion",
                "reason": "test",
                "existing_id": "a",
                "new_id": "b",
                "from": "a",
                "to": "b",
            }
        ]
        state_file = state_dir / "active_intent_test-sess.json"
        state_file.write_text(json.dumps({"pending_conflicts": conflicts}), encoding="utf-8")

        # Call with no actions — should reload from disk and return pending.
        # agent required on the MCP tool; passing test_agent which the
        # test's standalone fixture setup doesn't declare, so KG lookup either
        # degrades gracefully (_kg is None or no matching edge) or we rely on
        # the helper's except-Exception passthrough for fresh test KGs.
        result = mcp_server.tool_resolve_conflicts(agent="test_agent")
        assert result["success"] is False
        assert "Must provide actions" in result["error"]
        assert len(result["pending"]) == 1

        # Call with valid action — should process successfully
        result = mcp_server.tool_resolve_conflicts(
            actions=[
                {
                    "id": "c1",
                    "action": "keep",
                    "reason": "Both items are valid, keeping both in the graph",
                }
            ],
            agent="test_agent",
        )
        assert result["success"] is True
        assert len(result["resolved"]) == 1

    def test_declare_intent_blocks_on_pending_conflicts(self, tmp_path, monkeypatch):
        """declare_intent must block when _pending_conflicts is set (not just legacy)."""
        from mempalace import mcp_server

        monkeypatch.setattr(
            mcp_server,
            "_pending_conflicts",
            [{"id": "c1", "conflict_type": "edge_suggestion"}],
        )
        monkeypatch.setattr(mcp_server, "_active_intent", None)
        # test uses agent="test" (not declared); patch _STATE.kg=None so
        # _require_agent takes the graceful-fallback path and we still see
        # the pending_conflicts error this test is actually checking for.
        monkeypatch.setattr(
            mcp_server._STATE,
            "pending_conflicts",
            [{"id": "c1", "conflict_type": "edge_suggestion"}],
        )
        monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
        monkeypatch.setattr(mcp_server._STATE, "kg", None)

        result = mcp_server.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["thing"]},
            context={
                "queries": ["test inspection", "test perspective"],
                "keywords": ["test", "inspect"],
            },
            agent="test",
            budget={"Read": 1},
        )
        assert result["success"] is False
        assert "conflicts pending" in result["error"]
        assert "pending_conflicts" in result

    def test_legacy_resolve_suggestions_tool_removed(self):
        """After P3.9, tool_resolve_suggestions is removed and is no longer in
        the MCP tool registry. Agents must use resolve_conflicts instead."""
        from mempalace import mcp_server

        assert not hasattr(mcp_server, "tool_resolve_suggestions"), (
            "tool_resolve_suggestions must be deleted in P3.9"
        )
        assert "mempalace_resolve_suggestions" not in mcp_server.TOOLS, (
            "resolve_suggestions must not be in MCP tool registry"
        )
        assert not hasattr(mcp_server, "_pending_edge_suggestions"), (
            "_pending_edge_suggestions global must be removed in P3.12"
        )

    def test_mcp_dispatcher_includes_exception_details(self):
        """When a tool handler raises, the error response must include
        the exception type and message — not a generic 'Internal tool error'."""
        import inspect

        from mempalace import mcp_server

        # Check the dispatcher source includes exception details
        src = inspect.getsource(mcp_server)
        assert "type(e).__name__" in src, (
            "MCP dispatcher must include exception type in error response for debuggability"
        )


@pytest.mark.skipif(
    not os.path.exists(os.path.expanduser("~/.mempalace")),
    reason="requires production palace at ~/.mempalace",
)
class TestProductionDatabase:
    """Smoke tests against the user's actual production palace.

    These run only if ~/.mempalace exists — they verify that the current
    code can still read/use the existing production database.
    """

    def test_production_db_loads(self):
        """The production database loads without errors."""
        from mempalace.knowledge_graph import KnowledgeGraph

        palace = os.path.expanduser("~/.mempalace")
        db_path = os.path.join(palace, "kg.db")
        if not os.path.exists(db_path):
            pytest.skip("no production kg.db")

        kg = KnowledgeGraph(db_path)
        # Should be able to query — if migrations ran, this works
        count = kg._conn().execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        assert count >= 0, "production DB unreadable"
