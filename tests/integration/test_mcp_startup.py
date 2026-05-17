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
        add the columns the index depends on. P2 cutover dropped
        edge_traversal_feedback entirely; the legacy simulation now covers
        a pre-P2 database that STILL has the retired table -- migration 015
        should drop it cleanly on boot.
        """
        palace = tmp_path / "legacy_palace"
        palace.mkdir()
        db_path = palace / "kg.db"

        # Simulate a pre-P2 database that still has the retired tables.
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
            -- Pre-P2 retired tables that migration 015 should drop.
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
            CREATE TABLE keyword_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                keyword TEXT NOT NULL,
                was_useful BOOLEAN,
                context_id TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        conn.close()

        # Now try to import/use knowledge_graph -- triggers _init_db + migrations
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

        # Verify the retired tables are gone after migration 015.
        conn = sqlite3.connect(str(db_path))
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        conn.close()
        assert "edge_traversal_feedback" not in tables, "migration 015 should drop it"
        assert "keyword_feedback" not in tables, "migration 015 should drop it"

    def test_tool_registry_complete(self):
        """Core Phase 2 tools are registered and deprecated tools are absent."""
        from mempalace import mcp_server

        required = {
            "mempalace_wake_up",
            "mempalace_declare_intent",
            "mempalace_finalize_intent",
            "mempalace_kg_add",
            "mempalace_kg_declare_entity",  # also handles memories via kind='record'
            "mempalace_kg_search",  # unified memory+entity search
        }
        missing = required - set(mcp_server.TOOLS.keys())
        assert not missing, f"Missing required tools: {missing}"

        # mempalace_search → kg_search in P3.2; add_drawer → kg_declare_entity(memory) in P3.3.
        # v3.7.20 (Adrian directive 2026-05-17): resolve_conflicts +
        # list_pending_conflicts removed -- conflicts are resolved by Haiku
        # in the background via mempalace/conflict_resolver_auto.py.
        removed = {
            "mempalace_check_duplicate",
            "mempalace_search",
            "mempalace_add_drawer",
            "mempalace_resolve_conflicts",
            "mempalace_list_pending_conflicts",
        }
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
        """End-to-end JSON-RPC: initialize + tools/list round-trip.

        IMPORTANT: stderr is redirected to a file (not subprocess.PIPE)
        because the MCP server can emit a large burst of boot output --
        e.g. yoyo migration progress on a fresh palace can be 100+
        lines. A captured-but-undrained PIPE deadlocks at the OS pipe
        buffer threshold (~4 KiB on Windows), which silently hangs the
        server before its first JSON-RPC read. Writing stderr to a real
        file removes the back-pressure entirely.
        """
        palace = tmp_path / "rpc_palace"
        palace.mkdir()
        stderr_log = tmp_path / "mcp_stderr.log"
        with open(stderr_log, "w", encoding="utf-8") as stderr_fh:
            proc = subprocess.Popen(
                [sys.executable, "-u", "-m", "mempalace.mcp_server", "--palace", str(palace)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_fh,
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
                assert "result" in resp, (
                    f"initialize failed: {resp}; stderr={stderr_log.read_text(errors='replace')[-2000:]}"
                )
                assert resp["result"]["serverInfo"]["name"] == "mempalace"

                # tools/list
                req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
                proc.stdin.write(json.dumps(req) + "\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
                resp = json.loads(line)
                assert "result" in resp, f"tools/list failed: {resp}"
                tool_names = {t["name"] for t in resp["result"]["tools"]}
                # v3.7.20: resolve_conflicts removed; tool must be absent.
                assert "mempalace_resolve_conflicts" not in tool_names
                assert "mempalace_list_pending_conflicts" not in tool_names
                assert "mempalace_check_duplicate" not in tool_names
            finally:
                proc.stdin.close()
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()


# TestPendingConflictsRecovery removed in v3.7.20 (Adrian directive
# 2026-05-17). The deadlock-recovery scenario it guarded against is
# moot: there is no longer a blocking gate to deadlock on, and the
# tool_resolve_conflicts handler the recovery tests called has been
# removed in favour of the bg Haiku resolver
# (mempalace/conflict_resolver_auto.py).


class TestMCPTransportLevelFinalizeRoundTrip:
    """Adrian directive 2026-05-12 (Phase 5 post-mortem): the in-process
    integration suite went green for Phase 5 (sqlite_vec default + lazy
    chromadb) but the v3.1.12 plugin still crashed the MCP transport on
    the user's *first real* ``mempalace_finalize_intent`` call after
    reinstall (``MCP error -32000: Connection closed``). The crash
    happened in the **subprocess** -- a Python-level exception inside
    the handler, or a C-level SIGSEGV inside chromadb -- not a
    `return error response` path. Our in-process tests can't see that
    class of failure because they call ``handle_request`` directly in
    the test process; if Python dies in there, pytest dies with it and
    the failure is reported as "pytest crashed" instead of "this test
    failed".

    These tests spawn the actual ``mempalace`` MCP server as a
    **subprocess**, drive it over **stdio** JSON-RPC the same way
    Claude Code does, run the full intent lifecycle that the user hit
    (wake_up -> declare_intent -> declare_operation -> kg_declare_entity
    -> finalize_intent), and assert the subprocess is still alive after
    every call. If finalize_intent SIGSEGVs the server, the assertion
    on ``proc.poll() is None`` immediately catches it -- the next time
    we say "tests are green", that statement covers the round-trip the
    user actually runs.

    Lifecycle of one rpc():
      - write JSON line to stdin, flush
      - readline from stdout with a generous timeout
      - assert proc.poll() is None (server still running)
      - return parsed response

    A hang on readline means the server is alive but stuck -- still a
    failure, surfaced via the per-call timeout. A SIGSEGV closes
    stdout, which means readline returns "" -- caught by the `assert
    line` check below with a clear "server died" message.
    """

    # Keep these wide enough to cover Chroma model-cache cold-start on
    # first call (~5s on a warm machine, ~15s on a cold ARM box with
    # x64 emulation) but tight enough that a hang surfaces fast.
    INIT_TIMEOUT_S = 60.0
    CALL_TIMEOUT_S = 60.0

    def _rpc(self, proc, req: dict, timeout: float, stderr_log: Path) -> dict:
        """Send one JSON-RPC request and receive the response. Assert the
        server is still running after the response lands."""
        assert proc.poll() is None, (
            f"server died BEFORE request id={req.get('id')} method={req.get('method')!r}; "
            f"exit code={proc.returncode}; "
            f"stderr tail=\n{stderr_log.read_text(errors='replace')[-2000:]}"
        )
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()

        # Read with a wall-clock deadline so a true hang surfaces.
        # readline() doesn't accept a timeout natively, so we use a
        # background thread + Queue. Standard library only, no async.
        import queue
        import threading

        q: "queue.Queue[str]" = queue.Queue()

        def _reader():
            try:
                q.put(proc.stdout.readline())
            except Exception as exc:  # pragma: no cover
                q.put(f"__READER_EXC__:{type(exc).__name__}:{exc}")

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        try:
            line = q.get(timeout=timeout)
        except queue.Empty:
            raise AssertionError(
                f"server HUNG on request id={req.get('id')} "
                f"method={req.get('method')!r} tool={req.get('params', {}).get('name')!r} "
                f"(no response in {timeout}s); proc.poll()={proc.poll()}; "
                f"stderr tail=\n{stderr_log.read_text(errors='replace')[-2000:]}"
            )

        if line.startswith("__READER_EXC__:"):  # pragma: no cover
            raise AssertionError(f"stdout reader raised: {line}")

        if not line:
            # EOF on stdout = subprocess died mid-handler. This is THE
            # signal that pre-Phase-5 v3.1.12 emitted for the user --
            # the test must fail clearly here, not produce a confusing
            # JSON decode error two assertions down.
            assert proc.poll() is not None, (
                "stdout EOF but proc.poll() is None -- transport closed but server still running?"
            )
            raise AssertionError(
                f"server DIED during request id={req.get('id')} "
                f"method={req.get('method')!r} tool={req.get('params', {}).get('name')!r}; "
                f"exit code={proc.returncode}; "
                f"this is the MCP -32000 / Connection closed class -- the very thing "
                f"Phase 5 was meant to retire; "
                f"stderr tail=\n{stderr_log.read_text(errors='replace')[-3000:]}"
            )

        try:
            resp = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"server returned non-JSON for request id={req.get('id')} "
                f"method={req.get('method')!r}: {line[:500]!r}; decode error={exc}; "
                f"stderr tail=\n{stderr_log.read_text(errors='replace')[-2000:]}"
            )

        assert proc.poll() is None, (
            f"server died AFTER responding to id={req.get('id')} method={req.get('method')!r}; "
            f"exit code={proc.returncode}; "
            f"stderr tail=\n{stderr_log.read_text(errors='replace')[-2000:]}"
        )
        return resp

    def _spawn(
        self, palace: Path, stderr_log: Path, *, backend: str = "sqlite_vec"
    ) -> subprocess.Popen:
        """Spawn the MCP server subprocess. stderr -> real file (NOT
        PIPE) to avoid the OS pipe-buffer deadlock that hangs the
        server on its first JSON-RPC read once it bursts >4 KiB of
        boot output (see test_jsonrpc_initialize_and_list_tools).

        ``backend`` selects ``MEMPALACE_VECTOR_BACKEND`` -- pass
        ``"chroma"`` to seed a legacy-shaped palace for the upgrade
        smoke; default ``"sqlite_vec"`` covers the Phase-5 path."""
        # Append mode: legacy-upgrade test runs the server twice
        # against the same stderr log; first-pass content is part of
        # the failure context if the second pass crashes.
        stderr_fh = open(stderr_log, "a", encoding="utf-8")
        env = dict(os.environ)
        env["MEMPALACE_VECTOR_BACKEND"] = backend
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "mempalace.mcp_server", "--palace", str(palace)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_fh,
            text=True,
            bufsize=1,
            encoding="utf-8",
            cwd=str(Path(__file__).parent.parent),
            env=env,
        )
        # Smuggle the stderr file handle on the proc so the caller can
        # close it during teardown without re-opening the file.
        proc._mempalace_stderr_fh = stderr_fh  # type: ignore[attr-defined]
        return proc

    def _shutdown(self, proc: subprocess.Popen) -> None:
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        fh = getattr(proc, "_mempalace_stderr_fh", None)
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass

    def test_finalize_intent_round_trip_does_not_crash_server(self, tmp_path):
        """Full intent lifecycle over MCP stdio against sqlite_vec on a
        fresh palace. THIS is the test that pre-Phase-5 v3.1.12 would
        have failed -- finalize would crash the server, ``proc.poll()``
        would return a non-None exit code, and the rpc() helper would
        raise with a clear "server DIED during finalize_intent" message
        instead of a silent pass.
        """
        palace = tmp_path / "rpc_palace"
        palace.mkdir()
        stderr_log = tmp_path / "mcp_stderr.log"
        proc = self._spawn(palace, stderr_log, backend="sqlite_vec")
        try:
            self._run_full_intent_lifecycle(
                proc,
                stderr_log,
                agent="transport_test_agent",
                slug="transport-smoke-finalize",
                id_base=0,
            )
        finally:
            self._shutdown(proc)

    def _run_full_intent_lifecycle(
        self,
        proc: subprocess.Popen,
        stderr_log: Path,
        *,
        agent: str,
        slug: str,
        id_base: int = 0,
    ) -> None:
        """Drive one wake_up -> declare_intent -> finalize_intent
        cycle. Shared between the fresh-palace test and the
        chroma-built-palace upgrade test so the round-trip shape stays
        identical and any change to the lifecycle is picked up by both
        tests."""
        # initialize
        init_resp = self._rpc(
            proc,
            {
                "jsonrpc": "2.0",
                "id": id_base + 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "mempalace-transport-test", "version": "1"},
                },
            },
            timeout=self.INIT_TIMEOUT_S,
            stderr_log=stderr_log,
        )
        assert "result" in init_resp, f"initialize failed: {init_resp}"

        # wake_up
        wake_resp = self._rpc(
            proc,
            {
                "jsonrpc": "2.0",
                "id": id_base + 2,
                "method": "tools/call",
                "params": {
                    "name": "mempalace_wake_up",
                    "arguments": {"agent": agent},
                },
            },
            timeout=self.CALL_TIMEOUT_S,
            stderr_log=stderr_log,
        )
        assert "result" in wake_resp, f"wake_up failed: {wake_resp}"

        # declare_intent
        declare_resp = self._rpc(
            proc,
            {
                "jsonrpc": "2.0",
                "id": id_base + 3,
                "method": "tools/call",
                "params": {
                    "name": "mempalace_declare_intent",
                    "arguments": {
                        "intent_type": "kg_curate",
                        "slots": {"subject": ["transport_smoke"]},
                        "agent": agent,
                    },
                },
            },
            timeout=self.CALL_TIMEOUT_S,
            stderr_log=stderr_log,
        )
        assert "result" in declare_resp, f"declare_intent failed: {declare_resp}"

        # finalize_intent -- the call that crashed v3.1.12 against the
        # user's real palace
        finalize_resp = self._rpc(
            proc,
            {
                "jsonrpc": "2.0",
                "id": id_base + 4,
                "method": "tools/call",
                "params": {
                    "name": "mempalace_finalize_intent",
                    "arguments": {
                        "agent": agent,
                        "slug": slug,
                        "outcome": "completed",
                        "summary": {
                            "what": "transport smoke completed",
                            "why": "verify finalize round-trip does not crash MCP server",
                            "scope": "transport-test",
                        },
                        "content": (
                            "Transport-level smoke: this finalize must return a "
                            "response without crashing the MCP server subprocess."
                        ),
                    },
                },
            },
            timeout=self.CALL_TIMEOUT_S,
            stderr_log=stderr_log,
        )
        assert "result" in finalize_resp or "error" in finalize_resp, (
            f"finalize_intent returned malformed JSON-RPC: {finalize_resp}"
        )

        # Post-finalize liveness probe
        after_resp = self._rpc(
            proc,
            {"jsonrpc": "2.0", "id": id_base + 5, "method": "tools/list"},
            timeout=self.CALL_TIMEOUT_S,
            stderr_log=stderr_log,
        )
        assert "result" in after_resp, f"tools/list after finalize failed: {after_resp}"

    def test_finalize_intent_round_trip_survives_server_restart(self, tmp_path):
        """Boot the server once, complete a finalize lifecycle, shut
        down, boot again against the SAME palace, complete a second
        lifecycle. This is the real-world reinstall path the user
        hit -- the previous version of this test seeded the first
        pass under chromadb, but with chromadb removed the
        relevant invariant is just "a finalized palace can be
        reopened by a new server process without crashing the second
        finalize". Tests the persistence + reopen path end-to-end."""
        palace = tmp_path / "restart_palace"
        palace.mkdir()
        stderr_log = tmp_path / "mcp_stderr.log"

        proc1 = self._spawn(palace, stderr_log, backend="sqlite_vec")
        try:
            self._run_full_intent_lifecycle(
                proc1,
                stderr_log,
                agent="restart_smoke_agent",
                slug="first-pass-finalize",
                id_base=0,
            )
        finally:
            self._shutdown(proc1)

        proc2 = self._spawn(palace, stderr_log, backend="sqlite_vec")
        try:
            self._run_full_intent_lifecycle(
                proc2,
                stderr_log,
                agent="restart_smoke_agent",
                slug="second-pass-finalize-after-restart",
                id_base=100,
            )
        finally:
            self._shutdown(proc2)


class TestLegacyPalaceDbLoads:
    """Adrian directive 2026-05-12: the prior ``TestProductionDatabase``
    class probed ``~/.mempalace`` directly, but conftest.py
    deliberately redirects HOME to a temp dir for the entire test
    session (test hygiene -- the suite must not touch the user's real
    palace). That made the class skip on every modern install with
    "no production kg.db", which was a *structurally broken* test --
    it could never run from inside pytest, so it gave no signal
    either way.

    The replacement here exercises the same invariant ("a real-shaped
    KG sqlite file can be opened and queried") against a synthetic
    KG-DB built fresh inside ``tmp_path``. It deliberately does NOT
    look at the user's home directory -- if you want to verify an
    actual production palace, run mempalace against it manually
    (which is what the user does on every wake_up).
    """

    def test_synthetic_palace_db_loads_and_counts(self, tmp_path):
        """KnowledgeGraph can open a fresh sqlite file, run migrations,
        and answer a count query -- the bare-minimum production-shape
        smoke."""
        from mempalace.knowledge_graph import KnowledgeGraph

        db_path = str(tmp_path / "knowledge_graph.sqlite3")
        kg = KnowledgeGraph(db_path)
        count = kg._conn().execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        assert count >= 0, "fresh palace DB unreadable after migrations"


pytestmark = pytest.mark.integration
