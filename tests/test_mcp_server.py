"""
test_mcp_server.py — Tests for the MCP server tool handlers and dispatch.

Tests each tool handler directly (unit-level) and the handle_request
dispatch layer (integration-level). Uses isolated palace + KG fixtures
via monkeypatch to avoid touching real data.
"""

import json


def _patch_mcp_server(monkeypatch, config, kg):
    """Patch the mcp_server module globals to use test fixtures."""
    from mempalace import mcp_server

    monkeypatch.setattr(mcp_server._STATE, "config", config)
    monkeypatch.setattr(mcp_server._STATE, "kg", kg)
    # Every state-writing tool now requires a real session_id (no
    # default/unknown fallback). Give fixtures a stable one; individual
    # tests that need to exercise the "no sid" path override explicitly.
    monkeypatch.setattr(mcp_server._STATE, "session_id", "test-session")

    # Seed agent class + test_agent so added_by validation passes
    kg.add_entity("agent", kind="class", description="An AI agent", importance=5)
    kg.add_entity(
        "test_agent", kind="entity", description="Test agent for unit tests", importance=3
    )
    kg.add_triple("test_agent", "is_a", "agent")


def _get_collection(palace_path, create=False):
    """Helper to get collection from test palace.

    Returns (client, collection) so callers can clean up the client
    when they are done.
    """
    import chromadb

    client = chromadb.PersistentClient(path=palace_path)
    if create:
        return client, client.get_or_create_collection("mempalace_records")
    return client, client.get_collection("mempalace_records")


# ── Protocol Layer ──────────────────────────────────────────────────────


class TestHandleRequest:
    def test_initialize(self):
        from mempalace.mcp_server import handle_request

        resp = handle_request({"method": "initialize", "id": 1, "params": {}})
        assert resp["result"]["serverInfo"]["name"] == "mempalace"
        assert resp["id"] == 1

    def test_initialize_negotiates_client_version(self):
        from mempalace.mcp_server import handle_request

        resp = handle_request(
            {
                "method": "initialize",
                "id": 1,
                "params": {"protocolVersion": "2025-11-25"},
            }
        )
        assert resp["result"]["protocolVersion"] == "2025-11-25"

    def test_initialize_negotiates_older_supported_version(self):
        from mempalace.mcp_server import handle_request

        resp = handle_request(
            {
                "method": "initialize",
                "id": 1,
                "params": {"protocolVersion": "2025-03-26"},
            }
        )
        assert resp["result"]["protocolVersion"] == "2025-03-26"

    def test_initialize_unknown_version_falls_back_to_latest(self):
        from mempalace.mcp_server import handle_request

        resp = handle_request(
            {
                "method": "initialize",
                "id": 1,
                "params": {"protocolVersion": "9999-12-31"},
            }
        )
        from mempalace.mcp_server import SUPPORTED_PROTOCOL_VERSIONS

        assert resp["result"]["protocolVersion"] == SUPPORTED_PROTOCOL_VERSIONS[0]

    def test_initialize_missing_version_uses_oldest(self):
        from mempalace.mcp_server import handle_request, SUPPORTED_PROTOCOL_VERSIONS

        resp = handle_request({"method": "initialize", "id": 1, "params": {}})
        assert resp["result"]["protocolVersion"] == SUPPORTED_PROTOCOL_VERSIONS[-1]

    def test_notifications_initialized_returns_none(self):
        from mempalace.mcp_server import handle_request

        resp = handle_request({"method": "notifications/initialized", "id": None, "params": {}})
        assert resp is None

    def test_tools_list(self):
        from mempalace.mcp_server import handle_request

        resp = handle_request({"method": "tools/list", "id": 2, "params": {}})
        tools = resp["result"]["tools"]
        names = {t["name"] for t in tools}
        assert "mempalace_kg_search" in names
        assert "mempalace_search" not in names  # merged into kg_search
        assert "mempalace_add_drawer" not in names  # merged into kg_declare_entity
        assert "mempalace_kg_declare_entity" in names
        assert "mempalace_kg_add" in names
        assert "mempalace_resolve_conflicts" in names

    def test_null_arguments_does_not_hang(self, monkeypatch, config, palace_path, seeded_kg):
        """Sending arguments: null should return a result, not hang (#394)."""
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import handle_request

        _client, _col = _get_collection(palace_path, create=True)
        del _client
        resp = handle_request(
            {
                "method": "tools/call",
                "id": 10,
                "params": {"name": "mempalace_kg_stats", "arguments": None},
            }
        )
        assert "error" not in resp
        assert resp["result"] is not None

    def test_unknown_tool(self):
        from mempalace.mcp_server import handle_request

        resp = handle_request(
            {
                "method": "tools/call",
                "id": 3,
                "params": {"name": "nonexistent_tool", "arguments": {}},
            }
        )
        assert resp["error"]["code"] == -32601

    def test_unknown_method(self):
        from mempalace.mcp_server import handle_request

        resp = handle_request({"method": "unknown/method", "id": 4, "params": {}})
        assert resp["error"]["code"] == -32601

    def test_tools_call_dispatches(self, monkeypatch, config, palace_path, seeded_kg):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import handle_request

        # Create a collection so the dispatcher can load state
        _client, _col = _get_collection(palace_path, create=True)
        del _client

        resp = handle_request(
            {
                "method": "tools/call",
                "id": 5,
                "params": {"name": "mempalace_kg_stats", "arguments": {}},
            }
        )
        assert "result" in resp
        content = json.loads(resp["result"]["content"][0]["text"])
        assert "entities" in content or "triples" in content


# ── Crash visibility ────────────────────────────────────────────────────
# Two layers of crash visibility added 2026-04-25 after a Chroma HNSW
# C-level segfault produced "Connection closed -32000" with zero
# diagnostic detail (~90 minutes of opaque debugging). Tests below lock
# both layers in so a future refactor can't silently regress.


class TestCrashVisibility:
    def test_faulthandler_enabled_at_module_load(self):
        """faulthandler.is_enabled() must be True after mcp_server is
        imported. C-level access violations (Chroma HNSW, onnx) bypass
        Python's exception machinery; without faulthandler the process
        is killed with no traceback and the MCP client sees only a
        connection drop. Enabling it ensures stderr gets a Python
        stack frame at the crash site before the OS kills us."""
        # Import triggers the module-level enable() call.
        import faulthandler

        from mempalace import mcp_server  # noqa: F401

        assert faulthandler.is_enabled(), (
            "mcp_server module load must enable faulthandler so C-level "
            "crashes (Chroma HNSW segfaults) leave a Python stack on "
            "stderr instead of a silent connection-close."
        )

    def test_main_does_not_auto_run_hyphen_id_migration(self):
        """The N3 hyphen-id migration is RETIRED from auto-startup
        (2026-04-25). It was a one-shot legacy migration whose
        ``col.get(include=['embeddings'])`` call could trip a known
        Chroma v0.6.0 bug ('list assignment index out of range') and
        leave the HNSW vector index half-written, causing C-level
        access violations on subsequent queries (Adrian's session
        2026-04-25). The migration code is preserved in
        ``hyphen_id_migration.py`` for users with genuinely legacy
        data, but must NOT be called automatically. This test
        statically asserts the call site is gone so a future refactor
        can't silently regress.
        """
        from pathlib import Path

        src = Path(__file__).parent.parent / "mempalace" / "mcp_server.py"
        text = src.read_text(encoding="utf-8")
        # The function definition itself is fine — only the auto-call
        # from main() startup is forbidden. Reject any line that calls
        # the function (with or without leading whitespace).
        offending = [
            line
            for line in text.splitlines()
            if "_run_hyphen_id_migration_once()" in line
            and not line.lstrip().startswith(("#", '"', "'"))
            and not line.lstrip().startswith("def ")
        ]
        assert offending == [], (
            "Auto-call to _run_hyphen_id_migration_once() must not be "
            "reintroduced in mcp_server.py main() startup. The migration "
            "is preserved as a manual recovery path only. Found: "
            f"{offending!r}"
        )

    def test_tools_call_emits_heartbeat_to_stderr(self, capfd, monkeypatch, config, kg):
        """Every tools/call dispatch must write a heartbeat line to
        stderr BEFORE invoking the handler so post-mortem stderr can
        identify which tool killed the process when a C-level segfault
        is the next event. Without this, the only signal in the log is
        a generic Windows fatal exception with no tool context."""
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import handle_request

        handle_request(
            {
                "method": "tools/call",
                "id": 99,
                "params": {"name": "mempalace_kg_stats", "arguments": {}},
            }
        )
        _stdout, stderr = capfd.readouterr()
        assert "[mcp] tools/call: mempalace_kg_stats" in stderr, (
            f"heartbeat missing from stderr; got: {stderr!r}"
        )


# ── Search Tool ─────────────────────────────────────────────────────────


class TestSearchTool:
    def test_search_basic(self, monkeypatch, config, palace_path, seeded_collection, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_kg_search

        result = tool_kg_search(
            context={
                "queries": ["JWT authentication tokens", "test perspective"],
                "keywords": ["test", "search"],
                "entities": ["jwt"],
            }
        )
        assert "results" in result
        assert len(result["results"]) > 0
        # Memory result should surface — top hit should be the auth memory content
        memory_hits = [r for r in result["results"] if r.get("source") == "memory"]
        assert memory_hits, "expected at least one memory hit for JWT query"
        assert any("JWT" in r["text"] or "authentication" in r["text"].lower() for r in memory_hits)

    def test_search_with_agent_affinity(
        self, monkeypatch, config, palace_path, seeded_collection, kg
    ):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_kg_search

        # agent param provides affinity scoring (not hard filter)
        result = tool_kg_search(
            context={
                "queries": ["planning", "test perspective"],
                "keywords": ["test", "search"],
                "entities": ["planning"],
            },
            agent="miner",
        )
        assert "results" in result
        assert len(result["results"]) > 0


# ── Write Tools ─────────────────────────────────────────────────────────


_MEMORY_CONTEXT = {
    "queries": ["python decorators primer", "metaclass guide", "advanced python features"],
    "keywords": ["python", "decorators", "metaclass"],
    "entities": ["python", "decorators"],
}
_RUST_CONTEXT = {
    "queries": ["rust ownership rules", "borrow checker basics"],
    "keywords": ["rust", "ownership", "borrow"],
    "entities": ["rust", "ownership"],
}


class TestWriteTools:
    def test_add_memory_via_kg_declare_entity(self, monkeypatch, config, palace_path, kg):
        """P3.3 + memories are created via kg_declare_entity(kind='record') with Context."""
        _patch_mcp_server(monkeypatch, config, kg)
        _client, _col = _get_collection(palace_path, create=True)
        del _client
        from mempalace.mcp_server import tool_kg_declare_entity

        result = tool_kg_declare_entity(
            kind="record",
            slug="python-decorators-metaclasses",
            content="This is a test memory about Python decorators and metaclasses.",
            summary="Notes on Python decorators and metaclasses for later reference.",
            content_type="fact",
            context=_MEMORY_CONTEXT,
            added_by="test_agent",
        )
        assert result["success"] is True, result
        assert result["memory_id"] == "record_test_agent_python_decorators_metaclasses"

    def test_add_memory_duplicate_detection(self, monkeypatch, config, palace_path, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        _client, _col = _get_collection(palace_path, create=True)
        del _client
        from mempalace.mcp_server import tool_kg_declare_entity

        content = "This is a unique test memory about Rust ownership and borrowing."
        result1 = tool_kg_declare_entity(
            kind="record",
            slug="rust-ownership",
            content=content,
            summary="Rust ownership and borrowing notes.",
            content_type="fact",
            context=_RUST_CONTEXT,
            added_by="test_agent",
        )
        assert result1["success"] is True, result1

        # Same slug for same agent → collision
        result2 = tool_kg_declare_entity(
            kind="record",
            slug="rust-ownership",
            content="different content",
            summary="Rust ownership placeholder collision.",
            content_type="fact",
            context=_RUST_CONTEXT,
            added_by="test_agent",
        )
        assert result2["success"] is False
        assert "already exists" in result2["error"]
        assert "existing_memory" in result2

    def test_kg_declare_entity_memory_requires_slug(self, monkeypatch, config, palace_path, kg):
        """kind='record' rejects calls missing slug with helpful error."""
        _patch_mcp_server(monkeypatch, config, kg)
        _client, _col = _get_collection(palace_path, create=True)
        del _client
        from mempalace.mcp_server import tool_kg_declare_entity

        result = tool_kg_declare_entity(
            kind="record",
            content="some content",
            context=_MEMORY_CONTEXT,
            added_by="test_agent",
            # slug missing
        )
        assert result["success"] is False
        assert "slug" in result["error"]

    def test_kg_declare_entity_rejects_legacy_description(
        self, monkeypatch, config, palace_path, kg
    ):
        """passing the old `description` kwarg with no `context` returns a loud error."""
        _patch_mcp_server(monkeypatch, config, kg)
        _client, _col = _get_collection(palace_path, create=True)
        del _client
        from mempalace.mcp_server import tool_kg_declare_entity

        result = tool_kg_declare_entity(
            name="LegacyEntity",
            description="some text",
            kind="entity",
            added_by="test_agent",
        )
        assert result["success"] is False
        # Error should mention 'context' and that 'description' is gone
        assert "context" in result["error"]
        assert "description" in result["error"].lower()

    def test_kg_declare_entity_rejects_single_string_queries(
        self, monkeypatch, config, palace_path, kg
    ):
        """validate_context rejects single-string queries on writes too."""
        _patch_mcp_server(monkeypatch, config, kg)
        _client, _col = _get_collection(palace_path, create=True)
        del _client
        from mempalace.mcp_server import tool_kg_declare_entity

        result = tool_kg_declare_entity(
            name="ShouldFail",
            kind="entity",
            added_by="test_agent",
            context={"queries": "one string", "keywords": ["a", "b"]},
        )
        assert result["success"] is False
        assert "must be a LIST" in result["error"]

    def test_kg_declare_entity_keywords_persisted(self, monkeypatch, config, palace_path, kg):
        """caller-provided keywords land in the entity_keywords table."""
        _patch_mcp_server(monkeypatch, config, kg)
        _client, _col = _get_collection(palace_path, create=True)
        del _client
        from mempalace.mcp_server import tool_kg_declare_entity

        result = tool_kg_declare_entity(
            name="LoginService",
            kind="entity",
            added_by="test_agent",
            importance=4,
            context={
                "queries": ["the auth login service", "JWT issuer endpoint"],
                "keywords": ["login", "auth", "jwt"],
                "entities": ["LoginService"],
            },
        )
        assert result["success"] is True, result
        # Keywords stored in the new entity_keywords table
        assert kg.get_entity_keywords("login_service") == ["login", "auth", "jwt"]
        # Keyword channel can locate the entity by literal term
        assert "login_service" in kg.entity_ids_for_keyword("jwt")
        # Creation Context: legacy persist_context is retired (empty string);
        # the context-as-entity substrate now records provenance via
        # created_under edges, exercised in test_context_emit_sites.py.

    def test_kg_delete_entity_record(self, monkeypatch, config, palace_path, seeded_collection, kg):
        """unified kg_delete_entity works on record IDs (record_ prefix)."""
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_kg_delete_entity

        result = tool_kg_delete_entity("record_proj_backend_aaa", agent="test_agent")
        assert result["success"] is True
        assert result["source"] == "memory"
        assert seeded_collection.count() == 3

    def test_kg_delete_entity_not_found(
        self, monkeypatch, config, palace_path, seeded_collection, kg
    ):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_kg_delete_entity

        result = tool_kg_delete_entity("record_nonexistent", agent="test_agent")
        assert result["success"] is False

    def test_kg_delete_entity_marks_sqlite_status_deleted(
        self, monkeypatch, config, palace_path, seeded_collection, kg
    ):
        """Regression test for the 2026-04-25 prune-anomaly bug.

        Before the fix, tool_kg_delete_entity removed the row from chroma
        but never updated the SQLite entities table, leaving status='active'
        and downstream readers (get_entity, list_declared) treating the
        entity as still present.

        Strategy: piggy-back on the seeded_collection record (chroma side
        already populated), seed a matching entities row by hand, then
        verify tool_kg_delete_entity flips its status to 'deleted'.
        """
        _patch_mcp_server(monkeypatch, config, kg)

        target_id = "record_proj_backend_aaa"
        # Seed the SQLite entities row. seeded_collection only writes
        # the chroma side; the entities table is empty for this id.
        conn = kg._conn()
        conn.execute(
            "INSERT INTO entities (id, name, type, kind, status, description, "
            "importance, last_touched) "
            "VALUES (?, ?, 'unknown', 'record', 'active', 'doomed content', 3, "
            "'2026-04-25T00:00:00')",
            (target_id, target_id),
        )
        conn.commit()

        from mempalace.mcp_server import tool_kg_delete_entity

        result = tool_kg_delete_entity(target_id, agent="test_agent")
        assert result["success"] is True, result

        # The entities row must still exist (audit-only soft delete) and
        # carry status='deleted'.
        row = conn.execute("SELECT status FROM entities WHERE id=?", (target_id,)).fetchone()
        assert row is not None, "entity row should still exist for audit"
        assert row["status"] == "deleted", (
            f"expected status='deleted', got status={row['status']!r}. "
            "Pre-fix bug: chroma deleted but SQLite row left status='active'."
        )

    # test_check_duplicate removed: tool_check_duplicate deleted.
    # Dedup is embedded in _add_memory_internal (called by
    # kg_declare_entity kind='record').


# ── KG Tools ────────────────────────────────────────────────────────────


class TestKGTools:
    def test_kg_add(self, monkeypatch, config, palace_path, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_kg_add, _STATE

        # Must declare entities AND predicate before using kg_add
        _STATE.declared_entities.add("alice")
        _STATE.declared_entities.add("coffee")
        _STATE.declared_entities.add("likes")
        kg.add_entity("Alice", kind="entity", description="A person named Alice")
        kg.add_entity("coffee", kind="entity", description="The beverage coffee")
        kg.add_entity(
            "likes", kind="predicate", description="Subject enjoys or has preference for object"
        )

        result = tool_kg_add(
            subject="Alice",
            predicate="likes",
            object="coffee",
            valid_from="2025-01-01",
            context={
                "queries": ["Alice likes coffee", "preference for coffee beverage"],
                "keywords": ["alice", "coffee", "likes"],
                "entities": ["Alice", "coffee"],
            },
            agent="test_agent",
            statement="Alice likes coffee.",
        )
        assert result["success"] is True, result
        # Legacy persist_context is retired — creation_context_id on
        # triples is no longer populated by the old path. The context-
        # as-entity substrate carries provenance at the entity layer via
        # created_under; triples store the active_context_id in their
        # row's creation_context_id only when an active context exists.
        triple_id = result["triple_id"]
        conn = kg._conn()
        row = conn.execute(
            "SELECT creation_context_id FROM triples WHERE id=?", (triple_id,)
        ).fetchone()
        assert row is not None

    def test_kg_query(self, monkeypatch, config, palace_path, seeded_kg):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_query

        result = tool_kg_query(entity="Max")
        assert result["count"] > 0

    def test_kg_invalidate(self, monkeypatch, config, palace_path, seeded_kg):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_invalidate

        result = tool_kg_invalidate(
            subject="Max",
            predicate="does",
            object="chess",
            ended="2026-03-01",
            agent="test_agent",
        )
        assert result["success"] is True

    def test_kg_timeline(self, monkeypatch, config, palace_path, seeded_kg):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_timeline

        result = tool_kg_timeline(entity="Alice")
        assert result["count"] > 0

    def test_kg_stats(self, monkeypatch, config, palace_path, seeded_kg):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_stats

        result = tool_kg_stats()
        assert result["entities"] >= 4


# ── Diary Tools ─────────────────────────────────────────────────────────


class TestDiaryTools:
    def test_diary_write_and_read(self, monkeypatch, config, palace_path, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        _client, _col = _get_collection(palace_path, create=True)
        del _client
        from mempalace.mcp_server import tool_diary_write, tool_diary_read

        w = tool_diary_write(
            agent_name="TestAgent",
            entry="Today we discussed authentication patterns.",
            topic="architecture",
        )
        assert w["success"] is True

        r = tool_diary_read(agent_name="TestAgent")
        assert r["total"] == 1
        assert r["entries"][0]["topic"] == "architecture"
        assert "authentication" in r["entries"][0]["content"]

    def test_diary_read_empty(self, monkeypatch, config, palace_path, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        _client, _col = _get_collection(palace_path, create=True)
        del _client
        from mempalace.mcp_server import tool_diary_read

        r = tool_diary_read(agent_name="Nobody")
        assert r["entries"] == []
