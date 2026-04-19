"""
test_session_marker.py — Hook-to-server sid propagation via the
session-marker file, bypassing the unreliable ``updatedInput``.

Background (the live bug on 2026-04-19, round 2):

The PreToolUse hook was producing the correct JSON response with
``hookSpecificOutput.updatedInput.sessionId`` for every mempalace MCP
tool call. The server's dispatch loop was reading ``tool_args.sessionId``
to drive its session-scoping logic. But Claude Code, for MCP tool calls
specifically, does NOT propagate the hook's ``updatedInput`` mutations
through to the tool's arguments. The server therefore never saw the
injected sid, ``_STATE.session_id`` stayed at ``""`` forever, and every
``_persist_active_intent`` call wrote to ``active_intent_default.json``.

On-disk evidence: a single default-named file containing
``"session_id": ""`` that accumulated pending state from every logical
session — main and every subagent, collapsed into one queue.
``resolve_enrichments`` cleared a pending item, the next tool call came
in with a new sessionId that the server didn't apply, and the previous
queue's state (still on disk) would reappear as "phantom pending" on the
next declare_intent. Reproduced the user's deadlock exactly.

Fix: the hook now writes the effective sid to a well-known marker file
on every PreToolUse. The server reads that file as the AUTHORITATIVE
sid source, independent of ``tool_args.sessionId``. The legacy
``tool_args.sessionId`` path is still accepted as a fallback for
non-hook environments.

These tests pin the contract:

  - The hook writes the marker on every invocation, for any tool
    (mempalace MCP or otherwise).
  - The marker is atomic — a partial read never yields torn JSON.
  - Subagent calls write the composite sid.
  - ``_read_hook_session_marker`` returns the sanitized effective sid.
  - A legitimate pre-fix default state file's pending items are
    migrated INTO the new sid's file on the first session switch,
    never silently dropped.
  - Missing / corrupted marker gracefully degrades to empty sid.
"""

from __future__ import annotations

import io
import json
from unittest.mock import patch

import pytest


# ═══════════════════════════════════════════════════════════════════════
#  Hook-side: _write_session_marker + hook_pretooluse integration
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def isolated_hook_dir(tmp_path, monkeypatch):
    """Point the hook's STATE_DIR + _TRACE_DIR at a tmp location."""
    from mempalace import hooks_cli

    monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)
    monkeypatch.setattr(hooks_cli, "_TRACE_DIR", tmp_path)
    monkeypatch.setattr(hooks_cli, "INTENT_STATE_DIR", tmp_path)
    monkeypatch.setattr(hooks_cli, "_SESSION_MARKER_FILE", tmp_path / "current_session.json")
    return tmp_path


def _run_pretooluse(data):
    from mempalace.hooks_cli import hook_pretooluse

    buf = io.StringIO()
    with patch("sys.stdout", buf):
        hook_pretooluse(data, "claude-code")
    return buf.getvalue()


class TestHookWritesMarker:
    def test_marker_written_for_mcp_call(self, isolated_hook_dir):
        _run_pretooluse(
            {
                "session_id": "parent-abc",
                "tool_name": "mcp__plugin_3_0_14_mempalace__mempalace_kg_stats",
                "tool_input": {},
            }
        )
        marker = isolated_hook_dir / "current_session.json"
        assert marker.is_file()
        data = json.loads(marker.read_text())
        assert data["effective_sid"] == "parent-abc"
        assert data["base_session_id"] == "parent-abc"
        assert data["agent_id"] == ""
        assert "written_at" in data

    def test_marker_written_for_non_mcp_call(self, isolated_hook_dir):
        """Even when the tool is gated (no active intent → deny), the
        marker must still be written so the server sees the current sid
        on the NEXT mempalace call."""
        _run_pretooluse(
            {
                "session_id": "parent-abc",
                "tool_name": "Read",
                "tool_input": {"file_path": "/tmp/foo"},
            }
        )
        marker = isolated_hook_dir / "current_session.json"
        assert marker.is_file()
        assert json.loads(marker.read_text())["effective_sid"] == "parent-abc"

    def test_marker_composite_sid_for_subagent(self, isolated_hook_dir):
        _run_pretooluse(
            {
                "session_id": "parent-abc",
                "agent_id": "explore-42",
                "agent_type": "Explore",
                "tool_name": "mcp__plugin_3_0_14_mempalace__mempalace_kg_stats",
                "tool_input": {},
            }
        )
        data = json.loads((isolated_hook_dir / "current_session.json").read_text())
        assert data["effective_sid"] == "parent-abc__sub_explore-42"
        assert data["agent_id"] == "explore-42"

    def test_marker_overwritten_on_each_call(self, isolated_hook_dir):
        """The marker is the CURRENT sid — newer invocations overwrite it."""
        _run_pretooluse(
            {
                "session_id": "first",
                "tool_name": "mcp__plugin_3_0_14_mempalace__mempalace_kg_stats",
                "tool_input": {},
            }
        )
        _run_pretooluse(
            {
                "session_id": "second",
                "tool_name": "mcp__plugin_3_0_14_mempalace__mempalace_kg_stats",
                "tool_input": {},
            }
        )
        data = json.loads((isolated_hook_dir / "current_session.json").read_text())
        assert data["effective_sid"] == "second"

    def test_marker_skipped_when_session_id_missing(self, isolated_hook_dir):
        """No session_id in payload → no marker write. The server's legacy
        fallback is exercised."""
        _run_pretooluse(
            {
                "session_id": "",
                "tool_name": "mcp__plugin_3_0_14_mempalace__mempalace_kg_stats",
                "tool_input": {},
            }
        )
        marker = isolated_hook_dir / "current_session.json"
        assert not marker.is_file()

    def test_marker_write_is_atomic(self, isolated_hook_dir):
        """Use the tmp-rename pattern — never leaves a torn file that a
        concurrent reader could see mid-write."""
        from mempalace.hooks_cli import _write_session_marker

        _write_session_marker("abc", "abc", "")
        marker = isolated_hook_dir / "current_session.json"
        tmp = isolated_hook_dir / "current_session.json.tmp"
        assert marker.is_file()
        # The tmp should not linger after a normal write.
        assert not tmp.is_file()
        # Re-parseable.
        assert json.loads(marker.read_text())["effective_sid"] == "abc"


# ═══════════════════════════════════════════════════════════════════════
#  Server-side: _read_hook_session_marker
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def isolated_server(tmp_path, monkeypatch):
    from mempalace import mcp_server

    monkeypatch.setattr(mcp_server, "_INTENT_STATE_DIR", tmp_path)
    monkeypatch.setattr(mcp_server._STATE, "session_id", "")
    return mcp_server, tmp_path


class TestServerReadsMarker:
    def test_missing_marker_returns_empty(self, isolated_server):
        mcp, _ = isolated_server
        assert mcp._read_hook_session_marker() == ""

    def test_valid_marker_returns_sanitized_sid(self, isolated_server):
        mcp, dir_ = isolated_server
        (dir_ / "current_session.json").write_text(
            json.dumps({"effective_sid": "abc-123", "base_session_id": "abc-123"})
        )
        assert mcp._read_hook_session_marker() == "abc-123"

    def test_composite_sid_round_trips(self, isolated_server):
        mcp, dir_ = isolated_server
        (dir_ / "current_session.json").write_text(
            json.dumps(
                {
                    "effective_sid": "parent__sub_child",
                    "base_session_id": "parent",
                    "agent_id": "child",
                }
            )
        )
        assert mcp._read_hook_session_marker() == "parent__sub_child"

    def test_malicious_sid_gets_sanitized(self, isolated_server):
        mcp, dir_ = isolated_server
        (dir_ / "current_session.json").write_text(
            json.dumps({"effective_sid": "../../etc/passwd"})
        )
        sid = mcp._read_hook_session_marker()
        assert "/" not in sid
        assert ".." not in sid
        assert sid == "etcpasswd"

    def test_corrupted_marker_returns_empty(self, isolated_server):
        mcp, dir_ = isolated_server
        (dir_ / "current_session.json").write_text("{ not valid json")
        assert mcp._read_hook_session_marker() == ""

    def test_marker_missing_effective_sid_key_returns_empty(self, isolated_server):
        mcp, dir_ = isolated_server
        (dir_ / "current_session.json").write_text(json.dumps({"base_session_id": "x"}))
        assert mcp._read_hook_session_marker() == ""


# ═══════════════════════════════════════════════════════════════════════
#  Integration: dispatch uses marker as authoritative sid source
# ═══════════════════════════════════════════════════════════════════════


def _seed_minimal(mcp, palace_path, kg_path):
    """Bare minimum KG + Chroma state so handle_request + tool_active_intent
    don't crash on missing fixtures."""
    import chromadb
    from mempalace.knowledge_graph import KnowledgeGraph
    from mempalace.config import MempalaceConfig

    client = chromadb.PersistentClient(path=palace_path)
    client.get_or_create_collection("mempalace_records")
    client.get_or_create_collection("mempalace_entities")
    del client
    kg = KnowledgeGraph(db_path=kg_path)
    cfg = MempalaceConfig.__new__(MempalaceConfig)
    cfg._palace_path = palace_path  # minimal shim

    mcp._STATE.kg = kg
    mcp._STATE.config = cfg
    return kg


class TestDispatchUsesMarker:
    def test_marker_sid_overrides_tool_args_sessionId(self, tmp_path, monkeypatch):
        """If marker says sid=A and tool_args.sessionId=B, marker wins."""
        from mempalace import mcp_server

        palace = tmp_path / "palace"
        palace.mkdir()
        monkeypatch.setattr(mcp_server, "_INTENT_STATE_DIR", tmp_path)
        monkeypatch.setattr(mcp_server._STATE, "session_id", "")
        monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
        monkeypatch.setattr(mcp_server._STATE, "pending_conflicts", None)
        monkeypatch.setattr(mcp_server._STATE, "pending_enrichments", None)
        monkeypatch.setattr(mcp_server._STATE, "session_state", {})
        _seed_minimal(mcp_server, str(palace), str(tmp_path / "kg.sqlite3"))

        # Write marker with sid "A".
        (tmp_path / "current_session.json").write_text(
            json.dumps({"effective_sid": "marker-sid-A"})
        )
        # Dispatch with tool_args.sessionId = "B" (legacy path).
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "mempalace_active_intent",
                "arguments": {"sessionId": "args-sid-B"},
            },
        }
        mcp_server.handle_request(req)
        assert mcp_server._STATE.session_id == "marker-sid-A", (
            "marker sid must take precedence over tool_args.sessionId"
        )

    def test_no_marker_falls_back_to_tool_args_sessionId(self, tmp_path, monkeypatch):
        """Legacy environments without the marker still work via tool_args."""
        from mempalace import mcp_server

        palace = tmp_path / "palace"
        palace.mkdir()
        monkeypatch.setattr(mcp_server, "_INTENT_STATE_DIR", tmp_path)
        monkeypatch.setattr(mcp_server._STATE, "session_id", "")
        monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
        monkeypatch.setattr(mcp_server._STATE, "pending_conflicts", None)
        monkeypatch.setattr(mcp_server._STATE, "pending_enrichments", None)
        monkeypatch.setattr(mcp_server._STATE, "session_state", {})
        _seed_minimal(mcp_server, str(palace), str(tmp_path / "kg.sqlite3"))
        # no marker file
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "mempalace_active_intent",
                "arguments": {"sessionId": "legacy-sid"},
            },
        }
        mcp_server.handle_request(req)
        assert mcp_server._STATE.session_id == "legacy-sid"

    def test_neither_marker_nor_args_keeps_empty_sid(self, tmp_path, monkeypatch):
        """Graceful degradation — no sid sources, no crash."""
        from mempalace import mcp_server

        palace = tmp_path / "palace"
        palace.mkdir()
        monkeypatch.setattr(mcp_server, "_INTENT_STATE_DIR", tmp_path)
        monkeypatch.setattr(mcp_server._STATE, "session_id", "")
        monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
        monkeypatch.setattr(mcp_server._STATE, "pending_conflicts", None)
        monkeypatch.setattr(mcp_server._STATE, "pending_enrichments", None)
        monkeypatch.setattr(mcp_server._STATE, "session_state", {})
        _seed_minimal(mcp_server, str(palace), str(tmp_path / "kg.sqlite3"))
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "mempalace_active_intent", "arguments": {}},
        }
        mcp_server.handle_request(req)
        # No crash; sid stays empty.
        assert mcp_server._STATE.session_id == ""


# ═══════════════════════════════════════════════════════════════════════
#  Default-file migration: stale pre-fix state migrates to real sid
# ═══════════════════════════════════════════════════════════════════════


class TestDefaultFileMigration:
    """Users running the broken pre-fix server accumulated state in
    ``active_intent_default.json`` instead of ``active_intent_<sid>.json``.
    When they upgrade and the first real tool call arrives, that stale
    state MUST migrate into the new sid's file — not get silently unlinked.
    """

    def test_pending_enrichments_migrate_from_default_to_sid_file(self, tmp_path, monkeypatch):
        from mempalace import mcp_server

        palace = tmp_path / "palace"
        palace.mkdir()
        monkeypatch.setattr(mcp_server, "_INTENT_STATE_DIR", tmp_path)
        monkeypatch.setattr(mcp_server._STATE, "session_id", "")
        monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
        monkeypatch.setattr(mcp_server._STATE, "pending_conflicts", None)
        monkeypatch.setattr(mcp_server._STATE, "pending_enrichments", None)
        monkeypatch.setattr(mcp_server._STATE, "session_state", {})
        _seed_minimal(mcp_server, str(palace), str(tmp_path / "kg.sqlite3"))

        # Seed the stale default file.
        (tmp_path / "active_intent_default.json").write_text(
            json.dumps(
                {
                    "intent_id": "",
                    "session_id": "",
                    "pending_conflicts": [],
                    "pending_enrichments": [
                        {
                            "id": "e_stale",
                            "from_entity": "a",
                            "to_entity": "b",
                            "reason": "stranded",
                        }
                    ],
                }
            )
        )
        # Write marker so dispatch knows the real sid.
        (tmp_path / "current_session.json").write_text(json.dumps({"effective_sid": "fresh-sid"}))
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "mempalace_active_intent", "arguments": {}},
        }
        mcp_server.handle_request(req)

        # Default file is gone.
        assert not (tmp_path / "active_intent_default.json").is_file()
        # Real-sid file exists and contains the migrated enrichment.
        real_file = tmp_path / "active_intent_fresh-sid.json"
        assert real_file.is_file()
        data = json.loads(real_file.read_text())
        assert data.get("pending_enrichments")
        assert data["pending_enrichments"][0]["id"] == "e_stale"

    def test_active_intent_migrates_from_default(self, tmp_path, monkeypatch):
        from mempalace import mcp_server

        palace = tmp_path / "palace"
        palace.mkdir()
        monkeypatch.setattr(mcp_server, "_INTENT_STATE_DIR", tmp_path)
        monkeypatch.setattr(mcp_server._STATE, "session_id", "")
        monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
        monkeypatch.setattr(mcp_server._STATE, "pending_conflicts", None)
        monkeypatch.setattr(mcp_server._STATE, "pending_enrichments", None)
        monkeypatch.setattr(mcp_server._STATE, "session_state", {})
        _seed_minimal(mcp_server, str(palace), str(tmp_path / "kg.sqlite3"))

        (tmp_path / "active_intent_default.json").write_text(
            json.dumps(
                {
                    "intent_id": "intent_stale_1",
                    "intent_type": "research",
                    "slots": {"subject": ["x"]},
                    "effective_permissions": ["Read(*)"],
                    "description": "stranded intent",
                    "agent": "test_agent",
                    "session_id": "",
                    "budget": {"Read": 5},
                    "used": {},
                    "pending_conflicts": [],
                    "pending_enrichments": [],
                }
            )
        )
        (tmp_path / "current_session.json").write_text(json.dumps({"effective_sid": "new-sid"}))
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "mempalace_active_intent", "arguments": {}},
        }
        mcp_server.handle_request(req)

        assert not (tmp_path / "active_intent_default.json").is_file()
        real = tmp_path / "active_intent_new-sid.json"
        assert real.is_file()
        data = json.loads(real.read_text())
        assert data["intent_id"] == "intent_stale_1"

    def test_no_default_file_is_a_noop(self, tmp_path, monkeypatch):
        """Clean-install scenario — no default file, nothing to migrate."""
        from mempalace import mcp_server

        palace = tmp_path / "palace"
        palace.mkdir()
        monkeypatch.setattr(mcp_server, "_INTENT_STATE_DIR", tmp_path)
        monkeypatch.setattr(mcp_server._STATE, "session_id", "")
        monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
        monkeypatch.setattr(mcp_server._STATE, "pending_conflicts", None)
        monkeypatch.setattr(mcp_server._STATE, "pending_enrichments", None)
        monkeypatch.setattr(mcp_server._STATE, "session_state", {})
        _seed_minimal(mcp_server, str(palace), str(tmp_path / "kg.sqlite3"))

        (tmp_path / "current_session.json").write_text(json.dumps({"effective_sid": "clean-sid"}))
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "mempalace_active_intent", "arguments": {}},
        }
        mcp_server.handle_request(req)
        # Nothing blew up; sid applied; no residual files.
        assert mcp_server._STATE.session_id == "clean-sid"
