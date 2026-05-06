"""
test_no_cross_agent_fallbacks.py -- Invariant: no code path EVER writes
or reads a shared state file when session_id is empty.

Every pre-2026-04-19 deadlock traced back to the same shape: when a
tool call arrived without a real session_id, the codebase silently
substituted ``"default"`` or ``"unknown"`` for the file-name suffix.
Every agent running through that MCP server then read and wrote to
the SAME file -- one agent's resolve cleared state, the next agent's
tool call re-loaded the first agent's pending items as its own
blocker. Impossible to recover from.

Policy: NO CROSS-AGENT FALLBACK. If session_id is empty at the moment
a state-file operation is about to run, we skip the operation and
make the absence loud (log line, no-file-on-disk). The next tool
call under a real sid will pick up from a clean slate.

These tests lock that policy in.
"""

from __future__ import annotations

import io
import json

import pytest


# ═══════════════════════════════════════════════════════════════════════
#  hooks_cli sanitizer -- NO 'unknown' fallback
# ═══════════════════════════════════════════════════════════════════════


class TestSanitizerNoFallback:
    def test_hook_sanitizer_empty_stays_empty(self):
        from mempalace.hooks_cli import _sanitize_session_id

        assert _sanitize_session_id("") == ""
        assert _sanitize_session_id("!!!") == ""

    def test_server_sanitizer_empty_stays_empty(self):
        from mempalace.mcp_server import _sanitize_session_id

        assert _sanitize_session_id("") == ""
        assert _sanitize_session_id("!!!") == ""
        assert _sanitize_session_id(None) == ""


# ═══════════════════════════════════════════════════════════════════════
#  hooks_cli _append_trace -- skip on empty sid
# ═══════════════════════════════════════════════════════════════════════


class TestAppendTraceNoFallback:
    def test_empty_sid_does_not_create_default_trace_file(self, tmp_path, monkeypatch):
        from mempalace import hooks_cli

        monkeypatch.setattr(hooks_cli, "_TRACE_DIR", tmp_path)
        hooks_cli._append_trace("", "Read", {"file_path": "x"})
        # No execution_trace_*.jsonl, and specifically no
        # execution_trace_default.jsonl.
        files = list(tmp_path.glob("execution_trace_*.jsonl"))
        assert files == [], f"_append_trace leaked a trace file for empty sid: {files}"

    def test_real_sid_writes_its_own_trace(self, tmp_path, monkeypatch):
        from mempalace import hooks_cli

        monkeypatch.setattr(hooks_cli, "_TRACE_DIR", tmp_path)
        hooks_cli._append_trace("abc-123", "Read", {"file_path": "/tmp/x"})
        assert (tmp_path / "execution_trace_abc-123.jsonl").is_file()


# ═══════════════════════════════════════════════════════════════════════
#  hooks_cli _read_active_intent -- no default fallback
# ═══════════════════════════════════════════════════════════════════════


class TestReadActiveIntentNoFallback:
    def test_empty_sid_returns_none_even_if_default_file_exists(self, tmp_path, monkeypatch):
        """Someone else's shared default file MUST NOT be accepted as
        our active intent when we call with empty sid."""
        from mempalace import hooks_cli

        monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)
        # Plant a decoy shared default file that used to be the fallback.
        (tmp_path / "active_intent_default.json").write_text(
            json.dumps(
                {
                    "intent_id": "intent_from_someone_else",
                    "intent_type": "research",
                    "session_id": "",
                    "pending_conflicts": [],
                }
            )
        )
        assert hooks_cli._read_active_intent("") is None
        assert hooks_cli._read_active_intent(None) is None

    def test_real_sid_misses_when_its_own_file_absent(self, tmp_path, monkeypatch):
        """Even if a shared default exists, a real sid whose OWN file is
        missing must return None -- never silently adopt default's
        intent."""
        from mempalace import hooks_cli

        monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)
        (tmp_path / "active_intent_default.json").write_text(
            json.dumps({"intent_id": "stranger_intent", "session_id": ""})
        )
        assert hooks_cli._read_active_intent("my-sid") is None


# ═══════════════════════════════════════════════════════════════════════
#  intent._intent_state_path -- None on empty sid (no file name synthesis)
# ═══════════════════════════════════════════════════════════════════════


class TestIntentStatePathNoFallback:
    def test_returns_none_when_empty_sid(self, tmp_path, monkeypatch):
        from mempalace import intent, mcp_server

        monkeypatch.setattr(mcp_server, "_INTENT_STATE_DIR", tmp_path)
        monkeypatch.setattr(mcp_server._STATE, "session_id", "")
        assert intent._intent_state_path() is None

    def test_returns_sid_named_path_when_sid_set(self, tmp_path, monkeypatch):
        from mempalace import intent, mcp_server

        monkeypatch.setattr(mcp_server, "_INTENT_STATE_DIR", tmp_path)
        monkeypatch.setattr(mcp_server._STATE, "session_id", "my-real-sid")
        p = intent._intent_state_path()
        assert p is not None
        assert p.name == "active_intent_my-real-sid.json"


# ═══════════════════════════════════════════════════════════════════════
#  intent._persist_active_intent -- no-op on empty sid
# ═══════════════════════════════════════════════════════════════════════


class TestPersistNoFallback:
    def test_empty_sid_persist_is_noop_no_default_file(self, tmp_path, monkeypatch):
        from mempalace import intent, mcp_server

        monkeypatch.setattr(mcp_server, "_INTENT_STATE_DIR", tmp_path)
        monkeypatch.setattr(mcp_server._STATE, "session_id", "")
        monkeypatch.setattr(
            mcp_server._STATE,
            "active_intent",
            {
                "intent_id": "would_be_stranded",
                "intent_type": "research",
                "slots": {},
                "effective_permissions": [],
                "session_id": "",
            },
        )
        intent._persist_active_intent()

        # No file of any kind got written -- especially not
        # active_intent_default.json.
        files = list(tmp_path.glob("active_intent_*.json"))
        assert files == [], f"persist leaked a state file despite empty sid: {files}"

    def test_empty_sid_persist_with_pending_conflicts_is_noop(self, tmp_path, monkeypatch):
        from mempalace import intent, mcp_server

        monkeypatch.setattr(mcp_server, "_INTENT_STATE_DIR", tmp_path)
        monkeypatch.setattr(mcp_server._STATE, "session_id", "")
        monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
        monkeypatch.setattr(
            mcp_server._STATE,
            "pending_conflicts",
            [{"id": "c1", "subject": "a", "predicate": "p", "object": "b"}],
        )
        intent._persist_active_intent()

        files = list(tmp_path.glob("active_intent_*.json"))
        assert files == [], f"pending-only persist leaked a shared file: {files}"


# ═══════════════════════════════════════════════════════════════════════
#  mcp_server _load_pending_conflicts_from_disk -- [] on empty sid
# ═══════════════════════════════════════════════════════════════════════


class TestLoadPendingNoFallback:
    def test_empty_sid_returns_empty_even_if_default_has_data(self, tmp_path, monkeypatch):
        from mempalace import mcp_server

        monkeypatch.setattr(mcp_server, "_INTENT_STATE_DIR", tmp_path)
        monkeypatch.setattr(mcp_server._STATE, "session_id", "")
        (tmp_path / "active_intent_default.json").write_text(
            json.dumps({"pending_conflicts": [{"id": "stranger_pending"}]})
        )
        assert mcp_server._load_pending_conflicts_from_disk() == []


# ═══════════════════════════════════════════════════════════════════════
#  handle_request -- no sid switch when sid is missing, no default file
#  synthesis anywhere in the pipeline
# ═══════════════════════════════════════════════════════════════════════


class TestHandleRequestNoFallback:
    def test_tool_call_without_sessionId_does_not_create_default_file(self, tmp_path, monkeypatch):
        """Full dispatch through handle_request with NO sessionId in
        tool_args must NOT cause any file-system write under 'default'."""
        from mempalace import mcp_server
        from mempalace.config import MempalaceConfig
        from mempalace.knowledge_graph import KnowledgeGraph
        import chromadb

        palace = tmp_path / "palace"
        palace.mkdir()
        cfg_dir = tmp_path / "cfg"
        cfg_dir.mkdir()
        (cfg_dir / "config.json").write_text(json.dumps({"palace_path": str(palace)}))
        cfg = MempalaceConfig(config_dir=str(cfg_dir))
        kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
        # Minimal KG to avoid crashes on read-only tool.
        client = chromadb.PersistentClient(path=str(palace))
        client.get_or_create_collection("mempalace_records")
        client.get_or_create_collection("mempalace_entities")
        del client

        state_dir = tmp_path / "hook_state"
        state_dir.mkdir()
        monkeypatch.setattr(mcp_server, "_INTENT_STATE_DIR", state_dir)
        monkeypatch.setattr(mcp_server._STATE, "kg", kg)
        monkeypatch.setattr(mcp_server._STATE, "config", cfg)
        monkeypatch.setattr(mcp_server._STATE, "session_id", "")
        monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
        monkeypatch.setattr(mcp_server._STATE, "pending_conflicts", None)

        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "mempalace_active_intent", "arguments": {}},
        }
        mcp_server.handle_request(req)

        # Nothing called 'default' anywhere.
        leaked = list(state_dir.glob("*default*"))
        assert leaked == [], f"handle_request synthesized a default-named file: {leaked}"
        # And _STATE.session_id did not magically become 'unknown' or 'default'.
        assert mcp_server._STATE.session_id in ("", None)


# ═══════════════════════════════════════════════════════════════════════
#  Full grep-level invariant: the string 'default' does not appear as a
#  session_id file-name suffix anywhere in production code.
# ═══════════════════════════════════════════════════════════════════════


class TestSourceGrep:
    """Prevents regressions by reading production code, stripping
    comments and docstrings, and asserting the forbidden code-level
    patterns do not appear in any executable expression.

    Policy: No active expression may substitute "default" or "unknown"
    for an empty session_id. Comments describing the policy (including
    this docstring) are allowed.
    """

    @staticmethod
    def _code_only(rel: str) -> str:
        """Return the production code stripped of comments and string
        literals, so grep-based assertions don't match prose that
        describes what NOT to do."""
        import tokenize
        from pathlib import Path

        root = Path(__file__).parent.parent / "mempalace"
        src = (root / rel).read_text(encoding="utf-8")
        out = []
        try:
            tokens = tokenize.generate_tokens(io.StringIO(src).readline)
            for tok in tokens:
                ttype = tok.type
                if ttype in (tokenize.COMMENT, tokenize.STRING):
                    # Replace with a neutral marker so grep hits in
                    # f-strings / formatted code still get caught --
                    # but plain comments and docstrings do not.
                    #
                    # f-string path-format strings like
                    # f"active_intent_{_STATE.session_id or 'default'}.json"
                    # are emitted as a single STRING token; this means
                    # such code-level patterns slip through this filter.
                    # The in-process tests above (TestIntentStatePathNoFallback,
                    # TestPersistNoFallback, TestHandleRequestNoFallback,
                    # TestLoadPendingNoFallback, TestReadActiveIntentNoFallback)
                    # catch those cases end-to-end, so this static check
                    # is a belt-and-suspenders line-level scan.
                    continue
                out.append(tok.string)
        except tokenize.TokenizeError:
            return src  # Fail open: return raw source.
        return " ".join(out)

    def test_no_active_code_uses_or_default_outside_strings(self):
        for rel in ("hooks_cli.py", "intent.py", "mcp_server.py"):
            code = self._code_only(rel)
            # These shapes would only be present as real code expressions.
            assert 'or "default"' not in code, f"{rel}: active 'or default' expression"
            assert "or 'default'" not in code, f"{rel}: active 'or default' expression"

    def test_no_active_code_synthesizes_unknown_for_session_id(self):
        for rel in ("hooks_cli.py", "intent.py", "mcp_server.py"):
            code = self._code_only(rel)
            assert 'or "unknown"' not in code, (
                f"{rel}: active 'or unknown' expression (likely sid synth)"
            )
            assert "or 'unknown'" not in code, (
                f"{rel}: active 'or unknown' expression (likely sid synth)"
            )


# ═══════════════════════════════════════════════════════════════════════
#  _require_sid -- state-writing tools error LOUDLY on empty sid
# ═══════════════════════════════════════════════════════════════════════


class TestRequireSidFailsLoud:
    """Every state-writing mempalace tool must REFUSE on empty sid with
    a clear error pointing at the root cause (hook didn't inject
    sessionId), NOT silently no-op. Silent skip was the 2026-04-19
    deadlock's quiet amplifier -- the agent thought a resolve succeeded
    when nothing happened. Now the agent sees the problem immediately
    and fixes the hook wiring."""

    @pytest.fixture
    def empty_sid_mcp(self, monkeypatch):
        from mempalace import mcp_server

        monkeypatch.setattr(mcp_server._STATE, "session_id", "")
        return mcp_server

    def _assert_sid_error(self, result):
        """A _require_sid failure has shape {success: False, error: str}
        with the error string naming session_id."""
        assert isinstance(result, dict), result
        assert result.get("success") is False, result
        assert "session_id" in result.get("error", ""), result

    def test_kg_add_refuses(self, empty_sid_mcp):
        r = empty_sid_mcp.tool_kg_add(
            subject="a",
            predicate="b",
            object="c",
            context={
                "queries": ["q1", "q2"],
                "keywords": ["k1", "k2"],
                "entities": ["a"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="ga_agent",
        )
        self._assert_sid_error(r)

    def test_kg_declare_entity_refuses(self, empty_sid_mcp):
        r = empty_sid_mcp.tool_kg_declare_entity(
            name="x",
            kind="entity",
            importance=3,
            context={
                "queries": ["q1", "q2"],
                "keywords": ["k1", "k2"],
                "entities": ["x"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            added_by="ga_agent",
        )
        self._assert_sid_error(r)

    def test_kg_invalidate_refuses(self, empty_sid_mcp):
        r = empty_sid_mcp.tool_kg_invalidate(
            subject="a", predicate="b", object="c", agent="ga_agent"
        )
        self._assert_sid_error(r)

    def test_kg_delete_entity_refuses(self, empty_sid_mcp):
        r = empty_sid_mcp.tool_kg_delete_entity(entity="x", agent="ga_agent")
        self._assert_sid_error(r)

    def test_resolve_conflicts_refuses(self, empty_sid_mcp):
        r = empty_sid_mcp.tool_resolve_conflicts(actions=[], agent="ga_agent")
        self._assert_sid_error(r)

    def test_diary_write_refuses(self, empty_sid_mcp):
        r = empty_sid_mcp.tool_diary_write(
            agent_name="ga_agent",
            entry="long enough entry text for the write",
        )
        self._assert_sid_error(r)

    def test_declare_intent_refuses(self, empty_sid_mcp):
        r = empty_sid_mcp.tool_declare_intent(
            intent_type="research",
            slots={"subject": ["x"]},
            context={
                "queries": ["q1", "q2"],
                "keywords": ["k1", "k2"],
                "entities": ["x"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="ga_agent",
            budget={"Read": 5},
        )
        self._assert_sid_error(r)

    def test_finalize_intent_refuses(self, empty_sid_mcp):
        r = empty_sid_mcp.tool_finalize_intent(
            agent="ga_agent",
            slug="x",
            outcome="abandoned",
            content="long enough content for the finalize call right here",
            summary={
                "what": "test fixture record",
                "why": "long enough summary for the finalize call right here",
                "scope": "tests",
            },
            memory_feedback=[],
        )
        self._assert_sid_error(r)

    def test_extend_intent_refuses(self, empty_sid_mcp):
        r = empty_sid_mcp.tool_extend_intent(budget={"Read": 1}, agent="ga_agent")
        self._assert_sid_error(r)


pytestmark = pytest.mark.integration
