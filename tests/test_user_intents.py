"""
test_user_intents.py — Slice B-1 unit tests for tool_declare_user_intents.

Slice B-1 ships:
  * pending_user_messages persistence helpers in hooks_cli (read /
    append / clear) — disk-backed per-session JSON queue.
  * tool_declare_user_intents handler in intent.py — validates pending
    coverage, mints user_message records (kind='record'), runs
    context_lookup_or_create per declared user-context, returns memories
    per context, clears pending.
  * MCP schema registration in mcp_server.py.

Slice B-2 (next commit) wires the UserPromptSubmit hook to write
pending entries and the PreToolUse hook to block non-allowed tools
while pending > 0. Slice B-3 adds optional cause_id on declare_intent
+ finalize coverage rule.

These tests exercise the tool in isolation by writing the pending
file directly (simulating what UserPromptSubmit will do once Slice B-2
lands). Validation rules — coverage, unknown ids, no_intent proof —
are locked here so Slice B-2's hook write doesn't drift.

Grounding: STITCH (arXiv:2601.10702), Agent-Sentry (arXiv:2603.22868),
BDI (Rao & Georgeff 1995). See diary
diary_ga_agent_user_intent_tier_design_locked_2026_04_24.
"""

from __future__ import annotations

import json


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _bootstrap(monkeypatch, tmp_path):
    """Build a hermetic mempalace state for a single test.

    Mirrors the pattern in test_intent_system.py — point STATE_DIR at
    tmp_path, reset _STATE.session_id / active_intent / kg, return the
    mcp_server module so tests can poke handler shims directly.
    """
    from mempalace import hooks_cli, intent, mcp_server
    from mempalace.knowledge_graph import KnowledgeGraph

    monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)
    monkeypatch.setattr(mcp_server, "_INTENT_STATE_DIR", tmp_path)
    monkeypatch.setattr(intent, "_INTENT_STATE_DIR", tmp_path, raising=False)

    db = tmp_path / "palace.db"
    kg = KnowledgeGraph(str(db))
    monkeypatch.setattr(mcp_server._STATE, "kg", kg)
    monkeypatch.setattr(mcp_server._STATE, "session_id", "test-sid-b1")
    monkeypatch.setattr(mcp_server._STATE, "active_intent", None)

    # Declare an agent so added_by validation passes.
    kg.add_entity("ga_agent", kind="entity", description="ga test agent", importance=3)
    kg.add_triple("ga_agent", "is_a", "agent")

    return mcp_server, kg


def _write_pending(hooks_cli_mod, sid, messages):
    """Simulate UserPromptSubmit having written a pending queue."""
    from pathlib import Path

    path = Path(hooks_cli_mod.STATE_DIR) / f"pending_user_messages_{sid}.json"
    path.write_text(
        json.dumps({"session_id": sid, "messages": messages}, indent=2),
        encoding="utf-8",
    )
    return path


def _msg(idx, text="hello world"):
    """Build a pending message dict with a deterministic id."""
    from mempalace import hooks_cli

    mid = hooks_cli._make_user_message_id("test-sid-b1", idx, text)
    return {
        "id": mid,
        "text": text,
        "turn_idx": idx,
        "ts": f"2026-04-26T00:00:0{idx}Z",
    }


# ─────────────────────────────────────────────────────────────────────
# Pending-file helpers
# ─────────────────────────────────────────────────────────────────────


class TestPendingHelpers:
    """The disk-backed pending queue underpins everything else."""

    def test_read_returns_empty_for_missing_file(self, tmp_path, monkeypatch):
        from mempalace import hooks_cli

        monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)
        assert hooks_cli._read_pending_user_messages("sid-none") == []

    def test_append_then_read_roundtrips(self, tmp_path, monkeypatch):
        from mempalace import hooks_cli

        monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)
        m = {"id": "msg_aaa", "text": "hi", "turn_idx": 1, "ts": "2026-04-26T00:00:00Z"}
        assert hooks_cli._append_pending_user_message("sid-1", m) is True
        out = hooks_cli._read_pending_user_messages("sid-1")
        assert out == [m]

    def test_append_is_idempotent_on_duplicate_id(self, tmp_path, monkeypatch):
        from mempalace import hooks_cli

        monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)
        m = {"id": "msg_aaa", "text": "hi", "turn_idx": 1, "ts": "t"}
        hooks_cli._append_pending_user_message("sid-2", m)
        hooks_cli._append_pending_user_message("sid-2", m)
        assert hooks_cli._read_pending_user_messages("sid-2") == [m]

    def test_clear_drains_and_returns_count(self, tmp_path, monkeypatch):
        from mempalace import hooks_cli

        monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)
        for i in range(3):
            hooks_cli._append_pending_user_message(
                "sid-3", {"id": f"msg_{i}", "text": "x", "turn_idx": i, "ts": "t"}
            )
        n = hooks_cli._clear_pending_user_messages("sid-3")
        assert n == 3
        assert hooks_cli._read_pending_user_messages("sid-3") == []

    def test_make_user_message_id_is_deterministic_per_input(self, monkeypatch):
        """Same (session, turn, text) yields the same digest prefix.

        Note: the suffix is a nanosecond timestamp so full ids differ
        across calls — only the digest portion needs to be stable. Lock
        that contract: the prefix up to and including the second
        underscore must match for identical inputs."""
        from mempalace import hooks_cli

        a = hooks_cli._make_user_message_id("sid-x", 5, "fix the auth bug")
        b = hooks_cli._make_user_message_id("sid-x", 5, "fix the auth bug")
        # Format: msg_<sha12>_<ns>
        assert a.split("_")[1] == b.split("_")[1]
        # Different inputs differ.
        c = hooks_cli._make_user_message_id("sid-x", 5, "different prompt")
        assert a.split("_")[1] != c.split("_")[1]


# ─────────────────────────────────────────────────────────────────────
# tool_declare_user_intents validation
# ─────────────────────────────────────────────────────────────────────


class TestDeclareUserIntentsValidation:
    """Locked-in rejection rules. Each violation surfaces a specific
    error message agents can act on without re-reading docs."""

    def test_empty_contexts_rejected(self, tmp_path, monkeypatch):
        mcp_server, _kg = _bootstrap(monkeypatch, tmp_path)
        result = mcp_server.tool_declare_user_intents(contexts=[], agent="ga_agent")
        assert result["success"] is False
        assert "non-empty list" in result["error"]

    def test_missing_user_message_ids_rejected(self, tmp_path, monkeypatch):
        mcp_server, _kg = _bootstrap(monkeypatch, tmp_path)
        result = mcp_server.tool_declare_user_intents(
            contexts=[
                {
                    "context": {
                        "queries": ["fix auth bug"],
                        "keywords": ["auth", "bug"],
                        "entities": ["auth_service"],
                        "summary": {
                            "what": "fix auth bug",
                            "why": "user asked to debug 401s on login endpoint",
                        },
                    },
                    # user_message_ids missing
                },
            ],
            agent="ga_agent",
        )
        assert result["success"] is False
        assert "user_message_ids is required" in result["error"]

    def test_unknown_user_message_id_rejected(self, tmp_path, monkeypatch):
        mcp_server, _kg = _bootstrap(monkeypatch, tmp_path)
        # No pending file exists, so any referenced id is unknown.
        result = mcp_server.tool_declare_user_intents(
            contexts=[
                {
                    "context": {
                        "queries": ["fix auth bug"],
                        "keywords": ["auth", "bug"],
                        "entities": ["auth_service"],
                        "summary": {
                            "what": "fix auth bug",
                            "why": "user asked to debug 401s on the login endpoint",
                        },
                    },
                    "user_message_ids": ["msg_unknown_abc"],
                },
            ],
            agent="ga_agent",
        )
        assert result["success"] is False
        assert "not in the pending user_message queue" in result["error"]

    def test_pending_not_fully_covered_rejected(self, tmp_path, monkeypatch):
        from mempalace import hooks_cli

        mcp_server, _kg = _bootstrap(monkeypatch, tmp_path)
        # Two pending messages; declare a context covering only one.
        m0 = _msg(0, "first prompt about auth")
        m1 = _msg(1, "second prompt about rate limiting")
        _write_pending(hooks_cli, "test-sid-b1", [m0, m1])

        result = mcp_server.tool_declare_user_intents(
            contexts=[
                {
                    "context": {
                        "queries": ["fix auth bug"],
                        "keywords": ["auth", "bug"],
                        "entities": ["auth_service"],
                        "summary": {
                            "what": "fix auth bug",
                            "why": "user asked to debug 401s on the login endpoint",
                        },
                    },
                    "user_message_ids": [m0["id"]],
                    # m1 unattributed → coverage failure
                },
            ],
            agent="ga_agent",
        )
        assert result["success"] is False
        assert "not covered" in result["error"]
        assert m1["id"] in result["missing_user_message_ids"]

    def test_no_intent_without_clarification_rejected(self, tmp_path, monkeypatch):
        from mempalace import hooks_cli

        mcp_server, _kg = _bootstrap(monkeypatch, tmp_path)
        m = _msg(0, "thanks")
        _write_pending(hooks_cli, "test-sid-b1", [m])

        result = mcp_server.tool_declare_user_intents(
            contexts=[
                {
                    "context": {
                        "queries": ["thanks ack"],
                        "keywords": ["ack", "thanks"],
                        "entities": ["ga_agent"],
                        "summary": {
                            "what": "trivial ack",
                            "why": "user just said thanks; no action required from agent",
                        },
                    },
                    "user_message_ids": [m["id"]],
                    "no_intent": True,
                    # no_intent_clarified_with_user not set → reject
                },
            ],
            agent="ga_agent",
        )
        assert result["success"] is False
        assert "no_intent_clarified_with_user" in result["error"]


# ─────────────────────────────────────────────────────────────────────
# Happy-path round trip
# ─────────────────────────────────────────────────────────────────────


class TestDeclareUserIntentsHappyPath:
    """Pending messages → context creation → record minting → pending cleared."""

    def test_round_trip_creates_context_and_clears_pending(self, tmp_path, monkeypatch):
        from mempalace import hooks_cli

        mcp_server, kg = _bootstrap(monkeypatch, tmp_path)
        m = _msg(0, "fix the auth bug")
        _write_pending(hooks_cli, "test-sid-b1", [m])

        result = mcp_server.tool_declare_user_intents(
            contexts=[
                {
                    "context": {
                        "queries": ["fix auth bug"],
                        "keywords": ["auth", "bug"],
                        "entities": ["ga_agent"],
                        "summary": {
                            "what": "fix auth bug",
                            "why": "user asked to debug 401s on the login endpoint",
                        },
                    },
                    "user_message_ids": [m["id"]],
                },
            ],
            agent="ga_agent",
        )
        assert result["success"] is True, result
        assert len(result["contexts"]) == 1
        block = result["contexts"][0]
        assert block["ctx_id"], "context id should be minted"
        assert isinstance(block["reused"], bool)
        assert result["cleared_pending_count"] == 1
        assert m["id"] in result["minted_user_message_ids"]

        # user_message record exists in kg with kind='record'.
        rec = kg.get_entity(m["id"])
        assert rec is not None
        assert rec.get("kind") == "record"

        # Pending queue is empty after clear.
        assert hooks_cli._read_pending_user_messages("test-sid-b1") == []

    def test_round_trip_no_intent_with_clarification_accepted(self, tmp_path, monkeypatch):
        from mempalace import hooks_cli

        mcp_server, kg = _bootstrap(monkeypatch, tmp_path)
        m = _msg(0, "thanks")
        _write_pending(hooks_cli, "test-sid-b1", [m])

        result = mcp_server.tool_declare_user_intents(
            contexts=[
                {
                    "context": {
                        "queries": ["trivial ack thanks"],
                        "keywords": ["ack", "thanks"],
                        "entities": ["ga_agent"],
                        "summary": {
                            "what": "trivial user ack",
                            "why": "user said thanks; agent confirmed via AskUserQuestion that no action is required",
                        },
                    },
                    "user_message_ids": [m["id"]],
                    "no_intent": True,
                    "no_intent_clarified_with_user": True,
                },
            ],
            agent="ga_agent",
        )
        assert result["success"] is True, result
        assert result["contexts"][0].get("no_intent") is True
        assert result["cleared_pending_count"] == 1
