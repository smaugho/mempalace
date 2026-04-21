"""
test_subagent_sid_isolation.py — Isolation of subagent state from parent
state via the composite session-id scheme.

Claude Code subagents (dispatched via the Task tool) make tool calls
inside the SAME MCP session as the top-level conversation — the
PreToolUse hook payload's ``session_id`` is identical for both. Without
further disambiguation, a subagent's ``kg_declare_entity`` call would
mutate the parent's ``_STATE.pending_enrichments`` and persist them to
the parent's active-intent file, trapping the parent with phantom
pending state it never created. This was the live failure mode on
2026-04-19.

Fix: the hook reads the ``agent_id`` field (present only when the
hook fires inside a subagent, unique per subagent invocation, stable
across every tool call in that invocation) and folds it into a
composite sid — ``<session>__sub_<agent_id>``. The server then
scopes its on-disk pending-state file and in-memory ``_STATE`` cache
by the composite sid, so subagent state lives in its own file and its
own cache slot.

These tests lock in:

  - The composite-sid format is deterministic for any (session_id,
    agent_id) pair.
  - The parent hook call emits the base sid; the subagent hook call
    emits a distinct composite sid.
  - Two DIFFERENT subagent invocations inside the same session emit
    DIFFERENT composite sids.
  - The server (via end-to-end round-trip) treats a composite sid like
    any other sid: pending state for a subagent's composite sid never
    contaminates the parent's base-sid state and vice versa.
"""

from __future__ import annotations

import io
import json
from unittest.mock import patch

import pytest

from mempalace.hooks_cli import _effective_session_id, _sanitize_session_id, hook_pretooluse


# ═══════════════════════════════════════════════════════════════════════
#  _effective_session_id — pure function
# ═══════════════════════════════════════════════════════════════════════


class TestEffectiveSessionId:
    def test_no_agent_id_returns_base_session(self):
        assert _effective_session_id({"session_id": "abc123"}) == "abc123"

    def test_empty_agent_id_returns_base_session(self):
        assert _effective_session_id({"session_id": "abc123", "agent_id": ""}) == "abc123"

    def test_none_agent_id_returns_base_session(self):
        assert _effective_session_id({"session_id": "abc123", "agent_id": None}) == "abc123"

    def test_with_agent_id_returns_composite(self):
        assert (
            _effective_session_id({"session_id": "abc123", "agent_id": "explore-7"})
            == "abc123__sub_explore-7"
        )

    def test_composite_is_sanitizer_safe(self):
        """Composite sid must pass through _sanitize_session_id unchanged."""
        composite = _effective_session_id({"session_id": "abc123", "agent_id": "a-b_c"})
        assert composite == "abc123__sub_a-b_c"
        assert _sanitize_session_id(composite) == composite

    def test_different_agent_ids_same_session_produce_different_sids(self):
        """Two subagents in the same conversation must not collide."""
        a = _effective_session_id({"session_id": "sess", "agent_id": "alpha"})
        b = _effective_session_id({"session_id": "sess", "agent_id": "beta"})
        assert a != b
        assert a == "sess__sub_alpha"
        assert b == "sess__sub_beta"

    def test_same_agent_id_different_sessions_differ(self):
        """Agent-id reuse across sessions still maps to distinct composite sids."""
        assert _effective_session_id(
            {"session_id": "s1", "agent_id": "x"}
        ) != _effective_session_id({"session_id": "s2", "agent_id": "x"})

    def test_agent_id_without_session_still_scoped(self):
        """Defensive: if base session_id is missing for some reason, we still
        emit a usable subagent-scoped sid so state isn't silently written to
        'unknown' or 'default'."""
        result = _effective_session_id({"session_id": "", "agent_id": "a1"})
        assert result.startswith("sub_")
        assert "a1" in result

    def test_malicious_agent_id_characters_are_stripped(self):
        """Path-traversal attempts in agent_id must not leak into the sid."""
        result = _effective_session_id({"session_id": "sess", "agent_id": "../../etc/passwd"})
        # The sanitizer drops slashes and dots.
        assert "/" not in result
        assert ".." not in result
        assert result == "sess__sub_etcpasswd"

    def test_non_string_agent_id_coerced(self):
        """Defensive against unusual payload shapes (numeric agent ids etc.)."""
        result = _effective_session_id({"session_id": "sess", "agent_id": 42})
        assert result == "sess__sub_42"


# ═══════════════════════════════════════════════════════════════════════
#  hook_pretooluse — subagent vs parent payload
# ═══════════════════════════════════════════════════════════════════════


def _capture_hook_output(data, harness="claude"):
    """Run hook_pretooluse and return the parsed JSON response."""
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        hook_pretooluse(data, harness)
    raw = buf.getvalue().strip()
    if not raw:
        return None
    return json.loads(raw)


class TestHookPreToolUseSubagentInjection:
    """The hook must inject the composite sid into the MCP tool input when
    the payload identifies a subagent call; it must inject the base sid
    (current behavior) for parent calls."""

    def test_parent_call_injects_base_sid(self):
        payload = {
            "session_id": "parent-sess",
            "tool_name": "mcp__plugin_3_0_14_mempalace__mempalace_kg_stats",
            "tool_input": {},
            # no agent_id
        }
        out = _capture_hook_output(payload)
        assert out is not None
        updated = out["hookSpecificOutput"].get("updatedInput")
        assert updated is not None
        assert updated["sessionId"] == "parent-sess"

    def test_subagent_call_injects_composite_sid(self):
        payload = {
            "session_id": "parent-sess",
            "agent_id": "explore-7",
            "agent_type": "Explore",
            "tool_name": "mcp__plugin_3_0_14_mempalace__mempalace_kg_stats",
            "tool_input": {},
        }
        out = _capture_hook_output(payload)
        assert out is not None
        updated = out["hookSpecificOutput"].get("updatedInput")
        assert updated is not None
        assert updated["sessionId"] == "parent-sess__sub_explore-7"

    def test_two_subagents_in_same_session_get_different_sids(self):
        base = {
            "session_id": "parent-sess",
            "tool_name": "mcp__plugin_3_0_14_mempalace__mempalace_kg_stats",
            "tool_input": {},
        }
        out_a = _capture_hook_output({**base, "agent_id": "alpha"})
        out_b = _capture_hook_output({**base, "agent_id": "beta"})
        sid_a = out_a["hookSpecificOutput"]["updatedInput"]["sessionId"]
        sid_b = out_b["hookSpecificOutput"]["updatedInput"]["sessionId"]
        assert sid_a != sid_b
        assert sid_a == "parent-sess__sub_alpha"
        assert sid_b == "parent-sess__sub_beta"


# ═══════════════════════════════════════════════════════════════════════
#  Server-side isolation: parent + subagent sid round-trip through
#  tool_declare_intent / pending state
# ═══════════════════════════════════════════════════════════════════════


_TEST_BUDGET = {"Read": 20, "Grep": 20, "Glob": 20}


def _seed_intent_types(kg, palace_path):
    """Minimal intent type hierarchy: research + agent + thing (shared with
    test_blocking_round_trips but re-inlined here to keep this file
    self-contained)."""
    import chromadb

    client = chromadb.PersistentClient(path=palace_path)
    ecol = client.get_or_create_collection("mempalace_entities")

    kg.add_entity(
        "intent_type", kind="class", description="Root class for all intent types", importance=5
    )
    kg.add_entity("thing", kind="class", description="Root class for all entities", importance=5)
    kg.add_entity("agent", kind="class", description="An AI agent", importance=5)
    kg.add_entity("test_agent", kind="entity", description="Test agent", importance=3)
    kg.add_triple("test_agent", "is_a", "agent")
    research_props = {
        "rules_profile": {
            "slots": {"subject": {"classes": ["thing"], "required": True, "multiple": True}},
            "tool_permissions": [
                {"tool": "Read", "scope": "*"},
                {"tool": "Grep", "scope": "*"},
                {"tool": "Glob", "scope": "*"},
            ],
        }
    }
    kg.add_entity(
        "research",
        kind="class",
        description="Exploratory research",
        importance=5,
        properties=research_props,
    )
    kg.add_triple("research", "is_a", "intent_type")
    kg.add_entity("test_target", kind="entity", description="A test target", importance=3)
    kg.add_triple("test_target", "is_a", "thing")
    for pred in (
        "is_a",
        "suggested_link",
        "found_useful",
        "found_irrelevant",
        "relates_to",
        "described_by",
        "executed_by",
        "targeted",
    ):
        kg.add_entity(pred, kind="predicate", description=f"p {pred}", importance=4)
    ecol.upsert(
        ids=["intent_type", "thing", "agent", "test_agent", "research", "test_target"],
        documents=["root", "root", "agent", "test", "research", "target"],
        metadatas=[
            {"name": "intent_type", "kind": "class", "importance": 5},
            {"name": "thing", "kind": "class", "importance": 5},
            {"name": "agent", "kind": "class", "importance": 5},
            {"name": "test_agent", "kind": "entity", "importance": 3, "added_by": "test_agent"},
            {"name": "research", "kind": "class", "importance": 5},
            {"name": "test_target", "kind": "entity", "importance": 3},
        ],
    )
    del client


@pytest.fixture
def mcp_env(monkeypatch, tmp_path):
    from mempalace import mcp_server
    from mempalace.config import MempalaceConfig
    from mempalace.knowledge_graph import KnowledgeGraph

    palace_path = tmp_path / "palace"
    palace_path.mkdir()
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(json.dumps({"palace_path": str(palace_path)}))
    cfg = MempalaceConfig(config_dir=str(cfg_dir))

    kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
    _seed_intent_types(kg, str(palace_path))

    monkeypatch.setattr(mcp_server._STATE, "config", cfg)
    monkeypatch.setattr(mcp_server._STATE, "kg", kg)
    monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
    monkeypatch.setattr(mcp_server._STATE, "pending_conflicts", None)
    monkeypatch.setattr(mcp_server._STATE, "pending_enrichments", None)
    monkeypatch.setattr(mcp_server._STATE, "session_id", "")
    monkeypatch.setattr(mcp_server._STATE, "declared_entities", set())

    state_dir = tmp_path / "hook_state"
    state_dir.mkdir()
    monkeypatch.setattr(mcp_server, "_INTENT_STATE_DIR", state_dir)

    return mcp_server


def _declare_research(mcp):
    return mcp.tool_declare_intent(
        intent_type="research",
        slots={"subject": ["test_target"]},
        context={
            "queries": ["subagent isolation test", "round-trip"],
            "keywords": ["subagent", "isolation"],
        },
        agent="test_agent",
        budget=_TEST_BUDGET,
    )


class TestSubagentIsolationRoundTrip:
    """End-to-end: simulate the parent and subagent each driving a tool
    call with their distinct sessionIds (as the hook would produce).
    State for one MUST NOT leak into the other."""

    def test_subagent_pending_does_not_block_parent_declare(self, mcp_env):
        """The live 2026-04-19 deadlock: parent is trying to declare_intent,
        but sees pending enrichments the subagent created. Under the
        composite-sid scheme, parent's declare sees the PARENT sid's pending
        (empty) and proceeds.
        """
        mcp = mcp_env
        from mempalace import intent

        parent_sid = "parent-abc"
        subagent_sid = "parent-abc__sub_explore-7"

        # Subagent dispatches a tool call; its sessionId is the composite.
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "mempalace_active_intent",
                "arguments": {"sessionId": subagent_sid},
            },
        }
        mcp.handle_request(req)
        assert mcp._STATE.session_id == subagent_sid

        # Subagent creates pending enrichments (in its own sid's state).
        mcp._STATE.pending_enrichments = [
            {"id": "sub_e1", "from_entity": "test_target", "to_entity": "test_agent"}
        ]
        intent._persist_active_intent()

        # Now the parent dispatches its declare_intent — sid switch to parent_sid.
        req2 = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "mempalace_declare_intent",
                "arguments": {
                    "sessionId": parent_sid,
                    "intent_type": "research",
                    "slots": {"subject": ["test_target"]},
                    "context": {
                        "queries": ["parent work", "unblock"],
                        "keywords": ["parent", "work"],
                    },
                    "agent": "test_agent",
                    "budget": _TEST_BUDGET,
                },
            },
        }
        response = mcp.handle_request(req2)
        # Extract the JSON-RPC result.
        assert "result" in response, f"expected jsonrpc result, got {response!r}"
        content = response["result"].get("content", [])
        # The tool returns its payload as a content block (text/json).
        # Load the first text block back as JSON.
        text_blocks = [c.get("text", "") for c in content if c.get("type") == "text"]
        assert text_blocks
        parsed = json.loads(text_blocks[0])
        assert parsed.get("success") is True, (
            f"parent declare blocked by subagent pending; response={parsed!r}"
        )

        # And subagent's disk file is still there untouched.
        sub_file = mcp._INTENT_STATE_DIR / f"active_intent_{subagent_sid}.json"
        assert sub_file.is_file()
        data = json.loads(sub_file.read_text())
        assert data.get("pending_enrichments")
        assert data["pending_enrichments"][0]["id"] == "sub_e1"

    def test_parent_pending_does_not_block_subagent_declare(self, mcp_env):
        """Mirror: parent has pending; subagent declare must succeed."""
        mcp = mcp_env
        from mempalace import intent

        parent_sid = "parent-xyz"
        subagent_sid = "parent-xyz__sub_search"

        mcp._STATE.session_id = parent_sid
        mcp._STATE.pending_enrichments = [
            {"id": "parent_e1", "from_entity": "test_target", "to_entity": "test_agent"}
        ]
        intent._persist_active_intent()

        # Subagent dispatches declare_intent.
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "mempalace_declare_intent",
                "arguments": {
                    "sessionId": subagent_sid,
                    "intent_type": "research",
                    "slots": {"subject": ["test_target"]},
                    "context": {
                        "queries": ["subagent job", "exploration"],
                        "keywords": ["subagent", "job"],
                    },
                    "agent": "test_agent",
                    "budget": _TEST_BUDGET,
                },
            },
        }
        response = mcp.handle_request(req)
        text = [
            c.get("text", "")
            for c in response["result"].get("content", [])
            if c.get("type") == "text"
        ]
        parsed = json.loads(text[0])
        assert parsed.get("success") is True, (
            f"subagent declare blocked by parent pending: {parsed!r}"
        )

        # Parent's disk file intact.
        parent_file = mcp._INTENT_STATE_DIR / f"active_intent_{parent_sid}.json"
        assert parent_file.is_file()
        data = json.loads(parent_file.read_text())
        assert data["pending_enrichments"][0]["id"] == "parent_e1"

    def test_two_subagents_same_parent_do_not_cross_contaminate(self, mcp_env):
        """Two subagents dispatched from the same parent — each gets its own
        composite sid, and their pending states are independent."""
        mcp = mcp_env
        from mempalace import intent

        sub_a = "parent__sub_alpha"
        sub_b = "parent__sub_beta"

        # Subagent alpha lands a pending.
        mcp._STATE.session_id = sub_a
        mcp._STATE.pending_enrichments = [
            {"id": "alpha_1", "from_entity": "test_target", "to_entity": "test_agent"}
        ]
        intent._persist_active_intent()

        # Subagent beta runs declare_intent via dispatch — sees NO pending.
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "mempalace_declare_intent",
                "arguments": {
                    "sessionId": sub_b,
                    "intent_type": "research",
                    "slots": {"subject": ["test_target"]},
                    "context": {
                        "queries": ["beta work", "different task"],
                        "keywords": ["beta", "work"],
                    },
                    "agent": "test_agent",
                    "budget": _TEST_BUDGET,
                },
            },
        }
        response = mcp.handle_request(req)
        text = [
            c.get("text", "")
            for c in response["result"].get("content", [])
            if c.get("type") == "text"
        ]
        parsed = json.loads(text[0])
        assert parsed.get("success") is True

        # Alpha's pending is still pending.
        alpha_file = mcp._INTENT_STATE_DIR / f"active_intent_{sub_a}.json"
        assert alpha_file.is_file()
        data = json.loads(alpha_file.read_text())
        assert data["pending_enrichments"][0]["id"] == "alpha_1"

    def test_subagent_finalize_does_not_touch_parent_state(self, mcp_env):
        """If a subagent finalizes its intent, only its OWN sid's disk file
        is affected — the parent's active-intent file stays intact."""
        mcp = mcp_env
        from mempalace import intent

        parent_sid = "parent-fin"
        sub_sid = "parent-fin__sub_exec"

        # Parent has an active intent.
        mcp._STATE.session_id = parent_sid
        mcp._STATE.active_intent = {
            "intent_id": "intent_parent_1",
            "intent_type": "research",
            "slots": {},
            "effective_permissions": [],
            "description": "parent work",
            "agent": "test_agent",
            "session_id": parent_sid,
        }
        intent._persist_active_intent()
        parent_file = mcp._INTENT_STATE_DIR / f"active_intent_{parent_sid}.json"
        assert parent_file.is_file()
        parent_snapshot = parent_file.read_text()

        # Subagent declares + finalizes.
        mcp._STATE.session_id = sub_sid
        mcp._STATE.active_intent = {
            "intent_id": "intent_sub_1",
            "intent_type": "research",
            "slots": {},
            "effective_permissions": [],
            "description": "sub work",
            "agent": "test_agent",
            "session_id": sub_sid,
        }
        intent._persist_active_intent()
        # Simulate subagent finalization by clearing its state.
        mcp._STATE.active_intent = None
        mcp._STATE.pending_conflicts = None
        mcp._STATE.pending_enrichments = None
        intent._persist_active_intent()
        sub_file = mcp._INTENT_STATE_DIR / f"active_intent_{sub_sid}.json"
        assert not sub_file.exists(), "finalized subagent sid file should be unlinked"

        # Parent's file is byte-identical to the snapshot.
        assert parent_file.is_file()
        assert parent_file.read_text() == parent_snapshot, (
            "subagent finalization mutated parent's active-intent file"
        )
