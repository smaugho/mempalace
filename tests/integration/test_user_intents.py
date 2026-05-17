"""
test_user_intents.py -- unit tests for tool_declare_user_intents.

ships:
  * pending_user_messages persistence helpers in hooks_cli (read /
    append / clear) -- disk-backed per-session JSON queue.
  * tool_declare_user_intents handler in intent.py -- validates pending
    coverage, mints user_message records (kind='record'), runs
    context_lookup_or_create per declared user-context, returns memories
    per context, clears pending.
  * MCP schema registration in mcp_server.py.

(next commit) wires the UserPromptSubmit hook to write
pending entries and the PreToolUse hook to block non-allowed tools
while pending > 0. adds optional cause_id on declare_intent
+ finalize coverage rule.

These tests exercise the tool in isolation by writing the pending
file directly (simulating what UserPromptSubmit will do once the future build
lands). Validation rules -- coverage, unknown ids, no_intent proof --
are locked here so Slice B-2's hook write doesn't drift.

Grounding: STITCH (arXiv:2601.10702), Agent-Sentry (arXiv:2603.22868),
BDI (Rao & Georgeff 1995). See diary
diary_ga_agent_user_intent_tier_design_locked_2026_04_24.
"""

from __future__ import annotations

import json

import pytest  # noqa: F401 -- test helpers reference pytest.raises etc.


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _bootstrap(monkeypatch, tmp_path):
    """Build a hermetic mempalace state for a single test.

    Mirrors the pattern in test_intent_system.py -- point STATE_DIR at
    tmp_path, reset _STATE.session_id / active_intent / kg, return the
    mcp_server module so tests can poke handler shims directly.
    """
    from mempalace import hooks_cli, intent, mcp_server
    from mempalace.knowledge_graph import KnowledgeGraph

    monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)
    monkeypatch.setattr(mcp_server, "_INTENT_STATE_DIR", tmp_path)
    monkeypatch.setattr(intent, "_INTENT_STATE_DIR", tmp_path, raising=False)

    # Reset Chroma caches so prior tests don't leak collection vectors
    # across tmp_path boundaries (mirrors _b3_bootstrap).
    intent._reset_rated_user_contexts()
    mcp_server._STATE.client_cache = None
    mcp_server._STATE.collection_cache = None

    # Point _STATE.config at this test's tmp_path so context_lookup_or_create's
    # Chroma client (which rebuilds via PersistentClient(_STATE.config.palace_path)
    # on every call) isolates Chroma data per-test. Without this, prior tests'
    # _STATE.config leaks and the chroma collection lives in a stale path --
    # context_lookup_or_create returns ('', False, ...) silently.
    _isolated_config = type("IsolatedTestConfig", (), {"palace_path": str(tmp_path)})()
    monkeypatch.setattr(mcp_server._STATE, "config", _isolated_config)

    db = tmp_path / "palace.db"
    kg = KnowledgeGraph(str(db))
    # Cold-start lock 2026-05-01: seed the base ontology (mints `thing`,
    # `agent` class, all canonical predicates including `caused_by`,
    # `fulfills_user_message`) AND the user-intent ontology so the
    # add_triple paths in declare_intent / declare_user_intents land
    # on real entities. Pre-cold-start, every missing parent was
    # phantom-created; the gate's hard-reject closes that surface,
    # and the right fix is to seed in the proper order.
    from mempalace.seed import _ensure_user_intent_ontology

    kg.seed_ontology()
    _ensure_user_intent_ontology(kg)
    monkeypatch.setattr(mcp_server._STATE, "kg", kg)
    monkeypatch.setattr(mcp_server._STATE, "session_id", "test-sid-b1")
    monkeypatch.setattr(mcp_server._STATE, "active_intent", None)

    # Declare an agent so added_by validation passes.
    kg.add_entity("ga_agent", kind="entity", content="ga test agent", importance=3)
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
        """2026-04-28 contract: ``msg_<sid_short>_<turn_idx>``.

        Disambiguator is turn_idx (monotonic per session), NOT text --
        turns are unique per session by definition, so hashing text was
        never adding information. Same (session, turn) yields the same
        id regardless of text payload; different session OR different
        turn yields different id. Format is ~12 chars (~3 tokens) vs
        the prior msg_<sha12>_<ns> which was ~22 chars (~6+ tokens).
        """
        from mempalace import hooks_cli

        # Same (sid, turn) -- same id regardless of text.
        a = hooks_cli._make_user_message_id("sid-x", 5, "fix the auth bug")
        b = hooks_cli._make_user_message_id("sid-x", 5, "fix the auth bug")
        c = hooks_cli._make_user_message_id("sid-x", 5, "different prompt")
        assert a == b == c, "msg_id depends on (sid, turn) only after slice 4"
        # Different turn -- different id.
        d = hooks_cli._make_user_message_id("sid-x", 6, "fix the auth bug")
        assert a != d, "different turn_idx must yield different id"
        # Different session -- different id.
        e = hooks_cli._make_user_message_id("sid-y", 5, "fix the auth bug")
        assert a != e, "different session must yield different id"
        # Format invariants: msg_ + 6-char hex digest + _ + turn_idx digits.
        parts = a.split("_")
        assert len(parts) == 3 and parts[0] == "msg" and len(parts[1]) == 6
        assert parts[2] == "5"


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
        # CI Linux flake debug: dump full state if ctx_id empty so we can diagnose
        if not block.get("ctx_id"):
            import sys

            print("DEBUG_FAIL_RESULT:", result, file=sys.stderr, flush=True)
            print("DEBUG_FAIL_STATE:", mcp_server._STATE.active_intent, file=sys.stderr, flush=True)
        assert block["ctx_id"], "context id should be minted"
        assert isinstance(block["reused"], bool)
        assert result["cleared_pending_count"] == 1
        # minted_user_message_ids field retired in 2026-05-04 token-diet pass
        # (server still mints the user_message records; the response just no
        # longer echoes them back).

        # Cold-start lock 2026-05-01 (Adrian's user-message analysis):
        # user_messages mint with kind='user_message', not 'record'.
        # The literal user text is value, not identity -- the user-
        # context that fulfills the message carries the searchable
        # identity. user_messages skip the Chroma sync and the summary
        # contract by design.
        rec = kg.get_entity(m["id"])
        assert rec is not None
        assert rec.get("kind") == "user_message"

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


# ─────────────────────────────────────────────────────────────────────
# cause_id wiring on tool_declare_intent
# ─────────────────────────────────────────────────────────────────────


def _b3_bootstrap(monkeypatch, tmp_path):
    """Bootstrap a fresh palace with the user-intent + task ontologies
    seeded so cause_id validation has the predicates it needs."""
    from mempalace import hooks_cli, intent, mcp_server
    from mempalace.knowledge_graph import KnowledgeGraph

    monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)
    monkeypatch.setattr(mcp_server, "_INTENT_STATE_DIR", tmp_path)
    monkeypatch.setattr(intent, "_INTENT_STATE_DIR", tmp_path, raising=False)

    # Cold-start lock 2026-05-01: reset module-level intent + Chroma
    # state so the previous test's (sid -> rated-user-contexts) bucket,
    # context_views collection cache, and entity-Chroma collection
    # cache don't leak across tmp_path boundaries. All tests in this
    # file use session_id="test-sid-b3" and share _RATED_USER_CONTEXTS,
    # _STATE.client_cache, and _STATE.collection_cache at module scope.
    # Without this clear, context_lookup_or_create reuses the previous
    # test's activity context (because Chroma still has its view
    # vectors), and the cold-start gate's assert_entity_exists then
    # rejects the caused_by edge against an id that exists only in
    # Chroma but not in this test's SQLite tmp file.
    intent._reset_rated_user_contexts()
    mcp_server._STATE.client_cache = None
    mcp_server._STATE.collection_cache = None

    db = tmp_path / "palace.db"
    kg = KnowledgeGraph(str(db))
    # Cold-start lock 2026-05-01: unique session_id per test (derived
    # from the tmp_path basename) so module-level state keyed by sid
    # can't leak across tests. ALSO point _STATE.config at this test's
    # tmp_path so context_lookup_or_create's Chroma client (which
    # rebuilds via PersistentClient(_STATE.config.palace_path) on every
    # call) isolates Chroma data per-test. Pre-cold-start the phantom
    # auto-create masked this leak; post-cold-start, the assert_entity
    # _exists check in add_triple surfaces it as a PhantomEntityRejected
    # because the activity context (ctx_1) lives only in the previous
    # test's Chroma collection, not in this test's SQLite tmp file.
    _unique_sid = f"test-sid-b3-{tmp_path.name}"
    _isolated_config = type(
        "IsolatedTestConfig",
        (),
        {"palace_path": str(tmp_path)},
    )()
    monkeypatch.setattr(mcp_server._STATE, "kg", kg)
    monkeypatch.setattr(mcp_server._STATE, "session_id", _unique_sid)
    monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
    monkeypatch.setattr(mcp_server._STATE, "config", _isolated_config)

    # Cold-start lock 2026-05-01: seed the base ontology first so the
    # `is_a thing` triples written by _ensure_task_ontology and
    # _ensure_user_intent_ontology land on a declared parent class.
    kg.seed_ontology()
    mcp_server._ensure_task_ontology(kg)
    mcp_server._ensure_user_intent_ontology(kg)
    kg.add_entity("ga_agent", kind="entity", content="ga test agent", importance=3)
    kg.add_triple("ga_agent", "is_a", "agent")

    # Minimal intent_type seed so tool_declare_intent('modify', ...)
    # resolves. Mirrors the relevant slice of test_intent_system._seed
    # _intent_types -- only the bits needed for the modify path.
    kg.add_entity(
        "intent_type",
        kind="class",
        content="Root class for all intent types",
        importance=5,
    )
    kg.add_entity(
        "modify",
        kind="class",
        content="Edit or create files in the codebase",
        importance=4,
        properties={
            "rules_profile": {
                "slots": {
                    "files": {"classes": ["thing"], "required": False, "multiple": True},
                    "paths": {"classes": ["thing"], "required": False, "multiple": True},
                    "commands": {
                        "classes": ["thing"],
                        "required": False,
                        "multiple": True,
                    },
                },
                "tool_permissions": [
                    {"tool": "Edit", "scope": "{files}"},
                    {"tool": "Write", "scope": "{files}"},
                    {"tool": "Read", "scope": "{paths}"},
                    {"tool": "Grep", "scope": "{paths}"},
                ],
            }
        },
    )
    kg.add_triple("modify", "is_a", "intent_type")

    return mcp_server, kg


def _b3_args(**overrides):
    base = {
        "intent_type": "modify",
        "slots": {"files": [], "paths": [], "commands": []},
        "context": {
            "queries": ["test cause_id wiring", "exercise the caused_by edge"],
            "keywords": ["cause_id", "caused_by", "test"],
            "entities": ["ga_agent"],
            "summary": {
                "what": "exercise cause_id wiring",
                "why": "verify caused_by edge writes when a valid parent cause is supplied",
            },
        },
        "agent": "ga_agent",
        "budget": {"Read": 1},
    }
    base.update(overrides)
    return base


class TestDeclareIntentCauseIdBackCompat:
    """cause_id is OPTIONAL. Existing call sites still work."""

    def test_declare_intent_without_cause_id_succeeds(self, tmp_path, monkeypatch):
        # retired the cause_id back-compat (Adrian directive 2026-05-04):
        # cause_id is now MANDATORY. The conftest shim auto-injects 'autonomous'
        # for tests that omit it, so the call still succeeds and active_intent
        # records the autonomous parent-cause sentinel.
        mcp_server, _kg = _b3_bootstrap(monkeypatch, tmp_path)
        result = mcp_server.tool_declare_intent(**_b3_args())
        assert result.get("success") is not False, result
        state = mcp_server._STATE.active_intent
        assert state is not None
        # The conftest shim auto-injects cause_id='autonomous' (the no-parent
        # sentinel). Production recognises the sentinel and persists cause_id
        # as empty -- it writes no caused_by edge. Agents must still pass the
        # arg explicitly; back-compat with omitted cause_id was retired.
        # cause_id is the sentinel literal; cause_kind records the sentinel too.
        # No caused_by edge is written for autonomous, but the explicit value
        # forces the agent to acknowledge no parent rather than silently skipping.
        # Production stores the literal sentinel as cause_kind only;
        # cause_id is empty (no parent). No caused_by edge is written.
        assert state.get("cause_id", "") == ""
        assert state.get("cause_kind", "") == "autonomous"


class TestDeclareIntentSubagentAutonomousRejection:
    """v3.6.0 Slice A regression: sub-agent sessions cannot self-declare
    cause_id='autonomous'. The parent dispatched them; they MUST inherit
    a Task entity from the parent. Detection is via the '__sub_' suffix
    that _effective_session_id mints for sub-agent sessions."""

    def test_subagent_cause_id_autonomous_is_rejected(self, tmp_path, monkeypatch):
        mcp_server, _kg = _b3_bootstrap(monkeypatch, tmp_path)
        # Simulate a sub-agent session: the __sub_<hash> suffix is what
        # _effective_session_id produces under the Task tool dispatch.
        mcp_server._STATE.session_id = "test_sid__sub_planner_a1b2"
        result = mcp_server.tool_declare_intent(**_b3_args(cause_id="autonomous"))
        assert result.get("success") is False, result
        assert result.get("error_kind") == "subagent_autonomous_rejected", result
        # Directive prose mentions BOTH the kg_declare_entity recipe and
        # the cause_id replacement so the parent operator can act on the
        # error message alone without grepping docs.
        msg = result.get("error", "")
        assert "kg_declare_entity" in msg
        assert "task_<" in msg or "is_a='task'" in msg
        assert "cause_id" in msg

    def test_non_subagent_cause_id_autonomous_still_works(self, tmp_path, monkeypatch):
        # Regression guard: the autonomous escape MUST still work for
        # non-sub-agent sessions (gardener / scheduled audits / the main
        # ga_agent's truly self-driven background work).
        mcp_server, _kg = _b3_bootstrap(monkeypatch, tmp_path)
        mcp_server._STATE.session_id = "test_sid_main_no_sub_suffix"
        result = mcp_server.tool_declare_intent(**_b3_args(cause_id="autonomous"))
        assert result.get("success") is not False, result
        state = mcp_server._STATE.active_intent
        assert state is not None
        assert state.get("cause_kind", "") == "autonomous"


class TestDeclareIntentSubagentTaskOnly:
    """v3.6.1 (Adrian directive 2026-05-16, follow-on to Slice A):
    sub-agent declare_intent MUST anchor cause_id to a Task entity.
    user-context cause_ids belong to the parent session; sub-agents
    inherit via the Task entity the parent scoped for them. Slice A
    closed the 'autonomous' magic-word loophole; this closes the
    second loophole where a sub-agent could pass the parent's
    user-context ctx_id and bypass Task-entity attribution."""

    def test_subagent_with_task_cause_succeeds(self, tmp_path, monkeypatch):
        mcp_server, kg = _b3_bootstrap(monkeypatch, tmp_path)
        mcp_server._STATE.session_id = "test_sid__sub_planner_c1d2"
        kg.add_entity(
            "task_subagent_alpha",
            kind="entity",
            content="Sub-agent task: planner alpha workstream",
            importance=4,
        )
        kg.add_triple("task_subagent_alpha", "is_a", "Task")
        result = mcp_server.tool_declare_intent(**_b3_args(cause_id="task_subagent_alpha"))
        assert result.get("success") is not False, result
        state = mcp_server._STATE.active_intent
        assert state is not None
        assert state["cause_id"] == "task_subagent_alpha"
        assert state["cause_kind"] == "task"

    def test_subagent_with_user_context_cause_is_rejected(self, tmp_path, monkeypatch):
        mcp_server, kg = _b3_bootstrap(monkeypatch, tmp_path)
        # Build a valid user-context (would normally accept under non-sub-
        # agent session per TestDeclareIntentCauseIdHappyPath).
        kg.add_entity(
            "ctx_user_beta",
            kind="context",
            content="user-intent context: pick up the next slice",
            importance=3,
        )
        kg.add_entity(
            "msg_beta",
            kind="record",
            content="pick up the next slice",
            importance=3,
            properties={"type": "user_message"},
        )
        kg.add_triple(
            "ctx_user_beta",
            "fulfills_user_message",
            "msg_beta",
            statement="ctx_user_beta fulfils msg_beta for the next-slice intent",
        )
        # Flip to a sub-agent session and try to use the user-context id.
        mcp_server._STATE.session_id = "test_sid__sub_planner_e3f4"
        result = mcp_server.tool_declare_intent(**_b3_args(cause_id="ctx_user_beta"))
        assert result.get("success") is False, result
        assert result.get("error_kind") == "subagent_non_task_cause_rejected", result
        msg = result.get("error", "")
        # Directive prose must mention the Task-entity recipe + the
        # task_id= prompt-prefix contract so the parent operator can
        # act on the error message alone.
        assert "Task entity" in msg
        assert "kg_declare_entity" in msg
        assert "is_a='Task'" in msg
        assert "task_id=task_" in msg
        assert "cause_id" in msg

    def test_non_subagent_with_user_context_cause_still_works(self, tmp_path, monkeypatch):
        # Regression guard: user-context cause_ids stay valid for
        # non-sub-agent sessions (the parent ga_agent's normal path).
        mcp_server, kg = _b3_bootstrap(monkeypatch, tmp_path)
        kg.add_entity(
            "ctx_user_gamma",
            kind="context",
            content="user-intent context: parent agent path",
            importance=3,
        )
        kg.add_entity(
            "msg_gamma",
            kind="record",
            content="parent agent path",
            importance=3,
            properties={"type": "user_message"},
        )
        kg.add_triple(
            "ctx_user_gamma",
            "fulfills_user_message",
            "msg_gamma",
            statement="ctx_user_gamma fulfils msg_gamma for the parent path",
        )
        mcp_server._STATE.session_id = "test_sid_main_parent_no_sub"
        result = mcp_server.tool_declare_intent(**_b3_args(cause_id="ctx_user_gamma"))
        assert result.get("success") is not False, result
        state = mcp_server._STATE.active_intent
        assert state is not None
        assert state["cause_kind"] == "user_context"
        assert state["cause_id"] == "ctx_user_gamma"


class TestDeclareIntentCauseIdHappyPath:
    """Both valid parent-cause kinds are accepted; caused_by edge lands;
    cause_id + cause_kind persist on active_intent state."""

    def test_user_context_cause_writes_caused_by_edge(self, tmp_path, monkeypatch):
        mcp_server, kg = _b3_bootstrap(monkeypatch, tmp_path)

        kg.add_entity(
            "ctx_user_alpha",
            kind="context",
            content="user-intent context: fix the auth bug",
            importance=3,
        )
        kg.add_entity(
            "msg_alpha",
            kind="record",
            content="fix the auth bug",
            importance=3,
            properties={"type": "user_message"},
        )
        kg.add_triple(
            "ctx_user_alpha",
            "fulfills_user_message",
            "msg_alpha",
            statement="ctx_user_alpha fulfils msg_alpha for the auth bug intent",
        )

        result = mcp_server.tool_declare_intent(**_b3_args(cause_id="ctx_user_alpha"))
        assert result.get("success") is not False, result

        state = mcp_server._STATE.active_intent
        assert state["cause_id"] == "ctx_user_alpha"
        assert state["cause_kind"] == "user_context"

        activity_ctx_id = state.get("active_context_id", "")
        assert activity_ctx_id
        edges = kg.query_entity(activity_ctx_id, direction="outgoing")
        targets = {
            e.get("object")
            for e in edges
            if e.get("predicate") == "caused_by" and e.get("current", True)
        }
        assert "ctx_user_alpha" in targets

    def test_task_cause_writes_caused_by_edge(self, tmp_path, monkeypatch):
        mcp_server, kg = _b3_bootstrap(monkeypatch, tmp_path)

        kg.add_entity(
            "TASK-fix-rate-limiter",
            kind="entity",
            content="Fix the 429 rate-limiter regression",
            importance=4,
        )
        kg.add_triple("TASK-fix-rate-limiter", "is_a", "Task")

        result = mcp_server.tool_declare_intent(**_b3_args(cause_id="TASK-fix-rate-limiter"))
        assert result.get("success") is not False, result

        state = mcp_server._STATE.active_intent
        assert state["cause_id"] == "TASK-fix-rate-limiter"
        assert state["cause_kind"] == "task"

        activity_ctx_id = state.get("active_context_id", "")
        edges = kg.query_entity(activity_ctx_id, direction="outgoing")
        targets = {
            e.get("object")
            for e in edges
            if e.get("predicate") == "caused_by" and e.get("current", True)
        }
        assert "TASK-fix-rate-limiter" in targets


class TestDeclareIntentCauseIdRejections:
    """validation rules: each violation surfaces a specific
    error so the agent can self-correct without re-reading docs."""

    def test_unknown_cause_id_rejected(self, tmp_path, monkeypatch):
        mcp_server, _kg = _b3_bootstrap(monkeypatch, tmp_path)
        result = mcp_server.tool_declare_intent(**_b3_args(cause_id="entity_does_not_exist"))
        assert result["success"] is False
        assert "does not resolve" in result["error"]

    def test_wrong_kind_cause_id_rejected(self, tmp_path, monkeypatch):
        mcp_server, kg = _b3_bootstrap(monkeypatch, tmp_path)
        kg.add_entity(
            "random_entity",
            kind="entity",
            content="some unrelated entity",
            importance=2,
        )
        result = mcp_server.tool_declare_intent(**_b3_args(cause_id="random_entity"))
        assert result["success"] is False
        err = result["error"]
        assert "not a valid parent cause" in err
        assert "is_a" in err
        assert "fulfills_user_message" in err

    def test_activity_context_cause_id_rejected(self, tmp_path, monkeypatch):
        """A kind='context' entity WITHOUT a fulfills_user_message edge
        is an activity-intent context, not a user-context. Reject so
        agents do not chain activity-to-activity (cause_id is for
        parent-tier linkage only)."""
        mcp_server, kg = _b3_bootstrap(monkeypatch, tmp_path)
        kg.add_entity(
            "ctx_activity_only",
            kind="context",
            content="activity-intent context with no user-message edge",
            importance=3,
        )
        result = mcp_server.tool_declare_intent(**_b3_args(cause_id="ctx_activity_only"))
        assert result["success"] is False
        assert "not a valid parent cause" in result["error"]


class TestUserIntentOntologySeeded:
    """The seeder mints caused_by + fulfills_user_message predicates
    with the right constraint shape so kg_add accepts them without
    soft-fail wrappers."""

    def test_caused_by_seeded(self, tmp_path, monkeypatch):
        mcp_server, kg = _b3_bootstrap(monkeypatch, tmp_path)
        ent = kg.get_entity("caused_by")
        assert ent is not None
        assert ent.get("kind") == "predicate"

    def test_fulfills_user_message_seeded(self, tmp_path, monkeypatch):
        mcp_server, kg = _b3_bootstrap(monkeypatch, tmp_path)
        ent = kg.get_entity("fulfills_user_message")
        assert ent is not None
        assert ent.get("kind") == "predicate"


# ─────────────────────────────────────────────────────────────────────
# finalize wiring + first-rater coverage rule
# ─────────────────────────────────────────────────────────────────────


def _b4_bootstrap(monkeypatch, tmp_path):
    """Bootstrap that builds on _b3_bootstrap and seeds the additional
    predicates Slice B-4's tests need (`surfaced`, `rated_useful`,
    `rated_irrelevant`). Also resets the module-level
    `_RATED_USER_CONTEXTS` dict so each test starts empty."""
    from mempalace import intent

    mcp_server, kg = _b3_bootstrap(monkeypatch, tmp_path)

    # Predicates required for finalize_intent's coverage path.
    for pred in ("surfaced", "rated_useful", "rated_irrelevant", "executed_by"):
        if kg.get_entity(pred) is None:
            kg.add_entity(pred, kind="predicate", content=f"{pred} pred", importance=4)

    # Reset the session-scoped first-rater set so cross-test state
    # cannot leak.
    intent._reset_rated_user_contexts()

    return mcp_server, kg


def _make_user_context(kg, ctx_id="ctx_user_alpha", msg_id="msg_alpha"):
    """Mint a user-context entity + its fulfills_user_message edge."""
    kg.add_entity(
        ctx_id,
        kind="context",
        content="user-intent context for b-4 tests",
        importance=3,
    )
    kg.add_entity(
        msg_id,
        kind="record",
        content="user message for b-4 tests",
        importance=3,
        properties={"type": "user_message"},
    )
    kg.add_triple(
        ctx_id,
        "fulfills_user_message",
        msg_id,
        statement=f"{ctx_id} fulfils {msg_id} for the b-4 first-rater suite",
    )


def _stage_user_context_surfaced(kg, ctx_id, mem_ids):
    """Wire `surfaced` edges from a user-context to the listed memories.
    Mirrors what declare_user_intents does at retrieval time."""
    for mid in mem_ids:
        if kg.get_entity(mid) is None:
            kg.add_entity(mid, kind="record", content=f"memory {mid}", importance=3)
        kg.add_triple(ctx_id, "surfaced", mid)


def _finalize_minimal(mcp_server, slug):
    """Call tool_finalize_intent without agent ratings.

    v3.5.0 (2026-05-14): the memory_feedback / operation_ratings agent
    coverage path is retired; mempalace.feedback_auto's Haiku rater
    handles rating out of band. This helper just shells out to
    tool_finalize_intent with the bookkeeping fields the b-4 test
    suite expects.
    """
    from mempalace.intent import tool_finalize_intent

    return tool_finalize_intent(
        slug=slug,
        outcome="success",
        content=(
            "Finalize end-to-end body for the b-4 first-rater test fixture; "
            "rephrases the summary to give the embedding two cosine views."
        ),
        summary={
            "what": "b-4 first-rater fixture",
            "why": "exercise finalize wiring of cause_id and the rated_user_contexts session set",
            "scope": "tests",
        },
        agent="ga_agent",
    )


class TestB4FinalizeCausedByEdge:
    """finalize_intent writes a caused_by edge from the
    execution entity to active_intent.cause_id when present."""

    def test_caused_by_edge_lands_on_execution_entity(self, tmp_path, monkeypatch):
        mcp_server, kg = _b4_bootstrap(monkeypatch, tmp_path)
        _make_user_context(kg)

        decl = mcp_server.tool_declare_intent(**_b3_args(cause_id="ctx_user_alpha"))
        assert decl.get("success") is not False, decl

        result = _finalize_minimal(mcp_server, "b4-caused-by-user-context")
        assert result.get("success") is True, result
        exec_id = result["execution_entity"]

        edges = kg.query_entity(exec_id, direction="outgoing")
        targets = {
            e.get("object")
            for e in edges
            if e.get("predicate") == "caused_by" and e.get("current", True)
        }
        assert "ctx_user_alpha" in targets, edges

    def test_no_cause_id_no_caused_by_edge(self, tmp_path, monkeypatch):
        mcp_server, kg = _b4_bootstrap(monkeypatch, tmp_path)

        decl = mcp_server.tool_declare_intent(**_b3_args())
        assert decl.get("success") is not False, decl

        result = _finalize_minimal(mcp_server, "b4-no-cause")
        assert result.get("success") is True, result
        exec_id = result["execution_entity"]

        edges = kg.query_entity(exec_id, direction="outgoing")
        caused_by_edges = [
            e for e in edges if e.get("predicate") == "caused_by" and e.get("current", True)
        ]
        assert caused_by_edges == [], (
            "no cause_id at declare time means no caused_by edge at finalize"
        )


class TestB4FirstRaterSnapshot:
    """declare_intent snapshots first-rater state onto
    active_intent so finalize knows whether to apply the user-context
    coverage exemption."""

    def test_first_intent_with_cause_id_is_first_rater(self, tmp_path, monkeypatch):
        mcp_server, kg = _b4_bootstrap(monkeypatch, tmp_path)
        _make_user_context(kg)
        _stage_user_context_surfaced(kg, "ctx_user_alpha", ["mem_b4_one", "mem_b4_two"])

        decl = mcp_server.tool_declare_intent(**_b3_args(cause_id="ctx_user_alpha"))
        assert decl.get("success") is not False, decl

        state = mcp_server._STATE.active_intent
        assert state["user_context_first_rater"] is True, state
        # First-rater: no exemption snapshot needed (full coverage required).
        assert state["user_context_exempt_ids"] == [], state

    def test_subsequent_intent_inherits_exemption(self, tmp_path, monkeypatch):
        mcp_server, kg = _b4_bootstrap(monkeypatch, tmp_path)
        _make_user_context(kg)
        _stage_user_context_surfaced(kg, "ctx_user_alpha", ["mem_b4_one", "mem_b4_two"])

        # First intent: declare → finalize. Adds cause_id to rated set.
        decl1 = mcp_server.tool_declare_intent(**_b3_args(cause_id="ctx_user_alpha"))
        assert decl1.get("success") is not False, decl1
        result1 = _finalize_minimal(mcp_server, "b4-first-rater")
        assert result1.get("success") is True, result1

        # Drop any pending duplicate-detection conflicts the first
        # finalize's result record triggered. They are not the subject
        # of this test and they would otherwise block the second
        # declare_intent call below. The disk-backed file is the
        # source-of-truth (mcp_server._restore_session_state reloads
        # from it on every declare), so wipe it as well.
        mcp_server._STATE.pending_conflicts = None
        sid = mcp_server._STATE.session_id
        state_file = tmp_path / f"active_intent_{sid}.json"
        if state_file.is_file():
            state_file.unlink()

        # Second intent: same cause_id. Snapshot must show NOT first
        # rater + exempt_ids = the user-context's surfaced memories.
        decl2 = mcp_server.tool_declare_intent(**_b3_args(cause_id="ctx_user_alpha"))
        assert decl2.get("success") is not False, decl2

        state2 = mcp_server._STATE.active_intent
        assert state2["user_context_first_rater"] is False, state2
        assert set(state2["user_context_exempt_ids"]) == {"mem_b4_one", "mem_b4_two"}, state2

    def test_task_cause_kind_does_not_use_first_rater_path(self, tmp_path, monkeypatch):
        """Task entities have no surfaced-memory inheritance contract.
        A Task cause leaves first_rater at the True default and stores
        no exempt ids regardless of the rated set."""
        mcp_server, kg = _b4_bootstrap(monkeypatch, tmp_path)
        kg.add_entity(
            "TASK-b4-task",
            kind="entity",
            content="b-4 task entity",
            importance=3,
        )
        kg.add_triple("TASK-b4-task", "is_a", "Task")

        decl = mcp_server.tool_declare_intent(**_b3_args(cause_id="TASK-b4-task"))
        assert decl.get("success") is not False, decl

        state = mcp_server._STATE.active_intent
        assert state["cause_kind"] == "task", state
        assert state["user_context_first_rater"] is True, state
        assert state["user_context_exempt_ids"] == [], state


class TestB4RatedSetUpdates:
    """session-scoped rated_user_contexts set behavior."""

    def test_finalize_adds_user_context_cause_to_rated_set(self, tmp_path, monkeypatch):
        from mempalace import intent

        mcp_server, kg = _b4_bootstrap(monkeypatch, tmp_path)
        _make_user_context(kg)

        decl = mcp_server.tool_declare_intent(**_b3_args(cause_id="ctx_user_alpha"))
        assert decl.get("success") is not False, decl

        # Pre-finalize: rated set is empty for this session.
        sid = mcp_server._STATE.session_id
        assert "ctx_user_alpha" not in intent._rated_user_contexts_for(sid)

        result = _finalize_minimal(mcp_server, "b4-rated-set-add")
        assert result.get("success") is True, result

        # Post-finalize: cause_id is registered as rated.
        assert "ctx_user_alpha" in intent._rated_user_contexts_for(sid)

    def test_finalize_does_not_add_task_cause_to_rated_set(self, tmp_path, monkeypatch):
        from mempalace import intent

        mcp_server, kg = _b4_bootstrap(monkeypatch, tmp_path)
        kg.add_entity(
            "TASK-b4-rated-skip",
            kind="entity",
            content="b-4 task that must not enter rated set",
            importance=3,
        )
        kg.add_triple("TASK-b4-rated-skip", "is_a", "Task")

        decl = mcp_server.tool_declare_intent(**_b3_args(cause_id="TASK-b4-rated-skip"))
        assert decl.get("success") is not False, decl
        result = _finalize_minimal(mcp_server, "b4-task-rated-skip")
        assert result.get("success") is True, result

        sid = mcp_server._STATE.session_id
        assert "TASK-b4-rated-skip" not in intent._rated_user_contexts_for(sid), (
            "Task cause must NOT enter rated_user_contexts; the first-rater "
            "rule scopes to user_context kind only"
        )


class TestB4ProtocolMentionsUserIntentTier:
    """the wake-up protocol prose teaches the new tier."""

    def test_protocol_has_user_intent_tier_section(self, tmp_path, monkeypatch):
        from mempalace import mcp_server

        proto = mcp_server.PALACE_PROTOCOL
        assert "USER-INTENT TIER" in proto
        assert "mempalace_declare_user_intents" in proto
        assert "cause_id" in proto


pytestmark = pytest.mark.integration
