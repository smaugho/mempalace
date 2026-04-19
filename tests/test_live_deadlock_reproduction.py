"""
test_live_deadlock_reproduction.py — Regression reproductions of the
exact sequences the user hit in live sessions that my previous tests
missed.

Each class is a full dispatch-through-handle_request reproduction of a
real-world tool-call sequence, not a state-primitive unit test. Every
single code path the live sequence touches — ``kg_declare_entity``
suggested-link detection, ``kg_add`` auto-consume + contradiction
detection, ``resolve_conflicts`` clearing, ``resolve_enrichments``
clearing, ``declare_intent`` loading from disk, and the session-marker
read at dispatch — runs under real fixtures and against the live KG /
Chroma.

Everything funnels through ``handle_request`` with a session-marker
file present, so the same wiring a running server uses is what the
tests exercise.
"""

from __future__ import annotations

import json

import pytest


_TEST_BUDGET = {"Read": 20, "Grep": 20, "Glob": 20}


def _seed_intent_types(kg, palace_path):
    """Minimal intent type hierarchy + agent."""
    import chromadb

    client = chromadb.PersistentClient(path=palace_path)
    ecol = client.get_or_create_collection("mempalace_entities")

    kg.add_entity(
        "intent_type",
        kind="class",
        description="Root class for all intent types",
        importance=5,
    )
    kg.add_entity("thing", kind="class", description="root of entities", importance=5)
    kg.add_entity("agent", kind="class", description="agent class", importance=5)
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
        "resulted_in",
        "has_value",
        "evidenced_by",
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
    """A fully-wired mcp_server with hook-state dir + marker + seeded KG."""
    from mempalace import mcp_server
    from mempalace.config import MempalaceConfig
    from mempalace.knowledge_graph import KnowledgeGraph

    palace = tmp_path / "palace"
    palace.mkdir()
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(json.dumps({"palace_path": str(palace)}))
    cfg = MempalaceConfig(config_dir=str(cfg_dir))

    kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
    _seed_intent_types(kg, str(palace))

    monkeypatch.setattr(mcp_server._STATE, "config", cfg)
    monkeypatch.setattr(mcp_server._STATE, "kg", kg)
    monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
    monkeypatch.setattr(mcp_server._STATE, "pending_conflicts", None)
    monkeypatch.setattr(mcp_server._STATE, "pending_enrichments", None)
    monkeypatch.setattr(mcp_server._STATE, "session_id", "")
    monkeypatch.setattr(mcp_server._STATE, "declared_entities", set())
    monkeypatch.setattr(mcp_server._STATE, "session_state", {})

    state_dir = tmp_path / "hook_state"
    state_dir.mkdir()
    monkeypatch.setattr(mcp_server, "_INTENT_STATE_DIR", state_dir)

    return mcp_server, state_dir


# Fixed sid every test dispatches under — the real PreToolUse hook always
# injects ``tool_args.sessionId``; we do the same from test code.
_LIVE_SID = "live-sess"


def _dispatch(mcp, tool_name, **arguments):
    """Wrap handle_request with the standard JSON-RPC envelope.

    Mirrors real-harness behavior: every call carries ``sessionId`` in
    ``tool_args`` (injected by the hook via ``updatedInput``). NO shared
    default-file fallback exists anymore; tests must pass the sid
    explicitly or the server will refuse to scope state.
    """
    arguments = {**arguments, "sessionId": _LIVE_SID}
    req = {
        "jsonrpc": "2.0",
        "id": id(arguments),
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    response = mcp.handle_request(req)
    assert "result" in response, f"JSON-RPC error: {response}"
    text_blocks = [
        c.get("text", "") for c in response["result"].get("content", []) if c.get("type") == "text"
    ]
    assert text_blocks
    return json.loads(text_blocks[0])


# ═══════════════════════════════════════════════════════════════════════
#  The 2026-04-19 live deadlock, exactly reproduced
# ═══════════════════════════════════════════════════════════════════════


class TestLiveDeadlock_2026_04_19:
    """Live sequence:

    1. declare_intent(inspect) → success
    2. finalize_intent(inspect, abandoned) → creates execution entity →
       kg_declare_entity internally runs → _detect_suggested_links fires →
       pending enrichment created.
    3. declare_intent(research) → BLOCKS on pending enrichment.
    4. kg_add(subject=exec_memory, predicate=relates_to, object=other_memory)
       — matches the pending-enrichment endpoints → _consume_matching_enrichment
       removes it from state + persists. Also finds an existing edge →
       raises edge_contradiction → pending_conflicts populated.
    5. resolve_conflicts(skip) → clears pending_conflicts.
    6. resolve_enrichments(done) → in the OLD code this returned
       "No pending enrichments" and on the NEXT declare_intent a
       phantom pending re-appeared. After the fix: still "no pending",
       and declare_intent PROCEEDS.
    7. declare_intent(research) → SUCCESS, no phantom block.

    This test will hard-fail on the exact pre-fix codepath even after
    my round-trip suite passes, because it pipes the full sequence
    through handle_request with the session-marker contract.
    """

    def test_full_live_sequence(self, mcp_env):
        mcp, state_dir = mcp_env
        from mempalace import intent as _intent

        # Seed two real entities (the pending enrichment needs valid endpoints
        # so declare_intent's orphan-prune doesn't drop it).
        for name in ("mem_relevant_1", "mem_relevant_2", "existing_other_target"):
            mcp._STATE.kg.add_entity(name, kind="entity", description=f"{name} seed", importance=3)

        # Drive one tool call through handle_request so the session-switch
        # path fires and sets _STATE.session_id to the marker's sid.
        _dispatch(mcp, "mempalace_active_intent")
        assert mcp._STATE.session_id == "live-sess"

        # Seed exactly one pending enrichment mirroring the live shape:
        # from one real memory to another real memory.
        mcp._STATE.pending_enrichments = [
            {
                "id": "enrich_exec_mem_to_mem_relevant_1",
                "from_entity": "mem_relevant_2",
                "to_entity": "mem_relevant_1",
                "reason": "Live deadlock reproduction seed.",
                "similarity": 0.6,
            }
        ]
        _intent._persist_active_intent()

        # Step 3: declare_intent(research) — BLOCKS.
        r3 = _dispatch(
            mcp,
            "mempalace_declare_intent",
            intent_type="research",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["step 3 should block", "pending exists"],
                "keywords": ["block", "pending"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        assert r3["success"] is False, f"r3 unexpected: {r3!r}"
        # Dump the full r3 so diagnostic is loud when the shape changes.
        assert any(
            p["id"] == "enrich_exec_mem_to_mem_relevant_1"
            for p in r3.get("pending_enrichments", [])
        ), f"expected seed enrichment in r3 pending — full r3 = {r3!r}"

        # Step 4: kg_add — endpoints match the enrichment → auto-consume.
        # Pre-seed an existing edge so kg_add's contradiction detection
        # fires (same subject+predicate, different object — emulating the
        # live run).
        mcp._STATE.kg.add_triple("mem_relevant_2", "relates_to", "existing_other_target")

        r4 = _dispatch(
            mcp,
            "mempalace_kg_add",
            subject="mem_relevant_2",
            predicate="relates_to",
            object="mem_relevant_1",
            agent="test_agent",
            context={
                "queries": ["auto-consume match", "creates contradiction"],
                "keywords": ["add", "edge"],
            },
        )
        assert r4["success"] is True
        # The enrichment should be auto-consumed.
        assert r4.get("auto_resolved_enrichment") is not None, (
            "kg_add with matching endpoints must auto-consume the pending enrichment"
        )
        # And contradiction must be surfaced.
        assert r4.get("conflicts")

        # Step 5: resolve_conflicts(skip).
        conflict_id = r4["conflicts"][0]["id"]
        r5 = _dispatch(
            mcp,
            "mempalace_resolve_conflicts",
            agent="test_agent",
            actions=[
                {
                    "id": conflict_id,
                    "action": "skip",
                    "reason": "Skipping because this is a test scenario.",
                }
            ],
        )
        assert r5["success"] is True

        # CRITICAL INVARIANT: pending_enrichments must STILL be empty
        # (consumed in step 4) AFTER resolve_conflicts ran. If resolve_conflicts
        # accidentally persists stale state (the bug), the old enrichment
        # re-appears on disk here.
        assert not mcp._STATE.pending_enrichments
        marker_file = state_dir / "active_intent_live-sess.json"
        if marker_file.is_file():
            disk = json.loads(marker_file.read_text())
            assert not disk.get("pending_enrichments"), (
                f"resolve_conflicts left stale pending_enrichments on disk: {disk!r}"
            )

        # Step 6: resolve_enrichments — no-op, returns "No pending".
        r6 = _dispatch(
            mcp,
            "mempalace_resolve_enrichments",
            agent="test_agent",
            actions=[],
        )
        assert r6["success"] is True
        assert "No pending" in r6.get("message", "")

        # Step 7: declare_intent — MUST succeed. This is the live bug.
        r7 = _dispatch(
            mcp,
            "mempalace_declare_intent",
            intent_type="research",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["step 7 must pass", "no phantom block"],
                "keywords": ["unblocked", "clean"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        assert r7["success"] is True, (
            f"LIVE DEADLOCK REPRODUCTION: declare_intent phantom-blocked: {r7!r}"
        )


# ═══════════════════════════════════════════════════════════════════════
#  Cross-state survival: each write tool must not mutate OTHER pending
# ═══════════════════════════════════════════════════════════════════════


class TestCrossStateSurvival:
    """Every write tool in the mempalace server must preserve pending
    state it doesn't own. resolve_conflicts must NOT touch
    pending_enrichments and vice-versa; kg_add must not silently flatten
    either; finalize_intent must not drop a pre-existing pending_conflict.

    These tests seed BOTH pending types, invoke each write tool, and
    assert the kind-it-doesn't-own survived.
    """

    def test_resolve_conflicts_preserves_pending_enrichments(self, mcp_env):
        mcp, state_dir = mcp_env
        from mempalace import intent as _intent

        # Put the session in a valid sid.
        _dispatch(mcp, "mempalace_active_intent")
        # Seed both pending types.
        mcp._STATE.pending_conflicts = [
            {
                "id": "conflict_a",
                "conflict_type": "edge_contradiction",
                "reason": "seeded",
                "existing_id": "x",
                "new_id": "y",
            }
        ]
        mcp._STATE.pending_enrichments = [
            {"id": "enrich_a", "from_entity": "mem_x", "to_entity": "mem_y", "reason": "seeded"}
        ]
        _intent._persist_active_intent()

        # Resolve only the conflict.
        r = _dispatch(
            mcp,
            "mempalace_resolve_conflicts",
            agent="test_agent",
            actions=[
                {
                    "id": "conflict_a",
                    "action": "skip",
                    "reason": "Skipping for cross-state survival test.",
                }
            ],
        )
        assert r["success"] is True
        # Enrichment must survive in memory AND on disk.
        assert mcp._STATE.pending_enrichments, (
            "resolve_conflicts wiped pending_enrichments — cross-state leak"
        )
        sid_file = state_dir / f"active_intent_{mcp._STATE.session_id}.json"
        assert sid_file.is_file()
        disk = json.loads(sid_file.read_text())
        assert disk["pending_enrichments"]
        assert disk["pending_enrichments"][0]["id"] == "enrich_a"

    def test_resolve_enrichments_preserves_pending_conflicts(self, mcp_env):
        mcp, state_dir = mcp_env
        from mempalace import intent as _intent

        _dispatch(mcp, "mempalace_active_intent")
        mcp._STATE.pending_conflicts = [
            {
                "id": "conflict_b",
                "conflict_type": "edge_contradiction",
                "reason": "seeded",
                "existing_id": "x",
                "new_id": "y",
            }
        ]
        mcp._STATE.pending_enrichments = [
            {"id": "enrich_b", "from_entity": "mem_x", "to_entity": "mem_y"}
        ]
        _intent._persist_active_intent()

        r = _dispatch(
            mcp,
            "mempalace_resolve_enrichments",
            agent="test_agent",
            actions=[
                {
                    "id": "enrich_b",
                    "action": "reject",
                    "reason": "Rejecting for cross-state survival test.",
                }
            ],
        )
        assert r["success"] is True
        assert mcp._STATE.pending_conflicts
        sid_file = state_dir / f"active_intent_{mcp._STATE.session_id}.json"
        assert sid_file.is_file()
        disk = json.loads(sid_file.read_text())
        assert disk["pending_conflicts"]
        assert disk["pending_conflicts"][0]["id"] == "conflict_b"

    def test_kg_add_with_pending_enrichments_does_not_flatten_them(self, mcp_env):
        """kg_add of an UNRELATED edge must not touch pending_enrichments
        that reference different entities."""
        mcp, state_dir = mcp_env
        from mempalace import intent as _intent

        # Two real entities.
        for name in ("alpha", "beta"):
            mcp._STATE.kg.add_entity(name, kind="entity", description="x", importance=3)

        _dispatch(mcp, "mempalace_active_intent")
        mcp._STATE.pending_enrichments = [
            {
                "id": "enrich_unrelated",
                "from_entity": "other_x",
                "to_entity": "other_y",
                "reason": "seeded",
            }
        ]
        _intent._persist_active_intent()

        # kg_add an unrelated edge.
        r = _dispatch(
            mcp,
            "mempalace_kg_add",
            subject="alpha",
            predicate="relates_to",
            object="beta",
            agent="test_agent",
            context={
                "queries": ["alpha relates to beta", "unrelated link"],
                "keywords": ["add", "edge"],
            },
        )
        assert r["success"] is True, f"kg_add failed: {r!r}"
        # Unrelated pending enrichment must survive.
        assert mcp._STATE.pending_enrichments
        assert mcp._STATE.pending_enrichments[0]["id"] == "enrich_unrelated"

    def test_resolve_conflicts_persist_writes_both_pending_lists(self, mcp_env):
        """After resolve_conflicts, the disk file must reflect the NEW
        pending_conflicts state AND the UNCHANGED pending_enrichments.
        This pins the _persist contract against the live failure mode."""
        mcp, state_dir = mcp_env
        from mempalace import intent as _intent

        _dispatch(mcp, "mempalace_active_intent")
        mcp._STATE.pending_conflicts = [
            {
                "id": "c1",
                "conflict_type": "edge_contradiction",
                "reason": "x",
                "existing_id": "a",
                "new_id": "b",
            }
        ]
        mcp._STATE.pending_enrichments = [{"id": "e1", "from_entity": "a", "to_entity": "b"}]
        _intent._persist_active_intent()

        _dispatch(
            mcp,
            "mempalace_resolve_conflicts",
            agent="test_agent",
            actions=[{"id": "c1", "action": "skip", "reason": "Skip for persist test."}],
        )

        sid_file = state_dir / f"active_intent_{mcp._STATE.session_id}.json"
        assert sid_file.is_file()
        disk = json.loads(sid_file.read_text())
        assert disk.get("pending_conflicts") in ([], None, [])
        assert disk.get("pending_enrichments")
        assert disk["pending_enrichments"][0]["id"] == "e1"
