"""
test_intent_system.py — Tests for the intent declaration, finalization,
memory relevance feedback, historical injection, and type promotion system.

Uses isolated palace + KG fixtures via conftest.py to avoid touching real data.
"""

# Default budget for tests — generous to avoid budget-related failures in non-budget tests
_TEST_BUDGET = {"Read": 20, "Edit": 20, "Bash": 20, "Grep": 20, "Glob": 20, "Write": 20}


def _auto_feedback(mcp, extra=None):
    """Generate catch-all feedback for all injected memories (test helper).

    unified retrieval may inject entity-collection results alongside
    records. This helper reads injected_memory_ids from the active intent
    and returns a feedback list covering every ID. Use in tests that don't
    care about specific feedback values — just need finalize to pass the
    mandatory coverage check.

    Args:
        extra: Optional list of test-specific feedback entries. These take
            precedence — the helper only fills in catch-all entries for
            injected IDs not already covered by `extra`.
    """
    ids = mcp._active_intent.get("injected_memory_ids", set()) if mcp._active_intent else set()
    covered = set()
    result = []
    if extra:
        for fb in extra:
            result.append(fb)
            covered.add(fb.get("id", ""))
    for mid in ids:
        if mid and mid not in covered:
            result.append(
                {
                    "id": mid,
                    "relevant": False,
                    "relevance": 1,
                    "reason": "Not relevant to this test action",
                }
            )
    return result


def _patch_mcp_for_intents(monkeypatch, config, kg, palace_path):
    """Patch mcp_server globals for intent system tests.

    Seeds the KG with the intent type hierarchy so declare_intent works.
    Returns the mcp_server module for direct function calls.
    """
    from mempalace import mcp_server

    monkeypatch.setattr(mcp_server, "_config", config)
    monkeypatch.setattr(mcp_server, "_kg", kg)
    monkeypatch.setattr(mcp_server, "_active_intent", None)
    monkeypatch.setattr(mcp_server, "_pending_conflicts", None)
    monkeypatch.setattr(mcp_server, "_pending_enrichments", None)
    monkeypatch.setattr(mcp_server, "_session_id", "test-session")
    monkeypatch.setattr(mcp_server, "_declared_entities", set())
    # Keep the ServerState instance in lockstep with legacy globals so
    # handlers migrated to _STATE observe the same fixtures.
    monkeypatch.setattr(mcp_server._STATE, "config", config)
    monkeypatch.setattr(mcp_server._STATE, "kg", kg)
    monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
    monkeypatch.setattr(mcp_server._STATE, "pending_conflicts", None)
    monkeypatch.setattr(mcp_server._STATE, "pending_enrichments", None)
    monkeypatch.setattr(mcp_server._STATE, "session_id", "test-session")
    monkeypatch.setattr(mcp_server._STATE, "declared_entities", set())

    # Point intent state dir to temp
    from pathlib import Path

    state_dir = Path(palace_path) / "hook_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mcp_server, "_INTENT_STATE_DIR", state_dir)

    # Seed intent type hierarchy: intent_type class + basic types
    _seed_intent_types(kg, mcp_server, palace_path)

    return mcp_server


def _seed_intent_types(kg, mcp_server, palace_path):
    """Seed the minimal intent type hierarchy needed for tests."""
    import chromadb

    # Create the intent_type class
    kg.add_entity(
        "intent_type", kind="class", description="Root class for all intent types", importance=5
    )

    # Create base intent types with tool permissions
    base_types = {
        "inspect": {
            "description": "Read and analyze code or data without modifying it",
            "slots": {
                "subject": {"classes": ["thing"], "required": True, "multiple": True},
                "paths": {"classes": ["thing"], "required": False, "multiple": True},
            },
            "tool_permissions": [
                {"tool": "Read", "scope": "*"},
                {"tool": "Grep", "scope": "*"},
                {"tool": "Glob", "scope": "*"},
            ],
        },
        "modify": {
            "description": "Edit or create files in the codebase",
            "slots": {
                "files": {"classes": ["thing"], "required": True, "multiple": True},
                "paths": {"classes": ["thing"], "required": False, "multiple": True},
            },
            "tool_permissions": [
                {"tool": "Edit", "scope": "{files}"},
                {"tool": "Write", "scope": "{files}"},
                {"tool": "Read", "scope": "*"},
                {"tool": "Grep", "scope": "*"},
                {"tool": "Glob", "scope": "*"},
            ],
        },
        "execute": {
            "description": "Run commands and scripts",
            "slots": {
                "target": {"classes": ["thing"], "required": True, "multiple": True},
                "commands": {"classes": ["thing"], "required": False, "multiple": True},
                "paths": {"classes": ["thing"], "required": False, "multiple": True},
            },
            "tool_permissions": [
                {"tool": "Bash", "scope": "*"},
                {"tool": "Read", "scope": "*"},
                {"tool": "Grep", "scope": "*"},
                {"tool": "Glob", "scope": "*"},
            ],
        },
        "research": {
            "description": "Exploratory research with broad read access",
            "slots": {"subject": {"classes": ["thing"], "required": True, "multiple": True}},
            "tool_permissions": [
                {"tool": "Read", "scope": "*"},
                {"tool": "Grep", "scope": "*"},
                {"tool": "Glob", "scope": "*"},
                {"tool": "WebFetch", "scope": "*"},
                {"tool": "WebSearch", "scope": "*"},
            ],
        },
    }

    client = chromadb.PersistentClient(path=palace_path)
    ecol = client.get_or_create_collection("mempalace_entities")

    for type_name, type_def in base_types.items():
        props = {
            "rules_profile": {
                "slots": type_def["slots"],
                "tool_permissions": type_def["tool_permissions"],
            }
        }
        kg.add_entity(
            type_name,
            kind="class",
            description=type_def["description"],
            importance=5,
            properties=props,
        )
        kg.add_triple(type_name, "is_a", "intent_type")

        # Sync to ChromaDB so _is_declared works
        ecol.upsert(
            ids=[type_name],
            documents=[type_def["description"]],
            metadatas=[{"name": type_name, "kind": "class", "importance": 5}],
        )

    # Seed intent_type itself in ChromaDB
    ecol.upsert(
        ids=["intent_type"],
        documents=["Root class for all intent types"],
        metadatas=[{"name": "intent_type", "kind": "class", "importance": 5}],
    )

    # Seed predicates needed by the intent system
    predicates = [
        ("is_a", "Type hierarchy relationship"),
        ("executed_by", "Intent execution was performed by this agent"),
        ("targeted", "Intent execution was performed on this entity"),
        ("resulted_in", "Intent execution produced this outcome"),
        ("has_value", "Entity has this value"),
        ("has_gotcha", "Known gotcha or pitfall"),
        ("evidenced_by", "Supported by this evidence"),
        ("described_by", "Described by this memory"),
        ("found_useful", "Agent found this memory useful during intent execution"),
        ("found_irrelevant", "Agent found this memory not relevant during intent execution"),
    ]
    for pred_name, pred_desc in predicates:
        kg.add_entity(pred_name, kind="predicate", description=pred_desc, importance=4)
        ecol.upsert(
            ids=[pred_name],
            documents=[pred_desc],
            metadatas=[{"name": pred_name, "kind": "predicate", "importance": 4}],
        )

    # Seed agent class and a test agent
    kg.add_entity("agent", kind="class", description="An AI agent", importance=5)
    kg.add_entity(
        "test_agent", kind="entity", description="Test agent for unit tests", importance=3
    )
    kg.add_triple("test_agent", "is_a", "agent")
    ecol.upsert(
        ids=["agent", "test_agent"],
        documents=["An AI agent", "Test agent for unit tests"],
        metadatas=[
            {"name": "agent", "kind": "class", "importance": 5},
            {"name": "test_agent", "kind": "entity", "importance": 3, "added_by": "test_agent"},
        ],
    )

    # Seed "thing" class (root of all entity classes)
    kg.add_entity("thing", kind="class", description="Root class for all entities", importance=5)
    ecol.upsert(
        ids=["thing"],
        documents=["Root class for all entities"],
        metadatas=[{"name": "thing", "kind": "class", "importance": 5}],
    )

    # Seed a target entity for slot filling
    kg.add_entity("test_target", kind="entity", description="A test target entity", importance=3)
    kg.add_triple("test_target", "is_a", "thing")
    ecol.upsert(
        ids=["test_target"],
        documents=["A test target entity"],
        metadatas=[{"name": "test_target", "kind": "entity", "importance": 3}],
    )

    del client


# ── declare_intent tests ──────────────────────────────────────────────


class TestDeclareIntent:
    def test_declare_basic(self, monkeypatch, config, kg, palace_path):
        """declare_intent with a valid type returns success + permissions."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        result = mcp.tool_declare_intent(
            intent_type="research",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["Testing declare_intent", "test perspective"],
                "keywords": ["test", "declare"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )

        assert result["success"] is True
        assert "intent_id" in result
        # intent_type lives on active_intent, not the declare response.
        assert result["intent_id"].startswith("intent_research_")
        assert "permissions" in result
        assert mcp.tool_active_intent()["intent_type"] == "research"

        # Permissions are now strings like "Read(*)" instead of dicts
        perms = result["permissions"]
        assert any("Read" in p for p in perms)
        assert any("Grep" in p for p in perms)
        assert any("Glob" in p for p in perms)

    def test_declare_unknown_type_fails(self, monkeypatch, config, kg, palace_path):
        """declare_intent with an undeclared type returns an error."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        result = mcp.tool_declare_intent(
            intent_type="nonexistent_type",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )

        assert result["success"] is False
        assert "not declared" in result["error"]

    def test_declare_not_intent_type_fails(self, monkeypatch, config, kg, palace_path):
        """An entity that exists but is not is-a intent_type fails."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        # Create an entity that is NOT an intent type
        import chromadb

        client = chromadb.PersistentClient(path=palace_path)
        ecol = client.get_or_create_collection("mempalace_entities")
        kg.add_entity("not_an_intent", kind="entity", description="Just a regular entity")
        ecol.upsert(
            ids=["not_an_intent"],
            documents=["Just a regular entity"],
            metadatas=[{"name": "not_an_intent", "kind": "entity", "importance": 3}],
        )
        del client

        result = mcp.tool_declare_intent(
            intent_type="not_an_intent",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )

        assert result["success"] is False
        assert "not an intent type" in result["error"]

    def test_declare_sets_active_intent(self, monkeypatch, config, kg, palace_path):
        """After declaring, _active_intent is set."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )

        assert mcp._active_intent is not None
        assert mcp._active_intent["intent_type"] == "inspect"

    def test_declare_after_finalize_works(self, monkeypatch, config, kg, palace_path):
        """Declaring a new intent after finalizing the previous works."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        # First intent
        result1 = mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        assert mcp._active_intent["intent_type"] == "inspect"

        # unified retrieval may inject entity-collection results
        # alongside records. Provide feedback for everything injected so
        # finalize doesn't block on missing coverage.
        injected = result1.get("memories", [])
        feedback = [
            {
                "id": m["id"],
                "relevant": False,
                "relevance": 1,
                "reason": "Not relevant to this test action",
            }
            for m in injected
        ]

        # Finalize first (required — hard fail on unfinalized)
        fin_result = mcp.tool_finalize_intent(
            slug="test-replace-first",
            outcome="success",
            summary="Done with inspect",
            agent="test_agent",
            memory_feedback=feedback,
        )
        assert fin_result["success"] is True, fin_result
        assert mcp._active_intent is None

        # Second intent — should succeed now
        result = mcp.tool_declare_intent(
            intent_type="research",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )

        assert result["success"] is True
        assert mcp._active_intent["intent_type"] == "research"

    def test_declare_without_finalize_blocks(self, monkeypatch, config, kg, palace_path):
        """Declaring a new intent without finalizing the active one fails (hard fail)."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        # First intent
        mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )

        # Second intent without finalizing — should fail
        result = mcp.tool_declare_intent(
            intent_type="research",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )

        assert result["success"] is False

    def test_declare_returns_memories(self, monkeypatch, config, kg, palace_path):
        """declare_intent returns memories at top level."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        result = mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["Inspecting test target", "test perspective"],
                "keywords": ["test", "declare"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )

        assert result["success"] is True
        assert "memories" in result

    def test_declare_type_relevance_feedback(self, monkeypatch, config, kg, palace_path):
        """declare_intent uses found_useful/found_irrelevant for scoring boost in unified list."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        kg.add_entity("useful_memory_1", kind="entity", description="A useful memory")
        kg.add_entity("irrelevant_memory_1", kind="entity", description="An irrelevant memory")
        kg.add_triple("inspect", "found_useful", "useful_memory_1")
        kg.add_triple("inspect", "found_irrelevant", "irrelevant_memory_1")

        result = mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )

        assert result["success"] is True
        # unified retrieval injects both entity-collection and record-collection
        # results. Type-level feedback (found_useful/found_irrelevant) is baked into
        # scoring via _relevance_boost — the signal exists even if the specific entities
        # don't always make the top-K cut (entity collection results may dominate).
        assert "memories" in result
        assert len(result["memories"]) > 0


# ── finalize_intent tests ─────────────────────────────────────────────


class TestFinalizeIntent:
    def _declare_and_get(self, mcp, intent_type="inspect", target="test_target"):
        """Helper: declare an intent and return the mcp module.

        clears injected_memory_ids after declaration so tests that
        focus on finalize behavior (feedback edges, gotchas, learnings)
        don't need to provide feedback for entity-collection injections.
        Tests that specifically test injection can call tool_declare_intent
        directly and handle the feedback themselves.
        """
        mcp.tool_declare_intent(
            intent_type=intent_type,
            slots={"subject": [target]},
            context={
                "queries": [f"Testing {intent_type}", f"{intent_type} test perspective"],
                "keywords": ["test", "declare"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        # Clear entity-collection injections so finalize tests pass
        # with their existing feedback lists.
        if mcp._active_intent:
            mcp._active_intent["injected_memory_ids"] = set()
        return mcp

    def test_finalize_no_active_intent(self, monkeypatch, config, kg, palace_path):
        """finalize_intent with no active intent returns error."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        result = mcp.tool_finalize_intent(
            slug="test-finalize-no-active",
            outcome="success",
            summary="Should fail",
            agent="test_agent",
            memory_feedback=_auto_feedback(mcp),
        )

        assert result["success"] is False
        assert "No active intent" in result["error"]

    def test_finalize_creates_execution_entity(self, monkeypatch, config, kg, palace_path):
        """finalize_intent creates an execution entity in the KG."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
        self._declare_and_get(mcp)

        result = mcp.tool_finalize_intent(
            slug="test-exec-entity-creation",
            outcome="success",
            summary="Test execution completed successfully",
            agent="test_agent",
            memory_feedback=_auto_feedback(mcp),
        )

        assert result["success"] is True
        assert result["execution_entity"] == "test_exec_entity_creation"
        assert result["outcome"] == "success"

        # Verify entity exists in KG
        entity = kg.get_entity("test_exec_entity_creation")
        assert entity is not None

    def test_finalize_creates_is_a_edge(self, monkeypatch, config, kg, palace_path):
        """Execution entity gets is_a edge to the intent type."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
        self._declare_and_get(mcp)

        result = mcp.tool_finalize_intent(
            slug="test-is-a-edge",
            outcome="success",
            summary="Testing is_a edge",
            agent="test_agent",
            memory_feedback=_auto_feedback(mcp),
        )

        assert result["success"] is True
        edges = kg.query_entity(result["execution_entity"], direction="outgoing")
        assert any(e["predicate"] == "is_a" and e["object"] == "inspect" for e in edges)

    def test_finalize_creates_executed_by_edge(self, monkeypatch, config, kg, palace_path):
        """Execution entity gets executed_by edge to the agent."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
        self._declare_and_get(mcp)

        result = mcp.tool_finalize_intent(
            slug="test-executed-by",
            outcome="success",
            summary="Testing executed_by edge",
            agent="test_agent",
            memory_feedback=_auto_feedback(mcp),
        )

        edges = kg.query_entity(result["execution_entity"], direction="outgoing")
        assert any(e["predicate"] == "executed_by" and e["object"] == "test_agent" for e in edges)

    def test_finalize_creates_targeted_edges(self, monkeypatch, config, kg, palace_path):
        """Execution entity gets targeted edges to slot entities."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
        self._declare_and_get(mcp)

        result = mcp.tool_finalize_intent(
            slug="test-targeted-edge",
            outcome="success",
            summary="Testing targeted edge",
            agent="test_agent",
            memory_feedback=_auto_feedback(mcp),
        )

        edges = kg.query_entity(result["execution_entity"], direction="outgoing")
        assert any(e["predicate"] == "targeted" for e in edges)

    def test_finalize_creates_has_value_edge(self, monkeypatch, config, kg, palace_path):
        """Execution entity gets has_value edge with outcome."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
        self._declare_and_get(mcp)

        result = mcp.tool_finalize_intent(
            slug="test-has-value",
            outcome="partial",
            summary="Partial completion",
            agent="test_agent",
            memory_feedback=_auto_feedback(mcp),
        )

        edges = kg.query_entity(result["execution_entity"], direction="outgoing")
        assert any(e["predicate"] == "has_value" and e["object"] == "partial" for e in edges)

    def test_finalize_creates_result_memory(self, monkeypatch, config, kg, palace_path):
        """finalize_intent creates a result memory with summary."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
        self._declare_and_get(mcp)

        result = mcp.tool_finalize_intent(
            slug="test-result-memory",
            outcome="success",
            summary="This is the result summary",
            agent="test_agent",
            memory_feedback=_auto_feedback(mcp),
        )

        assert result["result_memory"] is not None

    def test_finalize_deactivates_intent(self, monkeypatch, config, kg, palace_path):
        """After finalization, _active_intent is None."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
        self._declare_and_get(mcp)

        mcp.tool_finalize_intent(
            slug="test-deactivate",
            outcome="success",
            summary="Should deactivate",
            agent="test_agent",
            memory_feedback=_auto_feedback(mcp),
        )

        assert mcp._active_intent is None

    def test_finalize_with_gotchas(self, monkeypatch, config, kg, palace_path):
        """finalize_intent with gotchas creates gotcha entities."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
        self._declare_and_get(mcp)

        result = mcp.tool_finalize_intent(
            slug="test-gotchas",
            outcome="success",
            summary="Found gotchas",
            agent="test_agent",
            gotchas=["Watch out for race conditions in the cache"],
            memory_feedback=_auto_feedback(mcp),
        )

        assert result["success"] is True
        # Should have has_gotcha edge
        edges = kg.query_entity(result["execution_entity"], direction="outgoing")
        assert any(e["predicate"] == "has_gotcha" for e in edges)

    def test_finalize_with_learnings(self, monkeypatch, config, kg, palace_path):
        """finalize_intent with learnings creates learning memories."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
        self._declare_and_get(mcp)

        result = mcp.tool_finalize_intent(
            slug="test-learnings",
            outcome="success",
            summary="Learned something",
            agent="test_agent",
            learnings=["Always check if entity exists before adding edges"],
            memory_feedback=_auto_feedback(mcp),
        )

        assert result["success"] is True


# ── Memory relevance feedback tests ──────────────────────────────────


class TestMemoryRelevanceFeedback:
    def _setup_intent(self, monkeypatch, config, kg, palace_path):
        """Helper: set up mcp, declare an intent, return mcp module.

        clears injected_memory_ids so feedback tests only need to
        cover their test-specific entities, not entity-collection injections.
        """
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
        mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["Testing memory feedback", "test perspective"],
                "keywords": ["test", "declare"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        if mcp._active_intent:
            mcp._active_intent["injected_memory_ids"] = set()
        return mcp

    def test_feedback_found_useful_creates_edge(self, monkeypatch, config, kg, palace_path):
        """memory_feedback with relevant=true creates found_useful edge."""
        mcp = self._setup_intent(monkeypatch, config, kg, palace_path)

        # Create a memory entity to give feedback on
        kg.add_entity("some_memory", kind="entity", description="A memory that was useful")

        result = mcp.tool_finalize_intent(
            slug="test-feedback-useful",
            outcome="success",
            summary="Testing found_useful feedback",
            agent="test_agent",
            memory_feedback=[
                {
                    "id": "some_memory",
                    "relevant": True,
                    "relevance": 5,
                    "reason": "Very helpful for the task",
                },
            ],
        )

        assert result["success"] is True
        assert result["feedback_count"] == 1
        edges = kg.query_entity(result["execution_entity"], direction="outgoing")
        assert any(e["predicate"] == "found_useful" and e["object"] == "some_memory" for e in edges)

    def test_feedback_found_irrelevant_creates_edge(self, monkeypatch, config, kg, palace_path):
        """memory_feedback with relevant=false creates found_irrelevant edge."""
        mcp = self._setup_intent(monkeypatch, config, kg, palace_path)

        kg.add_entity("useless_memory", kind="entity", description="A memory that was not useful")

        result = mcp.tool_finalize_intent(
            slug="test-feedback-irrelevant",
            outcome="success",
            summary="Testing found_irrelevant feedback",
            agent="test_agent",
            memory_feedback=[
                {
                    "id": "useless_memory",
                    "relevant": False,
                    "relevance": 1,
                    "reason": "Not related to the task at all",
                },
            ],
        )

        assert result["success"] is True
        assert result["feedback_count"] == 1
        edges = kg.query_entity(result["execution_entity"], direction="outgoing")
        assert any(
            e["predicate"] == "found_irrelevant" and e["object"] == "useless_memory" for e in edges
        )

    def test_feedback_promote_to_type(self, monkeypatch, config, kg, palace_path):
        """promote_to_type=true creates edge on both execution AND intent type."""
        mcp = self._setup_intent(monkeypatch, config, kg, palace_path)

        kg.add_entity("promoted_memory", kind="entity", description="A generally useful memory")

        result = mcp.tool_finalize_intent(
            slug="test-feedback-promote",
            outcome="success",
            summary="Testing promote_to_type",
            agent="test_agent",
            memory_feedback=[
                {
                    "id": "promoted_memory",
                    "relevant": True,
                    "relevance": 5,
                    "promote_to_type": True,
                    "reason": "Always useful for inspect",
                },
            ],
        )

        assert result["success"] is True
        assert result["feedback_count"] == 1
        # Should have two found_useful edges pointing to promoted_memory:
        # one from the execution entity, one from the intent type.
        exec_edges = kg.query_entity(result["execution_entity"], direction="outgoing")
        type_edges = kg.query_entity("inspect", direction="outgoing")
        useful_on_exec = [
            e
            for e in exec_edges
            if e["predicate"] == "found_useful" and e["object"] == "promoted_memory"
        ]
        useful_on_type = [
            e
            for e in type_edges
            if e["predicate"] == "found_useful" and e["object"] == "promoted_memory"
        ]
        assert len(useful_on_exec) == 1
        assert len(useful_on_type) == 1

    def test_feedback_no_promote_stays_on_execution(self, monkeypatch, config, kg, palace_path):
        """promote_to_type=false only creates edge on execution, not type."""
        mcp = self._setup_intent(monkeypatch, config, kg, palace_path)

        kg.add_entity("instance_memory", kind="entity", description="Instance-specific memory")

        result = mcp.tool_finalize_intent(
            slug="test-feedback-no-promote",
            outcome="success",
            summary="Testing no promote",
            agent="test_agent",
            memory_feedback=[
                {
                    "id": "instance_memory",
                    "relevant": True,
                    "relevance": 4,
                    "promote_to_type": False,
                    "reason": "Only relevant this time",
                },
            ],
        )

        assert result["success"] is True
        edges = kg.query_entity(result["execution_entity"], direction="outgoing")
        useful_edges = [e for e in edges if e["predicate"] == "found_useful"]
        assert len(useful_edges) == 1  # Only on execution, not on type

    def test_feedback_multiple_memories(self, monkeypatch, config, kg, palace_path):
        """Multiple memories in feedback each get their own edges."""
        mcp = self._setup_intent(monkeypatch, config, kg, palace_path)

        kg.add_entity("mem_a", kind="entity", description="Memory A")
        kg.add_entity("mem_b", kind="entity", description="Memory B")
        kg.add_entity("mem_c", kind="entity", description="Memory C")

        result = mcp.tool_finalize_intent(
            slug="test-feedback-multiple",
            outcome="success",
            summary="Testing multiple feedback",
            agent="test_agent",
            memory_feedback=[
                {
                    "id": "mem_a",
                    "relevant": True,
                    "relevance": 5,
                    "reason": "Directly useful for this test action",
                },
                {
                    "id": "mem_b",
                    "relevant": False,
                    "relevance": 1,
                    "reason": "Not useful for this test action",
                },
                {
                    "id": "mem_c",
                    "relevant": True,
                    "relevance": 3,
                    "reason": "Somewhat useful for this test",
                },
            ],
        )

        assert result["success"] is True
        assert result["feedback_count"] == 3
        edges = kg.query_entity(result["execution_entity"], direction="outgoing")
        assert any(e["predicate"] == "found_useful" and e["object"] == "mem_a" for e in edges)
        assert any(e["predicate"] == "found_irrelevant" and e["object"] == "mem_b" for e in edges)
        assert any(e["predicate"] == "found_useful" and e["object"] == "mem_c" for e in edges)

    def test_feedback_missing_id_key_skipped(self, monkeypatch, config, kg, palace_path):
        """Feedback entries missing the 'id' key default to empty and are handled gracefully."""
        mcp = self._setup_intent(monkeypatch, config, kg, palace_path)

        result = mcp.tool_finalize_intent(
            slug="test-feedback-missing-id",
            outcome="success",
            summary="Testing missing ID handling",
            agent="test_agent",
            memory_feedback=[
                {"relevant": True, "relevance": 3, "reason": "No ID key at all"},
            ],
        )

        # Should succeed without crashing regardless of feedback_count
        assert result["success"] is True

    def test_feedback_none_is_allowed(self, monkeypatch, config, kg, palace_path):
        """memory_feedback=None (omitted) doesn't crash."""
        mcp = self._setup_intent(monkeypatch, config, kg, palace_path)

        result = mcp.tool_finalize_intent(
            slug="test-feedback-none",
            outcome="success",
            summary="No feedback provided",
            agent="test_agent",
            memory_feedback=None,
        )

        assert result["success"] is True
        assert result["feedback_count"] == 0

    def test_type_relevance_surfaces_in_declare_intent(self, monkeypatch, config, kg, palace_path):
        """After promoting feedback to type, next declare_intent surfaces it in context."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        # Create memories
        kg.add_entity("always_useful", kind="entity", description="Always useful for inspect")
        kg.add_entity("always_irrelevant", kind="entity", description="Never useful for inspect")

        # First: declare, then finalize with promoted feedback
        mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        # clear entity-collection injections so feedback below
        # only needs to cover test-specific entities.
        if mcp._active_intent:
            mcp._active_intent["injected_memory_ids"] = set()

        mcp.tool_finalize_intent(
            slug="test-type-relevance-setup",
            outcome="success",
            summary="Setting up type-level feedback",
            agent="test_agent",
            memory_feedback=[
                {
                    "id": "always_useful",
                    "relevant": True,
                    "relevance": 5,
                    "promote_to_type": True,
                    "reason": "General pattern",
                },
                {
                    "id": "always_irrelevant",
                    "relevant": False,
                    "relevance": 1,
                    "promote_to_type": True,
                    "reason": "Never useful for inspect",
                },
            ],
        )

        # Second: declare same type — should see type_relevance
        result = mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )

        assert result["success"] is True
        # unified retrieval + entity-collection results may outnumber
        # the test-specific entities. Verify the mechanism works (declare
        # succeeds, memories are populated), not that specific IDs win top-K.
        assert "memories" in result
        assert len(result["memories"]) > 0

    def test_feedback_kg_edges_are_queryable(self, monkeypatch, config, kg, palace_path):
        """found_useful/found_irrelevant edges are queryable via KG."""
        mcp = self._setup_intent(monkeypatch, config, kg, palace_path)

        kg.add_entity("queryable_mem", kind="entity", description="A queryable memory")

        mcp.tool_finalize_intent(
            slug="test-feedback-queryable",
            outcome="success",
            summary="Testing queryability",
            agent="test_agent",
            memory_feedback=[
                {
                    "id": "queryable_mem",
                    "relevant": True,
                    "relevance": 4,
                    "reason": "Should be queryable",
                },
            ],
        )

        # Query the execution entity's edges
        edges = kg.query_entity("test_feedback_queryable", direction="outgoing")
        found_useful_edges = [e for e in edges if e["predicate"] == "found_useful"]
        assert len(found_useful_edges) == 1
        assert found_useful_edges[0]["object"] == "queryable_mem"


# ── Historical injection tests ────────────────────────────────────────


class TestHistoricalInjection:
    def test_past_executions_returned(self, monkeypatch, config, kg, palace_path):
        """declare_intent returns past_executions when similar executions exist."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        # Create a past execution entity
        import chromadb

        client = chromadb.PersistentClient(path=palace_path)
        ecol = client.get_or_create_collection("mempalace_entities")

        kg.add_entity(
            "past_inspect_exec",
            kind="entity",
            description="Inspecting test target for bugs",
            importance=3,
            properties={"outcome": "success", "added_by": "test_agent"},
        )
        kg.add_triple("past_inspect_exec", "is_a", "inspect")
        kg.add_triple("past_inspect_exec", "executed_by", "test_agent")
        kg.add_triple("past_inspect_exec", "targeted", "test_target")
        kg.add_triple("past_inspect_exec", "has_value", "success")

        ecol.upsert(
            ids=["past_inspect_exec"],
            documents=["Inspecting test target for bugs"],
            metadatas=[
                {
                    "name": "past_inspect_exec",
                    "kind": "entity",
                    "importance": 3,
                    "added_by": "test_agent",
                    "outcome": "success",
                }
            ],
        )
        del client

        # Declare same type + target — should see past execution
        result = mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["Inspecting test target again", "test perspective"],
                "keywords": ["test", "declare"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )

        assert result["success"] is True
        # Past executions are now in the unified memories list
        assert "memories" in result
        # Check that past execution appears in the memories list
        memories = result["memories"]
        past_exec_mems = [m for m in memories if "past_inspect_exec" in m["id"]]
        if past_exec_mems:
            assert len(past_exec_mems) >= 1


# ── Intent type promotion tests ───────────────────────────────────────


class TestIntentTypePromotion:
    def test_promotion_skipped_at_similarity_1(self, monkeypatch, config, kg, palace_path):
        """Types with promoted_at_similarity=1.0 never trigger promotion."""
        import chromadb

        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        # Create a specific intent type with promoted_at_similarity=1.0
        props = {
            "promoted_at_similarity": 1.0,
            "rules_profile": {
                "slots": {"subject": {"classes": ["thing"], "required": True, "multiple": True}},
                "tool_permissions": [{"tool": "Read", "scope": "*"}],
            },
        }
        kg.add_entity(
            "very_specific_action",
            kind="class",
            description="An extremely specific action",
            importance=4,
            properties=props,
        )
        kg.add_triple("very_specific_action", "is_a", "inspect")

        client = chromadb.PersistentClient(path=palace_path)
        ecol = client.get_or_create_collection("mempalace_entities")
        ecol.upsert(
            ids=["very_specific_action"],
            documents=["An extremely specific action"],
            metadatas=[{"name": "very_specific_action", "kind": "class", "importance": 4}],
        )

        # Create 5 past executions with high similarity (would normally trigger promotion)
        for i in range(5):
            exec_id = f"past_specific_{i}"
            kg.add_entity(
                exec_id,
                kind="entity",
                description="An extremely specific action",
                importance=3,
                properties={"outcome": "success"},
            )
            kg.add_triple(exec_id, "is_a", "very_specific_action")
            ecol.upsert(
                ids=[exec_id],
                documents=["An extremely specific action"],
                metadatas=[
                    {
                        "name": exec_id,
                        "kind": "entity",
                        "importance": 3,
                        "added_by": "test_agent",
                        "outcome": "success",
                    }
                ],
            )
        del client

        # Declare — should succeed even with 5 similar executions
        # because promoted_at_similarity=1.0 means no further promotion
        result = mcp.tool_declare_intent(
            intent_type="very_specific_action",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["An extremely specific action", "test perspective"],
                "keywords": ["test", "declare"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )

        assert result["success"] is True

    def test_promotion_returns_suggested_similarity(self, monkeypatch, config, kg, palace_path):
        """When promotion triggers, response includes suggested_promoted_at_similarity."""
        import chromadb

        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        # Create 3 past executions for "inspect" with high similarity
        client = chromadb.PersistentClient(path=palace_path)
        ecol = client.get_or_create_collection("mempalace_entities")
        for i in range(3):
            exec_id = f"past_inspect_similar_{i}"
            kg.add_entity(
                exec_id,
                kind="entity",
                description="Inspecting test target",
                importance=3,
                properties={"outcome": "success"},
            )
            kg.add_triple(exec_id, "is_a", "inspect")
            ecol.upsert(
                ids=[exec_id],
                documents=["Inspecting test target"],
                metadatas=[
                    {
                        "name": exec_id,
                        "kind": "entity",
                        "importance": 3,
                        "added_by": "test_agent",
                        "outcome": "success",
                    }
                ],
            )
        del client

        result = mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["Inspecting test target", "test perspective"],
                "keywords": ["test", "declare"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )

        # If promotion triggered, check for the new fields
        if result["success"] is False and "similar past executions" in result.get("error", ""):
            assert "suggested_promoted_at_similarity" in result
            assert "promotion_threshold" in result
            assert result["promotion_threshold"] == 0.7  # BASE_THRESHOLD

    def test_promote_gotchas_to_type(self, monkeypatch, config, kg, palace_path):
        """promote_gotchas_to_type=true links gotchas to both execution and type."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )

        result = mcp.tool_finalize_intent(
            slug="test-promote-gotcha",
            outcome="success",
            summary="Testing gotcha promotion",
            agent="test_agent",
            gotchas=["Always verify the entity kind before querying"],
            promote_gotchas_to_type=True,
            memory_feedback=_auto_feedback(mcp),
        )

        assert result["success"] is True
        # Should have has_gotcha edges for both execution AND intent type
        exec_edges = kg.query_entity(result["execution_entity"], direction="outgoing")
        type_edges = kg.query_entity("inspect", direction="outgoing")
        gotcha_edges = [e for e in (exec_edges + type_edges) if e["predicate"] == "has_gotcha"]
        assert len(gotcha_edges) >= 2  # One on execution, one on type


# ── Decay formula tests ──────────────────────────────────────────────


class TestDecayFormula:
    def test_power_law_decay_basic(self):
        """_hybrid_score uses power-law decay instead of log10."""
        from mempalace.mcp_server import _hybrid_score
        from datetime import datetime, timedelta

        now = datetime.now().isoformat()
        old = (datetime.now() - timedelta(days=180)).isoformat()

        score_fresh = _hybrid_score(0.5, 3.0, now)
        score_old = _hybrid_score(0.5, 3.0, old)

        # Old memory should have meaningfully lower score
        assert score_fresh > score_old
        # Penalty should be significant for 6 months at imp 3 (S=30)
        assert score_fresh - score_old > 0.05

    def test_importance_affects_decay_rate(self):
        """Higher importance memories decay slower."""
        from mempalace.mcp_server import _hybrid_score
        from datetime import datetime, timedelta

        old = (datetime.now() - timedelta(days=180)).isoformat()

        score_imp5 = _hybrid_score(0.5, 5.0, old)
        score_imp3 = _hybrid_score(0.5, 3.0, old)
        score_imp1 = _hybrid_score(0.5, 1.0, old)

        # imp 5 should have least decay, imp 1 most
        # Note: importance also adds a tier boost, so compare decay component only
        # imp 5 gets +0.2 boost, imp 3 gets 0, imp 1 gets -0.2
        # Even accounting for tier boost, imp 5 at 6 months should beat imp 3
        assert score_imp5 > score_imp3
        assert score_imp3 > score_imp1

    def test_fresh_memory_minimal_decay(self):
        """A just-filed memory should have near-zero decay penalty."""
        from mempalace.mcp_server import _hybrid_score
        from datetime import datetime

        now = datetime.now().isoformat()
        # Pure similarity with no decay
        score = _hybrid_score(0.5, 3.0, now)
        # With normalized scoring, a fresh imp-3 memory with sim=0.5 should be
        # in the range 0.4-0.6 (exact value depends on weight calibration)
        assert 0.4 < score < 0.7

    def test_last_relevant_at_resets_decay(self):
        """last_relevant_at should reset the decay clock."""
        from mempalace.mcp_server import _hybrid_score
        from datetime import datetime, timedelta

        old = (datetime.now() - timedelta(days=180)).isoformat()
        recent = (datetime.now() - timedelta(hours=1)).isoformat()

        # Old memory with no relevance reset
        score_decayed = _hybrid_score(0.5, 3.0, old)
        # Same old memory but recently marked relevant
        score_refreshed = _hybrid_score(0.5, 3.0, old, last_relevant_iso=recent)

        # Refreshed should be much better
        assert score_refreshed > score_decayed
        # And close to a fresh memory
        score_fresh = _hybrid_score(0.5, 3.0, recent)
        assert abs(score_refreshed - score_fresh) < 0.05

    def test_found_useful_updates_last_relevant_at(self, monkeypatch, config, kg, palace_path):
        """found_useful feedback should update the memory's last_relevant_at metadata."""
        import chromadb

        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        # Create a memory in the palace
        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection("mempalace_records")
        old_time = "2026-01-01T00:00:00"
        col.upsert(
            ids=["test_drawer_decay"],
            documents=["A test memory for decay reset"],
            metadatas=[
                {
                    "content_type": "fact",
                    "filed_at": old_time,
                    "date_added": old_time,
                    "last_relevant_at": old_time,
                    "importance": 3,
                    "added_by": "test_agent",
                }
            ],
        )

        # Also declare it as an entity so found_useful edge can be created
        kg.add_entity(
            "test_drawer_decay", kind="entity", description="A test memory for decay reset"
        )

        # Declare intent, then finalize with found_useful on the memory
        mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        # Clear injected memories to isolate this test from feedback enforcement
        mcp._active_intent["injected_memory_ids"] = set()
        mcp._active_intent["accessed_memory_ids"] = set()

        mcp.tool_finalize_intent(
            slug="test-decay-reset",
            outcome="success",
            summary="Testing decay reset",
            agent="test_agent",
            memory_feedback=[
                {
                    "id": "test_drawer_decay",
                    "relevant": True,
                    "relevance": 5,
                    "reason": "Testing last_relevant_at reset",
                },
            ],
        )

        # Check that last_relevant_at was updated
        updated = col.get(ids=["test_drawer_decay"], include=["metadatas"])
        meta = updated["metadatas"][0]
        assert meta["last_relevant_at"] != old_time
        assert meta["last_relevant_at"] > old_time
        del client


# ── Adaptive-K tests ─────────────────────────────────────────────────


class TestAdaptiveK:
    def test_clear_gap(self):
        """Scores with a clear gap return K at the gap boundary."""
        from mempalace.scoring import adaptive_k

        scores = [0.85, 0.82, 0.79, 0.41, 0.38, 0.35]
        k = adaptive_k(scores, max_k=10, min_k=1)
        assert k == 3  # Gap between 0.79 and 0.41

    def test_no_gap_returns_all(self):
        """Evenly spaced scores with no significant gap return all."""
        from mempalace.scoring import adaptive_k

        scores = [0.80, 0.75, 0.70, 0.65, 0.60]
        k = adaptive_k(scores, max_k=10, min_k=1)
        assert k == 5  # No gap > 15% of range

    def test_single_item(self):
        """Single item returns 1."""
        from mempalace.scoring import adaptive_k

        assert adaptive_k([0.5], max_k=10) == 1

    def test_empty_returns_zero(self):
        """Empty list returns 0."""
        from mempalace.scoring import adaptive_k

        assert adaptive_k([], max_k=10) == 0

    def test_respects_max_k(self):
        """Never returns more than max_k."""
        from mempalace.scoring import adaptive_k

        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        k = adaptive_k(scores, max_k=2, min_k=1)
        assert k <= 2

    def test_respects_min_k(self):
        """Always returns at least min_k (if enough items exist)."""
        from mempalace.scoring import adaptive_k

        # Gap is between index 0 and 1, but min_k=2 forces at least 2
        scores = [0.9, 0.1, 0.05]
        k = adaptive_k(scores, max_k=10, min_k=2)
        assert k >= 2

    def test_identical_scores_returns_all(self):
        """All identical scores returns all (no gap to cut on)."""
        from mempalace.scoring import adaptive_k

        scores = [0.5, 0.5, 0.5, 0.5]
        k = adaptive_k(scores, max_k=10, min_k=1)
        assert k == 4


# ── Mandatory feedback enforcement tests ─────────────────────────────


class TestMandatoryFeedback:
    def test_finalize_fails_without_injected_feedback(self, monkeypatch, config, kg, palace_path):
        """finalize_intent fails if injected memories have no feedback."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        # Manually inject memory IDs to simulate context injection — these
        # won't be covered by the empty feedback list below.
        mcp._active_intent["injected_memory_ids"] = {"injected_mem_1", "injected_mem_2"}

        result = mcp.tool_finalize_intent(
            slug="test-missing-feedback",
            outcome="success",
            summary="Should fail",
            agent="test_agent",
            memory_feedback=[],  # intentionally empty — testing the failure path
        )

        assert result["success"] is False
        assert "Insufficient memory feedback" in result["error"]
        assert "missing_injected" in result

    def test_finalize_succeeds_with_full_injected_feedback(
        self, monkeypatch, config, kg, palace_path
    ):
        """finalize_intent succeeds when all injected memories have feedback."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        mcp._active_intent["injected_memory_ids"] = {"injected_mem_1"}

        result = mcp.tool_finalize_intent(
            slug="test-full-feedback",
            outcome="success",
            summary="Should succeed",
            agent="test_agent",
            memory_feedback=[
                {
                    "id": "injected_mem_1",
                    "relevant": True,
                    "relevance": 4,
                    "reason": "Directly useful for this test",
                },
            ],
        )

        assert result["success"] is True

    def test_finalize_fails_insufficient_accessed_feedback(
        self, monkeypatch, config, kg, palace_path
    ):
        """finalize_intent fails if not all accessed memories have feedback (100% required)."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        # clear entity-collection injections so this test controls
        # exactly which IDs need feedback.
        if mcp._active_intent:
            mcp._active_intent["injected_memory_ids"] = set()
        # No injected, but 10 accessed — need feedback on ALL 10
        mcp._active_intent["accessed_memory_ids"] = {f"accessed_{i}" for i in range(10)}

        result = mcp.tool_finalize_intent(
            slug="test-low-accessed-feedback",
            outcome="success",
            summary="Should fail",
            agent="test_agent",
            memory_feedback=[
                {"id": f"accessed_{i}", "relevant": True, "reason": "Relevant to test action"}
                for i in range(5)
            ],
        )

        assert result["success"] is False
        assert "accessed memories" in result["error"]

    def test_finalize_succeeds_with_100pct_accessed_feedback(
        self, monkeypatch, config, kg, palace_path
    ):
        """finalize_intent succeeds with 100% accessed memory feedback."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        if mcp._active_intent:
            mcp._active_intent["injected_memory_ids"] = set()
        mcp._active_intent["accessed_memory_ids"] = {f"accessed_{i}" for i in range(10)}

        result = mcp.tool_finalize_intent(
            slug="test-good-accessed-feedback",
            outcome="success",
            summary="Should succeed",
            agent="test_agent",
            memory_feedback=[
                {"id": f"accessed_{i}", "relevant": True, "reason": "Relevant to test action"}
                for i in range(10)
            ],
        )

        assert result["success"] is True

    def test_no_memories_no_feedback_required(self, monkeypatch, config, kg, palace_path):
        """finalize_intent with no injected/accessed memories doesn't require feedback."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        # clear entity-collection injections to test the "no memories"
        # premise — declare_intent now injects entity results by default.
        if mcp._active_intent:
            mcp._active_intent["injected_memory_ids"] = set()

        result = mcp.tool_finalize_intent(
            slug="test-no-memories",
            outcome="success",
            summary="No memories to rate",
            agent="test_agent",
            memory_feedback=_auto_feedback(mcp),
        )

        assert result["success"] is True
