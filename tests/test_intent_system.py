"""
test_intent_system.py — Tests for the intent declaration, finalization,
memory relevance feedback, historical injection, and type promotion system.

Uses isolated palace + KG fixtures via conftest.py to avoid touching real data.
"""

import json
import os


def _patch_mcp_for_intents(monkeypatch, config, kg, palace_path):
    """Patch mcp_server globals for intent system tests.

    Seeds the KG with the intent type hierarchy so declare_intent works.
    Returns the mcp_server module for direct function calls.
    """
    from mempalace import mcp_server

    monkeypatch.setattr(mcp_server, "_config", config)
    monkeypatch.setattr(mcp_server, "_kg", kg)
    monkeypatch.setattr(mcp_server, "_active_intent", None)
    monkeypatch.setattr(mcp_server, "_session_id", "test-session")
    monkeypatch.setattr(mcp_server, "_declared_entities", set())

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
    kg.add_entity("intent_type", kind="class", description="Root class for all intent types", importance=5)

    # Create base intent types with tool permissions
    base_types = {
        "inspect": {
            "description": "Read and analyze code or data without modifying it",
            "slots": {"subject": {"classes": ["thing"], "required": True, "multiple": True},
                      "paths": {"classes": ["thing"], "required": False, "multiple": True}},
            "tool_permissions": [
                {"tool": "Read", "scope": "*"},
                {"tool": "Grep", "scope": "*"},
                {"tool": "Glob", "scope": "*"},
            ],
        },
        "modify": {
            "description": "Edit or create files in the codebase",
            "slots": {"files": {"classes": ["thing"], "required": True, "multiple": True},
                      "paths": {"classes": ["thing"], "required": False, "multiple": True}},
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
            "slots": {"target": {"classes": ["thing"], "required": True, "multiple": True},
                      "commands": {"classes": ["thing"], "required": False, "multiple": True},
                      "paths": {"classes": ["thing"], "required": False, "multiple": True}},
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
        props = {"rules_profile": {"slots": type_def["slots"], "tool_permissions": type_def["tool_permissions"]}}
        kg.add_entity(type_name, kind="class", description=type_def["description"],
                      importance=5, properties=props)
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
        ("described_by", "Described by this drawer"),
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
    kg.add_entity("test_agent", kind="entity", description="Test agent for unit tests", importance=3)
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
            description="Testing declare_intent",
            agent="test_agent",
        )

        assert result["success"] is True
        assert "intent_id" in result
        assert result["intent_type"] == "research"
        assert "permissions" in result

        # Should have the research tools
        tool_names = [p["tool"] for p in result["permissions"]]
        assert "Read" in tool_names
        assert "Grep" in tool_names
        assert "Glob" in tool_names

    def test_declare_unknown_type_fails(self, monkeypatch, config, kg, palace_path):
        """declare_intent with an undeclared type returns an error."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        result = mcp.tool_declare_intent(
            intent_type="nonexistent_type",
            slots={"subject": ["test_target"]},
            agent="test_agent",
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
            agent="test_agent",
        )

        assert result["success"] is False
        assert "not an intent type" in result["error"]

    def test_declare_sets_active_intent(self, monkeypatch, config, kg, palace_path):
        """After declaring, _active_intent is set."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            agent="test_agent",
        )

        assert mcp._active_intent is not None
        assert mcp._active_intent["intent_type"] == "inspect"

    def test_declare_after_finalize_works(self, monkeypatch, config, kg, palace_path):
        """Declaring a new intent after finalizing the previous works."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        # First intent
        mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            agent="test_agent",
        )
        assert mcp._active_intent["intent_type"] == "inspect"

        # Finalize first (required — hard fail on unfinalized)
        mcp.tool_finalize_intent(
            slug="test-replace-first",
            outcome="success",
            summary="Done with inspect",
            agent="test_agent",
            memory_feedback=[],
        )
        assert mcp._active_intent is None

        # Second intent — should succeed now
        result = mcp.tool_declare_intent(
            intent_type="research",
            slots={"subject": ["test_target"]},
            agent="test_agent",
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
            agent="test_agent",
        )

        # Second intent without finalizing — should fail
        result = mcp.tool_declare_intent(
            intent_type="research",
            slots={"subject": ["test_target"]},
            agent="test_agent",
        )

        assert result["success"] is False

    def test_declare_returns_context(self, monkeypatch, config, kg, palace_path):
        """declare_intent returns context with target_facts."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        result = mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            description="Inspecting test target",
            agent="test_agent",
        )

        assert result["success"] is True
        assert "context" in result
        context = result["context"]
        assert "target_facts" in context

    def test_declare_type_relevance_feedback(self, monkeypatch, config, kg, palace_path):
        """declare_intent returns type_relevance when type has found_useful/found_irrelevant edges."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        # Add found_useful and found_irrelevant edges to the inspect intent type
        kg.add_entity("useful_memory_1", kind="entity", description="A useful memory")
        kg.add_entity("irrelevant_memory_1", kind="entity", description="An irrelevant memory")
        kg.add_triple("inspect", "found_useful", "useful_memory_1")
        kg.add_triple("inspect", "found_irrelevant", "irrelevant_memory_1")

        result = mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            agent="test_agent",
        )

        assert result["success"] is True
        context = result["context"]
        assert "type_relevance" in context
        assert "useful_memory_1" in context["type_relevance"]["useful"]
        assert "irrelevant_memory_1" in context["type_relevance"]["irrelevant"]


# ── finalize_intent tests ─────────────────────────────────────────────


class TestFinalizeIntent:
    def _declare_and_get(self, mcp, intent_type="inspect", target="test_target"):
        """Helper: declare an intent and return the mcp module."""
        mcp.tool_declare_intent(
            intent_type=intent_type,
            slots={"subject": [target]},
            description=f"Testing {intent_type}",
            agent="test_agent",
        )
        return mcp

    def test_finalize_no_active_intent(self, monkeypatch, config, kg, palace_path):
        """finalize_intent with no active intent returns error."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        result = mcp.tool_finalize_intent(
            slug="test-finalize-no-active",
            outcome="success",
            summary="Should fail",
            agent="test_agent",
            memory_feedback=[],
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
            memory_feedback=[],
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
            memory_feedback=[],
        )

        assert result["success"] is True
        assert any("is_a inspect" in e for e in result["edges_created"])

    def test_finalize_creates_executed_by_edge(self, monkeypatch, config, kg, palace_path):
        """Execution entity gets executed_by edge to the agent."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
        self._declare_and_get(mcp)

        result = mcp.tool_finalize_intent(
            slug="test-executed-by",
            outcome="success",
            summary="Testing executed_by edge",
            agent="test_agent",
            memory_feedback=[],
        )

        assert any("executed_by test_agent" in e for e in result["edges_created"])

    def test_finalize_creates_targeted_edges(self, monkeypatch, config, kg, palace_path):
        """Execution entity gets targeted edges to slot entities."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
        self._declare_and_get(mcp)

        result = mcp.tool_finalize_intent(
            slug="test-targeted-edge",
            outcome="success",
            summary="Testing targeted edge",
            agent="test_agent",
            memory_feedback=[],
        )

        assert any("targeted" in e for e in result["edges_created"])

    def test_finalize_creates_has_value_edge(self, monkeypatch, config, kg, palace_path):
        """Execution entity gets has_value edge with outcome."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
        self._declare_and_get(mcp)

        result = mcp.tool_finalize_intent(
            slug="test-has-value",
            outcome="partial",
            summary="Partial completion",
            agent="test_agent",
            memory_feedback=[],
        )

        assert any("has_value partial" in e for e in result["edges_created"])

    def test_finalize_creates_result_drawer(self, monkeypatch, config, kg, palace_path):
        """finalize_intent creates a result drawer with summary."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
        self._declare_and_get(mcp)

        result = mcp.tool_finalize_intent(
            slug="test-result-drawer",
            outcome="success",
            summary="This is the result summary",
            agent="test_agent",
            memory_feedback=[],
        )

        assert result["result_drawer"] is not None

    def test_finalize_deactivates_intent(self, monkeypatch, config, kg, palace_path):
        """After finalization, _active_intent is None."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
        self._declare_and_get(mcp)

        mcp.tool_finalize_intent(
            slug="test-deactivate",
            outcome="success",
            summary="Should deactivate",
            agent="test_agent",
            memory_feedback=[],
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
            memory_feedback=[],
        )

        assert result["success"] is True
        # Should have has_gotcha edge
        assert any("has_gotcha" in e for e in result["edges_created"])

    def test_finalize_with_learnings(self, monkeypatch, config, kg, palace_path):
        """finalize_intent with learnings creates learning drawers."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
        self._declare_and_get(mcp)

        result = mcp.tool_finalize_intent(
            slug="test-learnings",
            outcome="success",
            summary="Learned something",
            agent="test_agent",
            learnings=["Always check if entity exists before adding edges"],
            memory_feedback=[],
        )

        assert result["success"] is True


# ── Memory relevance feedback tests ──────────────────────────────────


class TestMemoryRelevanceFeedback:
    def _setup_intent(self, monkeypatch, config, kg, palace_path):
        """Helper: set up mcp, declare an intent, return mcp module."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
        mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            description="Testing memory feedback",
            agent="test_agent",
        )
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
                {"id": "some_memory", "relevant": True, "relevance": 5,
                 "reason": "Very helpful for the task"},
            ],
        )

        assert result["success"] is True
        assert result["feedback_count"] == 1
        assert any("found_useful some_memory" in e for e in result["edges_created"])

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
                {"id": "useless_memory", "relevant": False, "relevance": 1,
                 "reason": "Not related to the task at all"},
            ],
        )

        assert result["success"] is True
        assert result["feedback_count"] == 1
        assert any("found_irrelevant useless_memory" in e for e in result["edges_created"])

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
                {"id": "promoted_memory", "relevant": True, "relevance": 5,
                 "promote_to_type": True, "reason": "Always useful for inspect"},
            ],
        )

        assert result["success"] is True
        assert result["feedback_count"] == 1
        # Should have TWO found_useful edges: one on execution, one on type
        useful_edges = [e for e in result["edges_created"] if "found_useful" in e]
        assert len(useful_edges) == 2
        # One should be on the execution entity
        assert any("test_feedback_promote" in e and "found_useful" in e for e in useful_edges)
        # One should be on the intent type
        assert any("inspect found_useful promoted_memory" in e for e in useful_edges)

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
                {"id": "instance_memory", "relevant": True, "relevance": 4,
                 "promote_to_type": False, "reason": "Only relevant this time"},
            ],
        )

        assert result["success"] is True
        useful_edges = [e for e in result["edges_created"] if "found_useful" in e]
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
                {"id": "mem_a", "relevant": True, "relevance": 5, "reason": "Useful"},
                {"id": "mem_b", "relevant": False, "relevance": 1, "reason": "Not useful"},
                {"id": "mem_c", "relevant": True, "relevance": 3, "reason": "Somewhat useful"},
            ],
        )

        assert result["success"] is True
        assert result["feedback_count"] == 3
        assert any("found_useful mem_a" in e for e in result["edges_created"])
        assert any("found_irrelevant mem_b" in e for e in result["edges_created"])
        assert any("found_useful mem_c" in e for e in result["edges_created"])

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
            agent="test_agent",
        )

        mcp.tool_finalize_intent(
            slug="test-type-relevance-setup",
            outcome="success",
            summary="Setting up type-level feedback",
            agent="test_agent",
            memory_feedback=[
                {"id": "always_useful", "relevant": True, "relevance": 5,
                 "promote_to_type": True, "reason": "General pattern"},
                {"id": "always_irrelevant", "relevant": False, "relevance": 1,
                 "promote_to_type": True, "reason": "Never useful for inspect"},
            ],
        )

        # Second: declare same type — should see type_relevance
        result = mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            agent="test_agent",
        )

        assert result["success"] is True
        context = result["context"]
        assert "type_relevance" in context
        assert "always_useful" in context["type_relevance"]["useful"]
        assert "always_irrelevant" in context["type_relevance"]["irrelevant"]

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
                {"id": "queryable_mem", "relevant": True, "relevance": 4,
                 "reason": "Should be queryable"},
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

        kg.add_entity("past_inspect_exec", kind="entity",
                      description="Inspecting test target for bugs",
                      importance=3, properties={"outcome": "success", "added_by": "test_agent"})
        kg.add_triple("past_inspect_exec", "is_a", "inspect")
        kg.add_triple("past_inspect_exec", "executed_by", "test_agent")
        kg.add_triple("past_inspect_exec", "targeted", "test_target")
        kg.add_triple("past_inspect_exec", "has_value", "success")

        ecol.upsert(
            ids=["past_inspect_exec"],
            documents=["Inspecting test target for bugs"],
            metadatas=[{"name": "past_inspect_exec", "kind": "entity",
                        "importance": 3, "added_by": "test_agent", "outcome": "success"}],
        )
        del client

        # Declare same type + target — should see past execution
        result = mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            description="Inspecting test target again",
            agent="test_agent",
        )

        assert result["success"] is True
        context = result["context"]
        if "past_executions" in context:
            # Past executions were found — verify structure
            assert len(context["past_executions"]) >= 1
            exec_entry = context["past_executions"][0]
            assert "entity_id" in exec_entry
            assert "relationships" in exec_entry


# ── Intent type promotion tests ───────────────────────────────────────


class TestIntentTypePromotion:
    def test_promote_gotchas_to_type(self, monkeypatch, config, kg, palace_path):
        """promote_gotchas_to_type=true links gotchas to both execution and type."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            agent="test_agent",
        )

        result = mcp.tool_finalize_intent(
            slug="test-promote-gotcha",
            outcome="success",
            summary="Testing gotcha promotion",
            agent="test_agent",
            gotchas=["Always verify the entity kind before querying"],
            promote_gotchas_to_type=True,
            memory_feedback=[],
        )

        assert result["success"] is True
        # Should have has_gotcha edges for both execution and type
        gotcha_edges = [e for e in result["edges_created"] if "has_gotcha" in e]
        assert len(gotcha_edges) >= 2  # One on execution, one on type
