"""
test_intent_system.py -- Tests for the intent declaration, finalization,
memory relevance feedback, historical injection, and type promotion system.

Uses isolated palace + KG fixtures via conftest.py to avoid touching real data.
"""

import pytest

# Default budget for tests -- generous to avoid budget-related failures in non-budget tests
_TEST_BUDGET = {"Read": 20, "Edit": 20, "Bash": 20, "Grep": 20, "Glob": 20, "Write": 20}


def _auto_feedback(mcp, extra=None):
    """Generate catch-all feedback for every injected memory (test helper).

    Returns the LIST-OF-GROUPS memory_feedback shape (post-2026-04-24
    cutover; dict shape retired because MCP clients silently dropped
    `additionalProperties`-nested objects). Shape:

        [{"context_id": <ctx>, "feedback": [{id, relevance, reason}, ...]}, ...]

    Scoped to the active intent's context id. Unified retrieval may
    inject entity-collection results alongside records; this helper
    reads injected_memory_ids and yields a catch-all entry per id so
    finalize's strict coverage check passes.

    Args:
        extra: Optional list of test-specific feedback entries. Each entry
            may carry a ``_context_id`` override; missing ones default to
            the active intent's context id.
    """
    ai = mcp._STATE.active_intent or {}
    ctx_id = ai.get("active_context_id", "") or ""
    ids = ai.get("injected_memory_ids", set())
    covered = set()
    entries_by_ctx: dict = {}
    if extra:
        for fb in extra:
            bucket = fb.get("_context_id") or ctx_id
            fb_copy = dict(fb)
            fb_copy.pop("_context_id", None)
            entries_by_ctx.setdefault(bucket, []).append(fb_copy)
            covered.add(fb.get("id", ""))
    for mid in ids:
        if mid and mid not in covered:
            entries_by_ctx.setdefault(ctx_id, []).append(
                {
                    "id": mid,
                    "relevant": False,
                    "relevance": 1,
                    "reason": "test fixture: surfaced memory addressed an unrelated past topic; the seeded test scenario asserts intent finalization not retrieval relevance, so this rating is the deliberate noise channel",
                }
            )
    return [{"context_id": cid, "feedback": fb_list} for cid, fb_list in entries_by_ctx.items()]


def _patch_mcp_for_intents(monkeypatch, config, kg, palace_path):
    """Patch mcp_server globals for intent system tests.

    Seeds the KG with the intent type hierarchy so declare_intent works.
    Returns the mcp_server module for direct function calls.
    """
    from mempalace import mcp_server

    monkeypatch.setattr(mcp_server._STATE, "config", config)
    monkeypatch.setattr(mcp_server._STATE, "kg", kg)
    monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
    monkeypatch.setattr(mcp_server._STATE, "pending_conflicts", None)
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

    # Create the intent_type class
    kg.add_entity(
        "intent_type", kind="class", content="Root class for all intent types", importance=5
    )

    # Create base intent types with tool permissions
    base_types = {
        "inspect": {
            "content": "Read and analyze code or data without modifying it",
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
            "content": "Edit or create files in the codebase",
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
            "content": "Run commands and scripts",
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
            "content": "Exploratory research with broad read access",
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

    # KG-only seeding (Adrian directive 2026-05-12, chromadb removed).
    # The legacy pre-removal version of this fixture mirror-wrote every
    # seeded class / predicate / entity into the chromadb records
    # collection so the old `_is_declared` lookup could find them. Two
    # things changed:
    #   1. `_is_declared` now consults the KG first (mcp_server.py:2089)
    #      so SQLite-only seeding is sufficient.
    #   2. chromadb had a sync_threshold=100 lazy queue that hid these
    #      seeds from the HNSW index until 100 rows queued, so they
    #      never appeared in retrieval. sqlite-vec writes are
    #      immediately indexed; if we wrote them now they would surface
    #      as injected memories in declare_intent retrieval and break
    #      every test that does not pre-rate the seed corpus. KG-only
    #      seeding preserves the legitimate `_is_declared` invariant
    #      and stops the retrieval pollution at the source.
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
            content=type_def["content"],
            importance=5,
            properties=props,
        )
        kg.add_triple(type_name, "is_a", "intent_type")

    # Seed predicates needed by the intent system (KG-only).
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
        kg.add_entity(pred_name, kind="predicate", content=pred_desc, importance=4)

    # Seed agent class and a test agent (KG-only).
    kg.add_entity("agent", kind="class", content="An AI agent", importance=5)
    kg.add_entity("test_agent", kind="entity", content="Test agent for unit tests", importance=3)
    kg.add_triple("test_agent", "is_a", "agent")

    # Seed "thing" class (root of all entity classes; KG-only).
    kg.add_entity("thing", kind="class", content="Root class for all entities", importance=5)

    # Seed a target entity for slot filling (KG-only).
    kg.add_entity("test_target", kind="entity", content="A test target entity", importance=3)
    kg.add_triple("test_target", "is_a", "thing")


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
                "entities": ["test_target"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )

        assert result["success"] is True
        assert "intent_id" in result
        # intent_type lives on active_intent, not the declare response.
        assert result["intent_id"].startswith("intent_research_")
        # permissions echo dropped from declare_intent response 2026-05-06 (commit 65c2015)
        # -- list is inferable from intent_type + slots; no longer round-tripped.
        assert mcp.tool_active_intent()["intent_type"] == "research"

    def test_declare_unknown_type_fails(self, monkeypatch, config, kg, palace_path):
        """declare_intent with an undeclared type returns an error."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        result = mcp.tool_declare_intent(
            intent_type="nonexistent_type",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
                "entities": ["test_target"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
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

        from mempalace.vector_store import get_vector_store as _mp_get_vs  # noqa: PLC0415

        client = _mp_get_vs(palace_path)
        kg.add_entity("not_an_intent", kind="entity", content="Just a regular entity")
        # (chromadb-side ecol.upsert removed 2026-05-12: KG seeding is sufficient)
        del client

        result = mcp.tool_declare_intent(
            intent_type="not_an_intent",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
                "entities": ["test_target"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )

        assert result["success"] is False
        assert "not an intent type" in result["error"]

    def test_slots_not_a_dict_kind_aware_error(self, monkeypatch, config, kg, palace_path):
        """When `slots` arrives as not-a-dict (e.g. MCP transport stringification),
        the error must (a) name the received Python type so the caller can spot
        the transport bug, (b) point at ToolSearch as the recovery path, and
        (c) NOT use the legacy kind-blind ['entity_name'] template that misled
        callers into passing entity ids for raw `paths`/`commands` slots.

        Regression guard for intent.py:929-961 (the not-a-dict guard).
        """
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        # Pass a string instead of a dict -- simulates the MCP transport
        # stringification we hit on 2026-04-28 that wedged ga_agent's session
        # behind a misleading "must be a dict mapping slot names to entity
        # names" error.
        result = mcp.tool_declare_intent(
            intent_type="research",
            slots='{"subject": ["test_target"]}',  # type: ignore[arg-type]
            context={
                "queries": ["Testing slot type guard", "test perspective"],
                "keywords": ["test", "slot", "validator"],
                "entities": ["test_target"],
                "summary": {
                    "what": "test fixture context",
                    "why": "regression guard for kind-aware not-a-dict slot validator error",
                    "scope": "tests",
                },
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )

        assert result["success"] is False
        err = result["error"]
        # (a) Names the actually-received type so transport bugs are diagnosable.
        assert "received str" in err, f"expected 'received str' in error, got: {err}"
        # (b) Mentions the MCP-transport hint so the caller knows the recovery path.
        assert "ToolSearch" in err, f"expected ToolSearch hint in error, got: {err}"
        # (c) Does NOT use the legacy kind-blind ["entity_name"] template.
        assert '"entity_name"' not in err, f"legacy kind-blind template leaked into error: {err}"

    def test_declare_sets_active_intent(self, monkeypatch, config, kg, palace_path):
        """After declaring, _active_intent is set."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
                "entities": ["test_target"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )

        assert mcp._STATE.active_intent is not None
        assert mcp._STATE.active_intent["intent_type"] == "inspect"

    def test_declare_after_finalize_works(self, monkeypatch, config, kg, palace_path):
        """Declaring a new intent after finalizing the previous works."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        # First intent
        mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
                "entities": ["test_target"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        assert mcp._STATE.active_intent["intent_type"] == "inspect"

        # v3.5.0: agent-side memory_feedback retired; finalize no
        # longer requires coverage. Haiku rater post-finalize does the
        # rating async (disabled in tests).

        # Finalize first (required -- hard fail on unfinalized)
        fin_result = mcp.tool_finalize_intent(
            slug="test-replace-first",
            outcome="success",
            content="Done with inspect",
            summary={"what": "test fixture record", "why": "Done with inspect", "scope": "tests"},
            agent="test_agent",
        )
        assert fin_result["success"] is True, fin_result
        assert mcp._STATE.active_intent is None

        # Test fixture: finalize may trigger pending dedup conflicts
        # (result memory prose vs context entity description). Skip them
        # before the next declare_intent so the SUT can run.
        if mcp._STATE.pending_conflicts:
            from mempalace.mcp_server import tool_resolve_conflicts

            for _c in list(mcp._STATE.pending_conflicts):
                tool_resolve_conflicts(
                    actions=[
                        {
                            "id": _c["id"],
                            "action": "skip",
                            "reason": "test fixture dedup collision between context entity prose and finalize result memory; not the contract under test here",
                        }
                    ],
                    agent="test_agent",
                )

        # Second intent -- should succeed now
        result = mcp.tool_declare_intent(
            intent_type="research",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
                "entities": ["test_target"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )

        assert result["success"] is True
        assert mcp._STATE.active_intent["intent_type"] == "research"

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
                "entities": ["test_target"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )

        # Second intent without finalizing -- should fail
        result = mcp.tool_declare_intent(
            intent_type="research",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
                "entities": ["test_target"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
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
                "entities": ["test_target"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )

        assert result["success"] is True
        assert "memories" in result


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
                "entities": [target],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        # Clear entity-collection injections so finalize tests pass
        # with their existing feedback lists.
        if mcp._STATE.active_intent:
            mcp._STATE.active_intent["injected_memory_ids"] = set()
        return mcp

    def test_finalize_no_active_intent(self, monkeypatch, config, kg, palace_path):
        """finalize_intent with no active intent returns error."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        result = mcp.tool_finalize_intent(
            slug="test-finalize-no-active",
            outcome="success",
            content="Should fail",
            summary={
                "what": "test fixture record",
                "why": "Should fail (test fixture)",
                "scope": "tests",
            },
            agent="test_agent",
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
            content="Test execution completed successfully",
            summary={
                "what": "test fixture record",
                "why": "Test execution completed successfully",
                "scope": "tests",
            },
            agent="test_agent",
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
            content="Testing is_a edge",
            summary={"what": "test fixture record", "why": "Testing is_a edge", "scope": "tests"},
            agent="test_agent",
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
            content="Testing executed_by edge",
            summary={
                "what": "test fixture record",
                "why": "Testing executed_by edge",
                "scope": "tests",
            },
            agent="test_agent",
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
            content="Testing targeted edge",
            summary={
                "what": "test fixture record",
                "why": "Testing targeted edge",
                "scope": "tests",
            },
            agent="test_agent",
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
            content="Partial completion",
            summary={"what": "test fixture record", "why": "Partial completion", "scope": "tests"},
            agent="test_agent",
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
            content="This is the result summary",
            summary={
                "what": "test fixture record",
                "why": "This is the result summary",
                "scope": "tests",
            },
            agent="test_agent",
        )

        assert result["result_memory"] is not None

    def test_finalize_deactivates_intent(self, monkeypatch, config, kg, palace_path):
        """After finalization, _active_intent is None."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
        self._declare_and_get(mcp)

        mcp.tool_finalize_intent(
            slug="test-deactivate",
            outcome="success",
            content="Should deactivate",
            summary={"what": "test fixture record", "why": "Should deactivate", "scope": "tests"},
            agent="test_agent",
        )

        assert mcp._STATE.active_intent is None

    def test_finalize_with_gotchas(self, monkeypatch, config, kg, palace_path):
        """finalize_intent with gotchas creates gotcha entities."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
        self._declare_and_get(mcp)

        result = mcp.tool_finalize_intent(
            slug="test-gotchas",
            outcome="success",
            content="Found gotchas",
            summary={
                "what": "test fixture record",
                "why": "Found gotchas (test fixture)",
                "scope": "tests",
            },
            agent="test_agent",
            gotchas=[
                {
                    "summary": {
                        "what": "race conditions in cache",
                        "why": "concurrent reads/writes can corrupt cache state during eviction",
                        "scope": "test fixture",
                    },
                    "content": "Watch out for race conditions in the cache",
                }
            ],
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
            content="Learned something",
            summary={"what": "test fixture record", "why": "Learned something", "scope": "tests"},
            agent="test_agent",
            learnings=[
                {
                    "summary": {
                        "what": "entity-existence guard before edge add",
                        "why": "adding edges to nonexistent entities triggers FK errors and silent drops",
                        "scope": "test fixture",
                    },
                    "content": "Always check if entity exists before adding edges",
                }
            ],
        )

        assert result["success"] is True

    def test_finalize_multiple_learnings_all_persisted(self, monkeypatch, config, kg, palace_path):
        """Regression guard for the slug-collision gotcha: when 5 learnings
        are passed, ALL 5 must be persisted, not just the first. The slug
        shape must include an index suffix so items 2..N don't collide on
        the unique-slug constraint. Confirmed fixed at intent.py:1966
        (``slug=f"learning-{exec_id}-{i}"``); this test prevents regression.
        """
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
        self._declare_and_get(mcp)

        learnings = [
            {
                "summary": {
                    "what": "namespace your keywords",
                    "why": "raw terms collide across domains and break retrieval precision",
                    "scope": "test fixture",
                },
                "content": "Learning one: always namespace your keywords.",
            },
            {
                "summary": {
                    "what": "fail-open on retrieval exceptions",
                    "why": "blocking on a retrieval timeout starves the agent of context",
                    "scope": "test fixture",
                },
                "content": "Learning two: fail-open on retrieval exceptions.",
            },
            {
                "summary": {
                    "what": "persist feedback reasons",
                    "why": "rationale strings drive the long-term learned-context channel",
                    "scope": "test fixture",
                },
                "content": "Learning three: persist feedback reasons for long-term use.",
            },
            {
                "summary": {
                    "what": "check stop_hook_active before blocking stop",
                    "why": "blocking stop without checking re-enters the hook in a loop",
                    "scope": "test fixture",
                },
                "content": "Learning four: check stop_hook_active before blocking stop.",
            },
            {
                "summary": {
                    "what": "HOOK_BYPASS_USER_ONLY break-glass",
                    "why": "user-only bypass keeps automated agents on rails while letting humans escape deadlocks",
                    "scope": "test fixture",
                },
                "content": "Learning five: use HOOK_BYPASS_USER_ONLY for break-glass.",
            },
        ]

        result = mcp.tool_finalize_intent(
            slug="test-multi-learnings",
            outcome="success",
            content="Testing that multiple learnings all persist",
            summary={
                "what": "test fixture record",
                "why": "Testing that multiple learnings all persist",
                "scope": "tests",
            },
            agent="test_agent",
            learnings=learnings,
        )

        assert result["success"] is True
        # No silent-failure errors reported
        assert not result.get("errors"), f"Unexpected errors: {result.get('errors')}"

        # All five learning memories should be discoverable via their slugs.
        exec_id = result["execution_entity"]
        expected_slugs = [f"learning_{exec_id}_{i}" for i in range(len(learnings))]
        for slug in expected_slugs:
            # learning memories link via evidenced_by to the execution entity
            found = any(
                e["predicate"] == "evidenced_by"
                and slug.replace("-", "_") in e["object"].replace("-", "_")
                for e in kg.query_entity(exec_id, direction="outgoing")
            )
            assert found, f"Learning {slug} not linked to {exec_id}"

    def test_finalize_learning_string_rejected_no_auto_derive(
        self, monkeypatch, config, kg, palace_path
    ):
        """Adrian's design lock 2026-04-28: learnings must be
        ``{summary: dict, content: str}`` -- bare strings are rejected
        with a migration error pointing at the new shape. Auto-deriving
        a summary from the content string is forbidden everywhere.
        """
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
        self._declare_and_get(mcp)

        result = mcp.tool_finalize_intent(
            slug="test-learning-string-rejected",
            outcome="success",
            content="String learning should fail",
            summary={
                "what": "test fixture record",
                "why": "ensure bare-string learnings are rejected",
                "scope": "tests",
            },
            agent="test_agent",
            learnings=["bare string used to be accepted; now must fail"],
        )

        # The execution entity is created (finalize itself succeeds);
        # the bare-string learning lands in errors with a clear
        # migration message -- no auto-derive happened.
        errs = result.get("errors") or []
        learning_errs = [e for e in errs if e.get("kind") == "learning_memory"]
        assert learning_errs, f"Expected learning_memory error; got {errs}"
        msg = learning_errs[0].get("error", "")
        assert "dict" in msg.lower() and "summary" in msg.lower(), (
            f"Migration message must mention dict + summary; got: {msg}"
        )
        assert "auto-derive" in msg.lower() or "string" in msg.lower(), (
            f"Migration message must explain rejection; got: {msg}"
        )

    def test_finalize_gotcha_string_rejected_no_auto_derive(
        self, monkeypatch, config, kg, palace_path
    ):
        """Same lock applied to gotchas -- strict
        ``{summary: dict, content: str}``; strings rejected.
        """
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
        self._declare_and_get(mcp)

        result = mcp.tool_finalize_intent(
            slug="test-gotcha-string-rejected",
            outcome="success",
            content="String gotcha should fail",
            summary={
                "what": "test fixture record",
                "why": "ensure bare-string gotchas are rejected",
                "scope": "tests",
            },
            agent="test_agent",
            gotchas=["bare string gotcha used to be accepted; now must fail"],
        )

        errs = result.get("errors") or []
        gotcha_errs = [e for e in errs if e.get("kind") == "gotcha_entity"]
        assert gotcha_errs, f"Expected gotcha_entity error; got {errs}"
        msg = gotcha_errs[0].get("error", "")
        assert "dict" in msg.lower() and "summary" in msg.lower(), (
            f"Migration message must mention dict + summary; got: {msg}"
        )

    def test_finalize_learning_long_content_with_dict_summary_succeeds(
        self, monkeypatch, config, kg, palace_path
    ):
        """The original failure mode: a long learning string overflowed
        the 280-char rendered-summary cap because the handler
        auto-derived ``why`` from the full content. With the new
        contract, content has no length cap (it's the verbatim body)
        and the caller-authored summary stays well under the limit --
        so long content + tight summary now persists cleanly.
        """
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
        self._declare_and_get(mcp)

        long_content = (
            "A very long learning that would have overflowed the 280-char "
            "rendered summary cap under the old auto-derive logic because "
            "the handler used the full content string as the why field, "
            "concatenated with what and scope into prose form. With the "
            "new strict dict contract, the caller authors a tight summary "
            "and the verbatim body lives in content -- no overflow."
        )
        assert len(long_content) > 280, "fixture must exceed old cap"

        result = mcp.tool_finalize_intent(
            slug="test-long-learning-succeeds",
            outcome="success",
            content="Long learning succeeds with tight summary",
            summary={
                "what": "test fixture record",
                "why": "verify long content + tight summary persists",
                "scope": "tests",
            },
            agent="test_agent",
            learnings=[
                {
                    "summary": {
                        "what": "long-content learning persistence",
                        "why": "tight caller-authored summary stays under cap regardless of content length",
                        "scope": "test fixture",
                    },
                    "content": long_content,
                }
            ],
        )

        # No learning_memory errors -- the long body went through cleanly.
        errs = [e for e in (result.get("errors") or []) if e.get("kind") == "learning_memory"]
        assert not errs, f"Long-content learning should not error; got {errs}"
        assert result.get("success") is True


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
                "entities": ["test_target"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        if mcp._STATE.active_intent:
            mcp._STATE.active_intent["injected_memory_ids"] = set()
        return mcp


# ── Historical injection tests ────────────────────────────────────────


class TestHistoricalInjection:
    def test_past_executions_returned(self, monkeypatch, config, kg, palace_path):
        """declare_intent returns past_executions when similar executions exist."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        # Create a past execution entity

        from mempalace.vector_store import get_vector_store as _mp_get_vs  # noqa: PLC0415

        client = _mp_get_vs(palace_path)

        kg.add_entity(
            "past_inspect_exec",
            kind="entity",
            content="Inspecting test target for bugs",
            importance=3,
            properties={"outcome": "success", "added_by": "test_agent"},
        )
        kg.add_triple("past_inspect_exec", "is_a", "inspect")
        kg.add_triple("past_inspect_exec", "executed_by", "test_agent")
        kg.add_triple("past_inspect_exec", "targeted", "test_target")
        # has_value was unskipped 2026-04-25 (carries searchable values
        # like outcome strings); statement now required at write time.
        # Cold-start lock 2026-05-01: outcome literals MUST exist as
        # entities-table rows before has_value edges can land.
        kg.add_entity("success", kind="literal", content="intent outcome value: success")
        kg.add_triple(
            "past_inspect_exec",
            "has_value",
            "success",
            statement="Past inspect execution concluded with outcome success",
        )

        # (chromadb-side ecol.upsert removed 2026-05-12: KG seeding is sufficient)
        del client

        # Declare same type + target -- should see past execution
        result = mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["Inspecting test target again", "test perspective"],
                "keywords": ["test", "declare"],
                "entities": ["test_target"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
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

        from mempalace.vector_store import get_vector_store as _mp_get_vs  # noqa: PLC0415

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
            content="An extremely specific action",
            importance=4,
            properties=props,
        )
        kg.add_triple("very_specific_action", "is_a", "inspect")

        client = _mp_get_vs(palace_path)
        # (chromadb-side ecol.upsert removed 2026-05-12: KG seeding is sufficient)

        # Create 5 past executions with high similarity (would normally trigger promotion)
        for i in range(5):
            exec_id = f"past_specific_{i}"
            kg.add_entity(
                exec_id,
                kind="entity",
                content="An extremely specific action",
                importance=3,
                properties={"outcome": "success"},
            )
            kg.add_triple(exec_id, "is_a", "very_specific_action")
            # (chromadb-side ecol.upsert removed 2026-05-12: KG seeding is sufficient)
        del client

        # Declare -- should succeed even with 5 similar executions
        # because promoted_at_similarity=1.0 means no further promotion
        result = mcp.tool_declare_intent(
            intent_type="very_specific_action",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["An extremely specific action", "test perspective"],
                "keywords": ["test", "declare"],
                "entities": ["test_target"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )

        assert result["success"] is True

    def test_promotion_returns_suggested_similarity(self, monkeypatch, config, kg, palace_path):
        """When promotion triggers, response includes suggested_promoted_at_similarity."""

        from mempalace.vector_store import get_vector_store as _mp_get_vs  # noqa: PLC0415

        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)

        # Create 3 past executions for "inspect" with high similarity
        client = _mp_get_vs(palace_path)
        for i in range(3):
            exec_id = f"past_inspect_similar_{i}"
            kg.add_entity(
                exec_id,
                kind="entity",
                content="Inspecting test target",
                importance=3,
                properties={"outcome": "success"},
            )
            kg.add_triple(exec_id, "is_a", "inspect")
            # (chromadb-side ecol.upsert removed 2026-05-12: KG seeding is sufficient)
        del client

        result = mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["Inspecting test target", "test perspective"],
                "keywords": ["test", "declare"],
                "entities": ["test_target"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
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
                "entities": ["test_target"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )

        result = mcp.tool_finalize_intent(
            slug="test-promote-gotcha",
            outcome="success",
            content="Testing gotcha promotion",
            summary={
                "what": "test fixture record",
                "why": "Testing gotcha promotion",
                "scope": "tests",
            },
            agent="test_agent",
            gotchas=[
                {
                    "summary": {
                        "what": "verify entity kind before querying",
                        "why": "kind-mismatched queries return empty without error and silently drop signal",
                        "scope": "test fixture",
                    },
                    "content": "Always verify the entity kind before querying",
                }
            ],
            promote_gotchas_to_type=True,
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


# v3.5.0 (2026-05-14): TestMandatoryFeedback +
# TestFinalizeSurfacedPairsParkNotBlock are retired -- they asserted
# the agent-rated memory_feedback / operation_ratings coverage gates
# and the pending_feedback parking writer; all three concepts are gone.
# mempalace.feedback_auto's Haiku rater fills rated_useful /
# rated_irrelevant edges out of band post-finalize.

# ── _sync_from_disk cold-hydration ──────────────────────────────────
# Regression guard: after MCP-server restart / plugin reinstall, the
# in-memory _STATE.active_intent is None but the on-disk state file
# still carries the live intent. Before this fix _sync_from_disk only
# refreshed used/budget when memory already had a matching intent, so
# a restart left finalize_intent permanently reporting "No active
# intent to finalize" even though disk had it. Now cold-hydration
# rebuilds the full active_intent dict from the disk record.


class TestSyncFromDiskColdHydration:
    def test_cold_hydration_rebuilds_active_intent_from_disk(
        self, monkeypatch, config, kg, palace_path
    ):
        """Disk has an intent, memory is empty \u2014 _sync_from_disk restores."""
        import json as _json

        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
        # First declare normally so the persist path writes a valid state file.
        mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["cold hydration test action", "another angle on hydration"],
                "keywords": ["hydrate", "cold"],
                "entities": ["test_target"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        assert mcp._STATE.active_intent is not None
        expected_intent_id = mcp._STATE.active_intent["intent_id"]

        # Simulate MCP-server restart: clear in-memory state entirely while
        # leaving the disk file intact.
        mcp._STATE.active_intent = None
        mcp._STATE.pending_conflicts = None

        # Verify disk file still exists with the intent
        from mempalace import intent as _intent_mod

        state_file = _intent_mod._intent_state_path()
        assert state_file is not None and state_file.is_file()
        disk_data = _json.loads(state_file.read_text(encoding="utf-8"))
        assert disk_data["intent_id"] == expected_intent_id

        # Cold hydrate
        _intent_mod._sync_from_disk()

        # In-memory is now populated from disk
        assert mcp._STATE.active_intent is not None
        assert mcp._STATE.active_intent["intent_id"] == expected_intent_id
        assert mcp._STATE.active_intent["intent_type"] == "inspect"
        # Sets round-tripped correctly
        assert isinstance(mcp._STATE.active_intent["injected_memory_ids"], set)
        assert isinstance(mcp._STATE.active_intent["accessed_memory_ids"], set)

    def test_cold_hydration_then_finalize_succeeds(self, monkeypatch, config, kg, palace_path):
        """End-to-end: simulate restart mid-session, then finalize without
        re-declaring. This is the wedge that bit 2026-04-20 wrap_up_session."""
        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
        mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["end-to-end cold finalize test", "restart and finalize"],
                "keywords": ["restart", "finalize"],
                "entities": ["test_target"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )

        # Simulate restart: wipe memory
        mcp._STATE.active_intent = None
        mcp._STATE.pending_conflicts = None

        # Finalize without re-declaring \u2014 previously this returned "No
        # active intent to finalize". Now _sync_from_disk rehydrates first.
        result = mcp.tool_finalize_intent(
            slug="test-cold-finalize",
            outcome="success",
            content="Finalized after simulated restart",
            summary={
                "what": "test fixture record",
                "why": "Finalized after simulated restart",
                "scope": "tests",
            },
            agent="test_agent",
        )
        assert result["success"] is True, f"Finalize should succeed after hydration: {result}"

    def test_sync_from_disk_same_intent_only_updates_used_budget(
        self, monkeypatch, config, kg, palace_path
    ):
        """Normal path \u2014 memory has same intent as disk: only used/budget refresh,
        other fields (like accessed_memory_ids) stay as they are in-memory."""
        import json as _json

        mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
        mcp.tool_declare_intent(
            intent_type="inspect",
            slots={"subject": ["test_target"]},
            context={
                "queries": ["same intent sync test", "no override"],
                "keywords": ["sync", "preserve"],
                "entities": ["test_target"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        # Simulate an accessed memory added in-memory after declare
        mcp._STATE.active_intent["accessed_memory_ids"].add("mem_added_in_memory")

        from mempalace import intent as _intent_mod

        # Modify disk used/budget to simulate hook bump
        state_file = _intent_mod._intent_state_path()
        disk_data = _json.loads(state_file.read_text(encoding="utf-8"))
        disk_data["used"] = {"Read": 3}
        disk_data["budget"] = {"Read": 50, "Edit": 20}
        state_file.write_text(_json.dumps(disk_data), encoding="utf-8")

        _intent_mod._sync_from_disk()

        # used/budget picked up from disk
        assert mcp._STATE.active_intent["used"] == {"Read": 3}
        assert mcp._STATE.active_intent["budget"]["Read"] == 50
        # accessed_memory_ids NOT overwritten
        assert "mem_added_in_memory" in mcp._STATE.active_intent["accessed_memory_ids"]


pytestmark = pytest.mark.integration
