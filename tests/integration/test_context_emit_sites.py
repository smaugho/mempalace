"""Context emit sites + created_under writers (Phase 1).

Three emit sites:
  - ``tool_declare_intent``
  - ``tool_declare_operation``
  - ``tool_kg_search``

Each is expected to call ``context_lookup_or_create`` and stash the
returned context id on ``_STATE.active_intent['active_context_id']``
under most-recent-emit-wins precedence. Three writers
(``_add_memory_internal``, ``tool_kg_declare_entity``, and transitively
``tool_kg_add`` via its ``creation_context_id`` column) then reference
the active context in the write path.

This test file verifies the wiring end-to-end against a real KG + real
ChromaDB collection, not at unit level.
"""

from __future__ import annotations

import pytest
from tests.integration.test_intent_system import _patch_mcp_for_intents, _TEST_BUDGET


def _current_active_context_id(mcp_server):
    ai = mcp_server._STATE.active_intent
    if not ai:
        return ""
    return ai.get("active_context_id", "") or ""


def _context_entities(kg):
    return kg.list_entities(status="active", kind="context")


def test_declare_intent_mints_context_entity(monkeypatch, config, kg, palace_path):
    mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
    # Seed the context class + created_under predicate (ontology extras
    # normally land via seed_ontology; the intent fixture seeds a
    # minimal subset so we seed the extras explicitly here).
    kg.add_entity(
        "context",
        kind="class",
        content="retrieval Contexts",
        importance=5,
    )
    before = len(_context_entities(kg))

    result = mcp.tool_declare_intent(
        intent_type="research",
        slots={"subject": ["test_target"]},
        context={
            "queries": ["investigate auth token refresh", "trace the refresh path"],
            "keywords": ["auth", "refresh"],
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

    cid = _current_active_context_id(mcp)
    assert cid, "active_context_id should be set after declare_intent"

    ctx_entity = kg.get_entity(cid)
    assert ctx_entity is not None
    assert ctx_entity["kind"] == "context"

    # Exactly one new context entity materialised.
    assert len(_context_entities(kg)) == before + 1


def test_declare_intent_twice_same_queries_reuses_context(monkeypatch, config, kg, palace_path):
    """Two declare_intent calls with identical Context → reuse same context id."""
    mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
    kg.add_entity("context", kind="class", content="ctx root", importance=5)

    ctx = {
        "queries": ["repeat this exact text two times", "identical perspective"],
        "keywords": ["repeat", "identical"],
        "entities": ["test_target"],
        "summary": {
            "what": "context-reuse test fixture",
            "why": "exercises declare_intent context-id reuse on identical Context dicts",
            "scope": "tests",
        },
    }
    r1 = mcp.tool_declare_intent(
        intent_type="research",
        slots={"subject": ["test_target"]},
        context=ctx,
        agent="test_agent",
        budget=_TEST_BUDGET,
    )
    assert r1["success"] is True
    first_cid = _current_active_context_id(mcp)
    assert first_cid

    # Finalize the first intent so declare_intent-2 is allowed.
    from mempalace.intent import tool_finalize_intent

    tool_finalize_intent(
        slug="r1-wrap",
        outcome="complete",
        content="nothing interesting happened",
        summary={
            "what": "unrelated kitten-naming-fixture",
            "why": "placeholder finalize body deliberately on a totally different topic so dedup does not cross-flag the context entity description",
            "scope": "tests",
        },
        agent="test_agent",
    )

    # Finalize creates a result memory whose embedded prose may collide
    # with the context entity description on dedup distance. The test
    # setup is intentionally trivial; skip any pending conflicts so the
    # second declare_intent (the actual SUT) can run without preflight
    # blockers.
    if mcp._STATE.pending_conflicts:
        from mempalace.mcp_server import tool_resolve_conflicts

        for c in list(mcp._STATE.pending_conflicts):
            tool_resolve_conflicts(
                actions=[
                    {
                        "id": c["id"],
                        "action": "skip",
                        "reason": "test fixture: dedup collision between context entity prose and finalize result memory; not the contract under test here",
                    }
                ],
                agent="test_agent",
            )

    count_between = len(_context_entities(kg))

    r2 = mcp.tool_declare_intent(
        intent_type="research",
        slots={"subject": ["test_target"]},
        context=ctx,
        agent="test_agent",
        budget=_TEST_BUDGET,
    )
    assert r2["success"] is True
    second_cid = _current_active_context_id(mcp)

    assert second_cid == first_cid, "identical context dict should reuse the context id"
    # No new entity was minted on the second call.
    assert len(_context_entities(kg)) == count_between


def test_entity_creation_writes_created_under(monkeypatch, config, kg, palace_path):
    """kg_declare_entity writes (new_entity, created_under, active_context_id)."""
    mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
    kg.add_entity("context", kind="class", content="ctx", importance=5)
    kg.add_entity(
        "created_under",
        kind="predicate",
        content="provenance edge",
        importance=4,
    )

    mcp.tool_declare_intent(
        intent_type="research",
        slots={"subject": ["test_target"]},
        context={
            "queries": ["declare a new entity under an active context", "test provenance edge"],
            "keywords": ["declare", "provenance"],
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
    cid = _current_active_context_id(mcp)
    assert cid

    # Declare a fresh entity inside the intent.
    r = mcp.tool_kg_declare_entity(
        name="p1_provenance_fixture",
        context={
            "queries": [
                "a fixture entity for created_under round-trip",
                "second perspective to satisfy the 2-query minimum",
            ],
            "keywords": ["fixture", "p1"],
            "entities": ["test_target"],
            "summary": {
                "what": "test fixture context",
                "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                "scope": "tests",
            },
        },
        kind="entity",
        importance=3,
        added_by="test_agent",
    )
    assert r.get("success") is True

    out = kg.query_entity("p1_provenance_fixture", direction="outgoing")
    edges = [(e["predicate"], e["object"]) for e in out]
    assert ("created_under", cid) in edges


def test_record_creation_writes_created_under(monkeypatch, config, kg, palace_path):
    """_add_memory_internal writes (record_id, created_under, active_context_id)."""
    mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
    kg.add_entity("context", kind="class", content="ctx", importance=5)
    kg.add_entity(
        "created_under",
        kind="predicate",
        content="provenance edge",
        importance=4,
    )

    mcp.tool_declare_intent(
        intent_type="research",
        slots={"subject": ["test_target"]},
        context={
            "queries": ["record a memory under an active context", "verify memory provenance"],
            "keywords": ["memory", "provenance"],
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
    cid = _current_active_context_id(mcp)
    assert cid

    # Add a memory via the internal path (kind='record' dispatch).
    r = mcp._add_memory_internal(
        content="A fixture memory for created_under round-trip. ~40 chars body.",
        summary={
            "what": "test fixture record",
            "why": "fixture memory for p1 provenance",
            "scope": "tests",
        },
        slug="p1-provenance-memory",
        added_by="test_agent",
        content_type="fact",
        importance=3,
        entity="test_target",
    )
    assert r.get("success") is True
    memory_id = r.get("memory_id")
    assert memory_id

    out = kg.query_entity(memory_id, direction="outgoing")
    edges = [(e["predicate"], e["object"]) for e in out]
    assert ("created_under", cid) in edges


def test_kg_search_updates_active_context(monkeypatch, config, kg, palace_path):
    """kg_search is an emit site -- it overwrites active_context_id."""
    mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
    kg.add_entity("context", kind="class", content="ctx", importance=5)

    mcp.tool_declare_intent(
        intent_type="research",
        slots={"subject": ["test_target"]},
        context={
            "queries": [
                "rust borrow checker lifetime elision",
                "compile-time memory safety guarantees",
            ],
            "keywords": ["rust", "lifetime"],
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
    intent_cid = _current_active_context_id(mcp)
    assert intent_cid

    # Now run a search with disjoint queries -- different semantic domain
    # entirely. Most-recent-emit-wins means active_context_id should
    # switch to the search's context id.
    mcp.tool_kg_search(
        context={
            "queries": [
                "picking tomato varieties for sandy soil gardens",
                "companion planting basil and peppers",
            ],
            "keywords": ["tomato", "gardening"],
            "entities": ["test_target"],
            "summary": {
                "what": "test fixture context",
                "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                "scope": "tests",
            },
        },
        limit=3,
        agent="test_agent",
    )
    search_cid = _current_active_context_id(mcp)
    assert search_cid
    assert search_cid != intent_cid, (
        "kg_search should overwrite active_context_id with its own context id"
    )


pytestmark = pytest.mark.integration
