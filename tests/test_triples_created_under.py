"""Triple-layer provenance via creation_context_id column.

Triples aren't first-class entities (no kind='triple' entity row), so
the memory/entity ``created_under`` edge pattern doesn't apply to
them. Instead, ``tool_kg_add`` stamps the ``active_context_id`` onto
``triples.creation_context_id`` (already a column since migration
007); ``kg.triples_created_under(ctx_id)`` is the retrieval path.
"""

from __future__ import annotations

from tests.test_intent_system import _patch_mcp_for_intents, _TEST_BUDGET


def _declare(mcp):
    return mcp.tool_declare_intent(
        intent_type="research",
        slots={"subject": ["test_target"]},
        context={
            "queries": ["exercise triples_created_under provenance", "verify column populated"],
            "keywords": ["triple", "provenance"],
            "entities": ["test_target"],
        },
        agent="test_agent",
        budget=_TEST_BUDGET,
    )


def test_kg_add_stamps_creation_context_from_active_context(monkeypatch, config, kg, palace_path):
    mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
    kg.add_entity("context", kind="class", description="ctx root", importance=5)
    kg.add_entity("related_to", kind="predicate", description="generic link", importance=3)
    kg.add_entity("alpha", kind="entity", description="a", importance=3)
    kg.add_entity("beta", kind="entity", description="b", importance=3)
    # Declared entities cache is consulted by tool_kg_add's permission check.
    mcp._STATE.declared_entities.update({"alpha", "beta", "related_to"})

    _declare(mcp)
    ctx_id = mcp._STATE.active_intent["active_context_id"]
    assert ctx_id

    result = mcp.tool_kg_add(
        subject="alpha",
        predicate="related_to",
        object="beta",
        context={
            "queries": ["alpha relates to beta", "ensure prov column lands"],
            "keywords": ["alpha", "beta"],
            "entities": ["alpha", "beta"],
        },
        agent="test_agent",
        statement="alpha is related to beta (provenance test).",
    )
    assert result["success"] is True

    triple_id = result["triple_id"]
    conn = kg._conn()
    row = conn.execute(
        "SELECT creation_context_id FROM triples WHERE id=?", (triple_id,)
    ).fetchone()
    assert row is not None
    assert row[0] == ctx_id, f"expected creation_context_id={ctx_id}, got {row[0]}"

    # triples_created_under returns this triple id back.
    tids = kg.triples_created_under(ctx_id)
    assert triple_id in tids


def test_triples_created_under_returns_empty_for_unknown_context(kg):
    kg.seed_ontology()
    assert kg.triples_created_under("no_such_context") == []
    assert kg.triples_created_under("") == []


def test_triples_created_under_ignores_invalidated_rows(kg):
    kg.seed_ontology()
    kg.add_entity("ctx_x", kind="context", description="ctx_x", importance=3)
    kg.add_entity("s", kind="entity", description="s", importance=3)
    kg.add_entity("o", kind="entity", description="o", importance=3)
    kg.add_entity("relates_to", kind="predicate", description="generic", importance=3)

    triple_id = kg.add_triple(
        "s",
        "relates_to",
        "o",
        creation_context_id="ctx_x",
        statement="s relates_to o",
    )
    assert triple_id in kg.triples_created_under("ctx_x")

    # Invalidate it — triples_created_under drops it from the result.
    kg.invalidate("s", "relates_to", "o", ended="2026-04-22")
    assert triple_id not in kg.triples_created_under("ctx_x")
