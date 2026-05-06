"""Regression locks for Adrian's 2026-05-01 congruence audit.

Adrian flagged four gaps at session-close:

  1. **Vocab incoherence.** The rendered summary prose surfaced under
     three different response keys -- ``description`` (kg_declare_entity,
     kg_search entity hits), ``text`` (declare_intent / declare_user_intents
     memory previews, kg_search RRF channel), and a content/text collision
     in tool_read.py's projector. Renamed uniformly to ``summary_text``
     for memory + entity prose, ``statement_text`` for triple prose.

  2. **link_score is real cosine, not a bucket.** The 1.0 / 0.5 round
     values Adrian observed in similar_contexts entries looked like
     fallback constants. They are not -- they are real accumulated
     similar_to edge confidences. This test asserts the formula returns
     the actual edge confidence (and the geometric product on 2-hop).

  3. **kg_add echoes statement_text.** Pre-fix the response carried
     ``triple_id`` + ``fact`` only; the rendered statement that got
     persisted to the row + embedded into mempalace_triples was not
     echoed back, breaking caller-side verification.

  4. **diary_write registers as entity.** Pre-fix diary_write only
     wrote to the Chroma record collection; the SQLite entities table
     had no row. Feedback edges (rated_useful etc.) targeting a diary
     memory id phantom-rejected at add_rated_edge, blocking
     extend_feedback / finalize_intent coverage when a diary entry
     surfaced as a memory in the active context's neighbourhood.

These tests are the durable contract that the fixes hold.
"""

from __future__ import annotations


import pytest


def _seed_agent(kg, name: str = "ga_agent", *, queries=None, summary=None) -> None:
    """Seed an agent entity with the required is_a agent edge.

    Every kg_add / kg_declare_entity call demands ``added_by`` be an
    entity carrying an ``is_a agent`` edge. This helper gets a test
    set up to that contract in two steps so each regression test can
    focus on the behavior it locks rather than re-stating fixtures.
    """
    from mempalace.entity_gate import mint_entity

    kg.add_entity("agent", kind="class", content="Agent class for is_a")
    mint_entity(
        name,
        kind="entity",
        summary=summary
        or {
            "what": f"{name} test fixture",
            "why": "exists so write tools have a valid added_by attribution",
            "scope": "test_congruence_audit_2026_05_01",
        },
        queries=queries or [f"who is the {name} test fixture", "audit fixture probe"],
        importance=3,
    )
    kg.add_triple(name, "is_a", "agent")


# ── Issue 1: vocab uniformity ────────────────────────────────────────


def test_kg_declare_entity_response_uses_summary_text(monkeypatch, config, palace_path, kg):
    """kg_declare_entity returns the rendered prose under ``summary_text``,
    NOT ``description``. The ``description`` key was Adrian's primary
    flag in the congruence audit because it collided with the entity
    table's ``content`` column name and broke callers that consumed the
    rendered form generically."""
    from mempalace import mcp_server
    from mempalace.tool_mutate import tool_kg_declare_entity

    monkeypatch.setattr(mcp_server._STATE, "kg", kg)
    monkeypatch.setattr(mcp_server._STATE, "config", config)
    monkeypatch.setattr(mcp_server._STATE, "session_id", "test-vocab-uniform")
    monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
    monkeypatch.setattr(mcp_server._STATE, "client_cache", None)
    monkeypatch.setattr(mcp_server._STATE, "collection_cache", None)
    kg.seed_ontology()
    _seed_agent(kg)

    result = tool_kg_declare_entity(
        name="vocab_test_entity",
        kind="entity",
        added_by="ga_agent",
        importance=3,
        context={
            "queries": [
                "vocab uniformity entity for the congruence audit",
                "test that kg_declare_entity emits summary_text",
            ],
            "keywords": ["vocab", "summary_text", "audit"],
            "entities": ["ga_agent"],
            "summary": {
                "what": "vocab_test_entity",
                "why": "regression fixture for the 2026-05-01 vocab rename",
                "scope": "congruence audit",
            },
        },
    )
    assert result.get("success") is True, result
    assert "summary_text" in result, (
        f"kg_declare_entity must emit 'summary_text' under the vocab lock; got keys: {list(result)}"
    )
    assert "description" not in result, (
        "kg_declare_entity must NOT emit 'description' anymore -- "
        "Adrian's vocab audit replaced it with 'summary_text' uniformly."
    )
    assert isinstance(result["summary_text"], str) and result["summary_text"].strip()


# ── Issue 2: link_score is real cosine ──────────────────────────────


def test_link_score_is_real_edge_cosine_not_fallback(tmp_path):
    """walk_rated_neighbourhood's neighbourhood_weights for a 1-hop
    similar_to neighbour equals the actual confidence column on the
    similar_to triple -- NOT a hardcoded 1.0 fallback. Set the edge
    confidence to a deliberately non-round value (0.7321) and assert it
    surfaces verbatim through the link_score channel."""
    from mempalace.knowledge_graph import KnowledgeGraph
    from mempalace.scoring import walk_rated_neighbourhood

    db = tmp_path / "link_score.sqlite3"
    kg = KnowledgeGraph(db_path=str(db))
    kg.seed_ontology()
    # Two context entities + a similar_to edge between them with a
    # non-round confidence so any fallback constant would visibly fail.
    kg.add_entity("ctx_active", kind="entity")
    kg.add_entity("ctx_neighbour", kind="entity")
    NON_ROUND_SIM = 0.7321
    kg.add_triple(
        "ctx_active",
        "similar_to",
        "ctx_neighbour",
        confidence=NON_ROUND_SIM,
    )
    walk = walk_rated_neighbourhood("ctx_active", kg, hops=2, sim_decay=0.5)
    weights = walk["neighbourhood_weights"]
    assert "ctx_active" not in weights, (
        "active context is excluded from neighbourhood_weights by design"
    )
    assert "ctx_neighbour" in weights, "1-hop similar_to neighbour must surface"
    # The 1-hop link_score must equal the edge confidence (no decay
    # applied at depth 0). If the formula falls back to a constant
    # (1.0 / 0.5 etc.) this assertion fails immediately.
    assert abs(weights["ctx_neighbour"] - NON_ROUND_SIM) < 1e-6, (
        f"link_score for 1-hop similar_to neighbour must equal the edge "
        f"confidence ({NON_ROUND_SIM}); got {weights['ctx_neighbour']}. "
        f"If this returns 1.0 or 0.5 the formula has regressed to a "
        f"fallback constant -- see scoring.walk_rated_neighbourhood."
    )


def test_link_score_2_hop_geometric_product(tmp_path):
    """A 2-hop traversal multiplies edge_sim by parent_sim and sim_decay.
    Set both edges to non-round confidences and assert link_score equals
    the geometric product, proving the formula is real arithmetic and
    not bucketed."""
    from mempalace.knowledge_graph import KnowledgeGraph
    from mempalace.scoring import walk_rated_neighbourhood

    db = tmp_path / "link_score_2hop.sqlite3"
    kg = KnowledgeGraph(db_path=str(db))
    kg.seed_ontology()
    kg.add_entity("ctx_a", kind="entity")
    kg.add_entity("ctx_b", kind="entity")
    kg.add_entity("ctx_c", kind="entity")
    SIM_AB = 0.8123
    SIM_BC = 0.6512
    DECAY = 0.5
    kg.add_triple("ctx_a", "similar_to", "ctx_b", confidence=SIM_AB)
    kg.add_triple("ctx_b", "similar_to", "ctx_c", confidence=SIM_BC)
    walk = walk_rated_neighbourhood("ctx_a", kg, hops=2, sim_decay=DECAY)
    weights = walk["neighbourhood_weights"]
    assert "ctx_b" in weights and "ctx_c" in weights
    # 1-hop b: edge_sim only (decay applies starting at depth >= 1).
    assert abs(weights["ctx_b"] - SIM_AB) < 1e-6, weights
    # 2-hop c: parent_sim * edge_sim * decay.
    expected_c = SIM_AB * SIM_BC * DECAY
    assert abs(weights["ctx_c"] - expected_c) < 1e-6, (
        f"2-hop link_score must equal parent_sim * edge_sim * decay "
        f"({SIM_AB} * {SIM_BC} * {DECAY} = {expected_c}); "
        f"got {weights['ctx_c']}."
    )


# ── Issue 3: kg_add echoes statement_text ───────────────────────────


def test_kg_add_response_echoes_statement_text(monkeypatch, config, palace_path, kg):
    """kg_add returns the rendered statement prose under
    ``statement_text`` so callers can verify the form that got
    persisted to the row + embedded into mempalace_triples. Pre-fix
    only ``triple_id`` + ``fact`` came back, making caller-side
    verification impossible."""
    from mempalace import mcp_server
    from mempalace.entity_gate import mint_entity
    from mempalace.tool_mutate import tool_kg_add

    monkeypatch.setattr(mcp_server._STATE, "kg", kg)
    monkeypatch.setattr(mcp_server._STATE, "config", config)
    monkeypatch.setattr(mcp_server._STATE, "session_id", "test-kgadd-echo")
    monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
    monkeypatch.setattr(mcp_server._STATE, "client_cache", None)
    monkeypatch.setattr(mcp_server._STATE, "collection_cache", None)
    kg.seed_ontology()
    _seed_agent(kg)
    mint_entity(
        "person_a",
        kind="entity",
        summary={
            "what": "person_a kg_add subject fixture",
            "why": "exists so the kg_add edge has a real subject node",
            "scope": "kg_add echo test",
        },
        queries=["person_a fixture", "kg_add subject fixture"],
    )
    mint_entity(
        "city_b",
        kind="entity",
        summary={
            "what": "city_b kg_add object fixture",
            "why": "exists so the kg_add edge has a real object node",
            "scope": "kg_add echo test",
        },
        queries=["city_b fixture", "kg_add object fixture"],
    )
    # Need a non-skip-list predicate so statement is required + echoed.
    from mempalace.tool_mutate import tool_kg_declare_entity

    decl = tool_kg_declare_entity(
        name="lives_in",
        kind="predicate",
        added_by="ga_agent",
        importance=3,
        properties={
            "constraints": {
                "subject_kinds": ["entity"],
                "object_kinds": ["entity"],
                "subject_classes": ["thing"],
                "object_classes": ["thing"],
                "cardinality": "many-to-many",
            }
        },
        context={
            "queries": [
                "lives_in predicate for the kg_add echo regression test",
                "geographic relation between a person and a place",
            ],
            "keywords": ["lives_in", "predicate", "geographic"],
            "entities": ["ga_agent"],
            "summary": {
                "what": "lives_in predicate fixture",
                "why": "non-skip-list predicate so kg_add demands a statement and echoes it back",
            },
        },
    )
    assert decl.get("success") is True, decl
    statement_dict = {
        "what": "person_a lives in city_b",
        "why": "primary residence address recorded for the kg_add echo regression",
        "scope": "test fixture",
    }
    result = tool_kg_add(
        subject="person_a",
        predicate="lives_in",
        object="city_b",
        agent="ga_agent",
        statement=statement_dict,
        context={
            "queries": [
                "edge linking person_a to city_b via lives_in",
                "kg_add echo regression test fixture edge",
            ],
            "keywords": ["edge", "kg_add", "echo", "lives_in"],
            "entities": ["person_a", "city_b"],
            "summary": {
                "what": "edge person_a lives_in city_b",
                "why": "test that kg_add echoes the rendered statement prose back to the caller",
            },
        },
    )
    assert result.get("success") is True, result
    assert "statement_text" in result, (
        f"kg_add must echo 'statement_text' for non-skip predicates; got keys: {list(result)}"
    )
    # The echoed prose must include the structured fields rendered to
    # text -- so callers can verify the embedded form matches what they
    # passed in.
    assert "person_a lives in city_b" in result["statement_text"]
    assert "primary residence address" in result["statement_text"]


# ── Issue 4: diary_write registers as entity ────────────────────────


def test_diary_write_registers_entity_row(monkeypatch, config, palace_path, kg):
    """diary_write must insert a row into the SQLite entities table
    (kind='record') in addition to the Chroma record collection write,
    so that subsequent feedback edges (rated_useful / rated_irrelevant)
    targeting the diary memory id do NOT phantom-reject at
    add_rated_edge. Pre-fix this exact gap deadlocked the wrap_up
    finalize coverage check during Adrian's session-close 2026-05-01."""
    from mempalace import mcp_server
    from mempalace.tool_mutate import tool_diary_write

    monkeypatch.setattr(mcp_server._STATE, "kg", kg)
    monkeypatch.setattr(mcp_server._STATE, "config", config)
    monkeypatch.setattr(mcp_server._STATE, "session_id", "test-diary-entity")
    monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
    monkeypatch.setattr(mcp_server._STATE, "client_cache", None)
    monkeypatch.setattr(mcp_server._STATE, "collection_cache", None)
    kg.seed_ontology()
    _seed_agent(kg)

    summary = {
        "what": "diary entry test fixture",
        "why": "exercise diary_write so the SQLite entities-table registration fires",
        "scope": "diary entity audit",
    }
    result = tool_diary_write(
        agent_name="ga_agent",
        entry="Today I added the diary entity-registration regression. "
        "Pre-fix the missing entities-table row caused phantom-rejection "
        "of feedback edges and deadlocked finalize_intent coverage.",
        summary=summary,
        slug="diary-entity-registration-fixture",
        topic="testing",
    )
    assert result.get("success") is True, result
    entry_id = result["entry_id"]
    assert entry_id.startswith("diary_ga_agent_")

    # The fix: diary_write also inserts a row in entities (kind='record')
    # so feedback edges can attach without phantom-rejection.
    ent = kg.get_entity(entry_id)
    assert ent is not None, (
        f"diary_write did not register {entry_id!r} in the entities table; "
        f"feedback edges to this id will phantom-reject. This is the exact "
        f"gap Adrian's congruence audit identified at session-close 2026-05-01."
    )
    assert ent.get("kind") == "record", ent
    # Properties must carry the structured summary so render_memory_preview
    # returns the canonical prose for this id.
    props = ent.get("properties") or {}
    if isinstance(props, str):
        import json as _json

        props = _json.loads(props)
    assert isinstance(props.get("summary"), dict), (
        f"diary entity row must persist properties.summary as a dict so "
        f"render_memory_preview reads canonical prose; got {props}"
    )
    assert props["summary"]["what"] == summary["what"]
    assert props["summary"]["why"] == summary["why"]


def test_diary_entity_supports_rated_useful_edge(monkeypatch, config, palace_path, kg):
    """End-to-end proof of fix: a rated_useful edge from a context to a
    diary memory id succeeds (no phantom-rejection). This is the EXACT
    flow that broke wrap_up's feedback coverage during the 2026-05-01
    session-close deadlock -- the test prevents regression."""
    from mempalace import mcp_server
    from mempalace.entity_gate import mint_entity
    from mempalace.tool_mutate import tool_diary_write

    monkeypatch.setattr(mcp_server._STATE, "kg", kg)
    monkeypatch.setattr(mcp_server._STATE, "config", config)
    monkeypatch.setattr(mcp_server._STATE, "session_id", "test-diary-edge")
    monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
    monkeypatch.setattr(mcp_server._STATE, "client_cache", None)
    monkeypatch.setattr(mcp_server._STATE, "collection_cache", None)
    kg.seed_ontology()
    _seed_agent(kg)
    diary_summary = {
        "what": "diary entry for rated_useful edge test",
        "why": "verify that feedback edges can attach to the diary memory id without phantom-rejection",
        "scope": "diary feedback audit",
    }
    diary_res = tool_diary_write(
        agent_name="ga_agent",
        entry="Diary entry to be rated_useful by a feedback context.",
        summary=diary_summary,
        slug="diary-feedback-edge-fixture",
        topic="testing",
    )
    assert diary_res.get("success") is True, diary_res
    diary_id = diary_res["entry_id"]

    # Mint a context entity to anchor the rated_useful edge.
    mint_entity(
        "ctx_test_feedback",
        kind="entity",
        summary={
            "what": "test feedback context anchor",
            "why": "exists so a rated_useful edge has a real subject node",
            "scope": "diary feedback edge test",
        },
        queries=["context anchor for diary feedback edge", "rated_useful edge subject"],
    )

    # Use add_rated_edge directly (the same path extend_feedback uses).
    # Pre-fix this would phantom-reject because the diary id had no
    # entities-table row.
    kg.add_rated_edge(
        context="ctx_test_feedback",
        predicate="rated_useful",
        memory=diary_id,
        confidence=1.0,
    )
    # Verify the edge actually landed.
    edges = kg.query_entity("ctx_test_feedback", direction="outgoing")
    rated_targets = {
        e["object"]
        for e in edges
        if e.get("predicate") == "rated_useful" and e.get("current", True)
    }
    assert diary_id in rated_targets, (
        f"rated_useful edge from ctx_test_feedback to {diary_id!r} must land "
        f"after the diary registration fix; got rated targets: {rated_targets}"
    )


pytestmark = pytest.mark.integration
