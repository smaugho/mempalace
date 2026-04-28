"""Context-as-entity accretion semantics.

Exercises ``context_lookup_or_create`` threshold branches
(``T_reuse=0.90``, ``T_similar=0.70``):

  - Cold-start: first call mints a fresh context, ``reused=False``,
    ``max_sim=0.0``.
  - Cosmetic restate: second call with near-identical queries reuses
    (``max_sim >= T_reuse``), no new entity, no new similar_to edge.
  - Mid-similarity: new call with overlapping-but-distinct queries
    falls in ``[T_similar, T_reuse)`` -- creates a new context AND writes
    a ``similar_to`` edge to the nearest neighbour.
  - Disjoint: new call below ``T_similar`` creates a new context with
    no ``similar_to`` edge.

Underlying citation: ColBERT-style late-interaction MaxSim for context
similarity (Khattab & Zaharia 2020 arXiv:2004.12832); BIRCH-style
threshold accretion (Zhang/Ramakrishnan/Livny 1996 SIGMOD).
"""

from __future__ import annotations

from tests.test_mcp_server import _patch_mcp_server


def _boot(monkeypatch, config, palace_path, kg):
    _patch_mcp_server(monkeypatch, config, kg)
    from mempalace import mcp_server

    kg.seed_ontology()
    return mcp_server


def _contexts(kg):
    return kg.list_entities(status="active", kind="context")


def _similar_to_edges(kg, subject):
    out = kg.query_entity(subject, direction="outgoing")
    return [e for e in out if e["predicate"] == "similar_to"]


def test_cold_start_mints_fresh_context(monkeypatch, config, palace_path, kg):
    mcp_server = _boot(monkeypatch, config, palace_path, kg)
    cid, reused, max_sim = mcp_server.context_lookup_or_create(
        queries=["authenticate users via jwt", "session expiry handling"],
        keywords=["auth", "jwt"],
        entities=["LoginService"],
        agent="test_agent",
    )
    assert cid, "expected non-empty context id"
    assert reused is False
    assert max_sim == 0.0
    # The entity is materialised in the KG with kind=context.
    entity = kg.get_entity(cid)
    assert entity is not None
    assert entity["kind"] == "context"
    props = entity.get("properties", {}) or {}
    import json

    if isinstance(props, str):
        props = json.loads(props)
    assert "authenticate users via jwt" in props.get("queries", [])
    assert "auth" in props.get("keywords", [])
    assert "LoginService" in props.get("entities", [])


def test_cosmetic_restate_reuses_context(monkeypatch, config, palace_path, kg):
    """Same queries → same context id, no new entity row."""
    mcp_server = _boot(monkeypatch, config, palace_path, kg)
    queries = ["authenticate users via jwt", "session expiry handling"]
    keywords = ["auth", "jwt"]

    cid_a, reused_a, _ = mcp_server.context_lookup_or_create(
        queries=queries, keywords=keywords, entities=[], agent="test_agent"
    )
    assert reused_a is False
    count_after_first = len(_contexts(kg))

    cid_b, reused_b, max_sim_b = mcp_server.context_lookup_or_create(
        queries=queries, keywords=keywords, entities=[], agent="test_agent"
    )
    assert cid_b == cid_a, "identical queries should reuse the original context"
    assert reused_b is True
    assert max_sim_b >= mcp_server.CONTEXT_REUSE_THRESHOLD
    # No new entity row.
    assert len(_contexts(kg)) == count_after_first


def test_disjoint_queries_mint_fresh_no_similar_to(monkeypatch, config, palace_path, kg):
    """A second context about a totally different topic gets no similar_to."""
    mcp_server = _boot(monkeypatch, config, palace_path, kg)
    cid_a, _, _ = mcp_server.context_lookup_or_create(
        queries=["authenticate users via jwt", "session expiry handling"],
        keywords=["auth", "jwt"],
        entities=[],
        agent="test_agent",
    )
    cid_b, reused_b, max_sim_b = mcp_server.context_lookup_or_create(
        queries=[
            "vegetable garden soil composition",
            "tomato crop rotation practices",
        ],
        keywords=["garden", "tomato"],
        entities=[],
        agent="test_agent",
    )
    assert cid_b != cid_a, "disjoint queries must create a fresh context"
    assert reused_b is False
    assert max_sim_b < mcp_server.CONTEXT_SIMILAR_THRESHOLD
    # No similar_to edge out of the new context (or into the old one).
    assert _similar_to_edges(kg, cid_b) == []


def test_mid_similarity_mints_fresh_with_similar_to(monkeypatch, config, palace_path, kg):
    """Overlap but not enough to reuse → new context + similar_to edge.

    We seed one context, then query with paraphrased overlap that should
    land in the 0.70..0.90 window. If the embedding model classifies
    differently (either above T_reuse or below T_similar), assert on the
    branch invariant rather than on the specific window -- the function's
    contract is "either reuse OR (create AND similar_to if max_sim in
    window)", so at least one of those shapes must hold.
    """
    mcp_server = _boot(monkeypatch, config, palace_path, kg)
    cid_a, _, _ = mcp_server.context_lookup_or_create(
        queries=[
            "retrieve user sessions from the auth token store",
            "look up active jwt sessions by user id",
        ],
        keywords=["auth", "jwt", "session"],
        entities=[],
        agent="test_agent",
    )
    # Paraphrase with the same core domain but different phrasing.
    cid_b, reused_b, max_sim_b = mcp_server.context_lookup_or_create(
        queries=[
            "querying the session database for signed-in agents",
            "find tokens by caller id in the auth cache",
        ],
        keywords=["auth", "sessions"],
        entities=[],
        agent="test_agent",
    )
    if reused_b:
        # Embedding model considered them duplicates -- legitimate branch.
        assert cid_b == cid_a
        assert max_sim_b >= mcp_server.CONTEXT_REUSE_THRESHOLD
    else:
        assert cid_b != cid_a
        sim_edges = _similar_to_edges(kg, cid_b)
        if max_sim_b >= mcp_server.CONTEXT_SIMILAR_THRESHOLD:
            # Window branch: must have written a similar_to edge to cid_a.
            assert len(sim_edges) == 1
            assert sim_edges[0]["object"] == cid_a
            # Confidence carries the sim score.
            assert abs(float(sim_edges[0]["confidence"]) - max_sim_b) < 1e-3
        else:
            # Disjoint branch: no similar_to.
            assert sim_edges == []


def test_empty_queries_return_empty(monkeypatch, config, palace_path, kg):
    mcp_server = _boot(monkeypatch, config, palace_path, kg)
    cid, reused, max_sim = mcp_server.context_lookup_or_create(
        queries=[], keywords=[], entities=[], agent="test_agent"
    )
    assert cid == ""
    assert reused is False
    assert max_sim == 0.0


def test_keywords_persisted_to_entity_keywords(monkeypatch, config, palace_path, kg):
    mcp_server = _boot(monkeypatch, config, palace_path, kg)
    cid, _, _ = mcp_server.context_lookup_or_create(
        queries=["a kw persistence smoke test for channel C seeding"],
        keywords=["smoke", "keyword_channel", "idf_seed"],
        entities=[],
        agent="test_agent",
    )
    stored = set(kg.get_entity_keywords(cid))
    assert {"smoke", "keyword_channel", "idf_seed"} <= stored
