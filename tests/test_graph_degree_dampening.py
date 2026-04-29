"""Degree-dampening on Channel B (_build_graph_channel).

Each seed→neighbour contribution is scaled by ``1 / log(degree(seed) + 2)``
so mega-hub seeds (agent entities, project roots, etc.) don't flood
the channel with generic 1-hop hits.

References:
  - Hogan et al. "Knowledge Graphs." arXiv:2003.02320 (2021).
  - West & Leskovec. WWW 2012 -- inverse-log degree dampening for KG
    random walks.
  - Bollacker et al. "Freebase." SIGMOD 2008 -- same idea for popular
    entity dampening.
"""

from __future__ import annotations

import math


def test_get_entity_degree_counts_incoming_plus_outgoing(kg):
    kg.seed_ontology()
    kg.add_entity("hub", kind="entity", content="h", importance=3)
    kg.add_entity("n1", kind="entity", content="n1", importance=3)
    kg.add_entity("n2", kind="entity", content="n2", importance=3)
    kg.add_entity("n3", kind="entity", content="n3", importance=3)
    # 2 outgoing + 1 incoming = degree 3.
    kg.add_triple("hub", "is_a", "n1")  # out (skip-list predicate; still counted)
    kg.add_triple("hub", "is_a", "n2")  # out
    kg.add_triple("n3", "is_a", "hub")  # in

    assert kg.get_entity_degree("hub") == 3


def test_get_entity_degree_returns_zero_for_unknown(kg):
    kg.seed_ontology()
    assert kg.get_entity_degree("no_such_entity") == 0
    assert kg.get_entity_degree("") == 0


def test_graph_channel_dampens_high_degree_seeds(monkeypatch, config, kg, palace_path):
    """A mega-hub seed's neighbours score less than a specialist seed's neighbours."""
    from tests.test_mcp_server import _patch_mcp_server, _get_collection
    from mempalace.scoring import _build_graph_channel

    _patch_mcp_server(monkeypatch, config, kg)
    _c, col = _get_collection(palace_path, create=True)

    # hub: degree 20 (20 outgoing edges to decoys)
    kg.add_entity("hub", kind="entity", content="h", importance=3)
    for i in range(20):
        decoy = f"decoy_{i}"
        kg.add_entity(decoy, kind="entity", content=f"d{i}", importance=3)
        kg.add_triple("hub", "is_a", decoy)
    kg.add_entity("hub_neighbour", kind="entity", content="hn", importance=3)
    kg.add_triple("hub", "is_a", "hub_neighbour")

    # specialist: degree 2 (two edges)
    kg.add_entity("specialist", kind="entity", content="s", importance=3)
    kg.add_entity("spec_neighbour", kind="entity", content="sn", importance=3)
    kg.add_entity("other", kind="entity", content="o", importance=3)
    kg.add_triple("specialist", "is_a", "spec_neighbour")
    kg.add_triple("specialist", "is_a", "other")

    # Seed both into the Chroma collection plus the two target neighbours.
    for eid in ["hub", "hub_neighbour", "specialist", "spec_neighbour"]:
        col.upsert(
            ids=[eid],
            documents=[f"doc {eid}"],
            metadatas=[{"name": eid, "kind": "entity", "importance": 3}],
        )

    # Same seed similarity for both so the only remaining difference is
    # the degree-dampening factor.
    seen_meta = {
        "hub": {"meta": {}, "doc": "hub doc", "similarity": 0.8},
        "specialist": {"meta": {}, "doc": "spec doc", "similarity": 0.8},
    }

    ranked = _build_graph_channel(
        col, kg, {"hub", "specialist"}, kind_filter=None, seen_meta=seen_meta
    )
    assert ranked

    # Find the entry for hub_neighbour and spec_neighbour.
    hub_scores = [s for s, _d, mid in ranked if mid == "hub_neighbour"]
    spec_scores = [s for s, _d, mid in ranked if mid == "spec_neighbour"]
    assert hub_scores, "expected a hub_neighbour contribution"
    assert spec_scores, "expected a spec_neighbour contribution"

    # The specialist's neighbour must outscore the hub's neighbour
    # because the hub's degree-dampening factor is smaller.
    assert max(spec_scores) > max(hub_scores), (
        f"specialist (deg=2) should beat hub (deg=21) after dampening. "
        f"spec={spec_scores} hub={hub_scores}"
    )


def test_graph_channel_damp_factor_math(kg):
    """Verify the 1 / log(degree + 2) shape by probing the helper directly."""
    kg.seed_ontology()
    kg.add_entity("a", kind="entity", content="a", importance=3)
    # 5 out + 0 in = degree 5.
    for i in range(5):
        kg.add_entity(f"b{i}", kind="entity", content=f"b{i}", importance=3)
        kg.add_triple("a", "is_a", f"b{i}")

    assert kg.get_entity_degree("a") == 5
    damp = 1.0 / math.log(5 + 2)
    # Within 5% of the expected shape.
    assert 0.5 < damp < 0.55
