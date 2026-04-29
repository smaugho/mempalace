"""Migration 014: context-as-entity schema additions.

Phase 1 of docs/context_as_entity_redesign_plan.md. Migration 014 adds
four partial indexes for the hot paths introduced by the new
``created_under`` and ``similar_to`` predicates, plus a partial index
on ``entities.kind='context'`` so ``kg_stats`` and the context ANN
rebuild path stay narrow as contexts accrete.

This test set runs against a **fresh** KG (no legacy rows) because the
plan's baseline assumption is a cold-started palace. The redesign
explicitly carries no backward-compat.
"""

from __future__ import annotations


def _indexes(kg):
    conn = kg._conn()
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
    ).fetchall()
    return {r[0] for r in rows}


def test_migration_014_creates_partial_indexes(kg):
    kg.seed_ontology()
    idx = _indexes(kg)
    # The four indexes migration 014 is responsible for.
    assert "idx_triples_created_under_subject" in idx
    assert "idx_triples_created_under_object" in idx
    assert "idx_triples_similar_to" in idx
    assert "idx_entities_kind_context" in idx


def test_migration_014_seeds_context_class(kg):
    """The ontology seed now includes a taxonomic root class `context`."""
    kg.seed_ontology()
    ctx = kg.get_entity("context")
    assert ctx is not None, "context root class missing from seeded ontology"
    assert ctx["kind"] == "class"
    # is_a thing -- every non-thing class inherits from it.
    rels = kg.query_entity("context", direction="outgoing")
    parents = {e["object"] for e in rels if e["predicate"] == "is_a"}
    assert "thing" in parents


def test_migration_014_seeds_created_under_predicate(kg):
    """created_under is a seeded predicate with correct constraints."""
    kg.seed_ontology()
    cu = kg.get_entity("created_under")
    assert cu is not None
    assert cu["kind"] == "predicate"
    props = cu.get("properties", {}) or {}
    import json

    if isinstance(props, str):
        props = json.loads(props)
    constraints = props.get("constraints", {}) or {}
    subj_kinds = set(constraints.get("subject_kinds") or [])
    obj_kinds = set(constraints.get("object_kinds") or [])
    # Anything that can be written under an active context is a valid subject.
    assert subj_kinds == {"entity", "class", "predicate", "literal", "record"}
    # Only contexts are valid objects -- that's the whole point.
    assert obj_kinds == {"context"}
    assert constraints.get("cardinality") == "many-to-one"


def test_migration_014_seeds_similar_to_predicate(kg):
    """similar_to is context-to-context only, many-to-many."""
    kg.seed_ontology()
    st = kg.get_entity("similar_to")
    assert st is not None
    assert st["kind"] == "predicate"
    props = st.get("properties", {}) or {}
    import json

    if isinstance(props, str):
        props = json.loads(props)
    constraints = props.get("constraints", {}) or {}
    assert set(constraints.get("subject_kinds") or []) == {"context"}
    assert set(constraints.get("object_kinds") or []) == {"context"}
    assert constraints.get("cardinality") == "many-to-many"


def test_context_kind_accepted_by_add_entity(kg):
    """kind='context' is a first-class kind -- add_entity accepts it."""
    kg.seed_ontology()
    eid = kg.add_entity("ctx_test_fixture_1", kind="context", content="smoke", importance=3)
    assert eid == "ctx_test_fixture_1"
    got = kg.get_entity(eid)
    assert got is not None
    assert got["kind"] == "context"


def test_created_under_edge_writes_and_reads(kg):
    """created_under edge round-trips via add_triple + query_entity."""
    kg.seed_ontology()
    kg.add_entity("ctx_rt_fixture", kind="context", content="rt", importance=3)
    kg.add_entity("my_memory", kind="record", content="m", importance=3)
    kg.add_triple("my_memory", "created_under", "ctx_rt_fixture")
    out = kg.query_entity("my_memory", direction="outgoing")
    edges = [(e["predicate"], e["object"]) for e in out]
    assert ("created_under", "ctx_rt_fixture") in edges


def test_similar_to_edge_confidence_carries_sim(kg):
    """similar_to stores the MaxSim score in the confidence column."""
    kg.seed_ontology()
    kg.add_entity("ctx_a", kind="context", content="a", importance=3)
    kg.add_entity("ctx_b", kind="context", content="b", importance=3)
    kg.add_triple("ctx_a", "similar_to", "ctx_b", confidence=0.82)
    out = kg.query_entity("ctx_a", direction="outgoing")
    sim_edges = [e for e in out if e["predicate"] == "similar_to"]
    assert len(sim_edges) == 1
    assert abs(float(sim_edges[0]["confidence"]) - 0.82) < 1e-6


def test_get_similar_contexts_one_hop(kg):
    kg.seed_ontology()
    kg.add_entity("ctx_root", kind="context", content="r", importance=3)
    kg.add_entity("ctx_near", kind="context", content="n", importance=3)
    kg.add_triple("ctx_root", "similar_to", "ctx_near", confidence=0.85)
    hops = kg.get_similar_contexts("ctx_root", hops=1)
    assert hops == [("ctx_near", 0.85)]


def test_get_similar_contexts_two_hops_decays(kg):
    """2-hop path is decayed: sim_1 * decay * sim_2, default decay=0.5."""
    kg.seed_ontology()
    kg.add_entity("ctx_r", kind="context", content="r", importance=3)
    kg.add_entity("ctx_m", kind="context", content="m", importance=3)
    kg.add_entity("ctx_f", kind="context", content="f", importance=3)
    kg.add_triple("ctx_r", "similar_to", "ctx_m", confidence=0.9)
    kg.add_triple("ctx_m", "similar_to", "ctx_f", confidence=0.8)
    two_hops = dict(kg.get_similar_contexts("ctx_r", hops=2))
    assert abs(two_hops["ctx_m"] - 0.9) < 1e-6
    assert abs(two_hops["ctx_f"] - (0.9 * 0.8 * 0.5)) < 1e-6


def test_get_similar_contexts_isolated_returns_empty(kg):
    kg.seed_ontology()
    kg.add_entity("ctx_lone", kind="context", content="lone", importance=3)
    assert kg.get_similar_contexts("ctx_lone", hops=2) == []
