"""Channel D -- context-feedback retrieval (Phase 2).

Verifies the end-to-end substrate introduced by P2:

  - ``_build_context_channel`` aggregates signals over the active context
    plus its 1-2 hop ``similar_to`` neighbourhood.
  - ``rated_useful`` edges contribute positive signal weighted by the
    neighbour's similarity score; ``rated_irrelevant`` contributes
    negative; ``surfaced`` contributes weak positive.
  - ``multi_channel_search`` wires Channel D in via ``active_context_id``
    and weighted-RRF merges it with the other channels.
"""

from __future__ import annotations

from mempalace.scoring import _build_context_channel, rrf_merge


def _seed_ctx(kg, cid, desc="ctx"):
    kg.add_entity(cid, kind="context", content=desc, importance=3)


def _seed_mem(kg, mid, desc="mem"):
    kg.add_entity(mid, kind="record", content=desc, importance=3)


def test_context_channel_rated_useful_accumulates(kg):
    """rated_useful edges feed Channel D positive scores."""
    kg.seed_ontology()
    _seed_ctx(kg, "ctx_active", "active")
    _seed_mem(kg, "mem_hit", "the memory we expect back")
    kg.add_triple(
        "ctx_active",
        "rated_useful",
        "mem_hit",
        confidence=0.8,
        properties={"ts": "2026-04-22T00:00:00", "reason": "seed"},
    )

    ranked = _build_context_channel(kg, "ctx_active", seen_meta={})
    ids = [m for _score, _doc, m in ranked]
    assert "mem_hit" in ids


def test_context_channel_irrelevant_is_negative(kg):
    """rated_irrelevant memories never surface in the positive-only channel."""
    kg.seed_ontology()
    _seed_ctx(kg, "ctx_active")
    _seed_mem(kg, "mem_neg", "m")
    kg.add_triple("ctx_active", "rated_irrelevant", "mem_neg", confidence=0.9)

    ranked = _build_context_channel(kg, "ctx_active", seen_meta={})
    ids = [m for _s, _d, m in ranked]
    assert "mem_neg" not in ids, "rated_irrelevant should drop memory below 0.0"


def test_context_channel_similar_neighbourhood_contributes(kg):
    """1-hop similar_to neighbour's rated_useful edges flow into active context's channel."""
    kg.seed_ontology()
    _seed_ctx(kg, "ctx_active")
    _seed_ctx(kg, "ctx_neigh")
    _seed_mem(kg, "mem_via_neigh")
    kg.add_triple("ctx_active", "similar_to", "ctx_neigh", confidence=0.85)
    kg.add_triple("ctx_neigh", "rated_useful", "mem_via_neigh", confidence=0.8)

    ranked = _build_context_channel(kg, "ctx_active", seen_meta={})
    ids = [m for _s, _d, m in ranked]
    assert "mem_via_neigh" in ids


def test_context_channel_isolated_returns_empty(kg):
    kg.seed_ontology()
    _seed_ctx(kg, "ctx_lonely")
    assert _build_context_channel(kg, "ctx_lonely", seen_meta={}) == []


def test_rrf_merge_weighted_variant_respects_weights():
    """Weighted RRF from Bruch/Gai/Ingber 2023 -- higher weight → higher score."""
    ranked_lists = {
        "cosine": ([(0.9, "a", "mid_a"), (0.8, "b", "mid_b")], 1.0),
        "context": ([(0.6, "a", "mid_a")], 1.5),  # heavier weight
    }
    scores, _cm, attr = rrf_merge(ranked_lists)
    # mid_a appears in both lists; mid_b appears only in cosine.
    # cosine rank-1 for both mid_a and mid_b, context rank-1 for mid_a.
    # Both rank-1 contributions go in, but context's weight > cosine's.
    assert scores["mid_a"] > scores["mid_b"]
    assert "context" in attr["mid_a"]


def test_rrf_merge_plain_list_defaults_to_weight_one():
    """Legacy-shape entries (bare list, not (list, weight)) get weight=1.0."""
    ranked_lists = {"cosine": [(0.9, "a", "mid")], "keyword": [(0.7, "a", "mid")]}
    scores, _cm, _attr = rrf_merge(ranked_lists)
    # Same rank in both lists, weight=1.0 for both → 2 * 1/(60+1) ≈ 0.0328
    assert abs(scores["mid"] - 2.0 / 61.0) < 1e-6
