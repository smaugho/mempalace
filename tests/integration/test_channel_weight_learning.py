"""Weight-learning re-enable (P3).

P2 truncated scoring_weight_feedback (cold start) and P3 flips the
``_A6_WEIGHT_SELFTUNE_ENABLED`` gate back to True. These tests verify:

  1. record_scoring_feedback writes rows (no-op was fixed).
  2. compute_learned_weights drifts weights away from the base when
     feedback signals a biased correlation.
  3. Below min_samples, the learner falls back to the base weights
     (conservative default).

Weight damping is ±30% per component (Robertson & Zaragoza "Foundations
of BM25 and Beyond" 2009 for the saturating-adjustment rationale,
adapted here to the 4-component hybrid_score convex combination).
"""

from __future__ import annotations

import pytest
from mempalace.scoring import DEFAULT_SEARCH_WEIGHTS


def test_weight_selftune_is_enabled(kg):
    """P3 flipped the flag from False to True."""
    kg.seed_ontology()
    assert kg._A6_WEIGHT_SELFTUNE_ENABLED is True


def test_record_scoring_feedback_persists_rows(kg):
    kg.seed_ontology()
    kg.record_scoring_feedback({"sim": 0.9, "rel": 0.7, "imp": 0.6, "decay": 0.8}, was_useful=True)
    kg.record_scoring_feedback({"sim": 0.1, "rel": 0.2, "imp": 0.2, "decay": 0.3}, was_useful=False)
    conn = kg._conn()
    count = conn.execute("SELECT COUNT(*) FROM scoring_weight_feedback").fetchone()[0]
    assert count == 8  # 4 components × 2 outcomes


def test_compute_learned_weights_below_min_samples_is_noop(kg):
    kg.seed_ontology()
    kg.record_scoring_feedback({"sim": 0.9, "rel": 0.7, "imp": 0.6, "decay": 0.5}, True)
    out = kg.compute_learned_weights(dict(DEFAULT_SEARCH_WEIGHTS), min_samples=20)
    assert out == DEFAULT_SEARCH_WEIGHTS


def test_compute_learned_weights_drifts_with_biased_signal(kg):
    """When `sim` strongly correlates with usefulness and `rel` anti-
    correlates, the learner should boost sim's weight and sink rel's,
    while keeping the sum at ~1.0. (W_AGENT was retired 2026-05-01 --
    REL is now the second-most-volatile component, so it carries the
    same bias-direction shape this test depended on.)"""
    kg.seed_ontology()
    # 10 useful rows where sim was high and rel was low.
    for _ in range(10):
        kg.record_scoring_feedback(
            {"sim": 0.9, "rel": 0.1, "imp": 0.5, "decay": 0.5},
            was_useful=True,
        )
    # 10 irrelevant rows where sim was low and rel was high.
    for _ in range(10):
        kg.record_scoring_feedback(
            {"sim": 0.1, "rel": 0.9, "imp": 0.5, "decay": 0.5},
            was_useful=False,
        )

    out = kg.compute_learned_weights(dict(DEFAULT_SEARCH_WEIGHTS), min_samples=5)
    # Weights should still sum to 1.0 post-normalisation.
    total = sum(out.values())
    assert abs(total - 1.0) < 1e-6
    # Sim went up relative to baseline; rel went down.
    assert out["sim"] > DEFAULT_SEARCH_WEIGHTS["sim"]
    assert out["rel"] < DEFAULT_SEARCH_WEIGHTS["rel"]


pytestmark = pytest.mark.integration
