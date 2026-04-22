"""Per-channel RRF weight learning (P3 polish follow-up).

compute_learned_weights now supports a second scope 'channel' that
learns weights for the four RRF channels (cosine / graph / keyword /
context). Same mechanism as the hybrid-score weights, same damped
point-biserial correlation, same ±30% cap. Feedback is recorded at
finalize_intent from surfaced-edge channel props.
"""

from __future__ import annotations

from mempalace.scoring import DEFAULT_CHANNEL_WEIGHTS


def test_record_scoring_feedback_honors_scope(kg):
    """scope='channel' rows get a 'ch_' prefix; scope='hybrid' doesn't."""
    kg.seed_ontology()

    kg.record_scoring_feedback({"sim": 0.8, "imp": 0.5}, was_useful=True)
    kg.record_scoring_feedback({"cosine": 1.0, "context": 0.0}, was_useful=True, scope="channel")

    conn = kg._conn()
    rows = conn.execute(
        "SELECT component FROM scoring_weight_feedback ORDER BY component"
    ).fetchall()
    components = [r[0] for r in rows]
    assert "sim" in components
    assert "imp" in components
    assert "ch_cosine" in components
    assert "ch_context" in components
    # No cross-scope collisions.
    assert "cosine" not in components
    assert "ch_sim" not in components


def test_compute_learned_weights_hybrid_scope_untouched_by_channel_rows(kg):
    """Channel rows don't count toward hybrid-scope min_samples."""
    kg.seed_ontology()
    # 40 channel rows shouldn't cause hybrid's compute_learned_weights to
    # think it has enough data.
    for _ in range(10):
        kg.record_scoring_feedback(
            {"cosine": 1.0, "graph": 0.0, "keyword": 0.0, "context": 0.0},
            was_useful=True,
            scope="channel",
        )
    # Hybrid scope is empty -> stays at defaults.
    out = kg.compute_learned_weights({"sim": 0.5, "rel": 0.5}, min_samples=10, scope="hybrid")
    assert out == {"sim": 0.5, "rel": 0.5}


def test_compute_learned_weights_channel_scope_drifts_on_biased_signal(kg):
    """Bias the channel signal and verify the learner shifts weight accordingly."""
    kg.seed_ontology()
    # 15 rows where cosine fires on useful rows and context doesn't.
    for _ in range(15):
        kg.record_scoring_feedback(
            {"cosine": 1.0, "graph": 0.0, "keyword": 0.0, "context": 0.0},
            was_useful=True,
            scope="channel",
        )
    # 15 rows where context fires on irrelevant rows.
    for _ in range(15):
        kg.record_scoring_feedback(
            {"cosine": 0.0, "graph": 0.0, "keyword": 0.0, "context": 1.0},
            was_useful=False,
            scope="channel",
        )

    out = kg.compute_learned_weights(dict(DEFAULT_CHANNEL_WEIGHTS), min_samples=5, scope="channel")
    assert abs(sum(out.values()) - 1.0) < 1e-6
    # cosine correlates with useful -> weight up.
    # context correlates with irrelevant -> weight down.
    # normalized comparison (ratio) because magnitudes differ from baseline.
    assert (
        out["cosine"] / DEFAULT_CHANNEL_WEIGHTS["cosine"]
        > out["context"] / DEFAULT_CHANNEL_WEIGHTS["context"]
    )


def test_get_effective_channel_weights_falls_back_to_defaults():
    """When no learned weights set, effective = DEFAULT_CHANNEL_WEIGHTS."""
    from mempalace.scoring import get_effective_channel_weights, set_learned_channel_weights

    set_learned_channel_weights({})
    eff = get_effective_channel_weights()
    assert eff == DEFAULT_CHANNEL_WEIGHTS


def test_set_learned_channel_weights_overrides_defaults():
    from mempalace.scoring import get_effective_channel_weights, set_learned_channel_weights

    set_learned_channel_weights({"cosine": 2.0, "context": 0.3})
    eff = get_effective_channel_weights()
    assert eff["cosine"] == 2.0
    assert eff["context"] == 0.3
    # Unspecified channels keep the default.
    assert eff["graph"] == DEFAULT_CHANNEL_WEIGHTS["graph"]
    assert eff["keyword"] == DEFAULT_CHANNEL_WEIGHTS["keyword"]
    # Reset so we don't leak into other tests.
    set_learned_channel_weights({})
