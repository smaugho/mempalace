"""Regression locks for the cross-encoder eval harness (mempalace.eval_ce).

Adrian's CE eval design 2026-05-02 (record_ga_agent_damping_factor_literature_and_ce_correction).
The harness has three layers worth pinning:

  A. ``build_examples_from_palace`` joins feedback edges to context
     queries and memory content. This is pure SQLite -- no ML deps.
     Tests A.1-A.3 build a tiny synthetic palace, write fixtures
     directly into the schema, and assert the output is what the
     downstream scorer expects.
  B. ``stratify_by_length`` + ``bucket_for`` -- pure helpers, but
     load-bearing for the per-bucket stratification Q1 motivated.
     Tests B.1-B.2 lock the boundary semantics so a future refactor
     doesn't silently shift them.
  C. Metric math -- ``_ndcg_at_k``, ``_precision_at_k``,
     ``_recall_at_k``, ``_auc_pairwise``. These are pure functions of
     ranked-label lists; tests C.1-C.5 use textbook cases (perfect
     ranking, reverse, ties, all-positive, all-negative) so the
     harness reports correct metrics no matter the scorer.

D. ``cost_savings_curve`` is tested via the integration smoke test
   D.1 which confirms the curve monotonicity invariants we expect.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mempalace.eval_ce import (
    BUCKETS,
    LabeledPair,
    _auc_pairwise,
    _ndcg_at_k,
    _precision_at_k,
    _recall_at_k,
    bucket_for,
    build_examples_from_palace,
    compute_bucket_metrics,
    cost_savings_curve,
    group_by_context,
    stratify_by_length,
)


# ── helpers ────────────────────────────────────────────────────────


def _build_synthetic_db(tmp_path: Path) -> Path:
    """Write a minimal 'palace' SQLite db carrying just the entities +
    triples tables the harness reads. Avoids the full mempalace
    bootstrap (yoyo migrations, Chroma, etc.) so the unit test is fast
    and offline.

    Schema mirrors live mempalace exactly for the columns the harness
    touches; other columns are omitted (the harness only SELECTs the
    fields it needs)."""
    db = tmp_path / "kg.sqlite3"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE entities (
            id TEXT PRIMARY KEY,
            content TEXT,
            properties TEXT
        );
        CREATE TABLE triples (
            id TEXT PRIMARY KEY,
            subject TEXT,
            predicate TEXT,
            object TEXT,
            valid_from TEXT,
            valid_to TEXT,
            confidence REAL,
            properties TEXT
        );
        """
    )
    return db


def _add_context(conn, ctx_id: str, queries: list[str]) -> None:
    """Insert a synthetic context entity carrying queries[]."""
    import json

    props = {"queries": queries}
    conn.execute(
        "INSERT INTO entities (id, content, properties) VALUES (?, ?, ?)",
        (ctx_id, "ctx", json.dumps(props)),
    )


def _add_memory(conn, mem_id: str, content: str, summary_dict: dict | None = None) -> None:
    """Insert a synthetic memory entity."""
    import json

    props = {"summary": summary_dict} if summary_dict else {}
    conn.execute(
        "INSERT INTO entities (id, content, properties) VALUES (?, ?, ?)",
        (mem_id, content, json.dumps(props) if props else None),
    )


def _add_feedback(
    conn,
    ctx_id: str,
    mem_id: str,
    *,
    label: int,
    relevance: int,
    valid_to: str = "",
) -> None:
    """Insert a rated_useful or rated_irrelevant edge."""
    import json

    pred = "rated_useful" if label > 0 else "rated_irrelevant"
    tid = f"t_{ctx_id}_{pred}_{mem_id}"
    props = {"relevance": relevance, "ts": "2026-05-02T10:00:00", "reason": "synthetic"}
    conn.execute(
        "INSERT INTO triples (id, subject, predicate, object, valid_to, "
        "confidence, properties) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (tid, ctx_id, pred, mem_id, valid_to, 1.0, json.dumps(props)),
    )


# ── A: build_examples_from_palace ──────────────────────────────────


def test_build_examples_basic(tmp_path):
    """A.1: harvests rated_useful + rated_irrelevant, attaches queries
    and content, drops invalidated edges by default."""
    db = _build_synthetic_db(tmp_path)
    conn = sqlite3.connect(str(db))
    _add_context(conn, "ctx_a", ["how does the gate work", "injection filter design"])
    _add_memory(conn, "mem_useful", "The injection gate filters retrieved memories." * 5)
    _add_memory(conn, "mem_irrelevant", "Unrelated text about gardening." * 5)
    _add_feedback(conn, "ctx_a", "mem_useful", label=1, relevance=5)
    _add_feedback(conn, "ctx_a", "mem_irrelevant", label=0, relevance=2)
    # Stale invalidated edge -- should be filtered out by default
    _add_feedback(conn, "ctx_a", "mem_useful", label=0, relevance=1, valid_to="2026-04-01")
    conn.commit()

    pairs = build_examples_from_palace(db)
    assert len(pairs) == 2, f"expected 2 current edges; got {len(pairs)}"
    by_label = {p.label: p for p in pairs}
    assert 1 in by_label and 0 in by_label
    assert by_label[1].memory_id == "mem_useful"
    assert by_label[1].relevance == 5
    assert by_label[1].queries == ["how does the gate work", "injection filter design"]
    assert by_label[0].memory_id == "mem_irrelevant"
    assert by_label[0].relevance == 2


def test_build_examples_skips_no_queries(tmp_path):
    """A.2: contexts with no queries[] are skipped (no probes -> no
    bi-encoder ranking signal -> can't include in eval)."""
    db = _build_synthetic_db(tmp_path)
    conn = sqlite3.connect(str(db))
    # Context with no queries field at all
    conn.execute(
        "INSERT INTO entities (id, content, properties) VALUES (?, ?, ?)",
        ("ctx_naked", "ctx", None),
    )
    _add_memory(conn, "mem_x", "some content")
    _add_feedback(conn, "ctx_naked", "mem_x", label=1, relevance=4)
    conn.commit()

    pairs = build_examples_from_palace(db)
    assert pairs == [], "context without queries must be skipped"


def test_build_examples_min_pairs_filter(tmp_path):
    """A.3: ``min_pairs_per_context`` drops contexts below the floor.

    NDCG / precision are noisy on contexts with only 1 labeled pair --
    the harness's default for the live eval is to require >= 2."""
    db = _build_synthetic_db(tmp_path)
    conn = sqlite3.connect(str(db))
    _add_context(conn, "ctx_thin", ["q1", "q2"])
    _add_context(conn, "ctx_full", ["q3", "q4"])
    _add_memory(conn, "m1", "content one")
    _add_memory(conn, "m2", "content two")
    _add_memory(conn, "m3", "content three")
    _add_feedback(conn, "ctx_thin", "m1", label=1, relevance=4)  # only 1
    _add_feedback(conn, "ctx_full", "m2", label=1, relevance=5)
    _add_feedback(conn, "ctx_full", "m3", label=0, relevance=2)
    conn.commit()

    pairs = build_examples_from_palace(db, min_pairs_per_context=2)
    ctxs = {p.context_id for p in pairs}
    assert ctxs == {"ctx_full"}, f"thin ctx should be filtered out; got {ctxs}"


# ── B: bucketing ────────────────────────────────────────────────────


def test_bucket_for_boundaries():
    """B.1: bucket boundaries match Q1's content-length stratification.

    Q1 finding (record_ga_agent_channel_a_content_truncation_1800):
    Channel A truncates record embed_doc at 1800 chars. The xlong_>1800
    bucket is where CE's no-truncation advantage should be largest.
    Boundaries: <200 / 200-500 / 500-1800 / >1800."""
    assert bucket_for(0) == "short_<200"
    assert bucket_for(199) == "short_<200"
    assert bucket_for(200) == "medium_200-500"
    assert bucket_for(499) == "medium_200-500"
    assert bucket_for(500) == "long_500-1800"
    assert bucket_for(1799) == "long_500-1800"
    assert bucket_for(1800) == "xlong_>1800"
    assert bucket_for(50_000) == "xlong_>1800"
    # All four buckets must be represented in BUCKETS so callers can
    # iterate over a fixed set.
    bucket_names = [b[0] for b in BUCKETS]
    assert bucket_names == ["short_<200", "medium_200-500", "long_500-1800", "xlong_>1800"]


def test_stratify_by_length_returns_all_buckets_even_when_empty():
    """B.2: stratify_by_length always returns all four bucket keys
    (empty lists for empty buckets) so callers can iterate without
    KeyError."""
    pairs = [LabeledPair(context_id="c", memory_id="m", label=1, relevance=4, content_length=50)]
    out = stratify_by_length(pairs)
    assert set(out.keys()) == {b[0] for b in BUCKETS}
    assert len(out["short_<200"]) == 1
    assert out["medium_200-500"] == []
    assert out["long_500-1800"] == []
    assert out["xlong_>1800"] == []


# ── C: metric math ──────────────────────────────────────────────────


def test_ndcg_perfect_ranking():
    """C.1: positive labels at the top of the rank produce NDCG=1.0."""
    assert _ndcg_at_k([1, 1, 0, 0, 0], 5) == pytest.approx(1.0)
    assert _ndcg_at_k([1, 0, 0, 0, 0], 5) == pytest.approx(1.0)


def test_ndcg_reverse_ranking():
    """C.2: positive labels at the bottom produce NDCG strictly less
    than the perfect 1.0; should be small but nonzero."""
    score = _ndcg_at_k([0, 0, 0, 0, 1], 5)
    assert 0.0 < score < 0.5
    score_at_2 = _ndcg_at_k([0, 0, 0, 0, 1], 2)
    assert score_at_2 == 0.0  # k=2 doesn't reach the only positive at idx 4


def test_precision_recall_at_k():
    """C.3: precision and recall match the textbook definitions."""
    labels = [1, 0, 1, 0, 1, 0, 0]  # 3 positives total
    assert _precision_at_k(labels, 3) == pytest.approx(2 / 3)
    assert _precision_at_k(labels, 5) == pytest.approx(3 / 5)
    assert _recall_at_k(labels, 3) == pytest.approx(2 / 3)
    assert _recall_at_k(labels, 5) == pytest.approx(1.0)


def test_auc_perfect_separation():
    """C.4: positives all scored above negatives -> AUC=1.0; reverse
    -> AUC=0.0; identical scores in both classes -> AUC=0.5."""
    assert _auc_pairwise([(0.9, 1), (0.8, 1), (0.2, 0), (0.1, 0)]) == pytest.approx(1.0)
    assert _auc_pairwise([(0.1, 1), (0.2, 1), (0.8, 0), (0.9, 0)]) == pytest.approx(0.0)
    assert _auc_pairwise([(0.5, 1), (0.5, 0)]) == pytest.approx(0.5)


def test_auc_single_class_returns_neutral():
    """C.5: with only one class present, AUC is undefined; the harness
    returns 0.5 (neutral) so downstream means are still well-defined."""
    assert _auc_pairwise([(0.9, 1), (0.8, 1)]) == pytest.approx(0.5)
    assert _auc_pairwise([(0.9, 0), (0.8, 0)]) == pytest.approx(0.5)
    assert _auc_pairwise([]) == pytest.approx(0.5)


# ── D: integration via compute_bucket_metrics + cost_savings_curve ──


def test_compute_bucket_metrics_integration():
    """D.1: end-to-end metric computation on a small ranked fixture.

    Two contexts, four pairs, perfect bi-encoder scoring (positives
    score 0.9, negatives 0.1). Expect:
      * NDCG@10 = 1.0 for each context
      * P@10 = 0.5 (2 of top-2 to top-10 are positives among 2 total)
      * R@10 = 1.0
      * AUC = 1.0 (perfect separation across the bucket)
      * mean_pos > mean_neg by a wide margin
    """
    pairs = [
        LabeledPair(
            context_id="c1",
            memory_id="m1",
            label=1,
            relevance=5,
            content_length=100,
        ),
        LabeledPair(
            context_id="c1",
            memory_id="m2",
            label=0,
            relevance=2,
            content_length=120,
        ),
        LabeledPair(
            context_id="c2",
            memory_id="m3",
            label=1,
            relevance=4,
            content_length=180,
        ),
        LabeledPair(
            context_id="c2",
            memory_id="m4",
            label=0,
            relevance=1,
            content_length=190,
        ),
    ]
    scores = {
        ("c1", "m1"): 0.9,
        ("c1", "m2"): 0.1,
        ("c2", "m3"): 0.9,
        ("c2", "m4"): 0.1,
    }
    m = compute_bucket_metrics("short_<200", pairs, scores)
    assert m.n_pairs == 4
    assert m.n_positive == 2
    assert m.n_negative == 2
    assert m.ndcg_at_10 == pytest.approx(1.0)
    assert m.precision_at_10 == pytest.approx(0.5)  # 1 positive in top-2 per ctx
    assert m.recall_at_10 == pytest.approx(1.0)
    assert m.auc == pytest.approx(1.0)
    assert m.score_mean_pos == pytest.approx(0.9)
    assert m.score_mean_neg == pytest.approx(0.1)


def test_cost_savings_curve_invariants():
    """D.2: as T_HIGH rises, llm_savings_pct can only fall (fewer
    items above threshold), and recall_at_threshold can only fall
    too (a positive above T at 0.6 must also be above T at 0.5).
    Auto-accept precision is allowed to rise OR fall depending on
    where positives sit on the score axis -- we don't lock it."""
    pairs = [
        LabeledPair(
            context_id="c",
            memory_id=f"m{i}",
            label=1 if i < 3 else 0,
            relevance=5 if i < 3 else 1,
            content_length=100,
        )
        for i in range(6)
    ]
    # Positives cluster at high scores, negatives at low.
    scores = {
        ("c", "m0"): 0.95,
        ("c", "m1"): 0.85,
        ("c", "m2"): 0.75,
        ("c", "m3"): 0.30,
        ("c", "m4"): 0.20,
        ("c", "m5"): 0.10,
    }
    curve = cost_savings_curve(pairs, scores, thresholds=[0.0, 0.5, 0.7, 0.9, 1.0])
    # Monotone-decreasing in n above and llm_savings_pct
    counts = [pt.auto_accept_count for pt in curve]
    savings = [pt.llm_savings_pct for pt in curve]
    recalls = [pt.recall_at_threshold for pt in curve]
    for a, b in zip(counts, counts[1:]):
        assert a >= b, f"auto_accept_count must be monotone-decreasing: {counts}"
    for a, b in zip(savings, savings[1:]):
        assert a >= b, f"llm_savings must be monotone-decreasing: {savings}"
    for a, b in zip(recalls, recalls[1:]):
        assert a >= b, f"recall must be monotone-decreasing: {recalls}"
    # At T=0.0 everything is auto-accepted (full savings)
    assert curve[0].llm_savings_pct == pytest.approx(1.0)
    # At T=1.0 nothing is auto-accepted
    assert curve[-1].auto_accept_count == 0


def test_group_by_context_round_trip():
    """Sanity: group_by_context returns lists keyed by context_id, and
    every pair lands in exactly one bucket."""
    pairs = [
        LabeledPair(context_id="a", memory_id="1", label=1, relevance=5, content_length=10),
        LabeledPair(context_id="a", memory_id="2", label=0, relevance=2, content_length=20),
        LabeledPair(context_id="b", memory_id="3", label=1, relevance=4, content_length=30),
    ]
    grouped = group_by_context(pairs)
    assert set(grouped.keys()) == {"a", "b"}
    assert len(grouped["a"]) == 2
    assert len(grouped["b"]) == 1
    # Same total count as input
    assert sum(len(v) for v in grouped.values()) == len(pairs)
