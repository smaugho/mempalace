"""
test_link_prediction_candidates.py -- Unit tests for the analytical
half of the link-author redesign (Commit 2).

Covers ``mempalace.link_author.upsert_candidate`` / ``list_pending``
at the SQL level so the accumulator's math, dedup, canonical ordering,
and direct-edge short-circuit are locked in. The integration with
``tool_finalize_intent.contexts_touched_detail`` is exercised by
test_intent_system.py's existing finalize paths; here we focus on the
pure accumulator contract.

References: docs/link_author_plan.md §2.2, §2.4, §3, §6.
"""

from __future__ import annotations

import pytest
import math

from mempalace import link_author as la


# ─────────────────────────────────────────────────────────────────────
# Adamic-Adar math
# ─────────────────────────────────────────────────────────────────────


class TestAdamicAdarScore:
    def test_single_context_adds_weight(self, kg):
        """One context with the pair contributes ``1 / log(|entities|)``."""
        # 4-entity context → weight = 1 / log(4) ≈ 0.721
        weight = 1.0 / math.log(4)
        assert la.upsert_candidate(kg, "A", "B", weight, "ctx-1") is True

        rows = la.list_pending(kg, limit=10, threshold=0.0)
        assert len(rows) == 1
        assert rows[0]["from_entity"] == "A"
        assert rows[0]["to_entity"] == "B"
        assert rows[0]["score"] == weight
        assert rows[0]["shared_context_count"] == 1
        assert rows[0]["last_context_id"] == "ctx-1"

    def test_distinct_contexts_accumulate(self, kg):
        """Same pair, different contexts → score is sum of per-context weights."""
        # ctx-1: 2 entities (weight = 1/log(2))
        # ctx-2: 4 entities (weight = 1/log(4))
        # ctx-3: 8 entities (weight = 1/log(8))
        w1 = 1.0 / math.log(2)
        w2 = 1.0 / math.log(4)
        w3 = 1.0 / math.log(8)
        expected = w1 + w2 + w3

        assert la.upsert_candidate(kg, "A", "B", w1, "ctx-1") is True
        assert la.upsert_candidate(kg, "A", "B", w2, "ctx-2") is True
        assert la.upsert_candidate(kg, "A", "B", w3, "ctx-3") is True

        rows = la.list_pending(kg, limit=10, threshold=0.0)
        assert len(rows) == 1
        assert rows[0]["shared_context_count"] == 3
        assert rows[0]["score"] == expected

    def test_smaller_contexts_weigh_more(self, kg):
        """Pair in 2-entity context outscores pair in 30-entity context."""
        la.upsert_candidate(kg, "A", "B", 1.0 / math.log(2), "small-ctx")
        la.upsert_candidate(kg, "C", "D", 1.0 / math.log(30), "big-ctx")
        rows = la.list_pending(kg, limit=10, threshold=0.0)
        # Sort-order from list_pending: score DESC, so A/B first.
        assert rows[0]["from_entity"] == "A"
        assert rows[0]["to_entity"] == "B"
        assert rows[0]["score"] > rows[1]["score"]


# ─────────────────────────────────────────────────────────────────────
# Distinct-context dedup (Option A from §2.2)
# ─────────────────────────────────────────────────────────────────────


class TestDistinctContextDedup:
    def test_same_context_five_times_contributes_once(self, kg):
        """The Adamic-Adar contract: re-observing the same context is ONE
        observation seen N times, not N observations. Only the first call
        for a given (pair, ctx_id) tuple mutates the candidate row."""
        weight = 1.0 / math.log(3)

        first = la.upsert_candidate(kg, "A", "B", weight, "ctx-same")
        assert first is True

        for _ in range(4):
            again = la.upsert_candidate(kg, "A", "B", weight, "ctx-same")
            assert again is False

        rows = la.list_pending(kg, limit=10, threshold=0.0)
        assert len(rows) == 1
        assert rows[0]["score"] == weight
        assert rows[0]["shared_context_count"] == 1

    def test_dedup_is_pair_scoped(self, kg):
        """Same context reused for DIFFERENT pairs counts once per pair."""
        w = 1.0 / math.log(3)
        assert la.upsert_candidate(kg, "A", "B", w, "ctx-shared") is True
        assert la.upsert_candidate(kg, "A", "C", w, "ctx-shared") is True
        assert la.upsert_candidate(kg, "B", "C", w, "ctx-shared") is True

        rows = la.list_pending(kg, limit=10, threshold=0.0)
        assert len(rows) == 3
        for r in rows:
            assert r["shared_context_count"] == 1
            assert r["score"] == w


# ─────────────────────────────────────────────────────────────────────
# Canonical ordering PK
# ─────────────────────────────────────────────────────────────────────


class TestCanonicalOrdering:
    def test_reversed_pair_collapses_to_same_row(self, kg):
        """upsert_candidate(A, B, ...) and upsert_candidate(B, A, ...)
        target the same PK -- canonical order is lex-sort by id, so the
        second call hits the first's row via ON CONFLICT DO UPDATE."""
        w = 1.0 / math.log(4)
        la.upsert_candidate(kg, "zebra", "apple", w, "ctx-1")
        la.upsert_candidate(kg, "apple", "zebra", w, "ctx-2")

        rows = la.list_pending(kg, limit=10, threshold=0.0)
        assert len(rows) == 1
        # Canonical: "apple" < "zebra"
        assert rows[0]["from_entity"] == "apple"
        assert rows[0]["to_entity"] == "zebra"
        assert rows[0]["shared_context_count"] == 2
        assert rows[0]["score"] == 2 * w

    def test_equal_ids_refused(self, kg):
        """Self-pair has no meaning; upsert refuses it silently."""
        assert la.upsert_candidate(kg, "A", "A", 1.0, "ctx-1") is False
        assert la.list_pending(kg, limit=10, threshold=0.0) == []


# ─────────────────────────────────────────────────────────────────────
# Direct-edge short-circuit
# ─────────────────────────────────────────────────────────────────────


class TestDirectEdgeShortCircuit:
    def _seed_edge(self, kg, subject, predicate, obj):
        """Seed the bare minimum KG state for an 'already connected' edge.

        We bypass the full validator stack -- the accumulator only reads
        the triples table, so that's all we need.
        """
        kg.add_entity(subject, kind="entity", content="s")
        kg.add_entity(obj, kind="entity", content="o")
        kg.add_entity(predicate, kind="predicate", content="p")
        kg.add_triple(subject, predicate, obj, statement=f"{subject} {predicate} {obj}.")

    def test_existing_forward_edge_skips_upsert(self, kg):
        self._seed_edge(kg, "alice", "knows", "bob")
        # Forward direction: (alice, knows, bob) already exists
        result = la.upsert_candidate(kg, "alice", "bob", 1.0, "ctx-1")
        assert result is False
        assert la.list_pending(kg, limit=10, threshold=0.0) == []

    def test_existing_reverse_edge_skips_upsert(self, kg):
        """A 1-hop edge in EITHER direction qualifies -- the point of the
        accumulator is finding NEW edges, and the graph channel already
        surfaces directly-connected pairs regardless of direction."""
        self._seed_edge(kg, "bob", "knows", "alice")
        result = la.upsert_candidate(kg, "alice", "bob", 1.0, "ctx-1")
        assert result is False
        assert la.list_pending(kg, limit=10, threshold=0.0) == []

    def test_unrelated_pair_still_upserts_when_edge_exists_elsewhere(self, kg):
        """Direct-edge check is scoped to the PAIR, not the graph."""
        self._seed_edge(kg, "alice", "knows", "bob")
        result = la.upsert_candidate(kg, "alice", "charlie", 1.0, "ctx-1")
        assert result is True
        rows = la.list_pending(kg, limit=10, threshold=0.0)
        assert len(rows) == 1
        assert rows[0]["from_entity"] == "alice"
        assert rows[0]["to_entity"] == "charlie"


# ─────────────────────────────────────────────────────────────────────
# list_pending threshold + ordering
# ─────────────────────────────────────────────────────────────────────


class TestListPending:
    def test_threshold_filters_out_low_scores(self, kg):
        # One pair at score=0.5 (below 1.5 default), one at 3*log(2)
        low_w = 0.5
        high_w = 1.0 / math.log(2)  # ≈ 1.44
        la.upsert_candidate(kg, "A", "B", low_w, "ctx-low")
        la.upsert_candidate(kg, "C", "D", high_w, "ctx-c1")
        la.upsert_candidate(kg, "C", "D", high_w, "ctx-c2")  # total ≈ 2.88

        # threshold 1.5: only C/D qualifies
        rows = la.list_pending(kg, limit=10, threshold=1.5)
        assert len(rows) == 1
        assert rows[0]["from_entity"] == "C"
        assert rows[0]["to_entity"] == "D"

    def test_limit_respected(self, kg):
        for i in range(5):
            la.upsert_candidate(kg, f"A{i}", f"B{i}", 1.0 / math.log(2), f"ctx-{i}")
        rows = la.list_pending(kg, limit=3, threshold=0.0)
        assert len(rows) == 3

    def test_results_ordered_by_score_desc(self, kg):
        # Build candidates with distinct total scores.
        la.upsert_candidate(kg, "A", "B", 1.0 / math.log(2), "c1")  # high
        la.upsert_candidate(kg, "C", "D", 1.0 / math.log(10), "c2")  # low
        la.upsert_candidate(kg, "E", "F", 1.0 / math.log(5), "c3")  # mid

        rows = la.list_pending(kg, limit=10, threshold=0.0)
        scores = [r["score"] for r in rows]
        assert scores == sorted(scores, reverse=True)


# ─────────────────────────────────────────────────────────────────────
# Input-shape guards
# ─────────────────────────────────────────────────────────────────────


class TestInputGuards:
    def test_empty_from_entity_refused(self, kg):
        assert la.upsert_candidate(kg, "", "B", 1.0, "ctx-1") is False

    def test_empty_to_entity_refused(self, kg):
        assert la.upsert_candidate(kg, "A", "", 1.0, "ctx-1") is False

    def test_empty_context_id_refused(self, kg):
        assert la.upsert_candidate(kg, "A", "B", 1.0, "") is False


# ─────────────────────────────────────────────────────────────────────
# Dispatch stub (Commit 2)
# ─────────────────────────────────────────────────────────────────────


class TestDispatchStub:
    def test_dispatch_is_noop_in_commit_2(self, kg):
        """_dispatch_if_due is a stub until Commit 3 wires the CLI. It
        must never raise regardless of state."""
        assert la._dispatch_if_due(kg, interval_hours=1) is None


pytestmark = pytest.mark.integration
