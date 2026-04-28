"""
test_rated_edge_supersede.py -- Lock in the last-wins-across-directions
contract for rating edges introduced 2026-04-22.

Background: kg.add_triple short-circuits on duplicate PK
(subject, predicate, object) when valid_to IS NULL, silently returning
the existing row without writing the new data. That behaviour is
correct for structural predicates (you don't need two
'Max is_a person' rows), but wrong for rating edges where the
additional properties (confidence, reason, agent, ts) carry legitimate
new information.

kg.add_rated_edge fixes this for the rating-predicate closed set
{rated_useful, rated_irrelevant} with these semantics:
  - at most ONE current (valid_to IS NULL) rating edge per
    (context, memory) pair, regardless of direction
  - writing a new rating invalidates any prior rating on the same
    pair before inserting
  - history is preserved via valid_to (bitemporal)

Scope: these tests cover the KG-level contract only. End-to-end
feedback-loop tests (finalize_intent → add_rated_edge) live in
test_intent_system.py.
"""

from __future__ import annotations

import pytest


# ─────────────────────────────────────────────────────────────────────
# Helper: count current rating edges on a pair, scanning both directions.
# ─────────────────────────────────────────────────────────────────────


def _current_ratings(kg, ctx_id, mem_id):
    # Names must be resolved through kg._entity_id so the SELECT matches
    # the normalized ids add_rated_edge actually wrote (mem_X -> mem_x).
    sub = kg._entity_id(ctx_id)
    obj = kg._entity_id(mem_id)
    conn = kg._conn()
    rows = conn.execute(
        "SELECT id, predicate, confidence, properties, valid_to "
        "FROM triples "
        "WHERE subject = ? AND object = ? "
        "AND predicate IN ('rated_useful', 'rated_irrelevant') "
        "AND valid_to IS NULL",
        (sub, obj),
    ).fetchall()
    return [dict(r) for r in rows]


def _all_ratings_including_history(kg, ctx_id, mem_id):
    sub = kg._entity_id(ctx_id)
    obj = kg._entity_id(mem_id)
    conn = kg._conn()
    rows = conn.execute(
        "SELECT id, predicate, confidence, valid_to "
        "FROM triples "
        "WHERE subject = ? AND object = ? "
        "AND predicate IN ('rated_useful', 'rated_irrelevant')",
        (sub, obj),
    ).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────
# The four failure modes the fix closes
# ─────────────────────────────────────────────────────────────────────


class TestSupersedeSemantics:
    def test_same_direction_stronger_rating_supersedes_weaker(self, kg):
        """Re-rating the same (ctx, mem) with a stronger useful rating
        replaces the weaker one. The old rating becomes history."""
        kg.add_rated_edge(
            "ctx_auth",
            "rated_useful",
            "mem_X",
            confidence=0.8,
            properties={"relevance": 4, "agent": "a"},
        )
        kg.add_rated_edge(
            "ctx_auth",
            "rated_useful",
            "mem_X",
            confidence=1.0,
            properties={"relevance": 5, "agent": "a"},
        )
        current = _current_ratings(kg, "ctx_auth", "mem_X")
        assert len(current) == 1
        assert current[0]["predicate"] == "rated_useful"
        assert current[0]["confidence"] == pytest.approx(1.0)

    def test_same_direction_weaker_rating_supersedes_stronger(self, kg):
        """The 'I rated too high, let me correct down' case."""
        kg.add_rated_edge(
            "ctx_auth",
            "rated_useful",
            "mem_X",
            confidence=1.0,
            properties={"relevance": 5, "agent": "a"},
        )
        kg.add_rated_edge(
            "ctx_auth",
            "rated_useful",
            "mem_X",
            confidence=0.2,
            properties={"relevance": 3, "agent": "a"},
        )
        current = _current_ratings(kg, "ctx_auth", "mem_X")
        assert len(current) == 1
        assert current[0]["confidence"] == pytest.approx(0.2)

    def test_flip_useful_to_irrelevant_supersedes_cross_predicate(self, kg):
        """The 'I was wrong, this memory is actually bad for this task'
        case. Without the fix both edges would coexist and cancel in
        walk_rated_neighbourhood. With the fix, only the newer irrelevant
        is current."""
        kg.add_rated_edge(
            "ctx_auth",
            "rated_useful",
            "mem_X",
            confidence=1.0,
            properties={"relevance": 5, "agent": "a"},
        )
        kg.add_rated_edge(
            "ctx_auth",
            "rated_irrelevant",
            "mem_X",
            confidence=1.0,
            properties={"relevance": 1, "agent": "a"},
        )
        current = _current_ratings(kg, "ctx_auth", "mem_X")
        assert len(current) == 1
        assert current[0]["predicate"] == "rated_irrelevant"
        assert current[0]["confidence"] == pytest.approx(1.0)

    def test_flip_irrelevant_to_useful_supersedes_cross_predicate(self, kg):
        """Mirror of the above -- reassessing 'actually this IS useful'."""
        kg.add_rated_edge(
            "ctx_auth",
            "rated_irrelevant",
            "mem_X",
            confidence=1.0,
            properties={"relevance": 1, "agent": "a"},
        )
        kg.add_rated_edge(
            "ctx_auth",
            "rated_useful",
            "mem_X",
            confidence=0.8,
            properties={"relevance": 4, "agent": "a"},
        )
        current = _current_ratings(kg, "ctx_auth", "mem_X")
        assert len(current) == 1
        assert current[0]["predicate"] == "rated_useful"

    def test_different_agents_last_wins_globally(self, kg):
        """Simpler path chosen over per-agent supersede: whoever rates
        most recently wins. Multi-agent consensus does NOT stack under
        this rule -- that would require the per-agent supersede variant
        (deferred until cross-rater use cases are live)."""
        kg.add_rated_edge(
            "ctx_auth",
            "rated_useful",
            "mem_X",
            confidence=0.8,
            properties={"relevance": 4, "agent": "agent_a"},
        )
        kg.add_rated_edge(
            "ctx_auth",
            "rated_useful",
            "mem_X",
            confidence=1.0,
            properties={"relevance": 5, "agent": "agent_b"},
        )
        current = _current_ratings(kg, "ctx_auth", "mem_X")
        assert len(current) == 1
        assert current[0]["confidence"] == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────
# History preservation -- invalidated edges stay queryable
# ─────────────────────────────────────────────────────────────────────


class TestHistoryPreserved:
    def test_invalidated_edges_remain_queryable(self, kg):
        """Bitemporal: superseded edges get valid_to set, not deleted."""
        kg.add_rated_edge(
            "ctx_auth",
            "rated_useful",
            "mem_X",
            confidence=0.8,
            properties={"relevance": 4, "agent": "a"},
        )
        kg.add_rated_edge(
            "ctx_auth",
            "rated_irrelevant",
            "mem_X",
            confidence=1.0,
            properties={"relevance": 1, "agent": "a"},
        )
        all_edges = _all_ratings_including_history(kg, "ctx_auth", "mem_X")
        assert len(all_edges) == 2
        invalidated = [e for e in all_edges if e["valid_to"] is not None]
        current = [e for e in all_edges if e["valid_to"] is None]
        assert len(invalidated) == 1
        assert len(current) == 1
        assert invalidated[0]["predicate"] == "rated_useful"
        assert current[0]["predicate"] == "rated_irrelevant"


# ─────────────────────────────────────────────────────────────────────
# Structural-predicate non-regression -- add_triple still dedups
# ─────────────────────────────────────────────────────────────────────


class TestAddTripleStillDedups:
    def test_is_a_still_first_wins(self, kg):
        """add_triple's dedup behaviour is UNCHANGED for structural
        predicates. Only rating predicates get the supersede treatment."""
        kg.add_entity("person", kind="class", description="root")
        kg.add_entity("max", kind="entity", description="Max")
        id1 = kg.add_triple("max", "is_a", "person")
        id2 = kg.add_triple("max", "is_a", "person")
        # Same id returned -- no second row written.
        assert id1 == id2
        conn = kg._conn()
        rows = conn.execute(
            "SELECT id FROM triples WHERE subject = 'max' AND predicate = 'is_a'"
        ).fetchall()
        assert len(rows) == 1


# ─────────────────────────────────────────────────────────────────────
# Input guards
# ─────────────────────────────────────────────────────────────────────


class TestInputGuards:
    def test_non_rating_predicate_raises(self, kg):
        """add_rated_edge is strictly the rating writer; other predicates
        must go through add_triple (which keeps its dedup semantics)."""
        with pytest.raises(ValueError) as exc:
            kg.add_rated_edge(
                "ctx_auth",
                "loves",
                "mem_X",
                confidence=1.0,
                properties={"relevance": 5, "agent": "a"},
            )
        assert "rating predicates" in str(exc.value).lower()

    def test_empty_predicate_raises(self, kg):
        with pytest.raises(ValueError):
            kg.add_rated_edge(
                "ctx_auth",
                "",
                "mem_X",
                confidence=1.0,
                properties={},
            )
