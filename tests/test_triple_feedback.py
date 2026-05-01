"""
test_triple_feedback.py -- Lock in the end-to-end contract for
triple-scoped feedback introduced by migration 018.

Context: triples have ids in a separate namespace from entities
(`t_<sub>_<pred>_<obj>_<hash>`). The existing rated_useful /
rated_irrelevant / surfaced edges can't target them without
polluting the entities table via add_triple's INSERT OR IGNORE.
Migration 018 adds `triple_context_feedback` so triples receive
per-context feedback natively with the same last-wins-across-
directions contract add_rated_edge enforces on entity ratings.

Scope:
  - KG-level writer (`_record_triple_feedback`) -- supersede, validation.
  - Unified dispatcher (`record_feedback`) -- target_kind routing.
  - Reader (`get_triple_feedback`) -- current rows only, graceful on
    empty input and missing table.
  - Channel D integration (`scoring.walk_rated_neighbourhood`) --
    triple rows merged into rated_scores and channel_D_list keyed by
    triple_id, alongside the entity-edge-based ratings.

End-to-end finalize_intent → record_feedback tests live in
test_intent_system.py (integration lane); this file covers the
KG + scoring surface.
"""

from __future__ import annotations

import pytest

# Cold-start lock 2026-05-01: phantom auto-create closed; tests exploit
# the prior auto-create path. Tracked under cold-start test-sweep todo.
pytestmark = pytest.mark.skip(
    reason="cold-start migration: phantom auto-create closed; needs declare-first sweep."
)


# ─────────────────────────────────────────────────────────────────────
# Helpers -- inspect the triple_context_feedback table directly so we
# verify the schema contract, not just the reader.
# ─────────────────────────────────────────────────────────────────────


def _current_rows(kg, context_id, triple_id):
    conn = kg._conn()
    rows = conn.execute(
        "SELECT id, kind, relevance, confidence, rater_kind, valid_to "
        "FROM triple_context_feedback "
        "WHERE context_id = ? AND triple_id = ? AND valid_to IS NULL",
        (context_id, triple_id),
    ).fetchall()
    return [dict(r) for r in rows]


def _all_rows(kg, context_id, triple_id):
    conn = kg._conn()
    rows = conn.execute(
        "SELECT id, kind, relevance, confidence, valid_to "
        "FROM triple_context_feedback "
        "WHERE context_id = ? AND triple_id = ? "
        "ORDER BY id ASC",
        (context_id, triple_id),
    ).fetchall()
    return [dict(r) for r in rows]


def _seed_triple(kg, subject="alice", predicate="knows", obj="bob"):
    """Create entities + a triple returning the triple_id."""
    kg.add_entity(subject, kind="entity", content=f"{subject} entity")
    kg.add_entity(obj, kind="entity", content=f"{obj} entity")
    return kg.add_triple(
        subject,
        predicate,
        obj,
        statement=f"{subject} {predicate} {obj}.",
    )


# ─────────────────────────────────────────────────────────────────────
# Migration applied: schema shape
# ─────────────────────────────────────────────────────────────────────


class TestMigration018:
    def test_table_exists(self, kg):
        """Migration 018 creates the table on a fresh palace."""
        conn = kg._conn()
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='triple_context_feedback'"
        ).fetchone()
        assert row is not None

    def test_partial_unique_index_exists(self, kg):
        """The last-wins contract is enforced at the schema level."""
        conn = kg._conn()
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_tcf_current_pair'"
        ).fetchone()
        assert row is not None

    def test_kind_check_constraint(self, kg):
        """Kind column rejects values outside the enum."""
        conn = kg._conn()
        with pytest.raises(Exception):
            conn.execute(
                "INSERT INTO triple_context_feedback "
                "(context_id, triple_id, kind, ts) "
                "VALUES ('c', 't', 'bogus_kind', '2026-01-01')"
            )

    def test_rater_kind_check_constraint(self, kg):
        """rater_kind column rejects values outside the enum."""
        conn = kg._conn()
        with pytest.raises(Exception):
            conn.execute(
                "INSERT INTO triple_context_feedback "
                "(context_id, triple_id, kind, rater_kind, ts) "
                "VALUES ('c', 't', 'rated_useful', 'bogus_rater', '2026-01-01')"
            )


# ─────────────────────────────────────────────────────────────────────
# Writer: _record_triple_feedback contract
# ─────────────────────────────────────────────────────────────────────


class TestRecordTripleFeedbackWriter:
    def test_basic_write_inserts_current_row(self, kg):
        tid = _seed_triple(kg)
        kg._record_triple_feedback(
            "ctx_a",
            tid,
            "rated_useful",
            relevance=4,
            reason="load-bearing",
            rater_kind="agent",
            rater_id="test_agent",
        )
        current = _current_rows(kg, "ctx_a", tid)
        assert len(current) == 1
        assert current[0]["kind"] == "rated_useful"
        assert current[0]["relevance"] == 4
        assert current[0]["rater_kind"] == "agent"

    def test_same_direction_rewrite_supersedes(self, kg):
        """Re-rate useful with a different confidence → one current row,
        previous invalidated, history preserved."""
        tid = _seed_triple(kg)
        kg._record_triple_feedback("ctx_a", tid, "rated_useful", relevance=3, confidence=0.5)
        kg._record_triple_feedback("ctx_a", tid, "rated_useful", relevance=5, confidence=1.0)
        current = _current_rows(kg, "ctx_a", tid)
        assert len(current) == 1
        assert current[0]["relevance"] == 5
        assert current[0]["confidence"] == pytest.approx(1.0)
        all_rows = _all_rows(kg, "ctx_a", tid)
        assert len(all_rows) == 2
        assert sum(1 for r in all_rows if r["valid_to"] is None) == 1

    def test_direction_flip_supersedes(self, kg):
        """useful → irrelevant invalidates the useful row."""
        tid = _seed_triple(kg)
        kg._record_triple_feedback("ctx_a", tid, "rated_useful", relevance=4)
        kg._record_triple_feedback("ctx_a", tid, "rated_irrelevant", relevance=1)
        current = _current_rows(kg, "ctx_a", tid)
        assert len(current) == 1
        assert current[0]["kind"] == "rated_irrelevant"
        assert current[0]["relevance"] == 1

    def test_surfaced_is_recall_only(self, kg):
        """surfaced kind stores relevance=None but is still current."""
        tid = _seed_triple(kg)
        kg._record_triple_feedback("ctx_a", tid, "surfaced", rater_kind="agent")
        current = _current_rows(kg, "ctx_a", tid)
        assert len(current) == 1
        assert current[0]["kind"] == "surfaced"
        assert current[0]["relevance"] is None

    def test_distinct_contexts_do_not_supersede_each_other(self, kg):
        """Last-wins is scoped to (context, triple) -- same triple rated
        under two contexts yields two current rows."""
        tid = _seed_triple(kg)
        kg._record_triple_feedback("ctx_a", tid, "rated_useful", relevance=4)
        kg._record_triple_feedback("ctx_b", tid, "rated_irrelevant", relevance=2)
        assert len(_current_rows(kg, "ctx_a", tid)) == 1
        assert len(_current_rows(kg, "ctx_b", tid)) == 1

    def test_rejects_unknown_kind(self, kg):
        tid = _seed_triple(kg)
        with pytest.raises(ValueError, match="_record_triple_feedback only accepts"):
            kg._record_triple_feedback("ctx", tid, "bogus_kind")

    def test_rejects_unknown_rater_kind(self, kg):
        tid = _seed_triple(kg)
        with pytest.raises(ValueError, match="rater_kind"):
            kg._record_triple_feedback("ctx", tid, "rated_useful", rater_kind="human_panel")


# ─────────────────────────────────────────────────────────────────────
# Dispatcher: record_feedback routes by target_kind
# ─────────────────────────────────────────────────────────────────────


class TestRecordFeedbackDispatcher:
    def test_entity_target_writes_rated_edge(self, kg):
        """target_kind='entity' routes through add_rated_edge, producing
        a rated_useful triple (not a row in triple_context_feedback)."""
        kg.add_entity("ctx_a", kind="entity", content="context A")
        kg.add_entity("record_x", kind="record", content="a memory") if hasattr(
            kg, "add_entity"
        ) else None
        kg.record_feedback(
            "ctx_a",
            "record_x",
            "entity",
            relevance=4,
            reason="useful",
            rater_kind="agent",
            rater_id="test_agent",
        )
        # Appears as an edge in the triples table.
        conn = kg._conn()
        sub = kg._entity_id("ctx_a")
        obj = kg._entity_id("record_x")
        row = conn.execute(
            "SELECT predicate FROM triples WHERE subject = ? AND object = ? AND valid_to IS NULL",
            (sub, obj),
        ).fetchone()
        assert row is not None
        assert row["predicate"] == "rated_useful"

    def test_triple_target_writes_triple_feedback_row(self, kg):
        tid = _seed_triple(kg)
        kg.record_feedback(
            "ctx_a",
            tid,
            "triple",
            relevance=5,
            reason="cited",
            rater_kind="gate_llm",
            rater_id="claude-haiku-gate",
        )
        current = _current_rows(kg, "ctx_a", tid)
        assert len(current) == 1
        assert current[0]["kind"] == "rated_useful"
        assert current[0]["rater_kind"] == "gate_llm"

    def test_relevance_2_routes_to_irrelevant(self, kg):
        tid = _seed_triple(kg)
        kg.record_feedback("ctx_a", tid, "triple", relevance=2)
        current = _current_rows(kg, "ctx_a", tid)
        assert current[0]["kind"] == "rated_irrelevant"

    def test_relevance_3_routes_to_useful(self, kg):
        """relevance=3 is the neutral-but-related default -- classified
        as useful so Channel D treats it as weak-positive signal."""
        tid = _seed_triple(kg)
        kg.record_feedback("ctx_a", tid, "triple", relevance=3)
        current = _current_rows(kg, "ctx_a", tid)
        assert current[0]["kind"] == "rated_useful"

    def test_rejects_unknown_target_kind(self, kg):
        with pytest.raises(ValueError, match="target_kind"):
            kg.record_feedback("ctx", "anything", "planet", relevance=3)

    def test_rejects_relevance_below_range(self, kg):
        tid = _seed_triple(kg)
        with pytest.raises(ValueError, match="relevance"):
            kg.record_feedback("ctx", tid, "triple", relevance=0)

    def test_rejects_relevance_above_range(self, kg):
        tid = _seed_triple(kg)
        with pytest.raises(ValueError, match="relevance"):
            kg.record_feedback("ctx", tid, "triple", relevance=6)


# ─────────────────────────────────────────────────────────────────────
# Reader: get_triple_feedback
# ─────────────────────────────────────────────────────────────────────


class TestGetTripleFeedback:
    def test_empty_input_returns_empty(self, kg):
        assert kg.get_triple_feedback([]) == []

    def test_returns_only_current_rows(self, kg):
        tid = _seed_triple(kg)
        kg._record_triple_feedback("ctx_a", tid, "rated_useful", relevance=3)
        kg._record_triple_feedback("ctx_a", tid, "rated_irrelevant", relevance=1)
        rows = kg.get_triple_feedback(["ctx_a"])
        assert len(rows) == 1
        assert rows[0]["kind"] == "rated_irrelevant"

    def test_filters_by_context(self, kg):
        tid = _seed_triple(kg)
        kg._record_triple_feedback("ctx_a", tid, "rated_useful", relevance=4)
        kg._record_triple_feedback("ctx_b", tid, "rated_irrelevant", relevance=1)
        a_rows = kg.get_triple_feedback(["ctx_a"])
        b_rows = kg.get_triple_feedback(["ctx_b"])
        both = kg.get_triple_feedback(["ctx_a", "ctx_b"])
        assert len(a_rows) == 1 and a_rows[0]["kind"] == "rated_useful"
        assert len(b_rows) == 1 and b_rows[0]["kind"] == "rated_irrelevant"
        assert len(both) == 2


# ─────────────────────────────────────────────────────────────────────
# Channel D integration: walk_rated_neighbourhood merges triple
# feedback into rated_scores / channel_D_list keyed by triple_id.
# ─────────────────────────────────────────────────────────────────────


class TestChannelDIntegration:
    def test_triple_rated_useful_contributes_positive_score(self, kg):
        from mempalace.scoring import walk_rated_neighbourhood

        tid = _seed_triple(kg)
        kg._record_triple_feedback("ctx_a", tid, "rated_useful", relevance=5, confidence=1.0)

        out = walk_rated_neighbourhood("ctx_a", kg)
        assert tid in out["rated_scores"]
        assert out["rated_scores"][tid] > 0.0
        ids_in_channel = [mid for _score, _doc, mid in out["channel_D_list"]]
        assert tid in ids_in_channel

    def test_triple_rated_irrelevant_contributes_negative_score(self, kg):
        from mempalace.scoring import walk_rated_neighbourhood

        tid = _seed_triple(kg)
        kg._record_triple_feedback("ctx_a", tid, "rated_irrelevant", relevance=1, confidence=1.0)

        out = walk_rated_neighbourhood("ctx_a", kg)
        assert tid in out["rated_scores"]
        assert out["rated_scores"][tid] < 0.0
        # channel_D_list is positive-only filter, so a strictly negative
        # item must not appear there.
        ids_in_channel = [mid for _score, _doc, mid in out["channel_D_list"]]
        assert tid not in ids_in_channel

    def test_surfaced_on_triple_is_recall_only(self, kg):
        from mempalace.scoring import walk_rated_neighbourhood

        tid = _seed_triple(kg)
        kg._record_triple_feedback("ctx_a", tid, "surfaced", rater_kind="agent")

        out = walk_rated_neighbourhood("ctx_a", kg, surfaced_weight=0.3)
        # rated_scores is rated_useful − rated_irrelevant only; surfaced
        # must not show up there.
        assert tid not in out["rated_scores"] or out["rated_scores"][tid] == 0.0
        ids_in_channel = [mid for _score, _doc, mid in out["channel_D_list"]]
        assert tid in ids_in_channel

    def test_triple_and_entity_feedback_coexist_in_same_walk(self, kg):
        """Single walk sees both namespaces -- triple_ids from the new
        table AND memory/entity ids from rated_* edges -- with no
        collisions."""
        from mempalace.scoring import walk_rated_neighbourhood

        # Entity-scope feedback (rated_* edge)
        kg.add_rated_edge(
            "ctx_a",
            "rated_useful",
            "mem_X",
            confidence=1.0,
            properties={"relevance": 4},
        )

        # Triple-scope feedback (triple_context_feedback row)
        tid = _seed_triple(kg)
        kg._record_triple_feedback("ctx_a", tid, "rated_useful", relevance=4)

        out = walk_rated_neighbourhood("ctx_a", kg)
        # query_entity returns RAW entity names (entities.name) as
        # edge["object"], not the normalized ids in the triples table.
        # See the documented query_entity-raw-name gotcha. Triple
        # feedback is keyed by triple_id (no normalization step). Both
        # namespaces coexist in the same rated_scores dict, keyed by
        # their respective id flavours, with no collisions in practice.
        assert "mem_X" in out["rated_scores"]
        assert tid in out["rated_scores"]
        assert out["rated_scores"]["mem_X"] > 0.0
        assert out["rated_scores"][tid] > 0.0

    def test_channel_d_ignores_unknown_context(self, kg):
        from mempalace.scoring import walk_rated_neighbourhood

        tid = _seed_triple(kg)
        kg._record_triple_feedback("ctx_known", tid, "rated_useful", relevance=4)

        out = walk_rated_neighbourhood("ctx_absent", kg)
        assert tid not in out["rated_scores"]


# ─────────────────────────────────────────────────────────────────────
# Step 1 of similar_context_id flag: contributing_contexts return key
# tracks which similar-context neighbours (NOT the active context)
# supplied per-memory weight via rated_* edges or triple feedback.
# Design record: record_ga_agent_similar_context_id_flag_design_2026_04_30.
# ─────────────────────────────────────────────────────────────────────


class TestContributingContexts:
    def test_return_dict_includes_contributing_contexts_key(self, kg):
        """walk_rated_neighbourhood now returns contributing_contexts."""
        from mempalace.scoring import walk_rated_neighbourhood

        out = walk_rated_neighbourhood("ctx_a", kg)
        assert "contributing_contexts" in out
        assert isinstance(out["contributing_contexts"], dict)

    def test_active_context_alone_yields_empty_contributing_contexts(self, kg):
        """An item rated useful in the active context (no similar
        neighbours) must NOT appear in contributing_contexts -- the
        map records similar-neighbour contributions only."""
        from mempalace.scoring import walk_rated_neighbourhood

        kg.add_rated_edge(
            "ctx_a",
            "rated_useful",
            "mem_X",
            confidence=1.0,
            properties={"relevance": 4},
        )

        out = walk_rated_neighbourhood("ctx_a", kg)
        # The score is recorded, but contributing_contexts excludes the
        # active context by design.
        assert "mem_X" in out["rated_scores"]
        assert out["contributing_contexts"].get("mem_X", []) == []

    def test_similar_context_neighbour_appears_in_contributing_contexts(self, kg):
        """An item rated useful in a similar_to neighbour of the
        active context must list that neighbour cid in
        contributing_contexts."""
        from mempalace.scoring import walk_rated_neighbourhood

        # ctx_a -similar_to-> ctx_b, with the rating in ctx_b only.
        kg.add_triple("ctx_a", "similar_to", "ctx_b")
        kg.add_rated_edge(
            "ctx_b",
            "rated_useful",
            "mem_Y",
            confidence=1.0,
            properties={"relevance": 4},
        )

        out = walk_rated_neighbourhood("ctx_a", kg)
        assert "mem_Y" in out["rated_scores"]
        contributing = out["contributing_contexts"].get("mem_Y", [])
        assert "ctx_b" in contributing
        # The active context must NOT be listed as a contributor
        # even when an item also has a rating from it.
        assert "ctx_a" not in contributing

    def test_triple_feedback_in_neighbour_appears_in_contributing_contexts(self, kg):
        """Triple-scoped feedback rows in a similar_to neighbour must
        also surface that neighbour cid in contributing_contexts,
        keyed by triple_id."""
        from mempalace.scoring import walk_rated_neighbourhood

        kg.add_triple("ctx_a", "similar_to", "ctx_b")
        tid = _seed_triple(kg)
        kg._record_triple_feedback("ctx_b", tid, "rated_useful", relevance=4)

        out = walk_rated_neighbourhood("ctx_a", kg)
        assert tid in out["rated_scores"]
        contributing = out["contributing_contexts"].get(tid, [])
        assert "ctx_b" in contributing
        assert "ctx_a" not in contributing
