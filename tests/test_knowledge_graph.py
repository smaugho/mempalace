"""
test_knowledge_graph.py — Tests for the temporal knowledge graph.

Covers: entity CRUD, triple CRUD, temporal queries, invalidation,
timeline, stats, and edge cases (duplicate triples, ID collisions).
"""


class TestEntityOperations:
    def test_add_entity(self, kg):
        eid = kg.add_entity("Alice", kind="entity", description="A person named Alice")
        assert eid == "alice"

    def test_add_entity_normalizes_id(self, kg):
        eid = kg.add_entity("Dr. Chen", kind="entity", description="A person named Dr. Chen")
        assert eid == "dr_chen"

    def test_add_entity_upsert(self, kg):
        kg.add_entity("Alice", kind="entity", description="A person named Alice")
        kg.add_entity("Alice", kind="entity", description="An engineer named Alice")
        # Should not raise — INSERT OR REPLACE
        stats = kg.stats()
        assert stats["entities"] == 1


class TestTripleOperations:
    def test_add_triple_creates_entities(self, kg):
        tid = kg.add_triple("Alice", "knows", "Bob", statement="Alice knows Bob.")
        assert tid.startswith("t_alice_knows_bob_")
        stats = kg.stats()
        assert stats["entities"] == 2  # auto-created

    def test_add_triple_with_dates(self, kg):
        tid = kg.add_triple(
            "Max",
            "does",
            "swimming",
            valid_from="2025-01-01",
            statement="Max started doing swimming on 2025-01-01.",
        )
        assert tid.startswith("t_max_does_swimming_")

    def test_duplicate_triple_returns_existing_id(self, kg):
        tid1 = kg.add_triple("Alice", "knows", "Bob", statement="Alice knows Bob.")
        tid2 = kg.add_triple("Alice", "knows", "Bob", statement="Alice knows Bob.")
        assert tid1 == tid2

    def test_add_triple_normalizes_hyphenated_predicate(self, kg):
        """Predicate 'is-a' must be stored as 'is_a' so hyphen/underscore
        callers land on the same edge (regression: storage boundary was
        only collapsing spaces, hyphens survived as a separate predicate).

        Note: ``is_a`` is a skip-list predicate, so no `statement` is needed.
        """
        kg.add_triple("Alice", "is-a", "person")
        edges = kg.query_entity("Alice", direction="outgoing")
        preds = [e["predicate"] for e in edges]
        assert "is_a" in preds
        assert "is-a" not in preds

    def test_invalidated_triple_allows_re_add(self, kg):
        tid1 = kg.add_triple("Alice", "works_at", "Acme", statement="Alice works at Acme.")
        kg.invalidate("Alice", "works_at", "Acme", ended="2025-01-01")
        tid2 = kg.add_triple(
            "Alice",
            "works_at",
            "Acme",
            statement="Alice re-joined Acme after the previous engagement was closed.",
        )
        assert tid1 != tid2  # new triple since old one was closed


class TestQueries:
    def test_query_outgoing(self, seeded_kg):
        results = seeded_kg.query_entity("Alice", direction="outgoing")
        predicates = {r["predicate"] for r in results}
        assert "parent_of" in predicates
        assert "works_at" in predicates

    def test_query_incoming(self, seeded_kg):
        results = seeded_kg.query_entity("Max", direction="incoming")
        assert any(r["subject"] == "Alice" and r["predicate"] == "parent_of" for r in results)

    def test_query_both_directions(self, seeded_kg):
        results = seeded_kg.query_entity("Max", direction="both")
        directions = {r["direction"] for r in results}
        assert "outgoing" in directions
        assert "incoming" in directions

    def test_query_as_of_filters_expired(self, seeded_kg):
        results = seeded_kg.query_entity("Alice", as_of="2023-06-01", direction="outgoing")
        employers = [r["object"] for r in results if r["predicate"] == "works_at"]
        assert "Acme Corp" in employers
        assert "NewCo" not in employers

    def test_query_as_of_shows_current(self, seeded_kg):
        results = seeded_kg.query_entity("Alice", as_of="2025-06-01", direction="outgoing")
        employers = [r["object"] for r in results if r["predicate"] == "works_at"]
        assert "NewCo" in employers
        assert "Acme Corp" not in employers

    def test_query_relationship(self, seeded_kg):
        results = seeded_kg.query_relationship("does")
        assert len(results) == 2  # swimming + chess


class TestInvalidation:
    def test_invalidate_sets_valid_to(self, seeded_kg):
        seeded_kg.invalidate("Max", "does", "chess", ended="2026-01-01")
        results = seeded_kg.query_entity("Max", direction="outgoing")
        chess = [r for r in results if r["object"] == "chess"]
        assert len(chess) == 1
        assert chess[0]["valid_to"] == "2026-01-01"
        assert chess[0]["current"] is False


class TestTimeline:
    def test_timeline_all(self, seeded_kg):
        tl = seeded_kg.timeline()
        assert len(tl) >= 4

    def test_timeline_entity(self, seeded_kg):
        tl = seeded_kg.timeline("Max")
        subjects_and_objects = {t["subject"] for t in tl} | {t["object"] for t in tl}
        assert "Max" in subjects_and_objects

    def test_timeline_global_has_limit(self, kg):
        # Add > 100 triples
        for i in range(105):
            kg.add_triple(
                f"entity_{i}",
                "relates_to",
                f"entity_{i + 1}",
                statement=f"entity_{i} relates to entity_{i + 1}.",
            )
        tl = kg.timeline()
        assert len(tl) == 100  # LIMIT 100

    def test_timeline_entity_has_limit(self, kg):
        # Add > 100 triples all connected to a single entity
        for i in range(105):
            kg.add_triple(
                "hub",
                "connects_to",
                f"spoke_{i}",
                valid_from=f"2025-01-{(i % 28) + 1:02d}",
                statement=f"The hub connects to spoke_{i}.",
            )
        tl = kg.timeline("hub")
        assert len(tl) == 100  # LIMIT 100 on entity-filtered branch


class TestWALMode:
    def test_wal_mode_enabled(self, kg):
        conn = kg._conn()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal"


class TestStats:
    def test_stats_empty(self, kg):
        stats = kg.stats()
        assert stats["entities"] == 0
        assert stats["triples"] == 0

    def test_stats_seeded(self, seeded_kg):
        stats = seeded_kg.stats()
        # seeded_kg now also declares `test_agent is_a agent` so
        # write tools validate against a real declared agent; that adds
        # 2 entities (test_agent + agent class) and 1 triple (is_a edge)
        # to the baseline.
        assert stats["entities"] >= 4
        assert stats["triples"] == 6
        assert stats["current_facts"] == 5  # 1 expired (Acme Corp), 1 new is_a
        assert stats["expired_facts"] == 1


# ── Skip-list audit (2026-04-25) ──────────────────────────────────────
# Adrian's audit identified two corrections to _TRIPLE_SKIP_PREDICATES:
#   1. has_value REMOVED: it carries actual values (port numbers, etc.)
#      that benefit from cosine search; skipping its statement embedding
#      meant those values were unreachable via semantic retrieval.
#   2. performed_well / performed_poorly / executed_op / superseded_by
#      ADDED: these are pure graph topology written by finalize_intent
#      and walked by retrieve_past_operations via similar_to neighbours,
#      never searched by statement. Without skip-list inclusion, the
#      caller-provided statement was required and silent try/except
#      around add_triple would drop edges (same class as the templatizes
#      bug fixed earlier today).


class TestTripleSkipListAudit:
    """Pin the 2026-04-25 audit decisions so a future refactor can't
    silently re-add has_value or drop the op-tier predicates.
    """

    def test_has_value_is_NOT_skip_listed(self):
        """Adrian's call: has_value carries actual values that should
        be cosine-searchable. Statements like 'server has_value port
        8080' need to be embedded for retrieval to find them."""
        from mempalace.knowledge_graph import _TRIPLE_SKIP_PREDICATES

        assert "has_value" not in _TRIPLE_SKIP_PREDICATES, (
            "has_value re-added to skip list — values would no longer "
            "embed for semantic search. See the comment above "
            "_TRIPLE_SKIP_PREDICATES in knowledge_graph.py."
        )

    def test_op_tier_rating_predicates_are_skip_listed(self):
        """performed_well/poorly/executed_op/superseded_by are pure
        graph topology. Their statement embedding adds noise without
        retrieval signal — they're walked by retrieve_past_operations
        via similar_to neighbours, never matched by cosine on
        statement text.
        """
        from mempalace.knowledge_graph import _TRIPLE_SKIP_PREDICATES

        for pred in (
            "executed_op",
            "performed_well",
            "performed_poorly",
            "superseded_by",
        ):
            assert pred in _TRIPLE_SKIP_PREDICATES, (
                f"{pred!r} dropped from skip list — finalize_intent's silent "
                "try/except around add_triple would now drop these edges."
            )

    def test_canonical_skip_list_predicates_still_skip(self):
        """Regression guard: the structural / context-bookkeeping
        predicates that always belonged in the skip list must stay."""
        from mempalace.knowledge_graph import _TRIPLE_SKIP_PREDICATES

        for pred in (
            "is_a",
            "described_by",
            "rated_useful",
            "rated_irrelevant",
            "similar_to",
            "surfaced",
            "templatizes",
        ):
            assert pred in _TRIPLE_SKIP_PREDICATES, (
                f"{pred!r} dropped from skip list — non-structural callers "
                "would now be forced to provide statements."
            )


# ── Structured-summary validation (2026-04-25) ────────────────────────
# WHAT+WHY+SCOPE? shape locked with Adrian. Storage is the dict for
# field-level validation; embedding is prose via
# serialize_summary_for_embedding. Legacy strings accepted with a
# heuristic clause-separator check (em-dash, semicolon, role-verbs).


class TestValidateSummary:
    """Lock the structural-summary contract.

    Each test names the failure mode it guards against in plain
    language so a future refactor can't silently relax the rule.
    """

    def test_dict_with_what_and_why_passes(self):
        from mempalace.knowledge_graph import validate_summary

        assert validate_summary(
            {
                "what": "InjectionGate",
                "why": "filters retrieved memories before injection via Haiku",
            }
        )

    def test_dict_with_scope_passes(self):
        from mempalace.knowledge_graph import validate_summary

        assert validate_summary(
            {
                "what": "InjectionGate",
                "why": "filters retrieved memories before injection",
                "scope": "one instance per palace process",
            }
        )

    def test_dict_missing_what_rejects(self):
        from mempalace.knowledge_graph import (
            SummaryStructureRequired,
            validate_summary,
        )
        import pytest

        with pytest.raises(SummaryStructureRequired, match="'what'"):
            validate_summary({"why": "filters retrieved memories"})

    def test_dict_missing_why_rejects(self):
        from mempalace.knowledge_graph import (
            SummaryStructureRequired,
            validate_summary,
        )
        import pytest

        with pytest.raises(SummaryStructureRequired, match="'why'"):
            validate_summary({"what": "InjectionGate"})

    def test_dict_stub_what_rejects(self):
        from mempalace.knowledge_graph import (
            SummaryStructureRequired,
            validate_summary,
        )
        import pytest

        with pytest.raises(SummaryStructureRequired, match="'what'"):
            validate_summary({"what": "X", "why": "filters retrieved memories before injection"})

    def test_dict_stub_why_rejects(self):
        from mempalace.knowledge_graph import (
            SummaryStructureRequired,
            validate_summary,
        )
        import pytest

        with pytest.raises(SummaryStructureRequired, match="'why'"):
            validate_summary({"what": "InjectionGate", "why": "is real"})

    def test_dict_oversized_scope_rejects(self):
        from mempalace.knowledge_graph import (
            SummaryStructureRequired,
            validate_summary,
        )
        import pytest

        with pytest.raises(SummaryStructureRequired, match="scope"):
            validate_summary(
                {
                    "what": "InjectionGate",
                    "why": "filters retrieved memories before injection",
                    "scope": "X" * 200,
                }
            )

    def test_dict_total_oversized_rejects(self):
        from mempalace.knowledge_graph import (
            SummaryStructureRequired,
            validate_summary,
        )
        import pytest

        with pytest.raises(SummaryStructureRequired, match="exceeds"):
            validate_summary(
                {
                    "what": "X" * 80,
                    "why": "Y" * 200,
                    "scope": "Z" * 50,
                }
            )

    def test_string_with_em_dash_passes(self):
        from mempalace.knowledge_graph import validate_summary

        assert validate_summary("InjectionGate — runtime gate that filters retrieved memories")

    def test_string_with_role_verb_passes(self):
        """Role-verbs without em-dash should pass — role verb is a
        sufficient signal of WHY-clause presence."""
        from mempalace.knowledge_graph import validate_summary

        assert validate_summary("InjectionGate orchestrates the Haiku gate for retrieved memories")

    def test_string_too_short_rejects(self):
        from mempalace.knowledge_graph import (
            SummaryStructureRequired,
            validate_summary,
        )
        import pytest

        with pytest.raises(SummaryStructureRequired, match="too short"):
            validate_summary("InjectionGate")

    def test_string_no_separator_rejects(self):
        """Single noun phrase or name-restating placeholder rejects."""
        from mempalace.knowledge_graph import (
            SummaryStructureRequired,
            validate_summary,
        )
        import pytest

        with pytest.raises(SummaryStructureRequired, match="WHAT\\+WHY"):
            validate_summary("Adrian Rivero is the project owner of mempalace today")

    def test_string_too_long_rejects(self):
        from mempalace.knowledge_graph import (
            SummaryStructureRequired,
            validate_summary,
        )
        import pytest

        with pytest.raises(SummaryStructureRequired, match="too long"):
            validate_summary("X — " + "Y" * 300)

    def test_legacy_string_disallowed_when_strict(self):
        from mempalace.knowledge_graph import (
            SummaryStructureRequired,
            validate_summary,
        )
        import pytest

        with pytest.raises(SummaryStructureRequired, match="legacy string"):
            validate_summary(
                "InjectionGate — filters retrieved memories",
                allow_legacy_string=False,
            )

    def test_non_string_non_dict_rejects(self):
        from mempalace.knowledge_graph import (
            SummaryStructureRequired,
            validate_summary,
        )
        import pytest

        with pytest.raises(SummaryStructureRequired, match="must be"):
            validate_summary(123)

    def test_context_for_error_appears_in_message(self):
        """Error messages must name the call site so callers know
        which write rejected. Without this, agents debugging a
        rejection have to grep call sites blind."""
        from mempalace.knowledge_graph import (
            SummaryStructureRequired,
            validate_summary,
        )
        import pytest

        with pytest.raises(SummaryStructureRequired, match="kg_declare_entity\\.summary"):
            validate_summary("X", context_for_error="kg_declare_entity.summary")


class TestSerializeSummaryForEmbedding:
    """The dict-storage / prose-embedding split: dict goes through
    validation, prose hits Chroma. Test the projection.
    """

    def test_dict_renders_with_em_dash(self):
        from mempalace.knowledge_graph import serialize_summary_for_embedding

        text = serialize_summary_for_embedding(
            {"what": "InjectionGate", "why": "filters retrieved memories"}
        )
        assert "—" in text  # em-dash separator
        assert "InjectionGate" in text
        assert "filters retrieved memories" in text

    def test_dict_with_scope_appends_with_semicolon(self):
        from mempalace.knowledge_graph import serialize_summary_for_embedding

        text = serialize_summary_for_embedding(
            {
                "what": "InjectionGate",
                "why": "filters retrieved memories",
                "scope": "one instance per palace process",
            }
        )
        assert text.endswith("one instance per palace process")
        assert "; one instance" in text

    def test_string_passes_through(self):
        from mempalace.knowledge_graph import serialize_summary_for_embedding

        text = serialize_summary_for_embedding("InjectionGate — runtime gate")
        assert text == "InjectionGate — runtime gate"

    def test_empty_dict_renders_empty_string(self):
        from mempalace.knowledge_graph import serialize_summary_for_embedding

        assert serialize_summary_for_embedding({}) == ""
