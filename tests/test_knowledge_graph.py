"""
test_knowledge_graph.py -- Tests for the temporal knowledge graph.

Covers: entity CRUD, triple CRUD, temporal queries, invalidation,
timeline, stats, and edge cases (duplicate triples, ID collisions).
"""


class TestEntityOperations:
    def test_add_entity(self, kg):
        eid = kg.add_entity("Alice", kind="entity", content="A person named Alice")
        assert eid == "alice"

    def test_add_entity_normalizes_id(self, kg):
        eid = kg.add_entity("Dr. Chen", kind="entity", content="A person named Dr. Chen")
        assert eid == "dr_chen"

    def test_add_entity_upsert(self, kg):
        kg.add_entity("Alice", kind="entity", content="A person named Alice")
        kg.add_entity("Alice", kind="entity", content="An engineer named Alice")
        # Should not raise -- INSERT OR REPLACE
        stats = kg.stats()
        assert stats["entities"] == 1


class TestTripleOperations:
    def test_add_triple_rejects_undeclared_endpoints(self, kg):
        """Cold-start lock 2026-05-01: add_triple no longer phantom-
        creates missing endpoints. Both subject and object must exist
        before any edge is written. Replaces the prior
        ``test_add_triple_creates_entities`` which exploited the
        phantom auto-create path.
        """
        from mempalace.entity_gate import PhantomEntityRejected

        import pytest

        with pytest.raises(PhantomEntityRejected):
            kg.add_triple("Alice", "knows", "Bob", statement="Alice knows Bob.")

    def test_add_triple_with_declared_endpoints(self, kg):
        """Happy path: both endpoints declared via add_entity, then
        add_triple writes the edge."""
        kg.add_entity("Alice", kind="entity", content="A person named Alice")
        kg.add_entity("Bob", kind="entity", content="A person named Bob")
        tid = kg.add_triple("Alice", "knows", "Bob", statement="Alice knows Bob.")
        assert tid.startswith("t_alice_knows_bob_")
        stats = kg.stats()
        assert stats["entities"] == 2

    def test_add_triple_with_dates(self, kg):
        kg.add_entity("Max", kind="entity", content="A person named Max")
        kg.add_entity("swimming", kind="entity", content="The sport of swimming")
        tid = kg.add_triple(
            "Max",
            "does",
            "swimming",
            valid_from="2025-01-01",
            statement="Max started doing swimming on 2025-01-01.",
        )
        assert tid.startswith("t_max_does_swimming_")

    def test_duplicate_triple_returns_existing_id(self, kg):
        kg.add_entity("Alice", kind="entity", content="A person named Alice")
        kg.add_entity("Bob", kind="entity", content="A person named Bob")
        tid1 = kg.add_triple("Alice", "knows", "Bob", statement="Alice knows Bob.")
        tid2 = kg.add_triple("Alice", "knows", "Bob", statement="Alice knows Bob.")
        assert tid1 == tid2

    def test_add_triple_normalizes_hyphenated_predicate(self, kg):
        """Predicate 'is-a' must be stored as 'is_a' so hyphen/underscore
        callers land on the same edge (regression: storage boundary was
        only collapsing spaces, hyphens survived as a separate predicate).

        Note: ``is_a`` is a skip-list predicate, so no `statement` is needed.
        """
        kg.add_entity("Alice", kind="entity", content="A person named Alice")
        kg.add_entity("person", kind="class", content="A human individual")
        kg.add_triple("Alice", "is-a", "person")
        edges = kg.query_entity("Alice", direction="outgoing")
        preds = [e["predicate"] for e in edges]
        assert "is_a" in preds
        assert "is-a" not in preds

    def test_invalidated_triple_allows_re_add(self, kg):
        kg.add_entity("Alice", kind="entity", content="A person named Alice")
        kg.add_entity("Acme", kind="entity", content="A company named Acme")
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
        # Add > 100 triples. Cold-start lock 2026-05-01: pre-declare
        # every endpoint since add_triple no longer auto-creates them.
        for i in range(106):
            kg.add_entity(f"entity_{i}", kind="entity", content=f"Test entity {i}")
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
        # Add > 100 triples all connected to a single entity. Cold-start
        # lock 2026-05-01: pre-declare hub + every spoke endpoint.
        kg.add_entity("hub", kind="entity", content="The hub entity")
        for i in range(105):
            kg.add_entity(f"spoke_{i}", kind="entity", content=f"Spoke {i}")
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
            "has_value re-added to skip list -- values would no longer "
            "embed for semantic search. See the comment above "
            "_TRIPLE_SKIP_PREDICATES in knowledge_graph.py."
        )

    def test_op_tier_rating_predicates_are_skip_listed(self):
        """performed_well/poorly/executed_op/superseded_by are pure
        graph topology. Their statement embedding adds noise without
        retrieval signal -- they're walked by retrieve_past_operations
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
                f"{pred!r} dropped from skip list -- finalize_intent's silent "
                "try/except around add_triple would now drop these edges."
            )


# ── Slice 1b 2026-04-28 ──────────────────────────────────────────────
# Render-time text fallback in query_entity. Honors Adrian's design lock
# 2026-04-25 (no auto-derivation at storage) -- the synthesis happens only
# when serving facts to a caller, never at write time.


class TestRenderFactDisplay:
    """_render_fact_display synthesis + query_entity 'text' field wiring."""

    def test_synthesizes_from_spo_when_statement_null(self):
        """Bare (s, p, o) tuple synthesizes a structural restatement
        with predicate underscores converted to spaces and a trailing
        period. This is structural, not auto-derived MEANING."""
        from mempalace.knowledge_graph import _render_fact_display

        out = _render_fact_display(
            {
                "subject": "Adrian",
                "predicate": "lives_in",
                "object": "Warsaw",
                "statement": None,
            }
        )
        assert out == "Adrian lives in Warsaw."

    def test_uses_statement_what_when_dict_authored(self):
        """Authored dict statement {what,why,scope?} surfaces 'what' as
        the display string -- that's the writer's natural-language
        verbalization."""
        from mempalace.knowledge_graph import _render_fact_display

        out = _render_fact_display(
            {
                "subject": "Adrian",
                "predicate": "lives_in",
                "object": "Warsaw",
                "statement": {
                    "what": "Adrian has lived in Warsaw since 2019",
                    "why": "primary residence; reflects current legal address",
                },
            }
        )
        assert out == "Adrian has lived in Warsaw since 2019"

    def test_uses_statement_what_when_json_string_authored(self):
        """Stored statements may arrive as JSON-encoded dict strings
        (the persisted form). Helper parses and surfaces 'what'."""
        import json

        from mempalace.knowledge_graph import _render_fact_display

        stmt_json = json.dumps({"what": "Server X is healthy", "why": "passing all probes"})
        out = _render_fact_display(
            {
                "subject": "ServerX",
                "predicate": "is_healthy",
                "object": "true",
                "statement": stmt_json,
            }
        )
        assert out == "Server X is healthy"

    def test_falls_back_to_raw_string_for_legacy_plain_statement(self):
        """Pre-dict-contract legacy edges may have raw-string statements.
        Helper passes them through as-is rather than synthesizing."""
        from mempalace.knowledge_graph import _render_fact_display

        out = _render_fact_display(
            {
                "subject": "X",
                "predicate": "rel",
                "object": "Y",
                "statement": "Some legacy plain-prose statement.",
            }
        )
        assert out == "Some legacy plain-prose statement."

    def test_query_entity_facts_carry_text_field(self, kg):
        """End-to-end: query_entity rows carry the synthesized 'text'
        field so kg_query callers always get a natural-language label
        without composing (s, p, o) themselves.

        Uses ``is_a`` (a skip-listed structural predicate, see
        ``_TRIPLE_SKIP_PREDICATES``) so the edge can be added without a
        caller-provided statement -- exercising exactly the synthetic
        fallback path the slice 1b helper is designed for. Uses the
        conftest ``kg`` fixture (isolated palace per test).
        """
        kg.add_entity("Adrian", kind="entity", content="A person")
        kg.add_entity("person", kind="class", content="Class for people")
        # is_a is in _TRIPLE_SKIP_PREDICATES so statement may be omitted.
        kg.add_triple("Adrian", "is_a", "person")
        facts = kg.query_entity("Adrian", direction="outgoing")
        assert facts, "expected at least one outgoing fact"
        is_a_edges = [f for f in facts if f["predicate"] == "is_a"]
        assert is_a_edges, "expected the is_a edge"
        assert "text" in is_a_edges[0], "query_entity rows must carry a 'text' field after slice 1b"
        # Synthetic fallback: predicate underscores become spaces, period appended.
        assert is_a_edges[0]["text"] == "Adrian is a person."

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
                f"{pred!r} dropped from skip list -- non-structural callers "
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

    def test_legacy_string_rejected_with_migration_message(self):
        """Adrian's design lock 2026-04-25: legacy strings on writes
        are rejected with a migration message pointing to the dict
        shape. Stored prose strings still serialize correctly through
        serialize_summary_for_embedding (legacy-read tolerance) but
        cannot be written through validate_summary."""
        from mempalace.knowledge_graph import (
            SummaryStructureRequired,
            validate_summary,
        )
        import pytest

        with pytest.raises(
            SummaryStructureRequired, match="legacy string form is no longer accepted"
        ):
            validate_summary("InjectionGate -- runtime gate that filters retrieved memories")

    def test_legacy_string_role_verb_also_rejected(self):
        """No regex on prose anywhere; even a structurally-good
        legacy string is rejected with the migration message."""
        from mempalace.knowledge_graph import (
            SummaryStructureRequired,
            validate_summary,
        )
        import pytest

        with pytest.raises(
            SummaryStructureRequired, match="legacy string form is no longer accepted"
        ):
            validate_summary("InjectionGate orchestrates the Haiku gate for retrieved memories")

    def test_dict_render_too_long_rejects(self):
        """When the rendered prose form exceeds 280 chars, validate
        rejects with the embedding-budget error."""
        from mempalace.knowledge_graph import (
            SummaryStructureRequired,
            validate_summary,
        )
        import pytest

        with pytest.raises(SummaryStructureRequired, match="rendered summary exceeds"):
            validate_summary(
                {
                    "what": "InjectionGate",
                    "why": ("filters retrieved memories before injection " * 20),
                }
            )

    def test_non_string_non_dict_rejects(self):
        from mempalace.knowledge_graph import (
            SummaryStructureRequired,
            validate_summary,
        )
        import pytest

        with pytest.raises(SummaryStructureRequired, match="must be a dict"):
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

    def test_dict_renders_with_ascii_double_hyphen_separator(self):
        """Adrian 2026-04-28 directive: no U+2014 anywhere. The renderer
        must use ' -- ' (ASCII double-hyphen) as the what/why separator
        so post-validation rendered prose is idempotent with the
        anyascii fold (fold_ascii('--') stays '--').
        """
        from mempalace.knowledge_graph import serialize_summary_for_embedding

        text = serialize_summary_for_embedding(
            {"what": "InjectionGate", "why": "filters retrieved memories"}
        )
        em_dash = chr(0x2014)
        en_dash = chr(0x2013)
        assert " -- " in text
        assert em_dash not in text  # NO em-dash
        assert en_dash not in text  # NO en-dash
        assert text.isascii(), f"renderer leaked non-ASCII: {text!r}"
        assert "InjectionGate" in text
        assert "filters retrieved memories" in text

    def test_renderer_output_is_ascii_for_ascii_input(self):
        """Property: ASCII-only summary fields produce ASCII-only prose.
        Regression for the renderer leak Adrian caught on the
        wrap_up_session record (knowledge_graph.py:309 was hardcoding
        U+2014 even when fold_summary had cleaned the input dict).
        """
        from mempalace.knowledge_graph import serialize_summary_for_embedding

        cases = [
            {"what": "x" * 5, "why": "y" * 15},
            {"what": "ascii thing", "why": "an ascii why clause", "scope": "everywhere"},
            {"what": "name", "why": "purpose role claim"},
        ]
        for s in cases:
            text = serialize_summary_for_embedding(s)
            assert text.isascii(), f"renderer leaked non-ASCII for {s!r}: {text!r}"

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

        text = serialize_summary_for_embedding("InjectionGate -- runtime gate")
        assert text == "InjectionGate -- runtime gate"

    def test_empty_dict_renders_empty_string(self):
        from mempalace.knowledge_graph import serialize_summary_for_embedding

        assert serialize_summary_for_embedding({}) == ""


class TestNoEmDashInSource:
    """Adrian's directive 2026-04-28: 'I don't want them AT ALL'.

    The em-dash purge replaced 1490 occurrences of U+2014 (and 13 of
    U+2013, en-dash) across mempalace/, tests/, and scripts/ .py files
    with their ASCII transliterations (' -- ' and ' - '). This test
    locks the purge against re-introduction by any future commit.

    If this test fails, run the purge inline (or scripts/purge_em_dashes_in_source.py
    if it exists) before committing.
    """

    def test_no_em_dash_in_any_py_source(self):
        from pathlib import Path

        em_dash = chr(0x2014)
        en_dash = chr(0x2013)
        repo_root = Path(__file__).resolve().parents[1]
        offenders: list[tuple[str, int, int]] = []
        for sub in ("mempalace", "tests", "scripts"):
            sub_root = repo_root / sub
            if not sub_root.is_dir():
                continue
            for py in sorted(sub_root.rglob("*.py")):
                try:
                    text = py.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                em = text.count(em_dash)
                en = text.count(en_dash)
                if em or en:
                    offenders.append((str(py.relative_to(repo_root)), em, en))
        assert not offenders, (
            "non-ASCII dashes found in source (Adrian's purge directive): "
            + ", ".join(f"{p} em={em} en={en}" for p, em, en in offenders)
        )

    def test_no_em_dash_in_any_text_artifact(self):
        """AT-ALL purge extends to docs + configs + shell hooks.

        Walks the repo for .md / .yaml / .yml / .toml / .json / .txt /
        .ini / .cfg / .sh / .cmd / .ps1 files (excluding vendored /
        cache / venv trees) and asserts none carry U+2014 or U+2013.
        """
        from pathlib import Path

        em_dash = chr(0x2014)
        en_dash = chr(0x2013)
        repo_root = Path(__file__).resolve().parents[1]
        text_exts = {
            ".yaml",
            ".yml",
            ".toml",
            ".md",
            ".json",
            ".txt",
            ".ini",
            ".cfg",
            ".sh",
            ".cmd",
            ".ps1",
        }
        skip_parts = {
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            "node_modules",
            ".pytest_cache",
            ".ruff_cache",
            "build",
            "dist",
            ".mypy_cache",
        }
        offenders: list[tuple[str, int, int]] = []
        for f in sorted(repo_root.rglob("*")):
            if not f.is_file():
                continue
            if any(part in skip_parts for part in f.parts):
                continue
            if f.suffix.lower() not in text_exts:
                continue
            try:
                text = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            em = text.count(em_dash)
            en = text.count(en_dash)
            if em or en:
                offenders.append((str(f.relative_to(repo_root)), em, en))
        assert not offenders, (
            "non-ASCII dashes found in text artifacts (Adrian's AT-ALL purge): "
            + ", ".join(f"{p} em={em} en={en}" for p, em, en in offenders)
        )


class TestUpdateEntityChromadbMetadata:
    """FINDING-1 (2026-04-28): kg_update_entity wrote importance to SQLite
    but kg_query read importance from Chroma multi-view metadata, which
    _sync_entity_to_chromadb did not patch (it orphan-wrote a single-view
    record under id=entity_id with no entity_id back-pointer instead).
    The fix adds _update_entity_chromadb_metadata, which iterates ALL
    Chroma records bound to the entity (multi-view via where + legacy
    single-id) and merges metadata field updates onto each.

    These tests pin the helper's contract against re-introduction.
    """

    def _make_fake_collection(self, multi_view_records=None, single_id_record=None):
        """Build a stub Chroma collection that mimics get/update.

        ``multi_view_records``: list of (id, metadata) for records carrying
        metadata['entity_id']. ``single_id_record``: optional (id, metadata)
        tuple representing a legacy single-id record (no entity_id back-pointer).
        """
        multi_view_records = multi_view_records or []
        captured_updates = []

        class FakeCol:
            def get(_self, where=None, ids=None, include=None):
                if where and "entity_id" in where:
                    target = where["entity_id"]
                    rows = [
                        (rid, meta)
                        for rid, meta in multi_view_records
                        if (meta or {}).get("entity_id") == target
                    ]
                    if not rows:
                        return {"ids": [], "metadatas": []}
                    return {
                        "ids": [r[0] for r in rows],
                        "metadatas": [r[1] for r in rows],
                    }
                if ids:
                    rows = []
                    for rid in ids:
                        if single_id_record and rid == single_id_record[0]:
                            rows.append(single_id_record)
                    if not rows:
                        return {"ids": [], "metadatas": []}
                    return {
                        "ids": [r[0] for r in rows],
                        "metadatas": [r[1] for r in rows],
                    }
                return {"ids": [], "metadatas": []}

            def update(_self, ids, metadatas):
                captured_updates.append((list(ids), [dict(m) for m in metadatas]))

        return FakeCol(), captured_updates

    def test_helper_patches_multi_view_records_metadata(self, monkeypatch):
        from mempalace import mcp_server

        col, captured = self._make_fake_collection(
            multi_view_records=[
                ("ent_xyz__v0", {"entity_id": "ent_xyz", "name": "Ent", "importance": 2}),
                ("ent_xyz__v1", {"entity_id": "ent_xyz", "name": "Ent", "importance": 2}),
            ],
        )
        monkeypatch.setattr(mcp_server, "_get_entity_collection", lambda create=False: col)

        mcp_server._update_entity_chromadb_metadata("ent_xyz", importance=5)

        assert len(captured) == 1, "exactly one col.update call expected"
        ids, metas = captured[0]
        assert sorted(ids) == ["ent_xyz__v0", "ent_xyz__v1"]
        for m in metas:
            assert m["importance"] == 5, f"new importance must be 5, got {m!r}"
            # Other keys preserved.
            assert m["name"] == "Ent"
            assert m["entity_id"] == "ent_xyz"

    def test_helper_also_patches_legacy_single_id_record(self, monkeypatch):
        from mempalace import mcp_server

        col, captured = self._make_fake_collection(
            multi_view_records=[],  # no multi-view records (legacy / pre-P5.2)
            single_id_record=("ent_legacy", {"name": "Legacy", "importance": 3}),
        )
        monkeypatch.setattr(mcp_server, "_get_entity_collection", lambda create=False: col)

        mcp_server._update_entity_chromadb_metadata("ent_legacy", importance=4)

        assert len(captured) == 1
        ids, metas = captured[0]
        assert ids == ["ent_legacy"]
        assert metas[0]["importance"] == 4
        assert metas[0]["name"] == "Legacy"

    def test_helper_patches_both_paths_when_present(self, monkeypatch):
        """Defensive: an entity with BOTH multi-view records and a stale
        single-id record (e.g. from a prior buggy update) should get
        both kinds patched in the same call.
        """
        from mempalace import mcp_server

        col, captured = self._make_fake_collection(
            multi_view_records=[
                ("ent_both__v0", {"entity_id": "ent_both", "importance": 2}),
            ],
            single_id_record=("ent_both", {"importance": 2, "name": "Both"}),
        )
        monkeypatch.setattr(mcp_server, "_get_entity_collection", lambda create=False: col)

        mcp_server._update_entity_chromadb_metadata("ent_both", importance=5)

        assert len(captured) == 1
        ids, metas = captured[0]
        assert sorted(ids) == sorted(["ent_both", "ent_both__v0"])
        for m in metas:
            assert m["importance"] == 5

    def test_helper_no_op_when_no_records(self, monkeypatch):
        from mempalace import mcp_server

        col, captured = self._make_fake_collection()  # empty
        monkeypatch.setattr(mcp_server, "_get_entity_collection", lambda create=False: col)

        mcp_server._update_entity_chromadb_metadata("ent_missing", importance=5)
        assert captured == []

    def test_helper_no_op_when_fields_empty(self, monkeypatch):
        from mempalace import mcp_server

        col, captured = self._make_fake_collection(
            multi_view_records=[
                ("ent_x__v0", {"entity_id": "ent_x", "importance": 2}),
            ],
        )
        monkeypatch.setattr(mcp_server, "_get_entity_collection", lambda create=False: col)

        # No fields to patch -> early-return, no col.update call.
        mcp_server._update_entity_chromadb_metadata("ent_x")
        assert captured == []

    def test_helper_silent_on_chroma_errors(self, monkeypatch):
        from mempalace import mcp_server

        class BrokenCol:
            def get(_self, *a, **kw):
                raise RuntimeError("chroma boom")

            def update(_self, *a, **kw):
                raise RuntimeError("chroma boom")

        monkeypatch.setattr(mcp_server, "_get_entity_collection", lambda create=False: BrokenCol())
        # Best-effort: must not raise even if Chroma is broken (SQLite write
        # already succeeded; transient Chroma failures shouldn't fail the
        # user-visible kg_update_entity).
        mcp_server._update_entity_chromadb_metadata("ent_z", importance=5)
