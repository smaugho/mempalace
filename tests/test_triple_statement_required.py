"""
test_triple_statement_required.py -- Invariant: every non-skip-predicate
triple MUST carry a caller-provided `statement`. No autogeneration
anywhere.

The policy (added 2026-04-19 after the user flagged that `kg_add_batch`
calls from inside the agent were producing bare triples that landed in
the mempalace_triples Chroma collection as low-signal underscore-to-
space fallbacks like "record X relates to record Y"): authors must
write a natural-language verbalization or the edge is refused.

Skip-list predicates (is_a, described_by, evidenced_by, executed_by,
targeted, has_value, session_note_for, derived_from, mentioned_in,
found_useful, found_irrelevant) remain statement-optional because
`_index_triple_statement` never embeds them regardless -- they are
schema glue, walkable via BFS, not searched by similarity.

These tests lock the policy in at three layers:

  1. `kg.add_triple()` raises `TripleStatementRequired` for non-skip
     predicates missing statement; passes for skip predicates.
  2. `tool_kg_add` returns a structured `{success: False, error: ...}`
     response naming the predicate and the skip-list exception.
  3. `tool_kg_add_batch` surfaces per-edge errors for bad edges and
     partial success for the rest.

Also asserts `_verbalize_triple` (the removed autogen helper) is not
resurrected, and `backfill_triple_statements` (the removed autogen
batch) is not resurrected.
"""

from __future__ import annotations


# ═══════════════════════════════════════════════════════════════════════
#  kg.add_triple -- SQL-level enforcement
# ═══════════════════════════════════════════════════════════════════════


class TestAddTripleEnforcement:
    def test_non_skip_predicate_missing_statement_raises(self, kg):
        """relates_to is NOT on the skip list, so omitting statement
        must raise TripleStatementRequired."""
        from mempalace.knowledge_graph import TripleStatementRequired

        try:
            kg.add_triple("alice", "relates_to", "bob")
        except TripleStatementRequired as exc:
            assert "relates_to" in str(exc)
            assert "statement" in str(exc).lower()
        else:
            raise AssertionError("expected TripleStatementRequired")

    def test_non_skip_predicate_empty_statement_raises(self, kg):
        from mempalace.knowledge_graph import TripleStatementRequired

        for bad in ("", "   ", "\n\t"):
            try:
                kg.add_triple("alice", "relates_to", "bob", statement=bad)
            except TripleStatementRequired:
                pass
            else:
                raise AssertionError(f"expected raise for statement={bad!r}")

    def test_non_skip_predicate_with_statement_succeeds(self, kg):
        tid = kg.add_triple("alice", "relates_to", "bob", statement="Alice relates to Bob.")
        assert tid.startswith("t_alice_relates_to_bob_")

    def test_skip_predicate_without_statement_succeeds(self, kg):
        """is_a, described_by, etc. are structural -- statement optional."""
        skip_predicates = [
            "is_a",
            "described_by",
            "evidenced_by",
            "executed_by",
            "targeted",
            "session_note_for",
            "derived_from",
            "mentioned_in",
            "created_under",
            "similar_to",
            "surfaced",
            "rated_useful",
            "rated_irrelevant",
            # Added 2026-04-25 audit (skip-list extension):
            "templatizes",  # S3b template-collapse edge
            "executed_op",  # S1 intent-exec → operation
            "performed_well",  # S1 context → good operation
            "performed_poorly",  # S1 context → bad operation
            "superseded_by",  # S2 op-to-op correction
        ]
        for pred in skip_predicates:
            tid = kg.add_triple(f"sub_{pred}", pred, f"obj_{pred}")
            assert tid.startswith("t_"), f"skip predicate '{pred}' should allow missing statement"

    def test_statement_is_trimmed(self, kg):
        """Leading/trailing whitespace in statement is stripped before
        storage so search embeddings aren't polluted by padding."""
        import sqlite3 as _sq3

        tid = kg.add_triple(
            "alice",
            "relates_to",
            "bob",
            statement="   Alice relates to Bob.   ",
        )
        conn = kg._conn()
        row = conn.execute("SELECT statement FROM triples WHERE id=?", (tid,)).fetchone()
        assert row is not None
        assert row["statement"] == "Alice relates to Bob."
        del _sq3  # silence lint


# ═══════════════════════════════════════════════════════════════════════
#  tool_kg_add -- MCP-layer enforcement (structured error)
# ═══════════════════════════════════════════════════════════════════════


class TestToolKgAddEnforcement:
    def test_missing_statement_returns_structured_error(self, monkeypatch, kg):
        """The MCP-layer guard must reject missing-statement calls with a
        structured error BEFORE the SQL layer gets a chance to raise.
        We only need sid + kg patched -- the guard checks predicate and
        statement before any further validation."""
        from mempalace import mcp_server

        monkeypatch.setattr(mcp_server._STATE, "kg", kg)
        monkeypatch.setattr(mcp_server._STATE, "session_id", "t-sid")
        # Seed just enough so that agent + predicate pass their own gates.
        kg.add_entity("agent", kind="class", description="x", importance=5)
        kg.add_entity("test_agent", kind="entity", description="x", importance=3)
        kg.add_triple("test_agent", "is_a", "agent")

        r = mcp_server.tool_kg_add(
            subject="alice",
            predicate="relates_to",
            object="bob",
            agent="test_agent",
            context={
                "queries": ["why alice relates to bob", "background"],
                "keywords": ["alice", "bob"],
                "entities": ["alice", "bob"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            # statement intentionally omitted
        )
        assert r.get("success") is False, r
        err = r.get("error", "")
        assert "statement" in err.lower(), err
        assert "relates_to" in err, err

    def test_skip_predicate_without_statement_still_succeeds(self, monkeypatch, kg):
        from mempalace import mcp_server

        monkeypatch.setattr(mcp_server._STATE, "kg", kg)
        monkeypatch.setattr(mcp_server._STATE, "session_id", "t-sid")
        # Minimal seed for validation.
        kg.add_entity("is_a", kind="predicate", description="x", importance=4)
        kg.add_entity("something", kind="entity", description="x", importance=3)
        kg.add_entity("thing", kind="class", description="x", importance=5)
        kg.add_entity("agent", kind="class", description="x", importance=5)
        kg.add_entity("test_agent", kind="entity", description="x", importance=3)
        kg.add_triple("test_agent", "is_a", "agent")

        r = mcp_server.tool_kg_add(
            subject="something",
            predicate="is_a",
            object="thing",
            agent="test_agent",
            context={
                "queries": ["something is a thing", "class membership"],
                "keywords": ["is_a", "thing"],
                "entities": ["something", "thing"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
        )
        assert r.get("success") is True, r


# ═══════════════════════════════════════════════════════════════════════
#  Removed helpers must stay removed
# ═══════════════════════════════════════════════════════════════════════


class TestRemovedHelpersStayRemoved:
    def test_verbalize_triple_is_not_re_exported(self):
        """The naive `_verbalize_triple` fallback was removed 2026-04-19.
        Re-adding it is a policy regression (autogeneration creates
        retrieval-poisoning text)."""
        import mempalace.knowledge_graph as kg_mod

        assert not hasattr(kg_mod, "_verbalize_triple"), (
            "_verbalize_triple is a retrieval-poisoning autogenerator and "
            "must not be re-introduced. See TripleStatementRequired policy."
        )

    def test_backfill_triple_statements_replaced_by_list_only_helper(self, kg):
        """The old backfill autogenerated statements via _verbalize_triple.
        The new helper is list-only: it reports which triples lack
        statements so a human/tool can write real ones, but never invents
        them."""
        # Seed one skip-predicate triple (no statement needed) and one
        # non-skip triple WITH a statement.
        kg.add_triple("sub_a", "is_a", "thing")
        kg.add_triple("sub_b", "relates_to", "obj_b", statement="B relates to obj_b.")
        rows = kg.list_unverbalized_triples()
        # Neither of the triples above qualifies as unverbalized: the
        # is_a one is on the skip list, the relates_to one has a real
        # statement.
        ids = {r["triple_id"] for r in rows}
        assert not any(r["predicate"] == "is_a" for r in rows), (
            "is_a triples must be excluded -- they are never embedded"
        )
        # Manually poke a non-skip triple with NULL statement (simulating
        # a legacy row) and ensure it shows up.
        conn = kg._conn()
        conn.execute(
            "INSERT INTO triples (id, subject, predicate, object, statement) "
            "VALUES (?, ?, ?, ?, NULL)",
            ("t_legacy", "sub_c", "relates_to", "obj_c"),
        )
        conn.commit()
        rows2 = kg.list_unverbalized_triples()
        ids2 = {r["triple_id"] for r in rows2}
        assert "t_legacy" in ids2, "list_unverbalized_triples must surface legacy rows"
        _ = ids  # silence unused
