"""
test_triple_bfs_emission.py -- Lock in Channel B triple emission.

Background: before this change, graph-BFS (Channel B) emitted only the
visited entity/memory ids into the fused ranking. Triples themselves --
which carry their own ids in a separate namespace -- only ever appeared
in Channel A cosine over the mempalace_triples collection. That meant
triples never accumulated RRF cross-channel rank contributions the way
memories/entities do, making them structurally second-class in scoring.

The fix has three moving parts:
  1. knowledge_graph.query_entity now includes triple_id + statement
     in its returned edge dicts (additive keys; old callers unaffected).
  2. intent.py BFS (declare_intent path) emits the traversed triple
     into _channel_b_list alongside the neighbour entity/memory.
  3. scoring._build_graph_channel (kg_search path) does the same into
     graph_ranked.

Skip-list predicates (is_a, described_by, executed_by, targeted,
created_under, similar_to, session_note_for, surfaced, rated_useful,
rated_irrelevant, evidenced_by, derived_from, mentioned_in, has_value)
are excluded -- same filter _index_triple_statement uses at embed time
for the same reason: these statements are schema glue or feedback
topology, not retrieval-worthy text.

Scope: these tests cover query_entity's extended shape and the
_build_graph_channel emission. End-to-end Channel-B-in-declare_intent
tests are exercised by the broader test_intent_system integration
lane because the BFS is deep inside finalize.
"""

from __future__ import annotations

import chromadb
import pytest


# ─────────────────────────────────────────────────────────────────────
# query_entity: extended return shape
# ─────────────────────────────────────────────────────────────────────


class TestQueryEntityExtendedShape:
    def test_outgoing_edges_include_triple_id_and_statement(self, kg):
        kg.add_entity("alice", kind="entity", description="Alice")
        kg.add_entity("bob", kind="entity", description="Bob")
        tid = kg.add_triple("alice", "knows", "bob", statement="Alice knows Bob.")

        edges = kg.query_entity("alice", direction="outgoing")
        assert len(edges) == 1
        assert edges[0]["triple_id"] == tid
        assert edges[0]["statement"] == "Alice knows Bob."
        assert edges[0]["predicate"] == "knows"

    def test_incoming_edges_include_triple_id_and_statement(self, kg):
        kg.add_entity("alice", kind="entity", description="Alice")
        kg.add_entity("bob", kind="entity", description="Bob")
        tid = kg.add_triple("alice", "knows", "bob", statement="Alice knows Bob.")

        edges = kg.query_entity("bob", direction="incoming")
        assert len(edges) == 1
        assert edges[0]["triple_id"] == tid
        assert edges[0]["statement"] == "Alice knows Bob."

    def test_both_directions_carry_the_keys(self, kg):
        kg.add_entity("alice", kind="entity", description="Alice")
        kg.add_entity("bob", kind="entity", description="Bob")
        kg.add_triple("alice", "knows", "bob", statement="Alice knows Bob.")
        kg.add_triple("bob", "knows", "alice", statement="Bob knows Alice.")

        edges = kg.query_entity("alice", direction="both")
        assert all("triple_id" in e and "statement" in e for e in edges)

    def test_old_keys_still_present(self, kg):
        """Additive extension: existing keys unchanged."""
        kg.add_entity("alice", kind="entity", description="Alice")
        kg.add_entity("bob", kind="entity", description="Bob")
        kg.add_triple("alice", "knows", "bob", statement="x")

        edges = kg.query_entity("alice", direction="outgoing")
        e = edges[0]
        for key in (
            "direction",
            "subject",
            "predicate",
            "object",
            "valid_from",
            "valid_to",
            "confidence",
            "current",
        ):
            assert key in e


# ─────────────────────────────────────────────────────────────────────
# _build_graph_channel: triple emission
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def graph_setup(kg, palace_path):
    """Seed a tiny graph + a Chroma collection that holds the neighbour
    so _build_graph_channel can fetch its doc/meta. Returns (kg, col,
    seed_id, neighbour_id, triple_id, statement)."""
    kg.add_entity("alice", kind="entity", description="Alice")
    kg.add_entity("bob", kind="entity", description="Bob")
    tid = kg.add_triple("alice", "works_with", "bob", statement="Alice works with Bob at Acme.")

    client = chromadb.PersistentClient(path=palace_path)
    col = client.get_or_create_collection(
        "mempalace_records_bfs_test", metadata={"hnsw:space": "cosine"}
    )
    col.add(
        ids=["bob"],
        documents=["Bob entity doc"],
        metadatas=[{"kind": "entity"}],
    )
    yield kg, col, "alice", "bob", tid, "Alice works with Bob at Acme."
    try:
        client.delete_collection("mempalace_records_bfs_test")
    except Exception:
        pass


class TestBuildGraphChannelTripleEmission:
    def test_triple_emitted_alongside_neighbour(self, graph_setup):
        from mempalace.scoring import _build_graph_channel

        kg, col, seed, neighbour, tid, statement = graph_setup
        # Pre-populate seen_meta so the seed has a similarity (≥0.2 base
        # floor kicks in either way, but mirror the multi_channel path).
        seen_meta = {seed: {"meta": {}, "doc": "", "similarity": 0.5}}
        ranked = _build_graph_channel(col, kg, [seed], kind_filter=None, seen_meta=seen_meta)
        ids = [mid for _score, _doc, mid in ranked]
        assert neighbour in ids, "neighbour entity must still emit"
        assert tid in ids, "triple itself must emit so it gets RRF cross-channel boost"
        # Triple's entry has the statement as its doc.
        triple_entries = [(score, doc, mid) for score, doc, mid in ranked if mid == tid]
        assert len(triple_entries) == 1
        assert statement[:50] in triple_entries[0][1]

    def test_triple_meta_populated_in_seen_meta(self, graph_setup):
        from mempalace.scoring import _build_graph_channel

        kg, col, seed, neighbour, tid, statement = graph_setup
        seen_meta = {seed: {"meta": {}, "doc": "", "similarity": 0.5}}
        _build_graph_channel(col, kg, [seed], kind_filter=None, seen_meta=seen_meta)
        assert tid in seen_meta
        assert seen_meta[tid]["meta"].get("source") == "triple"
        assert seen_meta[tid]["meta"].get("predicate") == "works_with"

    def test_skip_list_predicate_not_emitted(self, kg, palace_path):
        """is_a edges are schema glue; triple should NOT be emitted."""
        from mempalace.scoring import _build_graph_channel

        kg.add_entity("max", kind="entity", description="Max")
        kg.add_entity("person", kind="class", description="Person class")
        # is_a is in _TRIPLE_SKIP_PREDICATES; statement optional per
        # TripleStatementRequired carve-out.
        tid = kg.add_triple("max", "is_a", "person")

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection(
            "mempalace_records_skip_test", metadata={"hnsw:space": "cosine"}
        )
        col.add(ids=["person"], documents=["Person class"], metadatas=[{"kind": "class"}])

        seen_meta = {"max": {"meta": {}, "doc": "", "similarity": 0.5}}
        ranked = _build_graph_channel(col, kg, ["max"], kind_filter=None, seen_meta=seen_meta)
        ids = [mid for _score, _doc, mid in ranked]
        assert tid not in ids, "is_a edge must not surface in Channel B"
        try:
            client.delete_collection("mempalace_records_skip_test")
        except Exception:
            pass

    def test_kind_filter_suppresses_triple_emission(self, graph_setup):
        """When a caller pins a kind filter, the pipeline is entity-only
        and triple emission is off by design."""
        from mempalace.scoring import _build_graph_channel

        kg, col, seed, neighbour, tid, _statement = graph_setup
        seen_meta = {seed: {"meta": {}, "doc": "", "similarity": 0.5}}
        ranked = _build_graph_channel(col, kg, [seed], kind_filter="entity", seen_meta=seen_meta)
        ids = [mid for _score, _doc, mid in ranked]
        # Triple ids are never in the entities table; when kind_filter
        # is set the graph channel must stay entity-scoped.
        assert tid not in ids

    def test_no_emission_without_statement(self, kg, palace_path):
        """A triple stored without a statement (skip-list-only today, but
        defensive) must not be emitted as a Channel B item."""
        from mempalace.scoring import _build_graph_channel

        kg.add_entity("x", kind="entity", description="X")
        kg.add_entity("y", kind="entity", description="Y")
        kg.add_triple("x", "described_by", "y")  # skip-list: no statement

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection(
            "mempalace_records_no_stmt", metadata={"hnsw:space": "cosine"}
        )
        col.add(ids=["y"], documents=["Y doc"], metadatas=[{"kind": "entity"}])

        seen_meta = {"x": {"meta": {}, "doc": "", "similarity": 0.5}}
        ranked = _build_graph_channel(col, kg, ["x"], kind_filter=None, seen_meta=seen_meta)
        # No triple id of form t_* should appear.
        t_ids = [mid for _s, _d, mid in ranked if str(mid).startswith("t_")]
        assert t_ids == []
        try:
            client.delete_collection("mempalace_records_no_stmt")
        except Exception:
            pass
