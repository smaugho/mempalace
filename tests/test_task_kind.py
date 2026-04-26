"""
test_task_kind.py — Slice A unit tests for the Task ontology seeder.

Slice A adds:
  * Task class entity (kind='class', is_a thing) — domain class for
    external work items that cause activity-intents.
  * has_status predicate (entity → literal) — current task state.
  * external_ref predicate (entity → literal) — opaque external id.
  * Status literals (open / in_progress / done / canceled).

This is the standalone foundational slice of the user-intent tier
design. After ship: paperclip and any non-interactive agent can declare
a task entity (kind='entity', is_a Task) with a status edge and use it
as the cause_id for activity-intents in Slice B. Zero dependency on the
rest of the user-intent flow.

Tests focus on the seeder side-effects on a cold tmp palace — same
shape as test_op_memory.py::TestTemplatizesPredicateRegistered. We do
not require the live MCP startup hook here; we call the seeder
function directly so the test is hermetic.
"""

from __future__ import annotations

import json

import pytest


def _bootstrap_kg(tmp_path):
    """Build a fresh KnowledgeGraph + run the Task ontology seeder.

    Mirrors the bootstrap pattern in test_op_memory.py — direct seeder
    invocation against a tmp_path palace so we exercise the real
    add_entity / add_triple paths, not mocks.
    """
    from mempalace.knowledge_graph import KnowledgeGraph
    from mempalace.mcp_server import _ensure_task_ontology

    db = tmp_path / "palace.db"
    kg = KnowledgeGraph(str(db))
    _ensure_task_ontology(kg)
    return kg


# ─────────────────────────────────────────────────────────────────────
# Class seed
# ─────────────────────────────────────────────────────────────────────


class TestTaskClassSeeded:
    """The seeder mints a Task class entity that other agents can
    reference via is_a edges."""

    def test_task_class_exists(self, tmp_path):
        kg = _bootstrap_kg(tmp_path)
        ent = kg.get_entity("Task")
        assert ent is not None, "Task class entity must be seeded"
        assert ent.get("kind") == "class"

    def test_task_class_is_a_thing(self, tmp_path):
        """Every class in mempalace's ontology is_a thing — gives
        retrieval a uniform top-level anchor for class entities."""
        kg = _bootstrap_kg(tmp_path)
        edges = kg.query_entity("Task", direction="outgoing")
        is_a_edges = [e for e in edges if e.get("predicate") == "is_a" and e.get("current", True)]
        targets = {e.get("object") for e in is_a_edges}
        assert "thing" in targets, (
            f"Task must have is_a thing edge for ontology consistency; got {targets}"
        )


# ─────────────────────────────────────────────────────────────────────
# Predicate seeds — has_status + external_ref
# ─────────────────────────────────────────────────────────────────────


class TestTaskPredicatesSeeded:
    """Both task-tier predicates must be declared before agents can
    write task→literal edges via kg_add. Constraints lock the
    subject/object kinds at the schema layer."""

    def test_has_status_predicate_seeded(self, tmp_path):
        kg = _bootstrap_kg(tmp_path)
        ent = kg.get_entity("has_status")
        assert ent is not None
        assert ent.get("kind") == "predicate"

        props = ent.get("properties") or {}
        if isinstance(props, str):
            props = json.loads(props)
        constraints = props.get("constraints") or {}
        assert "entity" in (constraints.get("subject_kinds") or [])
        assert "literal" in (constraints.get("object_kinds") or [])
        assert "Task" in (constraints.get("subject_classes") or [])
        assert constraints.get("cardinality") == "one-to-one"

    def test_external_ref_predicate_seeded(self, tmp_path):
        kg = _bootstrap_kg(tmp_path)
        ent = kg.get_entity("external_ref")
        assert ent is not None
        assert ent.get("kind") == "predicate"

        props = ent.get("properties") or {}
        if isinstance(props, str):
            props = json.loads(props)
        constraints = props.get("constraints") or {}
        assert "entity" in (constraints.get("subject_kinds") or [])
        assert "literal" in (constraints.get("object_kinds") or [])
        assert "Task" in (constraints.get("subject_classes") or [])


# ─────────────────────────────────────────────────────────────────────
# Status literals
# ─────────────────────────────────────────────────────────────────────


class TestTaskStatusLiterals:
    """Status literals must be declared as kind='literal' nodes so
    has_status edges target real ontology members and queries like
    'all currently-open tasks' resolve cleanly."""

    @pytest.mark.parametrize("status", ["open", "in_progress", "done", "canceled"])
    def test_status_literal_seeded(self, tmp_path, status):
        kg = _bootstrap_kg(tmp_path)
        ent = kg.get_entity(status)
        assert ent is not None, f"status literal {status!r} must be seeded"
        assert ent.get("kind") == "literal"

    def test_no_extra_status_literals(self, tmp_path):
        """Lock the canonical set so an inadvertent status addition
        forces a deliberate seeder + test update."""
        kg = _bootstrap_kg(tmp_path)
        for status in ("open", "in_progress", "done", "canceled"):
            assert kg.get_entity(status) is not None


# ─────────────────────────────────────────────────────────────────────
# End-to-end: declare a task entity + write status edge
# ─────────────────────────────────────────────────────────────────────


class TestTaskEndToEnd:
    """After the seeder runs, an agent must be able to:
      1. Declare a task as kind='entity' + is_a Task.
      2. Write a has_status edge to a status literal.
      3. Write an external_ref edge to a fresh literal.
    No predicate-validation errors, no missing-target errors. This is
    the contract Slice B's cause_id resolver will rely on."""

    def test_task_entity_with_status_and_external_ref(self, tmp_path):
        kg = _bootstrap_kg(tmp_path)

        # Caller declares the literal and the agent first — same as a
        # paperclip agent would on its first run. Agent-attribution
        # rides via an explicit is_a edge, not an add_entity kwarg
        # (add_entity has no added_by parameter — attribution is in
        # the graph).
        kg.add_entity(
            "paperclip_agent",
            kind="entity",
            description="paperclip subagent — automation runner",
            importance=3,
        )
        kg.add_triple("paperclip_agent", "is_a", "agent")

        kg.add_entity(
            "TASK-fix-auth-bug",
            kind="entity",
            description="Fix the JWT-auth 401 bug surfacing on /login.",
            importance=4,
        )
        kg.add_triple("TASK-fix-auth-bug", "is_a", "Task")

        # Status edge against the seeded literal. has_status is not in
        # the structural skip-list (is_a, described_by, etc.), so a
        # natural-language statement is required (2026-04-19 design
        # lock — autogeneration was retired because naive fallbacks
        # poisoned retrieval).
        kg.add_triple(
            "TASK-fix-auth-bug",
            "has_status",
            "open",
            statement="TASK-fix-auth-bug is currently open and ready for an agent to start work on",
        )

        # External ref against a freshly declared literal (agents declare
        # their own opaque ids; we don't pre-seed those).
        kg.add_entity(
            "PAPERCLIP-1234",
            kind="literal",
            description="Paperclip issue PAPERCLIP-1234",
            importance=2,
        )
        kg.add_triple(
            "TASK-fix-auth-bug",
            "external_ref",
            "PAPERCLIP-1234",
            statement="TASK-fix-auth-bug round-trips with the Paperclip issue PAPERCLIP-1234 in the source tracker",
        )

        # Verify all three edges landed and are current.
        outgoing = kg.query_entity("TASK-fix-auth-bug", direction="outgoing")
        live = {(e["predicate"], e["object"]) for e in outgoing if e.get("current", True)}
        assert ("is_a", "Task") in live
        assert ("has_status", "open") in live
        assert ("external_ref", "PAPERCLIP-1234") in live
