"""TDD smoke test for state-protocol v3 slice 12 -- declare_operation gates.

Adrian directive 2026-05-05: ``declare_operation`` must enforce two
distinct gates that today only fire at finalize_intent time, letting
agents skip both for an entire intent and only get caught at the end.

Gate A -- slot-class enforcement:
  Per-tool ``operation_class`` entities (kind='class', is_a='operation',
  carrying ``properties.rules_profile.tool == <tool>`` and
  ``rules_profile.slots = {...}``) define the slot schema for that
  tool. ``declare_operation`` must validate the schema the same way
  ``declare_intent`` validates intent_type slots: required slots
  present, classes match, entities pre-declared, ``multiple=false``
  enforced. Tools without a registered class skip validation
  (back-compat).

Gate B -- operation-time state-delta enforcement:
  When retrieval surfaces a state-bearing INSTANCE (an entity whose
  is_a chain reaches a class with ``state_updatable=True``, e.g.
  Task / agent / intent_type instances), the agent MUST provide a
  ``state_deltas`` entry covering it on THIS op (changed/unchanged).
  Today the gate fires only at finalize_intent; slice 12 moves it to
  declare_operation so the agent cannot defer for a whole intent.

Test cases:
  Gate A (slots)
    1. Unclassified tool: bare context succeeds.
    2a. Required slot missing -> error.
    2b. Undeclared entity in slot -> error.
    2d. ``multiple=false`` violated -> error.
    2e. Valid slot persists on cue.

  Gate B (state-delta)
    3a. State-bearing instance surfaced and state_deltas missing -> BLOCK.
    3b. State-bearing instance surfaced and covered as 'unchanged' -> OK.
    3c. Non-state-bearing surface -> no requirement (back-compat).

Pre-fix the gate-B cases all PASS the call (no enforcement); post-fix
3a fails the call. All cases run against the real mempalace.intent
module via the singleton ``_STATE`` -- no module reload.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest


class _Slice12Fixture(unittest.TestCase):
    """Shared fixture wiring an isolated KG into the mempalace singletons."""

    def setUp(self):
        self.palace = tempfile.mkdtemp(prefix="mempalace_slice12_")

        # Use the singleton mempalace.mcp_server / intent modules. The
        # mcp_server module's top-level call ``intent.init(sys.modules[__name__])``
        # already wires intent._mcp at import time -- DO NOT reload.
        from mempalace import mcp_server as _mcp
        from mempalace import intent as _intent
        from mempalace.knowledge_graph import KnowledgeGraph
        from mempalace.seed import _ensure_operation_ontology

        # Fresh isolated KG.
        kg = KnowledgeGraph(db_path=os.path.join(self.palace, "kg.sqlite3"))
        # Pre-seed `thing` class -- _ensure_operation_ontology writes
        # an is_a edge from operation to thing and the cold-start gate
        # rejects the triple if thing isn't declared.
        kg.add_entity("thing", kind="class", content="Universal base class")
        # Seed the `operation` class + S1 predicates so add_triple
        # references resolve.
        _ensure_operation_ontology(kg)
        # Seed the `agent` + `file` classes. `agent` carries
        # state_updatable=True so its instances (ga_agent below) are
        # treated as state-bearing by the gate-B per-op enforcement
        # (mirrors the production seed). `file` is plain.
        kg.add_entity(
            "agent",
            kind="class",
            content="Agent class",
            properties={"state_updatable": True},
        )
        kg.add_entity("file", kind="class", content="File class")
        kg.add_entity(
            "ga_agent",
            kind="entity",
            content="ga_agent test fixture",
            properties={
                "summary": {
                    "what": "ga_agent test fixture",
                    "why": "agent identity used by declare_operation tests",
                }
            },
        )
        kg.add_triple("ga_agent", "is_a", "agent")

        # Replace the singleton's KG so intent.py's _mcp._STATE.kg
        # accesses point at our fresh DB. reset_transient (autouse)
        # clears active_intent + collection cache between tests.
        _mcp._STATE.kg = kg
        _mcp._STATE.declared_entities.add("ga_agent")
        _mcp._STATE.declared_entities.add("agent")
        _mcp._STATE.declared_entities.add("file")
        _mcp._STATE.declared_entities.add("operation")
        # Pre-mint a deterministic context entity for the active op.
        # Gate B reads active_context_id; without a known id the test
        # can't pre-thread state_deltas covering it. We pin
        # context_lookup_or_create to return this id so every op in
        # the test points at ctx_test_op.
        kg.add_entity(
            "ctx_test_op",
            kind="context",
            content="test op context for slice 12 smoke",
            properties={
                "summary": {
                    "what": "ctx_test_op deterministic test context",
                    "why": "pinned context id so gate B coverage in tests is deterministic",
                }
            },
        )
        _mcp._STATE.declared_entities.add("ctx_test_op")

        # Stub session id + active intent so declare_operation has
        # something to attach to.
        _mcp._STATE.session_id = "sid_test_slice12"
        _mcp._STATE.active_intent = {
            "intent_id": "intent_test_slice12",
            "intent_type": "edit_mempalace",
            "agent": "ga_agent",
            "session_id": "sid_test_slice12",
            "accessed_memory_ids": [],
            "injected_memory_ids": [],
            "pending_operation_cues": [],
            "active_context_id": "ctx_test_op",
            # _persist_active_intent reads these keys; provide empty
            # defaults so a minimal fixture doesn't KeyError.
            "slots": {},
            "intent_type_id": "edit_mempalace",
            "permissions": [],
            "budget": {},
            "budget_used": {},
            "raw_paths": {},
        }

        # Monkey-patch context_lookup_or_create so every declare_operation
        # in this test routes through the same deterministic ctx id.
        self._orig_ctx_lookup = _mcp.context_lookup_or_create

        def _fixed_ctx(queries, keywords, entities, agent, summary):
            return ("ctx_test_op", True, [])

        _mcp.context_lookup_or_create = _fixed_ctx

        # Monkey-patch _persist_active_intent to no-op. Persistence
        # reads many active_intent keys (effective_permissions, etc.)
        # that this minimal fixture doesn't carry; persistence is not
        # under test in slice 12 -- the gate-A and gate-B return paths
        # are. The original is restored in tearDown.
        self._orig_persist = _intent._persist_active_intent
        _intent._persist_active_intent = lambda: None

        self._mcp = _mcp
        self._intent = _intent
        self.kg = kg
        self.agent = "ga_agent"
        # State_deltas the gate-B implicit-active-set will demand on
        # every successful op in this fixture. Tests append to / merge
        # with this baseline as needed.
        self._baseline_deltas = [
            {
                "entity_id": "ga_agent",
                "status": "unchanged",
                "justification": "test fixture: agent state unchanged",
            },
            {
                "entity_id": "ctx_test_op",
                "status": "unchanged",
                "justification": "test fixture: op context unchanged",
            },
        ]

    def tearDown(self):
        try:
            self._mcp.context_lookup_or_create = self._orig_ctx_lookup
        except Exception:
            pass
        try:
            self._intent._persist_active_intent = self._orig_persist
        except Exception:
            pass
        try:
            self._mcp._STATE.active_intent = None
            self._mcp._STATE.declared_entities.clear()
        except Exception:
            pass
        shutil.rmtree(self.palace, ignore_errors=True)

    def _ctx(self, what: str = "test op", why: str = "exercising slot validation"):
        return {
            "queries": ["test operation cue", "slice 12 slot validation probe"],
            "keywords": ["test", "slice-12"],
            "entities": ["ga_agent"],
            "summary": {"what": what, "why": why},
        }

    def _declare_op_class(self, name: str, tool: str, slots: dict) -> None:
        self.kg.add_entity(
            name,
            kind="class",
            content=f"{tool} operation class for slice 12 tests",
            importance=3,
            properties={
                "summary": {
                    "what": f"{tool} operation class with slot schema",
                    "why": "slice 12 test fixture binding tool to slot definitions",
                },
                "rules_profile": {
                    "tool": tool,
                    "slots": slots,
                },
            },
        )
        self.kg.add_triple(name, "is_a", "operation")
        self._mcp._STATE.declared_entities.add(name)


class TestGateA_OperationSlots(_Slice12Fixture):
    """Gate A: slot-class enforcement at declare_operation time."""

    def test_unclassified_tool_accepts_no_slots(self):
        result = self._intent.tool_declare_operation(
            tool="Read",
            args_summary="Read {test_path}",
            context=self._ctx(),
            agent=self.agent,
            state_deltas=self._baseline_deltas,
        )
        self.assertTrue(
            result.get("success"),
            f"unclassified tool should succeed without slots; got {result}",
        )

    def test_classified_tool_missing_required_slot_fails(self):
        self._declare_op_class(
            "read_operation",
            tool="Read",
            slots={
                "target_file": {
                    "classes": ["file"],
                    "required": True,
                    "multiple": False,
                }
            },
        )
        result = self._intent.tool_declare_operation(
            tool="Read",
            args_summary="Read {test_path}",
            context=self._ctx(),
            agent=self.agent,
        )
        self.assertFalse(result.get("success"), f"expected failure, got {result}")
        err = (result.get("error") or "") + str(result.get("slot_issues") or "")
        self.assertIn("target_file", err)

    def test_classified_tool_undeclared_entity_fails(self):
        self._declare_op_class(
            "read_operation",
            tool="Read",
            slots={
                "target_file": {
                    "classes": ["file"],
                    "required": True,
                    "multiple": False,
                }
            },
        )
        result = self._intent.tool_declare_operation(
            tool="Read",
            args_summary="Read {test_path}",
            context=self._ctx(),
            agent=self.agent,
            slots={"target_file": ["definitely_not_declared_py"]},
        )
        self.assertFalse(result.get("success"), f"expected failure, got {result}")
        err = (result.get("error") or "") + str(result.get("slot_issues") or "")
        self.assertTrue(
            "not declared" in err.lower() or "kg_declare_entity" in err.lower(),
            f"expected 'declare first' style error; got {err}",
        )

    def test_classified_tool_multiple_false_rejected(self):
        self._declare_op_class(
            "read_operation",
            tool="Read",
            slots={
                "target_file": {
                    "classes": ["file"],
                    "required": True,
                    "multiple": False,
                }
            },
        )
        for fname in ("alpha_py", "beta_py"):
            self.kg.add_entity(
                fname,
                kind="entity",
                content=f"file fixture {fname}",
                properties={
                    "file_path": f"{fname}.py",
                    "summary": {
                        "what": f"{fname} file fixture",
                        "why": "slice 12 multiple=false test fixture",
                    },
                },
            )
            self.kg.add_triple(fname, "is_a", "file")
            self._mcp._STATE.declared_entities.add(fname)

        result = self._intent.tool_declare_operation(
            tool="Read",
            args_summary="Read {test_path}",
            context=self._ctx(),
            agent=self.agent,
            slots={"target_file": ["alpha_py", "beta_py"]},
        )
        self.assertFalse(result.get("success"), f"expected failure, got {result}")
        err = (result.get("error") or "") + str(result.get("slot_issues") or "")
        self.assertIn("multiple", err.lower())

    def test_classified_tool_valid_slot_persists_on_cue(self):
        self._declare_op_class(
            "read_operation",
            tool="Read",
            slots={
                "target_file": {
                    "classes": ["file"],
                    "required": True,
                    "multiple": False,
                }
            },
        )
        self.kg.add_entity(
            "good_target_py",
            kind="entity",
            content="valid file fixture",
            properties={
                "file_path": "good_target.py",
                "summary": {
                    "what": "good_target file fixture",
                    "why": "slice 12 valid-slot test fixture",
                },
            },
        )
        self.kg.add_triple("good_target_py", "is_a", "file")
        self._mcp._STATE.declared_entities.add("good_target_py")

        result = self._intent.tool_declare_operation(
            tool="Read",
            args_summary="Read {test_path}",
            context=self._ctx(),
            agent=self.agent,
            slots={"target_file": ["good_target_py"]},
            state_deltas=self._baseline_deltas,
        )
        self.assertTrue(result.get("success"), f"expected success, got {result}")

        cues = self._mcp._STATE.active_intent.get("pending_operation_cues") or []
        self.assertGreaterEqual(len(cues), 1, "expected at least one pending cue")
        last = cues[-1]
        self.assertEqual(last.get("tool"), "Read")
        resolved = last.get("resolved_slots") or {}
        self.assertIn(
            "target_file",
            resolved,
            f"resolved_slots missing target_file: {last}",
        )
        self.assertIn("good_target_py", resolved["target_file"])


class TestGateB_StateDeltaAtOpTime(_Slice12Fixture):
    """Gate B: state-delta enforcement at declare_operation time.

    The fixture overrides _run_local_retrieval so we get deterministic
    surfaces without standing up Chroma -- the gate-B logic must fire
    on whatever the retrieval pipeline returns, regardless of source.
    """

    def setUp(self):
        super().setUp()
        # Mark the `agent` class as state-bearing for this test (its
        # instances must commit to state on any op that surfaces them).
        # Mirrors the production seed where Task / agent / intent_type
        # carry state_updatable=True via state_schema_id.
        self.kg.add_entity(
            "task",
            kind="class",
            content="Task class for state delta tests",
            properties={
                "summary": {
                    "what": "task class -- state-bearing for slice 12 gate B",
                    "why": "instances must commit to changed/unchanged on every op that surfaces them",
                },
                "state_schema_id": "task_state",
                # Gate-B SQL filter looks for state_updatable=True;
                # without this the task class won't be picked up and
                # task_alpha won't surface as state-bearing.
                "state_updatable": True,
            },
        )
        # Declare a Task instance that retrieval will return.
        self.kg.add_entity(
            "task_alpha",
            kind="entity",
            content="task_alpha is a state-bearing instance",
            properties={
                "summary": {
                    "what": "task_alpha state-bearing instance",
                    "why": "gate B fixture surfaces this task to assert enforcement",
                },
            },
        )
        self.kg.add_triple("task_alpha", "is_a", "task")
        self._mcp._STATE.declared_entities.add("task")
        self._mcp._STATE.declared_entities.add("task_alpha")

        # Patch _run_local_retrieval to return a deterministic hit.
        from mempalace import hooks_cli as _hc

        self._orig_retrieval = _hc._run_local_retrieval

        def _fake_retrieval(cue, accessed, top_k):
            return (
                [{"id": "task_alpha", "summary_text": "task_alpha state-bearing instance"}],
                None,
            )

        _hc._run_local_retrieval = _fake_retrieval
        self._hc = _hc

    def tearDown(self):
        try:
            self._hc._run_local_retrieval = self._orig_retrieval
        except Exception:
            pass
        super().tearDown()

    def test_state_bearing_surface_without_deltas_blocks(self):
        """Gate B (post-fix): state-bearing entity surfaced + no
        state_deltas -> declare_operation must BLOCK."""
        result = self._intent.tool_declare_operation(
            tool="Read",
            args_summary="Read {test_path}",
            context=self._ctx(),
            agent=self.agent,
            # state_deltas omitted on purpose
        )
        self.assertFalse(
            result.get("success"),
            f"expected gate B to block; got success: {result}",
        )
        err = (result.get("error") or "") + str(result.get("missing_state_deltas") or "")
        self.assertIn("task_alpha", err)

    def test_state_bearing_surface_covered_unchanged_passes(self):
        """Gate B (post-fix): state_deltas covering the surfaced
        instance + implicit-active-set lets the call through."""
        result = self._intent.tool_declare_operation(
            tool="Read",
            args_summary="Read {test_path}",
            context=self._ctx(),
            agent=self.agent,
            state_deltas=self._baseline_deltas
            + [
                {
                    "entity_id": "task_alpha",
                    "status": "unchanged",
                    "justification": "gate B test: explicit no-op acknowledgement",
                }
            ],
        )
        self.assertTrue(
            result.get("success"),
            f"expected gate B to pass with covered delta; got {result}",
        )


if __name__ == "__main__":
    unittest.main()
