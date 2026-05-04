"""Ontology seeders -- single home for class/predicate/literal bootstrap.

Adrian's directive 2026-04-26: seeding logic must live in its own
script, not buried inside mcp_server.py. Three seeders live here, each
idempotent and safe to re-run:

  * ``_ensure_operation_ontology(kg)``  -- S1 op-memory tier
  * ``_ensure_task_ontology(kg)``       -- Slice-A task tier
  * ``_ensure_user_intent_ontology(kg)``-- Slice-B user-intent tier

``seed_all(kg)`` runs every seeder, swallowing+logging individual
failures so a broken seeder cannot wedge startup. ``mcp_server.py``
calls ``seed_all`` at module import (gated on
``MEMPALACE_SKIP_SEED``); tests import individual seeders directly.

Why a separate module:
  * Gives every ontology one canonical location instead of being
    smeared across mcp_server.
  * Lets the gardener / migrations / fresh-palace boot scripts call the
    same seeders without importing the MCP transport layer.
  * Makes audits ("are all op predicates seeded?") a single-file read.

The three seeders rely on KnowledgeGraph's ``ON CONFLICT DO UPDATE``
upserts in ``add_entity`` / ``add_triple``: re-running is a no-op on
populated palaces and a one-shot fill on fresh ones.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


# ==================== S1 OPERATION ONTOLOGY ====================
# Op-memory tier (Leontiev 1981 Activity Theory -- Operation level;
# cf. arXiv 2512.18950 Learning Hierarchical Procedural Memory). Seeds
# one class (`operation`) plus four predicates unconditionally on every
# startup. add_entity / add_triple both upsert via ON CONFLICT DO UPDATE,
# so re-running is a no-op on existing palaces and a one-shot fill on
# fresh ones. Separating from KnowledgeGraph.seed_ontology keeps
# knowledge_graph.py free of S1-specific code.


def _ensure_operation_ontology(kg) -> None:
    """Idempotently seed the `operation` class + S1 predicates.

    Predicates:
      * executed_op -- intent_exec → op (parent/child; statement required)
      * performed_well -- context → op (agent-rated quality ≥4)
      * performed_poorly -- context → op (agent-rated quality ≤2)
      * superseded_by -- op → op correction edge (S2)
      * templatizes -- record → op template-collapse edge (S3b)

    The `operation` class is a subclass of `thing`. Operations are
    kind='operation' entities (see VALID_KINDS); they are NEVER embedded
    into Chroma collections -- retrieval reaches them only via graph
    traversal from their context's performed_well / performed_poorly
    edges.

    Cold-start lock 2026-05-01: hand-curated inline {what, why, scope}
    summaries per entry. No helper, no template -- the seed list IS
    the curated source of truth, so the summaries live where the
    callers can audit them in one read.
    """
    _op_class_desc = (
        "A recorded tool invocation (tool + truncated args + context_id). "
        "Graph-only -- never embedded. Attached to an intent execution via "
        "executed_op, and to its operation-context via performed_well / "
        "performed_poorly. Cf. Leontiev 1981 Operation tier; arXiv "
        "2512.18950 hierarchical procedural memory."
    )
    kg.add_entity(
        "operation",
        kind="class",
        content=_op_class_desc,
        importance=4,
        properties={
            "summary": {
                "what": "operation class -- recorded tool invocations",
                "why": "graph-only entities (tool + args + context_id) attached via executed_op + performed_well/poorly; Leontiev 1981 / arXiv 2512.18950 op-tier",
                "scope": "graph-only; never embedded; one row per parametrized fingerprint",
            },
        },
    )
    kg.add_triple("operation", "is_a", "thing")

    # Each entry: (name, what, why, importance, constraints)
    # `what`  : hand-authored identity phrase
    # `why`   : curated description used as the canonical why (must fit <=160 chars)
    # `scope` : derived inline from cardinality + subject/object kinds
    _op_pred_defs = [
        (
            "executed_op",
            "executed_op -- intent-execution to operation predicate",
            "Parent-child edge from an intent execution to an operation entity it performed; written by finalize_intent on rated trace promotion",
            4,
            {
                "subject_kinds": ["entity"],
                "object_kinds": ["operation"],
                "subject_classes": ["intent_type", "thing"],
                "object_classes": ["operation", "thing"],
                "cardinality": "one-to-many",
            },
        ),
        (
            "performed_well",
            "performed_well -- positive op-quality predicate",
            "Edge: in the given operation-context the agent rated this op good (quality>=4); read at declare_operation to surface precedent patterns",
            4,
            {
                "subject_kinds": ["context"],
                "object_kinds": ["operation"],
                "subject_classes": ["thing"],
                "object_classes": ["thing"],
                "cardinality": "many-to-many",
            },
        ),
        (
            "performed_poorly",
            "performed_poorly -- negative op-quality predicate",
            "Edge: in the given operation-context the agent rated this op poor (quality<=2); surfaces cautionary precedent alongside performed_well",
            3,
            {
                "subject_kinds": ["context"],
                "object_kinds": ["operation"],
                "subject_classes": ["thing"],
                "object_classes": ["thing"],
                "cardinality": "many-to-many",
            },
        ),
        (
            "superseded_by",
            "superseded_by -- op-to-op correction predicate",
            "S2 correction edge: a poorly-rated op points to the op that would have been the correct move in the same context (better_alternative)",
            4,
            {
                "subject_kinds": ["operation"],
                "object_kinds": ["operation"],
                "subject_classes": ["operation", "thing"],
                "object_classes": ["operation", "thing"],
                "cardinality": "many-to-one",
            },
        ),
        (
            "templatizes",
            "templatizes -- template-collapse provenance predicate",
            "S3b edge: a template record points at each op it distilled; written by memory_gardener.synthesize_operation_template on cluster resolution",
            4,
            {
                "subject_kinds": ["record"],
                "object_kinds": ["operation"],
                "subject_classes": ["record", "thing"],
                "object_classes": ["operation", "thing"],
                "cardinality": "one-to-many",
            },
        ),
    ]
    for name, what, why, imp, constraints in _op_pred_defs:
        # Cold-start lock 2026-05-01: hand-curated {what, why, scope}
        # built inline from the per-entry fields. No helper, no derivation.
        cardinality = constraints.get("cardinality", "?")
        subj = ",".join(constraints.get("subject_kinds") or []) or "any"
        obj = ",".join(constraints.get("object_kinds") or []) or "any"
        scope = f"{cardinality}; subj={subj}; obj={obj}"[:100]
        summary = {"what": what, "why": why, "scope": scope}
        kg.add_entity(
            name,
            kind="predicate",
            content=why,
            importance=imp,
            properties={
                "constraints": constraints,
                "summary": summary,
            },
        )


def _ensure_task_ontology(kg) -> None:
    """Idempotently seed the `Task` class + Slice-A task predicates.

    Tasks are kind='entity' nodes with an is_a Task edge -- the canonical
    "domain types are classes not kinds" pattern (protocol design lock).
    They serve as the parallel parent-cause path for activity intents
    declared by non-interactive agents (paperclip, scheduled runs) where
    no user_message exists. Slice B's `declare_intent.cause_id` accepts
    either a user-context (kind='context' with fulfills_user_message
    edges) or an entity with is_a Task -- this seeder makes the latter
    half resolvable.

    Predicates seeded:
      * has_status   -- task entity → status literal (current state).
      * external_ref -- task entity → external_ref literal (Paperclip /
                       Flowsev / GitHub issue id, opaque string).

    Status literals seeded as `kind='literal'` entities so `has_status`
    edges land on declared targets and queries like "all open tasks"
    resolve via a single predicate scan.

    Cold-start lock 2026-05-01: hand-curated inline summaries.
    """
    _task_class_desc = (
        "An external work item that causes activity-intents in mempalace. "
        "Tasks are kind='entity' nodes with an is_a Task edge -- they hold "
        "a description, an optional external_ref (issue tracker key), and "
        "a has_status edge to a status literal. Used as cause_id by "
        "non-interactive agents (paperclip, scheduled runs) where no "
        "user_message is available."
    )
    kg.add_entity(
        "Task",
        kind="class",
        content=_task_class_desc,
        importance=4,
        properties={
            "summary": {
                "what": "Task class -- external work items",
                "why": "kind='entity' nodes with is_a Task edge; carry has_status + external_ref edges; used as cause_id by non-interactive agents (paperclip, schedules)",
                "scope": "task tier; one row per externally-tracked work item",
            },
            # State-protocol v1 (Adrian Option B 2026-05-03): Task instances
            # are state-bearing; their slot payload is validated against the
            # task_state schema in state_schemas.STATE_SCHEMAS at delta time.
            "state_updatable": True,
            "state_schema_id": "task_state",
        },
    )
    kg.add_triple("Task", "is_a", "thing")

    # Each entry: (name, what, why, importance, constraints)
    _task_pred_defs = [
        (
            "has_status",
            "has_status -- task-to-status-literal predicate",
            "Edge from a task entity to its current status literal (open, in_progress, done, canceled); versioned via valid_from/valid_to",
            4,
            {
                "subject_kinds": ["entity"],
                "object_kinds": ["literal"],
                "subject_classes": ["Task", "thing"],
                "object_classes": ["thing"],
                "cardinality": "one-to-one",
            },
        ),
        (
            "external_ref",
            "external_ref -- task-to-external-id predicate",
            "Edge from a task entity to an opaque external identifier literal (Paperclip / Flowsev / GitHub issue key) for round-trip integration",
            3,
            {
                "subject_kinds": ["entity"],
                "object_kinds": ["literal"],
                "subject_classes": ["Task", "thing"],
                "object_classes": ["thing"],
                "cardinality": "one-to-one",
            },
        ),
    ]
    for name, what, why, imp, constraints in _task_pred_defs:
        cardinality = constraints.get("cardinality", "?")
        subj = ",".join(constraints.get("subject_kinds") or []) or "any"
        obj = ",".join(constraints.get("object_kinds") or []) or "any"
        scope = f"{cardinality}; subj={subj}; obj={obj}"[:100]
        kg.add_entity(
            name,
            kind="predicate",
            content=why,
            importance=imp,
            properties={
                "constraints": constraints,
                "summary": {"what": what, "why": why, "scope": scope},
            },
        )

    # Status literals -- declared so has_status edges target real nodes.
    # Each entry: (name, what, why, scope)
    _task_status_literals = [
        (
            "open",
            "open status -- task ready to start",
            "Task is filed and ready to be worked on; no agent has started it",
            "task lifecycle entry state",
        ),
        (
            "in_progress",
            "in_progress status -- task being worked",
            "An agent is actively working on this task; intent execution under way",
            "task lifecycle middle state",
        ),
        (
            "done",
            "done status -- task completed",
            "Task completed successfully; resolution captured in linked records",
            "task lifecycle terminal state",
        ),
        (
            "canceled",
            "canceled status -- task abandoned",
            "Task closed without completion; not started or abandoned mid-flight",
            "task lifecycle terminal state",
        ),
    ]
    for status_name, status_what, status_why, status_scope in _task_status_literals:
        kg.add_entity(
            status_name,
            kind="literal",
            content=status_why,
            importance=3,
            properties={
                "summary": {
                    "what": status_what,
                    "why": status_why,
                    "scope": status_scope,
                },
            },
        )


def _ensure_user_intent_ontology(kg) -> None:
    """Idempotently seed Slice B user-intent tier predicates.

    Seeded:
      * fulfills_user_message - context entity (user-context) -> record
        entity (user_message). Written by mempalace_declare_user_intents
        for each user-context to its referenced user_message records.
        Identifies the "user-tier" subclass of contexts: a context with
        >=1 fulfills_user_message edge is a user-intent; without one it
        is an activity-intent or operation-intent context.
      * caused_by - context entity (activity-intent ctx) -> entity
        (user-context OR Task). Written by tool_declare_intent when an
        agent supplies cause_id. Cardinality many-to-one: an activity
        traces to exactly one parent cause; a parent can have many
        downstream activities. Slice B-3 wiring.

    Both predicates are skip-listable for the structural-edge fast path
    (no statement required) since they carry pure structural meaning.
    """
    # Each entry: (name, what, why, importance, constraints).
    # Cold-start lock 2026-05-01: hand-curated inline summaries.
    _ui_pred_defs = [
        (
            "fulfills_user_message",
            "fulfills_user_message -- user-context coverage predicate",
            "Edge from a user-context entity to a user_message entity it covers; presence of >=1 such edge marks the context as a user-tier context",
            4,
            {
                "subject_kinds": ["context"],
                "object_kinds": ["user_message"],
                "subject_classes": ["thing"],
                "object_classes": ["thing"],
                "cardinality": "many-to-many",
            },
        ),
        (
            "caused_by",
            "caused_by -- activity-to-parent-cause predicate",
            "Edge from an activity-intent context to its parent cause (a user-context for interactive agents OR a Task entity for non-interactive)",
            4,
            {
                "subject_kinds": ["context"],
                "object_kinds": ["context", "entity"],
                "subject_classes": ["thing"],
                "object_classes": ["Task", "thing"],
                "cardinality": "many-to-one",
            },
        ),
    ]
    for name, what, why, imp, constraints in _ui_pred_defs:
        cardinality = constraints.get("cardinality", "?")
        subj = ",".join(constraints.get("subject_kinds") or []) or "any"
        obj = ",".join(constraints.get("object_kinds") or []) or "any"
        scope = f"{cardinality}; subj={subj}; obj={obj}"[:100]
        kg.add_entity(
            name,
            kind="predicate",
            content=why,
            importance=imp,
            properties={
                "constraints": constraints,
                "summary": {"what": what, "why": why, "scope": scope},
            },
        )


# ==================== STATE SCHEMA ONTOLOGY ====================
# State-protocol v1 design lock 2026-05-03 (Adrian green-lit). The four
# core state schemas live in state_schemas.STATE_SCHEMAS as hardcoded
# Python literals (json_schema + slot_descriptions per kind). This
# seeder materialises each entry as a kind=entity with is_a state_schema
# alongside the state_schema class itself. Adding a fifth state-bearing
# kind requires editing state_schemas.py + adding a tuple below +
# cold-restart per no_back_compat_principle.


def _ensure_state_schema_ontology(kg) -> None:
    """Idempotently seed the `state_schema` class + four core state schemas.

    Cold-start lock 2026-05-01 + Adrian directive 2026-04-30: hand-curated
    inline {what, why, scope} summaries per entry. STATE_SCHEMAS in
    state_schemas.py supplies the json_schema + slot_descriptions payload
    (also hand-curated). add_entity / add_triple upsert via ON CONFLICT
    DO UPDATE so re-running is a no-op on populated palaces.
    """
    from .state_schemas import STATE_SCHEMAS

    _state_schema_class_desc = (
        "Registry entry describing the live state of a state-bearing kind. "
        "Each instance carries json_schema (RFC 6902-compatible JSON Schema), "
        "slot_descriptions (NL per slot, SGD-style after Rastogi 2020), and "
        "parent_schema_id (inheritance pointer or null). Consumed by "
        "injection_gate when surfacing memories and by declare_operation "
        "when validating state deltas."
    )
    kg.add_entity(
        "state_schema",
        kind="class",
        content=_state_schema_class_desc,
        importance=4,
        properties={
            "summary": {
                "what": "state_schema class -- registry per state-bearing kind",
                "why": "each instance carries json_schema + slot_descriptions + parent_schema_id; consumed by injection_gate and declare_operation delta validators",
                "scope": "state-protocol v1; hardcoded set",
            },
        },
    )
    kg.add_triple("state_schema", "is_a", "thing")

    # Each entry: (name, what, why, scope, content, importance).
    # All free-text fields hand-curated per Adrian directive 2026-04-30 --
    # no template helpers, no auto-derivation. STATE_SCHEMAS supplies the
    # json_schema + slot_descriptions payload (also hand-curated, in
    # state_schemas.py).
    _state_schema_defs = [
        (
            "project_state",
            "project_state schema -- live state of a mempalace project",
            "captures current_phase + active_branches + blockers + recent_milestones + open_questions per project; consumed when surfacing project-bearing memories",
            "state-protocol v1",
            "Live state of a project entity. Slots: current_phase (free-form label), active_branches (str list), blockers (str list), recent_milestones (objects with name + optional date), open_questions (str list). Required: current_phase.",
            4,
        ),
        (
            "intent_state",
            "intent_state schema -- live state of an in-flight intent",
            "captures current_step + progress_pct + blockers + latest_observation + open_questions; revised on every declare_operation throughout the intent",
            "state-protocol v1",
            "Live state of an intent execution. Slots: current_step (verb + object), progress_pct (0-100, coarse buckets), blockers (str list), latest_observation (one sentence since last op), open_questions (str list). Required: current_step.",
            4,
        ),
        (
            "agent_state",
            "agent_state schema -- live state of a mempalace agent",
            "captures current_focus + active_intent_id + recent_findings + pending_followups; refreshed across intents within a session",
            "state-protocol v1",
            "Live state of a declared agent. Slots: current_focus (topic label), active_intent_id (str or null), recent_findings (str list cap ~5), pending_followups (str list of one-sentence actions). Required: current_focus.",
            4,
        ),
        (
            "task_state",
            "task_state schema -- live state of a Task entity",
            "captures status enum + assignee + due_date + blockers + progress_pct; complements the has_status edge with structured slots",
            "state-protocol v1",
            "Live state of a Task entity. Slots: status (open / in_progress / blocked / done / cancelled), assignee (str or null), due_date (ISO date or null), blockers (str list, empty when status != blocked), progress_pct (0-100). Required: status.",
            4,
        ),
    ]
    for name, what, why, scope, content, imp in _state_schema_defs:
        schema_def = STATE_SCHEMAS.get(name)
        if schema_def is None:
            logger.warning(
                "state_schema %r missing from STATE_SCHEMAS; skipping seed",
                name,
            )
            continue
        # Option B (Adrian 2026-05-03): kind='state_schema' is graph-only.
        # mcp_server.py VALID_KINDS lists it; both _sync_entity_to_chromadb
        # and _sync_entity_views_to_chromadb skip it so no Chroma rows land.
        kg.add_entity(
            name,
            kind="state_schema",
            content=content,
            importance=imp,
            properties={
                "summary": {"what": what, "why": why, "scope": scope},
                "json_schema": schema_def.get("json_schema"),
                "slot_descriptions": schema_def.get("slot_descriptions"),
                "parent_schema_id": schema_def.get("parent_schema_id"),
            },
        )
        kg.add_triple(name, "is_a", "state_schema")

    # State-protocol v1 predicates (Adrian 2026-05-03): state revisions
    # are linked to the operation that caused them via state_changed_by --
    # the JTMS justification edge per Doyle 1979 + the v2 design lock at
    # record_ga_agent_state_protocol_design_locked_v2_2026_05_03 rule 8.
    # When an operation gets performed_poorly and is later invalidated,
    # dependent state revisions retract one hop deep (cap per the v2 lock).
    # Hand-curated per Adrian directive 2026-04-30; mirrors _op_pred_defs.
    _state_pred_defs = [
        (
            "state_changed_by",
            "state_changed_by -- state-revision-to-operation predicate",
            "JTMS justification edge from a StateRevision row to the operation context that caused it; written by declare_operation when state_deltas include a changed entry; one revision attributable to one op",
            4,
            {
                "subject_kinds": ["entity"],
                "object_kinds": ["operation"],
                "subject_classes": ["thing"],
                "object_classes": ["operation", "thing"],
                "cardinality": "many-to-one",
            },
        ),
    ]
    for name, what, why, imp, constraints in _state_pred_defs:
        cardinality = constraints.get("cardinality", "?")
        subj = ",".join(constraints.get("subject_kinds") or []) or "any"
        obj = ",".join(constraints.get("object_kinds") or []) or "any"
        scope = f"{cardinality}; subj={subj}; obj={obj}"[:100]
        kg.add_entity(
            name,
            kind="predicate",
            content=why,
            importance=imp,
            properties={
                "constraints": constraints,
                "summary": {"what": what, "why": why, "scope": scope},
            },
        )


# ==================== TOP-LEVEL RUNNER ====================


_SEEDERS = (
    ("ensure_operation_ontology", _ensure_operation_ontology),
    ("ensure_task_ontology", _ensure_task_ontology),
    ("ensure_user_intent_ontology", _ensure_user_intent_ontology),
    ("ensure_state_schema_ontology", _ensure_state_schema_ontology),
)


def seed_all(kg, *, skip_env: str = "MEMPALACE_SKIP_SEED") -> None:
    """Run every ontology seeder against ``kg``.

    Safe to call repeatedly -- every seeder is idempotent. A failure in
    one seeder is logged at WARNING and does NOT prevent later seeders
    from running. Honors ``MEMPALACE_SKIP_SEED`` env var: when set, the
    runner is a no-op (used by tests that build their own KnowledgeGraph
    fixtures).
    """
    if skip_env and os.environ.get(skip_env):
        return
    for name, fn in _SEEDERS:
        try:
            fn(kg)
        except Exception as err:  # pragma: no cover - defensive
            logger.warning("%s failed: %r", name, err)
