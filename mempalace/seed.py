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
    """
    kg.add_entity(
        "operation",
        kind="class",
        content=(
            "A recorded tool invocation (tool + truncated args + context_id). "
            "Graph-only -- never embedded. Attached to an intent execution via "
            "executed_op, and to its operation-context via performed_well / "
            "performed_poorly. Cf. Leontiev 1981 Operation tier; arXiv "
            "2512.18950 hierarchical procedural memory."
        ),
        importance=4,
    )
    kg.add_triple("operation", "is_a", "thing")

    _op_pred_defs = [
        (
            "executed_op",
            "Parent-child edge from an intent execution to an operation "
            "entity it performed. Written by finalize_intent when promoting "
            "a rated trace entry.",
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
            "Positive op-quality edge: in the given operation-context the "
            "agent rated this op as good (quality ≥4). Read at declare_"
            "operation time to surface precedent patterns. Distinct from "
            "rated_useful -- that is memory-retrieval relevance; this is "
            "tool+args correctness.",
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
            "Negative op-quality edge: in the given operation-context the "
            "agent rated this op as wrong or suboptimal (quality ≤2). "
            "Surfaced alongside performed_well so the agent sees both "
            "precedent and cautionary cases.",
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
            "S2 correction edge: a poorly-rated op points to the op that "
            "would have been the correct move in the same context. "
            "Written when the agent provides `better_alternative` on an "
            "operation_ratings entry (quality ≤2). Read at declare_operation "
            "time to present cautionary precedent PLUS a concrete "
            "alternative -- not just 'don't do this' but 'do THIS instead'. "
            "op-to-op edge (both subject and object are kind='operation').",
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
            "S3b template-collapse edge: a reusable template record points "
            "at each source operation it distilled. Written by the "
            "memory_gardener's synthesize_operation_template tool when it "
            "resolves an op_cluster_templatizable flag (>=3 same-tool "
            "same-sign precedents that surfaced together at declare_operation "
            "time). Read by retrieve_past_operations (S3c) which hoists the "
            "template into its own lane and suppresses the raw ops the "
            "template covers -- replace-not-append keeps the response "
            "bounded. record→operation edge; one template covers many ops.",
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
    for name, desc, imp, constraints in _op_pred_defs:
        kg.add_entity(
            name,
            kind="predicate",
            content=desc,
            importance=imp,
            properties={"constraints": constraints},
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
    """
    kg.add_entity(
        "Task",
        kind="class",
        content=(
            "An external work item that causes activity-intents in mempalace. "
            "Tasks are kind='entity' nodes with an is_a Task edge -- they hold "
            "a description, an optional external_ref (issue tracker key), and "
            "a has_status edge to a status literal. Used as cause_id by "
            "non-interactive agents (paperclip, scheduled runs) where no "
            "user_message is available."
        ),
        importance=4,
    )
    kg.add_triple("Task", "is_a", "thing")

    _task_pred_defs = [
        (
            "has_status",
            "Edge from a task entity to its current status literal "
            "(open, in_progress, done, canceled). Versioned via "
            "valid_from/valid_to so historical queries resolve "
            "as-of-date. Use kg_invalidate + kg_add to transition.",
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
            "Edge from a task entity to an opaque external identifier "
            "literal (Paperclip / Flowsev / GitHub issue key). Read by "
            "integrators to round-trip task state with the source system. "
            "Optional -- only present when the task originated externally.",
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
    for name, desc, imp, constraints in _task_pred_defs:
        kg.add_entity(
            name,
            kind="predicate",
            content=desc,
            importance=imp,
            properties={"constraints": constraints},
        )

    # Status literals -- declared so has_status edges target real nodes.
    # Single set; new statuses get added here later as needs emerge.
    _task_status_literals = [
        ("open", "Task is filed and ready to be worked on; no agent has started it."),
        ("in_progress", "An agent is actively working on this task; intent execution under way."),
        ("done", "Task completed successfully; resolution captured in linked records."),
        ("canceled", "Task closed without completion; not started or abandoned mid-flight."),
    ]
    for status_name, status_desc in _task_status_literals:
        kg.add_entity(
            status_name,
            kind="literal",
            content=status_desc,
            importance=3,
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
    _ui_pred_defs = [
        (
            "fulfills_user_message",
            "Edge from a user-context entity (kind='context') to a "
            "user_message entity (kind='user_message') that the context "
            "covers. Written by mempalace_declare_user_intents per "
            "declared user-intent. Presence of >=1 such outgoing edge "
            "marks the context as a user-tier context (vs activity-"
            "intent or operation context). Cold-start lock 2026-05-01 "
            "(Adrian's user-message analysis): user_message is its own "
            "kind, not 'record'. The literal user text is value, not "
            "identity -- the user-context that fulfills the message "
            "carries the searchable identity. user_messages skip the "
            "Chroma sync and the summary contract by design.",
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
            "Edge from an activity-intent context (kind='context') to "
            "its parent cause - either a user-context (kind='context' "
            "with at least one fulfills_user_message edge) for "
            "interactive agents, or a Task entity (kind='entity' with "
            "is_a Task) for non-interactive agents. Written by "
            "tool_declare_intent when cause_id is supplied. Cardinality "
            "many-to-one: each activity has exactly one cause; each "
            "cause can spawn many activities.",
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
    for name, desc, imp, constraints in _ui_pred_defs:
        kg.add_entity(
            name,
            kind="predicate",
            content=desc,
            importance=imp,
            properties={"constraints": constraints},
        )


# ==================== TOP-LEVEL RUNNER ====================


_SEEDERS = (
    ("ensure_operation_ontology", _ensure_operation_ontology),
    ("ensure_task_ontology", _ensure_task_ontology),
    ("ensure_user_intent_ontology", _ensure_user_intent_ontology),
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
