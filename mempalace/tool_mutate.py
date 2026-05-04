"""Mutating mempalace tool handlers (mutate bucket).

This module re-exports the state-mutating tool handlers from
``mempalace.mcp_server`` so the PreToolUse carve-out hook can determine
bucket membership by reading ``__all__``. Handler bodies stay in
``mcp_server`` -- moving them here would shuffle code for zero behavioural
change.

The hook does NOT import this module at runtime (that would chain into the
heavy ``mcp_server`` import on every PreToolUse call). Instead, the hook
hardcodes the bucket basenames in ``hooks_cli._MUTATE_BUCKET_BASENAMES``
and ``tests/test_hook_buckets.py::test_mutate_bucket_matches_module_all``
enforces the two stay in sync. If a handler is added or moves bucket,
update BOTH sides -- the drift-sentinel test breaks loudly otherwise.

Bucket semantics: mutate tools change KG / diary state. The gate hook
REQUIRES an active intent for any mutate-bucket call; without an intent
the call is denied with guidance. Under user-message preemption, mutates
are blocked entirely along with everything else except the user-intent
tier-0 carve-outs (``declare_user_intents``, ``extend_feedback``) and
``AskUserQuestion``. ``declare_operation`` lives in this bucket because
it mints a retrieval cue that has to attach to an active intent.
"""

import json  # noqa: E402
from datetime import datetime  # noqa: E402

from mempalace.config import sanitize_content, sanitize_name  # noqa: E402

# Phase 2: lazy mcp_server imports inside each function body to
# avoid the circular when this module is imported BEFORE mcp_server
# has finished loading. Each function imports only what it uses.

# tool_declare_operation stays sourced from intent.py (its canonical home);
# we re-export it through this bucket file because the spec assigns it to
# the mutate bucket (it mints retrieval cues attached to an active intent).
from mempalace.intent import tool_declare_operation  # noqa: E402


def tool_kg_delete_entity(entity: str, agent: str = None):
    """Delete an entity (memory or KG node) and invalidate every edge touching it.

    Works for both memories (ids starting with 'record_' or 'diary_' --
    historical prefixes kept for DB compatibility) and KG entities.
    Invalidates all current edges where the target is subject or object
    (soft-delete, temporal audit trail preserved), then removes its
    description from the appropriate Chroma collection.

    Use this when an entity is truly obsolete (superseded concept, stale memory,
    deleted person). For updating a single fact (one edge becomes untrue while
    the entity itself remains valid), use kg_invalidate(subject, predicate,
    object) on the specific triple instead.
    """
    from mempalace.mcp_server import (
        _STATE,
        _get_collection,
        _get_entity_collection,
        _no_palace,
        _require_agent,
        _require_sid,
        _wal_log,
        logger,
    )

    sid_err = _require_sid(action="kg_delete_entity")
    if sid_err:
        return sid_err
    # mandatory agent attribution.
    agent_err = _require_agent(agent, action="kg_delete_entity")
    if agent_err:
        return agent_err
    if not entity or not isinstance(entity, str):
        return {"success": False, "error": "entity is required (string)."}

    # Determine which collection to target: records live in the record
    # collection; everything else in the entity collection. The 'record_' /
    # 'diary_' id prefixes route to the record collection.
    is_record_id = entity.startswith(("record_", "diary_"))
    col = _get_collection() if is_record_id else _get_entity_collection(create=False)
    if not col:
        return (
            _no_palace()
            if is_record_id
            else {
                "success": False,
                "error": "Entity collection not found.",
            }
        )

    existing = None
    try:
        existing = col.get(ids=[entity])
    except Exception as e:
        return {"success": False, "error": f"lookup failed: {e}"}
    if not existing or not existing.get("ids"):
        return {
            "success": False,
            "error": f"Not found in {'memories' if is_record_id else 'entities'}: {entity}",
        }

    deleted_content = (existing.get("documents") or [""])[0] or ""
    deleted_meta = (existing.get("metadatas") or [{}])[0] or {}

    # Invalidate every current edge involving this entity (both directions).
    invalidated = 0
    try:
        edges = _STATE.kg.query_entity(entity, direction="both") or []
        for e in edges:
            if not e.get("current", True):
                continue
            subj = e.get("subject") or ""
            pred = e.get("predicate") or ""
            obj = e.get("object") or ""
            if not (subj and pred and obj):
                continue
            try:
                _STATE.kg.invalidate(subj, pred, obj)
                invalidated += 1
            except Exception:
                continue
    except Exception:
        pass  # kg lookup failure is non-fatal; we still remove from Chroma

    _wal_log(
        "kg_delete_entity",
        {
            "entity": entity,
            "collection": "memory" if is_record_id else "entity",
            "edges_invalidated": invalidated,
            "deleted_meta": deleted_meta,
            "content_preview": deleted_content[:200],
        },
    )

    try:
        col.delete(ids=[entity])
        # Mark the SQLite entities row as deleted so downstream readers
        # that filter by status='active' stop returning it. Without this
        # update the chroma side is gone but the entities row still
        # appears active in get_entity / list_declared / kg_query, which
        # is the bug the 2026-04-25 audit caught (record_ga_agent_
        # gardener_prune_anomaly_blanks_only).
        try:
            conn = _STATE.kg._conn()
            now = datetime.now().isoformat()
            conn.execute(
                "UPDATE entities SET status='deleted', last_touched=? WHERE id=?",
                (now, entity),
            )
            # Slice C-1 lifecycle hardening (Adrian 2026-05-03):
            # cascade-delete state revisions for state-bearing
            # entities. Migration 024 has no FK CASCADE; without this
            # DELETE the state_revisions rows orphan and the gardener
            # / projection materializer would still see them. Hard
            # delete (not soft) because the entity itself is gone --
            # superseded-state semantics are reserved for invalidated
            # operation contexts (JTMS retraction sweep), not deleted
            # entities. Companion fixes in knowledge_graph.py:
            # latest_state_for_entity status filter, merge_entities
            # cascade UPDATE, record_state_revision deleted-status
            # guard.
            conn.execute(
                "DELETE FROM mempalace_state_revisions WHERE entity_id=?",
                (entity,),
            )
            conn.commit()
        except Exception as sql_err:
            logger.warning(f"kg_delete_entity: SQL status update failed for {entity}: {sql_err}")
        logger.info(
            f"Deleted {'memory' if is_record_id else 'entity'}: {entity} ({invalidated} edges invalidated)"
        )
        return {
            "success": True,
            "entity": entity,
            "source": "memory" if is_record_id else "entity",
            "edges_invalidated": invalidated,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_kg_add(  # noqa: C901
    subject: str,
    predicate: str,
    object: str,
    context: dict = None,  # mandatory Context fingerprint for the edge
    agent: str = None,  # mandatory attribution
    valid_from: str = None,
    statement: str = None,  # natural-language verbalization for retrieval
    # v3 slice 4: required when predicate=='is_a' AND object is a
    # state-bearing class AND subject is kind='entity' (instance).
    initial_state: dict = None,
):
    """Add a relationship to the knowledge graph (Context mandatory).

    IMPORTANT: All three parts must be declared in this session:
    - subject: declared entity (any type EXCEPT predicate)
    - predicate: declared entity with type="predicate"
    - object: declared entity (any type EXCEPT predicate)

    The MANDATORY `context` records WHY this edge is being added -- the
    multi-view perspectives + caller-provided keywords + optional related
    entities. Its view vectors are persisted in mempalace_feedback_contexts
    and the resulting context_id is stored on the triple's
    creation_context_id column. Future feedback (found_useful etc.) applies
    by MaxSim against this fingerprint.

    `agent` is mandatory. Every write operation must be attributed
    to a declared agent (is_a agent); undeclared agents are rejected
    up-front with a declaration recipe.

    `statement` is REQUIRED for every predicate OUTSIDE the skip list
    (is_a, described_by, evidenced_by, executed_by, targeted, has_value,
    session_note_for, derived_from, mentioned_in, found_useful,
    found_irrelevant). It is the natural-language verbalization of the
    triple -- e.g. statement="Adrian lives in Warsaw" for
    ('adrian','lives_in','warsaw'). The statement is stored on the row
    AND embedded into the mempalace_triples Chroma collection so the
    triple becomes a first-class semantic-search result. Autogeneration
    was retired 2026-04-19 -- naive fallbacks poisoned retrieval with
    low-signal text like "record X relates to record Y".

    Call kg_declare_entity for subject/object entities, and
    kg_declare_entity with kind="predicate" for predicates.
    """
    from mempalace.mcp_server import (
        _STATE,
        _active_context_id,
        _declare_entity_recipe,
        _is_declared,
        _past_resolution_hint,
        _require_agent,
        _require_sid,
        _wal_log,
        intent,
    )
    from .knowledge_graph import (
        normalize_entity_name,
        _TRIPLE_SKIP_PREDICATES,
        _normalize_predicate,
        TripleStatementRequired,
    )
    from .scoring import validate_context

    # ── Validate Context (mandatory) -- summary is required on this
    # write tool (Adrian's design lock 2026-04-25). The dict
    # {what, why, scope?} explains the WHAT+WHY of THIS edge -- not
    # of the subject or object nodes (those have their own
    # entity-level summaries from kg_declare_entity). The edge-summary
    # is what context_lookup_or_create persists on the new context
    # entity, and it's what the gardener reads when refining edges.
    clean_context, ctx_err = validate_context(
        context,
        require_summary=True,
        summary_context_for_error="kg_add.context.summary",
    )
    if ctx_err:
        return ctx_err

    # ── mandatory agent attribution ──
    sid_err = _require_sid(action="kg_add")
    if sid_err:
        return sid_err
    agent_err = _require_agent(agent, action="kg_add")
    if agent_err:
        return agent_err

    # ── mandatory statement for non-skip predicates ──
    # Surfaced here at the MCP tool layer (instead of only trusting
    # add_triple's TripleStatementRequired raise) so the agent sees a
    # clean structured error long before the SQL write path.
    _pred_for_check = _normalize_predicate(predicate or "")
    if _pred_for_check and _pred_for_check not in _TRIPLE_SKIP_PREDICATES:
        # Statement is now a structured dict {what, why, scope?}
        # (Adrian's design lock 2026-04-25, edges aligned with summary
        # contract). Reject empty/None up front; bare strings get
        # rejected by validate_statement below with the migration
        # message.
        _stmt_missing = statement is None or (isinstance(statement, str) and not statement.strip())
        if _stmt_missing:
            return {
                "success": False,
                "error": (
                    f"`statement` is required for predicate '{_pred_for_check}'. "
                    f"Pass a structured dict {{what, why, scope?}} verbalizing "
                    f'the triple (e.g. statement={{"what": "Adrian lives in '
                    f'Warsaw", "why": "primary residence; reflects current '
                    f'legal address", "scope": "since 2019"}}). The rendered '
                    f"prose form is stored on the triple row AND embedded into "
                    f"mempalace_triples for semantic search. Skip-list "
                    f"predicates (is_a, described_by, "
                    f"evidenced_by, executed_by, targeted, has_value, "
                    f"session_note_for, derived_from, mentioned_in, found_useful, "
                    f"found_irrelevant) may omit statement \u2014 they are never "
                    f"embedded regardless. Autogeneration was retired 2026-04-19 "
                    f"because naive fallbacks produced retrieval-poisoning text."
                ),
            }
        # Validate dict shape and coerce to prose. add_triple still
        # accepts a string for the statement column; the dict-only
        # contract sits at this MCP-tool boundary.
        try:
            from .knowledge_graph import (
                TripleStatementStructureRequired,
                coerce_statement_for_persist,
                serialize_summary_for_embedding,
            )

            _stmt_dict = coerce_statement_for_persist(
                statement,
                context_for_error="kg_add.statement",
            )
            statement = serialize_summary_for_embedding(_stmt_dict)
        except TripleStatementStructureRequired as _vs_err:
            return {"success": False, "error": str(_vs_err)}

    try:
        subject = sanitize_name(subject, "subject")
        predicate = sanitize_name(predicate, "predicate")
        object = sanitize_name(object, "object")
    except ValueError as e:
        return {"success": False, "error": str(e)}

    # Enforce entity declaration: subject, predicate, and object must all be declared
    sub_normalized = normalize_entity_name(subject)
    pred_normalized = normalize_entity_name(predicate)
    obj_normalized = normalize_entity_name(object)

    errors = []

    # Check subject (must be declared, must NOT be a predicate)
    if not _is_declared(sub_normalized):
        errors.append(
            f"subject '{sub_normalized}' not declared. Call: "
            + _declare_entity_recipe(subject, kind="entity")
        )
    else:
        sub_entity = _STATE.kg.get_entity(sub_normalized)
        if sub_entity and sub_entity.get("kind") == "predicate":
            errors.append(
                f"subject '{sub_normalized}' is kind=predicate, not an entity. "
                f"Subjects must be kind=entity (or class/literal)."
            )

    # Check predicate (must be declared as type="predicate")
    if not _is_declared(pred_normalized):
        errors.append(
            f"predicate '{pred_normalized}' not declared. Call: "
            + _declare_entity_recipe(predicate, kind="predicate")
        )
    else:
        pred_entity = _STATE.kg.get_entity(pred_normalized)
        if pred_entity and pred_entity.get("kind") != "predicate":
            errors.append(
                f"'{pred_normalized}' is kind='{pred_entity.get('kind')}', not 'predicate'. "
                f"Predicates must be declared with kind='predicate'."
            )

    # Check object (must be declared, must NOT be a predicate)
    if not _is_declared(obj_normalized):
        errors.append(
            f"object '{obj_normalized}' not declared. Call: "
            + _declare_entity_recipe(object, kind="entity")
        )
    else:
        obj_entity = _STATE.kg.get_entity(obj_normalized)
        if obj_entity and obj_entity.get("kind") == "predicate":
            errors.append(
                f"object '{obj_normalized}' is kind=predicate, not an entity. "
                f"Objects must be kind=entity (or class/literal)."
            )

    if errors:
        return {
            "success": False,
            "error": "Declaration validation failed for kg_add.",
            "issues": errors,
        }

    # ── Class inheritance helper ──
    def _is_subclass_of(entity_classes, allowed_classes, max_depth=5):
        """Check if any of entity_classes is a subclass of any allowed_class.

        Walks is-a edges upward from each entity class. If it reaches an
        allowed class within max_depth hops, returns True. This enables
        class inheritance: if 'system is-a thing' and constraint allows
        'thing', then any system entity passes.
        """
        if not allowed_classes:
            return True  # no constraint = pass
        # Direct match first
        if any(c in allowed_classes for c in entity_classes):
            return True
        # Walk is-a hierarchy upward
        visited = set(entity_classes)
        frontier = list(entity_classes)
        for _ in range(max_depth):
            next_frontier = []
            for cls in frontier:
                parent_edges = _STATE.kg.query_entity(cls, direction="outgoing")
                for e in parent_edges:
                    if e["predicate"] == "is_a" and e["current"]:
                        parent = e["object"]
                        if parent in allowed_classes:
                            return True
                        if parent not in visited:
                            visited.add(parent)
                            next_frontier.append(parent)
            frontier = next_frontier
            if not frontier:
                break
        return False

    # ── Constraint enforcement ──
    constraint_errors = []
    if pred_entity:
        props = pred_entity.get("properties", {})
        if isinstance(props, str):
            import json as _json

            try:
                props = _json.loads(props)
            except Exception:
                props = {}
        pred_constraints = props.get("constraints", {})

        if pred_constraints:
            # Check subject kind constraint
            allowed_sub_kinds = pred_constraints.get("subject_kinds", [])
            if allowed_sub_kinds and sub_entity:
                sub_kind = sub_entity.get("kind", "entity")
                if sub_kind not in allowed_sub_kinds:
                    constraint_errors.append(
                        f"Subject kind mismatch: '{sub_normalized}' is kind='{sub_kind}', "
                        f"but predicate '{pred_normalized}' expects subject kind in {allowed_sub_kinds}."
                    )

            # Check subject class constraint (via is-a edges)
            allowed_sub_classes = pred_constraints.get("subject_classes", [])
            if allowed_sub_classes and sub_entity:
                sub_classes = [
                    e["object"]
                    for e in _STATE.kg.query_entity(sub_normalized, direction="outgoing")
                    if e["predicate"] == "is_a" and e["current"]
                ]
                if sub_classes and not _is_subclass_of(sub_classes, allowed_sub_classes):
                    constraint_errors.append(
                        f"Subject class mismatch: '{sub_normalized}' is_a {sub_classes}, "
                        f"but predicate '{pred_normalized}' expects subject is_a {allowed_sub_classes}. "
                        f"Options: (1) wrong edge -- use a different subject, "
                        f"(2) wrong predicate -- check kg_list_declared() for a better fit, "
                        f"(3) missing classification -- add is_a edge for '{sub_normalized}', "
                        f"(4) update predicate constraints, "
                        f"(5) create a more specific predicate, "
                        f"(6) rephrase with a more specific entity."
                    )

            # Check object kind constraint
            allowed_obj_kinds = pred_constraints.get("object_kinds", [])
            if allowed_obj_kinds and obj_entity:
                obj_kind = obj_entity.get("kind", "entity")
                if obj_kind not in allowed_obj_kinds:
                    constraint_errors.append(
                        f"Object kind mismatch: '{obj_normalized}' is kind='{obj_kind}', "
                        f"but predicate '{pred_normalized}' expects object kind in {allowed_obj_kinds}."
                    )

            # Check object class constraint
            allowed_obj_classes = pred_constraints.get("object_classes", [])
            if allowed_obj_classes and obj_entity:
                obj_classes = [
                    e["object"]
                    for e in _STATE.kg.query_entity(obj_normalized, direction="outgoing")
                    if e["predicate"] == "is_a" and e["current"]
                ]
                if obj_classes and not _is_subclass_of(obj_classes, allowed_obj_classes):
                    constraint_errors.append(
                        f"Object class mismatch: '{obj_normalized}' is_a {obj_classes}, "
                        f"but predicate '{pred_normalized}' expects object is_a {allowed_obj_classes}. "
                        f"Options: (1) wrong edge, (2) wrong predicate, (3) missing classification, "
                        f"(4) update constraints, (5) new predicate, (6) rephrase with specific entity."
                    )

            # Check cardinality constraint
            cardinality = pred_constraints.get("cardinality", "many-to-many")
            if cardinality in ("many-to-one", "one-to-one"):
                # Subject can have at most 1 edge with this predicate
                existing_sub = [
                    e
                    for e in _STATE.kg.query_entity(sub_normalized, direction="outgoing")
                    if e["predicate"] == pred_normalized and e["current"]
                ]
                if existing_sub:
                    existing_obj = existing_sub[0]["object"]
                    constraint_errors.append(
                        f"Cardinality violation: '{sub_normalized}' already has "
                        f"'{pred_normalized}' -> '{existing_obj}'. "
                        f"Predicate cardinality is {cardinality} (one target per subject). "
                        f"Options: (1) REPLACE -- invalidate the old edge first, then add new, "
                        f"(2) MISTAKE -- you meant a different predicate or entity, "
                        f"(3) EXPAND -- change predicate cardinality to many-to-many."
                    )
            if cardinality in ("one-to-many", "one-to-one"):
                # Object can have at most 1 incoming edge with this predicate
                existing_obj = [
                    e
                    for e in _STATE.kg.query_entity(obj_normalized, direction="incoming")
                    if e["predicate"] == pred_normalized and e["current"]
                ]
                if existing_obj:
                    existing_sub = existing_obj[0]["subject"]
                    constraint_errors.append(
                        f"Cardinality violation: '{obj_normalized}' already has incoming "
                        f"'{existing_sub}' -> '{pred_normalized}'. "
                        f"Predicate cardinality is {cardinality} (one source per object). "
                        f"Options: (1) REPLACE, (2) MISTAKE, (3) EXPAND cardinality."
                    )

    if constraint_errors:
        return {
            "success": False,
            "error": "Predicate constraint violation.",
            "constraint_issues": constraint_errors,
        }

    # ── Provenance: triples.creation_context_id ──
    # Points at the active context entity (kind='context') so the
    # context's outgoing created_under accretion is mirrored on triples
    # via this column. triples aren't first-class entities (no entity
    # row), so a direct created_under edge is inappropriate -- the
    # column is the provenance vehicle. Backed by:
    #   _STATE.kg.triples_created_under(context_id) -> [triple_ids]
    # and the existing JOIN-friendly idx_triples_creation_ctx index.
    # The old persist_context (retired) returned an empty string; we
    # now use the active context id directly when one exists.
    edge_context_id = _active_context_id() or ""

    _wal_log(
        "kg_add",
        {
            "subject": subject,
            "predicate": predicate,
            "object": object,
            "valid_from": valid_from,
            "context_id": edge_context_id,
        },
    )
    try:
        triple_id = _STATE.kg.add_triple(
            sub_normalized,
            pred_normalized,
            obj_normalized,
            valid_from=valid_from,
            creation_context_id=edge_context_id,
            statement=statement,
        )
    except TripleStatementRequired as exc:
        # Should already have been caught by the guard above, but keep
        # this as defense-in-depth in case a future refactor changes the
        # predicate-normalization order.
        return {"success": False, "error": str(exc)}

    # ── v3 slice 4: eager state init on is_a edges ──────────────────
    # State-protocol v3 (Adrian directive 2026-05-04). Mirror of slice
    # 3's atomic is_a + initial_state in kg_declare_entity, but here
    # the edge is being written standalone via kg_add. When predicate
    # is 'is_a' AND object is a state-bearing class (state_updatable=
    # True) AND subject is kind='entity' (instance, not subclass),
    # the caller must supply initial_state -- it is validated against
    # the target class's state_schema and persisted as rev0 via
    # record_state_revision, atomically with the edge write.
    #
    # Idempotent: if the subject already has a revision under this
    # schema_id, we skip rev0 (the existing revision wins; new state
    # changes land via state_deltas at finalize_intent, not through
    # kg_add side effects).
    if pred_normalized == "is_a":
        try:
            _obj_ent = _STATE.kg.get_entity(obj_normalized)
        except Exception:
            _obj_ent = None
        if _obj_ent and _obj_ent.get("kind") == "class":
            import json as _ej

            _obj_props = _obj_ent.get("properties") or {}
            if isinstance(_obj_props, str):
                try:
                    _obj_props = _ej.loads(_obj_props)
                except Exception:
                    _obj_props = {}
            _is_state_bearing = (
                isinstance(_obj_props, dict) and _obj_props.get("state_updatable") is True
            )
            if _is_state_bearing:
                # Only require initial_state when subject is an instance
                # (kind='entity'). Sub-class declarations inherit
                # transitively but carry no instance state.
                try:
                    _sub_ent = _STATE.kg.get_entity(sub_normalized)
                except Exception:
                    _sub_ent = None
                _sub_kind = (_sub_ent or {}).get("kind") or ""
                if _sub_kind == "entity":
                    _sid = _obj_props.get("state_schema_id") or ""
                    if isinstance(_sid, str) and _sid:
                        # Idempotency check -- if subject already has a
                        # revision for this schema, skip rev0.
                        try:
                            _existing = _STATE.kg.latest_state_for_entity(sub_normalized)
                        except Exception:
                            _existing = None
                        if _existing is None:
                            if not isinstance(initial_state, dict):
                                return {
                                    "success": False,
                                    "error": (
                                        f"kg_add: predicate='is_a' object='{obj_normalized}' "
                                        f"is a state-bearing class (schema='{_sid}') and "
                                        f"subject='{sub_normalized}' has no recorded state. "
                                        f"Pass initial_state={{...}} matching the schema. "
                                        f"See wake_up.schemas.{_sid} for the shape."
                                    ),
                                }
                            try:
                                _STATE.kg.record_state_revision(
                                    entity_id=sub_normalized,
                                    schema_id=_sid,
                                    payload=initial_state,
                                    op_context_id=edge_context_id or "",
                                    agent=agent or "",
                                )
                            except ValueError as _ve:
                                return {
                                    "success": False,
                                    "error": (
                                        f"kg_add.initial_state failed schema "
                                        f"'{_sid}' validation: {_ve}. See "
                                        f"wake_up.schemas.{_sid} for the shape."
                                    ),
                                }
                            except Exception:
                                pass  # Substrate-level fallback to gardener

    # ── Contradiction detection: find existing edges that may conflict ──
    conflicts = []
    try:
        # Skip is_a -- those aren't factual contradictions
        if pred_normalized != "is_a":
            existing_edges = _STATE.kg.query_entity(sub_normalized, direction="outgoing")
            for e in existing_edges:
                if not e.get("current", True):
                    continue
                if e["predicate"] != pred_normalized:
                    continue
                existing_obj = e["object"]
                if existing_obj == obj_normalized:
                    continue  # Same edge -- not a contradiction
                # Found: same subject + same predicate + different object
                # Slice 3b 2026-04-28: same conf_<N> integer pattern as
                # mcp_server.py:1007 -- batch-local 1-indexed counter; the
                # conflict_type field below ("edge_contradiction") carries
                # the type info that used to be in the prefix.
                conflict_id = f"conf_{len(conflicts) + 1}"
                past = None
                try:
                    past = _STATE.kg.get_past_conflict_resolution(
                        existing_obj, obj_normalized, "edge_contradiction"
                    )
                except Exception:
                    pass
                conflict_entry = {
                    "id": conflict_id,
                    "conflict_type": "edge_contradiction",
                    "reason": (
                        f"Same subject+predicate, different object: "
                        f"existing '{existing_obj}' vs new '{obj_normalized}'"
                    ),
                    "existing_id": existing_obj,
                    "existing_subject": sub_normalized,
                    "existing_predicate": pred_normalized,
                    "existing_object": existing_obj,
                    "new_id": obj_normalized,
                    "new_subject": sub_normalized,
                    "new_predicate": pred_normalized,
                    "new_object": obj_normalized,
                }
                if past:
                    conflict_entry["past_resolution"] = past
                conflicts.append(conflict_entry)
    except Exception:
        pass  # Non-fatal -- contradiction detection is best-effort

    result = {
        "success": True,
        "triple_id": triple_id,
        "fact": f"{sub_normalized} -> {pred_normalized} -> {obj_normalized}",
    }
    # Echo the rendered statement prose back so the caller can verify
    # the form that got persisted to the row + embedded into
    # mempalace_triples (Adrian's congruence audit 2026-05-01: kg_add
    # was the only write surface that did not echo its rendered form,
    # making it impossible to write tools that consume the persisted
    # statement generically). Skip-list predicates legitimately have no
    # statement -- omit the key in that case so the response shape stays
    # honest about whether anything was embedded.
    if statement:
        result["statement_text"] = statement

    if conflicts:
        _STATE.pending_conflicts = conflicts
        intent._persist_active_intent()
        result["conflicts"] = conflicts
        past_hint = _past_resolution_hint(conflicts)
        result["conflicts_prompt"] = (
            f"{len(conflicts)} potential contradiction(s) found. "
            f"You MUST call mempalace_resolve_conflicts to address each: "
            f"invalidate (old is stale), keep (both valid), or skip (undo new)." + past_hint
        )

    return result


def tool_kg_add_batch(triples: list = None, agent: str = None, edges: list = None):
    """Add multiple KG triples in one call (same params as kg_add, batched).

    Per Adrian's design lock 2026-04-28: each item in `triples` mirrors
    kg_add exactly -- subject, predicate, object, context (mandatory
    per-item), plus optional statement and valid_from. NOTHING different
    just for being in a batch; in particular there is no top-level
    shared-context shortcut, because that would mean the batch entry
    point can take fields kg_add cannot.

    `agent` is the only shared top-level field, mandatory, applied to
    every triple in the batch. Per-item agent overrides are not
    supported (batches are single-author by design; same validation as
    kg_add).

    `statement` (per-item) is REQUIRED for every triple whose predicate
    is OUTSIDE the skip list (is_a, described_by, evidenced_by,
    executed_by, targeted, has_value, session_note_for, derived_from,
    mentioned_in, found_useful, found_irrelevant). Writing a proper
    natural-language verbalization -- e.g. "Adrian lives in Warsaw" for
    ('adrian','lives_in','warsaw') -- lets the triple surface via
    semantic search in the mempalace_triples Chroma collection. Omitting
    it on a non-skip predicate returns a per-item error; skip-list
    predicates may omit it (never embedded).

    Validates each triple independently -- partial success OK. The
    legacy `edges` parameter name is accepted as a back-compat alias
    for `triples`; if both are passed, `triples` wins.
    """
    from mempalace.mcp_server import (
        _require_agent,
        _require_sid,
    )

    # Resolve the input list -- canonical name is `triples`; accept the
    # legacy `edges` alias so older callers continue to work.
    items = triples if triples is not None else edges

    # Some MCP transports stringify top-level array parameters.
    if isinstance(items, str):
        try:
            items = json.loads(items)
        except Exception:
            return {
                "success": False,
                "error": (
                    "`triples` arrived as an unparseable string. Pass a JSON array "
                    "of {subject, predicate, object, context, ...} objects."
                ),
            }
    if not items or not isinstance(items, list):
        return {
            "success": False,
            "error": (
                "triples must be a non-empty list of "
                "{subject, predicate, object, context, ...} dicts."
            ),
        }

    # ── agent validation up-front so we don't partially apply ──
    sid_err = _require_sid(action="kg_add_batch")
    if sid_err:
        return sid_err
    agent_err = _require_agent(agent, action="kg_add_batch")
    if agent_err:
        return agent_err

    failures = []
    succeeded_triples = []
    all_conflicts = []
    for idx, edge in enumerate(items):
        if not isinstance(edge, dict):
            failures.append({"index": idx, "error": "triple item must be a dict"})
            continue
        edge_context = edge.get("context")
        if edge_context is None:
            failures.append(
                {
                    "index": idx,
                    "subject": edge.get("subject"),
                    "predicate": edge.get("predicate"),
                    "object": edge.get("object"),
                    "error": (
                        "Each triple needs its own context dict -- same shape "
                        "as kg_add.context. Per Adrian's design lock 2026-04-28 "
                        "the batch entry point does not accept a shared "
                        "top-level context (nothing different just for being "
                        "in a batch)."
                    ),
                }
            )
            continue
        r = tool_kg_add(
            subject=edge.get("subject", ""),
            predicate=edge.get("predicate", ""),
            object=edge.get("object", ""),
            context=edge_context,
            agent=agent,
            valid_from=edge.get("valid_from"),
            statement=edge.get("statement"),
        )
        if r.get("success"):
            # Keep only the triple_id -- caller supplied the s/p/o.
            if r.get("triple_id"):
                succeeded_triples.append(r["triple_id"])
            if r.get("conflicts"):
                all_conflicts.extend(r["conflicts"])
        else:
            failures.append(
                {
                    "index": idx,
                    "subject": edge.get("subject"),
                    "predicate": edge.get("predicate"),
                    "object": edge.get("object"),
                    "error": r.get("error"),
                    "issues": r.get("issues") or r.get("constraint_issues"),
                }
            )

    # Caller supplied the s/p/o/statement for each edge -- echoing them back
    # is pure token waste. Return counts on success; surface per-edge detail
    # only for failures and any surfaced conflicts.
    response = {
        "success": len(succeeded_triples) > 0,
        "total": len(items),
        "succeeded": len(succeeded_triples),
        "failed": len(failures),
    }
    if succeeded_triples:
        response["triple_ids"] = succeeded_triples
    if failures:
        response["failures"] = failures
    if all_conflicts:
        response["conflicts"] = all_conflicts
    return response


def tool_kg_invalidate(
    subject: str,
    predicate: str,
    object: str,
    ended: str = None,
    agent: str = None,
):
    """Mark a fact as no longer true (set end date). agent required."""
    from mempalace.mcp_server import (
        _STATE,
        _require_agent,
        _require_sid,
        _wal_log,
    )

    sid_err = _require_sid(action="kg_invalidate")
    if sid_err:
        return sid_err
    agent_err = _require_agent(agent, action="kg_invalidate")
    if agent_err:
        return agent_err
    try:
        _wal_log(
            "kg_invalidate",
            {
                "subject": subject,
                "predicate": predicate,
                "object": object,
                "ended": ended,
                "agent": agent,
            },
        )
        _STATE.kg.invalidate(subject, predicate, object, ended=ended)
        return {"success": True, "ended": ended or "today"}
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


def tool_kg_declare_entity(  # noqa: C901
    name: str = None,
    context: dict = None,  # mandatory: {queries, keywords, entities?, summary} -- see validate_context
    kind: str = None,  # REQUIRED -- no default, model must choose
    importance: int = 3,
    properties: dict = None,  # General-purpose metadata
    user_approved_star_scope: bool = False,  # Required for * scope
    added_by: str = None,  # REQUIRED -- agent who declared this entity
    # Record-kind specific (REQUIRED when kind='record').
    slug: str = None,
    content: str = None,  # verbatim record text (kind='record' only)
    content_type: str = None,  # one of: fact, event, discovery, preference, advice, diary
    source_file: str = None,
    entity: str = None,  # entity name(s) to link this record to
    predicate: str = "described_by",  # link predicate
    # v3 slice 3: optional atomic is_a edge + eager state init.
    is_a=None,  # str | list[str] | None -- class(es) this entity is_a
    initial_state: dict = None,  # required when is_a target is state-bearing + kind='entity'
):
    """Declare an entity before using it in KG edges. REQUIRED per session.

    EVERY declaration speaks the unified Context object:

        context = {
          "queries":  list[str]   # 2-5 perspectives on what this entity is
          "keywords": list[str]   # 2-5 caller-provided exact terms
          "entities": list[str]   # 1-10 related entity ids
          "summary":  dict        # MANDATORY {what, why, scope?} -- the
                                  #   structured WHAT+WHY+SCOPE? anchor
                                  #   for the entity (rendered to prose for
                                  #   the cosine view). No auto-derive
                                  #   from queries[0] -- Adrian's design
                                  #   lock 2026-04-25.
        }

    Each query gets embedded as a separate Chroma record under
    '{entity_id}__v{N}' with metadata.entity_id=entity_id, so
    collision detection is multi-view RRF rather than single-vector
    cosine. Readers look up entities via where={"entity_id": X} -- the
    suffix is cosmetic, the metadata is load-bearing. Keywords are stored
    in entity_keywords (the keyword channel reads them directly --
    auto-extraction is gone). The Context's view vectors are also
    persisted in mempalace_feedback_contexts under a generated
    context_id, recorded on the entity, so future feedback (found_useful
    / found_irrelevant) applies via MaxSim by context similarity.

    Args:
        name: Entity name (REQUIRED for kind=entity/class/predicate/literal;
              auto-computed from added_by/slug for kind='record').
        context: MANDATORY Context dict -- see above. Carries the summary
              dict.
        kind: 'entity' | 'class' | 'predicate' | 'literal' | 'record'.
        content: VERBATIM text for kind='record' (the actual record body).
              For non-record kinds, context.summary carries the prose form;
              `content` is record-only.
        importance: 1-5.
        properties: predicate constraints / intent type rules_profile / arbitrary metadata.
        user_approved_star_scope: required only for "*" tool scopes.
        added_by: declared agent name (REQUIRED).
        slug/content_type/source_file/entity/predicate: kind='record' only.

    Returns: status "created" | "exists" | "collision".
    """
    from mempalace.mcp_server import (
        VALID_CARDINALITIES,
        VALID_KINDS,
        _STATE,
        _active_context_id,
        _add_memory_internal,
        _check_entity_similarity_multiview,
        _past_resolution_hint,
        _require_sid,
        _sync_entity_views_to_chromadb,
        _validate_importance,
        _validate_kind,
        _wal_log,
        intent,
    )
    from .knowledge_graph import normalize_entity_name
    from .scoring import validate_context

    sid_err = _require_sid(action="kg_declare_entity")
    if sid_err:
        return sid_err

    # ── Validate Context (mandatory) -- summary is required on this
    # write tool (Adrian's design lock 2026-04-25). Pass dict
    # {what, why, scope?} inside context.
    clean_context, ctx_err = validate_context(
        context,
        require_summary=True,
        summary_context_for_error="kg_declare_entity.context.summary",
    )
    if ctx_err:
        return ctx_err
    queries = clean_context["queries"]
    keywords = clean_context["keywords"]
    summary_dict = clean_context["summary"]
    # clean_context["entities"] is reserved for graph-anchor wiring in P4.3+ (kg_add).

    # ── kind='record' dispatch -- records are first-class entities.
    if kind == "record":
        if content is None or not str(content).strip():
            return {
                "success": False,
                "error": (
                    "kind='record' requires `content` -- the verbatim record text. "
                    "(`context.queries` are search angles, not the body.) "
                    "Use kg_declare_entity(kind='record', slug=..., "
                    "content='<full text>', context={...}, added_by=..., ...)."
                ),
            }
        if not slug:
            return {
                "success": False,
                "error": (
                    "kind='record' requires slug and added_by. "
                    "Slug is a short human-readable identifier (3-6 hyphenated words)."
                ),
            }
        return _add_memory_internal(
            content=content,
            slug=slug,
            added_by=added_by,
            content_type=content_type,
            importance=importance,
            entity=entity,
            predicate=predicate,
            context=clean_context,
            source_file=source_file,
            summary=summary_dict,  # dict-only, sourced from context.summary
        )

    # Non-record: render context.summary into the prose form used as
    # the entity's SQLite description + first chroma vector. The
    # queries[0] auto-derive that used to live here was retired
    # (Adrian's design lock 2026-04-25) -- auto-derivation lets stub
    # placeholders through; the writer must supply the WHAT+WHY
    # explicitly via context.summary.
    from .knowledge_graph import serialize_summary_for_embedding

    description = serialize_summary_for_embedding(summary_dict)

    try:
        description = sanitize_content(description, max_length=5000)
        importance = _validate_importance(importance)
        kind = _validate_kind(kind)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    if not name or not str(name).strip():
        return {
            "success": False,
            "error": (
                "name is required for kind='entity', 'class', 'predicate', or 'literal'. "
                "(For kind='record', use slug + content + added_by instead.)"
            ),
        }

    # Validate added_by: REQUIRED, must be a declared agent (is_a agent)
    if not added_by:
        return {
            "success": False,
            "error": "added_by is required. Pass your agent entity name (e.g., 'ga_agent', 'technical_lead_agent').",
        }
    agent_id_check = normalize_entity_name(added_by)
    if _STATE.kg:
        agent_edges = _STATE.kg.query_entity(agent_id_check, direction="outgoing")
        is_agent = any(
            e["predicate"] == "is_a" and e["object"] == "agent" and e.get("current", True)
            for e in agent_edges
        )
        if not is_agent:
            return {
                "success": False,
                "error": (
                    f"added_by '{added_by}' is not a declared agent (missing is_a agent edge). "
                    f"Declare it as an agent first: "
                    f"kg_declare_entity(name='{added_by}', kind='entity', ...) + "
                    f"kg_add(subject='{added_by}', predicate='is_a', object='agent')"
                ),
            }

    # Check for * scope in tool_permissions -- requires user approval
    if properties and not user_approved_star_scope:
        rules_profile = properties.get("rules_profile", {})
        tool_perms = rules_profile.get("tool_permissions", [])
        star_tools = [p["tool"] for p in tool_perms if p.get("scope") == "*"]
        if star_tools:
            return {
                "success": False,
                "error": (
                    f'BLOCKED: Unrestricted scope ("*") for tools: {star_tools}.\n\n'
                    f"MANDATORY: You MUST ask the user RIGHT NOW and get an explicit YES "
                    f"before proceeding. Do NOT self-approve. Do NOT assume prior approval. "
                    f"Do NOT set user_approved_star_scope=true without asking.\n\n"
                    f"Ask the user exactly this:\n"
                    f"  \"I need to create intent type '{name}' with unrestricted (*) access "
                    f'to {", ".join(star_tools)}. This bypasses scope restrictions. Approve? (yes/no)"\n\n'
                    f"ONLY if the user responds YES to that question in this conversation turn, "
                    f"retry with user_approved_star_scope=true.\n"
                    f"If the user says NO or does not respond: use scoped permissions instead.\n"
                    f"If this is a non-interactive session: this is a BLOCKER. Do not proceed."
                ),
                "star_tools": star_tools,
                "needs_approval": True,
            }

    # Validate constraints for predicates (read from properties.constraints)
    constraints = (properties or {}).get("constraints") if properties else None
    if kind == "predicate":
        if not constraints:
            return {
                "success": False,
                "error": (
                    "Predicates REQUIRE constraints in properties. ALL fields are mandatory: "
                    "subject_kinds, object_kinds, subject_classes, object_classes, cardinality. "
                    'Example: properties={"constraints": {"subject_kinds": ["entity"], "object_kinds": ["entity"], '
                    '"subject_classes": ["system","process"], "object_classes": ["thing"], '
                    '"cardinality": "many-to-one"}}'
                ),
            }
        # ALL 5 constraint fields are REQUIRED -- no optionals
        for field in (
            "subject_kinds",
            "object_kinds",
            "subject_classes",
            "object_classes",
            "cardinality",
        ):
            if field not in constraints:
                return {
                    "success": False,
                    "error": f"constraints must include '{field}'. ALL 5 fields are required: subject_kinds, object_kinds, subject_classes, object_classes, cardinality.",
                }
        # Validate kind lists
        for field in ("subject_kinds", "object_kinds"):
            vals = constraints[field]
            if not isinstance(vals, list) or not vals:
                return {
                    "success": False,
                    "error": f"constraints['{field}'] must be a non-empty list of kinds.",
                }
            for v in vals:
                if v not in VALID_KINDS:
                    return {
                        "success": False,
                        "error": f"constraints['{field}'] contains invalid kind '{v}'. Valid: {sorted(VALID_KINDS)}.",
                    }
        # Validate cardinality
        if constraints["cardinality"] not in VALID_CARDINALITIES:
            return {
                "success": False,
                "error": f"constraints['cardinality'] must be one of {sorted(VALID_CARDINALITIES)}.",
            }
        # Validate subject_classes / object_classes reference real class-kind entities
        for cls_field in ("subject_classes", "object_classes"):
            cls_list = constraints[cls_field]
            if not isinstance(cls_list, list) or not cls_list:
                return {
                    "success": False,
                    "error": f"constraints['{cls_field}'] must be a non-empty list of class entity names. Use ['thing'] for any class.",
                }
            for cls_name in cls_list:
                from .knowledge_graph import normalize_entity_name as _norm

                cls_entity = _STATE.kg.get_entity(_norm(cls_name))
                if not cls_entity:
                    return {
                        "success": False,
                        "error": f"constraints['{cls_field}'] references class '{cls_name}' which doesn't exist. Declare it first with kind='class'.",
                    }
                if cls_entity.get("kind") != "class":
                    return {
                        "success": False,
                        "error": f"constraints['{cls_field}'] references '{cls_name}' which is kind='{cls_entity.get('kind')}', not 'class'.",
                    }

    normalized = normalize_entity_name(name)
    if not normalized or normalized == "unknown":
        return {
            "success": False,
            "error": f"Entity name '{name}' normalizes to nothing. Use a more descriptive name.",
        }

    # Check for exact match (already exists)
    existing = _STATE.kg.get_entity(normalized)
    if existing:
        # Check for collisions with OTHER entities of SAME KIND (not self) -- multi-view
        similar = _check_entity_similarity_multiview(
            queries, kind_filter=kind, exclude_id=normalized
        )
        if similar:
            return {
                "success": False,
                "status": "collision",
                "entity_id": normalized,
                "kind": kind,
                "message": (
                    f"Entity '{normalized}' (kind={kind}) collides with other {kind}s. "
                    f"Disambiguate via kg_update_entity or merge via kg_merge_entities."
                ),
                "collisions": similar,
            }
        # No collisions -- register in session
        _STATE.declared_entities.add(normalized)
        # Update description + importance + kind if provided and different
        if description and description != existing.get("content", ""):
            _STATE.kg.update_entity_content(normalized, description, importance)
            _sync_entity_views_to_chromadb(
                normalized,
                name,
                queries,
                kind,
                importance or 3,
                added_by=added_by,
                summary_view=description,
            )
        # Update properties if provided (merge with existing)
        if properties and isinstance(properties, dict):
            _STATE.kg.update_entity_properties(normalized, properties)
        # Refresh keywords (caller may have updated them)
        _STATE.kg.add_entity_keywords(normalized, keywords)
        return {
            "success": True,
            "status": "exists",
            "entity_id": normalized,
            "kind": existing.get("kind", "entity"),
            "summary_text": existing.get("content") or description,
            "importance": existing.get("importance", 3),
            "edge_count": _STATE.kg.entity_edge_count(normalized),
        }

    # New entity -- multi-view collision check
    similar = _check_entity_similarity_multiview(queries, kind_filter=kind)

    # Create the entity regardless -- conflicts are resolved after creation
    props = properties if isinstance(properties, dict) else {}
    if added_by:
        props["added_by"] = added_by
    # Universal dict-summary storage (followup_universal_dict_summary_storage,
    # design lock 2026-04-26): preserve the validated summary dict
    # {what, why, scope?} alongside the rendered description so the gardener
    # can patch properties.summary in-place without recomputing the prose.
    # summary_dict came from clean_context["summary"] (line 907); description
    # was rendered from the same dict via serialize_summary_for_embedding
    # (line 951). Caller-supplied properties keys take precedence so the
    # dict only lands when the caller hasn't already pre-shaped the field.
    if "summary" not in props:
        props["summary"] = summary_dict
    # SQLite row first (with content rendered from context.summary).
    _STATE.kg.add_entity(
        name, kind=kind, content=description, importance=importance or 3, properties=props
    )
    # Multi-vector embedding into the entity Chroma collection (one record per view).
    # Summary-as-view 2026-04-30: append the rendered summary prose as view N+1
    # so multi_view_max_sim sees the structured WHAT+WHY+SCOPE alongside the
    # query views. CE-with-summary eval showed orthogonal signal cosine misses.
    _sync_entity_views_to_chromadb(
        normalized,
        name,
        queries,
        kind,
        importance or 3,
        added_by=added_by,
        summary_view=description,
    )
    # Caller-provided keywords → entity_keywords table
    _STATE.kg.add_entity_keywords(normalized, keywords)
    # Stamp creation_context_id from the active context entity when
    # one exists. Replaces the retired persist_context path.
    active_ctx = _active_context_id()
    if active_ctx:
        _STATE.kg.set_entity_creation_context(normalized, active_ctx)
    _STATE.declared_entities.add(normalized)

    # ── P1 created_under provenance edge ──
    # Every declared entity records the link to the active context
    # entity. Skip when normalized refers to a context itself (no
    # self-reference) or when the taxonomic root classes are re-seeded.
    _active_ctx = _active_context_id()
    if _active_ctx and normalized != _active_ctx and kind != "context":
        try:
            _STATE.kg.add_triple(normalized, "created_under", _active_ctx)
        except Exception:
            pass  # Non-fatal -- entity exists regardless

    # Auto-add is-a thing for new class entities (ensures class inheritance works)
    if kind == "class" and normalized != "thing":
        try:
            _STATE.kg.add_triple(normalized, "is_a", "thing")
        except Exception:
            pass  # Non-fatal if thing doesn't exist yet

    # ── v3 slice 3: atomic is_a edges + eager state init ─────────────
    # State-protocol v3 (Adrian directive 2026-05-04). Optional `is_a`
    # parameter writes the is_a edge(s) inline so the entity lands
    # ontologically anchored from creation, not in two write calls.
    # When any is_a target is a state-bearing class (state_updatable=
    # True) AND the new entity is kind='entity' (instance, not a
    # subclass), the caller must supply initial_state -- it is
    # validated against the target's state_schema and written as rev0
    # via record_state_revision, atomically with the entity insert.
    # Subclass declarations (kind='class' with is_a → state-bearing
    # class) inherit state_updatable transitively but carry no
    # instance state, so initial_state is not required.
    if is_a:
        _is_a_targets = [is_a] if isinstance(is_a, str) else list(is_a)
        _state_bearing_pairs: list = []  # (class_name, schema_id) for instances
        for _ia_name in _is_a_targets:
            if not isinstance(_ia_name, str) or not _ia_name.strip():
                continue
            _ia_norm = normalize_entity_name(_ia_name)
            _ia_ent = _STATE.kg.get_entity(_ia_norm)
            if not _ia_ent:
                return {
                    "success": False,
                    "error": (
                        f"is_a target '{_ia_name}' (resolved to '{_ia_norm}') "
                        f"is not a declared entity. Declare the class first via "
                        f"kg_declare_entity(name='{_ia_name}', kind='class', ...)."
                    ),
                }
            if _ia_ent.get("kind") != "class":
                return {
                    "success": False,
                    "error": (
                        f"is_a target '{_ia_name}' is kind="
                        f"'{_ia_ent.get('kind')}', not 'class'. is_a edges "
                        f"only target classes."
                    ),
                }
            # Detect state-bearing class -- only matters when WE are an
            # instance (kind='entity'). Class-on-class is_a inherits the
            # contract but no instance-state row is written.
            if kind == "entity":
                import json as _ej

                _ia_props = _ia_ent.get("properties") or {}
                if isinstance(_ia_props, str):
                    try:
                        _ia_props = _ej.loads(_ia_props)
                    except Exception:
                        _ia_props = {}
                if isinstance(_ia_props, dict) and _ia_props.get("state_updatable") is True:
                    _sid = _ia_props.get("state_schema_id") or ""
                    if isinstance(_sid, str) and _sid:
                        _state_bearing_pairs.append((_ia_norm, _sid))
            # Write the is_a edge (idempotent in add_triple).
            try:
                _STATE.kg.add_triple(normalized, "is_a", _ia_norm)
            except Exception:
                pass  # Non-fatal -- entity exists, edge can be retried later

        # If any is_a target is state-bearing AND we're an instance,
        # require initial_state and write rev0 against THIS entity.
        if _state_bearing_pairs:
            if not isinstance(initial_state, dict):
                _schema_ids = ", ".join(sorted({sid for _, sid in _state_bearing_pairs}))
                return {
                    "success": False,
                    "error": (
                        f"kg_declare_entity: is_a target(s) include state-"
                        f"bearing class(es) requiring initial_state. Pass "
                        f"initial_state={{...}} matching the schema(s): "
                        f"{_schema_ids}. See wake_up.schemas for the shapes."
                    ),
                }
            # When the instance is_a multiple state-bearing classes (rare),
            # write one rev0 per schema. Each schema validates the same
            # payload independently; agents authoring multi-class instances
            # must shape initial_state to satisfy every schema.
            _written_schemas: set = set()
            for _, _sid in _state_bearing_pairs:
                if _sid in _written_schemas:
                    continue
                try:
                    _STATE.kg.record_state_revision(
                        entity_id=normalized,
                        schema_id=_sid,
                        payload=initial_state,
                        op_context_id="",
                        agent=added_by or "",
                    )
                    _written_schemas.add(_sid)
                except ValueError as _ve:
                    return {
                        "success": False,
                        "error": (
                            f"kg_declare_entity.initial_state failed schema "
                            f"'{_sid}' validation: {_ve}. See wake_up.schemas"
                            f".{_sid} for the required shape."
                        ),
                    }
                except Exception:
                    # Substrate-level failure (table missing, transient
                    # SQLite error). Don't roll back the entity insert;
                    # the gardener retrofit path remains as fallback.
                    pass

    _wal_log(
        "kg_declare_entity",
        {
            "entity_id": normalized,
            "name": name,
            "description": description[:200],
            "kind": kind,
            "importance": importance,
        },
    )

    result = {
        "success": True,
        "status": "created",
        "entity_id": normalized,
        "kind": kind,
        "summary_text": description,
        "importance": importance or 3,
    }

    # ── Conflict detection: flag similar entities for resolution ──
    if similar:
        conflicts = []
        for s in similar:
            # Slice 3b 2026-04-28: same conf_<N> integer pattern as
            # mcp_server.py:1007 -- batch-local 1-indexed counter; the
            # conflict_type field below ("entity_duplicate") carries the
            # type info that used to be in the prefix.
            conflict_id = f"conf_{len(conflicts) + 1}"
            past = None
            try:
                past = _STATE.kg.get_past_conflict_resolution(
                    s["entity_id"], normalized, "entity_duplicate"
                )
            except Exception:
                pass
            conflict_entry = {
                "id": conflict_id,
                "conflict_type": "entity_duplicate",
                "reason": (
                    f"New entity '{normalized}' has similar description to "
                    f"existing '{s['entity_id']}' (similarity: {s.get('similarity', '?')})"
                ),
                "existing_id": s["entity_id"],
                "existing_description": s.get("content", "")[:200],
                "new_id": normalized,
                "new_description": description[:200],
            }
            if past:
                conflict_entry["past_resolution"] = past
            conflicts.append(conflict_entry)
        _STATE.pending_conflicts = conflicts
        intent._persist_active_intent()
        result["conflicts"] = conflicts
        past_hint = _past_resolution_hint(conflicts)
        result["conflicts_prompt"] = (
            f"{len(conflicts)} similar entity/entities found. "
            f"Call mempalace_resolve_conflicts: merge (combine both), "
            f"keep (both are distinct), or skip (undo new entity)." + past_hint
        )

    return result


def tool_kg_update_entity(  # noqa: C901
    entity: str,
    summary=None,  # dict {what, why, scope?} -- see validate_summary
    importance: int = None,
    properties: dict = None,
    context: dict = None,  # optional: re-record creation_context when meaning changes
    agent: str = None,  # mandatory attribution
    # Record-specific (only meaningful when entity is a kind='record')
    content_type: str = None,
):
    """Update any entity (record or KG node) in place. Pass only the fields you want to change.

    `context` is OPTIONAL but RECOMMENDED whenever you change
    semantic fields (`summary` for entities, or `properties` that alter
    meaning like predicate constraints / intent-type rules). When present
    the Context's view vectors are persisted and the entity's
    creation_context_id is repointed to the new context -- future MaxSim
    feedback then transfers against the updated meaning, not the old one.

    Args:
        entity: Entity ID or record ID to update.
        summary: New structured summary {what, why, scope?}. For entities
            (kind=entity/class/predicate/literal): re-syncs to entity ChromaDB
            and runs collision distance check. For records: NOT supported here --
            use kg_delete_entity + kg_declare_entity to change record content.
            Strings are rejected by validate_summary (dict-only contract,
            Adrian's design lock 2026-04-25).
        importance: New importance (1-5). Works for both entities and records.
        properties: Merged INTO existing properties dict (shallow merge at top
            level). For predicates use {"constraints": {...}} to replace
            constraints. For intent types {"rules_profile": {...}} to update slots
            or tool_permissions.
        content_type: Record-only content type update (no re-embedding).
    """
    from mempalace.mcp_server import (
        ENTITY_SIMILARITY_THRESHOLD,
        VALID_CARDINALITIES,
        VALID_KINDS,
        _STATE,
        _active_context_id,
        _check_entity_similarity,
        _get_collection,
        _no_palace,
        _require_agent,
        _require_sid,
        _sync_entity_to_chromadb,
        _update_entity_chromadb_metadata,
        _validate_content_type,
        _validate_importance,
        _wal_log,
        logger,
    )
    from .knowledge_graph import normalize_entity_name
    import json as _json

    if not entity or not isinstance(entity, str):
        return {"success": False, "error": "entity is required (string)."}

    # ── mandatory agent attribution ──
    sid_err = _require_sid(action="kg_update_entity")
    if sid_err:
        return sid_err
    agent_err = _require_agent(agent, action="kg_update_entity")
    if agent_err:
        return agent_err

    is_record_id = entity.startswith(("record_", "diary_"))

    # ── Validate inputs ──
    # `summary` is the entity's structured WHAT/WHY/SCOPE? anchor -- rendered
    # to prose for the entity collection's embedded text. The dict-only
    # contract (Adrian's design lock 2026-04-25) requires {what, why, scope?};
    # coerce_summary_for_persist validates and normalises, then we render
    # prose for sanitize_content and storage. Records are rejected below
    # (delete-then-redeclare), so we only run the gate on entity-shaped IDs.
    rendered_summary = None
    summary_dict = None
    if summary is not None and not is_record_id:
        try:
            from .knowledge_graph import (
                SummaryStructureRequired,
                coerce_summary_for_persist,
                serialize_summary_for_embedding,
            )

            summary_dict = coerce_summary_for_persist(
                summary,
                context_for_error="kg_update_entity.summary",
            )
        except SummaryStructureRequired as _vs_err:
            return {"success": False, "error": str(_vs_err)}
        # Render prose form for the embedded text + downstream consumers
        rendered_summary = serialize_summary_for_embedding(summary_dict)

    try:
        if rendered_summary is not None:
            rendered_summary = sanitize_content(rendered_summary, max_length=5000)
        if importance is not None:
            importance = _validate_importance(importance)
        if content_type is not None:
            content_type = _validate_content_type(content_type)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    # Reject contradictory inputs early
    if is_record_id and summary is not None:
        return {
            "success": False,
            "error": (
                "Cannot update record summary in place -- embeddings would be "
                "stale. Use kg_delete_entity then kg_declare_entity(kind='record', ...) "
                "to replace record content."
            ),
        }
    if not is_record_id and content_type is not None:
        return {
            "success": False,
            "error": (
                "content_type is a record-only field. For non-record entities, "
                "use properties={...} to update metadata."
            ),
        }

    # ── Memory path: in-place metadata update on the memory collection ──
    if is_record_id:
        col = _get_collection()
        if not col:
            return _no_palace()
        existing = col.get(ids=[entity], include=["metadatas"])
        if not existing.get("ids"):
            return {"success": False, "error": f"Memory not found: {entity}"}

        old_meta = dict(existing["metadatas"][0] or {})
        new_meta = dict(old_meta)
        updated_fields = []
        if content_type is not None and old_meta.get("content_type") != content_type:
            new_meta["content_type"] = content_type
            updated_fields.append("content_type")
        if importance is not None and old_meta.get("importance") != importance:
            new_meta["importance"] = importance
            updated_fields.append("importance")

        if not updated_fields:
            return {"success": True, "reason": "no_change", "entity": entity}

        _wal_log(
            "kg_update_entity",
            {
                "entity": entity,
                "source": "memory",
                "old_meta": old_meta,
                "new_meta": new_meta,
                "updated_fields": updated_fields,
            },
        )
        try:
            col.update(ids=[entity], metadatas=[new_meta])
            logger.info(f"Updated memory: {entity} fields={updated_fields}")
            return {
                "success": True,
                "entity": entity,
                "source": "memory",
                "updated_fields": updated_fields,
                "new_metadata": new_meta,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Entity path: SQLite update + ChromaDB sync + collision check ──
    normalized = normalize_entity_name(entity)
    existing = _STATE.kg.get_entity(normalized)
    if not existing:
        # Bug 2 fix 2026-04-28: kg.get_entity is SQLite-only and fails for
        # entities that exist only in the Chroma multi-view collection
        # (post-M1 unification gap -- kg_query falls back to
        # _fetch_entity_details which queries Chroma with metadata.entity_id
        # filter + raw-id fallback). Before returning the opaque "not found"
        # error, probe Chroma via the same path kg_query uses; if the
        # entity IS there but missing from SQLite, surface the half-present
        # state and direct the caller at kg_declare_entity to re-sync. The
        # update itself isn't safe to fabricate from Chroma metadata alone
        # (no soft-delete history, properties JSON shape may not match), so
        # we hand the call back with an actionable error instead of
        # silently writing partial state.
        try:
            from .mcp_server import _fetch_entity_details

            _chroma_details = _fetch_entity_details(normalized)
        except Exception:
            _chroma_details = None
        if _chroma_details:
            return {
                "success": False,
                "error": (
                    f"Entity '{normalized}' is present in Chroma "
                    f"(kind={_chroma_details.get('kind', '?')!r}) but "
                    f"missing from the SQLite entities table -- likely a "
                    f"post-M1 sync gap. kg_update_entity needs the SQLite "
                    f"row to update properties safely; re-sync by calling "
                    f"kg_declare_entity({normalized!r}, ...) which acts as "
                    f"an upsert and restores both sides."
                ),
                "lookup_mismatch": True,
                "chroma_details": _chroma_details,
            }
        return {"success": False, "error": f"Entity '{normalized}' not found."}

    updated_fields = []
    final_description = existing["content"]
    final_importance = existing.get("importance", 3)

    # Rewrite-count limit (closes 2026-04-25 audit finding #5).
    # Bound how many times memory_gardener can rewrite an entity's
    # summary so unstable rewrites don't compound across runs.
    # Only enforced for the gardener; manual edits stay unconstrained.
    # The counter lives in entities.properties JSON so no migration is
    # required.
    if (
        rendered_summary is not None
        and rendered_summary != existing["content"]
        and (agent or "").strip() == "memory_gardener"
    ):
        _existing_props_for_count = existing.get("properties", {})
        if isinstance(_existing_props_for_count, str):
            try:
                _existing_props_for_count = _json.loads(_existing_props_for_count)
            except Exception:
                _existing_props_for_count = {}
        _rewrite_count = int(_existing_props_for_count.get("summary_rewrite_count", 0) or 0)
        if _rewrite_count >= 3:
            return {
                "success": False,
                "error": (
                    f"Refusing summary rewrite on '{normalized}': memory_gardener "
                    f"has already rewritten this entity {_rewrite_count} times. Multiple "
                    "rewrites without convergence usually indicate weak grounding "
                    "evidence; defer this flag instead of compounding the drift."
                ),
                "summary_rewrite_count": _rewrite_count,
            }
        # Inject the bumped counter into the user-supplied properties so
        # the existing merge-and-persist code at line ~4676 picks it up.
        properties = dict(properties or {})
        properties["summary_rewrite_count"] = _rewrite_count + 1

    # Summary update + ChromaDB resync (still writes to entities.description column)
    if rendered_summary is not None and rendered_summary != existing["content"]:
        _STATE.kg.update_entity_content(normalized, rendered_summary)
        final_description = rendered_summary
        updated_fields.append("summary")

    # Properties merge (constraints validation when kind='predicate')
    if properties is not None:
        existing_props = existing.get("properties", {})
        if isinstance(existing_props, str):
            try:
                existing_props = _json.loads(existing_props)
            except Exception:
                existing_props = {}

        # If updating constraints on a predicate, validate before persisting.
        if "constraints" in properties and existing.get("kind") == "predicate":
            constraints = properties["constraints"]
            for field in ("subject_kinds", "object_kinds"):
                if field not in constraints:
                    return {
                        "success": False,
                        "error": f"constraints must include '{field}'.",
                    }
                vals = constraints[field]
                if not isinstance(vals, list) or not vals:
                    return {
                        "success": False,
                        "error": f"constraints['{field}'] must be a non-empty list.",
                    }
                for v in vals:
                    if v not in VALID_KINDS:
                        return {
                            "success": False,
                            "error": f"Invalid kind '{v}' in constraints['{field}']. Valid: {sorted(VALID_KINDS)}.",
                        }
            if "cardinality" in constraints:
                if constraints["cardinality"] not in VALID_CARDINALITIES:
                    return {
                        "success": False,
                        "error": f"Invalid cardinality. Valid: {sorted(VALID_CARDINALITIES)}.",
                    }
            for cls_field in ("subject_classes", "object_classes"):
                if cls_field in constraints:
                    for cls_name in constraints[cls_field]:
                        cls_eid = normalize_entity_name(cls_name)
                        cls_ent = _STATE.kg.get_entity(cls_eid)
                        if not cls_ent:
                            return {
                                "success": False,
                                "error": f"Class '{cls_name}' not found. Declare with kind='class' first.",
                            }
                        if cls_ent.get("kind") != "class":
                            return {
                                "success": False,
                                "error": f"'{cls_name}' is kind='{cls_ent.get('kind')}', not 'class'.",
                            }

        merged_props = dict(existing_props or {})
        merged_props.update(properties)  # shallow merge
        conn = _STATE.kg._conn()
        conn.execute(
            "UPDATE entities SET properties = ? WHERE id = ?",
            (_json.dumps(merged_props), normalized),
        )
        conn.commit()
        updated_fields.append("properties")

    # Importance update
    if importance is not None and importance != existing.get("importance"):
        conn = _STATE.kg._conn()
        conn.execute(
            "UPDATE entities SET importance = ? WHERE id = ?",
            (importance, normalized),
        )
        conn.commit()
        final_importance = importance
        updated_fields.append("importance")

    if not updated_fields:
        return {"success": True, "reason": "no_change", "entity": normalized}

    # Re-sync ChromaDB if summary or importance changed.
    # Two distinct paths because they target DIFFERENT records:
    #  - summary changes re-embed (write a fresh single-view record via
    #    _sync_entity_to_chromadb; multi-view-aware re-embedding is a
    #    separate concern, tracked under followup_universal_dict_summary_storage).
    #  - importance changes only need a metadata patch across the EXISTING
    #    records. Calling _sync_entity_to_chromadb for importance-only
    #    updates was orphan-writing a single-view record under id=entity_id
    #    (no entity_id metadata back-pointer) while leaving the canonical
    #    multi-view records with stale importance, so _fetch_entity_details'
    #    where={'entity_id': eid} lookup never saw the new value (FINDING-1
    #    2026-04-28 -- write went to SQLite, read came from Chroma multi-view
    #    metadata which the sync missed).
    if "summary" in updated_fields:
        _sync_entity_to_chromadb(
            normalized,
            existing["name"],
            final_description,
            existing.get("kind") or existing.get("type", "entity"),
            final_importance,
        )
    if "importance" in updated_fields:
        _update_entity_chromadb_metadata(normalized, importance=final_importance)

    # ── re-record creation_context when meaning changed ──
    # A summary or properties change IS a semantic update -- future
    # MaxSim-graded feedback should attach to the new meaning, not the old.
    # Pure-importance updates don't move meaning, so we skip context re-persist
    # unless summary/properties changed too.
    semantic_change = any(f in updated_fields for f in ("summary", "properties"))
    if semantic_change and context is not None:
        from .scoring import validate_context as _validate_context

        clean_ctx, ctx_err = _validate_context(
            context,
            require_summary=True,
            summary_context_for_error="kg_update_entity.context.summary",
        )
        if ctx_err:
            return ctx_err
        # Stamp the active context on the entity; refresh its stored
        # keywords to the new Context.keywords.
        active_ctx = _active_context_id()
        if active_ctx:
            _STATE.kg.set_entity_creation_context(normalized, active_ctx)
        _STATE.kg.add_entity_keywords(normalized, clean_ctx["keywords"])
        updated_fields.append("creation_context")

    _wal_log(
        "kg_update_entity",
        {"entity": normalized, "source": "entity", "updated_fields": updated_fields},
    )

    result = {
        "success": True,
        "entity": normalized,
        "source": "entity",
        "updated_fields": updated_fields,
    }
    # Stamp the current active context on the response when we just
    # wrote it (semantic-change path above sets creation_context_id to
    # _active_context_id()). Empty when we're in a non-semantic update.
    _new_ctx = _active_context_id() if semantic_change else ""
    if _new_ctx:
        result["creation_context_id"] = _new_ctx

    # P5.10 hint: nudge callers to pass context when meaning changed
    # but they omitted the context dict entirely.
    if semantic_change and context is None:
        result["context_hint"] = (
            "Summary/properties changed but no `context` was provided -- "
            "future MaxSim feedback will still attach to the OLD creation_context_id. "
            "Pass `context={queries,keywords,entities?}` to re-anchor the entity."
        )

    # Collision distance check when summary changed (was the point of the
    # legacy update_entity_description tool -- keep that behaviour).
    if "summary" in updated_fields:
        similar = _check_entity_similarity(final_description, exclude_id=normalized, threshold=0.7)
        distance_checks = [
            {
                "compared_to": s["entity_id"],
                "similarity": s["similarity"],
                "is_distinct": s["similarity"] < ENTITY_SIMILARITY_THRESHOLD,
                "threshold": ENTITY_SIMILARITY_THRESHOLD,
            }
            for s in similar
        ]
        all_distinct = all(d["is_distinct"] for d in distance_checks) if distance_checks else True
        result["distance_checks"] = distance_checks
        result["all_distinct"] = all_distinct
        result["hint"] = (
            "All clear -- re-declare this entity to register it."
            if all_distinct
            else "Still too similar to some entities. Make your description more specific."
        )

    return result


def tool_kg_merge_entities(source: str, target: str, summary: dict = None, agent: str = None):
    """Merge source entity into target. All edges rewritten. Source becomes alias.

    Use when kg_declare_entity returns 'collision' and the entities are
    actually the same thing. All triples from source are moved to target.
    Source name becomes an alias that auto-resolves to target in future queries.

    Args:
        source: Entity to merge FROM (will be soft-deleted).
        target: Entity to merge INTO (will be kept, edges grow).
        summary: Optional new structured summary {what, why, scope?} for the
            merged entity (dict-only contract, Adrian's design lock 2026-04-25).
            Strings are rejected by validate_summary. Coerced + rendered to
            prose internally before persisting.
        agent: mandatory, declared agent attributing this merge.
    """
    from mempalace.mcp_server import (
        _STATE,
        _get_entity_collection,
        _require_agent,
        _require_sid,
        _sync_entity_to_chromadb,
        _wal_log,
    )

    sid_err = _require_sid(action="kg_merge_entities")
    if sid_err:
        return sid_err
    agent_err = _require_agent(agent, action="kg_merge_entities")
    if agent_err:
        return agent_err

    # ── Coerce summary dict → prose at the API edge (mirrors kg_update_entity) ──
    rendered_summary = None
    if summary is not None:
        try:
            from .knowledge_graph import (
                SummaryStructureRequired,
                coerce_summary_for_persist,
                serialize_summary_for_embedding,
            )

            summary_dict = coerce_summary_for_persist(
                summary,
                context_for_error="kg_merge_entities.summary",
            )
            rendered_summary = serialize_summary_for_embedding(summary_dict)
        except SummaryStructureRequired as _vs_err:
            return {"success": False, "error": str(_vs_err)}

    _wal_log(
        "kg_merge_entities",
        {
            "source": source,
            "target": target,
            "summary": rendered_summary[:200] if rendered_summary else None,
            "agent": agent,
        },
    )

    result = _STATE.kg.merge_entities(source, target, rendered_summary)
    if "error" in result:
        return {"success": False, "error": result["error"]}

    # Update ChromaDB: remove source, update target
    from .knowledge_graph import normalize_entity_name

    source_id = normalize_entity_name(source)
    target_id = normalize_entity_name(target)

    ecol = _get_entity_collection(create=False)
    if ecol:
        try:
            ecol.delete(ids=[source_id])
        except Exception:
            pass
        target_entity = _STATE.kg.get_entity(target_id)
        if target_entity:
            _sync_entity_to_chromadb(
                target_id,
                target_entity["name"],
                target_entity["content"],
                target_entity.get("type", "concept"),
                target_entity.get("importance", 3),
            )

    # Register target as declared (source is now alias for target)
    _STATE.declared_entities.discard(source_id)
    _STATE.declared_entities.add(target_id)

    return {
        "success": True,
        "source": result["source"],
        "target": result["target"],
        "edges_moved": result["edges_moved"],
        "aliases_created": result["aliases_created"],
    }


def tool_diary_write(
    agent_name: str,
    entry: str,
    summary: dict = None,
    slug: str = "",
    topic: str = "general",
    content_type: str = "diary",
    importance: int = None,
):
    """
    Write a diary entry for this agent. Entries are timestamped and
    accumulate over time, scoped by agent name.

    The diary is a HIGH-LEVEL SESSION NARRATIVE -- not a detailed log.
    Write in readable prose.

    WHAT TO INCLUDE:
    - Decisions made with the user (approved designs, rejected ideas)
    - Big-picture status and direction
    - Pending items and backlog
    - Cross-intent narrative (how multiple actions connected)

    WHAT NOT TO INCLUDE (already captured by intent results):
    - Individual commits or features shipped
    - Gotchas and learnings (already KG entities via finalize_intent)
    - Tool traces or detailed action logs

    Each entry should be a DELTA from the previous -- what changed,
    not a full restatement of everything.

    Args:
        summary: REQUIRED dict {what, why, scope?} (cold-start lock
              2026-05-01). Pre-cold-start, diary entries bypassed the
              summary contract by writing directly to Chroma -- one of
              the 12 catalogued bypass surfaces. The summary distills
              the entry's WHAT (what changed this session) and WHY
              (the decision/discovery that made it diary-worthy) so
              retrieval surfaces the right entry on later searches.
        slug: Descriptive identifier for this entry (e.g. 'session12-scoring-design').
              If not provided, falls back to date-topic format.
        topic: Topic tag (optional, default: general)
        content_type: default 'diary'. Override with 'discovery' for
              "today I learned" entries that deserve higher retrieval priority,
              or 'event' for plain activity logs.
        importance: 1-5. Defaults to unset (treated as 3 by L1). Use 4 for
                    entries with learned lessons, 5 only for agent-wide
                    critical notes.
    """
    from mempalace.mcp_server import (
        _STATE,
        _get_collection,
        _no_palace,
        _require_sid,
        _slugify,
        _validate_content_type,
        _validate_importance,
        _wal_log,
        logger,
    )

    sid_err = _require_sid(action="diary_write")
    if sid_err:
        return sid_err
    try:
        agent_name = sanitize_name(agent_name, "agent_name")
        entry = sanitize_content(entry)
        content_type = _validate_content_type(content_type)
        importance = _validate_importance(importance)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    # Cold-start lock 2026-05-01: summary is required on every diary
    # entry. Pre-cold-start the diary path bypassed the summary contract
    # by writing straight to Chroma; that put 1000+ diary records into
    # the corpus with no structured what/why and no field-level audit.
    if summary is None:
        return {
            "success": False,
            "error": (
                "`summary` is required on every diary entry (cold-start "
                "lock 2026-05-01). Pass a dict {'what': '<noun phrase>', "
                "'why': '<purpose / role / claim>', 'scope': "
                "'<temporal/domain qualifier>'?}. Example for a diary "
                "entry: {'what': 'session 12 cold-start gate locked', "
                "'why': 'Adrian approved 3-level identity model and we "
                "shipped mint_entity + 4 phantom-site refactors today', "
                "'scope': '2026-05-01'}. The summary distills WHAT "
                "changed this session and WHY it was diary-worthy."
            ),
        }
    try:
        from .knowledge_graph import (
            SummaryStructureRequired,
            coerce_summary_for_persist,
            serialize_summary_for_embedding,
        )

        summary_dict = coerce_summary_for_persist(summary, context_for_error="diary_write.summary")
    except SummaryStructureRequired as _vs_err:
        return {"success": False, "error": str(_vs_err)}
    summary_prose = serialize_summary_for_embedding(summary_dict)

    from .knowledge_graph import normalize_entity_name as _norm_eid

    agent_slug = _norm_eid(agent_name)
    col = _get_collection(create=True)
    if not col:
        return _no_palace()

    now = datetime.now()
    if slug and slug.strip():
        diary_slug = _slugify(slug)
    else:
        diary_slug = _slugify(f"{now.strftime('%Y%m%d-%H%M%S')}-{topic}")
    entry_id = f"diary_{agent_slug}_{diary_slug}"

    _wal_log(
        "diary_write",
        {
            "agent_name": agent_name,
            "topic": topic,
            "entry_id": entry_id,
            "entry_preview": entry[:200],
            "content_type": content_type,
            "importance": importance,
        },
    )

    try:
        meta = {
            "content_type": content_type or "diary",
            "topic": topic,
            "type": "diary_entry",
            "added_by": agent_name,
            "filed_at": now.isoformat(),
            "date_added": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            # Cold-start lock 2026-05-01: persist both the rendered
            # summary prose (for retrieval display) and structured
            # field-level shape (for the gardener's field-level patches
            # and audit trail). Mirrors _add_memory_internal's pattern
            # so diary records are introspectable on the same axes as
            # all other records.
            "summary": summary_prose,
            "summary_what": summary_dict["what"],
            "summary_why": summary_dict["why"],
        }
        if "scope" in summary_dict:
            meta["summary_scope"] = summary_dict["scope"]
        if importance is not None:
            meta["importance"] = importance
        # Mirror the _add_memory_internal diagnostic pattern: if chroma
        # throws 'TextInputSequence must be str in add', re-raise with
        # explicit memory_id/types/lens/meta so the next live occurrence
        # is actionable instead of a bare tokenizer message.
        if not isinstance(entry, str):
            return {
                "success": False,
                "error": (f"internal: diary entry must be str (got {type(entry).__name__})."),
            }
        try:
            col.add(
                ids=[entry_id],
                documents=[entry],
                metadatas=[meta],
            )
        except Exception as _add_err:
            _meta_types = {k: type(v).__name__ for k, v in meta.items()}
            _msg = (
                f"col.add failed on diary id={entry_id!r}: "
                f"{type(_add_err).__name__}: {_add_err}. "
                f"types[ids]=list[{type(entry_id).__name__}], "
                f"types[documents]=list[{type(entry).__name__}], "
                f"len(entry)={len(entry)}, meta_value_types={_meta_types}"
            )
            logger.error(_msg)
            raise RuntimeError(_msg) from _add_err

        # Cold-start lock 2026-05-01 (issue #4 follow-up): register the
        # diary entry as a row in the SQLite entities table too, mirroring
        # what mint_entity does for kind='record'. Pre-fix, diary_write
        # wrote ONLY to the Chroma record collection; SQLite had no row,
        # so the phantom-rejection in add_rated_edge / add_triple blocked
        # any feedback edge that pointed at a diary memory id. The
        # downstream symptom was finalize_intent / extend_feedback failing
        # to close out coverage when a diary entry surfaced as a memory
        # in the active context's neighbourhood. The entities-table
        # registration is the canonical source-of-truth that
        # render_memory_preview reads -- properties.summary mirrors the
        # structured field-level shape persisted in Chroma metadata.
        # Skip Chroma re-sync (already wrote above) by using add_entity
        # directly instead of mint_entity (which would re-embed and try
        # to mint identity/probe views the diary doesn't have).
        try:
            _ent_props = {
                "summary": dict(summary_dict),
                "topic": topic,
                "content_type": content_type or "diary",
                "filed_at": now.isoformat(),
            }
            _STATE.kg.add_entity(
                entry_id,
                properties=_ent_props,
                content=entry,
                importance=importance if importance is not None else 3,
                kind="record",
                session_id=_STATE.session_id or "",
            )
        except Exception as _ent_err:
            # Entity-table registration is required for feedback to attach
            # cleanly. Fail loud (no back-compat 2026-05-01) so silent
            # phantom-state regressions cannot resurface.
            logger.error(
                "diary_write: entities-table registration failed for %s: %s",
                entry_id,
                _ent_err,
            )
            return {
                "success": False,
                "error": (
                    f"diary_write: entities-table registration failed for "
                    f"{entry_id!r}: {type(_ent_err).__name__}: {_ent_err}. "
                    f"The Chroma row was written but the SQLite entity row "
                    f"was not -- this would cause phantom-rejection on any "
                    f"feedback edge pointing at this diary memory id. "
                    f"Fix the underlying cause; do not proceed."
                ),
            }
        logger.info(f"Diary entry: {entry_id} content_type={content_type} imp={importance}")

        # Update the stop hook save counter -- proves diary was actually written.
        # The stop hook writes a _pending_save marker but does NOT update
        # last_save itself. This prevents the dodge where agents ignore the
        # save prompt and the counter resets anyway. Only diary_write updates it.
        try:
            from .hooks_cli import STATE_DIR

            STATE_DIR.mkdir(parents=True, exist_ok=True)
            # sid is guaranteed non-empty by _require_sid at entry.
            sid = _STATE.session_id
            pending_file = STATE_DIR / f"{sid}_pending_save"
            if pending_file.is_file():
                exchange_count = pending_file.read_text(encoding="utf-8").strip()
                last_save_file = STATE_DIR / f"{sid}_last_save"
                last_save_file.write_text(exchange_count, encoding="utf-8")
                pending_file.unlink()  # Clear the marker
        except Exception:
            pass  # Non-fatal -- save counter is best-effort

        return {
            "success": True,
            "entry_id": entry_id,
            "topic": topic,
            "content_type": content_type,
            "importance": importance,
            "timestamp": now.isoformat(),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


__all__ = [
    "tool_declare_operation",
    "tool_diary_write",
    "tool_kg_add",
    "tool_kg_add_batch",
    "tool_kg_declare_entity",
    "tool_kg_delete_entity",
    "tool_kg_invalidate",
    "tool_kg_merge_entities",
    "tool_kg_update_entity",
]
