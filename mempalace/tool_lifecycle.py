"""Intent-lifecycle mempalace tool handlers (lifecycle bucket).

This module re-exports the intent-lifecycle tool handlers from
``mempalace.mcp_server`` (which in turn delegate to ``mempalace.intent``)
so the PreToolUse carve-out hook can determine bucket membership by
reading ``__all__``. Handler bodies stay in ``mcp_server`` / ``intent`` --
moving them here would shuffle code for zero behavioural change.

The hook does NOT import this module at runtime (that would chain into the
heavy ``mcp_server`` import on every PreToolUse call). Instead, the hook
hardcodes the bucket basenames in ``hooks_cli._LIFECYCLE_BUCKET_BASENAMES``
and ``tests/test_hook_buckets.py::test_lifecycle_bucket_matches_module_all``
enforces the two stay in sync. If a handler is added or moves bucket,
update BOTH sides -- the drift-sentinel test breaks loudly otherwise.

Bucket semantics: lifecycle tools manage the intent state machine itself --
declaring, extending, finalizing intents and recording feedback. They
bypass the active-intent check entirely (otherwise ``declare_intent``
itself would be deadlocked). Under user-message preemption, only the
two true tier-0 carve-outs proceed:

  - ``mempalace_declare_user_intents`` -- the only path that clears the
    pending queue, so it MUST stay reachable.
  - ``mempalace_extend_feedback`` -- finishes a prior incomplete finalize
    so the agent isn't trapped between an unfinished intent and a new
    user message.

Every other lifecycle call (``declare_intent``, ``finalize_intent``,
``extend_intent``, ``resolve_conflicts``, ``active_intent``, ``wake_up``)
is blocked under preemption. AskUserQuestion remains the always-allowed
clarify path. See ``hooks_cli._USER_INTENT_TIER0_BASENAMES``.
"""

import json  # noqa: E402

# Phase 2: lifecycle bucket loaded BEFORE mcp_server.py finishes when
# test_hook_buckets.py imports `__all__` directly. Top-level imports
# from mempalace.mcp_server would trigger a circular at that path --
# mcp_server's import-back finds tool_lifecycle still mid-load.
# Each function imports its mcp_server deps lazily inside the body.


def tool_wake_up(agent: str = None):
    """Boot context for a session. Call ONCE at start.

    Returns protocol (behavioral rules), text (identity + top memories),
    and declared (compact summary of auto-declared entities).

    Args:
        agent: Agent identity -- MANDATORY. Used for affinity scoring in L1
            AND for cold-restart bootstrap (auto-creates the agent entity
            + is_a agent edge if missing, so subsequent write tools can run
            without hitting the chicken-and-egg deadlock that bites on a
            fresh palace). If omitted, falls back to reading the first
            non-blank token of ``~/.mempalace/identity.txt``; if neither is
            present, wake_up fails with a clear bootstrap instruction.
    """
    from mempalace.mcp_server import (
        PALACE_PROTOCOL,
        _STATE,
        _bootstrap_agent_if_missing,
        _hybrid_score_fn,
        _resolve_wake_up_agent,
        _telemetry_append_jsonl,
        intent,
    )

    try:
        from .layers import MemoryStack
    except Exception as e:
        return {"success": False, "error": f"layers module unavailable: {e}"}

    agent, err = _resolve_wake_up_agent(agent)
    if err is not None:
        return err

    try:
        _bootstrap_agent_if_missing(agent)

        stack = MemoryStack()
        text = stack.wake_up(agent=agent)
        from .knowledge_graph import normalize_entity_name

        # 1. Predicates -- declare + collect names
        predicates = _STATE.kg.list_entities(status="active", kind="predicate")
        pred_names = []
        for p in predicates:
            _STATE.declared_entities.add(p["id"])
            pred_names.append(p["id"])

        # 2. Classes -- declare + collect names
        classes = _STATE.kg.list_entities(status="active", kind="class")
        class_names = []
        for c in classes:
            _STATE.declared_entities.add(c["id"])
            class_names.append(c["id"])

        # 3. Intent types -- walk is-a tree, compact format
        #    Intent types are kind=class (they are types, not instances).
        #    Intent executions are kind=entity with is_a pointing to a class.
        entities = _STATE.kg.list_entities(status="active", kind="class")
        intent_type_ids = set()
        intent_parents = {}
        frontier = {"intent_type"}
        visited_walk = set()
        for _ in range(5):
            if not frontier:
                break
            next_frontier = set()
            for parent_id in frontier:
                if parent_id in visited_walk:
                    continue
                visited_walk.add(parent_id)
                for e in entities:
                    e_edges = _STATE.kg.query_entity(e["id"], direction="outgoing")
                    for edge in e_edges:
                        if edge["predicate"] == "is_a" and edge["current"]:
                            if normalize_entity_name(edge["object"]) == parent_id:
                                intent_type_ids.add(e["id"])
                                intent_parents[e["id"]] = parent_id
                                next_frontier.add(e["id"])
            frontier = next_frontier

        intent_entries = []
        for e in entities:
            if e["id"] in intent_type_ids:
                score = _hybrid_score_fn(
                    similarity=0.0,
                    importance=e.get("importance", 3),
                    date_iso=e.get("last_touched", ""),
                    agent_match=False,
                    last_relevant_iso=None,
                    relevance_feedback=0,
                    mode="l1",
                )
                intent_entries.append((score, e))
        intent_entries.sort(key=lambda x: x[0], reverse=True)

        # Format: top-level as name(Tool1,Tool2), children as name<parent(+AddedTool)
        intent_parts = []
        for _score, e in intent_entries[:20]:
            _STATE.declared_entities.add(e["id"])
            eid = e["id"]
            parent = intent_parents.get(eid, "?")
            _, tools = intent._resolve_intent_profile(eid)
            tool_names = sorted(set(t["tool"] for t in tools)) if tools else []
            if parent == "intent_type":
                intent_parts.append(eid + "(" + ",".join(tool_names) + ")" if tool_names else eid)
            else:
                own_props = e.get("properties", {})
                if isinstance(own_props, str):
                    try:
                        own_props = json.loads(own_props)
                    except Exception:
                        own_props = {}
                own_tools = own_props.get("rules_profile", {}).get("tool_permissions", [])
                own_names = sorted(set(t["tool"] for t in own_tools))
                if own_names:
                    intent_parts.append(eid + "<" + parent + "(+" + ",".join(own_names) + ")")
                else:
                    intent_parts.append(eid + "<" + parent)

        # 4. Top entities (non-intent) -- name[importance]
        entity_parts = []
        top_ents = [e for e in entities if e["id"] not in intent_type_ids][:20]
        for e in top_ents:
            _STATE.declared_entities.add(e["id"])
            entity_parts.append(e["id"] + "[" + str(e.get("importance", 3)) + "]")

        # Load learned scoring weights from feedback history. Two scopes:
        #   1. Hybrid score weights (sim / rel / imp / decay / agent) --
        #      learned from per-memory relevance correlations recorded
        #      at finalize_intent.
        #   2. Per-channel RRF weights (cosine / graph / keyword /
        #      context) -- learned from which channels surfaced memories
        #      that the agent later rated useful. Same mechanism, same
        #      table, different 'scope'.
        try:
            from .scoring import (
                set_learned_weights,
                set_learned_channel_weights,
                DEFAULT_SEARCH_WEIGHTS,
                DEFAULT_CHANNEL_WEIGHTS,
            )

            learned_hybrid = _STATE.kg.compute_learned_weights(
                DEFAULT_SEARCH_WEIGHTS, scope="hybrid"
            )
            set_learned_weights(learned_hybrid)
            learned_channels = _STATE.kg.compute_learned_weights(
                DEFAULT_CHANNEL_WEIGHTS, scope="channel"
            )
            set_learned_channel_weights(learned_channels)
            # Telemetry: observability for the weight-learning loop.
            # Writes one line to ~/.mempalace/hook_state/weight_log.jsonl
            # each time set_learned_* is invoked (wake_up + finalize_intent).
            # `is_tuned` tells you whether compute_learned_weights actually
            # drifted from the static defaults (requires
            # _A6_WEIGHT_SELFTUNE_ENABLED=True AND ≥ min_samples rows).
            try:
                from datetime import datetime as _dt, timezone as _tz

                _h_tuned = any(
                    abs(float(learned_hybrid.get(k, 0.0)) - float(DEFAULT_SEARCH_WEIGHTS[k])) > 1e-6
                    for k in DEFAULT_SEARCH_WEIGHTS
                )
                _c_tuned = any(
                    abs(float(learned_channels.get(k, 0.0)) - float(DEFAULT_CHANNEL_WEIGHTS[k]))
                    > 1e-6
                    for k in DEFAULT_CHANNEL_WEIGHTS
                )
                _fb_rows = {"hybrid": 0, "channel": 0}
                try:
                    _conn = _STATE.kg._conn()
                    _fb_rows["hybrid"] = int(
                        _conn.execute(
                            "SELECT COUNT(*) FROM scoring_weight_feedback "
                            "WHERE component NOT LIKE 'ch_%'"
                        ).fetchone()[0]
                    )
                    _fb_rows["channel"] = int(
                        _conn.execute(
                            "SELECT COUNT(*) FROM scoring_weight_feedback "
                            "WHERE component LIKE 'ch_%'"
                        ).fetchone()[0]
                    )
                except Exception:
                    pass
                _telemetry_append_jsonl(
                    "weight_log.jsonl",
                    {
                        "ts": _dt.now(_tz.utc).isoformat(timespec="seconds"),
                        "trigger": "wake_up",
                        "selftune_enabled": bool(
                            getattr(_STATE.kg, "_A6_WEIGHT_SELFTUNE_ENABLED", False)
                        ),
                        "feedback_rows": _fb_rows,
                        "hybrid": {
                            "learned": {k: round(float(v), 4) for k, v in learned_hybrid.items()},
                            "default": {
                                k: round(float(v), 4) for k, v in DEFAULT_SEARCH_WEIGHTS.items()
                            },
                            "is_tuned": _h_tuned,
                        },
                        "channel": {
                            "learned": {k: round(float(v), 4) for k, v in learned_channels.items()},
                            "default": {
                                k: round(float(v), 4) for k, v in DEFAULT_CHANNEL_WEIGHTS.items()
                            },
                            "is_tuned": _c_tuned,
                        },
                    },
                )
            except Exception:
                pass
        except Exception:
            pass

        declared = {
            "predicates": ", ".join(sorted(pred_names)),
            "classes": ", ".join(sorted(class_names)),
            "intent_types": " | ".join(intent_parts),
            "entities": ", ".join(entity_parts),
            "count": len(_STATE.declared_entities),
        }
        # Count the whole payload the caller receives -- not just `text`.
        # Rough 4-chars-per-token heuristic over text + protocol + declared.
        token_estimate = (len(text) + len(PALACE_PROTOCOL) + len(json.dumps(declared))) // 4
        return {
            "success": True,
            "protocol": PALACE_PROTOCOL,
            "text": text,
            "estimated_tokens": token_estimate,
            "declared": declared,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_declare_intent(*args, **kwargs):
    from mempalace.mcp_server import (
        intent,
    )

    return intent.tool_declare_intent(*args, **kwargs)


def tool_active_intent(*args, **kwargs):
    from mempalace.mcp_server import (
        intent,
    )

    return intent.tool_active_intent(*args, **kwargs)


def tool_extend_intent(*args, **kwargs):
    from mempalace.mcp_server import (
        intent,
    )

    return intent.tool_extend_intent(*args, **kwargs)


def tool_declare_user_intents(*args, **kwargs):
    from mempalace.mcp_server import (
        intent,
    )

    return intent.tool_declare_user_intents(*args, **kwargs)


def tool_resolve_conflicts(actions: list = None, agent: str = None):  # noqa: C901
    """Resolve pending conflicts -- contradictions, duplicates, or suggestions.

    Unified conflict resolution for ALL data types: edges, entities, memories.
    Each action specifies what to do with a conflict.

    Args:
        actions: List of {id, action, into?, merged_content?} dicts.
            id: The conflict ID (from the pending conflicts list).
            action: One of:
                "invalidate" -- mark existing item as no longer current (sets valid_to)
                "merge" -- combine items (must provide into + merged_content)
                "keep" -- both items are valid, no conflict
                "skip" -- don't add the new item (remove it)
            into: Target entity/memory ID to merge into (required for "merge")
            merged_content: Merged description/content (required for "merge")
        agent: mandatory, declared agent resolving these conflicts.
    """
    from mempalace.mcp_server import (
        _STATE,
        _load_pending_conflicts_from_disk,
        _require_agent,
        _require_sid,
        intent,
        tool_kg_merge_entities,
    )

    sid_err = _require_sid(action="resolve_conflicts")
    if sid_err:
        return sid_err
    agent_err = _require_agent(agent, action="resolve_conflicts")
    if agent_err:
        return agent_err

    # Some MCP transports stringify top-level array parameters. Parse once
    # up front rather than iterating a JSON string character-by-character
    # (which produces thousands of bogus per-character error entries).
    if isinstance(actions, str):
        try:
            actions = json.loads(actions)
        except Exception:
            return {
                "success": False,
                "error": (
                    "`actions` arrived as an unparseable string. Pass a JSON array "
                    "of {id, action, reason, ...} objects."
                ),
            }
    if actions is not None and not isinstance(actions, list):
        return {
            "success": False,
            "error": f"`actions` must be a list, got {type(actions).__name__}.",
        }

    # Disk is source of truth -- reload _STATE.pending_conflicts from the active
    # intent state file if memory is empty (MCP restart scenario).
    if not _STATE.pending_conflicts:
        _STATE.pending_conflicts = _load_pending_conflicts_from_disk()

    if not _STATE.pending_conflicts:
        try:
            intent._persist_active_intent()
        except Exception:
            pass
        return {"success": True, "message": "No pending conflicts."}

    if not actions:
        return {
            "success": False,
            "error": "Must provide actions list. Each conflict needs: {id, action}.",
            "pending": _STATE.pending_conflicts,
        }

    # Index pending conflicts by ID -- defensively coerce if any entries are
    # JSON strings (some MCP transports serialize nested objects)
    _normalized_conflicts = []
    for c in _STATE.pending_conflicts:
        if isinstance(c, str):
            try:
                c = json.loads(c)
            except Exception:
                continue
        if isinstance(c, dict) and c.get("id"):
            _normalized_conflicts.append(c)
    conflict_map = {c["id"]: c for c in _normalized_conflicts}
    resolved_ids = set()
    results = []

    # Normalize actions too -- tolerate string-encoded dicts from some transports
    normalized_actions = []
    for act in actions:
        if isinstance(act, str):
            try:
                act = json.loads(act)
            except Exception:
                results.append(
                    {"id": "?", "status": "error", "reason": f"Unparseable action: {act!r}"}
                )
                continue
        if not isinstance(act, dict):
            results.append(
                {
                    "id": "?",
                    "status": "error",
                    "reason": f"Action must be an object, got {type(act).__name__}",
                }
            )
            continue
        normalized_actions.append(act)

    # ── Validate reason field on all actions + laziness detection ──
    MIN_REASON_LENGTH = 15
    for act in normalized_actions:
        reason = (act.get("reason") or "").strip()
        if len(reason) < MIN_REASON_LENGTH:
            return {
                "success": False,
                "error": (
                    f"Mandatory 'reason' field missing or too short on conflict '{act.get('id', '?')}'. "
                    f"Each conflict resolution requires a reason (minimum {MIN_REASON_LENGTH} characters) "
                    f"explaining WHY you chose this action. This is a real semantic decision -- "
                    f"evaluate each conflict individually."
                ),
            }

    # Laziness detection: reject if 3+ actions share identical reason text
    reason_counts: dict = {}
    for act in normalized_actions:
        r = (act.get("reason") or "").strip()
        reason_counts[r] = reason_counts.get(r, 0) + 1
    for r, count in reason_counts.items():
        if count >= 3:
            return {
                "success": False,
                "error": (
                    f"Laziness detected: {count} conflicts share identical reason '{r[:50]}...'. "
                    f"Each conflict is a unique semantic decision -- evaluate individually and "
                    f"provide a specific reason for each. Bulk-processing is not allowed."
                ),
            }

    for act in normalized_actions:
        cid = act.get("id", "")
        action = act.get("action", "")

        if cid not in conflict_map:
            results.append({"id": cid, "status": "error", "reason": f"Unknown conflict ID: {cid}"})
            continue

        conflict = conflict_map[cid]
        conflict_type = conflict.get("conflict_type", "unknown")
        existing_id = conflict.get("existing_id", "")
        new_id = conflict.get("new_id", "")

        try:
            if action == "invalidate":
                # Mark existing item as no longer current
                if conflict_type == "edge_contradiction":
                    # Invalidate the existing edge by setting valid_to
                    _STATE.kg.invalidate(
                        conflict["existing_subject"],
                        conflict["existing_predicate"],
                        conflict["existing_object"],
                    )
                elif conflict_type in ("entity_duplicate", "memory_duplicate"):
                    # Mark entity/memory as merged-out
                    try:
                        conn = _STATE.kg._conn()
                        conn.execute(
                            "UPDATE entities SET status='invalidated' WHERE id=?",
                            (existing_id,),
                        )
                        conn.commit()
                    except Exception:
                        pass
                results.append({"id": cid, "status": "invalidated", "target": existing_id})

            elif action == "merge":
                into = act.get("into", "")
                merged_content = act.get("merged_content", "")
                if not into:
                    results.append(
                        {"id": cid, "status": "error", "reason": "merge requires 'into' field"}
                    )
                    continue
                if not merged_content:
                    results.append(
                        {
                            "id": cid,
                            "status": "error",
                            "reason": "merge requires 'merged_content' -- read BOTH items in full, then provide combined content",
                        }
                    )
                    continue

                # Determine source (the one NOT being merged into)
                source = new_id if into == existing_id else existing_id

                if conflict_type in ("entity_duplicate", "memory_duplicate"):
                    # Use existing kg_merge_entities for the plumbing.
                    # Wrap user's merged_content prose into the dict-only
                    # summary contract (Adrian's design lock 2026-04-25).
                    merge_result = tool_kg_merge_entities(
                        source=source,
                        target=into,
                        summary={
                            "what": into,
                            "why": merged_content,
                        },
                        agent=agent,
                    )
                    if merge_result.get("success"):
                        results.append({"id": cid, "status": "merged", "into": into})
                    else:
                        results.append(
                            {
                                "id": cid,
                                "status": "error",
                                "reason": str(merge_result.get("error", "")),
                            }
                        )
                else:
                    results.append(
                        {
                            "id": cid,
                            "status": "error",
                            "reason": f"merge not supported for {conflict_type}",
                        }
                    )

            elif action == "keep":
                # Both items are valid -- no action needed
                results.append({"id": cid, "status": "kept"})

            elif action == "skip":
                # Don't add the new item -- remove it if already added
                if conflict_type == "edge_contradiction":
                    try:
                        _STATE.kg.invalidate(
                            conflict.get("new_subject", ""),
                            conflict.get("new_predicate", ""),
                            conflict.get("new_object", ""),
                        )
                    except Exception:
                        pass
                results.append({"id": cid, "status": "skipped"})

            else:
                results.append(
                    {"id": cid, "status": "error", "reason": f"Unknown action: {action}"}
                )
                continue

            # Persist the resolution so future audits + feedback loops can
            # learn from the decision instead of throwing the reason away.
            _intent_type = (
                _STATE.active_intent.get("intent_type", "") if _STATE.active_intent else ""
            )
            try:
                _STATE.kg.record_conflict_resolution(
                    conflict_id=cid,
                    conflict_type=conflict_type,
                    action=action,
                    reason=(act.get("reason") or "").strip(),
                    existing_id=existing_id,
                    new_id=new_id,
                    agent=agent,
                    intent_type=_intent_type,
                )
            except Exception:
                pass

            # (retired P3) Edge-contradiction resolutions used to emit a
            # negative signal on the losing edge via edge_traversal_feedback
            # (invalidate → loser = existing triple; skip → loser = new
            # triple). That signal now flows through rated_irrelevant
            # edges on the active context at finalize_intent time.

            resolved_ids.add(cid)
        except Exception as e:
            results.append({"id": cid, "status": "error", "reason": str(e)})

    # Check all conflicts are resolved
    unresolved = set(conflict_map.keys()) - resolved_ids
    errors = [r for r in results if r.get("status") == "error"]
    if unresolved:
        return {
            "success": False,
            "error": f"{len(unresolved)} conflicts not addressed. Provide action for each.",
            "unresolved_ids": sorted(unresolved),
            "errors": errors,
        }

    # Clear pending conflicts and persist state
    _STATE.pending_conflicts = None
    try:
        intent._persist_active_intent()
    except Exception:
        pass
    # Caller supplied the ids/actions/reasons -- echoing them back is pure
    # token waste. Return only the count on full success; surface errors
    # individually if any.
    response = {"success": True, "count": len(resolved_ids)}
    if errors:
        response["errors"] = errors
    return response


def tool_finalize_intent(*args, **kwargs):
    from mempalace.mcp_server import (
        intent,
    )

    return intent.tool_finalize_intent(*args, **kwargs)


def tool_extend_feedback(*args, **kwargs):
    from mempalace.mcp_server import (
        intent,
    )

    return intent.tool_extend_feedback(*args, **kwargs)


__all__ = [
    "tool_active_intent",
    "tool_declare_intent",
    "tool_declare_user_intents",
    "tool_extend_feedback",
    "tool_extend_intent",
    "tool_finalize_intent",
    "tool_resolve_conflicts",
    "tool_wake_up",
]
