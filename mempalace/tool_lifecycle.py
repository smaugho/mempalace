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
declaring, extending, finalizing intents. They bypass the active-intent
check entirely (otherwise ``declare_intent`` itself would be deadlocked).
Under user-message preemption, only the single true tier-0 carve-out
proceeds:

  - ``mempalace_declare_user_intents`` -- the only path that clears the
    pending queue, so it MUST stay reachable.

Every other lifecycle call (``declare_intent``, ``finalize_intent``,
``extend_intent``, ``active_intent``, ``wake_up``) is blocked under
preemption. AskUserQuestion remains the always-allowed clarify path.
See ``hooks_cli._USER_INTENT_TIER0_BASENAMES``.
"""

import json  # noqa: E402

# Phase 2: lifecycle bucket loaded BEFORE mcp_server.py finishes when
# test_hook_buckets.py imports `__all__` directly. Top-level imports
# from mempalace.mcp_server would trigger a circular at that path --
# mcp_server's import-back finds tool_lifecycle still mid-load.
# Each function imports its mcp_server deps lazily inside the body.


def _scrub_declared_for_state(declared: dict) -> dict:
    """Drop state-subsystem names from the wake_up `declared` block when
    the state keeper is off (Adrian directive 2026-05-28).

    Filters the state_schema class, state predicates (state_changed_by),
    state intent types, and any state-named entity out of the boot
    ontology listing so the agent never learns state exists from it.
    No-op when the keeper is enabled. Pure function -- unit-tested.
    """
    from mempalace.state_schemas import mentions_state, state_keeper_enabled

    if state_keeper_enabled():
        return declared
    out = dict(declared)
    for _key, _sep in (
        ("predicates", ", "),
        ("classes", ", "),
        ("intent_types", " | "),
        ("entities", ", "),
    ):
        _val = out.get(_key) or ""
        if _val:
            out[_key] = _sep.join(p for p in _val.split(_sep) if not mentions_state(p))
    return out


def _derive_subagent_identity_context(agent: str, task_id: str):
    """For a SUB-AGENT first-wake with a resolving task_id but no caller
    context, synthesize an identity Context dict FROM the Task entity.

    Sub-agents spawn with throwaway agent names (e.g. 'impl_347') and cannot
    author their own identity, so the cold-start lock would hard-fail their
    first wake_up. We derive {queries, keywords, entities, summary} from the
    Task -- per-task discriminative (NOT a generic template), so the lock's
    anti-collision rationale still holds. Returns the context dict, or None
    when this is not a sub-agent / the task_id does not resolve (caller then
    falls through to the normal cold-start contract). Best-effort: any error
    returns None.

    Adrian directive 2026-05-30 (impl_347 transcript diagnosis).
    """
    from mempalace.mcp_server import _STATE

    try:
        sid = _STATE.session_id or ""
    except Exception:
        return None
    if "__sub_" in sid:
        try:
            task = _STATE.kg.get_entity(str(task_id).strip())
        except Exception:
            return None
        if not task:
            return None
        tsum = task.get("summary") or task.get("content") or ""
        tsum = str(tsum).replace("\n", " ").strip()
        why = (
            f"sub-agent spawned to carry out {task_id}: {tsum}"
            if tsum
            else f"sub-agent spawned to carry out {task_id}"
        )
        # why is embedded in a <=280-char rendered summary; cap well under it.
        why = why[:180]
        return {
            "queries": [
                f"who is the {agent} sub-agent",
                f"{agent} sub-agent for {task_id}",
                f"what work does {agent} do",
            ],
            "keywords": [agent, "sub-agent", str(task_id)[:60]],
            "entities": [],
            "summary": {
                "what": f"{agent} (sub-agent for {task_id})",
                "why": why,
                "scope": "ephemeral sub-agent; one task",
            },
        }
    return None


def tool_wake_up(agent: str = None, context: dict = None, task_id: str = None):  # noqa: C901
    """Boot context for a session. Call ONCE at start.

    Returns protocol (behavioral rules), text (identity + top memories),
    and declared (compact summary of auto-declared entities). For a
    sub-agent, pass ``task_id`` (the task_<slug> your parent put in your
    spawn prompt) to also receive ``result['task']`` -- the Task entity's
    summary + content, i.e. your actual mission -- so you can act without a
    separate kg_query. The same task_<slug> is your cause_id on the first
    declare_intent.

    Args:
        agent: Agent identity -- MANDATORY. Used for affinity scoring in L1
            AND for cold-restart bootstrap (auto-creates the agent entity
            + is_a agent edge if missing, so subsequent write tools can run
            without hitting the chicken-and-egg deadlock that bites on a
            fresh palace). If omitted, falls back to reading the first
            non-blank token of ``~/.mempalace/identity.txt``; if neither is
            present, wake_up fails with a clear bootstrap instruction.
        context: REQUIRED on the FIRST wake_up of a given agent name on
            this palace (cold-start lock 2026-05-01, no back-compat).
            Idempotent on subsequent wake_ups -- the agent's identity is
            already in the KG so context is ignored. Required shape::

                {
                    "summary": {"what": <str>, "why": <str>, "scope": <str>?},
                    "queries":  [<probe>, ...],   # optional, recommended
                    "keywords": [<keyword>, ...], # optional
                }

            The summary must be a real {what, why, scope?} dict that
            discriminates THIS agent from others. Generic templates that
            differ only in the agent name produce ~0.95 cosine on the
            gate's identity layer and silently false-reuse across agents.
    """
    from mempalace.mcp_server import (
        _STATE,
        _bootstrap_agent_if_missing,
        _hybrid_score_fn,
        _resolve_wake_up_agent,
        _telemetry_append_jsonl,
        build_protocol,
        intent,
    )
    from mempalace.state_schemas import state_keeper_enabled

    try:
        from .layers import MemoryStack
    except Exception as e:
        return {"success": False, "error": f"layers module unavailable: {e}"}

    agent, err = _resolve_wake_up_agent(agent)
    if err is not None:
        return err

    # Sub-agent cold-start onboarding (Adrian directive 2026-05-30, from the
    # impl_347 transcript diagnosis): a sub-agent is spawned with a fresh,
    # throwaway agent name (e.g. 'impl_347') and a spawn prompt that just
    # points it at wake_up + a task_id. It has no good way to author its own
    # identity context, so the cold-start lock (which requires queries +
    # keywords + summary for any new agent name) would hard-fail its FIRST
    # wake_up and leave it un-woken / gate-blocked for the whole run. When
    # the caller is a sub-agent (session carries '__sub_'), passed a task_id
    # that RESOLVES to a Task, and supplied no usable context, derive the
    # identity FROM the Task entity. This is per-task discriminative (NOT a
    # generic template), so it honours the lock's anti-collision rationale
    # while letting the sub-agent onboard. Best-effort: any failure falls
    # through to the normal cold-start contract (which raises a clear error).
    if task_id and not isinstance(context, dict):
        _derived_ctx = _derive_subagent_identity_context(agent, task_id)
        if _derived_ctx is not None:
            context = _derived_ctx

    # AgentBootstrapContextRequired surfaces a structured error rather
    # than crashing wake_up; the message tells the agent how to retry
    # with a real context dict on a fresh palace.
    from mempalace.mcp_server import AgentBootstrapContextRequired

    try:
        _bootstrap_agent_if_missing(agent, context=context)
    except AgentBootstrapContextRequired as exc:
        return {
            "success": False,
            "error": str(exc),
            "error_kind": "agent_bootstrap_context_required",
            "agent": agent,
        }

    # ── agent_state eager-init at wake_up ──────────────
    # State-protocol v3 (Adrian directive 2026-05-04). The agent is an
    # implicit member of the active set every session -- agent_state
    # should exist from wake_up, not be retrofitted lazily by the
    # gardener. We seed a minimal default revision (current_focus="")
    # iff no existing agent_state revision exists for this session.
    # Phase D scope-aware reads (slice 6) ensure each session sees its
    # own seed even when prior sessions wrote to the same agent id;
    # the read-then-seed sequence is therefore safe across sessions.
    # Failures are silent -- wake_up must remain robust against state-
    # substrate issues; the gardener retrofit path stays as fallback.
    try:
        from .state_schemas import materialize_default as _materialize

        _existing_agent_state = _STATE.kg.latest_state_for_entity(
            agent, session_id=_STATE.session_id or None
        )
        if _existing_agent_state is None:
            _STATE.kg.record_state_revision(
                entity_id=agent,
                schema_id="agent_state",
                payload=_materialize("agent_state"),
                op_context_id="",
                agent=agent,
                session_id=_STATE.session_id or None,
            )
    except Exception:
        pass

    try:
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

        # 4. Operation classes -- tool -> slots map
        # follow-up (Adrian directive 2026-05-06: "important
        # to return the operations slots shape on the wake up! as
        # otherwise the agents won't know"). Surfacing the slot shape
        # for each registered operation_class lets agents fill slots
        # correctly on the FIRST declare_operation, instead of hitting
        # slot-validation errors and discovering the shape via
        # trial-and-error. Compact dict {tool: {slot_name: {classes,
        # required, multiple}, ...}, ...} -- absent tools have no
        # registered class (no slot constraints).
        operation_classes_map: dict = {}
        for e in entities:
            try:
                e_edges = _STATE.kg.query_entity(e["id"], direction="outgoing")
            except Exception:
                continue
            is_op_class = any(
                edge["predicate"] == "is_a"
                and edge["current"]
                and normalize_entity_name(edge["object"]) == "operation"
                for edge in e_edges
            )
            if not is_op_class:
                continue
            try:
                full = _STATE.kg.get_entity(e["id"])
            except Exception:
                full = None
            if not full:
                continue
            props = full.get("properties", {}) or {}
            if isinstance(props, str):
                try:
                    props = json.loads(props)
                except Exception:
                    props = {}
            profile = props.get("rules_profile", {}) or {}
            tool_name = profile.get("tool")
            if not tool_name:
                continue
            slots = profile.get("slots", {}) or {}
            operation_classes_map[tool_name] = slots
            _STATE.declared_entities.add(e["id"])

        # 5. Top entities (non-intent, non-op-class) -- name[importance]
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
            "operations": operation_classes_map,
            "entities": ", ".join(entity_parts),
            "count": len(_STATE.declared_entities),
        }
        # State keeper master switch (Adrian directive 2026-05-28): strip
        # the state subsystem from the boot ontology listing when off.
        declared = _scrub_declared_for_state(declared)

        # State-protocol (Adrian 2026-05-04): return the full
        # state-schema registry at boot so agents have the shapes in
        # hand at the moment they must author initial_state on
        # declare_intent / kg_declare_entity / kg_add(is_a). The MCP
        # json schema cannot hardcode these (Phase 6 supports
        # agent-authored schemas; the set is open-ended), so the
        # canonical surface is the wake_up response. Mirrors the
        # schemas{} block already shipped on declare_user_intents +
        # declare_operation -- those are scoped to surfaced memories
        # (lean), wake_up returns the full registry (catalog).
        # State keeper master switch (Adrian directive 2026-05-28): the
        # schema registry is only surfaced when the state keeper is ON.
        # When OFF (default) wake_up returns no schemas block at all.
        schemas = {}
        if state_keeper_enabled():
            try:
                from .state_schemas import STATE_SCHEMAS as _SS

                schemas = {sid: dict(sdef) for sid, sdef in _SS.items()}
            except Exception:
                schemas = {}

        # Count the whole payload the caller receives -- not just `text`.
        # Rough 4-chars-per-token heuristic over text + protocol +
        # declared + schemas.
        _protocol_text = build_protocol()
        token_estimate = (
            len(text) + len(_protocol_text) + len(json.dumps(declared)) + len(json.dumps(schemas))
        ) // 4
        result = {
            "success": True,
            "protocol": _protocol_text,
            "text": text,
            "estimated_tokens": token_estimate,
            "declared": declared,
        }
        if schemas:
            result["schemas"] = schemas

        # v3.7.16 parent-side spawn protocol (Adrian directive 2026-05-17):
        # always surface the sub-agent spawn contract so the parent
        # agent learns it BEFORE spawning, not after a rejection on
        # the sub-agent side. Key rule: the Task entity holds the
        # actual task content; the spawn prompt is minimal -- just the
        # task_id line plus "read your task from this id". Every
        # session sees this because any agent can become a parent.
        result["sub_agent_spawn_protocol"] = (
            "WHEN YOU SPAWN A SUB-AGENT via the Task tool, the parent "
            "(you) MUST follow this contract so the sub-agent's "
            "declare_intent does not get rejected:\n"
            "  1. Declare a Task entity that holds the ACTUAL work:\n"
            "     mempalace_kg_declare_entity(\n"
            "       kind='entity', is_a='Task',\n"
            "       name='task_<descriptive_slug>',\n"
            "       added_by='<your_agent>', importance=4,\n"
            "       context={ queries, keywords, entities,\n"
            "                 summary={what, why, scope?} -- the task's\n"
            "                 actual goal + scope + acceptance criteria\n"
            "                 go HERE, not in the spawn prompt })\n"
            "     This SINGLE call writes the is_a Task edge ATOMICALLY "
            "(is_a is a kg_declare_entity param -- no separate kg_add). "
            "CONFIRM it returns success (the entity now resolves + is_a "
            "Task) BEFORE you spawn. NEVER dispatch a sub-agent whose "
            "task_id does not yet exist -- a sub-agent pointed at a "
            "missing Task wastes its entire run. Pass the task_id EXACTLY "
            "ONCE, only after the Task is created.\n"
            "  2. The Task entity's summary + content carries the task. "
            "The sub-agent will retrieve it via mempalace_kg_query when "
            "it boots. Do NOT cram instructions into the spawn prompt; "
            "put them in the Task entity instead so they are durable + "
            "queryable + can be updated mid-task without re-spawning.\n"
            "  3. Spawn-prompt body should be MINIMAL -- the task_id line "
            "plus a one-line instruction to call mempalace_wake_up FIRST. "
            "Example: 'task_id=task_<slug>\\n\\nCall mempalace_wake_up "
            "first (it returns your task), then follow the protocol it "
            "gives you.'\n"
            "  4. Do NOT spell out tool PARAMETERS in the spawn prompt. "
            "Name the tool (mempalace_wake_up) and stop there -- do NOT "
            "write 'wake_up(agent=..., task_id=...)' or any param list. "
            "The sub-agent fetches the tool's own schema (via ToolSearch) "
            "and fills the params itself. Hardcoding params in the prompt "
            "propagates mistakes: a prompt that spelled out an INCOMPLETE "
            "param list (missing the required first-wake identity context) "
            "is exactly what made a past sub-agent's first wake_up fail and "
            "thrash for its whole run. State WHAT to call, never HOW.\n"
            "  5. Tell the sub-agent, in the prompt, that it MUST call "
            "mempalace_wake_up FIRST -- before ANY other tool call -- and "
            "must NOT issue any other tool call, PARALLEL OR SEQUENTIAL, "
            "until wake_up returns. (If a gate-blocked call is bundled in a "
            "parallel batch, the harness cancels EVERY sibling in that "
            "batch; onboarding must be strictly serial or the sub-agent "
            "burns its whole run on cancelled calls.)\n"
            "  6. The sub-agent passes that 'task_<slug>' string as "
            "cause_id in its first mempalace_declare_intent call. "
            "Without it, declare_intent rejects with "
            "subagent_non_task_cause_rejected (any non-Task cause). "
            "(cause_id='autonomous' was removed 2026-05-30 -- it is "
            "rejected for everyone; there is no parentless action.)\n"
            "Why: causal attribution chains user_message -> parent "
            "intent -> Task entity -> sub-agent intent. The Task is the "
            "durable handoff anchor; the spawn prompt is just the "
            "pointer to it."
        )

        # v3.7.5 sub-agent task_id sidecar (Adrian directive 2026-05-16):
        # when this session is a sub-agent (session_id carries the
        # '__sub_' suffix minted by _effective_session_id), inject a
        # directive telling the agent how to parse the parent-supplied
        # task_id from its first user message and pass it as cause_id
        # on the first declare_intent. The v3.6.0+v3.6.1 server-side
        # gates already reject any non-Task cause_id from sub-agents;
        # this sidecar surfaces the WHY + the parse contract
        # proactively so sub-agents do not have to learn it the hard
        # way through a rejection on their first declare_intent.
        try:
            _sid_for_subagent_hint = _STATE.session_id or ""
        except Exception:
            _sid_for_subagent_hint = ""
        if "__sub_" in _sid_for_subagent_hint:
            result["sub_agent_protocol"] = (
                "SUB-AGENT DETECTED. Your session_id carries the "
                "'__sub_' suffix, which means a parent agent spawned "
                "you via the Task tool.\n"
                "SERIAL ONBOARDING (critical): mempalace_wake_up MUST be "
                "your FIRST tool call, on its own. Do NOT bundle any other "
                "tool call -- PARALLEL OR SEQUENTIAL -- with your onboarding "
                "(wake_up, then declare_intent). If a gate-blocked call is "
                "in a parallel batch, the harness CANCELS every sibling in "
                "that batch, so a single mis-ordered batch can waste your "
                "entire run on 'Cancelled: parallel tool call' errors. Do "
                "onboarding one call at a time; only batch real work AFTER "
                "your intent is active.\n"
                "Before you call mempalace_declare_intent, you MUST:\n"
                "  1. Read your first user message. The parent should "
                "have prefixed it with 'task_id=task_<descriptive_slug>' "
                "on its own line (typically the first line).\n"
                "  2. Pass that 'task_<slug>' string as the cause_id "
                "argument on your FIRST mempalace_declare_intent call. "
                "The slug must resolve to an entity with kind='entity' "
                "is_a Task in the KG -- the parent should have declared "
                "it via mempalace_kg_declare_entity before dispatching "
                "you.\n"
                "  3. If the prompt does NOT carry a task_id line, "
                "ABORT and surface the missing task_id back to the "
                "user. Do NOT pass the parent's user-context ctx_id "
                "(v3.6.1 rejects it for sub-agents). Note cause_id="
                "'autonomous' no longer exists (removed 2026-05-30 -- "
                "there is no parentless action). Only a Task entity id "
                "is accepted. (Note: your parent's spawn is itself "
                "blocked unless it includes a task_id, so a missing "
                "task_id should be rare -- if it happens, the parent "
                "skipped the contract.)\n"
                "  4. Pass that same task_<slug> to mempalace_wake_up("
                "task_id=...) -- a second wake_up call is fine -- to get "
                "your Task's full content (your mission) back in "
                "result['task'], so you don't have to kg_query it "
                "separately.\n"
                "Why: causal attribution must chain user_message -> "
                "parent intent -> Task -> sub-agent intent. Without "
                "the Task anchor, your work floats free of the user "
                "message that triggered the whole flow."
            )

        # task_id surfacing (Adrian directive 2026-05-30): when a caller
        # passes task_id (the task_<slug> the parent put in the spawn
        # prompt), fetch that Task entity and return its summary + content
        # as result['task'] so a sub-agent sees its actual mission at boot
        # without a separate kg_query. If it does not resolve, surface a
        # clear note rather than silently omitting -- an unresolved task_id
        # means the parent failed the spawn contract. Best-effort: any KG
        # error must not fail wake_up itself.
        if task_id:
            try:
                _tid = str(task_id).strip()
                _task_entity = _STATE.kg.get_entity(_tid)
                if _task_entity:
                    result["task"] = {
                        "id": _task_entity.get("id", _tid),
                        "name": _task_entity.get("name", _tid),
                        "summary": _task_entity.get("summary") or _task_entity.get("content", ""),
                        "content": _task_entity.get("content", ""),
                        "importance": _task_entity.get("importance"),
                    }
                else:
                    result["task_error"] = (
                        f"task_id '{_tid}' did not resolve to any entity. The "
                        "parent must mempalace_kg_declare_entity(kind='entity', "
                        "is_a='Task', name='"
                        f"{_tid}', ...) -- describing the work in its summary/"
                        "content -- BEFORE spawning you. Surface this back to "
                        "the parent; do not invent a substitute cause."
                    )
            except Exception as _task_exc:  # pragma: no cover - defensive
                result["task_error"] = f"could not load task_id '{task_id}': {_task_exc}"
        # v3.7.20 (Adrian directive 2026-05-17): pending_conflicts wake_up
        # surfacing removed. With the agent no longer resolving conflicts,
        # there's nothing to enumerate -- the bg Haiku resolver in
        # mempalace/conflict_resolver_auto.py owns invalidate/merge/keep/
        # skip; mempalace_bg_status surfaces the audit trail via
        # conflict_resolver_log.jsonl for operators who want to see what
        # Haiku decided.
        #
        # Self-onboarding marker (2026-05-30, Adrian directive): record that
        # this session has called wake_up. The PreToolUse gate reads this so
        # that an agent which hits the no-active-intent block WITHOUT having
        # woken up first (notably sub-agents -- they spawn in a fresh context
        # and never receive ~/.claude/CLAUDE.md, which is the only thing that
        # tells an agent to call wake_up) is pointed at mempalace_wake_up
        # FIRST. wake_up is what teaches the protocol + the sub-agent
        # Task-cause contract, so mempalace self-onboards regardless of
        # CLAUDE.md. Best-effort; a marker-write failure must never fail
        # wake_up itself.
        try:
            import json as _json_wm
            from datetime import datetime as _dt_wm

            from .mcp_server import _INTENT_STATE_DIR as _isd_wm
            from .mcp_server import _sanitize_session_id as _san_wm

            # Sanitize identically to the PreToolUse gate reader
            # (hooks_cli._wake_up_seen). The reader strips to
            # [a-zA-Z0-9_-]; if we wrote the raw sid the filenames could
            # diverge and the gate would never see the marker.
            _sid_wm = _san_wm(_STATE.session_id or "")
            if _sid_wm:
                _wm_path = _isd_wm / f"wake_up_seen_{_sid_wm}.json"
                _wm_path.parent.mkdir(parents=True, exist_ok=True)
                _wm_path.write_text(
                    _json_wm.dumps({"agent": agent or "", "ts": _dt_wm.now().isoformat()}),
                    encoding="utf-8",
                )
        except Exception:
            pass
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


# tool_list_pending_conflicts removed in v3.7.20 (Adrian directive
# 2026-05-17). With resolve_conflicts gone, there is no agent-facing
# reason to enumerate pending_conflicts -- the bg Haiku resolver
# (conflict_resolver_auto) consumes them automatically and
# mempalace_bg_status surfaces the audit trail via
# conflict_resolver_log.jsonl.


def tool_challenge_state_change(
    rev_id: str,
    justification: str,
    restore_prior: bool = True,
    agent: str = "",
):
    """File an agent challenge against a state-keeper-auto-applied state revision.

    Adrian directive 2026-05-13. The state keeper (background
    state-change detector, formerly 'state_judge') auto-writes patches
    UNCONDITIONALLY (v3.10.3 removed the v2_visibility/v0_strict env
    flag -- it never blocks), attributed as agent='state_judge' on
    mempalace_state_revisions (attribution literal kept stable for
    audit history). This tool closes the deferred-write protocol by
    giving the agent an explicit override path with a JTMS retraction
    trail.

    Two modes:
      - restore_prior=True (default): writes a NEW state_revision
        restoring the entity's state to the revision PRECEDING rev_id
        (via the indexed entity_id/created_at scan), attributed to the
        challenging agent. The challenge row's retracted_rev_id points
        at this new revision so the audit trail reads cleanly.
      - restore_prior=False: info-only challenge. The state keeper's
        write stands; only the challenge row + state_challenged_by edge
        survive. Useful when the agent wants to flag a disputed write
        without rolling it back (e.g. the state keeper was right but for
        the wrong reason).

    Returns:
        {"success": True, "challenge_id": str,
         "retracted_rev_id": str | None,
         "restored_rev_id": str | None}
        or {"success": False, "error": str}.
    """
    try:
        from mempalace import mcp_server as _mcp_mod

        _STATE = _mcp_mod._STATE
        kg = _STATE.kg
        if kg is None:
            return {
                "success": False,
                "error": "challenge_state_change: KG unavailable",
            }

        # Resolve the agent + op context. challenge_op_id is the
        # current active context if available; empty string is OK
        # (the table allows '' default + the JTMS edge stays soft).
        _agent = (agent or "").strip()
        if not _agent:
            return {
                "success": False,
                "error": (
                    "challenge_state_change: agent is required "
                    "(non-empty); challenges must attribute to a "
                    "known agent for trust/accuracy telemetry."
                ),
            }
        active = getattr(_STATE, "active_intent", None) or {}
        challenge_op_id = (active.get("active_context_id") or "").strip()

        # Look up the revision being challenged so we can (a) reject
        # missing rev_ids cleanly and (b) compute the restore target.
        conn = kg._conn()
        row = conn.execute(
            "SELECT entity_id, schema_id, created_at, agent "
            "FROM mempalace_state_revisions WHERE rev_id = ?",
            (rev_id,),
        ).fetchone()
        if row is None:
            return {
                "success": False,
                "error": (
                    f"challenge_state_change: rev_id '{rev_id}' not "
                    "found in mempalace_state_revisions."
                ),
            }
        target_entity_id, target_schema_id, target_created_at, target_agent = row

        restored_rev_id: str | None = None
        if restore_prior:
            # Find the most recent revision for this entity that
            # PRECEDED rev_id by created_at. If none, the prior state
            # is "empty" -- write {} as the restore payload.
            prior = conn.execute(
                "SELECT payload FROM mempalace_state_revisions "
                "WHERE entity_id = ? AND created_at < ? "
                "ORDER BY created_at DESC LIMIT 1",
                (target_entity_id, target_created_at),
            ).fetchone()
            if prior is None:
                prior_payload: dict = {}
            else:
                import json as _json

                try:
                    prior_payload = _json.loads(prior[0]) or {}
                except Exception:
                    prior_payload = {}
            # Write the restore revision with the challenging agent's
            # attribution so the audit trail shows WHO retracted.
            restored_rev_id = kg.record_state_revision(
                target_entity_id,
                target_schema_id,
                prior_payload,
                op_context_id=challenge_op_id,
                agent=_agent,
            )

        challenge_id = kg.record_state_revision_challenge(
            rev_id=rev_id,
            challenge_op_id=challenge_op_id,
            agent=_agent,
            justification=justification,
            retracted_rev_id=restored_rev_id,
        )
        return {
            "success": True,
            "challenge_id": challenge_id,
            "retracted_rev_id": restored_rev_id,
            "restored_rev_id": restored_rev_id,
            "challenged_rev": {
                "rev_id": rev_id,
                "entity_id": target_entity_id,
                "schema_id": target_schema_id,
                "applied_by_agent": target_agent,
            },
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


# tool_resolve_conflicts removed in v3.7.20 (Adrian directive 2026-05-17).
# Conflicts are now resolved by Haiku in the background -- see
# mempalace/conflict_resolver_auto.py. The bg resolver mirrors the four
# action verbs (invalidate / merge / keep / skip) the old handler used
# and persists via the same kg primitives (kg.invalidate,
# tool_kg_merge_entities, record_conflict_resolution). No agent-facing
# resolution path remains; mempalace_bg_status surfaces the audit trail.


def tool_finalize_intent(*args, **kwargs):
    from mempalace.mcp_server import (
        intent,
    )

    return intent.tool_finalize_intent(*args, **kwargs)


__all__ = [
    "tool_active_intent",
    "tool_declare_intent",
    "tool_declare_user_intents",
    "tool_extend_intent",
    "tool_finalize_intent",
    "tool_wake_up",
]
