#!/usr/bin/env python3
"""
mempalace/intent.py -- Intent declaration, active-intent tracking, and finalization.

Extracted from mcp_server.py. Uses a module-reference pattern to access
mcp_server globals without circular imports.
"""

import json
import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

from .knowledge_graph import normalize_entity_name

# Module reference (set by init())
_mcp = None


# ── Debug-return overlay (on by default; set env var to "0" to suppress) ──
# DEBUG_RETURN_SCORES: attach the fused retrieval score (``hybrid_score``,
#   the post-RRF fused score that ranks the returned memories) to every
#   item in the ``memories`` list of declare_intent / declare_operation /
#   kg_search. Debug-only; callers that serialize the payload should
#   treat this field as optional.
# DEBUG_RETURN_CONTEXT: attach a top-level ``context: {id, queries}``
#   block to declare_intent / declare_operation / kg_search responses so
#   callers can see which context entity minted/reused for the call and
#   the exact queries that seeded retrieval.
# MEMORY_PREVIEW_MAX_CHARS: safety cap applied to every per-memory preview
#   returned in the three tools above. Summary-first records fit easily;
#   legacy records without the summary\n\ncontent split no longer leak the
#   full content into the injection payload.
def _env_flag_on(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


DEBUG_RETURN_SCORES = _env_flag_on("MEMPALACE_DEBUG_RETURN_SCORES", True)
DEBUG_RETURN_CONTEXT = _env_flag_on("MEMPALACE_DEBUG_RETURN_CONTEXT", True)
MEMORY_PREVIEW_MAX_CHARS = 400


def _shorten_preview(text):
    """Summary-first + length cap for a single memory preview.

    Splits on the first blank line so summary-first records (written as
    ``summary\\n\\ncontent``) render only the ≤280-char distilled summary.
    Then caps at ``MEMORY_PREVIEW_MAX_CHARS`` as a safety net for legacy
    records that pre-date summary-first indexing.
    """
    if not isinstance(text, str):
        return text
    if "\n\n" in text:
        text = text.split("\n\n", 1)[0]
    if len(text) > MEMORY_PREVIEW_MAX_CHARS:
        text = text[: MEMORY_PREVIEW_MAX_CHARS - 1].rstrip() + "\u2026"
    return text


def init(mcp_module):
    """Wire this module to mcp_server so we can access its globals/functions."""
    global _mcp
    _mcp = mcp_module


# ==================== INTENT DECLARATION ====================
# Note: _STATE (ServerState instance) and _INTENT_STATE_DIR live in mcp_server.py.
# We access active-intent state exclusively via _mcp._STATE.active_intent and the
# hook-state directory via _mcp._INTENT_STATE_DIR.


def _intent_state_path() -> Optional[Path]:
    """Session-scoped active-intent state file path, or None if no sid.

    Returns None when ``_STATE.session_id`` is empty. The caller MUST
    treat None as "no persist / no read" rather than substituting a
    shared default filename. A shared ``active_intent_default.json``
    was the cross-agent contamination vector behind the 2026-04-19
    deadlock — it collected every agent's pending state into one file,
    so one agent's resolve could be reloaded as another agent's block.
    """
    sid = _mcp._STATE.session_id
    if not sid:
        return None
    return _mcp._INTENT_STATE_DIR / f"active_intent_{sid}.json"


def _build_intent_hierarchy(context: dict = None) -> list:
    """Build a list of all intent types with their tools and is_a parent.

    Walks the KG to find all entities that is_a intent_type (directly or
    transitively). Returns a list of dicts with id, parent, tools,
    importance, added_by — plus context_rank / context_score when a
    Context is supplied.

    Context-ranked hierarchy. When the caller passes the active
    intent's Context (queries + keywords), we re-use the SAME 3-channel
    pipeline the rest of the palace uses (scoring.multi_channel_search
    against the entity collection with kind='class') to rank intent
    types by semantic similarity to what the agent is actually doing.
    The ranking is baked into the hierarchy entries and persisted to the
    session state file, so the PreToolUse hook — which must stay
    dep-free (no ChromaDB, no Torch) — reads a pre-sorted list with
    zero retrieval work at hook time.
    """

    hierarchy = []
    # Find all entities in the KG that might be intent types
    ecol = _mcp._get_entity_collection(create=False)
    if not ecol:
        return hierarchy

    try:
        all_entities = ecol.get(include=["metadatas"])
        if not all_entities or not all_entities["ids"]:
            return hierarchy
    except Exception:
        return hierarchy

    # Post-P5.2 the entity collection stores multi-view records keyed by
    # '{entity_id}__v{N}' — dedupe by logical entity_id from metadata so
    # we only walk each class once.
    seen_logical = set()
    for i, raw_id in enumerate(all_entities["ids"]):
        meta = all_entities["metadatas"][i] or {}
        if meta.get("kind") != "class":
            continue
        eid = meta.get("entity_id") or raw_id
        if eid in seen_logical:
            continue
        seen_logical.add(eid)

        # Check if this class is-a intent_type (direct or via parent)
        edges = _mcp._STATE.kg.query_entity(eid, direction="outgoing")
        parent_id = None
        for e in edges:
            if e["predicate"] == "is_a" and e["current"]:
                obj = normalize_entity_name(e["object"])
                if obj == "intent_type":
                    parent_id = "intent-type"
                    break
                # Check if parent is itself an intent type
                parent_edges = _mcp._STATE.kg.query_entity(obj, direction="outgoing")
                for pe in parent_edges:
                    if pe["predicate"] == "is_a" and pe["current"]:
                        if normalize_entity_name(pe["object"]) == "intent_type":
                            parent_id = obj
                            break
                if parent_id:
                    break

        if not parent_id:
            continue

        # Get tool permissions via hierarchy resolution
        _, tools = _resolve_intent_profile(eid)
        tool_names = sorted(set(t["tool"] for t in tools)) if tools else []

        importance = meta.get("importance", 3)
        added_by = meta.get("added_by", "")
        hierarchy.append(
            {
                "id": eid,
                "parent": parent_id,
                "tools": tool_names,
                "importance": importance,
                "added_by": added_by,
            }
        )

    # Optional Context-based rank. Uses the same 3-channel
    # pipeline as kg_search / declare_intent memory injection.
    if context:
        _attach_context_rank(hierarchy, context, ecol)

    # Sort: context_rank first (when present; None last), then importance
    # desc, then top-level before children, finally by id for stability.
    hierarchy.sort(
        key=lambda x: (
            x.get("context_rank") if x.get("context_rank") is not None else 10**6,
            -x.get("importance", 3),
            0 if x["parent"] == "intent-type" else 1,
            x["id"],
        )
    )
    return hierarchy


def _attach_context_rank(hierarchy: list, context: dict, ecol) -> None:
    """Attach context_rank + context_score to each hierarchy entry in-place.

    Reuses scoring.multi_channel_search against the entity collection
    filtered to kind='class'. Maps physical Chroma ids (post-P5.2
    '{eid}__v{N}') back to logical entity_ids via metadata.entity_id
    and keeps the max RRF score per logical id.
    """
    queries = context.get("queries") or []
    keywords = context.get("keywords") or []
    if not queries:
        return
    try:
        from . import scoring as _scoring

        pipe = _scoring.multi_channel_search(
            ecol,
            list(queries),
            keywords=list(keywords),
            kg=_mcp._STATE.kg,
            kind="class",
            fetch_limit_per_view=50,
            include_graph=False,
        )
    except Exception:
        return
    rrf_scores = pipe.get("rrf_scores") or {}
    seen_meta = pipe.get("seen_meta") or {}

    logical_scores: dict = {}
    for phys_id, score in rrf_scores.items():
        entry = seen_meta.get(phys_id) or {}
        meta = entry.get("meta") or {}
        logical_id = meta.get("entity_id") or phys_id
        if score > logical_scores.get(logical_id, float("-inf")):
            logical_scores[logical_id] = score

    ranked_ids = sorted(logical_scores.keys(), key=lambda k: -logical_scores[k])
    id_to_rank = {eid: i for i, eid in enumerate(ranked_ids)}

    for entry in hierarchy:
        if entry["id"] in id_to_rank:
            entry["context_rank"] = id_to_rank[entry["id"]]
            entry["context_score"] = round(float(logical_scores[entry["id"]]), 6)


def _build_intent_hierarchy_safe(context: dict = None) -> list:
    """Safe wrapper — never crashes, returns [] on any error."""
    try:
        return _build_intent_hierarchy(context)
    except Exception:
        return []


def _sync_from_disk():
    """Reload active intent state from disk.

    Two cases:
      - Normal sync: in-memory intent matches disk intent \u2014 merge back
        ``used`` and ``budget`` because the PreToolUse hook may have
        bumped them out-of-process.
      - Cold hydration: in-memory state is empty but disk has a valid
        intent. Restore the WHOLE intent record plus pending conflicts.
        Handles MCP-server restart, plugin reinstall,
        or any path that clears ``_STATE`` while an on-disk session is
        still active. Before this hydration path, a restart mid-session
        would leave ``finalize_intent`` returning "No active intent" even
        though the disk file still carried the intent \u2014 the wedge that
        bit the 2026-04-20 wrap_up_session cycle.

    Any read/parse error is non-fatal; the caller falls back to "no
    intent" which is the correct loud-by-absence behavior.
    """
    try:
        state_file = _intent_state_path()
        if state_file is None or not state_file.is_file():
            return
        data = json.loads(state_file.read_text(encoding="utf-8"))
        if not data.get("intent_id"):
            # Disk has only pending state (no intent) \u2014 restore pending
            # conflicts so the agent resolves them before the next declare.
            pending_c = data.get("pending_conflicts") or []
            if pending_c and not _mcp._STATE.pending_conflicts:
                _mcp._STATE.pending_conflicts = pending_c
            return

        if _mcp._STATE.active_intent:
            # Same-intent sync path \u2014 just refresh used/budget + cues.
            if data["intent_id"] == _mcp._STATE.active_intent["intent_id"]:
                _mcp._STATE.active_intent["used"] = data.get("used", {})
                _mcp._STATE.active_intent["budget"] = data.get("budget", {})
                # pending_operation_cues may have been mutated by the hook
                # (entries consumed / TTL-expired) between our last write
                # and this sync; mirror disk truth as the single source.
                _mcp._STATE.active_intent["pending_operation_cues"] = (
                    data.get("pending_operation_cues") or []
                )
            return

        # Cold-hydration path: memory empty, disk has a live intent. Rebuild
        # the full active_intent dict so the next finalize can find it.
        _mcp._STATE.active_intent = {
            "intent_id": data["intent_id"],
            "intent_type": data.get("intent_type", ""),
            "slots": data.get("slots", {}),
            "effective_permissions": data.get("effective_permissions", []),
            "description": data.get("description", ""),
            "agent": data.get("agent", ""),
            "injected_memory_ids": set(data.get("injected_memory_ids", []) or []),
            "accessed_memory_ids": set(data.get("accessed_memory_ids", []) or []),
            "budget": data.get("budget", {}),
            "used": data.get("used", {}),
            "intent_hierarchy": data.get("intent_hierarchy", []),
            "active_context_id": data.get("active_context_id", "") or "",
            "contexts_touched": list(data.get("contexts_touched") or []),
            "contexts_touched_detail": list(data.get("contexts_touched_detail") or []),
        }
        # Preserve pending_operation_cues across MCP restart so agents
        # who declared operations just before the restart don't lose
        # their cues when the server re-hydrates from disk.
        _mcp._STATE.active_intent["pending_operation_cues"] = (
            data.get("pending_operation_cues") or []
        )
        pending_c = data.get("pending_conflicts") or []
        if pending_c and not _mcp._STATE.pending_conflicts:
            _mcp._STATE.pending_conflicts = pending_c
    except Exception as _e:
        # NEVER silent: a failed sync means the in-memory state diverges
        # from disk (stale active_intent, missed hook-updated budget, etc).
        # Record so the next SessionStart surfaces the sync failure.
        try:
            from . import hooks_cli as _hc

            _hc._record_hook_error("_sync_from_disk", _e)
        except Exception:
            pass


def _persist_active_intent():
    """Write the session-scoped state file for the PreToolUse hook.

    Contract:
      - An active_intent without pending state → write intent block, no pending keys.
      - No active_intent but pending conflicts → write a "no-intent"
        state file with just the pending conflicts list. The PreToolUse
        hook does not gate tools on this case (no intent = no permissions),
        but declare_intent reads this pending list on its next call so
        conflicts are never lost just because the intent was finalized
        before the agent resolved them.
      - No active_intent AND no pending state → unlink the file.
    """
    state_file = _intent_state_path()
    if state_file is None:
        # No session_id → no per-agent state file → refuse to persist.
        # Writing to active_intent_default.json would cross-contaminate
        # every agent sharing this MCP server.
        try:
            log_path = _mcp._INTENT_STATE_DIR / "hook.log"
            _mcp._INTENT_STATE_DIR.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"PERSIST_SKIP: _STATE.session_id is empty; refusing to "
                    f"write active_intent_default.json (cross-agent risk). "
                    f"active_intent={bool(_mcp._STATE.active_intent)} "
                    f"pending_conflicts={bool(_mcp._STATE.pending_conflicts)}\n"
                )
        except OSError:
            pass
        return
    try:
        _mcp._INTENT_STATE_DIR.mkdir(parents=True, exist_ok=True)
        has_intent = bool(_mcp._STATE.active_intent)
        has_pending = bool(_mcp._STATE.pending_conflicts)

        if not has_intent and not has_pending:
            # Fully clean state — nothing to persist.
            if state_file.exists():
                state_file.unlink()
            return

        if has_intent:
            cached_hierarchy = _mcp._STATE.active_intent.get("intent_hierarchy")
            if cached_hierarchy is None:
                cached_hierarchy = _build_intent_hierarchy_safe()
            state = {
                "intent_id": _mcp._STATE.active_intent["intent_id"],
                "intent_type": _mcp._STATE.active_intent["intent_type"],
                "slots": _mcp._STATE.active_intent["slots"],
                "effective_permissions": _mcp._STATE.active_intent["effective_permissions"],
                "description": _mcp._STATE.active_intent.get("description", ""),
                "agent": _mcp._STATE.active_intent.get("agent", ""),
                "session_id": _mcp._STATE.session_id,
                "intent_hierarchy": cached_hierarchy,
                "injected_memory_ids": list(
                    _mcp._STATE.active_intent.get("injected_memory_ids", set())
                ),
                "accessed_memory_ids": list(
                    _mcp._STATE.active_intent.get("accessed_memory_ids", set())
                ),
                "budget": _mcp._STATE.active_intent.get("budget", {}),
                "used": _mcp._STATE.active_intent.get("used", {}),
                "pending_conflicts": _mcp._STATE.pending_conflicts or [],
                # pending_operation_cues (2026-04-20): list of agent-declared
                # operation cues from mempalace_declare_operation, consumed
                # by the PreToolUse hook subprocess. List form supports
                # Claude Code's parallel tool dispatch (N declares in one
                # message, N tool calls follow — each consumes its own cue
                # by tool-name match). Hook pops first matching entry on
                # consume, writes shortened list back. Entries carry
                # declared_at_ts; the hook expires stale entries on consume
                # (see OPERATION_CUE_TTL_SECONDS in hooks_cli.py).
                "pending_operation_cues": _mcp._STATE.active_intent.get("pending_operation_cues")
                or [],
                # P1 context-as-entity: the context entity id active for
                # this intent. Writers (kg_declare_entity, kg_add,
                # _add_memory_internal) read it from active_intent to
                # emit `created_under` edges on every write.
                "active_context_id": _mcp._STATE.active_intent.get("active_context_id", "") or "",
                "contexts_touched": list(_mcp._STATE.active_intent.get("contexts_touched") or []),
                "contexts_touched_detail": list(
                    _mcp._STATE.active_intent.get("contexts_touched_detail") or []
                ),
            }
        else:
            # No active intent but pending state must outlive the finalize.
            # Write a minimal placeholder file that the hook ignores (no
            # intent_id key) but the declare_intent pending-check reads via
            # _load_pending_*_from_disk.
            state = {
                "intent_id": "",
                "session_id": _mcp._STATE.session_id,
                "pending_conflicts": _mcp._STATE.pending_conflicts or [],
            }
        state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError as _e:
        # NEVER silent: record to hook_errors.jsonl so the next SessionStart
        # (or any hook output) surfaces the failure. A silent persist loss
        # means the hook + the server disagree about active_intent forever.
        try:
            from . import hooks_cli as _hc

            _hc._record_hook_error("_persist_active_intent", _e)
        except Exception:
            # Absolute last resort: even the recorder cannot raise or we'd
            # lose the whole tool call. Swallow but leave a log breadcrumb
            # via the module-level log file already written above.
            pass


def _resolve_intent_profile(intent_type_id: str):
    """Walk is-a hierarchy to resolve effective slots and tool_permissions.

    Returns (slots, tool_permissions) where:
    - slots: merged from child to parent (child wins on conflict)
    - tool_permissions: ADDITIVE — child tools are merged with parent tools.
      Child can only ADD tools, not remove parent tools. This prevents
      overreach: a child of inspect can add WebFetch but can't drop Read.
    """

    visited = set()
    current = intent_type_id
    merged_slots = {}
    merged_tools = []  # Additive: collect from all levels
    seen_tools = set()  # Deduplicate by tool name

    # Walk upward through is-a chain (max 5 hops)
    for _ in range(5):
        if current in visited:
            break
        visited.add(current)

        entity = _mcp._STATE.kg.get_entity(current)
        if not entity:
            break

        props = entity.get("properties", {})
        if isinstance(props, str):
            import json as _json

            try:
                props = _json.loads(props)
            except Exception:
                props = {}

        profile = props.get("rules_profile", {})

        # Slots: merge (child wins, so only add parent slots not already defined)
        for slot_name, slot_def in profile.get("slots", {}).items():
            if slot_name not in merged_slots:
                merged_slots[slot_name] = slot_def

        # Tool permissions: ADDITIVE — collect from all levels, child + parent
        for perm in profile.get("tool_permissions", []):
            tool_key = perm.get("tool", "")
            if tool_key not in seen_tools:
                seen_tools.add(tool_key)
                merged_tools.append(perm)

        # Walk to parent via is-a — prefer intent hierarchy over universal "thing"
        edges = _mcp._STATE.kg.query_entity(current, direction="outgoing")
        parent = None
        for e in edges:
            if e["predicate"] == "is_a" and e["current"]:
                parent_id = normalize_entity_name(e["object"])
                # Stop at the root intent_type class
                if parent_id == "intent_type":
                    break
                # Skip universal base class — not part of intent hierarchy
                if parent_id == "thing":
                    continue
                parent_entity = _mcp._STATE.kg.get_entity(parent_id)
                if parent_entity and parent_entity.get("kind") == "class":
                    parent = parent_id
                    break
        if not parent:
            break
        current = parent

    return merged_slots, merged_tools


def _is_intent_type(entity_id: str) -> bool:
    """Check if an entity is-a intent_type (direct or inherited)."""

    edges = _mcp._STATE.kg.query_entity(entity_id, direction="outgoing")
    for e in edges:
        if e["predicate"] == "is_a" and e["current"]:
            obj = normalize_entity_name(e["object"])
            if obj == "intent_type":
                return True
            # Check parent (one level — e.g., edit_file is-a modify is-a intent_type)
            parent_edges = _mcp._STATE.kg.query_entity(obj, direction="outgoing")
            for pe in parent_edges:
                if pe["predicate"] == "is_a" and pe["current"]:
                    if normalize_entity_name(pe["object"]) == "intent_type":
                        return True
    return False


def tool_declare_intent(  # noqa: C901
    intent_type: str,
    slots: dict,
    context: dict = None,  # mandatory unified Context
    descriptions=None,  # LEGACY: rejected when context is missing (see below)
    auto_declare_files: bool = False,
    agent: str = None,
    budget: dict = None,
):
    """Declare what you intend to do BEFORE doing it. Returns permissions + context.

    budget: MANDATORY dict of tool_name -> max_calls. E.g. {"Read": 5, "Edit": 3}.
            Must cover all tools you plan to use. Budget is tracked by the hook —
            when exhausted, the tool is blocked until you extend (mempalace_extend_intent)
            or finalize and redeclare. Keep budgets tight — inflated budgets waste context.

    One active intent at a time — declaring a new intent expires the previous.
    mempalace_* tools are always allowed (not gated by intent).

    Args:
        intent_type: A declared intent type entity (is-a intent_type).
            Built-in types: inspect, modify, execute, communicate.
            Domain-specific: edit_file, write_tests, deploy, run_tests, etc.
            Declare new types via kg_declare_entity with is-a <parent_intent_type>.

        slots: Named slots filled with entity names. Each intent type defines
            expected slots with class constraints. Example:
            For edit_file:  {"files": ["auth.test.ts", "auth.utils.ts"]}
            For deploy:     {"target": ["flowsev_repository"], "environment": ["staging"]}
            For inspect:    {"subject": ["paperclip_server"]}

            Slot definitions are stored in the intent type's rules_profile.slots.
            Each slot has: classes (accepted entity classes), required (bool),
            multiple (bool — accepts list vs single entity).

        context: MANDATORY Context fingerprint for this intent.
            {
              "queries":  list[str]   2-5 perspectives on what you're about to do
              "keywords": list[str]   2-5 caller-provided exact terms
              "entities": list[str]   0+ related/seed entity ids (defaults to slot
                                      entities when omitted — they ARE the entities
                                      this intent is about)
            }
            Each query becomes a separate cosine view for multi-view retrieval;
            keywords drive the keyword channel (no auto-extraction); entities
            seed Channel B graph BFS. The Context's view vectors are persisted
            so future feedback applies via MaxSim. Example:
            context={
              "queries": ["Editing auth rate limiter",
                          "Security hardening against brute force",
                          "Adding tests for login endpoint"],
              "keywords": ["auth", "rate-limit", "brute-force", "login"],
              "entities": ["LoginService", "AuthRateLimiter"]
            }

    Returns:
        permissions: Which tools are allowed and their scope (scoped to slots or unrestricted).
        memories: Relevant injected memories (multi-view retrieved using the Context).
        previous_expired: ID of the previous active intent if one was replaced.
    """

    # ── Reject the legacy `descriptions` path (Context mandatory) ──
    from .scoring import validate_context as _validate_context

    if context is None and descriptions is not None:
        return {
            "success": False,
            "error": (
                "`descriptions` is gone. Pass `context` instead — a dict "
                "with mandatory queries, keywords, and optional entities. Example:\n"
                '  context={"queries": ["Editing auth rate limiter", '
                '"Security hardening", "Login endpoint tests"], '
                '"keywords": ["auth", "rate-limit", "brute-force"], '
                '"entities": ["LoginService"]}\n'
                "queries[0] becomes the canonical description for the active intent."
            ),
        }

    clean_context, ctx_err = _validate_context(context)
    if ctx_err:
        return ctx_err
    _description_views = clean_context["queries"]
    _context_keywords = clean_context["keywords"]
    _context_entities = clean_context["entities"]
    description = _description_views[0]

    # fail-fast agent validation. Unified with finalize_intent and
    # every other write entry point: undeclared agents are rejected at
    # the boundary instead of causing silent downstream failures.
    sid_err = _mcp._require_sid(action="declare_intent")
    if sid_err:
        return sid_err
    agent_err = _mcp._require_agent(agent, action="declare_intent")
    if agent_err:
        return agent_err

    # ── Check for pending conflicts ──
    # Disk is source of truth — reload from disk if memory is empty (MCP restart scenario)
    pending_conflicts = _mcp._STATE.pending_conflicts
    if not pending_conflicts and hasattr(_mcp, "_load_pending_conflicts_from_disk"):
        pending_conflicts = _mcp._load_pending_conflicts_from_disk() or None
        if pending_conflicts:
            _mcp._STATE.pending_conflicts = pending_conflicts
    if pending_conflicts:
        return {
            "success": False,
            "error": (
                f"{len(pending_conflicts)} conflicts pending from previous activity. "
                f"You MUST resolve ALL before declaring a new intent. Call "
                f"mempalace_resolve_conflicts with an action for each: "
                f"invalidate (old is stale), merge (combine — read both in full first), "
                f"keep (both valid), or skip (undo new)."
            ),
            "pending_conflicts": pending_conflicts,
        }

    # ── Validate intent_type ──
    try:
        intent_type = _mcp.sanitize_name(intent_type, "intent_type")
    except ValueError as e:
        return {"success": False, "error": str(e)}

    intent_id = normalize_entity_name(intent_type)

    if not _mcp._is_declared(intent_id):
        return {
            "success": False,
            "error": (
                f"Intent type '{intent_id}' not declared in this session. "
                f"Specific intent types are preferred over broad ones — they carry domain-specific "
                f"rules (must, requires, has_gotcha) that broad types don't. "
                f"Create it now:\n"
                f"  1. "
                + _mcp._declare_entity_recipe(
                    intent_type,
                    kind="class",
                    hint="what this action does, when to use it",
                    extra_properties=(
                        "{'rules_profile': {'slots': {...}, 'tool_permissions': [...]}}"
                    ),
                )
                + "\n"
                f"  2. kg_add(subject='{intent_type}', predicate='is_a', "
                f"object='<parent>', context={{'queries': [...], 'keywords': [...]}}) "
                f"— where parent is the broad type it inherits from "
                f"(inspect, modify, execute, or communicate)\n"
                f"  3. Then retry declare_intent with this type.\n"
                f"This is a one-time cost — once created, the type persists across sessions "
                f"and accumulates rules that will be surfaced on every future use."
            ),
        }

    if not _is_intent_type(intent_id):
        return {
            "success": False,
            "error": (
                f"'{intent_id}' exists but is not an intent type (missing is_a edge to the hierarchy). "
                f"Link it to the parent it inherits from:\n"
                f"  kg_add(subject='{intent_id}', predicate='is_a', object='<parent>')\n"
                f"Where parent is the broad type it specializes "
                f"(inspect, modify, execute, or communicate). "
                f"The type will then inherit its parent's permissions and slots, "
                f"and you can attach domain-specific rules to it."
            ),
        }

    # ── Auto-narrow: use description to find best-fit child intent type ──
    narrowed_from = None
    subtypes = []
    child_scores = []
    # Only kind=class — execution instances (kind=entity) are NOT subtypes
    all_entities = _mcp._STATE.kg.list_entities(status="active", kind="class")
    for e in all_entities:
        e_edges = _mcp._STATE.kg.query_entity(e["id"], direction="outgoing")
        for edge in e_edges:
            if edge["predicate"] == "is_a" and edge["current"]:
                parent_id = normalize_entity_name(edge["object"])
                if parent_id == intent_id:
                    subtypes.append(
                        {
                            "id": e["id"],
                            "description": e.get("description", ""),
                        }
                    )
                    break

    if subtypes and description.strip():
        ecol = _mcp._get_entity_collection(create=False)
        if ecol:
            try:
                child_id_set = {s["id"] for s in subtypes}
                count = ecol.count()
                if count > 0:
                    results = ecol.query(
                        query_texts=[description],
                        n_results=min(count, 50),
                        include=["documents", "metadatas", "distances"],
                    )
                    # Collect distances for parent and children
                    parent_dist = None
                    child_scores = []  # (id, distance, description)
                    if results["ids"] and results["ids"][0]:
                        for i, eid in enumerate(results["ids"][0]):
                            dist = results["distances"][0][i]
                            if eid == intent_id:
                                parent_dist = dist
                            elif eid in child_id_set:
                                child_scores.append(
                                    {
                                        "id": eid,
                                        "distance": dist,
                                        "description": results["documents"][0][i],
                                    }
                                )
                    # Auto-narrow: if a child is closer than the parent, it's
                    # a better fit for the agent's description. Use it.
                    # But only if the child's slots are compatible with what was provided.
                    if parent_dist is not None and child_scores:
                        child_scores.sort(key=lambda c: c["distance"])
                        better = [c for c in child_scores if c["distance"] < parent_dist]
                        # Filter out children whose required slots don't match
                        compatible = []
                        for candidate in better:
                            child_slots, _ = _resolve_intent_profile(candidate["id"])
                            if not child_slots:
                                continue
                            # Check: all required child slots must be present in provided slots
                            missing = [
                                s
                                for s, d in child_slots.items()
                                if d.get("required", False) and s not in slots
                            ]
                            if not missing:
                                compatible.append(candidate)
                        if len(compatible) == 1:
                            narrowed_from = intent_id
                            intent_id = compatible[0]["id"]
                            _mcp._STATE.declared_entities.add(intent_id)
                        elif len(compatible) > 1:
                            # Multiple children beat the parent — disambiguate
                            return {
                                "success": False,
                                "error": (
                                    f"Description matches multiple subtypes of '{intent_id}' "
                                    f"better than '{intent_id}' itself. "
                                    f"Pick the most appropriate one and declare it directly."
                                ),
                                "matching_subtypes": [
                                    {"id": c["id"], "description": c["description"][:120]}
                                    for c in compatible
                                ],
                            }
            except Exception:
                child_scores = []  # Non-fatal — narrowing is best-effort

    # ── Resolve effective profile via inheritance ──
    effective_slots, effective_permissions = _resolve_intent_profile(intent_id)

    if not effective_slots:
        return {
            "success": False,
            "error": (
                f"Intent type '{intent_id}' has no slots defined in its rules_profile. "
                f"Update its properties to include rules_profile.slots. Example: "
                f'{{"slots": {{"files": {{"classes": ["file"], "required": true, "multiple": true}}}}}}'
            ),
        }

    # ── Validate slots ──
    if not isinstance(slots, dict):
        return {
            "success": False,
            "error": (
                f"slots must be a dict mapping slot names to entity names. "
                f"Expected slots for '{intent_id}': {list(effective_slots.keys())}. "
                "Example: {"
                + ", ".join(
                    chr(34) + k + chr(34) + ": [" + chr(34) + "entity_name" + chr(34) + "]"
                    for k in effective_slots
                )
                + "}"
            ),
        }

    slot_errors = []
    resolved_slots = {}  # slot_name -> list of normalized entity IDs

    # Check required slots are present
    for slot_name, slot_def in effective_slots.items():
        if slot_def.get("required", False) and slot_name not in slots:
            slot_errors.append(
                f"Required slot '{slot_name}' not provided. "
                f"Accepted classes: {slot_def.get('classes', ['thing'])}."
            )

    # Check provided slots are valid
    for slot_name, slot_values in slots.items():
        if slot_name not in effective_slots:
            slot_errors.append(
                f"Unknown slot '{slot_name}'. Valid slots: {list(effective_slots.keys())}."
            )
            continue

        slot_def = effective_slots[slot_name]

        # Normalize to list
        if isinstance(slot_values, str):
            slot_values = [slot_values]
        if not isinstance(slot_values, list):
            slot_errors.append(f"Slot '{slot_name}' must be a string or list of strings.")
            continue

        # Check multiple
        if not slot_def.get("multiple", False) and len(slot_values) > 1:
            slot_errors.append(
                f"Slot '{slot_name}' accepts only one entity (multiple=false), got {len(slot_values)}."
            )
            continue

        # Raw slots: accept strings as-is, no entity declaration needed
        # Used for command patterns, URLs, etc.
        if slot_def.get("raw", False):
            normalized_values = [{"id": val, "raw": val} for val in slot_values]
            resolved_slots[slot_name] = normalized_values
            continue

        # Validate each entity in slot
        normalized_values = []
        allowed_classes = slot_def.get("classes", ["thing"])
        is_file_slot = "file" in allowed_classes

        for val in slot_values:
            # For file slots: use basename for entity name, keep raw path for scoping
            if is_file_slot:
                file_basename = os.path.basename(val)
                val_id = normalize_entity_name(file_basename)
            else:
                val_id = normalize_entity_name(val)

            # Auto-declare file entities if slot expects class=file
            if not _mcp._is_declared(val_id) and is_file_slot:
                file_exists = os.path.exists(val) or os.path.exists(os.path.join(os.getcwd(), val))
                if file_exists or auto_declare_files:
                    # Auto-declare: create entity from basename + is-a file
                    _auto_desc = f"[AUTO — needs refinement] File: {val}" + (
                        " (new)" if not file_exists else ""
                    )
                    _mcp._create_entity(
                        file_basename,
                        kind="entity",
                        description=_auto_desc,
                        importance=2,
                        added_by=agent,
                    )
                    _mcp._STATE.kg.add_triple(val_id, "is_a", "file")
                    _mcp._STATE.declared_entities.add(val_id)
                    # Auto-mints get an immediate generic_summary flag so
                    # the memory_gardener picks them up and produces a
                    # real WHAT/WHY description from the file's first
                    # docstring (or sibling signals). The rule is:
                    # every summary is caller-authored; any path that
                    # cannot honour that (auto-declare files, phantom
                    # entities) must flag for refinement at mint time.
                    try:
                        _mcp._STATE.kg.record_memory_flags(
                            [
                                {
                                    "kind": "generic_summary",
                                    "memory_ids": [val_id],
                                    "detail": (
                                        "Auto-declared file entity; description is a "
                                        "'File: <path>' placeholder. Replace with a "
                                        "≤280-char WHAT/WHY summary — what this file "
                                        "does and why it exists — drawn from the "
                                        "first docstring or module-level comment."
                                    ),
                                    # context_id intentionally empty: _active_context_id is
                                    # not yet minted at slot-validation time. Gardener still
                                    # picks up context-less flags; dedup collapses repeated
                                    # mints of the same file via (kind, memory_key, '').
                                    "context_id": "",
                                }
                            ],
                            rater_model="auto_declare_files",
                        )
                    except Exception:
                        pass  # Non-fatal: missing the flag doesn't block declaration.
                elif not file_exists:
                    slot_errors.append(
                        f"File '{val}' does not exist on disk and auto_declare_files=false. "
                        f"Either provide an existing file path, or set auto_declare_files=true "
                        f"if you intend to create this file."
                    )
                    continue

            if not _mcp._is_declared(val_id):
                slot_errors.append(
                    f"Entity '{val_id}' in slot '{slot_name}' not declared. "
                    f"Call kg_declare_entity first."
                )
                continue

            # Check class constraint via is-a + inheritance
            if "thing" not in allowed_classes:
                entity_classes = [
                    e["object"]
                    for e in _mcp._STATE.kg.query_entity(val_id, direction="outgoing")
                    if e["predicate"] == "is_a" and e["current"]
                ]
                if entity_classes:
                    from .knowledge_graph import normalize_entity_name as _norm

                    norm_classes = [_norm(c) for c in entity_classes]
                    norm_allowed = [_norm(c) for c in allowed_classes]

                    def _check_subclass(classes, allowed, depth=5):
                        if any(c in allowed for c in classes):
                            return True
                        visited = set(classes)
                        frontier = list(classes)
                        for _ in range(depth):
                            nxt = []
                            for cls in frontier:
                                for e in _mcp._STATE.kg.query_entity(cls, direction="outgoing"):
                                    if e["predicate"] == "is_a" and e["current"]:
                                        p = _norm(e["object"])
                                        if p in allowed:
                                            return True
                                        if p not in visited:
                                            visited.add(p)
                                            nxt.append(p)
                            frontier = nxt
                            if not frontier:
                                break
                        return False

                    if not _check_subclass(norm_classes, norm_allowed):
                        slot_errors.append(
                            f"Entity '{val_id}' in slot '{slot_name}' is-a {entity_classes}, "
                            f"but slot requires classes {allowed_classes}."
                        )
                        continue

            normalized_values.append({"id": val_id, "raw": val})
        resolved_slots[slot_name] = normalized_values

    if slot_errors:
        return {
            "success": False,
            "error": "Slot validation failed for declare_intent.",
            "slot_issues": slot_errors,
            "expected_slots": {
                name: {
                    "classes": d.get("classes", ["thing"]),
                    "required": d.get("required", False),
                    "multiple": d.get("multiple", False),
                }
                for name, d in effective_slots.items()
            },
        }

    # ── Build permissions ──
    # Flatten resolved_slots for return (id only) and keep raw paths for scoping
    flat_slots = {}  # slot_name -> [entity_id, ...]
    raw_paths = {}  # slot_name -> [raw_value, ...]
    all_slot_entities = []
    raw_slot_names = set()
    for slot_name, entries in resolved_slots.items():
        flat_slots[slot_name] = [e["id"] for e in entries]
        raw_paths[slot_name] = [e["raw"] for e in entries]
        # Check if this is a raw slot (commands, etc.) — don't add to entity list
        slot_def = effective_slots.get(slot_name, {})
        if slot_def.get("raw", False):
            raw_slot_names.add(slot_name)
        else:
            all_slot_entities.extend(flat_slots[slot_name])

    def _resolve_file_path(entity_id):
        """Resolve actual file path for a file entity.

        Checks entity properties for 'file_path', then falls back to
        extracting the path from the description (format: 'path/to/file.py — ...')
        """
        entity = _mcp._STATE.kg.get_entity(entity_id)
        if not entity:
            return None
        # Check properties first
        props = entity.get("properties", {})
        if isinstance(props, str):
            import json as _json

            try:
                props = _json.loads(props)
            except Exception:
                props = {}
        fp = props.get("file_path")
        if fp:
            return fp
        # Fall back to description — extract path from known formats
        desc = entity.get("description", "")
        # Format: "File: /path/to/file.ext" or "File: /path/to/file.ext (new)"
        if desc.startswith("File: "):
            candidate = desc[6:].split("(")[0].strip()
            if "/" in candidate or "\\" in candidate:
                return candidate
        # Format: "path/to/file.py — description text"
        for sep in (" — ", " - ", " – "):
            if sep in desc:
                candidate = desc.split(sep, 1)[0].strip()
                if (
                    "/" in candidate
                    or "\\" in candidate
                    or candidate.endswith((".py", ".ts", ".js", ".json"))
                ):
                    return candidate
        return None

    permissions = []
    for slot_name, entity_ids in flat_slots.items():
        raws = raw_paths.get(slot_name, entity_ids)
        # Check if this slot contains file entities — resolve actual paths
        slot_def = effective_slots.get(slot_name, {})
        slot_classes = slot_def.get("classes", [])
        is_file_slot = "file" in slot_classes
        for perm in effective_permissions:
            scope = perm.get("scope", "*")
            if f"{{{slot_name}}}" in scope:
                for raw_val, entity_id in zip(raws, entity_ids):
                    resolved_scope = raw_val
                    if is_file_slot:
                        file_path = _resolve_file_path(entity_id)
                        if not file_path:
                            return {
                                "success": False,
                                "error": (
                                    f"File entity '{entity_id}' has no file_path configured. "
                                    f"Either re-declare it with properties={{'file_path': "
                                    f"'path/to/file.ext'}} using "
                                    + _mcp._declare_entity_recipe(
                                        entity_id,
                                        kind="entity",
                                        hint=f"file entity {entity_id}",
                                        extra_properties="{'file_path': 'path/to/file.ext'}",
                                    )
                                    + ", or update it via kg_update_entity(entity='"
                                    + entity_id
                                    + "', properties={'file_path': 'path/to/file.ext'})."
                                ),
                            }
                        resolved_scope = file_path
                    permissions.append(
                        {
                            "tool": perm["tool"],
                            "scope": scope.replace(f"{{{slot_name}}}", resolved_scope),
                            "slot": slot_name,
                            "entity": entity_id,
                        }
                    )
            elif scope == "*":
                if not any(p["tool"] == perm["tool"] and p["scope"] == "*" for p in permissions):
                    permissions.append({"tool": perm["tool"], "scope": "*"})
            else:
                if not any(p["tool"] == perm["tool"] and p["scope"] == scope for p in permissions):
                    permissions.append({"tool": perm["tool"], "scope": scope})

    # ── Validate budget (after permissions so slot/type errors come first) ──
    if not budget or not isinstance(budget, dict):
        return {
            "success": False,
            "error": (
                "budget is MANDATORY. Provide a dict of tool_name -> max_calls. "
                'Example: budget={"Read": 5, "Edit": 3, "Bash": 2}. '
                "Keep budgets tight — estimate the minimum calls needed for this task."
            ),
        }
    # Validate budget: only keep tools that are actually permitted
    permitted_tool_names = {p["tool"] for p in permissions}
    validated_budget = {}
    for tool_name, count in budget.items():
        if tool_name not in permitted_tool_names:
            continue  # Silently ignore — permission check blocks anyway
        try:
            n = int(count)
            if n < 1:
                return {
                    "success": False,
                    "error": f"Budget for '{tool_name}' must be >= 1, got {n}",
                }
            validated_budget[tool_name] = n
        except (TypeError, ValueError):
            return {
                "success": False,
                "error": f"Budget for '{tool_name}' must be int, got {count!r}",
            }
    if not validated_budget:
        return {
            "success": False,
            "error": (
                f"Budget has no permitted tools. Permitted: {sorted(permitted_tool_names)}. "
                f"Budget must include at least one of these."
            ),
        }

    # ── Collect context via 3-channel retrieval ──
    context = {"memories": []}

    # ── 3-channel retrieval: cosine + graph + keyword → RRF merge ──

    # ── Context-scoped relevance feedback (signed, confidence-graded) ──
    # The signal is read from rated_useful / rated_irrelevant edges on the
    # active context PLUS its 1-2 hop similar_to neighbourhood
    # (lookup_context_feedback). finalize_intent stores
    # confidence = relevance/5.0 on each rated_* edge, so the mapping is:
    #
    #   relevance 5 useful      → confidence 1.0 → boost +1.0
    #   relevance 1 useful      → confidence 0.2 → boost +0.2
    #   no feedback             → 0.0 (neutral)
    #   relevance 1 irrelevant  → confidence 0.2 → penalty -0.2
    #   relevance 5 irrelevant  → confidence 1.0 → penalty -1.0
    #
    # The dict is populated AFTER _views is built and context_lookup_or_create
    # has minted / reused an active_context_id (below). Until then _relevance_boost
    # returns 0 (no signal) — retrieval runs AFTER the populate step anyway.
    _context_feedback: dict = {}

    def _relevance_boost(memory_id):
        """Return continuous relevance signal from context feedback.

        Returns float in [-1.0, +1.0]. Feeds hybrid_score as the signed
        relevance_feedback term — rated_irrelevant memories drop below
        neutral, rated_useful rise above.
        """
        return _context_feedback.get(memory_id, 0.0)

    def _preview(entity_id_or_memory):
        """Get text preview for any ID — memory content or entity description."""
        if entity_id_or_memory.startswith(("record_", "diary_")):
            try:
                col = _mcp._get_collection(create=False)
                if col:
                    d = col.get(ids=[entity_id_or_memory], include=["documents"])
                    if d and d["documents"] and d["documents"][0]:
                        return d["documents"][0][:150].replace("\n", " ")
            except Exception:
                pass
        else:
            try:
                ent = _mcp._STATE.kg.get_entity(entity_id_or_memory)
                if ent and ent.get("description"):
                    return ent["description"][:150].replace("\n", " ")
            except Exception:
                pass
        return ""

    already_seen_ids = set()  # dedup across all channels

    # ── Build multi-view queries from description + context ──
    _views = list(_description_views)  # start with explicit views
    if not _views and description:
        _views.append(description)
    if intent_id and intent_id not in _views:
        _views.append(intent_id)
    for entity_id in all_slot_entities[:3]:
        try:
            ent = _mcp._STATE.kg.get_entity(entity_id)
            if ent and ent.get("description"):
                _views.append(ent["description"][:200])
        except Exception:
            pass
    _views = list(dict.fromkeys(_views))[:6]
    if not _views:
        _views = [intent_id or "unknown"]

    # ── Context as first-class entity ──
    # Mint or reuse a kind="context" entity BEFORE the retrieval loops
    # so _relevance_boost can read rated_* edges scoped to this context's
    # similar_to neighbourhood. declare_intent is an emit site; other
    # writers (kg_declare_entity, _add_memory_internal) will reference
    # this id via created_under.
    _active_context_id = ""
    _active_context_reused = False
    try:
        _cid, _reused, _cms = _mcp.context_lookup_or_create(
            queries=_views,
            keywords=_context_keywords,
            entities=_context_entities,
            agent=agent or "",
        )
        _active_context_id = _cid or ""
        _active_context_reused = bool(_reused)
    except Exception:
        _active_context_id = ""

    # Pre-compute the intent-level emit entry; merged into
    # active_intent.contexts_touched_detail right after the dict is
    # built (see below). Rocchio enrichment at finalize will iterate
    # every entry in that detail list, not just this one, so operation
    # and search contexts also qualify for enrichment when reused +
    # net-positive.
    _intent_emit_entry = {
        "ctx_id": _active_context_id,
        "reused": _active_context_reused,
        "scope": "intent",
        "queries": list(_views),
        "keywords": list(_context_keywords),
        "entities": list(_context_entities),
    }

    # ══════════════════════════════════════════════════════════════
    # CHANNELS A+C: Unified retrieval — BOTH collections.
    # Uses the SAME scoring.multi_channel_search as kg_search. Each
    # collection runs Channels A (multi-view cosine) and C (keyword
    # overlap) internally; results merge into a shared RRF pot with
    # Channel B (graph BFS, below). Entity AND record candidates
    # compete head-to-head for injection — rules, concepts, gotchas,
    # past executions, and prose records all surface by relevance.
    #
    # This replaces the pre-P6.6 split where entities were queried
    # for _entity_sim (thrown away as candidates) and only records
    # became Channel A results. Now EVERYTHING is a candidate.
    # ══════════════════════════════════════════════════════════════
    from . import scoring as _scoring

    _channel_a_lists = {}  # unified: "record_cosine_0", "entity_cosine_0", etc.
    _combined_meta = {}  # mid -> {"meta": {...}, "doc": "...", "similarity": float}
    _entity_sim = {}  # entity_id -> max similarity (still needed by Channel B)

    # Share ONE walk of the context neighbourhood across all three
    # collection pipes (record / entity / triple) AND _relevance_boost.
    # The walker returns two aggregates:
    #   - rated_scores: per-memory signed float for hybrid_score's W_REL
    #     (consumed via _relevance_boost / _context_feedback below).
    #   - channel_D_list: ranked list for Channel D (passed via
    #     rated_walk kwarg into multi_channel_search).
    _rated_walk = (
        _scoring.walk_rated_neighbourhood(_active_context_id, _mcp._STATE.kg)
        if _active_context_id
        else {"rated_scores": {}, "channel_D_list": []}
    )
    _context_feedback = _rated_walk.get("rated_scores") or {}

    # Record collection (prose records — the old "memory" collection)
    try:
        dcol = _mcp._get_collection(create=False)
        if dcol:
            record_pipe = _scoring.multi_channel_search(
                dcol,
                _views,
                keywords=_context_keywords,
                kg=_mcp._STATE.kg,
                fetch_limit_per_view=50,
                include_graph=False,
                active_context_id=_active_context_id,
                rated_walk=_rated_walk,
            )
            for name, lst in record_pipe.get("ranked_lists", {}).items():
                _channel_a_lists[f"record_{name}"] = lst
            for mid, info in record_pipe.get("seen_meta", {}).items():
                _combined_meta[mid] = {**info, "source": "record"}
    except Exception:
        pass

    # Entity collection (structured entities — rules, concepts, past execs)
    try:
        ecol = _mcp._get_entity_collection(create=False)
        if ecol:
            entity_pipe = _scoring.multi_channel_search(
                ecol,
                _views,
                keywords=_context_keywords,
                kg=_mcp._STATE.kg,
                fetch_limit_per_view=50,
                include_graph=False,
                active_context_id=_active_context_id,
                rated_walk=_rated_walk,
            )
            for name, lst in entity_pipe.get("ranked_lists", {}).items():
                _channel_a_lists[f"entity_{name}"] = lst
            # Build _entity_sim from entity pipe's seen_meta (Channel B needs it)
            for mid, info in entity_pipe.get("seen_meta", {}).items():
                meta = info.get("meta") or {}
                logical_id = meta.get("entity_id") or mid
                sim = info.get("similarity", 0.0)
                _entity_sim[logical_id] = max(_entity_sim.get(logical_id, 0.0), sim)
                _combined_meta[mid] = {**info, "source": "entity"}
    except Exception:
        pass

    # Triple verbalization collection — surfaces structured (subject, predicate,
    # object) facts as first-class injected context. Without this, declare_intent
    # only sees prose memories and entity descriptions; triples like
    # (adrian, lives_in, warsaw) only contribute to the BFS Channel B if an
    # entity in the walk happens to attach them.
    try:
        from .knowledge_graph import _get_triple_collection

        tcol = _get_triple_collection()
        if tcol is not None and tcol.count() > 0:
            triple_pipe = _scoring.multi_channel_search(
                tcol,
                _views,
                keywords=_context_keywords,
                kg=_mcp._STATE.kg,
                fetch_limit_per_view=50,
                include_graph=False,
                active_context_id=_active_context_id,
                rated_walk=_rated_walk,
            )
            for name, lst in triple_pipe.get("ranked_lists", {}).items():
                _channel_a_lists[f"triple_{name}"] = lst
            for mid, info in triple_pipe.get("seen_meta", {}).items():
                _combined_meta[mid] = {**info, "source": "triple"}
    except Exception:
        pass

    # ══════════════════════════════════════════════════════════════
    # CHANNEL B: Graph — BFS from slot entities + intent type
    # Subsumes old sources 1 (KG edges), 2 (intent rules),
    # 4 (past executions), 5 (graph memories).
    #
    # Graph-seed derivation strategy (P5.9 doc):
    # This is the CONTROLLED-BFS variant, complementing the autonomous
    # top-cosine-seeds strategy in scoring.multi_channel_search. The
    # intent declaration already NAMES the entities it's about (via
    # slots + context.entities), so we anchor the walk on those rather
    # than guessing from semantic similarity. Two modes are intentional:
    #   - declare_intent → controlled BFS (caller knows the anchors)
    #   - kg_search      → autonomous BFS (caller doesn't always know)
    # ══════════════════════════════════════════════════════════════
    GRAPH_BUDGET = 30
    _MAX_HOPS = 3
    _MIN_EDGE_USEFULNESS = -0.5
    _GRAPH_SIM = {1: 0.5, 2: 0.3, 3: 0.1}
    _graph_memories = {}  # memory_id -> distance (for hop-shortening in finalize)
    _graph_entities = {}  # entity_id -> distance
    _channel_b_list = []
    _past_exec_ids = []  # for promotion check
    try:
        # BFS seeds: slot entities + intent type
        # Channel B seeds: slot entities + intent type + caller-provided
        # context.entities (explicit graph anchors). Slots stay as the
        # default backbone; context.entities augments them.
        bfs_seeds = list(all_slot_entities)
        for cent in _context_entities or []:
            cent_id = normalize_entity_name(cent)
            if cent_id and cent_id not in bfs_seeds:
                bfs_seeds.append(cent_id)
        if intent_id and intent_id not in bfs_seeds:
            bfs_seeds.append(intent_id)
        bfs_queue = [(eid, 0) for eid in bfs_seeds]
        visited = set(bfs_seeds)
        items_explored = 0

        while bfs_queue and items_explored < GRAPH_BUDGET:
            current_id, distance = bfs_queue.pop(0)
            if distance >= _MAX_HOPS:
                continue

            edges = _mcp._STATE.kg.query_entity(current_id, direction="both")
            for e in edges:
                if items_explored >= GRAPH_BUDGET:
                    break
                if not e.get("current", True):
                    continue
                pred = e["predicate"]
                subj = e["subject"]
                obj = e["object"]
                # Skip OUTGOING is_a (don't walk up type hierarchy)
                # Allow INCOMING is_a (find instances: past executions is_a intent_type)
                if pred == "is_a" and subj == current_id:
                    continue

                # Edge-usefulness gating RETIRED (P2). The old
                # edge_traversal_feedback table was dropped in migration
                # 015; the signal it provided is now expressed by
                # context --rated_useful--> memory edges consumed by
                # Channel D at retrieval time. Keeping BFS unfiltered
                # here lets every current edge contribute; the final
                # hybrid-score reranker still applies the signed W_REL
                # term so rated-irrelevant memories sink.

                other = obj if subj == current_id else subj
                if other in visited:
                    continue
                visited.add(other)
                items_explored += 1

                new_dist = distance + 1
                graph_sim = _GRAPH_SIM.get(new_dist, 0.1)

                # ── Channel B score = pure graph-walk signal ──
                # Two-stage retrieval (Nogueira/Cho 2019; Bruch 2023): each
                # channel ranks by its own natural signal, RRF fuses the
                # ranks, and the post-RRF reranker applies the feature-rich
                # hybrid_score. Mixing importance/decay/relevance_feedback
                # into the channel rank would double-count those terms once
                # the post-fusion rerank runs (it applies hybrid_score over
                # every RRF winner). Keep this channel honest by scoring
                # only (distance-based graph_sim, cosine overlap, log-degree
                # dampening). The reranker handles the rest.
                try:
                    _deg = len(_mcp._STATE.kg.query_entity(other, direction="both") or [])
                except Exception:
                    _deg = 0
                import math as _math

                _degree_damp = 1.0 / _math.log(_deg + 2)

                if other.startswith(("record_", "diary_")):
                    _graph_memories.setdefault(other, new_dist)
                    try:
                        col = _mcp._get_collection(create=False)
                        if col:
                            d = col.get(ids=[other], include=["documents", "metadatas"])
                            if d and d["ids"]:
                                score = graph_sim * _degree_damp
                                snippet = (d["documents"][0] or "")[:150].replace("\n", " ")
                                _channel_b_list.append((score, snippet, other))
                    except Exception:
                        pass
                else:
                    _graph_entities.setdefault(other, new_dist)
                    # Track past executions (instances of intent type via is_a)
                    if pred == "is_a" and obj == current_id:
                        _past_exec_ids.append(other)
                    preview = _preview(other)
                    if preview:
                        arrow = "->" if subj == current_id else "<-"
                        text = f'{arrow} {pred} {arrow} {other}: "{preview}"'
                        # effective_sim = max(graph_sim, cosine_sim) keeps
                        # the channel aware of entity-level cosine overlap
                        # without pulling in the reranker's feature space.
                        cosine_sim = _entity_sim.get(other, 0.0)
                        effective_sim = max(graph_sim, cosine_sim)
                        score = effective_sim * _degree_damp
                        _channel_b_list.append((score, text, other))
                    # Continue BFS from entities (not memories)
                    if new_dist < _MAX_HOPS:
                        bfs_queue.append((other, new_dist))

                # Channel B triple emission: emit the traversed edge
                # itself (not just the neighbour entity) so triples get
                # RRF cross-channel boost. Without this, triples only
                # surface via Channel A cosine over mempalace_triples and
                # never accumulate rank contributions from multiple
                # channels the way memories/entities do. Skip-list
                # predicates (schema glue, feedback topology) are
                # excluded — same filter as _index_triple_statement uses
                # at embed time, for the same reason (low-signal text).
                from .knowledge_graph import _TRIPLE_SKIP_PREDICATES

                triple_id = e.get("triple_id")
                statement = e.get("statement")
                if triple_id and statement and pred not in _TRIPLE_SKIP_PREDICATES:
                    triple_text = (statement or "")[:200].replace("\n", " ")
                    _channel_b_list.append((graph_sim * _degree_damp, triple_text, triple_id))
                    _combined_meta[triple_id] = {
                        "meta": {
                            "subject": subj,
                            "predicate": pred,
                            "object": obj,
                            "confidence": e.get("confidence", 1.0),
                        },
                        "doc": triple_text,
                        "similarity": 0.0,
                        "source": "triple",
                    }
    except Exception:
        pass  # Non-fatal

    # Channel C (keyword) is now built INTO multi_channel_search — no
    # separate keyword pass needed. The record_pipe and entity_pipe above
    # already include keyword-ranked lists when _context_keywords is non-empty.

    # ══════════════════════════════════════════════════════════════
    # RRF MERGE — unified across A (cosine) + B (graph) + C (keyword)
    # All channels from both collections compete head-to-head.
    # ══════════════════════════════════════════════════════════════
    all_rrf_lists = dict(_channel_a_lists)
    if _channel_b_list:
        all_rrf_lists["graph"] = _channel_b_list

    # ══════════════════════════════════════════════════════════════
    # Canonical two-stage pipeline (rrf → hybrid_score rerank →
    # adaptive_k) centralised in scoring.two_stage_retrieve. Every
    # context-creating tool routes through this helper so declare_intent
    # / declare_operation / kg_search produce results on the same scale
    # with the same semantics.
    # ══════════════════════════════════════════════════════════════
    from .scoring import two_stage_retrieve as _two_stage

    reranked, rrf_scores, candidate_map = _two_stage(
        all_rrf_lists,
        _combined_meta,
        agent=agent or "",
        session_id=_mcp._STATE.session_id or "",
        intent_type_id=intent_id or "",
        context_feedback=_context_feedback,
        rerank_top_m=50,
        max_k=20,
        min_k=3,
    )

    already_injected = set()
    for r in reranked:
        memory_id = r["id"]
        # Summary-first: every record is written as ``summary\n\ncontent``
        # (Anthropic Contextual Retrieval 2024); _shorten_preview also caps
        # legacy records written before the summary-first gate landed.
        text = _shorten_preview(r["text"])
        already_seen_ids.add(memory_id)
        already_injected.add(memory_id)
        entry = {"id": memory_id, "text": text}
        if DEBUG_RETURN_SCORES:
            # hybrid_score = scoring.hybrid_score output after the post-RRF
            # rerank. Uniform across declare_intent / declare_operation /
            # kg_search — same function, same scale (0.3–0.8).
            entry["hybrid_score"] = round(float(r["hybrid_score"]), 6)
        context["memories"].append(entry)

    # Build past_exec_candidates for promotion check from graph-discovered executions
    past_exec_candidates = []
    for eid in _past_exec_ids:
        rrf_score = rrf_scores.get(eid, 0.0)
        text, _ = candidate_map.get(eid, ("", ""))
        past_exec_candidates.append((rrf_score, text, "graph", eid))

    # ── Mandatory type promotion check: 3+ similar executions ──
    PROMOTION_COUNT = 3
    BASE_THRESHOLD = 0.7
    if len(past_exec_candidates) >= PROMOTION_COUNT:
        parent_threshold = BASE_THRESHOLD
        try:
            type_entity = _mcp._STATE.kg.get_entity(intent_id)
            if type_entity:
                props = type_entity.get("properties", {})
                if isinstance(props, str):
                    props = json.loads(props)
                parent_threshold = props.get("promoted_at_similarity", BASE_THRESHOLD)
        except Exception:
            pass

        if parent_threshold < 1.0:
            # Use score as similarity proxy for promotion check
            high_sim = [c for c in past_exec_candidates if c[0] > parent_threshold]
            if len(high_sim) >= PROMOTION_COUNT:
                avg_sim = sum(c[0] for c in high_sim) / len(high_sim)
                exec_list = "\n".join(f"  - {c[3]}: {c[1][:100]}" for c in high_sim[:5])
                return {
                    "success": False,
                    "error": (
                        f"Intent type '{intent_id}' has {len(high_sim)} similar past executions "
                        f"above threshold {parent_threshold:.2f}. You MUST either:\n\n"
                        f"(a) Create a specific intent type (set promoted_at_similarity={avg_sim:.3f}):\n"
                        f"    "
                        + _mcp._declare_entity_recipe(
                            "<specific-type>",
                            kind="class",
                            hint="what this action does",
                            extra_properties=(
                                f"{{'promoted_at_similarity': {avg_sim:.3f}, "
                                f"'rules_profile': {{...}}}}"
                            ),
                        )
                        + "\n"
                        f"    kg_add(subject='<specific-type>', predicate='is_a', object='{intent_id}', "
                        f"context={{'queries': [...], 'keywords': [...]}})\n"
                        f"    Then re-declare with the specific type.\n\n"
                        f"(b) Disambiguate existing executions (if they're actually different):\n"
                        f"    kg_update_entity(entity='<exec_id>', description='<more specific>', "
                        f"context={{'queries': ['<new meaning>', '<angle 2>'], "
                        f"'keywords': ['<term1>', '<term2>']}})\n\n"
                        f"Similar executions (avg similarity {avg_sim:.3f}):\n{exec_list}"
                    ),
                    "similar_executions": [{"id": c[3], "text": c[1][:100]} for c in high_sim[:5]],
                    "promotion_threshold": parent_threshold,
                    "suggested_promoted_at_similarity": round(avg_sim, 3),
                }

    # ── Hard fail if previous intent not finalized ──
    if _mcp._STATE.active_intent:
        prev_id = _mcp._STATE.active_intent.get("intent_id")
        prev_type = _mcp._STATE.active_intent.get("intent_type", "unknown")
        prev_desc = _mcp._STATE.active_intent.get("description", "")
        return {
            "success": False,
            "error": (
                f"Active intent '{prev_type}' ({prev_id}) has not been finalized. "
                f"You MUST call mempalace_finalize_intent before declaring a new intent. "
                f"Only the agent knows how to properly summarize what happened.\n\n"
                f"Call: mempalace_finalize_intent(\n"
                f"  slug='<descriptive-slug>',\n"
                f"  outcome='success' | 'partial' | 'failed' | 'abandoned',\n"
                f"  content='<full narrative body — what happened in detail>',\n"
                f"  summary='<≤280-char distilled one-liner of the outcome>',\n"
                f"  agent='<your_agent_name>'\n"
                f")\n\n"
                f"Previous intent: {prev_type} — {prev_desc[:100]}"
            ),
            "active_intent": prev_id,
        }

    intent_hash = hashlib.md5(
        f"{intent_id}:{description}:{datetime.now().isoformat()}".encode()
    ).hexdigest()[:12]
    new_intent_id = f"intent_{intent_id}_{intent_hash}"

    # bake a Context-ranked intent_hierarchy ONCE here so the
    # PreToolUse hook has a pre-sorted list and never needs to retrieve.
    # Uses the same 3-channel pipeline as kg_search — no reinvented
    # similarity math.
    context_for_ranking = {
        "queries": list(_description_views),
        "keywords": list(_context_keywords),
    }
    ranked_hierarchy = _build_intent_hierarchy_safe(context_for_ranking)

    # _memory_scoring_snapshot retired (P3 polish): the weight-learning
    # feedback path now reads signals directly from the sim + rel
    # seen_meta + _context_feedback at finalize time rather than a
    # separate snapshot dict on active_intent. Cleaner — fewer
    # persistent fields to maintain.
    #
    # _active_context_id was minted earlier (before the retrieval loops
    # so _relevance_boost could consume context-scoped feedback). No
    # second call here.

    _mcp._STATE.active_intent = {
        "intent_id": new_intent_id,
        "intent_type": intent_id,
        "slots": flat_slots,
        "effective_permissions": permissions,
        "injected_memory_ids": already_injected,
        "accessed_memory_ids": set(),
        "_graph_memories_snapshot": dict(_graph_memories),  # distance map for hop-shortening
        "description": description,
        "_context_views": _views,  # multi-view query strings for context vector storage
        "active_context_id": _active_context_id,  # P1 context-as-entity
        # Every context id touched during this intent (intent-level +
        # any operation/search emits). Enumerated at finalize to build
        # the strict coverage set: every (ctx, memory) surfaced pair
        # must have a rated_* edge or finalize is rejected.
        "contexts_touched": [_active_context_id] if _active_context_id else [],
        # Per-emit detail list — one entry per context emit during the
        # intent's lifecycle. Finalize iterates this to run Rocchio
        # enrichment independently per reused context. Initialised with
        # the intent-level emit; declare_operation + kg_search append
        # their own entries via _record_context_emit.
        "contexts_touched_detail": ([_intent_emit_entry] if _active_context_id else []),
        "agent": agent or "",
        "budget": validated_budget,
        "used": {},  # tool_name -> count, incremented by hook
        "intent_hierarchy": ranked_hierarchy,  # cached, context-ranked
    }

    # Persist to state file for PreToolUse hook (runs in separate process)
    _persist_active_intent()

    _mcp._wal_log(
        "declare_intent",
        {
            "intent_id": new_intent_id,
            "intent_type": intent_id,
            "slots": flat_slots,
            "description": description[:200],
        },
    )

    # feedback_reminder removed 2026-04-21: rules live in wake_up protocol.

    # Ranked subtype suggestions — top 3 that score well AND have required tools
    ranked_suggestions = []
    needed_tools = set(validated_budget.keys()) if validated_budget else set()
    if not narrowed_from and subtypes and description.strip():
        try:
            for cs in sorted(child_scores, key=lambda c: c["distance"])[:10]:
                sim = round(1 - cs["distance"], 3)
                if sim <= 0.1:
                    continue
                # Check if this subtype has the tools we need
                if needed_tools:
                    _, sub_tools = _resolve_intent_profile(cs["id"])
                    sub_tool_names = {t["tool"] for t in sub_tools} if sub_tools else set()
                    if not needed_tools.issubset(sub_tool_names):
                        continue
                ranked_suggestions.append(
                    {
                        "id": cs["id"],
                        "similarity": sim,
                        "description": (cs.get("description") or "")[:100],
                    }
                )
                if len(ranked_suggestions) >= 3:
                    break
        except Exception:
            pass

    # ── Injection-stage gate ──
    # Filter the composed memories list via the Haiku-backed relevance
    # gate before returning to the main agent. Dropped items are
    # persisted as rated_irrelevant feedback (rater_kind='gate_llm')
    # on the active context via kg.record_feedback — entity drops
    # become rated_* edges; triple drops land in
    # triple_context_feedback. No phantom entities. Fail-open: any
    # gate exception passes memories through unchanged.
    _gate_status = None
    try:
        from .injection_gate import apply_gate as _apply_gate

        _gated, _gate_status = _apply_gate(
            memories=context["memories"],
            combined_meta=_combined_meta,
            primary_context={
                "source": "declare_intent",
                "queries": list(_description_views),
                "keywords": list(_context_keywords),
                "entities": list(_context_entities or []),
            },
            context_id=_active_context_id or "",
            kg=_mcp._STATE.kg,
            agent=agent,
            parent_intent=None,  # declare_intent IS the root frame
        )
        context["memories"] = _gated
    except Exception:
        # Any wiring bug must not kill the declare_intent path.
        pass

    # Token-diet response: we deliberately DON'T echo `intent_type`,
    # `slots`, or `budget` — the caller just sent them, and the intent_id
    # itself carries the type (intent_{type}_{hash}). Anyone who genuinely
    # needs the normalized slot values or remaining budget should call
    # mempalace_active_intent, which is the single source of truth for
    # reconstructing the declaration. Keeping the return lean saves ~100
    # tokens per declare on typical intents and prevents tests from
    # coupling to server-side echoes.
    result = {
        "success": True,
        "intent_id": new_intent_id,
        "permissions": [f"{p['tool']}({p.get('scope', '*')})" for p in permissions],
        "memories": context["memories"],
    }
    if _gate_status is not None:
        result["gate_status"] = _gate_status
    if DEBUG_RETURN_CONTEXT:
        # Token-diet 2026-04-23: echo queries ONLY when the context
        # was reused. On a fresh mint the caller just sent them; no
        # reason to bounce them back. On a reuse, the queries were
        # drawn from the stored context entity (often different from
        # what the caller sent), so showing them is informative.
        _ctx_block = {
            "id": _active_context_id,
            "reused": bool(_active_context_reused),
        }
        if _active_context_reused:
            _ctx_block["queries"] = list(_description_views)
        result["context"] = _ctx_block
    if narrowed_from:
        result["narrowed_from"] = narrowed_from
    if ranked_suggestions:
        result["better_intent_types"] = ranked_suggestions
    return result


def tool_active_intent():
    """Return the current active intent, or null if none declared.

    Shows: intent type, permissions, budget remaining. Use this to check what you're currently allowed
    to do before calling a tool.
    """
    _sync_from_disk()
    if not _mcp._STATE.active_intent:
        return {
            "active": False,
            "message": "No active intent. Call mempalace_declare_intent before acting.",
        }
    perms = _mcp._STATE.active_intent["effective_permissions"]
    budget = _mcp._STATE.active_intent.get("budget", {})
    used = _mcp._STATE.active_intent.get("used", {})
    remaining = {k: budget.get(k, 0) - used.get(k, 0) for k in budget}
    return {
        "active": True,
        "intent_id": _mcp._STATE.active_intent["intent_id"],
        "intent_type": _mcp._STATE.active_intent["intent_type"],
        "slots": _mcp._STATE.active_intent.get("slots", {}),
        "permissions": [f"{p['tool']}({p.get('scope', '*')})" for p in perms],
        "budget_remaining": remaining,
    }


def tool_extend_intent(budget: dict, agent: str = None):
    """Extend the active intent's tool budget without redeclaring.

    Use when your budget is exhausted but you're still working on the same task.
    Adds the specified counts to the existing budget.

    Args:
        budget: Dict of tool_name -> additional_calls. E.g. {"Read": 3, "Edit": 2}.
        agent: Your agent name (for logging).
    """
    sid_err = _mcp._require_sid(action="extend_intent")
    if sid_err:
        return sid_err
    _sync_from_disk()
    if not _mcp._STATE.active_intent:
        return {"success": False, "error": "No active intent to extend."}

    if not budget or not isinstance(budget, dict):
        return {"success": False, "error": "budget must be a dict of tool_name -> count."}

    current_budget = _mcp._STATE.active_intent.get("budget", {})

    for tool_name, count in budget.items():
        try:
            n = int(count)
            if n < 1:
                return {"success": False, "error": f"Extension for '{tool_name}' must be >= 1"}
            current_budget[tool_name] = current_budget.get(tool_name, 0) + n
        except (TypeError, ValueError):
            return {
                "success": False,
                "error": f"Extension for '{tool_name}' must be int, got {count!r}",
            }

    _mcp._STATE.active_intent["budget"] = current_budget
    _persist_active_intent()  # Sync to disk for hook

    used = _mcp._STATE.active_intent.get("used", {})
    remaining = {k: current_budget.get(k, 0) - used.get(k, 0) for k in current_budget}

    return {
        "success": True,
        "budget": current_budget,
        "used": used,
        "remaining": remaining,
    }


# ───────────────────────────────────────────────────────────────────────
# Operation-level declaration (2026-04-20)
# ───────────────────────────────────────────────────────────────────────
# Per-tool-call cue declared explicitly by the agent, replacing the
# auto-built cue-from-tool-args that the PreToolUse hook historically
# used. Motivation: 2026-04-20 empirical audit showed ~58% of surfaced
# memories during a normal working session were pure noise, driven by
# the fact that generic cues like "run pytest" / "edit test_file" / "read
# line range" have no topic anchor — nearest-neighbor returns whatever
# past traces also ran pytest / edited tests / read files, regardless of
# topic. Agent-declared queries+keywords raise cue specificity to the
# same bar as declare_intent's Context fingerprint and align with AAO
# (Activity-Action-Operation) hierarchy: the intent is the Activity, the
# tool call is the Operation, and this is where the Operation cue lives.
#
# Retrieval reuses hooks_cli._run_local_retrieval (same multi-view cosine
# + keyword channel + RRF + dedup pipeline the PreToolUse hook already
# uses — no new scoring code). The hook is then responsible for consuming
# the pending_operation_cue and emitting the injected memories as
# additionalContext; see hooks_cli.hook_pretooluse for the consumer.
#
# Enforcement is gated by env MEMPALACE_REQUIRE_DECLARE_OPERATION. When
# off (default during rollout), missing cues fall back to the legacy
# auto-build path so existing sessions don't break. When on, missing
# cues cause the hook to deny the tool call with a recipe. Flip this on
# only after telemetry shows the agent reliably declares.

MIN_OP_QUERIES = 2
MAX_OP_QUERIES = 5
MIN_OP_KEYWORDS = 2
MAX_OP_KEYWORDS = 5
# Mandatory under link-author: every operation lists the entities it
# touches (files it'll read, services it reasons about, agents involved,
# etc.). Capped to keep abuse + the candidate-upsert fanout bounded.
MIN_OP_ENTITIES = 1
MAX_OP_ENTITIES = 10
OP_CUE_TOP_K = 5  # same cap as PreToolUse retrieval today


def tool_declare_operation(
    tool: str,
    context: dict = None,
    agent: str = None,
):
    """Declare the operation (tool call) you are about to perform.

    Mandatory pre-step for every non-carve-out tool call under the
    2026-04-20 cue-quality redesign. The cue you provide drives the
    same retrieval pipeline the PreToolUse hook uses today; memories are
    returned here and the hook also surfaces them as additionalContext
    when the real tool call fires (one-turn lag, identical to today).

    Unified Context shape (same as declare_intent / kg_search / kg_add /
    kg_declare_entity / kg_add_batch — ONE shape for every emit site):

        context = {
          "queries":  [2-5 natural-language perspectives],
          "keywords": [2-5 exact domain terms],
          "entities": [1-10 entity ids the operation touches],
        }

    Args:
        tool: Name of the tool you are about to call (e.g. 'Read', 'Grep',
              'Bash', 'Edit'). Must be permitted under the active intent.
        context: Mandatory unified Context dict. See shape above. Validated
                 by ``scoring.validate_context`` — same validator every
                 other emit site uses, same error messages, same bounds.
        agent: Your agent name.

    Returns:
        {"success": true, "memories": [...], "feedback_reminder": "..."}
        on success. Memories carried through to finalize_intent's
        mandatory memory_feedback coverage via accessed_memory_ids.

    Carve-outs: mempalace_* tools and the ALWAYS_ALLOWED set in
    hooks_cli (TodoWrite, Skill, Agent, ToolSearch, AskUserQuestion,
    Task*, ExitPlanMode) do NOT need declare_operation — they skip
    retrieval entirely. Attempting to declare an operation for one of
    those returns an informative error.
    """
    sid_err = _mcp._require_sid(action="declare_operation")
    if sid_err:
        return sid_err
    _sync_from_disk()
    if not _mcp._STATE.active_intent:
        return {
            "success": False,
            "error": (
                "No active intent. Call mempalace_declare_intent first. "
                "Operation-level declarations live under an Activity-level "
                "intent — you cannot declare an operation with no intent."
            ),
        }

    agent_err = _mcp._require_agent(agent, action="declare_operation")
    if agent_err:
        return agent_err

    # ── Validate tool name ──
    if not isinstance(tool, str) or not tool.strip():
        return {"success": False, "error": "tool must be a non-empty string."}
    tool = tool.strip()

    # Carve-outs: mempalace_* and ALWAYS_ALLOWED skip retrieval, so
    # declaring an operation for them is a no-op at best and confusing
    # at worst. Teach the agent directly.
    try:
        from . import hooks_cli as _hc_mod

        always_allowed = _hc_mod.ALWAYS_ALLOWED_TOOLS
    except Exception:
        always_allowed = set()
    is_mempalace_mcp = tool.startswith("mcp__") and "__mempalace_" in tool
    if tool in always_allowed or is_mempalace_mcp:
        return {
            "success": False,
            "error": (
                f"Tool '{tool}' does not require declare_operation. "
                "mempalace_* tools and ALWAYS_ALLOWED tools (TodoWrite, "
                "Skill, Agent, ToolSearch, AskUserQuestion, Task*, "
                "ExitPlanMode) skip PreToolUse retrieval — just call "
                "them directly."
            ),
        }

    # ── Validate Context — same shared validator every emit site uses ──
    # Bounds (MIN_OP_QUERIES etc.) are passed explicitly so module-level
    # constants stay the authoritative source-of-truth the schema + tests
    # can reference. Matches declare_intent / kg_search / kg_add / etc.
    from .scoring import validate_context as _validate_context

    clean_context, ctx_err = _validate_context(
        context,
        queries_min=MIN_OP_QUERIES,
        queries_max=MAX_OP_QUERIES,
        keywords_min=MIN_OP_KEYWORDS,
        keywords_max=MAX_OP_KEYWORDS,
        entities_min=MIN_OP_ENTITIES,
        entities_max=MAX_OP_ENTITIES,
    )
    if ctx_err:
        return ctx_err
    queries = clean_context["queries"]
    keywords = clean_context["keywords"]
    entities = clean_context["entities"]

    # ── Run retrieval via the SAME pipeline the hook uses today ──
    # _run_local_retrieval handles lazy Chroma import, dedup against
    # accessed_memory_ids, top-K cap, timeout, fail-loud error recording.
    # Reusing it keeps scoring.multi_channel_search the single source of
    # truth for cue → ranked memories.
    from . import hooks_cli as _hc

    cue = {"queries": [q.strip() for q in queries], "keywords": [k.strip() for k in keywords]}
    # Dedup filter: every memory surfaced so far in this intent must be
    # excluded from operation-time retrieval. Two lists carry those ids:
    # accessed_memory_ids (populated by declare_operation and kg_search)
    # and injected_memory_ids (populated by declare_intent). The finalize
    # coverage validator treats them separately so the two must remain
    # distinct for rating purposes, but for "already shown" they are the
    # same signal and must be unioned here. Without the union,
    # declare_operation re-surfaces whatever declare_intent already showed.
    accessed = set(_mcp._STATE.active_intent.get("accessed_memory_ids") or []) | set(
        _mcp._STATE.active_intent.get("injected_memory_ids") or []
    )
    try:
        hits, notice = _hc._run_local_retrieval(cue, accessed, OP_CUE_TOP_K)
    except Exception as _e:
        hits, notice = [], {"fn": "_run_local_retrieval", "error": repr(_e)}

    # ── Context as first-class entity (P1) ──
    # declare_operation is an emit site. A fresh operation cue gets its
    # own context entity: future operations whose cue is MaxSim-similar
    # reuse it. The stored context is the "operation flavour" of the
    # active intent's context — they may be similar (both pertain to the
    # same task) but diverge enough to merit their own accretion.
    # The returned id becomes this operation's active_context_id for any
    # writes that happen during the triggered tool call. We stash it on
    # the pending cue so the hook can later advertise it.
    _op_context_id = ""
    _op_context_reused = False
    try:
        _cid, _reused, _ms = _mcp.context_lookup_or_create(
            queries=cue["queries"],
            keywords=cue["keywords"],
            entities=entities,
            agent=agent or _mcp._STATE.active_intent.get("agent", ""),
        )
        _op_context_id = _cid or ""
        _op_context_reused = bool(_reused)
    except Exception:
        _op_context_id = ""
    # Most-recent-emit precedence: a declare_operation supersedes the
    # intent-level context for any writes that fire between now and the
    # next emit (intent switch, next operation, kg_search).
    if _op_context_id:
        _mcp._STATE.active_intent["active_context_id"] = _op_context_id
        _mcp._record_context_emit(
            _op_context_id,
            reused=_op_context_reused,
            scope="operation",
            queries=cue["queries"],
            keywords=cue["keywords"],
            entities=entities,
        )

    # ── Persist pending_operation_cues (append) + accessed_memory_ids ──
    # The hook pops the first matching-tool entry on the next real tool
    # call, uses it as the retrieval cue (replacing the legacy heuristic
    # cue build), then writes the shortened list back. List form supports
    # parallel tool dispatch: agent can declare N operations in one
    # message and the subsequent N tool calls each consume their own cue.
    # Each cue carries declared_at_ts; the hook expires entries older than
    # OPERATION_CUE_TTL_SECONDS on consume so a forgotten declaration
    # doesn't poison future tool calls indefinitely.
    new_cue = {
        "tool": tool,
        "queries": cue["queries"],
        "keywords": cue["keywords"],
        "declared_at_ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "surfaced_ids": [h.get("id") for h in hits if h.get("id")],
        # P1: context entity id minted for this operation cue. Writers
        # that fire while this cue is the most-recent one use it as
        # active_context_id.
        "active_context_id": _op_context_id,
    }
    existing_cues = _mcp._STATE.active_intent.get("pending_operation_cues") or []
    if not isinstance(existing_cues, list):
        existing_cues = []
    _mcp._STATE.active_intent["pending_operation_cues"] = existing_cues + [new_cue]
    # Single-list design (2026-04-23 decision): every memory surfaced
    # by a declare_operation is added to accessed_memory_ids. It now
    # participates both in within-intent dedup (filter on retrieval)
    # and finalize coverage (the agent must rate it). Sessions with
    # many operation cues will demand many ratings at finalize; that
    # is expected. Background-jury rating is tracked as a separate
    # TODO for later consideration.
    _new_op_ids = [h.get("id") for h in hits if h.get("id")]
    if _new_op_ids:
        _acc_set = _mcp._STATE.active_intent.get("accessed_memory_ids")
        if not isinstance(_acc_set, set):
            _acc_set = set(_acc_set or [])
        _acc_set.update(_new_op_ids)
        _mcp._STATE.active_intent["accessed_memory_ids"] = _acc_set
    _persist_active_intent()

    # ── Build response ──
    # Rules (mandatory-coverage, fetch-full-via-kg_query, declare-gate)
    # live in the wake_up protocol — we no longer repeat them in every
    # operation response. See wake_up's protocol string for the contract.
    memories = []
    for h in hits:
        entry = {
            "id": h["id"],
            "text": _shorten_preview((h.get("preview") or "").strip()),
        }
        if DEBUG_RETURN_SCORES:
            entry["hybrid_score"] = round(float(h.get("score", 0.0) or 0.0), 6)
        memories.append(entry)

    # ── Injection-stage gate ──
    # Same wiring as declare_intent: filter memories via the Haiku
    # relevance gate, persist drops as rated_irrelevant feedback
    # (rater_kind='gate_llm'), fail-open on any bug. Parent frame =
    # the active intent (this operation is nested under it).
    _gate_status = None
    try:
        from .injection_gate import apply_gate as _apply_gate

        # hits provides a per-id lookup of the source + channel when we
        # have metadata; otherwise apply_gate infers from id prefix.
        _op_combined_meta = {
            h["id"]: {
                "source": ("triple" if str(h.get("id", "")).startswith("t_") else "memory"),
                "doc": (h.get("preview") or "").strip(),
                "similarity": float(h.get("score", 0.0) or 0.0),
            }
            for h in hits
            if h.get("id")
        }
        _parent_intent = None
        try:
            ai = _mcp._STATE.active_intent or {}
            _parent_intent = {
                "intent_type": ai.get("intent_type"),
                "subject": ", ".join((ai.get("slots", {}) or {}).get("subject", []) or []),
                "query": (ai.get("description_views") or [""])[0],
            }
        except Exception:
            _parent_intent = None

        _gated, _gate_status = _apply_gate(
            memories=memories,
            combined_meta=_op_combined_meta,
            primary_context={
                "source": "declare_operation",
                "queries": list(cue["queries"]),
                "keywords": list(cue["keywords"]),
                "entities": list(entities or []),
            },
            context_id=_op_context_id or "",
            kg=_mcp._STATE.kg,
            agent=agent,
            parent_intent=_parent_intent,
        )
        memories = _gated
    except Exception:
        pass

    result = {"success": True, "memories": memories}
    if _gate_status is not None:
        result["gate_status"] = _gate_status
    if DEBUG_RETURN_CONTEXT:
        # Token-diet 2026-04-23: echo queries ONLY on reuse. See the
        # matching block in tool_declare_intent for the rationale.
        _ctx_block = {
            "id": _op_context_id,
            "reused": bool(_op_context_reused),
        }
        if _op_context_reused:
            _ctx_block["queries"] = list(cue["queries"])
        result["context"] = _ctx_block
    if notice:
        # Fail-loud: retrieval error surfaces to agent, not silent.
        result["retrieval_notice"] = notice
    return result


# Single-field relevance → (relevant, confidence) mapping.
#
# Agents supply only ``relevance: 1-5`` in memory_feedback entries (per the
# 2026-04-22 API cutover). The server derives both the rated_useful /
# rated_irrelevant sign and the confidence-on-the-edge magnitude from the
# integer. The mapping preserves the symmetry of the pre-cutover two-field
# shape:
#
#     relevance 1 → relevant=False, confidence 1.0 → signal -1.0
#     relevance 2 → relevant=False, confidence 0.5 → signal -0.5
#     relevance 3 → relevant=True,  confidence 0.2 → signal +0.2   (weak-positive floor)
#     relevance 4 → relevant=True,  confidence 0.8 → signal +0.8
#     relevance 5 → relevant=True,  confidence 1.0 → signal +1.0
#
# Design rationale: relevance is inherently subjective — there is no
# ground truth, only "this memory helped THIS task for THIS agent".
# CrowdTruth 2.0 (Aroyo & Welty) and Davani et al. 2022 make the case
# for preserving disagreement as signal rather than collapsing to a
# scalar. The mapping above keeps the full signed [-1, +1] dynamic
# range that Channel D + hybrid_score's W_REL term consume, with 3 as
# the "default when unsure" anchor that contributes a small positive
# signal (legitimate since the agent marked it related-but-not-decisive).
#
# Callers may still pass ``relevant`` explicitly to override the derived
# sign (back-compat + tests). If they do, we respect it and use the
# derived confidence for magnitude.
_RELEVANCE_MAPPING = {
    1: (False, 1.0),
    2: (False, 0.5),
    3: (True, 0.2),
    4: (True, 0.8),
    5: (True, 1.0),
}


def _derive_feedback_pair(fb: dict) -> tuple[int, bool, float]:
    """Resolve a memory_feedback entry to ``(relevance_int, relevant, confidence)``.

    Reads ``fb["relevance"]`` (1-5), coerces out-of-range values to 3
    (default "related context" per the schema), and applies the mapping
    above. The ``relevant`` bool is DERIVED exclusively from the integer;
    there is no override path. Single signed scale is the whole API.
    """
    raw = fb.get("relevance", 3)
    try:
        score = int(raw)
    except (TypeError, ValueError):
        score = 3
    if score < 1 or score > 5:
        score = 3
    relevant, confidence = _RELEVANCE_MAPPING[score]
    return score, relevant, confidence


def _coerce_list_param(name: str, val):
    """Normalize an MCP list-shaped param, guarding against stringified JSON.

    Mirrors the guard already used by ``tool_resolve_conflicts``. Some MCP
    transports (and the Opus planner under load) serialize a top-level
    array argument as a JSON string. A naive ``for item in val`` then walks
    characters and emits one bogus error per char — the same bug that
    could balloon a response to ~61k chars of per-char entries.

    Returns ``(coerced, err_response)``. If ``err_response`` is not None the
    caller must return it unmodified. ``None`` and real lists pass through
    untouched.
    """
    if val is None or isinstance(val, list):
        return val, None
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
        except Exception:
            return None, {
                "success": False,
                "error": (
                    f"`{name}` arrived as an unparseable JSON string. "
                    f"Pass a JSON array of objects, not a string."
                ),
            }
        if not isinstance(parsed, list):
            return None, {
                "success": False,
                "error": (
                    f"`{name}` parsed from string must be a list, got {type(parsed).__name__}."
                ),
            }
        return parsed, None
    return None, {
        "success": False,
        "error": f"`{name}` must be a list, got {type(val).__name__}.",
    }


def tool_finalize_intent(  # noqa: C901
    slug: str,
    outcome: str,
    content: str,
    summary: str,
    agent: str,
    memory_feedback: list = None,
    key_actions: list = None,
    gotchas: list = None,
    learnings: list = None,
    promote_gotchas_to_type: bool = False,
):
    """Finalize the active intent — capture what happened as structured memory.

    MUST be called before declaring a new intent or exiting the session.
    Creates an execution entity (kind=entity, is_a intent_type) with
    relationships linking it to the agent, targets, result memory, gotchas,
    and execution trace.

    VOCABULARY — uniform across every record-write boundary in mempalace:
      ``content`` = full narrative body. FREE LENGTH — as detailed as needed.
        Stored verbatim.
      ``summary`` = ≤280-char distillation / reframe. ALWAYS required (no
        length threshold on content — every record gets a summary). For
        long content the summary distills the WHAT/WHY; for short content
        the summary should REPHRASE the same fact from a different angle
        (different keywords / framing) so the summary+content pair yields
        two distinct cosine views of the same semantic — real retrieval
        gain, not redundancy. Anthropic Contextual Retrieval (2024)
        prepends the summary to the content before embedding (single CR
        vector); the summary is also what injection-time previews display.
        No prefix slicing. No auto-derivation. The caller produces it.

    Args:
        slug: Human-readable ID for this execution (e.g. 'edit-auth-rate-limiter-2026-04-14')
        outcome: 'success', 'partial', 'failed', or 'abandoned'
        content: Full outcome narrative — the body of the result memory. Any
            length. Becomes the embedded document (with summary prepended).
        summary: ≤280-char distilled one-liner of the outcome (or a
            different-angle rephrase when content is short). Shown in
            injections and prepended to content for embedding.
        agent: Agent entity name (e.g. 'technical_lead_agent')
        memory_feedback: MANDATORY — MAP SHAPE ONLY. Contextual
            relevance feedback for every memory accessed during this
            intent, scoped by the CONTEXT that surfaced it:

              {
                "<context_id>": [
                  {"id": "<memory_id>", "relevant": true/false,
                   "relevance": 1-5, "reason": "why (>=10 chars)"},
                  ...
                ],
                ...
              }

            Flat-list form is rejected — it loses the per-context
            attribution the writer needs to attach rated_useful /
            rated_irrelevant edges to the right context entity
            (which Channel D reads on future intents). Context ids
            are returned from declare_intent / declare_operation /
            kg_search and tracked on active_intent.contexts_touched.
        key_actions: Abbreviated tool+params list (optional — auto-filled from trace if omitted)
        gotchas: List of gotcha descriptions discovered during execution
        learnings: List of lesson descriptions worth remembering
        promote_gotchas_to_type: Also link gotchas to the intent type (not just execution)
    """

    # Sid check FIRST — an empty sid means the tool call came in without
    # hook-injected sessionId, which makes every downstream state op a
    # potential cross-agent contamination risk. Fail loud at the boundary.
    sid_err = _mcp._require_sid(action="finalize_intent")
    if sid_err:
        return sid_err

    # ── Summary-first gate: strict validation at the boundary ──
    # Mirrors _add_memory_internal's ≤280-char rule. Enforced HERE (not
    # only inside the downstream result_memory upsert) because the old
    # behaviour collected the downstream rejection into `errors` and
    # returned success=True — so a 299-char summary would finalize the
    # intent, create the execution entity, but leave no result memory,
    # letting the caller assume everything was fine. Every method that
    # accepts a summary rejects over-length up front and fails the call.
    # Keep this in lockstep with _add_memory_internal so the two rules
    # never drift.
    if not isinstance(summary, str):
        return {
            "success": False,
            "error": (
                f"`summary` must be a string (got {type(summary).__name__}). "
                f"Pass a ≤{_mcp._RECORD_SUMMARY_MAX_LEN}-char distilled "
                f"one-liner of the outcome."
            ),
        }
    _summary_clean = summary.strip()
    if not _summary_clean:
        return {
            "success": False,
            "error": (
                f"`summary` is required (≤{_mcp._RECORD_SUMMARY_MAX_LEN} "
                f"chars). One-sentence distillation of the outcome — names "
                f"the WHAT and WHY, no filler."
            ),
        }
    if len(_summary_clean) > _mcp._RECORD_SUMMARY_MAX_LEN:
        return {
            "success": False,
            "error": (
                f"`summary` is {len(_summary_clean)} chars; maximum is "
                f"{_mcp._RECORD_SUMMARY_MAX_LEN}. Distill further — one "
                f"sentence, names the WHAT and WHY, no filler."
            ),
        }
    summary = _summary_clean

    # memory_feedback contract: MAP SHAPE ONLY (flat list retired).
    #   {context_id: [{id, relevant, relevance, reason, ...}, ...]}
    # Each entry is scoped to the context that surfaced it. The writer
    # attaches rated_useful/rated_irrelevant edges FROM the context TO
    # the memory — Channel D reads those edges on future intents. A
    # flat list would lose per-context attribution (the writer couldn't
    # decide which context the rated_* edge belongs to), so the map is
    # load-bearing, not cosmetic.
    _memory_feedback_by_context: dict = {}
    if memory_feedback is None:
        memory_feedback = {}
    if isinstance(memory_feedback, str):
        # Stringified-JSON delivery — parse loudly.
        try:
            memory_feedback = json.loads(memory_feedback)
        except Exception:
            return {
                "success": False,
                "error": (
                    "memory_feedback arrived as an unparseable string. "
                    "Pass a dict {context_id: [{id, relevant, relevance, reason}, ...]}"
                ),
            }
    if isinstance(memory_feedback, list):
        return {
            "success": False,
            "error": (
                "memory_feedback flat-list shape is retired. Use the map: "
                "{context_id: [{id, relevant, relevance, reason}, ...]}. "
                "Each entry attributes to the context that surfaced the "
                "memory (intent/operation/search). Channel D reads edges "
                "scoped to that context, so the map form is load-bearing."
            ),
        }
    if not isinstance(memory_feedback, dict):
        return {
            "success": False,
            "error": (
                "memory_feedback must be a dict "
                "{context_id: [{id, relevant, relevance, reason}, ...]}. "
                f"Got {type(memory_feedback).__name__}."
            ),
        }
    flat: list = []
    for ctx_id, entries in memory_feedback.items():
        if not isinstance(entries, list):
            return {
                "success": False,
                "error": (
                    "memory_feedback map shape: each value must be a list "
                    "of entry dicts. Got value of type "
                    f"{type(entries).__name__} for context_id {ctx_id!r}."
                ),
            }
        for e in entries:
            if isinstance(e, dict):
                e2 = dict(e)
                e2.setdefault("_context_id", str(ctx_id))
                flat.append(e2)
                _memory_feedback_by_context.setdefault(str(ctx_id), []).append(e2)
    memory_feedback = flat
    gotchas, _pe = _coerce_list_param("gotchas", gotchas)
    if _pe:
        return _pe
    learnings, _pe = _coerce_list_param("learnings", learnings)
    if _pe:
        return _pe
    key_actions, _pe = _coerce_list_param("key_actions", key_actions)
    if _pe:
        return _pe

    _sync_from_disk()
    if not _mcp._STATE.active_intent:
        return {"success": False, "error": "No active intent to finalize."}

    # fail-fast agent validation. Before P6.1 an undeclared agent
    # would silently break result/trace/learning memory creation deep
    # inside _add_memory_internal; now we reject upfront with the same
    # recipe the hook teaches.
    agent_err = _mcp._require_agent(agent, action="finalize_intent")
    if agent_err:
        return agent_err

    intent_type = _mcp._STATE.active_intent["intent_type"]
    intent_desc = _mcp._STATE.active_intent.get("description", "")
    slot_entities = []
    for slot_name, slot_vals in _mcp._STATE.active_intent.get("slots", {}).items():
        if isinstance(slot_vals, list):
            slot_entities.extend(slot_vals)
        elif isinstance(slot_vals, str):
            slot_entities.append(slot_vals)

    # Normalize slug
    exec_id = normalize_entity_name(slug)
    if not exec_id:
        return {"success": False, "error": "slug normalizes to empty."}

    # ── Validate memory feedback reason field ──
    MIN_FEEDBACK_REASON = 10
    if memory_feedback:
        for fb in memory_feedback:
            reason = (fb.get("reason") or "").strip()
            if len(reason) < MIN_FEEDBACK_REASON:
                return {
                    "success": False,
                    "error": (
                        f"Memory feedback for '{fb.get('id', '?')}' missing or has too short 'reason' "
                        f"(minimum {MIN_FEEDBACK_REASON} characters). Each feedback entry must explain "
                        f"WHY the memory was or wasn't relevant to THIS intent."
                    ),
                }

    # ── Validate memory feedback coverage ──
    injected_ids = {x for x in _mcp._STATE.active_intent.get("injected_memory_ids", set()) if x}
    accessed_ids = {x for x in _mcp._STATE.active_intent.get("accessed_memory_ids", set()) if x}

    feedback_ids = set()
    if memory_feedback:
        for fb in memory_feedback:
            raw_id = (fb.get("id") or "").strip()
            if raw_id:
                # Store both raw and normalized forms so either matches
                feedback_ids.add(raw_id)
                feedback_ids.add(normalize_entity_name(raw_id))

    # Injected memories: 100% feedback required
    if injected_ids:
        missing_injected = injected_ids - feedback_ids
        if missing_injected:
            coverage = (len(injected_ids) - len(missing_injected)) / len(injected_ids)
            return {
                "success": False,
                "error": (
                    f"Insufficient memory feedback for THIS INTENT. {len(missing_injected)} of "
                    f"{len(injected_ids)} injected memories have no feedback (100% required). "
                    f"Rate each memory's relevance TO THE CURRENT INTENT (1-5 scale). "
                    f"Review these before rating: {sorted(missing_injected)}"
                ),
                "missing_injected": sorted(missing_injected),
                "missing_accessed": [],
                "feedback_coverage": {"injected": round(coverage, 2), "accessed": 0},
            }

    # Accessed memories: 100% feedback required (excluding already-covered injected)
    MIN_ACCESSED_COVERAGE = 1.0
    accessed_only = accessed_ids - injected_ids
    if accessed_only:
        accessed_covered = len(accessed_only & feedback_ids)
        accessed_coverage = accessed_covered / len(accessed_only)
        if accessed_coverage < MIN_ACCESSED_COVERAGE:
            missing_accessed = sorted(accessed_only - feedback_ids)
            return {
                "success": False,
                "error": (
                    f"Insufficient memory feedback for THIS INTENT. Only {accessed_covered}/{len(accessed_only)} "
                    f"accessed memories rated ({accessed_coverage:.0%}, minimum {MIN_ACCESSED_COVERAGE:.0%}). "
                    f"Rate each memory's relevance TO THE CURRENT INTENT (1-5 scale). "
                    f"Missing: {missing_accessed}"
                ),
                "missing_injected": [],
                "missing_accessed": missing_accessed,
                "feedback_coverage": {"injected": 1.0, "accessed": round(accessed_coverage, 2)},
            }

    # ── Read execution trace from hook state file ──
    trace_entries = []
    if not _mcp._STATE.session_id:
        # No sid means we never had a private trace file. Skipping is
        # correct — falling back to execution_trace_default.jsonl would
        # pull another agent's trace into THIS agent's finalize.
        trace_file = None
    else:
        trace_file = _mcp._INTENT_STATE_DIR / f"execution_trace_{_mcp._STATE.session_id}.jsonl"
    try:
        if trace_file and trace_file.exists():
            with open(trace_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        trace_entries.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        pass
            # Clear trace file after reading
            trace_file.write_text("", encoding="utf-8")
    except Exception:
        pass

    # Auto-fill key_actions from trace if not provided
    if not key_actions and trace_entries:
        key_actions = [f"{e['tool']} {e.get('target', '')}".strip() for e in trace_entries[-20:]]

    # ── Create execution entity ──
    # Full description stored in SQLite (for display)
    # Execution-entity description shows the distilled summary directly —
    # summary is already ≤280 chars by construction, no slicing needed.
    exec_description = f"{intent_desc or intent_type}: {summary}"
    # Embedding uses description-only (no summary) so similar intents cluster
    embed_description = intent_desc or intent_type
    try:
        _mcp._create_entity(
            exec_id,
            kind="entity",
            description=exec_description,
            importance=3,
            properties={
                "outcome": outcome,
                "agent": agent,
                "added_by": agent,
                "intent_type": intent_type,
                "finalized_at": datetime.now().isoformat(),
            },
            added_by=agent,
            embed_text=embed_description,  # description-only, no summary
        )
    except Exception as e:
        return {"success": False, "error": f"Failed to create execution entity: {e}"}

    # ── KG relationships ──
    edges_created = []

    # is_a → intent type (entity is_a class = instantiation)
    try:
        _mcp._STATE.kg.add_triple(exec_id, "is_a", intent_type)
        edges_created.append(f"{exec_id} is_a {intent_type}")
    except Exception:
        pass

    # executed_by → agent
    try:
        _mcp._STATE.kg.add_triple(exec_id, "executed_by", agent)
        edges_created.append(f"{exec_id} executed_by {agent}")
    except Exception:
        pass

    # targeted → slot entities
    for target in slot_entities:
        try:
            target_id = normalize_entity_name(target)
            _mcp._STATE.kg.add_triple(exec_id, "targeted", target_id)
            edges_created.append(f"{exec_id} targeted {target_id}")
        except Exception:
            pass

    # outcome as has_value
    try:
        _mcp._STATE.kg.add_triple(exec_id, "has_value", outcome)
        edges_created.append(f"{exec_id} has_value {outcome}")
    except Exception:
        pass

    # ── Result memory (summary) ──
    # silent-failure surface: when _add_memory_internal rejects the
    # call (e.g. agent not declared, duplicate slug), we used to swallow
    # the error and return result_memory=null with no indication. Now
    # every failure is appended to `errors` and surfaced in the response.
    errors: list = []
    result_memory_id = None
    try:
        # Result memory: the body is the agent's `content` wrapped with an
        # intent/outcome header; the ≤280-char distilled summary is the
        # agent's `summary` verbatim. No slicing, no auto-derivation — the
        # summary-first contract requires the caller to have produced a
        # real distillation, and we honor it here.
        _result_body = f"## {intent_type}: {intent_desc}\n\n**Outcome:** {outcome}\n\n{content}"
        result = _mcp._add_memory_internal(
            content=_result_body,
            slug=f"result-{exec_id}",
            added_by=agent,
            content_type="event",
            importance=3,
            entity=exec_id,
            predicate="resulted_in",
            summary=summary,
        )
        if result.get("success"):
            result_memory_id = result.get("memory_id")
            edges_created.append(f"{exec_id} resulted_in {result_memory_id}")
        else:
            errors.append({"kind": "result_memory", "error": result.get("error", "unknown")})
    except Exception as e:
        errors.append({"kind": "result_memory", "error": f"exception: {e}"})

    # ── Trace memory ── (retired 2026-04-22)
    # Traces used to be filed as ``record_ga_agent_trace_<slug>`` prose
    # memories with importance=2 and a count-of-tool-calls summary. They
    # polluted retrieval — every finalize added another "Trace of X: N
    # tool call(s)" hit competing with actual prose. The same information
    # is already available without the embedded memory:
    #   - execution_trace_<sid>.jsonl on disk (the raw tool-call log,
    #     cleared after finalize reads it)
    #   - key_actions on the execution entity (distilled tool+target
    #     list, auto-filled from the trace above)
    #   - edges on the execution entity (executed_by, targeted, is_a)
    # If you ever need the blow-by-blow, read the JSONL file between
    # finalizes — do not re-introduce a prose memory for it.

    # ── Gotchas ──
    if gotchas:
        for gotcha_desc in gotchas:
            try:
                gotcha_id = normalize_entity_name(gotcha_desc[:50])
                if gotcha_id:
                    # Check if gotcha entity exists, create if not
                    existing = _mcp._STATE.kg.get_entity(gotcha_id)
                    if not existing:
                        _mcp._create_entity(
                            gotcha_id,
                            kind="entity",
                            description=gotcha_desc,
                            importance=3,
                            added_by=agent,
                        )
                    # has_gotcha is NOT a skip predicate; we need a real
                    # statement so the edge is searchable. The gotcha's
                    # own description is a natural sentence for this.
                    gotcha_sentence = f"Execution {exec_id} ran into this gotcha: {gotcha_desc}"
                    _mcp._STATE.kg.add_triple(
                        exec_id,
                        "has_gotcha",
                        gotcha_id,
                        statement=gotcha_sentence,
                    )
                    edges_created.append(f"{exec_id} has_gotcha {gotcha_id}")
                    if promote_gotchas_to_type:
                        type_sentence = (
                            f"Intent type '{intent_type}' has a recurring gotcha: {gotcha_desc}"
                        )
                        _mcp._STATE.kg.add_triple(
                            intent_type,
                            "has_gotcha",
                            gotcha_id,
                            statement=type_sentence,
                        )
                        edges_created.append(f"{intent_type} has_gotcha {gotcha_id}")
            except Exception:
                pass

    # ── Learnings ──
    if learnings:
        for i, learning in enumerate(learnings):
            try:
                # Learnings accept two shapes, both honoring the summary-first
                # contract (summary always required, no auto-derivation):
                #   • str — a one-liner learning of ≤280 chars. Used as both
                #     content AND summary (the record IS its own distillation
                #     at that length). Strings longer than 280 are rejected
                #     here with a clear error pointing at the dict form.
                #   • dict{content, summary} — explicit distillation. Use
                #     for any learning whose body exceeds ~280 chars, or
                #     whenever you want the summary to reframe the content
                #     from a different angle for two-view retrieval.
                if isinstance(learning, dict):
                    _l_content = str(learning.get("content") or "").strip()
                    _l_summary = str(learning.get("summary") or "").strip()
                elif isinstance(learning, str):
                    _l_content = learning.strip()
                    if len(_l_content) > _mcp._RECORD_SUMMARY_MAX_LEN:
                        errors.append(
                            {
                                "kind": "learning_memory",
                                "index": i,
                                "error": (
                                    f"learning[{i}] is {len(_l_content)} "
                                    f"chars; as a bare string it must be "
                                    f"≤{_mcp._RECORD_SUMMARY_MAX_LEN} "
                                    f"(used as both content and summary). "
                                    f"For longer learnings use "
                                    f"dict{{content, summary}}."
                                ),
                            }
                        )
                        continue
                    _l_summary = _l_content
                else:
                    errors.append(
                        {
                            "kind": "learning_memory",
                            "index": i,
                            "error": (
                                "learning must be str (≤280 chars) or "
                                "dict{content, summary}; got "
                                f"{type(learning).__name__}"
                            ),
                        }
                    )
                    continue
                learning_result = _mcp._add_memory_internal(
                    content=_l_content,
                    slug=f"learning-{exec_id}-{i}",
                    added_by=agent,
                    content_type="discovery",
                    importance=4,
                    entity=exec_id,
                    predicate="evidenced_by",
                    summary=_l_summary,
                )
                if not learning_result.get("success"):
                    errors.append(
                        {
                            "kind": "learning_memory",
                            "index": i,
                            "error": learning_result.get("error", "unknown"),
                        }
                    )
            except Exception as e:
                errors.append({"kind": "learning_memory", "index": i, "error": f"exception: {e}"})

    # ── Strict coverage validator (context-scoped) ──
    # Enumerate every (context, memory) pair with a `surfaced` edge
    # written during this intent. Every such pair MUST have a matching
    # rated_useful or rated_irrelevant entry in memory_feedback (map
    # shape) or memory_feedback must cover the memory in a way that
    # maps to the active context (flat-list fallback). Missing coverage
    # → finalize rejected with the exact list of unresolved pairs.
    #
    # Adrian's design rule: "suggestion is the DEATH of whatever is
    # suggested with LLMs" — every feedback path must be blocking,
    # never advisory. This is the P2 §12 "coverage validator" the plan
    # demanded.
    _contexts_touched = list(_mcp._STATE.active_intent.get("contexts_touched") or [])
    _required_pairs = set()  # {(context_id, memory_id), ...}
    for _ctx_id in _contexts_touched:
        if not _ctx_id:
            continue
        try:
            _ctx_edges = _mcp._STATE.kg.query_entity(_ctx_id, direction="outgoing")
        except Exception:
            continue
        for _e in _ctx_edges:
            if not _e.get("current", True):
                continue
            if _e.get("predicate") != "surfaced":
                continue
            # query_entity returns entities.name (raw caller-supplied
            # name) as `object`; triples.object stores the normalized
            # id. The covered-pairs side normalizes feedback ids via
            # normalize_entity_name, so normalize here too — otherwise
            # any raw name whose normalized form differs (e.g. the
            # multi-view `foo__v1` suffix collapsing to `foo_v1`)
            # becomes an unreachable required pair and finalize is
            # stuck complaining forever.
            _mid = normalize_entity_name(_e.get("object") or "")
            if _mid:
                _required_pairs.add((_ctx_id, _mid))

    _covered_pairs = set()
    _active_ctx_id = _mcp._STATE.active_intent.get("active_context_id", "") or ""
    if memory_feedback:
        for fb in memory_feedback:
            if not isinstance(fb, dict):
                continue
            _fb_mid = normalize_entity_name(fb.get("id", ""))
            if not _fb_mid:
                continue
            # Map-shape entries carry _context_id; list-shape entries
            # default to the active intent-level context so the legacy
            # flat form still satisfies coverage when there's a single
            # active context.
            _fb_ctx = fb.get("_context_id") or _active_ctx_id
            if _fb_ctx:
                _covered_pairs.add((_fb_ctx, _fb_mid))
            # A list-shape entry with no active_ctx covers ALL contexts
            # for that memory — best-effort fallback to preserve the
            # permissive legacy behaviour when contexts_touched was
            # empty.
            if not _fb_ctx:
                for _c in _contexts_touched:
                    _covered_pairs.add((_c, _fb_mid))

    _missing_pairs = sorted(_required_pairs - _covered_pairs)
    if _missing_pairs:
        _preview_list = [{"context_id": c, "memory_id": m} for c, m in _missing_pairs[:20]]
        return {
            "success": False,
            "error": (
                "Insufficient memory_feedback coverage. "
                f"{len(_missing_pairs)} (context, memory) pair(s) surfaced "
                "during this intent have no rating. Every surfaced memory "
                "must be rated useful OR irrelevant (plus a reason) for "
                "the context that surfaced it. Pass memory_feedback as "
                "a map {context_id: [{id, relevant, relevance, reason}, "
                "...]} so each rating attributes to the context that "
                "surfaced the memory. missing_pairs shows up to 20 "
                "unresolved pairs."
            ),
            "missing_pairs_count": len(_missing_pairs),
            "missing_pairs": _preview_list,
        }

    # ── Memory relevance feedback ──
    #
    # P2: two write paths run in this block:
    #   (1) Legacy found_useful / found_irrelevant edges attached to the
    #       EXECUTION entity — kept intact so the existing retrieval
    #       machinery that still reads them during the cutover window
    #       doesn't see a sudden signal drop.
    #   (2) NEW rated_useful / rated_irrelevant edges attached to the
    #       CONTEXT that surfaced the memory (or, when the flat-list
    #       shape is used, the active_context_id for this intent). These
    #       are what Channel D reads on subsequent intents.
    # The dual-write is the cutover bridge; a later step retires (1).
    feedback_count = 0
    if memory_feedback:
        for fb in memory_feedback:
            try:
                raw_id = fb.get("id", "")
                mem_id = normalize_entity_name(raw_id)
                if not mem_id:
                    continue
                relevance_score, relevant, confidence = _derive_feedback_pair(fb)

                # ── Write the rated_* edge on the active context ──
                # Source context: from the map shape if provided, else the
                # active intent's context id. Skip when neither is present.
                # (Legacy found_useful / found_irrelevant edges on the
                # execution entity + the promote_to_type flag were retired
                # in the P3 polish sweep — context-scoped feedback is the
                # only signal the retrieval pipeline reads now.)
                #
                # Routes through kg.record_feedback — the unified
                # dispatcher covers both entity-scope and triple-scope
                # feedback with the same last-wins-across-directions
                # contract (migration 018 added triple_context_feedback
                # for triple targets because the entity-only object
                # namespace of rated_* edges silently created phantom
                # entities for triple ids). target_kind is detected from
                # the id prefix: add_triple ids start with 't_'; every
                # other id is in the entities namespace (records are
                # kind='record' entities). See record_feedback and
                # add_rated_edge docstrings for the four failure modes
                # the supersede contract closes.
                ctx_source = fb.get("_context_id") or _active_ctx_id
                if ctx_source:
                    target_kind = "triple" if mem_id.startswith("t_") else "entity"
                    rated_pred = "rated_useful" if relevant else "rated_irrelevant"
                    try:
                        _mcp._STATE.kg.record_feedback(
                            ctx_source,
                            mem_id,
                            target_kind,
                            relevance=int(relevance_score),
                            reason=str(fb.get("reason", "") or ""),
                            rater_kind="agent",
                            rater_id=agent or "",
                            confidence=confidence,
                        )
                        edges_created.append(f"{ctx_source} {rated_pred} {mem_id}")
                    except Exception:
                        pass  # Non-fatal

                # Reset decay for useful memories by updating last_relevant_at.
                # Chroma stores the ORIGINAL hyphenated id, not the KG-normalized
                # (underscored) form, so look up by raw_id with mem_id as a fallback
                # for callers that already normalized. Without this split, the
                # update silently misses (outer try/except swallows the empty get).
                if relevant:
                    try:
                        col = _mcp._get_collection(create=False)
                        if col:
                            lookup_ids = [raw_id] if raw_id and raw_id != mem_id else [mem_id]
                            if raw_id and raw_id != mem_id:
                                lookup_ids.append(mem_id)  # Fallback for pre-normalized callers
                            existing = col.get(ids=lookup_ids, include=["metadatas"])
                            if existing and existing["ids"]:
                                chroma_id = existing["ids"][0]
                                meta = existing["metadatas"][0] or {}
                                meta["last_relevant_at"] = datetime.now().isoformat()
                                col.update(ids=[chroma_id], metadatas=[meta])
                    except Exception:
                        pass  # Non-fatal — decay reset is best-effort

                feedback_count += 1
            except Exception:
                pass

    # ── Link-prediction candidate upsert ──
    # For each reused context with net-positive per-context feedback,
    # accumulate Adamic-Adar evidence (1/log(|entities|)) on every
    # unordered entity pair inside that context. Dedup by (pair, ctx_id)
    # so re-observing the same context N times contributes exactly once.
    # Direct-edge short-circuit inside upsert_candidate drops pairs
    # already connected 1-hop in any direction — the graph channel
    # already finds them. Consumed offline by the `mempalace link-author`
    # CLI (Commit 3). Wrapped in a try/except so a schema or DB failure
    # never blocks finalize. See docs/link_author_plan.md §2.4 + §5.2.
    try:
        import math

        from . import link_author as _la

        _CANDIDATE_MEAN_REL_CUT = 4.0
        detail = _mcp._STATE.active_intent.get("contexts_touched_detail") or []
        for entry in detail:
            if not isinstance(entry, dict):
                continue
            if not entry.get("reused"):
                continue
            ctx_id = entry.get("ctx_id") or ""
            if not ctx_id:
                continue
            ents = [e for e in (entry.get("entities") or []) if isinstance(e, str) and e]
            if len(ents) < 2:
                continue
            # Per-context mean-relevance gate. Contexts with no feedback
            # attributed to them (fresh or skipped by the agent) don't
            # contribute — positive signal requires positive ratings.
            fb_entries = _memory_feedback_by_context.get(ctx_id) or []
            rels = [
                int(e.get("relevance"))
                for e in fb_entries
                if isinstance(e, dict) and isinstance(e.get("relevance"), (int, float))
            ]
            if not rels:
                continue
            if sum(rels) / len(rels) < _CANDIDATE_MEAN_REL_CUT:
                continue
            # Textbook Adamic-Adar weight: 1 / log(|entities|). len>=2
            # guarantees log(n) >= log(2) ≈ 0.693 — no zero-div risk and
            # no +1 fudge (which would underweight small focused contexts,
            # the opposite of what AA wants).
            weight = 1.0 / math.log(len(ents))
            for i, a in enumerate(ents):
                for b in ents[i + 1 :]:
                    try:
                        _la.upsert_candidate(
                            _mcp._STATE.kg,
                            from_entity=a,
                            to_entity=b,
                            weight=weight,
                            context_id=ctx_id,
                        )
                    except Exception:
                        # Per-pair failure must not abort the whole loop.
                        continue
    except Exception:
        # Schema missing / import error / anything else: never block
        # finalize on the candidate accumulator. Logged at DEBUG level
        # is overkill for now; Commit 3 adds proper error recording.
        pass

    # ── Finalize-triggered background dispatch (stub in Commit 2) ──
    try:
        from . import link_author as _la  # noqa: F811

        _la._dispatch_if_due(_mcp._STATE.kg, interval_hours=1)
    except Exception:
        pass

    # ── Rocchio enrichment: per-context + per-channel gating ──
    # Each context that was REUSED during this intent's lifecycle gets
    # its OWN Rocchio evaluation. Within each reused context, the
    # three enrichment fields (queries / keywords / entities) are
    # gated INDEPENDENTLY by the channel that consumes them:
    #   - queries  → Channel A (cosine) feedback
    #   - keywords → Channel C (keyword) feedback
    #   - entities → Channel B (graph) feedback
    # Channel D (context) doesn't map to any Rocchio field — it surfaces
    # memories via similar_to neighbourhood, which is orthogonal to
    # this context's own view/keyword/entity shape. So Channel-D
    # memories are skipped from the per-field buckets.
    #
    # Fallback: if no channel attribution is available for any rated
    # memory (pure flat-list feedback on memories not surfaced this
    # intent), fall back to an aggregate mean ≥ 4.0 check for all
    # three fields so Rocchio doesn't silently die on edge cases.
    #
    # Rationale (Rocchio 1971, Manning/Raghavan/Schütze IR book Ch.9):
    # the enrichment shift is per-query-axis, not aggregate. A bad
    # keyword set shouldn't drag down the enrichment of good queries,
    # and vice versa.
    try:
        detail = _mcp._STATE.active_intent.get("contexts_touched_detail") or []
        # TODO (threshold): both the 4.0 enrichment mean cut-off and
        # the per-channel _MIN_BUCKET=2 floor are hand-tuned. Outcome
        # signal = "did the enriched context land more future reuses
        # at the right MaxSim than the un-enriched baseline?" Learn by
        # A/B-ing enrichment decisions for the same context signature
        # and sweeping both parameters. Low signal per decision;
        # needs 200+ finalizes for meaningful statistics.
        _MIN_BUCKET = 2  # need >=2 memories from a channel to trust its mean
        _ROCCHIO_MEAN_REL_CUT = 4.0
        for entry in detail:
            if not isinstance(entry, dict) or not entry.get("reused"):
                continue
            ctx_id = entry.get("ctx_id")
            if not ctx_id:
                continue

            # Map field -> channel name used to bucket relevances.
            FIELD_CHANNEL = {"queries": "cosine", "keywords": "keyword", "entities": "graph"}
            buckets = {"cosine": [], "keyword": [], "graph": []}
            all_relevances = []  # aggregate fallback

            for _fb in memory_feedback or []:
                if not isinstance(_fb, dict):
                    continue
                _fb_ctx = _fb.get("_context_id") or ""
                # Match feedback entry to this context (map-shape
                # directly, flat-list only when active ctx == this
                # ctx at the time of ALL emits — permissive).
                matches = _fb_ctx == ctx_id or (not _fb_ctx and _active_ctx_id == ctx_id)
                if not matches:
                    continue
                # Rocchio bucket: use the derived (relevant, confidence)
                # pair so single-field callers behave identically to
                # legacy two-field callers. Value for bucket is the raw
                # 1-5 score when relevant, else 0 — matching historical
                # behaviour where "irrelevant" contributes no weight.
                _score, _relevant, _ = _derive_feedback_pair(_fb)
                rel_val = float(_score) if _relevant else 0.0
                all_relevances.append(rel_val)

                # Look up the surfaced edge's channel attribution. The
                # newer shape stores a comma-joined set in ``channels``
                # (so multi-channel hits contribute to EVERY channel
                # they fired through); the legacy ``channel`` is a
                # single string fallback. "mixed"/"" from either
                # doesn't map to any Rocchio field.
                try:
                    fb_mid = normalize_entity_name(_fb.get("id", ""))
                    if not fb_mid:
                        continue
                    srow = (
                        _mcp._STATE.kg._conn()
                        .execute(
                            "SELECT properties FROM triples "
                            "WHERE subject=? AND predicate='surfaced' "
                            "AND object=? AND (valid_to IS NULL OR valid_to='')",
                            (ctx_id, fb_mid),
                        )
                        .fetchone()
                    )
                    attributed: list = []
                    if srow and srow[0]:
                        try:
                            props_obj = json.loads(srow[0]) or {}
                        except Exception:
                            props_obj = {}
                        chans_str = props_obj.get("channels") or ""
                        attributed = [c.strip() for c in str(chans_str).split(",") if c.strip()]
                    for ch in attributed:
                        if ch in buckets:
                            buckets[ch].append(rel_val)
                except Exception:
                    continue

            # Decide per-field. Each field enriches only when its
            # channel has enough signal AND that signal is net-positive.
            enrich_queries = False
            enrich_keywords = False
            enrich_entities = False
            any_channel_attributed = any(len(b) > 0 for b in buckets.values())
            if any_channel_attributed:
                if (
                    len(buckets["cosine"]) >= _MIN_BUCKET
                    and sum(buckets["cosine"]) / len(buckets["cosine"]) >= _ROCCHIO_MEAN_REL_CUT
                ):
                    enrich_queries = True
                if (
                    len(buckets["keyword"]) >= _MIN_BUCKET
                    and sum(buckets["keyword"]) / len(buckets["keyword"]) >= _ROCCHIO_MEAN_REL_CUT
                ):
                    enrich_keywords = True
                if (
                    len(buckets["graph"]) >= _MIN_BUCKET
                    and sum(buckets["graph"]) / len(buckets["graph"]) >= _ROCCHIO_MEAN_REL_CUT
                ):
                    enrich_entities = True
            else:
                # Aggregate fallback when no channel attribution exists
                # (edge case; common when memory_feedback references
                # memories the agent added inline rather than ones
                # surfaced by retrieval).
                if (
                    all_relevances
                    and sum(all_relevances) / len(all_relevances) >= _ROCCHIO_MEAN_REL_CUT
                ):
                    enrich_queries = enrich_keywords = enrich_entities = True

            if not (enrich_queries or enrich_keywords or enrich_entities):
                continue

            _ = FIELD_CHANNEL  # silence linter
            try:
                _mcp.rocchio_enrich_context(
                    ctx_id,
                    new_queries=(entry.get("queries") or []) if enrich_queries else [],
                    new_keywords=(entry.get("keywords") or []) if enrich_keywords else [],
                    new_entities=(entry.get("entities") or []) if enrich_entities else [],
                )
            except Exception:
                pass
    except Exception:
        pass

    # ── Record scoring component feedback for weight learning ──
    if memory_feedback:
        from .scoring import compute_age_days, DEFAULT_SEARCH_WEIGHTS

        for fb in memory_feedback:
            try:
                raw_id = fb.get("id", "")
                mem_id = normalize_entity_name(raw_id)
                if not mem_id:
                    continue
                _, relevant, _ = _derive_feedback_pair(fb)
                # Look up metadata to compute component values. Chroma stores the
                # original hyphenated id; try raw_id first, fall back to the KG-
                # normalized form for callers that pre-normalized.
                lookup_ids = [raw_id] if raw_id else [mem_id]
                if raw_id and raw_id != mem_id:
                    lookup_ids.append(mem_id)
                meta = {}
                try:
                    col = _mcp._get_collection(create=False)
                    if col:
                        d = col.get(ids=lookup_ids, include=["metadatas"])
                        if d and d["ids"]:
                            meta = d["metadatas"][0] or {}
                except Exception:
                    pass
                if not meta:
                    try:
                        ecol = _mcp._get_entity_collection(create=False)
                        if ecol:
                            d = ecol.get(ids=lookup_ids, include=["metadatas"])
                            if d and d["ids"]:
                                meta = d["metadatas"][0] or {}
                    except Exception:
                        pass

                imp = float(meta.get("importance", 3))
                date_iso = meta.get("date_added") or meta.get("filed_at") or ""
                last_rel = meta.get("last_relevant_at") or ""
                age_days = compute_age_days(date_iso, last_rel)
                agent_match = bool(agent and meta.get("added_by") == agent)

                # P3 polish: sim and rel signals now come from the
                # shared rated-walk the hybrid_score reranker used at
                # search time. When they're not available (e.g. agent
                # gave feedback on a memory they added inline, with no
                # retrieval pass), default to 0.
                _context_feedback = _mcp._STATE.active_intent.get("_context_feedback_dict") or {}
                rel_raw = float(_context_feedback.get(mem_id, 0.0) or 0.0)
                # sim isn't tracked separately post-retirement; W_SIM
                # learning still works because finalize feedback on
                # retrieved memories provides the signal correlation.
                sim_val = 0.0

                components = {
                    "sim": max(0.0, min(1.0, sim_val)),
                    "rel": max(0.0, min(1.0, (rel_raw + 1.0) / 2.0)),  # signed [-1,+1] -> [0,1]
                    "imp": (imp - 1.0) / 4.0,
                    "decay": max(0.0, min(1.0, 1.0 / (1.0 + age_days / 30.0))),
                    "agent": 1.0 if agent_match else 0.0,
                }
                _mcp._STATE.kg.record_scoring_feedback(components, relevant)

                # ── Per-channel RRF weight feedback ──
                # Which channels surfaced this memory? Read the `channel`
                # prop from surfaced edges on any touched context. Binary
                # presence per channel: the channels that fired get 1.0,
                # the others 0.0. Correlated with `relevant` over time,
                # the channel-weight learner shifts weight toward
                # channels whose hits actually earned useful ratings.
                try:
                    import json as _json

                    channels_hit = set()
                    for _ctx_id in _contexts_touched:
                        if not _ctx_id:
                            continue
                        try:
                            row = (
                                _mcp._STATE.kg._conn()
                                .execute(
                                    "SELECT properties FROM triples "
                                    "WHERE subject=? AND predicate='surfaced' "
                                    "AND object=? AND (valid_to IS NULL OR valid_to='')",
                                    (_ctx_id, mem_id),
                                )
                                .fetchone()
                            )
                            if row and row[0]:
                                props = _json.loads(row[0]) or {}
                                chans_str = props.get("channels") or ""
                                for c in str(chans_str).split(","):
                                    c = c.strip()
                                    if c:
                                        channels_hit.add(c)
                        except Exception:
                            continue
                    if channels_hit:
                        channel_components = {
                            "cosine": 1.0 if "cosine" in channels_hit else 0.0,
                            "graph": 1.0 if "graph" in channels_hit else 0.0,
                            "keyword": 1.0 if "keyword" in channels_hit else 0.0,
                            "context": 1.0 if "context" in channels_hit else 0.0,
                        }
                        _mcp._STATE.kg.record_scoring_feedback(
                            channel_components, relevant, scope="channel"
                        )
                except Exception:
                    pass
            except Exception:
                pass

        # Update learned weights — both scopes.
        try:
            from .scoring import (
                set_learned_weights,
                set_learned_channel_weights,
                DEFAULT_CHANNEL_WEIGHTS,
            )

            learned_hybrid = _mcp._STATE.kg.compute_learned_weights(
                DEFAULT_SEARCH_WEIGHTS, scope="hybrid"
            )
            set_learned_weights(learned_hybrid)
            learned_channels = _mcp._STATE.kg.compute_learned_weights(
                DEFAULT_CHANNEL_WEIGHTS, scope="channel"
            )
            set_learned_channel_weights(learned_channels)
        except Exception:
            pass

    # Feedback context vectors (store_feedback_context), edge-traversal
    # feedback (record_edge_feedback), and keyword-suppression feedback
    # (record_keyword_suppression / reset_keyword_suppression) are all
    # RETIRED in the P3 polish sweep. Their signals now live on:
    #   - rated_useful / rated_irrelevant edges → Channel D retrieval +
    #     hybrid_score's signed W_REL term.
    #   - keyword_idf table → BM25-IDF dampens dominant keywords
    #     channel-wide (replacing per-memory suppression).
    #   - created_under / similar_to context edges → Channel D's
    #     neighbourhood expansion replaces edge-usefulness gating.
    # This block used to reach for all three retired APIs.

    # ── Deactivate intent ──
    _mcp._STATE.active_intent = None
    _persist_active_intent()

    # ── Write last-finalized marker for Stop-hook proof-of-done check ──
    # The never-stop rule requires the Stop hook to see that the LAST finalized
    # intent in this session was a wrap_up_session with outcome=success before
    # allowing a stop. Writing a session-scoped marker here gives the dep-free
    # hook a file to read without needing SQLite or Chroma. Best-effort — any
    # error is non-fatal to the finalize itself.
    try:
        sid = _mcp._STATE.session_id or ""
        if sid:
            marker_path = _mcp._INTENT_STATE_DIR / f"last_finalized_{sid}.json"
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text(
                json.dumps(
                    {
                        "intent_type": intent_type,
                        "execution_entity": exec_id,
                        "outcome": outcome,
                        "agent": agent,
                        "ts": datetime.now().isoformat(),
                    }
                ),
                encoding="utf-8",
            )
    except Exception as _e:
        # NEVER silent: a failed marker write means the Stop hook's
        # never-stop rule will block the next stop attempt forever (reads
        # missing/stale marker as "no wrap-up proof"). Record for
        # SessionStart to surface.
        try:
            from . import hooks_cli as _hc

            _hc._record_hook_error("tool_finalize_intent.last_finalized_marker", _e)
        except Exception:
            pass

    result = {
        "success": True,
        "execution_entity": exec_id,
        "outcome": outcome,
        "edges_created_count": len(edges_created),
        "trace_entries": len(trace_entries),
        "result_memory": result_memory_id,
        "feedback_count": feedback_count,
    }

    # ── Memory-gardener detached spawn ──
    # If the injection gate has accumulated enough quality flags on
    # memory_flags, kick off a gardener subprocess so the finalize
    # caller isn't blocked on Claude Code latency. Mirrors the
    # link-author finalize-triggered detached pattern. Fail-silent:
    # a spawn failure must not block finalize.
    try:
        from . import memory_gardener as _mg

        _mg.maybe_trigger_from_finalize(_mcp._STATE.kg)
    except Exception:
        pass

    # ── P3 telemetry: finalize trace for mempalace-eval ──
    try:
        from datetime import timezone as _tz

        contexts_used = sorted(set(_memory_feedback_by_context.keys()))
        if _active_ctx_id and _active_ctx_id not in contexts_used:
            contexts_used.append(_active_ctx_id)
        _mcp._telemetry_append_jsonl(
            "finalize_log.jsonl",
            {
                "ts": datetime.now(_tz.utc).isoformat(timespec="seconds"),
                "intent_id": exec_id,
                "contexts_used": contexts_used,
                "memories_rated": feedback_count,
                "outcome": outcome,
                "agent": agent or "",
            },
        )
    except Exception:
        pass
    if errors:
        result["errors"] = errors
        result["warning"] = (
            f"{len(errors)} side-memory creation(s) failed silently before "
            "see 'errors' for details. The execution entity itself was created and "
            "feedback/gotchas were recorded; only the filed memories were affected."
        )
    return result
