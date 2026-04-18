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

from .knowledge_graph import normalize_entity_name
from .scoring import adaptive_k, rrf_merge

# Module reference (set by init())
_mcp = None


def init(mcp_module):
    """Wire this module to mcp_server so we can access its globals/functions."""
    global _mcp
    _mcp = mcp_module


# ==================== INTENT DECLARATION ====================
# Note: _active_intent and _INTENT_STATE_DIR live in mcp_server.py
# so that test monkeypatching continues to work (tests patch mcp_server.*).
# We access them exclusively via _mcp._active_intent / _mcp._INTENT_STATE_DIR.


def _intent_state_path() -> Path:
    """Get session-scoped intent state file path."""
    return _mcp._INTENT_STATE_DIR / f"active_intent_{_mcp._session_id or 'default'}.json"


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
        edges = _mcp._kg.query_entity(eid, direction="outgoing")
        parent_id = None
        for e in edges:
            if e["predicate"] in ("is-a", "is_a") and e["current"]:
                obj = normalize_entity_name(e["object"])
                if obj == "intent_type":
                    parent_id = "intent-type"
                    break
                # Check if parent is itself an intent type
                parent_edges = _mcp._kg.query_entity(obj, direction="outgoing")
                for pe in parent_edges:
                    if pe["predicate"] in ("is-a", "is_a") and pe["current"]:
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
            kg=_mcp._kg,
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
    """Reload active intent state from disk — hook may have updated used counts."""
    try:
        state_file = _intent_state_path()
        if state_file.is_file():
            data = json.loads(state_file.read_text(encoding="utf-8"))
            if data.get("intent_id") and _mcp._active_intent:
                # Only sync if same intent — don't load a stale one
                if data["intent_id"] == _mcp._active_intent["intent_id"]:
                    _mcp._active_intent["used"] = data.get("used", {})
                    _mcp._active_intent["budget"] = data.get("budget", {})
    except Exception:
        pass  # Non-fatal


def _persist_active_intent():
    """Write active intent to session-scoped state file for PreToolUse hook."""
    try:
        _mcp._INTENT_STATE_DIR.mkdir(parents=True, exist_ok=True)
        state_file = _intent_state_path()
        if _mcp._active_intent:
            # the Context-ranked hierarchy is computed ONCE at
            # declare_intent time and cached on _active_intent. Subsequent
            # persists (extend_intent, finalize_intent) just re-serialize
            # the cached version — no repeat 3-channel work per tool call.
            cached_hierarchy = _mcp._active_intent.get("intent_hierarchy")
            if cached_hierarchy is None:
                cached_hierarchy = _build_intent_hierarchy_safe()
            state = {
                "intent_id": _mcp._active_intent["intent_id"],
                "intent_type": _mcp._active_intent["intent_type"],
                "slots": _mcp._active_intent["slots"],
                "effective_permissions": _mcp._active_intent["effective_permissions"],
                "description": _mcp._active_intent.get("description", ""),
                "agent": _mcp._active_intent.get("agent", ""),
                "session_id": _mcp._session_id,
                "intent_hierarchy": cached_hierarchy,
                "injected_memory_ids": list(_mcp._active_intent.get("injected_memory_ids", set())),
                "accessed_memory_ids": list(_mcp._active_intent.get("accessed_memory_ids", set())),
                "budget": _mcp._active_intent.get("budget", {}),
                "used": _mcp._active_intent.get("used", {}),
                "pending_conflicts": getattr(_mcp, "_pending_conflicts", None) or [],
                "pending_enrichments": getattr(_mcp, "_pending_enrichments", None) or [],
            }
            state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
        else:
            if state_file.exists():
                state_file.unlink()
    except OSError:
        pass  # Non-fatal


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

        entity = _mcp._kg.get_entity(current)
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
        edges = _mcp._kg.query_entity(current, direction="outgoing")
        parent = None
        for e in edges:
            if e["predicate"] in ("is-a", "is_a") and e["current"]:
                parent_id = normalize_entity_name(e["object"])
                # Stop at the root intent_type class
                if parent_id == "intent_type":
                    break
                # Skip universal base class — not part of intent hierarchy
                if parent_id == "thing":
                    continue
                parent_entity = _mcp._kg.get_entity(parent_id)
                if parent_entity and parent_entity.get("kind") == "class":
                    parent = parent_id
                    break
        if not parent:
            break
        current = parent

    return merged_slots, merged_tools


def _is_intent_type(entity_id: str) -> bool:
    """Check if an entity is-a intent_type (direct or inherited)."""

    edges = _mcp._kg.query_entity(entity_id, direction="outgoing")
    for e in edges:
        if e["predicate"] in ("is-a", "is_a") and e["current"]:
            obj = normalize_entity_name(e["object"])
            if obj == "intent_type":
                return True
            # Check parent (one level — e.g., edit_file is-a modify is-a intent_type)
            parent_edges = _mcp._kg.query_entity(obj, direction="outgoing")
            for pe in parent_edges:
                if pe["predicate"] in ("is-a", "is_a") and pe["current"]:
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
    agent_err = _mcp._require_agent(agent, action="declare_intent")
    if agent_err:
        return agent_err

    # ── Check for pending conflicts ──
    # Disk is source of truth — reload from disk if memory is empty (MCP restart scenario)
    pending_conflicts = getattr(_mcp, "_pending_conflicts", None)
    if not pending_conflicts and hasattr(_mcp, "_load_pending_conflicts_from_disk"):
        pending_conflicts = _mcp._load_pending_conflicts_from_disk() or None
        if pending_conflicts:
            _mcp._pending_conflicts = pending_conflicts
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

    # ── Check for pending enrichments ──
    pending_enrichments = getattr(_mcp, "_pending_enrichments", None)
    if not pending_enrichments and hasattr(_mcp, "_load_pending_enrichments_from_disk"):
        pending_enrichments = _mcp._load_pending_enrichments_from_disk() or None
        if pending_enrichments:
            _mcp._pending_enrichments = pending_enrichments
    if pending_enrichments:
        return {
            "success": False,
            "error": (
                f"{len(pending_enrichments)} graph enrichment tasks pending. "
                f"You MUST resolve ALL before declaring a new intent. For each: "
                f"call kg_add(subject, predicate, object) to create the edge with "
                f"a predicate you choose, then call mempalace_resolve_enrichments "
                f"to mark them done. Or call mempalace_resolve_enrichments with "
                f"action='reject' and a mandatory reason to reject."
            ),
            "pending_enrichments": pending_enrichments,
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
    all_entities = _mcp._kg.list_entities(status="active", kind="class")
    for e in all_entities:
        e_edges = _mcp._kg.query_entity(e["id"], direction="outgoing")
        for edge in e_edges:
            if edge["predicate"] in ("is-a", "is_a") and edge["current"]:
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
                            _mcp._declared_entities.add(intent_id)
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
                f"Example: {{{', '.join(f'"{k}": ["entity_name"]' for k in effective_slots)}}}"
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
                    _mcp._create_entity(
                        file_basename,
                        kind="entity",
                        description=f"File: {val}" + (" (new)" if not file_exists else ""),
                        importance=2,
                        added_by=agent,
                    )
                    _mcp._kg.add_triple(val_id, "is-a", "file")
                    _mcp._declared_entities.add(val_id)
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
                    for e in _mcp._kg.query_entity(val_id, direction="outgoing")
                    if e["predicate"] in ("is-a", "is_a") and e["current"]
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
                                for e in _mcp._kg.query_entity(cls, direction="outgoing"):
                                    if e["predicate"] in ("is-a", "is_a") and e["current"]:
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
        entity = _mcp._kg.get_entity(entity_id)
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
    from .scoring import hybrid_score as _score_fn

    # ── Type-level relevance feedback, confidence-graded ──
    # Every memory_feedback entry captures a 1-5 relevance score which
    # finalize_intent stores as confidence = relevance/5.0 ∈ [0.2, 1.0]
    # on the found_useful / found_irrelevant edge. Here we read it back
    # and map to the continuous [-1.0, +1.0] range hybrid_score expects:
    #
    #   relevance 5 useful      → confidence 1.0 → boost +1.0
    #   relevance 1 useful      → confidence 0.2 → boost +0.2
    #   no feedback             → 0.0 (neutral)
    #   relevance 1 irrelevant  → confidence 0.2 → penalty -0.2
    #   relevance 5 irrelevant  → confidence 1.0 → penalty -1.0
    #
    # Previously irrelevant was always -1.0 regardless of confidence — the
    # docstring in hybrid_score advertised "magnitude = confidence" but
    # the negative side ignored it. Fixed.
    _type_feedback = {}  # memory_id -> float ∈ [-1, +1]
    try:
        type_edges = _mcp._kg.query_entity(intent_id, direction="outgoing")
        for te in type_edges:
            if not te.get("current", True):
                continue
            conf = te.get("confidence", 1.0)
            try:
                conf = max(0.0, min(1.0, float(conf)))
            except (TypeError, ValueError):
                conf = 1.0
            if te["predicate"] == "found_useful":
                _type_feedback[te["object"]] = conf
            elif te["predicate"] == "found_irrelevant":
                _type_feedback[te["object"]] = -conf
    except Exception:
        pass

    def _relevance_boost(memory_id):
        """Return continuous relevance signal from type feedback.

        Returns float in [-1.0, +1.0]:
          +1.0 = strongly useful (relevance 5)
          +0.2 = weakly useful (relevance 1)
          0.0  = no feedback
          -0.2 = weakly irrelevant
          -1.0 = strongly irrelevant (relevance 5 irrelevant)
        """
        return _type_feedback.get(memory_id, 0.0)

    def _preview(entity_id_or_memory):
        """Get text preview for any ID — memory content or entity description."""
        if entity_id_or_memory.startswith("drawer_"):
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
                ent = _mcp._kg.get_entity(entity_id_or_memory)
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
            ent = _mcp._kg.get_entity(entity_id)
            if ent and ent.get("description"):
                _views.append(ent["description"][:200])
        except Exception:
            pass
    _views = list(dict.fromkeys(_views))[:6]
    if not _views:
        _views = [intent_id or "unknown"]

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

    # Record collection (prose records — the old "memory" collection)
    try:
        dcol = _mcp._get_collection(create=False)
        if dcol:
            record_pipe = _scoring.multi_channel_search(
                dcol,
                _views,
                keywords=_context_keywords,
                kg=_mcp._kg,
                fetch_limit_per_view=50,
                include_graph=False,
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
                kg=_mcp._kg,
                fetch_limit_per_view=50,
                include_graph=False,
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
    _traversed_edges = []  # for feedback recording
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

            edges = _mcp._kg.query_entity(current_id, direction="both")
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
                if pred in ("is-a", "is_a") and subj == current_id:
                    continue

                # Check edge usefulness — skip if strongly negative
                # Uses contextual MaxSim when context vectors are available
                try:
                    # Try contextual feedback first
                    usefulness = 0.0
                    ctx_ids = _mcp._kg.get_context_ids_for_edge(subj, pred, obj)
                    if ctx_ids and _views:
                        matches = _mcp.maxsim_context_match(_views, ctx_ids)
                        if matches:
                            # Use the most similar context's feedback
                            best_cid = max(matches, key=matches.get)
                            usefulness = _mcp._kg.get_edge_usefulness(
                                subj, pred, obj, context_id=best_cid
                            )
                        else:
                            # No contextual match — fall back to intent_type
                            usefulness = _mcp._kg.get_edge_usefulness(
                                subj, pred, obj, intent_type=intent_id
                            )
                    else:
                        usefulness = _mcp._kg.get_edge_usefulness(
                            subj, pred, obj, intent_type=intent_id
                        )
                    if usefulness < _MIN_EDGE_USEFULNESS:
                        continue
                except Exception:
                    pass

                other = obj if subj == current_id else subj
                if other in visited:
                    continue
                visited.add(other)
                items_explored += 1
                _traversed_edges.append((subj, pred, obj))

                new_dist = distance + 1
                graph_sim = _GRAPH_SIM.get(new_dist, 0.1)

                if other.startswith("drawer_"):
                    _graph_memories.setdefault(other, new_dist)
                    try:
                        col = _mcp._get_collection(create=False)
                        if col:
                            d = col.get(ids=[other], include=["documents", "metadatas"])
                            if d and d["ids"]:
                                meta = d["metadatas"][0] or {}
                                score = _score_fn(
                                    similarity=graph_sim,
                                    importance=float(meta.get("importance", 3)),
                                    date_iso=meta.get("date_added") or meta.get("filed_at") or "",
                                    agent_match=bool(agent and meta.get("added_by") == agent),
                                    last_relevant_iso=meta.get("last_relevant_at") or "",
                                    relevance_feedback=_relevance_boost(other),
                                    mode="search",
                                )
                                snippet = (d["documents"][0] or "")[:150].replace("\n", " ")
                                _channel_b_list.append((score, snippet, other))
                    except Exception:
                        pass
                else:
                    _graph_entities.setdefault(other, new_dist)
                    # Track past executions (instances of intent type via is_a)
                    if pred in ("is-a", "is_a") and obj == current_id:
                        _past_exec_ids.append(other)
                    # Score entity — combine graph distance with cosine similarity
                    preview = _preview(other)
                    if preview:
                        imp = (
                            5.0
                            if pred in ("has_gotcha", "must", "must_not", "requires", "forbids")
                            else 3.0
                        )
                        arrow = "->" if subj == current_id else "<-"
                        text = f'{arrow} {pred} {arrow} {other}: "{preview}"'
                        # Use max of graph_sim and entity cosine similarity
                        cosine_sim = _entity_sim.get(other, 0.0)
                        effective_sim = max(graph_sim, cosine_sim)
                        score = _score_fn(
                            similarity=effective_sim,
                            importance=imp,
                            date_iso="",
                            relevance_feedback=_relevance_boost(other),
                            mode="search",
                        )
                        _channel_b_list.append((score, text, other))
                    # Continue BFS from entities (not memories)
                    if new_dist < _MAX_HOPS:
                        bfs_queue.append((other, new_dist))
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

    rrf_scores, candidate_map, _channel_attribution = rrf_merge(all_rrf_lists)

    # Sort by RRF score, apply adaptive-K
    rrf_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    already_injected = set()
    if len(rrf_ranked) > 1:
        final_k = adaptive_k([s for _, s in rrf_ranked], max_k=20, min_k=3)
    else:
        final_k = len(rrf_ranked)

    for memory_id, rrf_score in rrf_ranked[:final_k]:
        text, channel = candidate_map.get(memory_id, ("", "unknown"))
        already_seen_ids.add(memory_id)
        already_injected.add(memory_id)
        context["memories"].append({"id": memory_id, "text": text})

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
            type_entity = _mcp._kg.get_entity(intent_id)
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
    if _mcp._active_intent:
        prev_id = _mcp._active_intent.get("intent_id")
        prev_type = _mcp._active_intent.get("intent_type", "unknown")
        prev_desc = _mcp._active_intent.get("description", "")
        return {
            "success": False,
            "error": (
                f"Active intent '{prev_type}' ({prev_id}) has not been finalized. "
                f"You MUST call mempalace_finalize_intent before declaring a new intent. "
                f"Only the agent knows how to properly summarize what happened.\n\n"
                f"Call: mempalace_finalize_intent(\n"
                f"  slug='<descriptive-slug>',\n"
                f"  outcome='success' | 'partial' | 'failed' | 'abandoned',\n"
                f"  summary='<what happened>',\n"
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

    _mcp._active_intent = {
        "intent_id": new_intent_id,
        "intent_type": intent_id,
        "slots": flat_slots,
        "effective_permissions": permissions,
        "injected_memory_ids": already_injected,
        "accessed_memory_ids": set(),
        "traversed_edges": _traversed_edges,  # for edge feedback in finalize
        "_graph_memories_snapshot": dict(_graph_memories),  # distance map for hop-shortening
        "_channel_attribution": {k: list(v) for k, v in _channel_attribution.items()},
        "description": description,
        "_context_views": _views,  # multi-view query strings for context vector storage
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

    feedback_reminder = None
    injected_count = len(already_injected)
    if injected_count:
        feedback_reminder = (
            f"IMPORTANT: {injected_count} memories were injected for this intent. "
            f"READ and USE these memories — they were selected as relevant to your task. "
            f"If a memory contains information needed for your work, apply it. "
            f"You MUST provide feedback on ALL of them (relevance 1-5 TO THIS INTENT) "
            f"when calling finalize_intent. Finalization will FAIL without 100% coverage."
        )

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
        "feedback_reminder": feedback_reminder,
    }
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
    if not _mcp._active_intent:
        return {
            "active": False,
            "message": "No active intent. Call mempalace_declare_intent before acting.",
        }
    perms = _mcp._active_intent["effective_permissions"]
    budget = _mcp._active_intent.get("budget", {})
    used = _mcp._active_intent.get("used", {})
    remaining = {k: budget.get(k, 0) - used.get(k, 0) for k in budget}
    return {
        "active": True,
        "intent_id": _mcp._active_intent["intent_id"],
        "intent_type": _mcp._active_intent["intent_type"],
        "slots": _mcp._active_intent.get("slots", {}),
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
    _sync_from_disk()
    if not _mcp._active_intent:
        return {"success": False, "error": "No active intent to extend."}

    if not budget or not isinstance(budget, dict):
        return {"success": False, "error": "budget must be a dict of tool_name -> count."}

    current_budget = _mcp._active_intent.get("budget", {})

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

    _mcp._active_intent["budget"] = current_budget
    _persist_active_intent()  # Sync to disk for hook

    used = _mcp._active_intent.get("used", {})
    remaining = {k: current_budget.get(k, 0) - used.get(k, 0) for k in current_budget}

    return {
        "success": True,
        "budget": current_budget,
        "used": used,
        "remaining": remaining,
    }


def tool_finalize_intent(  # noqa: C901
    slug: str,
    outcome: str,
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

    Args:
        slug: Human-readable ID for this execution (e.g. 'edit-auth-rate-limiter-2026-04-14')
        outcome: 'success', 'partial', 'failed', or 'abandoned'
        summary: What happened — broader result narrative. Becomes a memory.
        agent: Agent entity name (e.g. 'technical_lead_agent')
        memory_feedback: MANDATORY — contextual relevance feedback for ALL memories
            accessed during this intent. Include memories injected by declare_intent,
            memories you found via search, AND any new memories you created.
            Each entry: {"id": "memory_id_or_entity_id", "relevant": true/false,
            "relevance": 1-5, "promote_to_type": false, "reason": "why"}.
            promote_to_type controls whether the feedback propagates to future
            declares of the same intent type:
              - true  → the edge is attached to the intent TYPE entity (e.g. 'modify').
                        `_relevance_boost` reads type-entity edges on every future
                        declare of that type and uses the signal to rerank
                        injection. Set true when the rating generalizes (clearly
                        relevant or clearly irrelevant, and the reason is about
                        the task SHAPE rather than this specific instance).
              - false → the edge is attached only to this execution entity. No
                        future declare will see it; the signal is effectively
                        diary-only for retrieval purposes. Use when the rating
                        is genuinely instance-specific.
        key_actions: Abbreviated tool+params list (optional — auto-filled from trace if omitted)
        gotchas: List of gotcha descriptions discovered during execution
        learnings: List of lesson descriptions worth remembering
        promote_gotchas_to_type: Also link gotchas to the intent type (not just execution)
    """

    _sync_from_disk()
    if not _mcp._active_intent:
        return {"success": False, "error": "No active intent to finalize."}

    # fail-fast agent validation. Before P6.1 an undeclared agent
    # would silently break result/trace/learning memory creation deep
    # inside _add_memory_internal; now we reject upfront with the same
    # recipe the hook teaches.
    agent_err = _mcp._require_agent(agent, action="finalize_intent")
    if agent_err:
        return agent_err

    intent_type = _mcp._active_intent["intent_type"]
    intent_desc = _mcp._active_intent.get("description", "")
    slot_entities = []
    for slot_name, slot_vals in _mcp._active_intent.get("slots", {}).items():
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
    injected_ids = {x for x in _mcp._active_intent.get("injected_memory_ids", set()) if x}
    accessed_ids = {x for x in _mcp._active_intent.get("accessed_memory_ids", set()) if x}

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
    try:
        trace_file = (
            _mcp._INTENT_STATE_DIR / f"execution_trace_{_mcp._session_id or 'default'}.jsonl"
        )
        if trace_file.exists():
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
    exec_description = f"{intent_desc or intent_type}: {summary[:200]}"
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
        _mcp._kg.add_triple(exec_id, "is_a", intent_type)
        edges_created.append(f"{exec_id} is_a {intent_type}")
    except Exception:
        pass

    # executed_by → agent
    try:
        _mcp._kg.add_triple(exec_id, "executed_by", agent)
        edges_created.append(f"{exec_id} executed_by {agent}")
    except Exception:
        pass

    # targeted → slot entities
    for target in slot_entities:
        try:
            target_id = normalize_entity_name(target)
            _mcp._kg.add_triple(exec_id, "targeted", target_id)
            edges_created.append(f"{exec_id} targeted {target_id}")
        except Exception:
            pass

    # outcome as has_value
    try:
        _mcp._kg.add_triple(exec_id, "has_value", outcome)
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
        result = _mcp._add_memory_internal(
            content=f"## {intent_type}: {intent_desc}\n\n**Outcome:** {outcome}\n\n{summary}",
            slug=f"result-{exec_id}",
            added_by=agent,
            content_type="event",
            importance=3,
            entity=exec_id,
            predicate="resulted_in",
        )
        if result.get("success"):
            result_memory_id = result.get("memory_id")
            edges_created.append(f"{exec_id} resulted_in {result_memory_id}")
        else:
            errors.append({"kind": "result_memory", "error": result.get("error", "unknown")})
    except Exception as e:
        errors.append({"kind": "result_memory", "error": f"exception: {e}"})

    # ── Trace memory ──
    if trace_entries:
        try:
            trace_text = "\n".join(
                f"- [{e.get('ts', '')}] {e['tool']} {e.get('target', '')}" for e in trace_entries
            )
            trace_result = _mcp._add_memory_internal(
                content=f"## Execution trace: {exec_id}\n\n{trace_text}",
                slug=f"trace-{exec_id}",
                added_by=agent,
                content_type="event",
                importance=2,
                entity=exec_id,
                predicate="evidenced_by",
            )
            if trace_result.get("success"):
                edges_created.append(f"{exec_id} evidenced_by {trace_result.get('memory_id')}")
            else:
                errors.append(
                    {"kind": "trace_memory", "error": trace_result.get("error", "unknown")}
                )
        except Exception as e:
            errors.append({"kind": "trace_memory", "error": f"exception: {e}"})

    # ── Gotchas ──
    if gotchas:
        for gotcha_desc in gotchas:
            try:
                gotcha_id = normalize_entity_name(gotcha_desc[:50])
                if gotcha_id:
                    # Check if gotcha entity exists, create if not
                    existing = _mcp._kg.get_entity(gotcha_id)
                    if not existing:
                        _mcp._create_entity(
                            gotcha_id,
                            kind="entity",
                            description=gotcha_desc,
                            importance=3,
                            added_by=agent,
                        )
                    _mcp._kg.add_triple(exec_id, "has_gotcha", gotcha_id)
                    edges_created.append(f"{exec_id} has_gotcha {gotcha_id}")
                    if promote_gotchas_to_type:
                        _mcp._kg.add_triple(intent_type, "has_gotcha", gotcha_id)
                        edges_created.append(f"{intent_type} has_gotcha {gotcha_id}")
            except Exception:
                pass

    # ── Learnings ──
    if learnings:
        for i, learning in enumerate(learnings):
            try:
                learning_result = _mcp._add_memory_internal(
                    content=learning,
                    slug=f"learning-{exec_id}-{i}",
                    added_by=agent,
                    content_type="discovery",
                    importance=4,
                    entity=exec_id,
                    predicate="evidenced_by",
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

    # ── Memory relevance feedback ──
    feedback_count = 0
    if memory_feedback:
        for fb in memory_feedback:
            try:
                mem_id = normalize_entity_name(fb.get("id", ""))
                if not mem_id:
                    continue
                relevant = fb.get("relevant", True)
                promote = fb.get("promote_to_type", False)

                predicate = "found_useful" if relevant else "found_irrelevant"
                relevance_score = fb.get("relevance", 3)  # 1-5 scale
                confidence = max(0.0, min(1.0, relevance_score / 5.0))

                # Link to execution instance (store relevance score as confidence)
                _mcp._kg.add_triple(exec_id, predicate, mem_id, confidence=confidence)
                edges_created.append(f"{exec_id} {predicate} {mem_id}")

                # If promoted to type, also link to the intent type class
                if promote and intent_type:
                    _mcp._kg.add_triple(intent_type, predicate, mem_id, confidence=confidence)
                    edges_created.append(f"{intent_type} {predicate} {mem_id}")

                # Reset decay for useful memories by updating last_relevant_at
                if relevant:
                    try:
                        col = _mcp._get_collection(create=False)
                        if col:
                            existing = col.get(ids=[mem_id], include=["metadatas"])
                            if existing and existing["ids"]:
                                meta = existing["metadatas"][0] or {}
                                meta["last_relevant_at"] = datetime.now().isoformat()
                                col.update(ids=[mem_id], metadatas=[meta])
                    except Exception:
                        pass  # Non-fatal — decay reset is best-effort

                feedback_count += 1
            except Exception:
                pass

    # ── Record scoring component feedback for weight learning ──
    if memory_feedback:
        from .scoring import compute_age_days, DEFAULT_SEARCH_WEIGHTS

        for fb in memory_feedback:
            try:
                mem_id = normalize_entity_name(fb.get("id", ""))
                if not mem_id:
                    continue
                relevant = fb.get("relevant", True)
                # Look up metadata to compute component values
                meta = {}
                try:
                    col = _mcp._get_collection(create=False)
                    if col:
                        d = col.get(ids=[mem_id], include=["metadatas"])
                        if d and d["ids"]:
                            meta = d["metadatas"][0] or {}
                except Exception:
                    pass
                if not meta:
                    try:
                        ecol = _mcp._get_entity_collection(create=False)
                        if ecol:
                            d = ecol.get(ids=[mem_id], include=["metadatas"])
                            if d and d["ids"]:
                                meta = d["metadatas"][0] or {}
                    except Exception:
                        pass

                imp = float(meta.get("importance", 3))
                date_iso = meta.get("date_added") or meta.get("filed_at") or ""
                last_rel = meta.get("last_relevant_at") or ""
                age_days = compute_age_days(date_iso, last_rel)
                agent_match = bool(agent and meta.get("added_by") == agent)

                components = {
                    "imp": (imp - 1.0) / 4.0,
                    "decay": max(0.0, min(1.0, 1.0 / (1.0 + age_days / 30.0))),
                    "agent": 1.0 if agent_match else 0.0,
                }
                _mcp._kg.record_scoring_feedback(components, relevant)
            except Exception:
                pass

        # Update learned weights
        try:
            from .scoring import set_learned_weights

            learned = _mcp._kg.compute_learned_weights(DEFAULT_SEARCH_WEIGHTS)
            set_learned_weights(learned)
        except Exception:
            pass

    # ── Store context vectors for contextual feedback ──
    context_views = _mcp._active_intent.get("_context_views", [])
    feedback_context_id = ""
    if context_views:
        try:
            feedback_context_id = f"ctx_{slug}"
            _mcp.store_feedback_context(feedback_context_id, context_views)
        except Exception:
            feedback_context_id = ""

    # ── Record edge traversal feedback ──
    # For memories found via graph walk, record whether the edges that led
    # to them were useful. This trains the graph walk for future intents.
    traversed_edges = _mcp._active_intent.get("traversed_edges", [])
    if traversed_edges and memory_feedback:
        feedback_map = {}
        for fb in memory_feedback or []:
            fid = normalize_entity_name(fb.get("id", ""))
            if fid:
                feedback_map[fid] = fb.get("relevant", True)
        for subj, pred, obj in traversed_edges:
            # Check if any feedback target is reachable via this edge
            # Simple: if obj or subj was in feedback, record the edge feedback
            for target_id, was_useful in feedback_map.items():
                if target_id in (subj, obj) or target_id.startswith("drawer_"):
                    try:
                        _mcp._kg.record_edge_feedback(
                            subj,
                            pred,
                            obj,
                            intent_type,
                            was_useful,
                            context_id=feedback_context_id,
                        )
                    except Exception:
                        pass
                    break  # One feedback per edge per finalization

    # ── Record keyword suppression feedback ──
    # If a memory came ONLY from keyword channel and was marked irrelevant,
    # increment its suppression count. If it came from another channel AND was
    # marked relevant, reset suppression (the content IS relevant, keyword
    # was just not discriminating enough in other contexts).
    channel_attribution = _mcp._active_intent.get("_channel_attribution", {})
    if channel_attribution and memory_feedback:
        for fb in memory_feedback or []:
            fid = normalize_entity_name(fb.get("id", ""))
            if not fid:
                continue
            channels = set(channel_attribution.get(fid, []))
            was_relevant = fb.get("relevant", True)
            if not was_relevant and channels == {"keyword"}:
                # Keyword-only + irrelevant → suppress
                try:
                    _mcp._kg.record_keyword_suppression(fid, context_id=feedback_context_id)
                except Exception:
                    pass
            elif was_relevant and "keyword" in channels and len(channels) > 1:
                # Multi-channel + relevant → reset suppression (recovery)
                try:
                    _mcp._kg.reset_keyword_suppression(fid)
                except Exception:
                    pass

    # ── Graph enrichment: suggest edges for useful unconnected memories ──
    # Two cases:
    # 1. Hop-shortening: graph-discovered at distance > 1 → suggest direct edge
    # 2. New connection: found via similarity/keyword with NO graph path → suggest edge
    # Both make the graph richer for future retrieval.
    edge_suggestions = []
    graph_distances = _mcp._active_intent.get("_graph_memories_snapshot", {})
    # Build set of directly-connected IDs (distance 1 or slot entities)
    directly_connected = set(slot_entities)
    for did, dist in graph_distances.items():
        if dist <= 1:
            directly_connected.add(did)

    for fb in memory_feedback or []:
        fid = normalize_entity_name(fb.get("id", ""))
        if not fid or not fb.get("relevant", False):
            continue
        if fid in directly_connected:
            continue  # Already directly connected — no edge needed

        dist = graph_distances.get(fid, None)
        if dist is not None and dist > 1:
            reason = f"Useful at distance {dist} — shorten hop"
        elif dist is None and graph_distances:
            # Only suggest if graph walk ran (avoids spurious suggestions in tests)
            reason = "Useful but no graph connection — create new edge"
        else:
            continue

        for slot_eid in slot_entities[:2]:
            edge_suggestions.append({"from": slot_eid, "to": fid, "reason": reason})

    # ── Deactivate intent ──
    _mcp._active_intent = None
    _persist_active_intent()

    # Store edge suggestions as pending enrichments (NOT conflicts — different mechanism)
    if edge_suggestions:
        enrichments = []
        for es in edge_suggestions:
            enrichment_id = f"enrich_{es['from']}_{es['to']}"
            enrichments.append(
                {
                    "id": enrichment_id,
                    "reason": es.get(
                        "reason", "Graph enrichment — create edge with appropriate predicate"
                    ),
                    "from_entity": es["from"],
                    "to_entity": es["to"],
                }
            )
        _mcp._pending_enrichments = enrichments
        _persist_active_intent()

    result = {
        "success": True,
        "execution_entity": exec_id,
        "outcome": outcome,
        "edges_created_count": len(edges_created),
        "trace_entries": len(trace_entries),
        "result_memory": result_memory_id,
        "feedback_count": feedback_count,
    }
    if errors:
        result["errors"] = errors
        result["warning"] = (
            f"{len(errors)} side-memory creation(s) failed silently before "
            "see 'errors' for details. The execution entity itself was created and "
            "feedback/gotchas were recorded; only the filed memories were affected."
        )
    if edge_suggestions:
        result["enrichments_count"] = len(edge_suggestions)
        result["enrichments_prompt"] = (
            f"{len(edge_suggestions)} graph enrichment tasks pending. "
            "For each: call kg_add(subject, predicate, object) to create the edge "
            "with a predicate YOU choose from the declared predicates, then call "
            "mempalace_resolve_enrichments to mark done. Or reject with reason."
        )
    return result
