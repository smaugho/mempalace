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
from .searcher import search_memories
from .scoring import adaptive_k

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


def _build_intent_hierarchy() -> list:
    """Build a list of all intent types with their tools and is_a parent.

    Walks the KG to find all entities that is_a intent_type (directly or
    transitively). Returns a list of dicts with id, parent, tools.
    Used in error messages so the model knows what types exist and can
    create new ones with the right parent.
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

    for i, eid in enumerate(all_entities["ids"]):
        meta = all_entities["metadatas"][i] or {}
        # Intent types are kind=class (types that get instantiated).
        # Intent executions are kind=entity — skip those here.
        if meta.get("kind") != "class":
            continue

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

    # Sort by importance (highest first), then top-level before children
    hierarchy.sort(
        key=lambda x: (-x.get("importance", 3), 0 if x["parent"] == "intent-type" else 1, x["id"])
    )
    return hierarchy


def _build_intent_hierarchy_safe() -> list:
    """Safe wrapper — never crashes, returns [] on any error."""
    try:
        return _build_intent_hierarchy()
    except Exception:
        return []


def _persist_active_intent():
    """Write active intent to session-scoped state file for PreToolUse hook."""
    try:
        _mcp._INTENT_STATE_DIR.mkdir(parents=True, exist_ok=True)
        state_file = _intent_state_path()
        if _mcp._active_intent:
            state = {
                "intent_id": _mcp._active_intent["intent_id"],
                "intent_type": _mcp._active_intent["intent_type"],
                "slots": _mcp._active_intent["slots"],
                "effective_permissions": _mcp._active_intent["effective_permissions"],
                "description": _mcp._active_intent.get("description", ""),
                "agent": _mcp._active_intent.get("agent", ""),
                "session_id": _mcp._session_id,
                "intent_hierarchy": _build_intent_hierarchy_safe(),
                "injected_drawer_ids": list(_mcp._active_intent.get("injected_drawer_ids", set())),
                "accessed_memory_ids": list(_mcp._active_intent.get("accessed_memory_ids", set())),
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

        # Walk to parent via is-a
        edges = _mcp._kg.query_entity(current, direction="outgoing")
        parent = None
        for e in edges:
            if e["predicate"] in ("is-a", "is_a") and e["current"]:
                parent_id = normalize_entity_name(e["object"])
                # Intent types are kind=class. Stop at the root intent_type class.
                if parent_id == "intent_type":
                    break
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
    description: str = "",
    auto_declare_files: bool = False,
    agent: str = None,
):
    """Declare what you intend to do BEFORE doing it. Returns permissions + context.

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

        description: Free-text description of what you plan to do and why.

    Returns:
        permissions: Which tools are allowed and their scope (scoped to slots or unrestricted).
        context: Facts about slot entities, rules on the intent type, relevant memories.
        previous_expired: ID of the previous active intent if one was replaced.

    If intent_type is not declared or not is-a intent_type, returns an error
    with instructions on how to declare it. Same pattern as predicate constraints.
    """

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
                f"  1. kg_declare_entity(name='{intent_type}', "
                f"description='<what this action does, when to use it>', kind='class', importance=4)\n"
                f"  2. kg_add(subject='{intent_type}', predicate='is_a', "
                f"object='<parent>') — where parent is the broad type it inherits from "
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
                pass  # Non-fatal — narrowing is best-effort

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

    permissions = []
    for slot_name, entity_ids in flat_slots.items():
        raws = raw_paths.get(slot_name, entity_ids)
        for perm in effective_permissions:
            scope = perm.get("scope", "*")
            if f"{{{slot_name}}}" in scope:
                # Replace slot reference with RAW values (file paths, not normalized IDs)
                for raw_val, entity_id in zip(raws, entity_ids):
                    permissions.append(
                        {
                            "tool": perm["tool"],
                            "scope": scope.replace(f"{{{slot_name}}}", raw_val),
                            "slot": slot_name,
                            "entity": entity_id,
                        }
                    )
            elif scope == "*":
                if not any(p["tool"] == perm["tool"] and p["scope"] == "*" for p in permissions):
                    permissions.append({"tool": perm["tool"], "scope": "*"})

    # ── Collect context ──
    context = {"target_facts": [], "intent_rules": [], "relevant_memories": []}

    # Facts about slot entities — NO hardcoded predicate filters.
    # All facts are included, then adaptive-K selects the natural cluster.
    # Learned predicate relevance from found_useful/found_irrelevant feedback
    # suppresses predicates that have been consistently irrelevant.
    # New predicates are always included until feedback teaches otherwise.

    # Learn predicate relevance from KG feedback on the active intent type.
    learned_irrelevant_preds = set()
    try:
        type_edges = _mcp._kg.query_entity(intent_id, direction="outgoing")
        for te in type_edges:
            if not te.get("current", True):
                continue
            if te["predicate"] == "found_useful":
                # Look up predicates in useful facts — these override irrelevant
                ref_edges = _mcp._kg.query_entity(te["object"], direction="both")
                for re in ref_edges:
                    if re.get("current", True):
                        learned_irrelevant_preds.discard(re["predicate"])
            elif te["predicate"] == "found_irrelevant":
                ref_edges = _mcp._kg.query_entity(te["object"], direction="both")
                for re in ref_edges:
                    if re.get("current", True):
                        learned_irrelevant_preds.add(re["predicate"])
    except Exception:
        pass  # Non-fatal

    for entity_id in all_slot_entities:
        edges = _mcp._kg.query_entity(entity_id, direction="both")
        entity_facts = []
        for e in edges:
            if not e.get("current", True):
                continue
            pred = e["predicate"]
            # Only skip predicates that feedback has taught are irrelevant
            if pred in learned_irrelevant_preds:
                continue
            entity_facts.append(
                f"{e.get('subject', entity_id)} -> {pred} -> {e.get('object', '?')}"
            )
        # Cap per entity — adaptive-K with real fact scoring comes in future.
        # For now, max 20 facts per entity; learned_irrelevant_preds handles filtering.
        context["target_facts"].extend(entity_facts[:20])

    # Rules on the intent type — include all current edges except is-a (structural taxonomy)
    intent_edges = _mcp._kg.query_entity(intent_id, direction="outgoing")
    for e in intent_edges:
        if e.get("current", True) and e["predicate"] not in ("is-a", "is_a"):
            # Skip only is-a (taxonomy structure, not actionable)
            # found_useful/found_irrelevant ARE shown here — they're learned rules
            if e["predicate"] not in learned_irrelevant_preds:
                context["intent_rules"].append(
                    f"{intent_id} -> {e['predicate']} -> {e.get('object', '?')}"
                )

    # Relevant memories via search (deduped against prior injections)
    already_injected = set()
    if _mcp._active_intent:
        already_injected = _mcp._active_intent.get("injected_drawer_ids", set())

    for entity_id in all_slot_entities:
        entity = _mcp._kg.get_entity(entity_id)
        search_query = entity["name"] if entity else entity_id
        try:
            search_result = search_memories(
                search_query, palace_path=_mcp._config.palace_path, n_results=3
            )
            if isinstance(search_result, dict) and search_result.get("results"):
                for r in search_result["results"]:
                    drawer_id = r.get("id", "")
                    if drawer_id and drawer_id not in already_injected:
                        already_injected.add(drawer_id)
                        context["relevant_memories"].append(
                            {
                                "drawer_id": drawer_id,
                                "snippet": (r.get("text") or "")[:200],
                                "for_entity": entity_id,
                            }
                        )
        except Exception:
            pass  # Non-fatal — context injection is best-effort

    # ── Historical injection: surface past executions of this intent type ──
    past_executions = []
    try:
        ecol = _mcp._get_entity_collection(create=False)
        if ecol:
            # Search for entities that are is_a this intent type (execution instances)
            exec_search = ecol.query(
                query_texts=[description or intent_id],
                n_results=20,
                include=["documents", "metadatas", "distances"],
                where={"kind": "entity"},
            )
            if exec_search["ids"] and exec_search["ids"][0]:
                for i, eid in enumerate(exec_search["ids"][0]):
                    meta = exec_search["metadatas"][0][i] or {}
                    # Check if this entity is an execution of our intent type
                    edges = _mcp._kg.query_entity(eid, direction="outgoing")
                    is_execution = False
                    gotchas = []
                    for e in edges:
                        if not e.get("current", True):
                            continue
                        pred = e["predicate"]
                        obj = e.get("object", "")
                        if pred in ("is-a", "is_a") and obj in (intent_id,):
                            is_execution = True
                        # Only collect actionable relationships: gotchas
                        if pred == "has_gotcha":
                            gotchas.append(obj)

                    if is_execution:
                        dist = exec_search["distances"][0][i]
                        similarity = round(1 - dist, 3)
                        exec_data = {
                            "entity_id": eid,
                            "description": (exec_search["documents"][0][i] or "")[:200],
                            "outcome": meta.get("outcome", "unknown"),
                            "agent": meta.get("added_by", ""),
                            "similarity": similarity,
                        }
                        if gotchas:
                            exec_data["gotchas"] = gotchas[:5]
                        past_executions.append(exec_data)

                # Sort by similarity
                past_executions.sort(key=lambda x: x.get("similarity", 0), reverse=True)
                # Adaptive-K: take natural cluster above largest score gap
                if len(past_executions) > 1:
                    exec_scores = [e.get("similarity", 0) for e in past_executions]
                    k = adaptive_k(exec_scores, max_k=10, min_k=1)
                    past_executions = past_executions[:k]
                else:
                    past_executions = past_executions[:1]
    except Exception:
        pass  # Non-fatal

    if past_executions:
        context["past_executions"] = past_executions

    # ── Contextual relevance: query type-level feedback to boost/demote memories ──
    # Check if the intent type has found_useful/found_irrelevant edges
    type_feedback = {"useful": set(), "irrelevant": set()}
    try:
        type_edges = _mcp._kg.query_entity(intent_id, direction="outgoing")
        for e in type_edges:
            if e.get("current", True):
                if e["predicate"] == "found_useful":
                    type_feedback["useful"].add(e["object"])
                elif e["predicate"] == "found_irrelevant":
                    type_feedback["irrelevant"].add(e["object"])
        if type_feedback["useful"] or type_feedback["irrelevant"]:
            context["type_relevance"] = {
                "useful": list(type_feedback["useful"]),
                "irrelevant": list(type_feedback["irrelevant"]),
                "note": "Based on past executions of this intent type. Useful memories are boosted in ranking.",
            }
    except Exception:
        pass  # Non-fatal

    # ── Mandatory type promotion check: 3+ similar executions ──
    # Uses similarity threshold tightening: each intent type stores
    # promoted_at_similarity in its properties. Deeper promotions require
    # higher similarity. At similarity=1.0, promotion stops (the type IS the action).
    PROMOTION_COUNT = 3
    BASE_THRESHOLD = 0.7  # Default threshold for types without promoted_at_similarity
    if len(past_executions) >= PROMOTION_COUNT:
        # Get the current type's promoted_at_similarity (if it was itself promoted)
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

        # At similarity=1.0, no further promotion possible — type IS the action
        if parent_threshold < 1.0:
            high_sim = [e for e in past_executions if e.get("similarity", 0) > parent_threshold]
            if len(high_sim) >= PROMOTION_COUNT:
                avg_sim = sum(e.get("similarity", 0) for e in high_sim) / len(high_sim)
                exec_list = "\n".join(
                    f"  - {e['entity_id']}: {e['description'][:100]}" for e in high_sim[:5]
                )
                return {
                    "success": False,
                    "error": (
                        f"Intent type '{intent_id}' has {len(high_sim)} similar past executions "
                        f"above threshold {parent_threshold:.2f}. You MUST either:\n\n"
                        f"(a) Create a specific intent type (set promoted_at_similarity={avg_sim:.3f}):\n"
                        f"    kg_declare_entity(name='<specific-type>', kind='class', importance=4, "
                        f"description='<what this action does>', "
                        f"properties={{'promoted_at_similarity': {avg_sim:.3f}, 'rules_profile': ...}})\n"
                        f"    kg_add(subject='<specific-type>', predicate='is_a', object='{intent_id}')\n"
                        f"    Then re-declare with the specific type.\n\n"
                        f"(b) Disambiguate existing executions (if they're actually different):\n"
                        f"    kg_update_entity_description(entity='<exec_id>', description='<more specific>')\n\n"
                        f"Similar executions (avg similarity {avg_sim:.3f}):\n{exec_list}"
                    ),
                    "similar_executions": high_sim[:5],
                    "promotion_threshold": parent_threshold,
                    "suggested_promoted_at_similarity": round(avg_sim, 3),
                }

    # ── Hard fail if previous intent not finalized ──
    previous_expired = None
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

    _mcp._active_intent = {
        "intent_id": new_intent_id,
        "intent_type": intent_id,
        "slots": flat_slots,
        "effective_permissions": permissions,
        "injected_drawer_ids": already_injected,
        "accessed_memory_ids": set(),
        "description": description,
        "agent": agent or "",
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

    # ── Suggest more specific subtypes (reuse subtypes found during auto-narrow) ──
    # If we narrowed, re-discover subtypes of the NEW (narrowed) intent type
    if narrowed_from:
        subtypes = []
        for e in all_entities:
            e_edges = _mcp._kg.query_entity(e["id"], direction="outgoing")
            for edge in e_edges:
                if edge["predicate"] in ("is-a", "is_a") and edge["current"]:
                    parent_id = normalize_entity_name(edge["object"])
                    if parent_id == intent_id:
                        subtypes.append(
                            {
                                "id": e["id"],
                                "description": e.get("description", "")[:120],
                            }
                        )
                        break

    # Trim descriptions for response
    suggested = [{"id": s["id"], "description": s.get("description", "")[:120]} for s in subtypes]

    subtype_hint = None
    if narrowed_from:
        subtype_hint = (
            f"Auto-narrowed from '{narrowed_from}' to '{intent_id}' based on your description. "
            f"This type carries domain-specific rules that '{narrowed_from}' does not."
        )
    elif suggested:
        subtype_hint = (
            f"You declared '{intent_id}' but more specific intent types exist. "
            f"Specific types carry domain-specific rules (must, requires, has_gotcha) "
            f"that '{intent_id}' does not. Consider switching if one fits."
        )
    else:
        subtype_hint = (
            f"No specific subtypes of '{intent_id}' exist yet. If this is a recurring "
            f"action pattern, consider declaring a specific intent type: "
            f"kg_declare_entity(name='<specific_action>', description='...', kind='class') "
            f"+ kg_add(subject='<specific_action>', predicate='is_a', object='{intent_id}'). "
            f"Then attach rules: kg_add(subject='<specific_action>', predicate='must', object='<rule>'). "
            f"Future declarations of the specific type will surface those rules automatically."
        )

    feedback_reminder = None
    if already_injected:
        feedback_reminder = (
            f"IMPORTANT: {len(already_injected)} memories were injected for this intent. "
            f"You MUST provide feedback on ALL injected memories and at least 30% of "
            f"any additional memories you access via search when calling finalize_intent. "
            f"Finalization will FAIL without sufficient feedback."
        )

    result = {
        "success": True,
        "intent_id": new_intent_id,
        "intent_type": intent_id,
        "slots": flat_slots,
        "permissions": permissions,
        "context": context,
        "previous_expired": previous_expired,
        "suggested_subtypes": suggested,
        "subtype_hint": subtype_hint,
        "feedback_reminder": feedback_reminder,
    }
    if narrowed_from:
        result["narrowed_from"] = narrowed_from
    return result


def tool_active_intent():
    """Return the current active intent, or null if none declared.

    Shows: intent type, filled slots, effective permissions, and how many
    memories were injected. Use this to check what you're currently allowed
    to do before calling a tool.
    """
    if not _mcp._active_intent:
        return {
            "active": False,
            "message": "No active intent. Call mempalace_declare_intent before acting.",
        }
    return {
        "active": True,
        "intent_id": _mcp._active_intent["intent_id"],
        "intent_type": _mcp._active_intent["intent_type"],
        "slots": _mcp._active_intent["slots"],
        "permissions": _mcp._active_intent["effective_permissions"],
        "description": _mcp._active_intent.get("description", ""),
        "injected_memories": len(_mcp._active_intent.get("injected_drawer_ids", set())),
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
    relationships linking it to the agent, targets, result drawer, gotchas,
    and execution trace.

    Args:
        slug: Human-readable ID for this execution (e.g. 'edit-auth-rate-limiter-2026-04-14')
        outcome: 'success', 'partial', 'failed', or 'abandoned'
        summary: What happened — broader result narrative. Becomes a drawer.
        agent: Agent entity name (e.g. 'technical_lead_agent')
        memory_feedback: MANDATORY — contextual relevance feedback for ALL memories
            accessed during this intent. Include memories injected by declare_intent,
            memories you found via search, AND any new memories you created.
            Each entry: {"id": "drawer_or_entity_id", "relevant": true/false,
            "relevance": 1-5, "promote_to_type": false, "reason": "why"}.
            promote_to_type=true links feedback to the intent TYPE (generalizable pattern),
            false keeps it on this execution only (instance-specific).
        key_actions: Abbreviated tool+params list (optional — auto-filled from trace if omitted)
        gotchas: List of gotcha descriptions discovered during execution
        learnings: List of lesson descriptions worth remembering
        promote_gotchas_to_type: Also link gotchas to the intent type (not just execution)
    """

    if not _mcp._active_intent:
        return {"success": False, "error": "No active intent to finalize."}

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

    # ── Validate memory feedback coverage ──
    injected_ids = {x for x in _mcp._active_intent.get("injected_drawer_ids", set()) if x}
    accessed_ids = {x for x in _mcp._active_intent.get("accessed_memory_ids", set()) if x}

    feedback_ids = set()
    if memory_feedback:
        for fb in memory_feedback:
            fb_id = normalize_entity_name(fb.get("id", ""))
            if fb_id:
                feedback_ids.add(fb_id)

    # Injected memories: 100% feedback required
    if injected_ids:
        missing_injected = injected_ids - feedback_ids
        if missing_injected:
            coverage = (len(injected_ids) - len(missing_injected)) / len(injected_ids)
            return {
                "success": False,
                "error": (
                    f"Insufficient memory feedback. {len(missing_injected)} of {len(injected_ids)} "
                    f"injected memories have no feedback (100% required). "
                    f"Missing: {sorted(missing_injected)[:10]}"
                ),
                "missing_injected": sorted(missing_injected),
                "missing_accessed": [],
                "feedback_coverage": {"injected": round(coverage, 2), "accessed": 0},
            }

    # Accessed memories: 30% feedback required (excluding already-covered injected)
    MIN_ACCESSED_COVERAGE = 0.3
    accessed_only = accessed_ids - injected_ids
    if accessed_only:
        accessed_covered = len(accessed_only & feedback_ids)
        accessed_coverage = accessed_covered / len(accessed_only)
        if accessed_coverage < MIN_ACCESSED_COVERAGE:
            missing_accessed = sorted(accessed_only - feedback_ids)
            needed = max(1, int(len(accessed_only) * MIN_ACCESSED_COVERAGE) - accessed_covered)
            return {
                "success": False,
                "error": (
                    f"Insufficient memory feedback. Only {accessed_covered}/{len(accessed_only)} "
                    f"accessed memories rated ({accessed_coverage:.0%}, minimum {MIN_ACCESSED_COVERAGE:.0%}). "
                    f"Rate at least {needed} more."
                ),
                "missing_injected": [],
                "missing_accessed": missing_accessed[:10],
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

    # ── Result drawer (summary) ──
    result_drawer_id = None
    try:
        # Determine wing from agent
        agent_id = normalize_entity_name(agent)
        wing = f"wing_{agent_id.replace('_agent', '').replace('-agent', '')}"

        result = _mcp.tool_add_drawer(
            wing=wing,
            room="intent-results",
            content=f"## {intent_type}: {intent_desc}\n\n**Outcome:** {outcome}\n\n{summary}",
            slug=f"result-{exec_id}",
            hall="hall_events",
            importance=3,
            entity=exec_id,
            predicate="resulted_in",
            added_by=agent,
        )
        if result.get("success"):
            result_drawer_id = result.get("drawer_id")
            edges_created.append(f"{exec_id} resulted_in {result_drawer_id}")
    except Exception:
        pass

    # ── Trace drawer ──
    if trace_entries:
        try:
            trace_text = "\n".join(
                f"- [{e.get('ts', '')}] {e['tool']} {e.get('target', '')}" for e in trace_entries
            )
            trace_result = _mcp.tool_add_drawer(
                wing=wing,
                room="intent-results",
                content=f"## Execution trace: {exec_id}\n\n{trace_text}",
                slug=f"trace-{exec_id}",
                hall="hall_events",
                importance=2,
                entity=exec_id,
                predicate="evidenced_by",
                added_by=agent,
            )
            if trace_result.get("success"):
                edges_created.append(f"{exec_id} evidenced_by {trace_result.get('drawer_id')}")
        except Exception:
            pass

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
                _mcp.tool_add_drawer(
                    wing=wing,
                    room="lessons-learned",
                    content=learning,
                    slug=f"learning-{exec_id}-{i}",
                    hall="hall_discoveries",
                    importance=4,
                    entity=exec_id,
                    predicate="evidenced_by",
                    added_by=agent,
                )
            except Exception:
                pass

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

                # Link to execution instance
                _mcp._kg.add_triple(exec_id, predicate, mem_id)
                edges_created.append(f"{exec_id} {predicate} {mem_id}")

                # If promoted to type, also link to the intent type class
                if promote and intent_type:
                    _mcp._kg.add_triple(intent_type, predicate, mem_id)
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

    # ── Deactivate intent ──
    _mcp._active_intent = None
    _persist_active_intent()

    return {
        "success": True,
        "execution_entity": exec_id,
        "outcome": outcome,
        "edges_created": edges_created,
        "trace_entries": len(trace_entries),
        "result_drawer": result_drawer_id,
        "feedback_count": feedback_count,
    }
