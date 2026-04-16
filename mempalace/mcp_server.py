#!/usr/bin/env python3
"""
MemPalace MCP Server — read/write palace access for Claude Code
================================================================
Install: claude mcp add mempalace -- python -m mempalace.mcp_server [--palace /path/to/palace]

Tools (read):
  mempalace_kg_search       — unified 3-channel search over drawers AND entities
  mempalace_kg_query        — structured edge lookup by exact entity name
  mempalace_kg_stats        — palace overview: counts by wing/room/kind

Tools (write):
  mempalace_kg_declare_entity — declare an entity (kind=entity/class/predicate/literal/memory)
                                memory drawers are first-class entities (P3.3)
  mempalace_kg_delete_entity — soft-delete an entity or drawer (invalidates all edges)
  mempalace_resolve_conflicts — resolve contradictions, duplicates, merge candidates
"""

import argparse
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

from .config import MempalaceConfig, sanitize_name, sanitize_content
from .version import __version__
from .query_sanitizer import sanitize_query
from .palace_graph import traverse, graph_stats
import chromadb

from .knowledge_graph import KnowledgeGraph
from . import intent
from .scoring import (
    hybrid_score as _hybrid_score_fn,
    adaptive_k,
    multi_channel_search,
    validate_query_views,
    lookup_type_feedback,
)

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger("mempalace_mcp")


def _parse_args():
    parser = argparse.ArgumentParser(description="MemPalace MCP Server")
    parser.add_argument(
        "--palace",
        metavar="PATH",
        help="Path to the palace directory (overrides config file and env var)",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.debug("Ignoring unknown args: %s", unknown)
    return args


_args = _parse_args()

if _args.palace:
    os.environ["MEMPALACE_PALACE_PATH"] = os.path.abspath(_args.palace)

_config = MempalaceConfig()
if _args.palace:
    _kg = KnowledgeGraph(db_path=os.path.join(_config.palace_path, "knowledge_graph.sqlite3"))
else:
    _kg = KnowledgeGraph()


# Wire intent module to this module (after globals are set)
intent.init(sys.modules[__name__])

_client_cache = None
_collection_cache = None


# ==================== WRITE-AHEAD LOG ====================
# Every write operation is logged to a JSONL file before execution.
# This provides an audit trail for detecting memory poisoning and
# enables review/rollback of writes from external or untrusted sources.

_WAL_DIR = Path(os.path.expanduser("~/.mempalace/wal"))
_WAL_DIR.mkdir(parents=True, exist_ok=True)
try:
    _WAL_DIR.chmod(0o700)
except (OSError, NotImplementedError):
    pass
_WAL_FILE = _WAL_DIR / "write_log.jsonl"


def _wal_log(operation: str, params: dict, result: dict = None):
    """Append a write operation to the write-ahead log."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "operation": operation,
        "params": params,
        "result": result,
    }
    try:
        with open(_WAL_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        try:
            _WAL_FILE.chmod(0o600)
        except (OSError, NotImplementedError):
            pass
    except Exception as e:
        logger.error(f"WAL write failed: {e}")


_client_cache = None
_collection_cache = None


def _get_client():
    """Return a singleton ChromaDB PersistentClient."""
    global _client_cache
    if _client_cache is None:
        _client_cache = chromadb.PersistentClient(path=_config.palace_path)
    return _client_cache


def _get_collection(create=False):
    """Return the ChromaDB collection, caching the client between calls."""
    global _collection_cache
    try:
        client = _get_client()
        if create:
            _collection_cache = client.get_or_create_collection(_config.collection_name)
        elif _collection_cache is None:
            _collection_cache = client.get_collection(_config.collection_name)
        return _collection_cache
    except Exception:
        return None


def _no_palace():
    return {
        "error": "No palace found",
        "hint": "Run: mempalace init <dir> && mempalace mine <dir>",
    }


# ==================== READ TOOLS ====================


# ── AAAK Dialect Spec ─────────────────────────────────────────────────────────
# Included in wake_up response so the AI learns it on first call.
# Also available via mempalace_get_aaak_spec tool.

PALACE_PROTOCOL = """MemPalace Protocol — behavioral rules only. The system enforces the rest
(intent declaration, entity declaration, tool permissions, predicate constraints).

ON START:
  Call mempalace_wake_up. Read this protocol, the text (identity + rules),
  and declared (entities, predicates, intent types with their tools).

BEFORE ACTING ON ANY FACT:
  Use kg_query for exact entity-ID lookups when you know the name.
  Use kg_search for fuzzy discovery — it searches BOTH drawers (prose)
  and entities (KG nodes) in one call, with graph expansion. Never guess.

WHEN HITTING A BLOCKER:
  FIRST search mempalace for known solutions — gotchas, lessons-learned,
  past executions that solved similar problems. Only report a blocker to
  the user if memory has no answer. When you solve a new problem, persist
  the solution (drawer + KG triple) so future sessions find it.

WHEN FILING DRAWERS:
  - Choose the precise predicate for the entity link: described_by,
    evidenced_by, derived_from, mentioned_in, session_note_for.
  - Then extract at least one KG triple from the content (twin pattern).
    Drawer alone = semantic search only. KG triple = fast entity lookup.
  - Duplicate detection is automatic — if similar drawers exist, conflicts
    will be returned. Resolve via mempalace_resolve_conflicts.

WHEN ADDING KG FACTS:
  - Declare new entities first (kg_declare_entity) with kind and importance.
  - Use properties for metadata: predicates need constraints, intent types
    need rules_profile (slots + tool_permissions).
  - Contradiction detection is automatic — if an edge conflicts with existing
    edges (same subject+predicate, different object), conflicts are returned.

WHEN CONFLICTS ARE DETECTED:
  - Any write operation (kg_add, kg_declare_entity, finalize_intent)
    may return conflicts. Tools are BLOCKED until all conflicts are resolved.
  - Call mempalace_resolve_conflicts with actions for each conflict:
    invalidate (old is stale), merge (combine both — provide merged_content),
    keep (both are valid), or skip (undo the new item).
  - For merge: READ BOTH items in full first, then provide merged_content
    that preserves ALL unique information from each side.

WHEN USING TOOLS:
  - Declare intent first (mempalace_declare_intent). Check declared.intent_types
    for available types. If a tool is blocked, the error shows the hierarchy
    and teaches how to create or switch intent types.
  - Tool permissions are additive — child types inherit parent tools. Only specify
    tools the parent doesn't have. Use wildcards for MCP groups (mcp__provider__*).
  - Intent types are kind=class. Execution instances are kind=entity with is_a.
  - Always pass your agent entity name in searches and declarations.

BEFORE SWITCHING INTENTS:
  Call mempalace_finalize_intent BEFORE declaring a new intent. This captures:
  - What you did (execution entity + trace)
  - What you learned (gotchas, learnings)
  - How it went (outcome: success/partial/failed/abandoned)
  If you forget, declare_intent auto-finalizes with outcome='abandoned'.
  Explicit finalization creates better memories for future sessions.

INTENT TYPES:
  - New intent types use kind='class' (they are types, not instances).
  - If 3+ similar executions exist on a broad type, declaration fails —
    you must create a specific intent type for that recurring action.

AT SESSION END:
  First, finalize the active intent (mempalace_finalize_intent).
  Then persist new knowledge using the twin pattern:
  - Decisions, rules, discoveries, gotchas -> drawer + KG triple(s).
  - Changed facts -> kg_invalidate old + kg_add new.
  - New entities encountered -> kg_declare_entity if not yet declared.
  Don't just diary them — diary is a temporal log, KG + drawers are
  durable knowledge that future sessions can query structurally.
  Then call diary_write — but keep it CONCISE and NON-REDUNDANT:
  - Write readable prose, NOT AAAK compression.
  - Delta only: what changed SINCE the last diary entry (not a full restatement).
  - Focus on: decisions made with user, big-picture status, pending items.
  - Do NOT repeat: commits, gotchas, learnings, features (already in intent results).
  - The diary is a high-level narrative, not a detailed log."""

AAAK_SPEC = """AAAK is a compressed memory dialect that MemPalace uses for efficient storage.
It is designed to be readable by both humans and LLMs without decoding.

FORMAT:
  ENTITIES: 3-letter uppercase codes. ALC=Alice, JOR=Jordan, RIL=Riley, MAX=Max, BEN=Ben.
  EMOTIONS: *action markers* before/during text. *warm*=joy, *fierce*=determined, *raw*=vulnerable, *bloom*=tenderness.
  STRUCTURE: Pipe-separated fields. FAM: family | PROJ: projects | ⚠: warnings/reminders.
  DATES: ISO format (2026-03-31). COUNTS: Nx = N mentions (e.g., 570x).
  IMPORTANCE: ★ to ★★★★★ (1-5 scale).
  HALLS: hall_facts, hall_events, hall_discoveries, hall_preferences, hall_advice.
  WINGS: wing_user, wing_agent, wing_team, wing_code, wing_myproject, wing_hardware, wing_ue5, wing_ai_research.
  ROOMS: Hyphenated slugs representing named ideas (e.g., chromadb-setup, gpu-pricing).

EXAMPLE:
  FAM: ALC→♡JOR | 2D(kids): RIL(18,sports) MAX(11,chess+swimming) | BEN(contributor)

Read AAAK naturally — expand codes mentally, treat *markers* as emotional context.
When WRITING AAAK: use entity codes, mark emotions, keep structure tight."""


def _hybrid_score(
    similarity: float,
    importance: float,
    date_added_iso: str,
    agent_match: bool = False,
    last_relevant_iso: str = None,
) -> float:
    """Hybrid ranking score for search results. Delegates to scoring.hybrid_score."""
    return _hybrid_score_fn(
        similarity=similarity,
        importance=importance,
        date_iso=date_added_iso,
        agent_match=agent_match,
        last_relevant_iso=last_relevant_iso,
        relevance_feedback=0,
        mode="search",
    )


# tool_search removed (P3.2): merged into tool_kg_search, which now searches
# BOTH drawers and entities in a single cross-collection RRF. The "palace is
# a graph" unification — one search tool over all memory.


def tool_check_duplicate(content: str, threshold: float = 0.9):
    col = _get_collection()
    if not col:
        return _no_palace()
    try:
        results = col.query(
            query_texts=[content],
            n_results=5,
            include=["metadatas", "documents", "distances"],
        )
        duplicates = []
        if results["ids"] and results["ids"][0]:
            for i, drawer_id in enumerate(results["ids"][0]):
                dist = results["distances"][0][i]
                similarity = round(1 - dist, 3)
                if similarity >= threshold:
                    meta = results["metadatas"][0][i]
                    doc = results["documents"][0][i]
                    duplicates.append(
                        {
                            "id": drawer_id,
                            "wing": meta.get("wing", "?"),
                            "room": meta.get("room", "?"),
                            "similarity": similarity,
                            "content": doc[:200] + "..." if len(doc) > 200 else doc,
                        }
                    )
        return {
            "is_duplicate": len(duplicates) > 0,
            "matches": duplicates,
        }
    except Exception as e:
        return {"error": str(e)}


def tool_get_aaak_spec():
    """Return the AAAK dialect specification."""
    return {"aaak_spec": AAAK_SPEC}


def tool_traverse_graph(start_room: str, max_hops: int = 2):
    """Walk the palace graph from a room. Find connected ideas across wings."""
    col = _get_collection()
    if not col:
        return _no_palace()
    return traverse(start_room, col=col, max_hops=max_hops)


# tool_find_tunnels removed (P3.8): the wing/room "tunnel" concept was a
# legacy metaphor. To find related content across wings, use kg_search or
# kg_query — the graph knows the real connections.


# tool_graph_stats removed (P3.7): functionality merged into tool_kg_stats,
# which now also reports wing/room tunnels and cross-wing connectivity.


# ==================== WRITE TOOLS ====================


VALID_HALLS = {
    "hall_facts",
    "hall_events",
    "hall_discoveries",
    "hall_preferences",
    "hall_advice",
    "hall_diary",
}


def _validate_importance(importance):
    """Coerce and validate an importance value (1-5). Returns int or raises ValueError."""
    if importance is None:
        return None
    try:
        n = int(importance)
    except (TypeError, ValueError):
        raise ValueError(
            f"importance must be an integer 1-5 (got {importance!r}). "
            f"Importance scale: 5=critical, 4=canonical, 3=default, 2=low, 1=junk."
        )
    if n < 1 or n > 5:
        raise ValueError(
            f"importance must be between 1 and 5 (got {n}). "
            f"Importance scale: 5=critical unmissable, 4=canonical rules/cookbooks, "
            f"3=historical events (default), 2=low priority, 1=junk/quarantine."
        )
    return n


VALID_KINDS = {
    "entity",  # a concrete individual thing
    "predicate",  # a relationship type
    "class",  # a category/domain-type definition
    "literal",  # a raw value (string, integer, timestamp, URL, path)
    "memory",  # a stored memory/drawer — full text in ChromaDB, metadata in SQLite
}


def _validate_kind(kind):
    """Validate entity kind (ontological role). REQUIRED — no default."""
    if kind is None:
        raise ValueError(
            "kind is REQUIRED. Must be one of: 'entity' (concrete thing), "
            "'predicate' (relationship type), 'class' (category definition), "
            "'literal' (raw value). You must explicitly choose the ontological role."
        )
    if kind not in VALID_KINDS:
        raise ValueError(
            f"kind must be one of {sorted(VALID_KINDS)} (got {kind!r}). "
            f"entity=concrete thing (default), predicate=relationship type, "
            f"class=category/type definition, literal=raw value. "
            f"Domain types (system, person, project, etc.) are NOT kinds — "
            f"they are class-kind entities linked via is_a edges."
        )
    return kind


def _validate_hall(hall):
    """Validate a hall value. Returns string or raises ValueError."""
    if hall is None:
        return None
    if hall not in VALID_HALLS:
        raise ValueError(
            f"hall must be one of {sorted(VALID_HALLS)} (got {hall!r}). "
            f"hall_facts=stable truths, hall_events=things that happened, "
            f"hall_discoveries=lessons learned, hall_preferences=user rules, "
            f"hall_advice=how-to guides, hall_diary=chronological journal."
        )
    return hall


def _normalize_drawer_slug(slug: str, max_length: int = 50) -> str:
    """Normalize a drawer slug: lowercase, hyphens, alphanumeric, max length."""
    import re

    slug = slug.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("-")
    return slug


def _find_related_entities(content_or_desc: str, exclude_ids: set = None, max_results: int = 5):
    """Search entity collection for entities related to given text. Returns suggestions list."""
    suggestions = []
    exclude = exclude_ids or set()
    try:
        ecol = _get_entity_collection(create=False)
        if ecol and ecol.count() > 0:
            n = min(ecol.count(), 20)
            results = ecol.query(
                query_texts=[content_or_desc[:500]],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )
            if results["ids"] and results["ids"][0]:
                for i, eid in enumerate(results["ids"][0]):
                    if eid in exclude:
                        continue
                    meta_r = results["metadatas"][0][i] or {}
                    kind = meta_r.get("kind", "entity")
                    if kind not in ("entity", "class"):
                        continue
                    dist = results["distances"][0][i]
                    sim = round(max(0.0, 1.0 - dist), 3)
                    if sim < 0.15:
                        continue
                    desc = (results["documents"][0][i] or "")[:100]
                    suggestions.append({"entity_id": eid, "similarity": sim, "description": desc})
                    if len(suggestions) >= max_results:
                        break
    except Exception:
        pass
    return suggestions


def _add_drawer_internal(  # noqa: C901
    wing: str,
    room: str,
    content: str,
    slug: str,
    source_file: str = None,
    added_by: str = None,
    hall: str = None,
    importance: int = None,
    entity: str = None,
    predicate: str = "described_by",
    context: dict = None,  # P4.2 — Context fingerprint for keywords + creation_context_id
):
    """File verbatim content into a wing/room. Checks for duplicates first.

    ALL classification params are REQUIRED (no lazy defaults):
        slug: short human-readable identifier — REQUIRED. Used as part of the
              drawer ID. Must be unique within the wing/room. Examples:
              'intent-pre-activation-issues', 'db-credentials', 'ga-identity'.
        hall: content type — REQUIRED. One of hall_facts, hall_events,
              hall_discoveries, hall_preferences, hall_advice, hall_diary.
        importance: integer 1-5 — REQUIRED. 5=critical, 4=canonical,
                    3=default, 2=low, 1=junk.
        entity: entity name (or comma-separated list) — REQUIRED. Links this drawer
                to an entity in the KG. If not provided, defaults to the wing name.
                This prevents orphan drawers — every drawer should be discoverable
                via the entity graph.
        predicate: relationship type for the entity→drawer link. Default: described_by.
                   Use a precise predicate: described_by (canonical description),
                   evidenced_by (backs a rule/decision), derived_from (extracted from),
                   mentioned_in (referenced but not main topic), session_note_for
                   (diary/session entry).

    Note: date_added is always set to the current time. Diary drawers
    (via diary_write) are exempt from the entity/slug requirement.
    """
    try:
        wing = sanitize_name(wing, "wing")
        room = sanitize_name(room, "room")
        content = sanitize_content(content)
        hall = _validate_hall(hall)
        importance = _validate_importance(importance)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    # Validate added_by: REQUIRED, must be a declared agent (is_a agent)
    if not added_by:
        return {
            "success": False,
            "error": "added_by is required. Pass your agent entity name (e.g., 'ga_agent', 'technical_lead_agent').",
        }
    try:
        from .knowledge_graph import normalize_entity_name

        agent_id = normalize_entity_name(added_by)
        if _kg:
            agent_edges = _kg.query_entity(agent_id, direction="outgoing")
            is_agent = any(
                e["predicate"] in ("is-a", "is_a")
                and e["object"] == "agent"
                and e.get("current", True)
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
    except Exception:
        pass  # graceful fallback if KG unavailable

    if not slug or not slug.strip():
        return {
            "success": False,
            "error": "slug is required. Provide a short human-readable identifier (e.g. 'intent-pre-activation-issues').",
        }

    normalized_slug = _normalize_drawer_slug(slug)
    if not normalized_slug:
        return {
            "success": False,
            "error": f"slug '{slug}' normalizes to empty. Use alphanumeric words separated by hyphens.",
        }

    col = _get_collection(create=True)
    if not col:
        return _no_palace()

    drawer_id = f"drawer_{wing}_{room}_{normalized_slug}"

    # Uniqueness check — slug collision returns existing drawer info
    try:
        existing = col.get(ids=[drawer_id], include=["documents", "metadatas"])
        if existing and existing["ids"]:
            return {
                "success": False,
                "error": f"Slug '{normalized_slug}' already exists in {wing}/{room}.",
                "existing_drawer": {
                    "drawer_id": drawer_id,
                    "content_preview": (existing["documents"][0] or "")[:200],
                    "metadata": existing["metadatas"][0],
                },
                "hint": "Choose a different slug, or use kg_update_entity(entity=drawer_id, ...) to modify the existing drawer's metadata.",
            }
    except Exception:
        pass

    _wal_log(
        "add_drawer",
        {
            "drawer_id": drawer_id,
            "wing": wing,
            "room": room,
            "added_by": added_by,
            "content_length": len(content),
            "content_preview": content[:200],
            "hall": hall,
            "importance": importance,
        },
    )

    now_iso = datetime.now().isoformat()
    meta = {
        "wing": wing,
        "room": room,
        "source_file": source_file or "",
        "chunk_index": 0,
        "added_by": added_by,
        "filed_at": now_iso,
        "date_added": now_iso,
        "last_relevant_at": now_iso,
    }
    if hall is not None:
        meta["hall"] = hall
    if importance is not None:
        meta["importance"] = importance

    try:
        col.upsert(
            ids=[drawer_id],
            documents=[content],
            metadatas=[meta],
        )
        logger.info(f"Filed drawer: {drawer_id} -> {wing}/{room} hall={hall} imp={importance}")

        # Register drawer as a memory entity in SQLite (first-class graph node)
        try:
            _kg.add_entity(
                drawer_id,
                kind="memory",
                description=content[:200],
                importance=importance or 3,
                properties={
                    "wing": wing,
                    "room": room,
                    "hall": hall or "",
                    "added_by": added_by or "",
                },
            )
        except Exception:
            pass  # Non-fatal — drawer exists in ChromaDB regardless

        # ── Context fingerprint (P4.2): keywords → entity_keywords table,
        # view vectors → feedback_contexts collection, context_id → entities row.
        # context is optional here so legacy intent.py callers (which still pass
        # synthetic kwargs) keep working — when present, full Context wiring engages.
        if context:
            try:
                _kg.add_entity_keywords(drawer_id, context.get("keywords") or [])
                cid = persist_context(context, prefix="memory")
                if cid:
                    _kg.set_entity_creation_context(drawer_id, cid)
            except Exception:
                pass  # Non-fatal

        # Create entity→drawer link(s) using the specified predicate
        VALID_DRAWER_PREDICATES = {
            "described_by",
            "evidenced_by",
            "derived_from",
            "mentioned_in",
            "session_note_for",
        }
        link_predicate = predicate if predicate in VALID_DRAWER_PREDICATES else "described_by"

        linked_entities = []
        entity_names = []
        if entity:
            # Support comma-separated list
            entity_names = [e.strip() for e in entity.split(",") if e.strip()]
        else:
            # Default: link to wing name as entity
            entity_names = [wing]

        from .knowledge_graph import normalize_entity_name

        for ename in entity_names:
            eid = normalize_entity_name(ename)
            # Only link to entities that already exist — don't auto-create junk stubs
            existing_entity = _kg.get_entity(eid)
            if not existing_entity:
                # Entity doesn't exist — skip the link. Agent should declare entities
                # via kg_declare_entity before referencing them in drawers.
                continue
            try:
                _kg.add_triple(eid, link_predicate, drawer_id)
                linked_entities.append(eid)
            except Exception:
                pass  # Non-fatal: drawer exists, linking failed

        # ── Suggest related entities for linking (graph enrichment) ──
        # Search entity collection for entities related to this drawer's content.
        # Present suggestions to the agent — they must confirm/reject each.
        suggested_links = []
        try:
            ecol = _get_entity_collection(create=False)
            if ecol and ecol.count() > 0:
                n = min(ecol.count(), 20)
                results = ecol.query(
                    query_texts=[content[:500]],  # use first 500 chars of content
                    n_results=n,
                    include=["documents", "metadatas", "distances"],
                )
                if results["ids"] and results["ids"][0]:
                    already_linked = set(linked_entities)
                    for i, eid in enumerate(results["ids"][0]):
                        if eid in already_linked:
                            continue  # Already linked
                        meta_r = results["metadatas"][0][i] or {}
                        kind = meta_r.get("kind", "entity")
                        if kind not in ("entity", "class"):
                            continue  # Skip predicates, literals
                        dist = results["distances"][0][i]
                        sim = round(max(0.0, 1.0 - dist), 3)
                        if sim < 0.15:
                            continue  # Too dissimilar
                        desc = (results["documents"][0][i] or "")[:100]
                        suggested_links.append(
                            {
                                "entity_id": eid,
                                "similarity": sim,
                                "description": desc,
                            }
                        )
                        if len(suggested_links) >= 5:
                            break
        except Exception:
            pass  # Non-fatal

        result = {
            "success": True,
            "drawer_id": drawer_id,
            "wing": wing,
            "room": room,
            "hall": hall,
            "importance": importance,
            "linked_entities": linked_entities,
        }
        if suggested_links:
            # Informational — agent decides whether to create edges (kg_add) for
            # any of these. P3.13 removed the blocking pending_link_suggestions
            # gate; the resolve_suggestions tool was already deleted in P3.9.
            result["suggested_links"] = suggested_links
            result["link_prompt"] = (
                "Related entities found. For any that should be connected to "
                "this drawer, call kg_add with a precise predicate "
                "(described_by, evidenced_by, derived_from, mentioned_in, "
                "session_note_for). Skip those that shouldn't."
            )

        # ── Drawer duplicate detection ──
        try:
            dup_results = col.query(
                query_texts=[content[:500]],
                n_results=5,
                include=["documents", "distances"],
            )
            if dup_results["ids"] and dup_results["ids"][0]:
                global _pending_conflicts
                dup_conflicts = []
                for i, did in enumerate(dup_results["ids"][0]):
                    if did == drawer_id:
                        continue  # Skip self
                    dist = dup_results["distances"][0][i]
                    sim = round(max(0.0, 1.0 - dist), 3)
                    if sim < 0.85:
                        continue  # Not similar enough
                    conflict_id = f"conflict_drawer_{drawer_id}_{did}"
                    preview = (dup_results["documents"][0][i] or "")[:150]
                    dup_conflicts.append(
                        {
                            "id": conflict_id,
                            "conflict_type": "drawer_duplicate",
                            "reason": f"Similar drawer found (similarity: {sim})",
                            "existing_id": did,
                            "existing_preview": preview,
                            "new_id": drawer_id,
                            "similarity": sim,
                        }
                    )
                if dup_conflicts:
                    _pending_conflicts = dup_conflicts
                    from . import intent

                    intent._persist_active_intent()
                    result["conflicts"] = dup_conflicts
                    result["conflicts_prompt"] = (
                        f"{len(dup_conflicts)} similar drawer(s) found. "
                        f"Call mempalace_resolve_conflicts: merge, keep, or skip."
                    )
        except Exception:
            pass

        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_kg_delete_entity(entity_id: str):
    """Delete an entity (drawer or KG node) and invalidate every edge touching it.

    Works for both drawer memories (ids starting with 'drawer_' or 'diary_')
    and KG entities. Invalidates all current edges where the target is subject
    or object (soft-delete, temporal audit trail preserved), then removes its
    description from the appropriate Chroma collection.

    Use this when an entity is truly obsolete (superseded concept, stale memory,
    deleted person). For updating a single fact (one edge becomes untrue while
    the entity itself remains valid), use kg_invalidate(subject, predicate,
    object) on the specific triple instead.
    """
    if not entity_id or not isinstance(entity_id, str):
        return {"success": False, "error": "entity_id is required (string)."}

    # Determine which collection to target: drawers live in the main drawer
    # collection; everything else in the entity collection.
    is_drawer = entity_id.startswith("drawer_") or entity_id.startswith("diary_")
    col = _get_collection() if is_drawer else _get_entity_collection(create=False)
    if not col:
        return (
            _no_palace()
            if is_drawer
            else {
                "success": False,
                "error": "Entity collection not found.",
            }
        )

    existing = None
    try:
        existing = col.get(ids=[entity_id])
    except Exception as e:
        return {"success": False, "error": f"lookup failed: {e}"}
    if not existing or not existing.get("ids"):
        return {
            "success": False,
            "error": f"Not found in {'drawers' if is_drawer else 'entities'}: {entity_id}",
        }

    deleted_content = (existing.get("documents") or [""])[0] or ""
    deleted_meta = (existing.get("metadatas") or [{}])[0] or {}

    # Invalidate every current edge involving this entity (both directions).
    invalidated = 0
    try:
        edges = _kg.query_entity(entity_id, direction="both") or []
        for e in edges:
            if not e.get("current", True):
                continue
            subj = e.get("subject") or ""
            pred = e.get("predicate") or ""
            obj = e.get("object") or ""
            if not (subj and pred and obj):
                continue
            try:
                _kg.invalidate(subj, pred, obj)
                invalidated += 1
            except Exception:
                continue
    except Exception:
        pass  # kg lookup failure is non-fatal; we still remove from Chroma

    _wal_log(
        "kg_delete_entity",
        {
            "entity_id": entity_id,
            "collection": "drawer" if is_drawer else "entity",
            "edges_invalidated": invalidated,
            "deleted_meta": deleted_meta,
            "content_preview": deleted_content[:200],
        },
    )

    try:
        col.delete(ids=[entity_id])
        logger.info(
            f"Deleted {'drawer' if is_drawer else 'entity'}: {entity_id} ({invalidated} edges invalidated)"
        )
        return {
            "success": True,
            "entity_id": entity_id,
            "source": "drawer" if is_drawer else "entity",
            "edges_invalidated": invalidated,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_wake_up(wing: str = None, agent: str = None):
    """Boot context for a session. Call ONCE at start.

    Returns protocol (behavioral rules), text (identity + top drawers),
    and declared (compact summary of auto-declared entities).

    Args:
        wing: Optional wing filter. If set, L1 loads only drawers in that wing.
        agent: Optional agent identity. When set, drawers filed by this agent
               get a ranking boost in L1 selection.
    """
    try:
        from .layers import MemoryStack
    except Exception as e:
        return {"success": False, "error": f"layers module unavailable: {e}"}

    try:
        stack = MemoryStack()
        text = stack.wake_up(wing=wing, agent=agent)
        token_estimate = len(text) // 4
        from .knowledge_graph import normalize_entity_name

        # 1. Predicates — declare + collect names
        predicates = _kg.list_entities(status="active", kind="predicate")
        pred_names = []
        for p in predicates:
            _declared_entities.add(p["id"])
            pred_names.append(p["id"])

        # 2. Classes — declare + collect names
        classes = _kg.list_entities(status="active", kind="class")
        class_names = []
        for c in classes:
            _declared_entities.add(c["id"])
            class_names.append(c["id"])

        # 3. Intent types — walk is-a tree, compact format
        #    Intent types are kind=class (they are types, not instances).
        #    Intent executions are kind=entity with is_a pointing to a class.
        entities = _kg.list_entities(status="active", kind="class")
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
                    e_edges = _kg.query_entity(e["id"], direction="outgoing")
                    for edge in e_edges:
                        if edge["predicate"] in ("is-a", "is_a") and edge["current"]:
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
            _declared_entities.add(e["id"])
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

        # 4. Top entities (non-intent) — name[importance]
        entity_parts = []
        if wing:
            top_ents = [e for e in entities if e["id"] not in intent_type_ids][:20]
            for e in top_ents:
                _declared_entities.add(e["id"])
                entity_parts.append(e["id"] + "[" + str(e.get("importance", 3)) + "]")

        # Load learned scoring weights from feedback history
        try:
            from .scoring import set_learned_weights, DEFAULT_SEARCH_WEIGHTS

            learned = _kg.compute_learned_weights(DEFAULT_SEARCH_WEIGHTS)
            set_learned_weights(learned)
        except Exception:
            pass

        return {
            "success": True,
            "wing": wing,
            "protocol": PALACE_PROTOCOL,
            "text": text,
            "estimated_tokens": token_estimate,
            "declared": {
                "predicates": ", ".join(sorted(pred_names)),
                "classes": ", ".join(sorted(class_names)),
                "intent_types": " | ".join(intent_parts),
                "entities": ", ".join(entity_parts),
                "count": len(_declared_entities),
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# tool_update_drawer_metadata removed (P3.4): merged into tool_kg_update_entity.
# Call kg_update_entity(entity=drawer_id, wing=..., room=..., hall=..., importance=...).


# ==================== KNOWLEDGE GRAPH ====================


def tool_kg_query(entity: str, as_of: str = None, direction: str = "both"):
    """Query the knowledge graph for an entity's relationships.

    Supports batch queries: pass a comma-separated list of entity names
    to query multiple entities in one call. Returns results keyed by entity.
    """
    entities = [e.strip() for e in entity.split(",") if e.strip()]

    # Track queried entities for mandatory feedback enforcement
    if _active_intent and isinstance(_active_intent.get("accessed_memory_ids"), set):
        for ename in entities:
            _active_intent["accessed_memory_ids"].add(ename)

    if len(entities) == 1:
        # Single entity — original format for backwards compatibility
        results = _kg.query_entity(entities[0], as_of=as_of, direction=direction)
        return {"entity": entities[0], "as_of": as_of, "facts": results, "count": len(results)}

    # Batch query — return results keyed by entity name
    batch_results = {}
    total_count = 0
    for ename in entities:
        facts = _kg.query_entity(ename, as_of=as_of, direction=direction)
        batch_results[ename] = {"facts": facts, "count": len(facts)}
        total_count += len(facts)

    return {"entities": batch_results, "as_of": as_of, "total_count": total_count, "batch": True}


def tool_kg_search(  # noqa: C901
    queries,
    limit: int = 10,
    wing: str = None,
    room: str = None,
    kind: str = None,
    sort_by: str = "hybrid",
    agent: str = None,
):
    """Unified palace search — drawers (prose memories) + entities (KG nodes) in one call.

    Runs the shared multi-channel pipeline (cosine + keyword for drawers,
    cosine + keyword + 1-hop graph for entities), then RRF-merges across both
    collections into a single ranked result list. Each hit carries a `source`
    field ('drawer' | 'entity') and the fields relevant to its type.

    The merge is a true cross-collection RRF: every channel's ranked list
    participates, regardless of source, so a drawer and an entity answering
    the same question compete head-to-head.

    Args:
        queries: MANDATORY list of 2-5 perspective strings. Each becomes a
            separate cosine view for multi-view retrieval. Passing a plain
            string is REJECTED — multi-view is the whole point, same contract
            as declare_intent descriptions.
            Example: ["auth rate limiting", "brute force hardening", "login"]
        limit: Max results across drawers+entities (default 10; adaptive-K
            may trim based on score gaps).
        wing, room: Optional drawer filters (silently ignored for entities).
        kind: Optional entity kind filter ('entity', 'predicate', 'class',
            'literal'). Silently ignored for drawers.
        sort_by: 'hybrid' (default) — RRF order, reranked by hybrid_score
            (similarity + importance + decay + agent + intent feedback).
            'similarity' — pure cosine.
        agent: Agent name for affinity scoring.
    """
    from .scoring import rrf_merge

    # ── Shared validation ──
    query_views, err = validate_query_views(queries, min_views=2, max_views=5)
    if err:
        return err

    sanitized_views = [sanitize_query(v)["clean_query"] for v in query_views]
    sanitized_views = [v for v in sanitized_views if v]
    if not sanitized_views:
        return {"success": False, "error": "All queries were empty after sanitization."}

    # ── Source scoping: wing/room → drawers only; kind → entities only ──
    # If all three are set, it's a contradiction → empty result with hint.
    drawer_filter_set = bool(wing or room)
    entity_filter_set = bool(kind)
    if drawer_filter_set and entity_filter_set:
        return {
            "success": False,
            "error": (
                "Cannot combine drawer filters (wing/room) with entity filters (kind) — "
                "they target different sources. Use one or the other, or neither "
                "(to search both)."
            ),
        }
    search_drawers = not entity_filter_set
    search_entities = not drawer_filter_set

    try:
        # ── Run pipeline over selected collections ──
        all_lists = {}
        combined_meta = {}

        if search_drawers:
            drawer_col = _get_collection(create=False)
            drawer_pipe = multi_channel_search(
                drawer_col,
                sanitized_views,
                kg=_kg,
                wing=wing,
                fetch_limit_per_view=max(limit * 3, 30),
                include_graph=False,
            )
            for name, lst in drawer_pipe["ranked_lists"].items():
                all_lists[f"drawer_{name}"] = lst
            for mid, info in drawer_pipe["seen_meta"].items():
                combined_meta[mid] = {**info, "source": "drawer"}

        if search_entities:
            entity_col = _get_entity_collection(create=False)
            entity_pipe = multi_channel_search(
                entity_col,
                sanitized_views,
                kg=_kg,
                kind=kind,
                fetch_limit_per_view=max(limit * 3, 30),
                include_graph=True,
            )
            for name, lst in entity_pipe["ranked_lists"].items():
                all_lists[f"entity_{name}"] = lst
            for mid, info in entity_pipe["seen_meta"].items():
                # Entity overrides drawer if the same id lives in both (shouldn't happen)
                combined_meta[mid] = {**info, "source": "entity"}

        if not all_lists:
            return {"queries": sanitized_views, "results": [], "count": 0, "sort_by": sort_by}

        rrf_scores, _cm, _attr = rrf_merge(all_lists)

        # ── Post-merge room filter (drawers only, applied here since the
        # pipeline only handles wing) ──
        if room:
            filtered = {}
            for mid, score in rrf_scores.items():
                info = combined_meta.get(mid, {})
                if info.get("source") == "drawer":
                    if (info.get("meta") or {}).get("room") != room:
                        continue
                filtered[mid] = score
            rrf_scores = filtered

        # ── Relevance-feedback lookup (shared) ──
        useful_ids, irrelevant_ids = lookup_type_feedback(_active_intent, _kg)

        # ── Assemble candidates with source-specific shape ──
        candidates = []
        for mid, _rrf in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            info = combined_meta.get(mid)
            if not info:
                continue
            meta = info["meta"] or {}
            doc = info["doc"] or ""
            similarity = info.get("similarity", 0.0)
            source = info["source"]
            importance = meta.get("importance", 3)
            date_anchor = (
                meta.get("last_touched") or meta.get("date_added") or meta.get("filed_at") or ""
            )

            if sort_by == "hybrid":
                is_match = bool(agent and meta.get("added_by") == agent)
                last_relevant = meta.get("last_relevant_at", "")
                rel_fb = 1 if mid in useful_ids else (-1 if mid in irrelevant_ids else 0)
                final_score = _hybrid_score_fn(
                    similarity=similarity,
                    importance=importance,
                    date_iso=date_anchor,
                    agent_match=is_match,
                    last_relevant_iso=last_relevant,
                    relevance_feedback=rel_fb,
                    mode="search",
                )
            else:
                final_score = similarity

            entry = {
                "id": mid,
                "source": source,
                "importance": importance,
                "similarity": similarity,
                "score": round(final_score, 4),
            }
            if source == "drawer":
                entry["text"] = doc[:300]
                entry["wing"] = meta.get("wing", "")
                entry["room"] = meta.get("room", "")
            else:
                entry["name"] = meta.get("name", mid)
                entry["description"] = doc
                entry["kind"] = meta.get("kind", "entity")
            candidates.append(entry)

        # ── Adaptive-K ──
        if sort_by == "hybrid" and len(candidates) > 1:
            k = adaptive_k([c["score"] for c in candidates], max_k=limit, min_k=1)
            top = candidates[:k]
        else:
            top = candidates[:limit]

        # ── Attach current edges for entity results only ──
        for entry in top:
            if entry["source"] == "entity":
                edges = _kg.query_entity(entry["id"], direction="both")
                current_edges = [e for e in edges if e.get("current", True)]
                entry["edges"] = current_edges
                entry["edge_count"] = len(current_edges)

        # ── Track accessed items for mandatory feedback enforcement ──
        if _active_intent and isinstance(_active_intent.get("accessed_memory_ids"), set):
            for entry in top:
                _active_intent["accessed_memory_ids"].add(entry["id"])

        return {
            "queries": sanitized_views,
            "results": top,
            "count": len(top),
            "sort_by": sort_by,
        }
    except Exception as e:
        return {"success": False, "error": f"kg_search failed: {e}"}


def tool_kg_add(  # noqa: C901
    subject: str,
    predicate: str,
    object: str,
    context: dict = None,  # P4.3 — mandatory Context fingerprint for the edge
    valid_from: str = None,
    source_closet: str = None,
):
    """Add a relationship to the knowledge graph (P4.3 — Context mandatory).

    IMPORTANT: All three parts must be declared in this session:
    - subject: declared entity (any type EXCEPT predicate)
    - predicate: declared entity with type="predicate"
    - object: declared entity (any type EXCEPT predicate)

    The MANDATORY `context` records WHY this edge is being added — the
    multi-view perspectives + caller-provided keywords + optional related
    entities. Its view vectors are persisted in mempalace_feedback_contexts
    and the resulting context_id is stored on the triple's
    creation_context_id column. Future feedback (found_useful etc.) applies
    by MaxSim against this fingerprint.

    Call kg_declare_entity for subject/object entities, and
    kg_declare_entity with kind="predicate" for predicates.
    """
    from .knowledge_graph import normalize_entity_name
    from .scoring import validate_context

    # ── Validate Context (mandatory) ──
    clean_context, ctx_err = validate_context(context)
    if ctx_err:
        return ctx_err

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
            f"kg_declare_entity(name='{subject}', description='...', kind='entity')"
        )
    else:
        sub_entity = _kg.get_entity(sub_normalized)
        if sub_entity and sub_entity.get("kind") == "predicate":
            errors.append(
                f"subject '{sub_normalized}' is kind=predicate, not an entity. "
                f"Subjects must be kind=entity (or class/literal)."
            )

    # Check predicate (must be declared as type="predicate")
    if not _is_declared(pred_normalized):
        errors.append(
            f"predicate '{pred_normalized}' not declared. Call: "
            f"kg_declare_entity(name='{predicate}', description='...', kind='predicate')"
        )
    else:
        pred_entity = _kg.get_entity(pred_normalized)
        if pred_entity and pred_entity.get("kind") != "predicate":
            errors.append(
                f"'{pred_normalized}' is kind='{pred_entity.get('kind')}', not 'predicate'. "
                f"Predicates must be declared with kind='predicate'."
            )

    # Check object (must be declared, must NOT be a predicate)
    if not _is_declared(obj_normalized):
        errors.append(
            f"object '{obj_normalized}' not declared. Call: "
            f"kg_declare_entity(name='{object}', description='...', kind='entity')"
        )
    else:
        obj_entity = _kg.get_entity(obj_normalized)
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
                parent_edges = _kg.query_entity(cls, direction="outgoing")
                for e in parent_edges:
                    if e["predicate"] == "is-a" and e["current"]:
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
                    for e in _kg.query_entity(sub_normalized, direction="outgoing")
                    if e["predicate"] == "is-a" and e["current"]
                ]
                if sub_classes and not _is_subclass_of(sub_classes, allowed_sub_classes):
                    constraint_errors.append(
                        f"Subject class mismatch: '{sub_normalized}' is-a {sub_classes}, "
                        f"but predicate '{pred_normalized}' expects subject is-a {allowed_sub_classes}. "
                        f"Options: (1) wrong edge — use a different subject, "
                        f"(2) wrong predicate — check kg_list_declared() for a better fit, "
                        f"(3) missing classification — add is-a edge for '{sub_normalized}', "
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
                    for e in _kg.query_entity(obj_normalized, direction="outgoing")
                    if e["predicate"] == "is-a" and e["current"]
                ]
                if obj_classes and not _is_subclass_of(obj_classes, allowed_obj_classes):
                    constraint_errors.append(
                        f"Object class mismatch: '{obj_normalized}' is-a {obj_classes}, "
                        f"but predicate '{pred_normalized}' expects object is-a {allowed_obj_classes}. "
                        f"Options: (1) wrong edge, (2) wrong predicate, (3) missing classification, "
                        f"(4) update constraints, (5) new predicate, (6) rephrase with specific entity."
                    )

            # Check cardinality constraint
            cardinality = pred_constraints.get("cardinality", "many-to-many")
            if cardinality in ("many-to-one", "one-to-one"):
                # Subject can have at most 1 edge with this predicate
                existing_sub = [
                    e
                    for e in _kg.query_entity(sub_normalized, direction="outgoing")
                    if e["predicate"] == pred_normalized and e["current"]
                ]
                if existing_sub:
                    existing_obj = existing_sub[0]["object"]
                    constraint_errors.append(
                        f"Cardinality violation: '{sub_normalized}' already has "
                        f"'{pred_normalized}' -> '{existing_obj}'. "
                        f"Predicate cardinality is {cardinality} (one target per subject). "
                        f"Options: (1) REPLACE — invalidate the old edge first, then add new, "
                        f"(2) MISTAKE — you meant a different predicate or entity, "
                        f"(3) EXPAND — change predicate cardinality to many-to-many."
                    )
            if cardinality in ("one-to-many", "one-to-one"):
                # Object can have at most 1 incoming edge with this predicate
                existing_obj = [
                    e
                    for e in _kg.query_entity(obj_normalized, direction="incoming")
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

    # ── Persist the edge's creation Context (P4.3) — view vectors → feedback_contexts,
    # context_id → triples.creation_context_id.
    edge_context_id = persist_context(clean_context, prefix="edge")

    _wal_log(
        "kg_add",
        {
            "subject": subject,
            "predicate": predicate,
            "object": object,
            "valid_from": valid_from,
            "source_closet": source_closet,
            "context_id": edge_context_id,
        },
    )
    triple_id = _kg.add_triple(
        sub_normalized,
        pred_normalized,
        obj_normalized,
        valid_from=valid_from,
        source_closet=source_closet,
        creation_context_id=edge_context_id,
    )

    # ── Contradiction detection: find existing edges that may conflict ──
    global _pending_conflicts
    conflicts = []
    try:
        # Skip is_a — those aren't factual contradictions
        if pred_normalized not in ("is_a", "is-a"):
            existing_edges = _kg.query_entity(sub_normalized, direction="outgoing")
            for e in existing_edges:
                if not e.get("current", True):
                    continue
                if e["predicate"] != pred_normalized:
                    continue
                existing_obj = e["object"]
                if existing_obj == obj_normalized:
                    continue  # Same edge — not a contradiction
                # Found: same subject + same predicate + different object
                conflict_id = f"conflict_{sub_normalized}_{pred_normalized}_{existing_obj}"
                conflicts.append(
                    {
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
                )
    except Exception:
        pass  # Non-fatal — contradiction detection is best-effort

    result = {
        "success": True,
        "triple_id": triple_id,
        "fact": f"{sub_normalized} -> {pred_normalized} -> {obj_normalized}",
    }

    if conflicts:
        _pending_conflicts = conflicts
        intent._persist_active_intent()
        result["conflicts"] = conflicts
        result["conflicts_prompt"] = (
            f"{len(conflicts)} potential contradiction(s) found. "
            f"You MUST call mempalace_resolve_conflicts to address each: "
            f"invalidate (old is stale), keep (both valid), or skip (undo new)."
        )

    return result


def tool_kg_add_batch(edges: list, context: dict = None):
    """Add multiple KG edges in one call (P4.3 — Context mandatory).

    Each edge: {subject, predicate, object, valid_from?, source_closet?, context?}.

    The TOP-LEVEL `context` is the shared default applied to every edge that
    doesn't carry its own — most batches add edges that all reflect the same
    'why' (a single agent decision), so one Context covers them. An edge can
    still override with its own `context` dict if needed. Validates each edge
    independently — partial success OK.
    """
    from .scoring import validate_context

    if not edges or not isinstance(edges, list):
        return {
            "success": False,
            "error": "edges must be a non-empty list of {subject, predicate, object} dicts.",
        }

    # Validate the shared/default context (if provided) once up front so we
    # surface a clean error before doing any per-edge work.
    default_clean_context = None
    if context is not None:
        default_clean_context, ctx_err = validate_context(context)
        if ctx_err:
            return ctx_err

    results = []
    succeeded = 0
    for edge in edges:
        if not isinstance(edge, dict):
            results.append({"success": False, "error": "edge must be a dict"})
            continue
        edge_context = edge.get("context") or default_clean_context
        if edge_context is None:
            results.append(
                {
                    "success": False,
                    "error": (
                        "Each edge needs a context — pass one at the top level of "
                        "kg_add_batch (shared default for all edges) or per-edge."
                    ),
                }
            )
            continue
        r = tool_kg_add(
            subject=edge.get("subject", ""),
            predicate=edge.get("predicate", ""),
            object=edge.get("object", ""),
            context=edge_context,
            valid_from=edge.get("valid_from"),
            source_closet=edge.get("source_closet"),
        )
        results.append(r)
        if r.get("success"):
            succeeded += 1

    return {
        "success": succeeded > 0,
        "total": len(edges),
        "succeeded": succeeded,
        "failed": len(edges) - succeeded,
        "results": results,
    }


def tool_kg_invalidate(subject: str, predicate: str, object: str, ended: str = None):
    """Mark a fact as no longer true (set end date)."""
    try:
        _wal_log(
            "kg_invalidate",
            {"subject": subject, "predicate": predicate, "object": object, "ended": ended},
        )
        _kg.invalidate(subject, predicate, object, ended=ended)
        return {
            "success": True,
            "fact": f"{subject} → {predicate} → {object}",
            "ended": ended or "today",
        }
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


def tool_kg_timeline(entity: str = None):
    """Get chronological timeline of facts, optionally for one entity."""
    results = _kg.timeline(entity)
    return {"entity": entity or "all", "timeline": results, "count": len(results)}


def tool_kg_stats():
    """Palace overview — entities, triples, relationship types, and (if a
    drawer collection exists) connectivity stats from the underlying graph.
    """
    stats = _kg.stats() or {}
    # Also fold in drawer/wing graph metrics when available (was graph_stats)
    try:
        col = _get_collection()
        if col:
            stats["graph"] = graph_stats(col=col)
    except Exception:
        pass
    return stats


# ==================== ENTITY DECLARATION ====================

ENTITY_SIMILARITY_THRESHOLD = 0.85
ENTITY_COLLECTION_NAME = "mempalace_entities"

# Session-level declared entities (in-memory cache, falls back to persistent KG)
_declared_entities: set = set()
_session_id: str = ""
_pending_conflicts = None  # Unified conflict resolution — blocks ALL tools until resolved

# ── Session isolation: save/restore state per session_id ──
# When multiple callers (sub-agents) share the same MCP process but have different
# session IDs, this prevents them from overwriting each other's state.
_session_state: dict = {}  # session_id -> {active_intent, pending_edges, declared, pending_conflicts}


def _save_session_state():
    """Save current session state before switching to a different session."""
    if _session_id:
        _session_state[_session_id] = {
            "active_intent": _active_intent,
            "pending_conflicts": _pending_conflicts,
            "declared": _declared_entities,
        }


def _load_pending_conflicts_from_disk(session_id: str = None) -> list:
    """Load pending conflicts from the active intent state file.

    Disk is the source of truth for cross-session/cross-restart state.
    Returns empty list if no file or no conflicts pending.
    """
    sid = session_id or _session_id or "default"
    try:
        state_file = _INTENT_STATE_DIR / f"active_intent_{sid}.json"
        if not state_file.is_file():
            return []
        data = json.loads(state_file.read_text(encoding="utf-8"))
        return data.get("pending_conflicts") or []
    except Exception:
        return []


def _restore_session_state(sid: str):
    """Restore session state for the given session_id.

    Memory is a cache; disk is the source of truth. When switching sessions,
    we reload pending_conflicts from disk so blocking state is always correct.
    """
    global _active_intent, _pending_conflicts, _declared_entities
    if sid in _session_state:
        s = _session_state[sid]
        _active_intent = s["active_intent"]
        _pending_conflicts = s.get("pending_conflicts")
        _declared_entities = s["declared"]
    else:
        _active_intent = None
        _pending_conflicts = None
        _declared_entities = set()
    # Always reconcile pending_conflicts with disk (authoritative source)
    disk_conflicts = _load_pending_conflicts_from_disk(sid)
    if disk_conflicts:
        _pending_conflicts = disk_conflicts


def _is_declared(entity_id: str) -> bool:
    """Check if an entity is declared, with fallback to persistent KG.

    The in-memory _declared_entities set is a cache that gets cleared on
    MCP server restart. If an entity isn't in the cache but exists in the
    persistent KG (ChromaDB), it's auto-added to the cache and considered
    declared. This makes declarations survive MCP server restarts without
    requiring the model to re-call wake_up.
    """
    if entity_id in _declared_entities:
        return True

    # Fallback: check persistent KG
    ecol = _get_entity_collection(create=False)
    if ecol:
        try:
            result = ecol.get(ids=[entity_id])
            if result and result["ids"]:
                _declared_entities.add(entity_id)
                return True
        except Exception:
            pass

    return False


def _get_entity_collection(create: bool = True):
    """Get or create the mempalace_entities ChromaDB collection for description similarity."""
    try:
        client = chromadb.PersistentClient(path=_config.palace_path)
        if create:
            return client.get_or_create_collection(ENTITY_COLLECTION_NAME)
        else:
            return client.get_collection(ENTITY_COLLECTION_NAME)
    except Exception:
        if create:
            try:
                client = chromadb.PersistentClient(path=_config.palace_path)
                return client.create_collection(ENTITY_COLLECTION_NAME)
            except Exception:
                return None
        return None


FEEDBACK_CONTEXT_COLLECTION = "mempalace_feedback_contexts"


def _get_feedback_context_collection(create: bool = True):
    """Get or create the feedback contexts ChromaDB collection.

    Stores multi-view context vectors alongside feedback records.
    Each entry = one context snapshot (multiple views stored as separate embeddings).
    Used for MaxSim comparison when applying stored feedback.
    """
    try:
        client = chromadb.PersistentClient(path=_config.palace_path)
        if create:
            return client.get_or_create_collection(FEEDBACK_CONTEXT_COLLECTION)
        else:
            return client.get_collection(FEEDBACK_CONTEXT_COLLECTION)
    except Exception:
        if create:
            try:
                client = chromadb.PersistentClient(path=_config.palace_path)
                return client.create_collection(FEEDBACK_CONTEXT_COLLECTION)
            except Exception:
                return None
        return None


def _generate_context_id(prefix: str, views: list) -> str:
    """Deterministic-ish context id from views + a prefix + timestamp.

    Hash of the sorted view strings keeps it stable when the same Context is
    re-used; the timestamp suffix prevents the rare collision when the SAME
    views are filed under genuinely different intents at the same instant.
    """
    import hashlib
    from datetime import datetime as _dt

    text = "\n".join(sorted(v.strip() for v in (views or []) if isinstance(v, str)))
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    ts = _dt.now().strftime("%Y%m%dT%H%M%S")
    return f"ctx_{prefix}_{digest}_{ts}"


def persist_context(context: dict, *, prefix: str = "entity") -> str:
    """Persist a Context object's view vectors to the feedback_contexts collection.

    Returns the generated context_id (or empty string on failure). Used by
    write tools (kg_declare_entity, kg_add) to record the creation Context so
    later MaxSim comparisons can apply feedback by similarity.
    """
    if not context or not isinstance(context, dict):
        return ""
    views = context.get("queries") or []
    if not views:
        return ""
    cid = _generate_context_id(prefix, views)
    stored = store_feedback_context(cid, views)
    return stored or ""


def store_feedback_context(context_id: str, views: list):
    """Store multi-view context vectors in ChromaDB for MaxSim comparison.

    Each view is stored as a separate document with the same context_id prefix.
    To compute MaxSim, query with current views and find matching context_ids.
    """
    col = _get_feedback_context_collection(create=True)
    if not col or not views:
        return None
    try:
        ids = []
        docs = []
        metas = []
        for i, view in enumerate(views):
            if not view or not view.strip():
                continue
            vid = f"{context_id}_v{i}"
            ids.append(vid)
            docs.append(view)
            metas.append({"context_id": context_id, "view_index": i})
        if ids:
            col.upsert(ids=ids, documents=docs, metadatas=metas)
        return context_id
    except Exception:
        return None


def maxsim_context_match(current_views: list, stored_context_ids: list, threshold: float = 0.7):
    """Compute MaxSim between current context views and stored context(s).

    MaxSim(A, B) = (1/|A|) * sum(max(cos(a, b) for b in B) for a in A)

    Returns dict of context_id -> maxsim_score for contexts above threshold.
    """
    col = _get_feedback_context_collection(create=False)
    if not col or not current_views or not stored_context_ids:
        return {}

    try:
        # Get all stored view IDs for the given context_ids
        all_stored_ids = []
        for cid in stored_context_ids:
            # Query by metadata filter for this context_id
            try:
                stored = col.get(where={"context_id": cid}, include=["embeddings"])
                if stored and stored["ids"]:
                    all_stored_ids.extend(stored["ids"])
            except Exception:
                pass

        if not all_stored_ids:
            return {}

        # For each current view, find max similarity to any stored vector
        results = {}
        for cid in stored_context_ids:
            max_sims = []
            for view in current_views:
                if not view or not view.strip():
                    continue
                try:
                    # Query stored vectors filtered to this context_id
                    res = col.query(
                        query_texts=[view],
                        n_results=min(col.count(), 10),
                        where={"context_id": cid},
                        include=["distances"],
                    )
                    if res["distances"] and res["distances"][0]:
                        # Min distance = max similarity
                        min_dist = min(res["distances"][0])
                        max_sim = max(0.0, 1.0 - min_dist)
                        max_sims.append(max_sim)
                except Exception:
                    pass
            if max_sims:
                # MaxSim = average of per-view max similarities
                results[cid] = sum(max_sims) / len(max_sims)

        # Filter by threshold
        return {cid: score for cid, score in results.items() if score >= threshold}
    except Exception:
        return {}


def _reset_declared_entities():
    """Reset the session's declared entities set (called on compact/clear/restart)."""
    global _declared_entities, _session_id
    _declared_entities = set()
    _session_id = ""


def _check_entity_similarity_multiview(
    views: list,
    kind_filter: str = None,
    exclude_id: str = None,
    threshold: float = None,
):
    """Multi-view collision detection (P4.2).

    Each view is queried independently against the entity collection; the
    per-view ranked candidates are merged via Reciprocal Rank Fusion. A hit
    is reported as a collision when its highest single-view similarity is
    above threshold (so one strong match still flags, but multi-view gives
    catches that single-vector cosine misses).

    Logical entity ids are recovered from suffix-style chroma ids
    ('entity_id::view_N' → 'entity_id'). Returns the same shape as
    _check_entity_similarity for drop-in compatibility.
    """
    from .scoring import rrf_merge

    threshold = threshold or ENTITY_SIMILARITY_THRESHOLD
    ecol = _get_entity_collection(create=False)
    if not ecol or not views:
        return []
    try:
        count = ecol.count()
        if count == 0:
            return []
        per_view_lists = {}
        per_id_best = {}  # logical entity_id -> (best_similarity, doc, meta)
        for vi, view in enumerate(views):
            if not view or not view.strip():
                continue
            kwargs = {
                "query_texts": [view],
                "n_results": min(20, count),
                "include": ["documents", "metadatas", "distances"],
            }
            results = ecol.query(**kwargs)
            if not (results.get("ids") and results["ids"][0]):
                continue
            view_candidates = []
            for i, raw_id in enumerate(results["ids"][0]):
                # Strip the ::view_N suffix to get the logical entity id
                logical_id = raw_id.split("::", 1)[0]
                if logical_id == exclude_id:
                    continue
                dist = results["distances"][0][i]
                sim = round(1 - dist, 3)
                meta = results["metadatas"][0][i] or {}
                if kind_filter and meta.get("kind") != kind_filter:
                    continue
                doc = results["documents"][0][i] or ""
                view_candidates.append((sim, doc, logical_id))
                prev = per_id_best.get(logical_id)
                if prev is None or sim > prev[0]:
                    per_id_best[logical_id] = (sim, doc, meta)
            if view_candidates:
                per_view_lists[f"cosine_{vi}"] = view_candidates

        if not per_view_lists:
            return []

        rrf_scores, _cm, _attr = rrf_merge(per_view_lists)
        # Order by RRF, but only emit entities whose best single-view sim is above threshold.
        similar = []
        for eid, _rrf in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            best = per_id_best.get(eid)
            if not best:
                continue
            sim, doc, meta = best
            if sim < threshold:
                continue
            similar.append(
                {
                    "entity_id": eid,
                    "name": meta.get("name", eid),
                    "description": doc,
                    "similarity": sim,
                    "importance": meta.get("importance", 3),
                }
            )
        return similar
    except Exception:
        return []


def _check_entity_similarity(
    description: str,
    kind_filter: str = None,
    exclude_id: str = None,
    threshold: float = None,
):
    """LEGACY single-view collision check.

    Kept for callers that still pass a single description (intent.py finalize,
    update flows). New multi-view path is _check_entity_similarity_multiview.
    """
    threshold = threshold or ENTITY_SIMILARITY_THRESHOLD
    ecol = _get_entity_collection(create=False)
    if not ecol:
        return []
    try:
        count = ecol.count()
        if count == 0:
            return []
        query_kwargs = {
            "query_texts": [description],
            "n_results": min(10, count),
            "include": ["documents", "metadatas", "distances"],
        }
        # Kind-scoped collision: only check against same kind
        if kind_filter:
            query_kwargs["where"] = {}
        results = ecol.query(**query_kwargs)
        similar = []
        if results["ids"] and results["ids"][0]:
            for i, eid in enumerate(results["ids"][0]):
                if eid == exclude_id:
                    continue
                dist = results["distances"][0][i]
                similarity = round(1 - dist, 3)
                if similarity >= threshold:
                    meta = results["metadatas"][0][i]
                    doc = results["documents"][0][i]
                    similar.append(
                        {
                            "entity_id": eid,
                            "name": meta.get("name", eid),
                            "description": doc,
                            "similarity": similarity,
                            "importance": meta.get("importance", 3),
                        }
                    )
        return similar
    except Exception:
        return []


def _create_entity(
    name: str,
    kind: str = "entity",
    description: str = "",
    importance: int = 3,
    properties: dict = None,
    added_by: str = None,
    embed_text: str = None,
):
    """Create an entity in BOTH SQLite AND ChromaDB. Use this instead of _kg.add_entity directly.

    Args:
        embed_text: Optional override for what gets embedded in ChromaDB.
                    If None, uses description. Use for execution entities
                    where you want description-only embedding (no summary).
    """
    from .knowledge_graph import normalize_entity_name

    eid = _kg.add_entity(
        name, kind=kind, description=description, importance=importance, properties=properties
    )
    normalized = normalize_entity_name(name)
    _sync_entity_to_chromadb(
        normalized, name, embed_text or description, kind, importance, added_by=added_by
    )
    return eid


def _sync_entity_to_chromadb(
    entity_id: str, name: str, description: str, kind: str, importance: int, added_by: str = None
):
    """Sync an entity's description to the ChromaDB collection for similarity search.

    LEGACY single-view path. New code should call _sync_entity_views_to_chromadb
    with the full Context.queries list (P4.2). Kept for intent.py callers that
    still synthesize single descriptions.
    """
    ecol = _get_entity_collection(create=True)
    if not ecol:
        return
    meta = {
        "name": name,
        "kind": kind,
        "importance": importance,
        "last_touched": datetime.now().isoformat(),
    }
    if added_by:
        meta["added_by"] = added_by
    ecol.upsert(
        ids=[entity_id],
        documents=[description],
        metadatas=[meta],
    )


def _sync_entity_views_to_chromadb(
    entity_id: str,
    name: str,
    views: list,
    kind: str,
    importance: int,
    added_by: str = None,
):
    """Multi-view sync to the entity ChromaDB collection (P4.2).

    Each view is stored as a separate record under '{entity_id}::view_{N}' so
    every angle is independently searchable. The collision detector
    (_check_entity_similarity_multiview) strips the suffix to recover the
    logical entity_id and RRF-merges per-view rankings.

    For a single-view list, behaves equivalent to the legacy single-record
    upsert (just with the suffix scheme — caller-transparent).
    """
    ecol = _get_entity_collection(create=True)
    if not ecol or not views:
        return
    cleaned = [v for v in views if isinstance(v, str) and v.strip()]
    if not cleaned:
        return
    now_iso = datetime.now().isoformat()
    base_meta = {
        "name": name,
        "kind": kind,
        "importance": importance,
        "last_touched": now_iso,
    }
    if added_by:
        base_meta["added_by"] = added_by

    ids, docs, metas = [], [], []
    for i, view in enumerate(cleaned):
        ids.append(f"{entity_id}::view_{i}")
        docs.append(view)
        m = dict(base_meta)
        m["view_index"] = i
        m["entity_id"] = entity_id  # explicit reverse lookup, not just id-suffix parsing
        metas.append(m)
    ecol.upsert(ids=ids, documents=docs, metadatas=metas)


VALID_CARDINALITIES = {"many-to-many", "many-to-one", "one-to-many", "one-to-one"}


def tool_kg_declare_entity(  # noqa: C901
    name: str = None,
    context: dict = None,  # P4.2 — mandatory: {queries, keywords, entities?}
    kind: str = None,  # REQUIRED — no default, model must choose
    importance: int = 3,
    properties: dict = None,  # General-purpose metadata
    user_approved_star_scope: bool = False,  # Required for * scope
    added_by: str = None,  # REQUIRED — agent who declared this entity
    # Memory-kind specific (REQUIRED when kind='memory').
    wing: str = None,
    room: str = None,
    slug: str = None,
    content: str = None,  # verbatim memory text (kind='memory' only); for other kinds, queries[0] is canonical
    hall: str = None,
    source_file: str = None,
    entity: str = None,  # entity name(s) to link this memory to
    predicate: str = "described_by",  # link predicate
    # ── Legacy single-string description path (P4.2 — REMOVED) ──
    description: str = None,  # accepted only as a hard-error trigger, see below
):
    """Declare an entity before using it in KG edges. REQUIRED per session.

    EVERY declaration speaks the unified Context object (P4.2):

        context = {
          "queries":  list[str]   # 2-5 perspectives on what this entity is
          "keywords": list[str]   # 2-5 caller-provided exact terms
          "entities": list[str]   # 0+ related entity ids (optional)
        }

    Each query gets embedded as a separate Chroma record under
    '{entity_id}::view_N', so collision detection is multi-view RRF rather
    than single-vector cosine. Keywords are stored in entity_keywords (the
    keyword channel reads them directly — auto-extraction is gone). The
    Context's view vectors are also persisted in mempalace_feedback_contexts
    under a generated context_id, recorded on the entity, so future
    feedback (found_useful / found_irrelevant) applies via MaxSim by
    context similarity.

    Args:
        name: Entity name (REQUIRED for kind=entity/class/predicate/literal;
              auto-computed from wing/room/slug for kind='memory').
        context: MANDATORY Context dict — see above. Replaces the single
              `description` parameter (which is now rejected with an error).
        kind: 'entity' | 'class' | 'predicate' | 'literal' | 'memory'.
        content: VERBATIM text for kind='memory' (the actual memory body).
              For non-memory kinds, queries[0] is used as the canonical
              description; pass `content` only when you need to override it.
        importance: 1-5.
        properties: predicate constraints / intent type rules_profile / arbitrary metadata.
        user_approved_star_scope: required only for "*" tool scopes.
        added_by: declared agent name (REQUIRED).
        wing/room/slug/hall/source_file/entity/predicate: kind='memory' only.

    Returns: status "created" | "exists" | "collision".
    """
    from .knowledge_graph import normalize_entity_name
    from .scoring import validate_context

    # ── Reject the legacy single-string description path ──
    if description is not None and context is None:
        return {
            "success": False,
            "error": (
                "P4.2: `description` is gone. Pass `context` instead — a dict "
                "with mandatory queries (list of 2-5 perspectives) and keywords "
                "(list of 2-5 caller-provided terms). Example:\n"
                '  context={"queries": ["DSpot platform server", "paperclip backend on :3100"], '
                '"keywords": ["dspot", "paperclip", "server", "port-3100"]}\n'
                "queries[0] becomes the canonical description for non-memory kinds."
            ),
        }

    # ── Validate Context (mandatory) ──
    clean_context, ctx_err = validate_context(context)
    if ctx_err:
        return ctx_err
    queries = clean_context["queries"]
    keywords = clean_context["keywords"]
    # clean_context["entities"] is reserved for graph-anchor wiring in P4.3+ (kg_add).

    # ── kind='memory' dispatch — memories are first-class entities ──
    if kind == "memory":
        if content is None or not str(content).strip():
            return {
                "success": False,
                "error": (
                    "kind='memory' requires `content` — the verbatim memory text. "
                    "(`context.queries` are search angles, not the body.) "
                    "Use kg_declare_entity(kind='memory', wing=..., room=..., slug=..., "
                    "content='<full text>', context={...}, added_by=..., ...)."
                ),
            }
        if not (wing and room and slug):
            return {
                "success": False,
                "error": (
                    "kind='memory' requires wing, room, and slug to construct the "
                    "memory id. Memory entities are scoped by wing/room (think project + "
                    "subtopic) and identified by slug (3-6 hyphenated words)."
                ),
            }
        return _add_drawer_internal(
            wing=wing,
            room=room,
            content=content,
            slug=slug,
            source_file=source_file,
            added_by=added_by,
            hall=hall,
            importance=importance,
            entity=entity,
            predicate=predicate,
            context=clean_context,
        )

    # Non-memory: queries[0] is the canonical description used for SQLite + first chroma vector.
    description = queries[0]

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
                "(For kind='memory', use wing/room/slug instead — the memory id is computed.)"
            ),
        }

    # Validate added_by: REQUIRED, must be a declared agent (is_a agent)
    if not added_by:
        return {
            "success": False,
            "error": "added_by is required. Pass your agent entity name (e.g., 'ga_agent', 'technical_lead_agent').",
        }
    agent_id_check = normalize_entity_name(added_by)
    if _kg:
        agent_edges = _kg.query_entity(agent_id_check, direction="outgoing")
        is_agent = any(
            e["predicate"] in ("is-a", "is_a") and e["object"] == "agent" and e.get("current", True)
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

    # Check for * scope in tool_permissions — requires user approval
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
        # ALL 5 constraint fields are REQUIRED — no optionals
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

                cls_entity = _kg.get_entity(_norm(cls_name))
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
    existing = _kg.get_entity(normalized)
    if existing:
        # Check for collisions with OTHER entities of SAME KIND (not self) — multi-view (P4.2)
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
        # No collisions — register in session
        _declared_entities.add(normalized)
        # Update description + importance + kind if provided and different
        if description and description != existing.get("description", ""):
            _kg.update_entity_description(normalized, description, importance)
            _sync_entity_views_to_chromadb(
                normalized, name, queries, kind, importance or 3, added_by=added_by
            )
        # Update properties if provided (merge with existing)
        if properties and isinstance(properties, dict):
            _kg.update_entity_properties(normalized, properties)
        # Refresh keywords (caller may have updated them)
        _kg.add_entity_keywords(normalized, keywords)
        return {
            "success": True,
            "status": "exists",
            "entity_id": normalized,
            "kind": existing.get("kind", "entity"),
            "description": existing.get("description") or description,
            "importance": existing.get("importance", 3),
            "edge_count": _kg.entity_edge_count(normalized),
        }

    # New entity — multi-view collision check (P4.2)
    similar = _check_entity_similarity_multiview(queries, kind_filter=kind)

    # Create the entity regardless — conflicts are resolved after creation
    props = properties if isinstance(properties, dict) else {}
    if added_by:
        props["added_by"] = added_by
    # SQLite row first (with queries[0] as the canonical description)
    _kg.add_entity(
        name, kind=kind, description=description, importance=importance or 3, properties=props
    )
    # Multi-vector embedding into the entity Chroma collection (one record per view)
    _sync_entity_views_to_chromadb(
        normalized, name, queries, kind, importance or 3, added_by=added_by
    )
    # Caller-provided keywords → entity_keywords table
    _kg.add_entity_keywords(normalized, keywords)
    # Persist the creation Context's view vectors and link the context_id to the entity
    cid = persist_context(clean_context, prefix=kind or "entity")
    if cid:
        _kg.set_entity_creation_context(normalized, cid)
    _declared_entities.add(normalized)

    # Auto-add is-a thing for new class entities (ensures class inheritance works)
    if kind == "class" and normalized != "thing":
        try:
            _kg.add_triple(normalized, "is-a", "thing")
        except Exception:
            pass  # Non-fatal if thing doesn't exist yet

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

    # ── Suggest related entities for linking (graph enrichment) ──
    suggested_links = []
    if kind in ("entity", "class") and description:
        try:
            ecol = _get_entity_collection(create=False)
            if ecol and ecol.count() > 1:
                n = min(ecol.count(), 15)
                results = ecol.query(
                    query_texts=[description],
                    n_results=n,
                    include=["documents", "metadatas", "distances"],
                )
                if results["ids"] and results["ids"][0]:
                    for i, eid in enumerate(results["ids"][0]):
                        if eid == normalized:
                            continue  # Skip self
                        meta_r = results["metadatas"][0][i] or {}
                        eid_kind = meta_r.get("kind", "entity")
                        if eid_kind not in ("entity", "class"):
                            continue
                        dist = results["distances"][0][i]
                        sim = round(max(0.0, 1.0 - dist), 3)
                        if sim < 0.15:
                            continue
                        desc = (results["documents"][0][i] or "")[:100]
                        suggested_links.append(
                            {
                                "entity_id": eid,
                                "similarity": sim,
                                "description": desc,
                            }
                        )
                        if len(suggested_links) >= 5:
                            break
        except Exception:
            pass

    result = {
        "success": True,
        "status": "created",
        "entity_id": normalized,
        "kind": kind,
        "description": description,
        "importance": importance or 3,
    }
    if suggested_links:
        result["suggested_links"] = suggested_links
        result["link_prompt"] = (
            "Related entities found. Create edges (kg_add) with precise predicates "
            "for each that should be connected. Skip those that shouldn't."
        )

    # ── Conflict detection: flag similar entities for resolution ──
    if similar:
        global _pending_conflicts
        conflicts = []
        for s in similar:
            conflict_id = f"conflict_entity_{normalized}_{s['entity_id']}"
            conflicts.append(
                {
                    "id": conflict_id,
                    "conflict_type": "entity_duplicate",
                    "reason": (
                        f"New entity '{normalized}' has similar description to "
                        f"existing '{s['entity_id']}' (similarity: {s.get('similarity', '?')})"
                    ),
                    "existing_id": s["entity_id"],
                    "existing_description": s.get("description", "")[:200],
                    "new_id": normalized,
                    "new_description": description[:200],
                }
            )
        _pending_conflicts = conflicts
        intent._persist_active_intent()
        result["conflicts"] = conflicts
        result["conflicts_prompt"] = (
            f"{len(conflicts)} similar entity/entities found. "
            f"Call mempalace_resolve_conflicts: merge (combine both), "
            f"keep (both are distinct), or skip (undo new entity)."
        )

    return result


def tool_kg_update_entity(  # noqa: C901
    entity: str,
    description: str = None,
    importance: int = None,
    properties: dict = None,
    # Drawer-specific (only meaningful when entity is a kind='memory' drawer)
    wing: str = None,
    room: str = None,
    hall: str = None,
):
    """Update any entity (drawer or KG node) in place. Pass only the fields you want to change.

    Unified replacement for the three legacy update tools (P3.4):
      - update_drawer_metadata → kg_update_entity(entity=drawer_id, wing=..., room=..., hall=..., importance=...)
      - update_entity_description → kg_update_entity(entity=..., description=...)
        (always checks distance against colliding entities; returns is_distinct flags)
      - update_predicate_constraints → kg_update_entity(entity=predicate, properties={"constraints": {...}})

    Args:
        entity: Entity ID or drawer ID to update.
        description: New description. For entities (kind=entity/class/predicate/literal):
            re-syncs to entity ChromaDB and runs collision distance check.
            For memory drawers: NOT supported here — use kg_delete_entity +
            kg_declare_entity to change drawer content.
        importance: New importance (1-5). Works for both entities and drawers.
        properties: Merged INTO existing properties dict (shallow merge at top
            level). For predicates use {"constraints": {...}} to replace
            constraints. For intent types {"rules_profile": {...}} to update slots
            or tool_permissions.
        wing, room, hall: Drawer-only metadata move (no re-embedding).
    """
    from .knowledge_graph import normalize_entity_name
    import json as _json

    if not entity or not isinstance(entity, str):
        return {"success": False, "error": "entity is required (string)."}

    is_drawer = entity.startswith("drawer_") or entity.startswith("diary_")

    # ── Validate inputs ──
    try:
        if description is not None:
            description = sanitize_content(description, max_length=5000)
        if importance is not None:
            importance = _validate_importance(importance)
        if hall is not None:
            hall = _validate_hall(hall)
        if wing is not None:
            wing = sanitize_name(wing, "wing")
        if room is not None:
            room = sanitize_name(room, "room")
    except ValueError as e:
        return {"success": False, "error": str(e)}

    # Reject contradictory inputs early
    if is_drawer and description is not None:
        return {
            "success": False,
            "error": (
                "Cannot update drawer description in place — embeddings would be "
                "stale. Use kg_delete_entity then kg_declare_entity(kind='memory', ...) "
                "to replace drawer content."
            ),
        }
    if not is_drawer and (wing is not None or room is not None or hall is not None):
        return {
            "success": False,
            "error": (
                "wing/room/hall are drawer-only fields. For non-drawer entities, "
                "use properties={...} to update metadata."
            ),
        }

    # ── Drawer path: in-place metadata update on the drawer collection ──
    if is_drawer:
        col = _get_collection()
        if not col:
            return _no_palace()
        existing = col.get(ids=[entity], include=["metadatas"])
        if not existing.get("ids"):
            return {"success": False, "error": f"Drawer not found: {entity}"}

        old_meta = dict(existing["metadatas"][0] or {})
        new_meta = dict(old_meta)
        updated_fields = []
        if wing is not None and old_meta.get("wing") != wing:
            new_meta["wing"] = wing
            updated_fields.append("wing")
        if room is not None and old_meta.get("room") != room:
            new_meta["room"] = room
            updated_fields.append("room")
        if hall is not None and old_meta.get("hall") != hall:
            new_meta["hall"] = hall
            updated_fields.append("hall")
        if importance is not None and old_meta.get("importance") != importance:
            new_meta["importance"] = importance
            updated_fields.append("importance")

        if not updated_fields:
            return {"success": True, "reason": "no_change", "entity_id": entity}

        _wal_log(
            "kg_update_entity",
            {
                "entity_id": entity,
                "source": "drawer",
                "old_meta": old_meta,
                "new_meta": new_meta,
                "updated_fields": updated_fields,
            },
        )
        try:
            col.update(ids=[entity], metadatas=[new_meta])
            logger.info(f"Updated drawer: {entity} fields={updated_fields}")
            return {
                "success": True,
                "entity_id": entity,
                "source": "drawer",
                "updated_fields": updated_fields,
                "new_metadata": new_meta,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Entity path: SQLite update + ChromaDB sync + collision check ──
    normalized = normalize_entity_name(entity)
    existing = _kg.get_entity(normalized)
    if not existing:
        return {"success": False, "error": f"Entity '{normalized}' not found."}

    updated_fields = []
    final_description = existing["description"]
    final_importance = existing.get("importance", 3)

    # Description update + ChromaDB resync
    if description is not None and description != existing["description"]:
        _kg.update_entity_description(normalized, description)
        final_description = description
        updated_fields.append("description")

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
                        cls_ent = _kg.get_entity(cls_eid)
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
        conn = _kg._conn()
        conn.execute(
            "UPDATE entities SET properties = ? WHERE id = ?",
            (_json.dumps(merged_props), normalized),
        )
        conn.commit()
        updated_fields.append("properties")

    # Importance update
    if importance is not None and importance != existing.get("importance"):
        conn = _kg._conn()
        conn.execute(
            "UPDATE entities SET importance = ? WHERE id = ?",
            (importance, normalized),
        )
        conn.commit()
        final_importance = importance
        updated_fields.append("importance")

    if not updated_fields:
        return {"success": True, "reason": "no_change", "entity_id": normalized}

    # Re-sync ChromaDB if description or importance changed (description embeds, importance is metadata)
    if "description" in updated_fields or "importance" in updated_fields:
        _sync_entity_to_chromadb(
            normalized,
            existing["name"],
            final_description,
            existing.get("kind") or existing.get("type", "entity"),
            final_importance,
        )

    _wal_log(
        "kg_update_entity",
        {"entity_id": normalized, "source": "entity", "updated_fields": updated_fields},
    )

    result = {
        "success": True,
        "entity_id": normalized,
        "source": "entity",
        "updated_fields": updated_fields,
    }

    # Collision distance check when description changed (was the point of the
    # legacy update_entity_description tool — keep that behaviour).
    if "description" in updated_fields:
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
            "All clear — re-declare this entity to register it."
            if all_distinct
            else "Still too similar to some entities. Make your description more specific."
        )

    return result


def tool_kg_merge_entities(source: str, target: str, update_description: str = None):
    """Merge source entity into target. All edges rewritten. Source becomes alias.

    Use when kg_declare_entity returns 'collision' and the entities are
    actually the same thing. All triples from source are moved to target.
    Source name becomes an alias that auto-resolves to target in future queries.

    Args:
        source: Entity to merge FROM (will be soft-deleted).
        target: Entity to merge INTO (will be kept, edges grow).
        update_description: Optional new description for the merged entity.
    """
    _wal_log(
        "kg_merge_entities",
        {
            "source": source,
            "target": target,
            "update_description": update_description[:200] if update_description else None,
        },
    )

    result = _kg.merge_entities(source, target, update_description)
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
        target_entity = _kg.get_entity(target_id)
        if target_entity:
            _sync_entity_to_chromadb(
                target_id,
                target_entity["name"],
                target_entity["description"],
                target_entity.get("type", "concept"),
                target_entity.get("importance", 3),
            )

    # Register target as declared (source is now alias for target)
    _declared_entities.discard(source_id)
    _declared_entities.add(target_id)

    return {
        "success": True,
        "source": result["source"],
        "target": result["target"],
        "edges_moved": result["edges_moved"],
        "aliases_created": result["aliases_created"],
    }


# tool_kg_update_predicate_constraints removed (P3.4): merged into tool_kg_update_entity.
# Call kg_update_entity(entity=predicate, properties={"constraints": {...}}).


def tool_kg_list_declared():
    """List all entities declared in this session."""
    results = []
    for eid in sorted(_declared_entities):
        entity = _kg.get_entity(eid)
        if entity:
            results.append(
                {
                    "entity_id": eid,
                    "name": entity["name"],
                    "description": entity["description"],
                    "importance": entity["importance"],
                    "last_touched": entity["last_touched"],
                    "edge_count": _kg.entity_edge_count(eid),
                }
            )
    return {
        "declared_count": len(results),
        "entities": results,
    }


# tool_kg_entity_info removed (P3.5): use kg_query(entity=..., direction="both").


# ==================== INTENT DECLARATION ====================

_active_intent = None  # Session-level: only one active intent at a time
_INTENT_STATE_DIR = Path(os.path.expanduser("~/.mempalace/hook_state"))


# Intent functions are in intent.py; init() is called after module globals are set.
# Aliases so TOOLS dispatch continues to work:
def tool_declare_intent(*args, **kwargs):
    return intent.tool_declare_intent(*args, **kwargs)


def tool_active_intent(*args, **kwargs):
    return intent.tool_active_intent(*args, **kwargs)


def tool_extend_intent(*args, **kwargs):
    return intent.tool_extend_intent(*args, **kwargs)


def tool_resolve_conflicts(actions: list = None):  # noqa: C901
    """Resolve pending conflicts — contradictions, duplicates, or suggestions.

    Unified conflict resolution for ALL data types: edges, entities, drawers.
    Each action specifies what to do with a conflict.

    Args:
        actions: List of {id, action, into?, merged_content?} dicts.
            id: The conflict ID (from the pending conflicts list).
            action: One of:
                "invalidate" — mark existing item as no longer current (sets valid_to)
                "merge" — combine items (must provide into + merged_content)
                "keep" — both items are valid, no conflict
                "skip" — don't add the new item (remove it)
            into: Target entity/drawer ID to merge into (required for "merge")
            merged_content: Merged description/content (required for "merge")
    """
    global _pending_conflicts

    # Disk is source of truth — reload _pending_conflicts from the active
    # intent state file if memory is empty (MCP restart scenario).
    if not _pending_conflicts:
        _pending_conflicts = _load_pending_conflicts_from_disk()

    if not _pending_conflicts:
        try:
            intent._persist_active_intent()
        except Exception:
            pass
        return {"success": True, "message": "No pending conflicts."}

    if not actions:
        return {
            "success": False,
            "error": "Must provide actions list. Each conflict needs: {id, action}.",
            "pending": _pending_conflicts,
        }

    # Index pending conflicts by ID — defensively coerce if any entries are
    # JSON strings (some MCP transports serialize nested objects)
    _normalized_conflicts = []
    for c in _pending_conflicts:
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

    # Normalize actions too — tolerate string-encoded dicts from some transports
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
                    _kg.invalidate(
                        conflict["existing_subject"],
                        conflict["existing_predicate"],
                        conflict["existing_object"],
                    )
                elif conflict_type in ("entity_duplicate", "drawer_duplicate"):
                    # Mark entity/drawer as merged-out
                    try:
                        conn = _kg._conn()
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
                            "reason": "merge requires 'merged_content' — read BOTH items in full, then provide combined content",
                        }
                    )
                    continue

                # Determine source (the one NOT being merged into)
                source = new_id if into == existing_id else existing_id

                if conflict_type in ("entity_duplicate", "drawer_duplicate"):
                    # Use existing kg_merge_entities for the plumbing
                    merge_result = tool_kg_merge_entities(
                        source=source, target=into, update_description=merged_content
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
                # Both items are valid — no action needed
                results.append({"id": cid, "status": "kept"})

            elif action == "skip":
                # Don't add the new item — remove it if already added
                if conflict_type == "edge_contradiction":
                    try:
                        _kg.invalidate(
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

            resolved_ids.add(cid)
        except Exception as e:
            results.append({"id": cid, "status": "error", "reason": str(e)})

    # Check all conflicts are resolved
    unresolved = set(conflict_map.keys()) - resolved_ids
    if unresolved:
        return {
            "success": False,
            "error": f"{len(unresolved)} conflicts not addressed. Provide action for each.",
            "unresolved": [conflict_map[cid] for cid in unresolved],
            "resolved": results,
        }

    # Clear pending conflicts and persist state
    _pending_conflicts = None
    try:
        intent._persist_active_intent()
    except Exception:
        pass
    return {"success": True, "resolved": results}


def tool_finalize_intent(*args, **kwargs):
    return intent.tool_finalize_intent(*args, **kwargs)


# ==================== AGENT DIARY ====================


def tool_diary_write(
    agent_name: str,
    entry: str,
    slug: str = "",
    topic: str = "general",
    hall: str = "hall_diary",
    importance: int = None,
):
    """
    Write a diary entry for this agent. Each agent gets its own wing
    with a diary room. Entries are timestamped and accumulate over time.

    The diary is a HIGH-LEVEL SESSION NARRATIVE — not a detailed log.
    Write in readable prose (NOT AAAK compression).

    WHAT TO INCLUDE:
    - Decisions made with the user (approved designs, rejected ideas)
    - Big-picture status and direction
    - Pending items and backlog
    - Cross-intent narrative (how multiple actions connected)

    WHAT NOT TO INCLUDE (already captured by intent results):
    - Individual commits or features shipped
    - Gotchas and learnings (already KG entities via finalize_intent)
    - Tool traces or detailed action logs

    Each entry should be a DELTA from the previous — what changed,
    not a full restatement of everything.

    Args:
        slug: Descriptive identifier for this entry (e.g. 'session12-scoring-design').
              If not provided, falls back to date-topic format.
        topic: Topic tag (optional, default: general)
        hall: default 'hall_diary'. Override with 'hall_discoveries' for
              "today I learned" entries that deserve higher retrieval priority,
              or 'hall_events' for plain activity logs.
        importance: 1-5. Defaults to unset (treated as 3 by L1). Use 4 for
                    entries with learned lessons, 5 only for agent-wide
                    critical notes.
    """
    try:
        agent_name = sanitize_name(agent_name, "agent_name")
        entry = sanitize_content(entry)
        hall = _validate_hall(hall)
        importance = _validate_importance(importance)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    wing = f"wing_{agent_name.lower().replace(' ', '_')}"
    room = "diary"
    col = _get_collection(create=True)
    if not col:
        return _no_palace()

    now = datetime.now()
    if slug and slug.strip():
        diary_slug = _normalize_drawer_slug(slug)
    else:
        diary_slug = _normalize_drawer_slug(f"{now.strftime('%Y%m%d-%H%M%S')}-{topic}")
    entry_id = f"diary_{wing}_{diary_slug}"

    _wal_log(
        "diary_write",
        {
            "agent_name": agent_name,
            "topic": topic,
            "entry_id": entry_id,
            "entry_preview": entry[:200],
            "hall": hall,
            "importance": importance,
        },
    )

    try:
        # TODO: Future versions should expand AAAK before embedding to improve
        # semantic search quality. For now, store raw AAAK in metadata so it's
        # preserved, and keep the document as-is for embedding (even though
        # compressed AAAK degrades embedding quality).
        meta = {
            "wing": wing,
            "room": room,
            "hall": hall or "hall_diary",
            "topic": topic,
            "type": "diary_entry",
            "agent": agent_name,
            "added_by": agent_name,  # Same field as drawers/entities for agent affinity scoring
            "filed_at": now.isoformat(),
            "date_added": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
        }
        if importance is not None:
            meta["importance"] = importance
        col.add(
            ids=[entry_id],
            documents=[entry],
            metadatas=[meta],
        )
        logger.info(f"Diary entry: {entry_id} -> {wing}/diary/{topic} hall={hall} imp={importance}")
        return {
            "success": True,
            "entry_id": entry_id,
            "agent": agent_name,
            "topic": topic,
            "hall": hall,
            "importance": importance,
            "timestamp": now.isoformat(),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_diary_read(agent_name: str, last_n: int = 10):
    """
    Read an agent's recent diary entries. Returns the last N entries
    in chronological order — the agent's personal journal.
    """
    wing = f"wing_{agent_name.lower().replace(' ', '_')}"
    col = _get_collection()
    if not col:
        return _no_palace()

    try:
        results = col.get(
            where={"$and": [{"wing": wing}, {"room": "diary"}]},
            include=["documents", "metadatas"],
            limit=10000,
        )

        if not results["ids"]:
            return {"agent": agent_name, "entries": [], "message": "No diary entries yet."}

        # Combine and sort by timestamp
        entries = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            entries.append(
                {
                    "date": meta.get("date", ""),
                    "timestamp": meta.get("filed_at", ""),
                    "topic": meta.get("topic", ""),
                    "content": doc,
                }
            )

        entries.sort(key=lambda x: x["timestamp"], reverse=True)
        entries = entries[:last_n]

        return {
            "agent": agent_name,
            "entries": entries,
            "total": len(results["ids"]),
            "showing": len(entries),
        }
    except Exception as e:
        return {"error": str(e)}


# ==================== MCP PROTOCOL ====================

TOOLS = {
    "mempalace_get_aaak_spec": {
        "description": "Get the AAAK dialect specification — the compressed memory format MemPalace uses. Call this if you need to read or write AAAK-compressed memories.",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_get_aaak_spec,
    },
    "mempalace_kg_query": {
        "description": "Query the knowledge graph for an entity's relationships by EXACT entity name. Returns typed facts with temporal validity. Supports batch queries: pass comma-separated names to query multiple entities in one call. Use kg_search instead if you don't know the exact entity name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "Entity to query (e.g. 'Max', 'MyProject'). Supports comma-separated batch: 'Max, Alice, MyProject' returns results keyed by entity.",
                },
                "as_of": {
                    "type": "string",
                    "description": "Date filter — only facts valid at this date (YYYY-MM-DD, optional)",
                },
                "direction": {
                    "type": "string",
                    "description": "outgoing (entity→?), incoming (?→entity), or both (default: both)",
                },
            },
            "required": ["entity"],
        },
        "handler": tool_kg_query,
    },
    "mempalace_kg_search": {
        "description": "Unified palace search — drawers (prose) + entities (KG nodes) in one call. Multi-view cosine + keyword for drawers; cosine + keyword + 1-hop graph for entities. Cross-collection Reciprocal Rank Fusion across all channels. Each result carries source='drawer'|'entity' with type-specific fields (drawers: text/wing/room; entities: name/kind/description/edges). Unlike kg_query (exact entity ID), this fuzzy-matches across your whole memory palace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 5,
                    "description": "MANDATORY list of 2-5 perspective strings. Each becomes a separate cosine view for multi-view retrieval (strongly outperforms a single phrasing). Example: ['deployment process', 'release pipeline', 'production rollout']. A single string is REJECTED — the whole point is multi-view.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results across drawers+entities (default 10; adaptive-K may trim if scores drop off).",
                },
                "wing": {
                    "type": "string",
                    "description": "Optional drawer wing filter (e.g. 'ga'). Ignored for entity results.",
                },
                "room": {
                    "type": "string",
                    "description": "Optional drawer room filter. Ignored for entity results.",
                },
                "kind": {
                    "type": "string",
                    "description": "Optional entity kind filter: 'entity', 'predicate', 'class', 'literal'. Ignored for drawer results.",
                    "enum": ["entity", "predicate", "class", "literal"],
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["hybrid", "similarity"],
                    "description": "'hybrid' (default) = RRF + hybrid_score (importance + decay + agent + feedback). 'similarity' = pure cosine.",
                },
                "agent": {
                    "type": "string",
                    "description": "Your agent entity name for affinity scoring. Items created by you get a ranking boost in hybrid mode.",
                },
            },
            "required": ["queries", "agent"],
        },
        "handler": tool_kg_search,
    },
    "mempalace_kg_add": {
        "description": (
            "Add a fact to the knowledge graph (P4.3 — Context mandatory). "
            "Subject → predicate → object plus a Context fingerprint that captures "
            "WHY the edge is being added. The Context's view vectors are persisted; "
            "future feedback (found_useful etc.) applies by MaxSim against this "
            "fingerprint. E.g. ('Max', 'started_school', 'Year 7', valid_from='2026-09-01', "
            "context={'queries': ['Max enrolled in Year 7', 'school start 2026'], "
            "'keywords': ['max', 'school', 'year-7']})."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "The entity doing/being something"},
                "predicate": {
                    "type": "string",
                    "description": "The relationship type (e.g. 'loves', 'works_on', 'daughter_of')",
                },
                "object": {"type": "string", "description": "The entity being connected to"},
                "context": {
                    "type": "object",
                    "description": (
                        "MANDATORY Context fingerprint for the edge. "
                        '{"queries": list[str] (2-5 perspectives on why this edge), '
                        '"keywords": list[str] (2-5 caller-provided terms), '
                        '"entities": list[str] (0+ related entity ids)}'
                    ),
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 5,
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 5,
                        },
                        "entities": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["queries", "keywords"],
                },
                "valid_from": {
                    "type": "string",
                    "description": "When this became true (YYYY-MM-DD, optional)",
                },
                "source_closet": {
                    "type": "string",
                    "description": "Closet ID where this fact appears (optional)",
                },
            },
            "required": ["subject", "predicate", "object", "context"],
        },
        "handler": tool_kg_add,
    },
    "mempalace_kg_add_batch": {
        "description": (
            "Add multiple KG edges in one call (P4.3 — Context mandatory). Pass a "
            "single top-level `context` as the shared default for every edge in the "
            "batch — most batches add edges that all reflect the same agent decision. "
            "An edge can override with its own `context` if needed. Validates "
            "independently — partial success OK."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "predicate": {"type": "string"},
                            "object": {"type": "string"},
                            "context": {
                                "type": "object",
                                "description": "Per-edge Context override (optional if top-level context provided).",
                            },
                        },
                        "required": ["subject", "predicate", "object"],
                    },
                    "description": "List of edges to add.",
                },
                "context": {
                    "type": "object",
                    "description": (
                        "Shared Context for every edge in the batch. Required "
                        "unless every edge carries its own `context`."
                    ),
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 5,
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 5,
                        },
                        "entities": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["queries", "keywords"],
                },
            },
            "required": ["edges"],
        },
        "handler": tool_kg_add_batch,
    },
    "mempalace_kg_invalidate": {
        "description": "Mark a fact as no longer true. E.g. ankle injury resolved, job ended, moved house.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Entity"},
                "predicate": {"type": "string", "description": "Relationship"},
                "object": {"type": "string", "description": "Connected entity"},
                "ended": {
                    "type": "string",
                    "description": "When it stopped being true (YYYY-MM-DD, default: today)",
                },
            },
            "required": ["subject", "predicate", "object"],
        },
        "handler": tool_kg_invalidate,
    },
    "mempalace_kg_timeline": {
        "description": "Chronological timeline of facts. Shows the story of an entity (or everything) in order.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "Entity to get timeline for (optional — omit for full timeline)",
                },
            },
        },
        "handler": tool_kg_timeline,
    },
    "mempalace_kg_stats": {
        "description": "Knowledge graph overview: entities, triples, current vs expired facts, relationship types.",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_kg_stats,
    },
    "mempalace_kg_declare_entity": {
        "description": (
            "REQUIRED before using any entity in kg_add. Declares an entity using "
            "the unified Context object (P4.2). Each query is embedded as a "
            "separate Chroma vector; collision detection runs multi-view RRF; "
            "caller-provided keywords go into the keyword index; the Context's "
            "view vectors are persisted so future feedback applies by similarity.\n\n"
            "Kinds:\n"
            "  'entity'    — concrete thing.\n"
            "  'class'     — category definition (other entities is_a this).\n"
            "  'predicate' — relationship type for kg_add edges.\n"
            "  'literal'   — raw value.\n"
            "  'memory'    — prose memory. Requires wing/room/slug + `content` "
            "(verbatim text). `name` is auto-computed from wing/room/slug. "
            "Use `entity`+`predicate` to link the memory to another entity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Entity name (will be normalized). REQUIRED for kind=entity/class/predicate/literal. OMIT for kind='memory' — the memory id is computed from wing/room/slug.",
                },
                "context": {
                    "type": "object",
                    "description": (
                        "MANDATORY Context fingerprint. Replaces the legacy single-string `description`.\n"
                        "  queries:  list[str] (2-5)  perspectives on what this entity is.\n"
                        "             For non-memory kinds, queries[0] becomes the canonical description.\n"
                        "  keywords: list[str] (2-5)  exact terms — caller-provided, NEVER auto-extracted.\n"
                        "             Stored in the keyword index for fast exact-match retrieval.\n"
                        "  entities: list[str] (0+)   related/seed entity ids (optional graph anchors).\n"
                        'Example: context={"queries": ["DSpot platform server", "paperclip backend on :3100"], '
                        '"keywords": ["dspot", "paperclip", "server", "port-3100"], '
                        '"entities": ["DSpotInfra"]}'
                    ),
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 5,
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 5,
                        },
                        "entities": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["queries", "keywords"],
                },
                "kind": {
                    "type": "string",
                    "description": "Ontological role: 'entity' (concrete thing), 'class' (category), 'predicate' (relationship type), 'literal' (raw value), 'memory' (requires wing/room/slug + content).",
                    "enum": ["entity", "predicate", "class", "literal", "memory"],
                },
                "content": {
                    "type": "string",
                    "description": "Verbatim text — REQUIRED for kind='memory' (the actual memory body). Ignored for other kinds (queries[0] is the canonical description).",
                },
                "importance": {
                    "type": "integer",
                    "description": "1-5. 5=critical, 4=canonical, 3=default, 2=low, 1=junk.",
                    "minimum": 1,
                    "maximum": 5,
                },
                "properties": {
                    "type": "object",
                    "description": 'General-purpose metadata stored with the entity. For predicates: {"constraints": {"subject_kinds": [...], "object_kinds": [...], "subject_classes": [...], "object_classes": [...], "cardinality": "..."}} (ALL 5 fields REQUIRED). For intent types: {"rules_profile": {"slots": {...}, "tool_permissions": [...]}}.',
                },
                "added_by": {
                    "type": "string",
                    "description": "Agent who is declaring this entity. Must be a declared agent (is_a agent).",
                },
                "user_approved_star_scope": {
                    "type": "boolean",
                    "description": "NEVER set this to true unless the user JUST said YES in this conversation turn.",
                },
                # ── kind='memory' specific ──
                "wing": {"type": "string", "description": "REQUIRED when kind='memory'."},
                "room": {"type": "string", "description": "REQUIRED when kind='memory'."},
                "slug": {
                    "type": "string",
                    "description": "REQUIRED when kind='memory'. 3-6 hyphenated words, unique within wing/room.",
                },
                "hall": {
                    "type": "string",
                    "description": "Optional content-type tag for kind='memory'.",
                    "enum": [
                        "hall_facts",
                        "hall_events",
                        "hall_discoveries",
                        "hall_preferences",
                        "hall_advice",
                        "hall_diary",
                    ],
                },
                "source_file": {
                    "type": "string",
                    "description": "Optional source attribution for kind='memory'.",
                },
                "entity": {
                    "type": "string",
                    "description": "Entity name(s) to link this memory to. Defaults to the wing name.",
                },
                "predicate": {
                    "type": "string",
                    "description": "Predicate for the entity→memory link (default: described_by).",
                    "enum": [
                        "described_by",
                        "evidenced_by",
                        "derived_from",
                        "mentioned_in",
                        "session_note_for",
                    ],
                },
            },
            "required": ["context", "kind", "importance", "added_by"],
        },
        "handler": tool_kg_declare_entity,
    },
    "mempalace_kg_update_entity": {
        "description": (
            "Unified update for any entity (drawer or KG node). Pass only the fields "
            "you want to change (P3.4 — replaces the three legacy update tools).\n\n"
            "FOR ENTITIES (kind=entity/class/predicate/literal):\n"
            "  - description: re-syncs to entity ChromaDB and runs collision distance check.\n"
            "  - properties: shallow-merged into existing properties. For predicates, "
            'use {"constraints": {...}} to update predicate constraints (validated).\n'
            "  - importance: 1-5.\n\n"
            "FOR DRAWERS (kind='memory', ids starting with drawer_/diary_):\n"
            "  - wing/room/hall: in-place metadata move (no re-embedding).\n"
            "  - importance: in-place importance change.\n"
            "  - description: NOT supported — use kg_delete_entity + kg_declare_entity "
            "to replace drawer content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "Entity ID or drawer ID (drawer_/diary_ prefix routes to drawer collection).",
                },
                "description": {
                    "type": "string",
                    "description": "New description (entities only). Triggers collision distance check.",
                },
                "importance": {
                    "type": "integer",
                    "description": "New importance 1-5 (works for both entities and drawers).",
                    "minimum": 1,
                    "maximum": 5,
                },
                "properties": {
                    "type": "object",
                    "description": 'Properties to merge into the entity. For predicates, use {"constraints": {"subject_kinds": [...], "object_kinds": [...], "subject_classes": [...], "object_classes": [...], "cardinality": "..."}}.',
                },
                "wing": {
                    "type": "string",
                    "description": "(Drawers only) New wing — in-place move, preserves embedding.",
                },
                "room": {
                    "type": "string",
                    "description": "(Drawers only) New room — in-place move, preserves embedding.",
                },
                "hall": {
                    "type": "string",
                    "description": "(Drawers only) Hall classification.",
                    "enum": [
                        "hall_facts",
                        "hall_events",
                        "hall_discoveries",
                        "hall_preferences",
                        "hall_advice",
                        "hall_diary",
                    ],
                },
            },
            "required": ["entity"],
        },
        "handler": tool_kg_update_entity,
    },
    "mempalace_kg_merge_entities": {
        "description": "Merge source entity into target. All edges rewritten, source becomes alias. Use when kg_declare_entity returns 'collision' and the entities are actually the same thing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Entity to merge FROM (will be soft-deleted)",
                },
                "target": {"type": "string", "description": "Entity to merge INTO (will be kept)"},
                "update_description": {
                    "type": "string",
                    "description": "Optional new description for the merged entity",
                },
            },
            "required": ["source", "target"],
        },
        "handler": tool_kg_merge_entities,
    },
    # mempalace_kg_update_predicate_constraints removed (P3.4): merged into
    # mempalace_kg_update_entity. Call with properties={"constraints": {...}}.
    "mempalace_kg_list_declared": {
        "description": "List all entities declared in this session with their details (description, importance, edge count).",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_kg_list_declared,
    },
    "mempalace_declare_intent": {
        "description": (
            "Declare what you intend to do BEFORE doing it. Returns permissions + context. "
            "One active intent at a time — new intent expires the previous. "
            "mempalace_* tools are always allowed regardless of intent.\n\n"
            "SLOT RULES — most intent types require these slots:\n"
            '  paths:    (raw) directory patterns for Read/Grep/Glob scoping. E.g. ["D:/Flowsev/repo/**"]\n'
            '  commands: (raw) command patterns for Bash scoping. E.g. ["pytest", "git add"]\n'
            "  files:    file paths for Edit/Write scoping. Auto-declares existing files.\n"
            "  target:   entity names for context injection. Requires pre-declared entities.\n\n"
            "EXCEPTION: 'research' type needs NO paths — it has unrestricted Read/Grep/Glob/WebFetch/WebSearch.\n"
            "Use 'research' when you genuinely don't know what you'll read. Other types require declaring paths.\n\n"
            "Check declared.intent_types from wake_up for available types + their tools.\n"
            "If a tool is blocked, the error teaches how to create or switch types."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "intent_type": {
                    "type": "string",
                    "description": (
                        "The intent type to declare (must be is-a intent_type). "
                        "Use the MOST SPECIFIC type available — specific types carry domain rules. "
                        "Examples: 'edit_file', 'write_tests', 'deploy', 'run_tests', 'diagnose_failure'. "
                        "Broad types: 'inspect', 'modify', 'execute', 'communicate'."
                    ),
                },
                "slots": {
                    "type": "object",
                    "description": (
                        "Named slots filled with entity names, file paths, or command patterns. "
                        'Example for edit_file: {"files": ["src/auth.test.ts"], "paths": ["src/**"]}. '
                        'Example for execute: {"target": ["my_project"], "commands": ["pytest", "git add"], "paths": ["D:/Flowsev/mempalace/**"]}. '
                        'Example for inspect: {"subject": ["my_system"], "paths": ["D:/Flowsev/repo/**"]}. '
                        'Example for research: {"subject": ["some_topic"]} — NO paths needed, broad reads allowed. '
                        "File slots auto-declare existing files. Command slots (raw) accept strings directly. "
                        "Other slots require pre-declared entities."
                    ),
                },
                "descriptions": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 2,
                    "maxItems": 8,
                    "description": (
                        "MANDATORY list of 2-8 distinct perspective strings describing what "
                        "you plan to do. Each string becomes a separate view in multi-view "
                        "retrieval. DO NOT pass a single string — multi-view is the whole "
                        "point. One angle catches a gotcha, another finds a past execution, "
                        "a third surfaces a rule. Also used for auto-narrowing: if a more "
                        "specific child intent type matches your descriptions, the system "
                        "will auto-select it.\n\n"
                        "REQUIRED format (list of 2+ strings):\n"
                        "  ['Editing auth rate limiter',\n"
                        "   'Security hardening against brute force',\n"
                        "   'Adding tests for login endpoint']\n\n"
                        "A single-string passed here will be REJECTED with an error."
                    ),
                },
                "auto_declare_files": {
                    "type": "boolean",
                    "description": (
                        "Set to true when creating NEW files that don't exist on disk yet. "
                        "Existing files are auto-declared without this flag. Default: false."
                    ),
                },
                "agent": {
                    "type": "string",
                    "description": "Your agent entity name for affinity scoring in context injection. Examples: 'ga_agent', 'technical_lead_agent'.",
                },
                "budget": {
                    "type": "object",
                    "description": (
                        "MANDATORY tool call budget. Dict of tool_name -> max_calls. "
                        'E.g. {"Read": 5, "Edit": 3, "Bash": 2}. Must cover all tools you plan to use. '
                        "Keep budgets tight — estimate minimum needed. "
                        "When exhausted, use mempalace_extend_intent to add more."
                    ),
                },
            },
            "required": ["intent_type", "slots", "agent", "budget"],
        },
        "handler": tool_declare_intent,
    },
    "mempalace_active_intent": {
        "description": "Return the current active intent — type, slots, permissions, budget remaining.",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_active_intent,
    },
    "mempalace_resolve_conflicts": {
        "description": (
            "Resolve pending conflicts — contradictions, duplicates, or merge candidates. "
            "MANDATORY when conflicts are returned by kg_add or kg_declare_entity (including kind='memory'). "
            "Tools are BLOCKED until ALL conflicts are resolved. Batch-process in one call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Conflict ID from the pending conflicts list.",
                            },
                            "action": {
                                "type": "string",
                                "enum": ["invalidate", "merge", "keep", "skip"],
                                "description": (
                                    "invalidate: mark existing as no longer current. "
                                    "merge: combine both (requires into + merged_content). "
                                    "keep: both are valid, no conflict. "
                                    "skip: don't add the new item."
                                ),
                            },
                            "into": {
                                "type": "string",
                                "description": "Target ID to merge into (required for 'merge').",
                            },
                            "merged_content": {
                                "type": "string",
                                "description": (
                                    "Merged description/content preserving ALL unique info from both sides. "
                                    "Required for 'merge'. Read BOTH items in full before merging."
                                ),
                            },
                        },
                        "required": ["id", "action"],
                    },
                    "description": "List of conflict resolution actions.",
                },
            },
            "required": ["actions"],
        },
        "handler": tool_resolve_conflicts,
    },
    "mempalace_extend_intent": {
        "description": (
            "Extend the active intent's tool budget without redeclaring. "
            "Use when budget is exhausted but you're still on the same task. "
            "Adds counts to existing budget. Syncs with hook via disk state."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "budget": {
                    "type": "object",
                    "description": (
                        'Additional tool calls to add. E.g. {"Read": 3, "Edit": 2}. '
                        "These ADD to existing budget, not replace."
                    ),
                },
                "agent": {"type": "string", "description": "Your agent name."},
            },
            "required": ["budget"],
        },
        "handler": tool_extend_intent,
    },
    "mempalace_finalize_intent": {
        "description": (
            "Finalize the active intent — capture what happened as structured memory. "
            "MUST be called before declaring a new intent or exiting the session. "
            "Creates an execution entity (is_a intent_type) with relationships to agent, "
            "targets, result drawer, gotchas, execution trace, and memory relevance feedback. "
            "If not called explicitly, declare_intent will BLOCK until you finalize."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "slug": {
                    "type": "string",
                    "description": "Human-readable ID for this execution (e.g. 'edit-auth-rate-limiter-2026-04-14'). Must be unique.",
                },
                "outcome": {
                    "type": "string",
                    "description": "How the intent concluded.",
                    "enum": ["success", "partial", "failed", "abandoned"],
                },
                "summary": {
                    "type": "string",
                    "description": "What happened — the result narrative. Becomes a drawer.",
                },
                "agent": {
                    "type": "string",
                    "description": "Your agent entity name (e.g. 'ga_agent', 'technical_lead_agent').",
                },
                "memory_feedback": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Drawer ID or entity ID"},
                            "relevant": {
                                "type": "boolean",
                                "description": "Was this memory relevant to the intent?",
                            },
                            "relevance": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5,
                                "description": "Contextual relevance 1-5 (not global importance — how useful was this FOR THIS ACTION)",
                            },
                            "promote_to_type": {
                                "type": "boolean",
                                "description": "true = generalizable pattern (always relevant for this intent type), false = instance-specific",
                            },
                            "reason": {
                                "type": "string",
                                "description": "Brief reason for the rating",
                            },
                        },
                        "required": ["id", "relevant"],
                    },
                    "description": "MANDATORY — contextual relevance feedback for ALL memories accessed during this intent: memories injected by declare_intent, memories found via search, AND new memories created. Rate each for relevance to THIS action.",
                },
                "key_actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Abbreviated tool+params list. Auto-filled from execution trace if omitted.",
                },
                "gotchas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Gotcha descriptions discovered during execution. Each becomes a KG entity.",
                },
                "learnings": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Lesson descriptions worth remembering. Each becomes a drawer.",
                },
                "promote_gotchas_to_type": {
                    "type": "boolean",
                    "description": "Also link gotchas to the intent TYPE (not just this execution). Use for general gotchas.",
                },
            },
            "required": ["slug", "outcome", "summary", "agent", "memory_feedback"],
        },
        "handler": tool_finalize_intent,
    },
    "mempalace_traverse": {
        "description": "Walk the palace graph from a room. Shows connected ideas across wings — the tunnels. Like following a thread through the palace: start at 'chromadb-setup' in wing_code, discover it connects to wing_myproject (planning) and wing_user (feelings about it).",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_room": {
                    "type": "string",
                    "description": "Room to start from (e.g. 'chromadb-setup', 'riley-school')",
                },
                "max_hops": {
                    "type": "integer",
                    "description": "How many connections to follow (default: 2)",
                },
            },
            "required": ["start_room"],
        },
        "handler": tool_traverse_graph,
    },
    # mempalace_search removed (P3.2): merged into mempalace_kg_search.
    # mempalace_add_drawer removed (P3.3): merged into mempalace_kg_declare_entity
    # with kind='memory'. Drawers are first-class graph entities — there's no
    # reason to have a separate write tool for them.
    "mempalace_kg_delete_entity": {
        "description": "Delete an entity (drawer or KG node) and invalidate every current edge touching it. Works for both drawer memories (ids starting with 'drawer_' / 'diary_') and KG entities. Use this when an entity is TRULY obsolete. For stale single facts (one relationship untrue while entity stays valid), use kg_invalidate on that specific (subject, predicate, object) triple instead.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "ID of the entity or drawer to delete.",
                },
            },
            "required": ["entity_id"],
        },
        "handler": tool_kg_delete_entity,
    },
    "mempalace_wake_up": {
        "description": "Return L0 (identity) + L1 (importance-ranked essential story) wake-up text (~600-900 tokens total). Call this ONCE at session start to load project/agent boot context. Also returns the protocol, declared entities/predicates/intent types — everything you need to start. L1 is ranked with importance-weighted time decay — critical facts always surface first, within-tier newer wins. Pass wing='ga' for GA sessions, wing='wing_<agent>' for paperclip agents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "wing": {
                    "type": "string",
                    "description": "Optional wing filter. If unset, L1 loads globally. If set, L1 loads only drawers in that wing (project/agent-scoped boot).",
                },
                "agent": {
                    "type": "string",
                    "description": "Agent identity for affinity scoring. Drawers filed by this agent get a ranking boost in L1. Use your agent entity name (e.g., 'ga_agent', 'technical_lead_agent').",
                },
            },
            "required": ["agent"],
        },
        "handler": tool_wake_up,
    },
    # mempalace_update_drawer_metadata removed (P3.4): merged into mempalace_kg_update_entity.
    # Call kg_update_entity(entity=drawer_id, wing=..., room=..., hall=..., importance=...).
    "mempalace_diary_write": {
        "description": "Write to your personal agent diary in AAAK format. Your observations, thoughts, what you worked on, what matters. Each agent has their own diary with full history. Write in AAAK for compression — e.g. 'SESSION:2026-04-04|built.palace.graph+diary.tools|ALC.req:agent.diaries.in.aaak|★★★'. Use entity codes from the AAAK spec. Optional hall/importance for special entries.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Your name — each agent gets their own diary wing",
                },
                "entry": {
                    "type": "string",
                    "description": "Your diary entry in AAAK format — compressed, entity-coded, emotion-marked",
                },
                "slug": {
                    "type": "string",
                    "description": "Descriptive identifier for this diary entry (e.g. 'session12-intent-narrowing-shipped', 'migration-lesson-learned'). Used as part of the entry ID.",
                },
                "topic": {
                    "type": "string",
                    "description": "Topic tag (optional, default: general)",
                },
                "hall": {
                    "type": "string",
                    "description": "Override the default hall_diary classification. Use hall_discoveries for 'today I learned' entries worth higher retrieval priority, hall_events for plain activity logs.",
                    "enum": [
                        "hall_facts",
                        "hall_events",
                        "hall_discoveries",
                        "hall_preferences",
                        "hall_advice",
                        "hall_diary",
                    ],
                },
                "importance": {
                    "type": "integer",
                    "description": "Importance 1-5. Leave unset for default diary entries (treated as 3 by Layer1). Use 4 for entries with learned lessons, 5 only for agent-wide critical notes.",
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": ["agent_name", "entry"],
        },
        "handler": tool_diary_write,
    },
    "mempalace_diary_read": {
        "description": "Read your recent diary entries (in AAAK). See what past versions of yourself recorded — your journal across sessions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Your name — each agent gets their own diary wing",
                },
                "last_n": {
                    "type": "integer",
                    "description": "Number of recent entries to read (default: 10)",
                },
            },
            "required": ["agent_name"],
        },
        "handler": tool_diary_read,
    },
}


SUPPORTED_PROTOCOL_VERSIONS = [
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
]


def handle_request(request):
    method = request.get("method", "")
    params = request.get("params", {})
    req_id = request.get("id")

    if method == "initialize":
        client_version = params.get("protocolVersion", SUPPORTED_PROTOCOL_VERSIONS[-1])
        negotiated = (
            client_version
            if client_version in SUPPORTED_PROTOCOL_VERSIONS
            else SUPPORTED_PROTOCOL_VERSIONS[0]
        )
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": negotiated,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mempalace", "version": __version__},
            },
        }
    elif method == "notifications/initialized":
        return None
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {"name": n, "description": t["description"], "inputSchema": t["input_schema"]}
                    for n, t in TOOLS.items()
                ]
            },
        }
    elif method == "tools/call":
        tool_name = params.get("name")
        tool_args = params.get("arguments") or {}
        if tool_name not in TOOLS:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }
        # Extract sessionId injected by PreToolUse hook (not a tool parameter)
        injected_session_id = tool_args.pop("sessionId", None)
        if injected_session_id:
            global _session_id
            new_sid = str(injected_session_id)
            if new_sid != _session_id:
                _save_session_state()
                _session_id = new_sid
                _restore_session_state(new_sid)

        # Coerce argument types based on input_schema.
        # MCP JSON transport may deliver integers as floats or strings;
        # ChromaDB and Python slicing require native int.
        schema_props = TOOLS[tool_name]["input_schema"].get("properties", {})
        for key, value in list(tool_args.items()):
            prop_schema = schema_props.get(key, {})
            declared_type = prop_schema.get("type")
            if declared_type == "integer" and not isinstance(value, int):
                tool_args[key] = int(value)
            elif declared_type == "number" and not isinstance(value, (int, float)):
                tool_args[key] = float(value)
        try:
            result = TOOLS[tool_name]["handler"](**tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]},
            }
        except Exception as e:
            logger.exception(f"Tool error in {tool_name}")
            # Include the exception details so callers can diagnose.
            # Generic "Internal tool error" without context is a debugging nightmare.
            import traceback

            tb_summary = traceback.format_exc().splitlines()[-5:]
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32000,
                    "message": f"Tool '{tool_name}' failed: {type(e).__name__}: {e}",
                    "data": {"traceback": tb_summary},
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Unknown method: {method}"},
    }


def main():
    logger.info("MemPalace MCP Server starting...")
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            request = json.loads(line)
            response = handle_request(request)
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Server error: {e}")


if __name__ == "__main__":
    main()
