#!/usr/bin/env python3
"""
MemPalace MCP Server — read/write palace access for Claude Code
================================================================
Install: claude mcp add mempalace -- python -m mempalace.mcp_server [--palace /path/to/palace]

Tools (read):
  mempalace_status          — total drawers, wing/room breakdown
  mempalace_list_wings      — all wings with drawer counts
  mempalace_list_rooms      — rooms within a wing
  mempalace_get_taxonomy    — full wing → room → count tree
  mempalace_search          — semantic search, optional wing/room filter
  mempalace_check_duplicate — check if content already exists before filing

Tools (write):
  mempalace_add_drawer      — file verbatim content into a wing/room
  mempalace_delete_drawer   — remove a drawer by ID
"""

import argparse
import os
import sys
import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path

from .config import MempalaceConfig, sanitize_name, sanitize_content
from .version import __version__
from .query_sanitizer import sanitize_query
from .searcher import search_memories
from .palace_graph import traverse, find_tunnels, graph_stats
import chromadb

from .knowledge_graph import KnowledgeGraph

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


def tool_status():
    col = _get_collection()
    if not col:
        return _no_palace()
    count = col.count()
    wings = {}
    rooms = {}
    batch_size = 5000
    offset = 0
    error_info = None
    while True:
        try:
            batch = col.get(include=["metadatas"], limit=batch_size, offset=offset)
            rows = batch["metadatas"]
            for m in rows:
                w = m.get("wing", "unknown")
                r = m.get("room", "unknown")
                wings[w] = wings.get(w, 0) + 1
                rooms[r] = rooms.get(r, 0) + 1
            offset += len(rows)
            if len(rows) < batch_size:
                break
        except Exception as e:
            error_info = f"Partial result, failed at offset {offset}: {str(e)}"
            break
    result = {
        "total_drawers": count,
        "wings": wings,
        "rooms": rooms,
        "palace_path": _config.palace_path,
        "protocol": PALACE_PROTOCOL,
        "aaak_dialect": AAAK_SPEC,
    }
    if error_info:
        result["error"] = error_info
        result["partial"] = True
    return result


# ── AAAK Dialect Spec ─────────────────────────────────────────────────────────
# Included in status response so the AI learns it on first wake-up call.
# Also available via mempalace_get_aaak_spec tool.

PALACE_PROTOCOL = """MemPalace Protocol — behavioral rules only. The system enforces the rest
(intent declaration, entity declaration, tool permissions, predicate constraints).

ON START:
  Call mempalace_wake_up. Read this protocol, the text (identity + rules),
  and declared (entities, predicates, intent types with their tools).

BEFORE ACTING ON ANY FACT:
  Query BOTH systems — kg_query/kg_search for structured entity facts,
  mempalace_search for prose context in drawers. Never guess.

WHEN FILING DRAWERS:
  - Call check_duplicate first. Skip if similarity >= 0.9.
  - Choose the precise predicate for the entity link: described_by,
    evidenced_by, derived_from, mentioned_in, session_note_for.
  - Then extract at least one KG triple from the content (twin pattern).
    Drawer alone = semantic search only. KG triple = fast entity lookup.

WHEN ADDING KG FACTS:
  - Declare new entities first (kg_declare_entity) with kind and importance.
  - Use properties for metadata: predicates need constraints, intent types
    need rules_profile (slots + tool_permissions).

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
  Then call diary_write to record what happened as a session summary."""

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


def tool_list_wings():
    col = _get_collection()
    if not col:
        return _no_palace()
    wings = {}
    batch_size = 5000
    offset = 0
    try:
        col.count()  # verify collection is accessible
    except Exception as e:
        return {"wings": {}, "error": str(e)}
    while True:
        try:
            batch = col.get(include=["metadatas"], limit=batch_size, offset=offset)
            rows = batch["metadatas"]
            for m in rows:
                w = m.get("wing", "unknown")
                wings[w] = wings.get(w, 0) + 1
            offset += len(rows)
            if len(rows) < batch_size:
                break
        except Exception as e:
            return {
                "wings": wings,
                "error": f"Partial result, failed at offset {offset}: {str(e)}",
                "partial": True,
            }
    return {"wings": wings}


def tool_list_rooms(wing: str = None):
    col = _get_collection()
    if not col:
        return _no_palace()
    rooms = {}
    batch_size = 5000
    offset = 0
    where = {"wing": wing} if wing else None
    try:
        col.count()  # verify collection is accessible
    except Exception as e:
        return {"wing": wing or "all", "rooms": {}, "error": str(e)}
    while True:
        try:
            kwargs = {"include": ["metadatas"], "limit": batch_size, "offset": offset}
            if where:
                kwargs["where"] = where
            batch = col.get(**kwargs)
            rows = batch["metadatas"]
            for m in rows:
                r = m.get("room", "unknown")
                rooms[r] = rooms.get(r, 0) + 1
            offset += len(rows)
            if len(rows) < batch_size:
                break
        except Exception as e:
            return {
                "wing": wing or "all",
                "rooms": rooms,
                "error": f"Partial result, failed at offset {offset}: {str(e)}",
                "partial": True,
            }
    return {"wing": wing or "all", "rooms": rooms}


def tool_get_taxonomy():
    col = _get_collection()
    if not col:
        return _no_palace()
    taxonomy = {}
    batch_size = 5000
    offset = 0
    try:
        col.count()  # verify collection is accessible
    except Exception as e:
        return {"taxonomy": {}, "error": str(e)}
    while True:
        try:
            batch = col.get(include=["metadatas"], limit=batch_size, offset=offset)
            rows = batch["metadatas"]
            for m in rows:
                w = m.get("wing", "unknown")
                r = m.get("room", "unknown")
                if w not in taxonomy:
                    taxonomy[w] = {}
                taxonomy[w][r] = taxonomy[w].get(r, 0) + 1
            offset += len(rows)
            if len(rows) < batch_size:
                break
        except Exception as e:
            return {
                "taxonomy": taxonomy,
                "error": f"Partial result, failed at offset {offset}: {str(e)}",
                "partial": True,
            }
    return {"taxonomy": taxonomy}


def _hybrid_score(similarity: float, importance: float, date_added_iso: str,
                   agent_match: bool = False) -> float:
    """Hybrid ranking score for search results.

    Combines semantic similarity with importance tier, time-decay, and agent affinity:

        hybrid = similarity + (importance - 3) * 0.1 - log10(age_days + 1) * 0.02 + agent_boost

    Properties:
        - Similarity dominates the shape of results
        - Importance nudges critical drawers up
        - Log-decay gently demotes old content
        - Agent affinity: when searching agent matches the drawer's added_by,
          boost by 0.15 — enough to surface own knowledge first within a
          similarity band, but not enough to override a much better semantic match
    """
    import math

    try:
        sim = float(similarity or 0.0)
    except (TypeError, ValueError):
        sim = 0.0
    try:
        imp = float(importance or 3.0)
    except (TypeError, ValueError):
        imp = 3.0

    # Age from date_added (fall back to large age if unknown)
    dt = _parse_iso_datetime_safe(date_added_iso)
    if dt is None:
        age_days = 365.0
    else:
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        age_seconds = max(0.0, (now - dt).total_seconds())
        age_days = age_seconds / 86400.0

    agent_boost = 0.15 if agent_match else 0.0

    return sim + (imp - 3.0) * 0.1 - math.log10(age_days + 1.0) * 0.02 + agent_boost


def _parse_iso_datetime_safe(value):
    """Parse ISO-format datetime (mcp_server local helper, mirrors layers._parse_iso_datetime)."""
    if not value or not isinstance(value, str):
        return None
    try:
        from datetime import timezone
        s = value.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError):
        return None


def tool_search(
    query: str,
    limit: int = 5,
    wing: str = None,
    room: str = None,
    context: str = None,
    sort_by: str = "hybrid",
    agent: str = None,
):
    """Search palace drawers with hybrid importance-decay-aware ranking by default.

    sort_by:
        "hybrid" (DEFAULT) — similarity + importance bonus + time-decay penalty.
                             Semantically-matching drawers dominate, but importance
                             and recency break ties. This is what you want for
                             almost every query.
        "similarity"       — pure ChromaDB vector similarity (legacy behavior).
                             Use when you want raw semantic match with no
                             importance/recency influence.
        "score"            — pure Layer1 decay-aware importance ranking
                             (importance*10 - log10(age+1)*0.5). Ignores similarity.
                             Use for wing-browse "what's critical in X" queries.
        "importance"       — pure importance DESC, ties by recency. Admin view.
        "date"             — chronological, most recent first. Diary reads.

    All non-similarity modes fetch ~limit*5 candidates via similarity first,
    then re-rank client-side. For pure metadata-based queries (no semantic
    intent at all), pass query='' to skip similarity and sort the whole
    filtered set.
    """
    from .layers import compute_decay_score

    # Mitigate system prompt contamination (Issue #333)
    sanitized = sanitize_query(query)

    # Determine fetch size: re-ranking modes need more candidates
    needs_rerank = sort_by != "similarity"
    fetch_limit = max(limit * 5, 50) if needs_rerank else limit

    result = search_memories(
        sanitized["clean_query"],
        palace_path=_config.palace_path,
        wing=wing,
        room=room,
        n_results=fetch_limit,
    )

    # Re-rank if requested (always when sort_by != "similarity", including default "hybrid")
    if needs_rerank and isinstance(result, dict) and result.get("results"):
        items = result["results"]

        def _importance(item):
            meta = item.get("metadata") or {}
            try:
                return float(meta.get("importance", 3))
            except (TypeError, ValueError):
                return 3.0

        def _date(item):
            meta = item.get("metadata") or {}
            return meta.get("date_added") or meta.get("filed_at") or ""

        def _similarity(item):
            try:
                return float(item.get("similarity") or 0.0)
            except (TypeError, ValueError):
                return 0.0

        def _agent_match(item):
            if not agent:
                return False
            meta = item.get("metadata") or {}
            return meta.get("added_by", "") == agent

        if sort_by == "hybrid":
            items.sort(
                key=lambda x: _hybrid_score(_similarity(x), _importance(x), _date(x), _agent_match(x)),
                reverse=True,
            )
        elif sort_by == "score":
            items.sort(key=lambda x: compute_decay_score(_importance(x), _date(x)), reverse=True)
        elif sort_by == "importance":
            items.sort(key=lambda x: (_importance(x), _date(x)), reverse=True)
        elif sort_by == "date":
            items.sort(key=lambda x: _date(x), reverse=True)
        else:
            return {
                "error": (
                    f"sort_by '{sort_by}' not supported. "
                    f"Use: hybrid (default), similarity, score, importance, date."
                )
            }

        result["results"] = items[:limit]
        result["sort_by"] = sort_by
        result["reranked"] = True

    # Attach sanitizer metadata for transparency
    if sanitized["was_sanitized"]:
        result["query_sanitized"] = True
        result["sanitizer"] = {
            "method": sanitized["method"],
            "original_length": sanitized["original_length"],
            "clean_length": sanitized["clean_length"],
            "clean_query": sanitized["clean_query"],
        }
    if context:
        result["context_received"] = True
    return result


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


def tool_find_tunnels(wing_a: str = None, wing_b: str = None):
    """Find rooms that bridge two wings — the hallways connecting domains."""
    col = _get_collection()
    if not col:
        return _no_palace()
    return find_tunnels(wing_a, wing_b, col=col)


def tool_graph_stats():
    """Palace graph overview: nodes, tunnels, edges, connectivity."""
    col = _get_collection()
    if not col:
        return _no_palace()
    return graph_stats(col=col)


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
    "entity",      # a concrete individual thing
    "predicate",   # a relationship type
    "class",       # a category/domain-type definition
    "literal",     # a raw value (string, integer, timestamp, URL, path)
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


def tool_add_drawer(
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
        return {"success": False, "error": "added_by is required. Pass your agent entity name (e.g., 'ga_agent', 'technical_lead_agent')."}
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
        return {"success": False, "error": "slug is required. Provide a short human-readable identifier (e.g. 'intent-pre-activation-issues')."}

    normalized_slug = _normalize_drawer_slug(slug)
    if not normalized_slug:
        return {"success": False, "error": f"slug '{slug}' normalizes to empty. Use alphanumeric words separated by hyphens."}

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
                "hint": "Choose a different slug, or use update_drawer_metadata to modify the existing drawer.",
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

        # Create entity→drawer link(s) using the specified predicate
        VALID_DRAWER_PREDICATES = {"described_by", "evidenced_by", "derived_from", "mentioned_in", "session_note_for"}
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
            # Auto-create entity if it doesn't exist (soft — drawer linking shouldn't fail)
            existing_entity = _kg.get_entity(eid)
            if not existing_entity:
                _kg.add_entity(ename, kind="entity", description=f"Auto-created from drawer link in {wing}/{room}")
            try:
                _kg.add_triple(eid, link_predicate, drawer_id)
                linked_entities.append(eid)
            except Exception:
                pass  # Non-fatal: drawer exists, linking failed

        return {
            "success": True,
            "drawer_id": drawer_id,
            "wing": wing,
            "room": room,
            "hall": hall,
            "importance": importance,
            "linked_entities": linked_entities,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_delete_drawer(drawer_id: str):
    """Delete a single drawer by ID."""
    col = _get_collection()
    if not col:
        return _no_palace()
    existing = col.get(ids=[drawer_id])
    if not existing["ids"]:
        return {"success": False, "error": f"Drawer not found: {drawer_id}"}

    # Log the deletion with the content being removed for audit trail
    deleted_content = existing.get("documents", [""])[0] if existing.get("documents") else ""
    deleted_meta = existing.get("metadatas", [{}])[0] if existing.get("metadatas") else {}
    _wal_log(
        "delete_drawer",
        {
            "drawer_id": drawer_id,
            "deleted_meta": deleted_meta,
            "content_preview": deleted_content[:200],
        },
    )

    try:
        col.delete(ids=[drawer_id])
        logger.info(f"Deleted drawer: {drawer_id}")
        return {"success": True, "drawer_id": drawer_id}
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
        from .layers import MemoryStack, compute_decay_score
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
                score = compute_decay_score(e.get("importance", 3), e.get("last_touched", ""))
                intent_entries.append((score, e))
        intent_entries.sort(key=lambda x: x[0], reverse=True)

        # Format: top-level as name(Tool1,Tool2), children as name<parent(+AddedTool)
        intent_parts = []
        for _score, e in intent_entries[:20]:
            _declared_entities.add(e["id"])
            eid = e["id"]
            parent = intent_parents.get(eid, "?")
            _, tools = _resolve_intent_profile(eid)
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



def tool_update_drawer_metadata(
    drawer_id: str,
    wing: str = None,
    room: str = None,
    hall: str = None,
    importance: int = None,
):
    """Update metadata fields on an existing drawer in place.

    Preserves embeddings (no re-vectorization) — faster than delete+recreate
    for wing/room migrations and retroactive hall/importance tagging.

    Any param left as None preserves the existing value. Use this for:
    - Retroactive hall classification on legacy drawers
    - Bumping importance when a drawer turns out to be more critical
    - Moving drawers between wings/rooms without re-embedding

    Note: cannot change the drawer_id or content. For those, use delete + add.
    """
    try:
        if wing is not None:
            wing = sanitize_name(wing, "wing")
        if room is not None:
            room = sanitize_name(room, "room")
        hall = _validate_hall(hall)
        importance = _validate_importance(importance)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    col = _get_collection()
    if not col:
        return _no_palace()

    existing = col.get(ids=[drawer_id], include=["metadatas"])
    if not existing["ids"]:
        return {"success": False, "error": f"Drawer not found: {drawer_id}"}

    old_meta = dict(existing["metadatas"][0])
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
        return {
            "success": True,
            "reason": "no_change",
            "drawer_id": drawer_id,
        }

    _wal_log(
        "update_drawer_metadata",
        {
            "drawer_id": drawer_id,
            "old_meta": old_meta,
            "new_meta": new_meta,
            "updated_fields": updated_fields,
        },
    )

    try:
        col.update(ids=[drawer_id], metadatas=[new_meta])
        logger.info(f"Updated drawer metadata: {drawer_id} fields={updated_fields}")
        return {
            "success": True,
            "drawer_id": drawer_id,
            "updated_fields": updated_fields,
            "new_metadata": new_meta,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ==================== KNOWLEDGE GRAPH ====================


def tool_kg_query(entity: str, as_of: str = None, direction: str = "both"):
    """Query the knowledge graph for an entity's relationships.

    Supports batch queries: pass a comma-separated list of entity names
    to query multiple entities in one call. Returns results keyed by entity.
    """
    entities = [e.strip() for e in entity.split(",") if e.strip()]

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


def tool_kg_search(query: str, limit: int = 5, kind: str = None, sort_by: str = "hybrid", agent: str = None):
    """Semantic search over KG entities. Returns matching entities with their relationships.

    Unlike kg_query (exact entity ID match), this uses vector similarity to find
    entities whose DESCRIPTIONS match your query. Use when you don't know the
    exact entity name, or want to discover related entities.

    Args:
        query: Natural language search (e.g. "database server", "editing rules",
               "deployment process"). Matched against entity descriptions.
        limit: Max entities to return (default 5).
        kind: Filter by entity kind: 'entity', 'predicate', 'class', 'literal'.
              If omitted, searches all kinds.
        sort_by: "hybrid" (DEFAULT) — similarity + importance bonus + time-decay.
                 "similarity" — pure vector match, no importance/decay influence.
    """
    from .layers import compute_decay_score

    ecol = _get_entity_collection(create=False)
    if not ecol:
        return {"success": False, "error": "Entity collection not found. No entities declared yet."}

    try:
        count = ecol.count()
        if count == 0:
            return {"query": query, "results": [], "count": 0}

        # Fetch extra candidates for re-ranking
        fetch_limit = max(limit * 5, 50) if sort_by == "hybrid" else limit
        query_kwargs = {
            "query_texts": [query],
            "n_results": min(fetch_limit, count),
            "include": ["documents", "metadatas", "distances"],
        }
        if kind:
            query_kwargs["where"] = {"kind": kind}

        results = ecol.query(**query_kwargs)
        candidates = []

        if results["ids"] and results["ids"][0]:
            for i, eid in enumerate(results["ids"][0]):
                dist = results["distances"][0][i]
                similarity = round(1 - dist, 3)
                meta = results["metadatas"][0][i] or {}
                doc = results["documents"][0][i]

                importance = meta.get("importance", 3)
                last_touched = meta.get("last_touched", "")

                # Use unified _hybrid_score for consistent ranking across all tools
                if sort_by == "hybrid":
                    is_match = bool(agent and meta.get("added_by") == agent)
                    hybrid = _hybrid_score(similarity, importance, last_touched, is_match)
                else:
                    hybrid = similarity

                candidates.append({
                    "entity_id": eid,
                    "name": meta.get("name", eid),
                    "description": doc,
                    "kind": meta.get("kind", "entity"),
                    "importance": importance,
                    "similarity": similarity,
                    "score": round(hybrid, 4),
                })

        # Sort by score and take top N
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top = candidates[:limit]

        # Fetch KG edges for top results only (avoid expensive queries on all candidates)
        for entity_result in top:
            edges = _kg.query_entity(entity_result["entity_id"], direction="both")
            current_edges = [e for e in edges if e.get("current", True)]
            entity_result["edges"] = current_edges
            entity_result["edge_count"] = len(current_edges)

        return {"query": query, "results": top, "count": len(top), "sort_by": sort_by}
    except Exception as e:
        return {"success": False, "error": f"KG search failed: {e}"}


def tool_kg_add(
    subject: str, predicate: str, object: str, valid_from: str = None, source_closet: str = None
):
    """Add a relationship to the knowledge graph.

    IMPORTANT: All three parts must be declared in this session:
    - subject: declared entity (any type EXCEPT predicate)
    - predicate: declared entity with type="predicate"
    - object: declared entity (any type EXCEPT predicate)

    Call kg_declare_entity for subject/object entities, and
    kg_declare_entity with kind="predicate" for predicates.
    """
    from .knowledge_graph import normalize_entity_name

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
                    e["object"] for e in _kg.query_entity(sub_normalized, direction="outgoing")
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
                    e["object"] for e in _kg.query_entity(obj_normalized, direction="outgoing")
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
                    e for e in _kg.query_entity(sub_normalized, direction="outgoing")
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
                    e for e in _kg.query_entity(obj_normalized, direction="incoming")
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

    _wal_log(
        "kg_add",
        {
            "subject": subject,
            "predicate": predicate,
            "object": object,
            "valid_from": valid_from,
            "source_closet": source_closet,
        },
    )
    triple_id = _kg.add_triple(
        sub_normalized, pred_normalized, obj_normalized, valid_from=valid_from, source_closet=source_closet
    )
    return {"success": True, "triple_id": triple_id, "fact": f"{sub_normalized} -> {pred_normalized} -> {obj_normalized}"}


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
    """Knowledge graph overview: entities, triples, relationship types."""
    return _kg.stats()


# ==================== ENTITY DECLARATION ====================

ENTITY_SIMILARITY_THRESHOLD = 0.85
ENTITY_COLLECTION_NAME = "mempalace_entities"

# Session-level declared entities (in-memory cache, falls back to persistent KG)
_declared_entities: set = set()
_session_id: str = ""


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


def _reset_declared_entities():
    """Reset the session's declared entities set (called on compact/clear/restart)."""
    global _declared_entities, _session_id
    _declared_entities = set()
    _session_id = ""


def _check_entity_similarity(
    description: str,
    kind_filter: str = None,
    exclude_id: str = None,
    threshold: float = None,
):
    """Check if a description is semantically similar to existing entities.

    Args:
        description: the description to check against existing entities.
        kind_filter: if provided, only check against entities of this type.
                     This creates type-scoped collision domains: systems
                     only collide with systems, predicates only with predicates.
        exclude_id: entity ID to exclude from results (self-check).
        threshold: similarity threshold (default ENTITY_SIMILARITY_THRESHOLD).

    Returns list of similar entities above threshold.
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
                    similar.append({
                        "entity_id": eid,
                        "name": meta.get("name", eid),
                        "description": doc,
                        "similarity": similarity,
                        "importance": meta.get("importance", 3),
                    })
        return similar
    except Exception:
        return []


def _sync_entity_to_chromadb(entity_id: str, name: str, description: str, kind: str, importance: int, added_by: str = None):
    """Sync an entity's description to the ChromaDB collection for similarity search."""
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


VALID_CARDINALITIES = {"many-to-many", "many-to-one", "one-to-many", "one-to-one"}


def tool_kg_declare_entity(
    name: str,
    description: str,
    kind: str = None,  # REQUIRED — no default, model must choose
    importance: int = 3,
    properties: dict = None,  # General-purpose metadata
    user_approved_star_scope: bool = False,  # Required for * scope
    added_by: str = None,  # REQUIRED — agent who declared this entity
):
    """Declare an entity before using it in KG edges. REQUIRED per session.

    Every entity used in kg_add (subject, predicate, or object) must be
    declared first. Declaration triggers similarity check against existing
    entities OF THE SAME KIND. If a collision is found (similarity > 0.85),
    the entity is BLOCKED until disambiguated or merged.

    Args:
        name: Entity name. Aggressively normalized (hyphens, underscores,
              CamelCase, articles all collapsed).
        description: Precise, unambiguous description.
              BAD:  "a server" (too generic, will collide with many entities)
              GOOD: "The DSpot paperclip platform server, started via pnpm
                     dev:once, listening on port 3100"
        kind: Ontological role — FIXED enum:
              'entity' (default) — a concrete individual thing
              'predicate' — a relationship type (use for KG edge labels)
              'class' — a category definition (domain type that entities is_a)
              'literal' — a raw value (string, number, timestamp)
              Collision detection is KIND-SCOPED: predicates only collide with
              predicates, entities only with entities, etc.
        importance: 1-5. 5=critical (production systems, hard rules),
                    4=canonical, 3=default, 2=low, 1=junk.
        properties: General-purpose metadata dict stored with the entity.
              Content depends on entity type:
              - Predicates: {"constraints": {"subject_kinds": [...], "object_kinds": [...],
                "subject_classes": [...], "object_classes": [...], "cardinality": "..."}}
                ALL 5 constraint fields are REQUIRED for predicates.
              - Intent types: {"rules_profile": {"slots": {"<name>": {"classes": [...],
                "required": true}}, "tool_permissions": [{"tool": "<Name>", "scope": "<pattern>"}]}}
                Scope must be specific — file patterns, command patterns, MCP wildcards.
                "*" scope requires user_approved_star_scope=true.
              - Any entity: arbitrary metadata as needed.

    Returns:
        status "created" — new entity, registered in session declared set.
        status "exists"  — entity already exists, registered in session.
        status "collision" — similar entities found, NOT registered.
                            You MUST resolve before using this entity.
    """
    from .knowledge_graph import normalize_entity_name

    try:
        description = sanitize_content(description, max_length=5000)
        importance = _validate_importance(importance)
        kind = _validate_kind(kind)
    except ValueError as e:
        return {"success": False, "error": str(e)}

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

    # Check for * scope in tool_permissions — requires user approval
    if properties and not user_approved_star_scope:
        rules_profile = properties.get("rules_profile", {})
        tool_perms = rules_profile.get("tool_permissions", [])
        star_tools = [p["tool"] for p in tool_perms if p.get("scope") == "*"]
        if star_tools:
            return {
                "success": False,
                "error": (
                    f"BLOCKED: Unrestricted scope (\"*\") for tools: {star_tools}.\n\n"
                    f"MANDATORY: You MUST ask the user RIGHT NOW and get an explicit YES "
                    f"before proceeding. Do NOT self-approve. Do NOT assume prior approval. "
                    f"Do NOT set user_approved_star_scope=true without asking.\n\n"
                    f"Ask the user exactly this:\n"
                    f"  \"I need to create intent type '{name}' with unrestricted (*) access "
                    f"to {', '.join(star_tools)}. This bypasses scope restrictions. Approve? (yes/no)\"\n\n"
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
                    "Example: properties={\"constraints\": {\"subject_kinds\": [\"entity\"], \"object_kinds\": [\"entity\"], "
                    "\"subject_classes\": [\"system\",\"process\"], \"object_classes\": [\"thing\"], "
                    "\"cardinality\": \"many-to-one\"}}"
                ),
            }
        # ALL 5 constraint fields are REQUIRED — no optionals
        for field in ("subject_kinds", "object_kinds", "subject_classes", "object_classes", "cardinality"):
            if field not in constraints:
                return {"success": False, "error": f"constraints must include '{field}'. ALL 5 fields are required: subject_kinds, object_kinds, subject_classes, object_classes, cardinality."}
        # Validate kind lists
        for field in ("subject_kinds", "object_kinds"):
            vals = constraints[field]
            if not isinstance(vals, list) or not vals:
                return {"success": False, "error": f"constraints['{field}'] must be a non-empty list of kinds."}
            for v in vals:
                if v not in VALID_KINDS:
                    return {"success": False, "error": f"constraints['{field}'] contains invalid kind '{v}'. Valid: {sorted(VALID_KINDS)}."}
        # Validate cardinality
        if constraints["cardinality"] not in VALID_CARDINALITIES:
            return {"success": False, "error": f"constraints['cardinality'] must be one of {sorted(VALID_CARDINALITIES)}."}
        # Validate subject_classes / object_classes reference real class-kind entities
        for cls_field in ("subject_classes", "object_classes"):
            cls_list = constraints[cls_field]
            if not isinstance(cls_list, list) or not cls_list:
                return {"success": False, "error": f"constraints['{cls_field}'] must be a non-empty list of class entity names. Use ['thing'] for any class."}
            for cls_name in cls_list:
                from .knowledge_graph import normalize_entity_name as _norm
                cls_entity = _kg.get_entity(_norm(cls_name))
                if not cls_entity:
                    return {"success": False, "error": f"constraints['{cls_field}'] references class '{cls_name}' which doesn't exist. Declare it first with kind='class'."}
                if cls_entity.get("kind") != "class":
                    return {"success": False, "error": f"constraints['{cls_field}'] references '{cls_name}' which is kind='{cls_entity.get('kind')}', not 'class'."}

    normalized = normalize_entity_name(name)
    if not normalized or normalized == "unknown":
        return {"success": False, "error": f"Entity name '{name}' normalizes to nothing. Use a more descriptive name."}

    # Check for exact match (already exists)
    existing = _kg.get_entity(normalized)
    if existing:
        # Check for collisions with OTHER entities of SAME KIND (not self)
        similar = _check_entity_similarity(description, kind_filter=kind, exclude_id=normalized)
        if similar:
            return {
                "success": False,
                "status": "collision",
                "entity_id": normalized,
                "kind": kind,
                "message": (
                    f"Entity '{normalized}' (kind={kind}) collides with other {kind}s. "
                    f"Disambiguate via kg_update_entity_description or merge via kg_merge_entities."
                ),
                "collisions": similar,
            }
        # No collisions — register in session
        _declared_entities.add(normalized)
        # Update description + importance + kind if provided and different
        if description and description != existing.get("description", ""):
            _kg.update_entity_description(normalized, description, importance)
            _sync_entity_to_chromadb(normalized, name, description, kind, importance or 3)
        return {
            "success": True,
            "status": "exists",
            "entity_id": normalized,
            "kind": existing.get("kind", "entity"),
            "description": existing.get("description") or description,
            "importance": existing.get("importance", 3),
            "edge_count": _kg.entity_edge_count(normalized),
        }

    # New entity — check for collisions via description similarity (same KIND only)
    similar = _check_entity_similarity(description, kind_filter=kind)
    if similar:
        return {
            "success": False,
            "status": "collision",
            "entity_id": normalized,
            "message": (
                f"Cannot create entity '{normalized}': its description is too similar to existing entities. "
                f"Either (1) merge with the matching entity via kg_merge_entities(source='{normalized}', "
                f"target='{similar[0]['entity_id']}'), or (2) rewrite your description to be more "
                f"specific and re-declare. The entity is NOT usable until resolved."
            ),
            "collisions": similar,
        }

    # No collisions — create the entity
    props = properties if isinstance(properties, dict) else {}
    if added_by:
        props["added_by"] = added_by
    _kg.add_entity(name, description=description, importance=importance or 3, kind=kind, properties=props)
    _sync_entity_to_chromadb(normalized, name, description, kind, importance or 3, added_by=added_by)
    _declared_entities.add(normalized)

    # Auto-add is-a thing for new class entities (ensures class inheritance works)
    if kind == "class" and normalized != "thing":
        try:
            _kg.add_triple(normalized, "is-a", "thing")
        except Exception:
            pass  # Non-fatal if thing doesn't exist yet

    _wal_log("kg_declare_entity", {
        "entity_id": normalized,
        "name": name,
        "description": description[:200],
        "kind": kind,
        
        "importance": importance,
    })

    return {
        "success": True,
        "status": "created",
        "entity_id": normalized,
        "kind": kind,
        "description": description,
        "importance": importance or 3,
        
    }


def tool_kg_update_entity_description(
    entity: str,
    new_description: str,
    check_against: str = None,
):
    """Update an entity's description and check distance from colliding entities.

    Use after kg_declare_entity returns 'collision'. Update the description
    to make it semantically distinct from the colliding entity, then re-declare.

    Args:
        entity: The entity to update.
        new_description: The improved, more specific description.
        check_against: Entity to measure distance from (optional).
                       If not provided, checks against all entities above threshold.

    Returns:
        Distance checks with similarity scores and is_distinct flags.
    """
    from .knowledge_graph import normalize_entity_name

    try:
        new_description = sanitize_content(new_description, max_length=5000)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    normalized = normalize_entity_name(entity)
    existing = _kg.get_entity(normalized)
    if not existing:
        return {"success": False, "error": f"Entity '{normalized}' not found."}

    # Update in SQLite + ChromaDB
    updated = _kg.update_entity_description(normalized, new_description)
    _sync_entity_to_chromadb(
        normalized, existing["name"], new_description,
        existing.get("type", "concept"), existing.get("importance", 3)
    )

    # Distance checks
    distance_checks = []
    if check_against:
        check_id = normalize_entity_name(check_against)
        check_entity = _kg.get_entity(check_id)
        if check_entity:
            similar = _check_entity_similarity(new_description, exclude_id=normalized, threshold=0.0)
            for s in similar:
                if s["entity_id"] == check_id:
                    distance_checks.append({
                        "compared_to": check_id,
                        "similarity": s["similarity"],
                        "is_distinct": s["similarity"] < ENTITY_SIMILARITY_THRESHOLD,
                        "threshold": ENTITY_SIMILARITY_THRESHOLD,
                    })
                    break
            else:
                distance_checks.append({
                    "compared_to": check_id,
                    "similarity": 0.0,
                    "is_distinct": True,
                    "threshold": ENTITY_SIMILARITY_THRESHOLD,
                })
    else:
        # Check against all entities above a low threshold
        similar = _check_entity_similarity(new_description, exclude_id=normalized, threshold=0.7)
        for s in similar:
            distance_checks.append({
                "compared_to": s["entity_id"],
                "similarity": s["similarity"],
                "is_distinct": s["similarity"] < ENTITY_SIMILARITY_THRESHOLD,
                "threshold": ENTITY_SIMILARITY_THRESHOLD,
            })

    all_distinct = all(d["is_distinct"] for d in distance_checks) if distance_checks else True

    return {
        "success": True,
        "entity_id": normalized,
        "description_updated": True,
        "new_description": new_description,
        "distance_checks": distance_checks,
        "all_distinct": all_distinct,
        "hint": "All clear — re-declare this entity to register it." if all_distinct
                else "Still too similar to some entities. Make your description more specific.",
    }


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
    _wal_log("kg_merge_entities", {
        "source": source,
        "target": target,
        "update_description": update_description[:200] if update_description else None,
    })

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
                target_id, target_entity["name"],
                target_entity["description"], target_entity.get("type", "concept"),
                target_entity.get("importance", 3)
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


def tool_kg_update_predicate_constraints(
    predicate: str,
    constraints: dict,
):
    """Update constraints on an existing predicate entity.

    Use when: a predicate's constraints are too narrow or too broad,
    or when seeding constraints on predicates that lack them.

    Args:
        predicate: the predicate entity name
        constraints: new constraints dict with subject_kinds, object_kinds,
                     optional subject_classes, object_classes, cardinality.
    """
    from .knowledge_graph import normalize_entity_name
    import json as _json

    normalized = normalize_entity_name(predicate)
    entity = _kg.get_entity(normalized)
    if not entity:
        return {"success": False, "error": f"Predicate '{normalized}' not found."}
    if entity.get("kind") != "predicate":
        return {"success": False, "error": f"'{normalized}' is kind='{entity.get('kind')}', not 'predicate'."}

    # Validate constraints
    for field in ("subject_kinds", "object_kinds"):
        if field not in constraints:
            return {"success": False, "error": f"constraints must include '{field}'."}
        vals = constraints[field]
        if not isinstance(vals, list) or not vals:
            return {"success": False, "error": f"constraints['{field}'] must be a non-empty list."}
        for v in vals:
            if v not in VALID_KINDS:
                return {"success": False, "error": f"Invalid kind '{v}' in constraints['{field}']. Valid: {sorted(VALID_KINDS)}."}
    if "cardinality" in constraints:
        if constraints["cardinality"] not in VALID_CARDINALITIES:
            return {"success": False, "error": f"Invalid cardinality. Valid: {sorted(VALID_CARDINALITIES)}."}
    # Validate class references
    for cls_field in ("subject_classes", "object_classes"):
        if cls_field in constraints:
            for cls_name in constraints[cls_field]:
                cls_eid = normalize_entity_name(cls_name)
                cls_ent = _kg.get_entity(cls_eid)
                if not cls_ent:
                    return {"success": False, "error": f"Class '{cls_name}' not found. Declare with kind='class' first."}
                if cls_ent.get("kind") != "class":
                    return {"success": False, "error": f"'{cls_name}' is kind='{cls_ent.get('kind')}', not 'class'."}

    # Update properties
    conn = _kg._conn()
    existing_props = entity.get("properties", {})
    if isinstance(existing_props, str):
        try:
            existing_props = _json.loads(existing_props)
        except Exception:
            existing_props = {}
    existing_props["constraints"] = constraints
    conn.execute(
        "UPDATE entities SET properties = ? WHERE id = ?",
        (_json.dumps(existing_props), normalized),
    )
    conn.commit()

    return {
        "success": True,
        "predicate": normalized,
        "constraints": constraints,
    }


def tool_kg_list_declared():
    """List all entities declared in this session."""
    results = []
    for eid in sorted(_declared_entities):
        entity = _kg.get_entity(eid)
        if entity:
            results.append({
                "entity_id": eid,
                "name": entity["name"],
                "description": entity["description"],
                "importance": entity["importance"],
                "last_touched": entity["last_touched"],
                "edge_count": _kg.entity_edge_count(eid),
            })
    return {
        "declared_count": len(results),
        "entities": results,
    }


def tool_kg_entity_info(entity: str):
    """Get full details for an entity: description, edges, linked drawers.

    Entity must be declared in this session to query.
    """
    from .knowledge_graph import normalize_entity_name
    normalized = normalize_entity_name(entity)

    if not _is_declared(normalized):
        return {
            "success": False,
            "error": (
                f"Entity '{normalized}' has not been declared in this session. "
                f"Call kg_declare_entity(name='{entity}', description='...') first. "
                f"All entities must be declared before use."
            ),
        }

    info = _kg.get_entity(normalized)
    if not info:
        return {"success": False, "error": f"Entity '{normalized}' not found in KG."}

    edges = _kg.query_entity(normalized, direction="both")
    return {
        "success": True,
        "entity_id": normalized,
        "name": info["name"],
        "description": info["description"],
        "type": info["type"],
        "importance": info["importance"],
        "last_touched": info["last_touched"],
        "status": info["status"],
        "edge_count": len(edges),
        "edges": edges,
    }


# ==================== INTENT DECLARATION ====================

_active_intent = None  # Session-level: only one active intent at a time
_INTENT_STATE_DIR = Path(os.path.expanduser("~/.mempalace/hook_state"))


def _intent_state_path() -> Path:
    """Get session-scoped intent state file path."""
    return _INTENT_STATE_DIR / f"active_intent_{_session_id or 'default'}.json"


def _build_intent_hierarchy() -> list:
    """Build a list of all intent types with their tools and is_a parent.

    Walks the KG to find all entities that is_a intent_type (directly or
    transitively). Returns a list of dicts with id, parent, tools.
    Used in error messages so the model knows what types exist and can
    create new ones with the right parent.
    """
    from .knowledge_graph import normalize_entity_name

    hierarchy = []
    # Find all entities in the KG that might be intent types
    ecol = _get_entity_collection(create=False)
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
        edges = _kg.query_entity(eid, direction="outgoing")
        parent_id = None
        for e in edges:
            if e["predicate"] in ("is-a", "is_a") and e["current"]:
                obj = normalize_entity_name(e["object"])
                if obj == "intent_type":
                    parent_id = "intent-type"
                    break
                # Check if parent is itself an intent type
                parent_edges = _kg.query_entity(obj, direction="outgoing")
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

        hierarchy.append({
            "id": eid,
            "parent": parent_id,
            "tools": tool_names,
        })

    # Sort: top-level first, then children
    hierarchy.sort(key=lambda x: (0 if x["parent"] == "intent-type" else 1, x["id"]))
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
        _INTENT_STATE_DIR.mkdir(parents=True, exist_ok=True)
        state_file = _intent_state_path()
        if _active_intent:
            state = {
                "intent_id": _active_intent["intent_id"],
                "intent_type": _active_intent["intent_type"],
                "slots": _active_intent["slots"],
                "effective_permissions": _active_intent["effective_permissions"],
                "description": _active_intent.get("description", ""),
                "session_id": _session_id,
                "intent_hierarchy": _build_intent_hierarchy_safe(),
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
    from .knowledge_graph import normalize_entity_name

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

        entity = _kg.get_entity(current)
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
        edges = _kg.query_entity(current, direction="outgoing")
        parent = None
        for e in edges:
            if e["predicate"] in ("is-a", "is_a") and e["current"]:
                parent_id = normalize_entity_name(e["object"])
                # Skip if parent is a class (intent_type, thing) — stop at intent types
                parent_entity = _kg.get_entity(parent_id)
                if parent_entity and parent_entity.get("kind") == "entity":
                    parent = parent_id
                break
        if not parent:
            break
        current = parent

    return merged_slots, merged_tools


def _is_intent_type(entity_id: str) -> bool:
    """Check if an entity is-a intent_type (direct or inherited)."""
    from .knowledge_graph import normalize_entity_name

    edges = _kg.query_entity(entity_id, direction="outgoing")
    for e in edges:
        if e["predicate"] in ("is-a", "is_a") and e["current"]:
            obj = normalize_entity_name(e["object"])
            if obj == "intent_type":
                return True
            # Check parent (one level — e.g., edit_file is-a modify is-a intent_type)
            parent_edges = _kg.query_entity(obj, direction="outgoing")
            for pe in parent_edges:
                if pe["predicate"] in ("is-a", "is_a") and pe["current"]:
                    if normalize_entity_name(pe["object"]) == "intent_type":
                        return True
    return False


def tool_declare_intent(
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
    global _active_intent
    from .knowledge_graph import normalize_entity_name

    # ── Validate intent_type ──
    try:
        intent_type = sanitize_name(intent_type, "intent_type")
    except ValueError as e:
        return {"success": False, "error": str(e)}

    intent_id = normalize_entity_name(intent_type)

    if not _is_declared(intent_id):
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
    all_entities = _kg.list_entities(status="active", kind="class")
    for e in all_entities:
        e_edges = _kg.query_entity(e["id"], direction="outgoing")
        for edge in e_edges:
            if edge["predicate"] in ("is-a", "is_a") and edge["current"]:
                parent_id = normalize_entity_name(edge["object"])
                if parent_id == intent_id:
                    subtypes.append({
                        "id": e["id"],
                        "description": e.get("description", ""),
                    })
                    break

    if subtypes and description.strip():
        ecol = _get_entity_collection(create=False)
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
                                child_scores.append({
                                    "id": eid,
                                    "distance": dist,
                                    "description": results["documents"][0][i],
                                })
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
                                s for s, d in child_slots.items()
                                if d.get("required", False) and s not in slots
                            ]
                            if not missing:
                                compatible.append(candidate)
                        if len(compatible) == 1:
                            narrowed_from = intent_id
                            intent_id = compatible[0]["id"]
                            _declared_entities.add(intent_id)
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
                f"Example: {{{', '.join(f'\"{k}\": [\"entity_name\"]' for k in effective_slots)}}}"
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
            if not _is_declared(val_id) and is_file_slot:
                file_exists = os.path.exists(val) or os.path.exists(
                    os.path.join(os.getcwd(), val)
                )
                if file_exists or auto_declare_files:
                    # Auto-declare: create entity from basename + is-a file
                    _kg.add_entity(
                        file_basename, kind="entity",
                        description=f"File: {val}" + (" (new)" if not file_exists else ""),
                        importance=2,
                    )
                    _kg.add_triple(val_id, "is-a", "file")
                    _sync_entity_to_chromadb(val_id, file_basename, f"File: {val}", "entity", 2)
                    _declared_entities.add(val_id)
                elif not file_exists:
                    slot_errors.append(
                        f"File '{val}' does not exist on disk and auto_declare_files=false. "
                        f"Either provide an existing file path, or set auto_declare_files=true "
                        f"if you intend to create this file."
                    )
                    continue

            if not _is_declared(val_id):
                slot_errors.append(
                    f"Entity '{val_id}' in slot '{slot_name}' not declared. "
                    f"Call kg_declare_entity first."
                )
                continue

            # Check class constraint via is-a + inheritance
            if "thing" not in allowed_classes:
                entity_classes = [
                    e["object"] for e in _kg.query_entity(val_id, direction="outgoing")
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
                                for e in _kg.query_entity(cls, direction="outgoing"):
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
    raw_paths = {}   # slot_name -> [raw_value, ...]
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
                    permissions.append({
                        "tool": perm["tool"],
                        "scope": scope.replace(f"{{{slot_name}}}", raw_val),
                        "slot": slot_name,
                        "entity": entity_id,
                    })
            elif scope == "*":
                if not any(p["tool"] == perm["tool"] and p["scope"] == "*" for p in permissions):
                    permissions.append({"tool": perm["tool"], "scope": "*"})

    # ── Collect context ──
    context = {"target_facts": [], "intent_rules": [], "relevant_memories": []}

    # Facts about all slot entities
    for entity_id in all_slot_entities:
        edges = _kg.query_entity(entity_id, direction="both")
        for e in edges:
            if e.get("current", True):
                context["target_facts"].append(
                    f"{e.get('subject', entity_id)} -> {e['predicate']} -> {e.get('object', '?')}"
                )

    # Rules on the intent type itself
    intent_edges = _kg.query_entity(intent_id, direction="outgoing")
    for e in intent_edges:
        if e.get("current", True) and e["predicate"] not in ("is-a", "is_a"):
            context["intent_rules"].append(
                f"{intent_id} -> {e['predicate']} -> {e.get('object', '?')}"
            )

    # Relevant memories via search (deduped against prior injections)
    already_injected = set()
    if _active_intent:
        already_injected = _active_intent.get("injected_drawer_ids", set())

    for entity_id in all_slot_entities:
        entity = _kg.get_entity(entity_id)
        search_query = entity["name"] if entity else entity_id
        try:
            search_result = search_memories(
                search_query, palace_path=_config.palace_path, n_results=3
            )
            if isinstance(search_result, dict) and search_result.get("results"):
                for r in search_result["results"]:
                    drawer_id = r.get("id", "")
                    if drawer_id not in already_injected:
                        already_injected.add(drawer_id)
                        context["relevant_memories"].append({
                            "drawer_id": drawer_id,
                            "snippet": (r.get("text") or "")[:200],
                            "for_entity": entity_id,
                        })
        except Exception:
            pass  # Non-fatal — context injection is best-effort

    # ── Historical injection: surface past executions of this intent type ──
    past_executions = []
    try:
        ecol = _get_entity_collection(create=False)
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
                    edges = _kg.query_entity(eid, direction="outgoing")
                    is_execution = False
                    exec_data = {"entity_id": eid, "relationships": []}
                    for e in edges:
                        if not e.get("current", True):
                            continue
                        pred = e["predicate"]
                        obj = e.get("object", "")
                        # Check is_a matches our intent type (or parent)
                        if pred in ("is-a", "is_a") and obj in (intent_id, ):
                            is_execution = True
                        # Collect ALL relationships
                        exec_data["relationships"].append(
                            f"{pred} -> {obj}"
                        )

                    if is_execution:
                        dist = exec_search["distances"][0][i]
                        similarity = round(1 - dist, 3)
                        exec_data["similarity"] = similarity
                        exec_data["description"] = (exec_search["documents"][0][i] or "")[:200]
                        exec_data["outcome"] = meta.get("outcome", "unknown")
                        exec_data["agent"] = meta.get("added_by", "")
                        past_executions.append(exec_data)

                # Sort by similarity
                past_executions.sort(key=lambda x: x.get("similarity", 0), reverse=True)
                past_executions = past_executions[:5]  # Top 5
    except Exception:
        pass  # Non-fatal

    if past_executions:
        context["past_executions"] = past_executions

    # ── Mandatory type promotion check: 3+ similar executions without specific type ──
    # If this is a broad type (inspect, modify, execute, communicate) and there are
    # 3+ similar past executions, force the agent to create a specific type.
    BROAD_TYPES = {"inspect", "modify", "execute", "communicate"}
    if intent_id in BROAD_TYPES and len(past_executions) >= 3:
        # Check if the similar executions share high semantic similarity
        high_sim = [e for e in past_executions if e.get("similarity", 0) > 0.7]
        if len(high_sim) >= 3:
            exec_list = "\n".join(
                f"  - {e['entity_id']}: {e['description'][:100]}"
                for e in high_sim[:5]
            )
            return {
                "success": False,
                "error": (
                    f"Intent type '{intent_id}' is too broad — {len(high_sim)} similar past executions "
                    f"found without a specific type. You MUST either:\n\n"
                    f"(a) Create a specific intent type:\n"
                    f"    kg_declare_entity(name='<specific-type>', kind='class', importance=4, "
                    f"description='<what this action does>')\n"
                    f"    kg_add(subject='<specific-type>', predicate='is_a', object='{intent_id}')\n"
                    f"    Then re-declare with the specific type.\n\n"
                    f"(b) Disambiguate existing executions (if they're actually different):\n"
                    f"    kg_update_entity_description(entity='<exec_id>', description='<more specific>')\n\n"
                    f"Similar executions found:\n{exec_list}"
                ),
                "similar_executions": high_sim[:5],
            }

    # ── Hard fail if previous intent not finalized ──
    previous_expired = None
    if _active_intent:
        prev_id = _active_intent.get("intent_id")
        prev_type = _active_intent.get("intent_type", "unknown")
        prev_desc = _active_intent.get("description", "")
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

    import hashlib
    intent_hash = hashlib.md5(f"{intent_id}:{description}:{datetime.now().isoformat()}".encode()).hexdigest()[:12]
    new_intent_id = f"intent_{intent_id}_{intent_hash}"

    _active_intent = {
        "intent_id": new_intent_id,
        "intent_type": intent_id,
        "slots": flat_slots,
        "effective_permissions": permissions,
        "injected_drawer_ids": already_injected,
        "description": description,
    }

    # Persist to state file for PreToolUse hook (runs in separate process)
    _persist_active_intent()

    _wal_log("declare_intent", {
        "intent_id": new_intent_id,
        "intent_type": intent_id,
        "slots": flat_slots,
        "description": description[:200],
    })

    # ── Suggest more specific subtypes (reuse subtypes found during auto-narrow) ──
    # If we narrowed, re-discover subtypes of the NEW (narrowed) intent type
    if narrowed_from:
        subtypes = []
        for e in all_entities:
            e_edges = _kg.query_entity(e["id"], direction="outgoing")
            for edge in e_edges:
                if edge["predicate"] in ("is-a", "is_a") and edge["current"]:
                    parent_id = normalize_entity_name(edge["object"])
                    if parent_id == intent_id:
                        subtypes.append({
                            "id": e["id"],
                            "description": e.get("description", "")[:120],
                        })
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
    if not _active_intent:
        return {
            "active": False,
            "message": "No active intent. Call mempalace_declare_intent before acting.",
        }
    return {
        "active": True,
        "intent_id": _active_intent["intent_id"],
        "intent_type": _active_intent["intent_type"],
        "slots": _active_intent["slots"],
        "permissions": _active_intent["effective_permissions"],
        "description": _active_intent.get("description", ""),
        "injected_memories": len(_active_intent.get("injected_drawer_ids", set())),
    }


def tool_finalize_intent(
    slug: str,
    outcome: str,
    summary: str,
    agent: str,
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
        key_actions: Abbreviated tool+params list (optional — auto-filled from trace if omitted)
        gotchas: List of gotcha descriptions discovered during execution
        learnings: List of lesson descriptions worth remembering
        promote_gotchas_to_type: Also link gotchas to the intent type (not just execution)
    """
    global _active_intent
    from .knowledge_graph import normalize_entity_name

    if not _active_intent:
        return {"success": False, "error": "No active intent to finalize."}

    intent_type = _active_intent["intent_type"]
    intent_desc = _active_intent.get("description", "")
    slot_entities = []
    for slot_name, slot_vals in _active_intent.get("slots", {}).items():
        if isinstance(slot_vals, list):
            slot_entities.extend(slot_vals)
        elif isinstance(slot_vals, str):
            slot_entities.append(slot_vals)

    # Normalize slug
    exec_id = normalize_entity_name(slug)
    if not exec_id:
        return {"success": False, "error": "slug normalizes to empty."}

    # ── Read execution trace from hook state file ──
    trace_entries = []
    try:
        trace_file = _INTENT_STATE_DIR / f"execution_trace_{_session_id or 'default'}.jsonl"
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
    try:
        _kg.add_entity(
            exec_id,
            kind="entity",
            description=f"{intent_desc or intent_type}: {summary[:200]}",
            importance=3,
            properties={
                "outcome": outcome,
                "agent": agent,
                "intent_type": intent_type,
                "finalized_at": datetime.now().isoformat(),
            },
        )
    except Exception as e:
        return {"success": False, "error": f"Failed to create execution entity: {e}"}

    # ── KG relationships ──
    edges_created = []

    # is_a → intent type (entity is_a class = instantiation)
    try:
        _kg.add_triple(exec_id, "is_a", intent_type)
        edges_created.append(f"{exec_id} is_a {intent_type}")
    except Exception:
        pass

    # executed_by → agent
    try:
        _kg.add_triple(exec_id, "executed_by", agent)
        edges_created.append(f"{exec_id} executed_by {agent}")
    except Exception:
        pass

    # targeted → slot entities
    for target in slot_entities:
        try:
            target_id = normalize_entity_name(target)
            _kg.add_triple(exec_id, "targeted", target_id)
            edges_created.append(f"{exec_id} targeted {target_id}")
        except Exception:
            pass

    # outcome as has_value
    try:
        _kg.add_triple(exec_id, "has_value", outcome)
        edges_created.append(f"{exec_id} has_value {outcome}")
    except Exception:
        pass

    # ── Result drawer (summary) ──
    result_drawer_id = None
    try:
        # Determine wing from agent
        agent_id = normalize_entity_name(agent)
        wing = f"wing_{agent_id.replace('_agent', '').replace('-agent', '')}"

        result = tool_add_drawer(
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
                f"- [{e.get('ts', '')}] {e['tool']} {e.get('target', '')}"
                for e in trace_entries
            )
            trace_result = tool_add_drawer(
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
                    existing = _kg.get_entity(gotcha_id)
                    if not existing:
                        _kg.add_entity(gotcha_id, kind="entity",
                                       description=gotcha_desc, importance=3)
                    _kg.add_triple(exec_id, "has_gotcha", gotcha_id)
                    edges_created.append(f"{exec_id} has_gotcha {gotcha_id}")
                    if promote_gotchas_to_type:
                        _kg.add_triple(intent_type, "has_gotcha", gotcha_id)
                        edges_created.append(f"{intent_type} has_gotcha {gotcha_id}")
            except Exception:
                pass

    # ── Learnings ──
    if learnings:
        for i, learning in enumerate(learnings):
            try:
                tool_add_drawer(
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

    # ── Deactivate intent ──
    _active_intent = None
    _persist_active_intent()

    return {
        "success": True,
        "execution_entity": exec_id,
        "outcome": outcome,
        "edges_created": edges_created,
        "trace_entries": len(trace_entries),
        "result_drawer": result_drawer_id,
    }


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

    This is the agent's personal journal — observations, thoughts,
    what it worked on, what it noticed, what it thinks matters.

    Args:
        slug: Descriptive identifier for this entry (e.g. 'session12-intent-narrowing-shipped').
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
    "mempalace_status": {
        "description": "Palace overview — total drawers, wing and room counts",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_status,
    },
    "mempalace_list_wings": {
        "description": "List all wings with drawer counts",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_list_wings,
    },
    "mempalace_list_rooms": {
        "description": "List rooms within a wing (or all rooms if no wing given)",
        "input_schema": {
            "type": "object",
            "properties": {
                "wing": {"type": "string", "description": "Wing to list rooms for (optional)"},
            },
        },
        "handler": tool_list_rooms,
    },
    "mempalace_get_taxonomy": {
        "description": "Full taxonomy: wing → room → drawer count",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_get_taxonomy,
    },
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
        "description": "Semantic search over KG entities by description similarity. Returns matching entities WITH their current relationships. Use when you don't know the exact entity name, want to discover related entities, or need fuzzy matching. Unlike kg_query (exact ID), this finds entities whose descriptions match your natural language query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query matched against entity descriptions (e.g. 'database server', 'editing rules', 'deployment process')",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max entities to return (default 5)",
                },
                "kind": {
                    "type": "string",
                    "description": "Filter by entity kind: 'entity', 'predicate', 'class', 'literal'. Omit to search all kinds.",
                    "enum": ["entity", "predicate", "class", "literal"],
                },
                "agent": {
                    "type": "string",
                    "description": "Your agent entity name for affinity scoring. Entities created by you get a ranking boost in hybrid mode. Examples: 'ga_agent', 'technical_lead_agent'.",
                },
            },
            "required": ["query", "agent"],
        },
        "handler": tool_kg_search,
    },
    "mempalace_kg_add": {
        "description": "Add a fact to the knowledge graph. Subject → predicate → object with optional time window. E.g. ('Max', 'started_school', 'Year 7', valid_from='2026-09-01').",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "The entity doing/being something"},
                "predicate": {
                    "type": "string",
                    "description": "The relationship type (e.g. 'loves', 'works_on', 'daughter_of')",
                },
                "object": {"type": "string", "description": "The entity being connected to"},
                "valid_from": {
                    "type": "string",
                    "description": "When this became true (YYYY-MM-DD, optional)",
                },
                "source_closet": {
                    "type": "string",
                    "description": "Closet ID where this fact appears (optional)",
                },
            },
            "required": ["subject", "predicate", "object"],
        },
        "handler": tool_kg_add,
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
        "description": "REQUIRED before using any entity in kg_add. Declare an entity with a precise description. Triggers KIND-SCOPED similarity check (entities only collide with entities, predicates only with predicates). Collision BLOCKS the entity until disambiguated or merged. Use kind='predicate' for relationship types, kind='class' for category definitions, kind='entity' (default) for concrete things.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Entity name (will be normalized: hyphens/underscores/CamelCase all collapsed)"},
                "description": {
                    "type": "string",
                    "description": "Precise description. BAD: 'a server'. GOOD: 'The DSpot paperclip platform server, started via pnpm dev:once, listening on port 3100'. Generic descriptions collide with many entities, forcing disambiguation.",
                },
                "kind": {
                    "type": "string",
                    "description": "Ontological role (FIXED 4 values): 'entity' (default, concrete thing), 'predicate' (relationship type for kg_add edges), 'class' (category definition, other entities is_a this), 'literal' (raw value).",
                    "enum": ["entity", "predicate", "class", "literal"],
                },
                "importance": {
                    "type": "integer",
                    "description": "1-5. 5=critical, 4=canonical, 3=default, 2=low, 1=junk.",
                    "minimum": 1,
                    "maximum": 5,
                },
                "properties": {
                    "type": "object",
                    "description": "General-purpose metadata stored with the entity. Content depends on entity type. For predicates: {\"constraints\": {\"subject_kinds\": [\"entity\"], \"object_kinds\": [\"entity\"], \"subject_classes\": [\"thing\"], \"object_classes\": [\"thing\"], \"cardinality\": \"many-to-many\"}} (ALL 5 constraint fields REQUIRED). For intent types: {\"rules_profile\": {\"slots\": {\"<name>\": {\"classes\": [\"thing\"], \"required\": true}}, \"tool_permissions\": [{\"tool\": \"Read\", \"scope\": \"src/**\"}, {\"tool\": \"Bash\", \"scope\": \"pytest\"}]}}. Scope must be specific (file patterns, command patterns) — \"*\" requires user approval.",
                },
                "added_by": {
                    "type": "string",
                    "description": "Agent who is declaring this entity. Must be a declared agent (is_a agent). Used for agent affinity scoring in searches.",
                },
                "user_approved_star_scope": {
                    "type": "boolean",
                    "description": "NEVER set this to true unless the user JUST said YES in this conversation turn. You MUST ask the user and receive explicit approval RIGHT NOW — not before, not assumed, not inferred. If the user has not responded YES to your approval request in this turn, this MUST be false or omitted.",
                },
            },
            "required": ["name", "description", "kind", "importance", "added_by"],
        },
        "handler": tool_kg_declare_entity,
    },
    "mempalace_kg_update_entity_description": {
        "description": "Update an entity's description to disambiguate from colliding entities. Use after kg_declare_entity returns 'collision'. Returns distance checks showing whether the entities are now distinct enough.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {"type": "string", "description": "Entity to update"},
                "new_description": {"type": "string", "description": "Improved, more specific description"},
                "check_against": {"type": "string", "description": "Entity to measure distance from (optional — auto-checks all if omitted)"},
            },
            "required": ["entity", "new_description"],
        },
        "handler": tool_kg_update_entity_description,
    },
    "mempalace_kg_merge_entities": {
        "description": "Merge source entity into target. All edges rewritten, source becomes alias. Use when kg_declare_entity returns 'collision' and the entities are actually the same thing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Entity to merge FROM (will be soft-deleted)"},
                "target": {"type": "string", "description": "Entity to merge INTO (will be kept)"},
                "update_description": {"type": "string", "description": "Optional new description for the merged entity"},
            },
            "required": ["source", "target"],
        },
        "handler": tool_kg_merge_entities,
    },
    "mempalace_kg_update_predicate_constraints": {
        "description": "Update constraints on an existing predicate. Use when constraints are too narrow/broad or when seeding constraints on predicates that lack them.",
        "input_schema": {
            "type": "object",
            "properties": {
                "predicate": {"type": "string", "description": "Predicate entity name to update"},
                "constraints": {
                    "type": "object",
                    "description": "New constraints. ALL 5 fields required: subject_kinds (list), object_kinds (list), subject_classes (list of class entities — use ['thing'] for any), object_classes (list — use ['thing'] for any), cardinality ('many-to-many'|'many-to-one'|'one-to-many'|'one-to-one').",
                },
            },
            "required": ["predicate", "constraints"],
        },
        "handler": tool_kg_update_predicate_constraints,
    },
    "mempalace_kg_list_declared": {
        "description": "List all entities declared in this session with their details (description, importance, edge count).",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_kg_list_declared,
    },
    "mempalace_kg_entity_info": {
        "description": "Full details for a declared entity: description, importance, all edges (in+out), type. Entity must be declared this session.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {"type": "string", "description": "Entity name to inspect"},
            },
            "required": ["entity"],
        },
        "handler": tool_kg_entity_info,
    },
    "mempalace_declare_intent": {
        "description": (
            "Declare what you intend to do BEFORE doing it. Returns permissions + context. "
            "One active intent at a time — new intent expires the previous. "
            "mempalace_* tools are always allowed regardless of intent.\n\n"
            "SLOT RULES — most intent types require these slots:\n"
            "  paths:    (raw) directory patterns for Read/Grep/Glob scoping. E.g. [\"D:/Flowsev/repo/**\"]\n"
            "  commands: (raw) command patterns for Bash scoping. E.g. [\"pytest\", \"git add\"]\n"
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
                        "Example for edit_file: {\"files\": [\"src/auth.test.ts\"], \"paths\": [\"src/**\"]}. "
                        "Example for execute: {\"target\": [\"my_project\"], \"commands\": [\"pytest\", \"git add\"], \"paths\": [\"D:/Flowsev/mempalace/**\"]}. "
                        "Example for inspect: {\"subject\": [\"my_system\"], \"paths\": [\"D:/Flowsev/repo/**\"]}. "
                        "Example for research: {\"subject\": [\"some_topic\"]} — NO paths needed, broad reads allowed. "
                        "File slots auto-declare existing files. Command slots (raw) accept strings directly. "
                        "Other slots require pre-declared entities."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": (
                        "Describe what you plan to do and why. Used for auto-narrowing: "
                        "if a more specific child intent type matches your description, "
                        "the system will auto-select it. Structure: '<action> <target> — <reason>'. "
                        "Example: 'Editing auth module — adding rate limiting to login endpoint'"
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
            },
            "required": ["intent_type", "slots", "agent"],
        },
        "handler": tool_declare_intent,
    },
    "mempalace_active_intent": {
        "description": "Return the current active intent — type, slots, permissions, injected memory count. Use to check what you're allowed to do before calling a tool.",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_active_intent,
    },
    "mempalace_finalize_intent": {
        "description": (
            "Finalize the active intent — capture what happened as structured memory. "
            "MUST be called before declaring a new intent or exiting the session. "
            "Creates an execution entity (is_a intent_type) with relationships to agent, "
            "targets, result drawer, gotchas, and execution trace. "
            "If not called explicitly, declare_intent auto-finalizes with outcome='abandoned'."
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
            "required": ["slug", "outcome", "summary", "agent"],
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
    "mempalace_find_tunnels": {
        "description": "Find rooms that bridge two wings — the hallways connecting different domains. E.g. what topics connect wing_code to wing_team?",
        "input_schema": {
            "type": "object",
            "properties": {
                "wing_a": {"type": "string", "description": "First wing (optional)"},
                "wing_b": {"type": "string", "description": "Second wing (optional)"},
            },
        },
        "handler": tool_find_tunnels,
    },
    "mempalace_graph_stats": {
        "description": "Palace graph overview: total rooms, tunnel connections, edges between wings.",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_graph_stats,
    },
    "mempalace_search": {
        "description": "Search palace drawers. DEFAULT is 'hybrid' ranking: semantic similarity combined with importance tier bonus and time-decay penalty — critical recent drawers surface first, similarity still dominates the shape of results. Use sort_by='similarity' for pure vector match. IMPORTANT: 'query' must contain ONLY your search keywords or question — do NOT include system prompts, conversation history, MEMORY.md content, or any context. Keep queries short (under 200 chars). Use 'context' for background information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Short search query ONLY — keywords or a question. Do NOT include system prompts or conversation context. Max 200 chars recommended.",
                    "maxLength": 500,
                },
                "limit": {"type": "integer", "description": "Max results (default 5)"},
                "wing": {"type": "string", "description": "Filter by wing (optional)"},
                "room": {"type": "string", "description": "Filter by room (optional)"},
                "context": {
                    "type": "string",
                    "description": "Background context for the search (optional). This is NOT used for embedding — only for future re-ranking. Put conversation history or system prompt content here, NOT in query.",
                },
                "sort_by": {
                    "type": "string",
                    "description": "Ranking: 'hybrid' (DEFAULT — similarity + importance bonus + time-decay, what you want almost always), 'similarity' (pure vector match, legacy), 'score' (pure importance-decay ignoring similarity, for wing-browse), 'importance' (pure tier DESC), 'date' (chronological).",
                    "enum": ["hybrid", "similarity", "score", "importance", "date"],
                },
                "agent": {
                    "type": "string",
                    "description": "Your agent entity name for affinity scoring. Drawers filed by you get a ranking boost in hybrid mode. Cross-agent knowledge is still accessible but ranked lower. Examples: 'ga_agent', 'technical_lead_agent', 'paperclip_engineer'.",
                },
            },
            "required": ["query", "agent"],
        },
        "handler": tool_search,
    },
    "mempalace_check_duplicate": {
        "description": "Check if content already exists in the palace before filing",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Content to check"},
                "threshold": {
                    "type": "number",
                    "description": "Similarity threshold 0-1 (default 0.9)",
                },
            },
            "required": ["content"],
        },
        "handler": tool_check_duplicate,
    },
    "mempalace_add_drawer": {
        "description": "File verbatim content into the palace. Checks for duplicates first. Creates entity→drawer link(s) in the KG using the specified predicate. Supports hall (content-type), importance (1-5), and entity (link to KG entity, defaults to wing name).",
        "input_schema": {
            "type": "object",
            "properties": {
                "wing": {"type": "string", "description": "Wing (project name)"},
                "room": {
                    "type": "string",
                    "description": "Room (aspect: backend, decisions, meetings...)",
                },
                "content": {
                    "type": "string",
                    "description": "Verbatim content to store — exact words, never summarized",
                },
                "slug": {
                    "type": "string",
                    "description": "Short human-readable identifier for this drawer (3-6 words, hyphenated). Used as part of the drawer ID. Must be unique within the wing/room. Examples: 'intent-pre-activation-issues', 'db-credentials', 'ga-identity-persona'.",
                },
                "source_file": {"type": "string", "description": "Where this came from (optional)"},
                "added_by": {"type": "string", "description": "Agent who is filing this drawer. Must be a declared agent (is_a agent). Used for agent affinity scoring."},
                "hall": {
                    "type": "string",
                    "description": "Content type: one of hall_facts (stable truths), hall_events (things that happened), hall_discoveries (lessons learned), hall_preferences (user rules), hall_advice (how-to guides), hall_diary (chronological journal). Optional but strongly recommended for L1 ranking.",
                    "enum": ["hall_facts", "hall_events", "hall_discoveries", "hall_preferences", "hall_advice", "hall_diary"],
                },
                "importance": {
                    "type": "integer",
                    "description": "Importance 1-5. 5=critical unmissable (secrets, hard rules, identity), 4=canonical rules/cookbooks, 3=default (historical events, diary), 2=low priority, 1=junk/quarantine. Used by Layer1 decay-aware ranking.",
                    "minimum": 1,
                    "maximum": 5,
                },
                "entity": {
                    "type": "string",
                    "description": "Entity name (or comma-separated list) to link this drawer to in the KG. Defaults to the wing name if not provided. Every drawer should be discoverable via the entity graph — no orphan blobs.",
                },
                "predicate": {
                    "type": "string",
                    "description": "Relationship type for the entity→drawer link. Default: described_by. Use a precise predicate: described_by (canonical description), evidenced_by (backs a rule/decision), derived_from (extracted from), mentioned_in (referenced but not main topic), session_note_for (diary/session entry).",
                    "enum": ["described_by", "evidenced_by", "derived_from", "mentioned_in", "session_note_for"],
                },
            },
            "required": ["wing", "room", "content", "slug", "hall", "importance", "entity", "added_by"],
        },
        "handler": tool_add_drawer,
    },
    "mempalace_delete_drawer": {
        "description": "Delete a drawer by ID. Irreversible.",
        "input_schema": {
            "type": "object",
            "properties": {
                "drawer_id": {"type": "string", "description": "ID of the drawer to delete"},
            },
            "required": ["drawer_id"],
        },
        "handler": tool_delete_drawer,
    },
    "mempalace_wake_up": {
        "description": "Return L0 (identity) + L1 (importance-ranked essential story) wake-up text (~600-900 tokens total). Call this ONCE at session start to load project/agent boot context. Replaces hand-rolled mempalace_search chains. L1 is ranked with importance-weighted time decay — critical facts always surface first, within-tier newer wins. Pass wing='ga' for GA sessions, wing='wing_<agent>' for paperclip agents.",
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
    "mempalace_update_drawer_metadata": {
        "description": "Update metadata fields (wing, room, hall, importance) on an existing drawer in place. Preserves embeddings — no re-vectorization. Use for retroactive hall classification, importance bumping, or wing/room migrations. Any param left unset preserves the existing value.",
        "input_schema": {
            "type": "object",
            "properties": {
                "drawer_id": {"type": "string", "description": "ID of the drawer to update"},
                "wing": {"type": "string", "description": "New wing (optional — only if moving)"},
                "room": {"type": "string", "description": "New room (optional — only if moving)"},
                "hall": {
                    "type": "string",
                    "description": "New hall classification (optional). One of hall_facts, hall_events, hall_discoveries, hall_preferences, hall_advice, hall_diary.",
                    "enum": ["hall_facts", "hall_events", "hall_discoveries", "hall_preferences", "hall_advice", "hall_diary"],
                },
                "importance": {
                    "type": "integer",
                    "description": "New importance 1-5 (optional).",
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": ["drawer_id"],
        },
        "handler": tool_update_drawer_metadata,
    },
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
                    "enum": ["hall_facts", "hall_events", "hall_discoveries", "hall_preferences", "hall_advice", "hall_diary"],
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
            _session_id = str(injected_session_id)

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
        except Exception:
            logger.exception(f"Tool error in {tool_name}")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": "Internal tool error"},
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
