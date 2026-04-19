#!/usr/bin/env python3
"""
MemPalace MCP Server — read/write palace access for Claude Code
================================================================
Install: claude mcp add mempalace -- python -m mempalace.mcp_server [--palace /path/to/palace]

Tools (read):
  mempalace_kg_search       — unified 3-channel search over memories AND entities
  mempalace_kg_query        — structured edge lookup by exact entity name
  mempalace_kg_stats        — knowledge graph overview: entities, triples, relationship types

Tools (write):
  mempalace_kg_declare_entity — declare an entity (kind=entity/class/predicate/literal/record)
                                memory memories are first-class entities
  mempalace_kg_delete_entity — soft-delete an entity or memory (invalidates all edges)
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
import chromadb

from .knowledge_graph import KnowledgeGraph
from . import intent
from .server_state import ServerState
from .scoring import (
    hybrid_score as _hybrid_score_fn,
    adaptive_k,
    multi_channel_search,
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

_bootstrap_config = MempalaceConfig()
# BF1: KG file always lives inside the palace dir now. KnowledgeGraph()
# with no arg derives the path from MempalaceConfig().palace_path and
# migrates any legacy ~/.mempalace/knowledge_graph.sqlite3 in place on
# first run. Whether --palace was passed or not, the same code path runs.
_bootstrap_kg = KnowledgeGraph()

_STATE = ServerState(config=_bootstrap_config, kg=_bootstrap_kg)
del _bootstrap_config, _bootstrap_kg

# Wire intent module to this module so it can reach _STATE and other helpers.
intent.init(sys.modules[__name__])


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


def _get_client():
    """Return a singleton ChromaDB PersistentClient."""
    if _STATE.client_cache is None:
        _STATE.client_cache = chromadb.PersistentClient(path=_STATE.config.palace_path)
    return _STATE.client_cache


# Cosine is the ONLY supported distance metric across mempalace.
# MaxSim/ColBERT math assumes 1-distance = cosine_similarity and our
# retrieval scoring depends on [-1, +1] similarity semantics. Explicitly
# pinning the hnsw:space prevents a future ChromaDB default change (or
# a collection created by an older tool) from silently shifting math
# underneath us.
_CHROMA_METADATA = {"hnsw:space": "cosine"}


def _get_collection(create=False):
    """Return the ChromaDB collection, caching the client between calls."""
    try:
        client = _get_client()
        if create:
            _STATE.collection_cache = client.get_or_create_collection(
                _STATE.config.collection_name, metadata=_CHROMA_METADATA
            )
        elif _STATE.collection_cache is None:
            _STATE.collection_cache = client.get_collection(_STATE.config.collection_name)
        return _STATE.collection_cache
    except Exception:
        return None


def _no_palace():
    return {
        "error": "No palace found",
        "hint": "Run: mempalace init <dir> && mempalace mine <dir>",
    }


# ==================== READ TOOLS ====================


PALACE_PROTOCOL = """MemPalace Protocol — behavioral rules only. The system enforces the rest
(intent declaration, entity declaration, tool permissions, predicate constraints).

ON START:
  Call mempalace_wake_up. Read this protocol, the text (identity + rules),
  and declared (entities, predicates, intent types with their tools).

BEFORE ACTING ON ANY FACT:
  Use kg_query for exact entity-ID lookups when you know the name.
  Use kg_search for fuzzy discovery — it searches BOTH memories (prose)
  and entities (KG nodes) in one call, with graph expansion. Never guess.

WHEN HITTING A BLOCKER:
  FIRST search mempalace for known solutions — gotchas, lessons-learned,
  past executions that solved similar problems. Only report a blocker to
  the user if memory has no answer. When you solve a new problem, persist
  the solution (memory + KG triple) so future sessions find it.

WHEN FILING DRAWERS:
  - Choose the most accurate predicate for the entity link from the
    declared predicates list (returned by wake_up).
  - Then extract at least one KG triple from the content (twin pattern).
    Memory alone = semantic search only. KG triple = fast entity lookup.
  - Duplicate detection is automatic — if similar memories exist, conflicts
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

COMPLETION DISCIPLINE:
  There is no "wrapping the session" while pending work remains. Do NOT offer
  to continue "in a later session", do NOT summarize and stop, do NOT ask
  "should I keep going?" or "want me to pick this up next time?". If the
  TodoWrite list has pending items, DO THEM — 100%. The user does not care
  about session boundaries or context limits. Finish the work.
  Only pause when a tool call genuinely needs the user's answer (ambiguous
  requirement, missing credential, destructive action requiring consent).
  Everything else is your job to complete without asking.

AT SESSION END (only when all pending work is actually done):
  First, finalize the active intent (mempalace_finalize_intent).
  Then persist new knowledge using the twin pattern:
  - Decisions, rules, discoveries, gotchas -> memory + KG triple(s).
  - Changed facts -> kg_invalidate old + kg_add new.
  - New entities encountered -> kg_declare_entity if not yet declared.
  Don't just diary them — diary is a temporal log, KG + memories are
  durable knowledge that future sessions can query structurally.
  Then call diary_write — but keep it CONCISE and NON-REDUNDANT:
  - Write readable prose.
  - Delta only: what changed SINCE the last diary entry (not a full restatement).
  - Focus on: decisions made with user, big-picture status, pending items.
  - Do NOT repeat: commits, gotchas, learnings, features (already in intent results).
  - The diary is a high-level narrative, not a detailed log."""


def _hybrid_score(
    similarity: float,
    importance: float,
    date_added_iso: str,
    agent_match: bool = False,
    last_relevant_iso: str = None,
    session_match: bool = False,
    intent_type_match: bool = False,
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
        session_match=session_match,
        intent_type_match=intent_type_match,
    )


# tool_search removed: merged into tool_kg_search, which now searches
# BOTH memories and entities in a single cross-collection RRF. The "palace is
# a graph" unification — one search tool over all memory.


# tool_check_duplicate removed: dedup is now embedded in
# _add_memory_internal (called by kg_declare_entity kind='record').
# The standalone tool was already removed from the MCP registry;
# this deletes the orphaned function.


# ==================== WRITE TOOLS ====================


VALID_CONTENT_TYPES = {
    "fact",
    "event",
    "discovery",
    "preference",
    "advice",
    "diary",
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
    "record",  # a stored prose record — full text in ChromaDB, metadata in SQLite
}

# kind='memory' is GONE. The one-pass migration at startup
# rewrites existing metadata; the alias is removed so callers
# get a clear error instead of silent normalization. "memory" is the
# palace-level concept; "record" is the record-type kind.


def _validate_kind(kind):
    """Validate entity kind (ontological role). REQUIRED — no default."""
    if kind is None:
        raise ValueError(
            "kind is REQUIRED. Must be one of: 'entity' (concrete thing), "
            "'predicate' (relationship type), 'class' (category definition), "
            "'literal' (raw value), or 'record' (prose record — requires "
            "slug + content + added_by). You must explicitly choose the "
            "ontological role."
        )
    if kind == "memory":
        raise ValueError(
            "kind='memory' was renamed to 'record' in P6.2. Use kind='record'. "
            "The word 'memory' is reserved for the palace-level concept."
        )
    if kind not in VALID_KINDS:
        raise ValueError(
            f"kind must be one of {sorted(VALID_KINDS)} (got {kind!r}). "
            f"entity=concrete thing (default), predicate=relationship type, "
            f"class=category/type definition, literal=raw value, "
            f"record=prose record with slug + content + added_by. "
            f"Domain types (system, person, project, etc.) are NOT kinds — "
            f"they are class-kind entities linked via is_a edges."
        )
    return kind


def _validate_content_type(content_type):
    """Validate a content_type value. Returns string or raises ValueError."""
    if content_type is None:
        return None
    if content_type not in VALID_CONTENT_TYPES:
        raise ValueError(
            f"content_type must be one of {sorted(VALID_CONTENT_TYPES)} (got {content_type!r}). "
            f"fact=stable truths, event=things that happened, "
            f"discovery=lessons learned, preference=user rules, "
            f"advice=how-to guides, diary=chronological journal."
        )
    return content_type


def _slugify(text: str, max_length: int = 50) -> str:
    """Canonical slug for memory/diary/entity identifiers.

    Single source of truth for identifier normalization. Delegates to
    ``normalize_entity_name`` so every stored identifier uses the same
    separator convention (underscore) across Chroma IDs, SQLite entity
    IDs, and KG triple subjects/objects. The previous implementation
    emitted hyphens, which collided with every downstream callsite that
    re-normalized to underscores and then looked up the hyphenated ID —
    yielding silent Chroma misses (see A7, 9ecf234). DRY it here, fix it
    forever.
    """
    from .knowledge_graph import normalize_entity_name

    slug = normalize_entity_name(text)
    if slug == "unknown":
        return ""
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("_")
    return slug


_ENRICHMENT_SIM_THRESHOLD = 0.50
_ENRICHMENT_USEFULNESS_FLOOR = -0.3
_ENRICHMENT_MAX_SUGGESTIONS = 5
_REJECTION_REASON_OVERLAP_THRESHOLD = 0.35


def _tokenize(text: str) -> set:
    """Cheap lowercase tokenizer for Jaccard overlap.

    Splits on non-word characters, drops empty tokens and 1-char filler.
    Intentionally simple — we just need a stable bag of content words to
    compare two short strings.
    """
    if not text:
        return set()
    import re

    return {t for t in re.split(r"\W+", text.lower()) if t and len(t) > 1}


def _consume_matching_enrichment(
    sub_normalized: str, pred_normalized: str, obj_normalized: str
) -> dict:
    """Phase N5: auto-resolve a pending enrichment when the agent kg_adds
    an edge that matches its (from_entity, to_entity) pair.

    Records positive edge feedback on the predicate the agent actually
    chose (so the feedback system learns which predicates the agent uses
    for which enrichment shapes), removes the enrichment from the
    ``_STATE.pending_enrichments`` list, and re-persists the active
    intent so the PreToolUse hook sees the updated state on the next
    tool call. Returns the consumed enrichment dict, or {} if none
    matched.

    Without this step, every kg_add that implements a proposed edge
    leaves the enrichment marooned in pending state — declare_intent
    then refuses to proceed with "N graph enrichment tasks pending"
    even though the agent has already expressed the accept decision.
    """
    pending = _STATE.pending_enrichments
    if not pending:
        return {}
    from .knowledge_graph import normalize_entity_name

    match = None
    remaining = []
    for enr in pending:
        raw_from = enr.get("from_entity", "")
        raw_to = enr.get("to_entity", "")
        from_id = normalize_entity_name(raw_from) if raw_from else ""
        to_id = normalize_entity_name(raw_to) if raw_to else ""
        if match is None and from_id == sub_normalized and to_id == obj_normalized:
            match = enr
            continue  # drop from remaining
        remaining.append(enr)
    if match is None:
        return {}

    # Record positive feedback on the predicate the agent chose. Reusing
    # the enrichment's original 'reason' as context_keywords lets MaxSim
    # retrieve this as precedent on future similar enrichments.
    intent_type = _STATE.active_intent.get("intent_type", "") if _STATE.active_intent else ""
    try:
        _STATE.kg.record_edge_feedback(
            sub_normalized,
            pred_normalized,
            obj_normalized,
            intent_type,
            useful=True,
            context_keywords=(match.get("reason") or "")[:200],
        )
    except Exception:
        pass  # Non-fatal — feedback is best-effort

    _STATE.pending_enrichments = remaining or None
    try:
        intent._persist_active_intent()
    except Exception:
        pass
    return match


def _rejection_suppresses_enrichment(
    subject_id: str, object_id: str, candidate_text: str, kg
) -> bool:
    """B2b: return True if a past rejection reason semantically overlaps with
    the new enrichment's content. Uses a cheap Jaccard token overlap on the
    candidate text vs each past rejection reason; the candidate is built from
    subject + object + candidate_text so a rejection reason mentioning either
    end of the new pair counts.

    Past rejections are scoped to the suggested_link predicate (the only
    predicate edge_traversal_feedback writes via resolve_enrichments).

    Returns False on any error so a degraded similarity check never blocks
    legitimate enrichments.
    """
    try:
        rows = kg.get_recent_rejection_reasons(limit=200) if kg else []
    except Exception:
        return False
    if not rows:
        return False
    cand_tokens = _tokenize(f"{subject_id} {object_id} {candidate_text or ''}")
    if len(cand_tokens) < 3:
        return False  # Too little content to match meaningfully
    for past_subj, past_obj, past_reason in rows:
        past_tokens = _tokenize(f"{past_subj or ''} {past_obj or ''} {past_reason or ''}")
        if len(past_tokens) < 3:
            continue
        union = cand_tokens | past_tokens
        if not union:
            continue
        overlap = len(cand_tokens & past_tokens) / len(union)
        if overlap >= _REJECTION_REASON_OVERLAP_THRESHOLD:
            return True
    return False


def _past_resolution_hint(conflicts: list) -> str:
    """B1b: build a short suffix for conflicts_prompt summarizing prior decisions.

    When any conflict in the batch carries a `past_resolution` field (attached
    at detection time by `get_past_conflict_resolution`), render a compact
    hint so the agent sees their prior decisions without having to read the
    full conflicts array. Empty string when no past decisions exist \u2014 no
    change to the prompt in that case.
    """
    past_lines = []
    for c in conflicts or []:
        past = c.get("past_resolution")
        if not past:
            continue
        when = (past.get("when") or "")[:10]  # YYYY-MM-DD
        reason = (past.get("reason") or "").strip()
        if len(reason) > 100:
            reason = reason[:97] + "..."
        action = past.get("action") or "?"
        past_lines.append(
            f"  \u2022 {c.get('id', '?')}: last time you chose '{action}' ({when}) \u2014 {reason}"
        )
    if not past_lines:
        return ""
    return (
        "\n\nPast resolutions on matching conflicts (use as guidance, still your call):\n"
        + "\n".join(past_lines)
    )


def _detect_suggested_links(
    source_id: str,
    query_text: str,
    excluded_ids: set = None,
) -> list:
    """Find related entities worth suggesting as graph links.

    Queries the entity collection for semantic neighbors of query_text,
    filters out the source, already-excluded entities, and pairs the agent
    has rejected in the past (via record_edge_feedback on the
    'suggested_link' predicate). Returns up to _ENRICHMENT_MAX_SUGGESTIONS
    dicts with keys: entity_id, similarity, description.
    """
    suggestions = []
    excluded = excluded_ids or set()
    intent_type = _STATE.active_intent.get("intent_type", "") if _STATE.active_intent else ""
    try:
        ecol = _get_entity_collection(create=False)
        if not ecol or ecol.count() == 0:
            return []
        n = min(ecol.count(), 20)
        results = ecol.query(
            query_texts=[query_text[:500]],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []
    if not (results.get("ids") and results["ids"][0]):
        return []
    seen = set()
    for i, raw_id in enumerate(results["ids"][0]):
        meta_r = results["metadatas"][0][i] or {}
        logical_id = meta_r.get("entity_id") or raw_id
        if logical_id == source_id or logical_id in excluded or logical_id in seen:
            continue
        seen.add(logical_id)
        kind = meta_r.get("kind", "entity")
        if kind not in ("entity", "class"):
            continue
        dist = results["distances"][0][i]
        sim = round(max(0.0, 1.0 - dist), 3)
        if sim < _ENRICHMENT_SIM_THRESHOLD:
            continue
        try:
            usefulness = _STATE.kg.get_edge_usefulness(
                source_id, "suggested_link", logical_id, intent_type=intent_type
            )
            if usefulness < _ENRICHMENT_USEFULNESS_FLOOR:
                continue  # Agent has rejected this pair before — don't re-ask
        except Exception:
            pass
        desc = (results["documents"][0][i] or "")[:100]
        # B2b: suppress when a past rejection reason semantically overlaps,
        # even if THIS specific pair has no direct rejection history yet.
        # Catches the "rejected for one pair, similar new pair surfaces" case
        # the audit flagged. Cheap Jaccard on tokens — no embeddings.
        if _rejection_suppresses_enrichment(source_id, logical_id, desc, _STATE.kg):
            continue
        suggestions.append({"entity_id": logical_id, "similarity": sim, "description": desc})
        if len(suggestions) >= _ENRICHMENT_MAX_SUGGESTIONS:
            break
    return suggestions


def _add_memory_internal(  # noqa: C901
    content: str,
    slug: str,
    added_by: str = None,
    content_type: str = None,
    importance: int = None,
    entity: str = None,
    predicate: str = "described_by",
    context: dict = None,  # Context fingerprint for keywords + creation_context_id
    source_file: str = None,
):
    """File verbatim content as a flat record. Checks for duplicates first.

    ALL classification params are REQUIRED (no lazy defaults):
        slug: short human-readable identifier — REQUIRED. Used as part of the
              record ID. Must be unique per agent. Examples:
              'intent-pre-activation-issues', 'db-credentials', 'ga-identity'.
        content_type: one of fact, event, discovery, preference, advice, diary.
        importance: integer 1-5 — REQUIRED. 5=critical, 4=canonical,
                    3=default, 2=low, 1=junk.
        entity: entity name (or comma-separated list) — REQUIRED. Links this record
                to an entity in the KG. If not provided, the record is unlinked.
        predicate: relationship type for the entity→record link. Default: described_by.

    Note: date_added is always set to the current time. Diary records
    (via diary_write) are exempt from the entity/slug requirement.
    """
    try:
        content = sanitize_content(content)
        content_type = _validate_content_type(content_type)
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
        if _STATE.kg:
            agent_edges = _STATE.kg.query_entity(agent_id, direction="outgoing")
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
    except Exception:
        pass  # graceful fallback if KG unavailable

    if not slug or not slug.strip():
        return {
            "success": False,
            "error": "slug is required. Provide a short human-readable identifier (e.g. 'intent-pre-activation-issues').",
        }

    normalized_slug = _slugify(slug)
    if not normalized_slug:
        return {
            "success": False,
            "error": f"slug '{slug}' normalizes to empty. Use alphanumeric words separated by underscores or hyphens.",
        }

    col = _get_collection(create=True)
    if not col:
        return _no_palace()

    from .knowledge_graph import normalize_entity_name as _norm_eid

    agent_slug = _norm_eid(added_by) if added_by else "unknown"
    memory_id = f"record_{agent_slug}_{normalized_slug}"

    # Uniqueness check — slug collision returns existing record info
    try:
        existing = col.get(ids=[memory_id], include=["documents", "metadatas"])
        if existing and existing["ids"]:
            return {
                "success": False,
                "error": f"Slug '{normalized_slug}' already exists for agent {added_by}.",
                "existing_memory": {
                    "memory_id": memory_id,
                    "content_preview": (existing["documents"][0] or "")[:200],
                    "metadata": existing["metadatas"][0],
                },
                "hint": "Choose a different slug, or use kg_update_entity(entity=memory_id, ...) to modify the existing record's metadata.",
            }
    except Exception:
        pass

    _wal_log(
        "add_memory",
        {
            "memory_id": memory_id,
            "added_by": added_by,
            "content_length": len(content),
            "content_preview": content[:200],
            "content_type": content_type,
            "importance": importance,
        },
    )

    now_iso = datetime.now().isoformat()
    meta = {
        "source_file": source_file or "",
        "chunk_index": 0,
        "added_by": added_by,
        "filed_at": now_iso,
        "date_added": now_iso,
        "last_relevant_at": now_iso,
    }
    if content_type is not None:
        meta["content_type"] = content_type
    if importance is not None:
        meta["importance"] = importance

    # provenance auto-injection. Every record carries session_id
    # and intent_id from the active session/intent. System-injected, not
    # agent-provided. Queryable via Chroma where filters for session-
    # scoped retrieval; also written to SQLite (migration 009).
    if _STATE.session_id:
        meta["session_id"] = _STATE.session_id
    if _STATE.active_intent and isinstance(_STATE.active_intent, dict):
        meta["intent_id"] = _STATE.active_intent.get("intent_id", "")

    try:
        col.upsert(
            ids=[memory_id],
            documents=[content],
            metadatas=[meta],
        )
        logger.info(f"Filed record: {memory_id} content_type={content_type} imp={importance}")

        # Register record as a first-class graph node in SQLite.
        try:
            _STATE.kg.add_entity(
                memory_id,
                kind="record",
                description=content[:200],
                importance=importance or 3,
                properties={
                    "content_type": content_type or "",
                    "added_by": added_by or "",
                },
                session_id=_STATE.session_id or "",
                intent_id=(
                    _STATE.active_intent.get("intent_id", "") if _STATE.active_intent else ""
                ),
            )
        except Exception:
            pass  # Non-fatal — record exists in ChromaDB regardless

        # ── Context fingerprint: keywords → entity_keywords table,
        # view vectors → feedback_contexts collection, context_id → entities row.
        # context is optional here so legacy intent.py callers (which still pass
        # synthetic kwargs) keep working — when present, full Context wiring engages.
        if context:
            try:
                _STATE.kg.add_entity_keywords(memory_id, context.get("keywords") or [])
                cid = persist_context(context, prefix="memory")
                if cid:
                    _STATE.kg.set_entity_creation_context(memory_id, cid)
            except Exception:
                pass  # Non-fatal

        # Create entity→memory link(s) using the specified predicate
        VALID_MEMORY_PREDICATES = {
            "described_by",
            "evidenced_by",
            "derived_from",
            "mentioned_in",
            "session_note_for",
        }
        link_predicate = predicate if predicate in VALID_MEMORY_PREDICATES else "described_by"

        linked_entities = []
        entity_names = []
        if entity:
            # Support comma-separated list
            entity_names = [e.strip() for e in entity.split(",") if e.strip()]

        from .knowledge_graph import normalize_entity_name

        for ename in entity_names:
            eid = normalize_entity_name(ename)
            # Only link to entities that already exist — don't auto-create junk stubs
            existing_entity = _STATE.kg.get_entity(eid)
            if not existing_entity:
                # Entity doesn't exist — skip the link. Agent should declare entities
                # via kg_declare_entity before referencing them in memories.
                continue
            try:
                _STATE.kg.add_triple(eid, link_predicate, memory_id)
                linked_entities.append(eid)
            except Exception:
                pass  # Non-fatal: memory exists, linking failed

        # ── Suggest related entities for linking (graph enrichment) ──
        # Detector respects past feedback: pairs the agent has rejected are
        # filtered out via get_edge_usefulness on the 'suggested_link' predicate.
        suggested_links = _detect_suggested_links(
            source_id=memory_id,
            query_text=content,
            excluded_ids=set(linked_entities),
        )

        result = {
            "success": True,
            "memory_id": memory_id,
            "content_type": content_type,
            "importance": importance,
            "linked_entities": linked_entities,
        }
        if suggested_links:
            # Store as pending enrichments — blocks tools until agent creates edges or rejects
            enrichments = []
            for sl in suggested_links:
                eid = sl["entity_id"]
                enrichment_id = f"enrich_{memory_id}_{eid}"
                enrichments.append(
                    {
                        "id": enrichment_id,
                        "reason": f"Memory '{memory_id}' is related to entity '{eid}' (similarity: {sl['similarity']})",
                        "from_entity": memory_id,
                        "to_entity": eid,
                        "similarity": sl["similarity"],
                        "to_description": sl.get("description", "")[:100],
                    }
                )
            _STATE.pending_enrichments = enrichments
            from . import intent  # noqa: F811

            intent._persist_active_intent()
            result["suggested_links"] = suggested_links
            result["enrichments_count"] = len(enrichments)
            result["enrichments_prompt"] = (
                f"{len(enrichments)} graph enrichment tasks pending. "
                "For each: call kg_add(subject, predicate, object) to create the edge "
                "with a predicate YOU choose from the declared predicates (see wake_up), "
                "then call mempalace_resolve_enrichments to mark done. Or reject with reason."
            )

        # ── Memory duplicate detection ──
        try:
            dup_results = col.query(
                query_texts=[content[:500]],
                n_results=5,
                include=["documents", "distances"],
            )
            if dup_results["ids"] and dup_results["ids"][0]:
                dup_conflicts = []
                for i, did in enumerate(dup_results["ids"][0]):
                    if did == memory_id:
                        continue  # Skip self
                    dist = dup_results["distances"][0][i]
                    sim = round(max(0.0, 1.0 - dist), 3)
                    if sim < 0.85:
                        continue  # Not similar enough
                    conflict_id = f"conflict_memory_{memory_id}_{did}"
                    preview = (dup_results["documents"][0][i] or "")[:150]
                    past = None
                    try:
                        past = _STATE.kg.get_past_conflict_resolution(
                            did, memory_id, "memory_duplicate"
                        )
                    except Exception:
                        pass
                    conflict_entry = {
                        "id": conflict_id,
                        "conflict_type": "memory_duplicate",
                        "reason": f"Similar memory found (similarity: {sim})",
                        "existing_id": did,
                        "existing_preview": preview,
                        "new_id": memory_id,
                        "similarity": sim,
                    }
                    if past:
                        conflict_entry["past_resolution"] = past
                    dup_conflicts.append(conflict_entry)
                if dup_conflicts:
                    _STATE.pending_conflicts = dup_conflicts
                    from . import intent

                    intent._persist_active_intent()
                    result["conflicts"] = dup_conflicts
                    past_hint = _past_resolution_hint(dup_conflicts)
                    result["conflicts_prompt"] = (
                        f"{len(dup_conflicts)} similar memory(s) found. "
                        f"Call mempalace_resolve_conflicts: merge, keep, or skip." + past_hint
                    )
        except Exception:
            pass

        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_kg_delete_entity(entity_id: str, agent: str = None):
    """Delete an entity (memory or KG node) and invalidate every edge touching it.

    Works for both memories (ids starting with 'drawer_' or 'diary_' —
    historical prefixes kept for DB compatibility) and KG entities.
    Invalidates all current edges where the target is subject or object
    (soft-delete, temporal audit trail preserved), then removes its
    description from the appropriate Chroma collection.

    Use this when an entity is truly obsolete (superseded concept, stale memory,
    deleted person). For updating a single fact (one edge becomes untrue while
    the entity itself remains valid), use kg_invalidate(subject, predicate,
    object) on the specific triple instead.
    """
    sid_err = _require_sid(action="kg_delete_entity")
    if sid_err:
        return sid_err
    # mandatory agent attribution.
    agent_err = _require_agent(agent, action="kg_delete_entity")
    if agent_err:
        return agent_err
    if not entity_id or not isinstance(entity_id, str):
        return {"success": False, "error": "entity_id is required (string)."}

    # Determine which collection to target: records live in the record
    # collection; everything else in the entity collection. The 'record_' /
    # 'diary_' id prefixes route to the record collection.
    is_record_id = entity_id.startswith(("record_", "diary_"))
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
        existing = col.get(ids=[entity_id])
    except Exception as e:
        return {"success": False, "error": f"lookup failed: {e}"}
    if not existing or not existing.get("ids"):
        return {
            "success": False,
            "error": f"Not found in {'memories' if is_record_id else 'entities'}: {entity_id}",
        }

    deleted_content = (existing.get("documents") or [""])[0] or ""
    deleted_meta = (existing.get("metadatas") or [{}])[0] or {}

    # Invalidate every current edge involving this entity (both directions).
    invalidated = 0
    try:
        edges = _STATE.kg.query_entity(entity_id, direction="both") or []
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
            "entity_id": entity_id,
            "collection": "memory" if is_record_id else "entity",
            "edges_invalidated": invalidated,
            "deleted_meta": deleted_meta,
            "content_preview": deleted_content[:200],
        },
    )

    try:
        col.delete(ids=[entity_id])
        logger.info(
            f"Deleted {'memory' if is_record_id else 'entity'}: {entity_id} ({invalidated} edges invalidated)"
        )
        return {
            "success": True,
            "entity_id": entity_id,
            "source": "memory" if is_record_id else "entity",
            "edges_invalidated": invalidated,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_wake_up(agent: str = None):
    """Boot context for a session. Call ONCE at start.

    Returns protocol (behavioral rules), text (identity + top memories),
    and declared (compact summary of auto-declared entities).

    Args:
        agent: Agent identity (required). Used for affinity scoring in L1.
    """
    try:
        from .layers import MemoryStack
    except Exception as e:
        return {"success": False, "error": f"layers module unavailable: {e}"}

    try:
        # Auto-bootstrap agent if it doesn't exist (cold restart safe)
        if agent:
            from .knowledge_graph import normalize_entity_name as _norm

            _agent_id = _norm(agent)
            _agent_ent = _STATE.kg.get_entity(_agent_id)
            if not _agent_ent:
                _STATE.kg.add_entity(
                    _agent_id,
                    kind="entity",
                    description=f"Agent: {agent}",
                    importance=4,
                )
                _STATE.kg.add_triple(_agent_id, "is_a", "agent")
                _sync_entity_to_chromadb(_agent_id, agent, f"Agent: {agent}", "entity", 4)
                _STATE.declared_entities.add(_agent_id)

        stack = MemoryStack()
        text = stack.wake_up(agent=agent)
        token_estimate = len(text) // 4
        from .knowledge_graph import normalize_entity_name

        # 1. Predicates — declare + collect names
        predicates = _STATE.kg.list_entities(status="active", kind="predicate")
        pred_names = []
        for p in predicates:
            _STATE.declared_entities.add(p["id"])
            pred_names.append(p["id"])

        # 2. Classes — declare + collect names
        classes = _STATE.kg.list_entities(status="active", kind="class")
        class_names = []
        for c in classes:
            _STATE.declared_entities.add(c["id"])
            class_names.append(c["id"])

        # 3. Intent types — walk is-a tree, compact format
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

        # 4. Top entities (non-intent) — name[importance]
        entity_parts = []
        top_ents = [e for e in entities if e["id"] not in intent_type_ids][:20]
        for e in top_ents:
            _STATE.declared_entities.add(e["id"])
            entity_parts.append(e["id"] + "[" + str(e.get("importance", 3)) + "]")

        # Load learned scoring weights from feedback history
        try:
            from .scoring import set_learned_weights, DEFAULT_SEARCH_WEIGHTS

            learned = _STATE.kg.compute_learned_weights(DEFAULT_SEARCH_WEIGHTS)
            set_learned_weights(learned)
        except Exception:
            pass

        return {
            "success": True,
            "protocol": PALACE_PROTOCOL,
            "text": text,
            "estimated_tokens": token_estimate,
            "declared": {
                "predicates": ", ".join(sorted(pred_names)),
                "classes": ", ".join(sorted(class_names)),
                "intent_types": " | ".join(intent_parts),
                "entities": ", ".join(entity_parts),
                "count": len(_STATE.declared_entities),
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# tool_update_drawer_metadata removed: merged into tool_kg_update_entity.


# ==================== KNOWLEDGE GRAPH ====================


def tool_kg_query(entity: str, as_of: str = None, direction: str = "both"):
    """Query the knowledge graph for an entity's relationships.

    Supports batch queries: pass a comma-separated list of entity names
    to query multiple entities in one call. Returns results keyed by entity.
    """
    entities = [e.strip() for e in entity.split(",") if e.strip()]

    # Track queried entities for mandatory feedback enforcement
    if _STATE.active_intent and isinstance(_STATE.active_intent.get("accessed_memory_ids"), set):
        for ename in entities:
            _STATE.active_intent["accessed_memory_ids"].add(ename)

    if len(entities) == 1:
        # Single entity — original format for backwards compatibility
        results = _STATE.kg.query_entity(entities[0], as_of=as_of, direction=direction)
        return {"entity": entities[0], "as_of": as_of, "facts": results, "count": len(results)}

    # Batch query — return results keyed by entity name
    batch_results = {}
    total_count = 0
    for ename in entities:
        facts = _STATE.kg.query_entity(ename, as_of=as_of, direction=direction)
        batch_results[ename] = {"facts": facts, "count": len(facts)}
        total_count += len(facts)

    return {"entities": batch_results, "as_of": as_of, "total_count": total_count, "batch": True}


def tool_kg_search(  # noqa: C901
    context: dict = None,
    limit: int = 10,
    kind: str = None,
    sort_by: str = "hybrid",
    agent: str = None,
    time_window: dict = None,  # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    queries=None,  # LEGACY: rejected when context missing — see below
):
    """Unified search — records (prose) + entities (KG nodes) in one call.

    Speaks the unified Context object: queries drive Channel A (multi-view
    cosine), keywords drive Channel C (caller-provided exact terms — no
    auto-extraction), entities seed Channel B graph BFS when provided.
    Cross-collection RRF then competes record + entity hits head-to-head.

    Args:
        context: MANDATORY Context = {queries, keywords, entities?}.
        limit: Max results across records+entities (default 10; adaptive-K may trim).
        kind: Optional entity-only kind filter (excludes record results).
        sort_by: 'hybrid' (default) — RRF + hybrid_score tiebreaker. 'similarity'.
        agent: Agent name for affinity scoring.
        time_window: optional {start, end} date range (YYYY-MM-DD).
            SOFT DECAY: items inside the window get a scoring boost; items
            outside still appear but rank lower. NOT a hard filter — nothing
            is excluded. Use for temporal scoping ("what happened this week")
            without losing globally-important items that fall outside.
    """
    from .scoring import rrf_merge, validate_context as _validate_context
    from .knowledge_graph import normalize_entity_name

    # ── Reject the legacy `queries` path (Context mandatory) ──
    if context is None and queries is not None:
        return {
            "success": False,
            "error": (
                "`queries` is gone. Pass `context` instead — a dict "
                "with mandatory queries, keywords, and optional entities. Example:\n"
                '  context={"queries": ["auth rate limiting", "brute force hardening"], '
                '"keywords": ["auth", "rate-limit", "brute-force"]}'
            ),
        }

    # ── Validate Context (mandatory) ──
    clean_context, ctx_err = _validate_context(context)
    if ctx_err:
        return ctx_err
    query_views = clean_context["queries"]
    context_keywords = clean_context["keywords"]
    context_entities = clean_context["entities"]

    sanitized_views = [sanitize_query(v)["clean_query"] for v in query_views]
    sanitized_views = [v for v in sanitized_views if v]
    if not sanitized_views:
        return {"success": False, "error": "All queries were empty after sanitization."}

    # ── Source scoping: kind → entities only; otherwise search both ──
    search_memories = not bool(kind)
    search_entities = True

    try:
        # ── Run pipeline over selected collections ──
        all_lists = {}
        combined_meta = {}

        if search_memories:
            memory_col = _get_collection(create=False)
            memory_pipe = multi_channel_search(
                memory_col,
                sanitized_views,
                keywords=context_keywords,
                kg=_STATE.kg,
                added_by=agent,
                fetch_limit_per_view=max(limit * 3, 30),
                include_graph=False,
            )
            for name, lst in memory_pipe["ranked_lists"].items():
                all_lists[f"memory_{name}"] = lst
            for mid, info in memory_pipe["seen_meta"].items():
                combined_meta[mid] = {**info, "source": "memory"}

        if search_entities:
            entity_col = _get_entity_collection(create=False)
            # caller-provided context.entities become explicit graph seeds.
            # When omitted, multi_channel_search falls back to deriving seeds
            # from top cosine hits (current behaviour).
            seed_ids = (
                [normalize_entity_name(e) for e in context_entities] if context_entities else None
            )
            entity_pipe = multi_channel_search(
                entity_col,
                sanitized_views,
                keywords=context_keywords,
                kg=_STATE.kg,
                kind=kind,
                fetch_limit_per_view=max(limit * 3, 30),
                include_graph=True,
                seed_ids=seed_ids,
            )
            for name, lst in entity_pipe["ranked_lists"].items():
                all_lists[f"entity_{name}"] = lst
            for mid, info in entity_pipe["seen_meta"].items():
                # Entity overrides memory if the same id lives in both (shouldn't happen)
                combined_meta[mid] = {**info, "source": "entity"}

        # Triple verbalizations: query the dedicated mempalace_triples
        # collection alongside memories and entities so structured
        # knowledge surfaces as first-class search results. Skip when a
        # caller-pinned `kind` filter is active (kind targets entity
        # records, not triples).
        if not kind:
            try:
                from .knowledge_graph import _get_triple_collection

                triple_col = _get_triple_collection()
                if triple_col is not None and triple_col.count() > 0:
                    triple_pipe = multi_channel_search(
                        triple_col,
                        sanitized_views,
                        keywords=context_keywords,
                        kg=_STATE.kg,
                        fetch_limit_per_view=max(limit * 3, 30),
                        include_graph=False,
                    )
                    for name, lst in triple_pipe["ranked_lists"].items():
                        all_lists[f"triple_{name}"] = lst
                    for mid, info in triple_pipe["seen_meta"].items():
                        combined_meta[mid] = {**info, "source": "triple"}
            except Exception:
                pass  # triples are an optional enrichment of search results

        if not all_lists:
            return {"queries": sanitized_views, "results": [], "count": 0, "sort_by": sort_by}

        rrf_scores, _cm, _attr = rrf_merge(all_lists)

        # ── Relevance-feedback lookup (shared) ──
        # Continuous per-memory score in [-1, 1] from the intent_type's
        # found_useful / found_irrelevant edges weighted by confidence.
        feedback_scores = lookup_type_feedback(_STATE.active_intent, _STATE.kg)

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
                rel_fb = feedback_scores.get(mid, 0.0)
                # P6.7b provenance affinity
                sess_match = bool(_STATE.session_id and meta.get("session_id") == _STATE.session_id)
                final_score = _hybrid_score_fn(
                    similarity=similarity,
                    importance=importance,
                    date_iso=date_anchor,
                    agent_match=is_match,
                    last_relevant_iso=last_relevant,
                    relevance_feedback=rel_fb,
                    mode="search",
                    session_match=sess_match,
                )
                # time_window soft-decay boost. Items whose date_anchor
                # falls inside the window get an additive boost; items outside
                # are NOT excluded, just rank lower.
                if time_window and date_anchor:
                    tw_start = time_window.get("start", "")
                    tw_end = time_window.get("end", "")
                    if tw_start and tw_end and tw_start <= date_anchor <= tw_end:
                        final_score += 0.15  # inside window bonus
                    elif tw_start and date_anchor >= tw_start and not tw_end:
                        final_score += 0.15  # after start, no end
                    elif tw_end and date_anchor <= tw_end and not tw_start:
                        final_score += 0.15  # before end, no start
            else:
                final_score = similarity

            entry = {
                "id": mid,
                "source": source,
                "importance": importance,
                "similarity": similarity,
                "score": round(final_score, 4),
            }
            if source == "memory":
                entry["text"] = doc[:300]
            elif source == "triple":
                # Verbalized triple: present the natural-language statement plus the
                # underlying (subject, predicate, object) so callers can both read
                # the prose AND know the structured fact behind it.
                entry["statement"] = doc[:300]
                entry["subject"] = meta.get("subject", "")
                entry["predicate"] = meta.get("predicate", "")
                entry["object"] = meta.get("object", "")
                entry["confidence"] = meta.get("confidence", 1.0)
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
                edges = _STATE.kg.query_entity(entry["id"], direction="both")
                current_edges = [e for e in edges if e.get("current", True)]
                entry["edges"] = current_edges
                entry["edge_count"] = len(current_edges)

        # ── Track accessed items for mandatory feedback enforcement ──
        if _STATE.active_intent and isinstance(
            _STATE.active_intent.get("accessed_memory_ids"), set
        ):
            for entry in top:
                _STATE.active_intent["accessed_memory_ids"].add(entry["id"])

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
    context: dict = None,  # mandatory Context fingerprint for the edge
    agent: str = None,  # mandatory attribution
    valid_from: str = None,
    statement: str = None,  # natural-language verbalization for retrieval
):
    """Add a relationship to the knowledge graph (Context mandatory).

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

    `agent` is mandatory. Every write operation must be attributed
    to a declared agent (is_a agent); undeclared agents are rejected
    up-front with a declaration recipe.

    Call kg_declare_entity for subject/object entities, and
    kg_declare_entity with kind="predicate" for predicates.
    """
    from .knowledge_graph import normalize_entity_name
    from .scoring import validate_context

    # ── Validate Context (mandatory) ──
    clean_context, ctx_err = validate_context(context)
    if ctx_err:
        return ctx_err

    # ── mandatory agent attribution ──
    sid_err = _require_sid(action="kg_add")
    if sid_err:
        return sid_err
    agent_err = _require_agent(agent, action="kg_add")
    if agent_err:
        return agent_err

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
                        f"Options: (1) wrong edge — use a different subject, "
                        f"(2) wrong predicate — check kg_list_declared() for a better fit, "
                        f"(3) missing classification — add is_a edge for '{sub_normalized}', "
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
                        f"Options: (1) REPLACE — invalidate the old edge first, then add new, "
                        f"(2) MISTAKE — you meant a different predicate or entity, "
                        f"(3) EXPAND — change predicate cardinality to many-to-many."
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

    # ── Persist the edge's creation Context — view vectors → feedback_contexts,
    # context_id → triples.creation_context_id.
    edge_context_id = persist_context(clean_context, prefix="edge")

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
    triple_id = _STATE.kg.add_triple(
        sub_normalized,
        pred_normalized,
        obj_normalized,
        valid_from=valid_from,
        creation_context_id=edge_context_id,
        statement=statement,
    )

    # ── Implicit enrichment acceptance (N5) ──
    # If this edge connects a pair that was previously surfaced as a
    # pending enrichment, the agent's kg_add IS the accept signal. We:
    #   1) record positive edge feedback on the predicate the agent chose,
    #   2) remove the matching enrichment from pending state so it stops
    #      blocking the next declare_intent,
    #   3) re-persist active_intent so the hook sees the updated state.
    # Without this step rejected pairs had to be manually resolve_enrichments-
    # rejected even though the agent had already expressed the decision via
    # kg_add — a DRY miss in the feedback pipeline.
    _consumed_enrichment = _consume_matching_enrichment(
        sub_normalized, pred_normalized, obj_normalized
    )

    # ── Contradiction detection: find existing edges that may conflict ──
    conflicts = []
    try:
        # Skip is_a — those aren't factual contradictions
        if pred_normalized != "is_a":
            existing_edges = _STATE.kg.query_entity(sub_normalized, direction="outgoing")
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
        pass  # Non-fatal — contradiction detection is best-effort

    result = {
        "success": True,
        "triple_id": triple_id,
        "fact": f"{sub_normalized} -> {pred_normalized} -> {obj_normalized}",
    }

    if _consumed_enrichment:
        # Surface the auto-resolution so the caller sees that creating
        # this edge satisfied a pending enrichment proposal. The caller
        # does NOT need to call resolve_enrichments for this one.
        result["auto_resolved_enrichment"] = {
            "id": _consumed_enrichment.get("id"),
            "reason": _consumed_enrichment.get("reason"),
        }

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


def tool_kg_add_batch(edges: list, context: dict = None, agent: str = None):
    """Add multiple KG edges in one call (Context mandatory).

    Each edge: {subject, predicate, object, valid_from?, context?}.

    The TOP-LEVEL `context` is the shared default applied to every edge that
    doesn't carry its own — most batches add edges that all reflect the same
    'why' (a single agent decision), so one Context covers them. An edge can
    still override with its own `context` dict if needed. Validates each edge
    independently — partial success OK.

    `agent` is mandatory (same validation as kg_add). Applies to the
    whole batch; per-edge agent overrides are not supported (batches are
    single-author by design).
    """
    from .scoring import validate_context

    if not edges or not isinstance(edges, list):
        return {
            "success": False,
            "error": "edges must be a non-empty list of {subject, predicate, object} dicts.",
        }

    # ── agent validation up-front so we don't partially apply ──
    sid_err = _require_sid(action="kg_add_batch")
    if sid_err:
        return sid_err
    agent_err = _require_agent(agent, action="kg_add_batch")
    if agent_err:
        return agent_err

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
            agent=agent,
            valid_from=edge.get("valid_from"),
            statement=edge.get("statement"),
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


def tool_kg_invalidate(
    subject: str,
    predicate: str,
    object: str,
    ended: str = None,
    agent: str = None,
):
    """Mark a fact as no longer true (set end date). agent required."""
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
        return {
            "success": True,
            "fact": f"{subject} → {predicate} → {object}",
            "ended": ended or "today",
        }
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


def tool_kg_timeline(entity: str = None):
    """Get chronological timeline of facts, optionally for one entity."""
    results = _STATE.kg.timeline(entity)
    return {"entity": entity or "all", "timeline": results, "count": len(results)}


def tool_kg_stats():
    """Knowledge graph overview — entities, triples, relationship types."""
    stats = _STATE.kg.stats() or {}
    return stats


# ==================== ENTITY DECLARATION ====================

ENTITY_SIMILARITY_THRESHOLD = 0.85
# Legacy — mempalace_entities was absorbed into mempalace_records by the M1
# migration. Kept as a module constant only so the migration can look it
# up when scanning for legacy rows on startup.
ENTITY_COLLECTION_NAME = "mempalace_entities"

# Session-level declared entities (in-memory cache on _STATE, falls back to persistent KG).
# _STATE.pending_conflicts blocks all tools until resolved; _STATE.pending_enrichments
# holds graph enrichment tasks that block until resolved via kg_add or reject.
# Both default to None on ServerState construction — no explicit init needed here.

# ── Session isolation: save/restore state per session_id ──
# _STATE.session_state maps session_id -> {active_intent, pending_conflicts,
# pending_enrichments, declared}. When multiple callers (sub-agents) share the
# same MCP process but have different session IDs, this prevents them from
# overwriting each other's state.


def _sanitize_session_id(session_id: str) -> str:
    """Match hooks_cli._sanitize_session_id — only alnum/dash/underscore.

    Returns "" for empty or fully-stripped input. NO FALLBACK to
    'unknown' or 'default' — callers must handle empty sid explicitly.
    A shared sid value (like 'unknown') would merge every agent's
    state file into one, producing cross-agent contamination.
    """
    import re

    if not session_id:
        return ""
    return re.sub(r"[^a-zA-Z0-9_-]", "", str(session_id))


def _save_session_state():
    """Save current session state before switching to a different session.

    Disk is authoritative for pending state (see intent._persist_active_intent).
    The in-memory session_state cache snapshots ONLY the ephemeral fields
    (active_intent + declared_entities). Pending conflicts and enrichments
    are NOT cached here — they live on disk and are re-read via
    _load_pending_*_from_disk on every restore. That asymmetry is deliberate:
    once a caller clears pending state (resolve_conflicts / resolve_enrichments
    clear in-memory AND persist the cleared disk file), a later session-id
    switch must NOT resurrect the old pending items from a stale snapshot.
    Cashing them in session_state was the root of the "enrichment loop"
    where rejected enrichments re-appeared on every declare_intent.
    """
    if _STATE.session_id:
        _STATE.session_state[_STATE.session_id] = {
            "active_intent": _STATE.active_intent,
            "declared": _STATE.declared_entities,
        }


def _load_pending_from_disk(key: str, session_id: str = None) -> list:
    """Load pending items (conflicts or enrichments) from the active intent state file.

    Disk is the source of truth for cross-session/cross-restart state.
    Returns empty list if no sid, no file, or no items pending.

    NO FALLBACK to ``active_intent_default.json``. A request for a
    specific sid's pending state must NEVER read another sid's file.
    """
    sid = session_id or _STATE.session_id
    if not sid:
        return []
    try:
        state_file = _INTENT_STATE_DIR / f"active_intent_{sid}.json"
        if not state_file.is_file():
            return []
        data = json.loads(state_file.read_text(encoding="utf-8"))
        return data.get(key) or []
    except Exception:
        return []


def _load_pending_conflicts_from_disk(session_id: str = None) -> list:
    return _load_pending_from_disk("pending_conflicts", session_id)


def _load_pending_enrichments_from_disk(session_id: str = None) -> list:
    return _load_pending_from_disk("pending_enrichments", session_id)


def _restore_session_state(sid: str):
    """Restore session state for the given session_id.

    Pending conflicts + enrichments are NOT read from the in-memory cache —
    they always come from disk (the authoritative source, updated in lockstep
    by intent._persist_active_intent). Everything else (active_intent,
    declared_entities) is ephemeral and safe to cache in-process.
    """
    if sid in _STATE.session_state:
        s = _STATE.session_state[sid]
        _STATE.active_intent = s["active_intent"]
        _STATE.declared_entities = s["declared"]
    else:
        _STATE.active_intent = None
        _STATE.declared_entities = set()
    # Disk is authoritative for pending state. Always OVERWRITE (not
    # additive) so that a cleared file becomes cleared state — the old
    # "if disk has something, set; otherwise leave memory alone" logic
    # let a stale in-memory copy survive past a legitimate clear.
    _STATE.pending_conflicts = _load_pending_conflicts_from_disk(sid) or None
    _STATE.pending_enrichments = _load_pending_enrichments_from_disk(sid) or None


def _require_sid(action: str = "this operation") -> dict:
    """Validate an agent session_id is present. Returns error dict on
    failure, None on pass.

    Every write tool that touches state (active intent file, pending
    queues, trace file, save counter) must call this at the top:

        sid_err = _require_sid(action="resolve_enrichments")
        if sid_err:
            return sid_err

    On failure the agent sees a clear error pointing at the root cause
    (the hook didn't inject sessionId) rather than a silent "no-op"
    that looks like success. Agent can't proceed without fixing the
    hook wiring — which is what we want.

    NO FALLBACK TO A SHARED SID. A missing sid in a state-writing tool
    is a real error. Quietly substituting "default" / "unknown" would
    cause cross-agent contamination (the 2026-04-19 deadlock class).
    """
    if not _STATE.session_id:
        return {
            "success": False,
            "error": (
                f"'{action}' requires an active session_id but none was "
                f"propagated to the MCP server. Root cause: the PreToolUse "
                f"hook must inject sessionId via hookSpecificOutput."
                f"updatedInput on every MCP tool call. Check that the "
                f"mempalace plugin is installed and its hook is registered "
                f"in ~/.claude/settings.json. Refusing to proceed with an "
                f"empty sid — using a shared default would cross-contaminate "
                f"other agents' state."
            ),
        }
    return None


def _require_agent(agent: str, action: str = "this operation") -> dict:
    """Validate agent is declared (is_a agent). Returns {"success": False, ...} on failure, None on pass.

    centralized agent validation. Every write MCP tool that
    attributes an action to an agent should call this at the top:

        err = _require_agent(agent, action="kg_add")
        if err:
            return err

    Callers pass the RAW agent name (not normalized); this helper
    normalizes and walks the KG for an `is_a agent` edge. Missing agent
    or undeclared agent → structured error with the declaration recipe.
    Never raises — KG lookup failures are treated as pass (graceful
    degradation when KG is unavailable, e.g. in fresh test fixtures).
    """
    if not agent or not isinstance(agent, str) or not agent.strip():
        return {
            "success": False,
            "error": (
                f"`agent` is required for {action}. Pass your declared agent entity name "
                f"(e.g. 'ga_agent', 'technical_lead_agent'). Every write operation must "
                f"be attributed to a declared agent (is_a agent)."
            ),
        }
    try:
        from .knowledge_graph import normalize_entity_name

        agent_id = normalize_entity_name(agent)
        if _STATE.kg:
            edges = _STATE.kg.query_entity(agent_id, direction="outgoing")
            is_agent = any(
                e["predicate"] == "is_a" and e["object"] == "agent" and e.get("current", True)
                for e in edges
            )
            if not is_agent:
                return {
                    "success": False,
                    "error": (
                        f"`agent` '{agent}' is not a declared agent (missing is_a agent edge). "
                        f"Declare it first: "
                        f"kg_declare_entity(name='{agent}', kind='entity', importance=4, "
                        f"context={{'queries': ['<who you are>', '<your role>'], "
                        f"'keywords': ['agent', '<identifier>']}}, added_by='{agent}') "
                        f"then kg_add(subject='{agent}', predicate='is_a', object='agent', "
                        f"context={{'queries': [...], 'keywords': [...]}})."
                    ),
                }
    except Exception:
        pass  # graceful fallback if KG unavailable (fresh fixture, etc.)
    return None


def _declare_entity_recipe(
    name: str,
    kind: str = "entity",
    hint: str = None,
    extra_properties: str = "",
) -> str:
    """Canonical kg_declare_entity recipe used in error messages.

    Single source of truth — DO NOT hand-roll `description=...` in new
    error strings; the tool rejects it (see tool_kg_declare_entity, the
    P4.2 legacy-path block). `context={queries,keywords}` is mandatory.

    Args:
        name: entity name to insert into the example.
        kind: one of 'entity' | 'class' | 'predicate' | 'literal'.
        hint: short phrase describing what this entity is; becomes queries[0].
        extra_properties: optional trailing ', properties=...' fragment for
            callers that need to teach class/predicate-specific metadata
            (rules_profile, constraints, file_path, etc.).
    """
    hint = hint or f"what {name} represents"
    default_importance = 4 if kind in ("class", "predicate") else 3
    props = f", properties={extra_properties}" if extra_properties else ""
    return (
        f"kg_declare_entity(name='{name}', kind='{kind}', importance={default_importance}, "
        f"context={{'queries': ['{hint}', '<another perspective>'], "
        f"'keywords': ['<term1>', '<term2>']}}, "
        f"added_by='<your_agent>'"
        f"{props})"
    )


def _declare_intent_recipe(intent_type: str = "modify", slots: str = None) -> str:
    """Canonical mempalace_declare_intent recipe for error messages.

    Single source of truth — the tool requires `context={queries,keywords}`
    AND `budget`; the old `description=` path is gone.
    """
    slots = slots or '{"files": ["target_file"]}'
    return (
        f"mempalace_declare_intent(intent_type='{intent_type}', slots={slots}, "
        f"context={{'queries': ['<what you plan to do>', '<another angle>'], "
        f"'keywords': ['<term1>', '<term2>']}}, "
        f"agent='<your_agent>', budget={{'Read': 5, 'Edit': 3}})"
    )


def _is_declared(entity_id: str) -> bool:
    """Check if an entity is declared, with fallback to persistent KG.

    The in-memory _STATE.declared_entities set is a cache that gets cleared on
    MCP server restart. If an entity isn't in the cache but exists in the
    persistent KG (ChromaDB), it's auto-added to the cache and considered
    declared. This makes declarations survive MCP server restarts without
    requiring the model to re-call wake_up.

    P5.2 lookup order (must cover BOTH physical layouts):
      1. In-memory cache — _STATE.declared_entities (session-lifetime fast path).
      2. Multi-view entities — where={"entity_id": X}. Entities declared
         via kg_declare_entity(context=...) live under '{eid}__v{N}' ids
         with metadata.entity_id=eid (see _sync_entity_views_to_chromadb).
      3. Single-record entities — ids=[X]. Internal bookkeeping entities
         (execution traces, gotchas) written by _sync_entity_to_chromadb
         use raw '{eid}' as the Chroma id, so the id-based lookup still
         applies. Same for the memories collection.
    """
    if entity_id in _STATE.declared_entities:
        return True

    # KG fallback — SQLite is the authoritative source of truth.
    # Wake_up pulls classes/predicates/intents directly from
    # _STATE.kg.list_entities, so every entity surfaced to the model via
    # declared.* must also be considered declared for gating purposes.
    # Checking the KG before Chroma ensures that a transient in-memory
    # cache wipe (session_id switch, MCP restart) or a Chroma-side glitch
    # cannot silently downgrade a legitimate intent_type to "not declared".
    try:
        if _STATE.kg.get_entity(entity_id):
            _STATE.declared_entities.add(entity_id)
            return True
    except Exception:
        pass

    ecol = _get_entity_collection(create=False)
    if not ecol:
        return False

    # Multi-view lookup (post-P5.2 entities declared via Context)
    try:
        result = ecol.get(where={"entity_id": entity_id}, limit=1)
        if result and result.get("ids"):
            _STATE.declared_entities.add(entity_id)
            return True
    except Exception:
        pass

    # Fallback: raw-id lookup (single-record bookkeeping entities)
    try:
        result = ecol.get(ids=[entity_id])
        if result and result.get("ids"):
            _STATE.declared_entities.add(entity_id)
            return True
    except Exception:
        pass

    return False


def _get_entity_collection(create: bool = True):
    """Entity collection accessor.

    Phase M1 collapsed the two physical Chroma collections (records +
    entities) into a single ``mempalace_records`` collection discriminated
    by ``metadata.kind``. This helper remains for callsite compatibility —
    anywhere that previously fetched "the entity collection" now gets the
    unified collection instead, and any query it runs must filter on
    ``metadata.kind`` if it wants entity-only results.

    The view-schema migration (``__vN`` suffix) runs against the same
    collection on first touch.
    """
    col = _get_collection(create=create)
    if col is not None:
        try:
            _migrate_entity_views_schema(col)
        except Exception:
            pass
    return col


# Chroma migration that retires the old '::view_N' suffix scheme.
# Detects legacy records by the absence of metadata.entity_id (all records
# written by the new code path carry it). Migration is a single pass at
# first _get_entity_collection(create=True) call per process; a _STATE flag
# prevents re-runs.


def _migrate_entity_views_schema(col):
    """Retire ::view_N ids in favour of __vN + metadata.entity_id.

    Idempotent. Only touches records missing metadata.entity_id:
      - '{eid}::view_{N}'  →  id='{eid}__v{N}',  meta.entity_id=eid, meta.view_index=N
      - pre-P4.2 plain '{eid}'  →  id='{eid}__v0', meta.entity_id=eid, meta.view_index=0

    Embeddings are preserved via the include=['embeddings'] round-trip,
    avoiding an expensive re-embed pass.
    """
    if _STATE.entity_views_migrated:
        return
    _STATE.entity_views_migrated = True
    try:
        got = col.get(include=["documents", "metadatas", "embeddings"])
    except Exception:
        return
    if not got or not got.get("ids"):
        return

    to_upsert_ids, to_upsert_docs, to_upsert_metas, to_upsert_embs = [], [], [], []
    to_delete = []
    all_embs_present = True

    for i, raw_id in enumerate(got["ids"]):
        meta = (got["metadatas"][i] if got.get("metadatas") else {}) or {}
        if "entity_id" in meta:
            continue  # Already in the new shape
        if "::view_" in raw_id:
            eid, _sep, v = raw_id.rpartition("::view_")
            try:
                view_idx = int(v)
            except ValueError:
                view_idx = 0
            new_id = f"{eid}__v{view_idx}"
        else:
            eid = raw_id
            view_idx = 0
            new_id = f"{eid}__v0"

        new_meta = {**meta, "entity_id": eid, "view_index": view_idx}
        to_delete.append(raw_id)
        to_upsert_ids.append(new_id)
        to_upsert_docs.append((got["documents"][i] if got.get("documents") else "") or "")
        to_upsert_metas.append(new_meta)
        emb = got["embeddings"][i] if got.get("embeddings") else None
        if emb is None:
            all_embs_present = False
        to_upsert_embs.append(emb)

    if not to_upsert_ids:
        return
    try:
        kwargs = {"ids": to_upsert_ids, "documents": to_upsert_docs, "metadatas": to_upsert_metas}
        if all_embs_present:
            kwargs["embeddings"] = to_upsert_embs
        col.upsert(**kwargs)
        col.delete(ids=to_delete)
        logger.info(
            f"P5.2 entity_views migration: rewrote {len(to_upsert_ids)} records "
            f"({'preserved' if all_embs_present else 're-embedded'})"
        )
    except Exception as e:
        logger.warning(f"P5.2 entity_views migration failed: {e}")


# kind='record' → 'record' one-pass migration (flag lives on _STATE).


def _migrate_kind_memory_to_record():
    """Rewrite kind='record' → 'record' across ChromaDB memory collection
    + SQLite entities table. Idempotent, one-pass per process.

    Why: "memory" was overloaded (palace-level concept + record-type name).
    Renaming at the record-type layer lets "memory" stay the palace
    concept cleanly. Data layout is untouched — only the metadata.kind
    string changes.
    """
    if _STATE.kind_rename_migrated:
        return
    _STATE.kind_rename_migrated = True

    # 1) Rewrite Chroma memory collection metadata.
    try:
        col = _get_collection(create=False)
        if col is not None:
            got = col.get(where={"kind": "memory"}, include=["metadatas"])
            if got and got.get("ids"):
                new_metas = []
                for meta in got["metadatas"] or []:
                    m = dict(meta or {})
                    m["kind"] = "record"
                    new_metas.append(m)
                col.update(ids=got["ids"], metadatas=new_metas)
                logger.info(
                    f"P6.2 kind migration (chroma memory collection): "
                    f"rewrote {len(got['ids'])} records from 'memory' to 'record'."
                )
    except Exception as e:
        logger.warning(f"P6.2 kind migration (chroma) failed: {e}")

    # 2) Rewrite SQLite entities table.
    try:
        if _STATE.kg:
            conn = _STATE.kg._conn()
            cursor = conn.execute("UPDATE entities SET kind='record' WHERE kind='record'")
            conn.commit()
            if cursor.rowcount:
                logger.info(
                    f"P6.2 kind migration (sqlite entities): "
                    f"rewrote {cursor.rowcount} rows from 'memory' to 'record'."
                )
    except Exception as e:
        logger.warning(f"P6.2 kind migration (sqlite) failed: {e}")


# M1 one-shot migration: absorb mempalace_entities into mempalace_records.
# Gated on ServerState.entity_collection_merged (added to server_state.py).
# Idempotent: on a fresh palace the legacy collection may not exist, in
# which case this is a no-op.


def _migrate_entities_collection_into_records():
    """Copy every row from the legacy mempalace_entities collection into
    the unified mempalace_records collection, then delete the legacy
    collection. Runs once per process.

    Safe to run multiple times: subsequent calls see no entities
    collection and exit quickly.

    ID-space note: entity rows use ``<entity_id>__vN`` IDs while record
    rows use ``record_<agent>_<slug>`` IDs — two non-overlapping
    namespaces — so merging cannot create ID collisions in the target
    collection. metadata.kind stays the discriminator for kind-scoped
    queries (``kind="class"``, ``"entity"``, ``"predicate"`` vs
    ``"record"``).
    """
    if getattr(_STATE, "entity_collection_merged", False):
        return
    _STATE.entity_collection_merged = True

    try:
        client = chromadb.PersistentClient(path=_STATE.config.palace_path)
        try:
            legacy = client.get_collection("mempalace_entities")
        except Exception:
            return  # Legacy collection never existed — fresh palace.

        dest = _get_collection(create=True)
        if dest is None:
            logger.warning("M1 migration: target mempalace_records unavailable")
            return

        got = legacy.get(include=["documents", "metadatas", "embeddings"])
        if not got or not got.get("ids"):
            # Collection exists but is empty — drop it.
            try:
                client.delete_collection("mempalace_entities")
            except Exception:
                pass
            logger.info("M1 migration: legacy mempalace_entities was empty, dropped.")
            return

        ids = got["ids"]
        docs = got.get("documents") or [None] * len(ids)
        metas = got.get("metadatas") or [None] * len(ids)
        embs = got.get("embeddings") or [None] * len(ids)

        BATCH = 200
        moved = 0
        any_upsert_failed = False
        for start in range(0, len(ids), BATCH):
            chunk_ids = ids[start : start + BATCH]
            chunk_docs = docs[start : start + BATCH]
            chunk_metas = metas[start : start + BATCH]
            chunk_embs = embs[start : start + BATCH]
            upsert_kwargs = {
                "ids": chunk_ids,
                "documents": chunk_docs,
                "metadatas": chunk_metas,
            }
            if all(e is not None for e in chunk_embs):
                upsert_kwargs["embeddings"] = chunk_embs
            try:
                dest.upsert(**upsert_kwargs)
                moved += len(chunk_ids)
            except Exception as e:
                any_upsert_failed = True
                logger.warning(f"M1 migration upsert batch failed: {e}")

        # Safety: only delete the legacy collection if EVERY row landed in
        # the target. A partial copy must leave the source intact so no
        # embeddings are lost; the migration flag is already set, so the
        # next startup won't retry, but the data is still accessible.
        if any_upsert_failed or moved != len(ids):
            logger.warning(
                f"M1 migration: moved {moved}/{len(ids)} rows; legacy collection "
                f"NOT deleted — partial copy detected, data preserved."
            )
            return

        try:
            client.delete_collection("mempalace_entities")
        except Exception as e:
            logger.warning(f"M1 migration: delete_collection failed: {e}")

        logger.info(
            f"M1 migration: moved {moved} entity rows into mempalace_records, "
            f"dropped legacy mempalace_entities."
        )
    except Exception as e:
        logger.warning(f"M1 migration failed: {e}")


FEEDBACK_CONTEXT_COLLECTION = "mempalace_feedback_contexts"


def _get_feedback_context_collection(create: bool = True):
    """Get or create the feedback contexts ChromaDB collection.

    Stores multi-view context vectors alongside feedback records.
    Each entry = one context snapshot (multiple views stored as separate embeddings).
    Used for MaxSim comparison when applying stored feedback.

    Pinned to cosine distance.
    """
    try:
        client = chromadb.PersistentClient(path=_STATE.config.palace_path)
        if create:
            return client.get_or_create_collection(
                FEEDBACK_CONTEXT_COLLECTION, metadata=_CHROMA_METADATA
            )
        else:
            return client.get_collection(FEEDBACK_CONTEXT_COLLECTION)
    except Exception:
        if create:
            try:
                client = chromadb.PersistentClient(path=_STATE.config.palace_path)
                return client.create_collection(
                    FEEDBACK_CONTEXT_COLLECTION, metadata=_CHROMA_METADATA
                )
            except Exception:
                return None
        return None


def _generate_context_id(prefix: str, views: list) -> str:
    """Collision-resistant context id from views + prefix + nanos + nonce.

    Hash of sorted views keeps the id stable-flavoured when the same Context
    is re-used. Nanosecond timestamp + 6-char random nonce eliminate the
    same-second collision class that the old YYYYMMDDTHHMMSS suffix had —
    two identical Contexts filed in the same instant now get distinct ids.
    """
    import hashlib
    import secrets
    import time

    text = "\n".join(sorted(v.strip() for v in (views or []) if isinstance(v, str)))
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    ns = time.time_ns()
    nonce = secrets.token_hex(3)  # 6 hex chars
    return f"ctx_{prefix}_{digest}_{ns}_{nonce}"


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

    MaxSim(A, B) = (1/|A|) * Σ_a max_b cos(a, b)     (ColBERT late-interaction)

    For each view in the current Context we find the best-matching view in
    a stored Context and average those maxes. This lets feedback transfer
    by *context similarity* rather than exact context-id match — a new
    query that looks like a past useful one inherits the signal.

    Reference:
      Khattab & Zaharia. "ColBERT: Efficient and Effective Passage Search
      via Contextualized Late Interaction over BERT." SIGIR 2020.
      → https://arxiv.org/abs/2004.12832
      (We use their late-interaction MaxSim operator at the context level,
      not the token level — same math, coarser granularity.)

    Implementation note: ChromaDB returns cosine distance; MaxSim requires
    cosine similarity. `similarity = 1 - distance` holds ONLY for cosine
    (that's why _get_feedback_context_collection pins hnsw:space="cosine",
    see P5.7). Other distance metrics would need a different conversion.

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
    _STATE.declared_entities = set()
    _STATE.session_id = ""


def _check_entity_similarity_multiview(
    views: list,
    kind_filter: str = None,
    exclude_id: str = None,
    threshold: float = None,
):
    """Multi-view collision detection (P4.2, refined in P5.2).

    Each view is queried independently against the entity collection; the
    per-view ranked candidates are merged via Reciprocal Rank Fusion. A hit
    is reported as a collision when its highest single-view similarity is
    above threshold (so one strong match still flags, but multi-view gives
    catches that single-vector cosine misses).

    Logical entity ids are read from metadata.entity_id (no id
    string splitting). Returns the same shape as _check_entity_similarity
    for drop-in compatibility.
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
                meta = results["metadatas"][0][i] or {}
                # read logical entity id from metadata. Defensive fallback
                # to raw_id only for records the migration hasn't touched yet
                # (will disappear once _migrate_entity_views_schema runs).
                logical_id = meta.get("entity_id") or raw_id
                if logical_id == exclude_id:
                    continue
                dist = results["distances"][0][i]
                sim = round(1 - dist, 3)
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
    """Single-description collision check.

    Used by tool_kg_update_entity after a description-only update (we have
    one new description and want to know whether it still collides). For
    kg_declare_entity (multi-view Context), use
    _check_entity_similarity_multiview which RRF-merges per-view rankings.
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
        # Kind-scoped collision: only check against same kind (was a
        # silent no-op before, the empty dict dropped the filter entirely).
        if kind_filter:
            query_kwargs["where"] = {"kind": kind_filter}
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
    """Create an entity in BOTH SQLite AND ChromaDB. Use this instead of _STATE.kg.add_entity directly.

    Args:
        embed_text: Optional override for what gets embedded in ChromaDB.
                    If None, uses description. Use for execution entities
                    where you want description-only embedding (no summary).
    """
    from .knowledge_graph import normalize_entity_name

    # pass provenance to SQLite
    _prov_session = _STATE.session_id or ""
    _prov_intent = _STATE.active_intent.get("intent_id", "") if _STATE.active_intent else ""
    eid = _STATE.kg.add_entity(
        name,
        kind=kind,
        description=description,
        importance=importance,
        properties=properties,
        session_id=_prov_session,
        intent_id=_prov_intent,
    )
    normalized = normalize_entity_name(name)
    _sync_entity_to_chromadb(
        normalized, name, embed_text or description, kind, importance, added_by=added_by
    )
    return eid


def _sync_entity_to_chromadb(
    entity_id: str, name: str, description: str, kind: str, importance: int, added_by: str = None
):
    """Single-description Chroma sync for internal bookkeeping entities.

    Used by intent.py finalize (execution entities, gotcha entities) and by
    tool_kg_update_entity after a description-only update — both cases
    naturally have one description and don't carry a multi-view Context.
    For Context-driven entity declarations, use _sync_entity_views_to_chromadb
    (multi-vector storage under '{entity_id}::view_N').
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
    # provenance auto-injection on entity Chroma records
    if _STATE.session_id:
        meta["session_id"] = _STATE.session_id
    if _STATE.active_intent and isinstance(_STATE.active_intent, dict):
        meta["intent_id"] = _STATE.active_intent.get("intent_id", "")
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
    """Multi-view sync to the entity ChromaDB collection (P4.2, refined in P5.2).

    Each view is stored as a separate record under '{entity_id}__v{N}' AND
    every record carries metadata.entity_id explicitly. Readers group views
    by the metadata field (col.get(where={'entity_id': X})) — NOT by parsing
    ids — so the separator choice is cosmetic, not load-bearing.

    The '__v' separator is deliberately chosen because it cannot appear
    inside a normalized_entity_name (which uses single-underscore segments
    only), making the literal id unambiguous for humans skimming the db.
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
    # provenance auto-injection on multi-view entity records
    if _STATE.session_id:
        base_meta["session_id"] = _STATE.session_id
    if _STATE.active_intent and isinstance(_STATE.active_intent, dict):
        base_meta["intent_id"] = _STATE.active_intent.get("intent_id", "")

    ids, docs, metas = [], [], []
    for i, view in enumerate(cleaned):
        ids.append(f"{entity_id}__v{i}")
        docs.append(view)
        m = dict(base_meta)
        m["view_index"] = i
        m["entity_id"] = entity_id  # canonical reverse lookup
        metas.append(m)
    ecol.upsert(ids=ids, documents=docs, metadatas=metas)


VALID_CARDINALITIES = {"many-to-many", "many-to-one", "one-to-many", "one-to-one"}


def tool_kg_declare_entity(  # noqa: C901
    name: str = None,
    context: dict = None,  # mandatory: {queries, keywords, entities?}
    kind: str = None,  # REQUIRED — no default, model must choose
    importance: int = 3,
    properties: dict = None,  # General-purpose metadata
    user_approved_star_scope: bool = False,  # Required for * scope
    added_by: str = None,  # REQUIRED — agent who declared this entity
    # Record-kind specific (REQUIRED when kind='record').
    slug: str = None,
    content: str = None,  # verbatim record text (kind='record' only); for other kinds, queries[0] is canonical
    content_type: str = None,  # one of: fact, event, discovery, preference, advice, diary
    source_file: str = None,
    entity: str = None,  # entity name(s) to link this record to
    predicate: str = "described_by",  # link predicate
    # ── Legacy single-string description path (REMOVED) ──
    description: str = None,  # accepted only as a hard-error trigger, see below
):
    """Declare an entity before using it in KG edges. REQUIRED per session.

    EVERY declaration speaks the unified Context object:

        context = {
          "queries":  list[str]   # 2-5 perspectives on what this entity is
          "keywords": list[str]   # 2-5 caller-provided exact terms
          "entities": list[str]   # 0+ related entity ids (optional)
        }

    Each query gets embedded as a separate Chroma record under
    '{entity_id}__v{N}' with metadata.entity_id=entity_id, so
    collision detection is multi-view RRF rather than single-vector
    cosine. Readers look up entities via where={"entity_id": X} — the
    suffix is cosmetic, the metadata is load-bearing. Keywords are stored
    in entity_keywords (the keyword channel reads them directly —
    auto-extraction is gone). The Context's view vectors are also
    persisted in mempalace_feedback_contexts under a generated
    context_id, recorded on the entity, so future feedback (found_useful
    / found_irrelevant) applies via MaxSim by context similarity.

    Args:
        name: Entity name (REQUIRED for kind=entity/class/predicate/literal;
              auto-computed from added_by/slug for kind='record').
        context: MANDATORY Context dict — see above. Replaces the single
              `description` parameter (which is now rejected with an error).
        kind: 'entity' | 'class' | 'predicate' | 'literal' | 'memory'.
        content: VERBATIM text for kind='record' (the actual memory body).
              For non-memory kinds, queries[0] is used as the canonical
              description; pass `content` only when you need to override it.
        importance: 1-5.
        properties: predicate constraints / intent type rules_profile / arbitrary metadata.
        user_approved_star_scope: required only for "*" tool scopes.
        added_by: declared agent name (REQUIRED).
        slug/content_type/source_file/entity/predicate: kind='record' only.

    Returns: status "created" | "exists" | "collision".
    """
    from .knowledge_graph import normalize_entity_name
    from .scoring import validate_context

    sid_err = _require_sid(action="kg_declare_entity")
    if sid_err:
        return sid_err

    # ── Reject the legacy single-string description path ──
    if description is not None and context is None:
        return {
            "success": False,
            "error": (
                "`description` is gone. Pass `context` instead — a dict "
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

    # ── kind='record' dispatch — records are first-class entities.
    if kind == "record":
        if content is None or not str(content).strip():
            return {
                "success": False,
                "error": (
                    "kind='record' requires `content` — the verbatim record text. "
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
        # Check for collisions with OTHER entities of SAME KIND (not self) — multi-view
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
        _STATE.declared_entities.add(normalized)
        # Update description + importance + kind if provided and different
        if description and description != existing.get("description", ""):
            _STATE.kg.update_entity_description(normalized, description, importance)
            _sync_entity_views_to_chromadb(
                normalized, name, queries, kind, importance or 3, added_by=added_by
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
            "description": existing.get("description") or description,
            "importance": existing.get("importance", 3),
            "edge_count": _STATE.kg.entity_edge_count(normalized),
        }

    # New entity — multi-view collision check
    similar = _check_entity_similarity_multiview(queries, kind_filter=kind)

    # Create the entity regardless — conflicts are resolved after creation
    props = properties if isinstance(properties, dict) else {}
    if added_by:
        props["added_by"] = added_by
    # SQLite row first (with queries[0] as the canonical description)
    _STATE.kg.add_entity(
        name, kind=kind, description=description, importance=importance or 3, properties=props
    )
    # Multi-vector embedding into the entity Chroma collection (one record per view)
    _sync_entity_views_to_chromadb(
        normalized, name, queries, kind, importance or 3, added_by=added_by
    )
    # Caller-provided keywords → entity_keywords table
    _STATE.kg.add_entity_keywords(normalized, keywords)
    # Persist the creation Context's view vectors and link the context_id to the entity
    cid = persist_context(clean_context, prefix=kind or "entity")
    if cid:
        _STATE.kg.set_entity_creation_context(normalized, cid)
    _STATE.declared_entities.add(normalized)

    # Auto-add is-a thing for new class entities (ensures class inheritance works)
    if kind == "class" and normalized != "thing":
        try:
            _STATE.kg.add_triple(normalized, "is_a", "thing")
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
    # Detector respects past feedback via get_edge_usefulness on 'suggested_link'.
    suggested_links = (
        _detect_suggested_links(source_id=normalized, query_text=description)
        if kind in ("entity", "class") and description
        else []
    )

    result = {
        "success": True,
        "status": "created",
        "entity_id": normalized,
        "kind": kind,
        "description": description,
        "importance": importance or 3,
    }
    if suggested_links:
        # Store as pending enrichments — blocks tools until agent creates edges or rejects
        enrichments = []
        for sl in suggested_links:
            eid = sl["entity_id"]
            enrichment_id = f"enrich_{normalized}_{eid}"
            enrichments.append(
                {
                    "id": enrichment_id,
                    "reason": f"Entity '{normalized}' is related to '{eid}' (similarity: {sl['similarity']})",
                    "from_entity": normalized,
                    "to_entity": eid,
                    "similarity": sl["similarity"],
                    "to_description": sl.get("description", "")[:100],
                }
            )
        _STATE.pending_enrichments = enrichments
        intent._persist_active_intent()
        result["suggested_links"] = suggested_links
        result["enrichments_count"] = len(enrichments)
        result["enrichments_prompt"] = (
            f"{len(enrichments)} graph enrichment tasks pending. "
            "For each: call kg_add(subject, predicate, object) to create the edge "
            "with a predicate YOU choose from the declared predicates (see wake_up), "
            "then call mempalace_resolve_enrichments to mark done. Or reject with reason."
        )

    # ── Conflict detection: flag similar entities for resolution ──
    if similar:
        conflicts = []
        for s in similar:
            conflict_id = f"conflict_entity_{normalized}_{s['entity_id']}"
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
                "existing_description": s.get("description", "")[:200],
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
    description: str = None,
    importance: int = None,
    properties: dict = None,
    context: dict = None,  # optional: re-record creation_context when meaning changes
    agent: str = None,  # mandatory attribution
    # Record-specific (only meaningful when entity is a kind='record')
    content_type: str = None,
):
    """Update any entity (record or KG node) in place. Pass only the fields you want to change.

    `context` is OPTIONAL but RECOMMENDED whenever you change
    semantic fields (`description` for entities, or `properties` that alter
    meaning like predicate constraints / intent-type rules). When present
    the Context's view vectors are persisted and the entity's
    creation_context_id is repointed to the new context — future MaxSim
    feedback then transfers against the updated meaning, not the old one.

    Args:
        entity: Entity ID or record ID to update.
        description: New description. For entities (kind=entity/class/predicate/literal):
            re-syncs to entity ChromaDB and runs collision distance check.
            For records: NOT supported here — use kg_delete_entity +
            kg_declare_entity to change record content.
        importance: New importance (1-5). Works for both entities and records.
        properties: Merged INTO existing properties dict (shallow merge at top
            level). For predicates use {"constraints": {...}} to replace
            constraints. For intent types {"rules_profile": {...}} to update slots
            or tool_permissions.
        content_type: Record-only content type update (no re-embedding).
    """
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
    try:
        if description is not None:
            description = sanitize_content(description, max_length=5000)
        if importance is not None:
            importance = _validate_importance(importance)
        if content_type is not None:
            content_type = _validate_content_type(content_type)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    # Reject contradictory inputs early
    if is_record_id and description is not None:
        return {
            "success": False,
            "error": (
                "Cannot update record description in place — embeddings would be "
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
            return {"success": True, "reason": "no_change", "entity_id": entity}

        _wal_log(
            "kg_update_entity",
            {
                "entity_id": entity,
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
                "entity_id": entity,
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
        return {"success": False, "error": f"Entity '{normalized}' not found."}

    updated_fields = []
    final_description = existing["description"]
    final_importance = existing.get("importance", 3)

    # Description update + ChromaDB resync
    if description is not None and description != existing["description"]:
        _STATE.kg.update_entity_description(normalized, description)
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

    # ── re-record creation_context when meaning changed ──
    # A description or properties change IS a semantic update — future
    # MaxSim-graded feedback should attach to the new meaning, not the old.
    # Pure-importance updates don't move meaning, so we skip context re-persist
    # unless description/properties changed too.
    semantic_change = any(f in updated_fields for f in ("description", "properties"))
    new_context_id = ""
    if semantic_change and context is not None:
        from .scoring import validate_context as _validate_context

        clean_ctx, ctx_err = _validate_context(context)
        if ctx_err:
            return ctx_err
        new_context_id = persist_context(clean_ctx, prefix=existing.get("kind", "entity"))
        if new_context_id:
            _STATE.kg.set_entity_creation_context(normalized, new_context_id)
            _STATE.kg.add_entity_keywords(normalized, clean_ctx["keywords"])
            updated_fields.append("creation_context")

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
    if new_context_id:
        result["creation_context_id"] = new_context_id

    # P5.10 hint: nudge callers to pass context when meaning changed.
    if semantic_change and not new_context_id:
        result["context_hint"] = (
            "Description/properties changed but no `context` was provided — "
            "future MaxSim feedback will still attach to the OLD creation_context_id. "
            "Pass `context={queries,keywords,entities?}` to re-anchor the entity."
        )

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


def tool_kg_merge_entities(
    source: str, target: str, update_description: str = None, agent: str = None
):
    """Merge source entity into target. All edges rewritten. Source becomes alias.

    Use when kg_declare_entity returns 'collision' and the entities are
    actually the same thing. All triples from source are moved to target.
    Source name becomes an alias that auto-resolves to target in future queries.

    Args:
        source: Entity to merge FROM (will be soft-deleted).
        target: Entity to merge INTO (will be kept, edges grow).
        update_description: Optional new description for the merged entity.
        agent: mandatory, declared agent attributing this merge.
    """
    sid_err = _require_sid(action="kg_merge_entities")
    if sid_err:
        return sid_err
    agent_err = _require_agent(agent, action="kg_merge_entities")
    if agent_err:
        return agent_err
    _wal_log(
        "kg_merge_entities",
        {
            "source": source,
            "target": target,
            "update_description": update_description[:200] if update_description else None,
            "agent": agent,
        },
    )

    result = _STATE.kg.merge_entities(source, target, update_description)
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
                target_entity["description"],
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


# tool_kg_update_predicate_constraints removed: merged into tool_kg_update_entity.
# Call kg_update_entity(entity=predicate, properties={"constraints": {...}}).


def tool_kg_list_declared():
    """List all entities declared in this session."""
    results = []
    for eid in sorted(_STATE.declared_entities):
        entity = _STATE.kg.get_entity(eid)
        if entity:
            results.append(
                {
                    "entity_id": eid,
                    "name": entity["name"],
                    "description": entity["description"],
                    "importance": entity["importance"],
                    "last_touched": entity["last_touched"],
                    "edge_count": _STATE.kg.entity_edge_count(eid),
                }
            )
    return {
        "declared_count": len(results),
        "entities": results,
    }


# tool_kg_entity_info removed: use kg_query(entity=..., direction="both").


# ==================== INTENT DECLARATION ====================

# _STATE.active_intent holds the session-level active intent (at most one).
# Defaults to None on ServerState construction — no explicit init needed here.
_INTENT_STATE_DIR = Path(os.path.expanduser("~/.mempalace/hook_state"))


# Intent functions are in intent.py; init() is called after module globals are set.
# Aliases so TOOLS dispatch continues to work:
def tool_declare_intent(*args, **kwargs):
    return intent.tool_declare_intent(*args, **kwargs)


def tool_active_intent(*args, **kwargs):
    return intent.tool_active_intent(*args, **kwargs)


def tool_extend_intent(*args, **kwargs):
    return intent.tool_extend_intent(*args, **kwargs)


def tool_resolve_conflicts(actions: list = None, agent: str = None):  # noqa: C901
    """Resolve pending conflicts — contradictions, duplicates, or suggestions.

    Unified conflict resolution for ALL data types: edges, entities, memories.
    Each action specifies what to do with a conflict.

    Args:
        actions: List of {id, action, into?, merged_content?} dicts.
            id: The conflict ID (from the pending conflicts list).
            action: One of:
                "invalidate" — mark existing item as no longer current (sets valid_to)
                "merge" — combine items (must provide into + merged_content)
                "keep" — both items are valid, no conflict
                "skip" — don't add the new item (remove it)
            into: Target entity/memory ID to merge into (required for "merge")
            merged_content: Merged description/content (required for "merge")
        agent: mandatory, declared agent resolving these conflicts.
    """
    sid_err = _require_sid(action="resolve_conflicts")
    if sid_err:
        return sid_err
    agent_err = _require_agent(agent, action="resolve_conflicts")
    if agent_err:
        return agent_err

    # Disk is source of truth — reload _STATE.pending_conflicts from the active
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

    # Index pending conflicts by ID — defensively coerce if any entries are
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
                    f"explaining WHY you chose this action. This is a real semantic decision — "
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
                    f"Each conflict is a unique semantic decision — evaluate individually and "
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
                            "reason": "merge requires 'merged_content' — read BOTH items in full, then provide combined content",
                        }
                    )
                    continue

                # Determine source (the one NOT being merged into)
                source = new_id if into == existing_id else existing_id

                if conflict_type in ("entity_duplicate", "memory_duplicate"):
                    # Use existing kg_merge_entities for the plumbing
                    merge_result = tool_kg_merge_entities(
                        source=source,
                        target=into,
                        update_description=merged_content,
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
                # Both items are valid — no action needed
                results.append({"id": cid, "status": "kept"})

            elif action == "skip":
                # Don't add the new item — remove it if already added
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

            # For edge contradictions, also emit a negative signal on the
            # losing edge so BFS prunes it in future traversals.
            if conflict_type == "edge_contradiction":
                loser = None
                if action in ("invalidate",):
                    loser = (
                        conflict.get("existing_subject", ""),
                        conflict.get("existing_predicate", ""),
                        conflict.get("existing_object", ""),
                    )
                elif action == "skip":
                    loser = (
                        conflict.get("new_subject", ""),
                        conflict.get("new_predicate", ""),
                        conflict.get("new_object", ""),
                    )
                if loser and all(loser):
                    try:
                        _STATE.kg.record_edge_feedback(
                            loser[0],
                            loser[1],
                            loser[2],
                            _intent_type,
                            useful=False,
                        )
                    except Exception:
                        pass

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
    _STATE.pending_conflicts = None
    try:
        intent._persist_active_intent()
    except Exception:
        pass
    return {"success": True, "resolved": results}


def tool_resolve_enrichments(actions: list = None, agent: str = None):
    """Resolve pending graph enrichment tasks.

    Each enrichment represents two entities that should be connected.
    The agent must either:
      - 'done': confirm the edge was created (agent already called kg_add)
      - 'reject': decline to create the edge (mandatory reason, min 15 chars)
    """
    sid_err = _require_sid(action="resolve_enrichments")
    if sid_err:
        return sid_err
    agent_err = _require_agent(agent, action="resolve_enrichments")
    if agent_err:
        return agent_err

    # Disk is source of truth
    if not _STATE.pending_enrichments:
        _STATE.pending_enrichments = _load_pending_enrichments_from_disk()

    if not _STATE.pending_enrichments:
        try:
            intent._persist_active_intent()
        except Exception:
            pass
        return {"success": True, "message": "No pending enrichments."}

    if not actions:
        return {
            "success": False,
            "error": (
                "Must provide actions list. Each enrichment needs: "
                "{id, action ('done' or 'reject'), reason (mandatory for reject)}."
            ),
            "pending": _STATE.pending_enrichments,
        }

    enrichment_map = {
        e["id"]: e for e in _STATE.pending_enrichments if isinstance(e, dict) and e.get("id")
    }
    resolved_ids = set()
    results = []
    MIN_REASON = 15

    for act in actions:
        if isinstance(act, str):
            try:
                act = json.loads(act)
            except Exception:
                results.append({"id": "?", "status": "error", "reason": f"Unparseable: {act!r}"})
                continue
        if not isinstance(act, dict):
            results.append(
                {
                    "id": "?",
                    "status": "error",
                    "reason": f"Expected object, got {type(act).__name__}",
                }
            )
            continue

        eid = act.get("id", "")
        action = act.get("action", "")
        reason = (act.get("reason") or "").strip()

        if eid not in enrichment_map:
            results.append(
                {"id": eid, "status": "error", "reason": f"Unknown enrichment ID: {eid}"}
            )
            continue

        enrichment = enrichment_map[eid]
        from_entity = enrichment.get("from_entity", "")
        to_entity = enrichment.get("to_entity", "")
        intent_type = _STATE.active_intent.get("intent_type", "") if _STATE.active_intent else ""

        if action == "done":
            if from_entity and to_entity:
                try:
                    _STATE.kg.record_edge_feedback(
                        from_entity,
                        "suggested_link",
                        to_entity,
                        intent_type,
                        useful=True,
                    )
                except Exception:
                    pass
            results.append({"id": eid, "status": "done"})
            resolved_ids.add(eid)
        elif action == "reject":
            if len(reason) < MIN_REASON:
                return {
                    "success": False,
                    "error": (
                        f"Rejection of enrichment '{eid}' requires a reason "
                        f"(minimum {MIN_REASON} characters) explaining why these "
                        f"entities should NOT be connected."
                    ),
                }
            if from_entity and to_entity:
                try:
                    # Store reason in context_keywords so future MaxSim reads it.
                    _STATE.kg.record_edge_feedback(
                        from_entity,
                        "suggested_link",
                        to_entity,
                        intent_type,
                        useful=False,
                        context_keywords=reason[:200],
                    )
                except Exception:
                    pass
            results.append({"id": eid, "status": "rejected", "reason": reason})
            resolved_ids.add(eid)
        else:
            results.append(
                {
                    "id": eid,
                    "status": "error",
                    "reason": f"Unknown action: {action}. Use 'done' or 'reject'.",
                }
            )
            continue

    # Check all enrichments addressed
    unresolved = set(enrichment_map.keys()) - resolved_ids
    if unresolved:
        return {
            "success": False,
            "error": f"{len(unresolved)} enrichments not addressed. Provide action for each.",
            "unresolved": [enrichment_map[eid] for eid in unresolved],
            "resolved": results,
        }

    # Clear pending enrichments and persist
    _STATE.pending_enrichments = None
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
    content_type: str = "diary",
    importance: int = None,
):
    """
    Write a diary entry for this agent. Entries are timestamped and
    accumulate over time, scoped by agent name.

    The diary is a HIGH-LEVEL SESSION NARRATIVE — not a detailed log.
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

    Each entry should be a DELTA from the previous — what changed,
    not a full restatement of everything.

    Args:
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
            "agent": agent_name,
            "added_by": agent_name,
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
        logger.info(f"Diary entry: {entry_id} content_type={content_type} imp={importance}")

        # Update the stop hook save counter — proves diary was actually written.
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
            pass  # Non-fatal — save counter is best-effort

        return {
            "success": True,
            "entry_id": entry_id,
            "agent": agent_name,
            "topic": topic,
            "content_type": content_type,
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
    col = _get_collection()
    if not col:
        return _no_palace()

    try:
        results = col.get(
            where={"$and": [{"added_by": agent_name}, {"type": "diary_entry"}]},
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
        "description": (
            "Unified search — records (prose) + entities (KG nodes) in one "
            "call (Context-based). Speaks the unified Context object: "
            "queries drive Channel A multi-view cosine, keywords drive Channel C "
            "(caller-provided exact terms — no auto-extraction), entities seed "
            "Channel B graph BFS. Cross-collection Reciprocal Rank Fusion across "
            "all channels. Each result carries source='memory'|'entity' with "
            "type-specific fields (memories: text; entities: name/kind/"
            "description/edges). Unlike kg_query (exact entity ID), this "
            "fuzzy-matches across your whole memory palace."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "context": {
                    "type": "object",
                    "description": (
                        "MANDATORY Context fingerprint for the query.\n"
                        "  queries:  list[str] (2-5)  perspectives — each becomes a cosine view.\n"
                        "  keywords: list[str] (2-5)  caller-provided exact terms (no auto-extract).\n"
                        "  entities: list[str] (0+)   graph BFS seeds (defaults to top cosine hits).\n"
                        'Example: context={"queries": ["deployment process", "release pipeline"], '
                        '"keywords": ["deploy", "release", "rollout"]}'
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
                "limit": {
                    "type": "integer",
                    "description": "Max results across memories+entities (default 10; adaptive-K may trim if scores drop off).",
                },
                "kind": {
                    "type": "string",
                    "description": "Optional entity kind filter. When set, scopes to entities only.",
                    "enum": ["entity", "predicate", "class", "literal"],
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["hybrid", "similarity"],
                    "description": "'hybrid' (default) = RRF + hybrid_score. 'similarity' = pure cosine.",
                },
                "agent": {
                    "type": "string",
                    "description": "Your agent entity name for affinity scoring.",
                },
                "time_window": {
                    "type": "object",
                    "description": (
                        "optional temporal scoping. Items inside the window "
                        "get a scoring boost; items outside still appear but rank lower "
                        "(soft decay, NOT a hard filter). Example: "
                        '{"start": "2026-04-15", "end": "2026-04-17"}'
                    ),
                    "properties": {
                        "start": {
                            "type": "string",
                            "description": "Start date (YYYY-MM-DD). Items after this date get boosted.",
                        },
                        "end": {
                            "type": "string",
                            "description": "End date (YYYY-MM-DD). Items before this date get boosted.",
                        },
                    },
                },
            },
            "required": ["context", "agent"],
        },
        "handler": tool_kg_search,
    },
    "mempalace_kg_add": {
        "description": (
            "Add a fact to the knowledge graph (Context mandatory). "
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
                "agent": {
                    "type": "string",
                    "description": (
                        "MANDATORY — your declared agent entity name (is_a agent). "
                        "Every write operation is attributed to a declared agent; "
                        "undeclared agents are rejected up front with a declaration recipe."
                    ),
                },
                "valid_from": {
                    "type": "string",
                    "description": "When this became true (YYYY-MM-DD, optional)",
                },
            },
            "required": ["subject", "predicate", "object", "context", "agent"],
        },
        "handler": tool_kg_add,
    },
    "mempalace_kg_add_batch": {
        "description": (
            "Add multiple KG edges in one call (Context mandatory). Pass a "
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
                "agent": {
                    "type": "string",
                    "description": ("MANDATORY — declared agent attributing the whole batch."),
                },
            },
            "required": ["edges", "agent"],
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
                "agent": {
                    "type": "string",
                    "description": "MANDATORY — declared agent invalidating this fact.",
                },
            },
            "required": ["subject", "predicate", "object", "agent"],
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
            "the unified Context object. Each query is embedded as a "
            "separate Chroma vector; collision detection runs multi-view RRF; "
            "caller-provided keywords go into the keyword index; the Context's "
            "view vectors are persisted so future feedback applies by similarity.\n\n"
            "Kinds:\n"
            "  'entity'    — concrete thing.\n"
            "  'class'     — category definition (other entities is_a this).\n"
            "  'predicate' — relationship type for kg_add edges.\n"
            "  'literal'   — raw value.\n"
            "  'record'    — prose record. Requires slug + `content` "
            "(verbatim text) + `added_by`. `name` is auto-computed. "
            "Use `entity`+`predicate` to link the record to another entity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Entity name (will be normalized). REQUIRED for kind=entity/class/predicate/literal. OMIT for kind='record' — the record id is computed from added_by + slug.",
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
                    "description": "Ontological role: 'entity' (concrete thing), 'class' (category), 'predicate' (relationship type), 'literal' (raw value), 'record' (requires slug + content + added_by).",
                    "enum": ["entity", "predicate", "class", "literal", "record"],
                },
                "content": {
                    "type": "string",
                    "description": "Verbatim text — REQUIRED for kind='record' (the actual memory body). Ignored for other kinds (queries[0] is the canonical description).",
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
                # ── kind='record' specific ──
                "slug": {
                    "type": "string",
                    "description": "REQUIRED when kind='record'. 3-6 hyphenated words, unique per agent.",
                },
                "content_type": {
                    "type": "string",
                    "description": "Content classification for kind='record'.",
                    "enum": [
                        "fact",
                        "event",
                        "discovery",
                        "preference",
                        "advice",
                        "diary",
                    ],
                },
                "source_file": {
                    "type": "string",
                    "description": "Optional source attribution for kind='record'.",
                },
                "entity": {
                    "type": "string",
                    "description": "Entity name(s) to link this record to (comma-separated).",
                },
                "predicate": {
                    "type": "string",
                    "description": "Predicate for the entity→record link (default: described_by).",
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
            "Unified update for any entity (record or KG node). Pass only the fields "
            "you want to change.\n\n"
            "FOR ENTITIES (kind=entity/class/predicate/literal):\n"
            "  - description: re-syncs to entity ChromaDB and runs collision distance check.\n"
            "  - properties: shallow-merged into existing properties. For predicates, "
            'use {"constraints": {...}} to update predicate constraints (validated).\n'
            "  - importance: 1-5.\n\n"
            "FOR RECORDS (kind='record'):\n"
            "  - content_type: in-place classification change (no re-embedding).\n"
            "  - importance: in-place importance change.\n"
            "  - description: NOT supported — use kg_delete_entity + kg_declare_entity "
            "to replace record content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "Entity ID or record ID (record_/diary_ prefix routes to record collection).",
                },
                "description": {
                    "type": "string",
                    "description": "New description (entities only). Triggers collision distance check.",
                },
                "importance": {
                    "type": "integer",
                    "description": "New importance 1-5 (works for both entities and memories).",
                    "minimum": 1,
                    "maximum": 5,
                },
                "properties": {
                    "type": "object",
                    "description": 'Properties to merge into the entity. For predicates, use {"constraints": {"subject_kinds": [...], "object_kinds": [...], "subject_classes": [...], "object_classes": [...], "cardinality": "..."}}.',
                },
                "context": {
                    "type": "object",
                    "description": (
                        "OPTIONAL Context to re-anchor the entity when description or "
                        "properties semantically change. Same shape as in "
                        "kg_declare_entity. When provided, persists a new creation_context_id "
                        "so future MaxSim feedback attaches to the updated meaning."
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
                "content_type": {
                    "type": "string",
                    "description": "(Records only) Content type classification.",
                    "enum": [
                        "fact",
                        "event",
                        "discovery",
                        "preference",
                        "advice",
                        "diary",
                    ],
                },
                "agent": {
                    "type": "string",
                    "description": "MANDATORY — declared agent attributing this update.",
                },
            },
            "required": ["entity", "agent"],
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
                "agent": {
                    "type": "string",
                    "description": "MANDATORY — declared agent attributing this merge.",
                },
            },
            "required": ["source", "target", "agent"],
        },
        "handler": tool_kg_merge_entities,
    },
    # mempalace_kg_update_predicate_constraints removed: merged into
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
                        "The intent type to declare (must be is_a intent_type). "
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
                "context": {
                    "type": "object",
                    "description": (
                        "MANDATORY Context fingerprint for this intent (replaces "
                        "the legacy `descriptions` parameter).\n"
                        "  queries:  list[str] (2-5)  perspectives on what you're about to do.\n"
                        "             Each becomes a separate cosine view for retrieval.\n"
                        "             queries[0] is the canonical description for auto-narrowing.\n"
                        "  keywords: list[str] (2-5)  caller-provided exact terms — drives the\n"
                        "             keyword channel (no auto-extraction).\n"
                        "  entities: list[str] (0+)   related/seed entities for graph BFS\n"
                        "             (defaults to slot entities when omitted).\n"
                        'Example: context={"queries": ["Editing auth rate limiter", '
                        '"Security hardening against brute force", "Adding tests for login endpoint"], '
                        '"keywords": ["auth", "rate-limit", "brute-force", "login"], '
                        '"entities": ["LoginService"]}'
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
            "required": ["intent_type", "slots", "context", "agent", "budget"],
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
            "MANDATORY when conflicts are returned by kg_add or kg_declare_entity (including kind='record'). "
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
                            "reason": {
                                "type": "string",
                                "description": (
                                    "MANDATORY — why you chose this action (minimum 15 characters). "
                                    "Each conflict is a unique semantic decision. Evaluate individually. "
                                    "Bulk-identical reasons across 3+ conflicts will be rejected as laziness."
                                ),
                            },
                        },
                        "required": ["id", "action", "reason"],
                    },
                    "description": "List of conflict resolution actions.",
                },
                "agent": {
                    "type": "string",
                    "description": "MANDATORY — declared agent resolving these conflicts.",
                },
            },
            "required": ["actions", "agent"],
        },
        "handler": tool_resolve_conflicts,
    },
    "mempalace_resolve_enrichments": {
        "description": (
            "Resolve pending graph enrichment tasks. Enrichments are NOT conflicts — "
            "they are tasks requiring you to create edges between related entities. "
            "For each: first call kg_add(subject, predicate, object) with a predicate "
            "you choose from the declared predicates list, then mark as 'done' here. "
            "Or 'reject' with a mandatory reason (min 15 chars) if the entities should "
            "NOT be connected. Tools are BLOCKED until ALL enrichments are resolved."
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
                                "description": "Enrichment ID from the pending enrichments list.",
                            },
                            "action": {
                                "type": "string",
                                "enum": ["done", "reject"],
                                "description": (
                                    "done: edge was created via kg_add. "
                                    "reject: entities should NOT be connected (reason required)."
                                ),
                            },
                            "reason": {
                                "type": "string",
                                "description": "MANDATORY for reject — why these entities should NOT be connected (min 15 chars).",
                            },
                        },
                        "required": ["id", "action"],
                    },
                    "description": "List of enrichment resolution actions.",
                },
                "agent": {
                    "type": "string",
                    "description": "MANDATORY — declared agent resolving these enrichments.",
                },
            },
            "required": ["actions", "agent"],
        },
        "handler": tool_resolve_enrichments,
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
            "targets, result memory, gotchas, execution trace, and memory relevance feedback. "
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
                    "description": "What happened — the result narrative. Becomes a memory.",
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
                            "id": {"type": "string", "description": "Memory ID or entity ID"},
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
                                "description": (
                                    "Controls whether this feedback propagates to future declares. "
                                    "true = attach to the intent TYPE — future declares of the same type "
                                    "read this signal via _relevance_boost and rerank accordingly. "
                                    "false (default) = attach only to this execution entity — the signal is "
                                    "INVISIBLE to future declares and effectively diary-only. "
                                    "Set true when the rating should generalize (clearly relevant / clearly "
                                    "irrelevant signals that apply whenever this intent-type runs again); "
                                    "leave false only when the signal is genuinely instance-specific "
                                    "(e.g. a memory matched by coincidence of timing, not by task shape)."
                                ),
                            },
                            "reason": {
                                "type": "string",
                                "description": (
                                    "MANDATORY — why this memory was or wasn't relevant to THIS intent "
                                    "(minimum 10 characters). Evaluate each memory individually."
                                ),
                            },
                        },
                        "required": ["id", "relevant", "reason"],
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
                    "description": "Lesson descriptions worth remembering. Each becomes a memory.",
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
    "mempalace_kg_delete_entity": {
        "description": "Delete an entity (record or KG node) and invalidate every current edge touching it. Works for both records (ids starting with 'record_' / 'diary_') and KG entities. Use this when an entity is TRULY obsolete. For stale single facts (one relationship untrue while entity stays valid), use kg_invalidate on that specific (subject, predicate, object) triple instead.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "ID of the entity or record to delete.",
                },
                "agent": {
                    "type": "string",
                    "description": "MANDATORY — declared agent attributing this deletion.",
                },
            },
            "required": ["entity_id", "agent"],
        },
        "handler": tool_kg_delete_entity,
    },
    "mempalace_wake_up": {
        "description": (
            "Return L0 (identity) + L1 (importance-ranked essential story) wake-up "
            "text (~600-900 tokens total). Call this ONCE at session start to load "
            "project/agent boot context. Also returns the protocol, declared entities/"
            "predicates/intent types — everything you need to start. L1 is ranked "
            "with importance-weighted time decay — critical facts always surface first, "
            "within-tier newer wins."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "Agent identity (required). Used for affinity scoring in L1 and auto-derives the memory scope. Use your agent entity name (e.g., 'ga_agent', 'technical_lead_agent').",
                },
            },
            "required": ["agent"],
        },
        "handler": tool_wake_up,
    },
    # mempalace_update_drawer_metadata removed: merged into mempalace_kg_update_entity.
    "mempalace_diary_write": {
        "description": "Write to your personal agent diary. Your observations, thoughts, what you worked on, what matters. Each agent has their own diary with full history. Optional content_type/importance for special entries.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Your name — each agent gets their own diary",
                },
                "entry": {
                    "type": "string",
                    "description": "Your diary entry — readable prose.",
                },
                "slug": {
                    "type": "string",
                    "description": "Descriptive identifier for this diary entry (e.g. 'session12-intent-narrowing-shipped', 'migration-lesson-learned'). Used as part of the entry ID.",
                },
                "topic": {
                    "type": "string",
                    "description": "Topic tag (optional, default: general)",
                },
                "content_type": {
                    "type": "string",
                    "description": "Override the default diary classification. Use discovery for 'today I learned' entries worth higher retrieval priority, event for plain activity logs.",
                    "enum": [
                        "fact",
                        "event",
                        "discovery",
                        "preference",
                        "advice",
                        "diary",
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
        "description": "Read your recent diary entries. See what past versions of yourself recorded — your journal across sessions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Your name — each agent gets their own diary",
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
        # Extract sessionId injected by the PreToolUse hook via
        # hookSpecificOutput.updatedInput. Verified to propagate
        # correctly (measured 2026-04-19 via SID_PROBE).
        #
        # Sanitize identically to the hook so file names match.
        #
        # NO FALLBACK if it's empty. We refuse to synthesize
        # "default" / "unknown" — that would silently merge every
        # agent's state into a shared file. When sid is unknown we
        # simply don't switch; downstream state operations will
        # themselves refuse to read/write (see _intent_state_path and
        # _load_pending_from_disk).
        injected_session_id = tool_args.pop("sessionId", None)
        if injected_session_id:
            new_sid = _sanitize_session_id(injected_session_id)
            if new_sid and new_sid != _STATE.session_id:
                _save_session_state()
                _STATE.session_id = new_sid
                _restore_session_state(new_sid)
                if _STATE.active_intent:
                    try:
                        intent._persist_active_intent()
                    except Exception:
                        pass

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


def _run_hyphen_id_migration_once():
    """Rename legacy hyphenated IDs (Chroma + SQLite) to canonical form.

    Idempotent, one-pass per process. Gated on ``_STATE.hyphen_ids_migrated``.
    Uses ``normalize_entity_name`` as the single source of truth so the
    post-migration invariant ``stored_id == normalize(id)`` holds across
    every collection and table.
    """
    try:
        from . import hyphen_id_migration
        from .knowledge_graph import normalize_entity_name

        stats = hyphen_id_migration.run_migration(
            _STATE,
            chroma_record_col=_get_collection(create=False),
            chroma_entity_col=_get_entity_collection(create=False),
            chroma_feedback_col=_get_feedback_context_collection(create=False),
            normalize=normalize_entity_name,
        )
        if not stats.get("skipped"):
            logger.info(f"N3 hyphen-id migration completed: {stats}")
    except Exception as e:
        logger.warning(f"N3 hyphen-id migration failed: {e}")


def main():
    logger.info("MemPalace MCP Server starting...")
    # run the kind='record' → 'record' migration once at startup.
    # Idempotent, no-op on fresh palaces or on second invocation within
    # the process.
    try:
        _migrate_kind_memory_to_record()
    except Exception as e:
        logger.warning(f"P6.2 startup kind migration failed: {e}")
    # N3 hyphen-id migration — rename legacy hyphenated identifiers to
    # the canonical underscored form enforced by normalize_entity_name.
    _run_hyphen_id_migration_once()
    # M1 collection merge — absorb legacy mempalace_entities rows into
    # the unified mempalace_records collection and drop the legacy one.
    try:
        _migrate_entities_collection_into_records()
    except Exception as e:
        logger.warning(f"M1 startup collection-merge failed: {e}")
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
