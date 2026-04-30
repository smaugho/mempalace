#!/usr/bin/env python3
"""
mempalace/intent.py -- Intent declaration, active-intent tracking, and finalization.

Extracted from mcp_server.py. Uses a module-reference pattern to access
mcp_server globals without circular imports.
"""

import hashlib
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from .knowledge_graph import normalize_entity_name

# Module reference (set by init())
_mcp = None


# ── Cop-out reason detection -- semantic-similarity side of the hybrid gate ──
# Regex catches the obvious literal forms ("don't know", "N/A", "never used")
# but agents can evade with rephrasings ("lack of information to evaluate",
# "cannot determine relevance", "unclear to me whether this was useful") that
# are semantically identical cop-outs. Second-pass embedding check catches
# those by measuring cosine similarity against a small set of exemplars that
# span the cop-out intent space. Threshold 0.70 tuned conservatively -- better
# to let a borderline reason through than to reject a genuine short-but-real
# rating.
#
# Exemplars seeded from reasons I (the agent) caught myself writing during
# the 2026-04-24 session plus typical paraphrases. Extend as patterns are
# discovered post-deploy; cache invalidates at process restart.
_COPOUT_EXEMPLARS = [
    "I don't know what this memory contains, cannot evaluate",
    "I never used this memory, did not engage with it",
    "Not relevant, skipped without reading the content",
    "Aborted before running any verification, no rating available",
    "Placeholder rating, unclear whether the memory was useful",
    "Cannot determine relevance, no idea what this is about",
    "Did not fetch the memory content, N/A for this intent",
    "Unable to assess, insufficient context to rate fairly",
    "Skipped evaluation, moving past without reading",
]
_COPOUT_EMB_CACHE: list | None = None  # lazy-populated on first _semantic_copout_check call
_COPOUT_SIM_THRESHOLD = 0.70


# Regex side of the hybrid gate. Promoted to module level (2026-04-24)
# so tests can import and assert pattern behaviour directly without
# calling the full finalize pipeline. Patterns are narrow and literal
# by design -- only cheap-and-obvious cop-outs here; semantic similarity
# handles paraphrased evasions. Adding a pattern that false-positives
# on compound nouns (e.g. \bskip(ped)?\b hitting "skip-list") is a
# bug -- prefer standalone-verb forms like \bskipped\b.
_LOW_QUALITY_REASON_PATTERNS = [
    r"\bdon'?t know\b",
    r"\bnot used\b",
    r"\bnever used\b",
    r"\bdidn'?t use\b",
    r"\bnot sure\b",
    r"\bn\.?\s*/?\s*a\b",
    r"\bno idea\b",
    r"\bnot applicable\b",
    r"\baborted\b.*\brunning\b",
    r"\bnot rated\b",
    # Narrow to the verb "skipped" only -- standalone "skip" matches
    # legitimate data-structure terms (skip-list, skip-gram, etc.)
    # which ARE valid content references in rating reasons.
    r"\bskipped\b",
    r"^\s*(unclear|unknown|placeholder|tbd|todo)\s*$",
]
_LOW_QUALITY_RE = re.compile("|".join(_LOW_QUALITY_REASON_PATTERNS), re.IGNORECASE)


def _regex_copout_check(reason_text: str) -> bool:
    """Regex fast-path of the hybrid cop-out gate. Returns True when the
    reason matches any literal-cop-out pattern. Tests import this to
    assert pattern coverage without calling the full finalize pipeline.
    """
    if not isinstance(reason_text, str):
        return False
    return bool(_LOW_QUALITY_RE.search(reason_text))


def _semantic_copout_check(reason_text: str) -> tuple[bool, float]:
    """Hybrid-gate second pass: cosine similarity of reason vs cop-out exemplars.

    Returns ``(is_copout, max_similarity)``. Caches exemplar embeddings so the
    per-call cost is one embedder forward (reason only). Fail-open: any
    exception yields ``(False, 0.0)`` so a broken embedder can't block
    finalize -- the regex gate still catches the obvious cases.
    """
    global _COPOUT_EMB_CACHE
    text = (reason_text or "").strip()
    if not text:
        return False, 0.0
    try:
        from chromadb.utils import embedding_functions as ef

        efunc = ef.DefaultEmbeddingFunction()
        if efunc is None:
            return False, 0.0
        if _COPOUT_EMB_CACHE is None:
            vecs = efunc(_COPOUT_EXEMPLARS)
            _COPOUT_EMB_CACHE = [list(float(x) for x in v) for v in vecs]
        reason_vec = list(float(x) for x in efunc([text])[0])
        na = math.sqrt(sum(x * x for x in reason_vec))
        if na == 0:
            return False, 0.0
        max_sim = 0.0
        for ex_vec in _COPOUT_EMB_CACHE:
            nb = math.sqrt(sum(x * x for x in ex_vec))
            if nb == 0:
                continue
            dot = sum(x * y for x, y in zip(reason_vec, ex_vec))
            sim = dot / (na * nb)
            if sim > max_sim:
                max_sim = sim
        return max_sim >= _COPOUT_SIM_THRESHOLD, max_sim
    except Exception:
        # Fail-open: embedder down must not block finalize.
        return False, 0.0


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


DEBUG_RETURN_SCORES = _env_flag_on("MEMPALACE_DEBUG_RETURN_SCORES", False)
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


# \u2500\u2500 Summary co-render helpers (Slice 1a 2026-04-28) \u2500\u2500
# Used by finalize_intent's error renderers (missing_injected / missing_accessed)
# so the model sees what each missing reference is ABOUT alongside the bare id.
# Without this, callers had to do a follow-up kg_query per missing id just to
# rate it. The principle: "ids are pointers, summaries are meaning - never
# surface a pointer without its meaning, except in structured fields where the
# caller can look it up themselves." Renderer co-render is the cheapest way to
# honour that without bloating storage.
_SUMMARY_PREVIEW_MAX_CHARS = 80


def _short_summary_for_id(memory_id: str, max_len: int = _SUMMARY_PREVIEW_MAX_CHARS):
    """Resolve a short human-readable summary for a memory id.

    Falls back to the entity's representative Chroma ``content`` field
    (the rendered prose of summary.what + why), takes the first sentence,
    and caps at ``max_len`` chars. Returns None on lookup failure so
    callers can omit the field cleanly - typical for ephemeral context
    ids (ctx_*) that aren't first-class entities.
    """
    if not memory_id or _mcp is None:
        return None
    try:
        details = _mcp._fetch_entity_details(memory_id)
    except Exception:
        return None
    if not details:
        return None
    content = details.get("content") or ""
    if not content:
        return None
    # First sentence (period+space delimiter); fall back to whole content if
    # no sentence boundary. Then length-cap with ellipsis.
    first = content.split(". ", 1)[0]
    if len(first) > max_len:
        first = first[: max_len - 1].rstrip() + "\u2026"
    return first


def _enrich_ids_with_summaries(ids):
    """Map a list of memory_ids -> list of {"id": <id>, "what": <summary?>}.

    Unknown ids get ``what: None`` so the caller can still tell the id
    is missing AND that no summary was resolvable. Used by finalize_intent
    error rendering at the missing_injected / missing_accessed sites.
    """
    return [{"id": mid, "what": _short_summary_for_id(mid)} for mid in ids]


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
    deadlock -- it collected every agent's pending state into one file,
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
    importance, added_by -- plus context_rank / context_score when a
    Context is supplied.

    Context-ranked hierarchy. When the caller passes the active
    intent's Context (queries + keywords), we re-use the SAME 3-channel
    pipeline the rest of the palace uses (scoring.multi_channel_search
    against the entity collection with kind='class') to rank intent
    types by semantic similarity to what the agent is actually doing.
    The ranking is baked into the hierarchy entries and persisted to the
    session state file, so the PreToolUse hook -- which must stay
    dep-free (no ChromaDB, no Torch) -- reads a pre-sorted list with
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
    # '{entity_id}__v{N}' -- dedupe by logical entity_id from metadata so
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
    """Safe wrapper -- never crashes, returns [] on any error."""
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
                # 2026-04-29: refresh op_args_by_ctx_tool so the lookup at
                # finalize promotion sees args declared in any earlier turn
                # of the same intent. Disk is authoritative since the hook
                # subprocess may have written between our last sync.
                _mcp._STATE.active_intent["op_args_by_ctx_tool"] = dict(
                    data.get("op_args_by_ctx_tool") or {}
                )
            return

        # Cold-hydration path: memory empty, disk has a live intent. Rebuild
        # the full active_intent dict so the next finalize can find it.
        _mcp._STATE.active_intent = {
            "intent_id": data["intent_id"],
            "intent_type": data.get("intent_type", ""),
            "slots": data.get("slots", {}),
            "effective_permissions": data.get("effective_permissions", []),
            "content": data.get("content", ""),
            "agent": data.get("agent", ""),
            "injected_memory_ids": set(data.get("injected_memory_ids", []) or []),
            "injected_by_context": dict(data.get("injected_by_context", {}) or {}),
            "accessed_memory_ids": set(data.get("accessed_memory_ids", []) or []),
            "budget": data.get("budget", {}),
            "used": data.get("used", {}),
            "intent_hierarchy": data.get("intent_hierarchy", []),
            "active_context_id": data.get("active_context_id", "") or "",
            "contexts_touched": list(data.get("contexts_touched") or []),
            "contexts_touched_detail": list(data.get("contexts_touched_detail") or []),
            # 2026-04-29: cold-hydration must also restore op_args_by_ctx_tool
            # so finalize promotion can recover args_summary for ops declared
            # before a server restart. Without this, every cold-restored
            # intent loses its op-args store and finalizes with empty args.
            "op_args_by_ctx_tool": dict(data.get("op_args_by_ctx_tool") or {}),
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
            # Fully clean state -- nothing to persist.
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
                "content": _mcp._STATE.active_intent.get("content", ""),
                "agent": _mcp._STATE.active_intent.get("agent", ""),
                "session_id": _mcp._STATE.session_id,
                "intent_hierarchy": cached_hierarchy,
                "injected_memory_ids": list(
                    _mcp._STATE.active_intent.get("injected_memory_ids", set())
                ),
                "injected_by_context": dict(
                    _mcp._STATE.active_intent.get("injected_by_context", {})
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
                # message, N tool calls follow -- each consumes its own cue
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
                # 2026-04-29: persist op_args_by_ctx_tool across turn
                # boundaries. This dict maps "{context_id}|{tool}" →
                # parametrized args_summary, populated by
                # tool_declare_operation and consumed by finalize_intent
                # promotion. Without persistence, every op declared in a
                # prior turn lands with empty args_summary at finalize
                # because the in-memory state was lost. Audit on
                # 2026-04-29 found 224 post-mandate ops leaked this way.
                "op_args_by_ctx_tool": dict(
                    _mcp._STATE.active_intent.get("op_args_by_ctx_tool") or {}
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
    - tool_permissions: ADDITIVE -- child tools are merged with parent tools.
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

        # Tool permissions: ADDITIVE -- collect from all levels, child + parent
        for perm in profile.get("tool_permissions", []):
            tool_key = perm.get("tool", "")
            if tool_key not in seen_tools:
                seen_tools.add(tool_key)
                merged_tools.append(perm)

        # Walk to parent via is-a -- prefer intent hierarchy over universal "thing"
        edges = _mcp._STATE.kg.query_entity(current, direction="outgoing")
        parent = None
        for e in edges:
            if e["predicate"] == "is_a" and e["current"]:
                parent_id = normalize_entity_name(e["object"])
                # Stop at the root intent_type class
                if parent_id == "intent_type":
                    break
                # Skip universal base class -- not part of intent hierarchy
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
            # Check parent (one level -- e.g., edit_file is-a modify is-a intent_type)
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
    auto_declare_files: bool = False,
    agent: str = None,
    budget: dict = None,
    cause_id: str = None,  # Slice B-3: optional parent cause (user-ctx OR Task)
):
    """Declare what you intend to do BEFORE doing it. Returns permissions + context.

    budget: MANDATORY dict of tool_name -> max_calls. E.g. {"Read": 5, "Edit": 3}.
            Must cover all tools you plan to use. Budget is tracked by the hook --
            when exhausted, the tool is blocked until you extend (mempalace_extend_intent)
            or finalize and redeclare. Keep budgets tight -- inflated budgets waste context.

    One active intent at a time -- declaring a new intent expires the previous.
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
            multiple (bool -- accepts list vs single entity).

        context: MANDATORY Context fingerprint for this intent.
            {
              "queries":  list[str]   2-5 perspectives on what you're about to do
              "keywords": list[str]   2-5 caller-provided exact terms
              "entities": list[str]   0+ related/seed entity ids (defaults to slot
                                      entities when omitted -- they ARE the entities
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

    from .scoring import validate_context as _validate_context

    clean_context, ctx_err = _validate_context(
        context,
        require_summary=True,
        summary_context_for_error="declare_intent.context.summary",
    )
    if ctx_err:
        return ctx_err
    _description_views = clean_context["queries"]
    _context_keywords = clean_context["keywords"]
    _context_entities = clean_context["entities"]
    # Render context.summary to the canonical description prose. The
    # queries[0] auto-derive that used to live here was retired
    # (Adrian's design lock 2026-04-25) -- same principle as
    # tool_kg_declare_entity: no auto-derive of summary fields.
    from .knowledge_graph import serialize_summary_for_embedding as _serialize_summary

    description = _serialize_summary(clean_context["summary"])

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
    # Disk is source of truth -- reload from disk if memory is empty (MCP restart scenario)
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
                f"invalidate (old is stale), merge (combine -- read both in full first), "
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
                f"Specific intent types are preferred over broad ones -- they carry domain-specific "
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
                f"-- where parent is the broad type it inherits from "
                f"(inspect, modify, execute, or communicate)\n"
                f"  3. Then retry declare_intent with this type.\n"
                f"This is a one-time cost -- once created, the type persists across sessions "
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
    # Only kind=class -- execution instances (kind=entity) are NOT subtypes
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
                            "content": e.get("content", ""),
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
                                        "content": results["documents"][0][i],
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
                            # Multiple children beat the parent -- disambiguate
                            return {
                                "success": False,
                                "error": (
                                    f"Description matches multiple subtypes of '{intent_id}' "
                                    f"better than '{intent_id}' itself. "
                                    f"Pick the most appropriate one and declare it directly."
                                ),
                                "matching_subtypes": [
                                    {"id": c["id"], "content": c["content"][:120]}
                                    for c in compatible
                                ],
                            }
            except Exception:
                child_scores = []  # Non-fatal -- narrowing is best-effort

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
        # Kind-aware error: tell the writer (a) what type the validator
        # actually received (so they can spot MCP transport stringification
        # of dict args), and (b) per-slot example values that match each
        # slot's KIND -- raw glob/command strings, file paths, or
        # pre-declared entity names. The legacy template used
        # ["entity_name"] for every slot regardless of kind, which
        # actively misled callers into passing entity ids for raw `paths`
        # or `commands` slots.
        def _slot_example(name, sd):
            if sd.get("raw", False):
                if name == "paths":
                    return '["D:/Flowsev/repo/**"]'
                if name == "commands":
                    return '["pytest", "git status"]'
                return '["raw_string"]'
            classes = sd.get("classes", ["thing"])
            if "file" in classes:
                return '["src/auth.py"]'
            return '["my_entity_name"]'

        def _slot_legend(name, sd):
            if sd.get("raw", False):
                return f"{name}=raw string"
            classes = sd.get("classes", ["thing"])
            if "file" in classes:
                return f"{name}=file path (auto-declared)"
            return f"{name}=pre-declared entity (classes: {classes})"

        legend = ", ".join(_slot_legend(k, v) for k, v in effective_slots.items())
        example_body = ", ".join(
            f'"{k}": {_slot_example(k, v)}' for k, v in effective_slots.items()
        )
        return {
            "success": False,
            "error": (
                f"slots must be a JSON object/dict, received "
                f"{type(slots).__name__}. "
                f"(If you passed a dict and still see this, the MCP "
                f"transport may have stringified it -- re-fetch the "
                f"declare_intent schema via ToolSearch and retry.) "
                f"Expected slots for '{intent_id}': {legend}. "
                f"Example: {{{example_body}}}"
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

            # ── Slice 6 design-lock 2026-04-28 ──
            # This block is the SOLE entity auto-naming surface in mempalace.
            # Per the 2026-04-26/28 id-design discussion (Adrian): every
            # entity must carry a model/user-authored name; the only
            # exception is file entities whose name is the basename of an
            # already-known file path (deterministic, structural, not a
            # noun-phrase guess from prose). Other paths confirmed
            # auto-naming-free by the 2026-04-28 codebase audit:
            #   - kg_add / kg_add_batch: reject undeclared subject/object
            #     with a structured "declare first" error (tool_mutate.py).
            #   - Mining (miner.py): pure 800-char chunking, no entity
            #     extraction.
            #   - Gardener (link_author.py): operates on existing entity
            #     names; the LLM jury authors edges, not entity names.
            #   - Intent slot resolution: rejects unknown entity names
            #     except file slots routed through THIS block.
            #   - sanitize_name (config.py): validates shape via
            #     _SAFE_NAME_RE without auto-trimming; raises ValueError
            #     on bad shape rather than silently mangling.
            # Adding any new auto-naming codepath violates the design
            # lock -- route through kg_declare_entity with a model-authored
            # name instead, OR document the new structural-derivation rule
            # here alongside the file-basename case.
            # Auto-declare file entities if slot expects class=file
            if not _mcp._is_declared(val_id) and is_file_slot:
                file_exists = os.path.exists(val) or os.path.exists(os.path.join(os.getcwd(), val))
                if file_exists or auto_declare_files:
                    # Auto-declare: create entity from basename + is-a file.
                    #
                    # The legacy "File: <path>" prose stub was retired
                    # (Adrian's design lock 2026-04-25): all descriptions
                    # follow the strict dict-only WHAT+WHY+SCOPE? shape,
                    # validated by validate_summary. The auto-mint path
                    # cannot ask the writer for a real WHAT/WHY (slot
                    # validation runs before the writer sees the prompt),
                    # so we emit a STRUCTURED placeholder dict that
                    # passes validate_summary on field-level shape and
                    # carries an explicit "needs refinement" why-clause
                    # plus the source path for the gardener.
                    #
                    # Storage stores the rendered prose (via
                    # serialize_summary_for_embedding); the dict shape is
                    # honoured at the validation gate. The gardener
                    # picks up the generic_summary flag below and
                    # rewrites the description from the file's first
                    # docstring or sibling signals.
                    _auto_summary_dict = {
                        "what": file_basename,
                        "why": (
                            "auto-declared file entity at slot-validation "
                            "time; placeholder pending gardener refinement "
                            "from docstring/sibling signals"
                        ),
                        "scope": (f"source path {val}" + (" (new)" if not file_exists else ""))[
                            :100
                        ],
                    }
                    try:
                        from .knowledge_graph import (
                            serialize_summary_for_embedding,
                            validate_summary,
                        )

                        validate_summary(
                            _auto_summary_dict,
                            context_for_error="auto_declare_files.summary",
                        )
                        _auto_desc = serialize_summary_for_embedding(_auto_summary_dict)
                    except Exception:
                        # Final safety: never block slot validation on
                        # an auto-mint summary failure; fall back to a
                        # short marker. The gardener's generic_summary
                        # flag below still fires.
                        _auto_desc = (
                            f"{file_basename} -- auto-declared file entity pending refinement"
                        )
                    _mcp._create_entity(
                        file_basename,
                        kind="entity",
                        content=_auto_desc,
                        importance=2,
                        properties={"file_path": val},
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
                                        "structured placeholder. Replace with a real "
                                        "{what, why, scope?} dict -- what this file "
                                        "does and why it exists -- drawn from the "
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
        # Check if this is a raw slot (commands, etc.) -- don't add to entity list
        slot_def = effective_slots.get(slot_name, {})
        if slot_def.get("raw", False):
            raw_slot_names.add(slot_name)
        else:
            all_slot_entities.extend(flat_slots[slot_name])

    def _resolve_file_path(entity_id):
        """Resolve actual file path for a file entity.

        Checks entity properties for 'file_path', then falls back to
        extracting the path from the description (format: 'path/to/file.py -- ...')
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
        # Fall back to description -- extract path from known formats
        desc = entity.get("content", "")
        # Format: "File: /path/to/file.ext" or "File: /path/to/file.ext (new)"
        if desc.startswith("File: "):
            candidate = desc[6:].split("(")[0].strip()
            if "/" in candidate or "\\" in candidate:
                return candidate
        # Format: "path/to/file.py -- description text"
        for sep in (" -- ", " - ", " - "):
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
        # Check if this slot contains file entities -- resolve actual paths
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
                "Keep budgets tight -- estimate the minimum calls needed for this task."
            ),
        }
    # Validate budget: only keep tools that are actually permitted
    permitted_tool_names = {p["tool"] for p in permissions}
    validated_budget = {}
    for tool_name, count in budget.items():
        if tool_name not in permitted_tool_names:
            continue  # Silently ignore -- permission check blocks anyway
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
    # returns 0 (no signal) -- retrieval runs AFTER the populate step anyway.
    _context_feedback: dict = {}

    def _relevance_boost(memory_id):
        """Return continuous relevance signal from context feedback.

        Returns float in [-1.0, +1.0]. Feeds hybrid_score as the signed
        relevance_feedback term -- rated_irrelevant memories drop below
        neutral, rated_useful rise above.
        """
        return _context_feedback.get(memory_id, 0.0)

    def _preview(entity_id_or_memory):
        """Get text preview for any ID -- memory content or entity description."""
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
                if ent and ent.get("content"):
                    return ent["content"][:150].replace("\n", " ")
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
            if ent and ent.get("content"):
                _views.append(ent["content"][:200])
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
            summary=clean_context.get("summary"),
        )
        _active_context_id = _cid or ""
        _active_context_reused = bool(_reused)
    except Exception:
        _active_context_id = ""

    # ── Slice B-3: cause_id validation + caused_by edge ─────────────
    # Optional parent-cause linkage: when cause_id is provided, validate
    # it is either (a) a kind='context' entity with at least one
    # fulfills_user_message outgoing edge (i.e. a user-context minted by
    # mempalace_declare_user_intents earlier this turn), or (b) a
    # kind='entity' entity with an is_a Task edge (paperclip / scheduled
    # path). On success, write a caused_by edge from this activity-
    # intent's context to the cause. Telemetry: stash on active_intent
    # so finalize_intent (also Slice B-3) can apply the user-context
    # feedback coverage rule scoped to this cause.
    _resolved_cause_id = ""
    _resolved_cause_kind = ""  # "user_context" or "task"
    # Slice B-4b first-rater snapshot defaults -- populated only on the
    # cause_kind=='user_context' path below. For Task or no-cause cases
    # they stay at their first-rater=True / no-exemption defaults so the
    # active_intent dict reads them safely.
    _user_ctx_first_rater = True
    _user_ctx_exempt_ids: list = []
    if cause_id and isinstance(cause_id, str) and cause_id.strip():
        _cid_clean = cause_id.strip()
        try:
            _cause_ent = _mcp._STATE.kg.get_entity(_cid_clean)
        except Exception:
            _cause_ent = None
        if not _cause_ent:
            return {
                "success": False,
                "error": (
                    f"cause_id={_cid_clean!r} does not resolve to any entity. "
                    "Pass either a user-context id (returned by "
                    "mempalace_declare_user_intents.contexts[*].ctx_id) "
                    "or a Task entity id (kind='entity', is_a Task)."
                ),
            }
        _cause_kind = _cause_ent.get("kind")
        try:
            _cause_edges = _mcp._STATE.kg.query_entity(_cid_clean, direction="outgoing")
        except Exception:
            _cause_edges = []
        _is_a_targets = {
            e.get("object")
            for e in _cause_edges
            if e.get("predicate") == "is_a" and e.get("current", True)
        }
        _has_fulfills = any(
            e.get("predicate") == "fulfills_user_message" and e.get("current", True)
            for e in _cause_edges
        )
        if _cause_kind == "context" and _has_fulfills:
            _resolved_cause_kind = "user_context"
        elif _cause_kind == "entity" and "Task" in _is_a_targets:
            _resolved_cause_kind = "task"
        else:
            return {
                "success": False,
                "error": (
                    f"cause_id={_cid_clean!r} is not a valid parent cause. "
                    f"Got kind={_cause_kind!r}; expected either kind='context' "
                    f"with at least one fulfills_user_message edge (a user-"
                    f"context from declare_user_intents), or kind='entity' "
                    f"with is_a Task (a Task entity). is_a targets: "
                    f"{sorted(_is_a_targets)}; has_fulfills_user_message: "
                    f"{_has_fulfills}."
                ),
            }
        _resolved_cause_id = _cid_clean

        # Slice B-4b: snapshot first-rater state for cause_kind='user_context'.
        # The FIRST agent intent that finalizes against a given user-context
        # carries full feedback coverage of the user-context's surfaced
        # memories. Subsequent intents with the same cause_id inherit the
        # coverage and are exempt from re-rating those exact memories. We
        # snapshot AT DECLARE TIME (not finalize) so the rating contract is
        # established when the agent commits to the intent -- stable across
        # any later finalize / extend_feedback path.
        _user_ctx_first_rater = True
        _user_ctx_exempt_ids: list = []
        if _resolved_cause_kind == "user_context":
            _sid_for_rated = _mcp._STATE.session_id or ""
            _rated_set = _rated_user_contexts_for(_sid_for_rated)
            if _resolved_cause_id in _rated_set:
                _user_ctx_first_rater = False
                # Look up which memories the user-context surfaced via
                # `surfaced` outgoing edges. Soft-fail: a stale or empty
                # user-context just means no exemptions, which is the
                # safe default (full coverage required).
                try:
                    _ctx_edges = _mcp._STATE.kg.query_entity(
                        _resolved_cause_id,
                        direction="outgoing",
                    )
                    _user_ctx_exempt_ids = sorted(
                        {
                            e.get("object")
                            for e in _ctx_edges
                            if e.get("predicate") == "surfaced"
                            and e.get("current", True)
                            and e.get("object")
                        }
                    )
                except Exception:
                    _user_ctx_exempt_ids = []

        # Write the caused_by edge. The predicate is non-skip-list so
        # add_triple requires a natural-language statement (per the
        # 2026-04-19 lock that retired autogenerated verbalisations).
        # Build a short verbalisation of the parent linkage. Soft-fail
        # at edge level so a transient kg issue doesn't prevent intent
        # declaration entirely; cause_id persists on active_intent
        # regardless so finalize can apply its coverage rule.
        if _active_context_id:
            _caused_by_statement = (
                f"This activity-intent context ({_active_context_id}) "
                f"was caused by {_resolved_cause_kind.replace('_', ' ')} "
                f"{_resolved_cause_id} per the user-intent tier "
                f"hierarchy."
            )
            try:
                _mcp._STATE.kg.add_triple(
                    _active_context_id,
                    "caused_by",
                    _resolved_cause_id,
                    statement=_caused_by_statement,
                )
            except Exception:
                pass

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
    # CHANNELS A+C: Unified retrieval -- BOTH collections.
    # Uses the SAME scoring.multi_channel_search as kg_search. Each
    # collection runs Channels A (multi-view cosine) and C (keyword
    # overlap) internally; results merge into a shared RRF pot with
    # Channel B (graph BFS, below). Entity AND record candidates
    # compete head-to-head for injection -- rules, concepts, gotchas,
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
        else {"rated_scores": {}, "channel_D_list": [], "contributing_contexts": {}}
    )
    _context_feedback = _rated_walk.get("rated_scores") or {}
    # Step 2 of similar_context_id flag (record_ga_agent_similar_context_id_
    # flag_design_2026_04_30): surface neighbour contributions per memory so
    # the agent can monitor which similar_to neighbours contributed to each
    # injected memory. Default-on; the active context is excluded from the
    # map by walk_rated_neighbourhood itself.
    _contributing_contexts = _rated_walk.get("contributing_contexts") or {}

    # Record collection (prose records -- the old "memory" collection)
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

    # Entity collection (structured entities -- rules, concepts, past execs)
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

    # Triple verbalization collection -- surfaces structured (subject, predicate,
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
    # CHANNEL B: Graph -- BFS from slot entities + intent type
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
                # excluded -- same filter as _index_triple_statement uses
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

    # Channel C (keyword) is now built INTO multi_channel_search -- no
    # separate keyword pass needed. The record_pipe and entity_pipe above
    # already include keyword-ranked lists when _context_keywords is non-empty.

    # ══════════════════════════════════════════════════════════════
    # RRF MERGE -- unified across A (cosine) + B (graph) + C (keyword)
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
            # kg_search -- same function, same scale (0.3-0.8).
            entry["hybrid_score"] = round(float(r["hybrid_score"]), 6)
        # Step 2 of similar_context_id flag: surface the similar-context
        # neighbours (NOT the active context) that contributed weight to
        # this memory's Channel D / W_REL score. Default-on. The active
        # context is excluded by walk_rated_neighbourhood itself.
        _neighbour_cids = _contributing_contexts.get(memory_id) or []
        if _neighbour_cids:
            entry["similar_context_ids"] = list(_neighbour_cids)
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
                        f"    kg_update_entity(entity='<exec_id>', summary={{'what': ..., 'why': ...}}, "
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
        prev_desc = _mcp._STATE.active_intent.get("content", "")
        return {
            "success": False,
            "error": (
                f"Active intent '{prev_type}' ({prev_id}) has not been finalized. "
                f"You MUST call mempalace_finalize_intent before declaring a new intent. "
                f"Only the agent knows how to properly summarize what happened.\n\n"
                f"Call: mempalace_finalize_intent(\n"
                f"  slug='<descriptive-slug>',\n"
                f"  outcome='success' | 'partial' | 'failed' | 'abandoned',\n"
                f"  content='<full narrative body -- what happened in detail>',\n"
                f"  summary='<≤280-char distilled one-liner of the outcome>',\n"
                f"  agent='<your_agent_name>'\n"
                f")\n\n"
                f"Previous intent: {prev_type} -- {prev_desc[:100]}"
            ),
            "active_intent": prev_id,
        }

    intent_hash = hashlib.md5(
        f"{intent_id}:{description}:{datetime.now().isoformat()}".encode()
    ).hexdigest()[:12]
    new_intent_id = f"intent_{intent_id}_{intent_hash}"

    # bake a Context-ranked intent_hierarchy ONCE here so the
    # PreToolUse hook has a pre-sorted list and never needs to retrieve.
    # Uses the same 3-channel pipeline as kg_search -- no reinvented
    # similarity math.
    context_for_ranking = {
        "queries": list(_description_views),
        "keywords": list(_context_keywords),
    }
    ranked_hierarchy = _build_intent_hierarchy_safe(context_for_ranking)

    # _memory_scoring_snapshot retired (P3 polish): the weight-learning
    # feedback path now reads signals directly from the sim + rel
    # seen_meta + _context_feedback at finalize time rather than a
    # separate snapshot dict on active_intent. Cleaner -- fewer
    # persistent fields to maintain.
    #
    # _active_context_id was minted earlier (before the retrieval loops
    # so _relevance_boost could consume context-scoped feedback). No
    # second call here.

    # Parallel map to injected_memory_ids that preserves which context
    # surfaced each id. Used by finalize's coverage error to produce a
    # per-context breakdown so the agent can see WHICH context emission
    # an uncovered id came from -- critical when multiple emits happen
    # in one intent and the agent needs to rate each under the right
    # ctx_id key in memory_feedback.
    _injected_by_context = (
        {_active_context_id: sorted(already_injected)}
        if (_active_context_id and already_injected)
        else {}
    )

    _mcp._STATE.active_intent = {
        "intent_id": new_intent_id,
        "intent_type": intent_id,
        "slots": flat_slots,
        "effective_permissions": permissions,
        "injected_memory_ids": already_injected,
        "injected_by_context": _injected_by_context,
        "accessed_memory_ids": set(),
        "_graph_memories_snapshot": dict(_graph_memories),  # distance map for hop-shortening
        "content": description,
        "_context_views": _views,  # multi-view query strings for context vector storage
        "active_context_id": _active_context_id,  # P1 context-as-entity
        # Every context id touched during this intent (intent-level +
        # any operation/search emits). Enumerated at finalize to build
        # the strict coverage set: every (ctx, memory) surfaced pair
        # must have a rated_* edge or finalize is rejected.
        "contexts_touched": [_active_context_id] if _active_context_id else [],
        # Per-emit detail list -- one entry per context emit during the
        # intent's lifecycle. Finalize iterates this to run Rocchio
        # enrichment independently per reused context. Initialised with
        # the intent-level emit; declare_operation + kg_search append
        # their own entries via _record_context_emit.
        "contexts_touched_detail": ([_intent_emit_entry] if _active_context_id else []),
        "agent": agent or "",
        "budget": validated_budget,
        "used": {},  # tool_name -> count, incremented by hook
        "intent_hierarchy": ranked_hierarchy,  # cached, context-ranked
        # Slice B-3: parent-cause linkage. cause_id is the validated
        # entity id from declare_intent (user-context OR Task entity)
        # written to active_intent so finalize_intent can apply the
        # user-context feedback coverage rule scoped to that cause.
        # cause_kind is "user_context" / "task" / "" (none).
        "cause_id": _resolved_cause_id,
        "cause_kind": _resolved_cause_kind,
        # Slice B-4b: first-rater snapshot for cause_kind='user_context'.
        # user_context_first_rater is True iff this intent IS the first
        # one in this session to finalize against that user-context.
        # When False, user_context_exempt_ids enumerates the memory ids
        # that were surfaced under cause_id at declare-user-intents time
        # -- finalize subtracts them from injected/accessed coverage so
        # subsequent intents inherit the prior intent's ratings rather
        # than repeating them.
        "user_context_first_rater": bool(_user_ctx_first_rater),
        "user_context_exempt_ids": list(_user_ctx_exempt_ids),
    }

    # Persist to state file for PreToolUse hook (runs in separate process)
    _persist_active_intent()

    _mcp._wal_log(
        "declare_intent",
        {
            "intent_id": new_intent_id,
            "intent_type": intent_id,
            "slots": flat_slots,
            "content": description[:200],
        },
    )

    # feedback_reminder removed 2026-04-21: rules live in wake_up protocol.

    # Ranked subtype suggestions -- top 3 that score well AND have required tools
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
                        "content": (cs.get("content") or "")[:100],
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
    # on the active context via kg.record_feedback -- entity drops
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

    # ── Fix: re-derive injected_memory_ids from POST-GATE memories ──
    # already_injected was populated pre-gate and contained every item
    # that the retrieval pipeline surfaced. The injection gate then
    # filtered context["memories"], dropping items the agent never
    # actually saw. Persisting the PRE-gate set made finalize demand
    # feedback on gate-dropped items the agent couldn't possibly rate,
    # producing a perpetual `coverage 0%` failure mode (same pattern
    # hook_userpromptsubmit already fixed at hooks_cli.py:1518-1521;
    # also parallels _persist_accessed_memory_ids's rendered-only
    # contract).
    #
    # We rebuild from the filtered list so the persisted set contains
    # exactly what the agent saw in the response. Gate-dropped items
    # still get their rated_irrelevant edges from apply_gate, so the
    # retrieval-learning signal is preserved -- we just don't demand
    # agent re-rating of what it never received.
    already_injected = {m["id"] for m in context["memories"] if m.get("id")}

    # Step 2 of similar_context_id flag (default-on): for each unique
    # neighbour cid that contributed weight to ANY surviving memory,
    # render the neighbour's Context entity (queries/keywords/summary)
    # as a top-level similar_contexts list. The agent can then see WHY
    # a memory surfaced -- "this came in because context X is similar
    # to your active context, and X is about Y/Z." Mirrors the existing
    # context-block shape that surfaces on context reuse.
    _similar_contexts_block: list = []
    _seen_neighbour_cids: set = set()
    for _m in context["memories"]:
        for _cid in _m.get("similar_context_ids") or []:
            if _cid in _seen_neighbour_cids:
                continue
            _seen_neighbour_cids.add(_cid)
            try:
                _ent = _mcp._STATE.kg.get_entity(_cid)
            except Exception:
                _ent = None
            if not _ent:
                continue
            _props = _ent.get("properties") or {}
            if isinstance(_props, str):
                try:
                    _props = json.loads(_props)
                except Exception:
                    _props = {}
            _ctx_obj = {"id": _cid}
            _q = _props.get("queries")
            if _q:
                _ctx_obj["queries"] = list(_q)
            _kw = _props.get("keywords")
            if _kw:
                _ctx_obj["keywords"] = list(_kw)
            _sum = _props.get("summary")
            if _sum:
                _ctx_obj["summary"] = _sum
            _similar_contexts_block.append(_ctx_obj)

    # Token-diet response: we deliberately DON'T echo `intent_type`,
    # `slots`, or `budget` -- the caller just sent them, and the intent_id
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
    # Step 2 of similar_context_id flag (default-on): include the
    # similar_contexts block only when non-empty so token-diet stays
    # the default for sessions where the active context has no rated
    # similar-neighbours yet.
    if _similar_contexts_block:
        result["similar_contexts"] = _similar_contexts_block
    if _gate_status is not None:
        result["gate_status"] = _gate_status
    if DEBUG_RETURN_CONTEXT:
        # Token-diet 2026-04-24: non-reused contexts collapse to the
        # literal string "new" -- the caller just sent the cue, no
        # need to echo it back. On reuse we return the stored id +
        # the queries we retrieved under (often different from what
        # the caller sent), so the agent can see what matched. The
        # shape of the `context` field itself signals reuse: string
        # "new" = fresh mint; object = reused.
        if _active_context_reused:
            result["context"] = {
                "id": _active_context_id,
                "queries": list(_description_views),
            }
        else:
            result["context"] = "new"
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
# line range" have no topic anchor -- nearest-neighbor returns whatever
# past traces also ran pytest / edited tests / read files, regardless of
# topic. Agent-declared queries+keywords raise cue specificity to the
# same bar as declare_intent's Context fingerprint and align with AAO
# (Activity-Action-Operation) hierarchy: the intent is the Activity, the
# tool call is the Operation, and this is where the Operation cue lives.
#
# Retrieval reuses hooks_cli._run_local_retrieval (same multi-view cosine
# + keyword channel + RRF + dedup pipeline the PreToolUse hook already
# uses -- no new scoring code). The hook is then responsible for consuming
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


def _record_op_recall_diagnostic(op_context_id: str, populated: bool) -> None:
    """2026-04-26: track op-recall hit/miss per declare_operation call.

    Added per Adrian's directive after the ops-recall audit
    (``audit_operations_recall_end_to_end_2026_04_26``). The
    ``past_operations`` bucket is silently omitted from the response
    when the walker comes back with nothing, which masks two distinct
    failure modes the operator needs to tell apart:

      (a) graph genuinely has no rated ops in this context's
          ``similar_to`` neighbourhood -- fine, will warm up;
      (b) the walker keeps returning empty across many calls because
          contexts aren't getting ``similar_to`` edges
          (T_similar=0.70, MaxSim averaged across views, only top-1
          candidate gets the edge -- see the audit memo).

    Without a counter, (b) is invisible. Per-call DEBUG log plus a
    session-level counter on ``_STATE.session_state`` lets operators
    query "how often was past_operations empty / populated this
    session" without spamming production logs. Fire-and-forget:
    every branch swallows exceptions so a metrics failure cannot
    break the declare_operation response.
    """
    key = "op_recall_populated_count" if populated else "op_recall_empty_count"
    try:
        ss = _mcp._STATE.session_state
        ss[key] = int(ss.get(key, 0)) + 1
    except Exception:
        pass
    if not populated:
        try:
            import logging as _ops_log

            _ops_log.getLogger(__name__).debug(
                "past_operations empty for ctx=%s (no good/avoid in similar_to neighbourhood)",
                op_context_id,
            )
        except Exception:
            pass


def _emit_op_cluster_flags(past_ops: dict, op_context_id: str, kg) -> None:
    """S3a: detect same-tool same-sign clusters in past_operations and
    persist them as ``op_cluster_templatizable`` memory_flags rows.

    Split out of ``tool_declare_operation`` to keep that function below
    the ruff C901 complexity budget and to give the emission path its
    own testable seam. Fire-and-forget: swallows every exception so
    retrieval errors never break the declare_operation response.
    """
    try:
        from .scoring import detect_op_cluster_flags as _detect_clusters

        flags = _detect_clusters(past_ops)
        if not flags:
            return
        for flag in flags:
            flag["context_id"] = op_context_id
        kg.record_memory_flags(flags)
    except Exception:
        # S3a is advisory -- a failure here must not propagate. The
        # gardener simply won't get this cluster flag; it will fire
        # again next time the same cluster re-surfaces.
        pass


def tool_declare_operation(  # noqa: C901
    tool: str,
    args_summary: str = None,
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
    kg_declare_entity / kg_add_batch -- ONE shape for every emit site):

        context = {
          "queries":  [2-5 natural-language perspectives],
          "keywords": [2-5 exact domain terms],
          "entities": [1-10 entity ids the operation touches],
        }

    Args:
        tool: Name of the tool you are about to call (e.g. 'Read', 'Grep',
              'Bash', 'Edit'). Must be permitted under the active intent.
        context: Mandatory unified Context dict. See shape above. Validated
                 by ``scoring.validate_context`` -- same validator every
                 other emit site uses, same error messages, same bounds.
        agent: Your agent name.

    Returns:
        {"success": true, "memories": [...], "feedback_reminder": "..."}
        on success. Memories carried through to finalize_intent's
        mandatory memory_feedback coverage via accessed_memory_ids.

    Carve-outs: mempalace_* tools and the ALWAYS_ALLOWED set in
    hooks_cli (TodoWrite, Skill, Agent, ToolSearch, AskUserQuestion,
    Task*, ExitPlanMode) do NOT need declare_operation -- they skip
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
                "intent -- you cannot declare an operation with no intent."
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
                "ExitPlanMode) skip PreToolUse retrieval -- just call "
                "them directly."
            ),
        }

    # ── Validate args_summary (mandatory, parametrized-core form) ──
    # 2026-04-27 redesign: args_summary moved from optional rating-side
    # field to mandatory declare-time field. Two ops sharing the same
    # parametrized args_summary cluster as the SAME operation in the
    # past_operations neighbourhood walk and the gardener S3a templatize
    # detector -- so the fingerprint must capture INTENT, not literal
    # text. See the schema description for parametrization examples.
    if not isinstance(args_summary, str) or not args_summary.strip():
        return {
            "success": False,
            "error": (
                "args_summary is required (string, 5-400 chars). It is the "
                "PARAMETRIZED CORE of the operation -- invariant shape with "
                "per-execution variables abstracted as {placeholders}. "
                "Examples:\n"
                "  Bad:  'git commit -m \"feat: ship Slice C gate\"'\n"
                "  Good: 'git commit -m \"{commit_message}\"'\n"
                "  Bad:  'python -m pytest tests/test_intent.py -q'\n"
                "  Good: 'python -m pytest {test_path} -q'\n"
                "Strip plumbing (cd, env vars, redirects). Two ops with "
                "the same args_summary string cluster as the same operation."
            ),
        }
    args_summary = args_summary.strip()
    if len(args_summary) < 5:
        return {
            "success": False,
            "error": (
                f"args_summary too short ({len(args_summary)} chars; "
                f"minimum 5). It must be a parametrized-core fingerprint, "
                f"not a one-word label."
            ),
        }
    if len(args_summary) > 400:
        return {
            "success": False,
            "error": (
                f"args_summary too long ({len(args_summary)} chars; "
                f"maximum 400). Compress to the parametrized core; long "
                f"literal strings defeat the cluster-matching purpose."
            ),
        }

    # ── Validate Context -- same shared validator every emit site uses ──
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
        require_summary=True,
        summary_context_for_error="declare_operation.context.summary",
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
    # active intent's context -- they may be similar (both pertain to the
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
            summary=clean_context.get("summary"),
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
        "args_summary": args_summary,
        "queries": cue["queries"],
        "keywords": cue["keywords"],
        "declared_at_ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "surfaced_ids": [h.get("id") for h in hits if h.get("id")],
        # P1: context entity id minted for this operation cue. Writers
        # that fire while this cue is the most-recent one use it as
        # active_context_id.
        "active_context_id": _op_context_id,
    }
    # 2026-04-27 redesign: persist args_summary on the active intent
    # under (context_id, tool) so the finalize_intent / extend_feedback
    # promotion path can fetch the parametrized-core fingerprint instead
    # of the (now-removed) rating-side field. Last-write-wins per
    # (ctx_id, tool); rare in practice because two ops with the same
    # context_id are usually the same operation by design.
    if _op_context_id:
        _op_args_store = _mcp._STATE.active_intent.get("op_args_by_ctx_tool")
        if not isinstance(_op_args_store, dict):
            _op_args_store = {}
            _mcp._STATE.active_intent["op_args_by_ctx_tool"] = _op_args_store
        _op_args_store[f"{_op_context_id}|{tool}"] = args_summary
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
        # Bug 3 fix 2026-04-28: skip coverage tracking when intent is in
        # pending_feedback state (mid-finalize). Once tool_finalize_intent
        # has accepted the intent and is awaiting extend_feedback, any
        # further declare_operation calls are bookkeeping (the agent
        # rating its own retrievals or running pending-work probes) and
        # shouldn't grow the coverage requirement. Without this gate the
        # set snowballed to 60-90 entries per intent transition by the
        # 5th intent in long sessions, dwarfing actual code work. Within-
        # intent dedup is moot post-finalize since the intent is closing
        # -- no further operations should rely on the dedup filter.
        _is_finalizing = bool(_mcp._STATE.active_intent.get("pending_feedback"))
        if not _is_finalizing:
            _acc_set = _mcp._STATE.active_intent.get("accessed_memory_ids")
            if not isinstance(_acc_set, set):
                _acc_set = set(_acc_set or [])
            _acc_set.update(_new_op_ids)
            _mcp._STATE.active_intent["accessed_memory_ids"] = _acc_set
    _persist_active_intent()

    # ── Build response ──
    # Rules (mandatory-coverage, fetch-full-via-kg_query, declare-gate)
    # live in the wake_up protocol -- we no longer repeat them in every
    # operation response. See wake_up's protocol string for the contract.
    #
    # Step 3 of similar_context_id flag (default-on, parity with
    # declare_intent + kg_search): walk the rated-neighbourhood of the
    # operation's context (or fallback to the active intent's context)
    # to surface which similar_to neighbours contributed weight to each
    # retrieved item. declare_operation does not normally walk -- the
    # cosine retrieval here is op-cue-only -- so we do a fresh walk
    # purely to populate the monitoring fields.
    _op_walk_ctx = _op_context_id or (
        _mcp._STATE.active_intent.get("active_context_id") if _mcp._STATE.active_intent else ""
    )
    try:
        from . import scoring as _scoring_op

        _op_rated_walk = (
            _scoring_op.walk_rated_neighbourhood(_op_walk_ctx, _mcp._STATE.kg)
            if _op_walk_ctx
            else {"contributing_contexts": {}}
        )
    except Exception:
        _op_rated_walk = {"contributing_contexts": {}}
    _op_contributing_contexts = _op_rated_walk.get("contributing_contexts") or {}

    memories = []
    for h in hits:
        entry = {
            "id": h["id"],
            "text": _shorten_preview((h.get("preview") or "").strip()),
        }
        if DEBUG_RETURN_SCORES:
            entry["hybrid_score"] = round(float(h.get("score", 0.0) or 0.0), 6)
        _neighbour_cids = _op_contributing_contexts.get(h["id"]) or []
        if _neighbour_cids:
            entry["similar_context_ids"] = list(_neighbour_cids)
        memories.append(entry)

    # Build top-level similar_contexts block from the union of unique
    # contributing cids across surviving memory entries. Same shape as
    # declare_intent + kg_search.
    _op_similar_contexts: list = []
    _op_seen_cids: set = set()
    for _m in memories:
        for _cid in _m.get("similar_context_ids") or []:
            if _cid in _op_seen_cids:
                continue
            _op_seen_cids.add(_cid)
            try:
                _ent = _mcp._STATE.kg.get_entity(_cid)
            except Exception:
                _ent = None
            if not _ent:
                continue
            _props = _ent.get("properties") or {}
            if isinstance(_props, str):
                try:
                    _props = json.loads(_props)
                except Exception:
                    _props = {}
            _ctx_obj = {"id": _cid}
            _q = _props.get("queries")
            if _q:
                _ctx_obj["queries"] = list(_q)
            _kw = _props.get("keywords")
            if _kw:
                _ctx_obj["keywords"] = list(_kw)
            _sum = _props.get("summary")
            if _sum:
                _ctx_obj["summary"] = _sum
            _op_similar_contexts.append(_ctx_obj)

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
    # Step 3 of similar_context_id flag (default-on, parity with
    # declare_intent + kg_search): include similar_contexts only when
    # non-empty (token-diet default).
    if _op_similar_contexts:
        result["similar_contexts"] = _op_similar_contexts
    if _gate_status is not None:
        result["gate_status"] = _gate_status

    # ── S1: past_operations -- op-tier retrieval ──
    # Orthogonal to memories (Channels A-D). Walks performed_well /
    # performed_poorly edges from the current operation's context
    # neighbourhood, returning good precedents + cautionary patterns.
    # Not filtered through the memory gate; rendered in its own slot
    # so op-tier noise can't pollute the memory retrieval signal. Cf.
    # arXiv 2512.18950 (Operation tier), Leontiev 1981 AAO.
    if _op_context_id:
        try:
            from .scoring import retrieve_past_operations as _retrieve_ops

            # 2026-04-27: pass args_summary + op-Chroma collection so
            # retrieve_past_operations can populate the args_precedents
            # lane via cosine recall + BGE-reranker rerank, surfacing
            # ops with similar parametrized fingerprint regardless of
            # context. Tool filter eliminates cross-tool false matches.
            _op_chroma = _mcp._get_entity_collection(create=False)
            _past_ops = _retrieve_ops(
                _op_context_id,
                _mcp._STATE.kg,
                k=5,
                current_args_summary=args_summary,
                op_chroma_collection=_op_chroma,
                current_tool=tool,
            )
            _has_good = bool(_past_ops.get("good_precedents"))
            _has_bad = bool(_past_ops.get("avoid_patterns"))
            _has_args = bool(_past_ops.get("args_precedents"))
            # Only attach when there is something to say -- keeps the
            # response lean when the graph has no op history yet.
            if _has_good or _has_bad or _has_args:
                result["past_operations"] = _past_ops

            # 2026-04-26 diagnostic -- see _record_op_recall_diagnostic
            # for the rationale and full doc.
            _record_op_recall_diagnostic(_op_context_id, _has_good or _has_bad)

            # S3a: piggyback-flag same-tool same-sign clusters for the
            # gardener to templatize (S3b) and for retrieval to later
            # hoist as reusable patterns (S3c). Helper is
            # fire-and-forget; see _emit_op_cluster_flags.
            _emit_op_cluster_flags(_past_ops, _op_context_id, _mcp._STATE.kg)
        except Exception:
            # Fail-silent: op retrieval is a nice-to-have, not load-bearing.
            pass

    if DEBUG_RETURN_CONTEXT:
        # Token-diet 2026-04-24: non-reused collapses to "new"; reused
        # returns {id, queries}. See tool_declare_intent for the full
        # rationale. Shape-as-signal: string "new" = fresh, object = reused.
        if _op_context_reused:
            result["context"] = {
                "id": _op_context_id,
                "queries": list(cue["queries"]),
            }
        else:
            result["context"] = "new"
    if notice:
        # Fail-loud: retrieval error surfaces to agent, not silent.
        result["retrieval_notice"] = notice
    return result


# ═══════════════════════════════════════════════════════════════════
# Slice B (user-intent tier): tool_declare_user_intents
# ═══════════════════════════════════════════════════════════════════
#
# Top-tier (Motive / Strategy in Leontiev 1981) declaration. The agent
# calls this AFTER each user message (or batch of messages) to declare
# what the user is asking for, BEFORE proceeding to declare any
# activity-intent. The tool:
#
#   1. Reads pending_user_messages from session state (written by
#      UserPromptSubmit hook).
#   2. Validates that union(context.user_message_ids for context in
#      contexts) covers every pending user_message_id -- no message
#      falls through unnoticed.
#   3. Mints a kind='record' user_message entity for each pending
#      message (content = raw prompt text). Links each user-context
#      to its referenced messages via fulfills_user_message edges.
#   4. Calls context_lookup_or_create per declared context (MaxSim
#      reuse, similar_to graph wiring -- same path as declare_intent /
#      declare_operation / kg_search).
#   5. Runs retrieval per context, dedup'd against accessed/injected
#      memory ids accumulated in this session, returns top-K.
#   6. Clears the pending queue so the PreToolUse block (Slice B-2)
#      releases.
#
# Grounding: STITCH (arXiv:2601.10702) for the structured-intent-tuple
# pattern, Agent-Sentry (arXiv:2603.22868) for the forced-cause-linkage
# discipline, BDI (Rao & Georgeff 1995) for the hierarchical-cause
# invariant. See diary_ga_agent_user_intent_tier_design_locked_2026_04_24
# for the full design narrative.
#
# Slice B-1 scope: tool ships and works end-to-end (agent can call it,
# validates pending coverage, mints records, returns memories per
# context). Slice B-2 wires the PreToolUse block + UserPromptSubmit
# rewrite that produces the pending entries. Slice B-3 adds optional
# cause_id on declare_intent + finalize coverage rule.


# Per-context bounds for the user-intent tier. Mirrors MIN_OP_QUERIES
# / MAX_OP_QUERIES style -- kept module-level so schema + tests share
# the source-of-truth.
MIN_USER_INTENT_QUERIES = 1  # one perspective per intent is enough
MAX_USER_INTENT_QUERIES = 5
MIN_USER_INTENT_KEYWORDS = 2
MAX_USER_INTENT_KEYWORDS = 5
MIN_USER_INTENT_ENTITIES = 1
MAX_USER_INTENT_ENTITIES = 10
USER_INTENT_TOP_K = 5  # memories per context


# Slice B-4b: session-scoped first-rater set. Maps session_id → set of
# user-context entity ids whose surfaced memories have already been rated
# by a prior agent intent finalize in this session. The FIRST agent
# intent that finalizes against a user-context (cause_kind=='user_context')
# is required to cover its surfaced memories in memory_feedback;
# subsequent intents with the same cause_id inherit that coverage and are
# exempt from re-rating those same memories. In-memory only -- survives
# only as long as the MCP server process. Tests reset by reassigning to
# a fresh dict (see _reset_rated_user_contexts).
_RATED_USER_CONTEXTS: dict = {}


def _rated_user_contexts_for(sid: str) -> set:
    """Get-or-create the rated_user_contexts set for the given session."""
    if not isinstance(sid, str):
        sid = ""
    bucket = _RATED_USER_CONTEXTS.get(sid)
    if bucket is None:
        bucket = set()
        _RATED_USER_CONTEXTS[sid] = bucket
    return bucket


def _reset_rated_user_contexts() -> None:
    """Drop all session buckets. Used by tests to isolate state."""
    _RATED_USER_CONTEXTS.clear()


def tool_declare_user_intents(  # noqa: C901
    contexts: list = None,
    agent: str = None,
):
    """Declare the user-intent contexts that cover the pending user
    messages for this session. Top tier of the activity hierarchy
    (Motive/Strategy); activity-intents declared via declare_intent
    later this turn link upward via cause_id.

    Args:
        contexts: list of dicts, one per user-intent. Each dict carries:
            - context: {queries, keywords, entities, summary} --
                same unified Context shape as every other emit site.
            - user_message_ids: list[str] -- pending message ids this
                user-intent covers. Union across all contexts MUST
                equal the pending set (no message left unattributed).
            - time_window: {start, end} optional ISO dates for soft
                date-range boost in retrieval (same semantics as
                kg_search.time_window).
            - no_intent: bool default False -- set TRUE to declare that
                a covered user message has no actionable intent (ack,
                "thanks", clarifying question already answered, etc.).
                When TRUE, no_intent_clarified_with_user MUST be a
                truthful bool (set TRUE only if the agent actually
                asked the user via AskUserQuestion to confirm).
        agent: caller's agent name. Required.

    Returns:
        {
          "success": True,
          "contexts": [
            {"ctx_id": "...", "reused": bool, "memories": [...]},
            ...
          ],
          "cleared_pending_count": N,
        }

    Validation rejects (success=False with explicit error) on:
        * Empty contexts list.
        * Any context missing user_message_ids.
        * Any user_message_id not in the pending queue.
        * Pending queue not fully covered by union of user_message_ids.
        * Standard Context validator failures (per-context queries /
          keywords / entities / summary bounds).
        * no_intent=True without no_intent_clarified_with_user=True.
    """
    sid_err = _mcp._require_sid(action="declare_user_intents")
    if sid_err:
        return sid_err
    agent_err = _mcp._require_agent(agent, action="declare_user_intents")
    if agent_err:
        return agent_err

    contexts, _err = _coerce_list_param("contexts", contexts)
    if _err:
        return _err
    if not contexts:
        return {
            "success": False,
            "error": (
                "contexts is required and must be a non-empty list. "
                "At least one user-intent context per call. Each context "
                "carries {context: {queries, keywords, entities, summary}, "
                "user_message_ids: [...], time_window?: {...}, "
                "no_intent?: bool, no_intent_clarified_with_user?: bool}."
            ),
        }

    # ── Read pending user-messages for this session ──
    from . import hooks_cli as _hc

    sid = _mcp._STATE.session_id or ""
    pending_msgs = _hc._read_pending_user_messages(sid)
    pending_ids = {m["id"] for m in pending_msgs}

    # ── Per-context shape validation + collect referenced ids ──
    from .scoring import validate_context as _validate_context

    cleaned_contexts = []
    referenced_ids = set()
    for i, c in enumerate(contexts):
        if not isinstance(c, dict):
            return {
                "success": False,
                "error": (
                    f"contexts[{i}] must be a dict with keys "
                    "'context', 'user_message_ids', and optional "
                    "'time_window' / 'no_intent' / "
                    "'no_intent_clarified_with_user'."
                ),
            }
        raw_ctx = c.get("context")
        clean_ctx, ctx_err = _validate_context(
            raw_ctx,
            queries_min=MIN_USER_INTENT_QUERIES,
            queries_max=MAX_USER_INTENT_QUERIES,
            keywords_min=MIN_USER_INTENT_KEYWORDS,
            keywords_max=MAX_USER_INTENT_KEYWORDS,
            entities_min=MIN_USER_INTENT_ENTITIES,
            entities_max=MAX_USER_INTENT_ENTITIES,
            require_summary=True,
            summary_context_for_error=f"declare_user_intents.contexts[{i}].context.summary",
        )
        if ctx_err:
            return ctx_err

        umids = c.get("user_message_ids")
        if not isinstance(umids, list) or not umids:
            return {
                "success": False,
                "error": (
                    f"contexts[{i}].user_message_ids is required (non-empty list). "
                    "Reference at least one pending user_message id this "
                    "user-intent covers. See additionalContext from the "
                    "UserPromptSubmit hook for pending ids."
                ),
            }
        for mid in umids:
            if not isinstance(mid, str) or not mid.strip():
                return {
                    "success": False,
                    "error": f"contexts[{i}].user_message_ids contains a non-string entry.",
                }
            if mid not in pending_ids:
                return {
                    "success": False,
                    "error": (
                        f"contexts[{i}].user_message_ids references {mid!r} which is "
                        f"not in the pending user_message queue for this session. "
                        f"Pending ids: {sorted(pending_ids)}"
                    ),
                }
            referenced_ids.add(mid)

        no_intent = bool(c.get("no_intent", False))
        if no_intent:
            confirmed = bool(c.get("no_intent_clarified_with_user", False))
            if not confirmed:
                return {
                    "success": False,
                    "error": (
                        f"contexts[{i}].no_intent=True requires "
                        "no_intent_clarified_with_user=True -- the agent must "
                        "have actually asked the user (via AskUserQuestion) "
                        "to confirm the message has no actionable intent. "
                        "Self-asserting no_intent without proof is rejected."
                    ),
                }

        time_window = c.get("time_window")
        if time_window is not None and not isinstance(time_window, dict):
            return {
                "success": False,
                "error": f"contexts[{i}].time_window must be a dict {{start, end}} or omitted.",
            }

        cleaned_contexts.append(
            {
                "clean_ctx": clean_ctx,
                "user_message_ids": list(umids),
                "time_window": time_window,
                "no_intent": no_intent,
            }
        )

    # ── Coverage check: every pending id must be referenced by ≥1 context ──
    if pending_ids and pending_ids - referenced_ids:
        missing = sorted(pending_ids - referenced_ids)
        return {
            "success": False,
            "error": (
                f"Pending user_message ids not covered by any declared context: "
                f"{missing}. Every pending message must appear in at least one "
                f"context.user_message_ids. If a message has no actionable intent, "
                f"declare it under a no_intent=True context (with "
                f"no_intent_clarified_with_user=True after asking the user)."
            ),
            "missing_user_message_ids": missing,
        }

    # ── Mint user_message records for each pending entry ──
    # Idempotent: if the record already exists (re-run scenario), skip.
    minted_user_message_ids = []
    for m in pending_msgs:
        mid = m["id"]
        if not mid:
            continue
        existing = None
        try:
            existing = _mcp._STATE.kg.get_entity(mid)
        except Exception:
            existing = None
        if existing:
            minted_user_message_ids.append(mid)
            continue
        try:
            _mcp._STATE.kg.add_entity(
                mid,
                kind="record",
                content=(m.get("text") or "")[:500],
                importance=3,
                properties={
                    "type": "user_message",
                    "session_id": sid,
                    "turn_idx": int(m.get("turn_idx") or 0),
                    "ts": m.get("ts") or "",
                    "added_by": agent or "",
                },
            )
            minted_user_message_ids.append(mid)
        except Exception as _mint_err:
            return {
                "success": False,
                "error": f"Failed to mint user_message record {mid!r}: {_mint_err!r}",
            }

    # ── For each context: lookup_or_create + fulfills_user_message edges + retrieval ──
    response_contexts = []
    new_injected_ids = []
    for entry in cleaned_contexts:
        clean_ctx = entry["clean_ctx"]
        ctx_id = ""
        reused = False
        try:
            ctx_id, reused, _ms = _mcp.context_lookup_or_create(
                queries=clean_ctx["queries"],
                keywords=clean_ctx["keywords"],
                entities=clean_ctx["entities"],
                agent=agent,
                summary=clean_ctx.get("summary"),
            )
        except Exception:
            ctx_id = ""
            reused = False

        # Wire user_message → user-context coverage edges. fulfills_user_message
        # is the predicate Slice B-3 cause_id validator reads to identify
        # "user-tier" contexts. The predicate is non-skip-list so a natural
        # -language statement is required (2026-04-19 lock that retired
        # autogenerated verbalisations). Soft-fail at edge level so a
        # transient kg/seeder issue does not prevent the context creation.
        if ctx_id:
            for um_id in entry["user_message_ids"]:
                _ful_statement = (
                    f"User-context {ctx_id} fulfils user_message {um_id} "
                    f"by declaring an intent that covers this user prompt."
                )
                try:
                    _mcp._STATE.kg.add_triple(
                        ctx_id,
                        "fulfills_user_message",
                        um_id,
                        statement=_ful_statement,
                    )
                except Exception:
                    pass

        # Retrieval per context -- same pipeline as declare_operation.
        cue = {
            "queries": list(clean_ctx["queries"]),
            "keywords": list(clean_ctx["keywords"]),
        }
        accessed = (
            set(_mcp._STATE.active_intent.get("accessed_memory_ids") or [])
            | set(_mcp._STATE.active_intent.get("injected_memory_ids") or [])
            if _mcp._STATE.active_intent
            else set()
        )
        try:
            hits, _notice = _hc._run_local_retrieval(cue, accessed, USER_INTENT_TOP_K)
        except Exception:
            hits = []
        memories = []
        for h in hits:
            mid = h.get("id")
            if not mid:
                continue
            new_injected_ids.append(mid)
            memories.append(
                {
                    "id": mid,
                    "text": _shorten_preview((h.get("preview") or "").strip()),
                }
            )

        block = {"ctx_id": ctx_id, "reused": bool(reused)}
        # Token-diet: echo queries/keywords/entities only on reuse, mirroring
        # declare_intent / declare_operation convention.
        if reused:
            block["queries"] = list(clean_ctx["queries"])
            block["keywords"] = list(clean_ctx["keywords"])
            block["entities"] = list(clean_ctx["entities"])
        if memories:
            block["memories"] = memories
        if entry["no_intent"]:
            block["no_intent"] = True
        response_contexts.append(block)

    # ── Persist injected ids to active_intent (if any) ──
    # When no active_intent exists yet (early in the session), we still
    # cleared pending; the next declare_intent will inherit retrieval
    # via its own context. This matches the "user-tier sits ABOVE
    # activity" design -- user contexts can exist without an activity.
    if new_injected_ids and _mcp._STATE.active_intent:
        _inj = _mcp._STATE.active_intent.get("injected_memory_ids")
        if not isinstance(_inj, set):
            _inj = set(_inj or [])
        _inj.update(new_injected_ids)
        _mcp._STATE.active_intent["injected_memory_ids"] = _inj
        try:
            _persist_active_intent()
        except Exception:
            pass

    # ── Clear ONLY the declared ids from the pending queue ──
    # Adrian's spec 2026-04-29: per-id removal, not bulk-drain. Messages
    # still pending after this call remain in the queue and surface in
    # the next UserPromptSubmit / PreToolUse gate check. Preserves
    # next_turn_idx so future ids stay monotonic.
    cleared_n = _hc._remove_pending_user_messages(sid, referenced_ids)

    return {
        "success": True,
        "contexts": response_contexts,
        "cleared_pending_count": cleared_n,
        "minted_user_message_ids": minted_user_message_ids,
    }


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
# Design rationale: relevance is inherently subjective -- there is no
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
    characters and emits one bogus error per char -- the same bug that
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
    memory_feedback: dict = None,
    key_actions: list = None,
    gotchas: list = None,
    learnings: list = None,
    promote_gotchas_to_type: bool = False,
    operation_ratings: list = None,
):
    """Finalize the active intent -- capture what happened as structured memory.

    MUST be called before declaring a new intent or exiting the session.
    Creates an execution entity (kind=entity, is_a intent_type) with
    relationships linking it to the agent, targets, result memory, gotchas,
    and execution trace.

    VOCABULARY -- uniform across every record-write boundary in mempalace:
      ``content`` = full narrative body. FREE LENGTH -- as detailed as needed.
        Stored verbatim.
      ``summary`` = ≤280-char distillation / reframe. ALWAYS required (no
        length threshold on content -- every record gets a summary). For
        long content the summary distills the WHAT/WHY; for short content
        the summary should REPHRASE the same fact from a different angle
        (different keywords / framing) so the summary+content pair yields
        two distinct cosine views of the same semantic -- real retrieval
        gain, not redundancy. Anthropic Contextual Retrieval (2024)
        prepends the summary to the content before embedding (single CR
        vector); the summary is also what injection-time previews display.
        No prefix slicing. No auto-derivation. The caller produces it.

    Args:
        slug: Human-readable ID for this execution (e.g. 'edit-auth-rate-limiter-2026-04-14')
        outcome: 'success', 'partial', 'failed', or 'abandoned'
        content: Full outcome narrative -- the body of the result memory. Any
            length. Becomes the embedded document (with summary prepended).
        summary: ≤280-char distilled one-liner of the outcome (or a
            different-angle rephrase when content is short). Shown in
            injections and prepended to content for embedding.
        agent: Agent entity name (e.g. 'technical_lead_agent')
        memory_feedback: MANDATORY -- LIST-OF-GROUPS shape (dict shape
            retired 2026-04-24 due to MCP-client `additionalProperties`
            serialization dropping payloads). Contextual relevance
            feedback for every memory accessed during this intent,
            scoped by the CONTEXT that surfaced it:

              [
                {
                  "context_id": "<ctx_id>",
                  "feedback": [
                    {"id": "<memory_id>", "relevance": 1-5,
                     "reason": "why (>=10 chars)", "relevant": true/false},
                    ...
                  ]
                },
                ...
              ]

            Per-group attribution is load-bearing: the writer attaches
            rated_useful / rated_irrelevant edges FROM that context TO
            the memory, and Channel D reads those edges on future
            intents. Context ids are returned by declare_intent /
            declare_operation / kg_search (as `context.id` on reuse)
            and revealed by a failed finalize's `missing_injected` map
            response field.
        key_actions: Abbreviated tool+params list (optional -- auto-filled from trace if omitted)
        gotchas: List of gotchas discovered during execution. Each entry
            is ``{summary: {what, why, scope?}, content: str}`` --
            structured-anchor + verbatim body. Strings are rejected
            (no auto-derive; Adrian's design lock 2026-04-28).
        learnings: List of lessons worth remembering. Each entry is
            ``{summary: {what, why, scope?}, content: str}`` -- same
            strict dict shape. Strings are rejected.
        promote_gotchas_to_type: Also link gotchas to the intent type (not just execution)
        operation_ratings: MANDATORY (100% coverage over unique (tool,
            context_id) pairs in the execution trace) -- agent's rating
            of tool-invocation quality. Orthogonal to memory_feedback
            (which rates retrieval relevance). Each entry describes
            ONE operation you performed and how well it fit the goal:

              [{
                "tool": "Edit",              # required, the tool name
                "context_id": "ctx_abc123",  # required, from declare_operation
                "quality": 1..5,             # required. 1=wrong move,
                                             # 2=suboptimal, 3=ok (no-op
                                             # promotion -- skipped),
                                             # 4=good, 5=load-bearing
                "reason": "why (>=10 chars)",       # recommended
                "args_summary": "short arg sketch", # recommended -- stored
                                             # on the op entity, used for
                                             # gardener clustering in S3
                "better_alternative": "op_id",     # S2 -- superseded_by edge
              }, ...]

            Quality ≥4 writes a `performed_well` edge from the context to
            the op entity; quality ≤2 writes `performed_poorly`; quality=3
            is skipped (neutral). Distinct from rated_useful /
            rated_irrelevant -- those rate retrieved memories, NOT tool
            correctness. Cf. Leontiev 1981 (Operation tier), arXiv
            2512.18950 (hierarchical procedural memory).
    """

    # Sid check FIRST -- an empty sid means the tool call came in without
    # hook-injected sessionId, which makes every downstream state op a
    # potential cross-agent contamination risk. Fail loud at the boundary.
    sid_err = _mcp._require_sid(action="finalize_intent")
    if sid_err:
        return sid_err

    # ── Pending user-intent gate (Adrian's spec 2026-04-29) ──
    # Refuse to finalize if the session still has user_message ids in the
    # pending queue that haven't been declared via mempalace_declare_user_intents.
    # The agent must surface those messages and either declare an activity
    # intent for each or mark them no_intent (with confirmation) before
    # closing the current activity. Without this gate, finalize silently
    # drops user prompts the agent never addressed -- exactly the failure
    # mode that motivated the user-intent tier in the first place.
    try:
        from . import hooks_cli as _hc

        sid_for_check = _mcp._STATE.session_id or ""
        if sid_for_check:
            _pending_now = _hc._read_pending_user_messages(sid_for_check)
            _pending_ids_now = sorted(
                m.get("id") for m in (_pending_now or []) if isinstance(m, dict) and m.get("id")
            )
            if _pending_ids_now:
                return {
                    "success": False,
                    "error": (
                        f"finalize_intent refuses to close while user_message ids "
                        f"remain undeclared in this session's pending queue. "
                        f"Pending ids: {_pending_ids_now}. Call "
                        f"mempalace_declare_user_intents first to declare an "
                        f"intent context for each (or no_intent=True with "
                        f"no_intent_clarified_with_user=True after asking the "
                        f"user via AskUserQuestion). Adrian's design 2026-04-29: "
                        f"every user prompt must be acknowledged before any "
                        f"activity intent finalizes."
                    ),
                    "pending_user_message_ids": _pending_ids_now,
                }
    except Exception:
        # Fail-open on read errors -- the pending file is best-effort
        # state; a corrupt or missing file shouldn't block legitimate
        # finalize calls. The hook layer already records read errors via
        # _record_hook_error.
        pass

    # ── Summary-first gate: strict validation at the boundary ──
    # Mirrors _add_memory_internal's ≤280-char rule. Enforced HERE (not
    # only inside the downstream result_memory upsert) because the old
    # behaviour collected the downstream rejection into `errors` and
    # returned success=True -- so a 299-char summary would finalize the
    # intent, create the execution entity, but leave no result memory,
    # letting the caller assume everything was fine. Every method that
    # accepts a summary rejects over-length up front and fails the call.
    # Keep this in lockstep with _add_memory_internal so the two rules
    # never drift.
    # Dict-only contract (Adrian's design lock 2026-04-25): summary
    # is a structured {what, why, scope?} dict. coerce_summary_for_persist
    # validates and serialize_summary_for_embedding renders the prose form
    # downstream code reads as the human-facing one-liner. Keep this in
    # lockstep with _add_memory_internal so the two contracts never drift.
    if summary is None:
        return {
            "success": False,
            "error": (
                f"`summary` is required. Pass a dict {{what, why, scope?}}; "
                f"the rendered prose form is capped at "
                f"{_mcp._RECORD_SUMMARY_MAX_LEN} chars."
            ),
        }
    try:
        from .knowledge_graph import (
            SummaryStructureRequired as _SSR,
            coerce_summary_for_persist as _coerce_summary,
            serialize_summary_for_embedding as _ser_summary,
        )

        _summary_dict = _coerce_summary(
            summary,
            context_for_error="finalize_intent.summary",
        )
    except _SSR as _vs_err:
        return {"success": False, "error": str(_vs_err)}
    _summary_clean = _ser_summary(_summary_dict).strip()
    summary = _summary_clean  # downstream reads the rendered prose form
    if len(_summary_clean) > _mcp._RECORD_SUMMARY_MAX_LEN:
        return {
            "success": False,
            "error": (
                f"`summary` is {len(_summary_clean)} chars; maximum is "
                f"{_mcp._RECORD_SUMMARY_MAX_LEN}. Distill further -- one "
                f"sentence, names the WHAT and WHY, no filler."
            ),
        }
    summary = _summary_clean

    # memory_feedback contract: LIST OF GROUPS ONLY (dict-shape retired 2026-04-24).
    #   [{context_id: <ctx_id>, feedback: [{id, relevance, reason, ...}, ...]}, ...]
    # The dict-shape was retired because some MCP clients silently drop
    # object parameters whose schema uses `additionalProperties` with a
    # nested schema -- leaving the handler with memory_feedback=None and
    # no indication the payload was ever sent. List-of-objects is the
    # universally-supported JSON-Schema shape and round-trips cleanly
    # through every client.
    # Each group attributes its ratings to the context that surfaced
    # those memories. The writer attaches rated_useful /
    # rated_irrelevant edges FROM that context TO the memory -- Channel D
    # reads those edges on future intents. Per-group attribution is
    # load-bearing, not cosmetic.
    # ── DIAGNOSTIC: finalize coverage bug trace (env-gated) ──
    # Captures memory_feedback shape AT ENTRY before any normalization
    # so "arrived wrong" vs "normalized wrong" is distinguishable. ON
    # by default; set MEMPALACE_DISABLE_FINALIZE_DEBUG=1 to silence.
    # Sink: ~/.mempalace/finalize_debug.log.
    _dbg_enabled = not os.environ.get("MEMPALACE_DISABLE_FINALIZE_DEBUG")
    if _dbg_enabled:
        import logging as _dbg_logging

        _dbg = _dbg_logging.getLogger("mempalace.finalize_debug")
        try:
            _dbg.warning(
                "FINALIZE_IN mf_type=%s is_dict=%s is_list=%s is_str=%s preview=%r",
                type(memory_feedback).__name__,
                isinstance(memory_feedback, dict),
                isinstance(memory_feedback, list),
                isinstance(memory_feedback, str),
                (str(memory_feedback)[:500] if memory_feedback is not None else None),
            )
        except Exception:
            pass

    _memory_feedback_by_context: dict = {}
    if memory_feedback is None:
        memory_feedback = []
    if isinstance(memory_feedback, str):
        # Stringified-JSON delivery -- parse loudly.
        try:
            memory_feedback = json.loads(memory_feedback)
        except Exception:
            return {
                "success": False,
                "error": (
                    "memory_feedback arrived as an unparseable string. "
                    "Pass a list of groups: "
                    "[{context_id, feedback: [{id, relevance, reason}, ...]}, ...]"
                ),
            }
    if isinstance(memory_feedback, dict):
        return {
            "success": False,
            "error": (
                "memory_feedback dict shape is retired (2026-04-24). Use "
                "list-of-groups: "
                "[{context_id: '<ctx_id>', feedback: [{id, relevance, reason}, ...]}, ...]. "
                "Each group attributes ratings to the context that surfaced "
                "its memories. Dict shape was retired because some MCP "
                "clients silently drop object parameters whose schema uses "
                "`additionalProperties` with a nested schema -- leaving the "
                "handler with memory_feedback=None. List-of-objects "
                "round-trips cleanly through every client."
            ),
        }
    if not isinstance(memory_feedback, list):
        return {
            "success": False,
            "error": (
                "memory_feedback must be a list of groups: "
                "[{context_id, feedback: [{id, relevance, reason}, ...]}, ...]. "
                f"Got {type(memory_feedback).__name__}."
            ),
        }
    flat: list = []
    for gi, group in enumerate(memory_feedback):
        if not isinstance(group, dict):
            return {
                "success": False,
                "error": (
                    f"memory_feedback[{gi}] must be a group object "
                    "{context_id, feedback: [...]}. "
                    f"Got {type(group).__name__}."
                ),
            }
        ctx_id = group.get("context_id")
        entries = group.get("feedback")
        if not isinstance(ctx_id, str) or not ctx_id.strip():
            return {
                "success": False,
                "error": (
                    f"memory_feedback[{gi}].context_id is required "
                    "(non-empty string). The Context id that surfaced "
                    "the memories -- see `missing_injected` map in a "
                    "failed finalize for the expected values."
                ),
            }
        if not isinstance(entries, list):
            return {
                "success": False,
                "error": (
                    f"memory_feedback[{gi}].feedback must be a list of entry "
                    f"dicts. Got {type(entries).__name__} for context_id "
                    f"{ctx_id!r}."
                ),
            }
        for e in entries:
            if isinstance(e, dict):
                e2 = dict(e)
                e2.setdefault("_context_id", str(ctx_id))
                flat.append(e2)
                _memory_feedback_by_context.setdefault(str(ctx_id), []).append(e2)
    memory_feedback = flat

    # DIAGNOSTIC: post-normalization snapshot. If flat_len=0 despite
    # non-empty input, the dict→list expansion silently dropped every
    # entry (isinstance(e, dict) check at the inner loop).
    if _dbg_enabled:
        try:
            _dbg.warning(
                "FINALIZE_POST_NORM flat_len=%d first=%r by_ctx_keys=%r",
                len(memory_feedback) if isinstance(memory_feedback, list) else -1,
                (
                    memory_feedback[:1]
                    if isinstance(memory_feedback, list) and memory_feedback
                    else None
                ),
                list(_memory_feedback_by_context.keys()),
            )
        except Exception:
            pass

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

    # ── Two-tool redesign 2026-04-25: an intent that's been accepted
    # but is awaiting feedback may NOT be re-finalized. Use
    # mempalace_extend_feedback to provide the remaining ratings.
    if _mcp._STATE.active_intent.get("pending_feedback"):
        _pf = _mcp._STATE.active_intent["pending_feedback"]
        return {
            "success": False,
            "error": (
                "This intent has already been accepted (execution entity "
                f"{_pf.get('execution_entity', '?')} created). It is awaiting "
                "remaining feedback via mempalace_extend_feedback. Do NOT call "
                "mempalace_finalize_intent again on this intent -- it accepts "
                "metadata once and never re-runs."
            ),
            "execution_entity": _pf.get("execution_entity", ""),
        }

    # fail-fast agent validation. Before P6.1 an undeclared agent
    # would silently break result/trace/learning memory creation deep
    # inside _add_memory_internal; now we reject upfront with the same
    # recipe the hook teaches.
    agent_err = _mcp._require_agent(agent, action="finalize_intent")
    if agent_err:
        return agent_err

    intent_type = _mcp._STATE.active_intent["intent_type"]
    intent_desc = _mcp._STATE.active_intent.get("content", "")
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
    # Two gates:
    #   1. MIN_FEEDBACK_REASON char minimum (10) -- typing-time effort.
    #   2. Low-quality pattern blacklist -- catches cop-out reasons
    #      like "don't know" / "not used" / "N/A" that agents reach
    #      for to shortcut through coverage. These reasons poison
    #      Channel D (rated_useful / rated_irrelevant edges get written
    #      with fake context) and the DB accumulates misinformation.
    #      Force the agent to actually fetch the memory content via
    #      kg_query and evaluate it on its merits.
    MIN_FEEDBACK_REASON = 10
    # Patterns live at module level (see _LOW_QUALITY_REASON_PATTERNS
    # and _LOW_QUALITY_RE). Reference the compiled regex here.
    _low_quality_re = _LOW_QUALITY_RE

    def _reject_copout_memory(mem_id, ctx_id, reason, reason_detail):
        """Build a consistent cop-out rejection error for memory_feedback.
        reason_detail explains which gate fired (regex / semantic / both)
        so the agent can see why their reason was rejected."""
        return {
            "success": False,
            "error": (
                f"Memory feedback for '{mem_id or '?'}' has a LOW-QUALITY reason "
                f"({reason_detail}): {reason!r}. Cop-out reasons poison Channel D "
                f"-- rated edges get written with fake signal and the DB accumulates "
                f"misinformation session over session. "
                f"FIX: BEFORE rating, fetch BOTH sides -- "
                f"  (1) `mempalace_kg_query(entity='{ctx_id or '<context_id>'}')` "
                f"to see what the CONTEXT was about (what queries+keywords caused "
                f"this memory to surface), AND "
                f"  (2) `mempalace_kg_query(entity='{mem_id or '<memory_id>'}')` "
                f"to read the memory's actual content. "
                f"Then write a concrete reason that names: what the memory SAYS, "
                f"what the context ASKED, and why those two did or didn't match "
                f"for THIS intent. If the memory was truly unrelated, that IS the "
                f"rating -- relevance 1 or 2 with a reason explaining the topic "
                f"mismatch (e.g. 'memory is about X, context asked about Y, no overlap'). "
                f"'Never used' / 'don't know' / 'unclear' are not rating reasons -- "
                f"they are admissions you didn't read the memory."
            ),
        }

    def _reject_copout_op(tool, ctx_id, reason, reason_detail):
        """Parallel cop-out rejection for operation_ratings."""
        return {
            "success": False,
            "error": (
                f"operation_ratings entry for ({tool or '?'}, {ctx_id or '?'}) has a "
                f"LOW-QUALITY reason ({reason_detail}): {reason!r}. Cop-out reasons "
                f"corrupt performed_well / performed_poorly edges. "
                f"FIX: BEFORE rating, fetch the CONTEXT to see what intention the op "
                f"was serving -- `mempalace_kg_query(entity='{ctx_id or '<context_id>'}')` "
                f"returns the queries+keywords that drove your declare_operation. "
                f"Then evaluate: was the tool the right choice for THAT intention? "
                f"Were the args appropriate? If good: explain what made it the right "
                f"call against that context. If bad: name the specific failure mode "
                f"(wrong tool, wrong scope, redundant with earlier call, missed a "
                f"shortcut, etc.). 'Skipped' / 'N/A' / 'not sure' are not ratings."
            ),
        }

    # ── Partial-accept cop-out gate (Adrian's design 2026-04-25) ──
    # Per-entry validation: GOOD entries continue through to persistence
    # so the caller's effort isn't wasted; BAD entries (cop-out / too
    # short) are split out and reported in the response with a
    # `rejected_feedback` / `rejected_operations` list. The caller sees
    # exactly which ratings to redo and can re-submit them via
    # mempalace_extend_feedback. Previously a single bad reason aborted
    # the whole batch, so a mostly-good payload of 80 ratings had to be
    # re-typed in full to fix one cop-out -- that asymmetry is gone.
    rejected_feedback: list = []
    rejected_operations: list = []
    if memory_feedback:
        _kept_memory_feedback: list = []
        for fb in memory_feedback:
            reason = (fb.get("reason") or "").strip()
            fb_id = fb.get("id") or ""
            fb_ctx = fb.get("_context_id") or ""
            if len(reason) < MIN_FEEDBACK_REASON:
                rejected_feedback.append(
                    {
                        "id": fb_id,
                        "context_id": fb_ctx,
                        "reason": reason,
                        "rejection_detail": (
                            f"too short ({len(reason)} < {MIN_FEEDBACK_REASON} chars)"
                        ),
                    }
                )
                continue
            if _low_quality_re.search(reason):
                rejected_feedback.append(
                    {
                        "id": fb_id,
                        "context_id": fb_ctx,
                        "reason": reason,
                        "rejection_detail": "regex pattern match",
                    }
                )
                continue
            _sem_hit, _sim = _semantic_copout_check(reason)
            if _sem_hit:
                rejected_feedback.append(
                    {
                        "id": fb_id,
                        "context_id": fb_ctx,
                        "reason": reason,
                        "rejection_detail": (
                            f"semantic similarity {_sim:.2f} to cop-out exemplars"
                        ),
                    }
                )
                continue
            _kept_memory_feedback.append(fb)
        # Replace memory_feedback with the accepted-only list so
        # downstream persistence and coverage computation only sees
        # GOOD entries. Rejected ones surface in the response below.
        memory_feedback = _kept_memory_feedback

    # Same partial-accept gate for operation_ratings. performed_well /
    # performed_poorly edges written with "don't know" or "didn't use"
    # reasons corrupt the op-tier retrieval; rejected entries are
    # surfaced in the response so the caller can rate them properly.
    if operation_ratings and isinstance(operation_ratings, list):
        _kept_operation_ratings: list = []
        for _opr in operation_ratings:
            if not isinstance(_opr, dict):
                continue
            _opr_reason = (_opr.get("reason") or "").strip()
            _opr_tool = _opr.get("tool") or ""
            _opr_ctx = _opr.get("context_id") or ""
            if len(_opr_reason) < MIN_FEEDBACK_REASON:
                rejected_operations.append(
                    {
                        "tool": _opr_tool,
                        "context_id": _opr_ctx,
                        "reason": _opr_reason,
                        "rejection_detail": (
                            f"too short ({len(_opr_reason)} < {MIN_FEEDBACK_REASON} chars)"
                        ),
                    }
                )
                continue
            if _low_quality_re.search(_opr_reason):
                rejected_operations.append(
                    {
                        "tool": _opr_tool,
                        "context_id": _opr_ctx,
                        "reason": _opr_reason,
                        "rejection_detail": "regex pattern match",
                    }
                )
                continue
            _opr_sem_hit, _opr_sim = _semantic_copout_check(_opr_reason)
            if _opr_sem_hit:
                rejected_operations.append(
                    {
                        "tool": _opr_tool,
                        "context_id": _opr_ctx,
                        "reason": _opr_reason,
                        "rejection_detail": (
                            f"semantic similarity {_opr_sim:.2f} to cop-out exemplars"
                        ),
                    }
                )
                continue
            _kept_operation_ratings.append(_opr)
        operation_ratings = _kept_operation_ratings

    # Stash on the active intent so the partial-finalize / extend_feedback
    # response surface in this same call can include them. The downstream
    # response builders read `rejected_feedback` / `rejected_operations`
    # off this struct or via the closure below.
    _rejected_summary_for_response = {
        "rejected_feedback": list(rejected_feedback),
        "rejected_operations": list(rejected_operations),
    }

    # ── Validate memory feedback coverage ──
    # DIAGNOSTIC: raw state before coverage computation. If injected_ids
    # appear here but don't match feedback_ids below, it's an encoding /
    # normalization mismatch. If feedback_ids is empty despite provided
    # feedback, it's an entry-iteration silent-skip.
    if _dbg_enabled:
        try:
            _raw_injected_dbg = _mcp._STATE.active_intent.get("injected_memory_ids", set())
            _raw_accessed_dbg = _mcp._STATE.active_intent.get("accessed_memory_ids", set())
            _dbg.warning(
                "FINALIZE_COVERAGE_IN injected_type=%s injected=%r "
                "accessed_type=%s accessed_len=%d mf_len=%d",
                type(_raw_injected_dbg).__name__,
                sorted(list(_raw_injected_dbg))[:20],
                type(_raw_accessed_dbg).__name__,
                len(_raw_accessed_dbg) if hasattr(_raw_accessed_dbg, "__len__") else -1,
                len(memory_feedback) if isinstance(memory_feedback, list) else -1,
            )
        except Exception:
            pass

    injected_ids = {x for x in _mcp._STATE.active_intent.get("injected_memory_ids", set()) if x}
    accessed_ids = {x for x in _mcp._STATE.active_intent.get("accessed_memory_ids", set()) if x}

    # ── Slice B-4b: first-rater coverage exemption ──
    # When this intent inherits a user-context (cause_kind='user_context')
    # whose surfaced memories were already rated by some prior agent
    # intent in this session, subtract those ids from the coverage sets.
    # The snapshot was taken at declare_intent time and persists on
    # active_intent. First-rater intents see no exemption (full coverage
    # required); subsequent intents skip the user-context-surfaced subset.
    _is_first_rater = bool(_mcp._STATE.active_intent.get("user_context_first_rater", True))
    if not _is_first_rater:
        _exempt = {
            x for x in _mcp._STATE.active_intent.get("user_context_exempt_ids", []) or [] if x
        }
        if _exempt:
            injected_ids = injected_ids - _exempt
            accessed_ids = accessed_ids - _exempt

    feedback_ids = set()
    if memory_feedback:
        for fb in memory_feedback:
            raw_id = (fb.get("id") or "").strip()
            if raw_id:
                # Store both raw and normalized forms so either matches
                feedback_ids.add(raw_id)
                feedback_ids.add(normalize_entity_name(raw_id))

    # ── Two-tool redesign 2026-04-25: capture coverage misses instead
    # of early-return. Entity creation + writes proceed regardless;
    # the final all-complete check at the bottom decides whether to
    # finalize formally OR transition into pending_feedback state for
    # mempalace_extend_feedback to close out. NO partial-write loss:
    # whatever the agent provided gets persisted on the first call.
    _pending_missing_injected_by_ctx: dict = {}
    _pending_missing_accessed: list = []
    _pending_missing_op_keys: dict = {}
    _pending_injected_coverage: float = 1.0
    _pending_accessed_coverage: float = 1.0

    # Injected memories: 100% feedback required
    if injected_ids:
        missing_injected = injected_ids - feedback_ids
        # DIAGNOSTIC: show both sets side-by-side when coverage fails so
        # the root cause (empty feedback_ids vs ID mismatch) is obvious.
        if _dbg_enabled and missing_injected:
            try:
                _dbg.warning(
                    "FINALIZE_COVERAGE_MISS feedback_ids_len=%d "
                    "feedback_ids_sample=%r injected=%r missing=%r",
                    len(feedback_ids),
                    sorted(list(feedback_ids))[:20],
                    sorted(list(injected_ids))[:10],
                    sorted(list(missing_injected))[:10],
                )
            except Exception:
                pass
        if missing_injected:
            coverage = (len(injected_ids) - len(missing_injected)) / len(injected_ids)

            # Group missing ids by the context that surfaced them so
            # agents can attribute each rating to the correct ctx_id
            # key in the memory_feedback map. Without this grouping,
            # when two contexts surface overlapping sets the agent
            # has no way to know which ctx_id to attach a rating to,
            # so Channel D's per-context feedback edges get misrouted.
            # Uncovered bucket "(unknown_context)" collects any ids
            # whose origin wasn't tracked (legacy state files, hook
            # writers we haven't instrumented yet).
            _injected_by_ctx = _mcp._STATE.active_intent.get("injected_by_context", {}) or {}
            _id_to_ctx: dict = {}
            for _ctx, _ids in _injected_by_ctx.items():
                for _mid in _ids or []:
                    _id_to_ctx.setdefault(_mid, _ctx)
            missing_by_context: dict = {}
            for _mid in missing_injected:
                _ctx = _id_to_ctx.get(_mid, "(unknown_context)")
                missing_by_context.setdefault(_ctx, []).append(_mid)
            # Slice 1a 2026-04-28: enrich each id with its summary.what so
            # the model can read what each missing reference is about
            # without a follow-up kg_query.
            missing_by_context = {
                k: _enrich_ids_with_summaries(sorted(v)) for k, v in missing_by_context.items()
            }

            # CAPTURE -- do not return. The final all-complete check at the
            # bottom of finalize_intent decides whether to formally finalize
            # or transition into pending_feedback state.
            _pending_missing_injected_by_ctx = missing_by_context
            _pending_injected_coverage = coverage

    # Accessed memories: 100% feedback required (excluding already-covered injected)
    MIN_ACCESSED_COVERAGE = 1.0
    accessed_only = accessed_ids - injected_ids
    if accessed_only:
        accessed_covered = len(accessed_only & feedback_ids)
        accessed_coverage = accessed_covered / len(accessed_only)
        if accessed_coverage < MIN_ACCESSED_COVERAGE:
            # Slice 1a 2026-04-28: enrich each id with summary.what so the
            # model can read what each missing reference is about without a
            # follow-up kg_query.
            missing_accessed = _enrich_ids_with_summaries(sorted(accessed_only - feedback_ids))
            # CAPTURE -- do not return. Same rationale as the injected gate.
            _pending_missing_accessed = missing_accessed
            _pending_accessed_coverage = accessed_coverage

    # ── Read execution trace from hook state file ──
    trace_entries = []
    if not _mcp._STATE.session_id:
        # No sid means we never had a private trace file. Skipping is
        # correct -- falling back to execution_trace_default.jsonl would
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

    # ── MANDATORY operation_ratings coverage ──
    # Parallel to memory_feedback coverage: every (tool, context_id) pair
    # that appeared in the execution trace requires a rating entry in
    # operation_ratings. Empty trace → empty requirement (legitimate
    # no-ops intents succeed without ratings). Optional-fields-get-
    # ignored rule applies: if this weren't mandatory, agents would
    # skip and the performed_well / performed_poorly signal would
    # never accumulate.
    #
    # One rating per (tool, context_id) pair covers any number of
    # repeated calls within that pair -- the context fingerprint IS
    # the unit of learning, so rating once per unique pair is enough.
    _required_op_keys = set()
    for _te in trace_entries:
        _te_tool = (_te.get("tool") or "").strip()
        _te_ctx = (_te.get("context_id") or "").strip()
        if _te_tool and _te_ctx:
            _required_op_keys.add((_te_tool, _te_ctx))
    _rated_op_keys = set()
    if operation_ratings and isinstance(operation_ratings, list):
        for _r in operation_ratings:
            if not isinstance(_r, dict):
                continue
            _r_tool = (_r.get("tool") or "").strip()
            _r_ctx = (_r.get("context_id") or "").strip()
            if _r_tool and _r_ctx:
                _rated_op_keys.add((_r_tool, _r_ctx))
    _missing_op_keys = _required_op_keys - _rated_op_keys
    if _missing_op_keys:
        _missing_by_ctx: dict = {}
        for _t, _c in _missing_op_keys:
            _missing_by_ctx.setdefault(_c, []).append(_t)
        _missing_by_ctx = {_c: sorted(set(_ts)) for _c, _ts in _missing_by_ctx.items()}
        # CAPTURE -- do not return. The all-complete check at the bottom
        # decides finalize vs pending_feedback transition.
        _pending_missing_op_keys = _missing_by_ctx

    # ── Create execution entity ──
    # Full description stored in SQLite (for display)
    # Execution-entity description shows the distilled summary directly --
    # summary is already ≤280 chars by construction, no slicing needed.
    exec_description = f"{intent_desc or intent_type}: {summary}"
    # Embedding uses description-only (no summary) so similar intents cluster
    embed_description = intent_desc or intent_type
    try:
        _mcp._create_entity(
            exec_id,
            kind="entity",
            content=exec_description,
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
    # Shared errors list -- initialized here so the S1 operation-rating
    # promotion (which runs before the result-memory block) can append
    # without a NameError. The result-memory block below re-uses this
    # same list, so duplicate init is avoided.
    errors: list = []

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

    # outcome as has_value -- unskipped 2026-04-25 (see _TRIPLE_SKIP_PREDICATES
    # comment for rationale). The statement verbalises the value pair so the
    # triple becomes a first-class search target ("intent X concluded with
    # outcome success/partial/failed/abandoned"), which is exactly the lookup
    # future agents make when auditing past intents by outcome.
    try:
        _hv_stmt = f"Intent execution {exec_id} concluded with outcome {outcome}"
        _mcp._STATE.kg.add_triple(exec_id, "has_value", outcome, statement=_hv_stmt)
        edges_created.append(f"{exec_id} has_value {outcome}")
    except Exception:
        pass

    # ── Slice B-4a: caused_by edge from execution entity to parent cause ──
    # When declare_intent stashed cause_id / cause_kind on active_intent
    # (Slice B-3 path: optional parent linkage to a user-context or Task
    # entity), the execution entity inherits that linkage so future audits
    # can trace activity-tier executions back to the user message that
    # provoked them. caused_by is non-skip-list so add_triple requires a
    # natural-language statement (per the 2026-04-19 lock that retired
    # autogenerated verbalisations). Soft-fail at edge level so a transient
    # kg issue does not prevent finalization itself.
    _cause_id_for_edge = _mcp._STATE.active_intent.get("cause_id") or ""
    _cause_kind_for_edge = _mcp._STATE.active_intent.get("cause_kind") or ""
    if _cause_id_for_edge:
        _cb_stmt = (
            f"Execution {exec_id} was caused by "
            f"{_cause_kind_for_edge.replace('_', ' ') or 'parent context'} "
            f"{_cause_id_for_edge} per the user-intent tier hierarchy."
        )
        try:
            _mcp._STATE.kg.add_triple(
                exec_id,
                "caused_by",
                _cause_id_for_edge,
                statement=_cb_stmt,
            )
            edges_created.append(f"{exec_id} caused_by {_cause_id_for_edge}")
        except Exception:
            pass

    # ── S1: Operation-rating promotion ──
    # For each rating with quality != 3, create a kind='operation' entity
    # (graph-only -- _sync_entity_to_chromadb gates on kind) and attach:
    #   exec_id --executed_op--> op_id        (parent/child audit trail)
    #   context_id --performed_well--> op_id  (when quality >= 4)
    #   context_id --performed_poorly--> op_id (when quality <= 2)
    # The op entity's id is a deterministic sha12 fingerprint over the
    # salient components, so ratings of the "same shape" op across
    # sessions collide -- necessary for gardener clustering in S3.
    # Reference: arXiv 2512.18950 Operation tier; Leontiev 1981 AAO.
    promoted_op_ids = []
    if operation_ratings and isinstance(operation_ratings, list):
        import hashlib as _op_hashlib

        for i, _rating in enumerate(operation_ratings):
            if not isinstance(_rating, dict):
                continue
            _quality = _rating.get("quality")
            if not isinstance(_quality, int) or _quality < 1 or _quality > 5:
                continue
            if _quality == 3:
                continue  # Neutral -- skip promotion
            _ctx_id = str(_rating.get("context_id") or "").strip()
            if not _ctx_id:
                continue
            _tool = str(_rating.get("tool") or "").strip()
            if not _tool:
                continue
            # 2026-04-27 redesign: args_summary now lives on the active
            # intent's op_args_by_ctx_tool store, populated at declare-
            # time. Promotion looks it up by (context_id, tool) instead
            # of reading the rating-side field (which was optional and
            # universally skipped, leading to empty fingerprints).
            _op_args_store = _mcp._STATE.active_intent.get("op_args_by_ctx_tool") or {}
            _args_summary = str(_op_args_store.get(f"{_ctx_id}|{_tool}", ""))[:400]
            _reason = str(_rating.get("reason") or "")
            # Deterministic op_id fingerprint. Salient components only --
            # session id is NOT included because we want same-shape ops
            # across sessions to collide (gardener S3 relies on that).
            _fp = f"{_tool}|{_args_summary}|{_ctx_id}"
            _op_hash = _op_hashlib.sha256(_fp.encode("utf-8", errors="replace")).hexdigest()[:12]
            _tool_slug = re.sub(r"[^a-zA-Z0-9]+", "_", _tool).strip("_").lower() or "op"
            _op_id = f"op_{_tool_slug}_{_op_hash}"
            _op_desc = f"{_tool} op: {_args_summary[:200]}" if _args_summary else f"{_tool} op"
            try:
                _mcp._create_entity(
                    _op_id,
                    kind="operation",
                    content=_op_desc,
                    importance=2,
                    properties={
                        "tool": _tool,
                        "args_summary": _args_summary,
                        "context_id": _ctx_id,
                        "quality": _quality,
                        "reason": _reason,
                        "rated_at": datetime.now().isoformat(timespec="seconds"),
                    },
                    added_by=agent,
                )
            except Exception as _e:
                errors.append(
                    {"kind": "operation_promotion", "error": f"exception creating {_op_id}: {_e}"}
                )
                continue
            # executed_op edge -- exec → op
            try:
                _exec_stmt = f"Execution {exec_id} performed a {_tool} operation" + (
                    f" on args {_args_summary[:80]!r}" if _args_summary else ""
                )
                _mcp._STATE.kg.add_triple(exec_id, "executed_op", _op_id, statement=_exec_stmt)
                edges_created.append(f"{exec_id} executed_op {_op_id}")
            except Exception:
                pass
            # Quality edge -- performed_well or performed_poorly
            _quality_pred = "performed_well" if _quality >= 4 else "performed_poorly"
            _quality_verb = "rated well (quality=" if _quality >= 4 else "rated poorly (quality="
            _reason_suffix = f" -- {_reason[:80]}" if _reason else ""
            try:
                _q_stmt = (
                    f"In context {_ctx_id}, the {_tool} op was "
                    f"{_quality_verb}{_quality}){_reason_suffix}"
                )
                _mcp._STATE.kg.add_triple(_ctx_id, _quality_pred, _op_id, statement=_q_stmt)
                edges_created.append(f"{_ctx_id} {_quality_pred} {_op_id}")
            except Exception:
                pass

            # ── S2: superseded_by correction edge ──
            # When a poorly-rated op carries a `better_alternative`
            # pointing to the op_id that SHOULD have been used in the
            # same context, record it as `(bad_op) superseded_by
            # (good_op)`. retrieve_past_operations walks this edge at
            # declare_operation time to surface concrete corrections,
            # not just cautionary precedent. Only written for quality
            # ≤2 (good ops don't need corrections) and only when the
            # caller supplied a non-empty better_alternative.
            if _quality <= 2:
                _better_alt = str(_rating.get("better_alternative") or "").strip()
                if _better_alt and _better_alt != _op_id:
                    try:
                        _sup_stmt = (
                            f"The {_tool} op {_op_id} (rated quality={_quality}) "
                            f"is superseded by {_better_alt} which is the correct "
                            f"alternative in context {_ctx_id}"
                        )
                        _mcp._STATE.kg.add_triple(
                            _op_id, "superseded_by", _better_alt, statement=_sup_stmt
                        )
                        edges_created.append(f"{_op_id} superseded_by {_better_alt}")
                    except Exception:
                        pass
            promoted_op_ids.append(_op_id)

    # ── Result memory (summary) ──
    # silent-failure surface: when _add_memory_internal rejects the
    # call (e.g. agent not declared, duplicate slug), we used to swallow
    # the error and return result_memory=null with no indication. Now
    # every failure is appended to `errors` and surfaced in the response.
    # (errors list initialized above, shared with S1 operation promotion.)
    result_memory_id = None
    try:
        # Result memory: the body is the agent's `content` wrapped with an
        # intent/outcome header; the ≤280-char distilled summary is the
        # agent's `summary` verbatim. No slicing, no auto-derivation -- the
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
            summary=_summary_dict,
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
    # polluted retrieval -- every finalize added another "Trace of X: N
    # tool call(s)" hit competing with actual prose. The same information
    # is already available without the embedded memory:
    #   - execution_trace_<sid>.jsonl on disk (the raw tool-call log,
    #     cleared after finalize reads it)
    #   - key_actions on the execution entity (distilled tool+target
    #     list, auto-filled from the trace above)
    #   - edges on the execution entity (executed_by, targeted, is_a)
    # If you ever need the blow-by-blow, read the JSONL file between
    # finalizes -- do not re-introduce a prose memory for it.

    # ── Gotchas ──
    # Strict dict-only contract (Adrian's design lock 2026-04-28):
    # each gotcha is {summary: {what, why, scope?}, content: str}.
    # The summary is rendered to prose for the entity description
    # (validated at source -- no auto-derive of summary from content),
    # and the content is preserved verbatim in entity.properties for
    # full retrieval. Strings are rejected with a migration error.
    if gotchas:
        try:
            from .knowledge_graph import (
                SummaryStructureRequired,
                coerce_summary_for_persist,
                serialize_summary_for_embedding,
            )
        except Exception:
            coerce_summary_for_persist = None
            serialize_summary_for_embedding = None
            SummaryStructureRequired = Exception
        for i, gotcha in enumerate(gotchas):
            try:
                if not isinstance(gotcha, dict):
                    errors.append(
                        {
                            "kind": "gotcha_entity",
                            "index": i,
                            "error": (
                                "gotcha must be "
                                "dict{summary: {what, why, scope?}, "
                                "content: str}; got "
                                f"{type(gotcha).__name__}. "
                                "Strings are rejected -- Adrian's design "
                                "lock 2026-04-28 forbids auto-derive of "
                                "summary from content."
                            ),
                        }
                    )
                    continue
                _g_content = str(gotcha.get("content") or "").strip()
                _g_summary = gotcha.get("summary")
                if not _g_content:
                    errors.append(
                        {
                            "kind": "gotcha_entity",
                            "index": i,
                            "error": (
                                f"gotcha[{i}].content is empty. Provide the verbatim gotcha body."
                            ),
                        }
                    )
                    continue
                if not isinstance(_g_summary, dict):
                    errors.append(
                        {
                            "kind": "gotcha_entity",
                            "index": i,
                            "error": (
                                f"gotcha[{i}].summary must be a dict "
                                f"{{what, why, scope?}}; got "
                                f"{type(_g_summary).__name__}. No "
                                f"auto-derive -- caller authors the "
                                f"WHAT and WHY."
                            ),
                        }
                    )
                    continue
                # Validate the summary dict via the shared gate so we
                # surface the same error messages the rest of the
                # write surface uses (field-level + 280-char rendered
                # cap).
                try:
                    _g_summary_dict = (
                        coerce_summary_for_persist(
                            _g_summary,
                            context_for_error=f"finalize.gotchas[{i}].summary",
                        )
                        if coerce_summary_for_persist
                        else _g_summary
                    )
                except SummaryStructureRequired as _se:
                    errors.append(
                        {
                            "kind": "gotcha_entity",
                            "index": i,
                            "error": str(_se),
                        }
                    )
                    continue
                _g_prose = (
                    serialize_summary_for_embedding(_g_summary_dict)
                    if serialize_summary_for_embedding
                    else str(_g_summary_dict)
                )
                if len(_g_prose) > _mcp._RECORD_SUMMARY_MAX_LEN:
                    errors.append(
                        {
                            "kind": "gotcha_entity",
                            "index": i,
                            "error": (
                                f"gotcha[{i}].summary rendered prose is "
                                f"{len(_g_prose)} chars; maximum is "
                                f"{_mcp._RECORD_SUMMARY_MAX_LEN}. Trim "
                                f"'why' or 'scope' so the prose form "
                                f"fits the embedding budget."
                            ),
                        }
                    )
                    continue
                # Entity name derives from summary.what (first 50 chars
                # normalised) -- this is the structured anchor of the
                # gotcha. content is verbatim narrative; we store it in
                # properties so future retrieval can pull the full body
                # without losing it to the 280-char prose cap.
                _g_what = str(_g_summary_dict.get("what") or "").strip()
                gotcha_id = normalize_entity_name(_g_what[:50])
                if not gotcha_id:
                    errors.append(
                        {
                            "kind": "gotcha_entity",
                            "index": i,
                            "error": (
                                f"gotcha[{i}].summary.what normalises "
                                f"to empty; provide a meaningful WHAT."
                            ),
                        }
                    )
                    continue
                existing = _mcp._STATE.kg.get_entity(gotcha_id)
                if not existing:
                    _mcp._create_entity(
                        gotcha_id,
                        kind="entity",
                        content=_g_prose,
                        importance=3,
                        properties={
                            "summary": _g_summary_dict,
                            "content": _g_content,
                        },
                        added_by=agent,
                    )
                # has_gotcha is NOT a skip predicate; we need a real
                # statement so the edge is searchable. The rendered
                # summary prose serves as the natural sentence.
                gotcha_sentence = f"Execution {exec_id} ran into this gotcha: {_g_prose}"
                _mcp._STATE.kg.add_triple(
                    exec_id,
                    "has_gotcha",
                    gotcha_id,
                    statement=gotcha_sentence,
                )
                edges_created.append(f"{exec_id} has_gotcha {gotcha_id}")
                if promote_gotchas_to_type:
                    type_sentence = (
                        f"Intent type '{intent_type}' has a recurring gotcha: {_g_prose}"
                    )
                    _mcp._STATE.kg.add_triple(
                        intent_type,
                        "has_gotcha",
                        gotcha_id,
                        statement=type_sentence,
                    )
                    edges_created.append(f"{intent_type} has_gotcha {gotcha_id}")
            except Exception as e:
                errors.append(
                    {
                        "kind": "gotcha_entity",
                        "index": i,
                        "error": f"exception: {e}",
                    }
                )

    # ── Learnings ──
    # Strict dict-only contract (Adrian's design lock 2026-04-28):
    # each learning is {summary: {what, why, scope?}, content: str}.
    # No string fallback -- strings used to be accepted with an
    # auto-derived summary dict (what="learning N of <type>",
    # why=<the string>, scope=<exec_id>), which violated the
    # no-auto-derive rule and overflowed the 280-char rendered cap
    # whenever the string was long. Caller now passes the structured
    # summary upfront; we forward it verbatim to _add_memory_internal.
    if learnings:
        for i, learning in enumerate(learnings):
            try:
                if not isinstance(learning, dict):
                    errors.append(
                        {
                            "kind": "learning_memory",
                            "index": i,
                            "error": (
                                "learning must be "
                                "dict{summary: {what, why, scope?}, "
                                "content: str}; got "
                                f"{type(learning).__name__}. "
                                "Strings are rejected -- Adrian's design "
                                "lock 2026-04-28 forbids auto-derive of "
                                "summary from content."
                            ),
                        }
                    )
                    continue
                _l_content = str(learning.get("content") or "").strip()
                _l_summary = learning.get("summary")
                if not _l_content:
                    errors.append(
                        {
                            "kind": "learning_memory",
                            "index": i,
                            "error": (
                                f"learning[{i}].content is empty. Provide the verbatim lesson body."
                            ),
                        }
                    )
                    continue
                if not isinstance(_l_summary, dict):
                    errors.append(
                        {
                            "kind": "learning_memory",
                            "index": i,
                            "error": (
                                f"learning[{i}].summary must be a dict "
                                f"{{what, why, scope?}}; got "
                                f"{type(_l_summary).__name__}. No "
                                f"auto-derive -- caller authors the "
                                f"WHAT and WHY."
                            ),
                        }
                    )
                    continue
                # Caller-provided summary dict passes through verbatim;
                # _add_memory_internal validates it via
                # coerce_summary_for_persist + the 280-char rendered
                # prose cap. Failures surface as record errors with
                # a clear field-level message -- no auto-derive at any
                # layer between caller and gate.
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
    # suggested with LLMs" -- every feedback path must be blocking,
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
            # normalize_entity_name, so normalize here too -- otherwise
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
            # for that memory -- best-effort fallback to preserve the
            # permissive legacy behaviour when contexts_touched was
            # empty.
            if not _fb_ctx:
                for _c in _contexts_touched:
                    _covered_pairs.add((_c, _fb_mid))

    # Surfaced-pairs coverage gap is no longer a hard reject (2026-04-25
    # bugfix completing the 99f81f9 two-tool migration). The legacy
    # block here used to early-return with success=False whenever any
    # surfaced (context, memory) pair lacked feedback -- that was the
    # all-or-nothing contract Adrian's redesign explicitly retired.
    # The three downstream coverage gates (injected / accessed / op
    # keys) already CAPTURE missings into _pending_missing_* and let
    # the function fall through to the pending_feedback parking
    # writer at the bottom; this fourth gate was missed in the cutover
    # and silently re-imposed the legacy contract on every finalize
    # whose surfaced edges exceeded the agent's feedback payload.
    #
    # We still compute _missing_pairs for downstream visibility (the
    # injected-pairs gate uses the same set source via the active
    # intent's accessed_memory_ids), but we DO NOT block here. Any
    # remaining surfaced-pair coverage gap is now handled by the
    # extend_feedback round-trip -- exactly the architecture Adrian
    # designed in finalize_incremental_idempotent_design.
    _missing_pairs = sorted(_required_pairs - _covered_pairs)

    # ── Memory relevance feedback ──
    #
    # P2: two write paths run in this block:
    #   (1) Legacy found_useful / found_irrelevant edges attached to the
    #       EXECUTION entity -- kept intact so the existing retrieval
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
                # in the P3 polish sweep -- context-scoped feedback is the
                # only signal the retrieval pipeline reads now.)
                #
                # Routes through kg.record_feedback -- the unified
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
                        pass  # Non-fatal -- decay reset is best-effort

                feedback_count += 1
            except Exception:
                pass

    # ── Link-prediction candidate upsert ──
    # For each reused context with net-positive per-context feedback,
    # accumulate Adamic-Adar evidence (1/log(|entities|)) on every
    # unordered entity pair inside that context. Dedup by (pair, ctx_id)
    # so re-observing the same context N times contributes exactly once.
    # Direct-edge short-circuit inside upsert_candidate drops pairs
    # already connected 1-hop in any direction -- the graph channel
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
            # contribute -- positive signal requires positive ratings.
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
            # guarantees log(n) >= log(2) ≈ 0.693 -- no zero-div risk and
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
    # Default interval is 1 hour; operators can opt into aggressive
    # mode via MEMPALACE_LINK_AUTHOR_AGGRESSIVE=1 to dispatch on
    # every finalize that has pending candidates. Closes 2026-04-25
    # audit finding #14 (jury throughput bottleneck once the gardener
    # max_batches=10 change starts pumping candidates faster).
    try:
        import os as _os

        from . import link_author as _la  # noqa: F811

        _aggressive = _os.environ.get("MEMPALACE_LINK_AUTHOR_AGGRESSIVE", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        _la._dispatch_if_due(_mcp._STATE.kg, interval_hours=0 if _aggressive else 1)
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
    # Channel D (context) doesn't map to any Rocchio field -- it surfaces
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
                # ctx at the time of ALL emits -- permissive).
                matches = _fb_ctx == ctx_id or (not _fb_ctx and _active_ctx_id == ctx_id)
                if not matches:
                    continue
                # Rocchio bucket: use the derived (relevant, confidence)
                # pair so single-field callers behave identically to
                # legacy two-field callers. Value for bucket is the raw
                # 1-5 score when relevant, else 0 -- matching historical
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

        # Update learned weights -- both scopes.
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
            # Telemetry: observability for the weight-learning loop.
            # Writes one line to ~/.mempalace/hook_state/weight_log.jsonl
            # each time set_learned_* is invoked. `is_tuned` = did
            # compute_learned_weights actually drift from the static
            # defaults (needs _A6_WEIGHT_SELFTUNE_ENABLED=True AND
            # ≥ min_samples rows).
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
                    _conn = _mcp._STATE.kg._conn()
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
                _mcp._telemetry_append_jsonl(
                    "weight_log.jsonl",
                    {
                        "ts": _dt.now(_tz.utc).isoformat(timespec="seconds"),
                        "trigger": "finalize_intent",
                        "intent_id": exec_id,
                        "agent": agent or "",
                        "selftune_enabled": bool(
                            getattr(_mcp._STATE.kg, "_A6_WEIGHT_SELFTUNE_ENABLED", False)
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

    # ── Two-tool redesign: branch on coverage completeness ──
    # The three coverage gates above CAPTURED missings instead of
    # early-returning. Now decide:
    #   - all complete → continue to formal finalization (deactivate
    #     intent, write sentinel, gardener trigger, telemetry).
    #   - any miss → keep active_intent in pending_feedback state so
    #     mempalace_extend_feedback can close coverage later. Entity +
    #     provided feedback are already written above; nothing is lost.
    _all_complete = (
        not _pending_missing_injected_by_ctx
        and not _pending_missing_accessed
        and not _pending_missing_op_keys
    )
    if not _all_complete:
        # Persist what's needed for extend_feedback to recompute coverage
        # later (and survive MCP server restart via _sync_from_disk).
        try:
            _mcp._STATE.active_intent["pending_feedback"] = {
                "execution_entity": exec_id,
                "intent_type": intent_type,
                "outcome": outcome,
                "agent": agent,
                "required_injected_ids": sorted(injected_ids),
                "required_accessed_ids": sorted(accessed_ids - injected_ids),
                "required_op_keys": [list(p) for p in _required_op_keys],
                "injected_by_context": _mcp._STATE.active_intent.get("injected_by_context", {})
                or {},
                "since": datetime.now().isoformat(),
            }
            _persist_active_intent()
        except Exception:
            pass
        _resp = {
            "success": False,
            "error": (
                "Intent accepted but feedback incomplete. The execution "
                "entity has been created and partial feedback recorded -- "
                "use mempalace_extend_feedback to provide the remaining "
                "entries. DO NOT call mempalace_finalize_intent again on "
                "this intent."
            ),
            "execution_entity": exec_id,
            "missing_injected": _pending_missing_injected_by_ctx or {},
            "missing_accessed": _pending_missing_accessed or [],
            "missing_operations": _pending_missing_op_keys or {},
            "feedback_coverage": {
                "injected": round(_pending_injected_coverage, 2),
                "accessed": round(_pending_accessed_coverage, 2),
            },
        }
        # Partial-accept gate: surface entries rejected for low-quality
        # reason so the caller knows exactly what to retry. Good entries
        # already wrote to the DB so this list is the ONLY redo work.
        if _rejected_summary_for_response["rejected_feedback"]:
            _resp["rejected_feedback"] = _rejected_summary_for_response["rejected_feedback"]
        if _rejected_summary_for_response["rejected_operations"]:
            _resp["rejected_operations"] = _rejected_summary_for_response["rejected_operations"]
        return _resp

    # ── Slice B-4b: register cause_id in rated_user_contexts ──
    # The intent finalized successfully under cause_kind='user_context';
    # add cause_id to the session-scoped rated set so the NEXT agent
    # intent declared with the same cause_id inherits the coverage and
    # skips the user-context-surfaced memories. Skip on cause_kind=='task'
    # (Task entities have no surfaced-memory inheritance contract) and
    # on no-cause intents. Read AT FINALIZE because active_intent gets
    # cleared next.
    _final_cause_id = _mcp._STATE.active_intent.get("cause_id") or ""
    _final_cause_kind = _mcp._STATE.active_intent.get("cause_kind") or ""
    if _final_cause_kind == "user_context" and _final_cause_id:
        try:
            _rated_user_contexts_for(
                _mcp._STATE.session_id or "",
            ).add(_final_cause_id)
        except Exception:
            pass

    # ── Deactivate intent ──
    _mcp._STATE.active_intent = None
    _persist_active_intent()

    # ── Write last-finalized marker for Stop-hook proof-of-done check ──
    # The never-stop rule requires the Stop hook to see that the LAST finalized
    # intent in this session was a wrap_up_session with outcome=success before
    # allowing a stop. Writing a session-scoped marker here gives the dep-free
    # hook a file to read without needing SQLite or Chroma. Best-effort -- any
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
        # S1: op entity ids promoted from operation_ratings. Empty list
        # when caller provided no ratings or every rating was quality=3.
        "promoted_op_ids": promoted_op_ids,
    }
    # Partial-accept: even on success, the gate may have rejected
    # some low-quality entries. Surface them so the caller can
    # resubmit polished reasons via mempalace_extend_feedback.
    if _rejected_summary_for_response["rejected_feedback"]:
        result["rejected_feedback"] = _rejected_summary_for_response["rejected_feedback"]
        result["success"] = False
        result["error"] = (
            "Some memory_feedback entries were rejected for low-quality "
            "reasons (cop-out / too short). Good entries were accepted "
            "and persisted; resubmit only the rejected ones via "
            "mempalace_extend_feedback with concrete reasons."
        )
    if _rejected_summary_for_response["rejected_operations"]:
        result["rejected_operations"] = _rejected_summary_for_response["rejected_operations"]
        result["success"] = False
        if "error" not in result:
            result["error"] = (
                "Some operation_ratings entries were rejected for low-"
                "quality reasons (cop-out / too short). Good entries "
                "were accepted; resubmit only the rejected ones via "
                "mempalace_extend_feedback with concrete reasons."
            )

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


# ════════════════════════════════════════════════════════════════════
# tool_extend_feedback -- sibling of tool_finalize_intent (2026-04-25).
#
# Two-tool design: tool_finalize_intent is called ONCE per intent and
# accepts the metadata (slug/outcome/content/summary) plus as much
# feedback as the agent has. If feedback coverage hits 100% on that
# call, the intent finalizes formally. Otherwise the execution entity
# stays in pending_feedback state, and the agent calls THIS tool with
# the remaining ratings -- no re-sending of metadata or already-rated
# entries. When THIS tool closes coverage to 100%, the intent
# formally finalizes (deactivate active_intent, write last-finalized
# sentinel, fire memory-gardener trigger, append finalize telemetry).
#
# Args mirror tool_finalize_intent's same-named params byte-for-byte:
#   memory_feedback : LIST-OF-GROUPS (same shape as finalize_intent)
#   operation_ratings : LIST (same shape as finalize_intent)
# Same validation rules: reason >= 10 chars, cop-out gate.
#
# Survives MCP server restart: pending_feedback state lives on
# active_intent, which _persist_active_intent() writes to
# ~/.mempalace/hook_state/active_intent_<sid>.json. _sync_from_disk()
# rehydrates it on each tool entry.
# ════════════════════════════════════════════════════════════════════


def tool_extend_feedback(  # noqa: C901
    agent: str,
    memory_feedback: list = None,
    operation_ratings: list = None,
):
    """Extend an in-flight intent's feedback. Use ONLY after
    tool_finalize_intent has accepted the intent but reported
    incomplete coverage. Cannot be used to start an intent or
    change metadata -- those are one-shot via finalize_intent."""
    _sync_from_disk()
    if not _mcp._STATE.active_intent:
        return {"success": False, "error": "No active intent."}
    pf = _mcp._STATE.active_intent.get("pending_feedback")
    if not pf:
        return {
            "success": False,
            "error": (
                "No intent awaiting feedback. mempalace_extend_feedback is "
                "for closing coverage on an intent that "
                "mempalace_finalize_intent has already accepted. Call "
                "finalize_intent first."
            ),
        }

    agent_err = _mcp._require_agent(agent, action="extend_feedback")
    if agent_err:
        return agent_err

    # ── memory_feedback shape coercion (mirror of finalize_intent) ──
    _memory_feedback_by_context: dict = {}
    if memory_feedback is None:
        memory_feedback = []
    if isinstance(memory_feedback, str):
        try:
            memory_feedback = json.loads(memory_feedback)
        except Exception:
            return {
                "success": False,
                "error": (
                    "memory_feedback arrived as an unparseable string. "
                    "Pass a list of groups: "
                    "[{context_id, feedback: [{id, relevance, reason}, ...]}, ...]"
                ),
            }
    if isinstance(memory_feedback, dict):
        return {
            "success": False,
            "error": (
                "memory_feedback dict shape is retired; use list-of-groups "
                "[{context_id, feedback: [{id, relevance, reason}, ...]}, ...]."
            ),
        }
    if not isinstance(memory_feedback, list):
        return {
            "success": False,
            "error": (
                f"memory_feedback must be a list of groups; got {type(memory_feedback).__name__}."
            ),
        }
    flat: list = []
    for gi, group in enumerate(memory_feedback):
        if not isinstance(group, dict):
            continue
        ctx_id = group.get("context_id")
        entries = group.get("feedback")
        if not isinstance(ctx_id, str) or not ctx_id.strip():
            continue
        if not isinstance(entries, list):
            continue
        for e in entries:
            if isinstance(e, dict):
                e2 = dict(e)
                e2.setdefault("_context_id", str(ctx_id))
                flat.append(e2)
                _memory_feedback_by_context.setdefault(str(ctx_id), []).append(e2)
    memory_feedback = flat

    # ── Validate reasons (same partial-accept gate as finalize_intent) ──
    # Rejected entries are collected and surfaced in the response so the
    # caller can resubmit only the bad ones; good entries persist.
    MIN_FEEDBACK_REASON = 10
    _low_quality_re = _LOW_QUALITY_RE
    rejected_feedback: list = []
    rejected_operations: list = []
    _kept_memory_feedback: list = []
    for fb in memory_feedback:
        reason = (fb.get("reason") or "").strip()
        fb_id = fb.get("id") or ""
        fb_ctx = fb.get("_context_id") or ""
        if len(reason) < MIN_FEEDBACK_REASON:
            rejected_feedback.append(
                {
                    "id": fb_id,
                    "context_id": fb_ctx,
                    "reason": reason,
                    "rejection_detail": (
                        f"too short ({len(reason)} < {MIN_FEEDBACK_REASON} chars)"
                    ),
                }
            )
            continue
        if _low_quality_re.search(reason):
            rejected_feedback.append(
                {
                    "id": fb_id,
                    "context_id": fb_ctx,
                    "reason": reason,
                    "rejection_detail": "regex pattern match (cop-out)",
                }
            )
            continue
        _sem_hit, _sim = _semantic_copout_check(reason)
        if _sem_hit:
            rejected_feedback.append(
                {
                    "id": fb_id,
                    "context_id": fb_ctx,
                    "reason": reason,
                    "rejection_detail": (f"semantic similarity {_sim:.2f} to cop-out exemplars"),
                }
            )
            continue
        _kept_memory_feedback.append(fb)
    memory_feedback = _kept_memory_feedback

    exec_id = pf.get("execution_entity", "")
    if not exec_id:
        return {
            "success": False,
            "error": "pending_feedback state missing execution_entity id.",
        }

    # ── Write new memory_feedback edges via kg.record_feedback ──
    new_feedback_ids: set = set()
    feedback_count = 0
    errors: list = []
    for fb in memory_feedback:
        mem_id = (fb.get("id") or "").strip()
        ctx_id = (fb.get("_context_id") or "").strip()
        if not mem_id or not ctx_id:
            continue
        relevance = fb.get("relevance")
        relevant = fb.get("relevant")
        if relevant is None and isinstance(relevance, int):
            relevant = relevance >= 3
        # Normalize the memory id the same way finalize_intent does so
        # the rated edge lands on the canonical entity. mem_id we read
        # above is the raw user-supplied id; pass through normalize so
        # already_rated_mem covers both raw and normalized forms below.
        norm_mem_id = normalize_entity_name(mem_id)
        # record_feedback signature is (context_id, target_id, target_kind,
        # *, relevance, reason, rater_kind, rater_id, confidence). The
        # original extend_feedback call used the wrong kwargs (memory_id /
        # relevant / agent), so every call raised TypeError, was swallowed
        # by the bare-except, and feedback_count stayed 0 -- the partial-
        # coverage state could never close. 2026-04-25 fix: align with
        # the finalize_intent caller at line ~3562.
        target_kind = "triple" if norm_mem_id.startswith("t_") else "entity"
        try:
            _mcp._STATE.kg.record_feedback(
                ctx_id,
                norm_mem_id,
                target_kind,
                relevance=int(relevance) if isinstance(relevance, int) else 3,
                reason=str(fb.get("reason", "") or ""),
                rater_kind="agent",
                rater_id=agent or "",
            )
            new_feedback_ids.add(mem_id)
            new_feedback_ids.add(normalize_entity_name(mem_id))
            feedback_count += 1
        except Exception as _e:
            errors.append(f"record_feedback {mem_id}: {_e}")

    # ── Promote new op_ratings (same logic as finalize_intent S1 block) ──
    promoted_op_ids: list = []
    new_op_keys: set = set()
    if operation_ratings and isinstance(operation_ratings, list):
        import hashlib as _op_hashlib

        for _rating in operation_ratings:
            if not isinstance(_rating, dict):
                continue
            _q = _rating.get("quality")
            if not isinstance(_q, int) or _q < 1 or _q > 5:
                continue
            _ctx = str(_rating.get("context_id") or "").strip()
            _tool = str(_rating.get("tool") or "").strip()
            if not _ctx or not _tool:
                continue
            _reason = str(_rating.get("reason") or "")
            if len(_reason) < MIN_FEEDBACK_REASON:
                rejected_operations.append(
                    {
                        "tool": _tool,
                        "context_id": _ctx,
                        "reason": _reason,
                        "rejection_detail": (
                            f"too short ({len(_reason)} < {MIN_FEEDBACK_REASON} chars)"
                        ),
                    }
                )
                continue
            if _low_quality_re.search(_reason):
                rejected_operations.append(
                    {
                        "tool": _tool,
                        "context_id": _ctx,
                        "reason": _reason,
                        "rejection_detail": "regex pattern match (cop-out)",
                    }
                )
                continue
            _opr_sem_hit, _opr_sim = _semantic_copout_check(_reason)
            if _opr_sem_hit:
                rejected_operations.append(
                    {
                        "tool": _tool,
                        "context_id": _ctx,
                        "reason": _reason,
                        "rejection_detail": (
                            f"semantic similarity {_opr_sim:.2f} to cop-out exemplars"
                        ),
                    }
                )
                continue
            new_op_keys.add((_tool, _ctx))
            if _q == 3:
                continue  # neutral; skip promotion
            # 2026-04-27 redesign: read args_summary from op_args store,
            # populated at declare_operation time. The rating-side
            # field is gone from the schema.
            _op_args_store_ef = _mcp._STATE.active_intent.get("op_args_by_ctx_tool") or {}
            _args = str(_op_args_store_ef.get(f"{_ctx}|{_tool}", ""))[:400]
            _fp = f"{_tool}|{_args}|{_ctx}"
            _h = _op_hashlib.sha256(_fp.encode("utf-8", errors="replace")).hexdigest()[:12]
            _slug = re.sub(r"[^a-zA-Z0-9]+", "_", _tool).strip("_").lower() or "op"
            _op_id = f"op_{_slug}_{_h}"
            _desc = f"{_tool} op: {_args[:200]}" if _args else f"{_tool} op"
            try:
                _mcp._create_entity(
                    _op_id,
                    kind="operation",
                    content=_desc,
                    importance=2,
                    properties={
                        "tool": _tool,
                        "args_summary": _args,
                        "context_id": _ctx,
                        "quality": _q,
                        "reason": _reason,
                        "rated_at": datetime.now().isoformat(timespec="seconds"),
                    },
                    added_by=agent,
                )
                _mcp._STATE.kg.add_triple(exec_id, "executed_op", _op_id)
                if _q >= 4:
                    _mcp._STATE.kg.add_triple(_ctx, "performed_well", _op_id)
                elif _q <= 2:
                    _mcp._STATE.kg.add_triple(_ctx, "performed_poorly", _op_id)
                promoted_op_ids.append(_op_id)
            except Exception as _e:
                errors.append(f"op_promote {_op_id}: {_e}")

    # ── Update already-rated tracking on pending_feedback ──
    already_rated_mem = set(pf.get("already_rated_memory_ids", []))
    already_rated_mem |= new_feedback_ids
    already_rated_ops = {tuple(p) for p in pf.get("already_rated_op_keys", [])}
    already_rated_ops |= new_op_keys
    pf["already_rated_memory_ids"] = sorted(already_rated_mem)
    pf["already_rated_op_keys"] = [list(p) for p in already_rated_ops]
    _mcp._STATE.active_intent["pending_feedback"] = pf
    _persist_active_intent()

    # ── Recompute coverage ──
    required_injected = set(pf.get("required_injected_ids", []))
    required_accessed = set(pf.get("required_accessed_ids", []))
    required_ops = {tuple(p) for p in pf.get("required_op_keys", [])}
    missing_injected = required_injected - already_rated_mem
    missing_accessed = required_accessed - already_rated_mem
    missing_ops = required_ops - already_rated_ops

    if missing_injected or missing_accessed or missing_ops:
        # missing_by_context for injected
        injected_by_ctx = pf.get("injected_by_context", {}) or {}
        id_to_ctx: dict = {}
        for _ctx, _ids in injected_by_ctx.items():
            for _mid in _ids or []:
                id_to_ctx.setdefault(_mid, _ctx)
        missing_inj_by_ctx: dict = {}
        for _mid in missing_injected:
            _c = id_to_ctx.get(_mid, "(unknown_context)")
            missing_inj_by_ctx.setdefault(_c, []).append(_mid)
        # Slice 1a 2026-04-28: enrich missing ids with summary.what so the
        # model can read what each missing reference is about without a
        # follow-up kg_query. Sister of the same enrichment in
        # tool_finalize_intent.
        missing_inj_by_ctx = {
            k: _enrich_ids_with_summaries(sorted(v)) for k, v in missing_inj_by_ctx.items()
        }
        missing_ops_by_ctx: dict = {}
        for _t, _c in missing_ops:
            missing_ops_by_ctx.setdefault(_c, []).append(_t)
        missing_ops_by_ctx = {_c: sorted(set(_ts)) for _c, _ts in missing_ops_by_ctx.items()}
        _resp = {
            "success": False,
            "complete": False,
            "execution_entity": exec_id,
            "feedback_count": feedback_count,
            "promoted_op_ids": promoted_op_ids,
            "missing_injected": missing_inj_by_ctx,
            "missing_accessed": _enrich_ids_with_summaries(sorted(missing_accessed)),
            "missing_operations": missing_ops_by_ctx,
            "error": (
                "Coverage still incomplete after merge. Provide remaining "
                f"feedback via mempalace_extend_feedback. Missing: "
                f"{len(missing_injected)} injected, "
                f"{len(missing_accessed)} accessed, "
                f"{len(missing_ops)} operations."
            ),
            # Surface any per-entry exceptions captured during the merge
            # loop so silent failures (e.g. signature drift to record_feedback,
            # bad context_id) become loudly diagnosable instead of leaving
            # feedback_count=0 unexplained. Cap at 5 to keep the payload
            # small; the rest are still logged via record_hook_error.
            "errors": errors[:5],
        }
        # Partial-accept: surface low-quality rejections so the caller
        # knows exactly which entries to retype. The good ones already
        # persisted via record_feedback above and feedback_count
        # reflects them.
        if rejected_feedback:
            _resp["rejected_feedback"] = rejected_feedback
        if rejected_operations:
            _resp["rejected_operations"] = rejected_operations
        return _resp

    # ── Coverage closed: formal finalization ──
    intent_type = pf.get("intent_type", "")
    outcome = pf.get("outcome", "success")
    sid = _mcp._STATE.session_id or ""

    # ── Slice B-4b: register cause_id in rated_user_contexts ──
    # Mirror tool_finalize_intent: when this multi-call coverage flow
    # closes against a user-context cause, mark it rated so subsequent
    # intents inherit the coverage.
    _ext_cause_id = _mcp._STATE.active_intent.get("cause_id") or ""
    _ext_cause_kind = _mcp._STATE.active_intent.get("cause_kind") or ""
    if _ext_cause_kind == "user_context" and _ext_cause_id:
        try:
            _rated_user_contexts_for(sid).add(_ext_cause_id)
        except Exception:
            pass

    _mcp._STATE.active_intent = None
    _persist_active_intent()

    try:
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
        try:
            from . import hooks_cli as _hc

            _hc._record_hook_error("tool_extend_feedback.last_finalized_marker", _e)
        except Exception:
            pass

    try:
        from . import memory_gardener as _mg

        _mg.maybe_trigger_from_finalize(_mcp._STATE.kg)
    except Exception:
        pass

    try:
        from datetime import timezone as _tz

        contexts_used = sorted(set(_memory_feedback_by_context.keys()))
        _mcp._telemetry_append_jsonl(
            "finalize_log.jsonl",
            {
                "ts": datetime.now(_tz.utc).isoformat(timespec="seconds"),
                "intent_id": exec_id,
                "contexts_used": contexts_used,
                "memories_rated": feedback_count,
                "outcome": outcome,
                "agent": agent or "",
                "via": "extend_feedback",
            },
        )
    except Exception:
        pass

    result = {
        "success": True,
        "complete": True,
        "execution_entity": exec_id,
        "outcome": outcome,
        "feedback_count": feedback_count,
        "promoted_op_ids": promoted_op_ids,
    }
    if errors:
        result["errors"] = errors
        result["warning"] = f"{len(errors)} record error(s) during merge."
    # Partial-accept: even on a complete-coverage success some entries
    # may have been rejected for cop-out reasons. Surface them so the
    # caller can resubmit polished reasons (which will overwrite via
    # last-write-wins on the same memory_id+context_id pair).
    if rejected_feedback:
        result["rejected_feedback"] = rejected_feedback
        result["success"] = False
        result["error"] = (
            "Some memory_feedback entries were rejected for low-quality "
            "reasons. Coverage closed on the accepted ones; resubmit the "
            "rejected ones with concrete reasons to overwrite."
        )
    if rejected_operations:
        result["rejected_operations"] = rejected_operations
        result["success"] = False
        if "error" not in result:
            result["error"] = (
                "Some operation_ratings entries were rejected for low-"
                "quality reasons. Coverage closed on the accepted ones; "
                "resubmit the rejected ones with concrete reasons."
            )
    return result
