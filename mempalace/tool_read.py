"""Read-only mempalace tool handlers (read bucket).

This module re-exports the read-only tool handlers from ``mempalace.mcp_server``
so the PreToolUse carve-out hook can determine bucket membership by reading
``__all__``. Handler bodies stay in ``mcp_server`` -- moving them here would
shuffle code for zero behavioural change.

The hook does NOT import this module at runtime (that would chain into the
heavy ``mcp_server`` import on every PreToolUse call). Instead, the hook
hardcodes the bucket basenames in ``hooks_cli._READ_BUCKET_BASENAMES`` and
``tests/test_hook_buckets.py::test_read_bucket_matches_module_all`` enforces
the two stay in sync. If a handler is added or moves bucket, update BOTH
sides -- the drift-sentinel test breaks loudly otherwise.

Bucket semantics: read tools NEVER mutate state. The gate hook always allows
them outside user-message preemption; under preemption they're blocked along
with everything else except the user-intent tier-0 carve-outs
(``declare_user_intents``, ``extend_feedback``) and ``AskUserQuestion``.
"""

from datetime import datetime, timezone  # noqa: E402

from mempalace.query_sanitizer import sanitize_query  # noqa: E402
from mempalace.scoring import multi_channel_search, walk_rated_neighbourhood  # noqa: E402

# Phase 2: lazy mcp_server imports inside each function body to avoid
# the circular when this module is imported BEFORE mcp_server has
# finished loading. Each function imports only what it uses.


# ── Phase 2 (pilot): bodies migrated for tool_kg_stats + tool_kg_timeline.
# These two are the simplest read handlers -- only depend on _STATE -- so they
# pilot the circular-import pattern without risk. mcp_server.py imports
# these two back at end-of-file for TOOLS dispatch. Phase 2 will roll out
# to the larger handlers as their dependency footprints get analyzed.
def tool_kg_timeline(entity: str = None, limit: int = 100):
    """Get chronological timeline of facts, optionally for one entity.

    `limit` caps the most-recent N results. Defaults to 100. Pass a
    larger value if you need the full history (the underlying timeline
    fetch is unbounded; this is purely a response-size guard so the
    common case doesn't blast the agent's context window).
    """
    from mempalace.mcp_server import (
        _STATE,
    )

    results = _STATE.kg.timeline(entity)
    if isinstance(limit, int) and limit > 0:
        results = results[:limit]
    return {"timeline": results, "count": len(results)}


def tool_kg_stats():
    """Knowledge graph overview -- entities, triples, relationship types."""
    from mempalace.mcp_server import (
        _STATE,
    )

    stats = _STATE.kg.stats() or {}
    return stats


def tool_kg_query(
    entity: str,
    as_of: str = None,
    direction: str = "both",
    include_context_edges: bool = False,
):
    """Query the knowledge graph for an entity's relationships AND its own content.

    Supports batch queries: pass a comma-separated list of entity names
    to query multiple entities in one call. Returns results keyed by entity.

    Each response carries:
      - `facts`: list of (subject, predicate, object) triples -- the edges.
      - `details`: the entity's own content/summary/kind/importance pulled
        from its representative Chroma record. Omitted when the entity has
        no record (rare -- most entities carry at least a declaration view).

    By default, retrieval-bookkeeping edges (rated_useful,
    rated_irrelevant, surfaced) are omitted from `facts` -- they fill the
    fact list with per-context noise that drowns out domain knowledge.
    Pass include_context_edges=True to see them (e.g. for retrieval
    audits). When any are hidden, a hidden_context_edges count is
    included in the response so callers know they exist.
    """
    from mempalace.mcp_server import (
        _STATE,
        _fetch_entity_details,
        _filter_context_edges,
    )

    entities = [e.strip() for e in entity.split(",") if e.strip()]

    # Track queried entities for mandatory feedback enforcement.
    # Bug 3 Piece B 2026-04-28: skip the add when active_intent is in
    # pending_feedback state (mid-finalize). Coverage is frozen at the
    # snapshot taken when finalize first fired; subsequent reads are
    # allowed (the lockdown gate explicitly permits read-bucket tools so
    # the agent can look up content to rate it) but they don't grow
    # the coverage requirement. Without this, the lockdown alone wouldn't
    # stop the snowball -- read tools would keep adding new ids to the
    # set the agent has to rate.
    if (
        _STATE.active_intent
        and isinstance(_STATE.active_intent.get("accessed_memory_ids"), set)
        and not _STATE.active_intent.get("pending_feedback")
    ):
        for ename in entities:
            _STATE.active_intent["accessed_memory_ids"].add(ename)

    if len(entities) == 1:
        # Single entity -- original format for backwards compatibility
        results = _STATE.kg.query_entity(entities[0], as_of=as_of, direction=direction)
        hidden = 0
        if not include_context_edges:
            results, hidden = _filter_context_edges(results)
        out = {"entity": entities[0], "as_of": as_of, "facts": results, "count": len(results)}
        details = _fetch_entity_details(entities[0])
        if details:
            out["details"] = details
        if hidden:
            out["hidden_context_edges"] = hidden
        return out

    # Batch query -- return results keyed by entity name
    batch_results = {}
    total_count = 0
    total_hidden = 0
    for ename in entities:
        facts = _STATE.kg.query_entity(ename, as_of=as_of, direction=direction)
        hidden = 0
        if not include_context_edges:
            facts, hidden = _filter_context_edges(facts)
        entry = {"facts": facts, "count": len(facts)}
        details = _fetch_entity_details(ename)
        if details:
            entry["details"] = details
        if hidden:
            entry["hidden_context_edges"] = hidden
        batch_results[ename] = entry
        total_count += len(facts)
        total_hidden += hidden

    out = {"entities": batch_results, "as_of": as_of, "total_count": total_count, "batch": True}
    if total_hidden:
        out["total_hidden_context_edges"] = total_hidden
    return out


def tool_kg_search(  # noqa: C901
    context: dict = None,
    limit: int = 10,
    kind: str = None,
    sort_by: str = "hybrid",
    agent: str = None,
    time_window: dict = None,  # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    queries=None,  # LEGACY: rejected when context missing -- see below
):
    """Unified search -- records (prose) + entities (KG nodes) in one call.

    Speaks the unified Context object: queries drive Channel A (multi-view
    cosine), keywords drive Channel C (caller-provided exact terms -- no
    auto-extraction), entities seed Channel B graph BFS when provided.
    Cross-collection RRF then competes record + entity hits head-to-head.

    Returns the SAME lean shape declare_intent / declare_operation use --
    each hit is {id, text, hybrid_score} -- so agents see one uniform
    retrieval surface across injection-time and search-time.

    Args:
        context: MANDATORY Context = {queries, keywords, entities?}.
        limit: Max results across records+entities (default 10; adaptive-K may trim).
        kind: Optional entity-only kind filter (excludes record results).
        sort_by: 'hybrid' (default) -- RRF + hybrid_score tiebreaker. 'similarity'.
        agent: Agent name for affinity scoring.
        time_window: optional {start, end} date range (YYYY-MM-DD).
            SOFT DECAY: items inside the window get a scoring boost; items
            outside still appear but rank lower. NOT a hard filter -- nothing
            is excluded. Use for temporal scoping ("what happened this week")
            without losing globally-important items that fall outside.
    """
    from mempalace.mcp_server import (
        _STATE,
        _get_collection,
        _get_entity_collection,
        _record_context_emit,
        _telemetry_append_jsonl,
        context_lookup_or_create,
        intent,
    )
    from .scoring import rrf_merge, validate_context as _validate_context
    from .knowledge_graph import normalize_entity_name

    # ── Reject the legacy `queries` path (Context mandatory) ──
    if context is None and queries is not None:
        return {
            "success": False,
            "error": (
                "`queries` is gone. Pass `context` instead -- a dict "
                "with mandatory queries, keywords, and optional entities. Example:\n"
                '  context={"queries": ["auth rate limiting", "brute force hardening"], '
                '"keywords": ["auth", "rate-limit", "brute-force"]}'
            ),
        }

    # ── Validate Context (mandatory; read-side relaxed) ──
    # kg_search is read-side: entities + summary are optional
    # (entities_min=0). The shared _CONTEXT_SCHEMA_READ used by the
    # MCP schema also leaves them optional so caller and runtime agree.
    clean_context, ctx_err = _validate_context(
        context,
        entities_min=0,
    )
    if ctx_err:
        return ctx_err
    query_views = clean_context["queries"]
    context_keywords = clean_context["keywords"]
    context_entities = clean_context.get("entities", [])

    sanitized_views = [sanitize_query(v)["clean_query"] for v in query_views]
    sanitized_views = [v for v in sanitized_views if v]
    if not sanitized_views:
        return {"success": False, "error": "All queries were empty after sanitization."}

    # ── Context as first-class entity (P1) ──
    # kg_search is an emit site. Mint or reuse a kind="context" entity
    # for the search cue and update the active_context_id -- this is the
    # most-recent emit, so subsequent writes in the same tool call are
    # correctly provenanced to what actually triggered them.
    # Precedence: declare_intent sets it on intent creation, then
    # declare_operation / kg_search each overwrite on their own invocation.
    _search_context_id = ""
    _search_context_reused = False
    _search_context_max_sim = 0.0
    try:
        _sc_id, _sc_reused, _sc_ms = context_lookup_or_create(
            queries=sanitized_views,
            keywords=context_keywords,
            entities=context_entities,
            agent=agent or "",
            summary=clean_context.get("summary"),
        )
        _search_context_id = _sc_id or ""
        _search_context_reused = bool(_sc_reused)
        _search_context_max_sim = float(_sc_ms or 0.0)
    except Exception:
        _search_context_id = ""
    if _STATE.active_intent is not None and _search_context_id:
        _STATE.active_intent["active_context_id"] = _search_context_id
        _record_context_emit(
            _search_context_id,
            reused=_search_context_reused,
            scope="search",
            queries=sanitized_views,
            keywords=context_keywords,
            entities=context_entities,
        )

    # ── Source scoping: kind → entities only; otherwise search both ──
    search_memories = not bool(kind)
    search_entities = True

    # ── Walk the rated-context neighbourhood ONCE ──
    # Both Channel D (retrieval recall) and hybrid_score's W_REL term
    # (per-memory signed relevance) consume this walk. The walker
    # returns both aggregates; we pass the dict down to multi_channel_
    # search for Channel D and pull rated_scores out for hybrid_score
    # below.
    _rated_walk = (
        walk_rated_neighbourhood(_search_context_id, _STATE.kg)
        if _search_context_id
        else {"rated_scores": {}, "channel_D_list": [], "contributing_contexts": {}}
    )
    # Step 3 of similar_context_id flag (record_ga_agent_similar_context_id_
    # flag_design_2026_04_30): renderer parity with declare_intent. Surface
    # which similar_to neighbours (NOT the active context) contributed weight
    # to each retrieved item, plus the full Context object per neighbour.
    _contributing_contexts = _rated_walk.get("contributing_contexts") or {}

    try:
        # ── Run pipeline over selected collections ──
        all_lists = {}
        combined_meta = {}

        # Classify a seen_meta entry by its metadata.kind. Post-M1 records
        # AND entities live in the same mempalace_records collection, so a
        # query against "the memory collection" returns BOTH kinds mixed in
        # one result list; we can't infer source from which pipe raised the
        # hit. Derive it from metadata.kind instead.
        _ENTITY_KINDS = {"entity", "class", "predicate", "literal"}

        def _classify_source(info: dict) -> str:
            meta = info.get("meta") or {}
            kind_value = meta.get("kind", "")
            if kind_value in _ENTITY_KINDS:
                return "entity"
            # record, diary, or unlabeled prose → memory
            return "memory"

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
                active_context_id=_search_context_id,
                rated_walk=_rated_walk,
            )
            for name, lst in memory_pipe["ranked_lists"].items():
                all_lists[f"memory_{name}"] = lst
            for mid, info in memory_pipe["seen_meta"].items():
                combined_meta[mid] = {**info, "source": _classify_source(info)}

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
                active_context_id=_search_context_id,
                rated_walk=_rated_walk,
            )
            for name, lst in entity_pipe["ranked_lists"].items():
                all_lists[f"entity_{name}"] = lst
            for mid, info in entity_pipe["seen_meta"].items():
                # Post-M1 both pipes see the same collection; classify by
                # metadata.kind so memories keep source="memory" even if the
                # entity pipe also surfaced them.
                src = _classify_source(info)
                if mid in combined_meta:
                    # Prefer the existing classification when they agree; a
                    # disagreement means metadata.kind is missing or stale,
                    # so fall through to the freshly-computed value.
                    if combined_meta[mid].get("source") == src:
                        continue
                combined_meta[mid] = {**info, "source": src}

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
            return {"results": []}

        # ── Canonical two-stage pipeline ──
        # scoring.two_stage_retrieve runs RRF → hybrid_score rerank →
        # adaptive_k. Same helper declare_intent / declare_operation use,
        # so every tool returns hits on the same scale with the same
        # semantics. sort_by='similarity' bypasses the reranker by
        # routing through a cosine-only path below.
        from .scoring import two_stage_retrieve as _two_stage

        if sort_by == "hybrid":
            feedback_scores = _rated_walk.get("rated_scores", {}) or {}
            reranked, _rrf_scores, _cm = _two_stage(
                all_lists,
                combined_meta,
                agent=agent or "",
                session_id=_STATE.session_id or "",
                intent_type_id="",
                context_feedback=feedback_scores,
                rerank_top_m=max(limit * 3, 50),
                max_k=limit,
                min_k=1,
                time_window=time_window,
            )
        else:
            # similarity-sort: skip the rerank, order by raw cosine only.
            rrf_scores, _cm, _attr = rrf_merge(all_lists)
            reranked = []
            for mid, _rrf in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
                info = combined_meta.get(mid)
                if not info:
                    continue
                reranked.append(
                    {
                        "id": mid,
                        "hybrid_score": float(info.get("similarity", 0.0) or 0.0),
                        "rrf_score": float(_rrf),
                        "text": info.get("doc") or "",
                        "channel": "cosine",
                        "meta": info.get("meta") or {},
                        "similarity": float(info.get("similarity", 0.0) or 0.0),
                        "source": info.get("source", ""),
                    }
                )
            reranked = reranked[:limit]

        # Build a kg_search-shape `top` list so the surfaced-edge writer
        # and telemetry hooks below still work unchanged.
        top = []
        for entry in reranked:
            meta = entry.get("meta") or {}
            source = entry.get("source") or "memory"
            doc = entry.get("text") or ""
            proj = {
                "id": entry["id"],
                "source": source,
                "similarity": entry.get("similarity", 0.0),
                "score": round(float(entry["hybrid_score"]), 4),
                "hybrid_score": round(float(entry["hybrid_score"]), 6),
            }
            # Step 3 of similar_context_id flag (default-on): surface
            # contributing similar-context neighbours per retrieved item.
            _neighbour_cids = _contributing_contexts.get(entry["id"]) or []
            if _neighbour_cids:
                proj["similar_context_ids"] = list(_neighbour_cids)
            if source == "memory":
                summary_val = (meta.get("summary") or "").strip()
                proj["text"] = intent._shorten_preview(summary_val or doc)
            elif source == "triple":
                proj["statement"] = doc[:300]
                proj["subject"] = meta.get("subject", "")
                proj["predicate"] = meta.get("predicate", "")
                proj["object"] = meta.get("object", "")
            else:
                # entity
                proj["name"] = meta.get("name", entry["id"])
                proj["content"] = doc
                proj["kind"] = meta.get("kind", "entity")
                proj["text"] = intent._shorten_preview(doc or meta.get("name", entry["id"]))
            top.append(proj)

        # ── Attach current edges for entity results only ──
        for entry in top:
            if entry["source"] == "entity":
                edges = _STATE.kg.query_entity(entry["id"], direction="both")
                current_edges = [e for e in edges if e.get("current", True)]
                entry["edges"] = current_edges
                entry["edge_count"] = len(current_edges)

        # ── Track accessed items for mandatory feedback enforcement ──
        # Bug 3 Piece B 2026-04-28: skip the add when active_intent is in
        # pending_feedback state (mid-finalize). Same rationale as
        # tool_kg_query above -- reads are explicitly allowed by the
        # finalize-phase lockdown gate (so the agent can look up content
        # to rate it) but they must not grow the coverage requirement
        # while the intent is closing.
        if (
            _STATE.active_intent
            and isinstance(_STATE.active_intent.get("accessed_memory_ids"), set)
            and not _STATE.active_intent.get("pending_feedback")
        ):
            for entry in top:
                _STATE.active_intent["accessed_memory_ids"].add(entry["id"])

        # ── P2: write `surfaced` edges from active context to each top result ──
        # These are the consumer of finalize_intent's coverage check and
        # feed Channel D on subsequent intents. Each edge carries
        # {ts, rank, channel, sim_score} as structured props so the
        # downstream pipeline has everything it needs without a rejoin.
        if _search_context_id and top:
            now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
            for rank, entry in enumerate(top, start=1):
                _raw_channels = entry.get("channels")
                _chans = sorted(_raw_channels) if isinstance(_raw_channels, (list, set)) else []
                # `channels` stores the full comma-joined set so
                # downstream bucketing attributes memories to every
                # channel that surfaced them.
                props = {
                    "ts": now_iso,
                    "rank": rank,
                    "channels": ",".join(_chans),
                    "sim_score": float(entry.get("similarity", 0.0) or 0.0),
                }
                try:
                    _STATE.kg.add_triple(
                        _search_context_id,
                        "surfaced",
                        entry["id"],
                        properties=props,
                    )
                except Exception:
                    pass  # non-fatal -- the search result still returns

        # ── P3 telemetry: JSONL trace for mempalace-eval ──
        try:
            per_channel_hits: dict = {}
            for entry in top:
                chs = entry.get("channels") or []
                if not isinstance(chs, list):
                    continue
                for ch in chs:
                    per_channel_hits[ch] = per_channel_hits.get(ch, 0) + 1
            _telemetry_append_jsonl(
                "search_log.jsonl",
                {
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "active_context_id": _search_context_id,
                    "reused": _search_context_reused,
                    "max_sim": round(_search_context_max_sim, 4),
                    "per_channel_hits": per_channel_hits,
                    "top_k": len(top),
                    "queries": sanitized_views[:3],
                    "agent": agent or "",
                },
            )
        except Exception:
            pass

        # ── Output projection ──
        # Every hit gets the SAME lean shape declare_intent /
        # declare_operation return: {id, text, source, hybrid_score}.
        # `source` is load-bearing for kg_search callers that mix memory
        # / entity / triple hits -- the three carry different downstream
        # affordances (entity hits unlock kg_query for edges; memory hits
        # are ready to read). Fetch the full entity / triple / edges via
        # mempalace_kg_query when you need the structured detail.
        projected = []
        for entry in top:
            lean = {
                "id": entry["id"],
                "text": entry.get("text", ""),
                "source": entry.get("source") or "memory",
            }
            if intent.DEBUG_RETURN_SCORES and "hybrid_score" in entry:
                lean["hybrid_score"] = entry["hybrid_score"]
            projected.append(lean)

        # ── Injection-stage gate ──
        # Same wiring pattern as declare_intent / declare_operation.
        # Parent frame = the active intent (if any) that this search
        # is nested inside; standalone searches have no parent. Fail-
        # open on any error -- search must still work even if the gate
        # is broken.
        _kg_gate_status = None
        try:
            from .injection_gate import apply_gate as _apply_gate

            _kg_combined = {
                entry["id"]: {
                    "source": entry.get("source") or "memory",
                    "doc": entry.get("text") or "",
                    "similarity": float(entry.get("hybrid_score") or 0.0),
                }
                for entry in top
                if entry.get("id")
            }
            _kg_parent = None
            try:
                ai = _STATE.active_intent or {}
                if ai:
                    _kg_parent = {
                        "intent_type": ai.get("intent_type"),
                        "subject": ", ".join((ai.get("slots", {}) or {}).get("subject", []) or []),
                        "query": (ai.get("description_views") or [""])[0],
                    }
            except Exception:
                _kg_parent = None

            _gated, _kg_gate_status = _apply_gate(
                memories=projected,
                combined_meta=_kg_combined,
                primary_context={
                    "source": "kg_search",
                    "queries": list(sanitized_views),
                    "keywords": list(context_keywords or []),
                    "entities": list((context.get("entities") if context else None) or []),
                },
                context_id=_search_context_id or "",
                kg=_STATE.kg,
                agent=agent,
                parent_intent=_kg_parent,
            )
            projected = _gated
        except Exception:
            pass

        # Step 3 of similar_context_id flag (default-on): top-level
        # similar_contexts list with full Context objects (queries/
        # keywords/summary) for every unique contributing neighbour
        # cid across all surviving results. Mirrors the declare_intent
        # response shape; included only when non-empty (token-diet).
        _similar_contexts_block: list = []
        _seen_neighbour_cids: set = set()
        for _r in projected:
            for _cid in _r.get("similar_context_ids") or []:
                if _cid in _seen_neighbour_cids:
                    continue
                _seen_neighbour_cids.add(_cid)
                try:
                    _ent = _STATE.kg.get_entity(_cid)
                except Exception:
                    _ent = None
                if not _ent:
                    continue
                _props = _ent.get("properties") or {}
                if isinstance(_props, str):
                    try:
                        import json as _json

                        _props = _json.loads(_props)
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

        response = {"results": projected}
        if _similar_contexts_block:
            response["similar_contexts"] = _similar_contexts_block
        if _kg_gate_status is not None:
            response["gate_status"] = _kg_gate_status
        if intent.DEBUG_RETURN_CONTEXT:
            # Debug overlay mirroring declare_intent / declare_operation.
            # Token-diet 2026-04-23: queries are echoed ONLY on reuse;
            # a fresh-mint context carries the caller's own queries so
            # there's nothing new to show.
            # Token-diet 2026-04-24: non-reused collapses to "new";
            # reused returns {id, queries}. Shape-as-signal: string
            # "new" = fresh mint; object = reused.
            if _search_context_reused:
                response["context"] = {
                    "id": _search_context_id,
                    "queries": list(sanitized_views),
                }
            else:
                response["context"] = "new"
        return response
    except Exception as e:
        return {"success": False, "error": f"kg_search failed: {e}"}


def tool_kg_list_declared():
    """List all entities declared in this session."""
    from mempalace.mcp_server import (
        _STATE,
    )

    results = []
    for eid in sorted(_STATE.declared_entities):
        entity = _STATE.kg.get_entity(eid)
        if entity:
            results.append(
                {
                    "entity_id": eid,
                    "name": entity["name"],
                    "content": entity["content"],
                    "importance": entity["importance"],
                    "last_touched": entity["last_touched"],
                    "edge_count": _STATE.kg.entity_edge_count(eid),
                }
            )
    return {
        "declared_count": len(results),
        "entities": results,
    }


def tool_diary_read(
    agent_name: str,
    limit: int = None,
    last_n: int = None,
):
    """
    Read an agent's recent diary entries. Returns the last N entries
    in chronological order -- the agent's personal journal.

    `limit` (canonical) and `last_n` (deprecated alias) both control
    how many entries are returned. If both are passed, `limit` wins.
    Default 10 when neither is set.
    """
    from mempalace.mcp_server import (
        _get_collection,
        _no_palace,
    )

    # Resolve limit: explicit `limit` overrides `last_n`; default 10.
    if isinstance(limit, int) and limit > 0:
        last_n = limit
    elif isinstance(last_n, int) and last_n > 0:
        pass  # last_n already set
    else:
        last_n = 10

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
            return {"entries": [], "message": "No diary entries yet."}

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
            "entries": entries,
            "total": len(results["ids"]),
            "showing": len(entries),
        }
    except Exception as e:
        return {"error": str(e)}


__all__ = [
    "tool_diary_read",
    "tool_kg_list_declared",
    "tool_kg_query",
    "tool_kg_search",
    "tool_kg_stats",
    "tool_kg_timeline",
]
