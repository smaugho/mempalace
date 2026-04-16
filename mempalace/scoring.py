"""mempalace/scoring.py — Unified scoring for all retrieval and ranking."""

from datetime import datetime

# Power-law decay constants
STABILITY_DAYS = {5: 365.0, 4: 90.0, 3: 30.0, 2: 7.0, 1: 1.0}
DECAY_WEIGHT = 0.2
TIER_MULTIPLIER = 10.0  # For L1 mode: ensures importance tiers never cross

# Boost constants
AGENT_BOOST_SEARCH = 0.15
AGENT_BOOST_L1 = 0.5
RELEVANCE_BOOST = 0.1  # boost for found_useful, penalty for found_irrelevant

# Learned weights override — populated by set_learned_weights()
_learned_weights: dict = {}  # empty = use defaults

# Default search weights (used as base for learning)
DEFAULT_SEARCH_WEIGHTS = {
    "sim": 0.40,
    "rel": 0.20,
    "imp": 0.18,
    "decay": 0.12,
    "agent": 0.10,
}


def set_learned_weights(weights: dict):
    """Set learned weights for hybrid_score search mode.

    Call with kg.compute_learned_weights(DEFAULT_SEARCH_WEIGHTS) to apply
    feedback-learned adjustments. Call with {} to reset to defaults.
    """
    global _learned_weights
    _learned_weights = dict(weights) if weights else {}


def compute_age_days(date_iso: str, last_relevant_iso: str = None) -> float:
    """Compute age in days from the most recent time anchor."""
    time_anchor = last_relevant_iso or date_iso
    dt = _parse_iso_datetime_safe(time_anchor)
    if dt is None:
        return 365.0
    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    return max(0.0, (now - dt).total_seconds()) / 86400.0


def power_law_decay(age_days: float, importance: float) -> float:
    """FSRS-inspired power-law decay: R = (1 + age/S)^(-0.5).

    Returns a penalty in range [-DECAY_WEIGHT, 0].
    Fresh = 0 penalty, old = up to -DECAY_WEIGHT.
    """
    stability = STABILITY_DAYS.get(int(importance), 30.0)
    retrievability = (1.0 + age_days / stability) ** -0.5
    return -DECAY_WEIGHT * (1.0 - retrievability)


def hybrid_score(
    similarity: float = 0.0,
    importance: float = 3.0,
    date_iso: str = "",
    agent_match: bool = False,
    last_relevant_iso: str = None,
    relevance_feedback: float = 0.0,  # [-1.0, +1.0] from found_useful/found_irrelevant with confidence
    mode: str = "search",  # "search" or "l1"
) -> float:
    """Unified scoring function for all mempalace retrieval.

    Modes:
        "search" — similarity-primary, with relevance feedback as strong secondary.
                   Used by mempalace_kg_search and declare_intent context.
        "l1"    — importance-primary. Hard tier separation (imp 5 ALWAYS
                   outranks imp 4). Used by Layer1 wake-up.

    Components:
        - similarity: cosine similarity to query/intent description
        - importance: 1-5 tier
        - decay: power-law with importance-dependent stability
        - agent_match: boost for own content
        - last_relevant_at: decay reset on found_useful
        - relevance_feedback: continuous [-1.0, +1.0] from type-level feedback,
          confidence-graded on BOTH sides (P5.3).
            +1.0 = relevance-5 useful, +0.2 = relevance-1 useful
             0.0 = no feedback yet
            -0.2 = relevance-1 irrelevant, -1.0 = relevance-5 irrelevant
          Stored as confidence=relevance/5.0 on found_useful/found_irrelevant
          edges; intent.py flips the sign for irrelevant at read time.
    """
    try:
        sim = float(similarity or 0.0)
    except (TypeError, ValueError):
        sim = 0.0
    try:
        imp = float(importance or 3.0)
    except (TypeError, ValueError):
        imp = 3.0

    age_days = compute_age_days(date_iso, last_relevant_iso)
    decay = power_law_decay(age_days, imp)

    if mode == "l1":
        # L1: importance dominates via tier multiplication
        # Invariant: imp 5 ALWAYS outranks imp 4 regardless of age
        # max decay penalty at age=infinity: -DECAY_WEIGHT * 2.5 = -0.5, tier gap = 10
        agent_boost = AGENT_BOOST_L1 if agent_match else 0.0
        rel_boost = RELEVANCE_BOOST * relevance_feedback
        return imp * TIER_MULTIPLIER + decay * 2.5 + agent_boost + rel_boost
    else:
        # Search: normalized weighted combination (all components in [0,1])
        # Weights sum to 1.0 — similarity leads, feedback is strong secondary
        # These are defaults; use set_learned_weights() to override with feedback-learned values
        W_SIM = _learned_weights.get("sim", 0.40)
        W_REL = _learned_weights.get("rel", 0.20)
        W_IMP = _learned_weights.get("imp", 0.18)
        W_DECAY = _learned_weights.get("decay", 0.12)
        W_AGENT = _learned_weights.get("agent", 0.10)

        # Normalize each component to [0, 1]
        norm_sim = max(0.0, min(1.0, sim))  # already 0-1
        norm_imp = (imp - 1.0) / 4.0  # 1→0, 5→1
        norm_decay = 1.0 + (decay / DECAY_WEIGHT)  # -0.2→0, 0→1
        norm_agent = 1.0 if agent_match else 0.0  # binary
        # Continuous: -1.0→0, 0→0.5, +1.0→1. Granular from 1-5 relevance scale.
        rel_float = float(relevance_feedback)
        norm_rel = max(0.0, min(1.0, (rel_float + 1.0) / 2.0))

        return (
            W_SIM * norm_sim
            + W_IMP * norm_imp
            + W_DECAY * norm_decay
            + W_AGENT * norm_agent
            + W_REL * norm_rel
        )


def adaptive_k(scores: list, max_k: int = 20, min_k: int = 1, gap_multiplier: float = 2.0) -> int:
    """Determine optimal K using similarity gap detection.

    Finds the largest gap between consecutive scores that is significantly
    larger than the average gap (gap_multiplier times the mean). This
    naturally handles evenly spaced scores (no standout gap → return all)
    and clear clusters (one gap >> mean → cut there).

    Based on: "No Tuning, No Iteration, Just Adaptive-K" (EMNLP 2025).

    Args:
        scores: List of scores (higher = more relevant). Must be pre-sorted descending.
        max_k: Safety ceiling — never return more than this.
        min_k: Floor — always return at least this many (if available).
        gap_multiplier: A gap must be this many times the mean gap to trigger
            a cutoff. Default 2.0 = gap must be 2x average to be "significant".

    Returns:
        Optimal K (number of items to keep).
    """
    if not scores:
        return 0
    if len(scores) <= min_k:
        return len(scores)

    n = min(len(scores), max_k)
    top_scores = scores[:n]

    if len(top_scores) < 2:
        return len(top_scores)

    # Compute all gaps
    gaps = [top_scores[i - 1] - top_scores[i] for i in range(1, len(top_scores))]
    if not gaps:
        return len(top_scores)

    mean_gap = sum(gaps) / len(gaps)
    if mean_gap <= 0:
        return n  # All scores identical — return all

    # Find the largest gap that exceeds gap_multiplier * mean_gap
    best_gap = 0.0
    best_k = n  # Default: return all (no significant gap found)

    for i in range(min_k, n):
        gap = gaps[i - 1]  # gap between position i-1 and i
        if gap > best_gap and gap >= mean_gap * gap_multiplier:
            best_gap = gap
            best_k = i

    return best_k


def _parse_iso_datetime_safe(value):
    """Parse ISO-format datetime string to datetime object."""
    if not value or not isinstance(value, str):
        return None
    try:
        s = value.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        if "+" in s[10:] or s.count("-") > 2:
            return datetime.fromisoformat(s)
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


# ══════════════════════════════════════════════════════════════════════
# Shared search primitives — caller-keyword lookups (P4.6).
#
# Auto-keyword extraction (extract_keywords + STOP_WORDS) is GONE. The
# caller MUST provide context.keywords on every read AND every write.
# Keywords are stored in the entity_keywords table (kg.add_entity_keywords)
# and looked up via kg.entity_ids_for_keyword — fast, indexed, exact-match.
# ══════════════════════════════════════════════════════════════════════


def keyword_lookup(kg, keywords, *, wing=None, kind_filter=None, collection=None):
    """Look up entities by caller-provided keywords (P4.6).

    For each keyword, fetch entity_ids from the entity_keywords table, then
    pull metadata from the matching ChromaDB collection (handles both plain
    memory ids and the ::view_N suffix of the multi-vector entity store).

    Returns list of (entity_id, document, metadata, suppression_score) —
    same shape the legacy keyword_search() returned, so downstream
    _build_keyword_channel logic stays identical.
    """
    if not keywords or kg is None:
        return []
    results = []
    seen_ids = set()
    for kw in keywords:
        if not kw or not kw.strip():
            continue
        try:
            entity_ids = kg.entity_ids_for_keyword(kw)
        except Exception:
            continue
        for eid in entity_ids:
            if eid in seen_ids:
                continue
            seen_ids.add(eid)
            meta = None
            doc = ""
            if collection is not None:
                # P5.2: Look up the entity in Chroma by metadata.entity_id first
                # (covers the multi-view entity collection where id != entity_id),
                # then fall back to plain-id lookup (memories collection where
                # the drawer id IS the entity id).
                try:
                    got = collection.get(
                        where={"entity_id": eid},
                        include=["documents", "metadatas"],
                        limit=1,
                    )
                except Exception:
                    got = None
                if not (got and got.get("ids")):
                    try:
                        got = collection.get(
                            ids=[eid],
                            include=["documents", "metadatas"],
                        )
                    except Exception:
                        got = None
                if got and got.get("ids"):
                    meta = (got["metadatas"][0] if got.get("metadatas") else {}) or {}
                    doc = (got["documents"][0] if got.get("documents") else "") or ""
            if meta is None:
                # Entity exists in entity_keywords but not in this collection — skip.
                continue
            if wing and meta.get("wing") != wing:
                continue
            if kind_filter and meta.get("kind") != kind_filter:
                continue
            suppression = 1.0
            try:
                suppression = kg.get_keyword_suppression(eid)
            except Exception:
                pass
            results.append((eid, doc, meta, suppression))
    return results


def rrf_merge(ranked_lists, k=60):
    """Reciprocal Rank Fusion — merge multiple ranked lists into one.

    Args:
        ranked_lists: dict of list_name -> [(score, text, memory_id), ...]
        k: RRF constant (default 60, Cormack et al. 2009)

    Returns:
        (rrf_scores, candidate_map, channel_attribution) where:
        - rrf_scores: dict memory_id -> rrf_score
        - candidate_map: dict memory_id -> (text, channel_name)
        - channel_attribution: dict memory_id -> set of channel names
    """
    rrf_scores = {}
    candidate_map = {}
    channel_attribution = {}

    for list_name, candidates in ranked_lists.items():
        deduped = {}
        for score, text, mid in candidates:
            if mid not in deduped or score > deduped[mid][0]:
                deduped[mid] = (score, text)
        ranked = sorted(deduped.items(), key=lambda x: x[1][0], reverse=True)

        for rank, (mid, (score, text)) in enumerate(ranked):
            rrf_contribution = 1.0 / (k + rank + 1)
            rrf_scores[mid] = rrf_scores.get(mid, 0.0) + rrf_contribution
            if mid not in candidate_map:
                candidate_map[mid] = (text, list_name)
            if mid not in channel_attribution:
                channel_attribution[mid] = set()
            channel_attribution[mid].add(list_name.split("_")[0])

    return rrf_scores, candidate_map, channel_attribution


# ══════════════════════════════════════════════════════════════════════
# Unified multi-channel search pipeline — the ONE implementation used
# everywhere we do similarity + keyword + graph retrieval.
# Callers (tool_kg_search unified + declare_intent context) just pass
# a collection + mandatory multi-view queries + filters, then apply their
# own hybrid rerank + adaptive-K on the returned seen_meta.
# ══════════════════════════════════════════════════════════════════════


# ── Unified Context object (P4.1) ──────────────────────────────────────
#
# Every read AND write across the palace API speaks Context. It is the
# universal shape for "what is the agent thinking" — used as both the
# multi-view retrieval seed and as the creation/traversal fingerprint
# stored on entities and edges so future feedback applies by similarity.
#
#   Context = {
#     queries:  list[str]   2-5 perspectives             (mandatory)
#     keywords: list[str]   2-5 caller-provided terms    (mandatory)
#     entities: list[str]   0+ related/seed entity ids   (optional)
#   }
#
# Auto-keyword extraction is gone — the caller knows what terms matter
# for the thing they're filing or searching, and we refuse to guess.


def _validate_string_list(value, field_name, min_n, max_n, example):
    """Shared list-of-strings validator. Returns (cleaned_list, error_dict)."""
    if isinstance(value, str):
        return None, {
            "success": False,
            "error": (
                f"{field_name} must be a LIST of strings, not a single string. "
                f"Multi-view needs at least {min_n} distinct items. "
                f"Example: {example}"
            ),
        }
    if value is None:
        return None, {
            "success": False,
            "error": f"{field_name} is required (LIST of {min_n}-{max_n} strings).",
        }
    if not isinstance(value, list):
        return None, {
            "success": False,
            "error": f"{field_name} must be a list of strings, got {type(value).__name__}",
        }
    cleaned = [s for s in value if isinstance(s, str) and s.strip()]
    if len(cleaned) < min_n:
        return None, {
            "success": False,
            "error": (
                f"{field_name} must contain at least {min_n} non-empty entries "
                f"(got {len(cleaned)}). Pass {min_n}-{max_n} distinct items. "
                f"Example: {example}"
            ),
        }
    if len(cleaned) > max_n:
        cleaned = cleaned[:max_n]
    return cleaned, None


def validate_context(context, *, queries_min=2, queries_max=5, keywords_min=2, keywords_max=5):
    """Shared validation for the unified Context object (P4.1).

    Context = {
      queries:  list[str]  (mandatory, 2-5)
      keywords: list[str]  (mandatory, 2-5; caller-provided, no auto-extract)
      entities: list[str]  (optional, 0+ related/seed entity ids)
    }

    Returns (clean_context_dict, error_dict_or_None). If error is truthy,
    caller should return it directly. Used by every read AND every write tool.
    """
    if context is None or not isinstance(context, dict):
        return None, {
            "success": False,
            "error": (
                "context is required (dict with 'queries', 'keywords', and "
                "optional 'entities'). Example: "
                '{"queries": ["auth rate limiting", "brute force hardening"], '
                '"keywords": ["auth", "rate", "limiting"], "entities": ["LoginService"]}'
            ),
        }

    queries, err = _validate_string_list(
        context.get("queries"),
        "context.queries",
        queries_min,
        queries_max,
        '["auth rate limiting", "brute force hardening", "login endpoint"]',
    )
    if err:
        return None, err

    keywords, err = _validate_string_list(
        context.get("keywords"),
        "context.keywords",
        keywords_min,
        keywords_max,
        '["auth", "rate-limit", "brute-force", "login"]',
    )
    if err:
        return None, err

    raw_entities = context.get("entities") or []
    if isinstance(raw_entities, str):
        return None, {
            "success": False,
            "error": (
                "context.entities must be a list (or omitted), not a string. "
                'Example: ["LoginService", "AuthRateLimiter"]'
            ),
        }
    if not isinstance(raw_entities, list):
        raw_entities = []
    entities = [e for e in raw_entities if isinstance(e, str) and e.strip()]

    return (
        {"queries": queries, "keywords": keywords, "entities": entities},
        None,
    )


# embed_context removed (P5.4): had zero callers. Context view vectors are
# embedded inside ChromaDB by store_feedback_context / _sync_entity_views_to_chromadb
# — there's no external embedding path that needs a caller-side helper.

# validate_query_views removed (P4.11): legacy shim with no remaining callers.
# Use validate_context() — same loud single-string-rejection contract, but
# expects the full Context shape (queries + keywords + entities).


def lookup_type_feedback(active_intent, kg):
    """Load found_useful / found_irrelevant sets from the active intent type.

    Returns (useful_ids, irrelevant_ids). Both empty if no intent or on error.
    Used for relevance_feedback input to hybrid_score.
    """
    useful_ids = set()
    irrelevant_ids = set()
    try:
        if not (active_intent and kg):
            return useful_ids, irrelevant_ids
        intent_type_id = active_intent.get("intent_type", "")
        if not intent_type_id:
            return useful_ids, irrelevant_ids
        type_edges = kg.query_entity(intent_type_id, direction="outgoing")
        for edge in type_edges:
            if not edge.get("current", True):
                continue
            if edge["predicate"] == "found_useful":
                useful_ids.add(edge["object"])
            elif edge["predicate"] == "found_irrelevant":
                irrelevant_ids.add(edge["object"])
    except Exception:
        pass
    return useful_ids, irrelevant_ids


def _build_cosine_channel(collection, views, fetch_limit_per_view, where_filter, seen_meta):
    """CHANNEL A: multi-view cosine. Mutates seen_meta, returns ranked lists."""
    ranked_lists = {}
    try:
        count = collection.count()
    except Exception:
        return ranked_lists
    if count == 0:
        return ranked_lists
    for vi, view in enumerate(views):
        kwargs = {
            "query_texts": [view],
            "n_results": min(fetch_limit_per_view, count),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            kwargs["where"] = where_filter
        try:
            results = collection.query(**kwargs)
        except Exception:
            continue
        if not (results.get("ids") and results["ids"][0]):
            continue
        candidates = []
        for i, mid in enumerate(results["ids"][0]):
            dist = results["distances"][0][i]
            similarity = round(1 - dist, 3)
            meta = results["metadatas"][0][i] or {}
            doc = results["documents"][0][i] or ""
            candidates.append((similarity, doc, mid))
            prev = seen_meta.get(mid)
            if prev is None or prev.get("similarity", 0.0) < similarity:
                seen_meta[mid] = {"meta": meta, "doc": doc, "similarity": similarity}
        if candidates:
            ranked_lists[f"cosine_{vi}"] = candidates
    return ranked_lists


def _build_keyword_channel(
    collection,
    keywords,
    kg,
    wing,
    kind_filter,
    seen_meta,
    suppression_floor=0.125,
    base_weight=0.4,
):
    """CHANNEL C: caller-provided keyword lookup (P4.6).

    Each keyword resolves to entity_ids via the entity_keywords index, then
    we fetch the matching ChromaDB record to pull document + metadata.
    No more $contains scanning, no more auto-extraction. Mutates seen_meta.
    """
    if not keywords or kg is None:
        return []
    kw_ranked = []
    try:
        kw_hits = keyword_lookup(
            kg, keywords, wing=wing, kind_filter=kind_filter, collection=collection
        )
    except Exception:
        return []
    for mid, doc, meta, suppression in kw_hits:
        if suppression < suppression_floor:
            continue
        score = base_weight * suppression
        kw_ranked.append((score, (doc or "")[:300], mid))
        if mid not in seen_meta:
            seen_meta[mid] = {"meta": meta, "doc": doc, "similarity": 0.0}
    return kw_ranked


def _build_graph_channel(collection, kg, seed_ids, kind_filter, seen_meta, top_per_seed_limit=None):
    """CHANNEL B: 1-hop graph neighbors of seed entities. Mutates seen_meta.

    Returns a ranked list where each neighbor's score scales with the strongest
    seed's cosine similarity. If seed_ids is empty or kg is None, returns [].
    """
    graph_ranked = []
    if not seed_ids or kg is None:
        return graph_ranked
    for seed_id in seed_ids:
        try:
            edges = kg.query_entity(seed_id, direction="both")
        except Exception:
            continue
        seed_sim = seen_meta.get(seed_id, {}).get("similarity", 0.0)
        count = 0
        for e in edges:
            if not e.get("current", True):
                continue
            subj = e.get("subject") or e.get("from") or ""
            obj = e.get("object") or e.get("to") or ""
            neighbor = obj if subj == seed_id else subj
            if not neighbor or neighbor == seed_id:
                continue
            if neighbor not in seen_meta:
                if collection is None:
                    continue
                try:
                    got = collection.get(ids=[neighbor], include=["documents", "metadatas"])
                except Exception:
                    continue
                if not (got and got.get("ids")):
                    continue
                nmeta = (got["metadatas"][0] if got.get("metadatas") else {}) or {}
                ndoc = (got["documents"][0] if got.get("documents") else "") or ""
                if kind_filter and nmeta.get("kind") != kind_filter:
                    continue
                seen_meta[neighbor] = {"meta": nmeta, "doc": ndoc, "similarity": 0.0}
            score = max(0.2, seed_sim * 0.8)
            graph_ranked.append((score, seen_meta[neighbor]["doc"], neighbor))
            count += 1
            if top_per_seed_limit and count >= top_per_seed_limit:
                break
    return graph_ranked


def multi_channel_search(
    collection,
    views,
    *,
    keywords=None,  # P4.6 — caller-provided keyword list for Channel C
    kg=None,
    wing=None,
    kind=None,
    fetch_limit_per_view=50,
    include_graph=False,
    seed_ids=None,
    graph_seed_topk_per_view=3,
):
    """Unified 3-channel search pipeline. The ONE implementation used by
    every multi-view search tool (mempalace_kg_search + declare_intent).

    Channels:
        A (cosine): one ranked list per view, dense vector similarity.
        B (graph) : 1-hop neighbors of explicit seeds or top cosine hits
                    (only if include_graph=True and kg is provided).
        C (keyword): caller-provided keywords resolved via the
                     entity_keywords table (P4.6 — no auto-extraction).

    Merged via Reciprocal Rank Fusion. Caller applies hybrid rerank + adaptive-K
    using seen_meta.

    Args:
        collection: ChromaDB collection (memories, entities, …).
        views: already-validated + sanitized list of perspective strings.
        keywords: caller-provided keyword list (Context.keywords). Required for
                  Channel C — when None or empty, Channel C is silently skipped.
        kg: KnowledgeGraph — required for keyword channel + graph channel.
        wing: optional wing filter for memory collections.
        kind: optional kind filter for entity collection.
        fetch_limit_per_view: cosine n_results per view.
        include_graph: if True, run Channel B.
        seed_ids: explicit graph seeds. If None and include_graph=True, derive
                  from top-K cosine hits per view.
        graph_seed_topk_per_view: how many top cosine hits per view become seeds.

    Returns dict with: "rrf_scores", "seen_meta", "ranked_lists", "attribution".
    """
    if not views or collection is None:
        return {"rrf_scores": {}, "seen_meta": {}, "ranked_lists": {}, "attribution": {}}

    seen_meta = {}
    ranked_lists = {}

    where_filter = None
    if wing:
        where_filter = {"wing": wing}
    elif kind:
        where_filter = {"kind": kind}

    cosine_lists = _build_cosine_channel(
        collection, views, fetch_limit_per_view, where_filter, seen_meta
    )
    ranked_lists.update(cosine_lists)

    kw_ranked = _build_keyword_channel(
        collection,
        keywords or [],
        kg=kg,
        wing=wing,
        kind_filter=kind,
        seen_meta=seen_meta,
    )
    if kw_ranked:
        ranked_lists["keyword"] = kw_ranked

    if include_graph and kg is not None:
        # ── Graph-seed derivation strategy (P5.9 doc) ──
        # This is the AUTONOMOUS-search strategy: if the caller didn't pass
        # explicit seed_ids, we derive them from the top-K cosine hits per
        # view. Rationale: in open-ended search the agent doesn't always
        # know which entities are relevant; letting cosine pick the anchors
        # is a HyDE-style generate-then-expand move (Gao et al. 2022). The
        # opposite strategy — controlled BFS from caller-supplied slots —
        # lives in intent.py declare_intent, where the intent itself
        # names the entities it's about. Two legitimate modes; the split
        # is intentional.
        effective_seeds = set(seed_ids or [])
        if not effective_seeds:
            for rname, rlist in cosine_lists.items():
                if not rname.startswith("cosine_"):
                    continue
                for _sim, _doc, sid in sorted(rlist, key=lambda x: x[0], reverse=True)[
                    :graph_seed_topk_per_view
                ]:
                    effective_seeds.add(sid)
        graph_ranked = _build_graph_channel(
            collection, kg, effective_seeds, kind_filter=kind, seen_meta=seen_meta
        )
        if graph_ranked:
            ranked_lists["graph"] = graph_ranked

    rrf_scores, _candidate_map, attribution = rrf_merge(ranked_lists)
    return {
        "rrf_scores": rrf_scores,
        "seen_meta": seen_meta,
        "ranked_lists": ranked_lists,
        "attribution": attribution,
    }
