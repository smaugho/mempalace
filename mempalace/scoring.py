"""mempalace/scoring.py — Unified scoring for all retrieval and ranking.

Primary-source references
─────────────────────────
Retrieval fusion:
  Cormack, Clarke & Büttcher. "Reciprocal rank fusion outperforms Condorcet
    and individual rank learning methods." SIGIR 2009.
    → https://dl.acm.org/doi/10.1145/1571941.1572114
    Our `rrf_merge` uses the canonical k=60. RRF is rank-only and scale-free,
    which is why it happily merges cosine scores, keyword-suppression scores
    and graph-distance scores in the same pot.

  Chen, Fisch, Weston & Bordes. "Reading Wikipedia to Answer Open-Domain
    Questions." ACL 2017 (DrQA). → https://arxiv.org/abs/1704.00051
  Izacard & Grave. "Leveraging Passage Retrieval with Generative Models."
    EACL 2021. → https://arxiv.org/abs/2007.01282
  Thakur et al. "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation
    of IR Models." NeurIPS 2021. → https://arxiv.org/abs/2104.08663
    These three lines establish that hybrid (keyword + dense) retrieval
    beats either alone across tasks. Our 3-channel design (cosine + keyword
    + graph) is the same insight extended with graph traversal.

Multi-view queries:
  Gao, Ma, Lin & Callan. "Precise Zero-Shot Dense Retrieval without
    Relevance Labels" (HyDE). ACL 2023. → https://arxiv.org/abs/2212.10496
    The intuition that multiple "views" per query outperform single-
    phrasing retrieval. The Context.queries list is the caller-authored
    equivalent — the agent supplies its own perspectives rather than
    asking an LLM to hallucinate them.

Late-interaction / MaxSim:
  Khattab & Zaharia. "ColBERT: Efficient and Effective Passage Search via
    Contextualized Late Interaction over BERT." SIGIR 2020.
    → https://arxiv.org/abs/2004.12832
    MaxSim(Q, D) = Σ_i max_j cos(q_i, d_j). We use it in
    `mcp_server.maxsim_context_match` when applying type-level feedback:
    similar contexts inherit useful/irrelevant signal.

Graph-augmented retrieval:
  Rasmussen et al. (Zep AI). "Zep: A Temporal Knowledge Graph Architecture
    for Agent Memory." 2025. → https://arxiv.org/abs/2501.13956
    Validates that temporal KG + hybrid search substantially beats vector-
    only retrieval on agent-memory benchmarks. Our Channel B (BFS over
    declared edges with edge-usefulness gating) is the mempalace adaptation.

Temporal decay:
  Wixted & Carpenter. "The Wickelgren power law and the Ebbinghaus savings
    function." Psychological Science 2007. → https://pubmed.ncbi.nlm.nih.gov/17576278/
    Power-law R(t) = (1 + t/S)^(-C) fits human retention better than the
    exponential R(t) = e^(-t/S) of classic spaced-repetition literature.

  Open Spaced Repetition. "Free Spaced Repetition Scheduler (FSRS)."
    → https://github.com/open-spaced-repetition/free-spaced-repetition-scheduler
    FSRS fits a 19-parameter power-law DSR model against real user review
    data. We use a simplified form: importance-tiered stability days
    (STABILITY_DAYS below) and fixed C = 0.5.

Adaptive top-K:
  Taguchi, Maekawa & Bhutani (Megagon Labs). "Efficient Context Selection
    for Long-Context QA: No Tuning, No Iteration, Just Adaptive-k." EMNLP 2025.
    → https://aclanthology.org/2025.emnlp-main.1017/
    Gap-detection over sorted similarity scores: if a gap exceeds k × mean
    gap, cut there. We use k = 2.0 as the paper's default.
"""

from datetime import datetime

# ═════════════════════════════════════════════════════════════════════
# CONSTANTS — magic numbers used by hybrid_score, the 3 channels, and
# the learned-weights loop. Each one has: (a) what it controls,
# (b) rationale / source, (c) safe tuning range.
# ═════════════════════════════════════════════════════════════════════

# ── Power-law forgetting (Wixted & Carpenter 2007, FSRS-simplified) ──

# Importance-tiered stability S in R(t) = (1 + t/S)^(-0.5). Picking S by
# importance tier gives "critical facts half-life = 1yr, junk half-life = 1d"
# semantics. These numbers are hand-tuned for a human-timescale agent memory
# (seconds to months); a real FSRS fit on review data would replace them.
# Safe to tune: keep the monotonicity (5>4>3>2>1); factor-of-2 shifts are fine.
STABILITY_DAYS = {5: 365.0, 4: 90.0, 3: 30.0, 2: 7.0, 1: 1.0}

# Max decay penalty magnitude. Caps how much decay can pull a fresh hit
# under a stale one — 0.2 keeps decay secondary to similarity (weighted ~0.4)
# and importance (~0.18). Higher values push old results further down;
# lower values make mempalace more forgiving of stale memories.
# Safe range: 0.1–0.3.
DECAY_WEIGHT = 0.2

# Tier multiplier for L1 (wake-up) mode — makes importance 5 mathematically
# always outrank importance 4 regardless of age. 10.0 leaves plenty of
# dynamic range inside each tier for decay + agent + feedback to matter.
# Safe range: 5–20.
TIER_MULTIPLIER = 10.0

# ── Provenance affinity ──
# Additive boosts applied to hybrid_score when the candidate's provenance
# metadata matches the current session/intent/agent. These are ADDITIVE
# (not weighted components) so they don't disrupt the existing weight
# balance. Think of them as tiebreakers that prefer "recent, same-context"
# items when relevance is close.

# Agent affinity: candidate's `added_by` matches current agent.
# 0.15 is "meaningful but not dominant" in search mode.
# Safe range: 0.05–0.25.
AGENT_BOOST_SEARCH = 0.15

# L1 wake-up uses a much larger agent boost because L1 is "your own story"
# dominated by priorities and decisions authored by you.
AGENT_BOOST_L1 = 0.5

# Session affinity: candidate was created in the SAME MCP session.
# Smaller than agent because session is a narrower scope — many items
# from the same agent are cross-session. Safe range: 0.03–0.15.
SESSION_BOOST_SEARCH = 0.08
SESSION_BOOST_L1 = 0.3

# Intent affinity: candidate was created during the SAME intent type
# (not necessarily the same execution, but the same kind of task).
# Helps surface items from "last time I did this type of work".
# Safe range: 0.02–0.10.
INTENT_TYPE_BOOST_SEARCH = 0.05
INTENT_TYPE_BOOST_L1 = 0.2

# Generic per-doc relevance boost when feedback is present but un-graded.
# Safe range: 0.05–0.15.
RELEVANCE_BOOST = 0.1

# ── Default search-mode weight proportions ──

# Convex combination used by hybrid_score in "search" mode. Sum = 1.0.
# sim and rel dominate because they're the strongest task-specific signal;
# imp+decay+agent together handle the "all-else-equal" tiebreaks.
# compute_learned_weights can nudge these ±30% (see LEARN_DAMPING) based
# on feedback correlation; set_learned_weights pushes the nudged values
# into this module's _learned_weights global at runtime.
# Safe range: any non-negative proportions summing to 1.0.
DEFAULT_SEARCH_WEIGHTS = {
    "sim": 0.40,
    "rel": 0.20,
    "imp": 0.18,
    "decay": 0.12,
    "agent": 0.10,
}

# Runtime weight override — populated by set_learned_weights().
_learned_weights: dict = {}  # empty → use DEFAULT_SEARCH_WEIGHTS


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
    """Power-law decay (Wixted & Carpenter 2007; FSRS-simplified).

    Retrievability R(t) = (1 + age_days / S)^(-0.5), where S is the
    importance-tier stability from STABILITY_DAYS. The -0.5 exponent is
    the canonical "1/sqrt time" shape from the psychology-of-memory
    literature (Wickelgren/Wixted power law), and also the default
    retrievability shape in the FSRS open-source scheduler before its
    learned-C adjustment.

    Returns a penalty in [-DECAY_WEIGHT, 0]:
      fresh (age≈0)       →   0
      very old (age≫S)    →  -DECAY_WEIGHT

    References:
      Wixted & Carpenter 2007 (https://pubmed.ncbi.nlm.nih.gov/17576278/)
      FSRS (https://github.com/open-spaced-repetition/free-spaced-repetition-scheduler)
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
    session_match: bool = False,  # candidate from same MCP session
    intent_type_match: bool = False,  # candidate from same intent type
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
        - agent_match: boost for own content (P6.7b provenance)
        - session_match: boost for same-session content (P6.7b provenance)
        - intent_type_match: boost for same-intent-type content (P6.7b provenance)
        - last_relevant_at: decay reset on found_useful
        - relevance_feedback: continuous [-1.0, +1.0] from type-level feedback,
          confidence-graded on BOTH sides.
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
        session_boost = SESSION_BOOST_L1 if session_match else 0.0
        intent_boost = INTENT_TYPE_BOOST_L1 if intent_type_match else 0.0
        rel_boost = RELEVANCE_BOOST * relevance_feedback
        return (
            imp * TIER_MULTIPLIER
            + decay * 2.5
            + agent_boost
            + session_boost
            + intent_boost
            + rel_boost
        )
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

        # provenance affinity boosts are ADDITIVE (not weighted)
        # so they don't disrupt the existing weight balance.
        prov_boost = 0.0
        if session_match:
            prov_boost += SESSION_BOOST_SEARCH
        if intent_type_match:
            prov_boost += INTENT_TYPE_BOOST_SEARCH

        return (
            W_SIM * norm_sim
            + W_IMP * norm_imp
            + W_DECAY * norm_decay
            + W_AGENT * norm_agent
            + W_REL * norm_rel
            + prov_boost
        )


def adaptive_k(scores: list, max_k: int = 20, min_k: int = 1, gap_multiplier: float = 2.0) -> int:
    """Determine optimal K using similarity gap detection (Adaptive-k).

    Finds the largest gap between consecutive scores that is at least
    gap_multiplier × the mean gap. Cuts at that gap. When there's no
    standout gap (all-similar scores) the function returns max_k, which
    means "keep everything under the safety ceiling" — i.e. no early cut.

    Reference:
      Taguchi, Maekawa & Bhutani (Megagon Labs).
      "Efficient Context Selection for Long-Context QA:
       No Tuning, No Iteration, Just Adaptive-k." EMNLP 2025.
      → https://aclanthology.org/2025.emnlp-main.1017/
      → https://github.com/megagonlabs/adaptive-k-retrieval

    The paper's default gap_multiplier is 2.0 (we keep it).

    Args:
        scores: List of scores (higher = more relevant). Must be pre-sorted descending.
        max_k: Safety ceiling — never return more than this.
        min_k: Floor — always return at least this many (if available).
        gap_multiplier: A gap must be this many times the mean gap to trigger
            a cutoff. Default 2.0 per the paper.

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
# Shared search primitives — caller-keyword lookups.
#
# Auto-keyword extraction (extract_keywords + STOP_WORDS) is GONE. The
# caller MUST provide context.keywords on every read AND every write.
# Keywords are stored in the entity_keywords table (kg.add_entity_keywords)
# and looked up via kg.entity_ids_for_keyword — fast, indexed, exact-match.
# ══════════════════════════════════════════════════════════════════════


def keyword_lookup(kg, keywords, *, added_by=None, kind_filter=None, collection=None):
    """Channel C: exact-term lookup over caller-provided keywords.

    This is the keyword half of hybrid retrieval (Izacard & Grave 2020;
    BEIR, Thakur et al 2021) — complements the dense cosine channel with
    literal-term matching to catch out-of-distribution entity names and
    jargon that embeddings under-weight. We do NOT auto-extract keywords
    (that's a common source of over-matching in IR systems): every keyword
    came from the caller via Context.keywords.

    For each keyword, fetch entity_ids from the `entity_keywords` table
    (fast indexed lookup), then pull document+metadata from the matching
    ChromaDB collection. Metadata-indexed: we resolve via
    where={'entity_id': eid} with an id-match fallback for the memory
    collection where memory_id IS the entity_id.

    Returns list of (entity_id, document, metadata, suppression_score).
    The suppression_score is the decaying penalty from
    kg.get_keyword_suppression(eid) — heavily suppressed hits drop out at
    the channel C threshold upstream.
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
                # Look up the entity in Chroma by metadata.entity_id first
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
            if added_by and meta.get("added_by") != added_by:
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

    RRF_score(d) = Σ_list 1 / (k + rank_list(d))

    Rank-based and scale-free, which is why it can fuse channels that
    return wildly different score magnitudes (cosine in [0,1], keyword
    suppression in (0,1], graph distance in (0, ∞)) without per-channel
    normalisation.

    Reference:
      Cormack, Clarke & Büttcher. "Reciprocal rank fusion outperforms
      Condorcet and individual rank learning methods." SIGIR 2009.
      → https://dl.acm.org/doi/10.1145/1571941.1572114
      k = 60 is the paper's recommended default and is not worth tuning
      without empirical evidence.

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


# ── Unified Context object ──────────────────────────────────────
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
    """Shared validation for the unified Context object.

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


# embed_context removed: had zero callers. Context view vectors are
# embedded inside ChromaDB by store_feedback_context / _sync_entity_views_to_chromadb
# — there's no external embedding path that needs a caller-side helper.

# validate_query_views removed: legacy shim with no remaining callers.
# Use validate_context() — same loud single-string-rejection contract, but
# expects the full Context shape (queries + keywords + entities).


def lookup_type_feedback(active_intent, kg):
    """Per-memory continuous feedback signal from the active intent type.

    Reads found_useful / found_irrelevant edges on the active intent_type
    entity and returns a dict {memory_id: signed_score in [-1.0, 1.0]} based
    on edge confidence. found_useful contributes +confidence, found_irrelevant
    contributes -confidence; multiple edges accumulate and clamp.

    Consumed by tool_kg_search as the relevance_feedback input to
    hybrid_score (which already accepts float [-1, 1]).
    """
    scores: dict = {}
    try:
        if not (active_intent and kg):
            return scores
        intent_type_id = active_intent.get("intent_type", "")
        if not intent_type_id:
            return scores
        type_edges = kg.query_entity(intent_type_id, direction="outgoing")
        for edge in type_edges:
            if not edge.get("current", True):
                continue
            pred = edge.get("predicate")
            mid = edge.get("object")
            if not mid:
                continue
            try:
                conf = float(edge.get("confidence") or 1.0)
            except (TypeError, ValueError):
                conf = 1.0
            if pred == "found_useful":
                scores[mid] = scores.get(mid, 0.0) + conf
            elif pred == "found_irrelevant":
                scores[mid] = scores.get(mid, 0.0) - conf
        for mid in list(scores.keys()):
            if scores[mid] > 1.0:
                scores[mid] = 1.0
            elif scores[mid] < -1.0:
                scores[mid] = -1.0
    except Exception:
        pass
    return scores


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
    added_by,
    kind_filter,
    seen_meta,
    suppression_floor=0.125,
    base_weight=0.4,
    min_overlap_ratio=0.0,
):
    """CHANNEL C: caller-provided keyword lookup with overlap weighting.

    Each keyword resolves to entity_ids via the entity_keywords index, then
    we fetch the matching ChromaDB record to pull document + metadata.
    No more $contains scanning, no more auto-extraction.

    Overlap-weighted scoring. Before P6.3 the keyword channel was
    binary: every hit scored `base_weight * suppression` regardless of
    how many of the caller's keywords matched the entity. An entity
    matching 4 of 5 caller keywords scored the same as one matching 1 of 5,
    so the channel over-surfaced weak partial matches. Now we count the
    number of caller keywords each entity matches and scale score by the
    overlap ratio: `score = base_weight * suppression * (matched/total)`.

    This is the cheapest useful upgrade toward BM25/hybrid retrieval.
    Literature:
      Robertson & Zaragoza. "The Probabilistic Relevance Framework:
        BM25 and Beyond." Foundations and Trends in IR (2009).
        https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf
      Robertson, Walker & Hancock-Beaulieu. TREC-3 (1994) — the
        original BM25 paper.
      Izacard & Grave (2020) arxiv:2007.01282 — hybrid retrieval
        rationale (keyword + dense).
      Thakur et al. BEIR benchmark (2021) arxiv:2104.08663 — hybrid
        beats either alone across tasks.
    True BM25 adds idf + term-frequency saturation; our first step is
    term-overlap weighting, which is the leading-order correction.

    `min_overlap_ratio` is a soft floor: 0.0 keeps every match (legacy
    behaviour). Set e.g. 0.3 to drop hits that matched fewer than 30 %
    of the caller's keywords — useful when keywords are many but
    individually weak.

    Mutates seen_meta in place (inserts entries for new hits).
    """
    if not keywords or kg is None:
        return []
    # Normalize + dedupe the caller keywords for the overlap denominator.
    # Empty strings are ignored because they can't match anything useful.
    total_keywords = len({kw.strip().lower() for kw in keywords if kw and kw.strip()})
    if total_keywords == 0:
        return []

    # Walk each caller keyword individually so we know WHICH keywords
    # matched each entity. Accumulate (doc, meta, suppression, matched_set).
    per_entity: dict = {}
    for kw in keywords:
        if not kw or not kw.strip():
            continue
        kw_norm = kw.strip().lower()
        try:
            hits = keyword_lookup(
                kg,
                [kw],
                added_by=added_by,
                kind_filter=kind_filter,
                collection=collection,
            )
        except Exception:
            continue
        for mid, doc, meta, suppression in hits:
            if suppression < suppression_floor:
                continue
            entry = per_entity.setdefault(
                mid,
                {
                    "doc": doc,
                    "meta": meta,
                    "suppression": suppression,
                    "matched": set(),
                },
            )
            entry["matched"].add(kw_norm)
            # Use the strongest suppression seen across keywords (most forgiving).
            if suppression > entry["suppression"]:
                entry["suppression"] = suppression

    kw_ranked = []
    for mid, entry in per_entity.items():
        overlap_ratio = len(entry["matched"]) / float(total_keywords)
        if overlap_ratio < min_overlap_ratio:
            continue
        score = base_weight * entry["suppression"] * overlap_ratio
        kw_ranked.append((score, (entry["doc"] or "")[:300], mid))
        if mid not in seen_meta:
            seen_meta[mid] = {
                "meta": entry["meta"],
                "doc": entry["doc"],
                "similarity": 0.0,
            }
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
    keywords=None,  # caller-provided keyword list for Channel C
    kg=None,
    added_by=None,
    kind=None,
    fetch_limit_per_view=50,
    include_graph=False,
    seed_ids=None,
    graph_seed_topk_per_view=3,
):
    """Unified 3-channel search pipeline. The ONE implementation used by
    every multi-view search tool (mempalace_kg_search + declare_intent).

    Channels:
        A (cosine):  one ranked list per view, dense vector similarity.
                     Multi-view comes from Context.queries — the caller-
                     authored analogue of HyDE-style multi-embedding
                     retrieval (Gao et al 2022, arxiv:2212.10496).
        B (graph):   1-hop neighbours of explicit seeds or top cosine hits
                     (only when include_graph=True and kg is provided).
                     Graph-augmented retrieval motivation: Zep/Graphiti
                     (Rasmussen et al 2025, arxiv:2501.13956) showed
                     temporal-KG hybrid search beats vector-only on
                     agent-memory benchmarks.
        C (keyword): caller-provided keywords resolved via the
                     entity_keywords table (no auto-extraction — see
                     keyword_lookup). Hybrid retrieval rationale:
                     Izacard & Grave 2020 (arxiv:2007.01282), BEIR
                     (Thakur et al 2021, arxiv:2104.08663).

    Merged via Reciprocal Rank Fusion (Cormack et al 2009; see rrf_merge).
    Caller applies hybrid rerank + adaptive-K
    using seen_meta.

    Args:
        collection: ChromaDB collection (memories, entities, …).
        views: already-validated + sanitized list of perspective strings.
        keywords: caller-provided keyword list (Context.keywords). Required for
                  Channel C — when None or empty, Channel C is silently skipped.
        kg: KnowledgeGraph — required for keyword channel + graph channel.
        added_by: optional agent filter for memory collections.
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
    if added_by:
        where_filter = {"added_by": added_by}
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
        added_by=added_by,
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
