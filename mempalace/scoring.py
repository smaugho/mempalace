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
# Post-P2 revision (2026-04-22): W_REL restored in SIGNED form.
#
# P2's initial cutover retired W_REL on the reasoning that "Channel D
# replaces it". That was wrong. Channel D and W_REL do complementary
# things:
#   - Channel D is a RETRIEVAL channel: it surfaces memories from
#     similar past contexts that cosine alone would miss. Its
#     contribution is always ≥ 0 (rank-based RRF). It cannot demote
#     a high-cosine memory that the agent has explicitly rated
#     irrelevant.
#   - W_REL is a SCORING term: it acts on every retrieved memory
#     regardless of how it was retrieved, and is SIGNED — a memory
#     rated irrelevant drops BELOW neutral, symmetric to how rated-
#     useful rises above.
#
# The signed form drops the old `norm_rel = (rel+1)/2` squashing. A
# memory with no feedback contributes 0 to W_REL * rel (proper
# "no signal" semantics, not a free 0.5 * W_REL baseline). Weights
# no longer form a convex combination in [0,1], but nothing downstream
# (sort order, adaptive-K gap detection, logging) requires it.
#
# Weight split (user-approved, 2026-04-22):
#   W_SIM=0.45 W_REL=0.15 W_IMP=0.18 W_DECAY=0.12 W_AGENT=0.10
# (sum of magnitudes = 1.0; rel's max swing ±0.15 is intentionally
# less aggressive than the old 0.20 because doubling the dynamic
# range via the signed form would have over-weighted a noisy signal).
DEFAULT_SEARCH_WEIGHTS = {
    "sim": 0.45,
    "rel": 0.15,
    "imp": 0.18,
    "decay": 0.12,
    "agent": 0.10,
}

# Weighted-RRF channel seeds for tool_kg_search's 4-channel composition.
# A (cosine) leads; D (context-feedback) gets the strongest weight because
# it carries the personalised signal that cosine alone can't. B (graph)
# and C (keyword) are tiebreakers. Bruch/Gai/Ingber 2023 ACM TOIS:
# weighted RRF is competitive with learned-rank fusion.
# DEFAULT_CHANNEL_WEIGHTS seeds the weighted-RRF merge. P3 polish added
# learning (scope='channel' in scoring_weight_feedback via
# compute_learned_weights); tool_wake_up loads the learned values and
# get_effective_channel_weights() merges them over these defaults.
#
# Hand-picked starting point per the redesign plan — cosine leads,
# context (Channel D) dominant because personalised feedback is the
# highest-signal channel when data exists, keyword second (domain-term
# hits are surgical), graph smallest (most diffuse). No data yet to say
# these are right; the learner's job is to find out.
DEFAULT_CHANNEL_WEIGHTS = {
    "cosine": 1.0,
    "graph": 0.7,
    "keyword": 0.8,
    "context": 1.5,
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


# Runtime per-channel weight override — populated by
# set_learned_channel_weights(). Mirrors _learned_weights for hybrid_score.
_learned_channel_weights: dict = {}


def set_learned_channel_weights(weights: dict):
    """Set per-channel RRF weights (cosine, graph, keyword, context).

    Consumed by multi_channel_search via DEFAULT_CHANNEL_WEIGHTS lookup;
    when this module-level dict has entries they override the static
    defaults on a per-channel basis.
    """
    global _learned_channel_weights
    _learned_channel_weights = dict(weights) if weights else {}


def get_effective_channel_weights() -> dict:
    """Merge learned channel weights over the static defaults."""
    merged = dict(DEFAULT_CHANNEL_WEIGHTS)
    merged.update(_learned_channel_weights)
    return merged


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
    relevance_feedback: float = 0.0,  # signed [-1, +1]; see note below
    mode: str = "search",  # "search" or "l1"
    session_match: bool = False,  # candidate from same MCP session
    intent_type_match: bool = False,  # candidate from same intent type
) -> float:
    """Unified scoring function for all mempalace retrieval.

    Modes:
        "search" — similarity-primary, with SIGNED relevance feedback.
                   A memory rated irrelevant demotes symmetric to how
                   a rated-useful memory boosts. Channel D (retrieval
                   channel) is complementary — it surfaces recall the
                   scoring term can't find.
        "l1"    — importance-primary. Hard tier separation (imp 5 ALWAYS
                   outranks imp 4). Used by Layer1 wake-up.

    relevance_feedback ∈ [-1, +1] (clamped). Convention:
       +1.0 = relevance 5, relevant=True   (full confidence useful)
       +0.2 = relevance 1, relevant=True   (weak useful)
        0.0 = no feedback AT ALL           (no signal, neutral)
       -0.2 = relevance 1, relevant=False  (weak irrelevant)
       -1.0 = relevance 5, relevant=False  (full confidence irrelevant)
    intent.py produces this from the confidence column on the
    new ``rated_useful`` / ``rated_irrelevant`` edges; see
    ``lookup_context_feedback``.
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
        return imp * TIER_MULTIPLIER + decay * 2.5 + agent_boost + session_boost + intent_boost
    else:
        # Search: weighted sum with SIGNED relevance. Channel D contributes
        # recall upstream (rank-based, always ≥ 0); W_REL here contributes
        # signed demotion / boost per-memory — the two are complementary,
        # not substitutes.
        W_SIM = _learned_weights.get("sim", DEFAULT_SEARCH_WEIGHTS["sim"])
        W_REL = _learned_weights.get("rel", DEFAULT_SEARCH_WEIGHTS["rel"])
        W_IMP = _learned_weights.get("imp", DEFAULT_SEARCH_WEIGHTS["imp"])
        W_DECAY = _learned_weights.get("decay", DEFAULT_SEARCH_WEIGHTS["decay"])
        W_AGENT = _learned_weights.get("agent", DEFAULT_SEARCH_WEIGHTS["agent"])

        norm_sim = max(0.0, min(1.0, sim))  # already 0-1
        norm_imp = (imp - 1.0) / 4.0  # 1→0, 5→1
        norm_decay = 1.0 + (decay / DECAY_WEIGHT)  # -0.2→0, 0→1
        norm_agent = 1.0 if agent_match else 0.0  # binary
        # SIGNED, no squash. No-feedback (rel=0) contributes 0. Rated-
        # irrelevant memories drop below neutral; rated-useful rise above.
        signed_rel = max(-1.0, min(1.0, float(relevance_feedback)))

        # provenance affinity boosts are ADDITIVE (not weighted)
        # so they don't disrupt the existing weight balance.
        prov_boost = 0.0
        if session_match:
            prov_boost += SESSION_BOOST_SEARCH
        if intent_type_match:
            prov_boost += INTENT_TYPE_BOOST_SEARCH

        return (
            W_SIM * norm_sim
            + W_REL * signed_rel
            + W_IMP * norm_imp
            + W_DECAY * norm_decay
            + W_AGENT * norm_agent
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
                # the record id IS the entity id).
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
    """Weighted Reciprocal Rank Fusion (Cormack/Clarke/Büttcher 2009 with
    per-channel weighting from Bruch/Gai/Ingber 2023 ACM TOIS).

        score(d) = Σ_list w_list / (k + rank_list(d))

    Rank-based and scale-free, which is why it can fuse channels that
    return wildly different score magnitudes without per-channel
    normalisation. The per-channel weight lets a high-signal channel
    (e.g. Channel D / context-feedback) dominate a low-signal one
    without turning the whole pipeline into a reciprocal-rank arithmetic
    tie.

    References:
      Cormack/Clarke/Büttcher 2009 SIGIR (canonical RRF, k=60):
        https://dl.acm.org/doi/10.1145/1571941.1572114
      Bruch/Gai/Ingber 2023 ACM TOIS (weighted RRF is competitive with
      learned-rank fusion): https://dl.acm.org/doi/10.1145/3596512

    Args:
        ranked_lists: dict of ``list_name -> [(score, text, mid), ...]``
            OR ``list_name -> ([(score, text, mid), ...], weight)`` for the
            weighted form. Lists without a weight tuple default to w=1.0,
            preserving legacy callers.
        k: RRF constant (default 60, Cormack et al. 2009).

    Returns:
        (rrf_scores, candidate_map, channel_attribution).
    """
    rrf_scores = {}
    candidate_map = {}
    channel_attribution = {}

    for list_name, entry in ranked_lists.items():
        if isinstance(entry, tuple) and len(entry) == 2:
            candidates, weight = entry
            try:
                weight = float(weight)
            except (TypeError, ValueError):
                weight = 1.0
        else:
            candidates = entry
            weight = 1.0

        deduped = {}
        for score, text, mid in candidates:
            if mid not in deduped or score > deduped[mid][0]:
                deduped[mid] = (score, text)
        ranked = sorted(deduped.items(), key=lambda x: x[1][0], reverse=True)

        for rank, (mid, (_score, text)) in enumerate(ranked):
            rrf_contribution = weight / (k + rank + 1)
            rrf_scores[mid] = rrf_scores.get(mid, 0.0) + rrf_contribution
            if mid not in candidate_map:
                candidate_map[mid] = (text, list_name)
            if mid not in channel_attribution:
                channel_attribution[mid] = set()
            channel_attribution[mid].add(list_name.split("_")[0])

    return rrf_scores, candidate_map, channel_attribution


# ══════════════════════════════════════════════════════════════════════
# two_stage_retrieve — THE canonical retrieval pipeline. Every
# context-creating tool (declare_intent, declare_operation, kg_search)
# routes its per-channel ranked lists through this function. One
# implementation, one semantic, one scale. Do not re-implement
# Stage-2/Stage-3 in callers; extend this helper instead.
#
#   Stage 1  : rrf_merge fuses the per-channel ranked lists into
#              rrf_scores (rank-only, scale-free — Cormack 2009;
#              Bruch 2023).
#   Stage 2  : the top-M (default 50) RRF candidates are re-scored by
#              hybrid_score() using the feature-rich meta captured by
#              multi_channel_search (importance, decay, signed
#              relevance_feedback, agent/session/intent provenance).
#              Nogueira & Cho 2019: per-candidate rerank is where the
#              signals RRF cannot see get their say.
#   Stage 3  : adaptive_k gap-detects the cutoff on the reranked
#              score (0.3–0.8 dynamic range — Mao et al. EMNLP 2025).
#
# Returns a list of dicts in final rank order:
#   {"id": mid,
#    "hybrid_score": float,     # post-rerank score (the ONLY score
#                                # exposed to callers)
#    "text": str,                # candidate_map text, caller can
#                                # shorten / project as needed
#    "channel": str,             # primary channel that surfaced it
#    "meta": dict,               # raw metadata (importance, date,
#                                # added_by, session_id, etc.)
#    "similarity": float}        # max cosine similarity seen
# ══════════════════════════════════════════════════════════════════════


def two_stage_retrieve(
    ranked_lists: dict,
    seen_meta: dict,
    *,
    agent: str = "",
    session_id: str = "",
    intent_type_id: str = "",
    context_feedback: dict = None,
    rerank_top_m: int = 50,
    max_k: int = 20,
    min_k: int = 3,
    time_window: dict = None,
) -> list:
    """Canonical RRF → hybrid_score rerank → adaptive_K pipeline.

    Args:
        ranked_lists: dict of channel_name -> [(score, text, id)] OR
            channel_name -> ([(score, text, id)], weight). Same shape
            rrf_merge consumes.
        seen_meta: flat dict of id -> {meta, doc, similarity, source} as
            accumulated across the pipes that fed ``ranked_lists``. Any
            ids missing from seen_meta fall back to neutral meta when
            scored (importance=3, no date, no provenance).
        agent: agent name for agent_match boost in hybrid_score.
        session_id: session id for session_match boost.
        intent_type_id: active intent type for intent_type_match boost.
        context_feedback: dict of id -> signed [-1, +1] feedback from
            rated_useful / rated_irrelevant walk (see
            walk_rated_neighbourhood).
        rerank_top_m: take top-M from RRF into Stage 2. Default 50.
        max_k / min_k: adaptive_k bounds.
        time_window: optional {start, end} ISO date strings. Inside-window
            items get a +0.15 additive boost; outside items are NOT
            excluded (soft decay).

    Returns:
        (reranked, rrf_scores, candidate_map) where:
          - reranked: list of result dicts (see module docstring) in final order
          - rrf_scores: full Stage-1 dict of id -> fused rank score (for
            downstream callers that need access to the raw RRF signal,
            e.g. intent-type promotion checks on past executions).
          - candidate_map: id -> (text, primary_channel) from RRF.
    """
    context_feedback = context_feedback or {}

    # Stage 1 — RRF fuse
    rrf_scores, candidate_map, channel_attribution = rrf_merge(ranked_lists)
    if not rrf_scores:
        return [], {}, {}

    top_m = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:rerank_top_m]

    # Stage 2 — hybrid_score rerank
    reranked = []
    for mid, rrf in top_m:
        info = seen_meta.get(mid) or {}
        meta = info.get("meta") or {}
        similarity = float(info.get("similarity", 0.0) or 0.0)
        importance = float(meta.get("importance", 3) or 3)
        date_anchor = (
            meta.get("last_touched") or meta.get("date_added") or meta.get("filed_at") or ""
        )
        is_agent_match = bool(agent and meta.get("added_by") == agent)
        last_relevant = meta.get("last_relevant_at", "") or ""
        rel_fb = float(context_feedback.get(mid, 0.0) or 0.0)
        sess_match = bool(session_id and meta.get("session_id") == session_id)
        itype_match = bool(intent_type_id and meta.get("intent_type") == intent_type_id)

        score = hybrid_score(
            similarity=similarity,
            importance=importance,
            date_iso=date_anchor,
            agent_match=is_agent_match,
            last_relevant_iso=last_relevant,
            relevance_feedback=rel_fb,
            mode="search",
            session_match=sess_match,
            intent_type_match=itype_match,
        )

        # time_window soft-decay: inside-window items get a boost,
        # outside items still rank.
        if time_window and date_anchor:
            tw_start = time_window.get("start", "")
            tw_end = time_window.get("end", "")
            if tw_start and tw_end and tw_start <= date_anchor <= tw_end:
                score += 0.15
            elif tw_start and date_anchor >= tw_start and not tw_end:
                score += 0.15
            elif tw_end and date_anchor <= tw_end and not tw_start:
                score += 0.15

        text, channel = candidate_map.get(mid, ("", "unknown"))
        reranked.append(
            {
                "id": mid,
                "hybrid_score": score,
                "rrf_score": float(rrf),  # internal debug / downstream use
                "text": text,
                "channel": channel,
                "meta": meta,
                "similarity": similarity,
                "source": info.get("source", ""),
            }
        )

    # Stage 3 — adaptive_k cutoff
    reranked.sort(key=lambda x: x["hybrid_score"], reverse=True)
    if len(reranked) > 1:
        cut = adaptive_k(
            [r["hybrid_score"] for r in reranked],
            max_k=max_k,
            min_k=min_k,
        )
        reranked = reranked[:cut]
    return reranked, rrf_scores, candidate_map


# ══════════════════════════════════════════════════════════════════════
# Unified multi-channel search pipeline — the ONE implementation used
# everywhere we do similarity + keyword + graph retrieval.
# Callers (tool_kg_search unified + declare_intent context) just pass
# a collection + mandatory multi-view queries + filters, then feed the
# returned ranked_lists + seen_meta to two_stage_retrieve() above.
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


def validate_context(
    context,
    *,
    queries_min=2,
    queries_max=5,
    keywords_min=2,
    keywords_max=5,
    entities_min=1,
    entities_max=10,
):
    """Shared validation for the unified Context object.

    Context = {
      queries:  list[str]  (mandatory, 2-5)
      keywords: list[str]  (mandatory, 2-5; caller-provided, no auto-extract)
      entities: list[str]  (mandatory, 1-10 — the link-author pipeline
                            accumulates Adamic-Adar evidence from every
                            (entity, context) co-occurrence, so contexts
                            with zero entities produce no candidates at
                            all. See docs/link_author_plan.md §2.3.)
    }

    Returns (clean_context_dict, error_dict_or_None). If error is truthy,
    caller should return it directly. Used by every read AND every write tool.
    """
    # Permissive transport coercion: some MCP clients JSON-stringify
    # nested object args. Parse str-shaped context once before the
    # type-check rather than rejecting outright — mirrors the same
    # tolerance memory_feedback already has at the finalize boundary.
    if isinstance(context, str):
        try:
            import json as _json

            context = _json.loads(context)
        except Exception:
            return None, {
                "success": False,
                "error": (
                    "context arrived as an unparseable string. Pass a dict "
                    "{queries, keywords, entities} or a JSON-encoded object "
                    "of that shape."
                ),
            }
    if context is None or not isinstance(context, dict):
        return None, {
            "success": False,
            "error": (
                "context is required (dict with 'queries', 'keywords', "
                "'entities'). Example: "
                '{"queries": ["auth rate limiting", "brute force hardening"], '
                '"keywords": ["auth", "rate", "limiting"], '
                '"entities": ["LoginService", "AuthRateLimiter"]}'
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

    raw_entities = context.get("entities")
    if isinstance(raw_entities, str):
        return None, {
            "success": False,
            "error": (
                "context.entities must be a list of strings, not a string. "
                'Example: ["LoginService", "AuthRateLimiter"]'
            ),
        }
    entities, err = _validate_string_list(
        raw_entities,
        "context.entities",
        entities_min,
        entities_max,
        '["LoginService", "AuthRateLimiter"]',
    )
    if err:
        return None, err

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


def walk_rated_neighbourhood(
    active_context_id,
    kg,
    *,
    hops: int = 2,
    sim_decay: float = 0.5,
    surfaced_weight: float = 0.3,
) -> dict:
    """ONE walk over active context + similar_to neighbourhood; two aggregates.

    Both Channel D (retrieval recall) and hybrid_score's W_REL (signed
    per-memory scoring) read the same edges, but aggregate differently.
    This helper does the walk once and returns both — downstream callers
    pick the aggregate they need.

    Per-edge contribution: ``weight × confidence`` where ``weight`` is 1.0
    for the active context itself and the ``get_similar_contexts`` decayed
    value for 1-2 hop neighbours.

    Returns::

        {
          "rated_scores":     {memory_id: signed_float ∈ [-1, +1]},
          "channel_D_list":   [(score, doc_placeholder, memory_id), …]
        }

    `rated_scores` uses rated_useful (positive) and rated_irrelevant
    (negative) ONLY — `surfaced` is not a feedback signal.
    `channel_D_list` additionally includes `surfaced` contributions
    weighted by ``surfaced_weight`` (default 0.3) because Channel D's
    role is recall: "was previously surfaced in a similar context" is
    a useful retrieval hint even without explicit rating.
    """
    out = {"rated_scores": {}, "channel_D_list": []}
    if not (active_context_id and kg):
        return out
    # (context_id, weight) — active first at w=1.0, then similar neighbours.
    neighbourhood = [(active_context_id, 1.0)]
    try:
        for cid, sim in kg.get_similar_contexts(active_context_id, hops=hops, decay=sim_decay):
            neighbourhood.append((cid, float(sim)))
    except Exception:
        pass

    rated_scores: dict = {}  # rated_useful − rated_irrelevant
    channel_d: dict = {}  # rated_useful − rated_irrelevant + surfaced_weight * surfaced
    for cid, weight in neighbourhood:
        try:
            edges = kg.query_entity(cid, direction="outgoing")
        except Exception:
            continue
        for edge in edges:
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
            if pred == "rated_useful":
                delta = weight * conf
                rated_scores[mid] = rated_scores.get(mid, 0.0) + delta
                channel_d[mid] = channel_d.get(mid, 0.0) + delta
            elif pred == "rated_irrelevant":
                delta = weight * conf
                rated_scores[mid] = rated_scores.get(mid, 0.0) - delta
                channel_d[mid] = channel_d.get(mid, 0.0) - delta
            elif pred == "surfaced":
                # Recall-only; does not contribute to W_REL.
                channel_d[mid] = channel_d.get(mid, 0.0) + weight * surfaced_weight

    # ── Triple-scoped feedback (migration 018) ──
    # Triples cannot be the object of edge-based rated_* / surfaced
    # because add_triple's entity auto-creation would pollute the
    # entities table with phantom rows. Triple feedback lives in
    # triple_context_feedback and is read once here for the whole
    # neighbourhood, merged into the same rated_scores / channel_d
    # dicts keyed by triple_id. Downstream consumers (RRF fusion,
    # hybrid_score) treat triples and entities uniformly at the id
    # level, so once the dicts are merged the rest of the pipeline
    # sees no difference between the two namespaces.
    try:
        triple_rows = kg.get_triple_feedback([cid for cid, _ in neighbourhood])
    except Exception:
        triple_rows = []
    if triple_rows:
        weight_by_ctx = {cid: w for cid, w in neighbourhood}
        for row in triple_rows:
            cid = row.get("context_id")
            weight = weight_by_ctx.get(cid, 0.0)
            if not weight:
                continue
            tid = row.get("triple_id")
            if not tid:
                continue
            kind = row.get("kind")
            try:
                conf = float(row.get("confidence") or 1.0)
            except (TypeError, ValueError):
                conf = 1.0
            if kind == "rated_useful":
                delta = weight * conf
                rated_scores[tid] = rated_scores.get(tid, 0.0) + delta
                channel_d[tid] = channel_d.get(tid, 0.0) + delta
            elif kind == "rated_irrelevant":
                delta = weight * conf
                rated_scores[tid] = rated_scores.get(tid, 0.0) - delta
                channel_d[tid] = channel_d.get(tid, 0.0) - delta
            elif kind == "surfaced":
                # Recall-only for triples too; does not contribute to W_REL.
                channel_d[tid] = channel_d.get(tid, 0.0) + weight * surfaced_weight

    # Clamp rated_scores to [-1, +1].
    for mid in list(rated_scores.keys()):
        if rated_scores[mid] > 1.0:
            rated_scores[mid] = 1.0
        elif rated_scores[mid] < -1.0:
            rated_scores[mid] = -1.0

    # Channel D: positive-only filter for rank-based RRF.
    channel_list = [(score, "", mid) for mid, score in channel_d.items() if score > 0.0]
    channel_list.sort(key=lambda x: x[0], reverse=True)

    out["rated_scores"] = rated_scores
    out["channel_D_list"] = channel_list
    return out


def lookup_context_feedback(active_context_id, kg, *, hops=2, sim_decay=0.5):
    """Per-memory signed feedback signal from the active context's neighbourhood.

    Thin wrapper around ``walk_rated_neighbourhood`` that returns only the
    ``rated_scores`` aggregate — consumed by hybrid_score as signed
    relevance_feedback (the W_REL term). A memory rated irrelevant in
    similar contexts drops below neutral.

    If a caller ALSO needs Channel D's ranked list (retrieval recall),
    call ``walk_rated_neighbourhood`` directly to avoid a second walk
    over the same neighbourhood.
    """
    return walk_rated_neighbourhood(active_context_id, kg, hops=hops, sim_decay=sim_decay)[
        "rated_scores"
    ]


# ──────────────────── S1: OPERATION NEIGHBOURHOOD ────────────────────
# Parallel walker for the operation-tier graph. Channel D reads the
# memory-relevance edges (rated_useful / rated_irrelevant); this walker
# reads the operation-correctness edges (performed_well / performed_
# poorly). The two aggregates are orthogonal by design — conflating
# them would mean "memory X was useful" and "op Y was the right move"
# share a signal, which is not true. See Leontiev 1981 for the AAO
# rationale and arXiv 2512.18950 for the empirical case.


def walk_operation_neighbourhood(
    active_context_id,
    kg,
    *,
    hops: int = 2,
    sim_decay: float = 0.5,
) -> dict:
    """ONE walk over active context + similar_to neighbourhood for ops.

    Structurally identical to ``walk_rated_neighbourhood`` but reads
    ``performed_well`` (positive) and ``performed_poorly`` (negative)
    edges instead. Returns signed scores keyed by op entity id.

    Returns::

        {
          "op_scores":   {op_id: signed_float ∈ [-1, +1]},
          "good_ops":    [(score, op_id), …]   # score > 0, ranked desc
          "bad_ops":     [(score, op_id), …]   # score < 0, ranked asc
        }

    The ranked lists are consumed by the declare_operation response
    builder to populate ``past_operations.good_precedents`` (what
    worked) and ``past_operations.avoid_patterns`` (what was wrong) —
    rendered in their own gate-prompt section, kept separate from the
    memory list.
    """
    out = {"op_scores": {}, "good_ops": [], "bad_ops": []}
    if not (active_context_id and kg):
        return out
    neighbourhood = [(active_context_id, 1.0)]
    try:
        for cid, sim in kg.get_similar_contexts(active_context_id, hops=hops, decay=sim_decay):
            neighbourhood.append((cid, float(sim)))
    except Exception:
        pass

    op_scores: dict = {}
    for cid, weight in neighbourhood:
        try:
            edges = kg.query_entity(cid, direction="outgoing")
        except Exception:
            continue
        for edge in edges:
            if not edge.get("current", True):
                continue
            pred = edge.get("predicate")
            op_id = edge.get("object")
            if not op_id:
                continue
            try:
                conf = float(edge.get("confidence") or 1.0)
            except (TypeError, ValueError):
                conf = 1.0
            if pred == "performed_well":
                op_scores[op_id] = op_scores.get(op_id, 0.0) + weight * conf
            elif pred == "performed_poorly":
                op_scores[op_id] = op_scores.get(op_id, 0.0) - weight * conf

    # Clamp to [-1, +1]
    for op_id in list(op_scores.keys()):
        if op_scores[op_id] > 1.0:
            op_scores[op_id] = 1.0
        elif op_scores[op_id] < -1.0:
            op_scores[op_id] = -1.0

    good = sorted(
        [(s, op_id) for op_id, s in op_scores.items() if s > 0.0],
        key=lambda x: x[0],
        reverse=True,
    )
    bad = sorted(
        [(s, op_id) for op_id, s in op_scores.items() if s < 0.0],
        key=lambda x: x[0],
    )
    out["op_scores"] = op_scores
    out["good_ops"] = good
    out["bad_ops"] = bad
    return out


def retrieve_past_operations(active_context_id, kg, *, k: int = 5, hops: int = 2):
    """Build the `past_operations` bundle for declare_operation response.

    Returns::

        {
          "good_precedents": [{op_id, score, tool, args_summary, ...}, ...],
          "avoid_patterns":  [{op_id, score, tool, args_summary, ...}, ...],
        }

    Each entry is limited to the op's public-ish properties from the
    KG so the gate prompt has enough signal to include/exclude without
    needing a second query per op. k caps each list length.
    """
    result = {"good_precedents": [], "avoid_patterns": []}
    walk = walk_operation_neighbourhood(active_context_id, kg, hops=hops)

    def _hydrate(pairs):
        out = []
        for score, op_id in pairs:
            try:
                ent = kg.get_entity(op_id)
            except Exception:
                ent = None
            if not ent:
                out.append({"op_id": op_id, "score": round(float(score), 3)})
                continue
            # properties may be a dict or a JSON string depending on KG impl
            props = ent.get("properties") or {}
            if isinstance(props, str):
                try:
                    import json as _json

                    props = _json.loads(props)
                except Exception:
                    props = {}
            out.append(
                {
                    "op_id": op_id,
                    "score": round(float(score), 3),
                    "tool": props.get("tool", ""),
                    "args_summary": props.get("args_summary", ""),
                    "quality": props.get("quality"),
                    "reason": (props.get("reason") or "")[:200],
                }
            )
        return out

    result["good_precedents"] = _hydrate(walk["good_ops"][:k])
    result["avoid_patterns"] = _hydrate(walk["bad_ops"][:k])
    return result


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
    base_weight=1.0,
    min_idf=0.0,
):
    """CHANNEL C: caller-provided keyword lookup weighted by robust IDF.

    Each keyword resolves to entity_ids via the entity_keywords index, then
    we fetch the matching ChromaDB record to pull document + metadata.

    Scoring = ``base_weight × Σ idf(kw) for kw in matched_keywords``

    ``idf(t) = log( (N - freq(t) + 0.5) / (freq(t) + 0.5) + 1 )`` — the
    IDF formula from BM25 (Robertson & Jones 1976; standard form per
    Wikipedia / Gao et al. "Which BM25 Do You Mean?" 2020). The `+1`
    outside the log is the robust stabiliser — keeps the term
    non-negative even when a term appears in more than half the corpus.
    Freq and N come from the ``keyword_idf`` table maintained
    incrementally by ``knowledge_graph.record_keyword_observations``.

    Scope note on naming: what we call "BM25-IDF" is *just* the IDF
    component of BM25. Full BM25 also has (a) term-frequency
    saturation via k1 and (b) document-length normalisation via b.
    Neither applies here — this channel treats each keyword-entity
    match as binary (TF ≡ 1, no occurrence counting), so the k1 and
    b pieces are no-ops on our input shape. For this channel's data
    model, "IDF-alone" and "full BM25" produce identical rankings.

    Rare terms dominate the per-entity score; dominant corpus-wide
    terms collapse toward zero. ``min_idf`` is an early-exit floor —
    any keyword whose idf falls below it is dropped before the lookup
    even runs. Default is ``0.0`` (accept everything) because at
    personal-palace scale (N=10..1000) even the "common" terms have
    IDFs around 0.05–0.3, so a hard positive floor drops too much.
    The ordering still favours rare terms; the floor only matters if
    you want to prune stop-word-like terms at very large N.

    Cold-start: when the keyword_idf table is empty (fresh palace, no
    records yet), every keyword falls back to uniform weight=1.0 so
    the channel still returns something. As the corpus grows, the
    per-keyword IDFs differentiate and the signal sharpens.

    Honest framing: at personal-palace scale (N << 10K), IDF weighting
    is a modest, directionally-correct improvement over the old
    overlap_ratio heuristic — not a silver bullet. Empirical validation
    of IDF specifically at this scale is absent from the literature;
    the math (rare-term weight ratio remains meaningful even at N=10)
    plus zero-regression cold-start fallback made this a low-risk
    upgrade. BEIR (Thakur et al. 2021 arXiv:2104.08663) validates the
    broader hybrid approach; mempalace's 4-channel weighted RRF is
    what cashes that in. Gao/Lu/Lin 2020 "Which BM25 Do You Mean?"
    found no statistically significant differences between BM25
    variants — the specific IDF form doesn't matter much, only that
    IDF is there at all.

    Mutates seen_meta in place (inserts entries for new hits).
    """
    if not keywords or kg is None:
        return []
    # Normalise + dedupe.
    cleaned_kws = list({kw.strip().lower() for kw in keywords if kw and kw.strip()})
    if not cleaned_kws:
        return []

    # Fetch IDFs in one call and drop any keyword below the floor.
    try:
        idf_map = kg.get_keyword_idf(cleaned_kws)
    except Exception:
        idf_map = {}
    # When there's no idf data yet (cold-start), fall back to a
    # uniform weight of 1.0 per matched keyword so the channel still
    # returns something — the agent gets SOME keyword signal before
    # the corpus has populated the IDF table.
    cold_start = not any(v > 0.0 for v in idf_map.values())
    effective_idf = {}
    for kw in cleaned_kws:
        idf = float(idf_map.get(kw, 0.0) or 0.0)
        if cold_start:
            effective_idf[kw] = 1.0
        elif idf >= min_idf:
            effective_idf[kw] = idf

    if not effective_idf:
        return []

    per_entity: dict = {}
    for kw, idf in effective_idf.items():
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
        for mid, doc, meta, _suppression in hits:
            entry = per_entity.setdefault(
                mid,
                {"doc": doc, "meta": meta, "score": 0.0, "matched": 0},
            )
            entry["score"] += idf
            entry["matched"] += 1

    kw_ranked = []
    for mid, entry in per_entity.items():
        score = base_weight * entry["score"]
        if score <= 0.0:
            continue
        kw_ranked.append((score, (entry["doc"] or "")[:300], mid))
        if mid not in seen_meta:
            seen_meta[mid] = {
                "meta": entry["meta"],
                "doc": entry["doc"],
                "similarity": 0.0,
            }
    return kw_ranked


def _build_graph_channel(collection, kg, seed_ids, kind_filter, seen_meta, top_per_seed_limit=None):
    """CHANNEL B: 1-hop graph neighbours of seed entities. Mutates seen_meta.

    Each neighbour's score = ``max(0.2, seed_sim × 0.8) × 1/log(degree(seed) + 2)``.

    Degree-dampening rationale:
      A mega-hub like ``ga_agent`` has hundreds of incident edges; without
      dampening, every one of its 1-hop neighbours gets the same (strong)
      seed-similarity contribution and the channel floods with generic
      hits. The log-dampening shrinks the per-neighbour contribution of
      high-degree seeds so a degree-50 hub doesn't drown a degree-2
      specialist.

      The ``1 / log(d + 2)`` shape gives:
        degree 1  → 1/log(3)   ≈ 0.91
        degree 5  → 1/log(7)   ≈ 0.51
        degree 20 → 1/log(22)  ≈ 0.32
        degree 50 → 1/log(52)  ≈ 0.25
      i.e. degree-50 contributions are ~30% of degree-2.

    References:
      Hogan et al. "Knowledge Graphs." arXiv:2003.02320 (2021).
      West & Leskovec. "Human wayfinding in information networks."
        WWW 2012 — inverse-log degree is the standard dampening shape
        for random-walk-over-KG retrieval.
      Bollacker et al. "Freebase." SIGMOD 2008 — same dampening for
        popular-entity bias.

    Returns a ranked list; empty if seed_ids is empty or kg is None.
    """
    import math

    graph_ranked = []
    if not seed_ids or kg is None:
        return graph_ranked
    for seed_id in seed_ids:
        try:
            edges = kg.query_entity(seed_id, direction="both")
        except Exception:
            continue
        seed_sim = seen_meta.get(seed_id, {}).get("similarity", 0.0)
        # Degree dampening factor for this seed. get_entity_degree does
        # the SQL count; fall back to 0 on any error so we don't kill
        # the channel over a hiccup.
        try:
            deg = kg.get_entity_degree(seed_id)
        except Exception:
            deg = 0
        # +2 inside log keeps the output positive even at degree=0.
        damp = 1.0 / math.log(deg + 2) if deg >= 0 else 1.0
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
            base = max(0.2, seed_sim * 0.8)
            score = base * damp
            graph_ranked.append((score, seen_meta[neighbor]["doc"], neighbor))
            count += 1

            # Channel B triple emission: emit the traversed edge itself
            # (not just the neighbour) so triples get RRF cross-channel
            # boost. Without this, triples only hit Channel A cosine over
            # mempalace_triples and never accumulate multi-channel rank
            # contributions the way memories/entities do. Skip-list
            # predicates (schema glue, feedback topology) are excluded
            # — same filter as _index_triple_statement at embed time.
            # kind_filter (when set) targets entity kinds; triples are
            # not entity-kind so we skip triple emission whenever a
            # kind filter is active.
            if not kind_filter:
                pred = e.get("predicate") or ""
                triple_id = e.get("triple_id")
                statement = e.get("statement")
                from .knowledge_graph import _TRIPLE_SKIP_PREDICATES

                if triple_id and statement and pred not in _TRIPLE_SKIP_PREDICATES:
                    triple_text = (statement or "")[:200].replace("\n", " ")
                    graph_ranked.append((score, triple_text, triple_id))
                    if triple_id not in seen_meta:
                        seen_meta[triple_id] = {
                            "meta": {
                                "subject": subj,
                                "predicate": pred,
                                "object": obj,
                                "confidence": e.get("confidence", 1.0),
                                "source": "triple",
                            },
                            "doc": triple_text,
                            "similarity": 0.0,
                        }
            if top_per_seed_limit and count >= top_per_seed_limit:
                break
    return graph_ranked


def _build_context_channel(
    kg,
    active_context_id: str,
    seen_meta: dict,
    *,
    hops: int = 2,
    surfaced_weight: float = 0.3,
    precomputed=None,
) -> list:
    """CHANNEL D — context-feedback retrieval.

    Delegates the neighbourhood walk to ``walk_rated_neighbourhood`` so
    Channel D and hybrid_score's W_REL (``lookup_context_feedback``)
    share a single pass over the KG. When the caller has already walked
    (e.g. tool_kg_search computes W_REL first), it can pass the result
    as ``precomputed`` to skip re-walking.

    Returns the ``channel_D_list`` aggregate: rated_useful (positive) +
    ``surfaced_weight`` × surfaced (weak positive) − rated_irrelevant,
    each scaled by the neighbour's similarity weight, filtered to
    net-positive memories, sorted descending.

    References:
      Resnick et al. CSCW 1994 "GroupLens" — memory-based collaborative
        filtering; the context↔memory matrix is the same shape.
      Burke. UMUAI 2002 — hybrid recommender systems.
      Khattab & Zaharia. SIGIR 2020 arXiv:2004.12832 — ColBERT; the
        similar_to neighbourhood is the late-interaction expansion.
    """
    if not active_context_id or kg is None:
        return []
    if precomputed is None:
        precomputed = walk_rated_neighbourhood(
            active_context_id, kg, hops=hops, surfaced_weight=surfaced_weight
        )
    items = precomputed.get("channel_D_list") or []
    if not items:
        return []
    # Patch docs from seen_meta where available (cosmetic — RRF only needs rank).
    if seen_meta:
        items = [
            (score, (seen_meta.get(mid) or {}).get("doc", "") or "", mid)
            for score, _doc, mid in items
        ]
    return items


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
    active_context_id=None,  # P2: enables Channel D (context-feedback)
    channel_weights=None,  # P2: weighted RRF weights per channel
    rated_walk=None,  # precomputed walk_rated_neighbourhood output (avoids re-walk)
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

    # ── Channel D (P2): context-feedback ──
    # Reuses `rated_walk` when provided (tool_kg_search computes W_REL
    # via lookup_context_feedback first, then passes the walk result
    # here so we don't query the same neighbourhood twice).
    if active_context_id and kg is not None:
        context_ranked = _build_context_channel(
            kg,
            active_context_id,
            seen_meta,
            precomputed=rated_walk,
        )
        if context_ranked:
            ranked_lists["context"] = context_ranked

    # ── Weighted RRF merge ──
    # Each ranked list gets a per-channel weight; lists without an
    # explicit weight fall back to 1.0 (legacy-compatible). When the
    # caller didn't pass channel_weights explicitly, merge the learned
    # per-channel weights (populated by tool_wake_up from
    # kg.compute_learned_weights(...scope='channel')) over the static
    # DEFAULT_CHANNEL_WEIGHTS so learning has a live read path.
    if channel_weights is None:
        channel_weights = get_effective_channel_weights()
    weighted_lists = {}
    for name, entries in ranked_lists.items():
        base = name.split("_")[0]  # cosine_0 → cosine
        weight = float(channel_weights.get(base, 1.0))
        weighted_lists[name] = (entries, weight)

    rrf_scores, _candidate_map, attribution = rrf_merge(weighted_lists)
    return {
        "rrf_scores": rrf_scores,
        "seen_meta": seen_meta,
        "ranked_lists": ranked_lists,
        "attribution": attribution,
    }
