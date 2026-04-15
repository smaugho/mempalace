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
                   Used by mempalace_search, kg_search, declare_intent context.
        "l1"    — importance-primary. Hard tier separation (imp 5 ALWAYS
                   outranks imp 4). Used by Layer1 wake-up.

    Components:
        - similarity: cosine similarity to query/intent description
        - importance: 1-5 tier
        - decay: power-law with importance-dependent stability
        - agent_match: boost for own content
        - last_relevant_at: decay reset on found_useful
        - relevance_feedback: continuous [-1.0, +1.0] from type-level feedback.
          Positive = found_useful (magnitude = confidence from 1-5 relevance score).
          Negative = found_irrelevant. 0 = no feedback yet.
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
# Shared search primitives — used by both declare_intent and mempalace_search
# ══════════════════════════════════════════════════════════════════════

STOP_WORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "will",
        "what",
        "when",
        "where",
        "how",
        "all",
        "each",
        "then",
        "also",
        "been",
        "have",
        "does",
        "should",
        "would",
        "could",
    }
)


def extract_keywords(text, max_words=5):
    """Extract significant keywords from text for keyword search."""
    if not text or len(text) <= 5:
        return []
    return [w.lower() for w in text.split() if len(w) > 3 and w.lower() not in STOP_WORDS][
        :max_words
    ]


def keyword_search(collection, query_text, kg=None, wing=None, limit_per_word=5, max_words=5):
    """Search a ChromaDB collection by keyword ($contains).

    Returns list of (drawer_id, document, metadata, suppression_score).
    """
    words = extract_keywords(query_text, max_words=max_words)
    if not words or not collection:
        return []

    results = []
    seen_ids = set()
    where_filter = {"wing": wing} if wing else None

    for word in words:
        try:
            kw_results = collection.get(
                where_document={"$contains": word},
                where=where_filter,
                include=["documents", "metadatas"],
                limit=limit_per_word,
            )
            if kw_results and kw_results["ids"]:
                for i, did in enumerate(kw_results["ids"]):
                    if did in seen_ids:
                        continue
                    seen_ids.add(did)
                    meta = kw_results["metadatas"][i] or {}
                    doc = kw_results["documents"][i] or ""
                    suppression = 1.0
                    if kg:
                        try:
                            suppression = kg.get_keyword_suppression(did)
                        except Exception:
                            pass
                    results.append((did, doc, meta, suppression))
        except Exception:
            continue
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
