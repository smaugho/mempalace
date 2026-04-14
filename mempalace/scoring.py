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
    relevance_feedback: int = 0,  # +1 = found_useful, -1 = found_irrelevant, 0 = no feedback
    mode: str = "search",  # "search" or "l1"
) -> float:
    """Unified scoring function for all mempalace retrieval.

    Modes:
        "search" — similarity-primary. Importance, decay, agent, relevance
                   are tiebreakers. Used by mempalace_search, kg_search.
        "l1"    — importance-primary. Hard tier separation (imp 5 ALWAYS
                   outranks imp 4). Decay is within-tier tiebreaker.
                   Used by Layer1 wake-up, sort_by="score".

    Components:
        - similarity: cosine similarity (0 for L1 mode)
        - importance: 1-5 tier
        - decay: power-law with importance-dependent stability
        - agent_match: boost for own content
        - last_relevant_at: decay reset on found_useful
        - relevance_feedback: +1/-1 from found_useful/found_irrelevant edges
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
        # Search: similarity dominates, everything else is tiebreaker
        agent_boost = AGENT_BOOST_SEARCH if agent_match else 0.0
        rel_boost = RELEVANCE_BOOST * relevance_feedback
        return sim + (imp - 3.0) * 0.1 + decay + agent_boost + rel_boost


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
