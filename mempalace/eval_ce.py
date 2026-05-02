"""
Cross-encoder rerank evaluation harness for mempalace retrieval.

Adrian's CE eval design 2026-05-02. Goal: quantify whether a CE rerank
step between bi-encoder retrieval and the LLM injection gate (a) lifts
retrieval quality, especially for long content where Q1 found Channel A
truncates at 1800 chars (record_ga_agent_channel_a_content_truncation_1800),
and (b) reduces LLM gate cost via an asymmetric "auto-accept above
T_HIGH only" policy that preserves the rated_irrelevant signal Adrian
called out (record_ga_agent_damping_factor_literature_and_ce_correction).

Data source: live palace feedback edges. Every ``rated_useful`` triple
gives us a ``(context, memory, label=1)`` example; every
``rated_irrelevant`` triple gives ``label=0``. The context's
``properties.queries`` are the bi-encoder probes; the memory's
``content`` (or summary fallback) is the document. relevance 1-5 lives
in ``triples.properties`` and is preserved as a graded label for NDCG.

References
----------
* Khattab & Zaharia 2020 -- ColBERT (multi-view base; CE comparison).
* Anthropic 2024 -- Contextual Retrieval (rerank as recall lift step).
* Wadden et al. 2020 -- SciFact (retrieval+rerank metric stratification).
* Reimers & Gurevych 2019 -- Sentence-BERT (cross-encoder pattern).

Models -- caller picks via ``ce_model`` parameter:
* ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (default; 80MB, ~5ms/pair CPU)
* ``BAAI/bge-reranker-v2-m3`` (568MB, multilingual, ~30ms/pair)
* ``cross-encoder/ms-marco-MiniLM-L-12-v2`` (130MB, English)

Usage
-----
The harness has two layers:

1. ``build_examples_from_palace`` -- pure SQLite reader, no ML deps;
   pulls the labeled ``(context, memory)`` pairs and bucketises by
   memory content length. This works on any palace path.

2. ``score_pairs`` / ``rerank_with_ce`` / ``compute_metrics`` -- the
   bi-encoder side reads existing chroma views (no new deps); the CE
   side lazy-imports ``sentence_transformers`` and is skipped when
   that's not installed. ``mempalace[eval]`` extra installs it.

The harness never mutates the palace -- it's read-only against the
SQLite + Chroma stores. Safe to run against the live palace OR a
backup directory copy.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# ─── data shapes ─────────────────────────────────────────────────────


@dataclass
class LabeledPair:
    """One (context, memory) example with feedback label.

    label is 1 for rated_useful (relevance >= 3) and 0 for
    rated_irrelevant (relevance <= 2). relevance preserves the raw
    1-5 grade for NDCG / fine-grained metrics. content_length is the
    memory's body length used for stratification.
    """

    context_id: str
    memory_id: str
    label: int  # 0 or 1
    relevance: int  # 1-5
    content_length: int
    queries: list[str] = field(default_factory=list)
    memory_content: str = ""
    memory_summary: str = ""
    feedback_ts: str = ""
    feedback_reason: str = ""


@dataclass
class BucketMetrics:
    """Per-bucket eval metrics. ``ndcg`` and ``precision_at_k`` are
    reported as means over all contexts in the bucket; ``n_pairs`` is
    the example count used to compute them."""

    bucket_name: str
    n_pairs: int
    n_positive: int  # label=1
    n_negative: int  # label=0
    ndcg_at_10: float = 0.0
    precision_at_10: float = 0.0
    recall_at_10: float = 0.0
    auc: float = 0.0  # rank-based discrimination
    score_mean_pos: float = 0.0
    score_mean_neg: float = 0.0


@dataclass
class CostCurvePoint:
    """One point on the auto-accept-only cost curve.

    threshold = T_HIGH ; auto_accept_count = #items with score >= T;
    auto_accept_precision = (#truly-positive among auto-accepted) /
    (auto-accepted total). llm_savings_pct = fraction of items the
    LLM gate skips at this threshold.
    """

    threshold: float
    auto_accept_count: int
    auto_accept_precision: float
    llm_savings_pct: float
    recall_at_threshold: float  # of all positives, fraction reached above T


# ─── content-length bucket boundaries ────────────────────────────────

# Stratification matches Q1 finding: Channel A truncates record embed_doc
# at 1800 chars (mcp_server.py:879 _EMBED_DOC_MAX_CHARS for all-MiniLM-L6-v2's
# 256-token ceiling). The >1800 bucket is where CE's no-truncation
# advantage should be largest.
BUCKETS: list[tuple[str, int, int]] = [
    ("short_<200", 0, 200),
    ("medium_200-500", 200, 500),
    ("long_500-1800", 500, 1800),
    ("xlong_>1800", 1800, 10**9),
]


def bucket_for(length: int) -> str:
    """Return the bucket name for a content length."""
    for name, lo, hi in BUCKETS:
        if lo <= length < hi:
            return name
    return "xlong_>1800"


# ─── palace reader (no ML deps) ──────────────────────────────────────


def default_palace_db() -> Path:
    """Default to ``~/.mempalace/palace/knowledge_graph.sqlite3``."""
    return Path.home() / ".mempalace" / "palace" / "knowledge_graph.sqlite3"


def _load_props(raw):
    """Best-effort parse triples.properties / entities.properties JSON."""
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_examples_from_palace(  # noqa: C901
    db_path: str | Path | None = None,
    *,
    only_current: bool = True,
    min_pairs_per_context: int = 1,
) -> list[LabeledPair]:
    """Materialise labeled examples from a palace SQLite db.

    Reads ``rated_useful`` (label=1) and ``rated_irrelevant`` (label=0)
    edges, joining each to the subject context's queries (from
    ``entities.properties.queries``) and the object memory's content.
    Returns the flattened list; bucket grouping is left to the caller
    so the same example list can power multiple stratifications.

    Filters
    -------
    only_current : bool
        Only include edges with empty/null ``valid_to``. Default True.
        Set False to include invalidated history if you want to study
        how feedback churns over time.
    min_pairs_per_context : int
        Drop contexts with fewer than this many labeled pairs. Default 1
        (no filter). NDCG / precision metrics need at least 2 examples
        per context to be meaningful; raise this for cleaner subsets.
    """
    path = Path(db_path) if db_path is not None else default_palace_db()
    if not path.exists():
        raise FileNotFoundError(f"palace SQLite db not found: {path}")
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row

    valid_clause = " AND (valid_to IS NULL OR valid_to = '')" if only_current else ""
    sql = (
        "SELECT subject, object, predicate, properties "
        "FROM triples WHERE predicate IN ('rated_useful', 'rated_irrelevant') " + valid_clause
    )
    rows = list(conn.execute(sql))

    # Pre-fetch context queries and memory content/summary in batches to
    # keep the per-pair SELECT count down on large palaces.
    context_ids = sorted({r["subject"] for r in rows})
    memory_ids = sorted({r["object"] for r in rows})

    ctx_props: dict[str, dict] = {}
    if context_ids:
        placeholders = ",".join("?" * len(context_ids))
        for ent in conn.execute(
            f"SELECT id, properties FROM entities WHERE id IN ({placeholders})",
            context_ids,
        ):
            ctx_props[ent["id"]] = _load_props(ent["properties"])

    mem_data: dict[str, dict] = {}
    if memory_ids:
        placeholders = ",".join("?" * len(memory_ids))
        for ent in conn.execute(
            f"SELECT id, content, properties FROM entities WHERE id IN ({placeholders})",
            memory_ids,
        ):
            mem_data[ent["id"]] = {
                "content": ent["content"] or "",
                "properties": _load_props(ent["properties"]),
            }

    pairs: list[LabeledPair] = []
    for r in rows:
        ctx_id = r["subject"]
        mem_id = r["object"]
        props = _load_props(r["properties"])
        relevance = _safe_int(props.get("relevance"))
        label = 1 if r["predicate"] == "rated_useful" else 0
        ctx_p = ctx_props.get(ctx_id, {})
        queries = list(ctx_p.get("queries") or [])
        if not queries:
            # Some context entities may have lost their queries during
            # gardening or never carried them; skip rather than fake.
            continue
        mem = mem_data.get(mem_id, {})
        memory_content = mem.get("content") or ""
        memory_summary = ""
        mem_p = mem.get("properties") or {}
        summary = mem_p.get("summary")
        if isinstance(summary, dict):
            # Render the dict to single-line prose so the CE sees a
            # meaningful surface. Lazy import to avoid pulling
            # knowledge_graph at module-load time.
            from .knowledge_graph import serialize_summary_for_embedding

            try:
                memory_summary = serialize_summary_for_embedding(summary)
            except Exception:
                memory_summary = ""
        elif isinstance(summary, str):
            memory_summary = summary
        pairs.append(
            LabeledPair(
                context_id=ctx_id,
                memory_id=mem_id,
                label=label,
                relevance=relevance,
                content_length=len(memory_content),
                queries=queries,
                memory_content=memory_content,
                memory_summary=memory_summary,
                feedback_ts=str(props.get("ts") or ""),
                feedback_reason=str(props.get("reason") or ""),
            )
        )

    if min_pairs_per_context > 1:
        from collections import Counter

        per_ctx = Counter(p.context_id for p in pairs)
        keep_ctx = {c for c, n in per_ctx.items() if n >= min_pairs_per_context}
        pairs = [p for p in pairs if p.context_id in keep_ctx]

    return pairs


def group_by_context(pairs: Iterable[LabeledPair]) -> dict[str, list[LabeledPair]]:
    """Group labeled pairs by context_id for per-context ranking."""
    out: dict[str, list[LabeledPair]] = {}
    for p in pairs:
        out.setdefault(p.context_id, []).append(p)
    return out


def stratify_by_length(
    pairs: Iterable[LabeledPair],
) -> dict[str, list[LabeledPair]]:
    """Bucketise pairs by ``content_length``. Empty buckets are
    represented as empty lists so callers always see all four keys."""
    out: dict[str, list[LabeledPair]] = {b[0]: [] for b in BUCKETS}
    for p in pairs:
        out[bucket_for(p.content_length)].append(p)
    return out


# ─── scoring (bi-encoder + cross-encoder) ────────────────────────────


def score_bi_encoder(pairs: list[LabeledPair]) -> dict[tuple[str, str], float]:
    """Score (context, memory) pairs via the same bi-encoder Channel A
    uses (chromadb's all-MiniLM-L6-v2 ONNX). Returns a dict keyed by
    (context_id, memory_id) → cosine similarity.

    Uses the maximum cosine across all of context.queries vs the
    memory's text -- mirrors ``multi_view_minmax_sim``'s max-of-max
    aggregator that production already uses for similar_to scoring.

    Implementation note: we re-run embeddings here rather than hit the
    live mempalace_entities collection, because the Chroma store may
    have stale rows (gardener-rewritten summaries don't always
    re-sync; see record_ga_agent_q3_corrected_gardener_rewrite_stale_views).
    Re-embedding from current SQLite content is the truth-y path.
    """
    if not pairs:
        return {}
    try:
        # chromadb already pulls onnxruntime + the all-MiniLM-L6-v2
        # default embedding function; reuse rather than add a new dep.
        from chromadb.utils import embedding_functions
    except ImportError as e:  # pragma: no cover -- defensive
        raise RuntimeError(
            "chromadb is required for bi-encoder scoring; install mempalace[base]."
        ) from e

    embed = embedding_functions.DefaultEmbeddingFunction()

    # Build the full corpus to embed once: every unique query + every
    # unique memory text. Cuts redundant embed cost when many contexts
    # share queries or memories.
    query_set: set[str] = set()
    for p in pairs:
        for q in p.queries:
            if q.strip():
                query_set.add(q)
    mem_set: set[str] = set()
    mem_text: dict[str, str] = {}
    for p in pairs:
        # Use summary+content (Anthropic Contextual Retrieval pattern,
        # same as _add_memory_internal). Truncate to 1800 chars to
        # match the bi-encoder's actual training-time ceiling (Q1).
        text = (
            (p.memory_summary + "\n\n" + p.memory_content) if p.memory_summary else p.memory_content
        )
        text = text[:1800]
        if not text:
            continue
        mem_text[p.memory_id] = text
        mem_set.add(p.memory_id)

    queries = sorted(query_set)
    mems = sorted(mem_set)
    if not queries or not mems:
        return {}

    q_vecs = embed(queries)
    m_vecs = embed([mem_text[m] for m in mems])
    q_idx = {q: i for i, q in enumerate(queries)}
    m_idx = {m: i for i, m in enumerate(mems)}

    # Cosine: chromadb's ONNX embedder returns L2-normalised vectors,
    # so dot product = cosine. Cast to native float at the boundary
    # because chromadb returns numpy float32 vectors and we want the
    # downstream JSONL telemetry write to work without ``np.ndarray``
    # JSON encoders.
    def dot(a, b):
        return float(sum(float(x) * float(y) for x, y in zip(a, b)))

    out: dict[tuple[str, str], float] = {}
    for p in pairs:
        if p.memory_id not in m_idx:
            continue
        mv = m_vecs[m_idx[p.memory_id]]
        best = -1.0
        for q in p.queries:
            if q not in q_idx:
                continue
            qv = q_vecs[q_idx[q]]
            score = dot(qv, mv)
            if score > best:
                best = score
        out[(p.context_id, p.memory_id)] = float(max(0.0, best))
    return out


def score_cross_encoder(
    pairs: list[LabeledPair],
    *,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> dict[tuple[str, str], float]:
    """Score (context, memory) pairs via a cross-encoder.

    Uses ``sentence_transformers.CrossEncoder``. Returns a dict keyed by
    (context_id, memory_id) → CE score (raw model output, NOT
    normalised; some CEs return logits, some sigmoid probs).

    Lazy-imports sentence-transformers; raises a clear error if the
    eval extra isn't installed.

    Aggregator: max-over-queries, mirroring score_bi_encoder so the
    two are directly comparable.

    For each (context, memory) we feed the FULL memory_content (not
    the 1800-char-truncated bi-encoder input) -- this is the whole
    point of the eval, capturing the asymmetric advantage Q1
    predicted for the >1800 bucket. CE input length is bounded by
    the model's max_seq_length (typically 512 tokens for MiniLM, up
    to 8192 for some BGE variants).
    """
    if not pairs:
        return {}
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise RuntimeError(
            "sentence-transformers is required for CE scoring. "
            "Install mempalace[eval] or `pip install sentence-transformers`."
        ) from e

    ce = CrossEncoder(model_name)

    # Build per-pair (query, doc) batch with max-aggregation marker so
    # we can group rows back. Format: list[(idx, query, doc)] then
    # CE.predict the (query,doc) pairs in a single batch.
    rows: list[tuple[int, str, str]] = []
    pair_index: list[tuple[str, str]] = []
    for p in pairs:
        if not p.memory_content and not p.memory_summary:
            pair_index.append((p.context_id, p.memory_id))
            continue
        doc = (
            (p.memory_summary + "\n\n" + p.memory_content) if p.memory_summary else p.memory_content
        )
        idx = len(pair_index)
        pair_index.append((p.context_id, p.memory_id))
        for q in p.queries:
            if q.strip():
                rows.append((idx, q, doc))

    if not rows:
        return {(c, m): 0.0 for (c, m) in pair_index}

    qd_pairs = [(q, d) for (_i, q, d) in rows]
    raw_scores = ce.predict(qd_pairs, show_progress_bar=False)
    # Aggregate max-over-queries per pair index.
    best: dict[int, float] = {}
    for (idx, _q, _d), s in zip(rows, raw_scores):
        s = float(s)
        if idx not in best or s > best[idx]:
            best[idx] = s

    out: dict[tuple[str, str], float] = {}
    for i, (c, m) in enumerate(pair_index):
        out[(c, m)] = best.get(i, 0.0)
    return out


# ─── metrics ─────────────────────────────────────────────────────────


def _ndcg_at_k(labels_in_rank_order: list[int], k: int) -> float:
    """NDCG@k for a single ranked list of binary/graded labels.

    Uses the standard 2^rel - 1 gain over log2(rank+2) discount.
    Returns 0.0 when there are no positive labels (perfect ideal == 0
    means we can't normalise; treating undefined as 0 keeps means
    well-defined when buckets contain hard-negative-only contexts).
    """
    if not labels_in_rank_order:
        return 0.0
    k = min(k, len(labels_in_rank_order))
    actual = sum((2 ** labels_in_rank_order[i] - 1) / math.log2(i + 2) for i in range(k))
    ideal_order = sorted(labels_in_rank_order, reverse=True)
    ideal = sum((2 ** ideal_order[i] - 1) / math.log2(i + 2) for i in range(k))
    if ideal <= 0:
        return 0.0
    return actual / ideal


def _precision_at_k(labels_in_rank_order: list[int], k: int) -> float:
    """Fraction of the top-k that are positive (label > 0)."""
    if not labels_in_rank_order:
        return 0.0
    k = min(k, len(labels_in_rank_order))
    return sum(1 for x in labels_in_rank_order[:k] if x > 0) / k


def _recall_at_k(labels_in_rank_order: list[int], k: int) -> float:
    """Of all positives in the list, fraction reached in top-k."""
    if not labels_in_rank_order:
        return 0.0
    total_pos = sum(1 for x in labels_in_rank_order if x > 0)
    if total_pos == 0:
        return 0.0
    k = min(k, len(labels_in_rank_order))
    return sum(1 for x in labels_in_rank_order[:k] if x > 0) / total_pos


def _auc_pairwise(scored: list[tuple[float, int]]) -> float:
    """ROC-AUC via the rank-based U statistic. Returns 0.5 when only
    one class is present (uninformative).

    O(n log n) via sort; suitable for the few-thousand-pairs scale we
    have here. For larger eval sets, switch to numpy."""
    if not scored:
        return 0.5
    n_pos = sum(1 for _s, lbl in scored if lbl > 0)
    n_neg = len(scored) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Sort ascending by score; ranks start at 1.
    s = sorted(scored, key=lambda x: x[0])
    # Average ranks for ties to keep AUC unbiased.
    ranks = [0.0] * len(s)
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j][0] == s[i][0]:
            j += 1
        avg = (i + j + 1) / 2  # 1-based mean rank for the tie group
        for k in range(i, j):
            ranks[k] = avg
        i = j
    pos_rank_sum = sum(ranks[i] for i, (_s, lbl) in enumerate(s) if lbl > 0)
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def compute_bucket_metrics(
    bucket_name: str,
    pairs: list[LabeledPair],
    scores: dict[tuple[str, str], float],
    *,
    k: int = 10,
) -> BucketMetrics:
    """Per-bucket NDCG / precision / recall / AUC using the supplied
    ``scores`` dict to rank within each context.

    ``label`` is binary in the rank order; ``relevance`` (1-5) is used
    only for the bucket-level statistics, not the NDCG gain (we
    intentionally stay binary here so 'rated_useful but barely' (rel=3)
    isn't double-counted vs 'load-bearing' (rel=5); a follow-up eval
    can switch to graded NDCG by feeding ``relevance`` directly)."""
    if not pairs:
        return BucketMetrics(bucket_name=bucket_name, n_pairs=0, n_positive=0, n_negative=0)

    n_pos = sum(1 for p in pairs if p.label > 0)
    n_neg = len(pairs) - n_pos

    by_ctx = group_by_context(pairs)
    ndcgs: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    pos_scores: list[float] = []
    neg_scores: list[float] = []

    for ctx_id, ctx_pairs in by_ctx.items():
        # Rank this context's pairs by score, descending.
        scored = []
        for p in ctx_pairs:
            sc = scores.get((ctx_id, p.memory_id), 0.0)
            scored.append((sc, p.label))
            (pos_scores if p.label > 0 else neg_scores).append(sc)
        scored.sort(key=lambda x: x[0], reverse=True)
        labels = [lbl for _s, lbl in scored]
        ndcgs.append(_ndcg_at_k(labels, k))
        precisions.append(_precision_at_k(labels, k))
        recalls.append(_recall_at_k(labels, k))

    flat_scored = [(scores.get((p.context_id, p.memory_id), 0.0), p.label) for p in pairs]

    return BucketMetrics(
        bucket_name=bucket_name,
        n_pairs=len(pairs),
        n_positive=n_pos,
        n_negative=n_neg,
        ndcg_at_10=sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        precision_at_10=sum(precisions) / len(precisions) if precisions else 0.0,
        recall_at_10=sum(recalls) / len(recalls) if recalls else 0.0,
        auc=_auc_pairwise(flat_scored),
        score_mean_pos=sum(pos_scores) / len(pos_scores) if pos_scores else 0.0,
        score_mean_neg=sum(neg_scores) / len(neg_scores) if neg_scores else 0.0,
    )


def cost_savings_curve(
    pairs: list[LabeledPair],
    scores: dict[tuple[str, str], float],
    *,
    thresholds: Iterable[float] | None = None,
) -> list[CostCurvePoint]:
    """Sweep T_HIGH from low to high and report the asymmetric
    auto-accept-only policy's cost/quality trade-off.

    For each threshold T:
      - auto_accept_count: items with score >= T
      - auto_accept_precision: of those, fraction that are truly label=1
      - llm_savings_pct: 1 - (auto_accept_count / total_items) -- not
        quite right since llm-side handles the OTHER half too. Reframe:
        llm_savings_pct = auto_accept_count / total_items (the fraction
        that skips the LLM gate).
      - recall_at_threshold: of all label=1 positives in the corpus,
        fraction that are above T (auto-accepted correctly).

    Use this to pick T_HIGH for production. A good T_HIGH gives high
    auto_accept_precision (so the items we skip the gate on are
    confidently relevant) AND non-trivial llm_savings_pct (so the cost
    win is real). Recall at T_HIGH should track precision; if recall
    collapses while precision stays high, we're auto-accepting only
    the few obvious wins and most useful items still hit the LLM."""
    if not pairs:
        return []
    if thresholds is None:
        # 0.0 to 1.0 in 0.05 steps. Adapt to score scale if model
        # outputs are out-of-range; the sweep is still informative.
        thresholds = [round(0.05 * i, 2) for i in range(0, 21)]
    flat = [(scores.get((p.context_id, p.memory_id), 0.0), p.label) for p in pairs]
    n_total = len(flat)
    n_pos_total = sum(1 for _s, lbl in flat if lbl > 0)
    out: list[CostCurvePoint] = []
    for T in thresholds:
        above = [(s, lbl) for s, lbl in flat if s >= T]
        n_above = len(above)
        n_above_pos = sum(1 for _s, lbl in above if lbl > 0)
        prec = n_above_pos / n_above if n_above else 0.0
        savings = n_above / n_total if n_total else 0.0
        recall = n_above_pos / n_pos_total if n_pos_total else 0.0
        out.append(
            CostCurvePoint(
                threshold=float(T),
                auto_accept_count=n_above,
                auto_accept_precision=prec,
                llm_savings_pct=savings,
                recall_at_threshold=recall,
            )
        )
    return out


# ─── reporting ───────────────────────────────────────────────────────


def format_bucket_table(metrics: list[BucketMetrics]) -> str:
    """Render bucket metrics as a fixed-width text table, ready for a
    terminal log or markdown code block."""
    header = (
        f"{'bucket':<18} {'n':>5} {'pos':>5} {'neg':>5} "
        f"{'ndcg@10':>8} {'p@10':>6} {'r@10':>6} {'auc':>5} "
        f"{'mean+':>6} {'mean-':>6}"
    )
    lines = [header, "-" * len(header)]
    for m in metrics:
        lines.append(
            f"{m.bucket_name:<18} {m.n_pairs:>5} {m.n_positive:>5} "
            f"{m.n_negative:>5} {m.ndcg_at_10:>8.3f} {m.precision_at_10:>6.3f} "
            f"{m.recall_at_10:>6.3f} {m.auc:>5.3f} "
            f"{m.score_mean_pos:>6.3f} {m.score_mean_neg:>6.3f}"
        )
    return "\n".join(lines)


def format_cost_curve(points: list[CostCurvePoint]) -> str:
    """Render the auto-accept cost curve as a small text table."""
    header = f"{'T_HIGH':>7} {'n_above':>8} {'precision':>10} {'llm_savings':>12} {'recall':>8}"
    lines = [header, "-" * len(header)]
    for p in points:
        lines.append(
            f"{p.threshold:>7.2f} {p.auto_accept_count:>8} "
            f"{p.auto_accept_precision:>10.3f} "
            f"{p.llm_savings_pct:>12.3f} {p.recall_at_threshold:>8.3f}"
        )
    return "\n".join(lines)


# ─── top-level eval driver ───────────────────────────────────────────


def run_eval(
    db_path: str | Path | None = None,
    *,
    ce_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    skip_ce: bool = False,
    output_jsonl: str | Path | None = None,
    min_pairs_per_context: int = 2,
) -> dict:
    """Top-level driver. Reads the palace, scores both pipelines,
    computes per-bucket metrics + cost curves for each, and optionally
    appends a JSONL telemetry row at ``output_jsonl``.

    Returns the aggregated result dict so callers can also use this
    programmatically (eg from a notebook or pytest fixture):

        {
          "n_pairs": int, "n_contexts": int,
          "bi_encoder": {"buckets": [...], "cost_curve": [...]},
          "cross_encoder": {"buckets": [...], "cost_curve": [...]},
        }

    skip_ce: if True (or sentence-transformers isn't installed), only
    the bi-encoder side runs. Useful for fast smoke tests in CI."""
    pairs = build_examples_from_palace(
        db_path, only_current=True, min_pairs_per_context=min_pairs_per_context
    )
    by_bucket = stratify_by_length(pairs)
    n_contexts = len({p.context_id for p in pairs})
    result: dict = {
        "n_pairs": len(pairs),
        "n_contexts": n_contexts,
        "buckets_n": {b: len(v) for b, v in by_bucket.items()},
        "bi_encoder": None,
        "cross_encoder": None,
    }

    bi_scores = score_bi_encoder(pairs)
    bi_buckets = [
        compute_bucket_metrics(name, by_bucket[name], bi_scores) for name, _, _ in BUCKETS
    ]
    bi_curve = cost_savings_curve(pairs, bi_scores)
    result["bi_encoder"] = {
        "buckets": [m.__dict__ for m in bi_buckets],
        "cost_curve": [pt.__dict__ for pt in bi_curve],
    }

    if not skip_ce:
        try:
            ce_scores = score_cross_encoder(pairs, model_name=ce_model)
            ce_buckets = [
                compute_bucket_metrics(name, by_bucket[name], ce_scores) for name, _, _ in BUCKETS
            ]
            ce_curve = cost_savings_curve(pairs, ce_scores)
            result["cross_encoder"] = {
                "model": ce_model,
                "buckets": [m.__dict__ for m in ce_buckets],
                "cost_curve": [pt.__dict__ for pt in ce_curve],
            }
        except RuntimeError as e:
            result["cross_encoder"] = {"error": str(e)}

    if output_jsonl is not None:
        out_path = Path(output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as f:
            from datetime import datetime, timezone

            f.write(
                json.dumps(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "db_path": str(db_path or default_palace_db()),
                        **result,
                    }
                )
                + "\n"
            )

    return result


# ─── CLI shim ────────────────────────────────────────────────────────


def _cli(argv: list[str] | None = None) -> int:
    """Minimal CLI: ``python -m mempalace.eval_ce [--skip-ce] [--db PATH]``.

    Prints the bucket tables and cost curves to stdout; appends a JSONL
    row to ``~/.mempalace/hook_state/eval_ce_log.jsonl`` so runs are
    durably comparable across days."""
    import argparse

    parser = argparse.ArgumentParser(prog="mempalace.eval_ce")
    parser.add_argument("--db", default=None, help="Path to palace SQLite db")
    parser.add_argument("--skip-ce", action="store_true", help="Run bi-encoder only (no CE)")
    parser.add_argument(
        "--ce-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="HuggingFace cross-encoder model id",
    )
    parser.add_argument(
        "--min-pairs-per-context",
        type=int,
        default=2,
        help="Drop contexts with fewer than this many labeled pairs",
    )
    parser.add_argument(
        "--output",
        default=str(Path.home() / ".mempalace" / "hook_state" / "eval_ce_log.jsonl"),
        help="JSONL telemetry log path",
    )
    args = parser.parse_args(argv)

    result = run_eval(
        db_path=args.db,
        ce_model=args.ce_model,
        skip_ce=args.skip_ce,
        output_jsonl=args.output,
        min_pairs_per_context=args.min_pairs_per_context,
    )

    print(f"\nEvaluated {result['n_pairs']} labeled pairs across {result['n_contexts']} contexts.")
    print(f"Bucket sizes: {result['buckets_n']}\n")

    if result["bi_encoder"]:
        print("=== Bi-encoder (chromadb default all-MiniLM-L6-v2) ===")
        bms = [BucketMetrics(**m) for m in result["bi_encoder"]["buckets"]]
        print(format_bucket_table(bms))
        print("\n-- auto-accept cost curve --")
        bcs = [CostCurvePoint(**p) for p in result["bi_encoder"]["cost_curve"]]
        print(format_cost_curve(bcs))
        print()

    ce = result.get("cross_encoder")
    if ce:
        if "error" in ce:
            print(f"=== Cross-encoder skipped: {ce['error']}")
        else:
            print(f"=== Cross-encoder ({ce['model']}) ===")
            cms = [BucketMetrics(**m) for m in ce["buckets"]]
            print(format_bucket_table(cms))
            print("\n-- auto-accept cost curve --")
            ccs = [CostCurvePoint(**p) for p in ce["cost_curve"]]
            print(format_cost_curve(ccs))
            print()

    return 0


if __name__ == "__main__":  # pragma: no cover -- CLI shim
    raise SystemExit(_cli())
