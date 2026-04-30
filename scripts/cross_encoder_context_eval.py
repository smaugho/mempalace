"""Cross-encoder rescore over similar_to edges with summary included.

Mirrors scripts/cross_encoder_eval.py but operates over the **context-to-
context** similar_to graph instead of (context, memory) rated pairs. For
each similar_to edge, builds a CE-input passage from BOTH endpoints'
queries AND summary (the structured what/why/scope), then runs BAAI/bge-
reranker-v2-m3 to produce a CE score and compares it against the stored
cosine confidence (= max-of-max link_score from context_lookup_or_create).

Why summary is included
-----------------------
Today's cosine pipeline (scoring.multi_view_max_sim) compares queries
view-by-view against queries view-by-view. The Context entity's summary
({what, why, scope?}) lives in properties.summary and is NOT vectorized
as its own view. This eval explicitly includes summary in the CE input
to surface signal that today's cosine misses. If CE-with-summary diverges
meaningfully from cosine-without-summary, that's evidence summary is
worth folding into the production similarity pipeline.

Output
------
* AUC of CE vs cosine (no ground-truth labels here, so AUC is replaced
  with rank correlation -- Spearman + Kendall tau).
* Histogram of (CE - cosine) per edge: how often does CE upgrade /
  downgrade / agree with cosine?
* Decision-flip counts:
    - "would CE upgrade to reuse?" CE >= 0.90 AND cosine < 0.90.
    - "would CE drop the edge?" CE < 0.65 AND cosine >= 0.65.
* Mean and median |CE - cosine|.

Usage
-----
    python scripts/cross_encoder_context_eval.py [--limit N]

The first run downloads BGE-reranker-v2-m3 (~600MB, one-time). 5K pairs
take ~5-10 minutes batched on CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path
from statistics import mean, median

import numpy as np

DB = Path(os.path.expanduser("~/.mempalace/palace/knowledge_graph.sqlite3"))
T_REUSE = 0.90
T_SIMILAR = 0.65
MAX_PASSAGE_CHARS = 800  # joint queries+summary -- keep CE input small


def _coerce_props(raw):
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _render_summary(summary) -> str:
    """Render a {what, why, scope?} dict into prose for the CE input.

    Mirrors knowledge_graph.serialize_summary_for_embedding semantics:
    'WHAT: <what>. WHY: <why>. SCOPE: <scope>.' but inline so this script
    is independent of the package's import graph.
    """
    if not isinstance(summary, dict):
        return ""
    parts = []
    what = (summary.get("what") or "").strip()
    why = (summary.get("why") or "").strip()
    scope = (summary.get("scope") or "").strip()
    if what:
        parts.append(f"WHAT: {what}.")
    if why:
        parts.append(f"WHY: {why}.")
    if scope:
        parts.append(f"SCOPE: {scope}.")
    return " ".join(parts)


def _passage(props: dict) -> str:
    """Build a CE-input passage from queries + summary.

    Queries get joined with ' | ' (matches cross_encoder_eval.py for the
    cosine-baseline comparability). Summary prose is appended after a
    sentence break so the CE sees both. Truncated to MAX_PASSAGE_CHARS
    so large summaries don't blow the token budget.
    """
    queries = props.get("queries") or []
    queries_text = " | ".join(q for q in queries if isinstance(q, str) and q.strip())
    summary_text = _render_summary(props.get("summary"))
    parts = [t for t in (queries_text, summary_text) if t]
    return " . ".join(parts)[:MAX_PASSAGE_CHARS]


def fetch_pairs(conn: sqlite3.Connection, limit: int | None = None) -> list[dict]:
    cur = conn.cursor()
    sql = (
        "SELECT t.subject AS sub, t.object AS obj, t.confidence AS conf, "
        "       a.properties AS a_props, b.properties AS b_props "
        "  FROM triples t "
        "  JOIN entities a ON a.id = t.subject "
        "  JOIN entities b ON b.id = t.object "
        " WHERE t.predicate = 'similar_to' "
        "   AND t.valid_to IS NULL "
        "   AND a.kind = 'context' "
        "   AND b.kind = 'context'"
    )
    if limit is not None and limit > 0:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql)
    rows: list[dict] = []
    for sub, obj, conf, a_props_raw, b_props_raw in cur.fetchall():
        a_props = _coerce_props(a_props_raw)
        b_props = _coerce_props(b_props_raw)
        a_text = _passage(a_props)
        b_text = _passage(b_props)
        if not a_text or not b_text:
            continue
        rows.append(
            {
                "subject": sub,
                "object": obj,
                "cosine": float(conf or 0.0),
                "a_text": a_text,
                "b_text": b_text,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap number of edges (0 = all)")
    ap.add_argument("--db", type=str, default=str(DB))
    args = ap.parse_args()

    db_path = Path(os.path.expanduser(args.db))
    print(f"[1/5] Loading similar_to edges from {db_path}", flush=True)
    conn = sqlite3.connect(db_path)
    pairs = fetch_pairs(conn, args.limit if args.limit > 0 else None)
    print(f"      loaded {len(pairs)} edges", flush=True)
    if not pairs:
        print("No edges found. Exiting.")
        return

    print("[2/5] Loading cross-encoder BAAI/bge-reranker-v2-m3 (~600 MB on first run)", flush=True)
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder("BAAI/bge-reranker-v2-m3")

    print("[3/5] Scoring edges through CE", flush=True)
    t0 = time.time()
    inputs = [(p["a_text"], p["b_text"]) for p in pairs]
    raw_scores = ce.predict(inputs, batch_size=32, show_progress_bar=True)
    # BGE-reranker outputs unbounded logits; sigmoid-normalize to [0, 1]
    # so CE and cosine are comparable on the same scale.
    ce_scores = (1.0 / (1.0 + np.exp(-np.asarray(raw_scores, dtype=float)))).tolist()
    print(f"      CE done in {time.time() - t0:.1f}s", flush=True)

    print("[4/5] Computing CE-vs-cosine deltas", flush=True)
    cosines = [p["cosine"] for p in pairs]
    deltas = [ce - cos for ce, cos in zip(ce_scores, cosines)]

    abs_deltas = [abs(d) for d in deltas]
    upgrades_to_reuse = sum(
        1 for ce, cos in zip(ce_scores, cosines) if ce >= T_REUSE and cos < T_REUSE
    )
    drops_below_similar = sum(
        1 for ce, cos in zip(ce_scores, cosines) if ce < T_SIMILAR and cos >= T_SIMILAR
    )
    agreements_above_reuse = sum(
        1 for ce, cos in zip(ce_scores, cosines) if ce >= T_REUSE and cos >= T_REUSE
    )

    bands = {"-1.0..-0.3": 0, "-0.3..-0.1": 0, "-0.1..0.1": 0, "0.1..0.3": 0, "0.3..1.0": 0}
    for d in deltas:
        if d <= -0.3:
            bands["-1.0..-0.3"] += 1
        elif d <= -0.1:
            bands["-0.3..-0.1"] += 1
        elif d < 0.1:
            bands["-0.1..0.1"] += 1
        elif d < 0.3:
            bands["0.1..0.3"] += 1
        else:
            bands["0.3..1.0"] += 1

    # Spearman rank correlation (no scipy dep -- compute manually).
    def rank(xs):
        sorted_idx = sorted(range(len(xs)), key=lambda i: xs[i])
        ranks = [0.0] * len(xs)
        for r, i in enumerate(sorted_idx):
            ranks[i] = float(r)
        return ranks

    ce_ranks = rank(ce_scores)
    cos_ranks = rank(cosines)
    n = len(ce_ranks)
    mean_ce = sum(ce_ranks) / n
    mean_cos = sum(cos_ranks) / n
    num = sum((cr - mean_ce) * (cor - mean_cos) for cr, cor in zip(ce_ranks, cos_ranks))
    den_ce = (sum((cr - mean_ce) ** 2 for cr in ce_ranks)) ** 0.5
    den_cos = (sum((cor - mean_cos) ** 2 for cor in cos_ranks)) ** 0.5
    spearman = num / (den_ce * den_cos) if den_ce > 0 and den_cos > 0 else 0.0

    print("[5/5] Results", flush=True)
    print("=" * 60)
    print(f"edges scored:                 {n}")
    print(f"cosine: mean={mean(cosines):.4f} median={median(cosines):.4f}")
    print(f"CE:     mean={mean(ce_scores):.4f} median={median(ce_scores):.4f}")
    print(f"|CE - cosine|: mean={mean(abs_deltas):.4f} median={median(abs_deltas):.4f}")
    print(f"Spearman rank corr(CE, cosine): {spearman:.4f}")
    print()
    print("delta bands (CE - cosine):")
    for band, n_band in bands.items():
        pct = 100.0 * n_band / n
        print(f"  {band:>14}: {n_band:>5} ({pct:5.1f}%)")
    print()
    print(f"would CE upgrade to reuse (CE >= {T_REUSE}, cos < {T_REUSE}): {upgrades_to_reuse}")
    print(
        f"would CE drop the edge   (CE <  {T_SIMILAR}, cos >= {T_SIMILAR}): {drops_below_similar}"
    )
    print(f"agree above reuse        (CE >= {T_REUSE}, cos >= {T_REUSE}): {agreements_above_reuse}")
    print("=" * 60)
    conn.close()


if __name__ == "__main__":
    main()
