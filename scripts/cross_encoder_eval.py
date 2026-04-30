"""Cross-encoder vs cosine baseline on rated_useful / rated_irrelevant pairs.

Reads our rated edges from the live palace, scores each (context, memory) pair
with both the current bi-encoder (cosine over MiniLM embeddings via Chroma)
and a pre-trained cross-encoder (BAAI/bge-reranker-v2-m3), then reports AUC,
calibrated thresholds, and the size of the uncertainty band.

Run: python scripts/cross_encoder_eval.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

DB = Path(os.path.expanduser("~/.mempalace/palace/knowledge_graph.sqlite3"))
SAMPLE_LIMIT = 1500  # cap pairs per label for runtime; balanced sampling


def fetch_pairs(conn: sqlite3.Connection) -> list[dict]:
    """Pull labeled (ctx_text, memory_text, label) triples from the palace.

    Reads `entities.content` (the long-prose body); migration 023 dropped the
    earlier dual-column shape so `content` is now the single source of truth.
    """
    cur = conn.cursor()
    rows: list[dict] = []
    for pred, label in (("rated_useful", 1), ("rated_irrelevant", 0)):
        cur.execute(
            """
            SELECT t.subject, t.object,
                   ctx.properties as ctx_props, ctx.content as ctx_content,
                   mem.content as mem_content
              FROM triples t
              JOIN entities ctx ON ctx.id = t.subject
              LEFT JOIN entities mem ON mem.id = t.object
             WHERE t.predicate = ?
               AND t.valid_to IS NULL
               AND ctx.kind = 'context'
               AND mem.content IS NOT NULL
               AND mem.content <> ''
             ORDER BY RANDOM()
             LIMIT ?
            """,
            (pred, SAMPLE_LIMIT),
        )
        for sub, obj, ctx_props, ctx_content, mem_content in cur.fetchall():
            try:
                props = json.loads(ctx_props) if ctx_props else {}
            except Exception:
                props = {}
            queries = props.get("queries") or []
            ctx_text = " | ".join(queries) if queries else (ctx_content or "")
            if not ctx_text or not mem_content:
                continue
            rows.append(
                {
                    "ctx_id": sub,
                    "mem_id": obj,
                    "ctx_text": ctx_text[:600],
                    "mem_text": mem_content[:600],
                    "label": label,
                }
            )
    return rows


def cosine_score(model, ctx_text: str, mem_text: str) -> float:
    a = model.encode(ctx_text, normalize_embeddings=True)
    b = model.encode(mem_text, normalize_embeddings=True)
    return float(np.dot(a, b))


def main() -> None:
    print(f"[1/5] Loading rated pairs from {DB}", flush=True)
    conn = sqlite3.connect(DB)
    pairs = fetch_pairs(conn)
    n_pos = sum(p["label"] for p in pairs)
    n_neg = len(pairs) - n_pos
    print(f"      Loaded {len(pairs)} pairs ({n_pos} useful, {n_neg} irrelevant)", flush=True)

    print("[2/5] Loading bi-encoder all-MiniLM-L6-v2", flush=True)
    from sentence_transformers import SentenceTransformer

    bi = SentenceTransformer("all-MiniLM-L6-v2")

    print("[3/5] Scoring with cosine", flush=True)
    t0 = time.time()
    cos_scores = []
    for i, p in enumerate(pairs):
        cos_scores.append(cosine_score(bi, p["ctx_text"], p["mem_text"]))
        if (i + 1) % 200 == 0:
            print(f"      cosine {i + 1}/{len(pairs)}", flush=True)
    print(f"      cosine done in {time.time() - t0:.1f}s", flush=True)

    print(
        "[4/5] Loading cross-encoder BAAI/bge-reranker-v2-m3 (one-time download ~600MB)", flush=True
    )
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder("BAAI/bge-reranker-v2-m3")

    print("[5/5] Scoring with cross-encoder", flush=True)
    t0 = time.time()
    ce_pairs = [(p["ctx_text"], p["mem_text"]) for p in pairs]
    ce_scores = ce.predict(ce_pairs, batch_size=32, show_progress_bar=True)
    print(f"      cross-encoder done in {time.time() - t0:.1f}s", flush=True)

    labels = np.array([p["label"] for p in pairs])
    cos_arr = np.array(cos_scores)
    ce_arr = np.array(ce_scores)

    from sklearn.metrics import roc_auc_score

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  pairs               {len(pairs)}  (useful={n_pos} / irrelevant={n_neg})")
    print(f"  cosine AUC          {roc_auc_score(labels, cos_arr):.4f}")
    print(f"  cross-encoder AUC   {roc_auc_score(labels, ce_arr):.4f}")
    print()

    # Calibrate cross-encoder via isotonic regression
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(ce_arr, labels)
    cal_arr = iso.predict(ce_arr)
    print(f"  calibrated CE AUC   {roc_auc_score(labels, cal_arr):.4f}")
    print()

    # Find uncertainty band: where calibrated p ∈ (p_low, p_high) and accuracy < target
    # Simpler: report % of pairs in [0.30, 0.85], [0.40, 0.80], [0.45, 0.75]
    print("  Uncertainty band coverage (calibrated probability):")
    for lo, hi in ((0.30, 0.85), (0.40, 0.80), (0.45, 0.75)):
        in_band = ((cal_arr > lo) & (cal_arr < hi)).sum()
        # accuracy outside the band (using majority-vote per side)
        outside_high = cal_arr >= hi
        outside_low = cal_arr <= lo
        acc_high = float((labels[outside_high] == 1).mean()) if outside_high.sum() else float("nan")
        acc_low = float((labels[outside_low] == 0).mean()) if outside_low.sum() else float("nan")
        print(
            f"    band ({lo:.2f}, {hi:.2f}):  "
            f"in_band={in_band} ({100 * in_band / len(pairs):.1f}%)   "
            f"acc above hi (predict-useful)={acc_high:.3f}   "
            f"acc below lo (predict-irrelevant)={acc_low:.3f}"
        )

    # Score distributions for the labels
    print()
    print("  Cosine score distribution:")
    for lab, name in ((1, "useful"), (0, "irrelevant")):
        s = cos_arr[labels == lab]
        print(
            f"    {name:11} mean={s.mean():.3f}  median={np.median(s):.3f}  p25={np.percentile(s, 25):.3f}  p75={np.percentile(s, 75):.3f}"
        )
    print("  Cross-encoder score distribution (raw):")
    for lab, name in ((1, "useful"), (0, "irrelevant")):
        s = ce_arr[labels == lab]
        print(
            f"    {name:11} mean={s.mean():.3f}  median={np.median(s):.3f}  p25={np.percentile(s, 25):.3f}  p75={np.percentile(s, 75):.3f}"
        )
    print("  Calibrated CE distribution:")
    for lab, name in ((1, "useful"), (0, "irrelevant")):
        s = cal_arr[labels == lab]
        print(
            f"    {name:11} mean={s.mean():.3f}  median={np.median(s):.3f}  p25={np.percentile(s, 25):.3f}  p75={np.percentile(s, 75):.3f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())
