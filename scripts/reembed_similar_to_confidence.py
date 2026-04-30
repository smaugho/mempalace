"""Re-embed similar_to confidence after summary-as-view backfill.

Adrian's design lock 2026-04-30: "summary should be a view EVERYWHERE
wherever there is a summary." The production code path now appends
the rendered summary as view N+1 in mempalace_context_views (commit
ceb372d) and ``scripts/backfill_summary_views.py`` re-embedded the
4584 existing contexts to add their summary view (commit ceb372d).

But the **stored confidence on each similar_to edge** is the
``max-of-max`` link_score from the moment the edge was first written
-- BEFORE summary-as-view existed. Today's downstream consumers
(``walk_rated_neighbourhood``'s ``link_score``, scoring.py's
``get_similar_contexts``) read that stale stored value, so the
summary signal is invisible at the link layer.

This script closes the loop by recomputing each similar_to edge's
max-of-max confidence against the current (summary-aware) view
collection and updating the triple if the new score differs from
the stored value by more than ``THRESHOLD`` (default 0.001).

Mirrors production semantics exactly:
* Subject side: queries-only views (matches what
  ``context_lookup_or_create`` passes as ``views`` to
  ``multi_view_max_sim``).
* Target side: all views including the summary view N+1 (since the
  Chroma collection holds them both, max-of-max picks the best
  alignment regardless of view kind).

The asymmetric compute is intentional and matches the production
lookup path. If you want a symmetric "summary on both sides" recompute
that's a separate design call beyond structural completion.

Usage::

    python scripts/reembed_similar_to_confidence.py [--dry-run] [--limit N]

    --dry-run  count what would change without writing.
    --limit N  cap edges processed (0 = all).

Idempotent: re-running on an unchanged corpus rewrites the same
confidence values. Safe to interrupt mid-run; partial updates are
committed in batches of 100.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DB = Path(os.path.expanduser("~/.mempalace/palace/knowledge_graph.sqlite3"))
THRESHOLD = 0.001
BATCH_SIZE = 100


def _coerce_props(raw):
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _eligible_triples(conn: sqlite3.Connection, limit: int | None):
    cur = conn.cursor()
    sql = (
        "SELECT t.id AS tid, t.subject AS sub, t.object AS obj, t.confidence AS conf, "
        "       a.properties AS a_props "
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
    for tid, sub, obj, conf, a_props_raw in cur.fetchall():
        a_props = _coerce_props(a_props_raw)
        queries = a_props.get("queries") or []
        queries = [q for q in queries if isinstance(q, str) and q.strip()]
        if not queries:
            continue
        rows.append(
            {
                "triple_id": tid,
                "subject": sub,
                "object": obj,
                "stored_conf": float(conf or 0.0),
                "queries": queries,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--db", type=str, default=str(DB))
    ap.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help="minimum |new - stored| to trigger an update",
    )
    args = ap.parse_args()

    db_path = Path(os.path.expanduser(args.db))
    print(f"[1/5] Loading similar_to edges from {db_path}", flush=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = None
    eligible = _eligible_triples(conn, args.limit if args.limit > 0 else None)
    print(f"      eligible edges: {len(eligible)}", flush=True)
    if not eligible:
        print("nothing to re-embed; exiting.")
        return

    print("[2/5] Loading mempalace context-views collection", flush=True)
    from mempalace.mcp_server import _get_context_views_collection
    from mempalace.scoring import multi_view_max_sim

    col = _get_context_views_collection(create=False)
    if col is None:
        print("ERROR: context-views collection unavailable", flush=True)
        return

    print("[3/5] Re-computing max-of-max per edge", flush=True)
    t0 = time.time()
    new_scores: list[tuple[str, float, float]] = []  # (triple_id, old, new)
    failures = 0
    skipped_missing_target = 0
    for i, edge in enumerate(eligible):
        # Critical guard: only recompute when the target has Chroma records.
        # SQLite/Chroma drift can leave target contexts present in entities
        # but absent from mempalace_context_views; querying those returns
        # empty score_map which would falsely zero out stored confidence.
        # Skipping is the safe choice -- the stored value remains the best
        # available signal until the drift is repaired.
        try:
            target_check = col.get(where={"context_id": edge["object"]}, include=[])
            if not target_check.get("ids"):
                skipped_missing_target += 1
                continue
        except Exception as exc:
            failures += 1
            if failures <= 5:
                print(
                    f"  target-check failed for {edge['triple_id']}: {exc!r}",
                    flush=True,
                )
            continue
        try:
            score_map = multi_view_max_sim(
                edge["queries"],
                [edge["object"]],
                col,
                where_key="context_id",
                n_results=10,
            )
            new_score = float(score_map.get(edge["object"], 0.0) or 0.0)
        except Exception as exc:
            failures += 1
            if failures <= 5:
                print(f"  triple {edge['triple_id']} failed: {exc!r}", flush=True)
            continue
        new_scores.append((edge["triple_id"], edge["stored_conf"], new_score))
        if (i + 1) % 250 == 0:
            print(f"      {i + 1}/{len(eligible)}  ({time.time() - t0:.1f}s)", flush=True)
    print(
        f"      scored {len(new_scores)} edges in {time.time() - t0:.1f}s; "
        f"skipped (target missing from Chroma): {skipped_missing_target}; "
        f"failures: {failures}",
        flush=True,
    )

    deltas = [(tid, new - old) for tid, old, new in new_scores]
    abs_deltas = [abs(d) for _tid, d in deltas]
    pos_changes = sum(1 for _tid, d in deltas if d > args.threshold)
    neg_changes = sum(1 for _tid, d in deltas if d < -args.threshold)
    no_change = sum(1 for _tid, d in deltas if abs(d) <= args.threshold)
    will_update = pos_changes + neg_changes

    print()
    print("=" * 60)
    print(f"edges scored:                 {len(new_scores)}")
    if abs_deltas:
        print(
            f"|delta|: mean={mean(abs_deltas):.4f} median={median(abs_deltas):.4f} max={max(abs_deltas):.4f}"
        )
    print(f"new > old (above threshold):  {pos_changes}")
    print(f"new < old (below threshold):  {neg_changes}")
    print(f"|delta| <= {args.threshold}: no change: {no_change}")
    print(f"would update:                 {will_update}")
    print("=" * 60)

    if args.dry_run:
        print("[5/5] --dry-run -- no SQLite writes")
        conn.close()
        return

    print("[4/5] Updating SQLite triples.confidence in batches", flush=True)
    cur = conn.cursor()
    n_updated = 0
    pending: list[tuple[float, str]] = []
    for tid, old, new in new_scores:
        if abs(new - old) <= args.threshold:
            continue
        pending.append((round(float(new), 4), tid))
        if len(pending) >= BATCH_SIZE:
            cur.executemany("UPDATE triples SET confidence=? WHERE id=?", pending)
            conn.commit()
            n_updated += len(pending)
            pending = []
            print(f"      committed {n_updated}/{will_update}", flush=True)
    if pending:
        cur.executemany("UPDATE triples SET confidence=? WHERE id=?", pending)
        conn.commit()
        n_updated += len(pending)
    print(f"      total updated: {n_updated}/{will_update}", flush=True)
    conn.close()

    print("[5/5] Re-embed complete", flush=True)


if __name__ == "__main__":
    main()
