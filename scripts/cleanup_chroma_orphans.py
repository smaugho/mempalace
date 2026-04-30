"""Cleanup Chroma records whose SQLite entity row is gone (orphans).

Background: today's Chroma+SQLite drift audit found ~883 records in
``mempalace_context_views`` whose ``context_id`` metadata points at an
entity that no longer exists in the SQLite ``entities`` table. These
orphans are unrecoverable (no source row to re-derive views from), waste
storage, and may pollute retrieval if Chroma returns them on a where-
filter that doesn't include the orphan check.

This script:
  1. Collects all distinct ``context_id`` metadata values from
     ``mempalace_context_views``.
  2. Diffs against ``SELECT id FROM entities`` (all kinds, not just
     context -- a few orphans may have been other kinds upstream).
  3. For every orphan ``cid``, deletes the matching Chroma records via
     ``col.delete(where={'context_id': cid})``.

Same logic applied to the entity multi-view collection
``mempalace_records_entities`` (whatever the production helper returns
from ``_get_entity_collection``) using the ``entity_id`` metadata key.

Idempotent. Safe to interrupt mid-run -- Chroma deletes are atomic per
where-filter call. Supports ``--dry-run`` and ``--limit``.

Usage::

    python scripts/cleanup_chroma_orphans.py [--dry-run] [--limit N]
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DB = Path(os.path.expanduser("~/.mempalace/palace/knowledge_graph.sqlite3"))


def _all_sqlite_ids(conn: sqlite3.Connection) -> set[str]:
    cur = conn.cursor()
    cur.execute("SELECT id FROM entities")
    return {row[0] for row in cur.fetchall()}


def _distinct_meta_values(col, key: str) -> set[str]:
    """Return the set of distinct values for a metadata key in the collection."""
    if col is None:
        return set()
    out: set[str] = set()
    raw = col.get(include=["metadatas"])
    for m in raw.get("metadatas") or []:
        if not isinstance(m, dict):
            continue
        v = m.get(key)
        if isinstance(v, str) and v:
            out.add(v)
    return out


def _delete_orphans(col, key: str, orphans: set[str], dry_run: bool) -> int:
    if not orphans:
        return 0
    if dry_run:
        return len(orphans)
    n_done = 0
    for cid in sorted(orphans):
        try:
            col.delete(where={key: cid})
            n_done += 1
        except Exception as exc:
            print(f"  delete failed for {cid}: {exc!r}", flush=True)
    return n_done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--db", type=str, default=str(DB))
    args = ap.parse_args()

    db_path = Path(os.path.expanduser(args.db))
    print(f"[1/4] Loading SQLite ids from {db_path}", flush=True)
    conn = sqlite3.connect(db_path)
    sqlite_ids = _all_sqlite_ids(conn)
    print(f"      total entity rows in SQLite: {len(sqlite_ids)}", flush=True)

    print("[2/4] Loading context-views Chroma collection", flush=True)
    # Only context_views is touched here. _get_entity_collection triggers
    # _migrate_entity_views_schema on first access, which segfaults under
    # Chroma-contention with a running MCP server (Windows local_hnsw
    # fatal exception). Entity-collection orphan cleanup needs MCP server
    # offline OR a restart-aware schema; defer to a separate pass.
    from mempalace.mcp_server import _get_context_views_collection

    ctx_col = _get_context_views_collection(create=False)

    print("[3/4] Diffing context_views vs SQLite", flush=True)
    t0 = time.time()
    ctx_chroma_cids = _distinct_meta_values(ctx_col, "context_id")
    ctx_orphans = ctx_chroma_cids - sqlite_ids
    if args.limit > 0:
        ctx_orphans = set(sorted(ctx_orphans)[: args.limit])
    print(
        f"      context_views: {len(ctx_chroma_cids)} distinct cids, "
        f"{len(ctx_orphans)} orphans ({time.time() - t0:.1f}s)",
        flush=True,
    )

    if args.dry_run:
        print()
        print("[4/4] --dry-run -- no Chroma writes")
        print(f"      would delete: {len(ctx_orphans)} context orphans")
        return

    print()
    print("[4/4] Deleting orphans", flush=True)
    t0 = time.time()
    n_ctx_deleted = _delete_orphans(ctx_col, "context_id", ctx_orphans, dry_run=False)
    print(
        f"      context orphans deleted: {n_ctx_deleted}/{len(ctx_orphans)} in {time.time() - t0:.1f}s"
    )
    print()
    print(f"Total: {n_ctx_deleted} context orphans deleted.")
    print(
        "Note: entity-views orphans are NOT touched here -- run separately when MCP server can be restarted."
    )
    conn.close()


if __name__ == "__main__":
    main()
