"""
repair.py -- Scan, prune corrupt entries, and rebuild HNSW index
================================================================

When ChromaDB's HNSW index accumulates duplicate entries (from repeated
add() calls with the same ID), link_lists.bin can grow unbounded --
terabytes on large palaces -- eventually causing segfaults.

This module provides three operations:

  scan    -- find every corrupt/unfetchable ID in the palace
  prune   -- delete only the corrupt IDs (surgical)
  rebuild -- extract all memories, delete the collection, recreate with
            correct HNSW settings, and upsert everything back

The rebuild backs up ONLY chroma.sqlite3 (the source of truth), not the
full palace directory -- so it works even when link_lists.bin is bloated.

Usage (standalone):
    python -m mempalace.repair scan
    python -m mempalace.repair prune --confirm
    python -m mempalace.repair rebuild

Usage (from CLI):
    mempalace repair
    mempalace repair-scan
    mempalace repair-prune --confirm
"""

import argparse
import os
import shutil
import time

import chromadb


COLLECTION_NAME = "mempalace_records"


def _get_palace_path():
    """Resolve palace path from config."""
    try:
        from .config import MempalaceConfig

        return MempalaceConfig().palace_path
    except Exception:
        default = os.path.join(os.path.expanduser("~"), ".mempalace", "palace")
        return default


def _paginate_ids(col, where=None):
    """Pull all IDs in a collection using pagination."""
    ids = []
    page = 1000
    offset = 0
    while True:
        try:
            r = col.get(where=where, include=[], limit=page, offset=offset)
        except Exception:
            try:
                r = col.get(where=where, include=[], limit=page)
                new_ids = [i for i in r["ids"] if i not in set(ids)]
                if not new_ids:
                    break
                ids.extend(new_ids)
                offset += len(new_ids)
                continue
            except Exception:
                break
        n = len(r["ids"]) if r["ids"] else 0
        if n == 0:
            break
        ids.extend(r["ids"])
        offset += n
        if n < page:
            break
    return ids


def scan_palace(palace_path=None):
    """Scan the palace for corrupt/unfetchable IDs.

    Probes in batches of 100, falls back to per-ID on failure.
    Writes corrupt_ids.txt to the palace directory for the prune step.

    Returns (good_set, bad_set).
    """
    palace_path = palace_path or _get_palace_path()
    print(f"\n  Palace: {palace_path}")
    print("  Loading...")

    client = chromadb.PersistentClient(path=palace_path)
    col = client.get_collection(COLLECTION_NAME)

    total = col.count()
    print(f"  Collection: {COLLECTION_NAME}, total: {total:,}")

    print("\n  Step 1: listing all IDs...")
    t0 = time.time()
    all_ids = _paginate_ids(col)
    print(f"  Found {len(all_ids):,} IDs in {time.time() - t0:.1f}s\n")

    if not all_ids:
        print("  Nothing to scan.")
        return set(), set()

    print("  Step 2: probing each ID (batches of 100)...")
    t0 = time.time()
    good_set = set()
    bad_set = set()
    batch = 100

    for i in range(0, len(all_ids), batch):
        chunk = all_ids[i : i + batch]
        try:
            r = col.get(ids=chunk, include=["documents"])
            for got in r["ids"]:
                good_set.add(got)
            for mid in chunk:
                if mid not in good_set:
                    bad_set.add(mid)
        except Exception:
            for sid in chunk:
                try:
                    r = col.get(ids=[sid], include=["documents"])
                    if r["ids"]:
                        good_set.add(sid)
                    else:
                        bad_set.add(sid)
                except Exception:
                    bad_set.add(sid)

        if (i // batch) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + batch) / max(elapsed, 0.01)
            eta = (len(all_ids) - i - batch) / max(rate, 0.01)
            print(
                f"    {i + batch:>6}/{len(all_ids):>6}  "
                f"good={len(good_set):>6}  bad={len(bad_set):>6}  "
                f"eta={eta:.0f}s"
            )

    print(f"\n  Scan complete in {time.time() - t0:.1f}s")
    print(f"  GOOD: {len(good_set):,}")
    print(f"  BAD:  {len(bad_set):,}  ({len(bad_set) / max(len(all_ids), 1) * 100:.1f}%)")

    bad_file = os.path.join(palace_path, "corrupt_ids.txt")
    with open(bad_file, "w") as f:
        for bid in sorted(bad_set):
            f.write(bid + "\n")
    print(f"\n  Bad IDs written to: {bad_file}")
    return good_set, bad_set


def prune_corrupt(palace_path=None, confirm=False):
    """Delete corrupt IDs listed in corrupt_ids.txt."""
    palace_path = palace_path or _get_palace_path()
    bad_file = os.path.join(palace_path, "corrupt_ids.txt")

    if not os.path.exists(bad_file):
        print("  No corrupt_ids.txt found -- run scan first.")
        return

    with open(bad_file) as f:
        bad_ids = [line.strip() for line in f if line.strip()]
    print(f"  {len(bad_ids):,} corrupt IDs queued for deletion")

    if not confirm:
        print("\n  DRY RUN -- no deletions performed.")
        print("  Re-run with --confirm to actually delete.")
        return

    client = chromadb.PersistentClient(path=palace_path)
    col = client.get_collection(COLLECTION_NAME)
    before = col.count()
    print(f"  Collection size before: {before:,}")

    batch = 100
    deleted = 0
    failed = 0
    for i in range(0, len(bad_ids), batch):
        chunk = bad_ids[i : i + batch]
        try:
            col.delete(ids=chunk)
            deleted += len(chunk)
        except Exception:
            for sid in chunk:
                try:
                    col.delete(ids=[sid])
                    deleted += 1
                except Exception:
                    failed += 1
        if (i // batch) % 20 == 0:
            print(f"    deleted {deleted}/{len(bad_ids)}  (failed: {failed})")

    after = col.count()
    print(f"\n  Deleted: {deleted:,}")
    print(f"  Failed:  {failed:,}")
    print(f"  Collection size: {before:,} → {after:,}")


def _rebuild_one_collection(client, name: str, batch_size: int = 5000) -> int:
    """Rebuild a single collection. Returns count rebuilt. Used by
    rebuild_index for the all-collections sweep.

    Extracts via col.get() (read-only on SQLite, doesn't trigger HNSW
    init), then delete_collection + create_collection (clean HNSW
    files), then re-upsert in batches.
    """
    try:
        col = client.get_collection(name)
    except Exception as e:
        print(f"    {name}: not present or unreadable ({e}); skipping")
        return 0
    # Slice 15 (Adrian directive 2026-05-05): use col.get() to extract
    # rows, NOT col.count() which only reports HNSW-committed entries.
    # On a poisoned palace where the queue never flushed, col.count()
    # returns 0 while col.get() correctly reads from the metadata
    # segment (SQLite). The previous total = col.count() / if total==0
    # path silently skipped collections that needed rebuilding most.
    print(f"    {name}: extracting rows via metadata segment...")
    all_ids: list = []
    all_docs: list = []
    all_metas: list = []
    offset = 0
    while True:
        batch = col.get(limit=batch_size, offset=offset, include=["documents", "metadatas"])
        if not batch["ids"]:
            break
        all_ids.extend(batch["ids"])
        all_docs.extend(batch["documents"])
        all_metas.extend(batch["metadatas"])
        offset += len(batch["ids"])
        # Safety: abort runaway loops on a corrupt collection.
        if offset > 10_000_000:
            break
    total = len(all_ids)
    if total == 0:
        print(f"    {name}: empty, skipping")
        return 0
    print(f"    {name}: extracted {total} rows from metadata; rebuilding...")
    client.delete_collection(name)
    new_col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    filed = 0
    for i in range(0, len(all_ids), batch_size):
        new_col.upsert(
            ids=all_ids[i : i + batch_size],
            documents=all_docs[i : i + batch_size],
            metadatas=all_metas[i : i + batch_size],
        )
        filed += min(batch_size, len(all_ids) - i)
    print(f"    {name}: rebuilt {new_col.count()}/{total} rows")
    return new_col.count()


def rebuild_index(palace_path=None):
    """Rebuild the HNSW indices for ALL Chroma collections from scratch.

    The mempalace palace has THREE Chroma collections (mempalace_records
    for memories + entities, mempalace_context_views for retrieval
    contexts, mempalace_triples for KG edge statements). HNSW corruption
    can hit any of them independently -- rebuilding only mempalace_records
    leaves the others segfaulting on first query (Adrian 2026-05-04
    crash post-marathon-session reproduced this; the segfault was inside
    mempalace_context_views which the prior rebuild_index never touched).

    Steps:
      1. Extract all rows from each collection via col.get() (read-only
         SQLite reads, doesn't trigger HNSW init -- safe even when the
         index is corrupt).
      2. Back up ONLY chroma.sqlite3 (not the bloated HNSW files).
      3. delete_collection + create_collection per collection -- clears
         the HNSW link_lists.bin / data_level0.bin files.
      4. Re-upsert all rows.
    """
    palace_path = palace_path or _get_palace_path()

    if not os.path.isdir(palace_path):
        print(f"\n  No palace found at {palace_path}")
        return

    print(f"\n{'=' * 55}")
    print("  MemPalace Repair -- Index Rebuild (all collections)")
    print(f"{'=' * 55}\n")
    print(f"  Palace: {palace_path}")

    client = chromadb.PersistentClient(path=palace_path)

    # Back up ONLY the SQLite database once, before any rebuild touches.
    sqlite_path = os.path.join(palace_path, "chroma.sqlite3")
    if os.path.exists(sqlite_path):
        backup_path = sqlite_path + ".backup"
        print(f"  Backing up chroma.sqlite3 ({os.path.getsize(sqlite_path) / 1e6:.0f} MB)...")
        shutil.copy2(sqlite_path, backup_path)
        print(f"  Backup: {backup_path}\n")

    # Rebuild every known collection. Order doesn't matter -- each is
    # independent. Iterate the canonical set rather than
    # client.list_collections() so we get deterministic coverage even
    # if Chroma's listing API drifts (Chroma 0.6 changed the return
    # shape from objects to bare names).
    collections = [
        "mempalace_records",
        "mempalace_context_views",
        "mempalace_triples",
    ]
    total_rebuilt = 0
    print("  Rebuilding HNSW indices...")
    for name in collections:
        total_rebuilt += _rebuild_one_collection(client, name)

    print(
        f"\n  Repair complete. {total_rebuilt} rows rebuilt across {len(collections)} collections."
    )
    print("  HNSW indices are now clean with cosine distance metric.")
    print(f"\n{'=' * 55}\n")


# ── HNSW health check (slice 13, Adrian directive 2026-05-05) ──────────
# C-level segfault inside chromadb local_hnsw._apply_batch keeps
# recurring during context_lookup_or_create writes (observed twice in
# 24h: 2026-05-04 marathon-end, 2026-05-05 fresh-install). Python
# try/except cannot catch a SIGSEGV, so reactive recovery isn't an
# option -- by the time the crash surfaces, the MCP transport is dead
# and the agent surface is hard-locked because every fallback path
# (declare_user_intents → context mint → segfault) re-triggers the
# same crash. The fix is preventive: detect corruption signals on the
# filesystem BEFORE Chroma loads the index, and rebuild the suspect
# collection while the process is still healthy.
#
# Heuristic: HNSW link_lists.bin size relative to expected row count.
# A healthy collection's link_lists scales roughly with row_count *
# avg_neighbour_density (~50-200 bytes per row at default M=16,
# ef_construction=100). At 10K rows, expect 1-20 MB. Real corruption
# observed grew link_lists to gigabytes -- duplicate add() calls with
# same id accumulate phantom adjacency entries. Threshold of 500 MB
# absolute (with explicit row-count override) is conservative: no
# legitimate palace under 1M rows should approach this.

# Anything over this size is presumed corrupt regardless of row count.
# Real palaces in the 10K-100K row range produce <100 MB link_lists.
HNSW_LINK_LISTS_HARD_LIMIT_BYTES = 500 * 1024 * 1024  # 500 MB


def _hnsw_files_for_collection(palace_path: str, collection_name: str) -> list:
    """Locate the on-disk HNSW files for a collection.

    Chroma 0.6 stores HNSW under ``<palace>/<vector_segment_uuid>/``
    (NOT the collection uuid -- the per-collection directory is named
    after the VECTOR segment, of which there is one per collection).
    The mapping is two hops: collections.id → segments.collection
    where segments.scope='VECTOR' → segments.id is the dir name. Both
    hops via read-only SQLite so this stays safe when the HNSW index
    itself is corrupt.
    """
    import sqlite3 as _sqlite

    sqlite_path = os.path.join(palace_path, "chroma.sqlite3")
    if not os.path.exists(sqlite_path):
        return []
    try:
        conn = _sqlite.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT id FROM collections WHERE name = ?",
                (collection_name,),
            ).fetchone()
            if not row:
                return []
            coll_uuid = row[0]
            seg_row = conn.execute(
                "SELECT id FROM segments WHERE collection = ? AND scope = 'VECTOR' LIMIT 1",
                (coll_uuid,),
            ).fetchone()
        finally:
            conn.close()
    except Exception:
        return []
    if not seg_row:
        return []
    seg_uuid = seg_row[0]
    seg_dir = os.path.join(palace_path, seg_uuid)
    if not os.path.isdir(seg_dir):
        return []
    return [
        os.path.join(seg_dir, fname)
        for fname in ("link_lists.bin", "data_level0.bin", "header.bin", "length.bin")
        if os.path.exists(os.path.join(seg_dir, fname))
    ]


def _queue_lag_for_collection(palace_path: str, collection_name: str) -> dict:
    """Detect ``embeddings_queue`` rows beyond the per-segment watermark.

    Slice 15 (Adrian directive 2026-05-05 after the C-level segfault
    recurred on a palace with healthy link_lists.bin). Chroma 0.6.3
    persists every collection write into the ``embeddings_queue``
    SQLite table; the HNSW segment maintains a watermark in
    ``max_seq_id``. When a session crashes mid-backfill (or exits
    before the next 1000-row sync), the watermark doesn't advance and
    the queue accumulates ``unprocessed`` rows. The next session's
    first ``col.query()`` triggers ``_backfill`` which replays those
    rows into HNSW via ``_apply_batch``; on Windows + Python 3.13 +
    Chroma 0.6.3 this consistently crashes with a SIGSEGV inside the
    C-level HNSW append code.

    Returns ``{"queue_max": int, "watermark": int, "lag": int}`` --
    ``lag = max(0, queue_max - watermark)``. A non-zero lag is the
    actual signal of corruption-prone state, far more reliable than
    the link_lists.bin size heuristic which only catches one of the
    crash patterns.

    Read-only; uses ``mode=ro`` SQLite access. Returns zeros for
    every field when the collection is unknown or the metadata tables
    are absent (older Chroma versions).
    """
    import sqlite3 as _sqlite
    import struct

    sqlite_path = os.path.join(palace_path, "chroma.sqlite3")
    if not os.path.exists(sqlite_path):
        return {"queue_max": 0, "watermark": 0, "lag": 0}
    try:
        conn = _sqlite.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        try:
            coll_row = conn.execute(
                "SELECT id FROM collections WHERE name = ?",
                (collection_name,),
            ).fetchone()
            if not coll_row:
                return {"queue_max": 0, "watermark": 0, "lag": 0}
            coll_uuid = coll_row[0]
            topic = f"persistent://default/default/{coll_uuid}"
            qmax_row = conn.execute(
                "SELECT MAX(seq_id) FROM embeddings_queue WHERE topic = ?",
                (topic,),
            ).fetchone()
            queue_max = int(qmax_row[0]) if qmax_row and qmax_row[0] is not None else 0

            seg_row = conn.execute(
                "SELECT id FROM segments WHERE collection = ? AND scope = 'VECTOR' LIMIT 1",
                (coll_uuid,),
            ).fetchone()
            watermark = 0
            if seg_row:
                wm_row = conn.execute(
                    "SELECT seq_id FROM max_seq_id WHERE segment_id = ?",
                    (seg_row[0],),
                ).fetchone()
                if wm_row and wm_row[0] is not None:
                    raw = wm_row[0]
                    if isinstance(raw, bytes):
                        watermark = struct.unpack(">Q", raw.ljust(8, b"\x00")[:8])[0]
                    else:
                        watermark = int(raw)
        finally:
            conn.close()
    except Exception:
        return {"queue_max": 0, "watermark": 0, "lag": 0}
    return {
        "queue_max": queue_max,
        "watermark": watermark,
        "lag": max(0, queue_max - watermark),
    }


def health_check(palace_path: str = None) -> dict:
    """Inspect each Chroma collection's HNSW files for corruption signals.

    Pure-filesystem + read-only-SQLite heuristic; does NOT load the
    HNSW index, so it is safe to run even when the index would
    segfault on query. Returns
    ``{collection_name: {"status": ..., "row_count": int,
    "link_lists_bytes": int, "queue_lag": int, "reason": str}}``
    where status is one of:
      ``"healthy"``     -- HNSW files exist and sizes are reasonable
                           AND queue lag is zero.
      ``"empty"``       -- collection has 0 rows; nothing to check.
      ``"oversized"``   -- link_lists.bin exceeds the hard limit;
                           rebuild recommended.
      ``"queue_lag"``   -- watermark < queue max; backfill replay
                           would risk a SIGSEGV. Rebuild required.
      ``"missing"``     -- collection registered in sqlite but no
                           HNSW files on disk (orphan; rare).

    Used standalone via ``python -m mempalace.repair check`` and
    automatically by ``auto_repair_if_needed`` at MCP server startup.
    """
    palace_path = palace_path or _get_palace_path()
    if not os.path.isdir(palace_path):
        return {}

    import sqlite3 as _sqlite

    sqlite_path = os.path.join(palace_path, "chroma.sqlite3")
    if not os.path.exists(sqlite_path):
        return {}

    results: dict = {}
    collections = ["mempalace_records", "mempalace_context_views", "mempalace_triples"]
    for name in collections:
        # Row count from Chroma's segment_metadata (read-only SQL, no HNSW).
        row_count = 0
        try:
            conn = _sqlite.connect(f"file:{sqlite_path}?mode=ro", uri=True)
            try:
                row = conn.execute(
                    "SELECT id FROM collections WHERE name = ?",
                    (name,),
                ).fetchone()
                if row:
                    coll_id = row[0]
                    cnt_row = conn.execute(
                        "SELECT COUNT(*) FROM embeddings WHERE segment_id IN "
                        "(SELECT id FROM segments WHERE collection = ?)",
                        (coll_id,),
                    ).fetchone()
                    row_count = int(cnt_row[0]) if cnt_row else 0
            finally:
                conn.close()
        except Exception:
            row_count = 0

        files = _hnsw_files_for_collection(palace_path, name)
        link_lists_bytes = 0
        for f in files:
            if f.endswith("link_lists.bin"):
                try:
                    link_lists_bytes = os.path.getsize(f)
                except OSError:
                    link_lists_bytes = 0
                break

        # Slice 15: queue-lag detection. Watermark < queue_max means
        # a previous session left unprocessed rows in
        # embeddings_queue; the next col.query() will _backfill them
        # via _apply_batch and may segfault. Detected via read-only
        # SQLite, no HNSW load.
        queue_info = _queue_lag_for_collection(palace_path, name)
        queue_lag = queue_info["lag"]

        if row_count == 0:
            status = "empty"
            reason = "no rows; nothing to validate"
        elif not files:
            status = "missing"
            reason = "collection registered but no HNSW files on disk"
        elif link_lists_bytes > HNSW_LINK_LISTS_HARD_LIMIT_BYTES:
            status = "oversized"
            reason = (
                f"link_lists.bin is {link_lists_bytes / 1e6:.0f} MB for {row_count:,} rows "
                f"(hard limit {HNSW_LINK_LISTS_HARD_LIMIT_BYTES / 1e6:.0f} MB); "
                "duplicate-add corruption likely"
            )
        elif queue_lag > 0 and queue_info["watermark"] > 0:
            # Slice 15: corruption signal is "watermark started
            # advancing then stopped" -- watermark > 0 AND lag > 0.
            # A fresh collection legitimately has watermark=0 with
            # queue_max=N until the first backfill catches up, so
            # don't flag that case (no prior session was partially
            # in-flight; the queue rows are valid first-write state).
            status = "queue_lag"
            reason = (
                f"embeddings_queue has {queue_lag} unprocessed row(s) "
                f"(queue_max={queue_info['queue_max']}, "
                f"watermark={queue_info['watermark']}); a prior session "
                "partially processed then crashed -- next backfill "
                "may SIGSEGV in HNSW _apply_batch, rebuild required"
            )
        else:
            status = "healthy"
            reason = f"{link_lists_bytes / 1e6:.1f} MB for {row_count:,} rows, queue in sync"

        results[name] = {
            "status": status,
            "row_count": row_count,
            "link_lists_bytes": link_lists_bytes,
            "queue_lag": queue_lag,
            "reason": reason,
        }
    return results


def auto_repair_if_needed(palace_path: str = None, *, verbose: bool = True) -> int:
    """Run health_check; rebuild any oversized collections in place.

    Returns the number of collections rebuilt. Designed to run at MCP
    server startup behind ``MEMPALACE_AUTO_REPAIR=1`` so a corrupted
    palace heals itself before the first context_lookup_or_create
    query has a chance to segfault. Idempotent: a healthy palace
    incurs only the filesystem-stat cost (<1ms).

    Stays silent on healthy palaces (verbose still prints the per-
    collection summary line); prints a loud rebuild banner when
    rebuild_index runs so the user notices in the MCP log.
    """
    palace_path = palace_path or _get_palace_path()
    report = health_check(palace_path)
    if not report:
        return 0

    # Slice 15: rebuild on EITHER oversized link_lists OR queue lag.
    # Queue lag is the more common production trigger (any session
    # that crashes mid-backfill leaves unprocessed rows that the next
    # session will replay -- and crash again). Link-lists oversize
    # catches the slower long-term accumulation.
    suspect = [
        name for name, info in report.items() if info["status"] in ("oversized", "queue_lag")
    ]
    if verbose:
        for name, info in report.items():
            print(f"  [repair check] {name}: {info['status']} ({info['reason']})")

    if not suspect:
        return 0

    # Rebuild touches all collections in one pass (the existing
    # rebuild_index does that for free); a per-collection rebuild
    # path would duplicate too much logic. Cost difference is small
    # because healthy collections rebuild fast (low row counts).
    if verbose:
        print(
            f"\n  [repair check] AUTO-REPAIR triggered: {len(suspect)} oversized "
            f"collection(s) detected ({', '.join(suspect)}). Running rebuild_index..."
        )
    try:
        rebuild_index(palace_path=palace_path)
    except Exception as exc:
        if verbose:
            print(f"  [repair check] auto_repair failed: {exc}")
        return 0
    return len(suspect)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MemPalace repair tools")
    p.add_argument("command", choices=["scan", "prune", "rebuild", "check", "auto"])
    p.add_argument("--palace", default=None, help="Palace directory path")
    p.add_argument("--confirm", action="store_true", help="Actually delete corrupt IDs")
    args = p.parse_args()

    path = os.path.expanduser(args.palace) if args.palace else None

    if args.command == "scan":
        scan_palace(palace_path=path)
    elif args.command == "prune":
        prune_corrupt(palace_path=path, confirm=args.confirm)
    elif args.command == "rebuild":
        rebuild_index(palace_path=path)
    elif args.command == "check":
        report = health_check(palace_path=path)
        if not report:
            print("  No palace found.")
        else:
            print(f"\n{'=' * 55}")
            print("  MemPalace HNSW Health Check")
            print(f"{'=' * 55}")
            for name, info in report.items():
                print(f"\n  {name}: {info['status']}")
                print(f"    rows: {info['row_count']:,}")
                print(f"    link_lists.bin: {info['link_lists_bytes'] / 1e6:.1f} MB")
                print(f"    {info['reason']}")
            suspect = [n for n, i in report.items() if i["status"] == "oversized"]
            if suspect:
                print(
                    f"\n  Recommendation: run 'python -m mempalace.repair rebuild' "
                    f"to fix {len(suspect)} oversized collection(s)."
                )
            print(f"\n{'=' * 55}\n")
    elif args.command == "auto":
        n = auto_repair_if_needed(palace_path=path)
        print(f"\n  auto_repair: {n} collection(s) rebuilt.\n")
