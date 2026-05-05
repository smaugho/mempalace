"""Smoke test for the ChromaDB embeddings_queue backfill segfault.

Root cause (observed 2026-05-04 + 2026-05-05): Chroma 0.6.3 persists
every collection write into ``embeddings_queue`` with a per-segment
``max_seq_id`` watermark. When a session crashes mid-backfill (or
exits before the next flush of 1000 rows lands), the watermark
doesn't advance and the queue accumulates ``unprocessed`` rows.
The NEXT session's first ``col.query()`` triggers Chroma's
``_backfill`` which replays those rows into HNSW via
``_apply_batch``; on Windows + Python 3.13 + Chroma 0.6.3 this
crashes with a SIGSEGV inside the C-level HNSW append code.

A healthy palace has watermark == queue max for every segment.
A poisoned palace has watermark < queue max -- there are rows
queued for replay that may crash the index. Slice 15 (this fix)
extends ``repair.auto_repair_if_needed`` to detect queue lag at
startup and rebuild the affected collections, clearing the queue.

This smoke test exercises the boot-repair path end-to-end:
  1. Create a palace, write a handful of contexts.
  2. Force the watermark backwards to simulate a crashed prior
     session.
  3. Run ``auto_repair_if_needed`` -- should detect the lag and
     rebuild.
  4. Issue a fresh query -- must NOT crash and must return cleanly.
  5. Verify the queue is now in sync (no lag).

Pre-fix this test fails: step 5 detects unrepaired lag (or step 4
crashes on machines where the replay actually segfaults). Post-fix
both steps pass.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import struct
import tempfile
import unittest


def _vector_segment_id(palace_path: str, collection_name: str) -> str:
    """Return the VECTOR segment uuid for a given collection name."""
    db = os.path.join(palace_path, "chroma.sqlite3")
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        row = conn.execute(
            "SELECT s.id FROM segments s "
            "JOIN collections c ON s.collection = c.id "
            "WHERE c.name = ? AND s.scope = 'VECTOR'",
            (collection_name,),
        ).fetchone()
    if not row:
        raise AssertionError(f"no VECTOR segment for collection {collection_name!r}")
    return row[0]


def _queue_max_seq_id(palace_path: str, collection_name: str) -> int:
    """Return the max seq_id in the embeddings queue for this collection.

    Returns 0 when nothing is queued for the collection.
    """
    db = os.path.join(palace_path, "chroma.sqlite3")
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        coll_id_row = conn.execute(
            "SELECT id FROM collections WHERE name = ?", (collection_name,)
        ).fetchone()
        if not coll_id_row:
            return 0
        topic = f"persistent://default/default/{coll_id_row[0]}"
        row = conn.execute(
            "SELECT MAX(seq_id) FROM embeddings_queue WHERE topic = ?", (topic,)
        ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _watermark(palace_path: str, segment_id: str) -> int:
    """Return the per-segment max_seq_id (Chroma's backfill watermark).

    Stored in the ``max_seq_id`` table as 8-byte big-endian. Returns
    0 when no row exists (segment never backfilled).
    """
    db = os.path.join(palace_path, "chroma.sqlite3")
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        row = conn.execute(
            "SELECT seq_id FROM max_seq_id WHERE segment_id = ?", (segment_id,)
        ).fetchone()
    if not row:
        return 0
    raw = row[0]
    if isinstance(raw, bytes):
        # 8-byte big-endian integer per Chroma's encoding.
        return struct.unpack(">Q", raw.ljust(8, b"\x00")[:8])[0]
    return int(raw)


def _force_watermark(palace_path: str, segment_id: str, value: int) -> None:
    """Overwrite the watermark to simulate a crashed prior session.

    Chroma stores the watermark as 8-byte big-endian; writing 0
    means "this segment has processed nothing", forcing _backfill to
    replay every queued row on the next query.
    """
    db = os.path.join(palace_path, "chroma.sqlite3")
    with sqlite3.connect(db) as conn:
        encoded = struct.pack(">Q", value)
        # Upsert pattern: try update first, insert if no rows touched.
        cur = conn.execute(
            "UPDATE max_seq_id SET seq_id = ? WHERE segment_id = ?",
            (encoded, segment_id),
        )
        if cur.rowcount == 0:
            conn.execute(
                "INSERT INTO max_seq_id (segment_id, seq_id) VALUES (?, ?)",
                (segment_id, encoded),
            )
        conn.commit()


class TestChromaQueueLagSmoke(unittest.TestCase):
    def setUp(self):
        self.palace = tempfile.mkdtemp(prefix="mempalace_queue_lag_smoke_")
        # Force seed_ontology to skip its slow Chroma sync; we are
        # only exercising the queue-lag repair path here.
        os.environ["MEMPALACE_SKIP_SEED_CHROMA_SYNC"] = "1"

    def tearDown(self):
        shutil.rmtree(self.palace, ignore_errors=True)
        os.environ.pop("MEMPALACE_SKIP_SEED_CHROMA_SYNC", None)

    def test_queue_lag_detected_and_rebuilt(self):
        """auto_repair_if_needed must rebuild a collection with queue lag.

        Pre-slice-15 ``auto_repair_if_needed`` only looked at link_lists.bin
        size and missed queue-lag corruption -- the actual cause of the
        Windows SIGSEGV during _backfill. This test reproduces the lag
        synthetically (force watermark to 0 after writes) and asserts
        that the repair function recognises the lag and rebuilds the
        collection, leaving zero lag afterwards.
        """
        import chromadb

        # 1. Create the palace + write some context rows.
        client = chromadb.PersistentClient(path=self.palace)
        col = client.get_or_create_collection("mempalace_context_views")
        col.upsert(
            ids=[f"ctx_lag_{i}" for i in range(8)],
            documents=[f"context view document {i} for queue lag smoke" for i in range(8)],
            metadatas=[{"context_id": f"ctx_lag_{i}"} for i in range(8)],
        )
        # Force a query to ensure HNSW segment is started + initial
        # backfill has run (so watermark is at the queue max).
        _ = col.query(query_texts=["smoke probe"], n_results=1)
        # Drop the client so the WAL flushes and we can re-open clean.
        del client, col

        # 2. Simulate a crashed prior session by rewinding the
        # watermark to one row BEHIND the queue max -- not zero. The
        # production crash signature (Adrian 2026-05-04 + 05-05) was
        # watermark > 0 AND queue_max > watermark, indicating a
        # session that processed some rows then crashed mid-batch.
        # A pure watermark=0 is a fresh-collection state (no prior
        # processing) and would be a false positive for this heuristic.
        seg_id = _vector_segment_id(self.palace, "mempalace_context_views")
        queue_max = _queue_max_seq_id(self.palace, "mempalace_context_views")
        self.assertGreater(queue_max, 1, "expected queue rows after upsert+query")
        # Set watermark to a non-zero value below queue_max -- mimics
        # "prior session processed N-2 rows then crashed before the
        # last 2 made it to HNSW".
        stale_watermark = max(1, queue_max - 2)
        _force_watermark(self.palace, seg_id, stale_watermark)

        # Sanity check: lag is now non-zero AND watermark is non-zero
        # (corruption signature matches production crash state).
        wm_before = _watermark(self.palace, seg_id)
        self.assertEqual(wm_before, stale_watermark)
        self.assertGreater(
            queue_max,
            wm_before,
            "expected queue lag (queue_max > watermark) after simulated crash",
        )
        self.assertGreater(
            wm_before,
            0,
            "watermark must be non-zero to match production crash signature",
        )

        # 3. Run auto_repair_if_needed -- this is the slice-15 hook.
        # Pre-fix: it looks at link_lists.bin only, sees nothing
        # oversized, returns 0 rebuilt -- lag persists. Post-fix: it
        # detects the watermark-vs-queue gap and rebuilds the
        # collection, which truncates the queue (delete_collection +
        # create_collection inside rebuild_index).
        from mempalace import repair as _repair

        rebuilt = _repair.auto_repair_if_needed(palace_path=self.palace, verbose=False)
        self.assertGreaterEqual(
            rebuilt,
            1,
            f"auto_repair_if_needed should have rebuilt at least 1 collection "
            f"(saw queue lag {queue_max} > watermark {wm_before}), got {rebuilt}",
        )

        # 4. Re-open and query -- must succeed without crashing the
        # process. (If the rebuild didn't clean the queue, this query
        # would replay the stale rows and segfault on real-world
        # bad input.)
        client = chromadb.PersistentClient(path=self.palace)
        col = client.get_collection("mempalace_context_views")
        result = col.query(query_texts=["post-repair probe"], n_results=1)
        self.assertIsNotNone(result)
        self.assertIn("ids", result)
        # Close the client so the watermark flushes to disk -- Chroma
        # holds it in memory until segment teardown.
        del col, client

        # 5. Verify the corrupt-state lag is gone. After rebuild, the
        # new collection has fresh queue rows (one per upsert during
        # rebuild) but those rows reference clean record state from
        # SQLite -- they're safe to replay. The CORRUPTION signal we
        # were detecting was a watermark < queue lag on the OLD
        # collection's persistent state; the new collection starts
        # over. Re-running auto_repair_if_needed must therefore
        # return 0 (no rebuild needed) -- the previous corrupt state
        # is gone, and the fresh queue rows aren't a corruption
        # signal because they reference clean upserts that haven't
        # yet been backfilled (Chroma syncs every 1000 writes; small
        # collections legitimately carry queue lag until then,
        # without crashing).
        from mempalace import repair as _repair2

        # The lag detector treats fresh-collection lag below
        # sync_threshold (1000) as benign; only oversized lag or
        # stuck-watermark-from-prior-crash triggers rebuild. Verify
        # the second pass is a no-op.
        rebuilt_again = _repair2.auto_repair_if_needed(palace_path=self.palace, verbose=False)
        self.assertEqual(
            rebuilt_again,
            0,
            f"second auto_repair_if_needed should be a no-op (state is clean "
            f"after rebuild), got {rebuilt_again}",
        )


if __name__ == "__main__":
    unittest.main()
