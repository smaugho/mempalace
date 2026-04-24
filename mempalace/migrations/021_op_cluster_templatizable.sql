-- 021: extend memory_flags CHECK constraint with op_cluster_templatizable.
-- depends: 020_memory_gardener_runs
--
-- S3a of the operation-memory tier (arXiv 2512.18950 / Leontiev 1981 AAO).
-- declare_operation now emits this flag kind when retrieve_past_operations
-- surfaces >=3 same-tool same-sign precedents in a single retrieval. The
-- gardener consumes the flag in S3b by Haiku-synthesising one reusable
-- operation-template record per cluster (kind='record' with a new
-- `templatizes` predicate pointing back at the source op entities).
--
-- memory_ids: JSON array of the op entity ids in the cluster.
-- detail: "positive" (performed_well cluster) or "negative"
--         (performed_poorly cluster) — lets the gardener ask the right
--         template question (recipe vs. failure-mode).
-- context_id: the active operation-context that surfaced the cluster;
--             scopes dedup so the same cluster under the same context
--             bumps attempted_count instead of inserting a duplicate.
--
-- SQLite can't ALTER a CHECK constraint in place, so we rebuild the
-- table, copy rows, drop the old, rename. No FKs on memory_flags —
-- rebuild is safe. Indexes are recreated with the original names.

CREATE TABLE memory_flags_new (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    kind             TEXT    NOT NULL
                     CHECK (kind IN (
                         'duplicate_pair',
                         'contradiction_pair',
                         'stale',
                         'unlinked_entity',
                         'orphan',
                         'generic_summary',
                         'edge_candidate',
                         'op_cluster_templatizable'
                     )),
    memory_ids       TEXT    NOT NULL,
    memory_key       TEXT    NOT NULL,
    detail           TEXT    NOT NULL DEFAULT '',
    context_id       TEXT    NOT NULL DEFAULT '',
    gate_run_ts      TEXT    NOT NULL,
    rater_model      TEXT    NOT NULL DEFAULT '',
    attempted_count  INTEGER NOT NULL DEFAULT 0,
    last_attempt_ts  TEXT,
    resolved_ts      TEXT,
    resolution       TEXT,
    resolution_note  TEXT
);

INSERT INTO memory_flags_new
    (id, kind, memory_ids, memory_key, detail, context_id,
     gate_run_ts, rater_model, attempted_count, last_attempt_ts,
     resolved_ts, resolution, resolution_note)
SELECT id, kind, memory_ids, memory_key, detail, context_id,
       gate_run_ts, rater_model, attempted_count, last_attempt_ts,
       resolved_ts, resolution, resolution_note
FROM memory_flags;

DROP TABLE memory_flags;

ALTER TABLE memory_flags_new RENAME TO memory_flags;

CREATE INDEX idx_memflags_pending
    ON memory_flags (attempted_count ASC, gate_run_ts DESC)
    WHERE resolved_ts IS NULL;

CREATE INDEX idx_memflags_key ON memory_flags (memory_key);

CREATE INDEX idx_memflags_context ON memory_flags (context_id);

CREATE UNIQUE INDEX idx_memflags_unique_pending
    ON memory_flags (kind, memory_key, context_id)
    WHERE resolved_ts IS NULL;
