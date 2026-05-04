-- 025: extend memory_flags CHECK constraint with state_init_needed.
-- depends: 024_state_revisions
--
-- State-protocol v1 piece 6 (retrofit, Adrian 2026-05-03). When an
-- instance entity of a state-bearing class (Task / agent / intent_type
-- carry state_schema_id properties) surfaces with no row in
-- mempalace_state_revisions, the runtime writes a memory_flag with
-- kind='state_init_needed'. The retrofit gardener handler drains the
-- flag, calls state_schemas.materialize_default(schema_id), and writes
-- the initial revision via knowledge_graph.record_state_revision with
-- agent='memory_gardener' and an empty op_context_id (so no spurious
-- state_changed_by JTMS edge lands -- the seed isn't caused by an op).
--
-- memory_ids: JSON array containing the single instance entity id that
--             needs initialization. Single-element list keeps the
--             gardener's batch-shape consistent with other flag kinds.
-- detail:     the schema_id (e.g. 'task_state') so the handler skips
--             the class-resolution lookup at drain time.
-- context_id: the user/intent context that surfaced the instance --
--             scopes dedup so a re-surface in the same context bumps
--             attempted_count rather than inserting a duplicate.
--
-- SQLite cannot ALTER a CHECK constraint in place, so we rebuild the
-- table per the 021 pattern, copy rows, drop the old, rename. No FKs on
-- memory_flags -- rebuild is safe. Indexes are recreated with the
-- original names + the same WHERE clauses.

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
                         'op_cluster_templatizable',
                         'state_init_needed'
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
