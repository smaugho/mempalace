-- 030: extend memory_flags CHECK constraint with the three ontology-
-- review flag kinds.
-- depends: 029_state_revision_challenges
--
-- v3.10.0 (Adrian msg_176/178 2026-05-21). The bg quality pass now
-- emits three new gardener flags so the corpus self-heals its ontology:
--   is_a_review            -- an entity's is_a class chain looks wrong /
--                             missing / too coarse. memory_ids=[entity_id];
--                             detail = the problem + the expected is_a.
--   kind_misclassification -- an entity's `kind` is wrong for what it is
--                             (e.g. a category tagged kind=entity that
--                             should be kind=class). memory_ids=[entity_id];
--                             detail = current kind + correct kind.
--   class_id_improvement   -- a kind=class id/name is opaque/poor vs its
--                             description. memory_ids=[class_id];
--                             detail = suggested better id + why.
--
-- The gardener resolves them via knowledge_graph._FLAG_RESOLUTIONS values
-- is_a_corrected / kind_corrected / class_renamed (class_renamed cascades
-- through the existing merge_entities edge-rewrite path).
--
-- SQLite cannot ALTER a CHECK constraint in place, so we rebuild the
-- table per the 021/025 pattern: create _new with the extended CHECK,
-- copy rows, drop old, rename. No FKs on memory_flags -- rebuild is safe.
-- Indexes recreated with the original names + WHERE clauses.

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
                         'state_init_needed',
                         'is_a_review',
                         'kind_misclassification',
                         'class_id_improvement'
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
