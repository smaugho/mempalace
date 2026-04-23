-- 020: memory_gardener_runs — per-batch audit log for the background
-- corpus-refinement process.
-- depends: 019_memory_flags
--
-- Mirrors link_author_runs (migration 017) so operators learn one shape
-- once. Every gardener invocation writes one row here: start time, end
-- time, which flags it processed (by id), how many outcomes of each
-- type, subprocess exit code, and any errors encountered.
--
-- Why mirror link-author: both are out-of-session Anthropic-driven
-- subsystems; identical run-log shape means monitoring / alerting /
-- dashboards can be unified. The differences are domain-specific
-- (gardener has per-action counters; link-author has jury-specific
-- counters) but the envelope is the same.

CREATE TABLE memory_gardener_runs (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    started_ts            TEXT    NOT NULL,
    completed_ts          TEXT,
    flags_processed       INTEGER NOT NULL DEFAULT 0,
    flag_ids              TEXT    NOT NULL DEFAULT '[]',  -- JSON array
    merges                INTEGER NOT NULL DEFAULT 0,
    invalidations         INTEGER NOT NULL DEFAULT 0,
    links_created         INTEGER NOT NULL DEFAULT 0,
    edges_proposed        INTEGER NOT NULL DEFAULT 0,      -- pushed to link_prediction_candidates
    summary_rewrites      INTEGER NOT NULL DEFAULT 0,
    prunes                INTEGER NOT NULL DEFAULT 0,
    deferrals             INTEGER NOT NULL DEFAULT 0,
    no_action             INTEGER NOT NULL DEFAULT 0,
    subprocess_exit_code  INTEGER,
    gardener_model        TEXT,
    errors                TEXT                              -- free-form string
);

-- Recent runs first for the CLI status command.
CREATE INDEX idx_gardener_runs_recent ON memory_gardener_runs (started_ts DESC);
