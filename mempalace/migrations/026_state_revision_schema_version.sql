-- 026: schema_version column on mempalace_state_revisions for Phase 6
-- depends: 025_state_init_needed_flag
--
-- State-protocol Phase 6 (Adrian design lock 2026-05-03): lazy migration
-- at injection-gate time. Each state revision carries the STATE_SCHEMAS
-- version it was written against. When the InjectionGate filters in an
-- entity whose latest revision is at version < current schema version,
-- the apply_gate hook walks the migration chain in
-- mempalace/state_migrations/{schema_id}/v{N}_to_v{N+1}.py and writes a
-- new revision at the current version. Old revisions stay as audit
-- history; the latest is always at-or-newer-than the schema's current
-- version after a successful gate pass. Dormant entities never trigger
-- migration cost -- the cost is paid only when the entity is surfaced
-- to the LLM and survives gating.
--
-- Default 1: existing rows pre-Phase-6 are treated as v1 (the only
-- version that ever existed before this column landed). New writes
-- read STATE_SCHEMAS[schema_id].version and stamp explicitly.
--
-- Migration is purely additive -- ADD COLUMN with a NOT NULL DEFAULT
-- back-fills existing rows in a single SQLite operation, no trigger or
-- backfill loop needed.

ALTER TABLE mempalace_state_revisions
    ADD COLUMN schema_version INTEGER NOT NULL DEFAULT 1;

-- Index helps the migration runner answer "are any rows at version < N
-- for this schema?" without a full table scan. Composite (schema_id,
-- schema_version) covers both the per-schema-id filter and the
-- range-on-version query.
CREATE INDEX idx_state_revisions_schema_version
    ON mempalace_state_revisions (schema_id, schema_version);
