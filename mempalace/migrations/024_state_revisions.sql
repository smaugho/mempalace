-- 024: mempalace_state_revisions table for state-protocol v1.
-- depends: 023_drop_description_column
--
-- State-protocol v1 design lock 2026-05-03 (Adrian Option B). When an
-- agent declares a state delta on a state-bearing entity (declare_operation
-- state_deltas with status='changed'), the validator applies the JSON Patch
-- against the current projection, validates the result against
-- state_schemas.STATE_SCHEMAS[schema_id].json_schema, and writes one row
-- here. The row id then becomes the subject of a state_changed_by edge to
-- the operation context (JTMS justification per Doyle 1979).
--
-- Columns:
--   rev_id          stable string id (e.g. 'srv_<entity>_<ts>') -- KG triple
--                   subject for state_changed_by edges. Must be unique.
--   entity_id       the entity whose state changed (Task instance, agent,
--                   intent execution). Indexed for "latest state per
--                   entity" queries used by the projection materializer.
--   schema_id       state_schemas.STATE_SCHEMAS key (e.g. 'task_state').
--                   Records which schema validated the payload at write
--                   time so future validators can detect schema drift.
--   payload         JSON-encoded slot dict that satisfies the schema at
--                   write time. The current state IS the latest payload
--                   for the entity; older payloads are history.
--   created_at      ISO timestamp; ordering key for projection replay.
--   op_context_id   the operation context that caused the revision; the
--                   target of the state_changed_by KG edge. Indexed for
--                   the JTMS retraction sweep when an op is invalidated.
--   agent           who wrote the revision -- declared agent for delta
--                   writes, 'memory_gardener' for retrofit-default writes.
--
-- Retraction: when an op is invalidated, the gardener walks
-- state_changed_by edges from this op_context_id and either deletes the
-- revision (if it's the latest for its entity_id and the next-latest
-- exists) or marks it superseded. Cap at 1-hop per v2 design lock rule 9.
--
-- Pure additive migration -- no existing table touched. ON CONFLICT not
-- needed because rev_id is generated unique per write.

CREATE TABLE mempalace_state_revisions (
    rev_id         TEXT    PRIMARY KEY,
    entity_id      TEXT    NOT NULL,
    schema_id      TEXT    NOT NULL,
    payload        TEXT    NOT NULL,
    created_at     TEXT    NOT NULL,
    op_context_id  TEXT    NOT NULL DEFAULT '',
    agent          TEXT    NOT NULL DEFAULT ''
);

-- Latest state per entity: ORDER BY created_at DESC LIMIT 1 with a
-- WHERE entity_id = ? lookup is the hot read path. Composite index
-- covers both the equality filter and the ORDER.
CREATE INDEX idx_state_revisions_entity_time
    ON mempalace_state_revisions (entity_id, created_at DESC);

-- JTMS retraction sweep: WHERE op_context_id = ? scans every revision
-- caused by an invalidated op. Standalone index because op_context_id
-- has higher selectivity than entity_id for retraction queries.
CREATE INDEX idx_state_revisions_op_context
    ON mempalace_state_revisions (op_context_id);

-- Schema-drift detection: WHERE schema_id = ? counts revisions per
-- schema, used by the gardener when a schema change lands and existing
-- revisions need revalidation.
CREATE INDEX idx_state_revisions_schema
    ON mempalace_state_revisions (schema_id);
