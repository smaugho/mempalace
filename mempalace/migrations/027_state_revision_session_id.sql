-- 027: session_id column on mempalace_state_revisions for Phase D
-- depends: 026_state_revision_schema_version
--
-- State-protocol v3 Phase D (Adrian design lock 2026-05-04). The v2
-- schemas declare a per-schema scope policy:
--   scope='session' -- per-session state stream (intent_state,
--                      agent_state). Each MCP-server session has its
--                      own state; reads see only this session's
--                      writes.
--   scope='global'  -- canonical state shared across sessions
--                      (project_state, task_state). Last-write-wins;
--                      reads return the latest revision regardless
--                      of which session wrote it.
--
-- Implementing the policy needs a session_id stamp on every revision
-- so latest_state_for_entity can filter:
--   * scope='session'  : WHERE session_id = :current_session
--   * scope='global'   : ORDER BY created_at DESC LIMIT 1 (no filter)
--
-- session_id source: a UUID minted at MCP-server process start, held
-- on _STATE.session_id, propagated through record_state_revision.
-- Pre-Phase-D rows + gardener retrofit + back-compat callers leave
-- session_id NULL -- the read path treats NULL as "any session" which
-- preserves existing behaviour (writes from before this column existed
-- still surface correctly).
--
-- Migration is purely additive -- ADD COLUMN with no DEFAULT leaves
-- existing rows at NULL, no trigger or backfill loop.

ALTER TABLE mempalace_state_revisions
    ADD COLUMN session_id TEXT;

-- Index covers the session-scoped read query:
--   SELECT * FROM mempalace_state_revisions
--   WHERE entity_id = ? AND schema_id = ? AND session_id = ?
--   ORDER BY created_at DESC LIMIT 1
-- The composite (entity_id, schema_id, session_id) lets SQLite
-- short-circuit straight to the right partition. Global-scope reads
-- ignore session_id and use the existing entity_id+schema_id index
-- chain, so they stay fast without depending on this index.
CREATE INDEX idx_state_revisions_session
    ON mempalace_state_revisions (entity_id, schema_id, session_id);
