-- 029: mempalace_state_revision_challenges table for state-judge v2 Phase 3 Slice B.
-- depends: 028_enforce_foreign_keys
--
-- Phase 3 Slice B (Adrian directive 2026-05-13). When the state_judge
-- auto-applies a patch under MEMPALACE_STATE_PROTOCOL=v2_visibility
-- (Slice A, shipped in v3.2.9), the agent may disagree with the
-- judge's call. The challenge_state_change MCP files a challenge
-- against a specific revision; this table is its audit trail.
--
-- Two challenge modes:
--   1. Restore (restore_prior=True): the handler writes a NEW
--      state_revision restoring the state to the revision that
--      preceded rev_id, and stamps retracted_rev_id on the
--      challenge row.  Net effect: the judge's write is undone but
--      both rows survive for forensics.
--   2. Info-only (restore_prior=False): retracted_rev_id is null;
--      only the JTMS edge + this row exist.  Use when the agent
--      wants to flag a disputed write without rolling it back.
--
-- Columns:
--   challenge_id      stable string id (e.g. 'src_<rev>_<ts>') -- KG
--                     triple subject for state_challenged_by edges.
--                     Unique.
--   rev_id            FK to mempalace_state_revisions(rev_id) with
--                     ON DELETE CASCADE: if the revision is purged
--                     the challenge row goes with it.
--   challenge_op_id   the operation context that filed the challenge.
--                     Becomes the OBJECT of the state_challenged_by
--                     edge for graph queries.
--   agent             challenging agent id.  Cross-checked at
--                     write time against the entity's owning agent
--                     so cross-agent state stomp is rejected.
--   justification     why the judge was wrong / why the agent
--                     disagreed.  Required.  Free-form text.
--   created_at        ISO timestamp.
--   retracted_rev_id  null on info-only challenges; otherwise the
--                     rev_id of the new state_revision that restored
--                     prior state.  No FK so a later GC of
--                     state_revisions doesn't strand the column
--                     (the audit trail still reads cleanly with a
--                     null retracted_rev_id meaning "row deleted").
--
-- Purely additive migration; no existing table touched.

CREATE TABLE mempalace_state_revision_challenges (
    challenge_id      TEXT    PRIMARY KEY,
    rev_id            TEXT    NOT NULL REFERENCES mempalace_state_revisions(rev_id) ON DELETE CASCADE,
    challenge_op_id   TEXT    NOT NULL DEFAULT '',
    agent             TEXT    NOT NULL DEFAULT '',
    justification     TEXT    NOT NULL,
    created_at        TEXT    NOT NULL,
    retracted_rev_id  TEXT
);

-- Lookup by revision: WHERE rev_id = ? finds every challenge filed
-- against a single auto-applied write.  Used by the challenge MCP
-- idempotency check and the gardener's judge-accuracy analysis.
CREATE INDEX idx_state_revision_challenges_rev
    ON mempalace_state_revision_challenges (rev_id);

-- JTMS sweep: WHERE challenge_op_id = ? walks every challenge filed
-- by a single operation when that op is invalidated.  Mirrors the
-- idx_state_revisions_op_context index pattern from migration 024.
CREATE INDEX idx_state_revision_challenges_op_context
    ON mempalace_state_revision_challenges (challenge_op_id);

-- Per-agent audit: WHERE agent = ? counts how often a specific agent
-- challenges judge writes -- input to the gardener's judge-accuracy
-- + agent-trust telemetry.
CREATE INDEX idx_state_revision_challenges_agent
    ON mempalace_state_revision_challenges (agent);
