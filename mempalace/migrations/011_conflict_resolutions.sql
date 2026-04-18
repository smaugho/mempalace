-- 011: Persist conflict-resolution decisions with their reasons.
-- depends: 010_normalize_predicate_hyphens
--
-- tool_resolve_conflicts validates a mandatory 15-char reason and runs a
-- laziness check, then throws the reason away. This table captures every
-- resolution (invalidate/merge/keep/skip) with the reason, the resolving
-- agent, and the active intent_type so future audits and feedback loops
-- can learn from past decisions.

CREATE TABLE IF NOT EXISTS conflict_resolutions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    conflict_id    TEXT    NOT NULL,
    conflict_type  TEXT    NOT NULL,
    action         TEXT    NOT NULL,
    reason         TEXT    NOT NULL,
    existing_id    TEXT,
    new_id         TEXT,
    agent          TEXT,
    intent_type    TEXT,
    context_id     TEXT    DEFAULT '',
    created_at     TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cr_conflict_id
    ON conflict_resolutions(conflict_id);
CREATE INDEX IF NOT EXISTS idx_cr_existing_id
    ON conflict_resolutions(existing_id);
CREATE INDEX IF NOT EXISTS idx_cr_new_id
    ON conflict_resolutions(new_id);
CREATE INDEX IF NOT EXISTS idx_cr_agent_intent
    ON conflict_resolutions(agent, intent_type);
