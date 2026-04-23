-- 019: memory_flags — quality-issue flags emitted by the injection gate.
-- depends: 018_triple_context_feedback
--
-- The injection gate reads K memories at once when deciding keep/drop.
-- That joint pass is a rare multi-memory vantage point: the judge can
-- spot duplicate pairs, contradictions, stale facts, orphan memories,
-- generic summaries, memories that imply an edge the KG doesn't have,
-- and memories that mention an entity with no link. Capturing those
-- flags costs ~0 extra tokens (one more array in the tool output); the
-- memory_gardener background process consumes them asynchronously,
-- investigating each via a Claude Code subprocess and acting (merge,
-- invalidate, link, propose edge to link-author jury, rewrite summary).
--
-- Flag kinds (closed set):
--   duplicate_pair      — two memories state the same thing
--   contradiction_pair  — they contradict on a fact
--   stale               — facts look outdated vs current context
--   unlinked_entity     — memory mentions X; no entity / link present
--   orphan              — no KG entities; graph-disconnected
--   generic_summary     — summary doesn't reflect content
--   edge_candidate      — implied relationship the LLM sees (routed
--                         to link_prediction_candidates, NOT authored
--                         directly — single graph-mutation gatekeeper)
--
-- memory_ids is JSON so multi-memory flags (pair-shaped) fit the same
-- row shape as single-memory flags. Pair kinds carry [a_id, b_id];
-- single-memory kinds carry [a_id]. The gardener parses on read.
--
-- Dedup: partial unique index on (kind, memory_key, context_id)
-- while unresolved so the gate can't duplicate-file the same issue
-- from the same retrieval context. memory_key is the sorted-joined
-- member ids so {A,B} and {B,A} collapse to one row.
--
-- attempted_count tracks gardener retries; after 3 attempts a flag is
-- frozen until manually released (mirrors link-author rejection
-- cooldown). resolved_ts + resolution capture the outcome for audit.

CREATE TABLE memory_flags (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    kind             TEXT    NOT NULL
                     CHECK (kind IN (
                         'duplicate_pair',
                         'contradiction_pair',
                         'stale',
                         'unlinked_entity',
                         'orphan',
                         'generic_summary',
                         'edge_candidate'
                     )),
    memory_ids       TEXT    NOT NULL,   -- JSON array of ids involved
    memory_key       TEXT    NOT NULL,   -- canonicalised key for dedup
    detail           TEXT    NOT NULL DEFAULT '',
    context_id       TEXT    NOT NULL DEFAULT '',
    gate_run_ts      TEXT    NOT NULL,
    rater_model      TEXT    NOT NULL DEFAULT '',
    attempted_count  INTEGER NOT NULL DEFAULT 0,
    last_attempt_ts  TEXT,
    resolved_ts      TEXT,
    resolution       TEXT,                -- 'merged' | 'invalidated' | 'linked' | 'edge_proposed' | 'summary_rewritten' | 'pruned' | 'deferred' | 'no_action' | NULL
    resolution_note  TEXT
);

-- Hot-path: gardener reads unresolved flags ordered by attempt count
-- (freshest first at the same count, so stuck retries don't block
-- new work).
CREATE INDEX idx_memflags_pending
    ON memory_flags (attempted_count ASC, gate_run_ts DESC)
    WHERE resolved_ts IS NULL;

-- Audit: find everything the gate ever flagged about a given memory.
CREATE INDEX idx_memflags_key ON memory_flags (memory_key);

-- Context-scoped audit.
CREATE INDEX idx_memflags_context ON memory_flags (context_id);

-- Dedup contract: one unresolved row per (kind, memory_key, context).
-- Gate re-observing the same issue from the same context bumps
-- attempted_count instead of creating a new row.
CREATE UNIQUE INDEX idx_memflags_unique_pending
    ON memory_flags (kind, memory_key, context_id)
    WHERE resolved_ts IS NULL;
