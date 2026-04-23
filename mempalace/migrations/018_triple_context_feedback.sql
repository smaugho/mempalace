-- 018: per-context feedback on TRIPLES.
-- depends: 017_link_prediction
--
-- Triples live in a separate id namespace from entities; the existing
-- rated_useful / rated_irrelevant / surfaced edges take an entity id as
-- object (add_triple auto-creates entities for any object string via
-- INSERT OR IGNORE). Routing triple feedback through edges would
-- silently pollute the entities table with phantom rows whose ids
-- happen to look like triple ids (t_<sub>_<pred>_<obj>_<hash>).
--
-- This table stores triple-scoped feedback natively:
--   * rated_useful / rated_irrelevant: signed per-memory rating, consumed
--     by scoring.walk_rated_neighbourhood into rated_scores and
--     channel_D_list alongside the edge-based entity/memory ratings.
--   * surfaced: recall-only signal emitted on retrieval events. Channel
--     D adds a weak positive contribution (surfaced_weight × weight)
--     for items previously surfaced in similar contexts.
--
-- Last-wins across directions: at most ONE current (valid_to IS NULL)
-- feedback row per (context_id, triple_id) regardless of kind. Writing
-- a new rating or flipping useful↔irrelevant invalidates any prior row
-- for the pair before insert. Mirrors the KnowledgeGraph.add_rated_edge
-- contract shipped 2026-04-22; the partial unique index enforces it at
-- the schema level. See docs/link_author_plan.md §5.1 and the
-- add_rated_edge docstring for the four failure modes this closes.
--
-- rater_kind distinguishes agent-authored feedback (main Claude agent
-- calling finalize_intent via record_feedback) from gate-authored
-- feedback (Haiku injection-stage gate dropping a triple). Consumers
-- can weight these differently; today Channel D treats both uniformly
-- but the column is there for future calibration.

CREATE TABLE triple_context_feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    context_id  TEXT    NOT NULL,
    triple_id   TEXT    NOT NULL,
    kind        TEXT    NOT NULL
                CHECK (kind IN ('rated_useful', 'rated_irrelevant', 'surfaced')),
    relevance   INTEGER,                                  -- 1..5, NULL for surfaced
    reason      TEXT    NOT NULL DEFAULT '',
    rater_kind  TEXT    NOT NULL DEFAULT 'agent'
                CHECK (rater_kind IN ('agent', 'gate_llm')),
    rater_id    TEXT    NOT NULL DEFAULT '',
    confidence  REAL    NOT NULL DEFAULT 1.0,
    valid_from  TEXT,
    valid_to    TEXT,
    ts          TEXT    NOT NULL,
    FOREIGN KEY (triple_id) REFERENCES triples(id) ON DELETE CASCADE
);

-- Context-scoped reads: walk_rated_neighbourhood pulls all current rows
-- for the active + similar contexts.
CREATE INDEX idx_tcf_context_current
    ON triple_context_feedback (context_id)
    WHERE valid_to IS NULL;

-- Triple-scoped audit: what feedback has this triple accumulated?
CREATE INDEX idx_tcf_triple
    ON triple_context_feedback (triple_id);

-- Last-wins contract: exactly one current feedback row per (ctx, triple).
CREATE UNIQUE INDEX idx_tcf_current_pair
    ON triple_context_feedback (context_id, triple_id)
    WHERE valid_to IS NULL;
