-- 015: Phase 2 cutover — retire old feedback tables, add triple properties,
-- index the three new feedback predicates.
-- depends: 014_context_as_entity
--
-- This migration is part of the atomic P2 cutover
-- (docs/context_as_entity_redesign_plan.md §2). Old feedback mechanisms
-- are replaced, not wrapped — there is no "both work" window.
--
-- - keyword_feedback: the old per-memory keyword-suppression table is
--   retired. Its role as "filter dominant keywords from generic corpus
--   noise" is now played by BM25-IDF dampening on Channel C (see
--   migration 016's keyword_idf + the _build_keyword_channel rewrite).
-- - edge_traversal_feedback: the old edge-usefulness table is retired.
--   Its role as "remember which graph edges led to useful memories" is
--   now played by Channel D, which re-derives the signal from the
--   context--rated_useful-->memory edges written at finalize time.
-- - scoring_weight_feedback: truncated for a clean cold-start. The 5
--   hybrid_score weights drop to 4 (W_REL retires), so the existing
--   rows would refer to a component that no longer exists. P3
--   re-enables learning against the new shape.
--
-- Triples get a generic `properties TEXT` column so the new feedback
-- predicates (surfaced, rated_useful, rated_irrelevant) can carry
-- rich structured props — rank, channel, sim_score, reason, relevance,
-- etc. — without smuggling them into pre-existing columns.
--
-- NB: dropping the ChromaDB `mempalace_feedback_contexts` collection
-- can't be done in SQL. That drop is handled in mcp_server startup by
-- ServerState._feedback_contexts_dropped (one-shot migration flag).

-- ── Drop retired tables ──
DROP TABLE IF EXISTS keyword_feedback;
DROP TABLE IF EXISTS edge_traversal_feedback;

-- ── Truncate scoring_weight_feedback (cold-start for P3 weight learning) ──
DELETE FROM scoring_weight_feedback;

-- ── Generic props column on triples ──
ALTER TABLE triples ADD COLUMN properties TEXT DEFAULT '{}';

-- ── Indexes for the new feedback predicates ──
CREATE INDEX IF NOT EXISTS idx_triples_surfaced_subject
    ON triples(subject) WHERE predicate = 'surfaced';
CREATE INDEX IF NOT EXISTS idx_triples_surfaced_object
    ON triples(object) WHERE predicate = 'surfaced';
CREATE INDEX IF NOT EXISTS idx_triples_rated_useful_subject
    ON triples(subject) WHERE predicate = 'rated_useful';
CREATE INDEX IF NOT EXISTS idx_triples_rated_useful_object
    ON triples(object) WHERE predicate = 'rated_useful';
CREATE INDEX IF NOT EXISTS idx_triples_rated_irrelevant_subject
    ON triples(subject) WHERE predicate = 'rated_irrelevant';
CREATE INDEX IF NOT EXISTS idx_triples_rated_irrelevant_object
    ON triples(object) WHERE predicate = 'rated_irrelevant';
