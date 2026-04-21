-- 014: Context as first-class KG entity — indexes for the new predicates.
-- depends: 013_triple_statement
--
-- The context-as-entity redesign (docs/context_as_entity_redesign_plan.md)
-- introduces two predicates in P1:
--   - created_under: <any> --> <context>        (provenance)
--   - similar_to:    <context> --> <context>    (MaxSim neighborhood)
--
-- Both are hot-read paths once P2's Channel D is live. Partial indexes
-- keep the existing idx_triples_subject / idx_triples_object indexes
-- small (they already cover generic lookups) and add narrow indexes for
-- the two predicates that will dominate queries on context entities.
--
-- kind='context' gets its own partial index on entities so
-- `SELECT * FROM entities WHERE kind='context'` (used by kg_stats and by
-- the context ANN rebuild path) stays O(context_count) rather than
-- O(entity_count).

CREATE INDEX IF NOT EXISTS idx_triples_created_under_subject
    ON triples(subject) WHERE predicate = 'created_under';

CREATE INDEX IF NOT EXISTS idx_triples_created_under_object
    ON triples(object) WHERE predicate = 'created_under';

CREATE INDEX IF NOT EXISTS idx_triples_similar_to
    ON triples(subject, object) WHERE predicate = 'similar_to';

CREATE INDEX IF NOT EXISTS idx_entities_kind_context
    ON entities(id) WHERE kind = 'context';
