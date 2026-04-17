-- P6.8: composite indexes for graph traversal performance.
-- The BFS in declare_intent does WHERE subject=? queries (covered by
-- idx_triples_subject) but edge-usefulness lookups often add AND predicate=?.
-- Composite indexes let SQLite satisfy both columns in one index scan.
--
-- P6.7a: provenance columns on entities table.
-- session_id and intent_id are auto-injected by the write paths so every
-- entity/record carries provenance metadata. Stored as SQLite columns
-- (queryable, indexable) in addition to Chroma metadata.

-- Composite indexes for faster graph traversal
CREATE INDEX IF NOT EXISTS idx_triples_subject_predicate
    ON triples(subject, predicate);
CREATE INDEX IF NOT EXISTS idx_triples_object_predicate
    ON triples(object, predicate);

-- Entity kind index for kind-filtered queries (e.g. kind='class' in
-- _build_intent_hierarchy, kind='record' in record-collection routing)
CREATE INDEX IF NOT EXISTS idx_entities_kind
    ON entities(kind);

-- Provenance columns (nullable — old entities won't have them)
ALTER TABLE entities ADD COLUMN session_id TEXT DEFAULT NULL;
ALTER TABLE entities ADD COLUMN intent_id TEXT DEFAULT NULL;

-- Index provenance for session-scoped queries
CREATE INDEX IF NOT EXISTS idx_entities_session
    ON entities(session_id);
CREATE INDEX IF NOT EXISTS idx_entities_intent
    ON entities(intent_id);
