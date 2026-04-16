-- 007: Unified Context object (P4) — caller-provided keywords stored as
-- entity_keywords (no more auto-extraction) + creation_context_id on every
-- entity and edge so MaxSim/Chamfer feedback applies by similarity to the
-- exact context the thing was filed under.
-- depends: 006_scoring_weight_feedback

-- Caller-provided keywords for entities (memories, classes, predicates, etc.)
-- source = 'caller'      → set explicitly via the Context API
-- source = 'auto_legacy' → backfilled from existing descriptions during the
--                          initial schema migration; safe to clean up.
CREATE TABLE IF NOT EXISTS entity_keywords (
    entity_id  TEXT NOT NULL,
    keyword    TEXT NOT NULL,
    source     TEXT DEFAULT 'caller',
    added_at   TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (entity_id, keyword),
    FOREIGN KEY (entity_id) REFERENCES entities(id)
);
CREATE INDEX IF NOT EXISTS idx_entity_keywords_keyword ON entity_keywords(keyword);
CREATE INDEX IF NOT EXISTS idx_entity_keywords_entity  ON entity_keywords(entity_id);

-- Creation Context fingerprint for entities. The actual view vectors live in
-- the mempalace_feedback_contexts ChromaDB collection (one row per view,
-- shared context_id metadata) — this column just points at which context_id.
ALTER TABLE entities ADD COLUMN creation_context_id TEXT DEFAULT '';

-- Same for edges. P4.3 wires kg_add to populate this. Existing
-- edge_traversal_feedback.context_id stays as the *traversal* context.
ALTER TABLE triples ADD COLUMN creation_context_id TEXT DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_entities_creation_ctx ON entities(creation_context_id);
CREATE INDEX IF NOT EXISTS idx_triples_creation_ctx  ON triples(creation_context_id);
