-- 004: Add context_id to edge_traversal_feedback for contextual MaxSim
-- feedback (P2 — stored context vectors in ChromaDB).
-- depends: 003_edge_traversal_feedback

ALTER TABLE edge_traversal_feedback ADD COLUMN context_id TEXT DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_etf_context
    ON edge_traversal_feedback(context_id);
