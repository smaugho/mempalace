-- 003: Edge traversal feedback — tracks whether a graph edge was useful for
-- an intent_type. Feeds back into Channel B (graph) scoring.
-- depends: 001_initial_schema

CREATE TABLE IF NOT EXISTS edge_traversal_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    intent_type TEXT NOT NULL,
    useful BOOLEAN NOT NULL,
    context_keywords TEXT DEFAULT '',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_etf_edge
    ON edge_traversal_feedback(subject, predicate, object);
CREATE INDEX IF NOT EXISTS idx_etf_intent
    ON edge_traversal_feedback(intent_type);
