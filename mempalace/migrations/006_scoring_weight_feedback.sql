-- 006: Scoring component feedback — records (component, value, was_useful)
-- triples that compute_learned_weights uses to adapt hybrid_score weights.
-- depends: 001_initial_schema

CREATE TABLE IF NOT EXISTS scoring_weight_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    component TEXT NOT NULL,
    component_value REAL NOT NULL,
    was_useful BOOLEAN NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_swf_component ON scoring_weight_feedback(component);
