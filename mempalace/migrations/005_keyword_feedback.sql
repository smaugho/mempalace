-- 005: Keyword suppression feedback — exponential decay for keyword-only
-- results marked irrelevant. Feeds back into Channel C scoring.
-- depends: 001_initial_schema

CREATE TABLE IF NOT EXISTS keyword_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    drawer_id TEXT NOT NULL,
    context_id TEXT DEFAULT '',
    suppression_count INTEGER DEFAULT 1,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    last_updated TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_kwf_drawer ON keyword_feedback(drawer_id);
CREATE INDEX IF NOT EXISTS idx_kwf_context ON keyword_feedback(context_id);
