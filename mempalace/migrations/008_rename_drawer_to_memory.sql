-- 008: P4.7 terminology — rename keyword_feedback.drawer_id to memory_id
-- so SQL columns match the rest of the codebase (drawers are memories now).
-- depends: 007_context_and_keywords

-- SQLite 3.25+ supports ALTER TABLE RENAME COLUMN. Drop+recreate the index too.
DROP INDEX IF EXISTS idx_kwf_drawer;

ALTER TABLE keyword_feedback RENAME COLUMN drawer_id TO memory_id;

CREATE INDEX IF NOT EXISTS idx_kwf_memory ON keyword_feedback(memory_id);
