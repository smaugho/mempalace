-- 016: Phase 2 — keyword IDF table for BM25-dampened Channel C.
-- depends: 015_retire_old_feedback
--
-- Before P2, the keyword channel used a naive overlap_ratio (matched
-- keywords / total keywords). That let dominant generic keywords like
-- "the", "file", "intent" dominate search results on a personal palace
-- whose entire corpus is narrow domain. BM25 robust IDF
-- (Robertson & Jones 1976 JASIS; Robertson & Zaragoza 2009
-- "Foundations of BM25") dampens each keyword by its corpus rarity:
--
--     idf(term) = log((N - freq(term) + 0.5) / (freq(term) + 0.5))
--
-- where N = total number of memories and freq(term) = number of
-- memories that contain the term. Rare terms get high idf
-- contribution; dominant ones get near-zero or even negative (which
-- the channel clamps at min_idf=0.5 for early-exit).
--
-- The table is populated incrementally from _add_memory_internal (see
-- the keyword_idf_hook) so IDF stays current without a bulk rebuild.
-- Initial backfill happens at the end of this migration from the
-- existing entity_keywords distribution — memories only (kind='record')
-- so IDF reflects the retrieval corpus, not the schema entities.

CREATE TABLE IF NOT EXISTS keyword_idf (
    keyword         TEXT PRIMARY KEY,
    freq            INTEGER NOT NULL DEFAULT 0,
    idf             REAL NOT NULL DEFAULT 0.0,
    last_updated_ts TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_keyword_idf_freq ON keyword_idf(freq);

-- Backfill freq from the existing entity_keywords table, scoped to
-- record-kind entities so IDF matches the retrieval corpus. idf is
-- recomputed by _keyword_idf_recompute in mcp_server on first boot
-- after this migration lands.
INSERT OR IGNORE INTO keyword_idf (keyword, freq, idf, last_updated_ts)
SELECT
    ek.keyword AS keyword,
    COUNT(DISTINCT ek.entity_id) AS freq,
    0.0 AS idf,
    datetime('now') AS last_updated_ts
FROM entity_keywords ek
JOIN entities e ON e.id = ek.entity_id
WHERE e.kind = 'record'
GROUP BY ek.keyword;
