-- 017: link prediction candidate queue + author runs + source dedup.
-- depends: 016_keyword_idf
--
-- Bipartite context-entity Adamic-Adar candidate signal. Entity pairs
-- that co-appear in DISTINCT reused + positively-rated contexts
-- accumulate a score here. Each (pair, context) tuple contributes at
-- most once (dedup via link_prediction_sources). The mempalace
-- link-author CLI reads above-threshold rows, asks an Opus-designed
-- Haiku jury to author the edge (or reject), writes the verdict
-- back, and creates the edge via kg_add on accepts.
--
-- Canonical ordering: from_entity lexically < to_entity. The upsert
-- must canonicalise before INSERT to keep the PK unique across
-- symmetric pair permutations.

CREATE TABLE link_prediction_candidates (
    from_entity            TEXT NOT NULL,
    to_entity              TEXT NOT NULL,
    score                  REAL NOT NULL DEFAULT 0.0,
    shared_context_count   INTEGER NOT NULL DEFAULT 0,
    last_context_id        TEXT NOT NULL DEFAULT '',
    last_updated_ts        TEXT NOT NULL DEFAULT '',
    processed_ts           TEXT,
    llm_verdict            TEXT,   -- 'edge' | 'no_edge' | 'uncertain' | 'jury_design_failed' | NULL
    llm_predicate          TEXT,   -- name of chosen or newly-created predicate
    llm_statement          TEXT,   -- natural-language verbalization for the edge
    llm_reason             TEXT,   -- jury's synthesis reasoning
    llm_jury_personas      TEXT,   -- JSON: the Opus-designed personas used for this candidate
    llm_jury_design_model  TEXT,   -- e.g. 'claude-opus-4-7-<date>'
    llm_jury_exec_model    TEXT,   -- e.g. 'claude-haiku-4-7-<date>'
    PRIMARY KEY (from_entity, to_entity)
);

-- Hot-path read: pending rows ordered by score desc.
CREATE INDEX idx_link_cands_pending
    ON link_prediction_candidates (score DESC)
    WHERE processed_ts IS NULL;

-- Debug / audit: recently-processed rows.
CREATE INDEX idx_link_cands_processed
    ON link_prediction_candidates (processed_ts DESC)
    WHERE processed_ts IS NOT NULL;

-- Distinct-context dedup: each context contributes to a given pair
-- exactly once regardless of how many times it's reused. INSERT OR
-- IGNORE on the upsert path; only increment link_prediction_candidates
-- when the insert here actually inserted (rowcount == 1).
CREATE TABLE link_prediction_sources (
    from_entity    TEXT NOT NULL,
    to_entity      TEXT NOT NULL,
    ctx_id         TEXT NOT NULL,
    contributed_ts TEXT NOT NULL,
    PRIMARY KEY (from_entity, to_entity, ctx_id)
);

CREATE INDEX idx_link_sources_ctx ON link_prediction_sources (ctx_id);

CREATE TABLE link_author_runs (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    started_ts             TEXT NOT NULL,
    completed_ts           TEXT,
    candidates_processed   INTEGER DEFAULT 0,
    edges_created          INTEGER DEFAULT 0,
    edges_rejected         INTEGER DEFAULT 0,
    new_predicates_created INTEGER DEFAULT 0,
    design_calls           INTEGER DEFAULT 0, -- Opus invocations (fewer than candidates when batched)
    jury_design_failures   INTEGER DEFAULT 0,
    design_model           TEXT,
    exec_model             TEXT,
    errors                 TEXT
);

CREATE INDEX idx_link_runs_recent ON link_author_runs (started_ts DESC);
