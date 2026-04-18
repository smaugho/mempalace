-- 013: Add a verbalized statement column to triples.
-- depends: 012_drop_source_closet
--
-- Triples are stored structurally (subject, predicate, object) but were
-- never indexed for semantic search. Asking "who lives in Warsaw" missed
-- a clear `(adrian, lives_in, warsaw)` triple unless prose memory text
-- happened to use the same words. This column holds the natural-language
-- form of the triple ("Adrian lives in Warsaw") for embedding into the
-- mempalace_triples Chroma collection so triples become first-class
-- search results alongside prose memories and entities.

ALTER TABLE triples ADD COLUMN statement TEXT;

CREATE INDEX IF NOT EXISTS idx_triples_has_statement
    ON triples(id) WHERE statement IS NOT NULL;
