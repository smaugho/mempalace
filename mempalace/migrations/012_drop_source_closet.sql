-- 012: Drop the source_closet column from triples.
-- depends: 011_conflict_resolutions
--
-- source_closet was a vestige of the palace "closet" metaphor that has
-- now been removed from the API. No callers write to it anymore.
-- SQLite 3.35+ supports ALTER TABLE DROP COLUMN directly; older versions
-- would require a full table copy which we don't need here.

ALTER TABLE triples DROP COLUMN source_closet;
