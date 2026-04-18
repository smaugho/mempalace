-- 010: Normalize predicate strings by replacing hyphens with underscores.
-- depends: 009_composite_indexes_and_provenance
--
-- Root cause: predicate normalization at the storage boundary previously
-- only collapsed spaces (.lower().replace(" ", "_")). Hyphens survived, so
-- seeded `add_triple(X, "is-a", Y)` rows coexisted with caller writes of
-- `is_a`, producing two distinct predicate strings in the DB for the same
-- concept. This migration folds every hyphenated predicate to underscores.

UPDATE triples
   SET predicate = REPLACE(predicate, '-', '_')
 WHERE predicate LIKE '%-%';

UPDATE edge_traversal_feedback
   SET predicate = REPLACE(predicate, '-', '_')
 WHERE predicate LIKE '%-%';
