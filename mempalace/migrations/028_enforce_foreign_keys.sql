-- v3.2.5 (Adrian directive 2026-05-12): clean dangling FK rows so the
-- _conn() PRAGMA foreign_keys=ON flip is safe to enable across existing
-- palaces. Pre-v3.2.5 the FK clauses declared in migrations 001/007/018
-- were decorative -- PRAGMA defaulted off, so deleted parent rows left
-- orphaned children behind.
--
-- Production scan 2026-05-12 found exactly three dangling rows, all in
-- triple_context_feedback (rated_useful / rated_irrelevant / surfaced
-- bookkeeping against triples that were later invalidated without
-- cleaning their feedback). entity_keywords and the two triples FKs
-- (subject + object) were clean (0 dangling rows each).
--
-- After this migration applies and _conn() flips PRAGMA on, the
-- triples cascade-delete chain becomes a hard schema invariant
-- (deleting an entity drops its triples drops their feedback).
--
-- vec_rowid_map is NOT touched here -- its entity_id column stores
-- logical vec ids that span four id namespaces ({eid}, {eid}__v{i},
-- {cid}_v{i}, triple_id), which doesn't fit a single-target FK.
-- Cascade for vec_palace stays at the app layer (kg_delete_entity).

DELETE FROM triple_context_feedback
WHERE triple_id NOT IN (SELECT id FROM triples);
