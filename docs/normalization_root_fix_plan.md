# Normalization Root Fix — 2026-04-19

## The root cause

Two separate functions normalize identifiers in the codebase, producing incompatible outputs:

| Function | Location | Non-alphanumeric → |
|----------|----------|--------------------|
| `normalize_entity_name()` | `mempalace/knowledge_graph.py` (line 233) | `_` (underscore) |
| `_normalize_memory_slug()` | `mempalace/mcp_server.py` (line 364) | `-` (hyphen) |

Record IDs are constructed with BOTH at `mcp_server.py:611`:

```python
memory_id = f"record_{agent_slug}_{normalized_slug}"
#                     ↑ underscore    ↑ hyphen
```

This produces mixed-separator IDs like `record_ga_agent_learning-verify-hook-fix`. The ID is stored verbatim in Chroma AND in the SQLite `entities` table. Any later code path that re-normalizes the ID (via `normalize_entity_name`) rewrites the hyphens to underscores, yielding `record_ga_agent_learning_verify_hook_fix` — which no longer matches the stored key. `col.get(ids=[normalized])` returns empty, silently. The outer `try/except` in many callsites swallows the miss.

This violates DRY. It has been patched at least three times at individual callsites (most recently A7 in commit 9ecf234), each time for a single bug, never for the underlying split. Every new callsite that follows the pattern reintroduces the bug.

## The decision

**Option A — underscores everywhere.** One canonical normalizer, one convention.

- `normalize_entity_name` is the single normalization function in the codebase.
- `_normalize_memory_slug` is deleted.
- All identifier-producing callsites delegate to `normalize_entity_name` + optional length cap.
- Existing Chroma records and SQLite rows with hyphenated identifiers are migrated in place to the underscored form, preserving embeddings and metadata.
- A regression test locks in the invariant: `normalize(normalize(x)) == normalize(x)` for every valid input, AND no stored ID in any collection contains `-`.

## Execution phases

Each phase lands as its own commit. Full test suite (597/1 baseline) must stay green after each.

### Phase N1 — Collapse the normalizers

- Delete `_normalize_memory_slug` from `mcp_server.py`.
- Add a thin helper `_build_record_id(agent, slug, max_len=50)` that calls `normalize_entity_name` on both parts and applies the length cap.
- Replace every `_normalize_memory_slug(x)` call with `normalize_entity_name(x)` or the new helper as appropriate.
- Add a unit test asserting idempotence (`normalize(normalize(x)) == normalize(x)`) for a representative set of inputs including already-normalized IDs, raw slugs, strings with hyphens, and strings with mixed separators.

### Phase N2 — Callsite audit

For every callsite of:
- `col.get(ids=[...])`, `col.update(ids=[...])`, `col.upsert(ids=[...])`, `col.delete(ids=[...])`
- `kg.get_entity(...)`, `kg.add_entity(...)`, `kg.add_triple(...)`, `kg.entity_edge_count(...)`, any method that takes an `entity_id` / `name` argument

verify that the identifier passed in has been normalized through `normalize_entity_name` at or before the boundary — OR was received as an already-normalized ID from elsewhere (document which).

Produce a checklist in `docs/normalization_callsite_audit.md`. No functional changes in this phase — pure audit + doc.

### Phase N3 — Migration

One-shot migration, gated on a `ServerState.normalization_migrated` flag.

- Scan `mempalace_records` and `mempalace_entities` Chroma collections.
- For each ID containing `-`, compute its target underscored form via `normalize_entity_name`.
- Round-trip rename: `col.get(ids=[old], include=['documents','metadatas','embeddings'])` → `col.delete(ids=[old])` → `col.upsert(ids=[new], documents=..., metadatas=..., embeddings=...)`.
- Same pass for SQLite: `UPDATE entities SET id = new WHERE id = old`, cascade via `triples.subject`, `triples.object`, `entity_keywords.entity_id`, `scoring_weight_feedback`, `edge_traversal_feedback`, `keyword_feedback`, `conflict_resolutions`, `memories.id` if present.
- Collision check: if `new` already exists, log and skip (shouldn't happen but defend against it).
- Flag migration complete on `ServerState`; subsequent server starts no-op.

### Phase N4 — Invariant enforcement

- Add a regression test that iterates every row in both Chroma collections and asserts no ID contains `-`.
- Add a unit test that fails if a new module-level `def normalize_*` or `def _normalize_*` function is added that doesn't delegate to `normalize_entity_name`. (grep-based, runs in the fast lane.)
- Add a comment block at the top of `knowledge_graph.py` declaring this as the single source of truth.

### Phase N5 — Broaden the feedback surface

Independent of the normalization work but part of the same root-cause cleanup. Feedback-learning currently fires only on `memory_feedback` at `finalize_intent`. It should also fire on:

- **Enrichment decisions** — every `resolve_enrichments` call (accept via `kg_add` or reject) writes a feedback row keyed on `(subject_id, predicate, object_id, intent_type)`. Future enrichment proposals consult past decisions and suppress repeats that were rejected (already partially done by B2b, extend to cover reason-free rejections too).
- **Conflict resolution decisions** — every `resolve_conflicts` action records outcome so `get_past_conflict_resolution` can pre-surface the prior decision on similar conflicts (B1b already does this for reads; extend writes to capture action + reason + intent_type).
- **Explicit edge creation** — when a user calls `kg_add` directly for an edge that was previously proposed as an enrichment, that's implicit positive feedback on the proposal predicate — record it so the enrichment detector learns which predicates the user actually uses.

### Phase N6 — Commit the deferred hook fix

The two changes I made earlier to solve the hook/server session-file disagreement (`_is_declared` KG fallback, `_sanitize_session_id`, hook fallback to `active_intent_default.json`) are orthogonal to the normalization work but currently uncommitted. Commit them as their own commit with a distinct message; do not mix into normalization commits.

### Phase M1 — Merge mempalace_records + mempalace_entities

Today both Chroma collections live in the same palace dir, share most
metadata fields (``kind``, ``importance``, ``last_touched``, ``added_by``,
``session_id``, ``intent_id``), and are queried head-to-head in the
P6.6 unified-retrieval path. The physical separation is a historical
artefact:

- Records ID convention: ``record_<agent>_<slug>``, single doc per ID.
- Entities ID convention: ``<entity_id>__v<N>``, multi-view per entity
  (one Chroma row per ``context.queries`` entry).

Execution plan, smallest-first:

1. **M1a — Search-API unification.** Introduce ``search_nodes(...)``
   in ``mcp_server.py`` that fans out across both collections, merges
   via RRF, and returns a unified hit list. Every callsite that does
   two ``multi_channel_search`` calls collapses to one. DRY win at the
   callsite level without touching storage layout yet.
2. **M1b — Unified storage.** Create ``mempalace_nodes`` as the
   canonical collection. Update every write path to target it. Records
   gain a ``__v0`` suffix on their Chroma ID for schema uniformity
   (``metadata.entity_id`` = bare record id, ``view_index`` = 0). Extend
   ``hyphen_id_migration`` (or a sibling migration) to copy every row
   from ``mempalace_records`` and ``mempalace_entities`` into
   ``mempalace_nodes`` in a single pass, preserving embeddings.
3. **M1c — Retirement.** Delete the two legacy collections once the
   migration has been confirmed green in at least one cold start of
   the live palace. Clean up ``_get_collection`` / ``_get_entity_collection``
   helpers, leaving only ``_get_nodes_collection``.
4. **M1d — Schema-invariant test.** Add a test analogous to N4:
   assert that after any public write, every row in ``mempalace_nodes``
   has a valid ``metadata.kind`` and a canonical ID.

M1 is deferred behind phases N5 and N7 below because the current
separation, while inelegant, is functionally correct (P6.6 proved both
pools can be searched head-to-head). Storage-layer unification is a
refactor, not a bug fix — chasing the bug fixes first reduces the
blast radius of any collection-schema change.

### Phase N7 — Clean the stale diary

After all the above lands:

- Invalidate (not delete) the diary entries from 2026-04-19 that describe intermediate patch attempts and false starts. Keep them readable but marked stale.
- Write a single authoritative diary entry summarizing: "the normalization contract is X, enforced in normalize_entity_name, migrated in migration N, tested in test_normalization.py".
- Update the `has_gotcha` KG entity for the two gotchas (`pretooluse_hook_disagrees_with_mempalace_active_in`, `pre_tool_use_hook_file_lookup_mismatch_hook_reads_a`) with `invalidated_by` edges pointing at the resolution memory so future sessions don't re-surface them as unresolved problems.

## Non-goals for this plan

- Not changing the KG schema.
- Not changing the Chroma collection names or metadata shape.
- Not touching the scoring formulas or hybrid_score weights.
- Not changing the intent-type hierarchy.

## Success criteria

After all phases:

1. Every identifier in Chroma and SQLite uses underscore separators only.
2. `grep -rn "def _normalize" mempalace/` returns exactly one hit (the test lane lint).
3. `grep -rn "col\.\(get\|update\|upsert\|delete\)(ids=" mempalace/` — every hit either receives a pre-normalized ID or normalizes at the call boundary, documented in the callsite audit.
4. The full test suite passes including the new invariant test.
5. A cold-start of the MCP server runs the migration, reports how many rows it touched, and the subsequent start reports zero touched.
6. The feedback system records at least one row per `resolve_enrichments` call, per `resolve_conflicts` call, and per implicit-accept `kg_add` call.
