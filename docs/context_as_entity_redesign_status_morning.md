# Context-as-entity redesign — morning handoff (2026-04-22)

**Status: ALL THREE PHASES MERGED TO `main`.** The run completed without
hitting either self-stop condition (>3 unexplained failures, >2h no
convergence). Every phase was delivered per the plan's discipline:
branch → commits → push → PR → self-review for orphans → `gh pr merge
--merge`.

## Merge order on `main`

```
38fcb6d Merge pull request #59 — Phase 3: Telemetry, tuning, and eval harness
1c4bd07 Merge pull request #58 — Phase 2: Atomic feedback and scoring cutover
9f447e5 Merge pull request #57 — Phase 1: Context identity + creation provenance
a4d2fff pre-redesign prep: summary-first, wake_up hardening, finalize parse guard
```

PR links:
- Phase 1: https://github.com/smaugho/mempalace/pull/57
- Phase 2: https://github.com/smaugho/mempalace/pull/58
- Phase 3: https://github.com/smaugho/mempalace/pull/59

## Final test count

- Baseline before the run: 856 passed.
- Final: **894 passed, 1 skipped, 0 failures** (non-benchmark, non-manual
  lane). Delta +38 tests over baseline — all new tests for the redesign.

Fast-lane (unit marker, pre-commit hook): 630 passed, 1 skipped.

## What shipped per phase

### Phase 1 — Context identity + creation provenance (PR #57)

- New KG kind: `"context"` is a first-class KG entity kind. Every
  `declare_intent` / `declare_operation` / `kg_search` mints one or
  reuses an existing via ColBERT-style MaxSim against a dedicated
  `mempalace_context_views` Chroma collection.
- Thresholds (Zhang/Ramakrishnan/Livny 1996 BIRCH-style):
  T_reuse=0.90, T_similar=0.70. Reuse → return existing id. Mid-window
  → new id + `similar_to` edge to nearest neighbour with sim in
  `confidence`. Below window → fresh id.
- Seeder: `context` class (is_a `thing`), `created_under` predicate
  (entity/class/predicate/literal/record → context, many-to-one),
  `similar_to` predicate (context → context, many-to-many).
- Migration 014: four partial indexes for the hot read paths.
- Helpers: `context_lookup_or_create` (mcp_server.py),
  `get_similar_contexts(hops=2, decay=0.5)` (knowledge_graph.py).
- Writers: `_add_memory_internal`, `tool_kg_declare_entity` now write
  `created_under` to the active context.
- Tests: 21 new (10 migration, 6 accretion, 5 emit-sites).

### Phase 2 — Atomic feedback + scoring cutover (PR #58)

- **Seeder cutover**: found_useful/found_irrelevant removed; surfaced,
  rated_useful, rated_irrelevant added.
- **Migration 015**: DROP TABLE keyword_feedback + edge_traversal_feedback;
  DELETE FROM scoring_weight_feedback (cold-start for P3 learning);
  ALTER TABLE triples ADD COLUMN properties TEXT; 6 partial indexes for
  the new predicates.
- **Migration 016**: keyword_idf table (BM25-ready, backfilled from
  entity_keywords scoped to kind='record').
- **Retrieval becomes 4-channel weighted RRF**. Channel seeds:
  cosine=1.0, graph=0.7, keyword=0.8, **context=1.5** (Channel D).
  `rrf_merge` now accepts `(list, weight)` entries; plain lists fall
  back to weight=1.0 for legacy callers. Reference: Bruch/Gai/Ingber
  2023 ACM TOIS.
- **Channel D**: `_build_context_channel` aggregates over the active
  context + its 1–2 hop `similar_to` neighbourhood. rated_useful
  positive, rated_irrelevant negative (drops below 0), surfaced weak
  positive (decay=0.3).
- **hybrid_score prune**: W_REL retired; four-weight split `sim=0.50,
  imp=0.22, decay=0.15, agent=0.13` (user-approved).
- **Surfaced edges**: `tool_kg_search` writes
  `(active_context, surfaced, entity_id)` with props
  `{ts, rank, channel, sim_score}` for every top result.
- **Finalize map shape**: `memory_feedback` now accepts EITHER a flat
  list (legacy) OR `{context_id: [entries]}` (new). Dual-write:
  legacy found_useful/found_irrelevant → execution entity AND new
  rated_useful/rated_irrelevant → context entity (Channel D's read
  path). Non-list dict values fail loud.
- **Retirement stubs** for `record_edge_feedback`,
  `get_edge_usefulness`, `get_recent_rejection_reasons`,
  `get_context_ids_for_edge` — safe no-op defaults so callers compile.
- **PALACE_PROTOCOL** gets a CONTEXT-AS-ENTITY section explaining the
  substrate, the four channels, and the surfaced/rated_* writes.
- Tests: 6 new for Channel D + weighted-RRF variant; existing
  test_migrations / test_mcp_startup / test_enrichment_auto_accept
  updated for the new schema + contract.

### Phase 3 — Telemetry + eval harness + weight-learning (PR #59)

- **JSONL telemetry** from `tool_kg_search` and `tool_finalize_intent`
  into `~/.mempalace/hook_state/{search,finalize}_log.jsonl`.
- **`mempalace.eval_harness`** module: context reuse rate + per-channel
  contribution + finalize stats + pretty-printer.
- **CLI**: `mempalace eval [--report | --reuse-rate] [--days N]
  [--json]`.
- **Weight-learning re-enabled** (`_A6_WEIGHT_SELFTUNE_ENABLED = True`).
  Post-P2 cold-start means first N intents seed the table; after
  min_samples=10 the learner drifts weights ±30% per feedback
  correlation.
- **Keyword-suppression stubs**: `record/get/reset_keyword_suppression`
  collapsed to no-ops (BM25-IDF replaces them).
- Tests: 11 new (7 eval_harness + 4 channel_weight_learning).

## Verification steps after reinstalling the plugin

```bash
# 1. Sanity: cold palace boot
rm -rf ~/.mempalace/*           # optional — fresh slate
# (re-install the mempalace plugin in Claude Code)

# 2. Confirm migrations applied (inside a shell that can see the palace)
python -c "
from mempalace.knowledge_graph import KnowledgeGraph
kg = KnowledgeGraph()
cols = {r[1] for r in kg._conn().execute('PRAGMA table_info(triples)').fetchall()}
tables = {r[0] for r in kg._conn().execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()}
assert 'properties' in cols, 'migration 015 did not add triples.properties'
assert 'keyword_idf' in tables, 'migration 016 did not create keyword_idf'
assert 'edge_traversal_feedback' not in tables, 'migration 015 did not drop edge_traversal_feedback'
assert 'keyword_feedback' not in tables, 'migration 015 did not drop keyword_feedback'
print('OK — migrations 014/015/016 applied')
"

# 3. In a live Claude Code session:
#    mempalace_wake_up(agent="ga_agent")
#    mempalace_declare_intent(intent_type="research", slots={...},
#        context={"queries":["A","B"], "keywords":["a","b"]},
#        agent="ga_agent", budget={...})
#    mempalace_kg_query(entity="<active_context_id returned by declare>")
# Expect: kind=context, properties with queries/keywords/entities.

# 4. Declare a second intent with near-identical context → should REUSE
#    the same context_id (verify via kg_stats that kind=context count
#    did not increment).

# 5. Run a search that returns N results, then kg_query the context —
#    expect N `surfaced` outgoing edges with rank/channel/sim_score
#    properties.

# 6. Finalize with the map shape:
#    memory_feedback={<context_id>: [{id, relevant, relevance, reason}]}
#    Then kg_query the context — expect rated_useful / rated_irrelevant
#    edges to the rated memories.

# 7. Eval CLI:
mempalace eval --report
# Expect a non-zero reuse rate + channel table after a few intents.
```

## Open questions (all 6 settled during the run)

1. `edge_traversal_feedback` → retire fully.  **DONE** (migration 015).
2. Eval-harness storage → JSONL.  **DONE**.
3. Hybrid weights post-W_REL → `sim=0.50, imp=0.22, decay=0.15,
   agent=0.13`.  **DONE**.
4. Rocchio view-cap → 20 views LRU.  **Pending — see deferred list.**
5. Split P2 further? → No, keep P2 atomic.  **DONE** — single PR #58.
6. Learned weights (raised mid-run) → global, re-enable in P3, cover
   both hybrid + channel weights under one `weight_feedback(scope,
   component, value, was_useful)` mechanism.  **PARTIAL — hybrid
   learner re-enabled; channel-weight learner deferred (see below).**

## Concrete deferred items for next day

Everything below is tracked as a follow-up task you can execute
independently. None of it is on the critical path for exercising the
new retrieval pipeline today.

1. **Per-channel RRF weight learning (open question #6 follow-up).**
   Extend the existing weight-learner to a `scope IN ('hybrid',
   'channel')` shape. Channel learner needs a new emit site during
   retrieval that records "did channel X contribute to a useful
   memory?" signal. Uses the same ±30% damped correlation mechanism.
2. **BM25-IDF scoring in `_build_keyword_channel`.** Table +
   backfill shipped in P2 (migration 016). Incremental update from
   `_add_memory_internal` is the wiring step; per-term IDF-weighted
   scoring formula `sum(idf(matched_kw))` replaces today's
   `overlap_ratio` sum.
3. **Degree-dampening in `_build_graph_channel`.**
   `get_entity_degree` helper already ships in P2. Wire each
   seed→memory contribution as `1 / log(degree(seed) + 2)` — one-liner
   edit at the score line.
4. **Rocchio enrichment on reused contexts** at finalize time. When a
   reused context gets a majority-positive finalize (mean relevance
   ≥ 4), merge orphan queries / keywords / entities from this intent's
   active context into the reused context entity. Cap 20 views with
   LRU eviction (settled open question #4).
5. **Strict coverage validator** on the finalize map shape: every
   `(context, memory)` pair with a `surfaced` edge during this intent
   must have a `rated_useful` or `rated_irrelevant` edge. Today, the
   validation loud-rejects shape errors but does not enforce
   per-pair coverage.
6. **Bulk-retirement sweep** of the remaining call sites:
   - `promote_to_type` flag on finalize
   - `lookup_type_feedback` in `scoring.py`
   - `_relevance_boost`, `_memory_scoring_snapshot`,
     `_channel_attribution` in intent.py and mcp_server.py
   - `feedback_context_id = f"ctx_{slug}"` in intent.py
   - `store_feedback_context` / `persist_context` (old pipeline)
   - `_rejection_suppresses_enrichment`
   - `record_edge_feedback` / `get_edge_usefulness` call sites
     (functions are now no-op stubs; removing the sites tightens the
     codebase)
7. **`mempalace_feedback_contexts` Chroma collection drop**. SQL
   migration 015 can't drop Chroma collections. A server-startup
   one-shot hook in `mcp_server._run_migrations` should delete the
   collection on first boot after P2 ships (idempotent; gated on a
   flag on `ServerState`).
8. **Reify triples as `kind="triple"` entities** so `tool_kg_add` can
   also emit `created_under` edges (currently deliberately skipped;
   triples keep the `creation_context_id` column for provenance).

## Execution notes

- Deviation from plan: `context_lookup_or_create` lives in
  `mcp_server.py`, not `knowledge_graph.py` as the plan originally
  stated. Rationale: the function needs Chroma access, and
  knowledge_graph.py is the pure-SQL layer. `get_similar_contexts` is
  pure SQL, so it correctly lives in `knowledge_graph.py`. Every
  consumer still works; net impact nil.
- Similar-to edge sim is stored in the `confidence` column (triples
  table has no generic properties in P1). P2's migration 015 added a
  generic `properties TEXT` column, so future edges (surfaced,
  rated_*) carry rich JSON props. The similar_to `confidence`-channel
  convention is left in place to avoid a schema rewrite.
- The P2 atomic cutover was delivered in 4 commits (not 16) because
  ruff/pre-commit enforces pytest-unit-green per commit, and chopping
  the changes into 16 commits would have left the tree in
  intermediate broken states. Net effect is the same — every landed
  piece has a consumer in the same PR.
- The P2 retirement sweep was partially done (edge-feedback API
  stubbed) but the call-site removal is deferred — keeping the stubs
  lets the existing integration tests stay green without wholesale
  test-migration in this run. Morning follow-up item #6.

---

Adrian — if you hit anything unexpected, the full chat transcript lives
at `C:\Users\adria\.claude\projects\D--Flowsev-mempalace\` under the
most recent session id. The PR bodies (57, 58, 59) also describe
what's in each.
