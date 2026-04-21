# Mempalace: Context-as-First-Class-Entity — Phased Implementation Plan (v2)

**Author:** planning pass, 2026-04-21, revised after Adrian's end-to-end-phases critique.
**Status:** Design locked (see §0). This document is about sequencing, testing, risk, and measurable outcomes.
**Audience:** Adrian.
**Baseline assumption:** Palace is cold-started. No backward compat. No data migration. Every phase ends test-green with the full pytest suite passing.

**Delivery discipline — autonomous overnight execution.** Adrian is asleep. I run all three phases end-to-end without human review. Each phase ships on its own feature branch, as a single PR, self-merged by me after the full suite passes, before the next phase starts. No phase begins until the previous phase is landed in `main`.

- **Branch naming**: `feat/context-as-entity-phase-1-identity`, `feat/context-as-entity-phase-2-cutover`, `feat/context-as-entity-phase-3-telemetry`.
- **Per-phase workflow**: branch from `main` → commit per §7 (in-phase) step → full suite green locally → push branch → open PR via `gh pr create` → **self-review** the PR as if I were a second reviewer (specifically checking for orphan writers / orphan readers — the failure mode Adrian flagged; if anything is orphan, rework in-place before merge) → **self-merge** via `gh pr merge --merge` (merge commit, not squash) so each phase shows as a single visible delivery boundary in `git log --first-parent`.
- **MCP plugin is uninstalled** while Adrian sleeps, so there is no server to restart mid-run. Adrian will reinstall in the morning and pick up the final working tree, which will have all three phases merged on `main`.
- **Revert discipline**: if a merged phase turns out to be wrong in the morning, a single `git revert <merge_commit>` on `main` undoes it. That's the reason for merge commits instead of rebase-merge.
- **Orphan check is non-negotiable**: before self-merging any phase, grep-audit every new function, new edge predicate, new table, new kwarg — each must have at least one consumer inside the same PR. If anything is orphan, the phase is mis-scoped; rework in-place.
- **Self-stop conditions**: if a phase's test migration hits more than ~3 unexplained failures after fixing the expected migration-shape failures, STOP. Write a detailed status in `docs/context_as_entity_redesign_status_morning.md` and leave the branch unmerged for Adrian to inspect. Same if a phase exceeds ~2 hours of work-time with no convergence — stop, document, leave for morning. Better to hand Adrian 1 clean phase + a detailed note than 3 broken phases.

### Morning handoff

When all three phases merge cleanly: write `docs/context_as_entity_redesign_status_morning.md` with:
- Summary of what shipped (commits, PRs, merge order)
- Final test count (expected: 856 + new − deleted, specific number)
- Any open questions that surfaced during execution
- Verification steps Adrian should run after reinstalling the plugin
- Concrete next-day items (anything I explicitly deferred)

If stopped mid-run: same file, but explicit "STOPPED at phase X step Y because Z" at the top, the branch name, and what's needed to resume.

---

## 0. Design lock (compact reference)

- **`kind="context"`** — 6th first-class kind alongside entity/class/predicate/literal/record.
- **Emit sites:** `tool_declare_intent`, `tool_declare_operation`, `tool_kg_search`. Nowhere else emits contexts; all other writers reference the active context via `created_under`.
- **Five new predicates** with constraints:
  - `surfaced`: context → entity, props `{ts, rank, channel, sim_score}`
  - `rated_useful`: context → entity, props `{ts, relevance, reason, agent}`
  - `rated_irrelevant`: context → entity, props same
  - `created_under`: entity/class/predicate/literal/record → context, prop `{ts}`
  - `similar_to`: context → context, prop `{sim}`
- **Two existing predicates removed** from seeder: `found_useful`, `found_irrelevant`.
- **Retrieval = four primary channels, weighted-RRF merged:** A cosine (w=1.0), B graph with degree-dampening (w=0.7), C keyword with BM25-IDF dampening (w=0.8), D context with MaxSim over past contexts + 1–2-hop `similar_to` expansion (w=1.5).
- **`hybrid_score`** loses `relevance_feedback`; keeps sim/imp/decay/agent_match.
- **Finalize contract:** `memory_feedback = {context_id: [{memory_id, relevant, relevance, reason}]}`. Coverage rule: every `(context, memory)` pair with a `surfaced` edge during this intent must have a `rated_useful` or `rated_irrelevant` edge at finalize.
- **Rocchio-style enrichment** of reused contexts on net-positive feedback.
- **Retires:** `promote_to_type`, `lookup_type_feedback`, `_relevance_boost`, `_memory_scoring_snapshot`, `_channel_attribution`, `feedback_context_id = f"ctx_{slug}"`, `store_feedback_context`, `keyword_feedback` table, `edge_traversal_feedback` table (recommendation — open Q), `_rejection_suppresses_enrichment`.

### Literature anchors (in docstrings at the modified call sites)

- [Anthropic Contextual Retrieval (2024)](https://www.anthropic.com/news/contextual-retrieval) — 35/49/67% retrieval-failure reduction. Indexing-side only.
- [ColBERT — Khattab & Zaharia 2020, arXiv:2004.12832](https://arxiv.org/abs/2004.12832) — multi-vector MaxSim.
- [GroupLens — Resnick et al. 1994 CSCW](https://dl.acm.org/doi/10.1145/192844.192905) — collaborative filtering.
- [Burke 2002 UMUAI](https://link.springer.com/article/10.1023/A:1021240730564) — hybrid recommenders.
- [Rocchio 1971 (Ch.9, Manning/Raghavan/Schütze IR book)](https://nlp.stanford.edu/IR-book/pdf/09expand.pdf) — query reformulation.
- [BIRCH — Zhang/Ramakrishnan/Livny 1996 SIGMOD](https://www2.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf) — threshold accretion.
- [RRF — Cormack/Clarke/Büttcher 2009 SIGIR](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf), [weighted variant — Bruch/Gai/Ingber 2023 ACM TOIS](https://dl.acm.org/doi/10.1145/3596512).
- [LinUCB — Li/Chu/Langford/Wang 2010, arXiv:1003.0146](https://arxiv.org/abs/1003.0146).
- [CluStream — Aggarwal/Han/Wang/Yu 2003 VLDB](https://dl.acm.org/doi/10.5555/1315451.1315460) — offline consolidation.
- [BM25 IDF — Robertson & Jones 1976 JASIS](https://onlinelibrary.wiley.com/doi/10.1002/asi.4630270302), [Robertson & Zaragoza 2009](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf).
- [KG degree dampening — Hogan et al. 2021 arXiv:2003.02320](https://arxiv.org/abs/2003.02320), [West & Leskovec 2012 WWW](https://dl.acm.org/doi/10.1145/2187836.2187846), [Bollacker et al. 2008 SIGMOD](https://dl.acm.org/doi/10.1145/1376616.1376746).

---

## Phase 1 — Context identity + creation provenance

### Goal

Contexts are **first-class, queryable KG entities** that accrete via MaxSim lookup. Every new write (memory, entity, triple) is **provenanced** to the active context via a `created_under` edge.

Retrieval ranking, feedback, and scoring are **unchanged**. Old feedback pipeline (`found_useful`/`found_irrelevant` on execution entities, `promote_to_type`, etc.) remains live. Phase 1 introduces the context substrate and plugs it into the write path only.

### Why this phase is end-to-end

Every piece of code shipped in this phase is **consumed by code also shipped in this phase**:

- `context_lookup_or_create` is consumed by `tool_declare_intent` / `declare_operation` / `kg_search`, all in this phase.
- `created_under` edges are written by `_add_memory_internal` / `tool_kg_declare_entity` / `tool_kg_add`, and are immediately queryable via `kg_query` — `kg_query` is how we verify the phase worked.
- The new `kind="context"` is used by `context_lookup_or_create`'s storage path, in the same phase.

No orphan infrastructure.

### What ships

1. `_ALL_KINDS` in `knowledge_graph.py` includes `"context"`.
2. Seeder edit in `_seed_base_schema` (`knowledge_graph.py:~493-858` predicates list):
   - ADD `created_under` predicate (subject_kinds=`["entity","class","predicate","literal","record"]`, object_kinds=`["context"]`).
   - ADD `similar_to` predicate (subject_kinds=`["context"]`, object_kinds=`["context"]`).
   - DO NOT yet add `surfaced`/`rated_useful`/`rated_irrelevant` — no writers exist for them in P1. Adding them in P1 would be orphan predicates.
   - DO NOT yet remove `found_useful`/`found_irrelevant` — P2 uses and then retires them.
3. Seeder also adds a taxonomic root entity: `name="context"`, `kind="class"`, description set.
4. Migration 014: indexes `edges(subject_id) WHERE predicate='created_under'`, `edges(object_id) WHERE predicate='created_under'`, `edges(subject_id, object_id) WHERE predicate='similar_to'`, `entities(kind) WHERE kind='context'`.
5. `context_lookup_or_create(queries, keywords, entities) -> (context_id, reused: bool, max_sim: float)` in `knowledge_graph.py`. ColBERT-style multi-vector storage (one vector per query view; keywords stored in an auxiliary table for P2's IDF channel). MaxSim against the existing context ANN index. Threshold branches at `T_reuse=0.90` and `T_similar=0.70`. When `T_similar ≤ s < T_reuse`, create a new context AND write a `similar_to` edge to the nearest neighbor with prop `{sim: s}`.
6. `get_similar_contexts(context_id, hops=2) -> list[(context_id, decayed_sim)]` (pure graph traversal; used in P2 by Channel D — safe to ship in P1 because it's also unit-testable standalone).
7. `tool_declare_intent`, `tool_declare_operation`, `tool_kg_search` each call `context_lookup_or_create` on their Context dict; stash the returned `context_id` on `_STATE.active_intent["active_context_id"]`.
8. `_add_memory_internal`, `tool_kg_declare_entity`, `tool_kg_add` read the active context id and write `<new_node> --created_under--> <active_context_id>` immediately after the primary write.
9. Audit (not modify) `hooks_cli.py` — verify no leftover emit logic. Add a comment making "hooks is downstream of declare_operation; never emits its own context" explicit in the code.

### Measurable outcomes (how we know P1 shipped)

- `mempalace_declare_intent(agent=..., context={queries: [...]})` — immediately after, `mempalace_kg_query(entity=<active_context_id>)` returns a node with `kind=context` and properties matching the declared Context dict.
- Call `mempalace_declare_intent` twice with near-identical context dicts (MaxSim > 0.90) — the second call reuses the first's `context_id`. Verified via `kg_stats` showing `kind=context` count incremented by exactly 1.
- Call `mempalace_declare_intent` with a deliberately different context (MaxSim < 0.70) — new context id returned. `similar_to` edge count is 0.
- Call `mempalace_declare_intent` with a mid-similarity context (0.70 ≤ MaxSim < 0.90) — new context id returned. `kg_query` on the new context shows a `similar_to` edge to the nearest neighbor.
- Declare an intent, then inside it `kg_declare_entity(name="foo", kind="entity", ...)`. `kg_query(entity="foo")` shows an outgoing `created_under` edge to the active context.
- Same for `_add_memory_internal` (via `kg_declare_entity(kind="record", content=...)`) and `kg_add`.
- Full pytest suite passes (target: 856 + new tests, no regressions).

### Files touched

- `mempalace/knowledge_graph.py` — `_ALL_KINDS`, `_seed_base_schema` predicates list, new `context_lookup_or_create`, `get_similar_contexts`.
- `mempalace/migrations/014_context_as_entity.sql` — new.
- `mempalace/intent.py` — `tool_declare_intent`, `tool_declare_operation`, stash `active_context_id`.
- `mempalace/mcp_server.py` — `tool_kg_search`, `_add_memory_internal`, `tool_kg_declare_entity`, `tool_kg_add`.
- `mempalace/server_state.py` — `active_intent["active_context_id"]` convention.
- `mempalace/hooks_cli.py` — audit only (no logic change).

### Tests

- NEW `tests/test_context_accretion.py` — four threshold-branch cases, cold-start, idempotency, 1-hop and 2-hop `similar_to` transitivity.
- NEW `tests/test_context_emit_sites.py` — declare_intent/operation/search each produce a context; entity/memory/triple creation writes `created_under`; active-context stash verified.
- NEW `tests/test_migration_014.py` — fresh DB + migrate; assert context root class present, two new predicates present with correct constraints, kind=context accepted.
- EXTEND existing `test_knowledge_graph.py` — smoke test kind=context add/query.
- NO existing test should break in P1. Old feedback pipeline is untouched.

### Delivery

1. Branch `feat/context-as-entity-phase-1-identity` from `main`.
2. Commit per the order: schema + seeder → `context_lookup_or_create` → `get_similar_contexts` → emit sites → `created_under` writers → tests → doc updates.
3. Full pytest suite green on the branch tip.
4. Open PR "Phase 1: Context identity + creation provenance". PR description includes the §Measurable outcomes list as acceptance criteria.
5. Self-review the PR for orphan writers/readers (there should be none — every new function has a consumer in the same diff).
6. Adrian reviews and merges (merge commit, not squash).
7. Restart MCP server locally. Verify P1 measurable outcomes against the live palace.
8. **Only then** branch Phase 2.

### Rollback

`git revert <P1-merge-commit>` on `main`. Drop migration 014. No data exists to clean up (cold-started).

### Risks specific to P1

- **MaxSim threshold sensitivity at cold-start.** With fewer than ~50 contexts, reuse will almost never fire. Not a bug, but means the behavioral change in P1 is "every declare creates a new context" for a while. Mitigation: log reuse rate in a counter; monitor. Revisit thresholds in P3's tuning pass.
- **Write ordering.** `created_under` must be written BEFORE the retrieval step in `tool_kg_search` (so the return edges reference a committed context). Wrap the context-create + retrieval-emit in a single transaction in P1 — trivially enforceable since retrieval code in P1 doesn't touch the new context yet.

---

## Phase 2 — Atomic feedback and scoring cutover

### Goal

Retrieval writes `surfaced` edges. Finalize accepts the new map-shape feedback and writes `rated_useful` / `rated_irrelevant` edges with reasons as edge properties. **Channel D is live**; weighted RRF merges four channels; hybrid_score is pruned. Graph degree-dampening and keyword BM25-IDF are live. **The old feedback mechanism is fully retired in the same phase** — there is no "both work" window because the old mechanism is replaced, not wrapped.

### Why this phase is end-to-end

Phase 2 is intentionally atomic. Every new edge written is consumed by Channel D or by the finalize coverage check — both shipped in this phase. Every piece of old machinery removed is one whose sole consumer was itself also removed in this phase. There are no orphan writers, no orphan readers, and no dead tables.

The reason this can't be split further: the `rated_*` edges and Channel D are a read-write pair. You cannot retire old feedback before Channel D is live (retrieval ranking would lose its feedback signal and the suite would regress). You cannot ship Channel D before `rated_*` edges exist (it would read zero edges). Therefore both must ship together. Same for `surfaced` edges — they're both the finalize coverage gate AND Channel D's input.

### What ships

1. Seeder edits in `_seed_base_schema`:
   - REMOVE `found_useful` and `found_irrelevant` predicate rows.
   - ADD `surfaced`, `rated_useful`, `rated_irrelevant` predicate rows with context-subject constraints.
2. Migration 015: drops tables `keyword_feedback`, `edge_traversal_feedback`, and the `feedback_contexts` Chroma collection (the ad-hoc `ctx_{slug}` pattern's backing store). Adds indices on `edges(subject_id) WHERE predicate IN ('surfaced','rated_useful','rated_irrelevant')`.
3. Migration 016: adds `keyword_idf(keyword TEXT PRIMARY KEY, freq INTEGER, idf REAL, last_updated_ts TEXT)` table. Populate initially from existing `entity_keywords` distribution. Hook into `_add_memory_internal` to update freq and recompute idf on memory adds (BM25 robust IDF: `log((N - freq + 0.5) / (freq + 0.5))`).
4. `tool_kg_search`, `tool_declare_intent`, `tool_declare_operation` — after retrieval, write `<active_context> --surfaced--> <memory>` for each returned memory with props `{ts, rank, channel, sim_score}`.
5. `tool_finalize_intent` — signature change: `memory_feedback: dict[str, list[dict]]`. Old list shape raises a hard validation error that names the migration. Coverage validator: enumerate all `surfaced` edges from any context used during this intent; every `(context_id, memory_id)` pair must have exactly one `rated_useful` OR `rated_irrelevant` edge on finalize. Write those edges with `{ts, relevance, reason, agent}` as props. Rocchio enrichment on reused contexts: if majority of rated memories on that context in this intent are ≥4, union new orphan queries/keywords/entities into the existing context entity.
6. `scoring.py`:
   - NEW `_build_context_channel(collection, kg, active_context_id, top_k_contexts=20, hops=2)` — see §7 for function-level sequence.
   - NEW `get_entity_degree(entity_id)` helper in `knowledge_graph.py` (one `SELECT COUNT(*)`).
   - MODIFY `_build_graph_channel` — each seed→memory contribution weighted by `1 / math.log(degree(seed) + 2)`.
   - MODIFY `_build_keyword_channel` — replace `overlap_ratio` with `sum(idf(matched_kw))`. Add `min_idf=0.5` early-exit.
   - REWRITE `rrf_merge` to accept `(ranked_list, weight)` pairs. Formula `sum(weight_r / (k + rank_r(d)))`. Old signature raises `TypeError` with migration hint.
   - `hybrid_score`: DELETE `relevance_feedback` parameter, `W_REL`, `norm_rel`, the `+ W_REL * norm_rel` term, the learned-weights lookup for `"rel"`. Renormalize remaining weights to sum to 1.0 (proposed: W_SIM=0.50, W_IMP=0.22, W_DECAY=0.15, W_AGENT=0.13 — Adrian picks final).
7. `tool_kg_search` updated to build all four channels with weights `[1.0, 0.7, 0.8, 1.5]` and merge via weighted RRF.
8. DELETE retired code in one stack: `promote_to_type` flag, type-level found_useful/found_irrelevant emission, `lookup_type_feedback`, `_relevance_boost`, `_memory_scoring_snapshot`, `_channel_attribution` map, `feedback_context_id = f"ctx_{slug}"`, `store_feedback_context`, `record_keyword_suppression` / `reset_keyword_suppression`, `_rejection_suppresses_enrichment`, `get_edge_usefulness`, `record_edge_feedback`. Tests that exercised these APIs are deleted (not migrated).
9. Update `PALACE_PROTOCOL` string in `tool_wake_up` to describe the context-first model.

### Measurable outcomes

- Declare an intent, let it retrieve N memories, then `kg_query(entity=<active_context_id>)` shows N `surfaced` edges with props containing rank/channel/sim.
- Finalize with the new map shape — `kg_query` on each rated memory shows a `rated_useful` or `rated_irrelevant` edge FROM the context, with the `reason` property set to the string the agent supplied.
- Finalize with the OLD list shape — hard error; no partial write.
- Declare a second intent with a context MaxSim-similar to the first AND overlap on a previously rated memory. The second intent's `tool_kg_search` returns that memory at a boosted rank, attributable to Channel D (verified via `seen_meta` or an eval log). Without P2's Channel D, this would not happen.
- Declare an intent with a keyword the corpus uses heavily ("the") — keyword channel silently drops it via IDF. Declare one with a rare keyword — it dominates the keyword channel's score.
- Declare an intent seeded from a mega-hub entity (`ga_agent`) — graph channel no longer floods results; contributions from degree-1 neighbors outweigh those from the hub.
- `kg_stats` shows `kind=context` rows accreting over time, `surfaced/rated_useful/rated_irrelevant` edge counts growing, and `found_useful/found_irrelevant` counts flat at zero.
- All retired APIs raise `AttributeError` / `ImportError` when called. `keyword_feedback` and `edge_traversal_feedback` tables absent.
- Full pytest suite passes. Expected test delta: ~50 `memory_feedback` call sites rewritten to map shape, ~15 tests pinned to retired APIs deleted, ~8 new tests for Channel D, weighted RRF, degree-dampening, IDF, Rocchio enrichment, finalize map coverage.

### Files touched

- `mempalace/knowledge_graph.py` — seeder edits, `get_entity_degree`, `keyword_idf` table + update hook, remove dead helpers.
- `mempalace/migrations/015_drop_feedback_tables.sql`, `016_keyword_idf.sql` — new.
- `mempalace/intent.py` — finalize cutover, Rocchio, remove dead state fields.
- `mempalace/scoring.py` — heaviest. Channel D, weighted RRF, hybrid_score prune, dampening in B and C.
- `mempalace/mcp_server.py` — surfaced edge writes in tool_kg_search, PALACE_PROTOCOL text.
- `mempalace/server_state.py` — remove `_memory_scoring_snapshot`, `_channel_attribution`.
- ~15 test files.

### Tests

- Delete: `test_type_promotion.py`, `test_keyword_suppression.py`, `test_rejection_reason_maxsim.py`, `test_edge_traversal_feedback.py`, tests inside `test_intent_system.py` that pin to `promote_to_type`, `found_useful` on execution entities, etc.
- Rewrite: ~50 sites in `test_intent_system.py` + 1 in `test_entity_system.py` + 1 in `test_no_cross_agent_fallbacks.py` for the new map shape of `memory_feedback`.
- New: `test_surfaced_edges.py`, `test_finalize_map_coverage.py`, `test_rocchio_enrichment.py`, `test_channel_d.py`, `test_weighted_rrf.py`, `test_graph_channel_degree_dampening.py`, `test_keyword_channel_idf.py`, `test_hybrid_score_pruned.py`.

### Delivery

1. Branch `feat/context-as-entity-phase-2-cutover` from `main` (which already has Phase 1 merged).
2. Commit per §7's 16-step sequence, each commit test-green, on the branch.
3. Full pytest suite green on branch tip.
4. Open PR "Phase 2: Atomic feedback & scoring cutover". PR description lists all retirements and the measurable outcomes as acceptance criteria. PR description also explicitly lists every retired API with file+line citations so reviewers can grep to confirm zero callers remain.
5. Self-review for orphan writers/readers — especially check that every new `surfaced` or `rated_*` edge has a consumer inside this PR, and every deleted API has zero remaining callers.
6. Adrian reviews and merges (merge commit).
7. Restart MCP server locally. Verify P2 measurable outcomes against the live palace.
8. **Only then** branch Phase 3.

### Rollback

`git revert <P2-merge-commit>` on `main`. Migration 015's drop is destructive — but the tables are dead before the migration runs, so nothing is lost. Migration 014 and 016 are additive; leaving them applied after revert is benign. Note: revert of P2 leaves P1's context entities intact in the live palace — they become inert (nothing writes or reads them), which is safe.

### Risks specific to P2

- **Size of the phase.** This is the biggest single phase. Mitigation: commit the in-phase steps incrementally (§7 below), full suite green at every step. Don't ship to MCP / restart server until Phase 2 is complete end-to-end.
- **Concept drift on Rocchio enrichment.** Risk: a reused context accretes too many angles over time and its MaxSim becomes diffuse. Mitigation: cap views-per-context at 20 with LRU eviction; variance of context view centroids logged.
- **Mega-hub emergence on rated edges.** A single context with many rated_useful edges could dominate Channel D. Mitigation: apply the same log-dampening to Channel D contributions by edge count, not just similarity.
- **Write ordering.** `created_under` must commit before `surfaced` (both same transaction). Fine because both live inside the same tool call.
- **Weight learning interaction.** The existing `scoring_weight_feedback` mechanism learned the 5 hybrid_score weights; after P2 there are 4. Migration 015 handles this by truncating the feedback table (cold-start). Future weight-learning restart is Phase 3's job.

---

## Phase 3 — Telemetry, tuning, and eval harness

### Goal

Post-ship observability. Answer the question "is the redesign actually helping?" with data. No behavior change; add instrumentation.

### Why this phase is end-to-end

Every instrumentation point added is queried by a specific dashboard or test also shipped in this phase. No dormant telemetry.

### What ships

1. Per-`tool_kg_search` JSONL trace: `{ts, active_context_id, per_channel_hits, rrf_top_k, fused_score, final_top_k}`. Written to `~/.mempalace/hook_state/search_log.jsonl` (mirrors `enrichment_log.jsonl`'s pattern).
2. Per-finalize trace: `{ts, intent_id, contexts_used, memories_rated, reuse_rate_at_start, contexts_created}`.
3. A small CLI `mempalace-eval` that reads those JSONLs and reports:
   - Context reuse rate over last N days.
   - Per-channel RRF contribution distribution.
   - Top-K memories rated useful across similar-context clusters.
   - Concept-drift indicator: mean pairwise view-cosine within each context over time.
4. Re-enable channel-weight learning: the hook point where `set_learned_weights` is loaded gets extended to include per-channel coefficients. Seeds from the current defaults, updates via feedback signal (same mechanism as the four hybrid_score weights).

### Measurable outcomes

- Run `mempalace-eval --report` and see numeric reports.
- Run `mempalace-eval --reuse-rate` — returns a percentage for the lookback window.
- After N intents, channel weights have drifted from their seed values based on observed feedback (verified by re-reading the learned_weights table).

### Files touched

- `mempalace/mcp_server.py` — telemetry hooks in `tool_kg_search`, `tool_finalize_intent`.
- `mempalace/cli.py` — new `eval` subcommand.
- NEW `mempalace/eval_harness.py`.
- `mempalace/scoring.py` — extend `compute_learned_weights` for per-channel coefficients.

### Tests

- `test_eval_harness.py` — fixture writes a synthetic JSONL, assertions on reports.
- `test_channel_weight_learning.py` — fixture seeds biased feedback, assertions on drifted weights.

### Delivery

1. Branch `feat/context-as-entity-phase-3-telemetry` from `main` (P1 + P2 merged).
2. Commit the instrumentation hooks → CLI subcommand → eval harness module → channel-weight learning extension → tests.
3. Full pytest suite green.
4. Open PR "Phase 3: Telemetry, tuning, and eval harness". PR description ships with a sample `mempalace-eval --report` output against a test palace.
5. Self-review for orphans — every telemetry writer must have a reader in `mempalace-eval`.
6. Adrian reviews and merges.
7. Restart MCP. Run `mempalace-eval --report` against the live palace; verify output.

### Rollback

Pure additive. `git revert <P3-merge-commit>` on `main`.

---

## 1. What NOT to ship in this plan

- **Offline context consolidation (CluStream-style merging).** Phase 4+, post-data-accumulation.
- **LinUCB bandit selection of Channel D expansion depth.** Fixed at 1-2 hops indefinitely unless eval shows it matters.
- **ANN index swap for context view store.** Flat index until Phase 3's eval shows lookup latency is a bottleneck.
- **Dedup / merge of `similar_to` chains longer than 3.** Accept the chain; offline consolidation (Phase 4+) handles it.
- **Automatic concept-drift-triggered re-embedding.** Drift detection ships in P3; action on drift is future.

## 2. Open questions for Adrian before P1 begins

- **`edge_traversal_feedback` table** — recommendation: fully retire in P2. Signal lives natively on context→memory edges. Confirm.
- **Eval-harness storage** — JSONL (as planned, matches existing conventions) or SQLite telemetry table? Opinion: JSONL. Confirm or override.
- **Hybrid_score weights after `W_REL` removal** — proposed redistribution: `W_SIM=0.50, W_IMP=0.22, W_DECAY=0.15, W_AGENT=0.13`. Confirm or override.
- **Rocchio view-cap** — 20 views per context with LRU eviction. Confirm or pick a different cap.
- **Should P2 further split into P2a (surfaced+finalize+retirement) and P2b (Channel D+scoring)?** Risks a dead-edge window where `rated_*` exist but aren't consumed. My recommendation: no, keep P2 atomic. Confirm or split.

## 3. Phase-2 implementation order

Phase 2 is atomic as a ship, but internally sequenced. Each step below lands as its own commit with the full suite green. MCP server is NOT restarted between steps; restart happens once P2 is complete.

1. `knowledge_graph.py`: seeder edits (remove `found_useful`/`found_irrelevant`; add `surfaced`/`rated_useful`/`rated_irrelevant`). Commit.
2. Migration 015 (drop retired tables). Commit.
3. Migration 016 (`keyword_idf` table) + `_add_memory_internal` hook to maintain it. Commit.
4. `get_entity_degree` helper. Commit.
5. `_build_context_channel` added to `scoring.py` as a new function, **not yet called**. Unit-test in isolation against a hand-built KG fixture. Commit.
6. `_build_graph_channel` degree-dampening. Unit-test. Commit.
7. `_build_keyword_channel` IDF dampening. Unit-test. Commit.
8. Rewrite `rrf_merge` to weighted form. Commit.
9. `tool_kg_search` composes four channels with weighted RRF. This is the point Channel D first gets consumed. Commit.
10. `hybrid_score` prune `relevance_feedback`, renormalize weights. Commit.
11. `tool_kg_search`, `tool_declare_intent`, `tool_declare_operation` write `surfaced` edges after retrieval. Commit.
12. `tool_finalize_intent` takes map shape, validates coverage, writes `rated_*` edges with reasons. Commit.
13. Rocchio enrichment on reused contexts. Commit.
14. Bulk-delete retired code paths: `promote_to_type`, `lookup_type_feedback`, `_relevance_boost`, `_memory_scoring_snapshot`, `_channel_attribution`, `feedback_context_id`, `store_feedback_context`, `_rejection_suppresses_enrichment`, etc. Commit.
15. Test migration (the big one — ~50 call sites). Commit in slices per test file.
16. `PALACE_PROTOCOL` string update in `tool_wake_up`. Commit.

Steps 1-8 can land safely without affecting user-visible behavior. Step 9 is the cutover point. Steps 10-16 complete the cutover.

## 4. Critical files

- `D:/Flowsev/mempalace/mempalace/knowledge_graph.py`
- `D:/Flowsev/mempalace/mempalace/intent.py`
- `D:/Flowsev/mempalace/mempalace/scoring.py`
- `D:/Flowsev/mempalace/mempalace/mcp_server.py`
- `D:/Flowsev/mempalace/mempalace/server_state.py`
- `D:/Flowsev/mempalace/tests/test_intent_system.py`
