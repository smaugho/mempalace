# Feedback System Audit — 2026-04-18

Live audit of the mempalace feedback-from-learning system against the running MCP server (version 3.0.14). Seven loops tested end-to-end (A1–A7) plus a `kg_search` edge-format review and bonus structural findings. Six bugs or material concerns discovered; three loops pass cleanly.

Plan driving the audit: `C:/Users/adria/.claude/plans/elegant-swimming-sunset.md`. Memory twin: `record_ga_agent_plan-feedback-system-audit-2026-04-18`.

## Summary table

| Loop | Result | Short reason |
|------|--------|--------------|
| A1 | PASS | `found_useful`/`found_irrelevant` → `lookup_type_feedback` → `hybrid_score` works end-to-end. Gap between +1 and –1 feedback == `W_REL` (0.200) exactly. |
| A2 | PASS | `promote_to_type=false` keeps edges on the execution entity only; `lookup_type_feedback` correctly ignores them. |
| A3 | PARTIAL | Keyword suppression primitives work. The trigger condition `channels == {"keyword"}` almost never fires in practice — RRF fusion lets cosine results dominate. `keyword_feedback` table has **0 rows** despite hundreds of finalizations. |
| A4 | **BUG** | Enrichment rejections ARE persisted and `get_edge_usefulness` returns –1.0 for repeat rejects, but the finalize-time enrichment generator at `intent.py:1982-2012` does NOT call `get_edge_usefulness` to filter. Same rejected pairs keep surfacing. |
| A5 | PASS-with-concern | BFS pruning correctly reads `get_edge_usefulness` at `intent.py:1154` with `_MIN_EDGE_USEFULNESS = –0.5`. Concern: structural edges (`is_a`, `executed_by`, `described_by`) can accumulate negative feedback when unrelated memories sit downstream of them. |
| A6 | **BUG** | `scoring_weight_feedback` only logs 3 of 5 components (`imp`, `decay`, `agent`). The two missing — `sim` and `rel` — are 60% of `DEFAULT_SEARCH_WEIGHTS`. `compute_learned_weights` produces ±0.1% adjustments on the 3 it has data for, and zero adjustment on the 60% it doesn't. |
| A7 | **BUG** | `last_relevant_at` decay-reset update silently no-ops. `intent.py:1828` normalizes the feedback id via `normalize_entity_name` (hyphens → underscores), then `col.get(ids=[normalized])` misses because Chroma stores hyphenated ids. Outer `try/except` swallows the empty result. Any memory with a hyphenated slug — i.e. essentially all of them — never gets its decay clock reset. |

Plus bonus structural findings (see below).

## A1 — `found_useful` loop [PASS]

Seeded `audit_type_a1` class + two test memories. Finalized with `memory_feedback=[{relevant:true, relevance:5, promote_to_type:true}, {relevant:false, relevance:5, promote_to_type:true}]`. All assertions passed:

- SQLite `triples` has `audit_type_a1 –found_useful→ audit-a1-memory-relevant (conf 1.0)` and `audit_type_a1 –found_irrelevant→ audit-a1-memory-irrelevant (conf 1.0)` plus matching edges on the execution entity. Confidence formula verified: relevance 5 → 1.0, 2 → 0.4, 1 → 0.2.
- `lookup_type_feedback({intent_type: 'audit_type_a1'}, kg)` returns `{mem_relevant: +1.0, mem_irrelevant: -1.0, plan: +1.0}`.
- `hybrid_score(similarity=0.7, ..., relevance_feedback=+1.0)` – `hybrid_score(..., relevance_feedback=-1.0)` == **0.2000** (= `W_REL`).

All primitives and consumption paths verified.

## A2 — `promote_to_type=false` scope [PASS]

Two feedback entries on memories with `promote_to_type` omitted (default false): `enforcement_audit_2026_04_17` (relevance 1) and `rule_plans_go_to_plan_mode_and_memory` (relevance 2). Assertions:

- Both edges land on the execution entity with correct confidence (0.2 and 0.4 respectively).
- Neither edge appears on `audit_type_a1` class.
- `lookup_type_feedback({intent_type: 'audit_type_a1'}, kg)` returns none of them.

## A3 — Keyword suppression [PARTIAL — trigger dead in practice]

Primitives verified:

- `kg.entity_ids_for_keyword('zqkx7audit')` returns the seeded memory.
- `keyword_lookup(kg, ['zqkx7audit'], collection=col)` returns the memory with `suppression=1.0`.
- `entity_keywords` table has the four indexed keywords from caller context.

**The trigger never fires.** Finalize-time code at `intent.py:1969` reads:

```python
if not was_relevant and channels == {"keyword"}:
    kg.record_keyword_suppression(...)
```

This requires the memory to be surfaced EXCLUSIVELY via the keyword channel (no semantic, no graph). In live retrieval:

- Each cosine view fetches `fetch_limit_per_view=50` results. Multiple views → hundreds of cosine candidates.
- The keyword channel for a narrow keyword returns 1–5 candidates.
- `rrf_merge` fusion gives the keyword-only hit an RRF contribution of `1/(60+0) ≈ 0.0167` — below the `adaptive_k` top-20 cutoff when cosine dominates.
- Net effect: keyword-only memories almost never appear in `injected_memory_ids`, so `channel_attribution[fid]` rarely contains `{"keyword"}`.

Evidence: `keyword_feedback` table is at **0 rows** despite the KG having 702 entities, 2828 triples, 3228 scoring-feedback rows, 1528 edge-feedback rows. The loop is coded correctly but the triggering condition is too strict given how RRF fusion ranks keyword hits.

Verified by directly calling `multi_channel_search` with keywords matching the test memory exactly — the memory appears in the keyword ranked_list (score 0.4) but doesn't survive RRF merge into the top 15.

### A3 recommendation

One of:

- Relax the condition to `"keyword" in channels` (keyword was a contributor, even if not sole).
- Fire suppression whenever the agent explicitly marks a keyword-matching memory irrelevant (regardless of other channels).
- Boost the keyword channel's RRF contribution or its base score so keyword-only hits survive adaptive_k more often.

Any of these would make the suppression feedback loop meaningfully active.

## A4 — Enrichment rejection [BUG — half-wired]

`edge_traversal_feedback` grew 1366 → 1528 during this audit session (1239 rows today). Multiple pairs rejected 3+ times return `get_edge_usefulness=-1.000` (well below the `_ENRICHMENT_USEFULNESS_FLOOR` of –0.3).

But the enrichments **keep surfacing**. Root cause: there are TWO enrichment generators:

1. `_detect_suggested_links` at `mcp_server.py:382` — runs when a memory is added. **Respects** the usefulness floor at line 430.
2. The finalize-time `edge_suggestions` loop at `intent.py:1982-2012` — runs when an intent finalizes, suggesting edges from slot entities to relevant memories at graph distance > 1. **Does NOT call `get_edge_usefulness`.**

The second generator is the one firing during this audit. Every intent finalization re-proposes the same rejected pairs.

### A4 recommendation

Add a `get_edge_usefulness` check in the finalize-time loop:

```python
# at intent.py:~2010, inside the edge_suggestions build
for slot_eid in slot_entities[:2]:
    try:
        u = _mcp._STATE.kg.get_edge_usefulness(slot_eid, "suggested_link", fid, intent_type=intent_type)
        if u < _mcp._ENRICHMENT_USEFULNESS_FLOOR:
            continue  # agent rejected this pair before
    except Exception:
        pass
    edge_suggestions.append({"from": slot_eid, "to": fid, "reason": reason})
```

Approximately 6 LOC.

## A5 — Edge traversal feedback [PASS-with-concern]

Write path (1528 rows in `edge_traversal_feedback`) and read path (`get_edge_usefulness` with context_id + intent_type + global fallback) both verified. BFS at `intent.py:1131-1155` prunes edges where `usefulness < _MIN_EDGE_USEFULNESS (–0.5)` using contextual MaxSim when available.

Concern: the edge feedback is recorded per-edge based on whether the MEMORY at the end of the BFS path was marked relevant. That means structural edges like `browse_for_invoices –is_a→ inspect` or `agent_* –executed_by→ ga_agent` can get `useful=0` recorded because one unrelated memory downstream was marked irrelevant. Accumulated over sessions, widely-used structural edges can drift below the pruning floor.

Evidence: the following structural edges currently have `useful=0/1` (i.e. one negative feedback, no positives):

- `browse_for_invoices –is_a→ inspect`
- `agent_cli_searcher_updates_2026_04_17 –executed_by→ ga_agent`
- Several other `*_updates –executed_by→ ga_agent`

Single-event –1.0 ratings on foundational type-hierarchy edges are premature.

### A5 recommendation

- Exclude structural predicates (`is_a`, `described_by`, `evidenced_by`, `executed_by`) from `record_edge_feedback` writes. They're schema, not inference links.
- Or: require multiple negative events before counting them toward `get_edge_usefulness`'s denominator.

## A6 — Scoring weight learning [BUG — 40% of weights never learn]

`scoring_weight_feedback` total 3228 rows, split only across three components:

- `agent`: 1076 rows (548 not-useful / 528 useful)
- `decay`: 1076 rows (same split)
- `imp`:   1076 rows (same split)
- `sim`:   **0 rows**
- `rel`:   **0 rows**

Source: `intent.py:1900-1905` writes only three components:

```python
components = {
    "imp": (imp - 1.0) / 4.0,
    "decay": max(0.0, min(1.0, 1.0 / (1.0 + age_days / 30.0))),
    "agent": 1.0 if agent_match else 0.0,
}
_mcp._STATE.kg.record_scoring_feedback(components, relevant)
```

Missing: the similarity value and the relevance-feedback value applied to this memory in the search that surfaced it. Those are 0.40 + 0.20 = 0.60 of the total hybrid_score weight.

`compute_learned_weights` correctly returns tiny adjustments for the three it has data for (±0.1% this session) and no adjustment for the two it lacks. The self-tuning system is effectively inert for the majority of the model.

### A6 recommendation

Extend `components` to include `sim` and `rel` at write time:

```python
components = {
    "sim":   similarity_score,      # the similarity this memory received in search
    "rel":   (relevance_feedback + 1.0) / 2.0,  # normalize [-1,+1] → [0,1]
    "imp":   (imp - 1.0) / 4.0,
    "decay": max(0.0, min(1.0, 1.0 / (1.0 + age_days / 30.0))),
    "agent": 1.0 if agent_match else 0.0,
}
```

Requires plumbing the per-memory similarity + relevance_feedback through from the search call into `active_intent`. Estimate: 15–25 LOC including the plumbing.

## A7 — `last_relevant_at` decay reset [BUG — silent no-op]

The relevant-feedback update at `intent.py:1847-1858`:

```python
if relevant:
    try:
        col = _mcp._get_collection(create=False)
        if col:
            existing = col.get(ids=[mem_id], include=["metadatas"])
            if existing and existing["ids"]:
                meta = existing["metadatas"][0] or {}
                meta["last_relevant_at"] = datetime.now().isoformat()
                col.update(ids=[mem_id], metadatas=[meta])
    except Exception:
        pass
```

`mem_id` at line 1828 is `normalize_entity_name(fb.get("id", ""))`. `normalize_entity_name` converts hyphens to underscores. Chroma stores the hyphenated ids verbatim. So `col.get(ids=[mem_id])` returns `existing["ids"]=[]`, the `if` branch is skipped, no update happens. The outer `try/except` means no error surfaces.

Verified directly: running the same code with the hyphenated id succeeds and updates `last_relevant_at`. With the normalized (underscored) id, `col.get` returns empty. The three test memories from this audit all have `last_relevant_at = date_added` despite being marked relevant during finalize.

Consequence: the power-law decay system at `scoring.py:170-190` uses `last_relevant_at` as the clock reset anchor. If that never gets updated, every useful memory ages out identically to an unused one.

### A7 recommendation

Use the original (non-normalized) id for the Chroma lookup, OR extend `normalize_entity_name` callers to pass through the raw id when reaching into Chroma. Minimal fix:

```python
raw_id = fb.get("id", "")
mem_id = normalize_entity_name(raw_id)
...
# when touching Chroma, use raw_id:
existing = col.get(ids=[raw_id], include=["metadatas"])
```

~3 LOC change. Worth a broader sweep across the codebase for other `col.get(ids=[normalized_id])` occurrences.

## Dead writes from the original plan (B1 / B2)

### B1 — Conflict resolution reasons never read

Confirmed — zero `SELECT * FROM conflict_resolutions` queries anywhere in the codebase. 12 rows accumulated this session, all audit-only. Plan recommendation stands: activate the read path by letting `resolve_conflicts` consult past decisions on similar conflicts (`get_past_conflict_resolution(existing_id, new_id, conflict_type)`).

### B2 — Enrichment rejection `context_keywords` never read

Confirmed — `useful=False` is consumed by `get_edge_usefulness`, but the reason text stored in `context_keywords[:200]` is never retrieved. Plan recommendation stands: use MaxSim over past rejection reasons when evaluating new enrichments, so a single strong-context rejection can apply without needing aggregate `useful` to hit the floor.

## `kg_search` edge format (Phase C)

Still recommending C2 (top-N edges + `edge_count` + `edges_full` escape hatch). Confirmed: `mcp_server.py:1205-1211` returns every current edge without cap. For the 702-entity live palace I saw edge counts ranging from 3 (typical entity) to 37+ (high-degree entities like `full_feedback_audit_2026_04_18`). A 20-edge result set per search hit is common. Payload bloat is real.

## Bonus structural findings (not in A1–A7)

### BF1 — KG file split

Without `--palace`, the bootstrap routes `KnowledgeGraph()` to `~/.mempalace/knowledge_graph.sqlite3` (via `DEFAULT_KG_PATH` in `knowledge_graph.py:63`). Config's `palace_path` is `~/.mempalace/palace/`. The Chroma data lives in the palace dir; the KG sqlite lives one level up. A zero-byte `palace/knowledge_graph.sqlite3` was found — an orphan artifact likely created by a past invocation with `--palace` set.

Fix: unify the KG path with the palace dir, or explicitly document the split and remove the orphan file.

### BF2 — Enrichment seed-selection noise

The enrichment system repeatedly suggests edges of the form:

- `<slot-command-literal> → <memory>` — e.g. `python → record_ga_agent_...` or `sqlite3 → ...` or `git → ...`. Command strings in the `commands` slot are getting treated as semantic entities.
- `<path-glob-literal> → <memory>` — e.g. `D:/Flowsev/mempalace/** → record_ga_agent_...`. The slot path glob is getting treated as a semantic entity.
- `ga_agent → <own-result-memory>` — agent-to-own-output where `described_by`/`executed_by`/`session_note_for` edges already exist transitively.

These account for the majority of rejected enrichments this session. Seed selection should exclude:

- Raw slot strings for `commands` and `paths` slots (they aren't semantic entities).
- Pairs already connected via any existing predicate — not just via `suggested_link` usefulness history.

### BF3 — `mempalace_drawers` collection name legacy

`config.json` still points at `mempalace_drawers` as the Chroma collection. The code was renamed to default `mempalace_records` in an earlier phase but existing installations carry the legacy name via their config file. Worth a migration path so fresh installs agree with old ones.

## Proposed fix sequence (Phase B/C implementation)

Each a self-contained commit:

1. **A7 fix** (`last_relevant_at` update uses hyphen id) — smallest, highest impact on the decay subsystem. ~3 LOC. `mempalace/intent.py`.
2. **A4 fix** (finalize-time enrichment generator honours `get_edge_usefulness`). ~6 LOC. `mempalace/intent.py`.
3. **A3 fix** (relax keyword suppression trigger OR boost keyword channel in RRF). ~5 LOC. `mempalace/intent.py` or `mempalace/scoring.py`.
4. **A6 fix** (plumb `sim` + `rel` components through to `record_scoring_feedback`). ~20 LOC across `intent.py` + ChromaDB query callsites.
5. **BF2 fix** (enrichment seed-selection noise filter). ~10 LOC. `mempalace/mcp_server.py` + `mempalace/intent.py`.
6. **A5 concern** (exclude structural predicates from edge_traversal_feedback writes). ~5 LOC. `mempalace/intent.py`.
7. **B1b** (activate `conflict_resolutions` read path). ~30 LOC. `mempalace/mcp_server.py` + `mempalace/knowledge_graph.py`.
8. **B2b** (MaxSim over past enrichment rejection reasons). ~25 LOC. `mempalace/mcp_server.py` + `mempalace/knowledge_graph.py`.
9. **C2** (`kg_search` edge reduction with `edges_full` escape hatch). ~20 LOC. `mempalace/mcp_server.py`.
10. **BF1** (unify KG file path or remove orphan). ~5 LOC + cleanup script.

If you approve all 10, each lands as a separate commit with tests (new tests in `tests/test_scoring.py` and `tests/test_feedback_loops.py`). Full suite (597/1 baseline) must stay green.
