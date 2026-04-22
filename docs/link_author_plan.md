# Link-author redesign — implementation plan

**Status**: designed, awaiting execution.
**Audience**: fresh Claude Code session after context compaction.
**Baseline assumption**: cold-started palace, no legacy data. No back-compat anywhere.

This document is self-contained. It captures every decision from the
design discussion so a new session can pick up and execute without
replaying the conversation. All open questions have defaults filled in;
override them if you have reason to.

---

## 0. Where we are right now

### Branches

- `main` has the context-as-entity redesign end-to-end through the polish PR (#60 merged). Final commit on main: `bcbf3f9 docs: morning handoff ...` (or later depending on when this document is read).
- `feat/rocchio-multi-context-dedup` is the in-flight branch with three commits:
  1. `fce3b5b` — Rocchio: multi-context + per-channel gating + semantic dedup
  2. `ed27f67` — Strip back-compat shims (cold-start palace, no legacy data)
  3. `dcb5932` — Threshold-calibration TODOs on tunable constants + flat-list memory_feedback retirement + test migration

All three are green (suite: 916 passed, 1 skipped) and ready to ship. **Ship this branch FIRST** — all work below builds on it.

### What's in the codebase already

- Context-as-entity substrate (kind="context" KG entities, MaxSim lookup/reuse/similar_to).
- Four-channel weighted-RRF retrieval (cosine / graph / keyword / context).
- Channel D (context-feedback) reads rated_useful/rated_irrelevant edges over similar_to neighbourhood.
- `rocchio_enrich_context` — per-context enrichment at finalize with per-channel gating + semantic dedup.
- `lookup_context_feedback` (signed W_REL) wired to hybrid_score.
- BM25-IDF on keyword channel; degree-dampening on graph channel.
- Per-channel and per-hybrid weight learning via `compute_learned_weights(scope=...)`.
- JSONL telemetry + `mempalace eval` CLI.
- `memory_feedback` is **map-shape only** at finalize_intent — flat-list retired.
- Strict coverage validator at finalize: every (context, memory) surfaced pair must have a rated_* edge.

### What's NOT in the codebase

- Mandatory entities across emit sites (currently optional).
- Any form of link prediction or background edge authoring.
- The blind-cosine `_detect_suggested_links` mechanism IS still present and still runs — that's what this work retires.

---

## 1. Goal

Replace the current blind-cosine `_detect_suggested_links` enrichment mechanism (low-signal, high-friction, agent-ignored, not research-backed) with a two-layer system:

1. **Analytical candidate layer** — cheap, in-session, at finalize. Uses Adamic-Adar on the bipartite context-entity graph to build a prioritised queue of entity pairs that SHOULD be examined for a relationship.
2. **LLM-authored edge layer** — background, out-of-session, via Claude Code CLI. For each candidate, a Haiku jury decides: is there a real edge? Which predicate (select from existing OR propose new)? What's the natural-language statement?

Only LLM-authored edges land in the KG. The analytical layer is a filter, not an author.

### Research grounding (citations for docstrings)

- Liben-Nowell & Kleinberg 2007 JASIST "The link-prediction problem for social networks" — topology-based prediction beats random baseline up to 30× on sparse networks; Adamic-Adar beats plain common-neighbours consistently.
- Adamic & Adar 2003 — rare shared neighbours are stronger evidence; weight by 1/log(degree).
- Shu et al. 2024 "KG-LLM for Link Prediction" (arXiv:2403.07311) — LLM fine-tuning beats classical methods on labeled KGs; predicate selection requires semantic reasoning, not topology.
- Du et al. 2024 "Improving Factuality and Reasoning in Language Models through Multiagent Debate" — ensemble/jury reasoning improves factuality over single-shot.
- Graphiti (Zep, 2024) — LLM-only edge extraction from prose in agent KGs; no topology-based auto-creation.

---

## 2. Design decisions (all locked)

### 2.1 Split of labour

- **Analytical**: Adamic-Adar on contexts-entities bipartite projection. Pure SQL at finalize, zero LLM calls. High recall, low precision.
- **LLM**: jury of 3 Haiku calls + 1 synthesis step. Picks predicate from existing set OR proposes new one, writes statement, or returns no-edge verdict. High precision.

### 2.2 Adamic-Adar on our structure

For entity pair (A, B), the score over all contexts where both appear:

```
AA(A, B) = Σ  1 / log(|ctx.properties.entities|)
          over contexts ctx where A ∈ ctx.entities AND B ∈ ctx.entities
```

A context listing 2 entities contributes more than one listing 30. Each finalize-with-positive-feedback of a reused context increments this score incrementally (no full recomputation).

### 2.3 Mandatory entities

Entities are NOT optional. Every emit site requires them:

- `validate_context(..., entities_min=1)` — default `entities_min` raised from 0 to 1.
- `tool_declare_operation` gains a new `entities: list` parameter. Mandatory min 1.
- Docstrings + `PALACE_PROTOCOL` updated to teach: list every entity the task touches.

Without entities, the Adamic-Adar signal doesn't exist and Channel B has no seeds. Making them mandatory is the precondition.

### 2.4 When the candidate upsert fires

At `tool_finalize_intent`, for each entry in `contexts_touched_detail`:
- Skip if not reused (fresh contexts have no accumulating evidence).
- Skip if not net-positive on this context's per-channel Rocchio gating.
- For each unordered pair (A, B) in `ctx.properties.entities` where `A < B` lexicographically:
  - If a direct edge A—*→B or B—*→A already exists in triples: skip.
  - Else: UPSERT into `link_prediction_candidates`, adding `1 / log(|ctx.entities|)` to score; increment `shared_context_count`; update `last_context_id`, `last_updated_ts`.

Canonical ordering (smaller id first) keeps the PK unique across direction permutations.

### 2.5 LLM jury architecture

**3 Haiku calls in parallel, each with a distinct role**:

1. **Engineer**: "You are a senior engineer looking at this codebase/project. Given these two entities and their co-occurrence evidence, is there a real relationship between them? If yes, which predicate from the list best captures it?"
2. **Ontologist**: "You are an ontology designer. Which predicate semantically fits? If none of the existing predicates fit, propose a new one (name, description, subject_kinds, object_kinds). Otherwise select the best match."
3. **Skeptic**: "You are a skeptical reviewer. Argue whether the evidence actually supports a real relationship, or whether A and B just happened to co-appear in unrelated contexts. Return your confidence (0-1) and predicate pick OR 'no edge'."

**Synthesis step** (1 more Haiku call, or rule-based):
- If 2+ juries agree on the same predicate → accept, use their statement.
- If all three juries say "no edge" → reject, record reason.
- If juries disagree → mark `llm_verdict='uncertain'`, leave unprocessed or escalate to Sonnet (configurable).

### 2.6 Predicate creation

The Ontologist role is permitted to propose NEW predicates when existing ones don't fit. Shape:

```json
{
  "action": "propose_new_predicate",
  "name": "wraps",
  "description": "Subject provides a higher-level API around object",
  "subject_kinds": ["entity", "class"],
  "object_kinds": ["entity", "class"],
  "cardinality": "many-to-one",
  "reasoning": "Neither 'uses' nor 'depends_on' captures the wrapping semantic..."
}
```

Guardrails against predicate explosion:
- Only the synthesis step can CREATE a new predicate (not the individual jurors).
- Synthesis requires jury consensus (at least 2 of 3 agreeing new is needed, and agreeing on name + semantics).
- Before creating, check existing predicates for semantic near-duplicates via a quick lookup in the KG.
- New predicates go through `kg_declare_entity(kind='predicate', ...)` with the proposed constraints.
- Log to `~/.mempalace/hook_state/new_predicates.jsonl` so the user can review periodically.
- `mempalace link-author status --new-predicates` shows recently created predicates.

### 2.7 CLI + scheduling

**CLI**:
```
mempalace link-author process [--model haiku|sonnet] [--max N] [--threshold F] [--dry-run] [--jury|--no-jury]
mempalace link-author status [--recent N] [--new-predicates]
```

Defaults: `--model haiku --max 50 --threshold 1.5 --jury`.

**Scheduling — two independent triggers**, both OK to have:

1. **Cron/launchd/Task Scheduler** — deterministic cadence. User sets up once with commands documented in `docs/link_author_scheduling.md`.
2. **Finalize-triggered detached subprocess** — event-driven. At finalize, after the candidate upsert, check `NOW - last_run_ts >= 1h` AND there are pending candidates. If both, spawn `mempalace link-author process` detached via `subprocess.Popen(..., start_new_session=True)` on POSIX / `creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP` on Windows. Non-blocking. Update `started_ts` before the fork to prevent concurrent runs.

### 2.8 Claude Code as the LLM runtime

Spawn Claude via the non-interactive mode:
```
claude -p "<prompt>" --model haiku --output-format json
```

The exact flags need verification on first run; the CLI is being iterated. Structure the module to isolate the subprocess call so swapping flags later is trivial.

**Detection + fallback**: check `shutil.which("claude")` on startup of `link-author process`. If missing, exit 2 with a clear message. Don't pretend to process anything.

### 2.9 Claude is the only LLM authority

No API key. No Anthropic SDK. No OpenAI. Claude Code CLI only. This keeps:
- Zero API key management burden.
- Single-binary deployment (if claude is installed, mempalace works).
- User's existing Claude Code usage budget handles the cost, no separate billing.

### 2.10 What gets retired

**Deleted entirely**:
- `_detect_suggested_links` function in mcp_server.py.
- Its two call sites (in `_add_memory_internal` and `tool_kg_declare_entity`).
- The `pending_enrichments` field in `active_intent` (nothing writes to it anymore).
- The `enrichment_resolutions` parameter in `tool_finalize_intent` + its acceptance path.
- `_consume_matching_enrichment` in mcp_server.py.
- `tool_resolve_enrichments` + its MCP dispatch.
- `_ENRICHMENT_SIM_THRESHOLD`, `_ENRICHMENT_USEFULNESS_FLOOR`, `_ENRICHMENT_MAX_SUGGESTIONS`, `_REJECTION_REASON_OVERLAP_THRESHOLD` constants.
- `_pair_already_directly_connected` helper (subsumed by the Adamic-Adar upsert's "skip if direct edge exists" check).
- All tests pinned to the retired enrichment API (`test_enrichment_auto_accept.py`, enrichment-related tests in `test_mcp_server.py`).

**Replaced by**: the `link_prediction_candidates` table + `mempalace link-author` CLI.

---

## 3. Schema (migration 017)

File: `mempalace/migrations/017_link_prediction.sql`

```sql
-- 017: link prediction candidate queue + author runs.
-- depends: 016_keyword_idf
--
-- Bipartite context-entity Adamic-Adar candidate signal. Entity pairs
-- that co-appear in reused + positively-rated contexts accumulate a
-- score here. The mempalace link-author CLI reads above-threshold
-- rows, asks a Haiku jury to author the edge (or reject), writes
-- the verdict back, and creates the edge via kg_add on accepts.
--
-- Canonical ordering: from_entity lexically < to_entity. The upsert
-- must canonicalise before INSERT to keep the PK unique across
-- symmetric pair permutations.

CREATE TABLE link_prediction_candidates (
    from_entity          TEXT NOT NULL,
    to_entity            TEXT NOT NULL,
    score                REAL NOT NULL DEFAULT 0.0,
    shared_context_count INTEGER NOT NULL DEFAULT 0,
    last_context_id      TEXT NOT NULL DEFAULT '',
    last_updated_ts      TEXT NOT NULL DEFAULT '',
    processed_ts         TEXT,
    llm_verdict          TEXT,   -- 'edge' | 'no_edge' | 'uncertain' | NULL
    llm_predicate        TEXT,   -- name of chosen or newly-created predicate
    llm_statement        TEXT,   -- natural-language verbalization for the edge
    llm_reason           TEXT,   -- jury's synthesis reasoning
    llm_jury_model       TEXT,   -- e.g. 'haiku' or 'sonnet'
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

CREATE TABLE link_author_runs (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    started_ts            TEXT NOT NULL,
    completed_ts          TEXT,
    candidates_processed  INTEGER DEFAULT 0,
    edges_created         INTEGER DEFAULT 0,
    edges_rejected        INTEGER DEFAULT 0,
    new_predicates_created INTEGER DEFAULT 0,
    model                 TEXT DEFAULT 'haiku',
    errors                TEXT
);

CREATE INDEX idx_link_runs_recent ON link_author_runs (started_ts DESC);
```

Also add an entry to `_already_applied_checks` in `knowledge_graph.py`:
```python
"017_link_prediction": lambda: _has_table("link_prediction_candidates"),
```

---

## 4. Module: `mempalace/link_author.py`

Responsibilities:
- `upsert_candidate(kg, from_entity, to_entity, weight, context_id)` — single-pair SQL upsert. Called by finalize.
- `list_pending(kg, limit, threshold)` — returns candidates above threshold, ordered by score desc.
- `author_candidate(kg, row, model='haiku', jury=True)` — runs the jury, returns verdict dict + edge payload.
- `process(kg, max=50, threshold=1.5, model='haiku', jury=True, dry_run=False)` — the CLI-entry function. Iterates, writes edges, records verdicts.
- `_call_claude(prompt, model) -> str` — subprocess wrapper around `claude -p ...`. Returns raw stdout.
- `_parse_verdict(text) -> dict` — schema-validates the Claude response.
- `_build_juror_prompt(role, entity_a, entity_b, shared_contexts, predicates) -> str`.
- `_synthesise(juror_verdicts) -> dict` — consensus logic.
- `_maybe_create_predicate(kg, proposal)` — validates + creates via `kg.add_entity(kind='predicate', ...)`.
- `_dispatch_if_due(kg, interval_hours)` — checks last_run, spawns detached subprocess if due. Called by finalize.

### Jury prompt template (keep prompts in-module, not in a separate file)

Each juror gets:
1. Role persona.
2. Entity A: id, kind, description (truncated to 500 chars), top keywords.
3. Entity B: same.
4. Shared contexts (up to 5): for each, `queries[:2]` + `keywords[:5]` + `score` contribution.
5. Available predicates: name + description + constraints (subject_kinds, object_kinds, cardinality). Filtered to those whose constraints are compatible with (A.kind, B.kind).
6. Output schema: strict JSON, examples provided.

### Response schema

Single juror:
```json
{
  "role": "engineer",
  "verdict": "edge" | "no_edge" | "uncertain",
  "confidence": 0.0-1.0,
  "predicate_choice": "uses" | "<existing>" | null,
  "propose_new_predicate": null | {
    "name": "...",
    "description": "...",
    "subject_kinds": [...],
    "object_kinds": [...],
    "cardinality": "many-to-many" | "many-to-one" | "one-to-many" | "one-to-one"
  },
  "statement": "Alice uses Bob." | null,
  "reason": "short explanation"
}
```

Synthesis output:
```json
{
  "verdict": "edge" | "no_edge" | "uncertain",
  "predicate": "uses" | null,
  "new_predicate_proposed": null | {...},
  "statement": "...",
  "reason": "jury synthesis reason",
  "juror_agreement": "unanimous" | "majority" | "split"
}
```

### Consensus rules

- All 3 say `edge` with same predicate → accept that predicate, use the engineer's statement.
- All 3 say `edge` but disagree on predicate → synthesis Haiku call picks best from the 3 options. If it picks one: accept. If it says "uncertain": mark `uncertain`.
- 2 say `edge`, 1 says `no_edge` → accept if the 2 agree on predicate; mark uncertain otherwise.
- 2 say `no_edge`, 1 says `edge` → reject (`no_edge`), record reason.
- All 3 say `no_edge` → reject, record reason.
- Any `uncertain` or schema-violating response → retry once, then mark `uncertain` in the DB.
- New predicate proposed: require BOTH ontologist + one other juror to agree on the new name + broadly-compatible semantics. Synthesis step double-checks existing predicates for near-duplicates (cosine on predicate descriptions >= 0.75 means "use existing").

---

## 5. Changes per file

### 5.1 `mempalace/scoring.py`

- `validate_context` default `entities_min=1`.
- The `_validate_string_list` is fine as-is.

### 5.2 `mempalace/intent.py`

`tool_declare_operation` gains `entities: list` parameter. Validation:
- Mandatory. `MIN_OP_ENTITIES = 1`, `MAX_OP_ENTITIES = 10` (cap to prevent abuse).
- Each must be a non-empty string; normalize via `normalize_entity_name`.
- Include in the operation cue payload + `_record_context_emit` call (already supports entities).

`tool_finalize_intent` gets a new post-write block BEFORE the Rocchio block:

```python
# ── Link-prediction candidate upsert ──
# For each reused context with net-positive per-channel feedback,
# accumulate Adamic-Adar signal for each unconnected entity pair.
# The mempalace link-author CLI consumes this queue offline.
import math
from . import link_author as _la
detail = _mcp._STATE.active_intent.get("contexts_touched_detail") or []
for entry in detail:
    if not (entry.get("reused") and entry.get("ctx_id")):
        continue
    entities = entry.get("entities") or []
    if len(entities) < 2:
        continue
    # Per-context mean relevance gate (same as Rocchio's).
    rels = [...]  # compute from memory_feedback scoped to ctx
    if mean(rels) < _ROCCHIO_MEAN_REL_CUT:
        continue
    weight = 1.0 / math.log(len(entities) + 1)  # +1 for stability
    for i, a in enumerate(entities):
        for b in entities[i+1:]:
            _la.upsert_candidate(
                _mcp._STATE.kg,
                from_entity=min(a, b),
                to_entity=max(a, b),
                weight=weight,
                context_id=entry["ctx_id"],
            )

# ── Finalize-triggered background dispatch ──
try:
    _la._dispatch_if_due(_mcp._STATE.kg, interval_hours=1)
except Exception:
    pass  # never block finalize on dispatch failure
```

Retirement: remove `enrichment_resolutions` parameter and its acceptance block entirely.

### 5.3 `mempalace/mcp_server.py`

- DELETE `_detect_suggested_links`, `_pair_already_directly_connected` (unless used elsewhere), both callers in `_add_memory_internal` and `tool_kg_declare_entity`.
- DELETE `_consume_matching_enrichment` — kg_add no longer auto-consumes.
- DELETE `tool_resolve_enrichments` function + its dispatch table entry + its advertised MCP method.
- DELETE constants: `_ENRICHMENT_SIM_THRESHOLD`, `_ENRICHMENT_USEFULNESS_FLOOR`, `_ENRICHMENT_MAX_SUGGESTIONS`, `_REJECTION_REASON_OVERLAP_THRESHOLD`, `_MIN_ENRICHMENT_REJECT_REASON_CHARS`, `_ENRICHMENT_LOG_PATH`.
- DELETE `_log_enrichment_decision` helper.
- DELETE `_STATE.pending_enrichments` field and all its readers in intent.py.
- DELETE `_load_pending_enrichments_from_disk` helper.
- `tool_kg_declare_entity` loses its `suggested_links` return payload.
- `tool_kg_add` loses the `auto_resolved_enrichment` return field.

### 5.4 `mempalace/cli.py`

Add the `link-author` subcommand with `process` and `status` actions. Defaults match §2.7.

### 5.5 `mempalace/config.py`

Add a `link_author` config section:
```python
{
  "link_author": {
    "model": "haiku",
    "interval_hours": 1,
    "threshold": 1.5,
    "min_confirmations": 2,
    "max_per_run": 50,
    "jury": True,
    "escalate_uncertain_to": "sonnet" | null
  }
}
```

Loaded at CLI start, overridden by CLI flags.

### 5.6 `docs/link_author_scheduling.md`

One-page doc with cron / launchd / Task Scheduler examples. Reference the dispatch-on-finalize as the primary trigger and schedulers as a belt-and-suspenders backup.

### 5.7 `PALACE_PROTOCOL` update in `mcp_server.py`

Add a section about mandatory entities:

```
WHEN DECLARING INTENT / OPERATION / SEARCH:
  - `context.entities` is MANDATORY (≥1). List every entity the task
    touches — the files you'll edit, the services / concepts you're
    reasoning about, the agents involved, etc.
  - Slots on declare_intent are typed entity references; context.entities
    MAY overlap with slot values but should also include entities that
    don't fit the slot schema (concepts, tools, related systems).
  - declare_operation now also requires `entities` (1-10).
  - kg_search now requires `context.entities` (≥1).
  - Why: the link-author background process builds a relationship graph
    from co-occurrence in contexts. Without entities, that graph never
    grows and the graph channel stays sparse.
```

Also remove the existing enrichment section (graph enrichment suggestions, resolve_enrichments workflow).

---

## 6. Tests

### New files

- `tests/test_link_prediction_candidates.py`
  - Adamic-Adar math: 2 entities in 3 contexts of varying size → expected score.
  - Canonical-ordering PK: (B, A) and (A, B) collapse to same row.
  - Direct-edge short-circuit: entity pair with existing edge is NOT upserted.
  - Per-context gating: fresh context OR negative feedback OR low-relevance channel → no upsert.

- `tests/test_link_author_cli.py`
  - Happy-path single candidate, jury mocked to unanimous `edge` → edge created via kg_add, verdict recorded.
  - Jury disagreement → `uncertain` verdict, no edge, candidate unprocessed (for retry).
  - All-no_edge jury → rejection verdict, no edge.
  - New predicate proposal with consensus → predicate created + edge created.
  - New predicate proposal without consensus → rejected.
  - Claude CLI missing → exit 2 with clear message.

- `tests/test_mandatory_entities.py`
  - `declare_intent` with empty entities → validation error.
  - `declare_operation` without entities param → validation error.
  - `kg_search` with empty entities → validation error.
  - Happy paths succeed with ≥1 entity.

### Existing tests to delete

- `tests/test_enrichment_auto_accept.py` — entire file retired.
- Enrichment-related tests in `tests/test_mcp_server.py` if any remain (search for `pending_enrichments`, `resolve_enrichments`, `suggested_links`).

### Test migration

- All existing tests currently passing `context={queries, keywords}` (no entities) → add `"entities": ["test_target"]` (or similar). Grep for `"queries":.*"keywords":.*}` and audit.
- All existing tests calling `tool_declare_operation(...)` without entities → add `entities=["test_target"]`.

Budget: ~15-20 test-site migrations across 5-6 files.

---

## 7. Commit sequencing

Branch from `main` (after `feat/rocchio-multi-context-dedup` is merged) as `feat/link-author`. Each commit is independently green.

1. **Migration 017 + validate_context entities_min=1 + docstring updates**. Test migration for context-dict sites (just adding the entities list everywhere).
2. **`tool_declare_operation` entities parameter + test migration.**
3. **Retire `_detect_suggested_links` + pending_enrichments + resolve_enrichments tool + auto-consume in kg_add.** Delete retired tests.
4. **`mempalace/link_author.py` with `upsert_candidate`, `list_pending`, `_dispatch_if_due`** (skeleton only, no LLM yet). Tests for upsert + canonical ordering + dispatch gating.
5. **Finalize candidate-upsert loop + test.**
6. **`link_author.process` with LLM jury + Haiku subprocess call + synthesis + test with mocked claude subprocess.**
7. **CLI `mempalace link-author process|status` + scheduling docs + PALACE_PROTOCOL update.**
8. **Final test pass + PR + merge.**

Expected total: ~800-1100 LOC change + ~20-25 new tests + ~15-20 test migrations.

---

## 8. Locked decisions (all user-confirmed; no overrides needed)

| Question | Locked value |
|---|---|
| Model | Haiku (default); escalate to Sonnet on 'uncertain' |
| Interval | 1 hour between dispatches |
| Threshold | score ≥ 1.5 AND shared_context_count ≥ 2 |
| Max per run | 50 candidates |
| Jury | ON — 3 Haiku jurors + synthesis step |
| Escalation | ON — uncertain verdict re-runs on Sonnet |
| Claude CLI discovery | HARD REQUIRE. Exit with clear message if missing. |
| New-predicate creation | permitted via jury consensus (see §2.6 guardrails) |
| Dispatch on finalize | ON — detached subprocess, 1h gate, non-blocking |
| Retired enrichment machinery | DELETED ENTIRELY — no stubs, no tool endpoint, no fallback |
| Mandatory entities | ENFORCED on declare_intent / declare_operation / kg_search, ≥ 1 |
| flat-list memory_feedback | already retired (lands via feat/rocchio-multi-context-dedup) |

---

## 9. Ship order (reinstall boundary)

**Before reinstall**:
1. Merge `feat/rocchio-multi-context-dedup` to main.
2. Branch `feat/link-author` from main.
3. Execute this plan end-to-end.
4. Merge to main.
5. Adrian reinstalls with the complete pipeline.

**After reinstall**:
- Threshold calibration based on observed candidate queue depth + jury verdict distribution.
- Model tuning (Haiku vs Sonnet) based on observed edge quality.
- New-predicate review: periodic human check of the `new_predicates.jsonl` log to catch predicate-space explosion.

---

## 10. What-not-to-do (guard rails)

- **Do NOT auto-create `co_appears_in` or other generic predicates.** The LLM picks a real predicate or the edge doesn't get created.
- **Do NOT keep any enrichment machinery around as a fallback.** It was low-value busywork; it's retired.
- **Do NOT make entities optional "for backward compatibility".** There's no legacy data; the contract is mandatory from day 1.
- **Do NOT block finalize on link-author failures.** Dispatch is fire-and-forget.
- **Do NOT use an Anthropic API key.** Claude Code CLI only.
- **Do NOT allow individual jurors to create new predicates.** Only synthesis step, with consensus.
- **Do NOT re-process rejected candidates forever.** `processed_ts` + a cooling-off period (e.g. 30 days) covers re-evaluation without churn.

---

## 11. Morning sanity checks after implementation

1. Full suite green (`pytest -q --ignore=tests/benchmarks --ignore=tests/manual`).
2. `mempalace link-author process --dry-run` works against an empty palace (exits clean, zero candidates).
3. Reinstall mempalace plugin. Declare 3 intents touching a common entity across 2 of them. Finalize each with positive feedback. Check the `link_prediction_candidates` table — should have one row for each unconnected pair.
4. Run `mempalace link-author process` manually. Verify it calls `claude`, writes verdicts, creates edges.
5. Inspect the created edges via `mempalace_kg_query` — they should have non-generic predicates and meaningful statements.

---

## 12. References

- Liben-Nowell & Kleinberg 2007 JASIST — https://onlinelibrary.wiley.com/doi/abs/10.1002/asi.20591
- Adamic & Adar 2003 / Wikipedia — https://en.wikipedia.org/wiki/Adamic%E2%80%93Adar_index
- Shu et al. 2024 "KG-LLM for Link Prediction" — https://arxiv.org/abs/2403.07311
- Du et al. 2024 "Multiagent Debate" — search arXiv.
- Graphiti — https://github.com/getzep/graphiti
- Rocchio 1971 via Manning/Raghavan/Schütze IR book Ch.9 — https://nlp.stanford.edu/IR-book/pdf/09expand.pdf
