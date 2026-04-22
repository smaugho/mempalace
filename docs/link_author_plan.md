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

- `main` has the context-as-entity redesign + Rocchio multi-context + back-compat strip + link-author plan merged (PR #61, merge commit `da4afcb`). All 916 tests green, 1 skipped.
- Branch from main as `feat/link-author` to execute this plan.

### What's in the codebase already

- Context-as-entity substrate (kind="context" KG entities, MaxSim lookup/reuse/similar_to).
- Four-channel weighted-RRF retrieval (cosine / graph / keyword / context).
- Channel D (context-feedback) reads rated_useful/rated_irrelevant edges over similar_to neighbourhood.
- `rocchio_enrich_context` — per-context enrichment at finalize with per-channel gating + semantic dedup.
- `contexts_touched_detail` on `active_intent` — records every emit (intent/operation/search) with scope + reused flag + source queries/keywords/entities. **This is the sole input to candidate seeding** (§2.4).
- `lookup_context_feedback` (signed W_REL) wired to hybrid_score.
- BM25-IDF on keyword channel; degree-dampening on graph channel.
- Per-channel and per-hybrid weight learning via `compute_learned_weights(scope=...)`.
- JSONL telemetry + `mempalace eval` CLI.
- `memory_feedback` is **map-shape only** at finalize_intent — flat-list retired.
- Strict coverage validator at finalize: every (context, memory) surfaced pair must have a rated_* edge.

### What's NOT in the codebase

- Mandatory entities across emit sites (currently optional — `entities_min=0`).
- Any form of link prediction or background edge authoring.
- The blind-cosine `_detect_suggested_links` mechanism IS still present and still runs — that's what this work retires (commit 1 in §7).

---

## 1. Goal

Replace the current blind-cosine `_detect_suggested_links` enrichment mechanism (low-signal, high-friction, agent-ignored, not research-backed) with a two-layer system:

1. **Analytical candidate layer** — cheap, in-session, at finalize. Uses Adamic-Adar on the bipartite context-entity graph to build a prioritised queue of entity pairs that SHOULD be examined for a relationship. Each distinct context contributes to a pair's score exactly once (deduped via `link_prediction_sources`).
2. **LLM-authored edge layer** — background, out-of-session, via Claude Code CLI. Per candidate, a three-stage pipeline: Opus 4.7 designs 3 domain-appropriate juror personas from the candidate's shared-context fingerprint → Haiku runs the 3 jurors in parallel → Haiku synthesises the verdict. Jury decides: is there a real edge? Which predicate (select from existing OR propose new)? What's the natural-language statement?

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
- **LLM**: three-stage pipeline per candidate — Opus 4.7 designs jurors from the candidate's domain fingerprint, Haiku runs 3 jurors in parallel, Haiku synthesises. Picks predicate from existing set OR proposes new one, writes statement, or returns no-edge verdict. High precision.

### 2.2 Adamic-Adar on our structure

For entity pair (A, B), the score over all **distinct** contexts where both appear:

```
AA(A, B) = Σ  1 / log(|ctx.properties.entities|)
          over DISTINCT contexts ctx where A ∈ ctx.entities AND B ∈ ctx.entities
```

A context listing 2 entities contributes more than one listing 30. Each **distinct** context contributes exactly once — multiple reuses of the same context do NOT re-increment the score (enforced via the `link_prediction_sources` dedup table, §3).

**Rationale for distinct-contexts-only (Option A)**: Adamic-Adar is a count of distinct shared-neighbour nodes. Re-observing the same neighbour is one observation seen repeatedly, not multiple observations. Accumulating every reuse would let a single heavily-reused context trip the threshold alone; distinct-only ensures the threshold reflects co-occurrence in *different* situations, which is the real signal.

**Candidate seeding source (explicit)**: seeding reads from `active_intent.contexts_touched_detail` only — every directly-touched reused context, regardless of scope (intent / operation / search). `similar_to` neighbours do NOT seed candidates. similar_to is a retrieval mechanism (Channel C on hybrid_score, Channel D on context-feedback); giving it a third job in link prediction would launder blind-cosine back into the pipeline, which is exactly what this rewrite retires.

### 2.3 Mandatory entities

Entities are NOT optional. Every emit site requires them:

- `validate_context(..., entities_min=1)` — default `entities_min` raised from 0 to 1.
- `tool_declare_operation` gains a new `entities: list` parameter. Mandatory min 1.
- Docstrings + `PALACE_PROTOCOL` updated to teach: list every entity the task touches.

Without entities, the Adamic-Adar signal doesn't exist and Channel B has no seeds. Making them mandatory is the precondition.

### 2.4 When the candidate upsert fires

At `tool_finalize_intent`, for each entry in `contexts_touched_detail` (covers all scopes — intent, operation, search):
- Skip if not reused (fresh contexts have no accumulating evidence — they're first-time observations).
- Skip if not net-positive on this context's per-channel Rocchio gating.
- For each unordered pair (A, B) in `ctx.properties.entities` where `A < B` lexicographically:
  - If a direct edge A—*→B or B—*→A already exists in triples: skip (the jury's job is discovering NEW edges; existing edges need no discovery).
  - **Dedup check** — INSERT OR IGNORE into `link_prediction_sources (from_entity, to_entity, ctx_id)`. If the insert was ignored (tuple already present), SKIP the candidate upsert — this context has already contributed to this pair.
  - Else: UPSERT into `link_prediction_candidates`, adding `1 / log(|ctx.entities|)` to score; increment `shared_context_count`; update `last_context_id`, `last_updated_ts`.

Canonical ordering (smaller id first) keeps the PK unique across direction permutations.

**Note on indirect connections**: pairs connected via longer paths (A→X→B, etc.) are still eligible candidates — only DIRECT edges skip. Collapsing multi-hop paths into direct edges is the whole point, because the graph channel only walks 1-hop neighbours (scoring.py:981, `_build_graph_channel`). Direct edges are what make pairs visible to Channel B.

### 2.5 LLM jury architecture — three stages, per candidate

**Stage 1 — Opus 4.7 designs the jury** (one call per candidate, or per domain-cluster if batching):

The jurors are not hard-coded. Opus looks at the candidate's domain fingerprint and designs three personas tailored to evaluate this specific candidate.

- **Input**: a "domain hint" blob built from the candidate's top-5 shared contexts — concatenate each context's `queries[:2]`, `keywords[:5]`, and dominant entity kinds (~500 tokens total).
- **Prompt**: "You are designing a jury to assess whether two entities in a knowledge graph share a real relationship. Here is the domain they co-occurred in: `<domain hint>`. Design 3 juror personas. Always include an ontologist (picks the most-fitting predicate) and a skeptic (challenges the evidence) — both are domain-agnostic. The third juror must be a domain expert appropriate for this domain (e.g. senior software engineer, paralegal, accountant, marketing strategist, security analyst, clinical researcher — whatever fits). Return strict JSON: `[{role, persona_prompt}, {role, persona_prompt}, {role, persona_prompt}]`."
- **Output**: 3 `{role, persona_prompt}` objects, used directly as system prompts in Stage 2.
- **Model**: `claude -p --model opus-4.7` via the Claude Code CLI subprocess wrapper.

**Batching to save cost**: if multiple pending candidates share near-identical domain fingerprints (embedding cosine ≥ 0.9 on the domain hint), one Stage-1 call serves all of them. Bucket candidates by domain-hint cosine at the start of each `process` run.

**Stage 2 — Haiku runs the jury** (3 parallel Haiku calls per candidate):

Each juror receives:
1. Its Opus-designed `persona_prompt` as system prompt.
2. Entity A: id, kind, description (truncated to 500 chars), top keywords.
3. Entity B: same.
4. Shared contexts (up to 5): for each, `queries[:2]` + `keywords[:5]` + `score` contribution.
5. Available predicates: name + description + constraints (subject_kinds, object_kinds, cardinality). Filtered to those whose constraints are compatible with (A.kind, B.kind).
6. Output schema: strict JSON, examples provided.

Juror response schema (per juror):

```json
{
  "role": "<as assigned by Opus>",
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

**Stage 3 — Haiku synthesises** (one call):

Input: the 3 juror responses. Output: final verdict.

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

**Failure handling (removal-discipline strict — no fallback personas)**:
- If Stage 1 (Opus design) fails for any reason (CLI error, malformed JSON, model unavailable): mark candidate with `llm_verdict='jury_design_failed'`, leave `processed_ts` NULL, retry next run. **Do not fall back to hard-coded personas** — a poorly-designed jury authoring real edges is worse than no jury running.
- If Stage 2 (juror) fails: retry that juror once. If still failing, treat as `uncertain` response in synthesis.
- If Stage 3 (synthesis) fails: retry once. If still failing, mark `llm_verdict='uncertain'`, retry next run.
- Claude CLI entirely missing at startup: exit 2 with clear message (§2.8).

### 2.5.1 Consensus rules (Stage 3 logic)

- All 3 say `edge` with same predicate → accept that predicate, use the domain-expert's statement.
- All 3 say `edge` but disagree on predicate → synthesis step picks best from the 3 options based on juror reasoning. If it picks one: accept. If it says uncertain: mark `uncertain`.
- 2 say `edge`, 1 says `no_edge` → accept if the 2 agree on predicate; mark `uncertain` otherwise.
- 2 say `no_edge`, 1 says `edge` → reject (`no_edge`), record reason.
- All 3 say `no_edge` → reject, record reason.
- Any `uncertain` or schema-violating response at juror stage → retry once, then mark `uncertain` in the DB (and optionally escalate to Sonnet per config).
- New predicate proposed: require BOTH ontologist + one other juror to agree on the new name + broadly-compatible semantics. Synthesis step double-checks existing predicates for near-duplicates (cosine on predicate descriptions ≥ 0.75 means "use existing").

### 2.6 Predicate creation

The Opus-designed ontologist juror is permitted to propose NEW predicates when existing ones don't fit. Shape:

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
- Only the synthesis step (Stage 3) can CREATE a new predicate — individual jurors only *propose*.
- Synthesis requires jury consensus: at least the ontologist + one other juror agreeing a new predicate is needed, and agreeing on name + broadly-compatible semantics.
- Before creating, synthesis checks existing predicates for semantic near-duplicates (embedding cosine on predicate descriptions ≥ 0.75 means "use existing, not new").
- New predicates go through `kg_declare_entity(kind='predicate', ...)` with the proposed constraints.
- Log to `~/.mempalace/hook_state/new_predicates.jsonl` so the user can review periodically.
- `mempalace link-author status --new-predicates` shows recently created predicates.

### 2.7 CLI + scheduling

**CLI**:
```
mempalace link-author process [--max N] [--threshold F] [--dry-run] [--no-batch-design]
mempalace link-author status [--recent N] [--new-predicates]
```

Defaults: `--max 50 --threshold 1.5`. Models are not CLI flags — they're fixed by the pipeline (Opus for design, Haiku for jury + synthesis); overrides live in config only.

**Scheduling — two independent triggers**, both OK to have:

1. **Cron/launchd/Task Scheduler** — deterministic cadence. User sets up once with commands documented in `docs/link_author_scheduling.md`.
2. **Finalize-triggered detached subprocess** — event-driven. At finalize, after the candidate upsert, check `NOW - last_run_ts >= 1h` AND there are pending candidates. If both, spawn `mempalace link-author process` detached via `subprocess.Popen(..., start_new_session=True)` on POSIX / `creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP` on Windows. Non-blocking. Update `started_ts` before the fork to prevent concurrent runs.

### 2.8 Claude Code as the LLM runtime

Spawn Claude via the non-interactive mode:
```
claude -p "<prompt>" --model opus-4.7 --output-format json    # Stage 1
claude -p "<prompt>" --model haiku    --output-format json    # Stages 2 & 3
```

The exact flags need verification on first run; the CLI is being iterated. Structure the module to isolate the subprocess call so swapping flags later is trivial.

**Detection**: check `shutil.which("claude")` on startup of `link-author process`. If missing, exit 2 with a clear message. Don't pretend to process anything. No fallback to API clients.

### 2.9 Claude is the only LLM authority

No API key. No Anthropic SDK. No OpenAI. Claude Code CLI only. This keeps:
- Zero API key management burden.
- Single-binary deployment (if claude is installed, mempalace works).
- User's existing Claude Code subscription handles the cost, no separate billing.

### 2.10 What gets retired

**Deleted entirely** (removal is removal — no stubs, no compat shims, no dormant code paths):
- `_detect_suggested_links` function in mcp_server.py.
- Its two call sites (in `_add_memory_internal` and `tool_kg_declare_entity`).
- The `pending_enrichments` field in `active_intent`.
- The `enrichment_resolutions` parameter in `tool_finalize_intent` + its acceptance path.
- `_consume_matching_enrichment` in mcp_server.py.
- `tool_resolve_enrichments` + its MCP dispatch entry + its method advertisement.
- `_ENRICHMENT_SIM_THRESHOLD`, `_ENRICHMENT_USEFULNESS_FLOOR`, `_ENRICHMENT_MAX_SUGGESTIONS`, `_REJECTION_REASON_OVERLAP_THRESHOLD`, `_MIN_ENRICHMENT_REJECT_REASON_CHARS`, `_ENRICHMENT_LOG_PATH` constants.
- `_log_enrichment_decision` helper.
- `_load_pending_enrichments_from_disk` helper.
- `_pair_already_directly_connected` helper (subsumed by the Adamic-Adar upsert's "skip if direct edge exists" check).
- `suggested_links` return field from `tool_kg_declare_entity`.
- `auto_resolved_enrichment` return field from `tool_kg_add`.
- All tests pinned to the retired enrichment API (`test_enrichment_auto_accept.py` in full, enrichment-related tests in `test_mcp_server.py`).
- The enrichment section in `PALACE_PROTOCOL`.

**Replaced by**: the `link_prediction_candidates` + `link_prediction_sources` tables + `mempalace link-author` CLI.

### 2.11 Multi-project handling — contexts are the sole project signal

The palace is a multi-tenant KB spanning accountancy, security, mempalace development, Flowserv, Dstaff, other tools, project management, marketing/comms. A candidate pair could belong to ANY of these domains.

**No project tags on entities.** Retrofit labelling is a burden the user shouldn't carry, and tags go stale the moment an entity serves two domains.

**Signal: the candidate's shared contexts ARE its project signal.** Contexts carry queries + keywords + entity-kind profiles that unambiguously characterise a domain. mempalace-dev contexts talk about "JWT refresh", "auth middleware"; accountancy contexts talk about "quarterly close", "vendor reconciliation"; marketing contexts talk about "campaign launch", "funnel". The domain hint extracted from top-5 shared contexts (§2.5 Stage 1) is the sole input to juror design.

**Handling cross-domain pairs**: if a candidate legitimately spans domains (e.g. an entity used by both accountancy and mempalace-dev contexts), the domain hint reflects both, and Opus will design a hybrid jury — which is what we want for that candidate.

**Handling a new project type**: zero configuration. First candidate from a new domain (say, a new client project in 6 months) gets appropriately-designed jurors on day 1. No retraining, no tags, no persona library to maintain.

---

## 3. Schema (migration 017)

File: `mempalace/migrations/017_link_prediction.sql`

```sql
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
    llm_jury_design_model  TEXT,   -- e.g. 'opus-4.7'
    llm_jury_exec_model    TEXT,   -- e.g. 'haiku'
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
    from_entity   TEXT NOT NULL,
    to_entity     TEXT NOT NULL,
    ctx_id        TEXT NOT NULL,
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
    design_model           TEXT DEFAULT 'opus-4.7',
    exec_model             TEXT DEFAULT 'haiku',
    errors                 TEXT
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

**Candidate accumulation (Stage 0 — called from finalize)**:
- `upsert_candidate(kg, from_entity, to_entity, weight, context_id)` — SQL-level upsert with distinct-context dedup: INSERT OR IGNORE into `link_prediction_sources`; only increment `link_prediction_candidates` when the source insert actually inserted (rowcount == 1).
- `list_pending(kg, limit, threshold)` — returns candidates above threshold, ordered by score desc.

**LLM pipeline (Stages 1-3 — called from CLI)**:
- `_build_domain_hint(kg, candidate_row) -> str` — pulls top-5 shared contexts, concatenates queries[:2] + keywords[:5] + dominant entity kinds.
- `_embed_domain_hint(hint) -> list[float]` — for batching.
- `_cluster_candidates_by_domain(candidates, threshold=0.9) -> list[list]` — groups candidates whose domain-hint embeddings have cosine ≥ threshold, so one Opus design call serves the whole cluster.
- `_design_jury(domain_hint) -> list[dict]` — Stage 1. Opus 4.7 call. Returns `[{role, persona_prompt}, ...]` × 3. Raises on failure (caller marks `jury_design_failed`).
- `_run_juror(persona, entity_a, entity_b, shared_contexts, predicates) -> dict` — Stage 2. Single Haiku call with persona as system prompt.
- `_synthesise(juror_verdicts, existing_predicates) -> dict` — Stage 3. Haiku call. Applies consensus rules from §2.5.1. Handles new-predicate proposals with near-duplicate check (cosine ≥ 0.75).
- `author_candidate(kg, row) -> dict` — orchestrates Stages 1-3 for one candidate. Returns verdict dict + edge payload. Handles retries and failure modes per §2.5.
- `process(kg, max=50, threshold=1.5, dry_run=False, batch_design=True) -> dict` — the CLI-entry function. Iterates candidates (clustered by domain hint if batching enabled), runs the pipeline, writes edges, records verdicts, appends a row to `link_author_runs`.

**Subprocess + predicate + dispatch helpers**:
- `_call_claude(prompt, model) -> str` — subprocess wrapper around `claude -p --model <model> --output-format json`. Returns raw stdout. Single point of CLI coupling.
- `_parse_json_response(text, schema_name) -> dict` — schema-validates Claude's JSON output.
- `_maybe_create_predicate(kg, proposal) -> str | None` — validates proposal, checks for near-duplicates via cosine on predicate descriptions (≥ 0.75 → use existing), creates via `kg.add_entity(kind='predicate', ...)` if new. Logs to `~/.mempalace/hook_state/new_predicates.jsonl`. Returns final predicate name (existing or newly created).
- `_dispatch_if_due(kg, interval_hours)` — checks last `link_author_runs.started_ts`; if ≥ interval_hours old AND there are pending candidates, spawns `mempalace link-author process` as detached subprocess. Writes new `started_ts` BEFORE fork to prevent concurrent runs. Called by finalize.

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
    # Textbook Adamic-Adar: 1/log(|entities|). We already skipped
    # len < 2, so min denominator is log(2) ≈ 0.693 — no zero-div risk
    # and no +1 fudge (which would underweight small focused contexts,
    # the opposite of what AA wants).
    weight = 1.0 / math.log(len(entities))
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
    # Pipeline models (distinct for each stage)
    "jury_design_model": "opus-4.7",      # Stage 1
    "jury_execution_model": "haiku",      # Stage 2 (3 parallel)
    "synthesis_model": "haiku",           # Stage 3

    # Cost optimisation — batch design calls across similar candidates
    "batch_design_by_domain_similarity": True,
    "batch_domain_cosine_threshold": 0.9,

    # Dispatch cadence
    "interval_hours": 1,

    # Candidate filtering
    "threshold": 1.5,
    "max_per_run": 50,

    # Failure handling
    "retry_uncertain_next_run": True,
    "rejection_cooldown_days": 30,        # don't re-eval no_edge candidates for 30d
    "escalate_uncertain_to": "sonnet"     # null to disable; otherwise retry uncertain verdicts on this model
  }
}
```

Loaded at CLI start, overridden by CLI flags where applicable (threshold, max_per_run, dry_run, no_batch_design).

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
  - Adamic-Adar math: 2 entities in 3 distinct contexts of varying size → expected score = Σ 1/log(size).
  - **Distinct-context dedup**: reusing the same context 5 times with the same pair contributes exactly once. Second and subsequent reuses do NOT re-increment the score.
  - Canonical-ordering PK: (B, A) and (A, B) collapse to same row on upsert.
  - Direct-edge short-circuit: entity pair with existing triple is NOT upserted.
  - Per-context gating: fresh context OR negative feedback OR low-per-channel-relevance → no upsert.
  - Seeding covers all scopes: intent + operation + search contexts all feed the upsert when reused.
  - similar_to neighbours do NOT seed: a context that's only similar (not directly reused) contributes nothing.

- `tests/test_link_author_cli.py`
  - Happy-path single candidate with mocked subprocess: Opus returns 3 personas → 3 Haiku jurors unanimous `edge` → synthesis accepts → edge created via kg_add, verdict + personas recorded in `link_prediction_candidates`.
  - Jury disagreement (2-1 split, predicate mismatch) → `uncertain` verdict, no edge, candidate unprocessed (retries next run).
  - All-no_edge jury → rejection, no edge, `processed_ts` set, 30-day cooldown gate.
  - New predicate proposal with ontologist + one other agreeing → predicate created via `kg.add_entity(kind='predicate')` + edge created.
  - New predicate proposal that near-duplicates existing (cosine ≥ 0.75) → uses existing predicate instead.
  - New predicate proposal without consensus → rejected as `uncertain`, no predicate created.
  - Opus design call fails (subprocess error or malformed JSON) → candidate marked `jury_design_failed`, NO fallback personas used, retry next run.
  - Domain-hint batching: 3 candidates with cosine ≥ 0.9 on hints → one Opus call, one persona set used for all three.
  - Claude CLI missing at startup → exit 2 with clear message.

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

Branch from `main` as `feat/link-author`. **Three functional commits + PR**, each one landing a user-observable capability — no dormant artifacts, no intermediate states where plumbing sits unused waiting for a later commit.

### Commit 1 — Retire enrichment entirely

Pure-delete commit. Green after this commit with zero regressions for any real flow (enrichment was never actually useful).

- Delete `_detect_suggested_links`, `_consume_matching_enrichment`, `_pair_already_directly_connected`, `_log_enrichment_decision`, `_load_pending_enrichments_from_disk`.
- Delete `tool_resolve_enrichments` function + its MCP dispatch entry + method advertisement.
- Delete call sites of `_detect_suggested_links` in `_add_memory_internal` and `tool_kg_declare_entity`.
- Delete `suggested_links` return field from `tool_kg_declare_entity`.
- Delete `auto_resolved_enrichment` return field from `tool_kg_add`.
- Delete `enrichment_resolutions` parameter from `tool_finalize_intent` + acceptance path.
- Delete `_STATE.pending_enrichments` field and all readers.
- Delete constants: `_ENRICHMENT_SIM_THRESHOLD`, `_ENRICHMENT_USEFULNESS_FLOOR`, `_ENRICHMENT_MAX_SUGGESTIONS`, `_REJECTION_REASON_OVERLAP_THRESHOLD`, `_MIN_ENRICHMENT_REJECT_REASON_CHARS`, `_ENRICHMENT_LOG_PATH`.
- Delete enrichment section from `PALACE_PROTOCOL`.
- Delete `tests/test_enrichment_auto_accept.py` entirely.
- Grep + delete any remaining enrichment-related tests in `tests/test_mcp_server.py` (`pending_enrichments`, `resolve_enrichments`, `suggested_links`).

**User-observable result**: `resolve_enrichments` tool is gone. `kg_declare_entity` no longer returns `suggested_links`. No `pending_enrichments` in state. Suite green. Cleaner foundation for everything that follows.

### Commit 2 — Analytical candidate pipeline end-to-end

Signal starts accumulating in the KG. No LLM yet, but the user can `SELECT * FROM link_prediction_candidates ORDER BY score DESC` and watch real candidates appear as they work.

- Migration 017: `link_prediction_candidates` + `link_prediction_sources` + `link_author_runs` tables (§3).
- `validate_context` default `entities_min=1` (§5.1).
- `tool_declare_operation` gains mandatory `entities` parameter (§5.2).
- `PALACE_PROTOCOL` updated for mandatory entities (§5.7).
- `mempalace/link_author.py` created with:
  - `upsert_candidate` (distinct-context dedup via sources table).
  - `list_pending`.
  - `_dispatch_if_due` (stub — logs "CLI not yet registered" and returns; gets wired in commit 3).
- `tool_finalize_intent` gains the candidate-upsert loop iterating `contexts_touched_detail` (§5.2).
- Test migration across all context-dict and `tool_declare_operation` call sites (~15-20 sites).
- New tests: `test_link_prediction_candidates.py` (math + dedup + canonical ordering + direct-edge skip + per-scope seeding + no-similar_to-seeding).
- New tests: `test_mandatory_entities.py`.

**User-observable result**: every intent / operation / search now requires entities. Every finalize with net-positive reused-context feedback grows the candidate queue. The queue is queryable. The math is tested.

### Commit 3 — LLM jury pipeline end-to-end

The queue drains. Real edges land in the KG.

- `mempalace/link_author.py` gains:
  - `_build_domain_hint`, `_embed_domain_hint`, `_cluster_candidates_by_domain`.
  - `_design_jury` (Stage 1 — Opus 4.7 subprocess call).
  - `_run_juror` (Stage 2 — Haiku subprocess call with Opus-designed persona).
  - `_synthesise` (Stage 3 — Haiku subprocess call).
  - `author_candidate` (orchestrates Stages 1-3 with retries).
  - `_maybe_create_predicate` (near-duplicate check + logging).
  - `process` (CLI-entry function: cluster → design → execute → synthesise → write).
  - `_call_claude`, `_parse_json_response` (subprocess coupling).
- `_dispatch_if_due` wired fully (spawns detached subprocess, updates `started_ts` pre-fork).
- `mempalace/cli.py` gains `link-author process` and `link-author status` subcommands.
- `mempalace/config.py` gains `link_author` config section (§5.5).
- `docs/link_author_scheduling.md` created (cron / launchd / Task Scheduler examples).
- `PALACE_PROTOCOL` final update referencing the link-author pipeline for the agent's mental model.
- New tests: `test_link_author_cli.py` with fully-mocked `_call_claude` covering all verdict paths + Opus design failure + domain-hint batching + new-predicate consensus + near-duplicate dedup + CLI-missing exit.

**User-observable result**: `mempalace link-author process` actually works. Candidates get edges authored. Auto-dispatch fires on finalize when due.

### Commit 4 — PR and merge

`gh pr create`, run full suite, merge to main when green.

Expected total: ~800-1100 LOC change + ~20-25 new tests + ~15-20 test migrations + 1 deleted test file. Commit 1 is a net deletion; commits 2 and 3 are the substantive additions.

---

## 8. Locked decisions (all user-confirmed; no overrides needed)

| Question | Locked value |
|---|---|
| Jury-design model (Stage 1) | **Opus 4.7** — designs 3 personas per candidate (or per domain-cluster if batched) |
| Jury-execution model (Stage 2) | **Haiku** — runs 3 parallel jurors with Opus-designed personas |
| Synthesis model (Stage 3) | **Haiku** — applies consensus rules, handles new-predicate proposals |
| Persona fallback | **NONE** — jury design failure → mark `jury_design_failed`, retry next run. No hard-coded fallback personas. |
| Design-call batching | **ON** — candidates with domain-hint cosine ≥ 0.9 share one Opus call |
| Interval | 1 hour between dispatches |
| Threshold | score ≥ 1.5 (distinct-contexts-only accumulation makes `shared_context_count ≥ 2` redundant — any pair that crosses 1.5 came from ≥ 2 contexts by construction) |
| Accumulation rule | **Option A — distinct contexts only** (via `link_prediction_sources` dedup table). Same context reused N times contributes exactly once. |
| Max per run | 50 candidates |
| Candidate seeding scope | **All three emit sites** — intent, operation, search — via `contexts_touched_detail`. `similar_to` neighbours do NOT seed. |
| Indirect-connection handling | Pairs connected via ≥ 2-hop paths are still eligible candidates; only DIRECT edges skip |
| Escalation | ON — uncertain verdict re-runs on Sonnet (configurable) |
| Claude CLI discovery | HARD REQUIRE. Exit 2 with clear message if missing. No API-client fallback. |
| New-predicate creation | permitted via jury consensus (ontologist + ≥1 other), with near-duplicate check (cosine ≥ 0.75 on descriptions) |
| Dispatch on finalize | ON — detached subprocess, 1h gate, non-blocking |
| Project-type detection | **Contexts are the sole project signal** — no entity tags, no auto-clustering. Domain hint built from top-5 shared contexts per candidate drives jury design. |
| Rejection cooldown | 30 days before re-evaluating `no_edge` verdicts |
| Retired enrichment machinery | **DELETED ENTIRELY** — no stubs, no tool endpoint, no fallback code paths. Removal = removal. |
| Mandatory entities | ENFORCED on declare_intent / declare_operation / kg_search, ≥ 1 |
| flat-list memory_feedback | already retired (landed via feat/rocchio-multi-context-dedup) |
| Commit structure | 3 functional commits + PR (§7) — no dormant-artifact intermediate states |

---

## 9. Ship order (reinstall boundary)

**Before reinstall**:
1. `feat/rocchio-multi-context-dedup` merged to main via PR #61 (merge commit `da4afcb`). ✅ done.
2. Branch `feat/link-author` from main.
3. Execute this plan end-to-end (3 functional commits per §7).
4. Merge to main.
5. Adrian reinstalls with the complete pipeline.

**After reinstall**:
- Threshold calibration based on observed candidate queue depth + jury verdict distribution.
- Domain-hint batching threshold tuning (0.9 may be too tight or too loose in practice).
- Opus design prompt iteration — first few runs will show whether juror personas need prompt tweaks.
- New-predicate review: periodic human check of the `new_predicates.jsonl` log to catch predicate-space explosion.

---

## 10. What-not-to-do (guard rails)

- **Do NOT auto-create `co_appears_in` or other generic predicates.** The LLM picks a real predicate or the edge doesn't get created.
- **Do NOT seed candidates from `similar_to` neighbourhoods, cosine-surfaced memories, keyword-matched memories, or any embedding-/statistical-similarity channel.** Only direct entity-in-context co-occurrence seeds candidates. Anything else reintroduces the blind-cosine poison this rewrite exists to retire.
- **Do NOT fall back to hard-coded juror personas if Opus design fails.** Mark the candidate `jury_design_failed` and retry next run. A poorly-designed jury authoring real edges is worse than no jury running.
- **Do NOT keep any enrichment machinery around as a fallback.** It was low-value busywork; it's retired in full. Removal is removal — no stubs, no deprecated tool endpoints, no dormant code paths. Same rule applies to changes: if a function signature or behaviour changes, every call-site changes with it in the same commit, no parallel old/new paths.
- **Do NOT make entities optional "for backward compatibility".** There's no legacy data; the contract is mandatory from day 1.
- **Do NOT block finalize on link-author failures.** Dispatch is fire-and-forget (wrapped in try/except with zero propagation).
- **Do NOT use an Anthropic API key.** Claude Code CLI only. No `anthropic` / `openai` package imports.
- **Do NOT allow individual jurors to create new predicates.** Only the synthesis step, with jury consensus (ontologist + ≥1 other) and near-duplicate check.
- **Do NOT accumulate score per context reuse.** Distinct-contexts-only (Option A) via the `link_prediction_sources` dedup table. Re-observing the same context is one observation, not multiple.
- **Do NOT introduce project tags on entities.** Multi-project handling is derived purely from shared-context domain fingerprints.
- **Do NOT re-process rejected candidates forever.** `processed_ts` + 30-day cooling-off gate re-evaluation.
- **Do NOT land dormant artifacts.** Every commit in §7 must produce a user-observable change. No "plumbing commit now, wired up in 2 commits".

---

## 11. Morning sanity checks after implementation

1. Full suite green (`pytest -q --ignore=tests/benchmarks --ignore=tests/manual`). Expect ~930-950 passed, 1 skipped (base 916 + ~15-30 new tests).
2. `mempalace link-author process --dry-run` works against an empty palace (exits clean, zero candidates processed, reports "no pending candidates").
3. Reinstall mempalace plugin. Declare 3 intents, each with 3-5 entities overlapping across intents. Finalize each with positive feedback. Check `link_prediction_candidates` — should have rows for unconnected pairs, and `link_prediction_sources` should show one row per `(pair, ctx_id)` tuple with no duplicates even if contexts were reused.
4. Reuse the SAME context 3 times across 3 intents. Verify the candidate score does NOT triple — distinct-context dedup kicked in.
5. Run `mempalace link-author process` manually. Watch for:
   - Opus 4.7 being called via `claude -p --model opus-4.7`.
   - 3 Haiku juror calls in parallel per candidate (or per cluster).
   - `link_prediction_candidates` rows gaining `llm_verdict`, `llm_predicate`, `llm_statement`, `llm_jury_personas`.
   - `link_author_runs` getting a row with `design_calls < candidates_processed` (batching working).
6. Inspect the created edges via `mempalace_kg_query` — they should have non-generic predicates, meaningful statements, proper `statement` values (not autogenerated underscore-to-space fallbacks).
7. Verify `_dispatch_if_due` fires: finalize an intent, wait for the detached subprocess, check `link_author_runs` for a new row within a minute.
8. Kill the `claude` binary temporarily (rename it). Run `mempalace link-author process`. Expect exit 2 with clear message — no silent fallback.

---

## 12. References

- Liben-Nowell & Kleinberg 2007 JASIST — https://onlinelibrary.wiley.com/doi/abs/10.1002/asi.20591
- Adamic & Adar 2003 / Wikipedia — https://en.wikipedia.org/wiki/Adamic%E2%80%93Adar_index
- Shu et al. 2024 "KG-LLM for Link Prediction" — https://arxiv.org/abs/2403.07311
- Du et al. 2024 "Multiagent Debate" — search arXiv.
- Graphiti — https://github.com/getzep/graphiti
- Rocchio 1971 via Manning/Raghavan/Schütze IR book Ch.9 — https://nlp.stanford.edu/IR-book/pdf/09expand.pdf
