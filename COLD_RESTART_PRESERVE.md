# MemPalace Cold Restart -- Preservation Document

Created: 2026-04-17
Purpose: Reference document for data that must be manually recreated after cold restart.

---

## 1. DATABASE LOCATIONS (to wipe)

| DB | Path | Purpose |
|----|------|---------|
| **KG SQLite** | `~/.mempalace/knowledge_graph.sqlite3` | Entities, triples, edge feedback |
| **ChromaDB** | `~/.mempalace/palace/` (entire directory) | Vector store for memories (records) |
| **Entity Registry** | `~/.mempalace/entity_registry.json` | Person/project recognition |
| **Config** | `~/.mempalace/config.json` | Palace config (keep or recreate) |

To cold restart: delete `knowledge_graph.sqlite3` and the `palace/` directory.
Restarting MCP will trigger `seed_ontology()` automatically (checks for `thing` entity).

---

## 2. CREDENTIALS & SERVER ACCESS

### Database
```
DATABASE_URL: postgres://paperclip:paperclip@localhost:5432/paperclip
```
- User: `paperclip`
- Password: `paperclip` (reverted to default on 2026-04-14)
- Previous secure password: `Pc2026SecureDB99` (set 2026-04-09 for DSPA-1896, rolled back during recovery)
- Runs in Docker container `paperclip-db-1` on port 5432

### Server
- Paperclip server: `localhost:3100`
- Dev server (worktrees): `localhost:3101`
- Startup command: `pnpm dev:server` from `D:/Flowsev/repo`
- Depends on: Docker PostgreSQL running first

### Temporal (for workflows)
- Start via: `docker compose` (Temporal infrastructure)
- Temporal UI available after startup
- Worker runs inside the Paperclip server process

---

## 3. AGENT UUID REGISTRY

| Agent | UUID |
|-------|------|
| **Director** | `49893ba7-6992-49f2-be62-2fff2f784cb1` |
| **Manager** | `adbd2033-f9f4-4edf-8c09-efb26bab6de9` |
| **Accountant** | `04f1aede-d99e-4a06-a854-6db622f936a0` |
| Technical Lead | CHECK: `smaugho/dspot-company-data` repo |
| Paperclip Engineer | CHECK: `smaugho/dspot-company-data` repo |
| Frontend Engineer | CHECK: `smaugho/dspot-company-data` repo |
| DSO | CHECK: `smaugho/dspot-company-data` repo |
| DevSecFinOps | CHECK: `smaugho/dspot-company-data` repo |
| Communications | CHECK: `smaugho/dspot-company-data` repo |
| QA Engineer | CHECK: `smaugho/dspot-company-data` repo |

NOTE: Full UUID list lives in the Paperclip server DB (agents table) and in
`smaugho/dspot-company-data` onboarding assets. Verify before recreating.

---

## 4. WORKFLOW SYSTEM (HIGH PRIORITY)

### Architecture
- Step-based workflow execution engine with Temporal integration
- CEL validation, retry/timeout logic
- Backend: `server/src/services/workflows.ts` + `server/src/temporal/`

### Feature Flags (both default OFF, CS-9 compliant)
| Flag | Purpose | Introduced In |
|------|---------|---------------|
| `enableWorkflowEditor` | Gates workflow UI -- CRUD vs read-only | DSPA-1944, PR #9 |
| `enableTemporalWorkflowSignals` | Routes agent submissions through Temporal signals | DSPA-1848, PR #10 |

Both managed in `instanceExperimentalSettingsSchema`.

### Key PRs (all merged to main)
- PR #8 (DSPA-1943): Temporal Docker infrastructure
- PR #9 (DSPA-1944): Workflow editor UI feature flag
- PR #10 (DSPA-1848): Temporal signals for agent submissions
- PR #12 (DSPA-1857): Temporal integration refinements
- PR #13 (DSPA-945): Additional workflow fixes

### Known Gotchas
- Workflow `steps` field is an **array**, not an object
- Temporal DB desync bug -- worker and DB can get out of sync
- Server restart blocked by active workflow runs
- FF merge can crash Vite server

---

## 5. PREDICATES TO RECREATE (not in seeder)

These 7 predicates were created at runtime and should be re-declared manually:

| Predicate | Description | Importance |
|-----------|-------------|------------|
| `adopted_from` | Fork adoption -- subject adopted code from upstream PR | 3 |
| `has_status` | Lifecycle status (todo, in_progress, done, merged) | 3 |
| `implements` | PR/code implements a task | 3 |
| `introduced_in` | Feature first created in a task/PR | 2 |
| `parent_of` | Parent coordination issue -> child task | 3 |
| `reviews` | Review task reviews a PR/artifact | 3 |
| `supersedes` | One entity replaces/supersedes another | 3 |

NOTE: `has_memory` is DROPPED -- it was replaced by better relationships
(described_by, evidenced_by, mentioned_in, derived_from, session_note_for).

---

## 6. CLASS TO RECREATE

| Class | Description | Importance |
|-------|-------------|------------|
| `feature_flag` | Boolean experimental setting gating a feature (CS-9) | 3 |

---

## 7. INTENT TYPES TO RECREATE

### Generic (recommended for all projects)
| Type | Parent | Purpose |
|------|--------|---------|
| `edit_file` | modify | Editing existing files |
| `implement_feature` | modify | Full feature implementation (+ Bash) |
| `git_commit_push` | execute | Git add/commit/push operations |
| `run_tests` | execute | Running test suites |
| `write_tests` | modify | Writing + running tests (+ Bash) |
| `diagnose_failure` | inspect | Investigating errors (+ Bash) |

### DSpot-specific (recreate as needed by agents)
- `tl_heartbeat`, `tl_heartbeat_cycle`, `tl_light_heartbeat`
- `dso_ui_smoke_test`, `ui_smoke_test`
- `pfe_verify_ui`, `pfe_visual_evidence`
- `deploy`, `refactor`
- `implement_scoring_module`, `run_pytest_suite`

---

## 8. CORE ENTITIES TO RECREATE

| Entity | Kind | Importance | Description |
|--------|------|------------|-------------|
| `adrian` | person | 5 | Adrian Rivero, founder of DSpot Sp. z o.o., GitHub: smaugho / DSpotDevelopers |
| `paperclip-server` | system | 5 | Paperclip server on port 3100, depends on paperclip_database |
| `paperclip_database` | system | 5 | PostgreSQL in Docker (paperclip-db-1), port 5432 |
| `docker-environment` | environment | 4 | Docker runtime hosting DB and Temporal |
| `workflow_system` | entity | 5 | Workflow engine with Temporal, see section 4 |
| `enableWorkflowEditor` | feature_flag | 4 | Gates workflow editor UI |
| `enableTemporalWorkflowSignals` | feature_flag | 4 | Routes submissions through Temporal |
| `ga-agent` | agent | 4 | General Assistant agent |
| `technical_lead_agent` | agent | 4 | TL agent, runs_in paperclip-server |
| `flowsev-repository` | project | 4 | DSpotDevelopers/flowsev fork |
| `fork_customization_inventory` | entity | 4 | FC list in smaugho/dspot-company-data |

---

## 9. OPERATIONAL GOTCHAS TO RE-PERSIST

| Gotcha | Affects | Description |
|--------|---------|-------------|
| `server-restart-blocked-by-active-runs` | paperclip-server | Can't restart while workflow runs active |
| `issue_release_resets_status` | paperclip-server | Releasing an issue resets its status |
| `ff-merge-vite-crash` | paperclip-server | Fast-forward merge crashes Vite |
| `checkout_requires_agentId_expectedStatuses` | paperclip_api | API checkout needs both params |
| `seed_ontology ON CONFLICT` | mempalace | Seeder overwrites force-updated entities |
| `gh-pr-review-approve-fails-same-account` | technical_lead_agent | Can't approve own PRs |

---

## 10. COMPANY STANDARDS REFERENCE

CS-1 through CS-17 are defined in the DSO code review workflow.
Key ones to remember:
- **CS-9**: Mandatory feature flags for all new features (default OFF)
- **CS-15**: Visual evidence (Playwright screenshots) required for UI changes
- **CS-17**: Traceability -- DSPA ticket number in PR title

Full standards live in `smaugho/dspot-company-data` agent instructions.

---

## 11. CONTEXT ARCHITECTURE -- v1 Declaration

Post cold restart, the current architecture is **Context v1**:
- Unified `{queries, keywords, entities}` contract
- 3-channel retrieval: Channel A (multi-view cosine), Channel B (graph BFS), Channel C (keyword)
- Reciprocal Rank Fusion merging
- Provenance-aware scoring (session + intent + agent affinity)
- Time-window soft decay
- MaxSim feedback transfer

No legacy formats to maintain. Phase codes (P2-P6) stripped in commit `5ec1801`.

---

## 12. KEY PATHS

| Path | Purpose |
|------|---------|
| `D:/Flowsev/repo` | Main Paperclip checkout |
| `D:/Flowsev/mempalace` | MemPalace repository |
| `~/.mempalace/` | Palace data directory |
| `smaugho/dspot-company-data` | Private company data repo (agent instructions, fork inventory) |
