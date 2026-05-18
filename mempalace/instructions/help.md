# MemPalace

AI memory system. Store everything, find anything. Local, free, no API key.

---

## Slash Commands

| Command              | Description                    |
|----------------------|--------------------------------|
| /mempalace:init      | Install and set up MemPalace   |
| /mempalace:search    | Search your memories           |
| /mempalace:mine      | Mine projects and conversations|
| /mempalace:status    | Palace overview and stats      |
| /mempalace:help      | This help message              |

---

## MCP Tools

### The Context object (every read AND every write speaks it -- P4.1+)
All search and declaration tools take a single `context` object. Shape:

    context = {
      queries:  list[str]  # 2-5 perspectives -- drive Channel A multi-view cosine
      keywords: list[str]  # 2-5 caller-provided exact terms -- drive Channel C
                           # (no auto-extraction -- the caller knows what matters)
      entities: list[str]  # 0+ related/seed entity ids -- drive Channel B graph BFS
    }

Each query is embedded as a separate vector; the collection IS the multi-view
fingerprint. Stored on entities + edges so feedback applies by MaxSim similarity.

### Boot + Search (read)
- mempalace_wake_up(agent, context?) -- Session boot: protocol + L0 identity + L1 ranked context + declared entities/predicates/intent types. `context` REQUIRED on first wake_up for a new agent on this palace (cold-start lock 2026-05-01).
- mempalace_kg_search(context, agent, kind?, limit?, sort_by?, time_window?) -- Unified 3-channel search (cosine + keyword + graph, RRF merged) across memories + entities.
- mempalace_kg_query(entity) -- Exact entity-ID lookup, returns all current edges. Supports comma-separated batch.
- mempalace_kg_stats -- Palace overview: counts by wing/room/kind + graph connectivity in one call.
- mempalace_kg_timeline(entity?) -- Chronological story for an entity (or everything).
- mempalace_kg_list_declared -- Entities declared in this session.
- mempalace_bg_status(streams?, limit?) -- Tail per-stream telemetry from `~/.mempalace/hook_state/` (gate_log, state_judge_log, retrieval_log, feedback_auto_log, conflict_resolver_log, bg_quality_log, wrapper_log, mcp_io_log, search_log, hook_errors, faulthandler). Read-only diagnostic.
- mempalace_pending_user_intents -- Restart-recovery read. Returns `{session_id, count, pending}` for the pending user-message queue persisted at `~/.mempalace/hook_state/pending_user_messages_<sid>.json`. Call after a server restart or context compaction to rediscover which user messages still need a `declare_user_intents` coverage call. No intent required (read-only); safe pre-bootstrap.

### Knowledge Graph (write)
- mempalace_kg_declare_entity(name?, kind, context, content?, importance, added_by, ...) -- Declare any entity. `kind="memory"` creates a memory (requires wing/room/slug + `content` verbatim text); for other kinds `queries[0]` is the canonical description. `kind="predicate"` requires constraints in `properties`. Multi-vector storage, multi-view collision, keyword index (P4.2).
- mempalace_kg_add(subject, predicate, object, context, valid_from?) -- Add a triple. `context` records WHY the edge was added; persisted as `triples.creation_context_id` so feedback transfers via MaxSim (P4.3).
- mempalace_kg_add_batch(edges, context) -- Batch add with a shared Context (or per-edge overrides).
- mempalace_kg_update_entity(entity, ...) -- Unified update for both memories and KG nodes (P3.4).
- mempalace_kg_invalidate(subject, predicate, object) -- Soft-delete a single fact.
- mempalace_kg_delete_entity(entity) -- Soft-delete an entire entity or memory (P3.6).
- mempalace_kg_merge_entities(source, target) -- Merge entities; source becomes alias.

<!-- mempalace_resolve_conflicts removed in v3.7.20: conflicts are resolved
     end-to-end by the background Haiku resolver (conflict_resolver_auto).
     Inspect decisions via mempalace_bg_status(streams=['conflict_resolver_log']). -->

### Intent System
- mempalace_declare_user_intents(contexts, agent) -- MANDATORY before any other tool call when a user message is pending. Top tier of the activity hierarchy (Motive/Strategy). Declare one context per user-intent; the union of `user_message_ids` across contexts must cover every pending id. `no_intent=true` + `no_intent_clarified_with_user=true` allowed only after an explicit AskUserQuestion confirmation.
- mempalace_declare_intent(intent_type, slots, context, agent, budget, cause_id, initial_intent_state) -- Declare what you intend to do; returns permissions + injected memories (P4.4). `cause_id` links upward to the user-intent context (or 'autonomous'). `initial_intent_state` is validated against state_schemas.intent_state.
- mempalace_declare_operation(tool, args_summary, context, agent, slots?, state_deltas?) -- MANDATORY before every non-mempalace tool call (except ALWAYS_ALLOWED carve-outs: TodoWrite, Skill, Agent, ToolSearch, AskUserQuestion, ExitPlanMode, Task*). Records the cue so retrieval surfaces memories matching your actual intent (not the shape of the tool args). Returns memories + past_operations (good_precedents / avoid_patterns) drawn from MaxSim neighbourhood of this operation context.
- mempalace_active_intent -- Show current intent + remaining budget + auto-applied states.
- mempalace_extend_intent(budget) -- Add to budget without redeclaring.
- mempalace_finalize_intent(slug, outcome, summary, content, agent, state_deltas?, gotchas?, learnings?) -- Capture what happened. The async-Haiku rater (mempalace.feedback_auto) rates retrieved memories + operations post-finalize; no agent ratings required.
- mempalace_challenge_state_change(entity_id, schema_id, target_rev_id, justification, agent, action='restore'|'info_only') -- Phase 3 Slice B JTMS challenge: agent disputes a prior state revision. `restore` re-promotes a target_rev_id payload as the new current state; `info_only` records the dispute without rolling back. All challenges land in mempalace_state_revision_challenges table for audit.

### Agent Diary
- mempalace_diary_write -- Write a diary entry (concise prose, delta-only).
- mempalace_diary_read -- Read recent diary entries.

---

## CLI Commands

    mempalace init <dir>                  Initialize a new palace
    mempalace mine <dir>                  Mine a project (default mode)
    mempalace mine <dir> --mode convos    Mine conversation exports
    mempalace search "query"              Search your memories
    mempalace split <dir>                 Split large transcript files
    mempalace wake-up                     Load palace into context
    mempalace compress                    Compress palace storage
    mempalace status                      Show palace status
    mempalace repair                      Rebuild vector index
    mempalace mcp                         Show MCP setup command
    mempalace hook run                    Run hook logic (for harness integration)
    mempalace instructions <name>         Output skill instructions

---

## Auto-Save Hooks

- Stop hook -- Automatically saves memories every 15 messages. Counts human
  messages in the session transcript (skipping command-messages). When the
  threshold is reached, blocks the AI with a save instruction. Uses
  ~/.mempalace/hook_state/ to track save points per session. If
  stop_hook_active is true, passes through to prevent infinite loops.

- PreCompact hook -- Emergency save before context compaction. Always blocks
  with a comprehensive save instruction because compaction means the AI is
  about to lose detailed context.

Hooks read JSON from stdin and output JSON to stdout. They can be invoked via:

    echo '{"session_id":"abc","stop_hook_active":false,"transcript_path":"..."}' | mempalace hook run --hook stop --harness claude-code

---

## Architecture

    Wings (projects/people)
      +-- Rooms (topics)
            +-- Closets (summaries)
                  +-- Memories (verbatim memories)

    Halls connect rooms within a wing.
    Tunnels connect rooms across wings.

The palace is stored locally using ChromaDB for vector search and SQLite for
metadata. No cloud services or API keys required.

---

## Getting Started

1. /mempalace:init -- Set up your palace
2. /mempalace:mine -- Mine a project or conversation
3. /mempalace:search -- Find what you stored
