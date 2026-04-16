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

### The Context object (every read AND every write speaks it — P4.1+)
All search and declaration tools take a single `context` object. Shape:

    context = {
      queries:  list[str]  # 2-5 perspectives — drive Channel A multi-view cosine
      keywords: list[str]  # 2-5 caller-provided exact terms — drive Channel C
                           # (no auto-extraction — the caller knows what matters)
      entities: list[str]  # 0+ related/seed entity ids — drive Channel B graph BFS
    }

Each query is embedded as a separate vector; the collection IS the multi-view
fingerprint. Stored on entities + edges so feedback applies by MaxSim similarity.

### Boot + Search (read)
- mempalace_wake_up(wing="ga") -- Session boot: protocol + L0 identity + L1 ranked context + declared entities/predicates/intent types
- mempalace_kg_search(context, agent, wing?, room?, kind?, limit?) -- Unified 3-channel search (cosine + keyword + graph, RRF merged) across memories + entities.
- mempalace_kg_query(entity) -- Exact entity-ID lookup, returns all current edges. Supports comma-separated batch.
- mempalace_kg_stats -- Palace overview: counts by wing/room/kind + graph connectivity in one call.
- mempalace_kg_timeline(entity?) -- Chronological story for an entity (or everything).
- mempalace_kg_list_declared -- Entities declared in this session.
- mempalace_traverse(start_room, max_hops?) -- Walk the graph from a room across wings.
- mempalace_get_aaak_spec -- AAAK dialect spec.

### Knowledge Graph (write)
- mempalace_kg_declare_entity(name?, kind, context, content?, importance, added_by, ...) -- Declare any entity. `kind="memory"` creates a memory (requires wing/room/slug + `content` verbatim text); for other kinds `queries[0]` is the canonical description. `kind="predicate"` requires constraints in `properties`. Multi-vector storage, multi-view collision, keyword index (P4.2).
- mempalace_kg_add(subject, predicate, object, context, valid_from?) -- Add a triple. `context` records WHY the edge was added; persisted as `triples.creation_context_id` so feedback transfers via MaxSim (P4.3).
- mempalace_kg_add_batch(edges, context) -- Batch add with a shared Context (or per-edge overrides).
- mempalace_kg_update_entity(entity, ...) -- Unified update for both memories and KG nodes (P3.4).
- mempalace_kg_invalidate(subject, predicate, object) -- Soft-delete a single fact.
- mempalace_kg_delete_entity(entity_id) -- Soft-delete an entire entity or memory (P3.6).
- mempalace_kg_merge_entities(source, target) -- Merge entities; source becomes alias.
- mempalace_resolve_conflicts(actions=[...]) -- Resolve duplicates/contradictions.

### Intent System
- mempalace_declare_intent(intent_type, slots, context, agent, budget?) -- Declare what you intend to do; returns permissions + injected memories (P4.4). `context.queries` drive retrieval, `context.keywords` drive the keyword channel, `context.entities` seed the graph BFS.
- mempalace_active_intent -- Show current intent + remaining budget.
- mempalace_extend_intent(budget) -- Add to budget without redeclaring.
- mempalace_finalize_intent(slug, outcome, summary, agent, memory_feedback=[...]) -- Capture what happened. memory_feedback is MANDATORY.

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
