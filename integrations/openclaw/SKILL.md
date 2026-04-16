---
name: mempalace
description: "MemPalace — Local AI memory with 96.6% recall. Semantic search, temporal knowledge graph, palace architecture (wings/rooms/drawers). Free, no cloud, no API keys."
version: 3.1.0
homepage: https://github.com/milla-jovovich/mempalace
user-invocable: true
metadata:
  openclaw:
    emoji: "\U0001F3DB"
    os:
      - darwin
      - linux
      - win32
    requires:
      anyBins:
        - mempalace
        - python3
    install:
      - id: mempalace-pip
        kind: uv
        label: "Install MemPalace (Python, local ChromaDB)"
        package: mempalace
        bins:
          - mempalace
---

# MemPalace — Local AI Memory System

You have access to a local memory palace via MCP tools. The palace stores verbatim conversation history and a temporal knowledge graph — all on the user's machine, zero cloud, zero API calls.

## Architecture

- **Wings** = people or projects (e.g. `wing_alice`, `wing_myproject`)
- **Halls** = categories (facts, events, preferences, advice)
- **Rooms** = specific topics (e.g. `chromadb-setup`, `riley-school`)
- **Drawers** = individual memory chunks (verbatim text)
- **Knowledge Graph** = entity-relationship facts with time validity

## Protocol — FOLLOW THIS EVERY SESSION

1. **ON WAKE-UP**: Call `mempalace_wake_up(wing="ga")` to load protocol + L0 identity + L1 ranked context + declared entities/predicates/intent types.
2. **BEFORE RESPONDING** about any person, project, or past event: call `mempalace_kg_search` or `mempalace_kg_query` FIRST. Never guess from memory — verify from the palace.
3. **IF UNSURE** about a fact (name, age, relationship, preference): say "let me check" and query. Wrong is worse than slow.
4. **AFTER EACH SESSION**: Call `mempalace_diary_write` to record what happened, what you learned, what matters.
5. **WHEN FACTS CHANGE**: Call `mempalace_kg_invalidate` on the old fact, then `mempalace_kg_add` for the new one.

## Available Tools

### Search & Browse
- `mempalace_kg_search` — Unified 3-channel search across drawers + entities. Always start here.
  - `queries` (required): MANDATORY list of 2-5 perspective strings. Multi-view retrieval fuses cosine + keyword + graph via RRF. A single string is REJECTED. Example: `["auth rate limiting", "brute force hardening", "login endpoint"]`.
  - `agent` (required): your agent entity name for affinity scoring.
  - `wing`, `room`: scope to drawers only.
  - `kind`: scope to entities only (`entity`, `predicate`, `class`, `literal`).
  - `limit`: max results (default 10, adaptive-K may trim).
- `mempalace_kg_query` — Exact entity-ID lookup. Use when `kg_search` already surfaced the entity and you want its full fact set.
  - `entity` (required): e.g. "Max", "MyProject" (supports comma-separated batch)
  - `as_of`: date filter (YYYY-MM-DD) — what was true at that time
  - `direction`: "outgoing", "incoming", or "both" (default "both")
- `mempalace_kg_stats` — Palace overview: counts by wing/room/kind + graph connectivity in one call.
- `mempalace_kg_timeline(entity?)` — Chronological story for an entity (or everything).
- `mempalace_kg_list_declared` — Entities declared in this session.
- `mempalace_traverse(start_room, max_hops?)` — Walk the graph from a room across wings.
- `mempalace_get_aaak_spec` — Get AAAK compression dialect specification.

### Knowledge Graph (write)
- `mempalace_kg_declare_entity` — Declare any entity. Drawers are entities of `kind="memory"` (P3.3).
  - For drawers: `kind="memory"`, `wing`, `room`, `slug`, `description` (verbatim content), `added_by` (required); `entity`, `predicate`, `hall`, `importance`, `source_file` (optional).
  - For other entities: `kind` ∈ {entity, class, predicate, literal}, `name`, `description`, `added_by` (required); `properties` for predicate constraints / intent type rules_profile.
  - Duplicate detection runs automatically; resolve any conflicts via `mempalace_resolve_conflicts`.
- `mempalace_kg_add` — Add a triple: subject → predicate → object.
- `mempalace_kg_add_batch(edges)` — Batch add (partial success OK).
- `mempalace_kg_update_entity(entity, ...)` — Unified update for both drawers and KG nodes (P3.4).
- `mempalace_kg_invalidate(subject, predicate, object)` — Soft-delete a single fact.
- `mempalace_kg_delete_entity(entity_id)` — Soft-delete an entire entity or drawer + invalidate every edge touching it (P3.6). Use for truly obsolete items; use `kg_invalidate` for a single stale fact.
- `mempalace_kg_merge_entities(source, target)` — Merge entities; source becomes alias.
- `mempalace_resolve_conflicts(actions=[...])` — Resolve duplicates/contradictions: `invalidate`, `merge`, `keep`, or `skip`.

### Intent System
- `mempalace_declare_intent(intent_type, slots, descriptions=[..., ...], agent, budget?)` — Declare what you intend to do; returns permissions + injected memories. `descriptions` is a MANDATORY list of 2+ perspectives (P3.21).
- `mempalace_active_intent` — Show current intent + remaining budget.
- `mempalace_extend_intent(budget)` — Add to budget without redeclaring.
- `mempalace_finalize_intent(slug, outcome, summary, agent, memory_feedback=[...])` — Capture what happened. `memory_feedback` is MANDATORY — rate every accessed memory 1-5.

### Agent Diary
- `mempalace_diary_write` — Write a session diary entry (concise prose, delta-only).
  - `agent_name` (required): your name/identifier
  - `entry` (required): what happened, what you learned, what matters
  - `topic`: category tag (default "general")
- `mempalace_diary_read` — Read recent diary entries
  - `agent_name` (required)
  - `last_n`: number of entries (default 10)

## Setup

Install MemPalace and populate the palace:

```bash
pip install mempalace
mempalace init ~/my-convos
mempalace mine ~/my-convos
```

### OpenClaw MCP config

Add to your OpenClaw MCP configuration:

```json
{
  "mcpServers": {
    "mempalace": {
      "command": "python3",
      "args": ["-m", "mempalace.mcp_server"]
    }
  }
}
```

Or via CLI:

```bash
openclaw mcp set mempalace '{"command":"python3","args":["-m","mempalace.mcp_server"]}'
```

### Other MCP hosts

```bash
# Claude Code
claude mcp add mempalace -- python -m mempalace.mcp_server

# Cursor — add to .cursor/mcp.json
# Codex — add to .codex/mcp.json
```

## Tips

- Search is semantic (meaning-based), not keyword. "What did we discuss about database performance?" works better than "database".
- The knowledge graph stores typed relationships with time windows. Use it for facts about people and projects — it knows WHEN things were true.
- Diary entries accumulate across sessions. Write one at the end of each conversation to build continuity.
- Duplicate detection is automatic on `kg_declare_entity` (kind=memory) — no separate check call. Resolve any returned conflicts via `mempalace_resolve_conflicts`.
- The AAAK dialect (from `mempalace_get_aaak_spec`) is a compressed notation for efficient storage. Read it naturally — expand codes mentally, treat *markers* as emotional context.

## License

[MemPalace](https://github.com/milla-jovovich/mempalace) is MIT licensed. Created by Milla Jovovich, Ben Sigman, Igor Lins e Silva, and contributors.
