# MemPalace Search

When the user wants to search their MemPalace memories, follow these steps:

## 1. Parse the Search Query

Extract the core search intent from the user's message. Identify any explicit
or implicit filters:
- Wing -- a top-level category (e.g., "work", "personal", "research")
- Room -- a sub-category within a wing
- Keywords / semantic query -- the actual search terms

## 2. Determine Wing/Room Filters

If the user mentions a specific domain, topic area, or context, map it to the
appropriate wing and/or room. If unsure, omit filters to search globally. You
can discover the taxonomy first if needed.

## 3. Use MCP Tools (Preferred)

If MCP tools are available, use them in this priority order:

- mempalace_kg_search(context, agent, wing?, room?, kind?) -- Primary search tool (P4.5).
  Unified 3-channel pipeline (cosine + keyword + graph, RRF-merged) across both
  memories and entities. `context` MUST be a dict with:
    - `queries`:  list[str] (2-5 perspectives) -- drive the multi-view cosine channel.
    - `keywords`: list[str] (2-5 caller-provided exact terms) -- drive the keyword channel.
                  No auto-extraction; the caller says what matters.
    - `entities`: list[str] (0+ optional) -- seed the graph BFS explicitly.
  A single-string `queries` is rejected. Use `wing`/`room` to scope to memories
  only, or `kind` to scope to entities only.
- mempalace_kg_query(entity) -- Exact entity-ID lookup when you know the name.
  Returns all edges for that entity. Use this if `kg_search` already surfaced
  the entity and you want its full fact set.
- mempalace_kg_stats -- Palace overview: counts by wing/room/kind, cross-wing
  connectivity. Use when the user wants a structural overview.
- mempalace_traverse(start_room, max_hops?) -- Walk the knowledge graph from a
  room. Use when the user wants to explore connections and related memories.

## 4. CLI Fallback

If MCP tools are not available, fall back to the CLI:

    mempalace search "query" [--wing X] [--room Y]

## 5. Present Results

When presenting search results:
- Always include source attribution: wing, room, and memory for each result
- Show relevance or similarity scores if available
- Group results by wing/room when returning multiple hits
- Quote or summarize the memory content clearly

## 6. Offer Next Steps

After presenting results, offer the user options to go deeper:
- Drill deeper -- search within a specific room or narrow the query
- Traverse -- explore the knowledge graph from a related room
- Check tunnels -- look for cross-wing connections if the topic spans domains
- Browse taxonomy -- show the full structure for manual exploration
