# MCP Integration — Claude Code

## Setup

Run the MCP server:

```bash
python -m mempalace.mcp_server
```

Or add it to Claude Code:

```bash
claude mcp add mempalace -- python -m mempalace.mcp_server
```

## Available Tools

The server exposes the full MemPalace MCP toolset. Common entry points:

- **mempalace_wake_up** — session boot: protocol + identity + L1 ranked context + declared entities/predicates/intent types
- **mempalace_kg_search** — unified 3-channel search (cosine + keyword + graph, RRF merged) across memories (prose) AND entities (KG nodes); `queries` is a mandatory list of 2+ perspectives
- **mempalace_kg_query** — exact entity-ID lookup, returns all current edges
- **mempalace_kg_stats** — palace overview: counts by wing/room/kind + graph connectivity

## Usage in Claude Code

Once configured, Claude Code can search your memories directly during conversations.
