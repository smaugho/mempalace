# MemPalace Status

Display the current state of the user's memory palace.

## Step 1: Gather Palace Status

Check if MCP tools are available (look for mempalace_kg_stats in available tools).

- If MCP is available: Call mempalace_kg_stats to retrieve palace state.
  This single call returns counts by wing/room/kind PLUS graph connectivity
  (connected components, average degree).
- If MCP is not available: Run the CLI command: mempalace status

## Step 2: Display Counts

Present the palace structure counts clearly:
- Number of wings, rooms, memories, entities
- Total triples in the knowledge graph
- Cross-wing connectivity stats (from the `graph` section of kg_stats)

Keep the output concise -- use a brief summary format, not verbose tables.

## Step 4: Suggest Next Actions

Based on the current state, suggest one relevant action:

- Empty palace (zero memories): Suggest "Try /mempalace:mine to add data from
  files, URLs, or text."
- Has data but no knowledge graph (memories exist but KG stats show zero
  triples): Suggest "Consider adding knowledge graph triples for richer
  queries."
- Healthy palace (has memories and KG data): Suggest "Use /mempalace:search to
  query your memories."

## Output Style

- Be concise and informative -- aim for a quick glance, not a report.
- Use short labels and numbers, not prose paragraphs.
- If any step fails or a tool is unavailable, note it briefly and continue
  with what is available.
