"""
injection_gate.py -- Post-retrieval relevance gate.

The gate sits between the retriever and the main agent's working
context. It decides per-item keep/drop: the RRF-ranked list coming
out of retrieval is unchanged; only what the GENERATOR sees is
filtered. Dropped items never enter accessed_memory_ids -- the gate
writes their rated_irrelevant feedback itself via KnowledgeGraph
.record_feedback with rater_kind='gate_llm'.

Research grounding
------------------
* Empirical distraction effect: Zhou et al. ACL 2025
  (arXiv:2505.06914) -- irrelevant retrievals measurably degrade
  generation; stronger retrievers produce MORE distracting
  irrelevants because they look semantically plausible.
* Self-RAG (Asai et al. ICLR 2024) -- ISREL reflection tokens gate
  each retrieved passage post-retrieval. This gate is the
  palace-shaped equivalent.
* Adaptive-RAG (Jeong et al. 2024) -- classifier decides whether to
  retrieve; here we always retrieve but filter injection.

Design principles
-----------------
* Bias-to-keep: if an item relates to the primary context in any
  way, KEEP. Drop only when clearly unrelated (different project /
  domain / thread). Stated twice in the prompt because LLM judges
  default to being discriminating.
* Structured output: forced tool-use via ``tool_choice`` eliminates
  JSON-parse failures.
* Fail-open: any API or parse failure returns ``gate_status.state =
  'degraded'`` and passes all items through unfiltered. The main
  agent is instructed (via the session-start protocol) to surface
  degraded gates to the user.
* Project disambiguation via cwd anchor: cwd is included in the
  session frame ONLY when the directory contains a project anchor
  file (pyproject.toml / package.json / .git / …). Otherwise we
  omit it to avoid leaking misleading project tags.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# v3.7.2 Slice 2 (Adrian directive 2026-05-16, Option 3 architecture):
# single-worker daemon thread pool for the background quality pass.
# After the lean foreground gate (Slice 1, v3.7.1) returns its
# drops-only decision, apply_gate submits a closure to this executor
# that runs a SECOND Haiku call against the full padded
# _SYSTEM_PROMPT + GATE_DECISIONS_TOOL to re-emit quality flags
# asynchronously. Single worker bounds Anthropic rate-limit pressure
# and lets the bg call use the padded prompt's cache benefit on the
# 2nd+ submission within the 5-minute TTL. Daemon threads die with
# the process so a hung Anthropic call cannot block interpreter exit.
# Disable via MEMPALACE_BG_QUALITY=0 (e.g. when debugging or when
# the API budget needs throttling).
_BG_QUALITY_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="mempalace-bg-quality",
)

log = logging.getLogger(__name__)

# Strips lone UTF-16 surrogate codepoints (U+D800-U+DFFF) from retrieved
# text before it reaches the judge's prompt. Anthropic's HTTP client
# rejects these at JSON-serialize time with UnicodeEncodeError ("surrogates
# not allowed"), which fails the judge call, degrades the gate, and dumps
# K=20 unfiltered items into the agent context -- turning a relevance gate
# into a pass-through. Old records written before the sanitizer existed
# still carry these codepoints, so scrubbing at the gate inlet is load-
# bearing even after the write-side sanitize_content fix lands.
# Stripping (vs. replacing with '?') avoids injecting a spurious char into
# otherwise-clean prose.
_UTF16_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _scrub_surrogates(value: str) -> str:
    if not isinstance(value, str):
        return value
    return _UTF16_SURROGATE_RE.sub("", value)


# Directory anchor files used for project-root detection. cwd is
# injected into the session frame only when at least one of these is
# present in the process's current working directory. Keep this list
# in sync with the cwd-anchor decision recorded in the wrap-up memory
# wrap_up_injection_gate_design_decisions_2026_04_23.
_PROJECT_ANCHORS = (
    "pyproject.toml",
    "package.json",
    ".git",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "Gemfile",
    "CMakeLists.txt",
    "composer.json",
    "mix.exs",
    "requirements.txt",
    "setup.py",
)

# Hard cap on per-item content rendered into the prompt. Haiku 4.5
# has 200K context; K × 5K memories fits comfortably. The cap is
# defensive -- a single runaway memory shouldn't blow the budget.
_MAX_ITEM_CHARS = 6000

# Default model. The runtime can override via MempalaceConfig or the
# ``MEMPALACE_GATE_MODEL`` env var.
_DEFAULT_MODEL = "claude-haiku-4-5"


# ═══════════════════════════════════════════════════════════════════
# Data shapes
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class GateItem:
    """One retrieval candidate the gate must judge.

    Shape mirrors what the three retrieval sites (declare_intent,
    declare_operation, kg_search) naturally have on hand: an id, the
    source namespace (memory/entity/triple), the document/statement
    text, the channel that produced it (A/B/C/D), rank, and score.
    ``extra`` carries per-source detail the prompt renderer uses
    (e.g. triple subject/predicate/object, entity kind + description,
    memory summary).
    """

    id: str
    source: Literal["memory", "entity", "triple"]
    text: str
    channel: str  # "A" (cosine), "B" (graph), "C" (keyword), "D" (context)
    rank: int
    score: float
    extra: dict = field(default_factory=dict)


@dataclass
class GateDecision:
    id: str
    action: Literal["keep", "drop"]
    reasoning: str
    proposed_summary: str | None = None


@dataclass
class GateResult:
    kept: list[GateItem]
    dropped: list[tuple[GateItem, GateDecision]]
    gate_status: dict
    judge_tokens_in: int = 0
    judge_tokens_out: int = 0
    # Prompt-caching telemetry from Anthropic's usage block. cache_read
    # is the tokens served from cache (billed at ~10% of normal input);
    # cache_creation is the tokens written to cache on a miss (billed at
    # 125% for 5-min TTL). A healthy gate after warm-up shows high
    # cache_read and near-zero cache_creation across consecutive calls.
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    # Quality-issue flags the judge emitted across the K items
    # together. Each is a dict: {kind, memory_ids, detail}.
    # Persisted by apply_gate to the memory_flags table for the
    # memory_gardener to investigate later.
    flags: list[dict] = field(default_factory=list)
    # Per-call wall-clock breakdown in milliseconds. Populated by
    # filter() so callers can see exactly where the gate spent its
    # time (prompt build, LLM round-trip, decision parse). Mirrors
    # the judge_tokens_in/out pair for cost observability -- tokens
    # tell you how much you paid, timings tell you how long the user
    # waited. Shape: {"prompt_ms": float, "llm_ms": float,
    # "parse_ms": float, "total_ms": float, "attempts": int,
    # "n_items": int}.
    timings: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# Prompt builder
# ═══════════════════════════════════════════════════════════════════


# v3.7.0 Slice 1 (Adrian directive 2026-05-16, Option 3 architecture):
# Lean foreground gate prompt. Drops the JOB 2 (quality-flag) taxonomy
# and 12 worked examples that padded the full prompt above the Haiku
# 4.5 cache minimum -- foreground gate output tokens dominate streaming
# time (~80 tok/sec), and dropping ~95% of output (drops-only vs
# decisions+flags) cuts wall time ~6-9s -> ~2-3s. Cache stops hitting
# because the lean prefix is below the ~4096 token cache floor, but
# cold-call latency wins decisively when the output is the bottleneck.
# The full _SYSTEM_PROMPT below is preserved for the background quality
# pass (Slice 2 will wire it onto an async thread that consumes the
# padding-benefit cache hit without blocking the agent's foreground).
_LEAN_SYSTEM_PROMPT = (
    "You are the relevance gate for a memory palace. One job: for "
    "each retrieved item, decide INJECT (keep) or SUPPRESS (drop).\n\n"
    "BIAS TO KEEP. If an item relates to the primary context in any "
    "way -- shares a topic, touches a mentioned entity, records a "
    "prior thread on the same question, informs a tangential decision "
    "-- mark it KEEP. Mark DROP only if the item is clearly from a "
    "different project / domain / thread and would add noise without "
    "signal. A low-importance but on-topic item is KEEP. Project "
    "mismatch is a strong drop signal. Importance alone is never a "
    "keep signal.\n\n"
    "Channel provenance is informative. Channel D (context-walk) "
    "items are already upvoted by past behaviour on this or similar "
    "contexts -- lean toward keep there even if content looks "
    "tangential. Channel A (cosine) items matched the primary "
    "context's text; Channel B (graph) items are neighbours of seed "
    "entities; Channel C (keyword) hits are exact-term matches.\n\n"
    "Before emitting each decision, write one sentence explaining "
    "the item's relation (or non-relation) to the PRIMARY CONTEXT. "
    "If you keep an item whose current summary is generic while its "
    "content is specific, propose a better summary (<=280 chars, "
    "faithful to the content, written from a different angle than "
    "the content itself).\n\n"
    "Repeat: BIAS TO KEEP. DROP only when the item is clearly "
    "unrelated.\n\n"
    "Quality flags (duplicate_pair, stale, unlinked_entity, orphan, "
    "generic_summary, edge_candidate, contradiction_pair, etc.) are "
    "NOT your job in this call. A separate background quality pass "
    "handles them asynchronously and writes results back without "
    "blocking the agent. Focus only on the keep/drop decision here."
)


_SYSTEM_PROMPT = (
    "You are the relevance gate AND quality inspector for a memory "
    "palace. Two jobs per call, both important.\n\n"
    "JOB 1 -- KEEP / DROP. For each retrieved item, decide INJECT "
    "(keep) or SUPPRESS (drop). Be GENEROUS toward keep: if an item "
    "relates to the primary context in any way -- shares a topic, "
    "touches a mentioned entity, records a prior thread on the same "
    "question, informs a tangential decision -- mark it KEEP. Mark "
    "DROP only if the item is clearly from a different project / "
    "domain / thread and would add noise without signal. A low-"
    "importance but on-topic item is KEEP. Project mismatch is a "
    "strong drop signal. Importance alone is never a keep signal.\n\n"
    "Channel provenance is informative. Channel D (context-walk) "
    "items are already upvoted by past behaviour on this or similar "
    "contexts -- lean toward keep there even if content looks "
    "tangential. Channel A (cosine) items matched the primary "
    "context's text; Channel B (graph) items are neighbours of "
    "seed entities; Channel C (keyword) hits are exact-term matches.\n\n"
    "Before emitting each decision, write one sentence explaining "
    "the item's relation (or non-relation) to the PRIMARY CONTEXT. "
    "If you keep an item whose current summary is generic while "
    "its content is specific, propose a better summary (≤280 chars, "
    "faithful to the content, written from a different angle than "
    "the content itself).\n\n"
    "Repeat the rule to yourself: BIAS TO KEEP. DROP only when the "
    "item is clearly unrelated.\n\n"
    "JOB 2 -- FLAG QUALITY ISSUES. This is VERY IMPORTANT and you "
    "must attend to it every call. You are looking at K memories "
    "together -- a rare joint vantage point a single-memory process "
    "never has. Use it. Emit flags when you see:\n"
    "  • duplicate_pair -- two items state the same fact.\n"
    "  • contradiction_pair -- two items contradict each other on a "
    "specific claim (dates, identities, outcomes, …).\n"
    "  • stale -- a CURRENT FACT in an item is now wrong (a decision "
    "was reversed, an entity was renamed, a value changed). Important: "
    "records that document a PAST event truthfully are NOT stale, even "
    "if the situation has since changed. A diary entry saying 'X was "
    "broken yesterday and we fixed it today' is valid history, not a "
    "stale memory. Only flag stale when the item itself makes a "
    "forward-looking claim that is now false.\n"
    "  • unlinked_entity -- an item clearly mentions a person, "
    "project, location, file, or system that does NOT appear in "
    "the primary context's entities list and is probably missing a "
    "KG link.\n"
    "  • orphan -- an item lists no entities in its meta and seems "
    "to describe something concrete -- probably lost its entity "
    "links and needs re-anchoring.\n"
    "  • generic_summary -- the structured summary fails the WHAT/"
    "WHY contract for retrieval. Flag when ANY of these hold:\n"
    "    (a) the WHAT is not a discriminative noun phrase -- bare "
    "type names ('project', 'tool', 'concept'), single tokens, or "
    "keyword-soup concatenations like 'summary contract what why "
    "scope dict' (six space-joined keywords with no clause). The "
    "WHAT must distinguish this entity from others of the same "
    "kind. GOOD what: 'InjectionGate (post-retrieval relevance "
    "filter)', 'data_migrations stamp table pattern'. BAD what: "
    "'the project', 'summary contract', 'tool'.\n"
    "    (b) the WHY is not a real purpose / role / claim clause -- "
    "it just restates the WHAT, lists keywords, or is a placeholder. "
    "Test: replace WHAT with 'X' -- does WHY still make sense as an "
    "explanation? GOOD why: 'filters retrieved memories before "
    "injection via Haiku tool-use, emits quality flags', 'marks "
    "one-shot data migrations as applied so subsequent inits short-"
    "circuit O(1)'. BAD why: 'what why scope dict', 'the summary "
    "contract', 'is a project'.\n"
    "    (c) the description starts with '[AUTO' or contains 'needs "
    "refinement' -- self-identifying placeholder from an auto-mint "
    "path that the gardener never reached.\n"
    "    (d) the description is a bare 'File: <path>' stub with no "
    "WHAT or WHY at all.\n"
    "  Prefer PROPOSING a rewrite (better WHAT/WHY/SCOPE) over "
    "flagging when you can; the gardener accepts proposals via the "
    "memory_flags row's detail field. The structured render in this "
    "prompt (WHAT: / WHY: / SCOPE: lines) makes per-field judgment "
    "tractable -- evaluate each line against its rule above.\n"
    "  • edge_candidate -- the content strongly implies a factual "
    "relationship between two named entities that the KG probably "
    "doesn't have (e.g. 'A replaces B', 'A depends on B', 'A was "
    "built by B'). Include the two entity ids in memory_ids and the "
    "suggested predicate in detail. Do NOT author edges here -- the "
    "link-author jury owns that decision; you only flag the "
    "candidate for it.\n\n"
    "Flags are OPTIONAL -- if nothing stands out, return flags: []. "
    "But do not skip this job. A perfect call catches every issue "
    "a human operator would have caught reading the K items side-"
    "by-side. Under-flagging is a failure mode; over-flagging is "
    "recoverable (the memory_gardener investigates each and can "
    "defer). When in doubt, flag.\n\n"
    "## Worked examples\n\n"
    "These examples ground the abstract rules above and ALSO lift the "
    "static system-prompt prefix above the model's prompt-cache minimum "
    "(claude-haiku-4-5 silently declines to cache prefixes shorter than "
    "~4096 tokens; once over the floor, repeat calls within the 5-minute "
    "ephemeral TTL pay only ~10% of the input cost). Every example "
    "below is INVARIANT across calls -- do not paraphrase the rules "
    "into per-call user content; keep them here so the cache key stays "
    "stable.\n\n"
    "Example 1 -- KEEP an on-topic memory; flag generic_summary.\n\n"
    "Primary context: queries=['migrate scoring.py off ChromaDB', "
    "'replace col.query with vs.query'], keywords=['scoring.py', "
    "'vs.query'], entities=['scoring_py', 'mempalace'].\n"
    "Item: id='record_xyz', source='memory', channel='cosine', "
    "score=0.61.\n"
    "  WHAT: 'the project'\n"
    "  WHY: 'is a project'\n"
    "  CONTENT: 'Migrated scoring.py multi_channel_search to call "
    "vs.query directly; col.query helper retired with the Tier 2 "
    "VectorStore landing.'\n\n"
    "Correct decision: KEEP (content is exactly on-topic for the "
    "primary context's migration query). Reason: 'Records the same "
    "scoring.py / col.query -> vs.query migration the gate's primary "
    "context describes.'\n"
    "Correct flag: generic_summary -- WHAT 'the project' is a bare "
    "type name; WHY 'is a project' restates WHAT with no clause. "
    "Propose better summary: WHAT 'scoring.py multi_channel_search "
    "vs.query migration', WHY 'records the Tier 2 VectorStore landing "
    "where col.query helper was retired in scoring.py'.\n\n"
    "Example 2 -- DROP a project-mismatched memory; no flag needed.\n\n"
    "Primary context: queries=['mempalace declare_intent latency'], "
    "keywords=['mempalace', 'declare_intent'], "
    "entities=['mempalace'].\n"
    "Item: id='record_paperclip_setup', source='memory', "
    "channel='keyword', score=0.42.\n"
    "  WHAT: 'paperclip backend port-3100 setup'\n"
    "  WHY: 'records local-dev port assignment for the DSpot paperclip "
    "backend on port 3100'\n"
    "  CONTENT: 'paperclip backend listens on port 3100 in dev; "
    "configured via ~/.paperclip/instances/default/.env'\n\n"
    "Correct decision: DROP (paperclip is a different project entirely; "
    "the keyword hit is incidental on 'paperclip' substring or similar). "
    "Reason: 'Paperclip / DSpot project memory; primary context is "
    "mempalace internals -- different project, no signal here.'\n"
    "No flags: summary is well-formed; mismatched-project items don't "
    "need any flag, just a drop.\n\n"
    "Example 3 -- KEEP via Channel D upvote even though content "
    "looks tangential.\n\n"
    "Primary context: queries=['v3.5.4 cache padding fix'], "
    "keywords=['cache_control', 'haiku-4-5', 'padding'], "
    "entities=['injection_gate_py'].\n"
    "Item: id='record_state_judge_v0_2026_05_07', source='memory', "
    "channel='context_walk', score=0.55.\n"
    "  WHAT: 'state-judge v0 architectural pivot'\n"
    "  WHY: 'Adrian preferred out-of-loop Haiku judge over agent-rated "
    "unchanged-ack defaults; established judge as separate Haiku call "
    "alongside the gate'\n"
    "  CONTENT: '(commit 6b539fd) Pivoted state_deltas to a Haiku judge "
    "running parallel to apply_gate; both share the cache prefix.'\n\n"
    "Correct decision: KEEP. Reason: 'Channel D walk-upvote means past "
    "behaviour on similar contexts found this useful. Content also "
    "names the gate+judge parallel pair that the v3.5.4 cache fix "
    "directly affects (both calls share the broken cache).'\n"
    "No flags.\n\n"
    "Example 4 -- flag duplicate_pair.\n\n"
    "Primary context: queries=['edit_mempalace v3.5.0 ship'], "
    "keywords=['v3.5.0', 'feedback_auto'], entities=['mempalace'].\n"
    "Item A: id='record_v350_shipped_109be13', WHAT='v3.5.0 atomic "
    "feedback rip-out shipped 109be13'.\n"
    "Item B: id='diary_v350_feedback_rip_out_2026_05_14', WHAT='v3.5.0 "
    "feedback rip-out diary entry'.\n"
    "CONTENT-A: 'Removed memory_feedback + operation_ratings + "
    "extend_feedback. Async Haiku rater wired. 1413 pytest green. "
    "Commit 109be13 on origin/main.'\n"
    "CONTENT-B: 'v3.5.0 atomic feedback rip-out shipped to origin/main "
    "as commit 109be13. Removed memory_feedback + operation_ratings + "
    "extend_feedback. Async Haiku rater wired. 1413 pytest.'\n\n"
    "Correct decision: KEEP both (record + diary are complementary "
    "memory kinds; both load-bearing).\n"
    "Correct flag: duplicate_pair -- memory_ids=[record_xyz, "
    "diary_xyz] -- both record the same ship event with overlapping "
    "narrative; gardener may want to thin or canonicalise.\n\n"
    "Example 5 -- flag stale.\n\n"
    "Primary context: queries=['kg_delete_entity Chroma vector "
    "lookup'], keywords=['kg_delete_entity'].\n"
    "Item: id='record_kg_delete_chroma_only_2026_04_25', WHAT='kg_"
    "delete_entity is Chroma-only', WHY='deletion path uses col.get "
    "lookup; SQL-only entities get false-negative Not found'.\n"
    "CONTENT: 'kg_delete_entity uses Chroma col.get(ids=[entity]) "
    "exclusively -- entities living only in the SQL entities table "
    "return Not found.'\n\n"
    "Correct decision: KEEP (still relevant for retrieval).\n"
    "Correct flag: stale -- detail='kg_delete_entity gained SQL "
    "fallback in v3.5.1 (commit 68465e2); claim that it is "
    "Chroma-only is now false. Update content to reference the "
    "v3.5.1 fix or invalidate.' This catches the case where the "
    "claim was true historically and is now wrong.\n\n"
    "Example 6 -- flag edge_candidate.\n\n"
    "Primary context: queries=['feedback_auto module relations'], "
    "keywords=['feedback_auto'].\n"
    "Item: id='record_v350_arch_2026_05_14', WHAT='v3.5.0 feedback "
    "rip-out architecture'.\n"
    "CONTENT: 'mempalace_extend_feedback was deleted in v3.5.0; "
    "feedback_auto.submit_finalize_feedback now replaces it as the "
    "post-finalize Haiku-rater entrypoint, called from "
    "tool_finalize_intent at the end of every intent close.'\n\n"
    "Correct decision: KEEP.\n"
    "Correct flag: edge_candidate -- memory_ids=['feedback_auto_py', "
    "'mempalace_extend_feedback'] -- predicate='replaced_by' -- "
    "detail='content states feedback_auto.submit_finalize_feedback "
    "replaces the deleted mempalace_extend_feedback tool; suggested "
    "edge: feedback_auto_py replaced_by mempalace_extend_feedback '"
    "(or its inverse).' Do NOT author the edge here -- the link-author "
    "jury owns that decision.\n\n"
    "Example 7 -- DROP an off-topic but high-importance memory.\n\n"
    "Primary context: queries=['mempalace gate latency Haiku cache'], "
    "keywords=['Haiku', 'cache'], entities=['injection_gate_py'].\n"
    "Item: id='record_adrian_homeoffice_setup', importance=5, WHAT="
    "'Adrian home office hardware', WHY='dual 4K monitors + RTX 4090 "
    "for local dev'.\n"
    "CONTENT: 'Adrian's home dev box: AMD 7950X, RTX 4090, dual "
    "27\" 4K monitors, Windows 11.'\n\n"
    "Correct decision: DROP. Reason: 'High importance is for the "
    "agent-Adrian working relationship; for a Haiku-cache latency "
    "investigation it is pure noise.'\n"
    "Importance is NEVER a keep signal on its own -- only relevance "
    "to the primary context decides. No flag needed.\n\n"
    "Example 8 -- flag unlinked_entity.\n\n"
    "Primary context: queries=['mempalace internals'], "
    "keywords=['mempalace'], entities=['mempalace'].\n"
    "Item: id='record_xyz', WHAT='Adrian shipped fastembed swap', "
    "WHY='replaced chromadb embedder with fastembed for ~50MB dep '"
    "surface vs ~2GB PyTorch tower'.\n"
    "CONTENT: 'fastembed (Qdrant) replaced chromadb embedder; same "
    "all-MiniLM-L6-v2 model so existing palace vectors stay cosine-"
    "compatible.'\n\n"
    "Correct decision: KEEP.\n"
    "Correct flag: unlinked_entity -- memory_ids=['fastembed', "
    "'qdrant'] -- detail='content names fastembed and Qdrant as "
    "concrete dependencies/vendors; primary context entities list "
    "does not include them; probably missing kg_declare_entity for "
    "fastembed (kind=tool, dependency relationship to mempalace).'\n\n"
    "Calibration reminders (re-read every call):\n"
    "  - BIAS TO KEEP. The agent suffers from missing context far "
    "more than from extra context. Drop only when project mismatch "
    "is unambiguous OR the item is pure noise.\n"
    "  - Importance is never a keep signal alone.\n"
    "  - Channel D (context_walk) items already have positive "
    "feedback history -- lean keep.\n"
    "  - Quality flags are PER-MEMORY-PAIR observations -- you have "
    "the rare K-item joint vantage point; use it.\n"
    "  - PROPOSE rewrites in detail rather than just flagging when "
    "you can articulate the correction.\n"
    "  - The link-author jury -- not you -- owns edge authoring; you "
    "only flag candidates.\n"
    "  - History records of past events that were true at the time "
    "are NEVER stale, even if the world has since changed.\n"
    "  - Generic-summary catches: bare type names ('the project'), "
    "keyword soup ('summary contract what why scope dict'), AUTO "
    "stubs, bare 'File: <path>' descriptions.\n\n"
    "Example 9 -- KEEP a triple (KG fact) on-topic.\n\n"
    "Primary context: queries=['Adrian preferences for shipping "
    "cadence'], keywords=['Adrian', 'shipping'], entities=['Adrian'].\n"
    "Item: id='t_adrian_prefers_atomic_ships_98ab21', source='triple', "
    "channel='graph', score=0.71.\n"
    "  subject: 'Adrian'   predicate: 'prefers'   object: "
    "'atomic_ships_over_partial'   confidence: 0.95\n"
    "  STATEMENT-WHAT: 'Adrian prefers atomic ships over partial work'\n"
    "  STATEMENT-WHY: 'consistently directs the agent to ship complete "
    "features in one commit rather than landing a half-feature; "
    "established 2026-04 across multiple sessions'\n\n"
    "Correct decision: KEEP. Reason: 'Triple directly answers the "
    "primary context query about Adrian's shipping cadence "
    "preferences; subject + predicate + object are all on-topic.'\n"
    "No flags -- statement WHAT and WHY are well-formed and "
    "discriminative.\n\n"
    "Example 10 -- KEEP an entity (KG node).\n\n"
    "Primary context: queries=['injection gate architecture'], "
    "keywords=['injection_gate'], entities=['injection_gate_py'].\n"
    "Item: id='injection_gate_py', source='entity', channel='cosine', "
    "score=0.83.\n"
    "  name: 'injection_gate_py'   kind: 'file'\n"
    "  WHAT: 'mempalace/injection_gate.py module'\n"
    "  WHY: 'post-retrieval Haiku-tool-use gate that filters surfaced "
    "memories before injection AND emits quality flags for the "
    "memory_gardener; co-located with run_state_judge'\n\n"
    "Correct decision: KEEP. Reason: 'Entity IS the injection_gate "
    "the primary context names; canonical authority for the topic.'\n"
    "No flags.\n\n"
    "Example 11 -- DROP a context-walk hit that is genuinely off-topic "
    "despite high Channel D score.\n\n"
    "Primary context: queries=['mempalace declare_intent slow'], "
    "keywords=['declare_intent', 'latency'], entities=['mempalace'].\n"
    "Item: id='ctx_4071_grocery_list', source='memory', "
    "channel='context_walk', score=0.65.\n"
    "  WHAT: 'Adrian groceries 2026-03'\n"
    "  WHY: 'list of items Adrian buys weekly; oat milk, bread, eggs, "
    "coffee, frozen pizza'\n"
    "  CONTENT: 'oat milk, bread, eggs, coffee, frozen pizza, "
    "dish soap, paper towels.'\n\n"
    "Correct decision: DROP. Reason: 'Channel D walk-upvote is from a "
    "completely unrelated past surfacing (probably a context-collision "
    "with an Adrian-related token); content is grocery list with zero "
    "relation to declare_intent latency.' Channel D is informative, "
    "not authoritative -- if content is clearly unrelated, drop "
    "regardless of channel.\n"
    "Optional flag: noise_in_channel_walk if this happens repeatedly "
    "on the same item -- gardener can downweight or invalidate the "
    "stale walk edge.\n\n"
    "Example 12 -- multiple flags on one item (orphan + generic_summary).\n\n"
    "Primary context: queries=['mempalace internals'], "
    "keywords=['mempalace'], entities=['mempalace'].\n"
    "Item: id='record_floating_2026_02', meta.entities=[].\n"
    "  WHAT: 'a thing happened'\n"
    "  WHY: 'something occurred recently'\n"
    "  CONTENT: 'Migrated the embedder. Replaced 2 GB of PyTorch with "
    "50 MB ONNX runtime. Cosine sim still 1.0.'\n\n"
    "Correct decision: KEEP (content describes a real concrete "
    "migration event worth retaining).\n"
    "Correct flags (TWO): orphan -- meta.entities=[] yet content "
    "names concrete things; needs re-anchoring to fastembed + "
    "chromadb. AND generic_summary -- WHAT 'a thing happened' is a "
    "placeholder, WHY 'something occurred' is meaningless. Propose "
    "rewrite: WHAT 'fastembed embedder swap (chromadb retired)', "
    "WHY 'replaced ChromaDB's bundled embedder with fastembed ONNX "
    "runtime; ~50MB dep vs ~2GB PyTorch tower; cos_sim=1.0 with "
    "prior model verified empirically'."
)


def _detect_project_anchor(cwd: str | None) -> str | None:
    """Return project name (cwd basename) only when an anchor file
    exists in cwd. Otherwise return None.

    This is the only check we perform before injecting cwd-based
    project info into the gate prompt. If the anchor check fails we
    omit the project tag silently rather than inject a potentially
    wrong one.
    """
    if not cwd:
        return None
    try:
        p = Path(cwd)
        if not p.is_dir():
            return None
        for anchor in _PROJECT_ANCHORS:
            if (p / anchor).exists():
                return p.name
    except OSError:
        return None
    return None


def _render_item(item: GateItem) -> str:
    """Render one retrieved item for the judge prompt.

    Adrian's design lock 2026-05-02: when a structured ``{what, why,
    scope?}`` summary dict is available (in ``item.extra['summary_dict']``
    for memory and ``item.extra['properties_summary_dict']`` for
    entity/triple), render it as labeled WHAT/WHY/SCOPE lines via
    ``scoring.render_structured_summary`` so the gate can evaluate each
    component separately. Falls back to the cached single-line prose
    ``item.extra['summary']`` for legacy data without a structured
    dict. The structured form costs ~6-12 extra tokens per item and
    decomposes the gate's decision from "is this prose good?" to per-
    field judgments ("is the WHAT discriminative? is the WHY purpose-
    clear? is the SCOPE meaningful?").
    """
    from .scoring import render_structured_summary

    text = (item.text or "").strip()
    if len(text) > _MAX_ITEM_CHARS:
        text = text[:_MAX_ITEM_CHARS] + " …[truncated]"

    lines = [
        f"  [{item.rank}] id={item.id}",
        f"      source: {item.source}   channel: {item.channel}   "
        f"rank: {item.rank}   score: {item.score:.3f}",
    ]

    if item.source == "triple":
        subj = item.extra.get("subject", "?")
        pred = item.extra.get("predicate", "?")
        obj = item.extra.get("object", "?")
        conf = item.extra.get("confidence", 1.0)
        # Triples carry the rendered statement prose in `text`. When the
        # writer also persisted the structured statement dict, render it
        # labeled; otherwise the single-line prose is the only signal.
        statement_dict = item.extra.get("statement_dict")
        if isinstance(statement_dict, dict) and statement_dict:
            labeled = render_structured_summary(statement_dict, fallback_prose=text)
            lines.append(f"      statement:\n{_indent_block(labeled)}")
        else:
            lines.append(f"      statement: {text}")
        lines.append(
            f"      subject: {subj}   predicate: {pred}   object: {obj}   confidence: {conf}"
        )
    elif item.source == "entity":
        name = item.extra.get("name", item.id)
        kind = item.extra.get("kind", "entity")
        lines.append(f"      name: {name}   kind: {kind}")
        # Entity description is the rendered prose. When the structured
        # properties.summary dict is available, render it labeled.
        summary_dict = item.extra.get("properties_summary_dict") or item.extra.get("summary_dict")
        if isinstance(summary_dict, dict) and summary_dict:
            labeled = render_structured_summary(summary_dict, fallback_prose=text)
            lines.append(f"      description:\n{_indent_block(labeled)}")
        elif text:
            lines.append(f"      description: {text}")
    else:  # memory
        summary_dict = item.extra.get("summary_dict")
        summary_prose = item.extra.get("summary") or ""
        if isinstance(summary_dict, dict) and summary_dict:
            labeled = render_structured_summary(summary_dict, fallback_prose=summary_prose)
            lines.append(f"      summary:\n{_indent_block(labeled)}")
        elif summary_prose:
            lines.append(f"      summary: {summary_prose}")
        if text:
            lines.append(f"      content: {text}")
    return "\n".join(lines)


def _indent_block(text: str, indent: str = "        ") -> str:
    """Indent each line of a multi-line string for readability in the
    prompt. Used to align labeled WHAT/WHY/SCOPE lines under their
    parent summary/description/statement label."""
    if not text:
        return text
    return "\n".join(indent + line for line in text.split("\n"))


def build_prompt(
    *,
    primary_context: dict,
    items: list[GateItem],
    parent_intent: dict | None = None,
    session_frame: dict | None = None,
) -> str:
    """Compose the user-message body for the judge.

    primary_context: {queries: [...], keywords: [...], entities: [...]}
    parent_intent (optional): {intent_type, subject, queries[0]}
    session_frame (optional): {agent, project, recent_intents: [...]}
    """
    parts = []

    if session_frame:
        frame_lines = ["SESSION FRAME"]
        if session_frame.get("agent"):
            frame_lines.append(f"  agent: {session_frame['agent']}")
        if session_frame.get("project"):
            frame_lines.append(f"  project (cwd-inferred): {session_frame['project']}")
        if len(frame_lines) > 1:
            parts.append("\n".join(frame_lines))

    if parent_intent:
        p = parent_intent
        parent_lines = ["PARENT FRAME (enclosing intent)"]
        if p.get("intent_type"):
            parent_lines.append(f"  intent_type: {p['intent_type']}")
        if p.get("subject"):
            parent_lines.append(f"  subject: {p['subject']}")
        if p.get("query"):
            parent_lines.append(f"  query: {p['query']}")
        parts.append("\n".join(parent_lines))

    pc_lines = ["PRIMARY CONTEXT (this retrieval's context -- judge against THIS)"]
    if primary_context.get("source"):
        pc_lines.append(f"  source: {primary_context['source']}")
    for q in (primary_context.get("queries") or [])[:5]:
        pc_lines.append(f"  query: {q}")
    if primary_context.get("keywords"):
        pc_lines.append("  keywords: [" + ", ".join(primary_context["keywords"][:8]) + "]")
    if primary_context.get("entities"):
        pc_lines.append("  entities: [" + ", ".join(primary_context["entities"][:10]) + "]")
    parts.append("\n".join(pc_lines))

    retrieved_header = f"RETRIEVED ITEMS (K={len(items)})"
    parts.append(retrieved_header + "\n" + "\n".join(_render_item(it) for it in items))

    parts.append(
        "Emit one decision per item, in the SAME ORDER, via the "
        "gate_decisions tool. Every id must appear exactly once."
    )
    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════
# Tool schema -- forced structured output
# ═══════════════════════════════════════════════════════════════════


_FLAG_KINDS_ENUM = [
    "duplicate_pair",
    "contradiction_pair",
    "stale",
    "unlinked_entity",
    "orphan",
    "generic_summary",
    "edge_candidate",
    # S3a: emitted by declare_operation (NOT the gate) when
    # retrieve_past_operations surfaces >=3 same-tool same-sign
    # precedents. Listed here so the closed-set enum stays centralised.
    "op_cluster_templatizable",
]


GATE_DECISIONS_LEAN = {
    "name": "gate_decisions",
    "description": (
        "Emit a keep/drop decision for every input item, in input "
        "order. Exactly one entry per item id. Lean foreground gate "
        "shape (v3.7.0 Slice 1): no flags array, no quality job. "
        "Quality flags (duplicate_pair, stale, generic_summary, etc.) "
        "are handled by a separate background quality pass and never "
        "block the agent's foreground response."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "reasoning": {
                            "type": "string",
                            "description": (
                                "One sentence: how does this item relate "
                                "to the primary context, or why doesn't it?"
                            ),
                        },
                        "action": {"enum": ["keep", "drop"]},
                        "proposed_summary": {
                            "type": "string",
                            "description": (
                                "OPTIONAL. Set only if action=keep AND "
                                "the item's current summary is generic "
                                "while its content is specific. <=280 "
                                "chars, faithful to content, different "
                                "angle than the content."
                            ),
                        },
                    },
                    "required": ["id", "reasoning", "action"],
                },
            },
        },
        "required": ["decisions"],
    },
}


# Legacy full-shape tool schema preserved for the background quality
# pass (v3.7.0 Slice 2 will wire it). The foreground gate no longer
# uses this -- see GATE_DECISIONS_LEAN above and the messages.create
# call inside InjectionGate.filter(). Adrian directive 2026-05-16
# (Option 3 architecture): drops-only foreground, async quality pass.
GATE_DECISIONS_TOOL = {
    "name": "gate_decisions",
    "description": (
        "Emit a keep/drop decision for every input item, in input "
        "order. Exactly one entry per item id. ALSO emit a flags "
        "array capturing any quality issues visible across the K "
        "items together -- this is the second job of the call and "
        "is VERY IMPORTANT. Empty flags is allowed when nothing "
        "stands out, but under-flagging is a failure mode."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "reasoning": {
                            "type": "string",
                            "description": (
                                "One sentence: how does this item relate "
                                "to the primary context, or why doesn't it?"
                            ),
                        },
                        "action": {"enum": ["keep", "drop"]},
                        "proposed_summary": {
                            "type": "string",
                            "description": (
                                "OPTIONAL. Set only if action=keep AND "
                                "the item's current summary is generic "
                                "while its content is specific. ≤280 "
                                "chars, faithful to content, different "
                                "angle than the content."
                            ),
                        },
                    },
                    "required": ["id", "reasoning", "action"],
                },
            },
            "flags": {
                "type": "array",
                "description": (
                    "Quality issues visible across the K items. Empty "
                    "array if nothing stands out. See the system "
                    "prompt's JOB 2 section for the flag taxonomy."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"enum": _FLAG_KINDS_ENUM},
                        "memory_ids": {
                            "type": "array",
                            "description": (
                                "Item ids this flag is about. Pair kinds "
                                "(duplicate_pair, contradiction_pair, "
                                "edge_candidate) carry exactly 2 ids in "
                                "subject→object order for edge_candidate. "
                                "Single-memory kinds carry 1 id."
                            ),
                            "items": {"type": "string"},
                        },
                        "detail": {
                            "type": "string",
                            "description": (
                                "One sentence explaining the issue. For "
                                "edge_candidate, include the proposed "
                                "predicate (e.g. 'depends_on', "
                                "'replaces', 'built_by')."
                            ),
                        },
                    },
                    "required": ["kind", "memory_ids", "detail"],
                },
            },
        },
        "required": ["decisions"],
    },
}


# ═══════════════════════════════════════════════════════════════════
# Gate runtime
# ═══════════════════════════════════════════════════════════════════


class InjectionGate:
    """Relevance gate wrapper. One instance per palace process.

    Usage::

        gate = InjectionGate()
        result = gate.filter(
            primary_context={...},
            items=[GateItem(...), ...],
            parent_intent={...} or None,
        )
        # result.kept flows to the main agent
        # result.dropped is written back as rated_irrelevant by the
        # caller (via kg.record_feedback(rater_kind='gate_llm'))

    Safety: construction never contacts the API. The client is
    built lazily on first filter() call. If the SDK or key is
    missing, filter() returns GateResult with gate_status='degraded'
    and all items in kept (fail-open).
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_retries: int = 2,
        min_items: int = 3,
        _client=None,  # test injection
    ):
        self.model = model or os.environ.get("MEMPALACE_GATE_MODEL") or _DEFAULT_MODEL
        self.api_key_env = api_key_env
        self.max_retries = max_retries
        self.min_items = min_items
        self._client = _client
        self._client_initialized = _client is not None

    # ── Lazy client init ──

    def _get_client(self):
        if self._client_initialized:
            return self._client
        self._client_initialized = True
        try:
            import anthropic
        except ImportError as exc:
            log.warning("injection_gate: anthropic SDK not available: %s", exc)
            self._client = None
            return None
        # Load the palace .env if the key isn't already in the process
        # environment. Link-author's CLI does this at process start,
        # but the MCP server that hosts the gate does not -- without
        # this call the gate would be permanently key-blind despite
        # the operator having set the key in the documented place.
        # Mirrors mempalace.link_author._load_env with override=True
        # so a stale shell var can't shadow the .env.
        _ensure_palace_env_loaded(self.api_key_env)
        key = os.environ.get(self.api_key_env) or ""
        if not key.strip():
            log.info("injection_gate: %s not set, gate will fail-open", self.api_key_env)
            self._client = None
            return None
        # v3.5.5 hang fix (Adrian directive 2026-05-15): the SDK's
        # default timeout is 10 minutes. If Anthropic stalls the agent
        # waits forever inside apply_gate / run_state_judge -- both
        # called synchronously on the declare_intent / declare_operation
        # critical path. Cap each request at MEMPALACE_HAIKU_TIMEOUT_SEC
        # (default 60s) so a stalled API fails fast and the gate fail-
        # opens (memories pass through; judge returns empty changes).
        # Tune via env when working in a high-latency network.
        try:
            _timeout_s = float(os.environ.get("MEMPALACE_HAIKU_TIMEOUT_SEC", "60"))
        except (TypeError, ValueError):
            _timeout_s = 60.0
        try:
            self._client = anthropic.Anthropic(api_key=key, timeout=_timeout_s)
        except Exception as exc:
            log.warning("injection_gate: client construction failed: %s", exc)
            self._client = None
        return self._client

    # ── Public API ──

    def filter(
        self,
        *,
        primary_context: dict,
        items: list[GateItem],
        parent_intent: dict | None = None,
        session_frame: dict | None = None,
    ) -> GateResult:
        """Filter retrieved items. See module docstring for semantics."""
        import time as _time

        _t0 = _time.perf_counter()

        # K=0: pass-through, no API call.
        if not items:
            return GateResult(
                kept=[],
                dropped=[],
                gate_status={"state": "skipped_empty"},
                timings={
                    "total_ms": round((_time.perf_counter() - _t0) * 1000, 2),
                    "n_items": 0,
                    "attempts": 0,
                },
            )

        # K below min_items: not worth the latency; pass all through.
        # This matches the design decision that very small K doesn't
        # benefit from gating (the agent can eyeball two items).
        if len(items) < self.min_items:
            return GateResult(
                kept=list(items),
                dropped=[],
                gate_status={
                    "state": "skipped_small_k",
                    "k": len(items),
                    "min_k": self.min_items,
                },
                timings={
                    "total_ms": round((_time.perf_counter() - _t0) * 1000, 2),
                    "n_items": len(items),
                    "attempts": 0,
                },
            )

        client = self._get_client()
        if client is None:
            # Distinguish "no key configured" (operator chose not to
            # run the gate -- NOT a runtime failure) from "runtime
            # degradation" (network, timeout, malformed response).
            # Happy-path callers treat this as a silent pass-through
            # and do NOT inject gate_status into their response.
            return GateResult(
                kept=list(items),
                dropped=[],
                gate_status={
                    "state": "skipped_no_client",
                    "reason": "anthropic_sdk_or_key_missing",
                },
                timings={
                    "total_ms": round((_time.perf_counter() - _t0) * 1000, 2),
                    "n_items": len(items),
                    "attempts": 0,
                },
            )

        # Prompt build. Measured separately so callers can see when a
        # long/expensive prompt dominates latency.
        _t_prompt_start = _time.perf_counter()
        prompt = build_prompt(
            primary_context=primary_context,
            items=items,
            parent_intent=parent_intent,
            session_frame=session_frame,
        )
        prompt_ms = round((_time.perf_counter() - _t_prompt_start) * 1000, 2)

        # Forced tool-use: Anthropic guarantees the response uses the
        # named tool, so the decisions arrive as structured arguments
        # -- no free-text JSON parsing.
        last_err = None
        parsed: tuple[dict[str, GateDecision], list[dict]] | None = None
        tokens_in = 0
        tokens_out = 0
        cache_creation = 0
        cache_read = 0
        llm_ms_cum = 0.0  # cumulative LLM wall-clock across retries
        parse_ms = 0.0
        attempts_used = 0
        for attempt in range(self.max_retries):
            attempts_used = attempt + 1
            try:
                _t_llm_start = _time.perf_counter()
                # Prompt caching: the system prompt and tool schema are 100%
                # static across every gate call, so we mark them as cacheable
                # ephemeral blocks. Anthropic skips re-tokenising the prefix
                # on cache hits (≈90% input-token discount, measurable latency
                # cut). Shape: system as a list of content blocks with
                # v3.7.0 Slice 1 (Adrian directive 2026-05-16, Option 3
                # architecture): foreground gate uses _LEAN_SYSTEM_PROMPT
                # + GATE_DECISIONS_LEAN -- drops-only output, no quality
                # flags. Output tokens drop ~95% (K=10 with 2 drops:
                # ~1400 tok -> ~80 tok); at Haiku's ~80 tok/sec stream
                # rate that is the dominant wall-time win (gate falls
                # from ~6-9s to ~2-3s). Cache stops hitting because the
                # lean prefix is ~500 tokens, well below Haiku 4.5's
                # ~4096 cache floor -- intentional: cold-call latency
                # wins decisively when output is the bottleneck, and
                # the full padded prompt now lives on the async
                # background quality pass (Slice 2) where cache hits
                # amortise over its slower cadence. cache_control
                # blocks are kept on system + tools so when Slice 2's
                # bg pass routes through this client, the cache wakes
                # up naturally; for the lean foreground path they are
                # silently no-op (the API never refuses a too-small
                # block, it just does not cache).
                resp = client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=[
                        {
                            "type": "text",
                            "text": _LEAN_SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    tools=[
                        {
                            **GATE_DECISIONS_LEAN,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    tool_choice={"type": "tool", "name": "gate_decisions"},
                    messages=[{"role": "user", "content": prompt}],
                )
                llm_ms_cum += (_time.perf_counter() - _t_llm_start) * 1000
                usage = getattr(resp, "usage", None)
                if usage:
                    tokens_in = int(getattr(usage, "input_tokens", 0) or 0)
                    tokens_out = int(getattr(usage, "output_tokens", 0) or 0)
                    cache_creation = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
                    cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
                _t_parse_start = _time.perf_counter()
                parsed = _extract_decisions(resp, {it.id for it in items})
                parse_ms = round((_time.perf_counter() - _t_parse_start) * 1000, 2)
                if parsed is not None:
                    break
                last_err = "missing_decisions_in_tool_call"
            except Exception as exc:
                # Include elapsed time in the failed attempt so we can
                # see which retry actually consumed latency.
                llm_ms_cum += (_time.perf_counter() - _t_llm_start) * 1000
                last_err = f"{type(exc).__name__}: {exc}"
                log.info("injection_gate attempt %d failed: %s", attempt + 1, last_err)

        llm_ms = round(llm_ms_cum, 2)

        if parsed is None:
            result = self._fail_open(
                items,
                reason=f"judge_failed_after_{self.max_retries}_attempts: {last_err}",
                instruction=(
                    "Relevance gate failed this turn. All items injected "
                    "unfiltered. Surface this to the user -- retrieval "
                    "quality may be reduced. Consider whether to proceed "
                    "or abort, and note the failure in your response."
                ),
            )
            result.timings = {
                "total_ms": round((_time.perf_counter() - _t0) * 1000, 2),
                "prompt_ms": prompt_ms,
                "llm_ms": llm_ms,
                "parse_ms": parse_ms,
                "attempts": attempts_used,
                "n_items": len(items),
            }
            return result

        decisions_by_id, flags = parsed

        # Route items by decision. Missing decisions fail-open for
        # that item (kept with a synthetic decision noting the miss).
        kept: list[GateItem] = []
        dropped: list[tuple[GateItem, GateDecision]] = []
        for item in items:
            dec = decisions_by_id.get(item.id)
            if dec is None:
                kept.append(item)
                continue
            if dec.action == "drop":
                dropped.append((item, dec))
            else:
                kept.append(item)

        total_ms = round((_time.perf_counter() - _t0) * 1000, 2)
        # Single logger line per gate run -- visible in the MCP server
        # log even when the caller doesn't surface gate_status. Shape
        # is grep-friendly: `gate.timing` prefix + key=value pairs.
        log.info(
            "gate.timing n_items=%d kept=%d dropped=%d total_ms=%.1f "
            "prompt_ms=%.1f llm_ms=%.1f parse_ms=%.1f attempts=%d "
            "tokens_in=%d tokens_out=%d cache_read=%d cache_creation=%d",
            len(items),
            len(kept),
            len(dropped),
            total_ms,
            prompt_ms,
            llm_ms,
            parse_ms,
            attempts_used,
            tokens_in,
            tokens_out,
            cache_read,
            cache_creation,
        )
        return GateResult(
            kept=kept,
            dropped=dropped,
            gate_status={"state": "ok"},
            judge_tokens_in=tokens_in,
            judge_tokens_out=tokens_out,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
            flags=flags,
            timings={
                "total_ms": total_ms,
                "prompt_ms": prompt_ms,
                "llm_ms": llm_ms,
                "parse_ms": parse_ms,
                "attempts": attempts_used,
                "n_items": len(items),
            },
        )

    # ── Background quality pass (v3.7.2 Slice 2) ──

    def run_quality_pass(
        self,
        *,
        primary_context: dict,
        items: list[GateItem],
        parent_intent: dict | None = None,
        session_frame: dict | None = None,
    ) -> list[dict]:
        """v3.7.2 Slice 2 (Adrian directive 2026-05-16, Option 3
        architecture): background quality flag emission. Runs a SECOND
        Haiku call against the same items as filter() using the FULL
        padded _SYSTEM_PROMPT + GATE_DECISIONS_TOOL schema, then
        returns ONLY the flags list. Decisions are discarded -- the
        foreground lean gate (Slice 1) already made the keep/drop
        call.

        Designed to run inside ``_BG_QUALITY_EXECUTOR``'s worker
        thread. NEVER raises -- every failure path returns ``[]`` so
        the apply_gate spawner can submit + forget without try/except
        around the whole closure.

        The full prompt lifts the bg pass above Haiku 4.5's ~4096
        token cache floor, so a busy session's 2nd+ submission within
        the 5-minute TTL pays only ~10% of the input cost. Foreground
        latency is unaffected -- foreground returns immediately after
        gate.filter() while this method runs in the background.
        """
        if not items:
            return []
        try:
            client = self._get_client()
            if client is None:
                return []
            prompt = build_prompt(
                primary_context=primary_context,
                items=items,
                parent_intent=parent_intent,
                session_frame=session_frame,
            )
            resp = client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=[
                    {
                        **GATE_DECISIONS_TOOL,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tool_choice={"type": "tool", "name": "gate_decisions"},
                messages=[{"role": "user", "content": prompt}],
            )
            parsed = _extract_decisions(resp, {it.id for it in items})
            if parsed is None:
                return []
            _decisions_by_id, flags = parsed
            return flags
        except Exception as exc:
            log.info("injection_gate.run_quality_pass: failed: %s", exc)
            return []

    # ── Helpers ──

    def _fail_open(self, items: list[GateItem], *, reason: str, instruction: str) -> GateResult:
        return GateResult(
            kept=list(items),
            dropped=[],
            gate_status={
                "state": "degraded",
                "reason": reason,
                "fallback": f"all {len(items)} items injected unfiltered",
                "agent_instruction": instruction,
            },
        )


# ═══════════════════════════════════════════════════════════════════
# Tool-call parser
# ═══════════════════════════════════════════════════════════════════


def _extract_decisions(
    resp, known_ids: set[str]
) -> tuple[dict[str, GateDecision], list[dict]] | None:
    """Parse a forced tool-use response into (decisions_by_id, flags).

    Returns None if the expected tool_use block is absent or malformed
    -- the caller interprets that as "retry then fail-open".

    Unknown ids (hallucinated by the model) are dropped. Duplicate ids
    keep the first decision. Flags are filtered to valid kinds + at
    least one known memory id per flag; malformed entries are dropped
    silently so a bad flag can't sink the whole response.
    """
    blocks = getattr(resp, "content", None) or []
    # Diagnostic context (Adrian directive 2026-05-10): when extraction
    # returns None the caller emits the opaque
    # "missing_decisions_in_tool_call" string and retries -- but the
    # actual failure mode (no tool_use block, wrong tool name, decisions
    # field missing/wrong type) was lost. Capture the shape of every
    # block we saw so the next degraded log entry tells us which mode
    # Haiku hit. Sticks to log.info so it doesn't spam quiet runs.
    block_shapes: list[str] = []
    for block in blocks:
        b_type = getattr(block, "type", None)
        b_name = getattr(block, "name", None)
        b_input = getattr(block, "input", None)
        input_keys = list(b_input.keys()) if isinstance(b_input, dict) else type(b_input).__name__
        block_shapes.append(f"type={b_type} name={b_name} input_keys={input_keys}")
        if b_type != "tool_use":
            continue
        if b_name != "gate_decisions":
            continue
        inp = b_input or {}
        raw_decisions = inp.get("decisions") if isinstance(inp, dict) else None
        if not isinstance(raw_decisions, list):
            log.info(
                "injection_gate _extract_decisions: tool_use found but decisions "
                "shape unexpected -- input_keys=%s raw_decisions_type=%s; block_shapes=%s",
                input_keys,
                type(raw_decisions).__name__ if raw_decisions is not None else "None",
                block_shapes,
            )
            return None
        by_id: dict[str, GateDecision] = {}
        for d in raw_decisions:
            if not isinstance(d, dict):
                continue
            did = d.get("id")
            action = d.get("action")
            reasoning = d.get("reasoning") or ""
            proposed = d.get("proposed_summary")
            if not isinstance(did, str) or did not in known_ids or action not in ("keep", "drop"):
                continue
            if did in by_id:
                continue
            by_id[did] = GateDecision(
                id=did,
                action=action,
                reasoning=str(reasoning),
                proposed_summary=str(proposed) if isinstance(proposed, str) else None,
            )

        # Flags are optional. An absent key is not malformed -- it's
        # "judge had nothing to flag this call".
        raw_flags = inp.get("flags") if isinstance(inp, dict) else None
        flags: list[dict] = []
        if isinstance(raw_flags, list):
            for f in raw_flags:
                if not isinstance(f, dict):
                    continue
                kind = f.get("kind")
                if kind not in _FLAG_KINDS_ENUM:
                    continue
                mids = f.get("memory_ids")
                if not isinstance(mids, list):
                    continue
                cleaned_ids = [str(m) for m in mids if isinstance(m, str) and m]
                if not cleaned_ids:
                    continue
                flags.append(
                    {
                        "kind": kind,
                        "memory_ids": cleaned_ids,
                        "detail": str(f.get("detail") or ""),
                    }
                )
        return by_id, flags
    # Reached when NO block in resp.content was both type==tool_use and
    # name==gate_decisions. Most common cause: Haiku declined the
    # tool_choice (returned text-only). Surface the block shapes so the
    # caller's degraded-log entry tells us which mode hit instead of the
    # opaque "missing_decisions_in_tool_call" string.
    log.info(
        "injection_gate _extract_decisions: no gate_decisions tool_use block "
        "in response; block_shapes=%s",
        block_shapes,
    )
    return None


# ═══════════════════════════════════════════════════════════════════
# Caller-side helper: write dropped items back as rated_irrelevant.
# ═══════════════════════════════════════════════════════════════════


def persist_drops(
    kg,
    *,
    context_id: str,
    dropped: list[tuple[GateItem, GateDecision]],
    rater_id: str = "claude-haiku-gate",
) -> int:
    """Write rated_irrelevant feedback for every dropped item.

    Uses KnowledgeGraph.record_feedback so both entity-scope and
    triple-scope feedback go through the unified dispatcher --
    entity-target drops become rated_irrelevant edges on context →
    entity; triple-target drops become rows in
    triple_context_feedback. No phantom entities.

    relevance=2 (rated_irrelevant, non-misleading noise) mirrors the
    user's guidance that the gate is bias-to-keep; a gate DROP is
    "on-topic enough to have surfaced but still not useful for this
    context" -- squarely relevance=2 rather than =1 (misleading).

    Returns the number of successful writes. Failures are logged but
    non-fatal; the caller should treat this as best-effort.
    """
    if not dropped or not context_id:
        return 0
    n = 0
    for item, dec in dropped:
        target_kind = "triple" if item.source == "triple" else "entity"
        try:
            kg.record_feedback(
                context_id,
                item.id,
                target_kind,
                relevance=2,
                reason=f"[gate] {dec.reasoning[:400]}",
                rater_kind="gate_llm",
                rater_id=rater_id,
            )
            n += 1
        except Exception as exc:  # pragma: no cover -- best-effort
            log.info(
                "persist_drops: record_feedback failed for %s: %s",
                item.id,
                exc,
            )
    return n


# ═══════════════════════════════════════════════════════════════════
# Session-frame helper
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
# Palace .env loader -- shared with memory_gardener for env parity
# ═══════════════════════════════════════════════════════════════════


_PALACE_ENV_LOADED = False


def _ensure_palace_env_loaded(api_key_env: str = "ANTHROPIC_API_KEY") -> None:
    """Load <palace>/.env into os.environ the first time any gate /
    gardener code path needs an API key.

    Link-author's CLI calls ``_load_env`` at process start because its
    entry point is a dedicated subcommand. The gate instead runs inside
    the long-lived MCP server process, which never had a documented
    place to load the .env -- so the key was invisible even though the
    file was present. This helper closes that gap exactly once per
    process, at the moment the first client is constructed. override
    is True so a stale / empty shell var can't shadow the file.

    No-op when the key is already in os.environ (shell-set), when the
    palace path can't be resolved, when the .env file is absent, or
    when python-dotenv is missing (hard dep but defensive).
    """
    global _PALACE_ENV_LOADED
    if _PALACE_ENV_LOADED:
        return
    _PALACE_ENV_LOADED = True
    if os.environ.get(api_key_env, "").strip():
        # Already in process env -- shell-set, parent-inherited, or a
        # prior load. Nothing to do.
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        log.info("injection_gate: python-dotenv missing; skipping .env load")
        return
    try:
        from .config import MempalaceConfig

        cfg = MempalaceConfig()
        palace = getattr(cfg, "palace_path", None)
    except Exception as exc:
        log.info("injection_gate: MempalaceConfig unavailable: %s", exc)
        palace = None
    if not palace:
        return
    from pathlib import Path

    target = Path(palace) / ".env"
    if not target.is_file():
        log.info("injection_gate: no .env at %s; skipping", target)
        return
    try:
        load_dotenv(str(target), override=True)
        log.info("injection_gate: palace .env loaded from %s", target)
    except Exception as exc:
        log.info("injection_gate: load_dotenv failed: %s", exc)


def build_session_frame(
    *,
    agent: str | None,
    cwd: str | None = None,
) -> dict:
    """Assemble the optional session-frame block for the gate prompt.

    Only includes cwd-derived ``project`` when an anchor file is
    present in cwd (prevents misleading project tags from sessions
    launched outside a project root).
    """
    frame: dict = {}
    if agent:
        frame["agent"] = agent
    project = _detect_project_anchor(cwd)
    if project:
        frame["project"] = project
    return frame


# ═══════════════════════════════════════════════════════════════════
# Process-wide singleton + wiring helper
# ═══════════════════════════════════════════════════════════════════


_GATE_SINGLETON: InjectionGate | None = None


def get_gate() -> InjectionGate:
    """Lazy process-wide InjectionGate.

    One gate per process so the Anthropic client (held on the gate
    instance) isn't rebuilt per retrieval call. Tests inject their
    own by constructing InjectionGate(_client=...) directly.
    """
    global _GATE_SINGLETON
    if _GATE_SINGLETON is None:
        _GATE_SINGLETON = InjectionGate()
    return _GATE_SINGLETON


def _gate_disabled() -> bool:
    """Opt-out via env: MEMPALACE_GATE_DISABLED=1 turns the gate off
    entirely (apply_gate becomes a pass-through). Used in tests and
    by operators who want to roll back without a code change."""
    return os.environ.get("MEMPALACE_GATE_DISABLED", "").strip() in (
        "1",
        "true",
        "True",
        "yes",
    )


def _gate_report_disabled() -> bool:
    """Adrian directive 2026-05-06: agents asked for visibility into gate
    filtering -- how many memories went in, how many came out, how long
    the gate took. Returned by default on every memory-surfacing tool;
    operators can opt out via MEMPALACE_GATE_REPORT_DISABLED=1 if the
    extra ~3 fields per response are unwelcome.
    """
    return os.environ.get("MEMPALACE_GATE_REPORT_DISABLED", "").strip() in (
        "1",
        "true",
        "True",
        "yes",
    )


def apply_gate(  # noqa: C901
    *,
    memories: list[dict],
    combined_meta: dict | None,
    primary_context: dict,
    context_id: str,
    kg,
    agent: str | None,
    parent_intent: dict | None = None,
    gate: InjectionGate | None = None,
    default_channel: str = "A",
) -> tuple[list[dict], dict | None, dict | None]:
    """Run the injection gate on a built retrieval list.

    Shared by declare_intent, declare_operation, and kg_search so the
    wiring is one call per site. Input ``memories`` is the list the
    caller was about to return (dicts with ``id`` and ``text``;
    optional ``hybrid_score`` / ``score``). Output is the filtered
    list plus an optional ``gate_status`` dict to surface in the
    response ONLY when the state is not ``ok``/``skipped_empty``/
    ``skipped_small_k`` (callers check the second return for None).

    Dropped items are persisted via record_feedback with
    rater_kind='gate_llm' -- entity drops become rated_irrelevant
    edges, triple drops land in triple_context_feedback, no phantom
    entities.

    Fail-open: any exception is caught and the original ``memories``
    list passes through unchanged; callers never see the gate kill
    their payload on a bug in this module.
    """
    import time as _time

    _apply_t0 = _time.perf_counter()
    _input_count = len(memories or [])

    def _report_passthrough():
        # Build a passthrough gate_report when the gate didn't filter
        # (disabled, empty input, or early-failure). Counts are in==out;
        # elapsed reflects whatever ran. Returns None when gate_report
        # is disabled by env.
        if _gate_report_disabled():
            return None
        elapsed = round((_time.perf_counter() - _apply_t0) * 1000, 2)
        # v3.4.2 (Adrian post-reinstall 2026-05-13): OMIT the tokens
        # block when no Haiku call was made. The prior v3.3.0 design
        # always emitted tokens with all-zero values on the passthrough
        # path; that read as "broken telemetry" to anyone scanning the
        # response. Truth is: zero tokens means zero Haiku spend --
        # the gate short-circuited (gate disabled, empty memories,
        # get_gate() failure, or filter() raised before LLM call).
        # No tokens block in the response now signals "no LLM ran"
        # unambiguously; presence of tokens block always means real
        # usage.
        return {
            "input_count": _input_count,
            "output_count": _input_count,
            "elapsed_ms": elapsed,
        }

    if _gate_disabled() or not memories:
        return memories, None, _report_passthrough()
    try:
        gate = gate or get_gate()
    except Exception as exc:  # pragma: no cover -- defensive
        log.info("apply_gate: get_gate failed: %s", exc)
        return memories, None, _report_passthrough()

    items: list[GateItem] = []
    for i, m in enumerate(memories):
        mid = m.get("id")
        if not mid:
            continue
        meta_entry = (combined_meta or {}).get(mid, {}) or {}
        source = meta_entry.get("source") or ("triple" if str(mid).startswith("t_") else "memory")
        if source not in ("memory", "entity", "triple"):
            source = "memory"
        extras = {}
        raw_meta = meta_entry.get("meta") or {}
        if isinstance(raw_meta, dict):
            # Scrub surrogates from every string value -- name, summary,
            # description, statement, spo fields all land in the judge
            # prompt verbatim via _render_item, and any one of them can
            # carry a stray U+DC9D from a legacy record written before
            # the write-side sanitizer existed.
            extras = {
                k: _scrub_surrogates(v) if isinstance(v, str) else v for k, v in raw_meta.items()
            }
        doc = meta_entry.get("doc") or m.get("text") or ""
        score = m.get("hybrid_score")
        if score is None:
            score = m.get("score") or meta_entry.get("similarity") or 0.0
        # Adrian's design lock 2026-05-02: enrich extras with the
        # structured `properties.summary` dict so _render_item can
        # project labeled WHAT/WHY/SCOPE lines for the gate. SQLite is
        # the canonical source (gardener-rewritten summaries land there
        # first and may not be re-synced to Chroma metadata yet -- see
        # record_ga_agent_q3_corrected_gardener_rewrite_stale_views_2026_05).
        # Best-effort: any failure leaves extras['summary_dict'] absent
        # and the renderer falls back to the legacy single-line prose.
        if kg is not None:
            try:
                ent = kg.get_entity(str(mid))
                if ent:
                    props = ent.get("properties") or {}
                    if isinstance(props, str):
                        try:
                            import json as _json

                            props = _json.loads(props)
                        except Exception:
                            props = {}
                    summary_dict = props.get("summary") if isinstance(props, dict) else None
                    if isinstance(summary_dict, dict) and summary_dict:
                        if source == "entity":
                            extras["properties_summary_dict"] = summary_dict
                        else:
                            extras["summary_dict"] = summary_dict
                    # State-protocol v1 (Adrian Option B 2026-05-03):
                    # forward state_schema_id + state_updatable into
                    # extras when the surfaced entity is a state-bearing
                    # kind=class (Task / agent / intent_type carry these
                    # via seed_ontology + _ensure_task_ontology). The
                    # delta-coverage rule downstream reads these to
                    # identify state-bearing memories without a second
                    # graph hop. Instance entities (kind=entity is_a
                    # <Class>) do their class lookup at delta time --
                    # this branch only catches the class-level case.
                    if isinstance(props, dict):
                        sid = props.get("state_schema_id")
                        if isinstance(sid, str) and sid:
                            extras["state_schema_id"] = sid
                            extras["state_updatable"] = bool(props.get("state_updatable"))
            except Exception:  # pragma: no cover -- defensive
                pass
        items.append(
            GateItem(
                id=str(mid),
                source=source,  # type: ignore[arg-type]
                text=_scrub_surrogates(str(doc or "")),
                channel=default_channel,
                rank=i + 1,
                score=float(score or 0.0),
                extra=extras,
            )
        )

    frame = build_session_frame(agent=agent, cwd=os.getcwd())
    try:
        result = gate.filter(
            primary_context=primary_context,
            items=items,
            parent_intent=parent_intent,
            session_frame=frame,
        )
    except Exception as exc:  # pragma: no cover -- defensive
        log.info("apply_gate: filter failed: %s", exc)
        return memories, None, _report_passthrough()

    if result.dropped and context_id:
        try:
            persist_drops(kg, context_id=context_id, dropped=result.dropped)
        except Exception as exc:  # pragma: no cover -- best-effort
            log.info("apply_gate: persist_drops failed: %s", exc)

    # Persist quality flags for the memory_gardener background process.
    # Scoped to the active context so re-observing the same issue in
    # the same context bumps the existing row instead of duplicating.
    # Best-effort: a flag-write failure must not prevent returning
    # kept items.
    if result.flags and context_id:
        try:
            enriched = [{**f, "context_id": context_id} for f in result.flags]
            kg.record_memory_flags(enriched, rater_model=getattr(gate, "model", "") or "")
        except Exception as exc:  # pragma: no cover -- best-effort
            log.info("apply_gate: record_memory_flags failed: %s", exc)

    # v3.7.2 Slice 2 (Adrian directive 2026-05-16, Option 3
    # architecture): background quality pass. Slice 1 stripped flags
    # from the foreground gate; this re-introduces flag emission via
    # an async Haiku call against the full padded prompt + schema, so
    # the agent's foreground response is unaffected but the gardener
    # still gets flag rows. Single-worker daemon pool bounds API
    # pressure; daemon threads die with the process so a hung
    # Anthropic call cannot block interpreter exit. Disable via
    # MEMPALACE_BG_QUALITY=0.
    if (
        os.environ.get("MEMPALACE_BG_QUALITY", "1").strip() != "0"
        and items
        and context_id
        and kg is not None
    ):
        try:
            _bg_items = list(items)
            _bg_pctx = primary_context
            _bg_pi = parent_intent
            _bg_sf = frame
            _bg_cid = context_id
            _bg_kg = kg
            _bg_rater = getattr(gate, "model", "") or ""
            _bg_agent = agent

            def _run_bg_quality():
                _bg_t0 = _time.perf_counter()
                n_flags = 0
                try:
                    bg_flags = gate.run_quality_pass(
                        primary_context=_bg_pctx,
                        items=_bg_items,
                        parent_intent=_bg_pi,
                        session_frame=_bg_sf,
                    )
                except Exception:
                    bg_flags = []
                if bg_flags:
                    try:
                        enriched = [{**f, "context_id": _bg_cid} for f in bg_flags]
                        _bg_kg.record_memory_flags(enriched, rater_model=_bg_rater)
                        n_flags = len(bg_flags)
                    except Exception:
                        pass
                try:
                    from datetime import datetime as _dt2
                    from datetime import timezone as _tz2

                    from .mcp_server import _telemetry_append_jsonl as _tel2

                    _tel2(
                        "bg_quality_log.jsonl",
                        {
                            "ts": _dt2.now(_tz2.utc).isoformat(timespec="seconds"),
                            "context_id": _bg_cid or "",
                            "agent": _bg_agent or "",
                            "n_items": len(_bg_items),
                            "n_flags": n_flags,
                            "elapsed_ms": round((_time.perf_counter() - _bg_t0) * 1000, 2),
                            "model": _bg_rater,
                        },
                    )
                except Exception:
                    pass

            _BG_QUALITY_EXECUTOR.submit(_run_bg_quality)
        except Exception:
            pass  # bg spawn must not affect foreground

    kept_ids = {it.id for it in result.kept}

    # Phase 6 lazy-migration-at-injection (Adrian design lock 2026-05-03):
    # for each kept entity, check whether its latest state revision is at
    # a schema_version below the current registered version. If so, walk
    # the migration chain in mempalace/state_migrations/{schema_id}/
    # v{N}_to_v{N+1}.py and write a new revision at the current version.
    # The hook fires HERE -- after the gate dropped irrelevant items --
    # so dormant entities never pay migration cost; only entities the
    # LLM is about to consume get migrated. Failures are caught
    # per-entity inside migrate_state_for_entities; this whole block is
    # also wrapped to fail-open so a bug here cannot kill the gate
    # path.
    try:
        kg.migrate_state_for_entities(kept_ids)
    except Exception as exc:  # pragma: no cover - defensive
        log.info("apply_gate: migrate_state_for_entities failed: %s", exc)

    # Time-touch / decay-reset post-gate (Adrian directive 2026-05-04):
    # bump last_touched on the kept entities so the decay clock tracks
    # actual utility, not raw retrieval traffic. Doing this BEFORE the
    # gate would refresh decay for items the gate later filters as
    # irrelevant -- noise. Doing it HERE means only items the LLM is
    # about to consume get their decay clocks reset. Single batch
    # UPDATE; failures fail-open like the migration hook above.
    try:
        kg.touch_entities(kept_ids)
    except Exception as exc:  # pragma: no cover - defensive
        log.info("apply_gate: touch_entities failed: %s", exc)

    filtered = [m for m in memories if str(m.get("id")) in kept_ids]

    # Telemetry: one row per apply_gate call appended to
    # ~/.mempalace/hook_state/gate_log.jsonl. Mirrors search_log /
    # finalize_log so the eval harness can report on gate latency +
    # drop rate alongside retrieval metrics. Best-effort: telemetry
    # failures must not change returned items.
    try:
        from datetime import datetime as _dt, timezone as _tz

        from .mcp_server import _telemetry_append_jsonl as _tel

        apply_total_ms = round((_time.perf_counter() - _apply_t0) * 1000, 2)
        _tel(
            "gate_log.jsonl",
            {
                "ts": _dt.now(_tz.utc).isoformat(timespec="seconds"),
                "context_id": context_id or "",
                "agent": agent or "",
                "state": result.gate_status.get("state"),
                "n_items": len(items),
                "n_kept": len(result.kept),
                "n_dropped": len(result.dropped),
                "n_flags": len(result.flags),
                "tokens_in": result.judge_tokens_in,
                "tokens_out": result.judge_tokens_out,
                "cache_read_input_tokens": result.cache_read_input_tokens,
                "cache_creation_input_tokens": result.cache_creation_input_tokens,
                "timings": result.timings,
                "apply_total_ms": apply_total_ms,
                "model": getattr(gate, "model", "") or "",
            },
        )
    except Exception:
        pass  # telemetry is best-effort

    # Build gate_report (Adrian directive 2026-05-06): input/output
    # counts + elapsed ms, returned by default on every memory-surfacing
    # tool. None when MEMPALACE_GATE_REPORT_DISABLED=1.
    # v3.3.0 Phase 3 (Adrian directive 2026-05-13): mirror
    # state_judge_report.tokens block so prompt-cache effectiveness is
    # visible inline. result already carries the 4 fields from the
    # Anthropic usage block; we just rename to the canonical short
    # keys (input/output/cache_read/cache_creation) for parity with
    # state_judge_report.tokens.
    if _gate_report_disabled():
        _gate_report = None
    else:
        _gate_report = {
            "input_count": _input_count,
            "output_count": len(filtered),
            "elapsed_ms": round((_time.perf_counter() - _apply_t0) * 1000, 2),
        }
        # v3.4.2 (Adrian post-reinstall 2026-05-13): only emit the
        # tokens block when an LLM call actually happened. The filter
        # path also reaches here on K-below-threshold short-circuits
        # (gate.filter returns skipped_* state with zero tokens); in
        # that case the tokens block is misleading. Test the canonical
        # signal -- judge_tokens_in -- and omit the block entirely when
        # zero so the response unambiguously says "no Haiku call".
        _tokens_in = int(getattr(result, "judge_tokens_in", 0) or 0)
        _tokens_out = int(getattr(result, "judge_tokens_out", 0) or 0)
        if _tokens_in or _tokens_out:
            _gate_report["tokens"] = {
                "input": _tokens_in,
                "output": _tokens_out,
                "cache_read": int(getattr(result, "cache_read_input_tokens", 0) or 0),
                "cache_creation": int(getattr(result, "cache_creation_input_tokens", 0) or 0),
            }

    state = result.gate_status.get("state")
    # Only surface gate_status on non-happy-path outcomes. The default
    # path should add zero extra tokens to the response. "skipped_*"
    # states are all happy-path: empty input, K below threshold, or
    # no API key configured (operator-chosen opt-out). Only "degraded"
    # (actual runtime failure) reaches the main agent -- that's the
    # signal worth surfacing.
    if state in (
        "ok",
        "skipped_empty",
        "skipped_small_k",
        "skipped_no_client",
    ):
        return filtered, None, _gate_report
    return filtered, result.gate_status, _gate_report


# ═══════════════════════════════════════════════════════════════════
# State-judge (Adrian directive 2026-05-07)
# ═══════════════════════════════════════════════════════════════════
#
# Out-of-loop Haiku that detects state changes the main agent might
# overlook. Replaces the implicit-active-set unchanged-ack default,
# which agents always defaulted to (zero signal). The judge sees the
# intent's transcript-so-far + the current state of every entity in
# the followed set and outputs ``{changes: [{entity_id, reason}]}``.
# Empty list = call proceeds; non-empty = call blocked, agent must
# author RFC 6902 patches (or override with status='unchanged' +
# justification only when 100% sure the judge was wrong).
#
# Detection-only (Adrian directive): the judge does NOT propose the
# patch shape. The main agent is the one that has the diff in its
# head; the judge just flags that a diff exists.
#
# Failure policy: fail-open. If the SDK / API key / API call fails,
# return changes=[] so the call proceeds. The explicit-coverage rule
# at finalize_intent already enforces a baseline; the judge is the
# upgrade. Mirrors apply_gate's degraded-path behavior.


def _state_judge_disabled() -> bool:
    """Operator-level kill-switch -- skip judge entirely when set."""
    return os.environ.get("MEMPALACE_STATE_JUDGE_DISABLED", "").strip() in (
        "1",
        "true",
        "True",
        "yes",
    )


def _state_judge_report_disabled() -> bool:
    """Opt-out for the cost/timing report attached to responses."""
    return os.environ.get("MEMPALACE_STATE_JUDGE_REPORT_DISABLED", "").strip() in (
        "1",
        "true",
        "True",
        "yes",
    )


def _attach_patch_if_changes(
    change_out: dict,
    raw_patch,
    eid: str,
    current_by_eid: dict,
) -> None:
    """Attach `patch` to change_out only if it would actually change state.

    v3.7.17 (Adrian directive 2026-05-17). The judge LLM frequently
    proposes RFC 6902 patches whose value already equals what's at
    `path` in the entity's current_state (e.g. /status -> "done" when
    the entry is already "done"). Those produce zero behavioural change
    but still trigger record_state_revision writes and clutter the
    response. Apply the patch to current_state; if the projected
    payload equals current_state, drop the patch and stamp
    skip_reason='no_op_patch' instead. The change still surfaces with
    its `reason` text -- only the patch and downstream auto-apply are
    skipped.

    Degradation: if jsonpatch is missing, the entity has no entry in
    current_by_eid, or the patch is malformed, the patch passes
    through verbatim so the apply site can surface any error.
    """
    if not isinstance(raw_patch, list) or not raw_patch:
        return  # nothing to attach
    if eid not in current_by_eid:
        change_out["patch"] = raw_patch  # no current state, pass through
        return
    try:
        import jsonpatch as _jp_local
    except Exception:
        change_out["patch"] = raw_patch  # jsonpatch missing, pass through
        return
    current = current_by_eid.get(eid) or {}
    try:
        projected = _jp_local.apply_patch(current, raw_patch)
    except Exception:
        # Malformed patch -- pass through so apply site surfaces error.
        change_out["patch"] = raw_patch
        return
    if projected == current:
        change_out["skip_reason"] = "no_op_patch"
        return  # patch dropped (no behavioural change)
    change_out["patch"] = raw_patch


def run_state_judge(
    *,
    transcript_text: str,
    entity_states: list[dict],
    agent: str | None,
    gate: InjectionGate | None = None,
) -> tuple[list[dict], dict | None]:
    """Detect state changes in a running intent.

    Args:
        transcript_text: a rendered prose log of the intent so far
            (declare_intent + every declare_operation that has fired).
        entity_states: list of {entity_id, state_schema_id, current_state}
            for the followed set (agent + intent context + any
            state-bearing instance surfaced this intent).
        agent: requesting agent id (recorded in telemetry).
        gate: optional InjectionGate instance for client reuse + tests.

    Returns:
        (changes, report) where changes is a list of
        {entity_id, reason} dicts (possibly empty) and report is
        {elapsed_ms, detected_count, model, tokens: {input, output,
        cache_read, cache_creation}} or None when MEMPALACE_STATE_JUDGE_REPORT_DISABLED=1.

        Both come back as ([], None) when MEMPALACE_STATE_JUDGE_DISABLED=1.
    """
    import time as _time

    if _state_judge_disabled():
        return [], None

    _t0 = _time.perf_counter()
    try:
        gate = gate or get_gate()
        client = gate._get_client() if gate is not None else None
    except Exception as exc:
        log.info("run_state_judge: client init failed: %s", exc)
        return [], None
    if client is None:
        return [], None

    # Build the user message: entity states + transcript.
    import json as _json

    states_block = _json.dumps(entity_states, indent=2, ensure_ascii=False)
    user_content = (
        "## Followed entity states (current values)\n\n"
        f"```json\n{states_block}\n```\n\n"
        "## Intent transcript so far\n\n"
        f"{transcript_text or '(empty)'}\n"
    )

    # Build the schemas block: for each unique state_schema_id in the
    # followed set, dump the JSON Schema + slot descriptions. Goes into
    # the cacheable system prefix so within an intent's repeated
    # state_judge calls (same followed_set -> same schemas) the prefix
    # is stable and the 5-min ephemeral cache hits. Adrian directive
    # 2026-05-11: "show whatever is or should be enforced for the
    # current intent, all this in the beginning so it'll add up to
    # instructions and other things, and can be cached then." No
    # hardcoded enumeration of task/intent/agent -- whatever schemas
    # the entity_states actually carry get rendered.
    try:
        from mempalace.state_schemas import STATE_SCHEMAS as _STATE_SCHEMAS
    except Exception:
        _STATE_SCHEMAS = {}

    seen_schema_ids: list[str] = []
    for es in entity_states or []:
        sid = (es.get("state_schema_id") or "").strip()
        if sid and sid not in seen_schema_ids:
            seen_schema_ids.append(sid)

    schemas_chunks: list[str] = []
    for sid in seen_schema_ids:
        schema_def = _STATE_SCHEMAS.get(sid)
        if not schema_def:
            schemas_chunks.append(
                f"### {sid}\n(schema not registered -- judge must infer shape from current_state)"
            )
            continue
        json_schema = schema_def.get("json_schema") or {}
        slot_descriptions = schema_def.get("slot_descriptions") or {}
        json_block = _json.dumps(json_schema, indent=2, ensure_ascii=False)
        chunk = f"### {sid}\n\n```json\n{json_block}\n```"
        if slot_descriptions:
            desc_lines = [f"- `{field}`: {desc}" for field, desc in slot_descriptions.items()]
            chunk += "\n\nField meanings:\n" + "\n".join(desc_lines)
        schemas_chunks.append(chunk)

    if schemas_chunks:
        schemas_block = (
            "\n\n## State schemas (enforced for this intent's followed "
            "entities)\n\n" + "\n\n".join(schemas_chunks)
        )
    else:
        schemas_block = ""

    system_prompt = (
        "You are a state-change detector. You receive (1) the current "
        "state values of a small set of state-bearing entities the "
        "agent is following, and (2) a transcript of an intent's "
        "activity so far.\n\n"
        "Your ONLY job: decide, per entity, whether the transcript "
        "reveals that the entity's state HAS ALREADY changed (or is "
        "now stale relative to the activity that just occurred). "
        "Output, per flagged entity: entity_id, reason, schema_id, "
        "AND an RFC 6902 'patch' (JSON Patch ops moving current_state "
        "to the corrected value). The patch is REQUIRED whenever the "
        "fix is concrete -- e.g. a specific field changed value, a "
        "todo advanced status, an active_id should shift, a list "
        "needs to be initialized, a phase/step moved. If you can "
        "articulate WHAT the new value should be in your 'reason', "
        "you can construct the patch -- emit it. The agent's auto-"
        "apply path consumes the patch via record_state_revision so "
        "the agent never has to re-derive what you already figured "
        "out.\n\n"
        "Omit patch ONLY when the fix is genuinely ambiguous -- "
        "e.g. you see state is stale but the transcript doesn't "
        "pin down what the corrected value should be. 'Patch I "
        "could infer but didn't bother' is the v3.2.x failure mode; "
        "don't do that. 'Wrong patch I shouldn't have emitted' is "
        "rarer and recoverable -- the agent retracts wrong patches "
        "via mempalace_challenge_state_change. So when in doubt "
        "between emit-and-risk-wrong vs. omit-and-force-agent-to-"
        "redo-the-analysis, emit. Under-emitting wastes the agent's "
        "tokens; the agent retracts wrong patches cheaply.\n\n"
        "RFC 6902 ops you'll use most: 'replace' (set existing "
        "field to new value), 'add' (set a field that may not "
        "yet exist, or append to a list via path ending in '/-'), "
        "'remove' (drop a field). Always include `schema_id` "
        "matching the entity's state_schema_id so the agent can "
        "validate the patched payload against the schema before "
        "persisting.\n\n"
        "NO-OP GUARD (v3.7.18, Adrian directive 2026-05-17): before "
        "you emit any patch op, look at the entity's current_state at "
        "the path you're about to change. If your proposed value is "
        "ALREADY exactly what's there, DROP that op from your patch. "
        "If after dropping no-op ops your patch becomes empty, omit "
        "the patch field entirely (you can still flag the entity with "
        "reason text if state really is wrong but you can't articulate "
        "a fix; that's a flag-only change). A patch that sets a field "
        "to its existing value is pure waste -- it costs the agent's "
        "tokens, your next-call cache budget, and creates noise in the "
        "graph's revision history with zero behavioural change. "
        "Examples of no-ops to drop: '/todos/0/status' -> 'done' when "
        "current_state already has /todos/0/status='done'; "
        "'/current_focus' -> 'X' when current_state already has "
        "current_focus='X'; '/active_todo_id' -> 't3' when "
        "current_state already has active_todo_id='t3'. The pipeline "
        "WILL filter these no-ops at the boundary, but it's faster + "
        "cheaper for both sides if you don't emit them at all.\n\n"
        "You MUST flag every entity whose current_state does not "
        "exactly match the latest activity in the transcript. When "
        "in doubt about whether something changed, flag. There is no "
        "penalty for over-flagging -- the agent absorbs one "
        "state_deltas ack per flag. There is real cost to under-"
        "flagging -- stale state silently accumulates in the graph "
        "and downstream queries return wrong answers. Default to "
        "flag.\n\nIMPORTANT: 'default to flag' is about whether to "
        "emit the {entity_id, reason} CHANGE entry. It is NOT about "
        "the patch field. Patch is governed by the rules above: "
        "REQUIRED when the fix is concrete, omit when ambiguous, and "
        "ALWAYS subject to the NO-OP GUARD (don't emit a patch op "
        "whose value equals current_state at that path). Flagging "
        "without a patch is fine -- the agent sees reason and can "
        "ack it. Flagging WITH a no-op patch is waste.\n\n"
        "Examples that REQUIRE a flag: a todo advanced (pending -> "
        "in_progress / in_progress -> completed); active_todo_id "
        "still points to a done item; an agent's current_focus is "
        "empty or stale while the transcript shows active work; a "
        "Task entity's status / phase / step doesn't match what "
        "just occurred; any field that ought to reflect new "
        "activity but doesn't.\n\n"
        "Always emit via the report_state_changes tool. An empty "
        "changes array is ONLY appropriate when you can prove every "
        "followed entity's current_state still exactly matches the "
        "transcript -- i.e. nothing happened that would move any of "
        "them. If you can articulate any divergence at all, flag it.\n\n"
        "## Worked examples\n\n"
        "Example 1 -- todo advanced after a successful tool call:\n\n"
        "Followed entity states:\n"
        '[{"entity_id": "ctx_42", "state_schema_id": "intent_state", '
        '"current_state": {"todos": [{"id": "t1", "text": "Edit foo.py", '
        '"status": "in_progress"}, {"id": "t2", "text": "Run pytest", '
        '"status": "pending"}], "active_todo_id": "t1"}}]\n\n'
        "Intent transcript:\n"
        "[10:01] declare_operation(tool='Edit', args_summary='Edit foo.py: "
        "rename helper')\n"
        "[10:01] Edit tool succeeded -- foo.py modified.\n"
        "[10:02] declare_operation(tool='Bash', args_summary='python -m "
        "pytest tests/unit -q --no-cov')\n\n"
        "Correct output:\n"
        '{"changes": [{"entity_id": "ctx_42", '
        '"schema_id": "intent_state", '
        '"reason": "Transcript shows Edit succeeded and pytest is '
        "the next op; todo t1 should be 'completed' and "
        "active_todo_id should advance to 't2', but current_state "
        "still has t1 'in_progress' and active_todo_id='t1'.\", "
        '"patch": ['
        '{"op": "replace", "path": "/todos/0/status", "value": "completed"}, '
        '{"op": "replace", "path": "/active_todo_id", "value": "t2"}'
        "]}]}\n\n"
        "Example 2 -- agent current_focus is empty while transcript "
        "shows active work:\n\n"
        "Followed entity states:\n"
        '[{"entity_id": "ga_agent", "state_schema_id": "agent_state", '
        '"current_state": {"current_focus": "", "last_active_intent_id": '
        '"intent_audit_fix_3a"}}]\n\n'
        "Intent transcript:\n"
        "[14:22] declare_intent(intent_type='audit_fix_workflow', "
        "summary={what:'migrate scoring.py to vs.query', why:'Tier 2 "
        "VectorStore cleanup'})\n"
        "[14:23] declare_operation(tool='Read', args_summary='Read "
        "scoring.py')\n"
        "[14:24] declare_operation(tool='Edit', args_summary='Edit "
        "scoring.py: replace col.query with vs.query')\n\n"
        "Correct output:\n"
        '{"changes": [{"entity_id": "ga_agent", '
        '"schema_id": "agent_state", '
        '"reason": "Agent is actively reading and editing scoring.py '
        "for a VectorStore migration, but current_focus is empty "
        'string. Focus should reflect the active scoring.py migration.", '
        '"patch": ['
        '{"op": "replace", "path": "/current_focus", '
        '"value": "Tier 2 VectorStore cleanup: migrating scoring.py col.query to vs.query"}'
        "]}]}\n\n"
        "Example 3 -- Task entity phase/step mismatch:\n\n"
        "Followed entity states:\n"
        '[{"entity_id": "task_user_signup_flow", "state_schema_id": '
        '"task_state", "current_state": {"status": "pending", '
        '"phase": "design", "step": "wireframe"}}]\n\n'
        "Intent transcript:\n"
        "[16:00] declare_intent(intent_type='execute', "
        "summary={what:'implement signup form component', why:'Task "
        "user_signup_flow phase=build step=frontend-form'})\n"
        "[16:01] declare_operation(tool='Edit', args_summary='create "
        "SignupForm.tsx with email + password fields')\n"
        "[16:02] Edit tool succeeded -- SignupForm.tsx created.\n\n"
        "Correct output:\n"
        '{"changes": [{"entity_id": "task_user_signup_flow", '
        '"schema_id": "task_state", '
        '"reason": "Task current_state shows phase=\'design\' '
        "step='wireframe' status='pending', but transcript shows agent "
        "has moved to phase='build' step='frontend-form' and just "
        'created SignupForm.tsx. Task state is stale on all three fields.", '
        '"patch": ['
        '{"op": "replace", "path": "/status", "value": "in_progress"}, '
        '{"op": "replace", "path": "/phase", "value": "build"}, '
        '{"op": "replace", "path": "/step", "value": "frontend-form"}'
        "]}]}\n\n"
        "Example 4 -- read-only investigation, no state shift:\n\n"
        "Followed entity states:\n"
        '[{"entity_id": "ctx_88", "state_schema_id": "intent_state", '
        '"current_state": {"todos": [{"id": "t1", "text": '
        '"Investigate cache hit rate", "status": "in_progress"}], '
        '"active_todo_id": "t1"}}]\n\n'
        "Intent transcript:\n"
        "[09:15] declare_intent(intent_type='research', "
        "summary={what:'investigate state_judge cache', why:'cache_read "
        "is always 0'})\n"
        "[09:16] declare_operation(tool='Grep', args_summary='grep "
        "cache_control in injection_gate.py')\n\n"
        "Correct output:\n"
        '{"changes": []}\n\n'
        "The examples above show the standard shape: when the "
        "transcript moves an entity's state, flag with a reason that "
        "names the divergence; when the transcript is purely "
        "read-only and no followed entity moved, emit empty changes.\n\n"
        "Example 5 -- agent state untouched but intent state advanced.\n\n"
        "Followed entity states:\n"
        '[{"entity_id": "ga_agent", "state_schema_id": "agent_state", '
        '"current_state": {"current_focus": "Investigating Haiku '
        'cache behaviour", "active_intent_id": "ctx_12440"}}, '
        '{"entity_id": "ctx_12440", "state_schema_id": "intent_state", '
        '"current_state": {"todos": [{"id": "p1", "text": "Write '
        'cache_probe.py", "status": "in_progress"}, {"id": "p2", '
        '"text": "Run probe + interpret results", "status": "pending"}], '
        '"active_todo_id": "p1"}}]\n\n'
        "Intent transcript:\n"
        "[10:30] declare_operation(tool='Write', "
        "args_summary='write benchmarks/cache_probe.py')\n"
        "[10:30] Write tool succeeded.\n"
        "[10:31] declare_operation(tool='Bash', "
        "args_summary='python benchmarks/cache_probe.py')\n"
        "[10:31] Bash returned cache_create=5059 at 5K-token prefix.\n\n"
        "Correct output:\n"
        '{"changes": [{"entity_id": "ctx_12440", '
        '"schema_id": "intent_state", '
        '"reason": "Write succeeded for cache_probe.py and probe was '
        "run; p1 'Write cache_probe.py' should be 'completed' and "
        "active_todo_id should advance to 'p2' (whose work is "
        'underway).", '
        '"patch": ['
        '{"op": "replace", "path": "/todos/0/status", "value": "completed"}, '
        '{"op": "replace", "path": "/active_todo_id", "value": "p2"}, '
        '{"op": "replace", "path": "/todos/1/status", "value": "in_progress"}'
        "]}]}\n\n"
        "Note: ga_agent is NOT flagged here because current_focus "
        "still accurately describes the work happening; only ctx_12440's "
        "todo list moved.\n\n"
        "Example 6 -- both followed entities moved.\n\n"
        "Followed entity states:\n"
        '[{"entity_id": "ga_agent", "state_schema_id": "agent_state", '
        '"current_state": {"current_focus": "v3.4.5 dormant '
        'feedback_auto module"}}, '
        '{"entity_id": "ctx_99", "state_schema_id": "intent_state", '
        '"current_state": {"todos": [{"id": "s1", "text": "Ship '
        'v3.5.0 atomic rip-out", "status": "in_progress"}], '
        '"active_todo_id": "s1"}}]\n\n'
        "Intent transcript:\n"
        "[14:00] declare_operation(tool='Bash', "
        "args_summary='git push origin main -- v3.5.0 ship 109be13')\n"
        "[14:00] Push succeeded; commit on origin/main.\n"
        "[14:01] Adrian: 'reinstalled, run a manual test'.\n"
        "[14:02] declare_operation(tool='Bash', "
        "args_summary='pip install -e . && python -m pytest -q')\n\n"
        "Correct output:\n"
        '{"changes": ['
        '{"entity_id": "ga_agent", "schema_id": "agent_state", '
        '"reason": "Agent has shifted from dormant-foundation work '
        "to active v3.5.0 ship + manual-test verification; "
        'current_focus is stale.", '
        '"patch": ['
        '{"op": "replace", "path": "/current_focus", '
        '"value": "v3.5.0 atomic rip-out shipped + post-reinstall manual test"}'
        "]}, "
        '{"entity_id": "ctx_99", "schema_id": "intent_state", '
        '"reason": "v3.5.0 ship has landed (push succeeded); s1 should '
        'be completed and a new todo for the manual test should appear.", '
        '"patch": ['
        '{"op": "replace", "path": "/todos/0/status", "value": "completed"}, '
        '{"op": "add", "path": "/todos/-", "value": '
        '{"id": "s2", "text": "Run post-reinstall manual test", '
        '"status": "in_progress"}}, '
        '{"op": "replace", "path": "/active_todo_id", "value": "s2"}'
        "]}]}\n\n"
        "Example 7 -- Task entity's blocker resolved.\n\n"
        "Followed entity states:\n"
        '[{"entity_id": "task_chromadb_removal", "state_schema_id": '
        '"task_state", "current_state": {"status": "blocked", '
        '"blocker": "fastembed parity not verified"}}]\n\n'
        "Intent transcript:\n"
        "[11:00] declare_operation(tool='Bash', "
        "args_summary='python verify_fastembed_parity.py')\n"
        "[11:01] Result: cos_sim=1.000000 across 50 sample texts; parity verified.\n"
        "[11:02] declare_operation(tool='Edit', "
        "args_summary='Edit pyproject.toml: drop chromadb dep, add fastembed>=0.6')\n\n"
        "Correct output:\n"
        '{"changes": [{"entity_id": "task_chromadb_removal", '
        '"schema_id": "task_state", '
        '"reason": "Blocker (fastembed parity) is now verified '
        "(cos_sim=1.0); Edit op shows pyproject migration is "
        'underway. Status should move to in_progress and blocker should clear.", '
        '"patch": ['
        '{"op": "replace", "path": "/status", "value": "in_progress"}, '
        '{"op": "replace", "path": "/blocker", "value": null}'
        "]}]}\n\n"
        "Example 8 -- todo list needs initialization.\n\n"
        "Followed entity states:\n"
        '[{"entity_id": "ctx_47", "state_schema_id": "intent_state", '
        '"current_state": {"todos": []}}]\n\n'
        "Intent transcript:\n"
        "[09:00] declare_intent(intent_type='audit_fix_workflow', "
        "summary={what:'audit + fix mempalace cache; ship as v3.5.4', "
        "why:'cache_read=0 empirically'})\n"
        "[09:01] declare_operation(tool='Read', "
        "args_summary='Read injection_gate.py 172-260 _SYSTEM_PROMPT')\n\n"
        "Correct output:\n"
        '{"changes": [{"entity_id": "ctx_47", '
        '"schema_id": "intent_state", '
        '"reason": "Intent is just declared with a clear multi-step '
        "audit-fix-ship plan but todos list is empty; should be "
        'populated with the plan items so the agent can patch progress.", '
        '"patch": ['
        '{"op": "add", "path": "/todos", "value": ['
        '{"id": "a1", "text": "Read prompt construction", "status": "in_progress"}, '
        '{"id": "a2", "text": "Pad system prompt above cache minimum", "status": "pending"}, '
        '{"id": "a3", "text": "Verify cache_create > 0 empirically", "status": "pending"}, '
        '{"id": "a4", "text": "Run pytest + ship", "status": "pending"}]}, '
        '{"op": "add", "path": "/active_todo_id", "value": "a1"}'
        "]}]}\n\n"
        "Example 9 -- pure read-only investigation, NO state shift.\n\n"
        "Followed entity states:\n"
        '[{"entity_id": "ctx_55", "state_schema_id": "intent_state", '
        '"current_state": {"todos": [{"id": "r1", "text": "Investigate '
        'cache hit rate", "status": "in_progress"}], '
        '"active_todo_id": "r1"}}]\n\n'
        "Intent transcript:\n"
        "[09:15] declare_operation(tool='Grep', "
        "args_summary='grep cache_control in injection_gate.py')\n"
        "[09:16] declare_operation(tool='Read', "
        "args_summary='Read injection_gate.py 680-740 (gate Haiku call)')\n"
        "[09:17] declare_operation(tool='Read', "
        "args_summary='Read injection_gate.py 1700-1800 (judge Haiku call)')\n\n"
        "Correct output:\n"
        '{"changes": []}\n\n'
        "All ops are read-only investigation under r1 'Investigate "
        "cache hit rate' which is exactly what is happening; r1 "
        "stays in_progress; no other entity-state field shifted. "
        "Empty changes is the right answer.\n\n"
        "Example 10 -- agent's recent_findings should accumulate.\n\n"
        "Followed entity states:\n"
        '[{"entity_id": "ga_agent", "state_schema_id": "agent_state", '
        '"current_state": {"current_focus": "Cache root cause", '
        '"recent_findings": []}}]\n\n'
        "Intent transcript:\n"
        "[12:00] declare_operation(tool='Bash', "
        "args_summary='python benchmarks/cache_probe.py')\n"
        "[12:01] Result: cache_create=0 at 1259 + 2498 tok; cache_create=5059 "
        "at 5059 tok. Threshold between 2498 and 5059 -- almost certainly 4096.\n\n"
        "Correct output:\n"
        '{"changes": [{"entity_id": "ga_agent", '
        '"schema_id": "agent_state", '
        '"reason": "Probe established Haiku 4.5 cache minimum is ~4096 '
        'tokens; this is a load-bearing finding that should land in recent_findings.", '
        '"patch": ['
        '{"op": "add", "path": "/recent_findings/-", '
        '"value": "Haiku 4.5 cache minimum empirically pinned at ~4096 tokens (between 2498 and 5059 in probe)"}'
        "]}]}\n\n"
        "Calibration reminders (re-read every call):\n"
        "  - DEFAULT TO FLAG. Under-flagging silently rots the graph; "
        "over-flagging just costs one ack per false positive.\n"
        "  - The patch is part of the answer, not a bonus -- if you "
        "can articulate WHAT the new value should be in 'reason', you "
        "have already done the work, emit it.\n"
        "  - 'replace' for existing fields, 'add' for new fields or "
        "appending to lists via /-, 'remove' to drop. Always include "
        "schema_id when emitting a patch so the agent can validate.\n"
        "  - Read-only investigation that doesn't move any followed "
        "entity is a real and common case -- empty changes is correct "
        "there. Don't fabricate a patch just to feel productive.\n"
        "  - When BOTH ga_agent and the intent_context advance, flag "
        "BOTH; don't pick one.\n"
        "  - History is not stale: a current_state field that records "
        "what was true at a point in time is fine; only flag when the "
        "field is meant to track NOW and NOW has moved.\n"
        "  - Task-state phase/step/blocker shifts are common and "
        "easy to miss -- pay attention to declare_operation results "
        "that resolve a blocker or move a phase forward."
    )

    tool_def = {
        "name": "report_state_changes",
        "description": (
            "Report any state-bearing entities whose current_state is "
            "now stale relative to the transcript. Empty array when "
            "no changes detected."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "changes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity_id": {"type": "string"},
                            "reason": {
                                "type": "string",
                                "description": (
                                    "Brief explanation of why the state "
                                    "is stale -- what the transcript "
                                    "showed vs what current_state holds."
                                ),
                            },
                            # v3.2.9 Phase 3 optional fields
                            # the judge MAY emit when confident enough to
                            # propose a fix. When both are present and
                            # the env flag MEMPALACE_STATE_PROTOCOL is in
                            # an auto-apply mode, intent.py will apply
                            # the patch via record_state_revision with
                            # agent='state_judge'. Omitting either keeps
                            # v0/v2_visibility flag-only semantics for
                            # this change.
                            "schema_id": {
                                "type": "string",
                                "description": (
                                    "Optional state_schemas.STATE_SCHEMAS "
                                    "key for the flagged entity. Required "
                                    "alongside 'patch' for the agent to "
                                    "auto-apply; omit when proposing no "
                                    "fix."
                                ),
                            },
                            "patch": {
                                "type": "array",
                                "items": {"type": "object"},
                                "description": (
                                    "Optional RFC 6902 JSON Patch ops "
                                    "that move current_state to the "
                                    "judged-correct value. Each op is "
                                    "{op, path, value} (or {op, from, "
                                    "path} for move/copy). Omit when "
                                    "uncertain about the exact fix -- "
                                    "the agent will still see the flag "
                                    "via 'reason'."
                                ),
                            },
                        },
                        "required": ["entity_id", "reason"],
                    },
                }
            },
            "required": ["changes"],
        },
    }

    model = gate.model
    # Prompt caching (Adrian directive 2026-05-10 -- state_judge_report
    # cache_read=0 / cache_creation=0 observed across every op-declare,
    # ~1100 fresh input tokens per call). The system_prompt + tool_def
    # are static across the entire process lifetime; the per-call
    # user_content carries the only variation. Marking the LAST element
    # of the tools section with cache_control=ephemeral creates a cache
    # checkpoint that covers everything BEFORE it (system + tools), so
    # the next call within the 5-minute TTL pays only for the user
    # message diff. Anthropic prompt-caching docs 2024-08; cache hit
    # tier costs 10% of base input tokens.
    # Adrian directive 2026-05-11: cache_control belongs on the SYSTEM
    # text block, not on the tools array. Anthropic's prompt cache builds
    # cumulative prefixes in the order tools -> system -> messages, with
    # the breakpoint marking the END of the cached prefix. If the
    # breakpoint is on tools alone, the cached prefix is JUST tools
    # (~250 tok here) which is below the Haiku 2048 minimum, so the
    # cache silently no-ops. Putting the breakpoint on the system text
    # block makes the cached prefix = tools + system (~2200+ tok with
    # the worked examples + dynamic schemas), comfortably over the
    # minimum. Verified offline via .scratch/cache_probe.py: pre-fix
    # showed cache_creation=0 / cache_read=0 across 3 identical calls
    # at input=3204 tok; post-fix should show cache_creation>0 on call
    # 1 and cache_read>0 on calls 2+.
    cached_system = [
        {
            "type": "text",
            "text": system_prompt + schemas_block,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    cached_tools = [{**tool_def, "cache_control": {"type": "ephemeral"}}]
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            system=cached_system,
            tools=cached_tools,
            tool_choice={"type": "tool", "name": "report_state_changes"},
            messages=[{"role": "user", "content": user_content}],
        )
    except Exception as exc:
        log.info("run_state_judge: API call failed: %s", exc)
        return [], None

    # v3.7.17 (Adrian directive 2026-05-17): build per-entity current
    # state map so we can filter no-op patches before they propagate
    # downstream. See _attach_patch_if_changes() docstring.
    _current_by_eid: dict = {}
    for _e in entity_states or []:
        if isinstance(_e, dict):
            _eid_map = (_e.get("entity_id") or "").strip()
            if _eid_map:
                _current_by_eid[_eid_map] = _e.get("current_state") or {}

    changes: list[dict] = []
    try:
        for block in resp.content or []:
            if getattr(block, "type", None) == "tool_use":
                payload = block.input or {}
                raw_changes = payload.get("changes") or []
                for entry in raw_changes:
                    if not isinstance(entry, dict):
                        continue
                    eid = (entry.get("entity_id") or "").strip()
                    reason = (entry.get("reason") or "").strip()
                    if eid and reason:
                        change_out: dict = {"entity_id": eid, "reason": reason}
                        # v3.2.9 Phase 3 forward optional
                        # schema_id + patch when the judge supplied
                        # them. Both fields are passed through verbatim;
                        # intent.py validates schema_id against
                        # STATE_SCHEMAS and applies patch via
                        # record_state_revision when an auto-apply env
                        # mode is active. Empty / missing values are
                        # not attached so downstream truthiness checks
                        # stay clean.
                        _sid = (entry.get("schema_id") or "").strip()
                        if _sid:
                            change_out["schema_id"] = _sid
                        _attach_patch_if_changes(
                            change_out, entry.get("patch"), eid, _current_by_eid
                        )
                        changes.append(change_out)
    except Exception as exc:
        log.info("run_state_judge: response parse failed: %s", exc)
        # Fall through with whatever we collected.

    elapsed_ms = round((_time.perf_counter() - _t0) * 1000, 2)

    if _state_judge_report_disabled():
        report = None
    else:
        usage = getattr(resp, "usage", None)
        # `model` is intentionally NOT in the agent-facing report --
        # it's noise (every call uses gate.model, agent never branches
        # on it). The state_judge_log.jsonl telemetry below DOES carry
        # model so analysis can track per-model judge quality.
        report = {
            "elapsed_ms": elapsed_ms,
            "detected_count": len(changes),
            "tokens": {
                "input": getattr(usage, "input_tokens", 0) or 0,
                "output": getattr(usage, "output_tokens", 0) or 0,
                "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
                "cache_creation": getattr(usage, "cache_creation_input_tokens", 0) or 0,
            },
        }

    # Telemetry: one row per judge call. Best-effort; failures must
    # not change returned values.
    try:
        from datetime import datetime as _dt, timezone as _tz

        from .mcp_server import _telemetry_append_jsonl as _tel

        _tel(
            "state_judge_log.jsonl",
            {
                "ts": _dt.now(_tz.utc).isoformat(timespec="seconds"),
                "agent": agent or "",
                "elapsed_ms": elapsed_ms,
                "detected_count": len(changes),
                "model": model,
                "tokens": report["tokens"] if report else {},
            },
        )
    except Exception:
        pass

    return changes, report
