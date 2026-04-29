#!/usr/bin/env python3
"""
MemPalace MCP Server -- read/write palace access for Claude Code
================================================================
Install: claude mcp add mempalace -- python -m mempalace.mcp_server [--palace /path/to/palace]

Tools (read):
  mempalace_kg_search       -- unified 3-channel search over memories AND entities
  mempalace_kg_query        -- structured edge lookup by exact entity name
  mempalace_kg_stats        -- knowledge graph overview: entities, triples, relationship types

Tools (write):
  mempalace_kg_declare_entity -- declare an entity (kind=entity/class/predicate/literal/record)
                                memory memories are first-class entities
  mempalace_kg_delete_entity -- soft-delete an entity or memory (invalidates all edges)
  mempalace_resolve_conflicts -- resolve contradictions, duplicates, merge candidates
"""

import argparse
import os
import sys
import json
import logging
import faulthandler
from datetime import datetime, timezone
from pathlib import Path

# ── C-level crash visibility ────────────────────────────────────────────
# Chroma's HNSW vector segment runs in a C extension on Windows that can
# segfault (STATUS_ACCESS_VIOLATION 0xC0000005) when the on-disk index is
# corrupted or when a hyphen-id migration leaves the collection in an
# inconsistent state. A C-level access violation BYPASSES Python's
# exception machinery -- try/except wrappers around tool dispatch (see
# tools/call branch) catch nothing, the process is killed by the OS,
# stdio drops, and the MCP client sees only "Connection closed -32000"
# with zero diagnostic detail.
#
# `faulthandler.enable()` registers Python's native C-fault dumper:
# on SIGSEGV / access violation it walks the Python stack frames at the
# crash site and writes them to stderr BEFORE the process terminates.
# That gives us a real stack trace in stderr (which Claude Code captures
# to its MCP log) instead of a silent kill, turning a 90-minute opaque-
# debug session into a one-line "ah, it's HNSW" diagnosis. Cost: zero
# runtime overhead in the happy path; only fires on actual signals.
#
# Activated below after all module imports complete (placement avoids
# ruff E402 for subsequent imports). Imports themselves are protected
# by Python's normal import-error path; faulthandler is for runtime
# C-level crashes during tool calls.
#
# Recovery when this fires for HNSW: `mempalace doctor --rebuild` to
# re-extract memories from sqlite and re-index into a fresh collection.

# ── Prevent `python -m` double-load ────────────────────────────────────
#
# When invoked as ``python -m mempalace.mcp_server`` this module runs under
# the name ``__main__``. Any dependency that later does
# ``from mempalace import mcp_server`` or ``import mempalace.mcp_server``
# would find the canonical dotted name MISSING from ``sys.modules`` and
# Python would execute this file a SECOND time under that dotted name --
# producing a distinct module object with its OWN ``_STATE = ServerState()``
# at line ~71. The two copies silently diverge, and writes to
# ``_STATE.session_id`` on one copy are invisible to the other. This was
# the 2026-04-19 session-state deadlock: handle_request
# set sid on __main__'s _STATE; intent._persist_active_intent read sid
# from mempalace.mcp_server's _STATE (empty); file never persisted;
# hook denied every subsequent tool call.
#
# Fix: alias ``__main__`` into ``sys.modules["mempalace.mcp_server"]``
# BEFORE any dependent import runs. Future dotted-name imports hit the
# cache and return this same module. Only one ``_STATE`` can exist.
if __name__ == "__main__":
    sys.modules["mempalace.mcp_server"] = sys.modules["__main__"]

from .config import MempalaceConfig, sanitize_content, sanitize_name  # noqa: F401
from .version import __version__
import chromadb

from .knowledge_graph import KnowledgeGraph
from . import intent
from .server_state import ServerState
from .scoring import hybrid_score as _hybrid_score_fn

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger("mempalace_mcp")

# Activate the C-level fault dumper described in the import-section
# comment above. Must run before any Chroma collection access.
faulthandler.enable()


def _parse_args():
    parser = argparse.ArgumentParser(description="MemPalace MCP Server")
    parser.add_argument(
        "--palace",
        metavar="PATH",
        help="Path to the palace directory (overrides config file and env var)",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.debug("Ignoring unknown args: %s", unknown)
    return args


_args = _parse_args()

if _args.palace:
    os.environ["MEMPALACE_PALACE_PATH"] = os.path.abspath(_args.palace)

_bootstrap_config = MempalaceConfig()

# NOTE: Hint file (~/.mempalace/hook_state/active_palace.txt) is written
# in main() at actual server startup, NOT here. Writing at module-import
# time was a bug: every pytest run / CLI invocation that imports this
# module would clobber the real hint with a temp palace, causing hook
# subprocesses to open empty collections. (2026-04-20 incident.)
# BF1: KG file always lives inside the palace dir now. KnowledgeGraph()
# with no arg derives the path from MempalaceConfig().palace_path and
# migrates any legacy ~/.mempalace/knowledge_graph.sqlite3 in place on
# first run. Whether --palace was passed or not, the same code path runs.
_bootstrap_kg = KnowledgeGraph()

_STATE = ServerState(config=_bootstrap_config, kg=_bootstrap_kg)
del _bootstrap_config, _bootstrap_kg


# ==================== ONTOLOGY SEEDERS (re-exported) ====================
# Adrian's directive 2026-04-26: ontology seeding lives in mempalace.seed,
# a dedicated module that the MCP transport layer just calls. The names
# are re-exported below so existing test imports
# (``from mempalace.mcp_server import _ensure_operation_ontology``) keep
# working without churn. New code should import from ``mempalace.seed``
# directly.
from .seed import (  # noqa: E402  (kept after _STATE bootstrap on purpose)
    _ensure_operation_ontology,  # noqa: F401  (re-exported for tests)
    _ensure_task_ontology,  # noqa: F401  (re-exported for tests)
    _ensure_user_intent_ontology,  # noqa: F401  (re-exported for tests)
    seed_all,
)


# Run all seeders. seed_all itself honors MEMPALACE_SKIP_SEED and
# logs+swallows failures per-seeder so a broken ontology block cannot
# wedge startup. See mempalace/seed.py for the bodies.
seed_all(_STATE.kg)


# Wire intent module to this module so it can reach _STATE and other helpers.
intent.init(sys.modules[__name__])


# ==================== WRITE-AHEAD LOG ====================
# Every write operation is logged to a JSONL file before execution.
# This provides an audit trail for detecting memory poisoning and
# enables review/rollback of writes from external or untrusted sources.

_WAL_DIR = Path(os.path.expanduser("~/.mempalace/wal"))
_WAL_DIR.mkdir(parents=True, exist_ok=True)
try:
    _WAL_DIR.chmod(0o700)
except (OSError, NotImplementedError):
    pass
_WAL_FILE = _WAL_DIR / "write_log.jsonl"


def _wal_log(operation: str, params: dict, result: dict = None):
    """Append a write operation to the write-ahead log."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "operation": operation,
        "params": params,
        "result": result,
    }
    try:
        with open(_WAL_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        try:
            _WAL_FILE.chmod(0o600)
        except (OSError, NotImplementedError):
            pass
    except Exception as e:
        logger.error(f"WAL write failed: {e}")


def _get_client():
    """Return a singleton ChromaDB PersistentClient."""
    if _STATE.client_cache is None:
        _STATE.client_cache = chromadb.PersistentClient(path=_STATE.config.palace_path)
    return _STATE.client_cache


# Cosine is the ONLY supported distance metric across mempalace.
# MaxSim/ColBERT math assumes 1-distance = cosine_similarity and our
# retrieval scoring depends on [-1, +1] similarity semantics. Explicitly
# pinning the hnsw:space prevents a future ChromaDB default change (or
# a collection created by an older tool) from silently shifting math
# underneath us.
_CHROMA_METADATA = {"hnsw:space": "cosine"}


def _get_collection(create=False):
    """Return the ChromaDB collection, caching the client between calls."""
    try:
        client = _get_client()
        if create:
            _STATE.collection_cache = client.get_or_create_collection(
                _STATE.config.collection_name, metadata=_CHROMA_METADATA
            )
        elif _STATE.collection_cache is None:
            _STATE.collection_cache = client.get_collection(_STATE.config.collection_name)
        return _STATE.collection_cache
    except Exception:
        return None


def _no_palace():
    return {
        "error": "No palace found",
        "hint": "Run: mempalace init <dir> && mempalace mine <dir>",
    }


# ==================== READ TOOLS ====================


PALACE_PROTOCOL = """MemPalace Protocol -- rules for your behavior. The system enforces
structure (slot validation, tool permissions, conflict detection, feedback
coverage) and teaches through error messages; your job is to do the right
thing and let the errors tune the rest.

ON START:
  Call mempalace_wake_up. Read this protocol, the text (identity + rules),
  and declared (entities, predicates, intent types with their tools).

BEFORE ACTING ON ANY FACT:
  kg_query for exact entity-ID lookups when you know the name.
  kg_search for fuzzy discovery -- searches memories (prose) and entities
  (KG nodes) in one call, with graph expansion. Never guess.

WHEN HITTING A BLOCKER:
  Search mempalace first -- gotchas, lessons-learned, past executions on
  similar problems. Only report a blocker to the user if memory has no
  answer. Persist new solutions (record + KG triple) so future sessions
  find them.

WHEN FILING RECORDS:
  - Pick the most accurate predicate from the declared predicates list.
  - Extract at least one KG triple from the content (twin pattern).
    Record alone = semantic search only; triple = fast entity lookup.
  - Duplicate / contradiction detection is automatic. If conflicts are
    returned, resolve via mempalace_resolve_conflicts:
      invalidate (old is stale),
      merge (combine -- READ BOTH in full, provide merged_content that
        preserves ALL unique info from each side),
      keep (both are valid),
      skip (undo the new item).

SUMMARY DISCIPLINE (records, entities, edges, contexts -- Adrian's
design lock 2026-04-25):
  EVERY summary / description / statement is a structured dict:

      {"what": "<noun phrase>",
       "why":  "<purpose / role / claim>",
       "scope": "<temporal/domain qualifier, optional>"}

  Validation is field-level + length-bounded -- no regex, no
  auto-derive. WHAT (≥5 chars), WHY (≥15 chars), SCOPE (≤100 chars
  optional). Rendered prose form ≤280 chars.

  The dict lives EVERYWHERE summary lived before:
    - records (kg_declare_entity kind='record', kg_add for records)
    - entities (kg_declare_entity, kg_update_entity description)
    - edges (kg_add statement -- same shape via validate_statement)
    - contexts (context.summary -- declare_intent, declare_operation,
      kg_declare_entity, kg_add, kg_update_entity all carry it)

  Standalone summary parameters were retired -- summary is INSIDE
  context for every context-taking tool. queries[0] auto-derive is
  retired. "File: <path>" auto-stub is retired. Strings on writes
  are rejected with a migration message.

  Good shape examples:
    {"what": "InjectionGate",
     "why": "filters retrieved memories before injection via "
            "Haiku tool-use; emits quality flags",
     "scope": "one instance per palace process"}
    {"what": "intent.py",
     "why": "orchestrates declare_intent slot validation and "
            "finalize_intent feedback coverage; central glue "
            "between hooks and the gate"}

  Literature: Anthropic Contextual Retrieval 2024 (the WHAT+WHY+role
  context block lifted retrieval F1 35-50%); ColBERT 2020; SciFact
  2020; MemGPT/Letta 2023-24. The dict shape decouples STORAGE
  (validation-friendly fields) from EMBEDDING (rendered prose) so
  each layer optimises independently.

DECLARING INTENT / OPERATION / SEARCH:
  - Declare intent first (mempalace_declare_intent). Check
    declared.intent_types for available types. Blocked-tool errors
    teach how to create or switch intent types. Intent types are
    kind=class; executions are kind=entity with is_a.
  - Before EVERY non-carve-out tool call (Read, Grep, Glob, Bash, Edit,
    Write, WebFetch, WebSearch, etc.) call mempalace_declare_operation
    with a cue that reflects your ACTUAL intention -- not the shape of
    the tool call. Queries are 2-5 natural-language perspectives;
    keywords are 2-5 exact domain terms. The hook blocks any call
    without a matching cue. Parallel batches: emit all declares + real
    tool calls in the same assistant message.
  - Carve-outs that skip declare_operation: mempalace_* tools,
    TodoWrite, Skill, Agent, ToolSearch, AskUserQuestion, Task*,
    ExitPlanMode.
  - context.entities is MANDATORY (1-10) on every declare_intent /
    declare_operation / kg_search. List files you'll edit, services
    and concepts you're reasoning about, agents involved. May overlap
    with slot values; also include entities that don't fit slots.
    Entities feed the link-author background pipeline -- zero entities,
    no graph growth.

USER-INTENT TIER (Slice B, 2026-04-26):
  Above the activity (declare_intent) tier sits a user-intent tier that
  binds every activity-intent to the user message that provoked it.
  Three pieces, in order:

  1. UserPromptSubmit hook persists each user prompt into a pending
     queue and emits an additionalContext block that names the pending
     user_message ids and points you at mempalace_declare_user_intents.

  2. Call mempalace_declare_user_intents BEFORE the first
     declare_intent / declare_operation of the turn. Pass a list of
     user-intent contexts that COVER every pending user_message id:
       - Each context: {context: {queries, keywords, entities, summary},
         user_message_ids: [...], no_intent: false}
       - Union of user_message_ids across contexts must equal the
         pending set.
       - Set no_intent=true when the prompt was conversational only
         (greeting / clarification ack) and no activity-intent will run.
     The tool mints user-context entities (kind='context'), writes
     fulfills_user_message edges to each minted user_message record,
     and surfaces top-K memories per context. PreToolUse blocks every
     non-carve-out tool call until pending is drained.

  3. Pass cause_id on declare_intent for every activity-intent that
     fulfils a user-context. Accepts either:
       - the user-context id returned by declare_user_intents
         (`contexts[*].ctx_id`), OR
       - a Task entity id (kind='entity', is_a Task) for paperclip /
         scheduled flows where the parent cause is a long-running task.
     The activity-intent's context gets a caused_by edge to the cause;
     finalize_intent writes a caused_by edge from the execution entity
     too.

  FIRST-RATER COVERAGE RULE: when cause_kind=='user_context', the FIRST
  agent intent that finalizes against that user-context covers its
  surfaced memories in memory_feedback (full coverage required).
  Subsequent intents declared with the same cause_id INHERIT the prior
  ratings -- finalize subtracts the user-context-surfaced memory ids
  from required coverage. The signal is rated_user_contexts, a
  session-scoped set keyed by cause_id; in-memory only, scoped to the
  MCP server process.

  CARVE-OUTS: AskUserQuestion + every mempalace_* tool are exempt from
  the pending-block (clarification turns and mempalace bookkeeping run
  freely). To disable the user-intent tier entirely set
  MEMPALACE_USER_INTENT_DISABLED=1; to disable just the PreToolUse
  block set MEMPALACE_USER_INTENT_BLOCK_DISABLED=1.

WHEN RECEIVING INJECTED MEMORIES:
  - Every memory surfaced (by declare_intent, declare_operation, or
    kg_search) lands in accessed_memory_ids and REQUIRES feedback at
    finalize_intent: 100% coverage, 1-5 relevance, reason string.
    Finalize rejects without coverage.
  - memory_feedback shape: list of groups
    [{context_id: <ctx_id>, feedback: [entries]}, ...]. Each group
    attributes its ratings back to the context that surfaced the
    memories -- this is what future retrieval reads. (Dict shape was
    retired 2026-04-24; MCP clients silently dropped it.)
  - Relevance calibration: 3 = related context (default when unsure).
    4-5 = changed a decision / load-bearing. 1-2 = noise / misleading.
    If >50% of your ratings are >=4, demote -- inflating dampens the
    signal you're giving future-you.
  - Memories return in short form. For the full content, call
    kg_query(entity=<id>).
  - Zero hits for a cue is success -- proceed. Low-relevance hits are
    expected and useful as negative feedback, not errors.

BEFORE SWITCHING INTENTS:
  Call finalize_intent BEFORE declaring a new intent. Captures what you
  did (execution entity + trace), what you learned (gotchas, learnings),
  outcome (success / partial / failed / abandoned). If you forget,
  declare_intent auto-finalizes with outcome='abandoned' -- lower-quality
  memory. Explicit finalization is always better.

INTENT TYPES:
  - New intent types use kind='class'.
  - If 3+ similar executions exist on a broad type, declaration fails --
    create a specific intent type for the recurring action.

COMPLETION DISCIPLINE:
  There is no "wrapping the session" while pending work remains. Do NOT
  offer to continue "in a later session", summarize and stop, or ask
  "should I keep going?". If TodoWrite has pending items, DO THEM. The
  user does not care about session boundaries or context limits.
  Only pause when a tool call genuinely needs the user's answer
  (ambiguous requirement, missing credential, destructive action
  requiring consent). Everything else is your job to finish.

AT SESSION END (only when all pending work is actually done):
  1. finalize_intent on the active intent.
  2. Persist new knowledge via the twin pattern:
     - Decisions, rules, discoveries, gotchas -> record + KG triple(s).
     - Changed facts -> kg_invalidate old + kg_add new.
     - New entities -> kg_declare_entity.
     Don't just diary them -- the diary is a temporal log; KG + records
     are durable knowledge future sessions can query structurally.
  3. diary_write -- concise, non-redundant, prose.
     - Delta only: what changed since the last entry.
     - Focus: decisions made with user, big-picture status, pending items.
     - Do NOT repeat commits / gotchas / learnings (already in intent
       results). The diary is high-level narrative, not a detailed log.

BACKGROUND (FYI, no action required):
  A link-author jury runs out-of-session over entity co-occurrence
  evidence and authors new edges autonomously. If a bad edge lands,
  kg_invalidate it normally. Retrieval fuses cosine / graph / keyword /
  learned-context channels via weighted RRF; rated_useful and
  rated_irrelevant edges you write in memory_feedback are the learning
  signal that shapes what surfaces next time."""


def _hybrid_score(
    similarity: float,
    importance: float,
    date_added_iso: str,
    agent_match: bool = False,
    last_relevant_iso: str = None,
    session_match: bool = False,
    intent_type_match: bool = False,
) -> float:
    """Hybrid ranking score for search results. Delegates to scoring.hybrid_score."""
    return _hybrid_score_fn(
        similarity=similarity,
        importance=importance,
        date_iso=date_added_iso,
        agent_match=agent_match,
        last_relevant_iso=last_relevant_iso,
        relevance_feedback=0,
        mode="search",
        session_match=session_match,
        intent_type_match=intent_type_match,
    )


# tool_search removed: merged into tool_kg_search, which now searches
# BOTH memories and entities in a single cross-collection RRF. The "palace is
# a graph" unification -- one search tool over all memory.


# tool_check_duplicate removed: dedup is now embedded in
# _add_memory_internal (called by kg_declare_entity kind='record').
# The standalone tool was already removed from the MCP registry;
# this deletes the orphaned function.


# ==================== WRITE TOOLS ====================


VALID_CONTENT_TYPES = {
    "fact",
    "event",
    "discovery",
    "preference",
    "advice",
    "diary",
}


def _validate_importance(importance):
    """Coerce and validate an importance value (1-5). Returns int or raises ValueError."""
    if importance is None:
        return None
    try:
        n = int(importance)
    except (TypeError, ValueError):
        raise ValueError(
            f"importance must be an integer 1-5 (got {importance!r}). "
            f"Importance scale: 5=critical, 4=canonical, 3=default, 2=low, 1=junk."
        )
    if n < 1 or n > 5:
        raise ValueError(
            f"importance must be between 1 and 5 (got {n}). "
            f"Importance scale: 5=critical unmissable, 4=canonical rules/cookbooks, "
            f"3=historical events (default), 2=low priority, 1=junk/quarantine."
        )
    return n


VALID_KINDS = {
    "entity",  # a concrete individual thing
    "predicate",  # a relationship type
    "class",  # a category/domain-type definition
    "literal",  # a raw value (string, integer, timestamp, URL, path)
    "record",  # a stored prose record -- full text in ChromaDB, metadata in SQLite
    "operation",  # a tool invocation. Carries tool + args_summary +
    #              context_id; attached to an intent execution via
    #              executed_op and to the op-context via performed_well /
    #              performed_poorly. As of 2026-04-27 (commit b905373)
    #              args_summary is mandatory parametrized-core form, so
    #              ops embed in Channel A via the single-description path
    #              ('{tool} op: {args_summary}'). Multi-view sync is still
    #              skipped -- the parametrized fingerprint IS the view.
    #              Cf. arXiv 2512.18950 + Leontiev 1981 AAO.
}

# kind='memory' is GONE. The one-pass migration at startup
# rewrites existing metadata; the alias is removed so callers
# get a clear error instead of silent normalization. "memory" is the
# palace-level concept; "record" is the record-type kind.


def _validate_kind(kind):
    """Validate entity kind (ontological role). REQUIRED -- no default."""
    if kind is None:
        raise ValueError(
            "kind is REQUIRED. Must be one of: 'entity' (concrete thing), "
            "'predicate' (relationship type), 'class' (category definition), "
            "'literal' (raw value), 'record' (prose record -- requires "
            "slug + content + added_by), or 'operation' (tool invocation -- "
            "graph-only, never embedded). You must explicitly choose the "
            "ontological role."
        )
    if kind == "memory":
        raise ValueError(
            "kind='memory' was renamed to 'record' in P6.2. Use kind='record'. "
            "The word 'memory' is reserved for the palace-level concept."
        )
    if kind not in VALID_KINDS:
        raise ValueError(
            f"kind must be one of {sorted(VALID_KINDS)} (got {kind!r}). "
            f"entity=concrete thing (default), predicate=relationship type, "
            f"class=category/type definition, literal=raw value, "
            f"record=prose record with slug + content + added_by. "
            f"Domain types (system, person, project, etc.) are NOT kinds -- "
            f"they are class-kind entities linked via is_a edges."
        )
    return kind


def _validate_content_type(content_type):
    """Validate a content_type value. Returns string or raises ValueError."""
    if content_type is None:
        return None
    if content_type not in VALID_CONTENT_TYPES:
        raise ValueError(
            f"content_type must be one of {sorted(VALID_CONTENT_TYPES)} (got {content_type!r}). "
            f"fact=stable truths, event=things that happened, "
            f"discovery=lessons learned, preference=user rules, "
            f"advice=how-to guides, diary=chronological journal."
        )
    return content_type


def _slugify(text: str, max_length: int = 50) -> str:
    """Canonical slug for memory/diary/entity identifiers.

    Single source of truth for identifier normalization. Delegates to
    ``normalize_entity_name`` so every stored identifier uses the same
    separator convention (underscore) across Chroma IDs, SQLite entity
    IDs, and KG triple subjects/objects. The previous implementation
    emitted hyphens, which collided with every downstream callsite that
    re-normalized to underscores and then looked up the hyphenated ID --
    yielding silent Chroma misses (see A7, 9ecf234). DRY it here, fix it
    forever.
    """
    from .knowledge_graph import normalize_entity_name

    slug = normalize_entity_name(text)
    if slug == "unknown":
        return ""
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("_")
    return slug


def _past_resolution_hint(conflicts: list) -> str:
    """B1b: build a short suffix for conflicts_prompt summarizing prior decisions.

    When any conflict in the batch carries a `past_resolution` field (attached
    at detection time by `get_past_conflict_resolution`), render a compact
    hint so the agent sees their prior decisions without having to read the
    full conflicts array. Empty string when no past decisions exist \u2014 no
    change to the prompt in that case.
    """
    past_lines = []
    for c in conflicts or []:
        past = c.get("past_resolution")
        if not past:
            continue
        when = (past.get("when") or "")[:10]  # YYYY-MM-DD
        reason = (past.get("reason") or "").strip()
        if len(reason) > 100:
            reason = reason[:97] + "..."
        action = past.get("action") or "?"
        past_lines.append(
            f"  \u2022 {c.get('id', '?')}: last time you chose '{action}' ({when}) \u2014 {reason}"
        )
    if not past_lines:
        return ""
    return (
        "\n\nPast resolutions on matching conflicts (use as guidance, still your call):\n"
        + "\n".join(past_lines)
    )


# Summary-first indexing. Summary is ALWAYS required on every record
# (no length threshold) -- grounded in Anthropic's Contextual Retrieval
# (https://www.anthropic.com/news/contextual-retrieval, 2024 -- prepending
# explanatory context to each chunk before embedding cut retrieval
# failure rate by 49%) and Chen et al. "Dense X Retrieval"
# (arXiv:2312.06648, 2023 -- proposition-granularity embeddings outperform
# passage-granularity on open-domain QA). Neither paper threshold-gates by
# length; every chunk gets the treatment. For long content the summary
# is a distillation of WHAT/WHY; for short content the summary should
# REPHRASE the same fact from a different angle (different keywords /
# framing) so the summary+content pair produces TWO distinct cosine
# views -- genuine retrieval information gain, not redundancy. The
# summary is prepended to content before embedding (single CR vector)
# AND stored verbatim in metadata for injection-time display previews.
_RECORD_SUMMARY_MAX_LEN = 280


def _add_memory_internal(  # noqa: C901
    content: str,
    slug: str,
    added_by: str = None,
    content_type: str = None,
    importance: int = None,
    entity: str = None,
    predicate: str = "described_by",
    context: dict = None,  # Context fingerprint for keywords + creation_context_id
    source_file: str = None,
    summary=None,  # dict {what, why, scope?} -- see validate_summary
):
    """File verbatim content as a flat record. Checks for duplicates first.

    ALL classification params are REQUIRED (no lazy defaults):
        slug: short human-readable identifier -- REQUIRED. Used as part of the
              record ID. Must be unique per agent. Examples:
              'intent-pre-activation-issues', 'db-credentials', 'ga-identity'.
        content_type: one of fact, event, discovery, preference, advice, diary.
        importance: integer 1-5 -- REQUIRED. 5=critical, 4=canonical,
                    3=default, 2=low, 1=junk.
        entity: entity name (or comma-separated list) -- REQUIRED. Links this record
                to an entity in the KG. If not provided, the record is unlinked.
        predicate: relationship type for the entity→record link. Default: described_by.
        summary: ≤280-char distillation -- REQUIRED on every record (no
              length threshold). For long content the summary distills the
              WHAT/WHY; for short content the summary should REPHRASE the
              same fact from a different angle (different keywords / framing)
              so the summary+content pair yields two distinct cosine views of
              the same semantic (Anthropic Contextual Retrieval, 2024; Chen
              et al. Dense X Retrieval, arXiv:2312.06648, 2023 -- neither
              paper gates by length). The summary is prepended to content
              before embedding (single CR vector) AND stored verbatim in
              metadata so injections display the summary rather than a
              truncated content preview.

    Note: date_added is always set to the current time. Diary records
    (via diary_write) are exempt from the entity/slug requirement.
    """
    try:
        content = sanitize_content(content)
        content_type = _validate_content_type(content_type)
        importance = _validate_importance(importance)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    # ── Summary-first gate: dict-only, ALWAYS required ──
    # Every record carries a structured summary {what, why, scope?}.
    # Validation reduces to "fields present, non-empty, length-bounded";
    # no regex, no auto-derivation. Adrian's design lock 2026-04-25.
    # The dict is normalised by coerce_summary_for_persist; we keep BOTH
    # the dict (for write-time audits and the gardener's field-level
    # patches) AND the rendered prose (for embedding-as-one-cosine-view,
    # per Anthropic Contextual Retrieval 2024 / Chen et al. Dense X
    # Retrieval 2023). Strings are no longer accepted -- see
    # validate_summary for the migration message.
    if summary is None:
        return {
            "success": False,
            "error": (
                "`summary` is required on every record. Pass a dict "
                "{'what': '<noun phrase>', 'why': '<purpose / role / "
                "claim>', 'scope': '<temporal/domain qualifier>'?}. "
                "No auto-derivation; the writer is the only one who "
                "knows the WHAT and WHY of this record."
            ),
        }
    # Strict dict-only contract: strings reach validate_summary which
    # raises with the migration message. All fixtures provide explicit
    # dicts; no test-mode escape hatch (Adrian's design lock 2026-04-25).
    try:
        from .knowledge_graph import (
            SummaryStructureRequired,
            coerce_summary_for_persist,
            serialize_summary_for_embedding,
        )

        summary_dict = coerce_summary_for_persist(
            summary,
            context_for_error="kg_add(kind=record).summary",
        )
    except SummaryStructureRequired as _vs_err:
        return {"success": False, "error": str(_vs_err)}
    summary_prose = serialize_summary_for_embedding(summary_dict)
    if len(summary_prose) > _RECORD_SUMMARY_MAX_LEN:
        return {
            "success": False,
            "error": (
                f"`summary` rendered prose is {len(summary_prose)} chars; "
                f"maximum is {_RECORD_SUMMARY_MAX_LEN}. Trim 'why' or "
                f"'scope' so the prose form fits the embedding budget."
            ),
        }
    # Downstream code reads `summary` as the prose string used for
    # display + embedding; the dict is preserved in `summary_dict` for
    # field-level metadata.
    summary = summary_prose

    # Validate added_by: REQUIRED, must be a declared agent (is_a agent)
    if not added_by:
        return {
            "success": False,
            "error": "added_by is required. Pass your agent entity name (e.g., 'ga_agent', 'technical_lead_agent').",
        }
    try:
        from .knowledge_graph import normalize_entity_name

        agent_id = normalize_entity_name(added_by)
        if _STATE.kg:
            agent_edges = _STATE.kg.query_entity(agent_id, direction="outgoing")
            is_agent = any(
                e["predicate"] == "is_a" and e["object"] == "agent" and e.get("current", True)
                for e in agent_edges
            )
            if not is_agent:
                return {
                    "success": False,
                    "error": (
                        f"added_by '{added_by}' is not a declared agent (missing is_a agent edge). "
                        f"Declare it as an agent first: "
                        f"kg_declare_entity(name='{added_by}', kind='entity', ...) + "
                        f"kg_add(subject='{added_by}', predicate='is_a', object='agent')"
                    ),
                }
    except Exception:
        pass  # graceful fallback if KG unavailable

    if not slug or not slug.strip():
        return {
            "success": False,
            "error": "slug is required. Provide a short human-readable identifier (e.g. 'intent-pre-activation-issues').",
        }

    normalized_slug = _slugify(slug)
    if not normalized_slug:
        return {
            "success": False,
            "error": f"slug '{slug}' normalizes to empty. Use alphanumeric words separated by underscores or hyphens.",
        }

    col = _get_collection(create=True)
    if not col:
        return _no_palace()

    from .knowledge_graph import normalize_entity_name as _norm_eid

    agent_slug = _norm_eid(added_by) if added_by else "unknown"
    memory_id = f"record_{agent_slug}_{normalized_slug}"

    # Uniqueness check -- slug collision returns existing record info
    try:
        existing = col.get(ids=[memory_id], include=["documents", "metadatas"])
        if existing and existing["ids"]:
            return {
                "success": False,
                "error": f"Slug '{normalized_slug}' already exists for agent {added_by}.",
                "existing_memory": {
                    "memory_id": memory_id,
                    "content_preview": (existing["documents"][0] or "")[:200],
                    "metadata": existing["metadatas"][0],
                },
                "hint": "Choose a different slug, or use kg_update_entity(entity=memory_id, ...) to modify the existing record's metadata.",
            }
    except Exception:
        pass

    _wal_log(
        "add_memory",
        {
            "memory_id": memory_id,
            "added_by": added_by,
            "content_length": len(content),
            "content_preview": content[:200],
            "content_type": content_type,
            "importance": importance,
        },
    )

    now_iso = datetime.now().isoformat()
    meta = {
        "source_file": source_file or "",
        "chunk_index": 0,
        "added_by": added_by,
        "filed_at": now_iso,
        "date_added": now_iso,
        "last_relevant_at": now_iso,
    }
    if content_type is not None:
        meta["content_type"] = content_type
    if importance is not None:
        meta["importance"] = importance
    # Summary-first indexing: store verbatim so injections can display the
    # summary instead of truncating content. Chroma metadata only accepts
    # strings/primitives, so write empty string (not None) when absent.
    meta["summary"] = summary or ""

    # provenance auto-injection. Every record carries session_id
    # and intent_id from the active session/intent. System-injected, not
    # agent-provided. Queryable via Chroma where filters for session-
    # scoped retrieval; also written to SQLite (migration 009).
    if _STATE.session_id:
        meta["session_id"] = _STATE.session_id
    if _STATE.active_intent and isinstance(_STATE.active_intent, dict):
        meta["intent_id"] = _STATE.active_intent.get("intent_id", "")

    # Contextual Retrieval embedding: prepend the summary to the content
    # before embedding so the vector carries short-form semantic anchors
    # AND the long-form specifics
    # (https://www.anthropic.com/news/contextual-retrieval, 2024:
    # "prepends chunk-specific explanatory context to each chunk before
    # embedding" -- reported ~49% reduction in retrieval failure). The
    # stored document keeps the prepended shape so col.get() round-trips
    # don't lose the context; `metadata.summary` remains the canonical
    # short form for display.
    embed_doc = f"{summary}\n\n{content}" if summary else content
    # Defensive type check -- chroma's default ONNX embedder forwards
    # documents straight to the HuggingFace tokenizer, which raises the
    # opaque "TextInputSequence must be str in upsert" if any element is
    # not a str. sanitize_content guards content upstream and the summary
    # gate guarantees a str summary, but guard again here so a future
    # caller passing a list/dict by mistake gets a clear typed error
    # instead of the tokenizer-internal message.
    if not isinstance(embed_doc, str):
        return {
            "success": False,
            "error": (
                f"internal: embed_doc must be str (got "
                f"{type(embed_doc).__name__}). content type="
                f"{type(content).__name__}, summary type="
                f"{type(summary).__name__}."
            ),
        }
    # Bug 1 fix 2026-04-28: pre-truncate embed_doc to a token-safe budget
    # before col.upsert. Chroma's default ONNX embedder is all-MiniLM-L6-v2
    # which has a hard 256-token ceiling; sequences over that crash inside
    # the HuggingFace tokenizer with the opaque
    # 'TextInputSequence must be str in upsert' error. The character
    # gate at sanitize_content allows up to 100_000 chars, vastly above
    # the embedding token budget. ~4 chars/token is the empirical rule;
    # 240 tokens × 4 = 960 chars; we use 1800 as a soft ceiling that
    # leaves headroom for multi-byte UTF-8 sequences and tokenizer
    # padding. The FULL content stays in metadata.summary +
    # metadata.content (separate fields, no embedding pass), so callers
    # that need the original payload still get it via col.get(); only
    # the document-side embedding sees the truncated form. Truncation
    # is logged, not silent, so we can spot frequent overflow and
    # consider chunked-multi-embed if it becomes common.
    _EMBED_DOC_MAX_CHARS = 1800
    if len(embed_doc) > _EMBED_DOC_MAX_CHARS:
        logger.warning(
            "embed_doc truncated for col.upsert: id=%r orig_len=%d -> %d "
            "(token-safe ceiling for all-MiniLM-L6-v2 256-token cap; "
            "full content remains in metadata)",
            memory_id,
            len(embed_doc),
            _EMBED_DOC_MAX_CHARS,
        )
        embed_doc = embed_doc[: _EMBED_DOC_MAX_CHARS - 1].rstrip() + "…"
    # Dedicated upsert try so a TextInputSequence failure (HF tokenizer)
    # surfaces precise diagnostic fields instead of a bare error message.
    # The intermittent 'TextInputSequence must be str in upsert' we kept
    # seeing in the live MCP server -- despite embed_doc already being
    # str-typed here -- implies the chroma embedder was being fed something
    # unexpected downstream. Re-raise with id + types + lens so the next
    # occurrence in a running server tells us which record triggered it.
    try:
        col.upsert(
            ids=[memory_id],
            documents=[embed_doc],
            metadatas=[meta],
        )
        logger.info(f"Filed record: {memory_id} content_type={content_type} imp={importance}")
    except Exception as _upsert_err:
        _meta_types = {k: type(v).__name__ for k, v in meta.items()}
        _msg = (
            f"col.upsert failed on id={memory_id!r}: "
            f"{type(_upsert_err).__name__}: {_upsert_err}. "
            f"types[ids]=list[{type(memory_id).__name__}], "
            f"types[documents]=list[{type(embed_doc).__name__}], "
            f"len(embed_doc)={len(embed_doc) if isinstance(embed_doc, str) else 'n/a'}, "
            f"meta_value_types={_meta_types}"
        )
        logger.error(_msg)
        raise RuntimeError(_msg) from _upsert_err
    # Proceed with downstream (SQLite record node, context wiring, entity
    # link, duplicate detection) only after the upsert succeeded. The
    # original flow had no try/except split here -- a fresh block keeps
    # downstream failures distinguishable from the chroma upsert itself.
    try:
        # Register record as a first-class graph node in SQLite.
        try:
            _STATE.kg.add_entity(
                memory_id,
                kind="record",
                content=content[:200],
                importance=importance or 3,
                properties={
                    "content_type": content_type or "",
                    "added_by": added_by or "",
                },
                session_id=_STATE.session_id or "",
                intent_id=(
                    _STATE.active_intent.get("intent_id", "") if _STATE.active_intent else ""
                ),
            )
        except Exception:
            pass  # Non-fatal -- record exists in ChromaDB regardless

        # ── Context fingerprint: keywords → entity_keywords table,
        # view vectors → feedback_contexts collection, context_id → entities row.
        # context is optional here so legacy intent.py callers (which still pass
        # synthetic kwargs) keep working -- when present, full Context wiring engages.
        if context:
            try:
                ctx_keywords = context.get("keywords") or []
                _STATE.kg.add_entity_keywords(memory_id, ctx_keywords)
                # BM25-IDF maintenance: bump freq for each keyword and
                # recompute idf so the keyword channel dampens dominant
                # corpus-wide terms. Scoped to kind='record' to keep
                # the corpus meaningful for retrieval (schema entities
                # would inflate freq without contributing to recall).
                try:
                    _STATE.kg.record_keyword_observations(ctx_keywords)
                except Exception:
                    pass
                # Stamp creation_context_id from the active context
                # entity when one exists. Replaces the retired
                # persist_context path.
                active_ctx = _active_context_id()
                if active_ctx:
                    _STATE.kg.set_entity_creation_context(memory_id, active_ctx)
            except Exception:
                pass  # Non-fatal

        # ── P1 created_under provenance edge ──
        # Every memory emitted while a context is active records the link
        # from memory → context. Consumed by P2's Channel D + finalize
        # coverage check. See docs/context_as_entity_redesign_plan.md §1.
        _active_ctx = _active_context_id()
        if _active_ctx:
            try:
                _STATE.kg.add_triple(memory_id, "created_under", _active_ctx)
            except Exception:
                pass  # Non-fatal -- memory exists regardless

        # Create entity→memory link(s) using the specified predicate
        VALID_MEMORY_PREDICATES = {
            "described_by",
            "evidenced_by",
            "derived_from",
            "mentioned_in",
            "session_note_for",
        }
        link_predicate = predicate if predicate in VALID_MEMORY_PREDICATES else "described_by"

        linked_entities = []
        entity_names = []
        if entity:
            # Support comma-separated list
            entity_names = [e.strip() for e in entity.split(",") if e.strip()]

        from .knowledge_graph import normalize_entity_name

        for ename in entity_names:
            eid = normalize_entity_name(ename)
            # Only link to entities that already exist -- don't auto-create junk stubs
            existing_entity = _STATE.kg.get_entity(eid)
            if not existing_entity:
                # Entity doesn't exist -- skip the link. Agent should declare entities
                # via kg_declare_entity before referencing them in memories.
                continue
            try:
                _STATE.kg.add_triple(eid, link_predicate, memory_id)
                linked_entities.append(eid)
            except Exception:
                pass  # Non-fatal: memory exists, linking failed

        result = {
            "success": True,
            "memory_id": memory_id,
            "content_type": content_type,
            "importance": importance,
            "linked_entities": linked_entities,
        }

        # ── Memory duplicate detection ──
        try:
            dup_results = col.query(
                query_texts=[content[:500]],
                n_results=5,
                include=["documents", "distances"],
            )
            if dup_results["ids"] and dup_results["ids"][0]:
                dup_conflicts = []
                for i, did in enumerate(dup_results["ids"][0]):
                    if did == memory_id:
                        continue  # Skip self
                    dist = dup_results["distances"][0][i]
                    sim = round(max(0.0, 1.0 - dist), 3)
                    if sim < 0.85:
                        continue  # Not similar enough
                    # Slice 3 2026-04-28: integer conf_<N> id (1-indexed
                    # within the current batch). Replaces the prior
                    # conflict_memory_<memory_id>_<did> string concat which
                    # cost ~30 tokens per id. The handle's only purpose is
                    # to reference into resolve_conflicts within the same
                    # pending batch -- pending_conflicts must be cleared
                    # before any new tool fires (enforced by the
                    # pending-conflicts-block-tools rule), so a tiny
                    # batch-local counter is sufficient and conf_1 restart
                    # is safe across batches. The conflict_type field below
                    # carries the type info that used to be in the prefix.
                    conflict_id = f"conf_{len(dup_conflicts) + 1}"
                    preview = (dup_results["documents"][0][i] or "")[:150]
                    past = None
                    try:
                        past = _STATE.kg.get_past_conflict_resolution(
                            did, memory_id, "memory_duplicate"
                        )
                    except Exception:
                        pass
                    conflict_entry = {
                        "id": conflict_id,
                        "conflict_type": "memory_duplicate",
                        "reason": f"Similar memory found (similarity: {sim})",
                        "existing_id": did,
                        "existing_preview": preview,
                        "new_id": memory_id,
                        "similarity": sim,
                    }
                    if past:
                        conflict_entry["past_resolution"] = past
                    dup_conflicts.append(conflict_entry)
                if dup_conflicts:
                    _STATE.pending_conflicts = dup_conflicts
                    from . import intent

                    intent._persist_active_intent()
                    result["conflicts"] = dup_conflicts
                    past_hint = _past_resolution_hint(dup_conflicts)
                    result["conflicts_prompt"] = (
                        f"{len(dup_conflicts)} similar memory(s) found. "
                        f"Call mempalace_resolve_conflicts: merge, keep, or skip." + past_hint
                    )
        except Exception:
            pass

        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Phase 2: tool_kg_delete_entity moved to tool_mutate.py.


def _resolve_wake_up_agent(agent):
    """Resolve agent for wake_up: explicit arg > identity.txt first-token.

    Returns (agent_str_or_None, error_response_or_None). If first element
    is None, caller should return the error_response verbatim.
    """
    if not agent or not str(agent).strip():
        try:
            from pathlib import Path as _Path

            identity_file = _Path.home() / ".mempalace" / "identity.txt"
            if identity_file.exists():
                first_line = identity_file.read_text(encoding="utf-8").strip().splitlines()
                if first_line:
                    agent = first_line[0].split()[0] if first_line[0].split() else None
        except Exception:
            agent = None
    if not agent or not str(agent).strip():
        return None, {
            "success": False,
            "error": (
                "`agent` is required on mempalace_wake_up. Pass your agent name "
                "(e.g. mempalace_wake_up(agent='ga_agent')) -- wake_up uses it to "
                "auto-bootstrap the agent entity + is_a agent edge on a fresh "
                "palace and to scope affinity in L1 retrieval. Alternatively, "
                "create ~/.mempalace/identity.txt with the agent name as its "
                "first token and call wake_up again."
            ),
        }
    return str(agent).strip(), None


def _bootstrap_agent_if_missing(agent):
    """Auto-bootstrap the agent entity on a fresh palace (cold restart safe).

    Direct KG writes here bypass the normal added_by/_require_agent gate --
    that gate is circular on a fresh palace (no agent exists → no agent
    can be declared via kg_declare_entity → deadlock). wake_up is the
    single sanctioned bootstrap path.
    """
    from .knowledge_graph import normalize_entity_name as _norm

    _agent_id = _norm(agent)
    _agent_ent = _STATE.kg.get_entity(_agent_id)
    if not _agent_ent:
        _STATE.kg.add_entity(
            _agent_id,
            kind="entity",
            content=f"Agent: {agent}",
            importance=4,
        )
        _STATE.kg.add_triple(_agent_id, "is_a", "agent")
        _sync_entity_to_chromadb(_agent_id, agent, f"Agent: {agent}", "entity", 4)
        _STATE.declared_entities.add(_agent_id)


# ── Phase 2: tool_wake_up moved to tool_lifecycle.py.

# tool_update_drawer_metadata removed: merged into tool_kg_update_entity.


# ==================== KNOWLEDGE GRAPH ====================


CONTEXT_EDGE_PREDICATES = frozenset({"rated_useful", "rated_irrelevant", "surfaced"})

# Keys lifted from entity-record metadata into the kg_query `details` block.
# Kept narrow on purpose -- the goal is "what IS this entity" in a few fields,
# not a metadata dump. Absent values are dropped so the block stays terse.
_ENTITY_DETAIL_META_KEYS = ("kind", "summary", "importance", "content_type")


def _filter_context_edges(facts):
    kept = [f for f in facts if f.get("predicate") not in CONTEXT_EDGE_PREDICATES]
    return kept, len(facts) - len(kept)


def _fetch_entity_details(eid):
    """Return the entity's own content + short metadata, or None.

    kg_query historically returned only outgoing/incoming triples (edges),
    leaving the caller with no way to see what the entity IS without a
    follow-up kg_search. This helper pulls the entity's representative
    Chroma record so callers see content/summary/kind/importance inline.

    Lookup order matches _is_declared: prefer multi-view records keyed by
    metadata.entity_id (P5.2+ declarations), fall back to raw-id lookup
    for legacy single-record bookkeeping entities.
    """
    try:
        col = _get_entity_collection(create=False)
    except Exception:
        return None
    if col is None:
        return None

    doc = None
    meta = None

    # Multi-view lookup (post-P5.2 entities).
    try:
        got = col.get(
            where={"entity_id": eid},
            limit=1,
            include=["documents", "metadatas"],
        )
        if got and got.get("ids"):
            docs = got.get("documents") or []
            metas = got.get("metadatas") or []
            doc = docs[0] if docs else None
            meta = metas[0] if metas else None
    except Exception:
        pass

    # Fallback: raw-id lookup (legacy bookkeeping entities).
    if meta is None and doc is None:
        try:
            got = col.get(ids=[eid], include=["documents", "metadatas"])
            if got and got.get("ids"):
                docs = got.get("documents") or []
                metas = got.get("metadatas") or []
                doc = docs[0] if docs else None
                meta = metas[0] if metas else None
        except Exception:
            pass

    if meta is None and not doc:
        return None

    meta = meta or {}
    out = {}
    for key in _ENTITY_DETAIL_META_KEYS:
        val = meta.get(key)
        if val is None or val == "":
            continue
        out[key] = val
    if doc:
        out["content"] = doc
    return out or None


# ── Phase 2: tool_kg_query moved to tool_read.py.

# ── Phase 2: tool_kg_search moved to tool_read.py.


# ── Phase 2: tool_kg_add moved to tool_mutate.py.

# ── Phase 2: tool_kg_add_batch moved to tool_mutate.py.

# ── Phase 2: tool_kg_invalidate moved to tool_mutate.py.

# ── Phase 2: tool_kg_timeline + tool_kg_stats moved to tool_read.py.
# They're imported back at end-of-file for TOOLS dispatch.


# ==================== ENTITY DECLARATION ====================

# TODO (threshold): ENTITY_SIMILARITY_THRESHOLD is a strong learning
# candidate. Outcome signal = was the detected collision actually a
# duplicate (agent merged them) or distinct (agent kept both)?
# Correlate sim-at-detection with resolve_conflicts action and sweep
# the threshold. Needs ~50 collision decisions for meaningful signal.
ENTITY_SIMILARITY_THRESHOLD = 0.85
# Legacy -- mempalace_entities was absorbed into mempalace_records by the M1
# migration. Kept as a module constant only so the migration can look it
# up when scanning for legacy rows on startup.
ENTITY_COLLECTION_NAME = "mempalace_entities"

# Session-level declared entities (in-memory cache on _STATE, falls back to persistent KG).
# _STATE.pending_conflicts blocks all tools until resolved.
# Defaults to None on ServerState construction -- no explicit init needed here.

# ── Session isolation: save/restore state per session_id ──
# _STATE.session_state maps session_id -> {active_intent, pending_conflicts,
# declared}. When multiple callers (sub-agents) share the same MCP process
# but have different session IDs, this prevents them from overwriting each
# other's state.


def _sanitize_session_id(session_id: str) -> str:
    """Match hooks_cli._sanitize_session_id -- only alnum/dash/underscore.

    Returns "" for empty or fully-stripped input. NO FALLBACK to
    'unknown' or 'default' -- callers must handle empty sid explicitly.
    A shared sid value (like 'unknown') would merge every agent's
    state file into one, producing cross-agent contamination.
    """
    import re

    if not session_id:
        return ""
    return re.sub(r"[^a-zA-Z0-9_-]", "", str(session_id))


def _save_session_state():
    """Save current session state before switching to a different session.

    Disk is authoritative for pending state (see intent._persist_active_intent).
    The in-memory session_state cache snapshots ONLY the ephemeral fields
    (active_intent + declared_entities). Pending conflicts are NOT cached
    here -- they live on disk and are re-read via _load_pending_*_from_disk
    on every restore. That asymmetry is deliberate: once a caller clears
    pending state (resolve_conflicts clears in-memory AND persists the
    cleared disk file), a later session-id switch must NOT resurrect the
    old pending items from a stale snapshot.
    """
    if _STATE.session_id:
        _STATE.session_state[_STATE.session_id] = {
            "active_intent": _STATE.active_intent,
            "declared": _STATE.declared_entities,
        }


def _load_pending_from_disk(key: str, session_id: str = None) -> list:
    """Load pending items (conflicts) from the active intent state file.

    Disk is the source of truth for cross-session/cross-restart state.
    Returns empty list if no sid, no file, or no items pending.

    NO FALLBACK to ``active_intent_default.json``. A request for a
    specific sid's pending state must NEVER read another sid's file.
    """
    sid = session_id or _STATE.session_id
    if not sid:
        return []
    try:
        state_file = _INTENT_STATE_DIR / f"active_intent_{sid}.json"
        if not state_file.is_file():
            return []
        data = json.loads(state_file.read_text(encoding="utf-8"))
        return data.get(key) or []
    except Exception:
        return []


def _load_pending_conflicts_from_disk(session_id: str = None) -> list:
    return _load_pending_from_disk("pending_conflicts", session_id)


def _restore_session_state(sid: str):
    """Restore session state for the given session_id.

    Pending conflicts are NOT read from the in-memory cache -- they always
    come from disk (the authoritative source, updated in lockstep by
    intent._persist_active_intent). Everything else (active_intent,
    declared_entities) is ephemeral and safe to cache in-process.
    """
    if sid in _STATE.session_state:
        s = _STATE.session_state[sid]
        _STATE.active_intent = s["active_intent"]
        _STATE.declared_entities = s["declared"]
    else:
        _STATE.active_intent = None
        _STATE.declared_entities = set()
    # Disk is authoritative for pending state. Always OVERWRITE (not
    # additive) so that a cleared file becomes cleared state -- the old
    # "if disk has something, set; otherwise leave memory alone" logic
    # let a stale in-memory copy survive past a legitimate clear.
    _STATE.pending_conflicts = _load_pending_conflicts_from_disk(sid) or None


def _require_sid(action: str = "this operation") -> dict:
    """Validate an agent session_id is present. Returns error dict on
    failure, None on pass.

    Every write tool that touches state (active intent file, pending
    queues, trace file, save counter) must call this at the top:

        sid_err = _require_sid(action="resolve_conflicts")
        if sid_err:
            return sid_err

    On failure the agent sees a clear error pointing at the root cause
    (the hook didn't inject sessionId) rather than a silent "no-op"
    that looks like success. Agent can't proceed without fixing the
    hook wiring -- which is what we want.

    NO FALLBACK TO A SHARED SID. A missing sid in a state-writing tool
    is a real error. Quietly substituting "default" / "unknown" would
    cause cross-agent contamination (the 2026-04-19 deadlock class).

    ONE NARROW EXCEPTION: the memory_gardener subprocess (TS or Python)
    deliberately runs with all hooks disabled to avoid the fork-bomb
    cascade, so the PreToolUse sid-injection hook never fires for it.
    The gardener sets MEMPALACE_GARDENER_ACTIVE=1 in its env as a
    signal that it's operating in that headless mode. We synthesise
    a per-process sid for it so state-writing mutations can proceed.
    The synthesised sid includes a timestamp + pid so parallel
    gardener runs don't collide. Non-gardener callers without a sid
    still hit the loud error above.
    """
    if not _STATE.session_id:
        if os.environ.get("MEMPALACE_GARDENER_ACTIVE") == "1":
            import time

            _STATE.session_id = f"gardener_{int(time.time())}_{os.getpid()}"
            return None
        return {
            "success": False,
            "error": (
                f"'{action}' requires an active session_id but none was "
                f"propagated to the MCP server. Root cause: the PreToolUse "
                f"hook must inject sessionId via hookSpecificOutput."
                f"updatedInput on every MCP tool call. Check that the "
                f"mempalace plugin is installed and its hook is registered "
                f"in ~/.claude/settings.json. Refusing to proceed with an "
                f"empty sid -- using a shared default would cross-contaminate "
                f"other agents' state."
            ),
        }
    return None


def _require_agent(agent: str, action: str = "this operation") -> dict:
    """Validate agent is declared (is_a agent). Returns {"success": False, ...} on failure, None on pass.

    centralized agent validation. Every write MCP tool that
    attributes an action to an agent should call this at the top:

        err = _require_agent(agent, action="kg_add")
        if err:
            return err

    Callers pass the RAW agent name (not normalized); this helper
    normalizes and walks the KG for an `is_a agent` edge. Missing agent
    or undeclared agent → structured error with the declaration recipe.
    Never raises -- KG lookup failures are treated as pass (graceful
    degradation when KG is unavailable, e.g. in fresh test fixtures).
    """
    # Gardener bypass -- same rationale as _require_sid. The gardener
    # subprocess runs with hooks off and doesn't go through the
    # wake_up bootstrap, so the standard "is_a agent" check would
    # always fail. MEMPALACE_GARDENER_ACTIVE=1 is our signal that
    # we're operating in that narrow mode; allow the attribution to
    # "memory_gardener" without requiring the entity to be declared.
    if os.environ.get("MEMPALACE_GARDENER_ACTIVE") == "1":
        return None
    if not agent or not isinstance(agent, str) or not agent.strip():
        return {
            "success": False,
            "error": (
                f"`agent` is required for {action}. Pass your declared agent entity name "
                f"(e.g. 'ga_agent', 'technical_lead_agent'). Every write operation must "
                f"be attributed to a declared agent (is_a agent)."
            ),
        }
    try:
        from .knowledge_graph import normalize_entity_name

        agent_id = normalize_entity_name(agent)
        if _STATE.kg:
            edges = _STATE.kg.query_entity(agent_id, direction="outgoing")
            is_agent = any(
                e["predicate"] == "is_a" and e["object"] == "agent" and e.get("current", True)
                for e in edges
            )
            if not is_agent:
                # Detect the fresh-palace case: if NO agents exist globally,
                # the kg_declare_entity recipe below is circular (it requires
                # added_by to already be a declared agent). Point the caller
                # at the sanctioned bootstrap path -- wake_up -- instead.
                zero_agents = False
                try:
                    any_agent_edges = _STATE.kg.query_entity("agent", direction="incoming")
                    zero_agents = not any(
                        e["predicate"] == "is_a" and e.get("current", True) for e in any_agent_edges
                    )
                except Exception:
                    zero_agents = False

                if zero_agents:
                    return {
                        "success": False,
                        "error": (
                            f"`agent` '{agent}' is not declared AND no agents exist "
                            f"in this palace yet. This is a fresh / cold-started "
                            f"palace -- bootstrap via wake_up, not kg_declare_entity "
                            f"(which requires a pre-existing declared agent and is "
                            f"circular here). Call: "
                            f"mempalace_wake_up(agent='{agent}') -- it auto-creates "
                            f"the agent entity and the is_a agent edge in one shot. "
                            f"Then retry {action}."
                        ),
                    }
                return {
                    "success": False,
                    "error": (
                        f"`agent` '{agent}' is not a declared agent (missing is_a agent edge). "
                        f"Declare it first: "
                        f"kg_declare_entity(name='{agent}', kind='entity', importance=4, "
                        f"context={{'queries': ['<who you are>', '<your role>'], "
                        f"'keywords': ['agent', '<identifier>']}}, added_by='<an-already-declared-agent>') "
                        f"then kg_add(subject='{agent}', predicate='is_a', object='agent', "
                        f"context={{'queries': [...], 'keywords': [...]}}). "
                        f"Or, if this is a fresh palace, use "
                        f"mempalace_wake_up(agent='{agent}') for the sanctioned bootstrap."
                    ),
                }
    except Exception:
        pass  # graceful fallback if KG unavailable (fresh fixture, etc.)
    return None


def _declare_entity_recipe(
    name: str,
    kind: str = "entity",
    hint: str = None,
    extra_properties: str = "",
) -> str:
    """Canonical kg_declare_entity recipe used in error messages.

    Single source of truth -- DO NOT hand-roll `description=...` in new
    error strings; the tool rejects it (see tool_kg_declare_entity, the
    P4.2 legacy-path block). `context={queries, keywords, entities,
    summary}` is the dict-only mandatory shape (Adrian's design lock
    2026-04-25): every context-taking write tool requires the
    structured summary {what, why, scope?} inside context.

    Args:
        name: entity name to insert into the example.
        kind: one of 'entity' | 'class' | 'predicate' | 'literal'.
        hint: short phrase describing what this entity is; becomes queries[0].
        extra_properties: optional trailing ', properties=...' fragment for
            callers that need to teach class/predicate-specific metadata
            (rules_profile, constraints, file_path, etc.).
    """
    hint = hint or f"what {name} represents"
    default_importance = 4 if kind in ("class", "predicate") else 3
    props = f", properties={extra_properties}" if extra_properties else ""
    return (
        f"kg_declare_entity(name='{name}', kind='{kind}', importance={default_importance}, "
        f"context={{'queries': ['{hint}', '<another perspective>'], "
        f"'keywords': ['<term1>', '<term2>'], "
        f"'entities': ['<related_entity>'], "
        f"'summary': {{'what': '<noun phrase naming {name}>', "
        f"'why': '<purpose / role / claim clause, ≥15 chars>', "
        f"'scope': '<temporal/domain qualifier (optional)>'}}}}, "
        f"added_by='<your_agent>'"
        f"{props})"
    )


def _declare_intent_recipe(intent_type: str = "modify", slots: str = None) -> str:
    """Canonical mempalace_declare_intent recipe for error messages.

    Single source of truth -- the tool requires the unified Context object
    {queries, keywords, entities, summary} AND `budget`. Adrian's design
    lock 2026-04-25: every context-taking write tool requires the
    structured summary {what, why, scope?} inside context. The legacy
    `description=` path is gone.
    """
    slots = slots or '{"files": ["target_file"]}'
    return (
        f"mempalace_declare_intent(intent_type='{intent_type}', slots={slots}, "
        f"context={{'queries': ['<what you plan to do>', '<another angle>'], "
        f"'keywords': ['<term1>', '<term2>'], "
        f"'entities': ['<related_entity>'], "
        f"'summary': {{'what': '<noun phrase naming the action>', "
        f"'why': '<purpose / goal clause, ≥15 chars>', "
        f"'scope': '<temporal/domain qualifier (optional)>'}}}}, "
        f"agent='<your_agent>', budget={{'Read': 5, 'Edit': 3}})"
    )


def _is_declared(entity_id: str) -> bool:
    """Check if an entity is declared, with fallback to persistent KG.

    The in-memory _STATE.declared_entities set is a cache that gets cleared on
    MCP server restart. If an entity isn't in the cache but exists in the
    persistent KG (ChromaDB), it's auto-added to the cache and considered
    declared. This makes declarations survive MCP server restarts without
    requiring the model to re-call wake_up.

    P5.2 lookup order (must cover BOTH physical layouts):
      1. In-memory cache -- _STATE.declared_entities (session-lifetime fast path).
      2. Multi-view entities -- where={"entity_id": X}. Entities declared
         via kg_declare_entity(context=...) live under '{eid}__v{N}' ids
         with metadata.entity_id=eid (see _sync_entity_views_to_chromadb).
      3. Single-record entities -- ids=[X]. Internal bookkeeping entities
         (execution traces, gotchas) written by _sync_entity_to_chromadb
         use raw '{eid}' as the Chroma id, so the id-based lookup still
         applies. Same for the memories collection.
    """
    if entity_id in _STATE.declared_entities:
        return True

    # KG fallback -- SQLite is the authoritative source of truth.
    # Wake_up pulls classes/predicates/intents directly from
    # _STATE.kg.list_entities, so every entity surfaced to the model via
    # declared.* must also be considered declared for gating purposes.
    # Checking the KG before Chroma ensures that a transient in-memory
    # cache wipe (session_id switch, MCP restart) or a Chroma-side glitch
    # cannot silently downgrade a legitimate intent_type to "not declared".
    try:
        if _STATE.kg.get_entity(entity_id):
            _STATE.declared_entities.add(entity_id)
            return True
    except Exception:
        pass

    ecol = _get_entity_collection(create=False)
    if not ecol:
        return False

    # Multi-view lookup (post-P5.2 entities declared via Context)
    try:
        result = ecol.get(where={"entity_id": entity_id}, limit=1)
        if result and result.get("ids"):
            _STATE.declared_entities.add(entity_id)
            return True
    except Exception:
        pass

    # Fallback: raw-id lookup (single-record bookkeeping entities)
    try:
        result = ecol.get(ids=[entity_id])
        if result and result.get("ids"):
            _STATE.declared_entities.add(entity_id)
            return True
    except Exception:
        pass

    return False


def _get_entity_collection(create: bool = True):
    """Entity collection accessor.

    Phase M1 collapsed the two physical Chroma collections (records +
    entities) into a single ``mempalace_records`` collection discriminated
    by ``metadata.kind``. This helper remains for callsite compatibility --
    anywhere that previously fetched "the entity collection" now gets the
    unified collection instead, and any query it runs must filter on
    ``metadata.kind`` if it wants entity-only results.

    The view-schema migration (``__vN`` suffix) runs against the same
    collection on first touch.
    """
    col = _get_collection(create=create)
    if col is not None:
        try:
            _migrate_entity_views_schema(col)
        except Exception:
            pass
    return col


# Chroma migration that retires the old '::view_N' suffix scheme.
# Detects legacy records by the absence of metadata.entity_id (all records
# written by the new code path carry it). Migration is a single pass at
# first _get_entity_collection(create=True) call per process; a _STATE flag
# prevents re-runs.


def _migrate_entity_views_schema(col):
    """Retire ::view_N ids in favour of __vN + metadata.entity_id.

    Idempotent. Only touches records missing metadata.entity_id:
      - '{eid}::view_{N}'  →  id='{eid}__v{N}',  meta.entity_id=eid, meta.view_index=N
      - pre-P4.2 plain '{eid}'  →  id='{eid}__v0', meta.entity_id=eid, meta.view_index=0

    Embeddings are preserved via the include=['embeddings'] round-trip,
    avoiding an expensive re-embed pass.
    """
    if _STATE.entity_views_migrated:
        return
    _STATE.entity_views_migrated = True
    try:
        got = col.get(include=["documents", "metadatas", "embeddings"])
    except Exception:
        return
    if not got or not got.get("ids"):
        return

    to_upsert_ids, to_upsert_docs, to_upsert_metas, to_upsert_embs = [], [], [], []
    to_delete = []
    all_embs_present = True

    for i, raw_id in enumerate(got["ids"]):
        meta = (got["metadatas"][i] if got.get("metadatas") else {}) or {}
        if "entity_id" in meta:
            continue  # Already in the new shape
        if "::view_" in raw_id:
            eid, _sep, v = raw_id.rpartition("::view_")
            try:
                view_idx = int(v)
            except ValueError:
                view_idx = 0
            new_id = f"{eid}__v{view_idx}"
        else:
            eid = raw_id
            view_idx = 0
            new_id = f"{eid}__v0"

        new_meta = {**meta, "entity_id": eid, "view_index": view_idx}
        to_delete.append(raw_id)
        to_upsert_ids.append(new_id)
        to_upsert_docs.append((got["documents"][i] if got.get("documents") else "") or "")
        to_upsert_metas.append(new_meta)
        emb = got["embeddings"][i] if got.get("embeddings") else None
        if emb is None:
            all_embs_present = False
        to_upsert_embs.append(emb)

    if not to_upsert_ids:
        return
    try:
        kwargs = {"ids": to_upsert_ids, "documents": to_upsert_docs, "metadatas": to_upsert_metas}
        if all_embs_present:
            kwargs["embeddings"] = to_upsert_embs
        col.upsert(**kwargs)
        col.delete(ids=to_delete)
        logger.info(
            f"P5.2 entity_views migration: rewrote {len(to_upsert_ids)} records "
            f"({'preserved' if all_embs_present else 're-embedded'})"
        )
    except Exception as e:
        logger.warning(f"P5.2 entity_views migration failed: {e}")


# kind='record' → 'record' one-pass migration (flag lives on _STATE).


def _migrate_kind_memory_to_record():
    """Rewrite kind='record' → 'record' across ChromaDB memory collection
    + SQLite entities table. Idempotent, one-pass per process.

    Why: "memory" was overloaded (palace-level concept + record-type name).
    Renaming at the record-type layer lets "memory" stay the palace
    concept cleanly. Data layout is untouched -- only the metadata.kind
    string changes.
    """
    if _STATE.kind_rename_migrated:
        return
    _STATE.kind_rename_migrated = True

    # 1) Rewrite Chroma memory collection metadata.
    try:
        col = _get_collection(create=False)
        if col is not None:
            got = col.get(where={"kind": "memory"}, include=["metadatas"])
            if got and got.get("ids"):
                new_metas = []
                for meta in got["metadatas"] or []:
                    m = dict(meta or {})
                    m["kind"] = "record"
                    new_metas.append(m)
                col.update(ids=got["ids"], metadatas=new_metas)
                logger.info(
                    f"P6.2 kind migration (chroma memory collection): "
                    f"rewrote {len(got['ids'])} records from 'memory' to 'record'."
                )
    except Exception as e:
        logger.warning(f"P6.2 kind migration (chroma) failed: {e}")

    # 2) Rewrite SQLite entities table.
    try:
        if _STATE.kg:
            conn = _STATE.kg._conn()
            cursor = conn.execute("UPDATE entities SET kind='record' WHERE kind='record'")
            conn.commit()
            if cursor.rowcount:
                logger.info(
                    f"P6.2 kind migration (sqlite entities): "
                    f"rewrote {cursor.rowcount} rows from 'memory' to 'record'."
                )
    except Exception as e:
        logger.warning(f"P6.2 kind migration (sqlite) failed: {e}")


# M1 one-shot migration: absorb mempalace_entities into mempalace_records.
# Gated on ServerState.entity_collection_merged (added to server_state.py).
# Idempotent: on a fresh palace the legacy collection may not exist, in
# which case this is a no-op.


def _migrate_entities_collection_into_records():
    """Copy every row from the legacy mempalace_entities collection into
    the unified mempalace_records collection, then delete the legacy
    collection. Runs once per process.

    Safe to run multiple times: subsequent calls see no entities
    collection and exit quickly.

    ID-space note: entity rows use ``<entity_id>__vN`` IDs while record
    rows use ``record_<agent>_<slug>`` IDs -- two non-overlapping
    namespaces -- so merging cannot create ID collisions in the target
    collection. metadata.kind stays the discriminator for kind-scoped
    queries (``kind="class"``, ``"entity"``, ``"predicate"`` vs
    ``"record"``).
    """
    if getattr(_STATE, "entity_collection_merged", False):
        return
    _STATE.entity_collection_merged = True

    try:
        client = chromadb.PersistentClient(path=_STATE.config.palace_path)
        try:
            legacy = client.get_collection("mempalace_entities")
        except Exception:
            return  # Legacy collection never existed -- fresh palace.

        dest = _get_collection(create=True)
        if dest is None:
            logger.warning("M1 migration: target mempalace_records unavailable")
            return

        got = legacy.get(include=["documents", "metadatas", "embeddings"])
        if not got or not got.get("ids"):
            # Collection exists but is empty -- drop it.
            try:
                client.delete_collection("mempalace_entities")
            except Exception:
                pass
            logger.info("M1 migration: legacy mempalace_entities was empty, dropped.")
            return

        ids = got["ids"]
        docs = got.get("documents") or [None] * len(ids)
        metas = got.get("metadatas") or [None] * len(ids)
        embs = got.get("embeddings") or [None] * len(ids)

        BATCH = 200
        moved = 0
        any_upsert_failed = False
        for start in range(0, len(ids), BATCH):
            chunk_ids = ids[start : start + BATCH]
            chunk_docs = docs[start : start + BATCH]
            chunk_metas = metas[start : start + BATCH]
            chunk_embs = embs[start : start + BATCH]
            upsert_kwargs = {
                "ids": chunk_ids,
                "documents": chunk_docs,
                "metadatas": chunk_metas,
            }
            if all(e is not None for e in chunk_embs):
                upsert_kwargs["embeddings"] = chunk_embs
            try:
                dest.upsert(**upsert_kwargs)
                moved += len(chunk_ids)
            except Exception as e:
                any_upsert_failed = True
                logger.warning(f"M1 migration upsert batch failed: {e}")

        # Safety: only delete the legacy collection if EVERY row landed in
        # the target. A partial copy must leave the source intact so no
        # embeddings are lost; the migration flag is already set, so the
        # next startup won't retry, but the data is still accessible.
        if any_upsert_failed or moved != len(ids):
            logger.warning(
                f"M1 migration: moved {moved}/{len(ids)} rows; legacy collection "
                f"NOT deleted -- partial copy detected, data preserved."
            )
            return

        try:
            client.delete_collection("mempalace_entities")
        except Exception as e:
            logger.warning(f"M1 migration: delete_collection failed: {e}")

        logger.info(
            f"M1 migration: moved {moved} entity rows into mempalace_records, "
            f"dropped legacy mempalace_entities."
        )
    except Exception as e:
        logger.warning(f"M1 migration failed: {e}")


# Retired Chroma collection name -- kept ONLY as a string for the
# one-shot drop hook (_drop_feedback_contexts_collection_once). No
# accessor helper, no ID generator, no maxsim function. All of that
# served the pre-context-as-entity feedback pipeline which is gone.
FEEDBACK_CONTEXT_COLLECTION = "mempalace_feedback_contexts"


# ──────────────────────────────────────────────────────────────────────────
# Context-as-first-class-entity (P1 of docs/context_as_entity_redesign_plan.md)
#
# Every context created by declare_intent / declare_operation / kg_search is
# materialised as a KG entity with kind="context". View vectors live in the
# mempalace_context_views Chroma collection (one row per view). Lookup uses
# ColBERT-style MaxSim (Khattab & Zaharia 2020, arXiv:2004.12832) -- a new
# context whose MaxSim against any existing one is ≥ T_reuse (0.90) reuses
# that context's id; otherwise a fresh context entity is minted. When the
# best MaxSim falls in [T_similar, T_reuse) a `similar_to` edge is written
# to the nearest neighbour with prop {sim: float}. This matches BIRCH-style
# threshold accretion (Zhang/Ramakrishnan/Livny 1996 SIGMOD).
#
# This P1 pipeline is ADDITIVE to the old persist_context / store_feedback_
# context machinery (which still lives in FEEDBACK_CONTEXT_COLLECTION). P2
# drops the old via migration 015.
# ──────────────────────────────────────────────────────────────────────────

CONTEXT_VIEWS_COLLECTION = "mempalace_context_views"

# BIRCH-style accretion thresholds (Zhang/Ramakrishnan/Livny 1996 SIGMOD).
# Calibrated to the max-of-max aggregation in scoring.multi_view_max_sim
# (the centralized helper, 2026-04-26). Under max-of-max the score is
# the BEST single per-view cosine; calibration ranges on
# all-MiniLM-L6-v2 (the embedding model) place verbatim near 1.0,
# strong paraphrase 0.80-0.90, related topic 0.65-0.80, weak overlap
# 0.50-0.65, unrelated <0.50.
#
#   - 0.90 reuse cut: only verbatim or near-verbatim single-view
#     overlap reuses an existing context. Carried over from the
#     prior mean-of-max calibration on Adrian's call 2026-04-26 --
#     under max-of-max the practical difference between 0.90 and
#     0.92 is the narrow [0.90, 0.92] band of "near-identical view
#     paraphrase," which is rare in practice; verbatim hits 1.0
#     either way and strong paraphrases sit ~0.85, below either cut.
#   - 0.65 similar cut: any single strong topical overlap writes a
#     similar_to edge to the nearest neighbour. Permissive so
#     paraphrased same-topic and partial-aspect-overlap contexts get
#     linked into the recall neighbourhood.
#
# The prior mean-of-max formula (functionally equivalent up to
# scaling) was retired 2026-04-26 after empirical experiment +
# literature audit (see record_ga_agent_result_audit_similarity_
# decision_sites_2026_04_26 and Theorem 4.9 of arXiv 2512.12458).
#
# TODO (threshold calibration -- highest-ROI tunable left in the system):
#   1. Log max-of-max on every emit (already partially recorded via
#      search_log.jsonl telemetry in eval_harness; extend to
#      declare_intent and declare_operation too).
#   2. Correlate score-at-decision with mean relevance of the resulting
#      feedback on the reused context. High score + bad feedback →
#      T_reuse too loose. Low score + good feedback at T_similar border
#      → T_similar too loose.
#   3. Offline sweep T_reuse 0.85→0.97, T_similar 0.50→0.75 in 0.02
#      steps; maximise (useful_reuses / (useful_reuses + bad_reuses))
#      subject to a similar_to density floor.
#   4. Once stable, fold the learned values into a kv table that
#      tool_wake_up reads alongside the hybrid + channel weights.
# Needs ~50-100 intents with feedback before calibration is reliable.
CONTEXT_REUSE_THRESHOLD = 0.90
CONTEXT_SIMILAR_THRESHOLD = 0.65


def _telemetry_append_jsonl(filename: str, record: dict) -> None:
    """Append one JSONL line to ~/.mempalace/hook_state/<filename>.

    Best-effort: any error silently drops the record so telemetry
    failures never block the tool call. Read by mempalace.eval_harness.
    """
    try:
        from pathlib import Path as _Path

        directory = _Path.home() / ".mempalace" / "hook_state"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


ROCCHIO_MAX_VIEWS = 20
# TODO (threshold): ROCCHIO_QUERY_DEDUP_THRESHOLD is a plausible-but-weak
# learning target. Outcome signal = "dedup decision didn't lose a useful
# view later." Hard to attribute cleanly; probably not worth learning
# versus hand-tuning based on observed LRU eviction churn.
ROCCHIO_QUERY_DEDUP_THRESHOLD = 0.85  # drop novel query if MaxSim ≥ this


def rocchio_enrich_context(  # noqa: C901
    context_id: str,
    new_queries=None,
    new_keywords=None,
    new_entities=None,
    *,
    max_views: int = ROCCHIO_MAX_VIEWS,
) -> dict:
    """Rocchio-style enrichment of a reused context on positive feedback.

    Reference: Rocchio 1971 (Manning/Raghavan/Schütze IR book, Ch.9) --
    the query-reformulation algorithm shifts the retrieval query toward
    the centroid of relevant results and away from non-relevant ones.
    Our adaptation is to the *context entity itself*: when an existing
    context is reused (MaxSim ≥ T_reuse) AND the intent finishes with
    net-positive feedback, we merge the caller's NEW queries, keywords,
    and related entities into the context so future MaxSim lookups land
    on it more easily. The context accretes a shape that reflects every
    successful past use.

    LRU cap of ``max_views`` on the view vector list (default 20) -- when
    the count would exceed the cap, drop the oldest view_index. Every
    view added gets a fresh view_index slot so "oldest" is well-defined
    without an explicit timestamp.

    Dedup strategy per field:
      - queries: TWO-step dedup -- exact-text first, then semantic MaxSim
        against the context's existing view vectors (threshold 0.85 via
        ROCCHIO_QUERY_DEDUP_THRESHOLD). Redundant-angle queries are
        dropped before they eat an LRU slot. The semantic check is
        cheap -- one Chroma query per novel query, scoped to this
        context via metadata.context_id.
      - keywords: lowercased exact match only. Controlled-vocabulary
        tags are agent-curated; the IDF table handles downstream
        redundancy via freq-based dampening. Embedding per-keyword
        would be expensive + noisy on 1-3 word tags (see design note
        on the keyword channel).
      - entities: dedupe via ``normalize_entity_name`` on both sides
        so "LoginService", "login-service", and "Login Service" all
        canonicalise to the same id and merge cleanly.

    Returns a dict ``{added_queries, added_keywords, added_entities,
    evicted_views, dedup_dropped_queries}`` for telemetry.
    """
    stats = {
        "added_queries": 0,
        "added_keywords": 0,
        "added_entities": 0,
        "evicted_views": 0,
        "dedup_dropped_queries": 0,
    }
    if not context_id:
        return stats
    try:
        ctx_entity = _STATE.kg.get_entity(context_id)
    except Exception:
        return stats
    if not ctx_entity or ctx_entity.get("kind") != "context":
        return stats

    props = ctx_entity.get("properties", {}) or {}
    if isinstance(props, str):
        try:
            props = json.loads(props)
        except Exception:
            props = {}

    existing_queries = list(props.get("queries") or [])
    existing_keywords = list(props.get("keywords") or [])
    existing_entities = list(props.get("entities") or [])

    # Normalise incoming.
    nq = [q.strip() for q in (new_queries or []) if isinstance(q, str) and q.strip()]
    nk = [k.strip() for k in (new_keywords or []) if isinstance(k, str) and k.strip()]
    ne = [e.strip() for e in (new_entities or []) if isinstance(e, str) and e.strip()]

    # ── Queries: exact-text dedup + semantic MaxSim dedup ──
    existing_query_set = set(existing_queries)
    exact_novel = [q for q in nq if q not in existing_query_set]

    # Semantic dedup: drop novel queries whose MaxSim against any
    # existing view exceeds ROCCHIO_QUERY_DEDUP_THRESHOLD (0.85). No
    # point wasting an LRU slot on a near-paraphrase.
    semantically_novel = []
    if exact_novel and existing_queries:
        col = _get_context_views_collection(create=False)
        if col is not None:
            try:
                count = col.count()
            except Exception:
                count = 0
            if count > 0:
                for q in exact_novel:
                    try:
                        res = col.query(
                            query_texts=[q],
                            n_results=min(3, count),
                            where={"context_id": context_id},
                            include=["distances"],
                        )
                        if res.get("distances") and res["distances"] and res["distances"][0]:
                            min_dist = min(res["distances"][0])
                            max_sim = max(0.0, 1.0 - float(min_dist))
                            if max_sim >= ROCCHIO_QUERY_DEDUP_THRESHOLD:
                                stats["dedup_dropped_queries"] += 1
                                continue
                    except Exception:
                        pass
                    semantically_novel.append(q)
            else:
                semantically_novel = list(exact_novel)
        else:
            semantically_novel = list(exact_novel)
    else:
        semantically_novel = list(exact_novel)

    novel_queries = semantically_novel

    updated_queries = existing_queries + novel_queries
    # LRU cap on the views list.
    evicted_texts: list = []
    if len(updated_queries) > max_views:
        overflow = len(updated_queries) - max_views
        evicted_texts = updated_queries[:overflow]
        updated_queries = updated_queries[overflow:]
    stats["added_queries"] = len(novel_queries)
    stats["evicted_views"] = len(evicted_texts)

    # ── Keywords (dedup by lowered + punct-normalised text) ──
    # Controlled-vocabulary tags -- embedding per-keyword is noisy on
    # 1-3 word strings and expensive; the IDF table downweights
    # redundant tags automatically downstream. Exact-lowercase + a
    # light punctuation normalisation catches "Rate-Limit" vs
    # "rate-limit" vs " rate-limit " cheaply.
    def _normalise_keyword(k: str) -> str:
        return k.strip().lower()

    existing_kw_set = {_normalise_keyword(k) for k in existing_keywords if k.strip()}
    novel_keywords = []
    for kw in nk:
        norm = _normalise_keyword(kw)
        if not norm or norm in existing_kw_set:
            continue
        existing_kw_set.add(norm)
        novel_keywords.append(kw)
    updated_keywords = existing_keywords + novel_keywords
    stats["added_keywords"] = len(novel_keywords)

    # ── Related entities (dedup via normalize_entity_name) ──
    # "LoginService", "login-service", "Login Service" all canonicalise
    # to the same id -- so we compare normalised forms on both sides.
    try:
        from .knowledge_graph import normalize_entity_name as _norm_ent
    except Exception:

        def _norm_ent(x):
            return x

    existing_ent_norm = {_norm_ent(e) for e in existing_entities}
    novel_entities = []
    for e in ne:
        en = _norm_ent(e)
        if not en or en in existing_ent_norm:
            continue
        existing_ent_norm.add(en)
        novel_entities.append(en)  # store the canonical form
    updated_entities = existing_entities + novel_entities
    stats["added_entities"] = len(novel_entities)

    # Nothing changed → short-circuit.
    if not (novel_queries or novel_keywords or novel_entities):
        return stats

    # Persist updated properties on the context entity.
    try:
        new_props = dict(props)
        new_props["queries"] = updated_queries
        new_props["keywords"] = updated_keywords
        new_props["entities"] = updated_entities
        new_props["last_enriched_ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _STATE.kg.add_entity(
            context_id,
            kind="context",
            content=ctx_entity.get("content", "") or (updated_queries[0][:200]),
            importance=ctx_entity.get("importance", 3) or 3,
            properties=new_props,
        )
    except Exception:
        return stats

    # Persist new keywords into entity_keywords + BM25-IDF maintenance.
    if novel_keywords:
        try:
            _STATE.kg.add_entity_keywords(context_id, novel_keywords, source="rocchio")
        except Exception:
            pass
        # Record as observations so the IDF table reflects that these
        # keywords are now attached to more contexts (affects corpus-
        # wide rarity on the context-entity side).
        try:
            _STATE.kg.record_keyword_observations(novel_keywords)
        except Exception:
            pass

    # Update the context view vectors in Chroma.
    col = _get_context_views_collection(create=True)
    if col is not None and (novel_queries or evicted_texts):
        try:
            if evicted_texts:
                # Best-effort delete of stale vectors. The properties.queries
                # list on the context entity is the authoritative candidate
                # pool, so leaving stale vectors in Chroma is harmless -- the
                # channel-D walker only touches metadata.context_id, not
                # the per-view texts. Kept for cleanliness.
                try:
                    col.delete(where={"context_id": context_id, "_lru_evicted": True})
                except Exception:
                    pass
            if novel_queries:
                # Use view_index offsets past the current stored count
                # so we don't collide with earlier view ids.
                base = len(existing_queries)
                ids = [f"{context_id}_v{base + i}" for i in range(len(novel_queries))]
                metas = [
                    {"context_id": context_id, "view_index": base + i, "source": "rocchio"}
                    for i, _ in enumerate(novel_queries)
                ]
                col.upsert(ids=ids, documents=novel_queries, metadatas=metas)
        except Exception:
            pass

    return stats


def _record_context_emit(
    context_id: str,
    reused: bool,
    *,
    scope: str,
    queries=None,
    keywords=None,
    entities=None,
) -> None:
    """Record a context emit (intent / operation / search) on active_intent.

    Every emit site calls this after ``context_lookup_or_create`` so
    finalize_intent can:
      - Build the strict ``surfaced`` coverage set from every touched
        context (already via ``contexts_touched``).
      - Run Rocchio enrichment INDEPENDENTLY for each reused context
        using THAT emit's source queries/keywords/entities.

    Before this helper, only the intent-level context's reused flag and
    source data were tracked; operation and search reuses were invisible
    to Rocchio. This fixes that asymmetry (see the polish-PR discussion
    with Adrian on per-emit enrichment).
    """
    if not context_id:
        return
    ai = _STATE.active_intent
    if ai is None:
        return
    touched = ai.get("contexts_touched") or []
    if context_id not in touched:
        touched.append(context_id)
        ai["contexts_touched"] = touched
    detail = ai.get("contexts_touched_detail") or []
    detail.append(
        {
            "ctx_id": context_id,
            "reused": bool(reused),
            "scope": scope,
            "queries": list(queries or []),
            "keywords": list(keywords or []),
            "entities": list(entities or []),
        }
    )
    ai["contexts_touched_detail"] = detail


def _active_context_id() -> str:
    """Return the currently-active context entity id, or empty string.

    Source of truth: _STATE.active_intent['active_context_id']. Emit
    sites (declare_intent / declare_operation / kg_search) update this
    on each invocation under most-recent-emit-wins precedence. Writers
    (`_add_memory_internal`, `tool_kg_declare_entity`, `tool_kg_add`)
    read it to stamp `created_under` edges.
    """
    try:
        if _STATE.active_intent is None:
            return ""
        return _STATE.active_intent.get("active_context_id", "") or ""
    except Exception:
        return ""


def _get_context_views_collection(create: bool = True):
    """Get or create the per-view Chroma collection backing context entities.

    Pinned to cosine distance (so ``similarity = 1 - distance`` holds, as
    required by the MaxSim math -- see Khattab & Zaharia 2020).
    """
    try:
        client = chromadb.PersistentClient(path=_STATE.config.palace_path)
        if create:
            return client.get_or_create_collection(
                CONTEXT_VIEWS_COLLECTION, metadata=_CHROMA_METADATA
            )
        return client.get_collection(CONTEXT_VIEWS_COLLECTION)
    except Exception:
        if create:
            try:
                client = chromadb.PersistentClient(path=_STATE.config.palace_path)
                return client.create_collection(CONTEXT_VIEWS_COLLECTION, metadata=_CHROMA_METADATA)
            except Exception:
                return None
        return None


def _mint_context_entity_id(views: list) -> str:
    """Slice 5 2026-04-28: integer ``ctx_<N>`` id (~5 chars vs ~45 prior).

    Counter derives from MAX existing ``ctx_<int>`` id + 1 in the entities
    table -- zero-state, restart-safe, no new schema needed. Old composite
    ``ctx_<10hex>_<ns>_<6hex>`` ids in stored references continue to work
    as opaque strings; only newly-minted ids use the short integer form.
    Per Adrian's design lock 2026-04-26/28 id-design discussion: id is a
    pointer not a value -- short integer maximizes token efficiency,
    summary co-render carries the meaning at render time.

    The ``views`` parameter is preserved for call-site compatibility but
    no longer affects the id (the digest-of-views in the prior format was
    stable-ish but never used downstream -- context dedup goes through
    MaxSim against view embeddings, not id-based grouping).

    Race-safety note: SQLite read-then-write here means two near-
    simultaneous mints could in principle return the same N. Acceptable
    for the current single-writer mempalace deployment; if multi-writer
    materializes, swap to an atomic ``INSERT ... RETURNING rowid`` on a
    dedicated counter table.
    """
    import time

    try:
        conn = _STATE.kg._conn()
        # Match only ids whose post-prefix portion is purely digits.
        # GLOB '[0-9]*' tolerates leading digits but allows trailing
        # garbage; the regex-style match below adds a strict tail check
        # via length comparison.
        rows = conn.execute(
            "SELECT id FROM entities WHERE id LIKE 'ctx_%' AND substr(id, 5) GLOB '[0-9]*'"
        ).fetchall()
        max_seq = 0
        for row in rows:
            tail = row[0][4:]
            if tail.isdigit():
                n = int(tail)
                if n > max_seq:
                    max_seq = n
        next_seq = max_seq + 1
    except Exception:
        # Storage error fallback: monotonic ms timestamp avoids id reuse
        # without requiring the entities table. Cosmetically longer than
        # the integer form but rare and safe.
        next_seq = time.time_ns() // 1_000_000
    return f"ctx_{next_seq}"


def _compute_context_maxsim(current_views: list, candidate_context_ids: list, col) -> dict:
    """Multi-view max-of-max similarity per candidate context.

    Thin wrapper over :func:`mempalace.scoring.multi_view_max_sim`. The
    aggregation is **max-of-max** (best-view similarity) -- captures
    "any single strong overlap counts," the design intent that
    ``_check_entity_similarity_multiview`` already shipped and that the
    2026-04-26 audit promoted to a single source of truth in
    ``scoring.py``. Empirically validated on the 7-op live-palace
    experiment (2026-04-26): mean-of-max dropped Op B and Op G to zero
    similar_to edges; max-of-max captures the verbatim/file overlap.

    The previous mean-of-max formula (``sum / |views|``) lived here from
    P1 of the context-as-entity redesign. It was retired 2026-04-26
    after the empirical experiment + literature audit; see the module
    docstring on ``scoring.multi_view_max_sim`` for the rationale.
    """
    from .scoring import multi_view_max_sim

    return multi_view_max_sim(
        current_views,
        candidate_context_ids,
        col,
        where_key="context_id",
    )


def context_lookup_or_create(
    queries,
    keywords=None,
    entities=None,
    agent: str = None,
    summary=None,
    *,
    t_reuse: float = None,
    t_similar: float = None,
) -> tuple:
    """Reuse an existing context by MaxSim or mint a new first-class context entity.

    Returns ``(context_id, reused, max_sim)``.

    Used by the three emit sites (``tool_declare_intent``,
    ``tool_declare_operation``, ``tool_kg_search``). Every other writer
    references the active context via ``created_under`` instead of creating
    its own -- only these three sites emit contexts.

    ``summary`` is the structured ``{what, why, scope?}`` dict the writer
    supplied in ``context.summary`` (validated by ``validate_context``
    with ``require_summary=True``). When present, the rendered prose
    form becomes the new context entity's canonical description so
    retrieval and gardener flagging see a real WHAT+WHY anchor instead
    of a queries[0] truncation. When absent (read-side or legacy
    callers), falls back to ``views[0][:200]`` as before -- Adrian's
    design lock 2026-04-25 makes summary mandatory at every write
    boundary, so this fallback exists only to keep read paths and
    in-flight callers working during the rollout.

    The threshold branches use DUAL aggregation (BIRCH-inspired,
    Zhang et al. 1996; max-of-max from ColBERT 2020 + CRISP 2025):
      - reuse decision uses **min-of-max** -- every view must align,
        capturing "these are the same context." Threshold: t_reuse.
      - similar_to decision uses **max-of-max** -- any view aligns,
        capturing "they share at least one anchor." Threshold:
        t_similar.

    The two aggregations differ only in the final reduction over
    per-view-max scores; min-of-max blocks the over-clustering bug
    discovered 2026-04-26 where Op B (1 verbatim of A + 2 unrelated
    queries) was collapsing into A's context because the verbatim
    view scored 1.0. Now Op B's other-views drag min to ~0.4 → no
    reuse → keeps own context with similar_to edge to A.

    Branches:
      - reuse_score ≥ t_reuse        → return existing, no write.
      - reuse_score < t_reuse AND
        link_score  ≥ t_similar      → create new + write `similar_to`.
      - link_score  < t_similar      → create new, no `similar_to`.
    """
    t_reuse = CONTEXT_REUSE_THRESHOLD if t_reuse is None else float(t_reuse)
    t_similar = CONTEXT_SIMILAR_THRESHOLD if t_similar is None else float(t_similar)
    views = [q.strip() for q in (queries or []) if isinstance(q, str) and q.strip()]
    if not views:
        return "", False, 0.0

    col = _get_context_views_collection(create=True)
    if col is None:
        return "", False, 0.0

    # 1. Collect candidate context ids -- top-K per-view neighbours, union'd.
    candidate_ids: set = set()
    try:
        count = col.count()
    except Exception:
        count = 0
    if count > 0:
        for view in views:
            try:
                res = col.query(
                    query_texts=[view],
                    n_results=min(20, count),
                    include=["metadatas"],
                )
                if res.get("metadatas") and res["metadatas"] and res["metadatas"][0]:
                    for meta in res["metadatas"][0]:
                        if meta and meta.get("context_id"):
                            candidate_ids.add(meta["context_id"])
            except Exception:
                continue

    # 2. Dual-aggregation MaxSim -- min-of-max for reuse, max-of-max for
    # similar_to. Both derived from ONE Chroma pass via
    # scoring.multi_view_minmax_sim.
    best_reuse_id, best_reuse_sim = None, 0.0
    best_link_id, best_link_sim = None, 0.0
    if candidate_ids:
        from .scoring import multi_view_minmax_sim

        pairs = multi_view_minmax_sim(views, list(candidate_ids), col, where_key="context_id")
        if pairs:
            # argmax over min-of-max for reuse
            best_reuse_id, (best_reuse_sim, _) = max(pairs.items(), key=lambda kv: kv[1][0])
            # argmax over max-of-max for similar_to
            best_link_id, (_, best_link_sim) = max(pairs.items(), key=lambda kv: kv[1][1])

    # 3. Reuse branch -- min-of-max ≥ t_reuse (all views align).
    if best_reuse_id and best_reuse_sim >= t_reuse:
        # Touch last_touched by re-adding the entity (add_entity is upsert).
        try:
            existing = _STATE.kg.get_entity(best_reuse_id)
            if existing and existing.get("kind") == "context":
                _STATE.kg.add_entity(
                    best_reuse_id,
                    kind="context",
                    content=existing.get("content", "") or (views[0][:200]),
                    importance=existing.get("importance", 3) or 3,
                    properties=existing.get("properties", {}) or {},
                )
        except Exception:
            pass
        return best_reuse_id, True, float(best_reuse_sim)

    # 4. Mint a fresh context entity.
    new_cid = _mint_context_entity_id(views)
    props = {
        "queries": list(views),
        "keywords": [k for k in (keywords or []) if isinstance(k, str) and k.strip()],
        "entities": [e for e in (entities or []) if isinstance(e, str) and e.strip()],
        "agent": agent or "",
    }
    # Description: prefer the rendered summary prose (real WHAT+WHY),
    # fall back to views[0] truncation only if summary is missing
    # (read-side / legacy paths). Persist the structured summary dict
    # in properties so the gardener can patch fields independently.
    description = None
    if isinstance(summary, dict):
        try:
            from .knowledge_graph import serialize_summary_for_embedding

            description = serialize_summary_for_embedding(summary) or None
            props["summary"] = {
                k: v
                for k, v in summary.items()
                if k in ("what", "why", "scope") and isinstance(v, str)
            }
        except Exception:
            description = None
    if not description:
        description = views[0][:200] if views else "context"
    try:
        _STATE.kg.add_entity(
            new_cid,
            kind="context",
            content=description,
            importance=3,
            properties=props,
        )
    except Exception:
        return "", False, float(best_link_sim)
    # Persist the caller-supplied keywords in entity_keywords so P2's BM25
    # channel can IDF them without re-extracting.
    if props["keywords"]:
        try:
            _STATE.kg.add_entity_keywords(new_cid, props["keywords"], source="caller")
        except Exception:
            pass
    # 5. Store N view vectors.
    try:
        ids = [f"{new_cid}_v{i}" for i in range(len(views))]
        metas = [{"context_id": new_cid, "view_index": i} for i in range(len(views))]
        col.upsert(ids=ids, documents=views, metadatas=metas)
    except Exception:
        # View persistence failed -- the entity row still exists but the
        # context won't be lookup-able. Mark it so ops can find it.
        try:
            bad_props = dict(props)
            bad_props["_views_persisted"] = False
            _STATE.kg.add_entity(
                new_cid,
                kind="context",
                content=description,
                importance=3,
                properties=bad_props,
            )
        except Exception:
            pass

    # 6. similar_to edge -- max-of-max ≥ t_similar (any view aligns).
    # Reuse already failed (else we returned at step 3), so this branch
    # always writes when the link threshold is met. The triples table
    # has no generic properties column in P1 -- we stuff the MaxSim into
    # the `confidence` field (semantically compatible: a similar_to
    # edge's "confidence" is exactly how similar the two contexts are).
    if best_link_id and best_link_sim >= t_similar:
        try:
            _STATE.kg.add_triple(
                new_cid,
                "similar_to",
                best_link_id,
                confidence=round(float(best_link_sim), 4),
            )
        except Exception as exc:
            # similar_to is in _TRIPLE_SKIP_PREDICATES so this CANNOT
            # be a missing-statement issue; any failure here is a real
            # DB/constraint/programming bug. Log loudly so it surfaces
            # in operator logs, but do NOT crash declare_* -- a missing
            # similar_to edge degrades retrieval neighbourhood quality
            # for one context, not the entire intent flow. (Silent
            # bare-except retired 2026-04-25 per Adrian's rule:
            # "ensure this crashes, and is not silently omitted".)
            logger.warning(
                "context_lookup_or_create: similar_to write failed (%s -> %s, sim=%.4f): %s",
                new_cid,
                best_link_id,
                best_link_sim,
                exc,
            )

    # Report the link score as the "max_sim" return value -- that's the
    # informative number for callers (telemetry, debug logs).
    return new_cid, False, float(best_link_sim)


def _reset_declared_entities():
    """Reset the session's declared entities set (called on compact/clear/restart)."""
    _STATE.declared_entities = set()
    _STATE.session_id = ""


def _check_entity_similarity_multiview(
    views: list,
    kind_filter: str = None,
    exclude_id: str = None,
    threshold: float = None,
):
    """Multi-view collision detection (P4.2, refined in P5.2).

    Each view is queried independently against the entity collection; the
    per-view ranked candidates are merged via Reciprocal Rank Fusion for
    ORDERING, then filtered by the centralized max-of-max similarity
    score (``scoring.multi_view_max_sim``).

    A hit is reported as a collision when its **best single-view
    similarity** (max-of-max) is above threshold, so one strong match
    still flags but multi-view catches what single-vector cosine misses.

    2026-04-26: scoring delegated to ``scoring.multi_view_max_sim`` --
    the centralized max-of-max helper that also backs
    ``_compute_context_maxsim``. Discovery + RRF ranking + hydration
    stay here because they're entity-collision-specific; only the
    aggregation formula was duplicated and that's now shared. Adrian's
    DRY directive 2026-04-26.

    Logical entity ids are read from metadata.entity_id (no id
    string splitting). Returns the same shape as _check_entity_similarity
    for drop-in compatibility.
    """
    from .scoring import multi_view_max_sim, rrf_merge

    threshold = threshold or ENTITY_SIMILARITY_THRESHOLD
    ecol = _get_entity_collection(create=False)
    if not ecol or not views:
        return []
    try:
        count = ecol.count()
        if count == 0:
            return []
        # Discovery + RRF inputs: scan per-view top-K, build
        # per-view ranked lists for fusion AND collect per-id
        # hydration metadata (first-seen doc/meta is fine -- the
        # response just needs *some* description for each surfaced
        # entity, not necessarily the max-view's). The actual
        # collision score is computed via the centralized helper
        # below to keep the formula in one place.
        per_view_lists = {}
        per_id_meta: dict = {}  # logical entity_id -> (doc, meta)
        for vi, view in enumerate(views):
            if not view or not view.strip():
                continue
            kwargs = {
                "query_texts": [view],
                "n_results": min(20, count),
                "include": ["documents", "metadatas", "distances"],
            }
            results = ecol.query(**kwargs)
            if not (results.get("ids") and results["ids"][0]):
                continue
            view_candidates = []
            for i, raw_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] or {}
                # read logical entity id from metadata. Defensive fallback
                # to raw_id only for records the migration hasn't touched yet
                # (will disappear once _migrate_entity_views_schema runs).
                logical_id = meta.get("entity_id") or raw_id
                if logical_id == exclude_id:
                    continue
                if kind_filter and meta.get("kind") != kind_filter:
                    continue
                dist = results["distances"][0][i]
                sim = round(1 - dist, 3)
                doc = results["documents"][0][i] or ""
                view_candidates.append((sim, doc, logical_id))
                if logical_id not in per_id_meta:
                    per_id_meta[logical_id] = (doc, meta)
            if view_candidates:
                per_view_lists[f"cosine_{vi}"] = view_candidates

        if not per_view_lists:
            return []

        # Centralized max-of-max scoring -- single source of truth for
        # the formula. See scoring.multi_view_max_sim.
        candidate_ids = list(per_id_meta.keys())
        per_id_score = multi_view_max_sim(
            views,
            candidate_ids,
            ecol,
            where_key="entity_id",
        )

        rrf_scores, _cm, _attr = rrf_merge(per_view_lists)
        # Order by RRF, but only emit entities whose max-of-max
        # similarity is above threshold.
        similar = []
        for eid, _rrf in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            sim = per_id_score.get(eid, 0.0)
            if sim < threshold:
                continue
            doc_meta = per_id_meta.get(eid)
            if not doc_meta:
                continue
            doc, meta = doc_meta
            similar.append(
                {
                    "entity_id": eid,
                    "name": meta.get("name", eid),
                    "description": doc,
                    "similarity": round(float(sim), 3),
                    "importance": meta.get("importance", 3),
                }
            )
        return similar
    except Exception:
        return []


def _check_entity_similarity(
    description: str,
    kind_filter: str = None,
    exclude_id: str = None,
    threshold: float = None,
):
    """Single-description collision check.

    Used by tool_kg_update_entity after a description-only update (we have
    one new description and want to know whether it still collides). For
    kg_declare_entity (multi-view Context), use
    _check_entity_similarity_multiview which RRF-merges per-view rankings.
    """
    threshold = threshold or ENTITY_SIMILARITY_THRESHOLD
    ecol = _get_entity_collection(create=False)
    if not ecol:
        return []
    try:
        count = ecol.count()
        if count == 0:
            return []
        query_kwargs = {
            "query_texts": [description],
            "n_results": min(10, count),
            "include": ["documents", "metadatas", "distances"],
        }
        # Kind-scoped collision: only check against same kind (was a
        # silent no-op before, the empty dict dropped the filter entirely).
        if kind_filter:
            query_kwargs["where"] = {"kind": kind_filter}
        results = ecol.query(**query_kwargs)
        similar = []
        if results["ids"] and results["ids"][0]:
            for i, eid in enumerate(results["ids"][0]):
                if eid == exclude_id:
                    continue
                dist = results["distances"][0][i]
                similarity = round(1 - dist, 3)
                if similarity >= threshold:
                    meta = results["metadatas"][0][i]
                    doc = results["documents"][0][i]
                    similar.append(
                        {
                            "entity_id": eid,
                            "name": meta.get("name", eid),
                            "description": doc,
                            "similarity": similarity,
                            "importance": meta.get("importance", 3),
                        }
                    )
        return similar
    except Exception:
        return []


def _create_entity(
    name: str,
    kind: str = "entity",
    content: str = "",
    importance: int = 3,
    properties: dict = None,
    added_by: str = None,
    embed_text: str = None,
):
    """Create an entity in BOTH SQLite AND ChromaDB. Use this instead of _STATE.kg.add_entity directly.

    Args:
        content: long-form text describing this entity. (Renamed from
                ``description`` 2026-04-29 -- migration 023 dropped the
                legacy column; the kwarg followed.)
        embed_text: Optional override for what gets embedded in ChromaDB.
                    If None, uses content. Use for execution entities
                    where you want content-only embedding (no summary).
    """
    from .knowledge_graph import normalize_entity_name

    # pass provenance to SQLite
    _prov_session = _STATE.session_id or ""
    _prov_intent = _STATE.active_intent.get("intent_id", "") if _STATE.active_intent else ""
    eid = _STATE.kg.add_entity(
        name,
        kind=kind,
        content=content,
        importance=importance,
        properties=properties,
        session_id=_prov_session,
        intent_id=_prov_intent,
    )
    normalized = normalize_entity_name(name)
    _sync_entity_to_chromadb(
        normalized, name, embed_text or content, kind, importance, added_by=added_by
    )
    return eid


def _update_entity_chromadb_metadata(entity_id: str, **fields):
    """Patch metadata fields across every Chroma record bound to entity_id.

    FINDING-1 fix 2026-04-28: kg_update_entity used to call
    _sync_entity_to_chromadb after an importance change, which UPSERTs a
    SINGLE-view record under id=entity_id. For entities declared via
    kg_declare_entity, the canonical records live under {entity_id}__v0,
    {entity_id}__v1, ... with metadata.entity_id pointing back. The
    single-view upsert orphaned the multi-view records' importance
    metadata, so kg_query (which reads via where={'entity_id': eid})
    returned the OLD importance.

    This helper iterates BOTH lookup paths (multi-view via metadata,
    legacy single-id record), reads each record's existing metadata,
    merges the patch dict, and writes back via col.update. Chroma's
    update API requires a complete metadata dict per record because it
    replaces wholesale; we read-then-merge to preserve every other key.

    No-op if the field set is empty or no records are found. Silent on
    Chroma errors (best-effort -- the SQLite write already succeeded
    so a transient Chroma failure shouldn't fail the user-visible
    update).
    """
    if not fields:
        return
    try:
        col = _get_entity_collection(create=False)
    except Exception:
        return
    if col is None:
        return

    ids: list[str] = []
    metas: list[dict] = []

    # Multi-view lookup (post-P5.2 entities).
    try:
        got = col.get(
            where={"entity_id": entity_id},
            include=["metadatas"],
        )
        if got and got.get("ids"):
            for rid, meta in zip(got["ids"], got.get("metadatas") or []):
                if rid in ids:
                    continue
                ids.append(rid)
                metas.append(dict(meta or {}))
    except Exception:
        pass

    # Legacy / single-id record lookup (covers entities written via
    # _sync_entity_to_chromadb directly, e.g. execution + gotcha entities).
    try:
        got = col.get(ids=[entity_id], include=["metadatas"])
        if got and got.get("ids"):
            for rid, meta in zip(got["ids"], got.get("metadatas") or []):
                if rid in ids:
                    continue
                ids.append(rid)
                metas.append(dict(meta or {}))
    except Exception:
        pass

    if not ids:
        return

    # Merge patch into each existing metadata dict, preserving other keys.
    new_metas = []
    for m in metas:
        merged = dict(m)
        merged.update(fields)
        new_metas.append(merged)

    try:
        col.update(ids=ids, metadatas=new_metas)
    except Exception:
        pass


def _sync_entity_to_chromadb(
    entity_id: str, name: str, description: str, kind: str, importance: int, added_by: str = None
):
    """Single-description Chroma sync for internal bookkeeping entities.

    Used by intent.py finalize (execution entities, gotcha entities) and by
    tool_kg_update_entity after a description-only update -- both cases
    naturally have one description and don't carry a multi-view Context.
    For Context-driven entity declarations, use _sync_entity_views_to_chromadb
    (multi-vector storage under '{entity_id}::view_N').

    kind='operation': as of 2026-04-27 (commit b905373), args_summary is
    a MANDATORY parametrized-core fingerprint at declare_operation time,
    so the rendered description ('{tool} op: {args_summary}') is
    meaningful enough to embed. Ops now participate in Channel A cosine
    retrieval via this single description. They still skip the multi-view
    path (_sync_entity_views_to_chromadb below) -- there's no useful
    second view for an op beyond its parametrized fingerprint.
    Cf. arXiv 2512.18950 (Operation tier).
    """
    ecol = _get_entity_collection(create=True)
    if not ecol:
        return
    meta = {
        "name": name,
        "kind": kind,
        "importance": importance,
        "last_touched": datetime.now().isoformat(),
    }
    if added_by:
        meta["added_by"] = added_by
    # provenance auto-injection on entity Chroma records
    if _STATE.session_id:
        meta["session_id"] = _STATE.session_id
    if _STATE.active_intent and isinstance(_STATE.active_intent, dict):
        meta["intent_id"] = _STATE.active_intent.get("intent_id", "")
    ecol.upsert(
        ids=[entity_id],
        documents=[description],
        metadatas=[meta],
    )


def _sync_entity_views_to_chromadb(
    entity_id: str,
    name: str,
    views: list,
    kind: str,
    importance: int,
    added_by: str = None,
):
    """Multi-view sync to the entity ChromaDB collection (P4.2, refined in P5.2).

    Each view is stored as a separate record under '{entity_id}__v{N}' AND
    every record carries metadata.entity_id explicitly. Readers group views
    by the metadata field (col.get(where={'entity_id': X})) -- NOT by parsing
    ids -- so the separator choice is cosmetic, not load-bearing.

    The '__v' separator is deliberately chosen because it cannot appear
    inside a normalized_entity_name (which uses single-underscore segments
    only), making the literal id unambiguous for humans skimming the db.

    kind='operation' deliberately skips the multi-view path: the
    parametrized args_summary fingerprint is the ONLY useful "view" of
    an op, and it's already embedded via the single-description path
    in _sync_entity_to_chromadb above (2026-04-27 redesign). No second
    perspective adds signal -- splitting "git commit -m {commit_message}"
    into multiple views would just re-embed the same fingerprint.
    Cf. arXiv 2512.18950 (Operation tier).
    """
    if kind == "operation":
        return
    ecol = _get_entity_collection(create=True)
    if not ecol or not views:
        return
    cleaned = [v for v in views if isinstance(v, str) and v.strip()]
    if not cleaned:
        return
    now_iso = datetime.now().isoformat()
    base_meta = {
        "name": name,
        "kind": kind,
        "importance": importance,
        "last_touched": now_iso,
    }
    if added_by:
        base_meta["added_by"] = added_by
    # provenance auto-injection on multi-view entity records
    if _STATE.session_id:
        base_meta["session_id"] = _STATE.session_id
    if _STATE.active_intent and isinstance(_STATE.active_intent, dict):
        base_meta["intent_id"] = _STATE.active_intent.get("intent_id", "")

    ids, docs, metas = [], [], []
    for i, view in enumerate(cleaned):
        ids.append(f"{entity_id}__v{i}")
        docs.append(view)
        m = dict(base_meta)
        m["view_index"] = i
        m["entity_id"] = entity_id  # canonical reverse lookup
        metas.append(m)
    ecol.upsert(ids=ids, documents=docs, metadatas=metas)


VALID_CARDINALITIES = {"many-to-many", "many-to-one", "one-to-many", "one-to-one"}


# ── Phase 2: tool_kg_declare_entity moved to tool_mutate.py.

# ── Phase 2: tool_kg_update_entity moved to tool_mutate.py.

# ── Phase 2: tool_kg_merge_entities moved to tool_mutate.py.

# tool_kg_update_predicate_constraints removed: merged into tool_kg_update_entity.
# Call kg_update_entity(entity=predicate, properties={"constraints": {...}}).


# ── Phase 2: tool_kg_list_declared moved to tool_read.py.

# tool_kg_entity_info removed: use kg_query(entity=..., direction="both").


# ==================== INTENT DECLARATION ====================

# _STATE.active_intent holds the session-level active intent (at most one).
# Defaults to None on ServerState construction -- no explicit init needed here.
_INTENT_STATE_DIR = Path(os.path.expanduser("~/.mempalace/hook_state"))


# Intent functions are in intent.py; init() is called after module globals are set.
# Aliases so TOOLS dispatch continues to work:
# ── Phase 2: tool_declare_intent moved to tool_lifecycle.py.

# ── Phase 2: tool_active_intent moved to tool_lifecycle.py.

# ── Phase 2: tool_extend_intent moved to tool_lifecycle.py.


def tool_declare_operation(*args, **kwargs):
    return intent.tool_declare_operation(*args, **kwargs)


# ── Phase 2: tool_declare_user_intents moved to tool_lifecycle.py.

# ── Phase 2: tool_resolve_conflicts moved to tool_lifecycle.py.

# ── Phase 2: tool_finalize_intent moved to tool_lifecycle.py.

# ── Phase 2: tool_extend_feedback moved to tool_lifecycle.py.

# ==================== AGENT DIARY ====================


# ── Phase 2: tool_diary_write moved to tool_mutate.py.

# ── Phase 2: tool_diary_read moved to tool_read.py.

# ==================== MCP PROTOCOL ====================

# ── Shared Context schema (Adrian's design lock 2026-04-25) ──
#
# Every context-taking MCP tool speaks the SAME Context object so
# callers learn one shape and reuse it everywhere. The structured
# summary {what, why, scope?} is part of that shape. Two variants:
#
# - _CONTEXT_SCHEMA: write-side (entities 1-10, summary REQUIRED).
#   Used by declare_intent, declare_operation, kg_declare_entity,
#   kg_add, kg_add_batch, kg_update_entity, diary_write.
# - _CONTEXT_SCHEMA_READ: read-side (entities optional, summary
#   optional). Used by kg_search and other read-time fingerprints
#   where the caller's WHAT/WHY may not yet be authored.
#
# Both reference _SUMMARY_SUBSCHEMA below so the structured-summary
# shape is defined exactly once. Tool definitions reference these
# constants by name instead of duplicating the inline literal -- the
# canonical shape lives here, and adding a field touches one place.

_SUMMARY_SUBSCHEMA = {
    "type": "object",
    "description": (
        "Structured summary {what, why, scope?}. "
        "what: noun phrase naming the subject (≥5 chars). "
        "why: purpose / role / claim (≥15 chars). "
        "scope: optional temporal / domain qualifier (≤100 chars). "
        "Rendered prose form ≤280 chars. Field-level + length-bounded "
        "validation; no regex; no auto-derive (Adrian's design lock 2026-04-25)."
    ),
    "properties": {
        "what": {"type": "string"},
        "why": {"type": "string"},
        "scope": {"type": "string"},
    },
    "required": ["what", "why"],
}

_CONTEXT_SCHEMA = {
    "type": "object",
    "description": (
        "MANDATORY Context fingerprint -- shared across every "
        "context-taking tool.\n"
        "  queries:  list[str] (2-5)  perspectives -- each becomes a cosine view.\n"
        "  keywords: list[str] (2-5)  caller-provided exact terms (no auto-extract).\n"
        "  entities: list[str] (1-10) related entity ids; graph anchors.\n"
        "  summary:  dict {what, why, scope?} -- structured WHAT+WHY+SCOPE? "
        "anchor; required on every WRITE.\n"
        'Example: context={"queries": ["DSpot platform server", '
        '"paperclip backend on :3100"], "keywords": ["dspot", "paperclip", '
        '"server", "port-3100"], "entities": ["dspot_infra"], '
        '"summary": {"what": "DSpot platform", "why": "hosts the paperclip '
        'backend on port 3100; central API surface for the team", '
        '"scope": "production"}}'
    ),
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 5,
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 5,
        },
        "entities": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 10,
        },
        "summary": _SUMMARY_SUBSCHEMA,
    },
    "required": ["queries", "keywords", "entities", "summary"],
}

# Read-side variant: same shape, but entities and summary are optional.
# Used where the caller is searching/looking up rather than authoring;
# they may not yet have a structured summary to commit. validate_context
# at read sites leaves require_summary=False.
_CONTEXT_SCHEMA_READ = {
    "type": "object",
    "description": (
        "Context fingerprint for read-time queries (kg_search, kg_query). "
        "Same shape as the write Context but summary is optional -- the "
        "caller may not yet have authored a WHAT+WHY for what they're "
        "looking for. Pass it when known so the search context can be "
        "indexed; omit when not."
    ),
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 5,
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 5,
        },
        "entities": {
            "type": "array",
            "items": {"type": "string"},
        },
        "summary": _SUMMARY_SUBSCHEMA,
    },
    "required": ["queries", "keywords"],
}


# ── Phase 2: bucket re-imports for TOOLS dispatch back-compat ──────────
# Handler bodies for these tools have moved into the bucket files. The
# import-back keeps the original module-level names available so the
# TOOLS dispatch (below) and any internal callers keep working. The
# import is placed AFTER all helpers and read-handlers it depends on
# are defined, so the bucket file's circular `from mempalace.mcp_server
# import _STATE, ...` resolves cleanly.
from mempalace.tool_read import (  # noqa: E402, F401
    tool_diary_read,
    tool_kg_list_declared,
    tool_kg_query,
    tool_kg_search,
    tool_kg_stats,
    tool_kg_timeline,
)
from mempalace.tool_mutate import (  # noqa: E402, F401
    tool_diary_write,
    tool_kg_add,
    tool_kg_add_batch,
    tool_kg_declare_entity,
    tool_kg_delete_entity,
    tool_kg_invalidate,
    tool_kg_merge_entities,
    tool_kg_update_entity,
)
from mempalace.tool_lifecycle import (  # noqa: E402, F401
    tool_active_intent,
    tool_declare_intent,
    tool_declare_user_intents,
    tool_extend_feedback,
    tool_extend_intent,
    tool_finalize_intent,
    tool_resolve_conflicts,
    tool_wake_up,
)


TOOLS = {
    "mempalace_kg_query": {
        "description": "Query the knowledge graph for an entity's relationships AND its own content by EXACT entity name. Returns typed facts (edges) with temporal validity PLUS a `details` block with the entity's kind/summary/content/importance pulled from its representative Chroma record. Retrieval-bookkeeping edges (rated_useful, rated_irrelevant, surfaced) are omitted by default; pass include_context_edges=true to see them. Supports batch queries: pass comma-separated names to query multiple entities in one call. Use kg_search instead if you don't know the exact entity name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "Entity to query (e.g. 'Max', 'MyProject'). Supports comma-separated batch: 'Max, Alice, MyProject' returns results keyed by entity.",
                },
                "as_of": {
                    "type": "string",
                    "description": "Date filter -- only facts valid at this date (YYYY-MM-DD, optional)",
                },
                "direction": {
                    "type": "string",
                    "description": "outgoing (entity→?), incoming (?→entity), or both (default: both)",
                },
                "include_context_edges": {
                    "type": "boolean",
                    "description": "Include retrieval-bookkeeping edges (rated_useful, rated_irrelevant, surfaced) in the facts list. Default false -- they are filtered out because they drown domain edges in per-context noise. Set true for retrieval audits. When filtered, hidden_context_edges (or total_hidden_context_edges in batch mode) reports how many were hidden.",
                },
            },
            "required": ["entity"],
        },
        "handler": tool_kg_query,
    },
    "mempalace_kg_search": {
        "description": (
            "Unified search -- records (prose) + entities (KG nodes) in one "
            "call (Context-based). Speaks the unified Context object: "
            "queries drive Channel A multi-view cosine, keywords drive Channel C "
            "(caller-provided exact terms -- no auto-extraction), entities seed "
            "Channel B graph BFS. Cross-collection Reciprocal Rank Fusion across "
            "all channels. Each result carries source='memory'|'entity' with "
            "type-specific fields (memories: text; entities: name/kind/"
            "description/edges). Unlike kg_query (exact entity ID), this "
            "fuzzy-matches across your whole memory palace."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "context": _CONTEXT_SCHEMA_READ,
                "limit": {
                    "type": "integer",
                    "description": "Max results across memories+entities (default 10; adaptive-K may trim if scores drop off).",
                },
                "kind": {
                    "type": "string",
                    "description": "Optional entity kind filter. When set, scopes to entities only.",
                    "enum": ["entity", "predicate", "class", "literal"],
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["hybrid", "similarity"],
                    "description": "'hybrid' (default) = RRF + hybrid_score. 'similarity' = pure cosine.",
                },
                "agent": {
                    "type": "string",
                    "description": "Your agent entity name for affinity scoring.",
                },
                "time_window": {
                    "type": "object",
                    "description": (
                        "optional temporal scoping. Items inside the window "
                        "get a scoring boost; items outside still appear but rank lower "
                        "(soft decay, NOT a hard filter). Example: "
                        '{"start": "2026-04-15", "end": "2026-04-17"}'
                    ),
                    "properties": {
                        "start": {
                            "type": "string",
                            "description": "Start date (YYYY-MM-DD). Items after this date get boosted.",
                        },
                        "end": {
                            "type": "string",
                            "description": "End date (YYYY-MM-DD). Items before this date get boosted.",
                        },
                    },
                },
            },
            "required": ["context", "agent"],
        },
        "handler": tool_kg_search,
    },
    "mempalace_kg_add": {
        "description": (
            "Add a fact to the knowledge graph (Context mandatory). "
            "Subject → predicate → object plus a Context fingerprint that captures "
            "WHY the edge is being added. The Context's view vectors are persisted; "
            "future feedback (found_useful etc.) applies by MaxSim against this "
            "fingerprint. E.g. ('Max', 'started_school', 'Year 7', valid_from='2026-09-01', "
            "context={'queries': ['Max enrolled in Year 7', 'school start 2026'], "
            "'keywords': ['max', 'school', 'year-7']})."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "The entity doing/being something"},
                "predicate": {
                    "type": "string",
                    "description": "The relationship type (e.g. 'loves', 'works_on', 'daughter_of')",
                },
                "object": {"type": "string", "description": "The entity being connected to"},
                "context": _CONTEXT_SCHEMA,
                "agent": {
                    "type": "string",
                    "description": (
                        "MANDATORY -- your declared agent entity name (is_a agent). "
                        "Every write operation is attributed to a declared agent; "
                        "undeclared agents are rejected up front with a declaration recipe."
                    ),
                },
                "valid_from": {
                    "type": "string",
                    "description": "When this became true (YYYY-MM-DD, optional)",
                },
                "statement": {
                    "type": "object",
                    "description": (
                        "Structured verbalization of the triple -- same dict "
                        "shape as context.summary: {what, why, scope?}. The "
                        "rendered prose form gets embedded into "
                        "mempalace_triples so the edge is a first-class "
                        "semantic-search result. Adrian's design lock "
                        "2026-04-25: edges follow the same dict-only contract "
                        "as records and entities. REQUIRED for every "
                        "predicate OUTSIDE the skip list (is_a, described_by, "
                        "executed_by, targeted, has_value, session_note_for, "
                        "derived_from, mentioned_in, found_useful, "
                        "found_irrelevant, evidenced_by); for skip-list "
                        "predicates statement may be omitted because those "
                        "edges are never embedded anyway. Example: "
                        '{"what": "Adrian lives in Warsaw", "why": "primary '
                        'residence; reflects current legal address", '
                        '"scope": "since 2019"}.'
                    ),
                    "properties": {
                        "what": {"type": "string"},
                        "why": {"type": "string"},
                        "scope": {"type": "string"},
                    },
                    "required": ["what", "why"],
                },
            },
            "required": ["subject", "predicate", "object", "context", "agent"],
        },
        "handler": tool_kg_add,
    },
    "mempalace_kg_add_batch": {
        "description": (
            "Add multiple KG triples in one call (same params as kg_add, "
            "batched). Each item in `triples` mirrors kg_add exactly: "
            "subject, predicate, object, context (mandatory per-item), "
            "plus optional statement and valid_from. `agent` is the only "
            "shared top-level field -- batches are single-author by design. "
            "Validates each triple independently -- partial success OK. "
            "Adrian's design lock 2026-04-28: nothing different just for "
            "being in a batch; per-item shape mirrors kg_add."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "triples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "predicate": {"type": "string"},
                            "object": {"type": "string"},
                            "context": _CONTEXT_SCHEMA,
                            "statement": {
                                "type": "object",
                                "description": (
                                    "Structured verbalization of the triple "
                                    "{what, why, scope?} -- same shape as "
                                    "kg_add.statement. REQUIRED for every "
                                    "predicate outside the skip list "
                                    "(is_a, described_by, evidenced_by, "
                                    "executed_by, targeted, has_value, "
                                    "session_note_for, derived_from, "
                                    "mentioned_in, found_useful, "
                                    "found_irrelevant)."
                                ),
                                "properties": {
                                    "what": {"type": "string"},
                                    "why": {"type": "string"},
                                    "scope": {"type": "string"},
                                },
                                "required": ["what", "why"],
                            },
                            "valid_from": {
                                "type": "string",
                                "description": (
                                    "When this fact became true (YYYY-MM-DD, optional)."
                                ),
                            },
                        },
                        "required": ["subject", "predicate", "object", "context"],
                    },
                    "description": (
                        "List of triples to add. Each item has the same "
                        "shape as kg_add (subject/predicate/object/context/"
                        "statement?/valid_from?)."
                    ),
                },
                "agent": {
                    "type": "string",
                    "description": ("MANDATORY -- declared agent attributing the whole batch."),
                },
            },
            "required": ["triples", "agent"],
        },
        "handler": tool_kg_add_batch,
    },
    "mempalace_kg_invalidate": {
        "description": "Mark a fact as no longer true. E.g. ankle injury resolved, job ended, moved house.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Entity"},
                "predicate": {"type": "string", "description": "Relationship"},
                "object": {"type": "string", "description": "Connected entity"},
                "ended": {
                    "type": "string",
                    "description": "When it stopped being true (YYYY-MM-DD, default: today)",
                },
                "agent": {
                    "type": "string",
                    "description": "MANDATORY -- declared agent invalidating this fact.",
                },
            },
            "required": ["subject", "predicate", "object", "agent"],
        },
        "handler": tool_kg_invalidate,
    },
    "mempalace_kg_timeline": {
        "description": "Chronological timeline of facts. Shows the story of an entity (or everything) in order.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "Entity to get timeline for (optional -- omit for full timeline)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max entries to return (default: 100). Most recent first.",
                    "minimum": 1,
                },
            },
        },
        "handler": tool_kg_timeline,
    },
    "mempalace_kg_stats": {
        "description": "Knowledge graph overview: entities, triples, current vs expired facts, relationship types.",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_kg_stats,
    },
    "mempalace_kg_declare_entity": {
        "description": (
            "REQUIRED before using any entity in kg_add. Declares an entity using "
            "the unified Context object. Each query is embedded as a "
            "separate Chroma vector; collision detection runs multi-view RRF; "
            "caller-provided keywords go into the keyword index; the Context's "
            "view vectors are persisted so future feedback applies by similarity.\n\n"
            "Kinds -- STRICT enum, exactly five values:\n"
            "  'entity'    -- concrete thing. DEFAULT for most new nodes.\n"
            "  'class'     -- category definition (other entities is_a this).\n"
            "  'predicate' -- relationship type for kg_add edges.\n"
            "  'literal'   -- raw value.\n"
            "  'record'    -- prose record. Requires slug + `content` "
            "(verbatim text) + `added_by`. `name` is auto-computed. "
            "Use `entity`+`predicate` to link the record to another entity.\n"
            "If the value you want isn't in the enum, it's a domain "
            "class, not a kind -- declare the node with kind='entity' and "
            "add an is_a edge to the class node. The enum of kinds is "
            "fixed; the set of classes is open and grows over time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Entity name (will be normalized). REQUIRED for kind=entity/class/predicate/literal. OMIT for kind='record' -- the record id is computed from added_by + slug.",
                },
                "context": _CONTEXT_SCHEMA,
                "kind": {
                    "type": "string",
                    "description": "Ontological role -- STRICT enum, exactly five values: 'entity' (concrete thing -- DEFAULT), 'class' (category/type definition that other entities is_a), 'predicate' (relationship type for kg_add edges), 'literal' (raw value), 'record' (prose memory; requires slug + content + added_by). If the value you want isn't in the enum, it's a domain class, not a kind -- pass kind='entity' and add an is_a edge to the class node (kg_add(subject=name, predicate='is_a', object=<that_class>)). The set of classes is open and grows over time; the set of kinds is fixed.",
                    "enum": ["entity", "predicate", "class", "literal", "record"],
                },
                "content": {
                    "type": "string",
                    "description": "Verbatim text -- REQUIRED for kind='record' (the actual record body). Ignored for non-record kinds; their canonical description is rendered from context.summary.",
                },
                "importance": {
                    "type": "integer",
                    "description": "1-5. 5=critical, 4=canonical, 3=default, 2=low, 1=junk.",
                    "minimum": 1,
                    "maximum": 5,
                },
                "properties": {
                    "type": "object",
                    "description": 'General-purpose metadata stored with the entity. For predicates: {"constraints": {"subject_kinds": [...], "object_kinds": [...], "subject_classes": [...], "object_classes": [...], "cardinality": "..."}} (ALL 5 fields REQUIRED). For intent types: {"rules_profile": {"slots": {...}, "tool_permissions": [...]}}.',
                },
                "added_by": {
                    "type": "string",
                    "description": "Agent who is declaring this entity. Must be a declared agent (is_a agent).",
                },
                "user_approved_star_scope": {
                    "type": "boolean",
                    "description": "NEVER set this to true unless the user JUST said YES in this conversation turn.",
                },
                # ── kind='record' specific ──
                "slug": {
                    "type": "string",
                    "description": "REQUIRED when kind='record'. 3-6 hyphenated words, unique per agent.",
                },
                "content_type": {
                    "type": "string",
                    "description": "Content classification for kind='record'.",
                    "enum": [
                        "fact",
                        "event",
                        "discovery",
                        "preference",
                        "advice",
                        "diary",
                    ],
                },
                "source_file": {
                    "type": "string",
                    "description": "Optional source attribution for kind='record'.",
                },
                "entity": {
                    "type": "string",
                    "description": "Entity name(s) to link this record to (comma-separated).",
                },
                "predicate": {
                    "type": "string",
                    "description": "Predicate for the entity→record link (default: described_by).",
                    "enum": [
                        "described_by",
                        "evidenced_by",
                        "derived_from",
                        "mentioned_in",
                        "session_note_for",
                    ],
                },
            },
            "required": ["context", "kind", "importance", "added_by"],
        },
        "handler": tool_kg_declare_entity,
    },
    "mempalace_kg_update_entity": {
        "description": (
            "Unified update for any entity (record or KG node). Pass only the fields "
            "you want to change.\n\n"
            "FOR ENTITIES (kind=entity/class/predicate/literal):\n"
            "  - summary: re-syncs to entity ChromaDB and runs collision distance check.\n"
            "  - properties: shallow-merged into existing properties. For predicates, "
            'use {"constraints": {...}} to update predicate constraints (validated).\n'
            "  - importance: 1-5.\n\n"
            "FOR RECORDS (kind='record'):\n"
            "  - content_type: in-place classification change (no re-embedding).\n"
            "  - importance: in-place importance change.\n"
            "  - summary: NOT supported -- use kg_delete_entity + kg_declare_entity "
            "to replace record content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "Entity ID or record ID (record_/diary_ prefix routes to record collection).",
                },
                "summary": {
                    "type": "object",
                    "description": (
                        "New structured summary {what, why, scope?} "
                        "(entities only). The dict is rendered to prose for "
                        "the entity's embedded text; validation is "
                        "field-level via knowledge_graph."
                        "coerce_summary_for_persist. Triggers collision "
                        "distance check on the rendered prose. Strings are "
                        "rejected by validate_summary."
                    ),
                    "properties": {
                        "what": {"type": "string"},
                        "why": {"type": "string"},
                        "scope": {"type": "string"},
                    },
                    "required": ["what", "why"],
                },
                "importance": {
                    "type": "integer",
                    "description": "New importance 1-5 (works for both entities and memories).",
                    "minimum": 1,
                    "maximum": 5,
                },
                "properties": {
                    "type": "object",
                    "description": 'Properties to merge into the entity. For predicates, use {"constraints": {"subject_kinds": [...], "object_kinds": [...], "subject_classes": [...], "object_classes": [...], "cardinality": "..."}}.',
                },
                "context": _CONTEXT_SCHEMA,
                "content_type": {
                    "type": "string",
                    "description": "(Records only) Content type classification.",
                    "enum": [
                        "fact",
                        "event",
                        "discovery",
                        "preference",
                        "advice",
                        "diary",
                    ],
                },
                "agent": {
                    "type": "string",
                    "description": "MANDATORY -- declared agent attributing this update.",
                },
            },
            "required": ["entity", "agent"],
        },
        "handler": tool_kg_update_entity,
    },
    "mempalace_kg_merge_entities": {
        "description": "Merge source entity into target. All edges rewritten, source becomes alias. Use when kg_declare_entity returns 'collision' and the entities are actually the same thing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Entity to merge FROM (will be soft-deleted)",
                },
                "target": {"type": "string", "description": "Entity to merge INTO (will be kept)"},
                "summary": {
                    "type": "object",
                    "description": (
                        "Optional structured summary {what, why, scope?} for "
                        "the merged entity (dict-only contract). Coerced and "
                        "rendered to prose internally; strings rejected by "
                        "validate_summary. Mirrors kg_update_entity.summary."
                    ),
                    "properties": {
                        "what": {"type": "string"},
                        "why": {"type": "string"},
                        "scope": {"type": "string"},
                    },
                    "required": ["what", "why"],
                },
                "agent": {
                    "type": "string",
                    "description": "MANDATORY -- declared agent attributing this merge.",
                },
            },
            "required": ["source", "target", "agent"],
        },
        "handler": tool_kg_merge_entities,
    },
    # mempalace_kg_update_predicate_constraints removed: merged into
    # mempalace_kg_update_entity. Call with properties={"constraints": {...}}.
    "mempalace_kg_list_declared": {
        "description": "List all entities declared in this session with their details (description, importance, edge count).",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_kg_list_declared,
    },
    "mempalace_declare_intent": {
        "description": (
            "Declare what you intend to do BEFORE doing it. Returns permissions + context. "
            "One active intent at a time -- new intent expires the previous. "
            "mempalace_* tools are always allowed regardless of intent.\n\n"
            "SLOT RULES -- most intent types require these slots:\n"
            '  paths:    (raw) directory patterns for Read/Grep/Glob scoping. E.g. ["D:/Flowsev/repo/**"]\n'
            '  commands: (raw) command patterns for Bash scoping. E.g. ["pytest", "git add"]\n'
            "  files:    file paths for Edit/Write scoping. Auto-declares existing files.\n"
            "  target:   entity names for context injection. Requires pre-declared entities.\n\n"
            "EXCEPTION: 'research' type needs NO paths -- it has unrestricted Read/Grep/Glob/WebFetch/WebSearch.\n"
            "Use 'research' when you genuinely don't know what you'll read. Other types require declaring paths.\n\n"
            "Check declared.intent_types from wake_up for available types + their tools.\n"
            "If a tool is blocked, the error teaches how to create or switch types."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "intent_type": {
                    "type": "string",
                    "description": (
                        "The intent type to declare (must be is_a intent_type). "
                        "Use the MOST SPECIFIC type available -- specific types carry domain rules. "
                        "Examples: 'edit_file', 'write_tests', 'deploy', 'run_tests', 'diagnose_failure'. "
                        "Broad types: 'inspect', 'modify', 'execute', 'communicate'."
                    ),
                },
                "slots": {
                    "type": "object",
                    "description": (
                        "Named slots filled with entity names, file paths, or command patterns. "
                        'Example for edit_file: {"files": ["src/auth.test.ts"], "paths": ["src/**"]}. '
                        'Example for execute: {"target": ["my_project"], "commands": ["pytest", "git add"], "paths": ["D:/Flowsev/mempalace/**"]}. '
                        'Example for inspect: {"subject": ["my_system"], "paths": ["D:/Flowsev/repo/**"]}. '
                        'Example for research: {"subject": ["some_topic"]} -- NO paths needed, broad reads allowed. '
                        "File slots auto-declare existing files. Command slots (raw) accept strings directly. "
                        "Other slots require pre-declared entities."
                    ),
                },
                "context": _CONTEXT_SCHEMA,
                "auto_declare_files": {
                    "type": "boolean",
                    "description": (
                        "Set to true when creating NEW files that don't exist on disk yet. "
                        "Existing files are auto-declared without this flag. Default: false."
                    ),
                },
                "agent": {
                    "type": "string",
                    "description": "Your agent entity name for affinity scoring in context injection. Examples: 'ga_agent', 'technical_lead_agent'.",
                },
                "budget": {
                    "type": "object",
                    "description": (
                        "MANDATORY tool call budget. Dict of tool_name -> max_calls. "
                        'E.g. {"Read": 5, "Edit": 3, "Bash": 2}. Must cover all tools you plan to use. '
                        "Keep budgets tight -- estimate minimum needed. "
                        "When exhausted, use mempalace_extend_intent to add more."
                    ),
                },
                "cause_id": {
                    "type": "string",
                    "description": (
                        "Optional parent-cause id (Slice B-3). Accepts either a "
                        "user-context entity id (kind='context' with at least one "
                        "fulfills_user_message edge - typically returned by "
                        "mempalace_declare_user_intents earlier this turn) OR a "
                        "Task entity id (kind='entity' with is_a Task - for non"
                        "-interactive agents whose work is caused by an external "
                        "issue rather than a user message). When supplied, the "
                        "handler validates the kind/class and writes a caused_by "
                        "edge from this activity-intent's context to cause_id. "
                        "Slice B-3 keeps cause_id optional for back-compat; "
                        "agents are expected to provide it whenever they have "
                        "either parent available. Telemetry tracks adoption."
                    ),
                },
            },
            "required": ["intent_type", "slots", "context", "agent", "budget"],
        },
        "handler": tool_declare_intent,
    },
    "mempalace_active_intent": {
        "description": "Return the current active intent -- type, slots, permissions, budget remaining.",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_active_intent,
    },
    "mempalace_resolve_conflicts": {
        "description": (
            "Resolve pending conflicts -- contradictions, duplicates, or merge candidates. "
            "MANDATORY when conflicts are returned by kg_add or kg_declare_entity (including kind='record'). "
            "Tools are BLOCKED until ALL conflicts are resolved. Batch-process in one call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Conflict ID from the pending conflicts list.",
                            },
                            "action": {
                                "type": "string",
                                "enum": ["invalidate", "merge", "keep", "skip"],
                                "description": (
                                    "invalidate: mark existing as no longer current. "
                                    "merge: combine both (requires into + merged_content). "
                                    "keep: both are valid, no conflict. "
                                    "skip: don't add the new item."
                                ),
                            },
                            "into": {
                                "type": "string",
                                "description": "Target ID to merge into (required for 'merge').",
                            },
                            "merged_content": {
                                "type": "string",
                                "description": (
                                    "Merged description/content preserving ALL unique info from both sides. "
                                    "Required for 'merge'. Read BOTH items in full before merging."
                                ),
                            },
                            "reason": {
                                "type": "string",
                                "description": (
                                    "MANDATORY -- why you chose this action (minimum 15 characters). "
                                    "Each conflict is a unique semantic decision. Evaluate individually. "
                                    "Bulk-identical reasons across 3+ conflicts will be rejected as laziness."
                                ),
                            },
                        },
                        "required": ["id", "action", "reason"],
                    },
                    "description": "List of conflict resolution actions.",
                },
                "agent": {
                    "type": "string",
                    "description": "MANDATORY -- declared agent resolving these conflicts.",
                },
            },
            "required": ["actions", "agent"],
        },
        "handler": tool_resolve_conflicts,
    },
    "mempalace_extend_intent": {
        "description": (
            "Extend the active intent's tool budget without redeclaring. "
            "Use when budget is exhausted but you're still on the same task. "
            "Adds counts to existing budget. Syncs with hook via disk state."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "budget": {
                    "type": "object",
                    "description": (
                        'Additional tool calls to add. E.g. {"Read": 3, "Edit": 2}. '
                        "These ADD to existing budget, not replace."
                    ),
                },
                "agent": {"type": "string", "description": "Your agent name."},
            },
            "required": ["budget"],
        },
        "handler": tool_extend_intent,
    },
    "mempalace_declare_operation": {
        "description": (
            "Declare the operation (tool call) you are about to perform. "
            "Replaces the auto-built cue-from-tool-args that the PreToolUse "
            "hook used to construct -- you specify the cue directly so "
            "retrieval surfaces memories that match your actual intention, "
            "not the shape of the tool call. Queries and keywords have the "
            "same role and shape as declare_intent's Context fingerprint, "
            "just scoped to ONE operation. Memories returned here land in "
            "accessed_memory_ids and require feedback at finalize_intent "
            "(same 100% coverage rule as declare-time memories). "
            "Carve-outs: mempalace_* tools and ALWAYS_ALLOWED (TodoWrite, "
            "Skill, Agent, ToolSearch, AskUserQuestion, Task*, "
            "ExitPlanMode) skip retrieval entirely and do NOT need "
            "declare_operation. When env MEMPALACE_REQUIRE_DECLARE_OPERATION "
            "is set, the hook BLOCKS any non-carve-out tool call that "
            "doesn't have a matching pending_operation_cue; otherwise the "
            "hook falls back to the legacy auto-build path. "
            "S1: the response also carries an optional `past_operations` "
            "field -- `{good_precedents, avoid_patterns}` -- drawn from the "
            "performed_well / performed_poorly edges in the current "
            "operation-context's MaxSim neighbourhood. Distinct from "
            "`memories` (memory-retrieval relevance); this is tool+args "
            "correctness. Rate your ops at finalize via `operation_ratings` "
            "to feed this channel. "
            "MANDATORY `args_summary` (parametrized core of the operation) "
            "is the cluster fingerprint -- see the field description for "
            "examples of good vs bad parametrization. Two ops with the "
            "same args_summary cluster as the SAME operation in past_ops "
            "and the gardener's S3 templatize detector."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tool": {
                    "type": "string",
                    "description": (
                        "Name of the tool you are about to call "
                        "(e.g. 'Read', 'Grep', 'Bash', 'Edit'). Must be "
                        "permitted under the active intent."
                    ),
                },
                "args_summary": {
                    "type": "string",
                    "description": (
                        "MANDATORY: parametrized core of the operation -- "
                        "the INVARIANT shape of what you are about to do, "
                        "with per-execution variables abstracted as "
                        "{placeholders}. Two ops sharing the same "
                        "args_summary string are doing the SAME operation "
                        "and will cluster as good/avoid precedents in "
                        "future past_operations responses, so the "
                        "fingerprint must capture INTENT not literal text. "
                        "Strip plumbing (cd, env vars, redirects) and "
                        "abstract anything that varies per call. "
                        "Examples:\n"
                        "  Bad:  'git commit -m \"feat(carveout): enforce "
                        "three-bucket gate via Slice C\"'\n"
                        "  Good: 'git commit -m \"{commit_message}\"'\n"
                        "\n"
                        "  Bad:  'cd D:/Flowsev/mempalace && git commit ...'\n"
                        "  Good: 'git commit -m \"{commit_message}\"' "
                        "(the cd is plumbing)\n"
                        "\n"
                        "  Bad:  'python -m pytest "
                        "tests/test_intent_system.py -q'\n"
                        "  Good: 'python -m pytest {test_path} -q'\n"
                        "\n"
                        "  Bad:  'kg_search queries=[\"phase 2 body "
                        'migration", "bucket file refactor"]\'\n'
                        "  Good: 'kg_search context.queries=[{N "
                        "perspectives on a topic}]'\n"
                        "\n"
                        "  Bad:  'Bash python -X utf8 -c \"import json; "
                        "...long inline script...\"'\n"
                        "  Good: 'python -X utf8 -c \"{inline ad-hoc "
                        "audit script}\"'\n"
                        "\n"
                        "Length: 5-400 chars. Cluster matching for past "
                        "operations + gardener S3 templatize detection "
                        "depend on this being well-formed; an empty or "
                        "literal-only value collapses different ops into "
                        "the same fingerprint and breaks precedent recall."
                    ),
                    "minLength": 5,
                    "maxLength": 400,
                },
                "context": _CONTEXT_SCHEMA,
                "agent": {
                    "type": "string",
                    "description": "Your agent name.",
                },
            },
            "required": ["tool", "args_summary", "context"],
        },
        "handler": tool_declare_operation,
    },
    "mempalace_declare_user_intents": {
        "description": (
            "Declare the user-intent contexts that cover the pending user "
            "messages for this session. Top tier of the activity hierarchy "
            "(Motive/Strategy in Leontiev 1981); activity-intents declared "
            "via declare_intent later in the turn link upward via cause_id "
            "(Slice B-3 wiring). MUST cover every pending user_message id "
            "for this session -- the union of context.user_message_ids "
            "across declared contexts must equal the pending set. Missing "
            "ids are heavily penalised; if you genuinely cannot infer the "
            "user's intent, use AskUserQuestion to clarify before declaring "
            "(use mempalace_* tools freely if memory will help disambiguate). "
            "If a covered message truly carries no actionable intent (a "
            "trivial 'thanks' / 'ok' ack), declare it under a no_intent=True "
            "context AND set no_intent_clarified_with_user=True ONLY after "
            "confirming with the user via AskUserQuestion. Self-asserting "
            "no_intent without proof is rejected. Returns memories per "
            "context -- same retrieval pipeline as declare_operation, "
            "dedup'd against accessed/injected ids accumulated this session, "
            "subject to mandatory feedback at finalize_intent. Grounding: "
            "STITCH (arXiv:2601.10702), Agent-Sentry (arXiv:2603.22868), "
            "BDI (Rao & Georgeff 1995). See "
            "diary_ga_agent_user_intent_tier_design_locked_2026_04_24."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "contexts": {
                    "type": "array",
                    "minItems": 1,
                    "description": (
                        "List of user-intent context declarations. ≥1 entry. "
                        "Each entry covers one or more pending user messages "
                        "and creates / reuses a kind='context' entity via "
                        "MaxSim -- same pattern as declare_intent / "
                        "declare_operation / kg_search."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "context": _CONTEXT_SCHEMA,
                            "user_message_ids": {
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "string"},
                                "description": (
                                    "Pending user_message ids this user-intent "
                                    "covers. ≥1 per context. Every pending id "
                                    "for this session must be covered by ≥1 "
                                    "context across the batch. See "
                                    "additionalContext from the UserPromptSubmit "
                                    "hook for the pending id list."
                                ),
                            },
                            "time_window": {
                                "type": "object",
                                "properties": {
                                    "start": {"type": "string"},
                                    "end": {"type": "string"},
                                },
                                "description": (
                                    "Optional ISO date range for soft date-range "
                                    "boost in retrieval (same semantics as "
                                    "kg_search.time_window -- soft +0.15 boost "
                                    "for items dated inside the window, items "
                                    "outside still rank)."
                                ),
                            },
                            "no_intent": {
                                "type": "boolean",
                                "description": (
                                    "TRUE iff the covered user message(s) "
                                    "have no actionable intent (ack, 'thanks', "
                                    "etc.). Default FALSE. When TRUE, "
                                    "no_intent_clarified_with_user MUST also "
                                    "be TRUE -- agent must have asked the "
                                    "user via AskUserQuestion."
                                ),
                            },
                            "no_intent_clarified_with_user": {
                                "type": "boolean",
                                "description": (
                                    "Truthful flag -- set TRUE only when the "
                                    "agent actually asked the user via "
                                    "AskUserQuestion to confirm the message "
                                    "has no actionable intent. Self-asserting "
                                    "no_intent without proof is rejected."
                                ),
                            },
                        },
                        "required": ["context", "user_message_ids"],
                    },
                },
                "agent": {
                    "type": "string",
                    "description": "Your agent name.",
                },
            },
            "required": ["contexts", "agent"],
        },
        "handler": tool_declare_user_intents,
    },
    "mempalace_finalize_intent": {
        "description": (
            "Finalize the active intent -- capture what happened as structured memory. "
            "MUST be called before declaring a new intent or exiting the session. "
            "Creates an execution entity (is_a intent_type) with relationships to agent, "
            "targets, result memory, gotchas, execution trace, and memory relevance feedback. "
            "If not called explicitly, declare_intent will BLOCK until you finalize."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "slug": {
                    "type": "string",
                    "description": "Human-readable ID for this execution (e.g. 'edit-auth-rate-limiter-2026-04-14'). Must be unique.",
                },
                "outcome": {
                    "type": "string",
                    "description": "How the intent concluded.",
                    "enum": ["success", "partial", "failed", "abandoned"],
                },
                "content": {
                    "type": "string",
                    "description": (
                        "MANDATORY -- the full narrative body for the result memory. "
                        "Free length, stored verbatim. ALWAYS required on every record; "
                        "no auto-derivation. For long content, a distillation of WHAT/WHY; "
                        "for short content, a REPHRASE from a different angle so the "
                        "summary+content pair yields two distinct cosine views of the "
                        "same semantic (Anthropic Contextual Retrieval 2024)."
                    ),
                },
                "summary": _SUMMARY_SUBSCHEMA,
                "agent": {
                    "type": "string",
                    "description": "Your agent entity name (e.g. 'ga_agent', 'technical_lead_agent').",
                },
                "memory_feedback": {
                    "type": "array",
                    "description": (
                        "MANDATORY -- list of per-context feedback groups: "
                        "[{context_id: <ctx_id>, feedback: [{id, relevance, reason, relevant?}, ...]}, ...]. "
                        "Each group attributes its ratings back to the context that surfaced those memories "
                        "(from declare_intent, declare_operation, or kg_search). Channel D reads rated_useful / "
                        "rated_irrelevant edges scoped to that context on subsequent intents, so correct "
                        "attribution is load-bearing. Coverage rule: every memory in accessed_memory_ids must "
                        "appear in exactly one group's `feedback` list. Values 1-2 become rated_irrelevant edges; "
                        "3-5 become rated_useful. This list-of-groups shape replaced the retired dict shape "
                        "(2026-04-24) because some MCP clients silently drop object parameters whose schema uses "
                        "`additionalProperties`-with-nested-schema; list-of-objects round-trips through every "
                        "client cleanly. Use `missing_injected` (map) in the error response as the guide for "
                        "which ctx_ids need coverage."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "context_id": {
                                "type": "string",
                                "description": (
                                    "The Context entity id that surfaced these memories. Returned by "
                                    "declare_intent / declare_operation / kg_search in their `context.id` field "
                                    "(on reuse) or revealed by a failing finalize's `missing_injected` map. "
                                    "Do NOT confuse with intent_id -- the Context is a distinct KG entity."
                                ),
                            },
                            "feedback": {
                                "type": "array",
                                "description": "Per-memory rating entries for the memories surfaced under this context.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {
                                            "type": "string",
                                            "description": "Memory ID or entity ID",
                                        },
                                        "relevance": {
                                            "type": "integer",
                                            "minimum": 1,
                                            "maximum": 5,
                                            "description": (
                                                "1-5, signed scale -- what did you actually do with this memory when it surfaced? "
                                                "1=misleading (wasted attention / pointed me wrong; teach the context NOT to surface this again). "
                                                "2=noise (skimmed and dropped; same topic area, nothing to do with this specific task). "
                                                "3=related context (DEFAULT when unsure -- accurate and topical, didn't change what I did). "
                                                "4=informed (changed a decision or saved a lookup; want this again on similar tasks). "
                                                "5=load-bearing (the task fails or duplicates work without it). "
                                                "Values 1-2 become rated_irrelevant edges on the active context; "
                                                "values 3-5 become rated_useful edges. "
                                                "Calibration: if >50% of your ratings are >=4, re-read your task and demote. "
                                                "Clustering at the top compresses every downstream signal. "
                                                "The system learns from the skew; inflating ratings dampens the signal you're giving future-you."
                                            ),
                                        },
                                        "reason": {
                                            "type": "string",
                                            "description": (
                                                "MANDATORY -- why this memory was or wasn't relevant to THIS intent "
                                                "(minimum 10 characters). Evaluate each memory individually."
                                            ),
                                        },
                                        "relevant": {
                                            "type": "boolean",
                                            "description": (
                                                "Optional explicit override of the relevance→relevant mapping. "
                                                "Normally derived from relevance (1-2 → false / rated_irrelevant, "
                                                "3-5 → true / rated_useful). Set only when the derived value is wrong."
                                            ),
                                        },
                                    },
                                    "required": ["id", "relevance", "reason"],
                                },
                            },
                        },
                        "required": ["context_id", "feedback"],
                    },
                },
                "key_actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Abbreviated tool+params list. Auto-filled from execution trace if omitted.",
                },
                "gotchas": {
                    "type": "array",
                    "description": (
                        "Gotchas discovered during execution. Each is "
                        "{summary: {what, why, scope?}, content: str}. "
                        "summary is the structured anchor for retrieval "
                        "(rendered prose <=280 chars, validated at source "
                        "with no auto-derive). content is the verbatim "
                        "narrative body. The handler creates a gotcha "
                        "entity (description = rendered summary prose) "
                        "and a twin record carrying content. Strings are "
                        "rejected with a migration error -- Adrian's "
                        "design lock 2026-04-28: avoid auto-derive "
                        "everywhere."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "summary": {
                                "type": "object",
                                "properties": {
                                    "what": {"type": "string"},
                                    "why": {"type": "string"},
                                    "scope": {"type": "string"},
                                },
                                "required": ["what", "why"],
                            },
                            "content": {"type": "string"},
                        },
                        "required": ["summary", "content"],
                    },
                },
                "learnings": {
                    "type": "array",
                    "description": (
                        "Lessons worth remembering. Each is "
                        "{summary: {what, why, scope?}, content: str}. "
                        "summary is the structured anchor (rendered prose "
                        "<=280 chars, validated at source); content is "
                        "the verbatim lesson body. The handler files each "
                        "as a record via _add_memory_internal with the "
                        "caller-provided summary passed through directly "
                        "(no auto-derive). Strings are rejected with a "
                        "migration error -- Adrian's design lock "
                        "2026-04-28: avoid auto-derive everywhere."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "summary": {
                                "type": "object",
                                "properties": {
                                    "what": {"type": "string"},
                                    "why": {"type": "string"},
                                    "scope": {"type": "string"},
                                },
                                "required": ["what", "why"],
                            },
                            "content": {"type": "string"},
                        },
                        "required": ["summary", "content"],
                    },
                },
                "promote_gotchas_to_type": {
                    "type": "boolean",
                    "description": "Also link gotchas to the intent TYPE (not just this execution). Use for general gotchas.",
                },
                "operation_ratings": {
                    "type": "array",
                    "description": (
                        "MANDATORY -- your rating of tool-invocation quality. "
                        "100% coverage required: every (tool, context_id) "
                        "pair that appeared in the execution trace must "
                        "have a rating. ORTHOGONAL to memory_feedback: "
                        "memory_feedback rates retrieved MEMORIES (was the "
                        "surfaced info useful?); operation_ratings rates "
                        "OPERATIONS (was the tool+args you chose the right "
                        "move?). "
                        "Each entry: {tool (required), context_id "
                        "(required, from declare_operation), quality "
                        "(required, 1-5: 1=wrong move, 2=suboptimal, "
                        "3=ok -- neutral signal, no promotion, 4=good, "
                        "5=load-bearing), reason, better_alternative (S2)}. "
                        "Quality >=4 writes performed_well; <=2 writes "
                        "performed_poorly; =3 is neutral. One rating per "
                        "unique (tool, ctx_id) pair covers any number of "
                        "repeated calls under that pair. See "
                        "`missing_operations` map in a failed finalize for "
                        "the required pairs. Distinct from rated_useful / "
                        "rated_irrelevant. "
                        "NOTE: `args_summary` is no longer carried in the "
                        "rating -- it was moved to declare_operation as a "
                        "MANDATORY parametrized-core field at declare-time "
                        "(2026-04-27). Promotion now reads it from the "
                        "operation-context store keyed by (context_id, tool). "
                        "Cf. Leontiev 1981 Operation tier; arXiv 2512.18950."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": {"type": "string"},
                            "context_id": {"type": "string"},
                            "quality": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5,
                            },
                            "reason": {"type": "string"},
                            "better_alternative": {"type": "string"},
                        },
                        "required": ["tool", "context_id", "quality"],
                    },
                },
            },
            "required": ["slug", "outcome", "content", "summary", "agent", "memory_feedback"],
        },
        "handler": tool_finalize_intent,
    },
    "mempalace_extend_feedback": {
        "description": (
            "Sibling of mempalace_finalize_intent (2026-04-25 two-tool design). "
            "Use ONLY after finalize_intent has accepted the intent but reported "
            "incomplete coverage. This tool merges additional memory_feedback / "
            "operation_ratings into the existing execution entity -- same shape as "
            "finalize_intent's same-named params. When coverage hits 100%, the intent "
            "formally finalizes (active intent cleared, sentinel written, gardener "
            "triggered). Cannot start an intent or change metadata; cannot be called "
            "before finalize_intent. Survives MCP server restart via persisted "
            "active_intent.pending_feedback state."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "Your agent entity name (must match the agent that called finalize_intent).",
                },
                "memory_feedback": {
                    "type": "array",
                    "description": (
                        "List of per-context feedback groups -- same shape as "
                        "finalize_intent.memory_feedback: "
                        "[{context_id, feedback: [{id, relevance, reason, relevant?}, ...]}, ...]. "
                        "Last-write-wins per (memory_id, context_id) -- supplying a "
                        "rating for a memory already rated overwrites the prior one."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "context_id": {"type": "string"},
                            "feedback": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "relevance": {
                                            "type": "integer",
                                            "minimum": 1,
                                            "maximum": 5,
                                        },
                                        "reason": {"type": "string"},
                                        "relevant": {"type": "boolean"},
                                    },
                                    "required": ["id", "relevance", "reason"],
                                },
                            },
                        },
                        "required": ["context_id", "feedback"],
                    },
                },
                "operation_ratings": {
                    "type": "array",
                    "description": (
                        "List of operation rating entries -- same shape as "
                        "finalize_intent.operation_ratings: "
                        "[{tool, context_id, quality, reason}, ...]. "
                        "Last-write-wins per (tool, context_id). "
                        "args_summary is NOT carried here -- it was moved "
                        "to declare_operation as a mandatory parametrized-"
                        "core field at declare-time (2026-04-27)."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": {"type": "string"},
                            "context_id": {"type": "string"},
                            "quality": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5,
                            },
                            "reason": {"type": "string"},
                        },
                        "required": ["tool", "context_id", "quality", "reason"],
                    },
                },
            },
            "required": ["agent"],
        },
        "handler": tool_extend_feedback,
    },
    "mempalace_kg_delete_entity": {
        "description": "Delete an entity (record or KG node) and invalidate every current edge touching it. Works for both records (ids starting with 'record_' / 'diary_') and KG entities. Use this when an entity is TRULY obsolete. For stale single facts (one relationship untrue while entity stays valid), use kg_invalidate on that specific (subject, predicate, object) triple instead.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "ID of the entity or record to delete.",
                },
                "agent": {
                    "type": "string",
                    "description": "MANDATORY -- declared agent attributing this deletion.",
                },
            },
            "required": ["entity", "agent"],
        },
        "handler": tool_kg_delete_entity,
    },
    "mempalace_wake_up": {
        "description": (
            "Return L0 (identity) + L1 (importance-ranked essential story) wake-up "
            "text (~600-900 tokens total). Call this ONCE at session start to load "
            "project/agent boot context. Also returns the protocol, declared entities/"
            "predicates/intent types -- everything you need to start. L1 is ranked "
            "with importance-weighted time decay -- critical facts always surface first, "
            "within-tier newer wins."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "Agent identity (required). Used for affinity scoring in L1 and auto-derives the memory scope. Use your agent entity name (e.g., 'ga_agent', 'technical_lead_agent').",
                },
            },
            "required": ["agent"],
        },
        "handler": tool_wake_up,
    },
    # mempalace_update_drawer_metadata removed: merged into mempalace_kg_update_entity.
    "mempalace_diary_write": {
        "description": "Write to your personal agent diary. Your observations, thoughts, what you worked on, what matters. Each agent has their own diary with full history. Optional content_type/importance for special entries.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Your name -- each agent gets their own diary",
                },
                "entry": {
                    "type": "string",
                    "description": "Your diary entry -- readable prose.",
                },
                "slug": {
                    "type": "string",
                    "description": "Descriptive identifier for this diary entry (e.g. 'session12-intent-narrowing-shipped', 'migration-lesson-learned'). Used as part of the entry ID.",
                },
                "topic": {
                    "type": "string",
                    "description": "Topic tag (optional, default: general)",
                },
                "content_type": {
                    "type": "string",
                    "description": "Override the default diary classification. Use discovery for 'today I learned' entries worth higher retrieval priority, event for plain activity logs.",
                    "enum": [
                        "fact",
                        "event",
                        "discovery",
                        "preference",
                        "advice",
                        "diary",
                    ],
                },
                "importance": {
                    "type": "integer",
                    "description": "Importance 1-5. Leave unset for default diary entries (treated as 3 by Layer1). Use 4 for entries with learned lessons, 5 only for agent-wide critical notes.",
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": ["agent_name", "entry"],
        },
        "handler": tool_diary_write,
    },
    "mempalace_diary_read": {
        "description": "Read your recent diary entries. See what past versions of yourself recorded -- your journal across sessions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Your name -- each agent gets their own diary",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of recent entries to read (default: 10).",
                    "minimum": 1,
                },
                "last_n": {
                    "type": "integer",
                    "description": (
                        "DEPRECATED -- use `limit` instead. Kept for "
                        "back-compat with older agents; if both are passed, "
                        "`limit` wins."
                    ),
                    "minimum": 1,
                },
            },
            "required": ["agent_name"],
        },
        "handler": tool_diary_read,
    },
}


SUPPORTED_PROTOCOL_VERSIONS = [
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
]


def handle_request(request):
    method = request.get("method", "")
    params = request.get("params", {})
    req_id = request.get("id")

    if method == "initialize":
        client_version = params.get("protocolVersion", SUPPORTED_PROTOCOL_VERSIONS[-1])
        negotiated = (
            client_version
            if client_version in SUPPORTED_PROTOCOL_VERSIONS
            else SUPPORTED_PROTOCOL_VERSIONS[0]
        )
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": negotiated,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mempalace", "version": __version__},
            },
        }
    elif method == "notifications/initialized":
        return None
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {"name": n, "description": t["description"], "inputSchema": t["input_schema"]}
                    for n, t in TOOLS.items()
                ]
            },
        }
    elif method == "tools/call":
        tool_name = params.get("name")
        tool_args = params.get("arguments") or {}
        # Stderr heartbeat so when the next call segfaults at C-level
        # (Chroma HNSW, onnx) we know which tool killed the process.
        # Without this, post-mortem stderr just shows the generic
        # "Windows fatal exception: access violation" with no context
        # for which call triggered it. Heartbeat is cheap (~50 bytes
        # per call) and never gates dispatch.
        try:
            print(f"[mcp] tools/call: {tool_name}", file=sys.stderr, flush=True)
        except Exception:
            pass
        if tool_name not in TOOLS:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }
        # Extract sessionId injected by the PreToolUse hook via
        # hookSpecificOutput.updatedInput. Verified to propagate
        # correctly (measured 2026-04-19 via SID_PROBE).
        #
        # Sanitize identically to the hook so file names match.
        #
        # NO FALLBACK if it's empty. We refuse to synthesize
        # "default" / "unknown" -- that would silently merge every
        # agent's state into a shared file. When sid is unknown we
        # simply don't switch; downstream state operations will
        # themselves refuse to read/write (see _intent_state_path and
        # _load_pending_from_disk).
        injected_session_id = tool_args.pop("sessionId", None)
        if injected_session_id:
            new_sid = _sanitize_session_id(injected_session_id)
            if new_sid and new_sid != _STATE.session_id:
                _save_session_state()
                _STATE.session_id = new_sid
                _restore_session_state(new_sid)
                if _STATE.active_intent:
                    try:
                        intent._persist_active_intent()
                    except Exception:
                        pass

        # Coerce argument types based on input_schema.
        # MCP JSON transport may deliver integers as floats or strings;
        # ChromaDB and Python slicing require native int.
        schema_props = TOOLS[tool_name]["input_schema"].get("properties", {})
        for key, value in list(tool_args.items()):
            prop_schema = schema_props.get(key, {})
            declared_type = prop_schema.get("type")
            if declared_type == "integer" and not isinstance(value, int):
                tool_args[key] = int(value)
            elif declared_type == "number" and not isinstance(value, (int, float)):
                tool_args[key] = float(value)
        try:
            result = TOOLS[tool_name]["handler"](**tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]},
            }
        except Exception as e:
            logger.exception(f"Tool error in {tool_name}")
            # Include the exception details so callers can diagnose.
            # Generic "Internal tool error" without context is a debugging nightmare.
            import traceback

            tb_summary = traceback.format_exc().splitlines()[-5:]
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32000,
                    "message": f"Tool '{tool_name}' failed: {type(e).__name__}: {e}",
                    "data": {"traceback": tb_summary},
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Unknown method: {method}"},
    }


def _drop_feedback_contexts_collection_once():
    """One-shot: drop the retired mempalace_feedback_contexts Chroma collection.

    P3 polish -- migration 015 retired the SQLite companion tables
    (keyword_feedback, edge_traversal_feedback). This drops the Chroma
    collection they fed off. Idempotent and fail-open: if the collection
    doesn't exist, we just mark the flag and move on.

    Gated by ``ServerState.feedback_contexts_dropped`` so we only run
    once per server process. The SQLite `_yoyo_migration` table owns
    its own idempotence; this flag mirrors that for the Chroma side.
    """
    if _STATE.feedback_contexts_dropped:
        return
    _STATE.feedback_contexts_dropped = True
    try:
        client = chromadb.PersistentClient(path=_STATE.config.palace_path)
    except Exception:
        return
    try:
        client.delete_collection(FEEDBACK_CONTEXT_COLLECTION)
        logger.info("Dropped retired Chroma collection: %s", FEEDBACK_CONTEXT_COLLECTION)
    except Exception:
        # Most commonly: collection doesn't exist. Fresh palace -- nothing
        # to drop. Quiet success.
        pass


def _run_hyphen_id_migration_once():
    """Rename legacy hyphenated IDs (Chroma + SQLite) to canonical form.

    Idempotent, one-pass per process. Gated on ``_STATE.hyphen_ids_migrated``.
    Uses ``normalize_entity_name`` as the single source of truth so the
    post-migration invariant ``stored_id == normalize(id)`` holds across
    every collection and table.
    """
    try:
        from . import hyphen_id_migration
        from .knowledge_graph import normalize_entity_name

        stats = hyphen_id_migration.run_migration(
            _STATE,
            chroma_record_col=_get_collection(create=False),
            chroma_entity_col=_get_entity_collection(create=False),
            chroma_feedback_col=None,
            normalize=normalize_entity_name,
        )
        if not stats.get("skipped"):
            logger.info(f"N3 hyphen-id migration completed: {stats}")
    except Exception as e:
        logger.warning(f"N3 hyphen-id migration failed: {e}")


def main():
    logger.info("MemPalace MCP Server starting...")

    # Write hint file for hook subprocesses. Only fires here (actual
    # server startup), never at module-import time \u2014 so pytest / CLI
    # invocations that happen to import this module don't clobber the
    # production hint with a temp palace path.
    try:
        _hint_dir = Path(os.path.expanduser("~/.mempalace/hook_state"))
        _hint_dir.mkdir(parents=True, exist_ok=True)
        (_hint_dir / "active_palace.txt").write_text(_STATE.config.palace_path, encoding="utf-8")
    except OSError:
        pass  # best-effort: hooks fall through to other resolution paths

    # run the kind='record' → 'record' migration once at startup.
    # Idempotent, no-op on fresh palaces or on second invocation within
    # the process.
    try:
        _migrate_kind_memory_to_record()
    except Exception as e:
        logger.warning(f"P6.2 startup kind migration failed: {e}")
    # N3 hyphen-id migration -- RETIRED FROM AUTO-STARTUP (2026-04-25).
    #
    # This was a one-shot legacy migration that renamed hyphenated IDs
    # to the underscored canonical form. New palaces never produce
    # hyphenated IDs -- `normalize_entity_name` strips them at write
    # time, so any palace created post-N3 has nothing to migrate.
    # Already-migrated palaces re-walked thousands of Chroma rows on
    # every server boot, doing zero useful work, while exposing the
    # boot path to a known Chroma v0.6.0 internal bug ("list assignment
    # index out of range" inside `col.get(include=embeddings)`) that
    # could leave the HNSW vector index half-written and trigger a
    # C-level access violation on subsequent queries.
    #
    # The migration code (mempalace.hyphen_id_migration.run_migration)
    # is preserved verbatim -- anyone with genuinely legacy hyphenated
    # data can run it manually via:
    #   python -c "from mempalace import mcp_server; \
    #              mcp_server._run_hyphen_id_migration_once()"
    # -- but it is NOT invoked here on every boot.
    #
    # See: record_ga_agent_chroma_hnsw_segfault_root_cause_2026_04_25
    # for the corruption mechanism this removal closes off.
    # M1 collection merge -- absorb legacy mempalace_entities rows into
    # the unified mempalace_records collection and drop the legacy one.
    try:
        _migrate_entities_collection_into_records()
    except Exception as e:
        logger.warning(f"M1 startup collection-merge failed: {e}")
    # P3 polish one-shot -- drop the retired mempalace_feedback_contexts
    # Chroma collection. Its SQLite peers (keyword_feedback,
    # edge_traversal_feedback) were dropped by migration 015; this hook
    # takes care of the Chroma side (which can't be touched from SQL).
    try:
        _drop_feedback_contexts_collection_once()
    except Exception as e:
        logger.warning(f"feedback_contexts drop failed: {e}")
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            request = json.loads(line)
            response = handle_request(request)
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Server error: {e}")


if __name__ == "__main__":
    main()
