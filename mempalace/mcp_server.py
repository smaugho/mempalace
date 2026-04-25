#!/usr/bin/env python3
"""
MemPalace MCP Server — read/write palace access for Claude Code
================================================================
Install: claude mcp add mempalace -- python -m mempalace.mcp_server [--palace /path/to/palace]

Tools (read):
  mempalace_kg_search       — unified 3-channel search over memories AND entities
  mempalace_kg_query        — structured edge lookup by exact entity name
  mempalace_kg_stats        — knowledge graph overview: entities, triples, relationship types

Tools (write):
  mempalace_kg_declare_entity — declare an entity (kind=entity/class/predicate/literal/record)
                                memory memories are first-class entities
  mempalace_kg_delete_entity — soft-delete an entity or memory (invalidates all edges)
  mempalace_resolve_conflicts — resolve contradictions, duplicates, merge candidates
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
# exception machinery — try/except wrappers around tool dispatch (see
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
# Python would execute this file a SECOND time under that dotted name —
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

from .config import MempalaceConfig, sanitize_name, sanitize_content
from .version import __version__
from .query_sanitizer import sanitize_query
import chromadb

from .knowledge_graph import KnowledgeGraph
from . import intent
from .server_state import ServerState
from .scoring import (
    hybrid_score as _hybrid_score_fn,
    multi_channel_search,
    walk_rated_neighbourhood,
)

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


# ==================== S1 OPERATION ONTOLOGY ====================
# Op-memory tier (Leontiev 1981 Activity Theory — Operation level;
# cf. arXiv 2512.18950 Learning Hierarchical Procedural Memory). We seed
# one class (`operation`) plus three predicates unconditionally on every
# startup. add_entity / add_triple both upsert via ON CONFLICT DO UPDATE,
# so re-running is a no-op on existing palaces and a one-shot fill on
# fresh ones. Separating this from KnowledgeGraph.seed_ontology keeps
# knowledge_graph.py free of S1-specific code.


def _ensure_operation_ontology(kg):
    """Idempotently seed the `operation` class + S1 predicates.

    Predicates:
      * executed_op — intent_exec → op (parent/child; statement required)
      * performed_well — context → op (agent-rated quality ≥4)
      * performed_poorly — context → op (agent-rated quality ≤2)

    The `operation` class is a subclass of `thing`. Operations are
    kind='operation' entities (see VALID_KINDS); they are NEVER embedded
    into Chroma collections — retrieval reaches them only via graph
    traversal from their context's performed_well / performed_poorly
    edges.
    """
    kg.add_entity(
        "operation",
        kind="class",
        description=(
            "A recorded tool invocation (tool + truncated args + context_id). "
            "Graph-only — never embedded. Attached to an intent execution via "
            "executed_op, and to its operation-context via performed_well / "
            "performed_poorly. Cf. Leontiev 1981 Operation tier; arXiv "
            "2512.18950 hierarchical procedural memory."
        ),
        importance=4,
    )
    kg.add_triple("operation", "is_a", "thing")

    _op_pred_defs = [
        (
            "executed_op",
            "Parent-child edge from an intent execution to an operation "
            "entity it performed. Written by finalize_intent when promoting "
            "a rated trace entry.",
            4,
            {
                "subject_kinds": ["entity"],
                "object_kinds": ["operation"],
                "subject_classes": ["intent_type", "thing"],
                "object_classes": ["operation", "thing"],
                "cardinality": "one-to-many",
            },
        ),
        (
            "performed_well",
            "Positive op-quality edge: in the given operation-context the "
            "agent rated this op as good (quality ≥4). Read at declare_"
            "operation time to surface precedent patterns. Distinct from "
            "rated_useful — that is memory-retrieval relevance; this is "
            "tool+args correctness.",
            4,
            {
                "subject_kinds": ["context"],
                "object_kinds": ["operation"],
                "subject_classes": ["thing"],
                "object_classes": ["thing"],
                "cardinality": "many-to-many",
            },
        ),
        (
            "performed_poorly",
            "Negative op-quality edge: in the given operation-context the "
            "agent rated this op as wrong or suboptimal (quality ≤2). "
            "Surfaced alongside performed_well so the agent sees both "
            "precedent and cautionary cases.",
            3,
            {
                "subject_kinds": ["context"],
                "object_kinds": ["operation"],
                "subject_classes": ["thing"],
                "object_classes": ["thing"],
                "cardinality": "many-to-many",
            },
        ),
        (
            "superseded_by",
            "S2 correction edge: a poorly-rated op points to the op that "
            "would have been the correct move in the same context. "
            "Written when the agent provides `better_alternative` on an "
            "operation_ratings entry (quality ≤2). Read at declare_operation "
            "time to present cautionary precedent PLUS a concrete "
            "alternative — not just 'don't do this' but 'do THIS instead'. "
            "op-to-op edge (both subject and object are kind='operation').",
            4,
            {
                "subject_kinds": ["operation"],
                "object_kinds": ["operation"],
                "subject_classes": ["operation", "thing"],
                "object_classes": ["operation", "thing"],
                "cardinality": "many-to-one",
            },
        ),
        (
            "templatizes",
            "S3b template-collapse edge: a reusable template record points "
            "at each source operation it distilled. Written by the "
            "memory_gardener's synthesize_operation_template tool when it "
            "resolves an op_cluster_templatizable flag (>=3 same-tool "
            "same-sign precedents that surfaced together at declare_operation "
            "time). Read by retrieve_past_operations (S3c) which hoists the "
            "template into its own lane and suppresses the raw ops the "
            "template covers — replace-not-append keeps the response "
            "bounded. record→operation edge; one template covers many ops.",
            4,
            {
                "subject_kinds": ["record"],
                "object_kinds": ["operation"],
                "subject_classes": ["record", "thing"],
                "object_classes": ["operation", "thing"],
                "cardinality": "one-to-many",
            },
        ),
    ]
    for name, desc, imp, constraints in _op_pred_defs:
        kg.add_entity(
            name,
            kind="predicate",
            description=desc,
            importance=imp,
            properties={"constraints": constraints},
        )


if not os.environ.get("MEMPALACE_SKIP_SEED"):
    try:
        _ensure_operation_ontology(_STATE.kg)
    except Exception as _ensure_err:
        import logging as _ensure_log

        _ensure_log.getLogger(__name__).warning("ensure_operation_ontology failed: %r", _ensure_err)


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


PALACE_PROTOCOL = """MemPalace Protocol — rules for your behavior. The system enforces
structure (slot validation, tool permissions, conflict detection, feedback
coverage) and teaches through error messages; your job is to do the right
thing and let the errors tune the rest.

ON START:
  Call mempalace_wake_up. Read this protocol, the text (identity + rules),
  and declared (entities, predicates, intent types with their tools).

BEFORE ACTING ON ANY FACT:
  kg_query for exact entity-ID lookups when you know the name.
  kg_search for fuzzy discovery — searches memories (prose) and entities
  (KG nodes) in one call, with graph expansion. Never guess.

WHEN HITTING A BLOCKER:
  Search mempalace first — gotchas, lessons-learned, past executions on
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
      merge (combine — READ BOTH in full, provide merged_content that
        preserves ALL unique info from each side),
      keep (both are valid),
      skip (undo the new item).

DECLARING INTENT / OPERATION / SEARCH:
  - Declare intent first (mempalace_declare_intent). Check
    declared.intent_types for available types. Blocked-tool errors
    teach how to create or switch intent types. Intent types are
    kind=class; executions are kind=entity with is_a.
  - Before EVERY non-carve-out tool call (Read, Grep, Glob, Bash, Edit,
    Write, WebFetch, WebSearch, etc.) call mempalace_declare_operation
    with a cue that reflects your ACTUAL intention — not the shape of
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
    Entities feed the link-author background pipeline — zero entities,
    no graph growth.

WHEN RECEIVING INJECTED MEMORIES:
  - Every memory surfaced (by declare_intent, declare_operation, or
    kg_search) lands in accessed_memory_ids and REQUIRES feedback at
    finalize_intent: 100% coverage, 1-5 relevance, reason string.
    Finalize rejects without coverage.
  - memory_feedback shape: list of groups
    [{context_id: <ctx_id>, feedback: [entries]}, ...]. Each group
    attributes its ratings back to the context that surfaced the
    memories — this is what future retrieval reads. (Dict shape was
    retired 2026-04-24; MCP clients silently dropped it.)
  - Relevance calibration: 3 = related context (default when unsure).
    4-5 = changed a decision / load-bearing. 1-2 = noise / misleading.
    If >50% of your ratings are >=4, demote — inflating dampens the
    signal you're giving future-you.
  - Memories return in short form. For the full content, call
    kg_query(entity=<id>).
  - Zero hits for a cue is success — proceed. Low-relevance hits are
    expected and useful as negative feedback, not errors.

BEFORE SWITCHING INTENTS:
  Call finalize_intent BEFORE declaring a new intent. Captures what you
  did (execution entity + trace), what you learned (gotchas, learnings),
  outcome (success / partial / failed / abandoned). If you forget,
  declare_intent auto-finalizes with outcome='abandoned' — lower-quality
  memory. Explicit finalization is always better.

INTENT TYPES:
  - New intent types use kind='class'.
  - If 3+ similar executions exist on a broad type, declaration fails —
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
     Don't just diary them — the diary is a temporal log; KG + records
     are durable knowledge future sessions can query structurally.
  3. diary_write — concise, non-redundant, prose.
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
# a graph" unification — one search tool over all memory.


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
    "record",  # a stored prose record — full text in ChromaDB, metadata in SQLite
    "operation",  # a tool invocation: graph-only, NEVER embedded into Chroma.
    #              Carries tool + args_json + context_id; attached to an
    #              intent execution via executed_op, and to the operation's
    #              context via performed_well / performed_poorly. Cf. arXiv
    #              2512.18950 (hierarchical procedural memory) + Leontiev
    #              1981 (Activity Theory AAO — "operation" tier).
}

# kind='memory' is GONE. The one-pass migration at startup
# rewrites existing metadata; the alias is removed so callers
# get a clear error instead of silent normalization. "memory" is the
# palace-level concept; "record" is the record-type kind.


def _validate_kind(kind):
    """Validate entity kind (ontological role). REQUIRED — no default."""
    if kind is None:
        raise ValueError(
            "kind is REQUIRED. Must be one of: 'entity' (concrete thing), "
            "'predicate' (relationship type), 'class' (category definition), "
            "'literal' (raw value), 'record' (prose record — requires "
            "slug + content + added_by), or 'operation' (tool invocation — "
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
            f"Domain types (system, person, project, etc.) are NOT kinds — "
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
    re-normalized to underscores and then looked up the hyphenated ID —
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
# (no length threshold) — grounded in Anthropic's Contextual Retrieval
# (https://www.anthropic.com/news/contextual-retrieval, 2024 — prepending
# explanatory context to each chunk before embedding cut retrieval
# failure rate by 49%) and Chen et al. "Dense X Retrieval"
# (arXiv:2312.06648, 2023 — proposition-granularity embeddings outperform
# passage-granularity on open-domain QA). Neither paper threshold-gates by
# length; every chunk gets the treatment. For long content the summary
# is a distillation of WHAT/WHY; for short content the summary should
# REPHRASE the same fact from a different angle (different keywords /
# framing) so the summary+content pair produces TWO distinct cosine
# views — genuine retrieval information gain, not redundancy. The
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
    summary: str = None,
):
    """File verbatim content as a flat record. Checks for duplicates first.

    ALL classification params are REQUIRED (no lazy defaults):
        slug: short human-readable identifier — REQUIRED. Used as part of the
              record ID. Must be unique per agent. Examples:
              'intent-pre-activation-issues', 'db-credentials', 'ga-identity'.
        content_type: one of fact, event, discovery, preference, advice, diary.
        importance: integer 1-5 — REQUIRED. 5=critical, 4=canonical,
                    3=default, 2=low, 1=junk.
        entity: entity name (or comma-separated list) — REQUIRED. Links this record
                to an entity in the KG. If not provided, the record is unlinked.
        predicate: relationship type for the entity→record link. Default: described_by.
        summary: ≤280-char distillation — REQUIRED on every record (no
              length threshold). For long content the summary distills the
              WHAT/WHY; for short content the summary should REPHRASE the
              same fact from a different angle (different keywords / framing)
              so the summary+content pair yields two distinct cosine views of
              the same semantic (Anthropic Contextual Retrieval, 2024; Chen
              et al. Dense X Retrieval, arXiv:2312.06648, 2023 — neither
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

    # ── Summary-first gate: summary ALWAYS required ──
    # Every record carries a ≤280-char summary. For long content the
    # summary distills WHAT/WHY; for short content the summary should
    # REPHRASE the same fact from a different angle so the summary+content
    # pair produces two distinct cosine views of the same semantic (real
    # retrieval gain, not redundancy). Grounded in Anthropic Contextual
    # Retrieval (2024) and Chen et al. Dense X Retrieval (2023) — neither
    # gates by length; every chunk gets the treatment.
    summary = (summary or "").strip() if summary is not None else ""
    if not summary:
        return {
            "success": False,
            "error": (
                f"`summary` is required on every record (≤{_RECORD_SUMMARY_MAX_LEN} "
                f"chars). No auto-derivation. For long content, distill the "
                f"WHAT/WHY. For short content, REPHRASE the same fact from a "
                f"different angle (different keywords, different framing) so "
                f"the summary and content give two distinct retrieval views of "
                f"the same semantic."
            ),
        }
    if len(summary) > _RECORD_SUMMARY_MAX_LEN:
        return {
            "success": False,
            "error": (
                f"`summary` is {len(summary)} chars; maximum is "
                f"{_RECORD_SUMMARY_MAX_LEN}. Distill further — one sentence, "
                f"names the WHAT and WHY, no filler."
            ),
        }

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

    # Uniqueness check — slug collision returns existing record info
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
    # embedding" — reported ~49% reduction in retrieval failure). The
    # stored document keeps the prepended shape so col.get() round-trips
    # don't lose the context; `metadata.summary` remains the canonical
    # short form for display.
    embed_doc = f"{summary}\n\n{content}" if summary else content
    # Defensive type check — chroma's default ONNX embedder forwards
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
    # Dedicated upsert try so a TextInputSequence failure (HF tokenizer)
    # surfaces precise diagnostic fields instead of a bare error message.
    # The intermittent 'TextInputSequence must be str in upsert' we kept
    # seeing in the live MCP server — despite embed_doc already being
    # str-typed here — implies the chroma embedder was being fed something
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
    # original flow had no try/except split here — a fresh block keeps
    # downstream failures distinguishable from the chroma upsert itself.
    try:
        # Register record as a first-class graph node in SQLite.
        try:
            _STATE.kg.add_entity(
                memory_id,
                kind="record",
                description=content[:200],
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
            pass  # Non-fatal — record exists in ChromaDB regardless

        # ── Context fingerprint: keywords → entity_keywords table,
        # view vectors → feedback_contexts collection, context_id → entities row.
        # context is optional here so legacy intent.py callers (which still pass
        # synthetic kwargs) keep working — when present, full Context wiring engages.
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
                pass  # Non-fatal — memory exists regardless

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
            # Only link to entities that already exist — don't auto-create junk stubs
            existing_entity = _STATE.kg.get_entity(eid)
            if not existing_entity:
                # Entity doesn't exist — skip the link. Agent should declare entities
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
                    conflict_id = f"conflict_memory_{memory_id}_{did}"
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


def tool_kg_delete_entity(entity_id: str, agent: str = None):
    """Delete an entity (memory or KG node) and invalidate every edge touching it.

    Works for both memories (ids starting with 'drawer_' or 'diary_' —
    historical prefixes kept for DB compatibility) and KG entities.
    Invalidates all current edges where the target is subject or object
    (soft-delete, temporal audit trail preserved), then removes its
    description from the appropriate Chroma collection.

    Use this when an entity is truly obsolete (superseded concept, stale memory,
    deleted person). For updating a single fact (one edge becomes untrue while
    the entity itself remains valid), use kg_invalidate(subject, predicate,
    object) on the specific triple instead.
    """
    sid_err = _require_sid(action="kg_delete_entity")
    if sid_err:
        return sid_err
    # mandatory agent attribution.
    agent_err = _require_agent(agent, action="kg_delete_entity")
    if agent_err:
        return agent_err
    if not entity_id or not isinstance(entity_id, str):
        return {"success": False, "error": "entity_id is required (string)."}

    # Determine which collection to target: records live in the record
    # collection; everything else in the entity collection. The 'record_' /
    # 'diary_' id prefixes route to the record collection.
    is_record_id = entity_id.startswith(("record_", "diary_"))
    col = _get_collection() if is_record_id else _get_entity_collection(create=False)
    if not col:
        return (
            _no_palace()
            if is_record_id
            else {
                "success": False,
                "error": "Entity collection not found.",
            }
        )

    existing = None
    try:
        existing = col.get(ids=[entity_id])
    except Exception as e:
        return {"success": False, "error": f"lookup failed: {e}"}
    if not existing or not existing.get("ids"):
        return {
            "success": False,
            "error": f"Not found in {'memories' if is_record_id else 'entities'}: {entity_id}",
        }

    deleted_content = (existing.get("documents") or [""])[0] or ""
    deleted_meta = (existing.get("metadatas") or [{}])[0] or {}

    # Invalidate every current edge involving this entity (both directions).
    invalidated = 0
    try:
        edges = _STATE.kg.query_entity(entity_id, direction="both") or []
        for e in edges:
            if not e.get("current", True):
                continue
            subj = e.get("subject") or ""
            pred = e.get("predicate") or ""
            obj = e.get("object") or ""
            if not (subj and pred and obj):
                continue
            try:
                _STATE.kg.invalidate(subj, pred, obj)
                invalidated += 1
            except Exception:
                continue
    except Exception:
        pass  # kg lookup failure is non-fatal; we still remove from Chroma

    _wal_log(
        "kg_delete_entity",
        {
            "entity_id": entity_id,
            "collection": "memory" if is_record_id else "entity",
            "edges_invalidated": invalidated,
            "deleted_meta": deleted_meta,
            "content_preview": deleted_content[:200],
        },
    )

    try:
        col.delete(ids=[entity_id])
        # Mark the SQLite entities row as deleted so downstream readers
        # that filter by status='active' stop returning it. Without this
        # update the chroma side is gone but the entities row still
        # appears active in get_entity / list_declared / kg_query, which
        # is the bug the 2026-04-25 audit caught (record_ga_agent_
        # gardener_prune_anomaly_blanks_only).
        try:
            conn = _STATE.kg._conn()
            now = datetime.now().isoformat()
            conn.execute(
                "UPDATE entities SET status='deleted', last_touched=? WHERE id=?",
                (now, entity_id),
            )
            conn.commit()
        except Exception as sql_err:
            logger.warning(f"kg_delete_entity: SQL status update failed for {entity_id}: {sql_err}")
        logger.info(
            f"Deleted {'memory' if is_record_id else 'entity'}: {entity_id} ({invalidated} edges invalidated)"
        )
        return {
            "success": True,
            "entity_id": entity_id,
            "source": "memory" if is_record_id else "entity",
            "edges_invalidated": invalidated,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


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
                "(e.g. mempalace_wake_up(agent='ga_agent')) — wake_up uses it to "
                "auto-bootstrap the agent entity + is_a agent edge on a fresh "
                "palace and to scope affinity in L1 retrieval. Alternatively, "
                "create ~/.mempalace/identity.txt with the agent name as its "
                "first token and call wake_up again."
            ),
        }
    return str(agent).strip(), None


def _bootstrap_agent_if_missing(agent):
    """Auto-bootstrap the agent entity on a fresh palace (cold restart safe).

    Direct KG writes here bypass the normal added_by/_require_agent gate —
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
            description=f"Agent: {agent}",
            importance=4,
        )
        _STATE.kg.add_triple(_agent_id, "is_a", "agent")
        _sync_entity_to_chromadb(_agent_id, agent, f"Agent: {agent}", "entity", 4)
        _STATE.declared_entities.add(_agent_id)


def tool_wake_up(agent: str = None):
    """Boot context for a session. Call ONCE at start.

    Returns protocol (behavioral rules), text (identity + top memories),
    and declared (compact summary of auto-declared entities).

    Args:
        agent: Agent identity — MANDATORY. Used for affinity scoring in L1
            AND for cold-restart bootstrap (auto-creates the agent entity
            + is_a agent edge if missing, so subsequent write tools can run
            without hitting the chicken-and-egg deadlock that bites on a
            fresh palace). If omitted, falls back to reading the first
            non-blank token of ``~/.mempalace/identity.txt``; if neither is
            present, wake_up fails with a clear bootstrap instruction.
    """
    try:
        from .layers import MemoryStack
    except Exception as e:
        return {"success": False, "error": f"layers module unavailable: {e}"}

    agent, err = _resolve_wake_up_agent(agent)
    if err is not None:
        return err

    try:
        _bootstrap_agent_if_missing(agent)

        stack = MemoryStack()
        text = stack.wake_up(agent=agent)
        from .knowledge_graph import normalize_entity_name

        # 1. Predicates — declare + collect names
        predicates = _STATE.kg.list_entities(status="active", kind="predicate")
        pred_names = []
        for p in predicates:
            _STATE.declared_entities.add(p["id"])
            pred_names.append(p["id"])

        # 2. Classes — declare + collect names
        classes = _STATE.kg.list_entities(status="active", kind="class")
        class_names = []
        for c in classes:
            _STATE.declared_entities.add(c["id"])
            class_names.append(c["id"])

        # 3. Intent types — walk is-a tree, compact format
        #    Intent types are kind=class (they are types, not instances).
        #    Intent executions are kind=entity with is_a pointing to a class.
        entities = _STATE.kg.list_entities(status="active", kind="class")
        intent_type_ids = set()
        intent_parents = {}
        frontier = {"intent_type"}
        visited_walk = set()
        for _ in range(5):
            if not frontier:
                break
            next_frontier = set()
            for parent_id in frontier:
                if parent_id in visited_walk:
                    continue
                visited_walk.add(parent_id)
                for e in entities:
                    e_edges = _STATE.kg.query_entity(e["id"], direction="outgoing")
                    for edge in e_edges:
                        if edge["predicate"] == "is_a" and edge["current"]:
                            if normalize_entity_name(edge["object"]) == parent_id:
                                intent_type_ids.add(e["id"])
                                intent_parents[e["id"]] = parent_id
                                next_frontier.add(e["id"])
            frontier = next_frontier

        intent_entries = []
        for e in entities:
            if e["id"] in intent_type_ids:
                score = _hybrid_score_fn(
                    similarity=0.0,
                    importance=e.get("importance", 3),
                    date_iso=e.get("last_touched", ""),
                    agent_match=False,
                    last_relevant_iso=None,
                    relevance_feedback=0,
                    mode="l1",
                )
                intent_entries.append((score, e))
        intent_entries.sort(key=lambda x: x[0], reverse=True)

        # Format: top-level as name(Tool1,Tool2), children as name<parent(+AddedTool)
        intent_parts = []
        for _score, e in intent_entries[:20]:
            _STATE.declared_entities.add(e["id"])
            eid = e["id"]
            parent = intent_parents.get(eid, "?")
            _, tools = intent._resolve_intent_profile(eid)
            tool_names = sorted(set(t["tool"] for t in tools)) if tools else []
            if parent == "intent_type":
                intent_parts.append(eid + "(" + ",".join(tool_names) + ")" if tool_names else eid)
            else:
                own_props = e.get("properties", {})
                if isinstance(own_props, str):
                    try:
                        own_props = json.loads(own_props)
                    except Exception:
                        own_props = {}
                own_tools = own_props.get("rules_profile", {}).get("tool_permissions", [])
                own_names = sorted(set(t["tool"] for t in own_tools))
                if own_names:
                    intent_parts.append(eid + "<" + parent + "(+" + ",".join(own_names) + ")")
                else:
                    intent_parts.append(eid + "<" + parent)

        # 4. Top entities (non-intent) — name[importance]
        entity_parts = []
        top_ents = [e for e in entities if e["id"] not in intent_type_ids][:20]
        for e in top_ents:
            _STATE.declared_entities.add(e["id"])
            entity_parts.append(e["id"] + "[" + str(e.get("importance", 3)) + "]")

        # Load learned scoring weights from feedback history. Two scopes:
        #   1. Hybrid score weights (sim / rel / imp / decay / agent) —
        #      learned from per-memory relevance correlations recorded
        #      at finalize_intent.
        #   2. Per-channel RRF weights (cosine / graph / keyword /
        #      context) — learned from which channels surfaced memories
        #      that the agent later rated useful. Same mechanism, same
        #      table, different 'scope'.
        try:
            from .scoring import (
                set_learned_weights,
                set_learned_channel_weights,
                DEFAULT_SEARCH_WEIGHTS,
                DEFAULT_CHANNEL_WEIGHTS,
            )

            learned_hybrid = _STATE.kg.compute_learned_weights(
                DEFAULT_SEARCH_WEIGHTS, scope="hybrid"
            )
            set_learned_weights(learned_hybrid)
            learned_channels = _STATE.kg.compute_learned_weights(
                DEFAULT_CHANNEL_WEIGHTS, scope="channel"
            )
            set_learned_channel_weights(learned_channels)
            # Telemetry: observability for the weight-learning loop.
            # Writes one line to ~/.mempalace/hook_state/weight_log.jsonl
            # each time set_learned_* is invoked (wake_up + finalize_intent).
            # `is_tuned` tells you whether compute_learned_weights actually
            # drifted from the static defaults (requires
            # _A6_WEIGHT_SELFTUNE_ENABLED=True AND ≥ min_samples rows).
            try:
                from datetime import datetime as _dt, timezone as _tz

                _h_tuned = any(
                    abs(float(learned_hybrid.get(k, 0.0)) - float(DEFAULT_SEARCH_WEIGHTS[k])) > 1e-6
                    for k in DEFAULT_SEARCH_WEIGHTS
                )
                _c_tuned = any(
                    abs(float(learned_channels.get(k, 0.0)) - float(DEFAULT_CHANNEL_WEIGHTS[k]))
                    > 1e-6
                    for k in DEFAULT_CHANNEL_WEIGHTS
                )
                _fb_rows = {"hybrid": 0, "channel": 0}
                try:
                    _conn = _STATE.kg._conn()
                    _fb_rows["hybrid"] = int(
                        _conn.execute(
                            "SELECT COUNT(*) FROM scoring_weight_feedback "
                            "WHERE component NOT LIKE 'ch_%'"
                        ).fetchone()[0]
                    )
                    _fb_rows["channel"] = int(
                        _conn.execute(
                            "SELECT COUNT(*) FROM scoring_weight_feedback "
                            "WHERE component LIKE 'ch_%'"
                        ).fetchone()[0]
                    )
                except Exception:
                    pass
                _telemetry_append_jsonl(
                    "weight_log.jsonl",
                    {
                        "ts": _dt.now(_tz.utc).isoformat(timespec="seconds"),
                        "trigger": "wake_up",
                        "selftune_enabled": bool(
                            getattr(_STATE.kg, "_A6_WEIGHT_SELFTUNE_ENABLED", False)
                        ),
                        "feedback_rows": _fb_rows,
                        "hybrid": {
                            "learned": {k: round(float(v), 4) for k, v in learned_hybrid.items()},
                            "default": {
                                k: round(float(v), 4) for k, v in DEFAULT_SEARCH_WEIGHTS.items()
                            },
                            "is_tuned": _h_tuned,
                        },
                        "channel": {
                            "learned": {k: round(float(v), 4) for k, v in learned_channels.items()},
                            "default": {
                                k: round(float(v), 4) for k, v in DEFAULT_CHANNEL_WEIGHTS.items()
                            },
                            "is_tuned": _c_tuned,
                        },
                    },
                )
            except Exception:
                pass
        except Exception:
            pass

        declared = {
            "predicates": ", ".join(sorted(pred_names)),
            "classes": ", ".join(sorted(class_names)),
            "intent_types": " | ".join(intent_parts),
            "entities": ", ".join(entity_parts),
            "count": len(_STATE.declared_entities),
        }
        # Count the whole payload the caller receives — not just `text`.
        # Rough 4-chars-per-token heuristic over text + protocol + declared.
        token_estimate = (len(text) + len(PALACE_PROTOCOL) + len(json.dumps(declared))) // 4
        return {
            "success": True,
            "protocol": PALACE_PROTOCOL,
            "text": text,
            "estimated_tokens": token_estimate,
            "declared": declared,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# tool_update_drawer_metadata removed: merged into tool_kg_update_entity.


# ==================== KNOWLEDGE GRAPH ====================


CONTEXT_EDGE_PREDICATES = frozenset({"rated_useful", "rated_irrelevant", "surfaced"})

# Keys lifted from entity-record metadata into the kg_query `details` block.
# Kept narrow on purpose — the goal is "what IS this entity" in a few fields,
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


def tool_kg_query(
    entity: str,
    as_of: str = None,
    direction: str = "both",
    include_context_edges: bool = False,
):
    """Query the knowledge graph for an entity's relationships AND its own content.

    Supports batch queries: pass a comma-separated list of entity names
    to query multiple entities in one call. Returns results keyed by entity.

    Each response carries:
      - `facts`: list of (subject, predicate, object) triples — the edges.
      - `details`: the entity's own content/summary/kind/importance pulled
        from its representative Chroma record. Omitted when the entity has
        no record (rare — most entities carry at least a declaration view).

    By default, retrieval-bookkeeping edges (rated_useful,
    rated_irrelevant, surfaced) are omitted from `facts` — they fill the
    fact list with per-context noise that drowns out domain knowledge.
    Pass include_context_edges=True to see them (e.g. for retrieval
    audits). When any are hidden, a hidden_context_edges count is
    included in the response so callers know they exist.
    """
    entities = [e.strip() for e in entity.split(",") if e.strip()]

    # Track queried entities for mandatory feedback enforcement
    if _STATE.active_intent and isinstance(_STATE.active_intent.get("accessed_memory_ids"), set):
        for ename in entities:
            _STATE.active_intent["accessed_memory_ids"].add(ename)

    if len(entities) == 1:
        # Single entity — original format for backwards compatibility
        results = _STATE.kg.query_entity(entities[0], as_of=as_of, direction=direction)
        hidden = 0
        if not include_context_edges:
            results, hidden = _filter_context_edges(results)
        out = {"entity": entities[0], "as_of": as_of, "facts": results, "count": len(results)}
        details = _fetch_entity_details(entities[0])
        if details:
            out["details"] = details
        if hidden:
            out["hidden_context_edges"] = hidden
        return out

    # Batch query — return results keyed by entity name
    batch_results = {}
    total_count = 0
    total_hidden = 0
    for ename in entities:
        facts = _STATE.kg.query_entity(ename, as_of=as_of, direction=direction)
        hidden = 0
        if not include_context_edges:
            facts, hidden = _filter_context_edges(facts)
        entry = {"facts": facts, "count": len(facts)}
        details = _fetch_entity_details(ename)
        if details:
            entry["details"] = details
        if hidden:
            entry["hidden_context_edges"] = hidden
        batch_results[ename] = entry
        total_count += len(facts)
        total_hidden += hidden

    out = {"entities": batch_results, "as_of": as_of, "total_count": total_count, "batch": True}
    if total_hidden:
        out["total_hidden_context_edges"] = total_hidden
    return out


def tool_kg_search(  # noqa: C901
    context: dict = None,
    limit: int = 10,
    kind: str = None,
    sort_by: str = "hybrid",
    agent: str = None,
    time_window: dict = None,  # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    queries=None,  # LEGACY: rejected when context missing — see below
):
    """Unified search — records (prose) + entities (KG nodes) in one call.

    Speaks the unified Context object: queries drive Channel A (multi-view
    cosine), keywords drive Channel C (caller-provided exact terms — no
    auto-extraction), entities seed Channel B graph BFS when provided.
    Cross-collection RRF then competes record + entity hits head-to-head.

    Returns the SAME lean shape declare_intent / declare_operation use —
    each hit is {id, text, hybrid_score} — so agents see one uniform
    retrieval surface across injection-time and search-time.

    Args:
        context: MANDATORY Context = {queries, keywords, entities?}.
        limit: Max results across records+entities (default 10; adaptive-K may trim).
        kind: Optional entity-only kind filter (excludes record results).
        sort_by: 'hybrid' (default) — RRF + hybrid_score tiebreaker. 'similarity'.
        agent: Agent name for affinity scoring.
        time_window: optional {start, end} date range (YYYY-MM-DD).
            SOFT DECAY: items inside the window get a scoring boost; items
            outside still appear but rank lower. NOT a hard filter — nothing
            is excluded. Use for temporal scoping ("what happened this week")
            without losing globally-important items that fall outside.
    """
    from .scoring import rrf_merge, validate_context as _validate_context
    from .knowledge_graph import normalize_entity_name

    # ── Reject the legacy `queries` path (Context mandatory) ──
    if context is None and queries is not None:
        return {
            "success": False,
            "error": (
                "`queries` is gone. Pass `context` instead — a dict "
                "with mandatory queries, keywords, and optional entities. Example:\n"
                '  context={"queries": ["auth rate limiting", "brute force hardening"], '
                '"keywords": ["auth", "rate-limit", "brute-force"]}'
            ),
        }

    # ── Validate Context (mandatory) ──
    clean_context, ctx_err = _validate_context(context)
    if ctx_err:
        return ctx_err
    query_views = clean_context["queries"]
    context_keywords = clean_context["keywords"]
    context_entities = clean_context["entities"]

    sanitized_views = [sanitize_query(v)["clean_query"] for v in query_views]
    sanitized_views = [v for v in sanitized_views if v]
    if not sanitized_views:
        return {"success": False, "error": "All queries were empty after sanitization."}

    # ── Context as first-class entity (P1) ──
    # kg_search is an emit site. Mint or reuse a kind="context" entity
    # for the search cue and update the active_context_id — this is the
    # most-recent emit, so subsequent writes in the same tool call are
    # correctly provenanced to what actually triggered them.
    # Precedence: declare_intent sets it on intent creation, then
    # declare_operation / kg_search each overwrite on their own invocation.
    _search_context_id = ""
    _search_context_reused = False
    _search_context_max_sim = 0.0
    try:
        _sc_id, _sc_reused, _sc_ms = context_lookup_or_create(
            queries=sanitized_views,
            keywords=context_keywords,
            entities=context_entities,
            agent=agent or "",
        )
        _search_context_id = _sc_id or ""
        _search_context_reused = bool(_sc_reused)
        _search_context_max_sim = float(_sc_ms or 0.0)
    except Exception:
        _search_context_id = ""
    if _STATE.active_intent is not None and _search_context_id:
        _STATE.active_intent["active_context_id"] = _search_context_id
        _record_context_emit(
            _search_context_id,
            reused=_search_context_reused,
            scope="search",
            queries=sanitized_views,
            keywords=context_keywords,
            entities=context_entities,
        )

    # ── Source scoping: kind → entities only; otherwise search both ──
    search_memories = not bool(kind)
    search_entities = True

    # ── Walk the rated-context neighbourhood ONCE ──
    # Both Channel D (retrieval recall) and hybrid_score's W_REL term
    # (per-memory signed relevance) consume this walk. The walker
    # returns both aggregates; we pass the dict down to multi_channel_
    # search for Channel D and pull rated_scores out for hybrid_score
    # below.
    _rated_walk = (
        walk_rated_neighbourhood(_search_context_id, _STATE.kg)
        if _search_context_id
        else {"rated_scores": {}, "channel_D_list": []}
    )

    try:
        # ── Run pipeline over selected collections ──
        all_lists = {}
        combined_meta = {}

        # Classify a seen_meta entry by its metadata.kind. Post-M1 records
        # AND entities live in the same mempalace_records collection, so a
        # query against "the memory collection" returns BOTH kinds mixed in
        # one result list; we can't infer source from which pipe raised the
        # hit. Derive it from metadata.kind instead.
        _ENTITY_KINDS = {"entity", "class", "predicate", "literal"}

        def _classify_source(info: dict) -> str:
            meta = info.get("meta") or {}
            kind_value = meta.get("kind", "")
            if kind_value in _ENTITY_KINDS:
                return "entity"
            # record, diary, or unlabeled prose → memory
            return "memory"

        if search_memories:
            memory_col = _get_collection(create=False)
            memory_pipe = multi_channel_search(
                memory_col,
                sanitized_views,
                keywords=context_keywords,
                kg=_STATE.kg,
                added_by=agent,
                fetch_limit_per_view=max(limit * 3, 30),
                include_graph=False,
                active_context_id=_search_context_id,
                rated_walk=_rated_walk,
            )
            for name, lst in memory_pipe["ranked_lists"].items():
                all_lists[f"memory_{name}"] = lst
            for mid, info in memory_pipe["seen_meta"].items():
                combined_meta[mid] = {**info, "source": _classify_source(info)}

        if search_entities:
            entity_col = _get_entity_collection(create=False)
            # caller-provided context.entities become explicit graph seeds.
            # When omitted, multi_channel_search falls back to deriving seeds
            # from top cosine hits (current behaviour).
            seed_ids = (
                [normalize_entity_name(e) for e in context_entities] if context_entities else None
            )
            entity_pipe = multi_channel_search(
                entity_col,
                sanitized_views,
                keywords=context_keywords,
                kg=_STATE.kg,
                kind=kind,
                fetch_limit_per_view=max(limit * 3, 30),
                include_graph=True,
                seed_ids=seed_ids,
                active_context_id=_search_context_id,
                rated_walk=_rated_walk,
            )
            for name, lst in entity_pipe["ranked_lists"].items():
                all_lists[f"entity_{name}"] = lst
            for mid, info in entity_pipe["seen_meta"].items():
                # Post-M1 both pipes see the same collection; classify by
                # metadata.kind so memories keep source="memory" even if the
                # entity pipe also surfaced them.
                src = _classify_source(info)
                if mid in combined_meta:
                    # Prefer the existing classification when they agree; a
                    # disagreement means metadata.kind is missing or stale,
                    # so fall through to the freshly-computed value.
                    if combined_meta[mid].get("source") == src:
                        continue
                combined_meta[mid] = {**info, "source": src}

        # Triple verbalizations: query the dedicated mempalace_triples
        # collection alongside memories and entities so structured
        # knowledge surfaces as first-class search results. Skip when a
        # caller-pinned `kind` filter is active (kind targets entity
        # records, not triples).
        if not kind:
            try:
                from .knowledge_graph import _get_triple_collection

                triple_col = _get_triple_collection()
                if triple_col is not None and triple_col.count() > 0:
                    triple_pipe = multi_channel_search(
                        triple_col,
                        sanitized_views,
                        keywords=context_keywords,
                        kg=_STATE.kg,
                        fetch_limit_per_view=max(limit * 3, 30),
                        include_graph=False,
                    )
                    for name, lst in triple_pipe["ranked_lists"].items():
                        all_lists[f"triple_{name}"] = lst
                    for mid, info in triple_pipe["seen_meta"].items():
                        combined_meta[mid] = {**info, "source": "triple"}
            except Exception:
                pass  # triples are an optional enrichment of search results

        if not all_lists:
            return {"results": []}

        # ── Canonical two-stage pipeline ──
        # scoring.two_stage_retrieve runs RRF → hybrid_score rerank →
        # adaptive_k. Same helper declare_intent / declare_operation use,
        # so every tool returns hits on the same scale with the same
        # semantics. sort_by='similarity' bypasses the reranker by
        # routing through a cosine-only path below.
        from .scoring import two_stage_retrieve as _two_stage

        if sort_by == "hybrid":
            feedback_scores = _rated_walk.get("rated_scores", {}) or {}
            reranked, _rrf_scores, _cm = _two_stage(
                all_lists,
                combined_meta,
                agent=agent or "",
                session_id=_STATE.session_id or "",
                intent_type_id="",
                context_feedback=feedback_scores,
                rerank_top_m=max(limit * 3, 50),
                max_k=limit,
                min_k=1,
                time_window=time_window,
            )
        else:
            # similarity-sort: skip the rerank, order by raw cosine only.
            rrf_scores, _cm, _attr = rrf_merge(all_lists)
            reranked = []
            for mid, _rrf in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
                info = combined_meta.get(mid)
                if not info:
                    continue
                reranked.append(
                    {
                        "id": mid,
                        "hybrid_score": float(info.get("similarity", 0.0) or 0.0),
                        "rrf_score": float(_rrf),
                        "text": info.get("doc") or "",
                        "channel": "cosine",
                        "meta": info.get("meta") or {},
                        "similarity": float(info.get("similarity", 0.0) or 0.0),
                        "source": info.get("source", ""),
                    }
                )
            reranked = reranked[:limit]

        # Build a kg_search-shape `top` list so the surfaced-edge writer
        # and telemetry hooks below still work unchanged.
        top = []
        for entry in reranked:
            meta = entry.get("meta") or {}
            source = entry.get("source") or "memory"
            doc = entry.get("text") or ""
            proj = {
                "id": entry["id"],
                "source": source,
                "similarity": entry.get("similarity", 0.0),
                "score": round(float(entry["hybrid_score"]), 4),
                "hybrid_score": round(float(entry["hybrid_score"]), 6),
            }
            if source == "memory":
                summary_val = (meta.get("summary") or "").strip()
                proj["text"] = intent._shorten_preview(summary_val or doc)
            elif source == "triple":
                proj["statement"] = doc[:300]
                proj["subject"] = meta.get("subject", "")
                proj["predicate"] = meta.get("predicate", "")
                proj["object"] = meta.get("object", "")
            else:
                # entity
                proj["name"] = meta.get("name", entry["id"])
                proj["description"] = doc
                proj["kind"] = meta.get("kind", "entity")
                proj["text"] = intent._shorten_preview(doc or meta.get("name", entry["id"]))
            top.append(proj)

        # ── Attach current edges for entity results only ──
        for entry in top:
            if entry["source"] == "entity":
                edges = _STATE.kg.query_entity(entry["id"], direction="both")
                current_edges = [e for e in edges if e.get("current", True)]
                entry["edges"] = current_edges
                entry["edge_count"] = len(current_edges)

        # ── Track accessed items for mandatory feedback enforcement ──
        if _STATE.active_intent and isinstance(
            _STATE.active_intent.get("accessed_memory_ids"), set
        ):
            for entry in top:
                _STATE.active_intent["accessed_memory_ids"].add(entry["id"])

        # ── P2: write `surfaced` edges from active context to each top result ──
        # These are the consumer of finalize_intent's coverage check and
        # feed Channel D on subsequent intents. Each edge carries
        # {ts, rank, channel, sim_score} as structured props so the
        # downstream pipeline has everything it needs without a rejoin.
        if _search_context_id and top:
            now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
            for rank, entry in enumerate(top, start=1):
                _raw_channels = entry.get("channels")
                _chans = sorted(_raw_channels) if isinstance(_raw_channels, (list, set)) else []
                # `channels` stores the full comma-joined set so
                # downstream bucketing attributes memories to every
                # channel that surfaced them.
                props = {
                    "ts": now_iso,
                    "rank": rank,
                    "channels": ",".join(_chans),
                    "sim_score": float(entry.get("similarity", 0.0) or 0.0),
                }
                try:
                    _STATE.kg.add_triple(
                        _search_context_id,
                        "surfaced",
                        entry["id"],
                        properties=props,
                    )
                except Exception:
                    pass  # non-fatal — the search result still returns

        # ── P3 telemetry: JSONL trace for mempalace-eval ──
        try:
            per_channel_hits: dict = {}
            for entry in top:
                chs = entry.get("channels") or []
                if not isinstance(chs, list):
                    continue
                for ch in chs:
                    per_channel_hits[ch] = per_channel_hits.get(ch, 0) + 1
            _telemetry_append_jsonl(
                "search_log.jsonl",
                {
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "active_context_id": _search_context_id,
                    "reused": _search_context_reused,
                    "max_sim": round(_search_context_max_sim, 4),
                    "per_channel_hits": per_channel_hits,
                    "top_k": len(top),
                    "queries": sanitized_views[:3],
                    "agent": agent or "",
                },
            )
        except Exception:
            pass

        # ── Output projection ──
        # Every hit gets the SAME lean shape declare_intent /
        # declare_operation return: {id, text, source, hybrid_score}.
        # `source` is load-bearing for kg_search callers that mix memory
        # / entity / triple hits — the three carry different downstream
        # affordances (entity hits unlock kg_query for edges; memory hits
        # are ready to read). Fetch the full entity / triple / edges via
        # mempalace_kg_query when you need the structured detail.
        projected = []
        for entry in top:
            lean = {
                "id": entry["id"],
                "text": entry.get("text", ""),
                "source": entry.get("source") or "memory",
            }
            if intent.DEBUG_RETURN_SCORES and "hybrid_score" in entry:
                lean["hybrid_score"] = entry["hybrid_score"]
            projected.append(lean)

        # ── Injection-stage gate ──
        # Same wiring pattern as declare_intent / declare_operation.
        # Parent frame = the active intent (if any) that this search
        # is nested inside; standalone searches have no parent. Fail-
        # open on any error — search must still work even if the gate
        # is broken.
        _kg_gate_status = None
        try:
            from .injection_gate import apply_gate as _apply_gate

            _kg_combined = {
                entry["id"]: {
                    "source": entry.get("source") or "memory",
                    "doc": entry.get("text") or "",
                    "similarity": float(entry.get("hybrid_score") or 0.0),
                }
                for entry in top
                if entry.get("id")
            }
            _kg_parent = None
            try:
                ai = _STATE.active_intent or {}
                if ai:
                    _kg_parent = {
                        "intent_type": ai.get("intent_type"),
                        "subject": ", ".join((ai.get("slots", {}) or {}).get("subject", []) or []),
                        "query": (ai.get("description_views") or [""])[0],
                    }
            except Exception:
                _kg_parent = None

            _gated, _kg_gate_status = _apply_gate(
                memories=projected,
                combined_meta=_kg_combined,
                primary_context={
                    "source": "kg_search",
                    "queries": list(sanitized_views),
                    "keywords": list(context_keywords or []),
                    "entities": list((context.get("entities") if context else None) or []),
                },
                context_id=_search_context_id or "",
                kg=_STATE.kg,
                agent=agent,
                parent_intent=_kg_parent,
            )
            projected = _gated
        except Exception:
            pass

        response = {"results": projected}
        if _kg_gate_status is not None:
            response["gate_status"] = _kg_gate_status
        if intent.DEBUG_RETURN_CONTEXT:
            # Debug overlay mirroring declare_intent / declare_operation.
            # Token-diet 2026-04-23: queries are echoed ONLY on reuse;
            # a fresh-mint context carries the caller's own queries so
            # there's nothing new to show.
            # Token-diet 2026-04-24: non-reused collapses to "new";
            # reused returns {id, queries}. Shape-as-signal: string
            # "new" = fresh mint; object = reused.
            if _search_context_reused:
                response["context"] = {
                    "id": _search_context_id,
                    "queries": list(sanitized_views),
                }
            else:
                response["context"] = "new"
        return response
    except Exception as e:
        return {"success": False, "error": f"kg_search failed: {e}"}


def tool_kg_add(  # noqa: C901
    subject: str,
    predicate: str,
    object: str,
    context: dict = None,  # mandatory Context fingerprint for the edge
    agent: str = None,  # mandatory attribution
    valid_from: str = None,
    statement: str = None,  # natural-language verbalization for retrieval
):
    """Add a relationship to the knowledge graph (Context mandatory).

    IMPORTANT: All three parts must be declared in this session:
    - subject: declared entity (any type EXCEPT predicate)
    - predicate: declared entity with type="predicate"
    - object: declared entity (any type EXCEPT predicate)

    The MANDATORY `context` records WHY this edge is being added — the
    multi-view perspectives + caller-provided keywords + optional related
    entities. Its view vectors are persisted in mempalace_feedback_contexts
    and the resulting context_id is stored on the triple's
    creation_context_id column. Future feedback (found_useful etc.) applies
    by MaxSim against this fingerprint.

    `agent` is mandatory. Every write operation must be attributed
    to a declared agent (is_a agent); undeclared agents are rejected
    up-front with a declaration recipe.

    `statement` is REQUIRED for every predicate OUTSIDE the skip list
    (is_a, described_by, evidenced_by, executed_by, targeted, has_value,
    session_note_for, derived_from, mentioned_in, found_useful,
    found_irrelevant). It is the natural-language verbalization of the
    triple — e.g. statement="Adrian lives in Warsaw" for
    ('adrian','lives_in','warsaw'). The statement is stored on the row
    AND embedded into the mempalace_triples Chroma collection so the
    triple becomes a first-class semantic-search result. Autogeneration
    was retired 2026-04-19 — naive fallbacks poisoned retrieval with
    low-signal text like "record X relates to record Y".

    Call kg_declare_entity for subject/object entities, and
    kg_declare_entity with kind="predicate" for predicates.
    """
    from .knowledge_graph import (
        normalize_entity_name,
        _TRIPLE_SKIP_PREDICATES,
        _normalize_predicate,
        TripleStatementRequired,
    )
    from .scoring import validate_context

    # ── Validate Context (mandatory) ──
    clean_context, ctx_err = validate_context(context)
    if ctx_err:
        return ctx_err

    # ── mandatory agent attribution ──
    sid_err = _require_sid(action="kg_add")
    if sid_err:
        return sid_err
    agent_err = _require_agent(agent, action="kg_add")
    if agent_err:
        return agent_err

    # ── mandatory statement for non-skip predicates ──
    # Surfaced here at the MCP tool layer (instead of only trusting
    # add_triple's TripleStatementRequired raise) so the agent sees a
    # clean structured error long before the SQL write path.
    _pred_for_check = _normalize_predicate(predicate or "")
    if _pred_for_check and _pred_for_check not in _TRIPLE_SKIP_PREDICATES:
        if not statement or not str(statement).strip():
            return {
                "success": False,
                "error": (
                    f"`statement` is required for predicate '{_pred_for_check}'. "
                    f"Write a natural-language sentence verbalizing the triple "
                    f'(e.g. statement="Adrian lives in Warsaw"). It is stored '
                    f"on the triple row AND embedded into mempalace_triples for "
                    f"semantic search. Skip-list predicates (is_a, described_by, "
                    f"evidenced_by, executed_by, targeted, has_value, "
                    f"session_note_for, derived_from, mentioned_in, found_useful, "
                    f"found_irrelevant) may omit statement \u2014 they are never "
                    f"embedded regardless. Autogeneration was retired 2026-04-19 "
                    f"because naive fallbacks produced retrieval-poisoning text."
                ),
            }

    try:
        subject = sanitize_name(subject, "subject")
        predicate = sanitize_name(predicate, "predicate")
        object = sanitize_name(object, "object")
    except ValueError as e:
        return {"success": False, "error": str(e)}

    # Enforce entity declaration: subject, predicate, and object must all be declared
    sub_normalized = normalize_entity_name(subject)
    pred_normalized = normalize_entity_name(predicate)
    obj_normalized = normalize_entity_name(object)

    errors = []

    # Check subject (must be declared, must NOT be a predicate)
    if not _is_declared(sub_normalized):
        errors.append(
            f"subject '{sub_normalized}' not declared. Call: "
            + _declare_entity_recipe(subject, kind="entity")
        )
    else:
        sub_entity = _STATE.kg.get_entity(sub_normalized)
        if sub_entity and sub_entity.get("kind") == "predicate":
            errors.append(
                f"subject '{sub_normalized}' is kind=predicate, not an entity. "
                f"Subjects must be kind=entity (or class/literal)."
            )

    # Check predicate (must be declared as type="predicate")
    if not _is_declared(pred_normalized):
        errors.append(
            f"predicate '{pred_normalized}' not declared. Call: "
            + _declare_entity_recipe(predicate, kind="predicate")
        )
    else:
        pred_entity = _STATE.kg.get_entity(pred_normalized)
        if pred_entity and pred_entity.get("kind") != "predicate":
            errors.append(
                f"'{pred_normalized}' is kind='{pred_entity.get('kind')}', not 'predicate'. "
                f"Predicates must be declared with kind='predicate'."
            )

    # Check object (must be declared, must NOT be a predicate)
    if not _is_declared(obj_normalized):
        errors.append(
            f"object '{obj_normalized}' not declared. Call: "
            + _declare_entity_recipe(object, kind="entity")
        )
    else:
        obj_entity = _STATE.kg.get_entity(obj_normalized)
        if obj_entity and obj_entity.get("kind") == "predicate":
            errors.append(
                f"object '{obj_normalized}' is kind=predicate, not an entity. "
                f"Objects must be kind=entity (or class/literal)."
            )

    if errors:
        return {
            "success": False,
            "error": "Declaration validation failed for kg_add.",
            "issues": errors,
        }

    # ── Class inheritance helper ──
    def _is_subclass_of(entity_classes, allowed_classes, max_depth=5):
        """Check if any of entity_classes is a subclass of any allowed_class.

        Walks is-a edges upward from each entity class. If it reaches an
        allowed class within max_depth hops, returns True. This enables
        class inheritance: if 'system is-a thing' and constraint allows
        'thing', then any system entity passes.
        """
        if not allowed_classes:
            return True  # no constraint = pass
        # Direct match first
        if any(c in allowed_classes for c in entity_classes):
            return True
        # Walk is-a hierarchy upward
        visited = set(entity_classes)
        frontier = list(entity_classes)
        for _ in range(max_depth):
            next_frontier = []
            for cls in frontier:
                parent_edges = _STATE.kg.query_entity(cls, direction="outgoing")
                for e in parent_edges:
                    if e["predicate"] == "is_a" and e["current"]:
                        parent = e["object"]
                        if parent in allowed_classes:
                            return True
                        if parent not in visited:
                            visited.add(parent)
                            next_frontier.append(parent)
            frontier = next_frontier
            if not frontier:
                break
        return False

    # ── Constraint enforcement ──
    constraint_errors = []
    if pred_entity:
        props = pred_entity.get("properties", {})
        if isinstance(props, str):
            import json as _json

            try:
                props = _json.loads(props)
            except Exception:
                props = {}
        pred_constraints = props.get("constraints", {})

        if pred_constraints:
            # Check subject kind constraint
            allowed_sub_kinds = pred_constraints.get("subject_kinds", [])
            if allowed_sub_kinds and sub_entity:
                sub_kind = sub_entity.get("kind", "entity")
                if sub_kind not in allowed_sub_kinds:
                    constraint_errors.append(
                        f"Subject kind mismatch: '{sub_normalized}' is kind='{sub_kind}', "
                        f"but predicate '{pred_normalized}' expects subject kind in {allowed_sub_kinds}."
                    )

            # Check subject class constraint (via is-a edges)
            allowed_sub_classes = pred_constraints.get("subject_classes", [])
            if allowed_sub_classes and sub_entity:
                sub_classes = [
                    e["object"]
                    for e in _STATE.kg.query_entity(sub_normalized, direction="outgoing")
                    if e["predicate"] == "is_a" and e["current"]
                ]
                if sub_classes and not _is_subclass_of(sub_classes, allowed_sub_classes):
                    constraint_errors.append(
                        f"Subject class mismatch: '{sub_normalized}' is_a {sub_classes}, "
                        f"but predicate '{pred_normalized}' expects subject is_a {allowed_sub_classes}. "
                        f"Options: (1) wrong edge — use a different subject, "
                        f"(2) wrong predicate — check kg_list_declared() for a better fit, "
                        f"(3) missing classification — add is_a edge for '{sub_normalized}', "
                        f"(4) update predicate constraints, "
                        f"(5) create a more specific predicate, "
                        f"(6) rephrase with a more specific entity."
                    )

            # Check object kind constraint
            allowed_obj_kinds = pred_constraints.get("object_kinds", [])
            if allowed_obj_kinds and obj_entity:
                obj_kind = obj_entity.get("kind", "entity")
                if obj_kind not in allowed_obj_kinds:
                    constraint_errors.append(
                        f"Object kind mismatch: '{obj_normalized}' is kind='{obj_kind}', "
                        f"but predicate '{pred_normalized}' expects object kind in {allowed_obj_kinds}."
                    )

            # Check object class constraint
            allowed_obj_classes = pred_constraints.get("object_classes", [])
            if allowed_obj_classes and obj_entity:
                obj_classes = [
                    e["object"]
                    for e in _STATE.kg.query_entity(obj_normalized, direction="outgoing")
                    if e["predicate"] == "is_a" and e["current"]
                ]
                if obj_classes and not _is_subclass_of(obj_classes, allowed_obj_classes):
                    constraint_errors.append(
                        f"Object class mismatch: '{obj_normalized}' is_a {obj_classes}, "
                        f"but predicate '{pred_normalized}' expects object is_a {allowed_obj_classes}. "
                        f"Options: (1) wrong edge, (2) wrong predicate, (3) missing classification, "
                        f"(4) update constraints, (5) new predicate, (6) rephrase with specific entity."
                    )

            # Check cardinality constraint
            cardinality = pred_constraints.get("cardinality", "many-to-many")
            if cardinality in ("many-to-one", "one-to-one"):
                # Subject can have at most 1 edge with this predicate
                existing_sub = [
                    e
                    for e in _STATE.kg.query_entity(sub_normalized, direction="outgoing")
                    if e["predicate"] == pred_normalized and e["current"]
                ]
                if existing_sub:
                    existing_obj = existing_sub[0]["object"]
                    constraint_errors.append(
                        f"Cardinality violation: '{sub_normalized}' already has "
                        f"'{pred_normalized}' -> '{existing_obj}'. "
                        f"Predicate cardinality is {cardinality} (one target per subject). "
                        f"Options: (1) REPLACE — invalidate the old edge first, then add new, "
                        f"(2) MISTAKE — you meant a different predicate or entity, "
                        f"(3) EXPAND — change predicate cardinality to many-to-many."
                    )
            if cardinality in ("one-to-many", "one-to-one"):
                # Object can have at most 1 incoming edge with this predicate
                existing_obj = [
                    e
                    for e in _STATE.kg.query_entity(obj_normalized, direction="incoming")
                    if e["predicate"] == pred_normalized and e["current"]
                ]
                if existing_obj:
                    existing_sub = existing_obj[0]["subject"]
                    constraint_errors.append(
                        f"Cardinality violation: '{obj_normalized}' already has incoming "
                        f"'{existing_sub}' -> '{pred_normalized}'. "
                        f"Predicate cardinality is {cardinality} (one source per object). "
                        f"Options: (1) REPLACE, (2) MISTAKE, (3) EXPAND cardinality."
                    )

    if constraint_errors:
        return {
            "success": False,
            "error": "Predicate constraint violation.",
            "constraint_issues": constraint_errors,
        }

    # ── Provenance: triples.creation_context_id ──
    # Points at the active context entity (kind='context') so the
    # context's outgoing created_under accretion is mirrored on triples
    # via this column. triples aren't first-class entities (no entity
    # row), so a direct created_under edge is inappropriate — the
    # column is the provenance vehicle. Backed by:
    #   _STATE.kg.triples_created_under(context_id) -> [triple_ids]
    # and the existing JOIN-friendly idx_triples_creation_ctx index.
    # The old persist_context (retired) returned an empty string; we
    # now use the active context id directly when one exists.
    edge_context_id = _active_context_id() or ""

    _wal_log(
        "kg_add",
        {
            "subject": subject,
            "predicate": predicate,
            "object": object,
            "valid_from": valid_from,
            "context_id": edge_context_id,
        },
    )
    try:
        triple_id = _STATE.kg.add_triple(
            sub_normalized,
            pred_normalized,
            obj_normalized,
            valid_from=valid_from,
            creation_context_id=edge_context_id,
            statement=statement,
        )
    except TripleStatementRequired as exc:
        # Should already have been caught by the guard above, but keep
        # this as defense-in-depth in case a future refactor changes the
        # predicate-normalization order.
        return {"success": False, "error": str(exc)}

    # ── Contradiction detection: find existing edges that may conflict ──
    conflicts = []
    try:
        # Skip is_a — those aren't factual contradictions
        if pred_normalized != "is_a":
            existing_edges = _STATE.kg.query_entity(sub_normalized, direction="outgoing")
            for e in existing_edges:
                if not e.get("current", True):
                    continue
                if e["predicate"] != pred_normalized:
                    continue
                existing_obj = e["object"]
                if existing_obj == obj_normalized:
                    continue  # Same edge — not a contradiction
                # Found: same subject + same predicate + different object
                conflict_id = f"conflict_{sub_normalized}_{pred_normalized}_{existing_obj}"
                past = None
                try:
                    past = _STATE.kg.get_past_conflict_resolution(
                        existing_obj, obj_normalized, "edge_contradiction"
                    )
                except Exception:
                    pass
                conflict_entry = {
                    "id": conflict_id,
                    "conflict_type": "edge_contradiction",
                    "reason": (
                        f"Same subject+predicate, different object: "
                        f"existing '{existing_obj}' vs new '{obj_normalized}'"
                    ),
                    "existing_id": existing_obj,
                    "existing_subject": sub_normalized,
                    "existing_predicate": pred_normalized,
                    "existing_object": existing_obj,
                    "new_id": obj_normalized,
                    "new_subject": sub_normalized,
                    "new_predicate": pred_normalized,
                    "new_object": obj_normalized,
                }
                if past:
                    conflict_entry["past_resolution"] = past
                conflicts.append(conflict_entry)
    except Exception:
        pass  # Non-fatal — contradiction detection is best-effort

    result = {
        "success": True,
        "triple_id": triple_id,
        "fact": f"{sub_normalized} -> {pred_normalized} -> {obj_normalized}",
    }

    if conflicts:
        _STATE.pending_conflicts = conflicts
        intent._persist_active_intent()
        result["conflicts"] = conflicts
        past_hint = _past_resolution_hint(conflicts)
        result["conflicts_prompt"] = (
            f"{len(conflicts)} potential contradiction(s) found. "
            f"You MUST call mempalace_resolve_conflicts to address each: "
            f"invalidate (old is stale), keep (both valid), or skip (undo new)." + past_hint
        )

    return result


def tool_kg_add_batch(edges: list, context: dict = None, agent: str = None):
    """Add multiple KG edges in one call (Context mandatory).

    Each edge: {subject, predicate, object, statement?, valid_from?, context?}.

    `statement` (per-edge) is REQUIRED for every edge whose predicate is
    OUTSIDE the skip list (is_a, described_by, evidenced_by, executed_by,
    targeted, has_value, session_note_for, derived_from, mentioned_in,
    found_useful, found_irrelevant). Writing a proper natural-language
    verbalization — e.g. "Adrian lives in Warsaw" for ('adrian','lives_in',
    'warsaw') — lets the triple surface via semantic search in the
    mempalace_triples Chroma collection. Omitting it on a non-skip edge
    returns a per-edge error; skip-list edges may omit it (never embedded).

    The TOP-LEVEL `context` is the shared default applied to every edge that
    doesn't carry its own — most batches add edges that all reflect the same
    'why' (a single agent decision), so one Context covers them. An edge can
    still override with its own `context` dict if needed. Validates each edge
    independently — partial success OK.

    `agent` is mandatory (same validation as kg_add). Applies to the
    whole batch; per-edge agent overrides are not supported (batches are
    single-author by design).
    """
    from .scoring import validate_context

    # Some MCP transports stringify top-level array parameters.
    if isinstance(edges, str):
        try:
            edges = json.loads(edges)
        except Exception:
            return {
                "success": False,
                "error": (
                    "`edges` arrived as an unparseable string. Pass a JSON array "
                    "of {subject, predicate, object, ...} objects."
                ),
            }
    if not edges or not isinstance(edges, list):
        return {
            "success": False,
            "error": "edges must be a non-empty list of {subject, predicate, object} dicts.",
        }

    # ── agent validation up-front so we don't partially apply ──
    sid_err = _require_sid(action="kg_add_batch")
    if sid_err:
        return sid_err
    agent_err = _require_agent(agent, action="kg_add_batch")
    if agent_err:
        return agent_err

    # Validate the shared/default context (if provided) once up front so we
    # surface a clean error before doing any per-edge work.
    default_clean_context = None
    if context is not None:
        default_clean_context, ctx_err = validate_context(context)
        if ctx_err:
            return ctx_err

    failures = []
    succeeded_triples = []
    all_conflicts = []
    for idx, edge in enumerate(edges):
        if not isinstance(edge, dict):
            failures.append({"index": idx, "error": "edge must be a dict"})
            continue
        edge_context = edge.get("context") or default_clean_context
        if edge_context is None:
            failures.append(
                {
                    "index": idx,
                    "subject": edge.get("subject"),
                    "predicate": edge.get("predicate"),
                    "object": edge.get("object"),
                    "error": (
                        "Each edge needs a context — pass one at the top level of "
                        "kg_add_batch (shared default for all edges) or per-edge."
                    ),
                }
            )
            continue
        r = tool_kg_add(
            subject=edge.get("subject", ""),
            predicate=edge.get("predicate", ""),
            object=edge.get("object", ""),
            context=edge_context,
            agent=agent,
            valid_from=edge.get("valid_from"),
            statement=edge.get("statement"),
        )
        if r.get("success"):
            # Keep only the triple_id — caller supplied the s/p/o.
            if r.get("triple_id"):
                succeeded_triples.append(r["triple_id"])
            if r.get("conflicts"):
                all_conflicts.extend(r["conflicts"])
        else:
            failures.append(
                {
                    "index": idx,
                    "subject": edge.get("subject"),
                    "predicate": edge.get("predicate"),
                    "object": edge.get("object"),
                    "error": r.get("error"),
                    "issues": r.get("issues") or r.get("constraint_issues"),
                }
            )

    # Caller supplied the s/p/o/statement for each edge — echoing them back
    # is pure token waste. Return counts on success; surface per-edge detail
    # only for failures and any surfaced conflicts.
    response = {
        "success": len(succeeded_triples) > 0,
        "total": len(edges),
        "succeeded": len(succeeded_triples),
        "failed": len(failures),
    }
    if succeeded_triples:
        response["triple_ids"] = succeeded_triples
    if failures:
        response["failures"] = failures
    if all_conflicts:
        response["conflicts"] = all_conflicts
    return response


def tool_kg_invalidate(
    subject: str,
    predicate: str,
    object: str,
    ended: str = None,
    agent: str = None,
):
    """Mark a fact as no longer true (set end date). agent required."""
    sid_err = _require_sid(action="kg_invalidate")
    if sid_err:
        return sid_err
    agent_err = _require_agent(agent, action="kg_invalidate")
    if agent_err:
        return agent_err
    try:
        _wal_log(
            "kg_invalidate",
            {
                "subject": subject,
                "predicate": predicate,
                "object": object,
                "ended": ended,
                "agent": agent,
            },
        )
        _STATE.kg.invalidate(subject, predicate, object, ended=ended)
        return {"success": True, "ended": ended or "today"}
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


def tool_kg_timeline(entity: str = None):
    """Get chronological timeline of facts, optionally for one entity."""
    results = _STATE.kg.timeline(entity)
    return {"timeline": results, "count": len(results)}


def tool_kg_stats():
    """Knowledge graph overview — entities, triples, relationship types."""
    stats = _STATE.kg.stats() or {}
    return stats


# ==================== ENTITY DECLARATION ====================

# TODO (threshold): ENTITY_SIMILARITY_THRESHOLD is a strong learning
# candidate. Outcome signal = was the detected collision actually a
# duplicate (agent merged them) or distinct (agent kept both)?
# Correlate sim-at-detection with resolve_conflicts action and sweep
# the threshold. Needs ~50 collision decisions for meaningful signal.
ENTITY_SIMILARITY_THRESHOLD = 0.85
# Legacy — mempalace_entities was absorbed into mempalace_records by the M1
# migration. Kept as a module constant only so the migration can look it
# up when scanning for legacy rows on startup.
ENTITY_COLLECTION_NAME = "mempalace_entities"

# Session-level declared entities (in-memory cache on _STATE, falls back to persistent KG).
# _STATE.pending_conflicts blocks all tools until resolved.
# Defaults to None on ServerState construction — no explicit init needed here.

# ── Session isolation: save/restore state per session_id ──
# _STATE.session_state maps session_id -> {active_intent, pending_conflicts,
# declared}. When multiple callers (sub-agents) share the same MCP process
# but have different session IDs, this prevents them from overwriting each
# other's state.


def _sanitize_session_id(session_id: str) -> str:
    """Match hooks_cli._sanitize_session_id — only alnum/dash/underscore.

    Returns "" for empty or fully-stripped input. NO FALLBACK to
    'unknown' or 'default' — callers must handle empty sid explicitly.
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
    here — they live on disk and are re-read via _load_pending_*_from_disk
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

    Pending conflicts are NOT read from the in-memory cache — they always
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
    # additive) so that a cleared file becomes cleared state — the old
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
    hook wiring — which is what we want.

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
                f"empty sid — using a shared default would cross-contaminate "
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
    Never raises — KG lookup failures are treated as pass (graceful
    degradation when KG is unavailable, e.g. in fresh test fixtures).
    """
    # Gardener bypass — same rationale as _require_sid. The gardener
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
                # at the sanctioned bootstrap path — wake_up — instead.
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
                            f"palace — bootstrap via wake_up, not kg_declare_entity "
                            f"(which requires a pre-existing declared agent and is "
                            f"circular here). Call: "
                            f"mempalace_wake_up(agent='{agent}') — it auto-creates "
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

    Single source of truth — DO NOT hand-roll `description=...` in new
    error strings; the tool rejects it (see tool_kg_declare_entity, the
    P4.2 legacy-path block). `context={queries,keywords}` is mandatory.

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
        f"'keywords': ['<term1>', '<term2>']}}, "
        f"added_by='<your_agent>'"
        f"{props})"
    )


def _declare_intent_recipe(intent_type: str = "modify", slots: str = None) -> str:
    """Canonical mempalace_declare_intent recipe for error messages.

    Single source of truth — the tool requires `context={queries,keywords}`
    AND `budget`; the old `description=` path is gone.
    """
    slots = slots or '{"files": ["target_file"]}'
    return (
        f"mempalace_declare_intent(intent_type='{intent_type}', slots={slots}, "
        f"context={{'queries': ['<what you plan to do>', '<another angle>'], "
        f"'keywords': ['<term1>', '<term2>']}}, "
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
      1. In-memory cache — _STATE.declared_entities (session-lifetime fast path).
      2. Multi-view entities — where={"entity_id": X}. Entities declared
         via kg_declare_entity(context=...) live under '{eid}__v{N}' ids
         with metadata.entity_id=eid (see _sync_entity_views_to_chromadb).
      3. Single-record entities — ids=[X]. Internal bookkeeping entities
         (execution traces, gotchas) written by _sync_entity_to_chromadb
         use raw '{eid}' as the Chroma id, so the id-based lookup still
         applies. Same for the memories collection.
    """
    if entity_id in _STATE.declared_entities:
        return True

    # KG fallback — SQLite is the authoritative source of truth.
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
    by ``metadata.kind``. This helper remains for callsite compatibility —
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
    concept cleanly. Data layout is untouched — only the metadata.kind
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
    rows use ``record_<agent>_<slug>`` IDs — two non-overlapping
    namespaces — so merging cannot create ID collisions in the target
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
            return  # Legacy collection never existed — fresh palace.

        dest = _get_collection(create=True)
        if dest is None:
            logger.warning("M1 migration: target mempalace_records unavailable")
            return

        got = legacy.get(include=["documents", "metadatas", "embeddings"])
        if not got or not got.get("ids"):
            # Collection exists but is empty — drop it.
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
                f"NOT deleted — partial copy detected, data preserved."
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


# Retired Chroma collection name — kept ONLY as a string for the
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
# ColBERT-style MaxSim (Khattab & Zaharia 2020, arXiv:2004.12832) — a new
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
# Hardcoded intuition picks — NOT derived from data. 0.90 is the "clearly
# same topic" cut for ColBERT MaxSim on multi-view embeddings; 0.70 is the
# "similar but distinct" cut (triggers similar_to edge + new context).
#
# TODO (threshold calibration — highest-ROI tunable left in the system):
#   1. Log MaxSim-best on every emit (already partially recorded via
#      search_log.jsonl telemetry in eval_harness; extend to
#      declare_intent and declare_operation too).
#   2. Correlate MaxSim-at-decision with mean relevance of the resulting
#      feedback on the reused context. High-MaxSim + bad-feedback →
#      T_reuse too loose. Low-MaxSim + good-feedback at T_similar border →
#      T_similar too loose.
#   3. Offline sweep 0.80 → 0.95 in 0.01 steps; maximise
#      (useful_reuses / (useful_reuses + bad_reuses)) subject to a
#      reuse-rate floor (≥ ~30% or retrieval-memory signal disappears).
#   4. Once stable, fold the learned values into a kv table that
#      tool_wake_up reads alongside the hybrid + channel weights.
# Needs ~50-100 intents with feedback before calibration is reliable
# (binary decision outcome; most observations are either well-above or
# well-below threshold — the diagnostic band at 0.85-0.95 is narrow).
CONTEXT_REUSE_THRESHOLD = 0.90
CONTEXT_SIMILAR_THRESHOLD = 0.70


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

    Reference: Rocchio 1971 (Manning/Raghavan/Schütze IR book, Ch.9) —
    the query-reformulation algorithm shifts the retrieval query toward
    the centroid of relevant results and away from non-relevant ones.
    Our adaptation is to the *context entity itself*: when an existing
    context is reused (MaxSim ≥ T_reuse) AND the intent finishes with
    net-positive feedback, we merge the caller's NEW queries, keywords,
    and related entities into the context so future MaxSim lookups land
    on it more easily. The context accretes a shape that reflects every
    successful past use.

    LRU cap of ``max_views`` on the view vector list (default 20) — when
    the count would exceed the cap, drop the oldest view_index. Every
    view added gets a fresh view_index slot so "oldest" is well-defined
    without an explicit timestamp.

    Dedup strategy per field:
      - queries: TWO-step dedup — exact-text first, then semantic MaxSim
        against the context's existing view vectors (threshold 0.85 via
        ROCCHIO_QUERY_DEDUP_THRESHOLD). Redundant-angle queries are
        dropped before they eat an LRU slot. The semantic check is
        cheap — one Chroma query per novel query, scoped to this
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
    # Controlled-vocabulary tags — embedding per-keyword is noisy on
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
    # to the same id — so we compare normalised forms on both sides.
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
            description=ctx_entity.get("description", "") or (updated_queries[0][:200]),
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
                # pool, so leaving stale vectors in Chroma is harmless — the
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
    required by the MaxSim math — see Khattab & Zaharia 2020).
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
    """Mint a fresh, stable-ish context entity id.

    normalize_entity_name keeps `[a-z0-9_]+` strings unchanged, so the id
    survives round-tripping through add_entity.
    """
    import hashlib
    import secrets
    import time

    text = "\n".join(sorted(v.strip() for v in (views or []) if isinstance(v, str)))
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    ns = time.time_ns()
    nonce = secrets.token_hex(3)
    # "ctx_" prefix distinguishes context entities from everything else and
    # keeps them easy to grep; normalize_entity_name is a no-op on this shape.
    return f"ctx_{digest}_{ns}_{nonce}"


def _compute_context_maxsim(current_views: list, candidate_context_ids: list, col) -> dict:
    """MaxSim score per candidate context, from current views against stored ones.

    MaxSim(A, B) = (1/|A|) * Σ_a max_b cos(a, b)   — ColBERT late interaction.
    """
    results = {}
    if not current_views or not candidate_context_ids or col is None:
        return results
    try:
        total = col.count()
        if total == 0:
            return results
    except Exception:
        return results
    for cid in candidate_context_ids:
        max_sims = []
        for view in current_views:
            if not view or not view.strip():
                continue
            try:
                res = col.query(
                    query_texts=[view],
                    n_results=min(10, total),
                    where={"context_id": cid},
                    include=["distances"],
                )
                if res.get("distances") and res["distances"] and res["distances"][0]:
                    min_dist = min(res["distances"][0])
                    max_sim = max(0.0, 1.0 - float(min_dist))
                    max_sims.append(max_sim)
            except Exception:
                continue
        if max_sims:
            results[cid] = sum(max_sims) / len(max_sims)
    return results


def context_lookup_or_create(
    queries,
    keywords=None,
    entities=None,
    agent: str = None,
    *,
    t_reuse: float = None,
    t_similar: float = None,
) -> tuple:
    """Reuse an existing context by MaxSim or mint a new first-class context entity.

    Returns ``(context_id, reused, max_sim)``.

    Used by the three emit sites (``tool_declare_intent``,
    ``tool_declare_operation``, ``tool_kg_search``). Every other writer
    references the active context via ``created_under`` instead of creating
    its own — only these three sites emit contexts.

    The threshold branches are (BIRCH-inspired, Zhang et al. 1996):
      - max_sim ≥ t_reuse (0.90) → return existing, no write.
      - t_similar ≤ max_sim < t_reuse → create new + write `similar_to`
        edge to the nearest neighbour (prop {sim}).
      - max_sim < t_similar → create new, no `similar_to`.
    """
    t_reuse = CONTEXT_REUSE_THRESHOLD if t_reuse is None else float(t_reuse)
    t_similar = CONTEXT_SIMILAR_THRESHOLD if t_similar is None else float(t_similar)
    views = [q.strip() for q in (queries or []) if isinstance(q, str) and q.strip()]
    if not views:
        return "", False, 0.0

    col = _get_context_views_collection(create=True)
    if col is None:
        return "", False, 0.0

    # 1. Collect candidate context ids — top-K per-view neighbours, union'd.
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

    # 2. Full MaxSim for each candidate.
    best_id, best_sim = None, 0.0
    if candidate_ids:
        sims = _compute_context_maxsim(views, list(candidate_ids), col)
        if sims:
            best_id, best_sim = max(sims.items(), key=lambda kv: kv[1])

    # 3. Reuse branch.
    if best_id and best_sim >= t_reuse:
        # Touch last_touched by re-adding the entity (add_entity is upsert).
        try:
            existing = _STATE.kg.get_entity(best_id)
            if existing and existing.get("kind") == "context":
                _STATE.kg.add_entity(
                    best_id,
                    kind="context",
                    description=existing.get("description", "") or (views[0][:200]),
                    importance=existing.get("importance", 3) or 3,
                    properties=existing.get("properties", {}) or {},
                )
        except Exception:
            pass
        return best_id, True, float(best_sim)

    # 4. Mint a fresh context entity.
    new_cid = _mint_context_entity_id(views)
    props = {
        "queries": list(views),
        "keywords": [k for k in (keywords or []) if isinstance(k, str) and k.strip()],
        "entities": [e for e in (entities or []) if isinstance(e, str) and e.strip()],
        "agent": agent or "",
    }
    description = views[0][:200] if views else "context"
    try:
        _STATE.kg.add_entity(
            new_cid,
            kind="context",
            description=description,
            importance=3,
            properties=props,
        )
    except Exception:
        return "", False, float(best_sim)
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
        # View persistence failed — the entity row still exists but the
        # context won't be lookup-able. Mark it so ops can find it.
        try:
            bad_props = dict(props)
            bad_props["_views_persisted"] = False
            _STATE.kg.add_entity(
                new_cid,
                kind="context",
                description=description,
                importance=3,
                properties=bad_props,
            )
        except Exception:
            pass

    # 6. similar_to edge if best_id is in [t_similar, t_reuse).
    # The triples table has no generic properties column in P1 — we stuff
    # the MaxSim into the `confidence` field (semantically compatible: a
    # similar_to edge's "confidence" is exactly how similar the two
    # contexts are). P2 adds richer props for surfaced/rated_* via a
    # schema change.
    if best_id and t_similar <= best_sim < t_reuse:
        try:
            _STATE.kg.add_triple(
                new_cid,
                "similar_to",
                best_id,
                confidence=round(float(best_sim), 4),
            )
        except Exception as exc:
            # similar_to is in _TRIPLE_SKIP_PREDICATES so this CANNOT
            # be a missing-statement issue; any failure here is a real
            # DB/constraint/programming bug. Log loudly so it surfaces
            # in operator logs, but do NOT crash declare_* — a missing
            # similar_to edge degrades retrieval neighbourhood quality
            # for one context, not the entire intent flow. (Silent
            # bare-except retired 2026-04-25 per Adrian's rule:
            # "ensure this crashes, and is not silently omitted".)
            logger.warning(
                "context_lookup_or_create: similar_to write failed (%s -> %s, sim=%.4f): %s",
                new_cid,
                best_id,
                best_sim,
                exc,
            )

    return new_cid, False, float(best_sim)


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
    per-view ranked candidates are merged via Reciprocal Rank Fusion. A hit
    is reported as a collision when its highest single-view similarity is
    above threshold (so one strong match still flags, but multi-view gives
    catches that single-vector cosine misses).

    Logical entity ids are read from metadata.entity_id (no id
    string splitting). Returns the same shape as _check_entity_similarity
    for drop-in compatibility.
    """
    from .scoring import rrf_merge

    threshold = threshold or ENTITY_SIMILARITY_THRESHOLD
    ecol = _get_entity_collection(create=False)
    if not ecol or not views:
        return []
    try:
        count = ecol.count()
        if count == 0:
            return []
        per_view_lists = {}
        per_id_best = {}  # logical entity_id -> (best_similarity, doc, meta)
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
                dist = results["distances"][0][i]
                sim = round(1 - dist, 3)
                if kind_filter and meta.get("kind") != kind_filter:
                    continue
                doc = results["documents"][0][i] or ""
                view_candidates.append((sim, doc, logical_id))
                prev = per_id_best.get(logical_id)
                if prev is None or sim > prev[0]:
                    per_id_best[logical_id] = (sim, doc, meta)
            if view_candidates:
                per_view_lists[f"cosine_{vi}"] = view_candidates

        if not per_view_lists:
            return []

        rrf_scores, _cm, _attr = rrf_merge(per_view_lists)
        # Order by RRF, but only emit entities whose best single-view sim is above threshold.
        similar = []
        for eid, _rrf in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            best = per_id_best.get(eid)
            if not best:
                continue
            sim, doc, meta = best
            if sim < threshold:
                continue
            similar.append(
                {
                    "entity_id": eid,
                    "name": meta.get("name", eid),
                    "description": doc,
                    "similarity": sim,
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
    description: str = "",
    importance: int = 3,
    properties: dict = None,
    added_by: str = None,
    embed_text: str = None,
):
    """Create an entity in BOTH SQLite AND ChromaDB. Use this instead of _STATE.kg.add_entity directly.

    Args:
        embed_text: Optional override for what gets embedded in ChromaDB.
                    If None, uses description. Use for execution entities
                    where you want description-only embedding (no summary).
    """
    from .knowledge_graph import normalize_entity_name

    # pass provenance to SQLite
    _prov_session = _STATE.session_id or ""
    _prov_intent = _STATE.active_intent.get("intent_id", "") if _STATE.active_intent else ""
    eid = _STATE.kg.add_entity(
        name,
        kind=kind,
        description=description,
        importance=importance,
        properties=properties,
        session_id=_prov_session,
        intent_id=_prov_intent,
    )
    normalized = normalize_entity_name(name)
    _sync_entity_to_chromadb(
        normalized, name, embed_text or description, kind, importance, added_by=added_by
    )
    return eid


def _sync_entity_to_chromadb(
    entity_id: str, name: str, description: str, kind: str, importance: int, added_by: str = None
):
    """Single-description Chroma sync for internal bookkeeping entities.

    Used by intent.py finalize (execution entities, gotcha entities) and by
    tool_kg_update_entity after a description-only update — both cases
    naturally have one description and don't carry a multi-view Context.
    For Context-driven entity declarations, use _sync_entity_views_to_chromadb
    (multi-vector storage under '{entity_id}::view_N').

    kind='operation' is graph-only by design — ops are reachable via edges
    (executed_op, performed_well, performed_poorly) but never embedded. See
    arXiv 2512.18950 (Operation tier in hierarchical procedural memory).
    """
    if kind == "operation":
        return
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
    by the metadata field (col.get(where={'entity_id': X})) — NOT by parsing
    ids — so the separator choice is cosmetic, not load-bearing.

    The '__v' separator is deliberately chosen because it cannot appear
    inside a normalized_entity_name (which uses single-underscore segments
    only), making the literal id unambiguous for humans skimming the db.

    kind='operation' is graph-only and skipped here too — consistent with
    _sync_entity_to_chromadb. Cf. arXiv 2512.18950.
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


def tool_kg_declare_entity(  # noqa: C901
    name: str = None,
    context: dict = None,  # mandatory: {queries, keywords, entities?}
    kind: str = None,  # REQUIRED — no default, model must choose
    importance: int = 3,
    properties: dict = None,  # General-purpose metadata
    user_approved_star_scope: bool = False,  # Required for * scope
    added_by: str = None,  # REQUIRED — agent who declared this entity
    # Record-kind specific (REQUIRED when kind='record').
    slug: str = None,
    content: str = None,  # verbatim record text (kind='record' only); for other kinds, queries[0] is canonical
    content_type: str = None,  # one of: fact, event, discovery, preference, advice, diary
    source_file: str = None,
    entity: str = None,  # entity name(s) to link this record to
    predicate: str = "described_by",  # link predicate
    summary: str = None,  # ≤280-char distilled one-sentence WHAT+WHY — REQUIRED on every kind (Anthropic Contextual Retrieval 2024). For kind='record' it's enforced; for other kinds the value defaults to queries[0] as a legacy fallback AND the entity is auto-flagged for memory_gardener refinement. Always prefer supplying a real summary at declare time — the flag-then-refine path costs a gardener cycle.
    # ── Legacy single-string description path (REMOVED) ──
    description: str = None,  # accepted only as a hard-error trigger, see below
):
    """Declare an entity before using it in KG edges. REQUIRED per session.

    EVERY declaration speaks the unified Context object:

        context = {
          "queries":  list[str]   # 2-5 perspectives on what this entity is
          "keywords": list[str]   # 2-5 caller-provided exact terms
          "entities": list[str]   # 0+ related entity ids (optional)
        }

    Each query gets embedded as a separate Chroma record under
    '{entity_id}__v{N}' with metadata.entity_id=entity_id, so
    collision detection is multi-view RRF rather than single-vector
    cosine. Readers look up entities via where={"entity_id": X} — the
    suffix is cosmetic, the metadata is load-bearing. Keywords are stored
    in entity_keywords (the keyword channel reads them directly —
    auto-extraction is gone). The Context's view vectors are also
    persisted in mempalace_feedback_contexts under a generated
    context_id, recorded on the entity, so future feedback (found_useful
    / found_irrelevant) applies via MaxSim by context similarity.

    Args:
        name: Entity name (REQUIRED for kind=entity/class/predicate/literal;
              auto-computed from added_by/slug for kind='record').
        context: MANDATORY Context dict — see above. Replaces the single
              `description` parameter (which is now rejected with an error).
        kind: 'entity' | 'class' | 'predicate' | 'literal' | 'memory'.
        content: VERBATIM text for kind='record' (the actual memory body).
              For non-memory kinds, queries[0] is used as the canonical
              description; pass `content` only when you need to override it.
        importance: 1-5.
        properties: predicate constraints / intent type rules_profile / arbitrary metadata.
        user_approved_star_scope: required only for "*" tool scopes.
        added_by: declared agent name (REQUIRED).
        slug/content_type/source_file/entity/predicate: kind='record' only.

    Returns: status "created" | "exists" | "collision".
    """
    from .knowledge_graph import normalize_entity_name
    from .scoring import validate_context

    sid_err = _require_sid(action="kg_declare_entity")
    if sid_err:
        return sid_err

    # ── Reject the legacy single-string description path ──
    if description is not None and context is None:
        return {
            "success": False,
            "error": (
                "`description` is gone. Pass `context` instead — a dict "
                "with mandatory queries (list of 2-5 perspectives) and keywords "
                "(list of 2-5 caller-provided terms). Example:\n"
                '  context={"queries": ["DSpot platform server", "paperclip backend on :3100"], '
                '"keywords": ["dspot", "paperclip", "server", "port-3100"]}\n'
                "queries[0] becomes the canonical description for non-memory kinds."
            ),
        }

    # ── Validate Context (mandatory) ──
    clean_context, ctx_err = validate_context(context)
    if ctx_err:
        return ctx_err
    queries = clean_context["queries"]
    keywords = clean_context["keywords"]
    # clean_context["entities"] is reserved for graph-anchor wiring in P4.3+ (kg_add).

    # ── kind='record' dispatch — records are first-class entities.
    if kind == "record":
        if content is None or not str(content).strip():
            return {
                "success": False,
                "error": (
                    "kind='record' requires `content` — the verbatim record text. "
                    "(`context.queries` are search angles, not the body.) "
                    "Use kg_declare_entity(kind='record', slug=..., "
                    "content='<full text>', context={...}, added_by=..., ...)."
                ),
            }
        if not slug:
            return {
                "success": False,
                "error": (
                    "kind='record' requires slug and added_by. "
                    "Slug is a short human-readable identifier (3-6 hyphenated words)."
                ),
            }
        return _add_memory_internal(
            content=content,
            slug=slug,
            added_by=added_by,
            content_type=content_type,
            importance=importance,
            entity=entity,
            predicate=predicate,
            context=clean_context,
            source_file=source_file,
            summary=summary,
        )

    # Non-memory: queries[0] is the canonical description used for SQLite + first chroma vector.
    description = queries[0]

    try:
        description = sanitize_content(description, max_length=5000)
        importance = _validate_importance(importance)
        kind = _validate_kind(kind)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    if not name or not str(name).strip():
        return {
            "success": False,
            "error": (
                "name is required for kind='entity', 'class', 'predicate', or 'literal'. "
                "(For kind='record', use slug + content + added_by instead.)"
            ),
        }

    # Validate added_by: REQUIRED, must be a declared agent (is_a agent)
    if not added_by:
        return {
            "success": False,
            "error": "added_by is required. Pass your agent entity name (e.g., 'ga_agent', 'technical_lead_agent').",
        }
    agent_id_check = normalize_entity_name(added_by)
    if _STATE.kg:
        agent_edges = _STATE.kg.query_entity(agent_id_check, direction="outgoing")
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

    # Check for * scope in tool_permissions — requires user approval
    if properties and not user_approved_star_scope:
        rules_profile = properties.get("rules_profile", {})
        tool_perms = rules_profile.get("tool_permissions", [])
        star_tools = [p["tool"] for p in tool_perms if p.get("scope") == "*"]
        if star_tools:
            return {
                "success": False,
                "error": (
                    f'BLOCKED: Unrestricted scope ("*") for tools: {star_tools}.\n\n'
                    f"MANDATORY: You MUST ask the user RIGHT NOW and get an explicit YES "
                    f"before proceeding. Do NOT self-approve. Do NOT assume prior approval. "
                    f"Do NOT set user_approved_star_scope=true without asking.\n\n"
                    f"Ask the user exactly this:\n"
                    f"  \"I need to create intent type '{name}' with unrestricted (*) access "
                    f'to {", ".join(star_tools)}. This bypasses scope restrictions. Approve? (yes/no)"\n\n'
                    f"ONLY if the user responds YES to that question in this conversation turn, "
                    f"retry with user_approved_star_scope=true.\n"
                    f"If the user says NO or does not respond: use scoped permissions instead.\n"
                    f"If this is a non-interactive session: this is a BLOCKER. Do not proceed."
                ),
                "star_tools": star_tools,
                "needs_approval": True,
            }

    # Validate constraints for predicates (read from properties.constraints)
    constraints = (properties or {}).get("constraints") if properties else None
    if kind == "predicate":
        if not constraints:
            return {
                "success": False,
                "error": (
                    "Predicates REQUIRE constraints in properties. ALL fields are mandatory: "
                    "subject_kinds, object_kinds, subject_classes, object_classes, cardinality. "
                    'Example: properties={"constraints": {"subject_kinds": ["entity"], "object_kinds": ["entity"], '
                    '"subject_classes": ["system","process"], "object_classes": ["thing"], '
                    '"cardinality": "many-to-one"}}'
                ),
            }
        # ALL 5 constraint fields are REQUIRED — no optionals
        for field in (
            "subject_kinds",
            "object_kinds",
            "subject_classes",
            "object_classes",
            "cardinality",
        ):
            if field not in constraints:
                return {
                    "success": False,
                    "error": f"constraints must include '{field}'. ALL 5 fields are required: subject_kinds, object_kinds, subject_classes, object_classes, cardinality.",
                }
        # Validate kind lists
        for field in ("subject_kinds", "object_kinds"):
            vals = constraints[field]
            if not isinstance(vals, list) or not vals:
                return {
                    "success": False,
                    "error": f"constraints['{field}'] must be a non-empty list of kinds.",
                }
            for v in vals:
                if v not in VALID_KINDS:
                    return {
                        "success": False,
                        "error": f"constraints['{field}'] contains invalid kind '{v}'. Valid: {sorted(VALID_KINDS)}.",
                    }
        # Validate cardinality
        if constraints["cardinality"] not in VALID_CARDINALITIES:
            return {
                "success": False,
                "error": f"constraints['cardinality'] must be one of {sorted(VALID_CARDINALITIES)}.",
            }
        # Validate subject_classes / object_classes reference real class-kind entities
        for cls_field in ("subject_classes", "object_classes"):
            cls_list = constraints[cls_field]
            if not isinstance(cls_list, list) or not cls_list:
                return {
                    "success": False,
                    "error": f"constraints['{cls_field}'] must be a non-empty list of class entity names. Use ['thing'] for any class.",
                }
            for cls_name in cls_list:
                from .knowledge_graph import normalize_entity_name as _norm

                cls_entity = _STATE.kg.get_entity(_norm(cls_name))
                if not cls_entity:
                    return {
                        "success": False,
                        "error": f"constraints['{cls_field}'] references class '{cls_name}' which doesn't exist. Declare it first with kind='class'.",
                    }
                if cls_entity.get("kind") != "class":
                    return {
                        "success": False,
                        "error": f"constraints['{cls_field}'] references '{cls_name}' which is kind='{cls_entity.get('kind')}', not 'class'.",
                    }

    normalized = normalize_entity_name(name)
    if not normalized or normalized == "unknown":
        return {
            "success": False,
            "error": f"Entity name '{name}' normalizes to nothing. Use a more descriptive name.",
        }

    # Check for exact match (already exists)
    existing = _STATE.kg.get_entity(normalized)
    if existing:
        # Check for collisions with OTHER entities of SAME KIND (not self) — multi-view
        similar = _check_entity_similarity_multiview(
            queries, kind_filter=kind, exclude_id=normalized
        )
        if similar:
            return {
                "success": False,
                "status": "collision",
                "entity_id": normalized,
                "kind": kind,
                "message": (
                    f"Entity '{normalized}' (kind={kind}) collides with other {kind}s. "
                    f"Disambiguate via kg_update_entity or merge via kg_merge_entities."
                ),
                "collisions": similar,
            }
        # No collisions — register in session
        _STATE.declared_entities.add(normalized)
        # Update description + importance + kind if provided and different
        if description and description != existing.get("description", ""):
            _STATE.kg.update_entity_description(normalized, description, importance)
            _sync_entity_views_to_chromadb(
                normalized, name, queries, kind, importance or 3, added_by=added_by
            )
        # Update properties if provided (merge with existing)
        if properties and isinstance(properties, dict):
            _STATE.kg.update_entity_properties(normalized, properties)
        # Refresh keywords (caller may have updated them)
        _STATE.kg.add_entity_keywords(normalized, keywords)
        return {
            "success": True,
            "status": "exists",
            "entity_id": normalized,
            "kind": existing.get("kind", "entity"),
            "description": existing.get("description") or description,
            "importance": existing.get("importance", 3),
            "edge_count": _STATE.kg.entity_edge_count(normalized),
        }

    # New entity — multi-view collision check
    similar = _check_entity_similarity_multiview(queries, kind_filter=kind)

    # Create the entity regardless — conflicts are resolved after creation
    props = properties if isinstance(properties, dict) else {}
    if added_by:
        props["added_by"] = added_by
    # SQLite row first (with queries[0] as the canonical description)
    _STATE.kg.add_entity(
        name, kind=kind, description=description, importance=importance or 3, properties=props
    )
    # Multi-vector embedding into the entity Chroma collection (one record per view)
    _sync_entity_views_to_chromadb(
        normalized, name, queries, kind, importance or 3, added_by=added_by
    )
    # Caller-provided keywords → entity_keywords table
    _STATE.kg.add_entity_keywords(normalized, keywords)
    # Stamp creation_context_id from the active context entity when
    # one exists. Replaces the retired persist_context path.
    active_ctx = _active_context_id()
    if active_ctx:
        _STATE.kg.set_entity_creation_context(normalized, active_ctx)
    _STATE.declared_entities.add(normalized)

    # ── P1 created_under provenance edge ──
    # Every declared entity records the link to the active context
    # entity. Skip when normalized refers to a context itself (no
    # self-reference) or when the taxonomic root classes are re-seeded.
    _active_ctx = _active_context_id()
    if _active_ctx and normalized != _active_ctx and kind != "context":
        try:
            _STATE.kg.add_triple(normalized, "created_under", _active_ctx)
        except Exception:
            pass  # Non-fatal — entity exists regardless

    # Auto-add is-a thing for new class entities (ensures class inheritance works)
    if kind == "class" and normalized != "thing":
        try:
            _STATE.kg.add_triple(normalized, "is_a", "thing")
        except Exception:
            pass  # Non-fatal if thing doesn't exist yet

    _wal_log(
        "kg_declare_entity",
        {
            "entity_id": normalized,
            "name": name,
            "description": description[:200],
            "kind": kind,
            "importance": importance,
        },
    )

    result = {
        "success": True,
        "status": "created",
        "entity_id": normalized,
        "kind": kind,
        "description": description,
        "importance": importance or 3,
    }

    # ── Conflict detection: flag similar entities for resolution ──
    if similar:
        conflicts = []
        for s in similar:
            conflict_id = f"conflict_entity_{normalized}_{s['entity_id']}"
            past = None
            try:
                past = _STATE.kg.get_past_conflict_resolution(
                    s["entity_id"], normalized, "entity_duplicate"
                )
            except Exception:
                pass
            conflict_entry = {
                "id": conflict_id,
                "conflict_type": "entity_duplicate",
                "reason": (
                    f"New entity '{normalized}' has similar description to "
                    f"existing '{s['entity_id']}' (similarity: {s.get('similarity', '?')})"
                ),
                "existing_id": s["entity_id"],
                "existing_description": s.get("description", "")[:200],
                "new_id": normalized,
                "new_description": description[:200],
            }
            if past:
                conflict_entry["past_resolution"] = past
            conflicts.append(conflict_entry)
        _STATE.pending_conflicts = conflicts
        intent._persist_active_intent()
        result["conflicts"] = conflicts
        past_hint = _past_resolution_hint(conflicts)
        result["conflicts_prompt"] = (
            f"{len(conflicts)} similar entity/entities found. "
            f"Call mempalace_resolve_conflicts: merge (combine both), "
            f"keep (both are distinct), or skip (undo new entity)." + past_hint
        )

    return result


def tool_kg_update_entity(  # noqa: C901
    entity: str,
    description: str = None,
    importance: int = None,
    properties: dict = None,
    context: dict = None,  # optional: re-record creation_context when meaning changes
    agent: str = None,  # mandatory attribution
    # Record-specific (only meaningful when entity is a kind='record')
    content_type: str = None,
):
    """Update any entity (record or KG node) in place. Pass only the fields you want to change.

    `context` is OPTIONAL but RECOMMENDED whenever you change
    semantic fields (`description` for entities, or `properties` that alter
    meaning like predicate constraints / intent-type rules). When present
    the Context's view vectors are persisted and the entity's
    creation_context_id is repointed to the new context — future MaxSim
    feedback then transfers against the updated meaning, not the old one.

    Args:
        entity: Entity ID or record ID to update.
        description: New description. For entities (kind=entity/class/predicate/literal):
            re-syncs to entity ChromaDB and runs collision distance check.
            For records: NOT supported here — use kg_delete_entity +
            kg_declare_entity to change record content.
        importance: New importance (1-5). Works for both entities and records.
        properties: Merged INTO existing properties dict (shallow merge at top
            level). For predicates use {"constraints": {...}} to replace
            constraints. For intent types {"rules_profile": {...}} to update slots
            or tool_permissions.
        content_type: Record-only content type update (no re-embedding).
    """
    from .knowledge_graph import normalize_entity_name
    import json as _json

    if not entity or not isinstance(entity, str):
        return {"success": False, "error": "entity is required (string)."}

    # ── mandatory agent attribution ──
    sid_err = _require_sid(action="kg_update_entity")
    if sid_err:
        return sid_err
    agent_err = _require_agent(agent, action="kg_update_entity")
    if agent_err:
        return agent_err

    is_record_id = entity.startswith(("record_", "diary_"))

    # ── Validate inputs ──
    try:
        if description is not None:
            description = sanitize_content(description, max_length=5000)
        if importance is not None:
            importance = _validate_importance(importance)
        if content_type is not None:
            content_type = _validate_content_type(content_type)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    # Reject contradictory inputs early
    if is_record_id and description is not None:
        return {
            "success": False,
            "error": (
                "Cannot update record description in place — embeddings would be "
                "stale. Use kg_delete_entity then kg_declare_entity(kind='record', ...) "
                "to replace record content."
            ),
        }
    if not is_record_id and content_type is not None:
        return {
            "success": False,
            "error": (
                "content_type is a record-only field. For non-record entities, "
                "use properties={...} to update metadata."
            ),
        }

    # ── Memory path: in-place metadata update on the memory collection ──
    if is_record_id:
        col = _get_collection()
        if not col:
            return _no_palace()
        existing = col.get(ids=[entity], include=["metadatas"])
        if not existing.get("ids"):
            return {"success": False, "error": f"Memory not found: {entity}"}

        old_meta = dict(existing["metadatas"][0] or {})
        new_meta = dict(old_meta)
        updated_fields = []
        if content_type is not None and old_meta.get("content_type") != content_type:
            new_meta["content_type"] = content_type
            updated_fields.append("content_type")
        if importance is not None and old_meta.get("importance") != importance:
            new_meta["importance"] = importance
            updated_fields.append("importance")

        if not updated_fields:
            return {"success": True, "reason": "no_change", "entity_id": entity}

        _wal_log(
            "kg_update_entity",
            {
                "entity_id": entity,
                "source": "memory",
                "old_meta": old_meta,
                "new_meta": new_meta,
                "updated_fields": updated_fields,
            },
        )
        try:
            col.update(ids=[entity], metadatas=[new_meta])
            logger.info(f"Updated memory: {entity} fields={updated_fields}")
            return {
                "success": True,
                "entity_id": entity,
                "source": "memory",
                "updated_fields": updated_fields,
                "new_metadata": new_meta,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Entity path: SQLite update + ChromaDB sync + collision check ──
    normalized = normalize_entity_name(entity)
    existing = _STATE.kg.get_entity(normalized)
    if not existing:
        return {"success": False, "error": f"Entity '{normalized}' not found."}

    updated_fields = []
    final_description = existing["description"]
    final_importance = existing.get("importance", 3)

    # Rewrite-count limit (closes 2026-04-25 audit finding #5).
    # Bound how many times memory_gardener can rewrite an entity's
    # description so unstable rewrites don't compound across runs.
    # Only enforced for the gardener; manual edits stay unconstrained.
    # The counter lives in entities.properties JSON so no migration is
    # required.
    if (
        description is not None
        and description != existing["description"]
        and (agent or "").strip() == "memory_gardener"
    ):
        _existing_props_for_count = existing.get("properties", {})
        if isinstance(_existing_props_for_count, str):
            try:
                _existing_props_for_count = _json.loads(_existing_props_for_count)
            except Exception:
                _existing_props_for_count = {}
        _rewrite_count = int(_existing_props_for_count.get("summary_rewrite_count", 0) or 0)
        if _rewrite_count >= 3:
            return {
                "success": False,
                "error": (
                    f"Refusing description rewrite on '{normalized}': memory_gardener "
                    f"has already rewritten this entity {_rewrite_count} times. Multiple "
                    "rewrites without convergence usually indicate weak grounding "
                    "evidence; defer this flag instead of compounding the drift."
                ),
                "summary_rewrite_count": _rewrite_count,
            }
        # Inject the bumped counter into the user-supplied properties so
        # the existing merge-and-persist code at line ~4676 picks it up.
        properties = dict(properties or {})
        properties["summary_rewrite_count"] = _rewrite_count + 1

    # Description update + ChromaDB resync
    if description is not None and description != existing["description"]:
        _STATE.kg.update_entity_description(normalized, description)
        final_description = description
        updated_fields.append("description")

    # Properties merge (constraints validation when kind='predicate')
    if properties is not None:
        existing_props = existing.get("properties", {})
        if isinstance(existing_props, str):
            try:
                existing_props = _json.loads(existing_props)
            except Exception:
                existing_props = {}

        # If updating constraints on a predicate, validate before persisting.
        if "constraints" in properties and existing.get("kind") == "predicate":
            constraints = properties["constraints"]
            for field in ("subject_kinds", "object_kinds"):
                if field not in constraints:
                    return {
                        "success": False,
                        "error": f"constraints must include '{field}'.",
                    }
                vals = constraints[field]
                if not isinstance(vals, list) or not vals:
                    return {
                        "success": False,
                        "error": f"constraints['{field}'] must be a non-empty list.",
                    }
                for v in vals:
                    if v not in VALID_KINDS:
                        return {
                            "success": False,
                            "error": f"Invalid kind '{v}' in constraints['{field}']. Valid: {sorted(VALID_KINDS)}.",
                        }
            if "cardinality" in constraints:
                if constraints["cardinality"] not in VALID_CARDINALITIES:
                    return {
                        "success": False,
                        "error": f"Invalid cardinality. Valid: {sorted(VALID_CARDINALITIES)}.",
                    }
            for cls_field in ("subject_classes", "object_classes"):
                if cls_field in constraints:
                    for cls_name in constraints[cls_field]:
                        cls_eid = normalize_entity_name(cls_name)
                        cls_ent = _STATE.kg.get_entity(cls_eid)
                        if not cls_ent:
                            return {
                                "success": False,
                                "error": f"Class '{cls_name}' not found. Declare with kind='class' first.",
                            }
                        if cls_ent.get("kind") != "class":
                            return {
                                "success": False,
                                "error": f"'{cls_name}' is kind='{cls_ent.get('kind')}', not 'class'.",
                            }

        merged_props = dict(existing_props or {})
        merged_props.update(properties)  # shallow merge
        conn = _STATE.kg._conn()
        conn.execute(
            "UPDATE entities SET properties = ? WHERE id = ?",
            (_json.dumps(merged_props), normalized),
        )
        conn.commit()
        updated_fields.append("properties")

    # Importance update
    if importance is not None and importance != existing.get("importance"):
        conn = _STATE.kg._conn()
        conn.execute(
            "UPDATE entities SET importance = ? WHERE id = ?",
            (importance, normalized),
        )
        conn.commit()
        final_importance = importance
        updated_fields.append("importance")

    if not updated_fields:
        return {"success": True, "reason": "no_change", "entity_id": normalized}

    # Re-sync ChromaDB if description or importance changed (description embeds, importance is metadata)
    if "description" in updated_fields or "importance" in updated_fields:
        _sync_entity_to_chromadb(
            normalized,
            existing["name"],
            final_description,
            existing.get("kind") or existing.get("type", "entity"),
            final_importance,
        )

    # ── re-record creation_context when meaning changed ──
    # A description or properties change IS a semantic update — future
    # MaxSim-graded feedback should attach to the new meaning, not the old.
    # Pure-importance updates don't move meaning, so we skip context re-persist
    # unless description/properties changed too.
    semantic_change = any(f in updated_fields for f in ("description", "properties"))
    if semantic_change and context is not None:
        from .scoring import validate_context as _validate_context

        clean_ctx, ctx_err = _validate_context(context)
        if ctx_err:
            return ctx_err
        # Stamp the active context on the entity; refresh its stored
        # keywords to the new Context.keywords.
        active_ctx = _active_context_id()
        if active_ctx:
            _STATE.kg.set_entity_creation_context(normalized, active_ctx)
        _STATE.kg.add_entity_keywords(normalized, clean_ctx["keywords"])
        updated_fields.append("creation_context")

    _wal_log(
        "kg_update_entity",
        {"entity_id": normalized, "source": "entity", "updated_fields": updated_fields},
    )

    result = {
        "success": True,
        "entity_id": normalized,
        "source": "entity",
        "updated_fields": updated_fields,
    }
    # Stamp the current active context on the response when we just
    # wrote it (semantic-change path above sets creation_context_id to
    # _active_context_id()). Empty when we're in a non-semantic update.
    _new_ctx = _active_context_id() if semantic_change else ""
    if _new_ctx:
        result["creation_context_id"] = _new_ctx

    # P5.10 hint: nudge callers to pass context when meaning changed
    # but they omitted the context dict entirely.
    if semantic_change and context is None:
        result["context_hint"] = (
            "Description/properties changed but no `context` was provided — "
            "future MaxSim feedback will still attach to the OLD creation_context_id. "
            "Pass `context={queries,keywords,entities?}` to re-anchor the entity."
        )

    # Collision distance check when description changed (was the point of the
    # legacy update_entity_description tool — keep that behaviour).
    if "description" in updated_fields:
        similar = _check_entity_similarity(final_description, exclude_id=normalized, threshold=0.7)
        distance_checks = [
            {
                "compared_to": s["entity_id"],
                "similarity": s["similarity"],
                "is_distinct": s["similarity"] < ENTITY_SIMILARITY_THRESHOLD,
                "threshold": ENTITY_SIMILARITY_THRESHOLD,
            }
            for s in similar
        ]
        all_distinct = all(d["is_distinct"] for d in distance_checks) if distance_checks else True
        result["distance_checks"] = distance_checks
        result["all_distinct"] = all_distinct
        result["hint"] = (
            "All clear — re-declare this entity to register it."
            if all_distinct
            else "Still too similar to some entities. Make your description more specific."
        )

    return result


def tool_kg_merge_entities(
    source: str, target: str, update_description: str = None, agent: str = None
):
    """Merge source entity into target. All edges rewritten. Source becomes alias.

    Use when kg_declare_entity returns 'collision' and the entities are
    actually the same thing. All triples from source are moved to target.
    Source name becomes an alias that auto-resolves to target in future queries.

    Args:
        source: Entity to merge FROM (will be soft-deleted).
        target: Entity to merge INTO (will be kept, edges grow).
        update_description: Optional new description for the merged entity.
        agent: mandatory, declared agent attributing this merge.
    """
    sid_err = _require_sid(action="kg_merge_entities")
    if sid_err:
        return sid_err
    agent_err = _require_agent(agent, action="kg_merge_entities")
    if agent_err:
        return agent_err
    _wal_log(
        "kg_merge_entities",
        {
            "source": source,
            "target": target,
            "update_description": update_description[:200] if update_description else None,
            "agent": agent,
        },
    )

    result = _STATE.kg.merge_entities(source, target, update_description)
    if "error" in result:
        return {"success": False, "error": result["error"]}

    # Update ChromaDB: remove source, update target
    from .knowledge_graph import normalize_entity_name

    source_id = normalize_entity_name(source)
    target_id = normalize_entity_name(target)

    ecol = _get_entity_collection(create=False)
    if ecol:
        try:
            ecol.delete(ids=[source_id])
        except Exception:
            pass
        target_entity = _STATE.kg.get_entity(target_id)
        if target_entity:
            _sync_entity_to_chromadb(
                target_id,
                target_entity["name"],
                target_entity["description"],
                target_entity.get("type", "concept"),
                target_entity.get("importance", 3),
            )

    # Register target as declared (source is now alias for target)
    _STATE.declared_entities.discard(source_id)
    _STATE.declared_entities.add(target_id)

    return {
        "success": True,
        "source": result["source"],
        "target": result["target"],
        "edges_moved": result["edges_moved"],
        "aliases_created": result["aliases_created"],
    }


# tool_kg_update_predicate_constraints removed: merged into tool_kg_update_entity.
# Call kg_update_entity(entity=predicate, properties={"constraints": {...}}).


def tool_kg_list_declared():
    """List all entities declared in this session."""
    results = []
    for eid in sorted(_STATE.declared_entities):
        entity = _STATE.kg.get_entity(eid)
        if entity:
            results.append(
                {
                    "entity_id": eid,
                    "name": entity["name"],
                    "description": entity["description"],
                    "importance": entity["importance"],
                    "last_touched": entity["last_touched"],
                    "edge_count": _STATE.kg.entity_edge_count(eid),
                }
            )
    return {
        "declared_count": len(results),
        "entities": results,
    }


# tool_kg_entity_info removed: use kg_query(entity=..., direction="both").


# ==================== INTENT DECLARATION ====================

# _STATE.active_intent holds the session-level active intent (at most one).
# Defaults to None on ServerState construction — no explicit init needed here.
_INTENT_STATE_DIR = Path(os.path.expanduser("~/.mempalace/hook_state"))


# Intent functions are in intent.py; init() is called after module globals are set.
# Aliases so TOOLS dispatch continues to work:
def tool_declare_intent(*args, **kwargs):
    return intent.tool_declare_intent(*args, **kwargs)


def tool_active_intent(*args, **kwargs):
    return intent.tool_active_intent(*args, **kwargs)


def tool_extend_intent(*args, **kwargs):
    return intent.tool_extend_intent(*args, **kwargs)


def tool_declare_operation(*args, **kwargs):
    return intent.tool_declare_operation(*args, **kwargs)


def tool_resolve_conflicts(actions: list = None, agent: str = None):  # noqa: C901
    """Resolve pending conflicts — contradictions, duplicates, or suggestions.

    Unified conflict resolution for ALL data types: edges, entities, memories.
    Each action specifies what to do with a conflict.

    Args:
        actions: List of {id, action, into?, merged_content?} dicts.
            id: The conflict ID (from the pending conflicts list).
            action: One of:
                "invalidate" — mark existing item as no longer current (sets valid_to)
                "merge" — combine items (must provide into + merged_content)
                "keep" — both items are valid, no conflict
                "skip" — don't add the new item (remove it)
            into: Target entity/memory ID to merge into (required for "merge")
            merged_content: Merged description/content (required for "merge")
        agent: mandatory, declared agent resolving these conflicts.
    """
    sid_err = _require_sid(action="resolve_conflicts")
    if sid_err:
        return sid_err
    agent_err = _require_agent(agent, action="resolve_conflicts")
    if agent_err:
        return agent_err

    # Some MCP transports stringify top-level array parameters. Parse once
    # up front rather than iterating a JSON string character-by-character
    # (which produces thousands of bogus per-character error entries).
    if isinstance(actions, str):
        try:
            actions = json.loads(actions)
        except Exception:
            return {
                "success": False,
                "error": (
                    "`actions` arrived as an unparseable string. Pass a JSON array "
                    "of {id, action, reason, ...} objects."
                ),
            }
    if actions is not None and not isinstance(actions, list):
        return {
            "success": False,
            "error": f"`actions` must be a list, got {type(actions).__name__}.",
        }

    # Disk is source of truth — reload _STATE.pending_conflicts from the active
    # intent state file if memory is empty (MCP restart scenario).
    if not _STATE.pending_conflicts:
        _STATE.pending_conflicts = _load_pending_conflicts_from_disk()

    if not _STATE.pending_conflicts:
        try:
            intent._persist_active_intent()
        except Exception:
            pass
        return {"success": True, "message": "No pending conflicts."}

    if not actions:
        return {
            "success": False,
            "error": "Must provide actions list. Each conflict needs: {id, action}.",
            "pending": _STATE.pending_conflicts,
        }

    # Index pending conflicts by ID — defensively coerce if any entries are
    # JSON strings (some MCP transports serialize nested objects)
    _normalized_conflicts = []
    for c in _STATE.pending_conflicts:
        if isinstance(c, str):
            try:
                c = json.loads(c)
            except Exception:
                continue
        if isinstance(c, dict) and c.get("id"):
            _normalized_conflicts.append(c)
    conflict_map = {c["id"]: c for c in _normalized_conflicts}
    resolved_ids = set()
    results = []

    # Normalize actions too — tolerate string-encoded dicts from some transports
    normalized_actions = []
    for act in actions:
        if isinstance(act, str):
            try:
                act = json.loads(act)
            except Exception:
                results.append(
                    {"id": "?", "status": "error", "reason": f"Unparseable action: {act!r}"}
                )
                continue
        if not isinstance(act, dict):
            results.append(
                {
                    "id": "?",
                    "status": "error",
                    "reason": f"Action must be an object, got {type(act).__name__}",
                }
            )
            continue
        normalized_actions.append(act)

    # ── Validate reason field on all actions + laziness detection ──
    MIN_REASON_LENGTH = 15
    for act in normalized_actions:
        reason = (act.get("reason") or "").strip()
        if len(reason) < MIN_REASON_LENGTH:
            return {
                "success": False,
                "error": (
                    f"Mandatory 'reason' field missing or too short on conflict '{act.get('id', '?')}'. "
                    f"Each conflict resolution requires a reason (minimum {MIN_REASON_LENGTH} characters) "
                    f"explaining WHY you chose this action. This is a real semantic decision — "
                    f"evaluate each conflict individually."
                ),
            }

    # Laziness detection: reject if 3+ actions share identical reason text
    reason_counts: dict = {}
    for act in normalized_actions:
        r = (act.get("reason") or "").strip()
        reason_counts[r] = reason_counts.get(r, 0) + 1
    for r, count in reason_counts.items():
        if count >= 3:
            return {
                "success": False,
                "error": (
                    f"Laziness detected: {count} conflicts share identical reason '{r[:50]}...'. "
                    f"Each conflict is a unique semantic decision — evaluate individually and "
                    f"provide a specific reason for each. Bulk-processing is not allowed."
                ),
            }

    for act in normalized_actions:
        cid = act.get("id", "")
        action = act.get("action", "")

        if cid not in conflict_map:
            results.append({"id": cid, "status": "error", "reason": f"Unknown conflict ID: {cid}"})
            continue

        conflict = conflict_map[cid]
        conflict_type = conflict.get("conflict_type", "unknown")
        existing_id = conflict.get("existing_id", "")
        new_id = conflict.get("new_id", "")

        try:
            if action == "invalidate":
                # Mark existing item as no longer current
                if conflict_type == "edge_contradiction":
                    # Invalidate the existing edge by setting valid_to
                    _STATE.kg.invalidate(
                        conflict["existing_subject"],
                        conflict["existing_predicate"],
                        conflict["existing_object"],
                    )
                elif conflict_type in ("entity_duplicate", "memory_duplicate"):
                    # Mark entity/memory as merged-out
                    try:
                        conn = _STATE.kg._conn()
                        conn.execute(
                            "UPDATE entities SET status='invalidated' WHERE id=?",
                            (existing_id,),
                        )
                        conn.commit()
                    except Exception:
                        pass
                results.append({"id": cid, "status": "invalidated", "target": existing_id})

            elif action == "merge":
                into = act.get("into", "")
                merged_content = act.get("merged_content", "")
                if not into:
                    results.append(
                        {"id": cid, "status": "error", "reason": "merge requires 'into' field"}
                    )
                    continue
                if not merged_content:
                    results.append(
                        {
                            "id": cid,
                            "status": "error",
                            "reason": "merge requires 'merged_content' — read BOTH items in full, then provide combined content",
                        }
                    )
                    continue

                # Determine source (the one NOT being merged into)
                source = new_id if into == existing_id else existing_id

                if conflict_type in ("entity_duplicate", "memory_duplicate"):
                    # Use existing kg_merge_entities for the plumbing
                    merge_result = tool_kg_merge_entities(
                        source=source,
                        target=into,
                        update_description=merged_content,
                        agent=agent,
                    )
                    if merge_result.get("success"):
                        results.append({"id": cid, "status": "merged", "into": into})
                    else:
                        results.append(
                            {
                                "id": cid,
                                "status": "error",
                                "reason": str(merge_result.get("error", "")),
                            }
                        )
                else:
                    results.append(
                        {
                            "id": cid,
                            "status": "error",
                            "reason": f"merge not supported for {conflict_type}",
                        }
                    )

            elif action == "keep":
                # Both items are valid — no action needed
                results.append({"id": cid, "status": "kept"})

            elif action == "skip":
                # Don't add the new item — remove it if already added
                if conflict_type == "edge_contradiction":
                    try:
                        _STATE.kg.invalidate(
                            conflict.get("new_subject", ""),
                            conflict.get("new_predicate", ""),
                            conflict.get("new_object", ""),
                        )
                    except Exception:
                        pass
                results.append({"id": cid, "status": "skipped"})

            else:
                results.append(
                    {"id": cid, "status": "error", "reason": f"Unknown action: {action}"}
                )
                continue

            # Persist the resolution so future audits + feedback loops can
            # learn from the decision instead of throwing the reason away.
            _intent_type = (
                _STATE.active_intent.get("intent_type", "") if _STATE.active_intent else ""
            )
            try:
                _STATE.kg.record_conflict_resolution(
                    conflict_id=cid,
                    conflict_type=conflict_type,
                    action=action,
                    reason=(act.get("reason") or "").strip(),
                    existing_id=existing_id,
                    new_id=new_id,
                    agent=agent,
                    intent_type=_intent_type,
                )
            except Exception:
                pass

            # (retired P3) Edge-contradiction resolutions used to emit a
            # negative signal on the losing edge via edge_traversal_feedback
            # (invalidate → loser = existing triple; skip → loser = new
            # triple). That signal now flows through rated_irrelevant
            # edges on the active context at finalize_intent time.

            resolved_ids.add(cid)
        except Exception as e:
            results.append({"id": cid, "status": "error", "reason": str(e)})

    # Check all conflicts are resolved
    unresolved = set(conflict_map.keys()) - resolved_ids
    errors = [r for r in results if r.get("status") == "error"]
    if unresolved:
        return {
            "success": False,
            "error": f"{len(unresolved)} conflicts not addressed. Provide action for each.",
            "unresolved_ids": sorted(unresolved),
            "errors": errors,
        }

    # Clear pending conflicts and persist state
    _STATE.pending_conflicts = None
    try:
        intent._persist_active_intent()
    except Exception:
        pass
    # Caller supplied the ids/actions/reasons — echoing them back is pure
    # token waste. Return only the count on full success; surface errors
    # individually if any.
    response = {"success": True, "count": len(resolved_ids)}
    if errors:
        response["errors"] = errors
    return response


def tool_finalize_intent(*args, **kwargs):
    return intent.tool_finalize_intent(*args, **kwargs)


def tool_extend_feedback(*args, **kwargs):
    return intent.tool_extend_feedback(*args, **kwargs)


# ==================== AGENT DIARY ====================


def tool_diary_write(
    agent_name: str,
    entry: str,
    slug: str = "",
    topic: str = "general",
    content_type: str = "diary",
    importance: int = None,
):
    """
    Write a diary entry for this agent. Entries are timestamped and
    accumulate over time, scoped by agent name.

    The diary is a HIGH-LEVEL SESSION NARRATIVE — not a detailed log.
    Write in readable prose.

    WHAT TO INCLUDE:
    - Decisions made with the user (approved designs, rejected ideas)
    - Big-picture status and direction
    - Pending items and backlog
    - Cross-intent narrative (how multiple actions connected)

    WHAT NOT TO INCLUDE (already captured by intent results):
    - Individual commits or features shipped
    - Gotchas and learnings (already KG entities via finalize_intent)
    - Tool traces or detailed action logs

    Each entry should be a DELTA from the previous — what changed,
    not a full restatement of everything.

    Args:
        slug: Descriptive identifier for this entry (e.g. 'session12-scoring-design').
              If not provided, falls back to date-topic format.
        topic: Topic tag (optional, default: general)
        content_type: default 'diary'. Override with 'discovery' for
              "today I learned" entries that deserve higher retrieval priority,
              or 'event' for plain activity logs.
        importance: 1-5. Defaults to unset (treated as 3 by L1). Use 4 for
                    entries with learned lessons, 5 only for agent-wide
                    critical notes.
    """
    sid_err = _require_sid(action="diary_write")
    if sid_err:
        return sid_err
    try:
        agent_name = sanitize_name(agent_name, "agent_name")
        entry = sanitize_content(entry)
        content_type = _validate_content_type(content_type)
        importance = _validate_importance(importance)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    from .knowledge_graph import normalize_entity_name as _norm_eid

    agent_slug = _norm_eid(agent_name)
    col = _get_collection(create=True)
    if not col:
        return _no_palace()

    now = datetime.now()
    if slug and slug.strip():
        diary_slug = _slugify(slug)
    else:
        diary_slug = _slugify(f"{now.strftime('%Y%m%d-%H%M%S')}-{topic}")
    entry_id = f"diary_{agent_slug}_{diary_slug}"

    _wal_log(
        "diary_write",
        {
            "agent_name": agent_name,
            "topic": topic,
            "entry_id": entry_id,
            "entry_preview": entry[:200],
            "content_type": content_type,
            "importance": importance,
        },
    )

    try:
        meta = {
            "content_type": content_type or "diary",
            "topic": topic,
            "type": "diary_entry",
            "added_by": agent_name,
            "filed_at": now.isoformat(),
            "date_added": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
        }
        if importance is not None:
            meta["importance"] = importance
        # Mirror the _add_memory_internal diagnostic pattern: if chroma
        # throws 'TextInputSequence must be str in add', re-raise with
        # explicit memory_id/types/lens/meta so the next live occurrence
        # is actionable instead of a bare tokenizer message.
        if not isinstance(entry, str):
            return {
                "success": False,
                "error": (f"internal: diary entry must be str (got {type(entry).__name__})."),
            }
        try:
            col.add(
                ids=[entry_id],
                documents=[entry],
                metadatas=[meta],
            )
        except Exception as _add_err:
            _meta_types = {k: type(v).__name__ for k, v in meta.items()}
            _msg = (
                f"col.add failed on diary id={entry_id!r}: "
                f"{type(_add_err).__name__}: {_add_err}. "
                f"types[ids]=list[{type(entry_id).__name__}], "
                f"types[documents]=list[{type(entry).__name__}], "
                f"len(entry)={len(entry)}, meta_value_types={_meta_types}"
            )
            logger.error(_msg)
            raise RuntimeError(_msg) from _add_err
        logger.info(f"Diary entry: {entry_id} content_type={content_type} imp={importance}")

        # Update the stop hook save counter — proves diary was actually written.
        # The stop hook writes a _pending_save marker but does NOT update
        # last_save itself. This prevents the dodge where agents ignore the
        # save prompt and the counter resets anyway. Only diary_write updates it.
        try:
            from .hooks_cli import STATE_DIR

            STATE_DIR.mkdir(parents=True, exist_ok=True)
            # sid is guaranteed non-empty by _require_sid at entry.
            sid = _STATE.session_id
            pending_file = STATE_DIR / f"{sid}_pending_save"
            if pending_file.is_file():
                exchange_count = pending_file.read_text(encoding="utf-8").strip()
                last_save_file = STATE_DIR / f"{sid}_last_save"
                last_save_file.write_text(exchange_count, encoding="utf-8")
                pending_file.unlink()  # Clear the marker
        except Exception:
            pass  # Non-fatal — save counter is best-effort

        return {
            "success": True,
            "entry_id": entry_id,
            "topic": topic,
            "content_type": content_type,
            "importance": importance,
            "timestamp": now.isoformat(),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_diary_read(agent_name: str, last_n: int = 10):
    """
    Read an agent's recent diary entries. Returns the last N entries
    in chronological order — the agent's personal journal.
    """
    col = _get_collection()
    if not col:
        return _no_palace()

    try:
        results = col.get(
            where={"$and": [{"added_by": agent_name}, {"type": "diary_entry"}]},
            include=["documents", "metadatas"],
            limit=10000,
        )

        if not results["ids"]:
            return {"entries": [], "message": "No diary entries yet."}

        # Combine and sort by timestamp
        entries = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            entries.append(
                {
                    "date": meta.get("date", ""),
                    "timestamp": meta.get("filed_at", ""),
                    "topic": meta.get("topic", ""),
                    "content": doc,
                }
            )

        entries.sort(key=lambda x: x["timestamp"], reverse=True)
        entries = entries[:last_n]

        return {
            "entries": entries,
            "total": len(results["ids"]),
            "showing": len(entries),
        }
    except Exception as e:
        return {"error": str(e)}


# ==================== MCP PROTOCOL ====================

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
                    "description": "Date filter — only facts valid at this date (YYYY-MM-DD, optional)",
                },
                "direction": {
                    "type": "string",
                    "description": "outgoing (entity→?), incoming (?→entity), or both (default: both)",
                },
                "include_context_edges": {
                    "type": "boolean",
                    "description": "Include retrieval-bookkeeping edges (rated_useful, rated_irrelevant, surfaced) in the facts list. Default false — they are filtered out because they drown domain edges in per-context noise. Set true for retrieval audits. When filtered, hidden_context_edges (or total_hidden_context_edges in batch mode) reports how many were hidden.",
                },
            },
            "required": ["entity"],
        },
        "handler": tool_kg_query,
    },
    "mempalace_kg_search": {
        "description": (
            "Unified search — records (prose) + entities (KG nodes) in one "
            "call (Context-based). Speaks the unified Context object: "
            "queries drive Channel A multi-view cosine, keywords drive Channel C "
            "(caller-provided exact terms — no auto-extraction), entities seed "
            "Channel B graph BFS. Cross-collection Reciprocal Rank Fusion across "
            "all channels. Each result carries source='memory'|'entity' with "
            "type-specific fields (memories: text; entities: name/kind/"
            "description/edges). Unlike kg_query (exact entity ID), this "
            "fuzzy-matches across your whole memory palace."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "context": {
                    "type": "object",
                    "description": (
                        "MANDATORY Context fingerprint for the query.\n"
                        "  queries:  list[str] (2-5)  perspectives — each becomes a cosine view.\n"
                        "  keywords: list[str] (2-5)  caller-provided exact terms (no auto-extract).\n"
                        "  entities: list[str] (0+)   graph BFS seeds (defaults to top cosine hits).\n"
                        'Example: context={"queries": ["deployment process", "release pipeline"], '
                        '"keywords": ["deploy", "release", "rollout"]}'
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
                        "entities": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["queries", "keywords"],
                },
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
                "context": {
                    "type": "object",
                    "description": (
                        "MANDATORY Context fingerprint for the edge. "
                        '{"queries": list[str] (2-5 perspectives on why this edge), '
                        '"keywords": list[str] (2-5 caller-provided terms), '
                        '"entities": list[str] (0+ related entity ids)}'
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
                        "entities": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["queries", "keywords"],
                },
                "agent": {
                    "type": "string",
                    "description": (
                        "MANDATORY — your declared agent entity name (is_a agent). "
                        "Every write operation is attributed to a declared agent; "
                        "undeclared agents are rejected up front with a declaration recipe."
                    ),
                },
                "valid_from": {
                    "type": "string",
                    "description": "When this became true (YYYY-MM-DD, optional)",
                },
                "statement": {
                    "type": "string",
                    "description": (
                        "Natural-language verbalization of the triple (e.g. "
                        '"Max is a child of Alice"). REQUIRED for every '
                        "predicate OUTSIDE the skip list (is_a, described_by, "
                        "executed_by, targeted, has_value, session_note_for, "
                        "derived_from, mentioned_in, found_useful, "
                        "found_irrelevant, evidenced_by). For skip-list "
                        "predicates the statement may be omitted because "
                        "those edges are never embedded anyway. Auto-generation "
                        "was retired 2026-04-19 because naive fallbacks "
                        "produced retrieval-poisoning text."
                    ),
                },
            },
            "required": ["subject", "predicate", "object", "context", "agent"],
        },
        "handler": tool_kg_add,
    },
    "mempalace_kg_add_batch": {
        "description": (
            "Add multiple KG edges in one call (Context mandatory). Pass a "
            "single top-level `context` as the shared default for every edge in the "
            "batch — most batches add edges that all reflect the same agent decision. "
            "An edge can override with its own `context` if needed. Validates "
            "independently — partial success OK."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "predicate": {"type": "string"},
                            "object": {"type": "string"},
                            "context": {
                                "type": "object",
                                "description": "Per-edge Context override (optional if top-level context provided).",
                            },
                        },
                        "required": ["subject", "predicate", "object"],
                    },
                    "description": "List of edges to add.",
                },
                "context": {
                    "type": "object",
                    "description": (
                        "Shared Context for every edge in the batch. Required "
                        "unless every edge carries its own `context`."
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
                        "entities": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["queries", "keywords"],
                },
                "agent": {
                    "type": "string",
                    "description": ("MANDATORY — declared agent attributing the whole batch."),
                },
            },
            "required": ["edges", "agent"],
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
                    "description": "MANDATORY — declared agent invalidating this fact.",
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
                    "description": "Entity to get timeline for (optional — omit for full timeline)",
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
            "Kinds — STRICT enum, exactly five values:\n"
            "  'entity'    — concrete thing. DEFAULT for most new nodes.\n"
            "  'class'     — category definition (other entities is_a this).\n"
            "  'predicate' — relationship type for kg_add edges.\n"
            "  'literal'   — raw value.\n"
            "  'record'    — prose record. Requires slug + `content` "
            "(verbatim text) + `added_by`. `name` is auto-computed. "
            "Use `entity`+`predicate` to link the record to another entity.\n"
            "If the value you want isn't in the enum, it's a domain "
            "class, not a kind — declare the node with kind='entity' and "
            "add an is_a edge to the class node. The enum of kinds is "
            "fixed; the set of classes is open and grows over time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Entity name (will be normalized). REQUIRED for kind=entity/class/predicate/literal. OMIT for kind='record' — the record id is computed from added_by + slug.",
                },
                "context": {
                    "type": "object",
                    "description": (
                        "MANDATORY Context fingerprint. Replaces the legacy single-string `description`.\n"
                        "  queries:  list[str] (2-5)  perspectives on what this entity is.\n"
                        "             For non-memory kinds, queries[0] becomes the canonical description.\n"
                        "  keywords: list[str] (2-5)  exact terms — caller-provided, NEVER auto-extracted.\n"
                        "             Stored in the keyword index for fast exact-match retrieval.\n"
                        "  entities: list[str] (0+)   related/seed entity ids (optional graph anchors).\n"
                        'Example: context={"queries": ["DSpot platform server", "paperclip backend on :3100"], '
                        '"keywords": ["dspot", "paperclip", "server", "port-3100"], '
                        '"entities": ["DSpotInfra"]}'
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
                        "entities": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["queries", "keywords"],
                },
                "kind": {
                    "type": "string",
                    "description": "Ontological role — STRICT enum, exactly five values: 'entity' (concrete thing — DEFAULT), 'class' (category/type definition that other entities is_a), 'predicate' (relationship type for kg_add edges), 'literal' (raw value), 'record' (prose memory; requires slug + content + added_by). If the value you want isn't in the enum, it's a domain class, not a kind — pass kind='entity' and add an is_a edge to the class node (kg_add(subject=name, predicate='is_a', object=<that_class>)). The set of classes is open and grows over time; the set of kinds is fixed.",
                    "enum": ["entity", "predicate", "class", "literal", "record"],
                },
                "content": {
                    "type": "string",
                    "description": "Verbatim text — REQUIRED for kind='record' (the actual memory body). Ignored for other kinds (queries[0] is the canonical description).",
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
                "summary": {
                    "type": "string",
                    "description": (
                        "≤280-char distilled one-liner. REQUIRED when "
                        "kind='record' — every record carries a summary "
                        "independent of content length (Anthropic "
                        "Contextual Retrieval 2024). For long content the "
                        "summary distills WHAT/WHY; for short content the "
                        "summary REPHRASES the same fact from a different "
                        "angle so summary+content produce two distinct "
                        "cosine views. Ignored for non-record kinds."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": (
                        "Optional description override for non-record kinds "
                        "(entity/class/predicate/literal). Defaults to "
                        "queries[0] from the Context when omitted."
                    ),
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
            "  - description: re-syncs to entity ChromaDB and runs collision distance check.\n"
            "  - properties: shallow-merged into existing properties. For predicates, "
            'use {"constraints": {...}} to update predicate constraints (validated).\n'
            "  - importance: 1-5.\n\n"
            "FOR RECORDS (kind='record'):\n"
            "  - content_type: in-place classification change (no re-embedding).\n"
            "  - importance: in-place importance change.\n"
            "  - description: NOT supported — use kg_delete_entity + kg_declare_entity "
            "to replace record content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "Entity ID or record ID (record_/diary_ prefix routes to record collection).",
                },
                "description": {
                    "type": "string",
                    "description": "New description (entities only). Triggers collision distance check.",
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
                "context": {
                    "type": "object",
                    "description": (
                        "OPTIONAL Context to re-anchor the entity when description or "
                        "properties semantically change. Same shape as in "
                        "kg_declare_entity. When provided, persists a new creation_context_id "
                        "so future MaxSim feedback attaches to the updated meaning."
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
                        "entities": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["queries", "keywords"],
                },
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
                    "description": "MANDATORY — declared agent attributing this update.",
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
                "update_description": {
                    "type": "string",
                    "description": "Optional new description for the merged entity",
                },
                "agent": {
                    "type": "string",
                    "description": "MANDATORY — declared agent attributing this merge.",
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
            "One active intent at a time — new intent expires the previous. "
            "mempalace_* tools are always allowed regardless of intent.\n\n"
            "SLOT RULES — most intent types require these slots:\n"
            '  paths:    (raw) directory patterns for Read/Grep/Glob scoping. E.g. ["D:/Flowsev/repo/**"]\n'
            '  commands: (raw) command patterns for Bash scoping. E.g. ["pytest", "git add"]\n'
            "  files:    file paths for Edit/Write scoping. Auto-declares existing files.\n"
            "  target:   entity names for context injection. Requires pre-declared entities.\n\n"
            "EXCEPTION: 'research' type needs NO paths — it has unrestricted Read/Grep/Glob/WebFetch/WebSearch.\n"
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
                        "Use the MOST SPECIFIC type available — specific types carry domain rules. "
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
                        'Example for research: {"subject": ["some_topic"]} — NO paths needed, broad reads allowed. '
                        "File slots auto-declare existing files. Command slots (raw) accept strings directly. "
                        "Other slots require pre-declared entities."
                    ),
                },
                "context": {
                    "type": "object",
                    "description": (
                        "MANDATORY Context fingerprint for this intent (replaces "
                        "the legacy `descriptions` parameter).\n"
                        "  queries:  list[str] (2-5)  perspectives on what you're about to do.\n"
                        "             Each becomes a separate cosine view for retrieval.\n"
                        "             queries[0] is the canonical description for auto-narrowing.\n"
                        "  keywords: list[str] (2-5)  caller-provided exact terms — drives the\n"
                        "             keyword channel (no auto-extraction).\n"
                        "  entities: list[str] (0+)   related/seed entities for graph BFS\n"
                        "             (defaults to slot entities when omitted).\n"
                        'Example: context={"queries": ["Editing auth rate limiter", '
                        '"Security hardening against brute force", "Adding tests for login endpoint"], '
                        '"keywords": ["auth", "rate-limit", "brute-force", "login"], '
                        '"entities": ["LoginService"]}'
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
                        "entities": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["queries", "keywords"],
                },
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
                        "Keep budgets tight — estimate minimum needed. "
                        "When exhausted, use mempalace_extend_intent to add more."
                    ),
                },
            },
            "required": ["intent_type", "slots", "context", "agent", "budget"],
        },
        "handler": tool_declare_intent,
    },
    "mempalace_active_intent": {
        "description": "Return the current active intent — type, slots, permissions, budget remaining.",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_active_intent,
    },
    "mempalace_resolve_conflicts": {
        "description": (
            "Resolve pending conflicts — contradictions, duplicates, or merge candidates. "
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
                                    "MANDATORY — why you chose this action (minimum 15 characters). "
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
                    "description": "MANDATORY — declared agent resolving these conflicts.",
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
            "hook used to construct — you specify the cue directly so "
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
            "field — `{good_precedents, avoid_patterns}` — drawn from the "
            "performed_well / performed_poorly edges in the current "
            "operation-context's MaxSim neighbourhood. Distinct from "
            "`memories` (memory-retrieval relevance); this is tool+args "
            "correctness. Rate your ops at finalize via `operation_ratings` "
            "to feed this channel."
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
                "context": {
                    "type": "object",
                    "description": (
                        "Unified Context fingerprint — same shape as "
                        "declare_intent / kg_search / kg_add. Mandatory "
                        "object with queries (2-5), keywords (2-5), "
                        "entities (1-10)."
                    ),
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 5,
                            "description": (
                                "2-5 natural-language perspectives on WHAT you "
                                "are about to do and WHY. Specific, varied, "
                                "anchored on the task, not the tool. Bad: "
                                "['run pytest']. Good: ['verify the mandatory-"
                                "memory_feedback finalize contract', 'check "
                                "declare_intent slot validation']."
                            ),
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 5,
                            "description": (
                                "2-5 exact terms — domain vocabulary that will "
                                "hit the keyword channel. Bad: ['test', 'run']. "
                                "Good: ['memory_feedback', 'finalize_intent', "
                                "'declare_intent']."
                            ),
                        },
                        "entities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 10,
                            "description": (
                                "1-10 entity ids this operation touches — files, "
                                "services, agents, concepts the task handles. The "
                                "link-author pipeline accumulates Adamic-Adar "
                                "evidence from (entity, context) co-occurrence, so "
                                "this is the seed for discovering relationships "
                                "the agent should later author. MAY overlap with "
                                "declare_intent's slot values."
                            ),
                        },
                    },
                    "required": ["queries", "keywords", "entities"],
                },
                "agent": {
                    "type": "string",
                    "description": "Your agent name.",
                },
            },
            "required": ["tool", "context"],
        },
        "handler": tool_declare_operation,
    },
    "mempalace_finalize_intent": {
        "description": (
            "Finalize the active intent — capture what happened as structured memory. "
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
                        "MANDATORY — the full narrative body for the result memory. "
                        "Free length, stored verbatim. ALWAYS required on every record; "
                        "no auto-derivation. For long content, a distillation of WHAT/WHY; "
                        "for short content, a REPHRASE from a different angle so the "
                        "summary+content pair yields two distinct cosine views of the "
                        "same semantic (Anthropic Contextual Retrieval 2024)."
                    ),
                },
                "summary": {
                    "type": "string",
                    "description": (
                        "MANDATORY — ≤280-char distilled one-liner of the outcome. "
                        "Shown in injections and prepended to content for embedding. "
                        "For short content, REPHRASE from a different angle than content."
                    ),
                },
                "agent": {
                    "type": "string",
                    "description": "Your agent entity name (e.g. 'ga_agent', 'technical_lead_agent').",
                },
                "memory_feedback": {
                    "type": "array",
                    "description": (
                        "MANDATORY — list of per-context feedback groups: "
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
                                    "Do NOT confuse with intent_id — the Context is a distinct KG entity."
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
                                                "1-5, signed scale — what did you actually do with this memory when it surfaced? "
                                                "1=misleading (wasted attention / pointed me wrong; teach the context NOT to surface this again). "
                                                "2=noise (skimmed and dropped; same topic area, nothing to do with this specific task). "
                                                "3=related context (DEFAULT when unsure — accurate and topical, didn't change what I did). "
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
                                                "MANDATORY — why this memory was or wasn't relevant to THIS intent "
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
                    "items": {"type": "string"},
                    "description": "Gotcha descriptions discovered during execution. Each becomes a KG entity.",
                },
                "learnings": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Lesson descriptions worth remembering. Each becomes a memory.",
                },
                "promote_gotchas_to_type": {
                    "type": "boolean",
                    "description": "Also link gotchas to the intent TYPE (not just this execution). Use for general gotchas.",
                },
                "operation_ratings": {
                    "type": "array",
                    "description": (
                        "MANDATORY — your rating of tool-invocation quality. "
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
                        "3=ok — neutral signal, no promotion, 4=good, "
                        "5=load-bearing), reason, args_summary, "
                        "better_alternative (S2)}. "
                        "Quality >=4 writes performed_well; <=2 writes "
                        "performed_poorly; =3 is neutral. One rating per "
                        "unique (tool, ctx_id) pair covers any number of "
                        "repeated calls under that pair. See "
                        "`missing_operations` map in a failed finalize for "
                        "the required pairs. Distinct from rated_useful / "
                        "rated_irrelevant. "
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
                            "args_summary": {"type": "string"},
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
            "operation_ratings into the existing execution entity — same shape as "
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
                        "List of per-context feedback groups — same shape as "
                        "finalize_intent.memory_feedback: "
                        "[{context_id, feedback: [{id, relevance, reason, relevant?}, ...]}, ...]. "
                        "Last-write-wins per (memory_id, context_id) — supplying a "
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
                        "List of operation rating entries — same shape as "
                        "finalize_intent.operation_ratings: "
                        "[{tool, context_id, quality, reason, args_summary?}, ...]. "
                        "Last-write-wins per (tool, context_id)."
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
                            "args_summary": {"type": "string"},
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
                "entity_id": {
                    "type": "string",
                    "description": "ID of the entity or record to delete.",
                },
                "agent": {
                    "type": "string",
                    "description": "MANDATORY — declared agent attributing this deletion.",
                },
            },
            "required": ["entity_id", "agent"],
        },
        "handler": tool_kg_delete_entity,
    },
    "mempalace_wake_up": {
        "description": (
            "Return L0 (identity) + L1 (importance-ranked essential story) wake-up "
            "text (~600-900 tokens total). Call this ONCE at session start to load "
            "project/agent boot context. Also returns the protocol, declared entities/"
            "predicates/intent types — everything you need to start. L1 is ranked "
            "with importance-weighted time decay — critical facts always surface first, "
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
                    "description": "Your name — each agent gets their own diary",
                },
                "entry": {
                    "type": "string",
                    "description": "Your diary entry — readable prose.",
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
        "description": "Read your recent diary entries. See what past versions of yourself recorded — your journal across sessions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Your name — each agent gets their own diary",
                },
                "last_n": {
                    "type": "integer",
                    "description": "Number of recent entries to read (default: 10)",
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
        # "default" / "unknown" — that would silently merge every
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

    P3 polish — migration 015 retired the SQLite companion tables
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
        # Most commonly: collection doesn't exist. Fresh palace — nothing
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
    # N3 hyphen-id migration — RETIRED FROM AUTO-STARTUP (2026-04-25).
    #
    # This was a one-shot legacy migration that renamed hyphenated IDs
    # to the underscored canonical form. New palaces never produce
    # hyphenated IDs — `normalize_entity_name` strips them at write
    # time, so any palace created post-N3 has nothing to migrate.
    # Already-migrated palaces re-walked thousands of Chroma rows on
    # every server boot, doing zero useful work, while exposing the
    # boot path to a known Chroma v0.6.0 internal bug ("list assignment
    # index out of range" inside `col.get(include=embeddings)`) that
    # could leave the HNSW vector index half-written and trigger a
    # C-level access violation on subsequent queries.
    #
    # The migration code (mempalace.hyphen_id_migration.run_migration)
    # is preserved verbatim — anyone with genuinely legacy hyphenated
    # data can run it manually via:
    #   python -c "from mempalace import mcp_server; \
    #              mcp_server._run_hyphen_id_migration_once()"
    # — but it is NOT invoked here on every boot.
    #
    # See: record_ga_agent_chroma_hnsw_segfault_root_cause_2026_04_25
    # for the corruption mechanism this removal closes off.
    # M1 collection merge — absorb legacy mempalace_entities rows into
    # the unified mempalace_records collection and drop the legacy one.
    try:
        _migrate_entities_collection_into_records()
    except Exception as e:
        logger.warning(f"M1 startup collection-merge failed: {e}")
    # P3 polish one-shot — drop the retired mempalace_feedback_contexts
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
