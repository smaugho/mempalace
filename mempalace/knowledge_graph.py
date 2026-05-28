"""
knowledge_graph.py -- Temporal Entity-Relationship Graph for MemPalace
=====================================================================

Real knowledge graph with:
  - Entity nodes (people, projects, tools, concepts)
  - Typed relationship edges (daughter_of, does, loves, works_on, etc.)
  - Temporal validity (valid_from → valid_to -- knows WHEN facts are true)
  - Closet references (links back to the verbatim memory)

Storage: SQLite (local, no dependencies, no subscriptions)
Query: entity-first traversal with time filtering

This is what competes with Zep's temporal knowledge graph.
Zep uses Neo4j in the cloud ($25/mo+). We use SQLite locally (free).

Usage:
    from mempalace.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph()
    kg.add_triple("Max", "child_of", "Alice", valid_from="2015-04-01")
    kg.add_triple("Max", "does", "swimming", valid_from="2025-01-01")
    kg.add_triple("Max", "loves", "chess", valid_from="2025-10-01")

    # Query: everything about Max
    kg.query_entity("Max")

    # Query: what was true about Max in January 2026?
    kg.query_entity("Max", as_of="2026-01-15")

    # Query: who is connected to Alice?
    kg.query_entity("Alice", direction="both")

    # Invalidate: Max's sports injury resolved
    kg.invalidate("Max", "has_issue", "sports_injury", ended="2026-02-15")
"""

from __future__ import annotations


import atexit
import hashlib
import json
import os
import re
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

# Track all KG instances for cleanup on exit
_active_instances = []


def _cleanup_all():
    """Close all KG connections on process exit to release WAL locks."""
    for kg in _active_instances:
        try:
            kg.close()
        except Exception:
            pass


atexit.register(_cleanup_all)


# BF1: legacy default location used when KG was kept outside the palace dir.
# Pre-2026-04-18 default. The current canonical location is
# {config.palace_path}/knowledge_graph.sqlite3 (computed in __init__ when
# db_path is None). LEGACY_KG_PATH is checked at first init and migrated in
# place when the canonical file is missing or empty, so existing installs
# don't lose data on the path move.
LEGACY_KG_PATH = os.path.expanduser("~/.mempalace/knowledge_graph.sqlite3")
DEFAULT_KG_PATH = LEGACY_KG_PATH  # kept as alias for any external imports


def _resolve_default_kg_path() -> str:
    """Return the canonical KG path: inside the resolved palace directory.

    Falls back to LEGACY_KG_PATH if MempalaceConfig can't load (test setups,
    bootstrap edge cases) so the existing module-import behaviour stays safe.
    """
    try:
        from .config import MempalaceConfig

        return os.path.join(MempalaceConfig().palace_path, "knowledge_graph.sqlite3")
    except Exception:
        return LEGACY_KG_PATH


def _maybe_migrate_legacy_kg(canonical_path: str) -> None:
    """Move LEGACY_KG_PATH -> canonical_path on first init when canonical is
    missing or zero-byte and legacy has data. Idempotent and safe to call on
    every KG construction; no-op when there's nothing to migrate.
    """
    try:
        if canonical_path == LEGACY_KG_PATH:
            return  # nothing to migrate when both paths coincide
        if not os.path.exists(LEGACY_KG_PATH):
            return
        try:
            legacy_size = os.path.getsize(LEGACY_KG_PATH)
        except OSError:
            legacy_size = 0
        if legacy_size == 0:
            return
        canonical_size = os.path.getsize(canonical_path) if os.path.exists(canonical_path) else 0
        if canonical_size > 0:
            return  # canonical already has data; don't clobber it
        Path(canonical_path).parent.mkdir(parents=True, exist_ok=True)
        # Move legacy file plus any -wal/-shm sidecar files SQLite may have left.
        for suffix in ("", "-wal", "-shm"):
            src = LEGACY_KG_PATH + suffix
            dst = canonical_path + suffix
            if os.path.exists(src):
                if os.path.exists(dst):
                    os.remove(dst)  # only happens for empty/orphan dst
                os.replace(src, dst)
    except Exception:
        pass  # migration is best-effort; bad migration shouldn't crash startup


# ── Triple verbalization (research-style "triple-to-text" for retrieval) ──
# Each triple gets a natural-language sentence stored on the row + embedded
# into the mempalace_triples Chroma collection. That makes triples
# first-class search citizens alongside prose memories and entities; without
# this, a query like "who lives in Warsaw" misses an `(adrian, lives_in,
# warsaw)` triple unless prose memory text happens to match the words.

TRIPLE_COLLECTION_NAME = "mempalace_triples"

# Predicates that are pure schema glue (type membership, attribution,
# narrative back-references). Verbalizing them ("research is a inspect",
# "memory_X described_by memory_Y") just floods retrieval with low-signal
# generic statements that drown semantic content. Skip them at index time
# -- structural facts are still in the SQL triples table and walkable via
# BFS, just not embedded for similarity search.
_TRIPLE_SKIP_PREDICATES = {
    "is_a",
    "described_by",
    "evidenced_by",
    "executed_by",
    "targeted",
    # NOTE on `has_value` (REMOVED from skip list 2026-04-25):
    # Adrian's audit identified has_value as a content-bearing predicate,
    # not a structural one -- `(server has_value port=8080)` carries the
    # actual value 8080 that future agents may need to retrieve via
    # cosine search. Skipping its statement means the value never
    # gets embedded and is unreachable via semantic search. Callers
    # writing has_value triples MUST now provide a statement that
    # verbalises the value pair (see add_triple TripleStatementRequired).
    "session_note_for",
    "derived_from",
    "mentioned_in",
    # Context-as-entity predicates. All pure graph topology -- the
    # context system exposes them via kg_query, never via semantic
    # search over synthesised statements (which would add noise).
    "created_under",  # provenance from node to context
    "similar_to",  # context-to-context neighbourhood
    "surfaced",  # retrieval-event edge (context → surfaced entity)
    "rated_useful",  # positive feedback edge
    "rated_irrelevant",  # negative feedback edge
    # S3b template-collapse edge (record → operation). Pure graph
    # topology -- the template record itself is embedded; the edge is
    # structural and walking it is by KG traversal, not by cosine
    # search over a synthesised statement. Adding here also fixes the
    # silent-drop bug that hid TripleStatementRequired in the
    # gardener's _synthesize_operation_template_shim 2026-04-25.
    "templatizes",
    # Operation-tier rating + parent-child edges (S1/S2 of op-memory,
    # 2026-04-25 audit). Adrian flagged that any predicate ignored by
    # retrieval should also be in this skip-list so add_triple stops
    # demanding statements that wouldn't ever be embedded anyway.
    # These edges are pure graph topology:
    #   - executed_op: intent_exec parent → operation child
    #   - performed_well / performed_poorly: context → operation
    #     rating bookkeeping (cosine-walked via similar_to neighbours
    #     in retrieve_past_operations, never search-by-statement)
    #   - superseded_by: operation → operation correction edge,
    #     walked when the parent op surfaces in avoid_patterns
    # Without this skip-list inclusion, finalize_intent's silent
    # try/except around add_triple at lines 4420-4424 would hide
    # TripleStatementRequired and drop the edges -- the same class of
    # silent-drop bug that bit S3b's templatizes.
    "executed_op",
    "performed_well",
    "performed_poorly",
    "superseded_by",
    # State-protocol v1 JTMS justification edge (Adrian Option B
    # 2026-05-03; Doyle 1979). state_changed_by links a state-revision
    # row id to the op-context that caused the revision -- pure graph
    # topology like executed_op (parent-cause edge), walked by the
    # retraction sweep when an op is invalidated. Without this skip-list
    # inclusion record_state_revision's try/except around add_triple
    # would silently swallow TripleStatementRequired and the JTMS edge
    # would never land. Discovered by manual test step 5 on 2026-05-03.
    "state_changed_by",
    # Channel-B structural edge from a context entity to each anchor
    # entity the caller declared in context.entities. Pure graph
    # topology -- the relation is "this context references these
    # entities so BFS can find this context when walking out from
    # them," nothing semantic to embed. Pre-2026-05-02 fix the entity
    # *content* was auto-appended into the context's _views list which
    # saturated max-of-max similarity at 1.0 across contexts sharing
    # any slot entity (record_ga_agent_channel_violation_saturation).
    # The fix replaces that pollution with this skip-list edge so
    # entities are reachable via Channel B graph traversal without
    # leaking into Channel A cosine views.
    "anchored_by",
}


# NOTE: an older ``_verbalize_triple`` helper used to exist here as a naive
# "replace underscores with spaces" fallback for callers that omitted
# ``statement``. It was removed 2026-04-19 -- see the TripleStatementRequired
# policy below. Autogenerated statements produced low-signal text like
# "record ga agent a relates to record ga agent b" that drowned real
# retrievals. Now callers either supply a real sentence or the non-skip
# edge is rejected at write time.


class TripleStatementRequired(ValueError):
    """Raised by ``add_triple`` when a non-structural edge is created
    without a caller-provided ``statement``.

    Skip-list predicates (``is_a``, ``described_by``, ``executed_by``, …)
    remain statement-optional because ``_index_triple_statement`` never
    embeds them anyway -- they're schema glue, walkable via BFS, not
    searched by similarity. For every other predicate, the caller MUST
    supply a natural-language verbalization or the triple is refused.
    """


# ═══════════════════════════════════════════════════════════════════
# Structured-summary validation (2026-04-25 design lock with Adrian)
#
# Records, entities, predicate statements, and (eventually) contexts
# all carry a `summary` field. The retrieval pipeline embeds the
# summary as one cosine view alongside the prose body, so the WHAT
# the entity IS plus the WHY it matters land as a focused token-budget
# anchor in the embedding space.
#
# Without structure, summaries drift to single-noun stubs ("Adrian",
# "the project", "File: x.py") that contribute zero retrieval signal
# beyond the entity name itself -- and the entity name is already
# searchable by exact lookup. Adrian's audit (2026-04-25) confirmed
# the gap: most existing summaries are name-restating placeholders or
# auto-stubs from the file-mint path.
#
# Shape: {what, why, scope?} -- strict dict-only on writes (Adrian's
# design lock 2026-04-25). Stored prose strings from earlier writes
# remain readable (serialize_summary_for_embedding passes them
# through) but new writes must pass the dict; the legacy-string
# write path was retired because it let low-effort stubs through
# under the back-compat door.
#
# Why dict-storage / prose-embedding hybrid:
#   - Validation works on structured fields (each non-empty, length
#     bounds enforced) so silent-stub regressions get caught at write
#     time loudly, not silently at retrieval.
#   - Embedding text is concatenated prose without label tokens
#     ("InjectionGate -- runtime gate that filters retrieved memories;
#     one instance per palace process"). Literature converges that
#     embedding-quality is best on prose, not on labelled fields.
#
# Why "scope" (not "condition" / "when"):
#   - Generalises across record kinds: temporal qualifier for events,
#     domain qualifier for rules ("Windows-only"), role qualifier for
#     services ("one instance per palace process"). "Condition" reads
#     prescriptive; "when" too time-specific.
#
# Literature references baked into validate_summary's docstring:
#   - Anthropic Contextual Retrieval (2024): the prepended what/why/
#     role context block lifted retrieval F1 by 35-50% across five
#     embedding models. The big gain was the role/why piece, not
#     topic alone.
#   - Khattab & Zaharia 2020 (ColBERT, late-interaction): multi-view
#     storage benefits from focused per-view content; one summary view
#     anchored on what+why complements the body view.
#   - Wadden et al. 2020 (SciFact): structured claim+evidence
#     summaries beat freeform on fact-grounded retrieval by 10-20%
#     nDCG@10.
#   - Packer et al. 2023 (MemGPT) / Liu et al. 2024 (Letta): agent-
#     memory papers showing field-shaped summaries retrieve better
#     than freeform -- directly relevant to mempalace's design.
# ═══════════════════════════════════════════════════════════════════


class SummaryStructureRequired(ValueError):
    """Raised when a summary fails the WHAT+WHY+SCOPE? structural check.

    Carries a context-specific message naming the failing field and
    the call site so callers know exactly which write rejected.
    """


# Field-level minima for the structured-summary contract.
#
# Adrian's design lock 2026-04-25 (post-376347a refactor): summary is
# a dict ``{what, why, scope?}``, validation reduces to "fields present,
# non-empty, length-bounded". The previous regex-on-prose path and the
# legacy-string tolerance were both retired -- they let low-effort stubs
# through under the back-compat door. New writes pass dicts; existing
# stored prose remains readable but is not re-validated.
_SUMMARY_MAX_LEN = 280
_SUMMARY_WHAT_MIN = 5  # noun-phrase floor
_SUMMARY_WHY_MIN = 15  # purpose-clause floor
_SUMMARY_SCOPE_MAX = 100  # scope is optional and short


def serialize_summary_for_embedding(summary):
    """Project a summary dict into the prose form used as one of the
    embedding views.

    Storage is the dict ``{what, why, scope?}`` for validation +
    field-level audit; embedding text concatenates ``what -- why``
    plus ``; scope`` when present. Embeddings work measurably better
    on prose than on labelled fields (Anthropic Contextual Retrieval
    2024; replicated in BEIR / MS MARCO ablations), which is why
    serialization strips the keys before handing to chroma.

    Already-persisted prose strings pass through unchanged so reads
    of legacy data don't break. Validation rejects strings on NEW
    writes (see ``validate_summary``); this projection only runs on
    already-validated dicts or pre-existing prose.
    """
    if isinstance(summary, str):
        return summary
    if isinstance(summary, dict):
        what = str(summary.get("what", "")).strip()
        why = str(summary.get("why", "")).strip()
        scope = str(summary.get("scope", "")).strip()
        parts = [p for p in (what, why) if p]
        text = " -- ".join(parts) if parts else ""
        if scope:
            text = f"{text}; {scope}" if text else scope
        return text
    return str(summary)


def validate_summary(summary, *, context_for_error: str = "summary"):
    """Validate a summary against the WHAT+WHY+SCOPE? structural shape.

    Strict dict-only contract (Adrian's design lock 2026-04-25):

        {"what": str, "why": str, "scope": str?}

    Field intent (be EXPLICIT here -- the gardener has to fix what
    callers get wrong, and the gate has to flag what slips through):

    ``what`` (required, ≥5 chars after strip)
        A NOUN PHRASE that names the entity discriminatively. It
        must distinguish this entity from other entities of similar
        kind. Avoid bare type names ("project", "tool") and
        keyword-soup concatenations.

        GOOD: "InjectionGate (post-retrieval relevance filter)"
        GOOD: "data_migrations stamp table pattern"
        GOOD: "Adrian's primary mempalace dev companion"
        BAD:  "summary contract"  (too generic, doesn't discriminate)
        BAD:  "summary contract what why scope dict"  (keyword soup, not a phrase)
        BAD:  "the project"  (bare type, no identity)

    ``why`` (required, ≥15 chars after strip)
        A PURPOSE / ROLE / CLAIM CLAUSE explaining why this entity
        exists, what it does, or what's claimed about it. It must
        carry NEW information beyond restating ``what``. Test:
        replace ``what`` with "X" -- does ``why`` still make sense
        as an explanation? If ``why`` overlaps heavily with ``what``,
        you've got a redundancy not a why.

        GOOD: "filters retrieved memories before injection via
              Haiku tool-use, emits quality flags for the gardener"
        GOOD: "marks one-shot Python data migrations as applied so
              subsequent KG inits short-circuit O(1)"
        BAD:  "what why scope dict"  (no clause, just labels)
        BAD:  "the summary contract"  (restates 'what')
        BAD:  "is a project"  (placeholder, no real claim)

    ``scope`` (optional, ≤100 chars)
        A TEMPORAL OR DOMAIN qualifier that narrows applicability.
        Use it when the entity has a clear scope; omit it when the
        entity is universal/timeless. Don't pad scope just to fill
        the field.

        GOOD: "Adrian design lock 2026-04-25"
        GOOD: "mempalace internals; v3.1.x"
        GOOD: "Adrian Windows home office; long-context Opus sessions"
        BAD:  "dict"  (a single token, no qualifier)
        BAD:  "scope"  (literal placeholder)

    Embedding-budget cap: the rendered prose form
    (``serialize_summary_for_embedding``) must fit within
    ``_SUMMARY_MAX_LEN`` (280 chars) so it stays a focused embedding
    view per Anthropic Contextual Retrieval 2024.

    Returns ``True`` on success. Raises ``SummaryStructureRequired``
    with a precise message naming the failing field plus the call
    site (``context_for_error``).

    Validation is intentionally STRUCTURAL only -- fields present,
    non-empty, length-bounded. No regex on prose, no role-verb
    detection, no em-dash heuristics. Semantic quality (the keyword-
    soup vs real-clause distinction in the GOOD/BAD examples above)
    is the GARDENER'S job: it flags ``generic_summary`` items and
    proposes Haiku-rewritten replacements. The injection gate has the
    same examples in its system prompt for the ``generic_summary``
    flag rule (see injection_gate.py).

    Strings are NOT accepted on new writes. Callers that previously
    passed prose must migrate to the dict shape; the error message
    spells out the migration. Already-stored prose strings still
    serialize correctly through ``serialize_summary_for_embedding``
    (legacy-read tolerance) -- they just can't be re-written through
    this path.

    References
    ----------
    Anthropic 2024 -- *Introducing Contextual Retrieval*: prepending a
        what/why/role context block lifted retrieval F1 by 35-50%
        across five embedding models. The biggest gain came from the
        role/why piece, not the topic alone.
    Khattab & Zaharia 2020 -- *ColBERT*: late-interaction multi-view
        retrieval benefits from focused per-view content.
    Wadden et al. 2020 -- *SciFact*: structured claim+evidence
        summaries beat freeform on fact-grounded retrieval by 10-20%
        nDCG@10.
    Packer et al. 2023 / Liu et al. 2024 -- *MemGPT / Letta*: agent-
        memory papers showing field-shaped summaries retrieve better
        than freeform; the direct precedent for this dict-storage
        + prose-embedding hybrid.
    """
    if isinstance(summary, str):
        raise SummaryStructureRequired(
            f"{context_for_error}: legacy string form is no longer "
            "accepted on writes. Pass a dict instead: "
            "{'what': '<noun phrase>', 'why': '<purpose / role / "
            "claim>', 'scope': '<temporal/domain qualifier>'?}. "
            "Example: {'what': 'InjectionGate', 'why': 'filters "
            "retrieved memories before injection via Haiku tool-use, "
            "emits quality flags', 'scope': 'one instance per palace "
            "process'}."
        )

    if not isinstance(summary, dict):
        raise SummaryStructureRequired(
            f"{context_for_error}: summary must be a dict "
            f"{{what, why, scope?}}; got {type(summary).__name__}."
        )

    what = summary.get("what")
    why = summary.get("why")
    scope = summary.get("scope")

    if not isinstance(what, str) or len(what.strip()) < _SUMMARY_WHAT_MIN:
        raise SummaryStructureRequired(
            f"{context_for_error}: dict missing or stub 'what' (min "
            f"{_SUMMARY_WHAT_MIN} chars). Required shape: "
            "{'what': '<noun phrase naming the entity>', "
            "'why': '<purpose / role / claim>', "
            "'scope': '<temporal/domain qualifier>'?}. "
            "Example: {'what': 'InjectionGate', 'why': 'filters "
            "retrieved memories before injection via Haiku tool-use, "
            "emits quality flags', 'scope': 'one instance per palace "
            "process'}."
        )
    if not isinstance(why, str) or len(why.strip()) < _SUMMARY_WHY_MIN:
        raise SummaryStructureRequired(
            f"{context_for_error}: dict missing or stub 'why' (min "
            f"{_SUMMARY_WHY_MIN} chars). The 'why' clause must explain "
            "the entity's purpose, role, or claim -- not restate the "
            "name. Bad: 'why: \"the project\"'. Good: 'why: "
            '"orchestrates declare-time intent validation and '
            "retrieval\"'."
        )
    if scope is not None and not isinstance(scope, str):
        raise SummaryStructureRequired(
            f"{context_for_error}: 'scope' must be a string when present, "
            f"got {type(scope).__name__}."
        )
    if scope and len(scope) > _SUMMARY_SCOPE_MAX:
        raise SummaryStructureRequired(
            f"{context_for_error}: 'scope' exceeds {_SUMMARY_SCOPE_MAX} "
            f"chars ({len(scope)} given). Compress to a temporal/domain "
            "qualifier; longer detail belongs in the body."
        )
    # Final embedding-budget check -- even valid fields can blow the
    # 280-char cap if all three are at their max.
    rendered = serialize_summary_for_embedding(summary)
    if len(rendered) > _SUMMARY_MAX_LEN:
        raise SummaryStructureRequired(
            f"{context_for_error}: rendered summary exceeds "
            f"{_SUMMARY_MAX_LEN} chars ({len(rendered)} given). Trim "
            "'why' or 'scope' so the prose form fits the embedding budget."
        )
    return True


def coerce_summary_for_persist(
    summary,
    *,
    context_for_error: str = "summary",
    allow_haiku_coerce: bool = True,
):
    """Validate ``summary`` and return the canonical persisted form.

    Returns the dict ``{what, why, scope?}`` after silently transliterating
    every string field to ASCII via :func:`mempalace.ascii_fold.fold_summary`
    and then passing ``validate_summary``. Raises
    ``SummaryStructureRequired`` on bad input.

    Adrian's design lock 2026-04-27: metadata fields are ASCII-only.
    The fold runs BEFORE ``validate_summary`` so the 280-char rendered
    length cap, the ≥5/≥15 length floors, and the type checks all apply
    to the post-fold form actually persisted. anyascii occasionally
    EXPANDS strings (em-dash ``--`` -> ``--``, ellipsis ``…`` -> ``...``);
    validating the post-fold form keeps the storage contract honest
    rather than leaving a 282-char rendered summary on disk just because
    the pre-fold form clocked in at 280. Long-form ``content`` fields
    stay UTF-8 verbatim -- the fold is summary-scoped on purpose.

    v3.2.4 (Adrian directive 2026-05-12): when validation fails ONLY
    because the rendered prose exceeds ``_SUMMARY_MAX_LEN``, route the
    dict through :func:`mempalace.summary_coerce.haiku_coerce_summary_to_length`
    which asks Claude Haiku to trim wording while preserving meaning.
    If Haiku returns a valid dict that re-passes ``validate_summary``,
    persist that; otherwise raise the original length error. All other
    validation failures (missing/short ``what``/``why``, oversized
    ``scope``, string-instead-of-dict) still raise immediately --
    Haiku coerce is length-only by design. Set
    ``allow_haiku_coerce=False`` to bypass the Haiku path (used by the
    coerce module's own tests so it doesn't recurse).

    Strings raise -- see ``validate_summary`` for the migration path.
    """
    from .ascii_fold import fold_summary  # local import -- avoids circular import at module load

    folded = fold_summary(summary)
    try:
        validate_summary(folded, context_for_error=context_for_error)
    except SummaryStructureRequired as exc:
        if not allow_haiku_coerce or "rendered summary exceeds" not in str(exc):
            raise
        # Length-only failure -- ask Haiku to trim, then re-validate.
        try:
            from .summary_coerce import haiku_coerce_summary_to_length
        except Exception:
            raise exc from None
        trimmed = haiku_coerce_summary_to_length(
            folded,
            max_len=_SUMMARY_MAX_LEN,
            context_for_error=context_for_error,
        )
        if trimmed is None:
            raise
        # Re-fold (Haiku output may have new ASCII-unsafe chars) and re-
        # validate. Disable recursion: even if Haiku still over-shoots,
        # we raise rather than re-coerce.
        folded = fold_summary(trimmed)
        validate_summary(folded, context_for_error=context_for_error)
    # Normalise: strip whitespace, drop empty 'scope'.
    out = {
        "what": folded["what"].strip(),
        "why": folded["why"].strip(),
    }
    scope = folded.get("scope")
    if isinstance(scope, str) and scope.strip():
        out["scope"] = scope.strip()
    return out


# ── Hand-authored ``what`` clauses for seed predicates ───────────────
#
# Cold-start lock 2026-05-01 (Adrian's curation directive): seed
# predicates carry real curated summaries, not template-derived
# placeholders. The `why` is the existing curated description string
# inline at the seed callsite; the `scope` is constraint-derived; the
# `what` lives here so the human-authored identity phrase isn't buried
# in the long tuple. Predicate names alone (e.g. "is_a", 4 chars) fall
# below the gate's 8-char discrimination floor; the lookup phrases
# embed both the predicate name and a one-line role qualifier so the
# identity layer separates each predicate cleanly.
_PREDICATE_WHATS: dict[str, str] = {
    "is_a": "is_a -- taxonomic classification predicate",
    "has_value": "has_value -- attribute value predicate",
    "has_property": "has_property -- named-property predicate",
    "defaults_to": "defaults_to -- default-value predicate",
    "lives_at": "lives_at -- location/address predicate",
    "runs_in": "runs_in -- process-runtime hosting predicate",
    "stored_in": "stored_in -- data-persistence predicate",
    "depends_on": "depends_on -- runtime/build dependency predicate",
    "requires": "requires -- runtime prerequisite predicate",
    "blocks": "blocks -- progress-blocker predicate",
    "enables": "enables -- capability-unlock predicate",
    "must": "must -- positive-rule (required) predicate",
    "must_not": "must_not -- negative-rule (forbidden) predicate",
    "forbids": "forbids -- rule-source prohibition predicate",
    "has_gotcha": "has_gotcha -- known-pitfall predicate",
    "warns_about": "warns_about -- caution predicate",
    "replaced_by": "replaced_by -- supersession predicate",
    "invalidated_by": "invalidated_by -- obsolescence-event predicate",
    "described_by": "described_by -- canonical-description predicate",
    "evidenced_by": "evidenced_by -- supporting-evidence predicate",
    "mentioned_in": "mentioned_in -- passing-reference predicate",
    "session_note_for": "session_note_for -- diary/session-log predicate",
    "derived_from": "derived_from -- extraction provenance predicate",
    "tested_by": "tested_by -- test-coverage predicate",
    "executed_by": "executed_by -- intent-execution agent predicate",
    "targeted": "targeted -- intent-execution slot-target predicate",
    "resulted_in": "resulted_in -- intent-outcome predicate",
    "surfaced": "surfaced -- retrieval-event predicate",
    "rated_useful": "rated_useful -- positive feedback predicate",
    "rated_irrelevant": "rated_irrelevant -- negative feedback predicate",
    "created_under": "created_under -- context-provenance predicate",
    "similar_to": "similar_to -- context-similarity edge predicate",
    "anchored_by": "anchored_by -- context-to-entity Channel B anchor predicate",
}


_INTENT_TYPE_WHATS: dict[str, str] = {
    "inspect": "inspect intent_type -- read-only observation",
    "modify": "modify intent_type -- create/edit codebase artefacts",
    "execute": "execute intent_type -- run a command/script/process",
    "communicate": "communicate intent_type -- chat/notify/post output",
    "research": "research intent_type -- read+web+search compose",
    "wrap_up_session": "wrap_up_session intent_type -- session-finalisation ritual",
}


def _seed_intent_type_summary(name: str, desc: str, parent: str) -> dict:
    """Build a hand-curated ``{what, why, scope}`` summary for a seed intent_type.

    Cold-start lock 2026-05-01: ``what`` from ``_INTENT_TYPE_WHATS``
    (one phrase per declared intent_type), ``why`` is the existing
    curated desc, ``scope`` records the is_a parent in the intent
    hierarchy. New intent_types MUST register an explicit ``what``.
    """
    what = _INTENT_TYPE_WHATS[name]
    why = (desc or "").strip()
    if len(why) < 15:
        raise ValueError(
            f"_seed_intent_type_summary({name!r}): desc too short ({len(why)} chars). "
            f"Curate desc >=15 chars at the seed callsite."
        )
    if len(why) > 160:
        raise ValueError(
            f"_seed_intent_type_summary({name!r}): desc {len(why)} chars; trim to <=160."
        )
    scope = f"intent_type hierarchy; is_a parent={parent}"[:100]
    out = {"what": what, "why": why, "scope": scope}
    return coerce_summary_for_persist(out, context_for_error=f"seed_intent_type_summary({name!r})")


def _seed_predicate_summary(name: str, desc: str, constraints: dict) -> dict:
    """Build a hand-curated ``{what, why, scope}`` summary for a seed predicate.

    Cold-start lock 2026-05-01 (no derivation, no template): combines
    the hand-authored ``what`` from ``_PREDICATE_WHATS`` with the
    existing curated ``desc`` (used as ``why``) and a constraint-derived
    ``scope`` clause. Raises ``KeyError`` if the predicate name isn't
    in the lookup -- new seed predicates MUST register an explicit
    ``what`` phrase, no exceptions.
    """
    what = _PREDICATE_WHATS[name]
    why = (desc or "").strip()
    if len(why) < 15:
        raise ValueError(
            f"_seed_predicate_summary({name!r}): existing desc too short "
            f"({len(why)} chars). Curate a description >=15 chars at the "
            f"seed callsite -- the desc IS the predicate's canonical why."
        )
    cardinality = constraints.get("cardinality", "?")
    subj_kinds = ",".join(constraints.get("subject_kinds") or []) or "any"
    obj_kinds = ",".join(constraints.get("object_kinds") or []) or "any"
    scope = f"{cardinality}; subj={subj_kinds}; obj={obj_kinds}"[:100]
    # The rendered prose form is ``what -- why; scope`` and must fit
    # _SUMMARY_MAX_LEN (280 chars). ``what`` is ~40 chars; scope is
    # <=100; the four-char separator overhead. That leaves <=130 for
    # ``why`` in the worst case. We cap at 160 chars so the helper
    # accepts the great majority of curated descs without hand-trim;
    # any predicate whose curated desc exceeds 160 chars MUST be
    # hand-shortened at the seed callsite (cold-start lock: every
    # field is curated to fit, no programmatic degradation).
    if len(why) > 160:
        raise ValueError(
            f"_seed_predicate_summary({name!r}): curated desc is {len(why)} "
            f"chars; the rendered prose budget needs why<=160. Hand-trim "
            f"the desc at the seed callsite to a tighter purpose clause."
        )
    out = {"what": what, "why": why, "scope": scope}
    return coerce_summary_for_persist(out, context_for_error=f"seed_predicate_summary({name!r})")


# ── Triple statement validation (Adrian's design lock 2026-04-25) ──
#
# Triple statements (kg_add(statement=...)) are the natural-language
# verbalization of an edge -- "Adrian lives in Warsaw" for the triple
# ('adrian', 'lives_in', 'warsaw'). They get embedded into the
# mempalace_triples Chroma collection so the edge becomes a
# first-class semantic-search result. Same retrieval principles apply
# as for entity summaries: a focused WHAT+WHY block embeds better
# than freeform prose (Anthropic Contextual Retrieval 2024).
#
# Structurally identical to summary: {what, why, scope?}. The
# semantic mapping is:
#   - what:  who/what is the edge about (e.g. "Adrian lives in Warsaw")
#   - why:   why this edge exists / what claim it asserts / what evidence
#            (e.g. "primary residence since 2019; reflects current legal address")
#   - scope: optional temporal / domain qualifier (e.g. "since 2019")


class TripleStatementStructureRequired(SummaryStructureRequired):
    """Raised when a triple statement fails the WHAT+WHY+SCOPE? check.

    Subclasses SummaryStructureRequired so callers that catch the
    summary-level exception also catch statement-level failures --
    they share validation surface.
    """


def validate_statement(statement, *, context_for_error: str = "statement"):
    """Validate a triple statement against the WHAT+WHY+SCOPE? shape.

    Same strict dict-only contract as ``validate_summary`` -- passes
    through to it and re-raises any structural error under
    ``TripleStatementStructureRequired`` for caller-side discrimination.

    Per Adrian's design lock 2026-04-25: edges follow the same
    structured contract as records and entities. No regex, no
    auto-derivation; the writer supplies WHAT (the edge in plain
    language) + WHY (the claim / evidence / role) + optional SCOPE.
    """
    try:
        validate_summary(statement, context_for_error=context_for_error)
    except SummaryStructureRequired as exc:
        # Re-raise as the statement subclass so callers that want to
        # distinguish edge-level from entity-level failures can.
        raise TripleStatementStructureRequired(str(exc)) from exc
    return True


def coerce_statement_for_persist(statement, *, context_for_error: str = "statement"):
    """Validate ``statement`` and return the canonical persisted form.

    Mirrors ``coerce_summary_for_persist``. Returns the normalised
    dict; ``serialize_summary_for_embedding`` projects it to the
    prose form actually stored in the triple's `statement` column.
    """
    validate_statement(statement, context_for_error=context_for_error)
    out = {
        "what": statement["what"].strip(),
        "why": statement["why"].strip(),
    }
    scope = statement.get("scope")
    if isinstance(scope, str) and scope.strip():
        out["scope"] = scope.strip()
    return out


# ── 2026-04-28: render-time fact display ──
# Honors Adrian's design lock 2026-04-25 ("no auto-derivation at storage,
# the writer supplies WHAT+WHY+SCOPE?") by computing this purely at
# query-time. The underlying `statement` column stays nullable and
# writer-authored; this helper just gives kg_query callers a
# natural-language label they can read inline instead of having to
# concatenate (subject, predicate, object) tuples themselves every time.
#
# Two cases:
#   (a) statement is populated  → return its `what` (the WHAT clause is
#       the natural-language verbalization the writer authored).
#   (b) statement is null       → synthesize from S-P-O. The synthetic
#       form is structural restatement (predicate underscores → spaces,
#       trailing period), not auto-derived MEANING. The writer's
#       responsibility for authoring rich what/why is unchanged.
def _render_fact_display(fact: dict) -> str:
    """Return a display string for a kg_query fact row.

    See module-level comment block above for the design rationale.
    """
    stmt = fact.get("statement")
    if stmt:
        if isinstance(stmt, str):
            # Stored statements may be JSON-encoded dicts {what,why,scope?}
            # written by coerce_statement_for_persist callers, or raw
            # legacy strings. Try the dict path first; fall back to raw.
            try:
                obj = json.loads(stmt)
            except (ValueError, TypeError):
                obj = None
            if isinstance(obj, dict):
                return obj.get("what") or obj.get("why") or stmt
            return stmt
        if isinstance(stmt, dict):
            return stmt.get("what") or stmt.get("why") or ""
    # Synthetic fallback: structural restatement of the triple, NOT
    # auto-derived meaning. Predicate underscores become spaces so
    # "lives_in" reads as "lives in".
    s = fact.get("subject", "?")
    p = (fact.get("predicate") or "?").replace("_", " ")
    o = fact.get("object", "?")
    return f"{s} {p} {o}."


def _get_triple_collection(create: bool = False):
    """Return the mempalace_triples Chroma collection or None on any error.

    Lazy import + best-effort to avoid coupling the SQL layer to ChromaDB
    at construction time. Uses the live mcp_server _STATE.client_cache when
    available so we share the embedding model + persistent client.

    When `create=False` (default -- used by search-side callers) we only
    return the collection if it already exists, so a search call never
    has the side effect of creating a new Chroma collection in palaces
    that have no triples yet. Write-side callers (add_triple,
    backfill_triple_statements) pass create=True.
    """
    try:
        from . import mcp_server

        client = mcp_server._get_client()
        if client is None:
            return None
        if create:
            # ``hnsw:sync_threshold=100`` -- flush every 100
            # writes so a crashed session leaves at most 100 unprocessed
            # rows in embeddings_queue, well under the lag threshold
            # that triggers the C-level _apply_batch SIGSEGV. See
            # mcp_server._CHROMA_METADATA for the full root-cause note.
            return client.get_or_create_collection(
                TRIPLE_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine", "hnsw:sync_threshold": 100},
            )
        try:
            return client.get_collection(TRIPLE_COLLECTION_NAME)
        except Exception:
            return None
    except Exception:
        return None


def _index_triple_statement(kg, triple_id, sub_id, pred, obj_id, statement, confidence):
    """Upsert the verbalized statement into the triples Chroma collection.

    Best-effort: silent no-op on any failure so write-side errors never
    block the SQL insert. The SQL row remains the source of truth; the
    Chroma index is rebuildable via backfill_triple_statements().

    Structural predicates (is_a, described_by, executed_by, ...) are
    deliberately NOT embedded -- see _TRIPLE_SKIP_PREDICATES. They're high-
    cardinality glue that floods search results with generic statements
    like "research is a inspect" without adding retrievable signal.
    """
    if not statement:
        return
    if pred in _TRIPLE_SKIP_PREDICATES:
        return
    col = _get_triple_collection(create=True)
    if col is None:
        return
    try:
        col.upsert(
            ids=[triple_id],
            documents=[statement],
            metadatas=[
                {
                    "triple_id": triple_id,
                    "subject": sub_id,
                    "predicate": pred,
                    "object": obj_id,
                    "confidence": float(confidence) if confidence is not None else 1.0,
                }
            ],
        )
    except Exception:
        pass


def normalize_entity_name(name: str) -> str:
    """Aggressive entity name normalization for dedup.

    Collapses: hyphens, underscores, dots, spaces, colons, slashes,
    backslashes, CamelCase boundaries, leading articles.

    Does NOT collapse: plurals, abbreviations (handled by semantic
    similarity on entity descriptions instead).

    Examples:
        "The Flowsev Repository" -> "flowsev_repository"
        "flowsev_repository"     -> "flowsev_repository"
        "FlowsevRepository"      -> "flowsev_repository"
        "D:\\Flowsev\\repo"      -> "d_flowsev_repo"
        "paperclip-server"       -> "paperclip_server"
        "paperclip_server"       -> "paperclip_server"
        "the GA agent"           -> "ga_agent"
    """
    if not isinstance(name, str) or not name.strip():
        return "unknown"
    s = name.strip()
    # Adrian's design lock 2026-04-27: anyascii-fold first so unicode-name
    # inputs produce stable lossless ids -- "café" -> "cafe" instead of the
    # lossy "caf_" the [^a-z0-9]+ regex below would otherwise emit. Local
    # import: this module is itself imported during ascii_fold's module
    # load via the package __init__ chain, so the import has to be lazy.
    from .ascii_fold import fold_ascii

    s = fold_ascii(s)
    # Split CamelCase: "FlowsevRepo" -> "Flowsev Repo"
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    # Also split "HTTPServer" -> "HTTP Server"
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    # Lowercase
    s = s.lower()
    # Replace ALL non-alphanumeric with underscore (matches ChromaDB memory ID convention)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    # Collapse repeated underscores
    s = re.sub(r"_+", "_", s)
    # Strip leading/trailing underscores
    s = s.strip("_")
    # Strip leading articles
    for article in ("the_", "a_", "an_"):
        if s.startswith(article):
            s = s[len(article) :]
            break
    return s or "unknown"


def _normalize_predicate(predicate: str) -> str:
    """Normalize predicate strings at the storage boundary.

    Collapses hyphens, spaces, and repeated underscores. Matches how
    normalize_entity_name treats entity names, so `is-a` and `is_a` become
    the same predicate in the DB. Without this, seeded edges (`is-a`) and
    caller writes (`is_a`) were stored as distinct predicates.
    """
    if not isinstance(predicate, str):
        return ""
    s = predicate.strip().lower()
    s = re.sub(r"[-\s]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


class KnowledgeGraph:
    def __init__(self, db_path: str = None):
        # BF1: when the caller doesn't pin db_path, derive it from the live
        # palace_path so the KG always lives next to its Chroma data instead
        # of one directory up. Migrate any legacy ~/.mempalace/knowledge_graph.sqlite3
        # in place on first construction so existing installs keep their data.
        if db_path is None:
            db_path = _resolve_default_kg_path()
            _maybe_migrate_legacy_kg(db_path)
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = None
        self._init_db()
        _active_instances.append(self)

    def _init_db(self):
        """Initialize DB via yoyo-migrations + set PRAGMAs.

        Schema lives in per-file migrations under mempalace/migrations/.
        Each migration runs exactly once and is tracked by yoyo's version table.
        For legacy databases predating yoyo, set MEMPALACE_BOOTSTRAP_LEGACY=1 to
        mark all current migrations as applied without re-running them (since
        CREATE TABLE IF NOT EXISTS / ALTER on existing columns would fail).
        """
        from .migrations import apply_migrations

        # PRAGMAs first (yoyo opens its own connection briefly)
        conn = self._conn()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.commit()

        # For legacy databases that already have the schema but no yoyo marker:
        # detect and bootstrap (mark all migrations applied, run nothing).
        if self._is_legacy_unmarked_db(conn):
            self._bootstrap_yoyo_from_legacy_db()
        else:
            apply_migrations(self.db_path)

        # Data-migrations stamp table (Adrian's followup 2026-05-02).
        # One-shot data migrations -- backfill_seed_chroma,
        # migrate_strip_polluted_context_views,
        # migrate_recompute_similar_to_confidences -- used to iterate every
        # row on every KG init even when there was nothing to migrate. With
        # this table, each helper checks ``_data_migration_applied(name)``
        # first and exits early if stamped; helpers stamp themselves on
        # success via ``_stamp_data_migration(name)``. Two-column schema
        # (name PK, applied_at ISO ts) is intentionally simpler than yoyo's
        # ``_yoyo_migration`` -- yoyo owns SQL schema changes, this owns
        # idempotent Python data-shape migrations whose lifecycle is
        # "run once per palace, then dead-code." Future readers can grep for
        # stamp names to know which palace versions ran which fixes;
        # helpers can be deleted after the rollout window without affecting
        # any palace already stamped.
        with self._conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS data_migrations ("
                "  name TEXT PRIMARY KEY,"
                "  applied_at TEXT NOT NULL"
                ")"
            )

        # Seed canonical ontology on first run (no "thing" class yet)
        # Only for production palaces -- test KGs are empty by design
        if not os.environ.get("MEMPALACE_SKIP_SEED"):
            self.seed_ontology()
            # Channel-separation lock 2026-05-02: in-place strip + edge
            # backfill for existing palaces whose contexts were created
            # under the pre-fix auto-append path. Idempotent (no-op
            # after the first successful run); best-effort (logs and
            # skips per-row failures). See
            # ``migrate_strip_polluted_context_views`` for the
            # algorithm. Test KGs (MEMPALACE_SKIP_SEED=1) skip this.
            try:
                self.migrate_strip_polluted_context_views()
            except Exception:
                # Migration is opportunistic -- seed_ontology has
                # already laid the structural ground truth, so even if
                pass
            # Adrian's followup 2026-05-02: after the strip migration
            # cleans pre-fix contexts of their auto-appended structural
            # views, the existing similar_to edges still carry the stale
            # saturated confidences. The recompute helper rereads each
            # current similar_to edge and updates the confidence against
            # now-clean view geometry. Stamp-protected (skipped after
            # first successful run); best-effort on per-edge failures.
            try:
                self.migrate_recompute_similar_to_confidences()
            except Exception:
                # Like the strip migration, opportunistic -- the channel-
                # separation fix's INTEGRITY does not depend on this; it's
                # cleanup of stale numbers, not a correctness gate.
                pass

    def _is_legacy_unmarked_db(self, conn) -> bool:
        """True if the DB has our tables but no yoyo version marker.

        Such DBs must be bootstrapped (mark migrations applied without running)
        so yoyo doesn't try to re-CREATE tables that already exist.
        """
        # If _yoyo_migration table exists, yoyo has managed this DB before -- no
        # bootstrap needed.
        try:
            has_yoyo = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='_yoyo_migration'"
            ).fetchone()
            if has_yoyo:
                return False
        except sqlite3.OperationalError:
            return False
        # Otherwise: legacy only if we already have our tables
        try:
            has_entities = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='entities'"
            ).fetchone()
            return bool(has_entities)
        except sqlite3.OperationalError:
            return False

    def _bootstrap_yoyo_from_legacy_db(self) -> None:
        """Mark migrations as applied only for schema state already present.

        On a legacy DB (pre-yoyo) we inspect the actual columns/tables and
        mark migrations applied only when their effect is already in place.
        Remaining migrations then run normally to fill the gaps.
        """
        from yoyo import get_backend, read_migrations

        from .migrations import MIGRATIONS_DIR

        conn = self._conn()

        def _has_table(name: str) -> bool:
            return bool(
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (name,),
                ).fetchone()
            )

        def _has_column(table: str, col: str) -> bool:
            try:
                cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
                return col in cols
            except sqlite3.OperationalError:
                return False

        # Migration ID → predicate: does the DB already reflect this migration?
        # Each predicate returns True when the migration's effect is already in
        # place (so we should mark it applied and skip running it).
        already_applied_checks = {
            "001_initial_schema": lambda: _has_table("entities") and _has_table("triples"),
            "002_entity_metadata_columns": lambda: _has_column("entities", "kind"),
            "003_edge_traversal_feedback": lambda: _has_table("edge_traversal_feedback"),
            "004_edge_context_id": lambda: _has_column("edge_traversal_feedback", "context_id"),
            "005_keyword_feedback": lambda: _has_table("keyword_feedback"),
            "006_scoring_weight_feedback": lambda: _has_table("scoring_weight_feedback"),
            "007_context_and_keywords": lambda: _has_table("entity_keywords"),
            "008_rename_drawer_to_memory": lambda: _has_column("keyword_feedback", "memory_id"),
            "009_composite_indexes_and_provenance": lambda: _has_column("entities", "session_id"),
            "010_normalize_predicate_hyphens": lambda: (
                not bool(
                    conn.execute(
                        "SELECT 1 FROM triples WHERE predicate LIKE '%-%' LIMIT 1"
                    ).fetchone()
                )
            ),
            "011_conflict_resolutions": lambda: _has_table("conflict_resolutions"),
            "012_drop_source_closet": lambda: not _has_column("triples", "source_closet"),
            "013_triple_statement": lambda: _has_column("triples", "statement"),
            "014_context_as_entity": lambda: bool(
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='index' "
                    "AND name='idx_triples_created_under_subject' LIMIT 1"
                ).fetchone()
            ),
            "015_retire_old_feedback": lambda: (
                _has_column("triples", "properties") and not _has_table("keyword_feedback")
            ),
            "016_keyword_idf": lambda: _has_table("keyword_idf"),
            "017_link_prediction": lambda: _has_table("link_prediction_candidates"),
            "018_triple_context_feedback": lambda: _has_table("triple_context_feedback"),
            "019_memory_flags": lambda: _has_table("memory_flags"),
            "020_memory_gardener_runs": lambda: _has_table("memory_gardener_runs"),
        }

        backend = get_backend(f"sqlite:///{self.db_path}")
        all_migrations = read_migrations(str(MIGRATIONS_DIR))

        to_mark = []
        to_apply = []
        for m in all_migrations:
            check = already_applied_checks.get(m.id)
            if check is None:
                # Unknown migration (e.g. __init__ Python marker) -- apply normally
                to_apply.append(m)
                continue
            if check():
                to_mark.append(m)
            else:
                to_apply.append(m)

        with backend.lock():
            if to_mark:
                # mark_migrations needs a MigrationList, not a bare list
                try:
                    from yoyo.migrations import MigrationList

                    backend.mark_migrations(MigrationList(to_mark))
                except ImportError:
                    backend.mark_migrations(to_mark)
            if to_apply:
                backend.apply_migrations(backend.to_apply(all_migrations))

    def _conn(self):
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA busy_timeout=10000")
            # v3.2.6 (Adrian directive 2026-05-12): load sqlite-vec on
            # the KG connection too. The BEFORE DELETE trigger on
            # vec_rowid_map (created by sqlite_vec_store bootstrap)
            # cascades DELETE FROM vec_palace, which is a vec0 virtual
            # table -- and vec0 must be loaded on the connection that
            # executes the DELETE. Without this load, a plain DELETE
            # FROM entities through this connection would fire the
            # trigger and fail with 'no such module: vec0'. Best-
            # effort: pre-v3.2.0 environments may not have sqlite_vec
            # available; the cascade still drops the rowid_map row,
            # only the vec_palace cleanup degrades to app-layer.
            try:
                self._connection.enable_load_extension(True)
                try:
                    import sqlite_vec  # noqa: PLC0415

                    sqlite_vec.load(self._connection)
                finally:
                    self._connection.enable_load_extension(False)
            except Exception:
                pass
            # v3.2.5 (Adrian directive 2026-05-12): enforce the FK clauses
            # declared in migrations 001/007/018. Pre-v3.2.5 those clauses
            # were decorative -- PRAGMA defaulted off so dangling rows
            # could accumulate and CASCADE never fired. With this on,
            # deleting an entity automatically cleans up its triples
            # (subject + object), entity_keywords, and triple_context_
            # feedback (via triples cascade). Migration 028 cleans the 3
            # legacy dangling triple_context_feedback rows so this turn-on
            # is safe on existing palaces. vec_rowid_map is NOT FK'd by
            # design -- its entity_id column stores logical vec ids
            # ({eid}/{eid}__v{i}/{cid}_v{i}/triple_id) which span multiple
            # id namespaces, so cascade stays app-layer for that table.
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def _sync_seed_entity_to_chroma(
        self,
        entity_id: str,
        name: str,
        content: str,
        kind: str,
        importance: int,
    ) -> None:
        """Best-effort Chroma sync for a seeded ontology entity.

        Audit follow-up 2026-05-01 (db_audit_2026_05_01_findings): pre-fix
        ``seed_ontology`` called only ``self.add_entity`` which writes to
        SQLite but NOT to the ``mempalace_entities`` Chroma collection.
        That left every seed class / predicate / intent_type as a
        phantom-shaped row -- functional for is_a hierarchies but absent
        from kg_query.details and from kg_search retrieval. The fix is to
        mirror what ``mcp_server._create_entity`` does: SQLite write +
        single-description Chroma upsert. We can't import
        ``_create_entity`` itself at module load time without creating
        an import cycle (knowledge_graph -> mcp_server ->
        knowledge_graph), so the import is lazy and the call is best-
        effort: if mcp_server isn't ready (e.g. unit tests against a
        bare KG, or cold-start before the MCP server has bootstrapped
        its Chroma client cache), the seed proceeds with SQLite-only
        and the new ``backfill_seed_chroma`` helper can re-sync later.
        """
        try:
            from mempalace.mcp_server import _sync_entity_to_chromadb
        except Exception:
            return
        try:
            _sync_entity_to_chromadb(entity_id, name, content, kind, importance)
        except Exception:
            # Best-effort: SQLite is the source of truth. A missing
            # Chroma row is recoverable via backfill_seed_chroma; a
            # crashed seed_ontology is not.
            pass

    def _data_migration_applied(self, name: str) -> bool:
        """Return True if a one-shot data migration has already stamped itself.

        See ``data_migrations`` table bootstrap in ``__init__``. Helpers
        check this before iterating; stamped migrations skip the work.
        """
        try:
            row = (
                self._conn()
                .execute("SELECT 1 FROM data_migrations WHERE name = ?", (name,))
                .fetchone()
            )
            return row is not None
        except Exception:
            # Best-effort: if the table somehow isn't there we fall through
            # to the work. Worst case the helper iterates one extra time.
            return False

    def _stamp_data_migration(self, name: str) -> None:
        """Mark a one-shot data migration as completed.

        Idempotent: ``INSERT OR IGNORE`` so a re-stamp from a duplicate run
        is a no-op. Stamped name should be a stable identifier including a
        date suffix (e.g. ``"strip_polluted_context_views_2026_05_02"``)
        so future readers can match the helper to the rollout window.
        """
        try:
            now = datetime.now().isoformat()
            with self._conn() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO data_migrations (name, applied_at) VALUES (?, ?)",
                    (name, now),
                )
        except Exception:
            # Stamp failure is non-fatal -- if the helper succeeded but the
            # stamp didn't land, the next boot will re-run the helper, find
            # nothing to do (idempotent by construction), and try the stamp
            # again. The cost is a wasted scan, not data corruption.
            pass

    def migrate_recompute_similar_to_confidences(self) -> dict:
        """Recompute stored similar_to edge confidences against clean views.

        Adrian's followup 2026-05-02: after
        ``migrate_strip_polluted_context_views`` cleans pre-fix contexts of
        their auto-appended structural strings, the existing ``similar_to``
        edges still carry their pre-fix saturated confidences (1.0 from the
        byte-identical agent-content view, 0.5 from 2-hop decay of those).
        Those numbers no longer reflect the underlying view geometry. This
        helper rereads each current similar_to edge, recomputes the
        max-of-max similarity over the now-clean Chroma views, and writes
        the new confidence in place. Edges whose new sim falls below
        ``CONTEXT_SIMILAR_THRESHOLD`` are invalidated (valid_to set) since
        they shouldn't have been written under the post-fix path either.

        Stamped via ``_data_migration_applied`` /
        ``_stamp_data_migration`` so a re-run after the first success is
        an O(1) check. Best-effort: per-edge failures log via the standard
        path and continue; SQLite + Chroma stay consistent.

        Returns counts for observability:
        ``{considered, updated, invalidated, skipped, errors}``.

        Cold restart alternative: drops every rated_useful / surfaced /
        fulfills_user_message edge with the contexts. Channel D feedback
        signal goes to zero. Not worth the cleanup; this script preserves
        the history.
        """
        STAMP = "recompute_similar_to_confidences_2026_05_02"
        result = {
            "considered": 0,
            "updated": 0,
            "invalidated": 0,
            "skipped": 0,
            "errors": 0,
            "status": "applied",
        }
        if self._data_migration_applied(STAMP):
            result["status"] = "already_applied"
            return result
        try:
            from mempalace.mcp_server import (
                CONTEXT_SIMILAR_THRESHOLD,
                _get_context_views_collection,
            )
            from mempalace.scoring import multi_view_minmax_sim
            from mempalace.vector_store import (
                CONTEXT_VIEWS_COLLECTION as _CV_NAME,
                get_vector_store as _get_vs,
            )
        except Exception:
            # mcp_server not importable yet (cold-start before bootstrap);
            # bail without stamping so a later boot retries.
            result["status"] = "deferred"
            return result
        # Tier 2 migration 2026-05-10: scoring helpers take (vs,
        # collection_name). We still touch _get_context_views_collection
        # to ensure the chromadb collection exists -- VectorStore queries
        # silently degrade on missing collections.
        try:
            view_col = _get_context_views_collection(create=False)
        except Exception:
            view_col = None
        if view_col is None:
            result["status"] = "deferred"
            return result
        try:
            _vs = _get_vs(None)  # singleton; resolves active palace
        except Exception:
            result["status"] = "deferred"
            return result

        conn = self._conn()
        # Iterate every CURRENT similar_to edge between contexts.
        rows = conn.execute(
            "SELECT id, subject, object, confidence FROM triples "
            "WHERE predicate = 'similar_to' "
            "AND (valid_to IS NULL OR valid_to = '')"
        ).fetchall()
        now = datetime.now().date().isoformat()
        for row in rows:
            result["considered"] += 1
            tid = row["id"]
            sub = row["subject"]
            obj = row["object"]
            try:
                sub_ent = self.get_entity(sub)
                obj_ent = self.get_entity(obj)
                if not (sub_ent and obj_ent):
                    result["skipped"] += 1
                    continue
                sub_props = sub_ent.get("properties") or {}
                if isinstance(sub_props, str):
                    try:
                        sub_props = json.loads(sub_props)
                    except Exception:
                        sub_props = {}
                sub_views = sub_props.get("queries") or []
                if not sub_views:
                    result["skipped"] += 1
                    continue
                pairs = multi_view_minmax_sim(
                    list(sub_views), [obj], _vs, _CV_NAME, where_key="context_id"
                )
                if not pairs or obj not in pairs:
                    result["skipped"] += 1
                    continue
                # multi_view_minmax_sim returns (min_of_max, max_of_max).
                _, new_max_of_max = pairs[obj]
                new_conf = round(float(new_max_of_max), 4)
                if new_conf < CONTEXT_SIMILAR_THRESHOLD:
                    # Below threshold post-recompute -- the edge wouldn't
                    # have been written under the post-fix code path.
                    # Invalidate rather than delete so the audit trail
                    # survives.
                    with conn:
                        conn.execute(
                            "UPDATE triples SET valid_to = ? WHERE id = ?",
                            (now, tid),
                        )
                    result["invalidated"] += 1
                else:
                    with conn:
                        conn.execute(
                            "UPDATE triples SET confidence = ? WHERE id = ?",
                            (new_conf, tid),
                        )
                    result["updated"] += 1
            except Exception:
                result["errors"] += 1
                continue

        # Stamp on success (even partial) so we don't iterate again. If
        # the caller wants a force-rerun they can DELETE the stamp row
        # and re-init the KG.
        self._stamp_data_migration(STAMP)
        return result

    def migrate_strip_polluted_context_views(self) -> dict:  # noqa: C901
        """One-shot in-place migration for the 2026-05-02 channel-separation fix.

        Pre-fix, ``tool_declare_intent`` auto-appended slot-entity content
        and intent_id literals into the views passed to
        ``context_lookup_or_create``. Those polluted views landed in two
        places:

        1. The context entity's ``properties.queries`` JSON list.
        2. The ``mempalace_context_views`` Chroma collection as
           per-view rows under ``{context_id}_v{N}``.

        See ``record_ga_agent_channel_violation_saturation``. Symptoms
        Adrian flagged: similar_to confidence saturating at 1.0 because
        max-of-max picks up the byte-identical entity-content view
        across every context targeting the same slot entity.

        This migration:

        * Iterates every ``kind='context'`` entity in SQLite.
        * For each, computes the structural string set from
          ``properties.entities`` (each entity's ``content[:200]``) plus
          regex-matched intent-id literals (``^intent_[a-z0-9_]+$``).
        * Strips those strings from ``properties.queries`` and rewrites
          the entity row.
        * Deletes the matching rows from the Chroma context-views
          collection (matched by ``where={'context_id': cid}`` + document
          membership in the structural set).
        * Backfills missing ``anchored_by`` graph edges from the context
          to each entity in ``properties.entities`` so Channel B BFS
          becomes reachable for the cleaned context.

        Idempotent: re-running finds no further structural strings to
        strip, no missing edges to add, and the Chroma deletes are
        no-ops because the rows are already gone. Safe to wire into
        the cold-start boot path so existing palaces self-heal on next
        plugin restart.

        Best-effort: any per-context failure is logged and skipped so
        one bad row doesn't abort the whole sweep. Returns counts for
        observability.
        """
        STAMP = "strip_polluted_context_views_2026_05_02"
        result = {
            "considered": 0,
            "stripped_views": 0,
            "deleted_chroma_rows": 0,
            "added_anchored_by_edges": 0,
            "errors": 0,
            "status": "applied",
        }
        if self._data_migration_applied(STAMP):
            result["status"] = "already_applied"
            return result
        try:
            from mempalace.mcp_server import _get_context_views_collection
        except Exception:
            _get_context_views_collection = None
        try:
            view_col = (
                _get_context_views_collection(create=False)
                if _get_context_views_collection
                else None
            )
        except Exception:
            view_col = None

        conn = self._conn()
        rows = conn.execute(
            "SELECT id, properties FROM entities WHERE kind = 'context' AND status = 'active'"
        ).fetchall()
        intent_id_re = re.compile(r"^intent_[a-z0-9_]+$")
        for row in rows:
            result["considered"] += 1
            cid = row["id"]
            try:
                props_raw = row["properties"]
                if not props_raw:
                    continue
                if isinstance(props_raw, str):
                    try:
                        props = json.loads(props_raw)
                    except Exception:
                        continue
                else:
                    props = dict(props_raw)
                queries = props.get("queries") or []
                anchor_entities = props.get("entities") or []
                if not isinstance(queries, list) or not isinstance(anchor_entities, list):
                    continue

                # Build the structural set from anchor entities' content[:200]
                # plus literal intent_id strings ever stored as queries.
                structural: set[str] = set()
                for eid in anchor_entities:
                    if not isinstance(eid, str) or not eid.strip():
                        continue
                    ent_row = conn.execute(
                        "SELECT content FROM entities WHERE id = ?",
                        (eid.strip(),),
                    ).fetchone()
                    if not ent_row:
                        continue
                    content = ent_row["content"] or ""
                    if content:
                        structural.add(content[:200])

                cleaned_queries = []
                for q in queries:
                    if not isinstance(q, str):
                        continue
                    if q in structural or intent_id_re.match(q.strip()):
                        continue
                    cleaned_queries.append(q)

                stripped_count = len(queries) - len(cleaned_queries)
                if stripped_count > 0:
                    props["queries"] = cleaned_queries
                    # Rewrite the entity row with cleaned properties.
                    conn.execute(
                        "UPDATE entities SET properties = ? WHERE id = ?",
                        (json.dumps(props), cid),
                    )
                    conn.commit()
                    result["stripped_views"] += stripped_count

                    # Delete the polluted Chroma rows for this context.
                    if view_col is not None:
                        try:
                            got = view_col.get(
                                where={"context_id": cid},
                                include=["documents"],
                            )
                        except Exception:
                            got = None
                        if got and got.get("ids"):
                            ids_to_delete = []
                            for rid, doc in zip(got["ids"], got.get("documents") or []):
                                if not isinstance(doc, str):
                                    continue
                                if doc in structural or intent_id_re.match(doc.strip()):
                                    ids_to_delete.append(rid)
                            if ids_to_delete:
                                try:
                                    view_col.delete(ids=ids_to_delete)
                                    result["deleted_chroma_rows"] += len(ids_to_delete)
                                except Exception:
                                    result["errors"] += 1

                # Backfill anchored_by edges -- idempotent (add_triple
                # upsert is safe on identical triples).
                for eid in anchor_entities:
                    if not isinstance(eid, str) or not eid.strip():
                        continue
                    try:
                        # Gate-rejected if entity is phantom; that's fine
                        # -- log via the warning path and continue. The
                        # SQLite-level helper raises PhantomEntityRejected
                        # which we treat as a skip rather than a fail.
                        self.add_triple(cid, "anchored_by", eid.strip())
                        result["added_anchored_by_edges"] += 1
                    except Exception:
                        # Most common: already exists OR the entity row
                        # is missing (phantom). Either way, not fatal.
                        pass
            except Exception:
                result["errors"] += 1
                continue
        # Stamp on completion. Re-run is then O(1) -- check the stamp,
        # return early. If something went so wrong that the migration
        # actually couldn't proceed (e.g. mcp_server lazy-import failure
        # made _get_context_views_collection None), we still stamp because
        # the SQLite-side strip work HAS landed; only the Chroma deletes
        # were skipped, and Chroma will eventually be repopulated cleanly
        # via context_lookup_or_create's normal write path.
        self._stamp_data_migration(STAMP)
        return result

    def backfill_seed_chroma(self) -> dict:  # noqa: C901
        """Re-sync any seed-ontology entity that lacks a Chroma row.

        Audit follow-up 2026-05-01: existing palaces that were seeded
        before this fix carry phantom-shaped seed classes (no Chroma
        record). This helper iterates every kind in {class, predicate}
        plus the seeded intent_type entities, reads each row's content
        from SQLite, and calls ``_sync_entity_to_chromadb`` on it.
        Idempotent: ``ecol.upsert`` overwrites existing rows cleanly.

        Returns a dict with counts of entities considered + synced for
        observability. Best-effort on Chroma: if the entity collection
        isn't available the function returns zeros without raising so
        callers can run it during boot without crashing the palace.
        """
        STAMP = "backfill_seed_chroma_2026_05_01"
        result = {
            "considered": 0,
            "synced": 0,
            "skipped_no_content": 0,
            "status": "applied",
        }
        if self._data_migration_applied(STAMP):
            result["status"] = "already_applied"
            return result
        try:
            from mempalace.mcp_server import _sync_entity_to_chromadb
        except Exception:
            return result
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT id, name, kind, content, importance
            FROM entities
            WHERE kind IN ('class', 'predicate')
               OR id IN (
                  SELECT subject FROM triples
                  WHERE predicate='is_a' AND object='intent_type'
                    AND (valid_to IS NULL OR valid_to='')
               )
            """
        ).fetchall()
        for row in rows:
            result["considered"] += 1
            eid = row["id"]
            ename = row["name"] or eid
            content = row["content"] or ""
            kind = row["kind"] or "class"
            importance = int(row["importance"] or 3)
            if not content.strip():
                # Skip rows whose SQLite content is empty -- nothing to
                # embed, so a Chroma write would silently no-op anyway.
                result["skipped_no_content"] += 1
                continue
            try:
                _sync_entity_to_chromadb(eid, ename, content, kind, importance)
                result["synced"] += 1
            except Exception:
                # Best-effort. SQLite remains the source of truth; the
                # caller can retry the backfill later.
                pass
        # Stamp on completion -- subsequent boots become O(1).
        self._stamp_data_migration(STAMP)
        return result

    def backfill_all_entity_vectors(  # noqa: C901
        self,
        *,
        dry_run: bool = False,
        force: bool = False,
        kinds: tuple = ("entity", "class", "predicate", "record", "context"),
    ) -> dict:
        """Rebuild vec_palace from the entities table (v3.2.2 multi-view).

        Walks every active (status='active') row in entities and writes
        THREE flavours of vec rows so all four retrieval channels resolve
        against pre-v3.2.0 corpora:

        1. Single-row content (``mempalace_records``): ``id={eid}``,
           ``doc=entity.content``. Back-compat with the legacy single-
           vector channel A path + callers that look the entity up by
           its bare id (e.g. lazy-queue fallbacks).
        2. Multi-view rows (``mempalace_records``): ``id={eid}__v{i}``,
           ``doc=queries[i]``, ``metadata.entity_id=eid``, plus
           ``view_index`` and (last view) ``is_summary_view=True``.
           Source of ``queries[i]`` is the entity's creation Context
           entity (``properties.queries`` JSON). If unavailable, the
           rendered ``content`` becomes the single base view. The
           rendered summary is appended as the trailing view. Skipped
           for ``kind in ('operation','state_schema')`` -- those kinds
           never get multi-view rows by design.
        3. Context-view rows (``mempalace_context_views``):
           ``id={cid}_v{i}``, ``doc=queries[i]``,
           ``metadata.context_id=cid``. Only for ``kind='context'``
           entities. Powers Channel D similar_to walks.

        Idempotent via STAMP ``backfill_entity_vectors_v322_2026_05_12``
        in the data_migrations table.

        Adrian directive 2026-05-12: v3.2.1's single-row backfill left
        Channel A multi-view + Channel D cold against legacy palaces
        whose vectors lived in pre-removal chromadb. v3.2.2 rebuilds
        the full retrieval surface from the deterministic SQL inputs
        already in the same `.db` file.

        Parameters
        ----------
        dry_run : bool
            If True, walk + count but do not write. Honors no STAMP.
        force : bool
            If True, ignore the STAMP and re-run even if a prior pass
            stamped the data_migrations table.
        kinds : tuple[str, ...]
            Which entity kinds to rebuild.

        Returns
        -------
        dict
            ``{considered, single_synced, multi_view_synced,
            context_views_synced, skipped_no_content, errors, status,
            dry_run}``. ``status`` is one of 'applied' /
            'already_applied' / 'no_embedder' / 'no_vectorstore'.
        """
        import json as _json  # noqa: PLC0415
        import os as _os  # noqa: PLC0415

        STAMP = "backfill_entity_vectors_v323_2026_05_12"
        result = {
            "considered": 0,
            "single_synced": 0,
            "multi_view_synced": 0,
            "context_views_synced": 0,
            "skipped_no_content": 0,
            "errors": 0,
            "status": "applied",
            "dry_run": bool(dry_run),
        }
        if not dry_run and not force and self._data_migration_applied(STAMP):
            result["status"] = "already_applied"
            return result

        palace_path = _os.path.dirname(_os.path.abspath(self.db_path))

        from mempalace.embedder import get_default_embedder  # noqa: PLC0415
        from mempalace.vector_store import (  # noqa: PLC0415
            CONTEXT_VIEWS_COLLECTION,
            RECORDS_COLLECTION,
            get_vector_store,
        )

        embedder = get_default_embedder()
        if embedder is None:
            result["status"] = "no_embedder"
            return result

        try:
            vs = get_vector_store(palace_path)
        except Exception as exc:  # pragma: no cover - defensive
            result["status"] = "no_vectorstore: " + type(exc).__name__ + ": " + str(exc)
            return result

        conn = self._conn()

        # ── Pre-build {context_id -> queries[]} from kind=context rows ──
        # Lets us reconstruct the multi-view perspectives that originally
        # produced each entity's vec rows. Queries are persisted on the
        # context entity's properties JSON (mcp_server context_lookup_
        # or_create write site).
        ctx_queries: dict = {}
        try:
            ctx_rows = conn.execute(
                "SELECT id, properties FROM entities WHERE kind='context' AND status='active'"
            ).fetchall()
            for cr in ctx_rows:
                try:
                    props = _json.loads(cr["properties"] or "{}")
                    qs = props.get("queries") or []
                    if isinstance(qs, list):
                        cleaned = [q for q in qs if isinstance(q, str) and q.strip()]
                        if cleaned:
                            ctx_queries[cr["id"]] = cleaned
                except Exception:
                    continue
        except Exception:
            ctx_queries = {}

        kinds_csv = ",".join("'" + k + "'" for k in kinds)
        rows = conn.execute(
            "SELECT id, name, kind, content, importance, "
            "       creation_context_id, properties "
            "FROM entities "
            "WHERE kind IN (" + kinds_csv + ") "
            "  AND status = 'active'"
        ).fetchall()

        BATCH = 64
        # Shared records buffer + parallel bucket tags so we can
        # attribute each landed row to its write-path counter on flush.
        rec_ids: list = []
        rec_docs: list = []
        rec_metas: list = []
        rec_buckets: list = []  # entries: "single" or "multi"
        ctx_ids: list = []
        ctx_docs: list = []
        ctx_metas: list = []

        def _flush_rec():
            nonlocal rec_ids, rec_docs, rec_metas, rec_buckets
            if not rec_ids or dry_run:
                rec_ids, rec_docs, rec_metas, rec_buckets = [], [], [], []
                return
            try:
                emb = embedder(rec_docs)
                vs.upsert(
                    RECORDS_COLLECTION,
                    ids=rec_ids,
                    documents=rec_docs,
                    metadatas=rec_metas,
                    embeddings=emb,
                )
                for b in rec_buckets:
                    if b == "single":
                        result["single_synced"] += 1
                    else:
                        result["multi_view_synced"] += 1
            except Exception:
                result["errors"] += len(rec_ids)
            rec_ids, rec_docs, rec_metas, rec_buckets = [], [], [], []

        def _flush_ctx():
            nonlocal ctx_ids, ctx_docs, ctx_metas
            if not ctx_ids or dry_run:
                ctx_ids, ctx_docs, ctx_metas = [], [], []
                return
            try:
                emb = embedder(ctx_docs)
                vs.upsert(
                    CONTEXT_VIEWS_COLLECTION,
                    ids=ctx_ids,
                    documents=ctx_docs,
                    metadatas=ctx_metas,
                    embeddings=emb,
                )
                result["context_views_synced"] += len(ctx_ids)
            except Exception:
                result["errors"] += len(ctx_ids)
            ctx_ids, ctx_docs, ctx_metas = [], [], []

        for row in rows:
            result["considered"] += 1
            eid = row["id"]
            ename = row["name"] or eid
            content = (row["content"] or "").strip()
            kind = row["kind"] or "entity"
            importance = int(row["importance"] or 3)
            creation_cid = row["creation_context_id"] or ""
            if not content:
                result["skipped_no_content"] += 1
                continue

            # ── Derive rendered_summary ────────────────────────────────
            # Mirrors render_memory_preview's SQL-side render: parse the
            # properties JSON, extract properties["summary"] (the structured
            # {what, why, scope} dict), and render via
            # serialize_summary_for_embedding. For non-records the entity
            # row's content IS this rendered string already (the write
            # path stores it that way), so the fallback `content` keeps
            # non-record behaviour identical to v3.2.2. For records,
            # content holds the verbatim record body NOT the summary
            # prose -- so this derivation is what makes the record's
            # trailing summary view correct (Adrian's audit 2026-05-12).
            props_raw = row["properties"] if "properties" in row.keys() else None
            rendered_summary = ""
            if props_raw:
                try:
                    pd = _json.loads(props_raw) if isinstance(props_raw, str) else props_raw
                    if isinstance(pd, dict):
                        sd = pd.get("summary")
                        if isinstance(sd, dict):
                            rendered_summary = serialize_summary_for_embedding(sd).strip()
                except Exception:
                    rendered_summary = ""
            if not rendered_summary and kind != "record":
                rendered_summary = content

            # ── Path 1: single-row content ────────────────────────────
            rec_ids.append(eid)
            rec_docs.append(content)
            rec_metas.append(
                {
                    "name": ename,
                    "kind": kind,
                    "importance": importance,
                    "backfilled": True,
                }
            )
            rec_buckets.append("single")
            if not dry_run and len(rec_ids) >= BATCH:
                _flush_rec()

            # ── Path 2: multi-view rows ───────────────────────────────
            if kind not in ("operation", "state_schema"):
                base_views = list(ctx_queries.get(creation_cid, []))
                if not base_views:
                    base_views = [content]
                # Append rendered summary as the trailing summary view
                # (v3.2.3 -- formerly content; wrong for records).
                summary_view_index = -1
                if rendered_summary and (not base_views or base_views[-1] != rendered_summary):
                    base_views.append(rendered_summary)
                    summary_view_index = len(base_views) - 1
                base_meta = {
                    "name": ename,
                    "kind": kind,
                    "importance": importance,
                    "backfilled": True,
                }
                for i, vdoc in enumerate(base_views):
                    rec_ids.append(eid + "__v" + str(i))
                    rec_docs.append(vdoc)
                    m = dict(base_meta)
                    m["view_index"] = i
                    m["entity_id"] = eid
                    if i == summary_view_index:
                        m["is_summary_view"] = True
                    rec_metas.append(m)
                    rec_buckets.append("multi")
                    if not dry_run and len(rec_ids) >= BATCH:
                        _flush_rec()

            # ── Path 3: context-view rows ─────────────────────────────
            if kind == "context":
                cviews = list(ctx_queries.get(eid, []))
                # v3.2.3 -- mirror context_lookup_or_create: append the
                # rendered summary as the trailing context-view, flagged
                # is_summary_view so multi_view_max_sim can weight it.
                cv_summary_idx = -1
                if rendered_summary and (not cviews or cviews[-1] != rendered_summary):
                    cviews.append(rendered_summary)
                    cv_summary_idx = len(cviews) - 1
                for i, vdoc in enumerate(cviews):
                    ctx_ids.append(eid + "_v" + str(i))
                    ctx_docs.append(vdoc)
                    m = {
                        "context_id": eid,
                        "view_index": i,
                        "source": "backfill",
                    }
                    if i == cv_summary_idx:
                        m["is_summary_view"] = True
                    ctx_metas.append(m)
                    if not dry_run and len(ctx_ids) >= BATCH:
                        _flush_ctx()

        # Final partial flushes
        if not dry_run:
            _flush_rec()
            _flush_ctx()
            self._stamp_data_migration(STAMP)
        return result

    def backfill_l3_body_views(  # noqa: C901
        self,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> dict:
        """One-shot retrofit: write Level-3 ``{eid}__body`` vec rows for
        every existing entity/record whose content carries information
        beyond its L1 identity and L2 rendered-summary surfaces.

        Reinstated v3.7.29 (Adrian directive 2026-05-18). The 2026-04
        refactor dropped L3 BODY entirely on the theory that
        MiniLM-L6's 256-token ceiling made long-content embeddings
        noisy; Adrian re-locked the design: content MUST be a stored
        view in the multi-view system alongside summary + queries.
        Live writes after v3.7.29 emit the L3 view inline via
        ``entity_gate._write_identity_and_probe_views`` (for entities)
        and ``mcp_server._add_memory_internal`` (for records). This
        method retrofits the missing rows for everything that already
        existed in the palace before the cutover.

        Walks every active ``entities`` row whose ``content`` field is
        non-empty AND distinct from the rendered summary prose. For
        each, embeds the content (truncated to ``_EMBED_DOC_MAX_CHARS``
        for the MiniLM-L6 256-token ceiling) and upserts a single vec
        row at ``{eid}__body`` with ``view_kind='body'`` +
        ``view_index=-2`` metadata. Skips entities where ``{eid}__body``
        already exists in the vec store (incremental re-runs are
        cheap).

        Idempotent via STAMP ``backfill_l3_body_views_v3729_2026_05_18``
        in ``data_migrations``.

        Parameters
        ----------
        dry_run : bool
            Walk + count but do not write. Honors no STAMP.
        force : bool
            Ignore the STAMP and re-run even if a prior pass stamped.

        Returns
        -------
        dict
            ``{considered, body_synced, skipped_no_content,
            skipped_duplicate_of_summary, skipped_already_present,
            errors, status, dry_run}``. ``status`` is one of
            'applied' / 'already_applied' / 'no_embedder' /
            'no_vectorstore'.
        """
        import json as _json  # noqa: PLC0415
        import os as _os  # noqa: PLC0415

        STAMP = "backfill_l3_body_views_v3729_2026_05_18"
        result = {
            "considered": 0,
            "body_synced": 0,
            "skipped_no_content": 0,
            "skipped_duplicate_of_summary": 0,
            "skipped_already_present": 0,
            "errors": 0,
            "status": "applied",
            "dry_run": bool(dry_run),
        }
        if not dry_run and not force and self._data_migration_applied(STAMP):
            result["status"] = "already_applied"
            return result

        palace_path = _os.path.dirname(_os.path.abspath(self.db_path))

        from mempalace.embedder import get_default_embedder  # noqa: PLC0415
        from mempalace.vector_store import (  # noqa: PLC0415
            RECORDS_COLLECTION,
            get_vector_store,
        )

        embedder = get_default_embedder()
        if embedder is None:
            result["status"] = "no_embedder"
            return result

        try:
            vs = get_vector_store(palace_path)
        except Exception:
            result["status"] = "no_vectorstore"
            return result

        # Walk every active entity row that has content. The status
        # filter mirrors the canonical "live" set; soft-deleted rows
        # are not re-embedded.
        with self._conn() as conn:
            rows = list(
                conn.execute(
                    "SELECT id, name, kind, content, importance, properties "
                    "FROM entities "
                    "WHERE status = 'active' "
                    "AND content IS NOT NULL "
                    "AND length(trim(content)) > 0"
                ).fetchall()
            )

        # Truncation ceiling mirrors _add_memory_internal +
        # backfill_all_entity_vectors: MiniLM-L6's 256-token cap
        # means content beyond ~1800 chars gets truncated to fit.
        _EMBED_DOC_MAX_CHARS = 1800

        # Probe vec store for pre-existing __body rows so we skip
        # incremental re-embeds when nothing changed. Cheap: one
        # whole-collection list of ids filtered by suffix.
        try:
            present_ids = set(vs.all_ids(RECORDS_COLLECTION))
        except Exception:
            present_ids = set()

        BATCH = 64
        batch_ids: list[str] = []
        batch_docs: list[str] = []
        batch_metas: list[dict] = []
        batch_embeds: list[list[float]] = []

        def _flush() -> None:
            if not batch_ids:
                return
            try:
                vs.upsert(
                    RECORDS_COLLECTION,
                    ids=list(batch_ids),
                    documents=list(batch_docs),
                    metadatas=list(batch_metas),
                    embeddings=list(batch_embeds),
                )
                result["body_synced"] += len(batch_ids)
            except Exception:
                result["errors"] += len(batch_ids)
            batch_ids.clear()
            batch_docs.clear()
            batch_metas.clear()
            batch_embeds.clear()

        for row in rows:
            result["considered"] += 1
            eid = row["id"]
            ename = row["name"] or eid
            kind = row["kind"] or "entity"
            importance = int(row["importance"] or 3)
            content = (row["content"] or "").strip()
            if not content:
                result["skipped_no_content"] += 1
                continue

            body_view_id = f"{eid}__body"
            if not force and body_view_id in present_ids:
                result["skipped_already_present"] += 1
                continue

            # Derive rendered summary from properties.summary so we can
            # skip writes where content IS the summary (avoids
            # duplicate L2/L3 vectors). Mirrors the live-write
            # distinctness check.
            props_raw = row["properties"] if "properties" in row.keys() else None
            rendered_summary = ""
            if props_raw:
                try:
                    pd = _json.loads(props_raw) if isinstance(props_raw, str) else props_raw
                    if isinstance(pd, dict):
                        sd = pd.get("summary")
                        if isinstance(sd, dict):
                            rendered_summary = serialize_summary_for_embedding(sd).strip()
                except Exception:
                    rendered_summary = ""

            if content == rendered_summary:
                result["skipped_duplicate_of_summary"] += 1
                continue

            body_doc = content
            if len(body_doc) > _EMBED_DOC_MAX_CHARS:
                body_doc = body_doc[: _EMBED_DOC_MAX_CHARS - 1].rstrip() + "..."

            meta = {
                "name": ename,
                "kind": kind,
                "importance": importance,
                "entity_id": eid,
                "view_kind": "body",
                "view_index": -2,
                "backfilled": True,
            }
            batch_ids.append(body_view_id)
            batch_docs.append(body_doc)
            batch_metas.append(meta)

            if not dry_run and len(batch_ids) >= BATCH:
                # Embed + flush in batches of BATCH to amortise the
                # ONNX call overhead. Embedder.__call__ accepts a
                # list[str] and returns list[list[float]].
                try:
                    batch_embeds.extend(embedder(list(batch_docs[-len(batch_ids) :])))
                except Exception:
                    # Embedder failure: bail this batch, count as errors.
                    result["errors"] += len(batch_ids)
                    batch_ids.clear()
                    batch_docs.clear()
                    batch_metas.clear()
                    batch_embeds.clear()
                    continue
                _flush()

        # Final partial flush
        if not dry_run and batch_ids:
            try:
                batch_embeds.extend(embedder(list(batch_docs)))
                _flush()
            except Exception:
                result["errors"] += len(batch_ids)

        if not dry_run:
            self._stamp_data_migration(STAMP)
        return result

    def backfill_all_triple_statements(  # noqa: C901
        self,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> dict:
        """Rebuild the ``mempalace_triples`` vec rows from triples table.

        Walks every active triple (``valid_to IS NULL OR valid_to=''``)
        whose predicate is NOT in ``_TRIPLE_SKIP_PREDICATES`` and
        re-embeds its ``statement`` column into the triples vec
        collection. Mirrors the shape of :func:`_index_triple_statement`
        (id=triple_id, metadatas carry triple_id/subject/predicate/
        object/confidence) so post-backfill reads land on the same
        physical rows as live writes.

        Idempotent via STAMP
        ``backfill_triple_statements_v322_2026_05_12`` in
        data_migrations.

        Parameters
        ----------
        dry_run : bool
            Walk + count, no writes.
        force : bool
            Ignore STAMP, re-run.

        Returns
        -------
        dict
            ``{considered, synced, skipped_no_statement,
            skipped_predicate, errors, status, dry_run}``.
        """
        import os as _os  # noqa: PLC0415

        STAMP = "backfill_triple_statements_v322_2026_05_12"
        result = {
            "considered": 0,
            "synced": 0,
            "skipped_no_statement": 0,
            "skipped_predicate": 0,
            "errors": 0,
            "status": "applied",
            "dry_run": bool(dry_run),
        }
        if not dry_run and not force and self._data_migration_applied(STAMP):
            result["status"] = "already_applied"
            return result

        palace_path = _os.path.dirname(_os.path.abspath(self.db_path))

        from mempalace.embedder import get_default_embedder  # noqa: PLC0415
        from mempalace.vector_store import (  # noqa: PLC0415
            TRIPLES_COLLECTION,
            get_vector_store,
        )

        embedder = get_default_embedder()
        if embedder is None:
            result["status"] = "no_embedder"
            return result

        try:
            vs = get_vector_store(palace_path)
        except Exception as exc:  # pragma: no cover - defensive
            result["status"] = "no_vectorstore: " + type(exc).__name__ + ": " + str(exc)
            return result

        conn = self._conn()
        rows = conn.execute(
            "SELECT id, subject, predicate, object, statement, confidence "
            "FROM triples "
            "WHERE valid_to IS NULL OR valid_to = ''"
        ).fetchall()

        BATCH = 64
        b_ids: list = []
        b_docs: list = []
        b_metas: list = []

        def _flush():
            nonlocal b_ids, b_docs, b_metas
            if not b_ids or dry_run:
                b_ids, b_docs, b_metas = [], [], []
                return
            try:
                emb = embedder(b_docs)
                vs.upsert(
                    TRIPLES_COLLECTION,
                    ids=b_ids,
                    documents=b_docs,
                    metadatas=b_metas,
                    embeddings=emb,
                )
                result["synced"] += len(b_ids)
            except Exception:
                result["errors"] += len(b_ids)
            b_ids, b_docs, b_metas = [], [], []

        for row in rows:
            result["considered"] += 1
            pred = row["predicate"] or ""
            if pred in _TRIPLE_SKIP_PREDICATES:
                result["skipped_predicate"] += 1
                continue
            stmt = (row["statement"] or "").strip()
            if not stmt:
                result["skipped_no_statement"] += 1
                continue
            if dry_run:
                continue
            b_ids.append(row["id"])
            b_docs.append(stmt)
            b_metas.append(
                {
                    "triple_id": row["id"],
                    "subject": row["subject"],
                    "predicate": pred,
                    "object": row["object"],
                    "confidence": float(row["confidence"] or 1.0),
                }
            )
            if len(b_ids) >= BATCH:
                _flush()

        if not dry_run:
            _flush()
            self._stamp_data_migration(STAMP)
        return result

    def seed_ontology(self):
        """Seed canonical classes, predicates, and intent types. Idempotent.

        Called automatically on first run (empty entities table) or on demand.
        Uses add_entity + add_triple, so normalization and schema are consistent.

        Cold-start lock 2026-05-01 follow-up (audit
        ``record_ga_agent_db_audit_2026_05_01_findings``): the seed
        path now ALSO writes each entity to the ``mempalace_entities``
        Chroma collection via ``_sync_seed_entity_to_chroma``. Pre-fix
        the seed wrote SQLite only, leaving every root class
        phantom-shaped (no kg_query.details, invisible to kg_search).

        ENV: ``MEMPALACE_SKIP_SEED_CHROMA_SYNC=1`` makes the seed populate
        SQLite only and skip the per-entity Chroma sync (and the
        idempotent ``backfill_seed_chroma`` re-run on the early-return
        path). The Chroma sync's per-entity ONNX embedding dominates
        seed_ontology runtime (~30-78s on cold caches) so test harnesses
        that need fast canonical-ontology seeding -- and that don't read
        Chroma for seeded entities -- can opt out by setting this var
        before the first ``KnowledgeGraph(...)`` instantiation. Production
        callers should leave it unset; mirrors the existing
        ``MEMPALACE_SKIP_SEED`` / ``MEMPALACE_BOOTSTRAP_LEGACY`` env-var
        pattern. Callers that opt out and later need Chroma rows for
        seeded entities can call ``backfill_seed_chroma()`` explicitly.
        """
        # Env-var gate: when set, skip per-entity Chroma writes (and the
        # idempotent backfill on early-return). Read once up-front so a
        # mid-loop env mutation can't yield half-synced seed state.
        sync_chroma = not os.environ.get("MEMPALACE_SKIP_SEED_CHROMA_SYNC")

        conn = self._conn()
        # Check if ontology already seeded (look for root class "thing")
        thing = conn.execute("SELECT id FROM entities WHERE id = 'thing'").fetchone()
        if thing:
            # Audit follow-up 2026-05-01: existing palaces seeded
            # before the Chroma-sync fix carry phantom-shaped seed
            # classes (SQLite rows but no mempalace_entities Chroma
            # row). Auto-run the backfill on every seed invocation so
            # the next plugin restart heals them. Idempotent: chroma
            # upsert overwrites cleanly, and the helper is a no-op when
            # mcp_server isn't ready (returns considered=0).
            #
            # Gated by sync_chroma so opt-out callers don't pay the
            # backfill cost on the idempotent re-run path either.
            if sync_chroma:
                try:
                    self.backfill_seed_chroma()
                except Exception:
                    # Best-effort: SQLite remains the source of truth. A
                    # failed backfill leaves the existing phantom state
                    # intact and the next restart can retry.
                    pass
            return  # Already seeded

        # ── Classes (kind=class) ──
        # Cold-start lock 2026-05-01 (Adrian's curation directive): each
        # seed class carries an inline hand-curated {what, why, scope?}
        # summary. The shape is a list of dicts (not tuples) so the
        # semantic content is self-documenting. Curated once at design
        # time -- these are the ontology spine and rarely change.
        classes: list[dict] = [
            {
                "name": "thing",
                "summary": {
                    "what": "thing -- ontology root class",
                    "why": "universal taxonomic anchor; every other class is_a thing, so retrieval and walks have a shared top-level entrypoint",
                    "scope": "mempalace ontology root; never invalidated",
                },
                "importance": 5,
            },
            {
                "name": "system",
                "summary": {
                    "what": "system class -- running infrastructure",
                    "why": "names servers, databases, containers, and long-lived services as a distinct kind so retrieval can scope queries to operational components vs people, files, or concepts",
                    "scope": "infrastructure tier of the ontology",
                },
                "importance": 4,
            },
            {
                "name": "person",
                "summary": {
                    "what": "person class -- human individuals",
                    "why": "anchors humans (vs agents, vs systems) so social-graph triples (parent_of, works_at, knows) target a typed kind and retrieval can filter people-only",
                    "scope": "social tier of the ontology",
                },
                "importance": 4,
            },
            {
                "name": "agent",
                "summary": {
                    "what": "agent class -- AI agents in mempalace",
                    "why": "names the class every wake_up'd agent entity is_a, so cross-agent retrieval, diary scoping, and added_by validation all have a typed anchor",
                    "scope": "AI-runtime tier; one instance per declared agent identity",
                },
                "importance": 4,
                # State-protocol v1 (Adrian Option B 2026-05-03): agent
                # instances are state-bearing; their slot payload validates
                # against the agent_state schema in state_schemas.STATE_SCHEMAS.
                "state_schema_id": "agent_state",
            },
            {
                "name": "project",
                "summary": {
                    "what": "project class -- repos and software products",
                    "why": "groups files, tools, and processes under a top-level codebase identity so retrieval can scope 'within mempalace' vs 'within DSpot' vs cross-project",
                    "scope": "codebase tier of the ontology",
                },
                "importance": 4,
            },
            {
                "name": "file",
                "summary": {
                    "what": "file class -- paths in a project",
                    "why": "names individual source/config files as typed entities so slot validation, auto-declare, and gardener flagging all target the same kind",
                    "scope": "filesystem leaf of the project tier",
                },
                "importance": 3,
            },
            {
                "name": "rule",
                "summary": {
                    "what": "rule class -- human-authored directives",
                    "why": "anchors standing orders / constraints / preferences (Adrian's locks, project conventions) as a distinct kind so retrieval can surface 'what must I always do' separately from facts",
                    "scope": "behavioural-policy tier; persists across sessions",
                },
                "importance": 4,
            },
            {
                "name": "tool",
                "summary": {
                    "what": "tool class -- software tools and CLIs",
                    "why": "names invocable utilities (git, ruff, pytest, etc.) so depends_on / requires triples land on a typed target and the tool ecosystem is queryable",
                    "scope": "tooling tier of the project ontology",
                },
                "importance": 3,
            },
            {
                "name": "process",
                "summary": {
                    "what": "process class -- workflows and procedures",
                    "why": "names recurring multi-step operations (release, deploy, audit) so they can be cited as targets of has_status, blocks, enables triples without conflating with one-shot intents",
                    "scope": "procedural tier; instance-per-named-workflow",
                },
                "importance": 3,
            },
            {
                "name": "concept",
                "summary": {
                    "what": "concept class -- abstract ideas / patterns",
                    "why": "names design patterns, formulas, theorems so they can be cited as evidence and walked via described_by / mentioned_in",
                    "scope": "abstract tier; survives instances that reference it",
                },
                "importance": 3,
            },
            {
                "name": "environment",
                "summary": {
                    "what": "environment class -- runtime hosts",
                    "why": "names containers, VMs, OS environments where processes/services run, so runs_in / stored_in triples land on a typed target distinct from the project itself",
                    "scope": "runtime-host tier; one per logical environment",
                },
                "importance": 3,
            },
            {
                "name": "intent_type",
                "summary": {
                    "what": "intent_type class -- root for intent kinds",
                    "why": "every declared intent is_a some intent_type subclass; root anchors the is_a hierarchy so tool_permissions inherit",
                    "scope": "intent-protocol tier; root of the action vocabulary",
                },
                "importance": 5,
                # State-protocol v1 (Adrian Option B 2026-05-03): intent
                # executions are state-bearing; their slot payload validates
                # against the intent_state schema in state_schemas.STATE_SCHEMAS.
                "state_schema_id": "intent_state",
            },
            {
                "name": "context",
                "summary": {
                    "what": "context class -- first-class retrieval contexts",
                    "why": "kind='context' entities minted by declare_intent / declare_operation / kg_search; accrete via MaxSim and link via created_under",
                    "scope": "retrieval tier; one per distinct semantic context",
                },
                "importance": 5,
            },
        ]
        for entry in classes:
            name = entry["name"]
            summary = entry["summary"]
            imp = entry["importance"]
            # Content (long-form prose, used for embedding + display) is
            # the rendered summary itself -- the structured dict IS the
            # canonical description for ontology entries; no separate
            # legacy-content prose needed.
            content = serialize_summary_for_embedding(summary)
            # State-protocol v1 (Adrian Option B 2026-05-03): when an entry
            # carries a state_schema_id, forward it into properties along
            # with state_updatable=True. The validator path reads
            # class.properties.state_schema_id -> state_schemas.get_schema()
            # at delta time. Entries without state_schema_id default to
            # state_updatable=False (absent property) -- non-state-bearing.
            properties = {"summary": summary}
            state_schema_id = entry.get("state_schema_id")
            if state_schema_id:
                properties["state_updatable"] = True
                properties["state_schema_id"] = state_schema_id
            eid = self.add_entity(
                name,
                kind="class",
                content=content,
                importance=imp,
                properties=properties,
            )
            # State-protocol v1 (Adrian 2026-05-03): add_entity's
            # ON CONFLICT path may not refresh properties on existing
            # rows, so a class minted pre-Slice-A (and later mutated by
            # the gardener summary-rewrite path which only stores
            # summary_rewrite_count) loses the state_updatable +
            # state_schema_id fields the seeder is supposed to set. The
            # symptom: after a reinstall over an existing palace, agent +
            # intent_type lost their state-link properties even though
            # this seeder declared them. The fix is an explicit merge
            # write that always lands the seed properties without
            # clobbering anything else (e.g. summary_rewrite_count).
            # Idempotent on a fresh palace because the merge with empty
            # existing == identity. Bug confirmed by manual test
            # 2026-05-03 (state-protocol v1 audit pass).
            try:
                _conn = self._conn()
                _norm = self._entity_id(name)
                _existing_row = _conn.execute(
                    "SELECT properties FROM entities WHERE id=?",
                    (_norm,),
                ).fetchone()
                if _existing_row and _existing_row[0]:
                    try:
                        _existing_props = json.loads(_existing_row[0]) or {}
                    except Exception:
                        _existing_props = {}
                else:
                    _existing_props = {}
                _merged = dict(_existing_props)
                _merged.update(properties)  # seed wins on shared keys
                _conn.execute(
                    "UPDATE entities SET properties=? WHERE id=?",
                    (json.dumps(_merged), _norm),
                )
                _conn.commit()
            except Exception:
                pass  # best-effort; never break seed on this
            # Audit follow-up 2026-05-01: also sync to mempalace_entities
            # Chroma collection so the seed class is retrievable + has
            # full kg_query.details. Best-effort; falls through silently
            # if mcp_server isn't ready. Gated by sync_chroma so callers
            # with MEMPALACE_SKIP_SEED_CHROMA_SYNC=1 (e.g. test harnesses)
            # skip the per-entity ONNX embedding cost.
            if sync_chroma:
                self._sync_seed_entity_to_chroma(
                    entity_id=self._entity_id(name),
                    name=name,
                    content=content,
                    kind="class",
                    importance=imp,
                )
            if name != "thing":
                self.add_triple(name, "is_a", "thing")
            # Suppress unused-variable warning while keeping the return
            # value documented for future readers (eid is the normalized
            # id from add_entity; the chroma sync uses its own normalize
            # call so we don't depend on the return shape here).
            del eid

        # ── Predicates (kind=predicate) with constraints ──
        # Cold-start lock 2026-05-01 (Adrian's curation directive):
        # predicate summaries are hand-curated per entry. The `what`
        # field is the canonical hand-authored identity phrase from
        # ``_PREDICATE_WHATS`` below; the `why` is the existing curated
        # description string; the `scope` summarises the constraint
        # signature (cardinality + subject/object kinds). Each entry's
        # summary is a real {what, why, scope} dict that discriminates
        # the predicate from its peers at the gate's identity layer.
        predicates = [
            (
                "is_a",
                "Taxonomic classification: entity is_a class = instantiation, class is_a class = subtyping",
                5,
                {
                    "subject_kinds": ["entity", "class"],
                    "object_kinds": ["class", "entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "has_value",
                "Subject has a specific attribute value as object",
                4,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity", "literal"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "has_property",
                "Subject has a named property described by object",
                4,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity", "literal"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "defaults_to",
                "Subject has a default value of object",
                3,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity", "literal"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-one",
                },
            ),
            (
                "lives_at",
                "Subject is located at object (path, URL, address)",
                4,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity", "literal"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-one",
                },
            ),
            (
                "runs_in",
                "Subject operates as a process inside object runtime",
                4,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["system", "process"],
                    "object_classes": ["system", "environment"],
                    "cardinality": "many-to-one",
                },
            ),
            (
                "stored_in",
                "Subject data is persisted in object storage",
                3,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["system", "tool", "project"],
                    "object_classes": ["system", "tool", "environment"],
                    "cardinality": "many-to-one",
                },
            ),
            (
                "depends_on",
                "Subject requires object to function",
                4,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["system", "tool", "project", "process"],
                    "object_classes": ["system", "tool", "project", "process"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "requires",
                "Subject needs object as a runtime prerequisite",
                4,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["system", "tool", "project", "process"],
                    "object_classes": ["system", "tool", "project", "process"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "blocks",
                "Subject prevents object from proceeding",
                3,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "enables",
                "Subject unlocks object capability",
                3,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "must",
                "Subject is required to do/be object (positive rule)",
                4,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity", "literal"],
                    "subject_classes": [
                        "agent",
                        "system",
                        "tool",
                        "project",
                        "process",
                        "person",
                        "intent-type",
                    ],
                    "object_classes": ["rule"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "must_not",
                "Subject is forbidden from doing/being object (negative rule)",
                4,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity", "literal"],
                    "subject_classes": [
                        "agent",
                        "system",
                        "tool",
                        "project",
                        "process",
                        "person",
                        "intent-type",
                    ],
                    "object_classes": ["rule"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "forbids",
                "Subject prohibits object action",
                3,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity", "literal"],
                    "subject_classes": ["rule"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "has_gotcha",
                "Subject has a known pitfall described by object",
                4,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity", "literal"],
                    "subject_classes": ["system", "tool", "project", "process", "concept"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "warns_about",
                "Subject raises a caution about object",
                3,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity", "literal"],
                    "subject_classes": ["system", "tool", "project", "process", "concept"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "replaced_by",
                "Subject was superseded by object",
                3,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-one",
                },
            ),
            (
                "invalidated_by",
                "Subject was made obsolete by object event",
                3,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-one",
                },
            ),
            (
                "described_by",
                "Entity's canonical description lives in this memory",
                4,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "evidenced_by",
                "A rule or decision is backed by this memory's content",
                3,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "mentioned_in",
                "Entity is referenced in this memory but not its main topic",
                3,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "session_note_for",
                "A diary or session-log entry relevant to this entity",
                3,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "derived_from",
                "Entity was extracted or created from this memory's content",
                3,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "tested_by",
                "Subject is tested by the object test suite or entity",
                3,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["system", "tool", "project", "process"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "executed_by",
                "Intent execution was performed by this agent",
                4,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["agent"],
                    "cardinality": "many-to-one",
                },
            ),
            (
                "targeted",
                "Intent execution was performed on this entity (slot target)",
                4,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "resulted_in",
                "Intent execution produced this outcome memory",
                4,
                {
                    "subject_kinds": ["entity"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "surfaced",
                "Retrieval-event edge: a context surfaced this entity to the agent during search; consumed by finalize coverage and Channel D",
                4,
                {
                    "subject_kinds": ["context"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "rated_useful",
                "Positive feedback edge: the agent rated this surfaced entity as useful at finalize_intent; consumed by Channel D and Rocchio enrichment",
                4,
                {
                    "subject_kinds": ["context"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "rated_irrelevant",
                "Negative feedback edge: the agent rated this surfaced entity as not relevant at finalize_intent; Channel D demotes similar future contexts",
                3,
                {
                    "subject_kinds": ["context"],
                    "object_kinds": ["entity"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            (
                "created_under",
                "Provenance edge: a memory / entity / triple was written while this Context was active; consumed by Channel D and finalize coverage",
                4,
                {
                    "subject_kinds": ["entity", "class", "predicate", "literal", "record"],
                    "object_kinds": ["context"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-one",
                },
            ),
            (
                "similar_to",
                "Context-to-context similarity edge written when MaxSim falls in [T_similar, T_reuse); used for 1-2-hop expansion in Channel D",
                3,
                {
                    "subject_kinds": ["context"],
                    "object_kinds": ["context"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
            # Channel-B anchor edge: context -> each entity declared in
            # context.entities. Pure graph topology so BFS can find the
            # context by walking from the entities, without polluting the
            # context's cosine view set (the auto-append-into-views
            # antipattern, retired 2026-05-02 -- see
            # record_ga_agent_channel_violation_saturation).
            (
                "anchored_by",
                "Context-to-entity graph anchor edge; written for every entity in context.entities so Channel B BFS reaches the context from its anchors",
                4,
                {
                    "subject_kinds": ["context"],
                    "object_kinds": ["entity", "class", "predicate", "literal"],
                    "subject_classes": ["thing"],
                    "object_classes": ["thing"],
                    "cardinality": "many-to-many",
                },
            ),
        ]
        for name, desc, imp, constraints in predicates:
            # Cold-start lock 2026-05-01: hand-curated `what` (from
            # _PREDICATE_WHATS), existing curated `desc` as `why`,
            # constraint-derived `scope`. No template, no derivation.
            _seed_summary = _seed_predicate_summary(name, desc, constraints)
            self.add_entity(
                name,
                kind="predicate",
                content=desc,
                importance=imp,
                properties={"constraints": constraints, "summary": _seed_summary},
            )
            # Audit follow-up 2026-05-01: predicates also need a row in
            # mempalace_entities so kg_query.details + retrieval work.
            # Gated by sync_chroma -- see classes-loop comment above.
            if sync_chroma:
                self._sync_seed_entity_to_chroma(
                    entity_id=self._entity_id(name),
                    name=name,
                    content=desc,
                    kind="predicate",
                    importance=imp,
                )

        # ── Intent types (kind=class, is-a intent_type) ──
        intent_types = [
            # (name, description, importance, parent, slots, tool_permissions_or_None)
            (
                "inspect",
                "Intent type for read-only observation",
                4,
                "intent_type",
                {
                    "subject": {"classes": ["thing"], "required": True, "multiple": True},
                    "paths": {"raw": True, "required": True, "multiple": True},
                },
                [
                    {"tool": "Read", "scope": "{paths}"},
                    {"tool": "Grep", "scope": "{paths}"},
                    {"tool": "Glob", "scope": "{paths}"},
                ],
            ),
            (
                "modify",
                "Intent type for changing files",
                4,
                "intent_type",
                {
                    "files": {"classes": ["file"], "required": True, "multiple": True},
                    "paths": {"raw": True, "required": True, "multiple": True},
                },
                [
                    {"tool": "Edit", "scope": "{files}"},
                    {"tool": "Write", "scope": "{files}"},
                    {"tool": "Read", "scope": "{paths}"},
                    {"tool": "Grep", "scope": "{paths}"},
                    {"tool": "Glob", "scope": "{paths}"},
                ],
            ),
            (
                "execute",
                "Intent type for running commands and scripts",
                4,
                "intent_type",
                {
                    "target": {"classes": ["thing"], "required": True, "multiple": True},
                    "commands": {"raw": True, "required": True, "multiple": True},
                    "paths": {"raw": True, "required": True, "multiple": True},
                },
                [
                    {"tool": "Read", "scope": "{paths}"},
                    {"tool": "Grep", "scope": "{paths}"},
                    {"tool": "Glob", "scope": "{paths}"},
                    {"tool": "Bash", "scope": "{commands}"},
                ],
            ),
            (
                "communicate",
                "Intent type for external communication -- sending messages, creating issues, pushing to services, fetching web content",
                4,
                "intent_type",
                {
                    "target": {"classes": ["thing"], "required": True, "multiple": True},
                    "audience": {
                        "classes": ["person", "agent"],
                        "required": False,
                        "multiple": True,
                    },
                    "paths": {"raw": True, "required": True, "multiple": True},
                },
                [
                    {"tool": "Read", "scope": "{paths}"},
                    {"tool": "Grep", "scope": "{paths}"},
                    {"tool": "Glob", "scope": "{paths}"},
                    {"tool": "Bash", "scope": "{target}"},
                    {"tool": "WebFetch", "scope": "*"},
                    {"tool": "WebSearch", "scope": "*"},
                ],
            ),
            (
                "research",
                "Intent type for researching external documentation, APIs, and web resources -- read-only web access plus local code reading",
                4,
                "inspect",
                {
                    "subject": {"classes": ["thing"], "required": True, "multiple": True},
                    "paths": {"raw": True, "required": False, "multiple": True},
                },
                [
                    {"tool": "Read", "scope": "*"},
                    {"tool": "Grep", "scope": "*"},
                    {"tool": "Glob", "scope": "*"},
                    {"tool": "WebFetch", "scope": "*"},
                    {"tool": "WebSearch", "scope": "*"},
                ],
            ),
            # wrap_up_session: mandatory proof-of-done intent for the
            # never-stop rule. The Stop hook requires the LAST finalized
            # intent to be wrap_up_session(success) before it lets the
            # session stop. Must be seeded on every fresh palace or the
            # never-stop rule would wedge every install -- no way to stop.
            (
                "wrap_up_session",
                "Proof-of-done intent: agent runs >=2 kg_search passes against pending-work patterns and persists session delta so the Stop hook admits clean exit",
                4,
                "inspect",
                {
                    "subject": {"classes": ["thing"], "required": True, "multiple": False},
                    "paths": {"raw": True, "required": True, "multiple": True},
                },
                None,  # inherits inspect's tool_permissions
            ),
            # Only generic top-level types seeded here.
            # Domain-specific children (edit_file, deploy, etc.) are declared
            # by agents via kg_declare_entity -- not hardcoded in the seeder.
        ]
        for name, desc, imp, parent, slots, perms in intent_types:
            props = {"rules_profile": {"slots": slots}}
            if perms is not None:
                props["rules_profile"]["tool_permissions"] = perms
            # Cold-start lock 2026-05-01: hand-curated summary via
            # _seed_intent_type_summary -- explicit `what` from the
            # _INTENT_TYPE_WHATS lookup, existing curated desc as `why`.
            props["summary"] = _seed_intent_type_summary(name, desc, parent)
            self.add_entity(name, kind="class", content=desc, importance=imp, properties=props)
            # Audit follow-up 2026-05-01: intent_type entities also need
            # the Chroma row so declare_intent retrieval finds the type.
            # Gated by sync_chroma -- see classes-loop comment above.
            if sync_chroma:
                self._sync_seed_entity_to_chroma(
                    entity_id=self._entity_id(name),
                    name=name,
                    content=desc,
                    kind="class",
                    importance=imp,
                )
            self.add_triple(name, "is_a", parent)

    # Retired edge-feedback API (record_edge_feedback, get_edge_usefulness,
    # get_recent_rejection_reasons, get_context_ids_for_edge) deleted in
    # the cold-start cleanup -- there's no legacy data to shim for.
    # Signal now flows through context --rated_useful/rated_irrelevant-->
    # memory edges written at finalize_intent.

    def get_past_conflict_resolution(
        self,
        existing_id: str,
        new_id: str,
        conflict_type: str,
    ):
        """Return the most recent past resolution for a (existing_id, new_id,
        conflict_type) triple, or None if no row exists.

        B1b: surfaces past decisions as a hint on newly-detected conflicts so
        agents don't re-derive reasoning they already captured. Matches by
        normalized entity ids on both sides plus the conflict_type (so a
        past `edge_contradiction` decision doesn't apply to a new
        `memory_duplicate` between the same ids). Ordered by created_at DESC
        so we return the freshest decision.
        """
        if not (existing_id and new_id and conflict_type):
            return None
        conn = self._conn()
        try:
            ex = self._entity_id(existing_id)
            nw = self._entity_id(new_id)
        except Exception:
            ex, nw = existing_id, new_id
        try:
            row = conn.execute(
                """SELECT action, reason, agent, intent_type, created_at
                   FROM conflict_resolutions
                   WHERE existing_id = ? AND new_id = ? AND conflict_type = ?
                   ORDER BY created_at DESC
                   LIMIT 1""",
                (ex, nw, conflict_type),
            ).fetchone()
        except Exception:
            return None
        if not row:
            return None
        return {
            "action": row[0],
            "reason": row[1],
            "agent": row[2] or "",
            "intent_type": row[3] or "",
            "when": row[4] or "",
        }

    def record_conflict_resolution(
        self,
        conflict_id: str,
        conflict_type: str,
        action: str,
        reason: str,
        existing_id: str = "",
        new_id: str = "",
        agent: str = "",
        intent_type: str = "",
        context_id: str = "",
    ):
        """Persist the agent's resolution of a conflict.

        Captures invalidate/merge/keep/skip decisions plus the mandatory
        reason, so future audits and feedback loops can learn from past
        choices instead of losing the reasoning.
        """
        conn = self._conn()
        now = datetime.now().isoformat()
        with conn:
            conn.execute(
                """INSERT INTO conflict_resolutions
                   (conflict_id, conflict_type, action, reason,
                    existing_id, new_id, agent, intent_type,
                    context_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    conflict_id,
                    conflict_type,
                    action,
                    reason,
                    existing_id,
                    new_id,
                    agent,
                    intent_type,
                    context_id,
                    now,
                ),
            )

    # ── Caller-provided keywords (stored, not auto-extracted) ──
    def add_entity_keywords(self, entity_id, keywords, source="caller"):
        """Persist caller-provided keywords for an entity.

        Replaces any existing rows with the same (entity_id, keyword) -- idempotent.
        Used by kg_declare_entity (and friends) to store the Context.keywords list
        so the keyword channel can look entities up by literal term match without
        ever having to auto-extract from descriptions.
        """
        if not entity_id or not keywords:
            return 0
        cleaned = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
        if not cleaned:
            return 0
        conn = self._conn()
        rows = [(entity_id, k, source) for k in cleaned]
        conn.executemany(
            "INSERT OR REPLACE INTO entity_keywords (entity_id, keyword, source) VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()
        return len(rows)

    def get_entity_keywords(self, entity_id):
        """Return caller-provided keywords (lowercased str list) for an entity."""
        if not entity_id:
            return []
        conn = self._conn()
        rows = conn.execute(
            "SELECT keyword FROM entity_keywords WHERE entity_id=? ORDER BY added_at",
            (entity_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def entity_ids_for_keyword(self, keyword, limit=50):
        """Return entity_ids whose caller-provided keywords contain `keyword`.

        Case-insensitive exact match. Used by the keyword channel to
        surface entities by literal term hit -- fast, indexed, no $contains scan.
        """
        if not keyword or not keyword.strip():
            return []
        conn = self._conn()
        rows = conn.execute(
            "SELECT DISTINCT entity_id FROM entity_keywords WHERE keyword=? LIMIT ?",
            (keyword.strip().lower(), limit),
        ).fetchall()
        return [r[0] for r in rows]

    def is_a_parents(self, entity_id):
        """v3.9.6: return the current is_a parent objects of `entity_id`.

        One indexed query against triples (subject + predicate filter);
        only edges still valid (valid_to IS NULL) are returned. Used by
        the class-signature renderer to walk the is_a ancestor chain for
        the '(kind) a -> b' surface label. Does NOT recurse -- the caller
        walks transitively so it can bound depth + dedup across branches.
        """
        if not entity_id:
            return []
        eid = self._entity_id(entity_id)
        conn = self._conn()
        rows = conn.execute(
            "SELECT object FROM triples WHERE subject=? AND predicate='is_a' AND valid_to IS NULL",
            (eid,),
        ).fetchall()
        return [r[0] for r in rows]

    def set_entity_creation_context(self, entity_id, context_id):
        """Record the Context.id under which an entity was created.

        The actual view vectors live in the mempalace_feedback_contexts Chroma
        collection (set by store_feedback_context). This column points at it
        so MaxSim can later weight feedback transfer by context similarity.
        """
        if not entity_id or not context_id:
            return False
        conn = self._conn()
        conn.execute(
            "UPDATE entities SET creation_context_id=? WHERE id=?",
            (context_id, entity_id),
        )
        conn.commit()
        return True

    def get_entity_creation_context(self, entity_id):
        if not entity_id:
            return ""
        conn = self._conn()
        row = conn.execute(
            "SELECT creation_context_id FROM entities WHERE id=?",
            (entity_id,),
        ).fetchone()
        return (row[0] if row else "") or ""

    # Retired keyword-suppression API (record_keyword_suppression,
    # get_keyword_suppression, reset_keyword_suppression) deleted in
    # the cold-start cleanup. BM25-IDF on keyword_idf replaces the
    # channel-level dominance signal.

    # P3: weight self-tune is RE-ENABLED. P2 cutover retired W_REL so the
    # scoring_weight_feedback table was truncated in migration 015 -- the
    # learner now correlates against the four post-prune components
    # (sim, imp, decay, agent). Global weights (not per-context); see
    # docs/context_as_entity_redesign_plan.md -- personal-scale palaces
    # are too sparse for LinUCB-style per-context bandits (Li et al.
    # 2010 arXiv:1003.0146; they need hundreds of observations per
    # context to converge).
    _A6_WEIGHT_SELFTUNE_ENABLED = True

    # TODO (learning parameters):
    #   - LEARN_DAMPING = 0.30 (the ±30% cap inside compute_learned_weights)
    #     is a meta-parameter. Tuning it requires double-learning (learn
    #     the rate of learning); rabbit hole. Hand-set forever.
    #   - min_samples = 10 default gates the first adjustment. At personal
    #     palace scale this is ~1 week of active use. Dropping to 5 would
    #     let the learner bite earlier at the cost of noisier early moves.
    #     Not worth learning -- empirical call.

    def record_scoring_feedback(self, components: dict, was_useful: bool, *, scope: str = "hybrid"):
        """Record scoring component values alongside relevance outcome.

        Two scopes:
          - scope='hybrid' (default): hybrid_score's per-memory weights
            (sim, rel, imp, decay, agent). Each row stored with component
            in that namespace.
          - scope='channel': per-channel RRF weights (cosine, graph,
            keyword, context). Components land with a ``ch_`` prefix
            so the row space stays disjoint from hybrid and
            ``compute_learned_weights(base, scope='channel')`` can
            filter by prefix.

        DISABLED by ``_A6_WEIGHT_SELFTUNE_ENABLED`` -- currently a no-op
        when False. Keeping the body so flipping the flag re-enables
        data collection without touching the callers.
        """
        if not self._A6_WEIGHT_SELFTUNE_ENABLED:
            return
        conn = self._conn()
        now = datetime.now().isoformat()
        prefix = "ch_" if scope == "channel" else ""
        with conn:
            for comp, value in components.items():
                stored_name = f"{prefix}{comp}" if not comp.startswith(prefix) else comp
                conn.execute(
                    """INSERT INTO scoring_weight_feedback
                       (component, component_value, was_useful, created_at)
                       VALUES (?, ?, ?, ?)""",
                    (stored_name, float(value), was_useful, now),
                )

    def compute_learned_weights(
        self, base_weights: dict, min_samples: int = 10, *, scope: str = "hybrid"
    ):
        """Compute adjusted weights from feedback correlation.

        Works for either scope:
          - scope='hybrid': hybrid_score's per-memory weights (sim / rel
            / imp / decay / agent). Component names match base_weights
            keys exactly.
          - scope='channel': per-channel RRF weights (cosine / graph /
            keyword / context). Rows were stored with a ``ch_`` prefix
            by record_scoring_feedback; this method queries accordingly.

        Returns adjusted weights (same keys as base_weights), renormalised
        to sum to 1.0. Returns base_weights unchanged if insufficient
        feedback data or the self-tune flag is False.
        """
        if not self._A6_WEIGHT_SELFTUNE_ENABLED:
            return dict(base_weights)
        conn = self._conn()
        prefix = "ch_" if scope == "channel" else ""

        # Count rows in the relevant scope only.
        total = conn.execute(
            "SELECT COUNT(*) FROM scoring_weight_feedback WHERE component LIKE ?",
            (f"{prefix}%" if prefix else "%",),
        ).fetchone()[0]
        if total < min_samples:
            return dict(base_weights)

        adjustments = {}
        for comp in base_weights:
            stored_name = f"{prefix}{comp}" if prefix and not comp.startswith(prefix) else comp
            rows = conn.execute(
                """SELECT was_useful, AVG(component_value), COUNT(*)
                   FROM scoring_weight_feedback
                   WHERE component=?
                   GROUP BY was_useful""",
                (stored_name,),
            ).fetchall()
            avg_useful = 0.5
            avg_irrelevant = 0.5
            for row in rows:
                if row[0]:
                    avg_useful = row[1]
                else:
                    avg_irrelevant = row[1]
            correlation = avg_useful - avg_irrelevant
            adjustments[comp] = 1.0 + 0.3 * max(-1.0, min(1.0, correlation))

        adjusted = {}
        for comp, base_w in base_weights.items():
            adjusted[comp] = base_w * adjustments.get(comp, 1.0)
        total_w = sum(adjusted.values())
        if total_w > 0:
            for comp in adjusted:
                adjusted[comp] /= total_w
        return adjusted

    def close(self):
        """Close the database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _entity_id(self, name: str) -> str:
        """Normalize an entity name to a canonical ID.

        Uses aggressive normalization (hyphens, underscores, CamelCase, articles
        all collapsed). Also checks the alias table for merged entities.
        """
        normalized = normalize_entity_name(name)
        # Check if this normalized name is an alias for a merged entity
        conn = self._conn()
        alias_row = conn.execute(
            "SELECT canonical_id FROM entity_aliases WHERE alias = ?", (normalized,)
        ).fetchone()
        if alias_row:
            return alias_row["canonical_id"]
        return normalized

    def _touch_entity(self, entity_id: str):
        """Update last_touched timestamp on an entity."""
        conn = self._conn()
        now = datetime.now().isoformat()
        with conn:
            conn.execute("UPDATE entities SET last_touched = ? WHERE id = ?", (now, entity_id))

    def touch_entities(self, entity_ids) -> int:
        """Batch update last_touched on a list of entities (decay reset).

        Adrian directive 2026-05-04: the InjectionGate is the canonical
        "this entity is being used" signal. Bumping last_touched at
        retrieval time would refresh the decay clock for entities the
        gate later filters as irrelevant -- noise. Bumping it AFTER
        the gate filters means only entities the LLM actually consumes
        get their decay reset. apply_gate calls this on the kept_ids
        set so decay tracks utility, not raw retrieval traffic.

        Single SQL UPDATE with IN(...) keeps this O(1) round-trip
        regardless of K. Returns the number of rows updated (some
        ids may not resolve to entities table rows -- e.g. memory ids
        that live only in Chroma; those are silently skipped).
        """
        if not entity_ids:
            return 0
        ids = [self._entity_id(eid) for eid in entity_ids if eid]
        if not ids:
            return 0
        conn = self._conn()
        now = datetime.now().isoformat()
        placeholders = ",".join("?" * len(ids))
        with conn:
            cur = conn.execute(
                f"UPDATE entities SET last_touched = ? WHERE id IN ({placeholders})",
                (now, *ids),
            )
            return cur.rowcount or 0

    def soft_delete_entity(self, name: str):
        """Soft-delete an entity (set status='deleted'). Also invalidates all its edges."""
        eid = self._entity_id(name)
        conn = self._conn()
        ended = date.today().isoformat()
        with conn:
            conn.execute("UPDATE entities SET status='deleted' WHERE id=?", (eid,))
            conn.execute(
                "UPDATE triples SET valid_to=? WHERE (subject=? OR object=?) AND valid_to IS NULL",
                (ended, eid, eid),
            )
        return eid

    # ── Write operations ──────────────────────────────────────────────────

    def add_entity(
        self,
        name: str,
        properties: dict = None,
        content: str = "",
        importance: int = 3,
        kind: str = "entity",
        session_id: str = None,
        intent_id: str = None,
    ):
        """Add or update an entity node.

        Args:
            kind: ontological role -- 'entity' (concrete thing), 'predicate' (relationship type),
                  'class' (category/type), 'literal' (raw value). Fixed enum.
            content: precise text describing this entity. (Renamed from
                  ``description`` 2026-04-29; migration 023 dropped the legacy
                  column.)
            importance: 1-5 scale for decay-aware ranking.
            session_id: P6.7a provenance -- auto-injected by callers, stored for session-scoped queries.
            intent_id: P6.7a provenance -- auto-injected by callers, stored for intent-scoped queries.
        """
        eid = self._entity_id(name)
        # Adrian's design lock 2026-04-27: entities.name (the raw display
        # column) is ASCII-only metadata, same as the id family. Fold the
        # raw caller-supplied name before binding it to the INSERT so the
        # display label matches the id ("Café" -> "Cafe", not "Cafe" id +
        # "Café" name drift). entities.description (long-form content)
        # stays UTF-8 verbatim and is intentionally NOT folded.
        from .ascii_fold import fold_ascii

        display_name = fold_ascii(name) if isinstance(name, str) else name
        props = json.dumps(properties or {})
        now = datetime.now().isoformat()
        conn = self._conn()
        with conn:
            # provenance columns (session_id, intent_id) are added
            # by migration 009. Use a try/except fallback for pre-migration
            # DBs where the columns don't exist yet.
            try:
                conn.execute(
                    """INSERT INTO entities (id, name, type, kind, properties, content,
                                            importance, last_touched, status, session_id, intent_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                       ON CONFLICT(id) DO UPDATE SET
                           name = excluded.name,
                           type = excluded.type,
                           kind = excluded.kind,
                           properties = excluded.properties,
                           content = CASE WHEN excluded.content != '' THEN excluded.content ELSE entities.content END,
                           importance = CASE WHEN excluded.importance != 3 THEN excluded.importance ELSE entities.importance END,
                           last_touched = excluded.last_touched,
                           status = 'active',
                           merged_into = NULL,
                           session_id = COALESCE(excluded.session_id, entities.session_id),
                           intent_id = COALESCE(excluded.intent_id, entities.intent_id)
                    """,
                    (
                        eid,
                        display_name,
                        kind,
                        kind,
                        props,
                        content,
                        importance,
                        now,
                        session_id or "",
                        intent_id or "",
                    ),
                )
            except Exception:
                # Pre-migration fallback (columns don't exist yet) for palaces
                # that pre-date migration 009 (session_id/intent_id) AND have
                # already had migration 022 run (so content column exists).
                conn.execute(
                    """INSERT INTO entities (id, name, type, kind, properties, content, importance, last_touched, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active')
                       ON CONFLICT(id) DO UPDATE SET
                           name = excluded.name,
                           type = excluded.type,
                           kind = excluded.kind,
                           properties = excluded.properties,
                           content = CASE WHEN excluded.content != '' THEN excluded.content ELSE entities.content END,
                           importance = CASE WHEN excluded.importance != 3 THEN excluded.importance ELSE entities.importance END,
                           last_touched = excluded.last_touched,
                           status = 'active',
                           merged_into = NULL
                    """,
                    (eid, display_name, kind, kind, props, content, importance, now),
                )
        return eid

    def merge_entities(self, source_name: str, target_name: str, summary: str = None):
        """Merge source entity into target. All edges rewritten. Source becomes alias.

        `summary` is the already-rendered prose form (the dict-to-prose
        coercion happens at the tool-handler edge, not here).

        Returns dict with counts of edges_moved, aliases_created.
        """
        source_id = normalize_entity_name(source_name)
        target_id = self._entity_id(target_name)  # resolves aliases
        if source_id == target_id:
            return {"error": "source and target resolve to the same entity"}

        conn = self._conn()
        with conn:
            # Rewrite triples: subject
            r1 = conn.execute(
                "UPDATE triples SET subject = ? WHERE subject = ?", (target_id, source_id)
            )
            # Rewrite triples: object
            r2 = conn.execute(
                "UPDATE triples SET object = ? WHERE object = ?", (target_id, source_id)
            )
            edges_moved = r1.rowcount + r2.rowcount

            # Register alias
            now = datetime.now().isoformat()
            conn.execute(
                "INSERT OR REPLACE INTO entity_aliases (alias, canonical_id, merged_at) VALUES (?, ?, ?)",
                (source_id, target_id, now),
            )

            # Soft-delete source
            conn.execute(
                "UPDATE entities SET status = 'merged', merged_into = ? WHERE id = ?",
                (target_id, source_id),
            )

            # lifecycle hardening (Adrian 2026-05-03): cascade
            # state revisions from source to target. Without this,
            # source's state history orphans (entity_id=source_id) while
            # latest_state_for_entity(source) -- which calls
            # _entity_id(), now alias-resolving to target -- returns
            # None because no rows match target_id. The merged entity
            # appears stateless. Rewriting the entity_id column on
            # state revisions preserves history under the canonical
            # (target) id so reads continue to work.
            conn.execute(
                "UPDATE mempalace_state_revisions SET entity_id = ? WHERE entity_id = ?",
                (target_id, source_id),
            )

            # Update target content if provided (already rendered prose).
            if summary:
                conn.execute(
                    "UPDATE entities SET content = ?, last_touched = ? WHERE id = ?",
                    (summary, now, target_id),
                )

            # Touch target
            conn.execute("UPDATE entities SET last_touched = ? WHERE id = ?", (now, target_id))

        # v3.7.32 (FINDING #R fix 2026-05-18): post-merge content
        # rewrite must refresh the target's Level-3 body vector so
        # the merged-prose content is reflected in cosine retrieval.
        # Outside the with-conn block: the SQLite write is already
        # committed; vec store writes are non-transactional anyway.
        # Best-effort: a failed refresh leaves the prior __body in
        # place (slightly stale) but doesn't break the merge itself.
        if summary:
            try:
                self._refresh_body_view(target_id)
            except Exception:
                pass

        return {
            "source": source_id,
            "target": target_id,
            "edges_moved": edges_moved,
            "aliases_created": 1,
        }

    def list_unverbalized_triples(self, limit: int = None) -> list:
        """Return SQL rows for semantic triples missing a ``statement``.

        Each row is (id, subject, predicate, object, confidence) -- the
        raw material a human (or curation tool) needs to write a proper
        natural-language sentence for. Skip-list predicates are omitted
        because they're never embedded anyway.

        NO auto-generation happens anywhere. The previous
        ``backfill_triple_statements`` that fabricated statements from
        underscore-to-space substitution was retired 2026-04-19 -- see
        the TripleStatementRequired policy in add_triple. Legacy rows
        with ``statement IS NULL`` simply remain NULL and absent from
        the mempalace_triples Chroma collection; they're still walkable
        via BFS and queryable by exact id, just not similarity-searched
        until someone writes a real statement via kg_update_triple or
        equivalent curation.
        """
        conn = self._conn()
        skip_clause = ",".join("?" for _ in _TRIPLE_SKIP_PREDICATES)
        rows = conn.execute(
            f"""SELECT id, subject, predicate, object, confidence
               FROM triples
               WHERE statement IS NULL
                 AND predicate NOT IN ({skip_clause})
               ORDER BY id
               LIMIT ?""",
            (*sorted(_TRIPLE_SKIP_PREDICATES), int(limit) if limit else 1_000_000),
        ).fetchall()
        return [
            {
                "triple_id": r["id"],
                "subject": r["subject"],
                "predicate": r["predicate"],
                "object": r["object"],
                "confidence": r["confidence"] if r["confidence"] is not None else 1.0,
            }
            for r in rows
        ]

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        valid_from: str = None,
        valid_to: str = None,
        confidence: float = 1.0,
        source_file: str = None,
        creation_context_id: str = "",
        statement: str = None,
        properties: dict = None,
    ):
        """
        Add a relationship triple: subject → predicate → object.

        Examples:
            add_triple("Max", "child_of", "Alice", valid_from="2015-04-01")
            add_triple("Max", "does", "swimming", valid_from="2025-01-01")
            add_triple("Alice", "worried_about", "Max injury", valid_from="2026-01", valid_to="2026-02")

        `statement` is the natural-language verbalization of the triple
        ("Max is a child of Alice"). Stored on the row and embedded into
        the mempalace_triples Chroma collection so the triple becomes a
        first-class search target.

        REQUIRED for every predicate OUTSIDE the skip list
        (``_TRIPLE_SKIP_PREDICATES``). For skip-list predicates (``is_a``,
        ``described_by``, ``executed_by``, ``targeted``, …) the statement
        is allowed to be None because those are schema glue that never
        gets embedded regardless -- they're walkable via BFS, not searched
        by similarity.

        Rationale (2026-04-19 policy change): we used to fall back to a
        naive "replace underscores with spaces" verbalization when callers
        omitted ``statement``. That produced retrieval-poisoning text like
        ``"record ga agent a relates to record ga agent b"``. Callers now
        write a real sentence or the edge is rejected.
        """
        sub_id = self._entity_id(subject)
        obj_id = self._entity_id(obj)
        pred = _normalize_predicate(predicate)
        # Adrian's design lock 2026-04-27: triple statements are metadata
        # (verbalized edges), not long-form content; fold to ASCII via
        # anyascii at the write boundary so both the SQL row and the
        # mempalace_triples Chroma document/embedding are canonical
        # ASCII. Applies to every predicate (skip-list included) so that
        # any caller emitting a verbalization passes through one gate.
        # The dict-form ``{what, why, scope}`` statement path goes
        # through ``validate_statement`` -> ``coerce_summary_for_persist``
        # which already folds; this branch covers the plain-prose form
        # the operation-promotion auto-rater (intent.py) emits.
        if isinstance(statement, str) and statement:
            from .ascii_fold import fold_ascii

            statement = fold_ascii(statement.strip())
        # Require a caller-provided statement for non-skip predicates.
        # Skip predicates stay optional -- they're never embedded anyway.
        if pred not in _TRIPLE_SKIP_PREDICATES:
            if not statement or not statement.strip():
                raise TripleStatementRequired(
                    f"add_triple({subject!r}, {pred!r}, {obj!r}): predicate "
                    f"{pred!r} requires a caller-provided `statement` -- a "
                    f"natural-language verbalization of the fact. "
                    f"Structural predicates (is_a, described_by, "
                    f"executed_by, targeted, has_value, "
                    f"session_note_for, derived_from, mentioned_in, "
                    f"found_useful, found_irrelevant, evidenced_by) may "
                    f"omit `statement`; every other predicate must supply "
                    f"one. Autogeneration was retired 2026-04-19 because "
                    f"naive fallbacks produced low-signal text that "
                    f"poisoned retrieval."
                )
            statement = statement.strip()

        # Hard-reject phantom references (cold-start lock 2026-05-01).
        # Pre-cold-start, the lines below silently INSERT OR IGNORE
        # missing endpoints, creating phantom entities with no kind, no
        # summary, no is_a edge -- the root cause of the 1,780 untyped
        # entities counted in the live corpus on 2026-05-01. Both
        # endpoints must exist before an edge can be written; declare via
        # mempalace_kg_declare_entity (which routes through entity_gate.
        # mint_entity) so summary + identity-collision checks run.
        from .entity_gate import assert_entity_exists

        conn = self._conn()
        with conn:
            assert_entity_exists(sub_id, conn)
            assert_entity_exists(obj_id, conn)

            # Check for existing identical triple
            existing = conn.execute(
                "SELECT id FROM triples WHERE subject=? AND predicate=? AND object=? AND valid_to IS NULL",
                (sub_id, pred, obj_id),
            ).fetchone()

            if existing:
                return existing["id"]  # Already exists and still valid

            triple_id = f"t_{sub_id}_{pred}_{obj_id}_{hashlib.sha256(f'{valid_from}{datetime.now().isoformat()}'.encode()).hexdigest()[:12]}"

            props_json = json.dumps(properties or {})
            conn.execute(
                """INSERT INTO triples (id, subject, predicate, object, valid_from, valid_to,
                                        confidence, source_file, creation_context_id, statement,
                                        properties)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    triple_id,
                    sub_id,
                    pred,
                    obj_id,
                    valid_from,
                    valid_to,
                    confidence,
                    source_file,
                    creation_context_id or "",
                    statement,
                    props_json,
                ),
            )
        # Touch both entities (update last_touched for decay scoring)
        self._touch_entity(sub_id)
        self._touch_entity(obj_id)
        # Embed the verbalization so kg_search and multi_channel_search can
        # surface this triple as a first-class result. Best-effort: any
        # Chroma write failure is non-fatal (the SQL row is the source of
        # truth and the backfill helper can re-embed later).
        _index_triple_statement(self, triple_id, sub_id, pred, obj_id, statement, confidence)
        return triple_id

    def invalidate(self, subject: str, predicate: str, obj: str, ended: str = None):
        """Mark a relationship as no longer valid (set valid_to date)."""
        sub_id = self._entity_id(subject)
        obj_id = self._entity_id(obj)
        pred = _normalize_predicate(predicate)
        ended = ended or date.today().isoformat()

        conn = self._conn()
        with conn:
            conn.execute(
                "UPDATE triples SET valid_to=? WHERE subject=? AND predicate=? AND object=? AND valid_to IS NULL",
                (ended, sub_id, pred, obj_id),
            )

    # State-protocol v1 (Adrian Option B 2026-05-03): durable substrate
    # for state revisions. Migration 024 created the mempalace_state_revisions
    # table; these helpers are the single canonical write/read entry point.
    # Slice B's per-op delta coverage rule + piece 6's retrofit gardener
    # handler both call record_state_revision; the projection materializer
    # in state_projection.py (future) reads via latest_state_for_entity.

    def record_state_revision(
        self,
        entity_id: str,
        schema_id: str,
        payload: dict,
        op_context_id: str = "",
        agent: str = "",
        session_id: str | None = None,
    ) -> str:
        """Insert a state revision row + state_changed_by edge.

        Returns the rev_id. Caller is responsible for validating payload
        against state_schemas.STATE_SCHEMAS[schema_id].json_schema before
        calling -- this helper persists, it does not validate. The
        state_changed_by JTMS edge (Doyle 1979) is written only when
        op_context_id is non-empty; gardener retrofit-default writes
        leave op_context_id empty so no spurious edge lands.

        lifecycle hardening (Adrian 2026-05-03): refuses to
        write a revision when the entity row is missing or soft-deleted
        (status='deleted'). Without the check, a typo'd entity_id or a
        state_deltas write against a since-deleted entity would mint a
        'phantom state' row -- a revision whose entity_id has no
        corresponding entities table entry, leaving downstream readers
        and gardeners with dangling references. Tests that need to
        write revisions against ad-hoc entity_ids must first call the
        slice_b _ensure_entity helper to seed an entities row.
        """
        import json as _json

        eid = self._entity_id(entity_id)
        conn = self._conn()
        # Existence + status check before write. Refuse phantom + deleted.
        row = conn.execute("SELECT status FROM entities WHERE id = ?", (eid,)).fetchone()
        if row is None:
            raise ValueError(
                f"record_state_revision: entity '{entity_id}' "
                f"(resolved to id '{eid}') not found in entities table; "
                "phantom state writes are blocked. Declare the entity "
                "via mempalace_kg_declare_entity first."
            )
        if (row[0] or "") == "deleted":
            raise ValueError(
                f"record_state_revision: entity '{entity_id}' is "
                "soft-deleted (status='deleted'); cannot write state "
                "revision. Resurrect the entity via kg_declare_entity "
                "or write state on its replacement instead."
            )
        # schema validation hardening (Adrian corner-case
        # audit 2026-05-03). Two checks:
        #   1. schema_id must be a known STATE_SCHEMAS key (or empty
        #      for gardener-default writes / pre-Slice-C2 callers).
        #   2. payload must validate against the schema's json_schema
        #      via jsonschema.validate. Without these, agents could
        #      mint state revisions naming unknown schemas or carrying
        #      malformed payloads, and downstream readers (the
        #      projection materializer + per-memory state surfacing
        #      in _enrich_memories_with_state) would silently return
        #      shapes that don't match the schema agents author
        #      patches against. Empty schema_id stays allowed -- it
        #      signals "no schema known" (gardener-default writes,
        #      extend_feedback recovery deltas without explicit
        #      schema). jsonschema is an optional dep; skip silently
        #      if unavailable rather than block the write.
        if schema_id:
            from mempalace import state_schemas as _schemas

            if schema_id not in _schemas.STATE_SCHEMAS:
                raise ValueError(
                    f"record_state_revision: schema_id '{schema_id}' "
                    f"is not a known STATE_SCHEMAS key. Known: "
                    f"{sorted(_schemas.STATE_SCHEMAS.keys())}."
                )
            try:
                import jsonschema as _js
            except ImportError:
                _js = None
            if _js is not None:
                try:
                    _js.validate(
                        payload,
                        _schemas.STATE_SCHEMAS[schema_id]["json_schema"],
                    )
                except _js.ValidationError as _verr:
                    raise ValueError(
                        f"record_state_revision: payload failed schema "
                        f"'{schema_id}' validation: {_verr.message}"
                    ) from _verr
        # Phase 6 lazy-migration-at-injection (Adrian 2026-05-03): stamp
        # the current schema_version on every new revision. The
        # injection-gate hook later compares this column against
        # state_schemas.current_version(schema_id) to decide whether to
        # run the migration chain. Empty schema_id (gardener-default
        # writes) gets version=1 -- they fall outside the migration
        # path and stay at the floor.
        if schema_id:
            try:
                from . import state_schemas as _ss

                _schema_version = _ss.current_version(schema_id)
            except Exception:  # pragma: no cover - defensive
                _schema_version = 1
        else:
            _schema_version = 1
        # Microsecond timestamp keeps rev_ids unique under rapid-fire writes.
        now = datetime.now().isoformat()
        rev_suffix = now.replace(":", "").replace(".", "").replace("-", "")
        rev_id = f"srv_{eid}_{rev_suffix}"

        # State-protocol v3 Phase D (Adrian 2026-05-04): stamp the
        # process-scoped session_id on every revision so scope-aware
        # reads (latest_state_for_entity) can filter session-scoped
        # schemas (intent_state, agent_state) without seeing other
        # sessions' writes. Global-scope schemas (project_state,
        # task_state) ignore the column at read time. NULL is
        # acceptable -- pre-Phase-D rows + gardener retrofit + back-
        # compat callers leave it None and the read path treats NULL
        # as "any session" which preserves prior behaviour.
        with conn:
            conn.execute(
                "INSERT INTO mempalace_state_revisions "
                "(rev_id, entity_id, schema_id, payload, created_at, "
                "op_context_id, agent, schema_version, session_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    rev_id,
                    eid,
                    schema_id,
                    _json.dumps(payload),
                    now,
                    op_context_id or "",
                    agent or "",
                    int(_schema_version),
                    session_id,
                ),
            )

        # State-protocol v1 (Adrian Option B 2026-05-03): the JTMS
        # justification link is the op_context_id COLUMN above, not a KG
        # triple. Dropped the kg.add_triple call after manual test step 5
        # on 2026-05-03 revealed two compounding issues: (a) state_changed_by
        # is a non-structural predicate so add_triple required a
        # statement -- fixed by adding it to _TRIPLE_SKIP_PREDICATES, and
        # (b) rev_id lives in mempalace_state_revisions, NOT entities, so
        # the cold-start phantom-gate rejected it. The phantom-gate suggests
        # passing skip_existence_check=True but that kwarg is NOT plumbed
        # through add_triple's public signature. Rather than declare every
        # rev_id as a kind=entity (pollution) or modify add_triple's
        # internals (blast radius), v1 relies on the indexed
        # op_context_id column for retraction sweeps -- O(log n) via
        # idx_state_revisions_op_context, same query power as a triple
        # walk would give. The state_changed_by predicate stays seeded
        # for future use if we plumb skip_existence_check later.
        return rev_id

    def record_state_revision_challenge(
        self,
        rev_id: str,
        challenge_op_id: str,
        agent: str,
        justification: str,
        retracted_rev_id: str | None = None,
    ) -> str:
        """Insert a state_revision_challenges row + return challenge_id.

        v3.3.0 Phase 3 (Adrian directive 2026-05-13). Files the
        agent's challenge against a specific state_revisions row. The
        caller is responsible for (a) verifying the rev_id exists, (b)
        deciding whether to write a paired 'restore' revision via
        record_state_revision before calling this helper (and passing
        the new rev_id as retracted_rev_id), and (c) checking agent
        ownership / permission. This helper persists the audit row
        only; it does not gate on policy.

        Args:
            rev_id: target revision being challenged. Must exist in
                mempalace_state_revisions or the FK rejects the write.
            challenge_op_id: operation context that filed the
                challenge (becomes the JTMS-style object reference).
            agent: challenging agent id. Required (non-empty).
            justification: free-form explanation of why the challenge
                stands. Required (non-empty).
            retracted_rev_id: optional rev_id of a freshly-written
                state_revision that restored prior state.  None for
                info-only challenges.

        Returns:
            challenge_id: stable string id ('src_<rev>_<ts>').
        """
        import time as _time

        if not (justification or "").strip():
            raise ValueError(
                "record_state_revision_challenge: justification is "
                "required (non-empty); every challenge needs an audit "
                "rationale."
            )
        if not (agent or "").strip():
            raise ValueError(
                "record_state_revision_challenge: agent is required "
                "(non-empty); challenges must attribute to a known "
                "agent for trust/accuracy telemetry."
            )

        # Verify the target revision exists before the FK does (cleaner
        # error message than the raw FK violation).
        conn = self._conn()
        row = conn.execute(
            "SELECT 1 FROM mempalace_state_revisions WHERE rev_id = ?",
            (rev_id,),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"record_state_revision_challenge: rev_id '{rev_id}' "
                "not found in mempalace_state_revisions; cannot "
                "challenge a non-existent revision."
            )

        # Stable id derives from rev_id + monotonic ts so two challenges
        # on the same rev never collide. 6-char rev suffix is enough
        # because rev_id is already globally unique.
        ts_ms = int(_time.time() * 1000)
        rev_suffix = rev_id[-12:] if len(rev_id) > 12 else rev_id
        challenge_id = f"src_{rev_suffix}_{ts_ms}"
        # Mirror record_state_revision's timestamp pattern (datetime.now
        # isoformat, no Z). Microsecond precision keeps challenge_ids
        # unique under rapid-fire writes.
        created_at = datetime.now().isoformat()

        conn.execute(
            "INSERT INTO mempalace_state_revision_challenges ("
            "challenge_id, rev_id, challenge_op_id, agent, "
            "justification, created_at, retracted_rev_id"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                challenge_id,
                rev_id,
                challenge_op_id or "",
                agent,
                justification,
                created_at,
                retracted_rev_id,
            ),
        )
        conn.commit()
        return challenge_id

    def latest_state_for_entity(
        self,
        entity_id: str,
        session_id: str | None = None,
    ) -> dict | None:
        """Return the latest state payload for an entity, or None.

        Reads the most recent mempalace_state_revisions row by
        created_at; returns the parsed JSON dict. None when the entity
        has no revisions yet -- callers using state-protocol v1 retrofit
        logic treat None as the trigger to call
        state_schemas.materialize_default + record_state_revision with
        agent='memory_gardener' for the initial seed.

        lifecycle hardening (Adrian corner-case audit
        2026-05-03): also returns None when the entity is soft-deleted
        (entities.status='deleted'). Without this filter kg_query and
        the per-memory state-enrichment helper would surface stale
        state for entities the agent already retired -- silently
        misleading. Merged entities are handled by self._entity_id()
        following entity_aliases, so a query for the source name
        resolves to the canonical (target) id; cascade in
        merge_entities ensures the source's revisions move to the
        target's id at merge time.

        State-protocol v3 Phase D (Adrian 2026-05-04): scope-aware
        reads. The latest revision's schema_id is looked up in
        STATE_SCHEMAS to determine its scope policy:
          * scope='session'  : when ``session_id`` is provided, refilter
                               to revisions stamped with the same
                               session_id (or NULL pre-Phase-D rows --
                               treated as "any session"). When
                               session_id is None, behave as v2 (return
                               latest regardless).
          * scope='global'   : ignore session_id entirely; latest wins.
        Schemas without a scope field default to 'global' (back-compat
        for hand-built schemas in tests). The two-step lookup (latest
        row -> schema scope -> refilter if needed) avoids requiring
        callers to know the scope policy in advance.
        """
        import json as _json

        eid = self._entity_id(entity_id)
        conn = self._conn()
        # Status filter -- soft-deleted entities should not surface state.
        status_row = conn.execute("SELECT status FROM entities WHERE id = ?", (eid,)).fetchone()
        if status_row is not None and (status_row[0] or "") == "deleted":
            return None
        # First pass: latest row regardless of session. We need its
        # schema_id to decide whether to refilter for scope='session'.
        row = conn.execute(
            "SELECT payload, schema_id FROM mempalace_state_revisions "
            "WHERE entity_id = ? ORDER BY created_at DESC LIMIT 1",
            (eid,),
        ).fetchone()
        if row is None:
            return None
        _payload, _row_schema_id = row[0], row[1]
        # Phase D scope policy. When session_id is None (v2 callers,
        # gardener, back-compat reads) we skip the refilter and return
        # whatever the latest row says -- preserves prior behaviour.
        if session_id and _row_schema_id:
            try:
                from . import state_schemas as _ss

                _scope = (_ss.STATE_SCHEMAS.get(_row_schema_id) or {}).get("scope") or "global"
            except Exception:
                _scope = "global"
            if _scope == "session":
                # Refilter: latest row scoped to this session OR pre-
                # Phase-D rows (session_id IS NULL). NULL handling is
                # important so reinstall doesn't blank existing state.
                row = conn.execute(
                    "SELECT payload FROM mempalace_state_revisions "
                    "WHERE entity_id = ? AND schema_id = ? "
                    "AND (session_id = ? OR session_id IS NULL) "
                    "ORDER BY created_at DESC LIMIT 1",
                    (eid, _row_schema_id, session_id),
                ).fetchone()
                if row is None:
                    return None
                _payload = row[0]
        try:
            return _json.loads(_payload)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None

    def migrate_state_for_entities(
        self,
        entity_ids,
        agent: str = "memory_gardener",
    ) -> dict:
        """Phase 6 lazy-migration-at-injection runner.

        For each entity_id in ``entity_ids``: read the latest state
        revision (rev_id, schema_id, schema_version, payload). If
        schema_version < state_schemas.current_version(schema_id),
        walk the migration chain via
        mempalace.state_migrations.apply_pending_migrations and write
        a NEW revision at the current version (durable audit trail
        per Adrian's design lock). Entities with no revisions or
        empty schema_id are silently skipped (state_init_needed
        gardener handles those).

        Called from injection_gate.apply_gate after kept_ids is built,
        so only entities that survived the InjectionGate's relevance
        filter pay migration cost. Dormant entities never trigger.

        Returns a dict mapping entity_id -> {"version": int,
        "migrated": bool, "schema_id": str}. Failures are caught
        per-entity and logged via the standard logger; entries get a
        "error" key. Errors do not propagate -- the gate path must
        stay open.
        """
        import json as _json

        from . import state_migrations as _migrations
        from . import state_schemas as _schemas

        out: dict = {}
        if not entity_ids:
            return out
        conn = self._conn()
        for raw_eid in entity_ids:
            if not raw_eid:
                continue
            eid = self._entity_id(raw_eid)
            status_row = conn.execute("SELECT status FROM entities WHERE id = ?", (eid,)).fetchone()
            if status_row is None or (status_row[0] or "") == "deleted":
                out[raw_eid] = {"migrated": False, "version": 0, "schema_id": ""}
                continue
            row = conn.execute(
                "SELECT rev_id, schema_id, schema_version, payload "
                "FROM mempalace_state_revisions "
                "WHERE entity_id = ? ORDER BY created_at DESC LIMIT 1",
                (eid,),
            ).fetchone()
            if row is None:
                out[raw_eid] = {"migrated": False, "version": 0, "schema_id": ""}
                continue
            _, schema_id, schema_version, payload_json = row
            if not schema_id:
                out[raw_eid] = {"migrated": False, "version": 0, "schema_id": ""}
                continue
            try:
                current_v = _schemas.current_version(schema_id)
            except KeyError:
                out[raw_eid] = {
                    "migrated": False,
                    "version": int(schema_version or 1),
                    "schema_id": schema_id,
                    "error": f"unknown_schema:{schema_id}",
                }
                continue
            from_v = int(schema_version or 1)
            if from_v >= current_v:
                out[raw_eid] = {
                    "migrated": False,
                    "version": from_v,
                    "schema_id": schema_id,
                }
                continue
            try:
                payload = _json.loads(payload_json)
            except (TypeError, ValueError):
                out[raw_eid] = {
                    "migrated": False,
                    "version": from_v,
                    "schema_id": schema_id,
                    "error": "payload_parse_failed",
                }
                continue
            try:
                migrated = _migrations.apply_pending_migrations(
                    payload, schema_id, from_v, current_v
                )
            except _migrations.StateMigrationError as exc:
                out[raw_eid] = {
                    "migrated": False,
                    "version": from_v,
                    "schema_id": schema_id,
                    "error": f"migration_failed:{exc}",
                }
                continue
            try:
                self.record_state_revision(
                    entity_id=eid,
                    schema_id=schema_id,
                    payload=migrated,
                    op_context_id="",
                    agent=agent,
                )
                out[raw_eid] = {
                    "migrated": True,
                    "version": current_v,
                    "schema_id": schema_id,
                }
            except Exception as exc:  # pragma: no cover - defensive
                out[raw_eid] = {
                    "migrated": False,
                    "version": from_v,
                    "schema_id": schema_id,
                    "error": f"persist_failed:{type(exc).__name__}:{exc}",
                }
        return out

    # Rating predicates -- the closed set that add_rated_edge treats as a
    # single logical slot per (ctx, memory) pair. Writing ONE supersedes
    # any prior rating regardless of direction (useful→irrelevant flip
    # invalidates the useful edge too). Regular add_triple still dedups
    # identical edges for structural predicates; this is rating-only.
    _RATING_PREDICATES = frozenset(("rated_useful", "rated_irrelevant"))

    def add_rated_edge(
        self,
        context: str,
        predicate: str,
        memory: str,
        confidence: float = 1.0,
        statement: str = None,
        properties: dict = None,
        valid_from: str = None,
    ):
        """Write a rating edge with last-wins-across-predicates semantics.

        Contract: at most ONE current (valid_to IS NULL) rating edge exists
        per (context, memory) pair, regardless of direction. Writing a new
        rating -- useful OR irrelevant -- invalidates any prior rating on
        the same pair before inserting.

        This is the fix for the add_triple silent-drop bug on rating
        edges (documented 2026-04-22 in
        record_ga_agent_add_triple_first_wins_on_rated_edges). The
        generic add_triple short-circuits on duplicate PK, which drops
        same-direction re-ratings and prevents direction flips from
        replacing prior ratings.

        Four failure modes this fixes:
          1. same agent re-rates same direction stronger (4→5)
          2. same agent re-rates same direction weaker (5→3)
          3. same agent flips direction (useful→irrelevant, or the reverse)
          4. different agent re-rates (last-wins globally -- this is the
             accepted simpler path; per-agent supersede is a future design
             if multi-agent consensus stacking becomes needed)

        Rationale: rating edges carry additional state (confidence,
        reason, agent, ts) that isn't part of the PK. The generic
        dedup-on-PK is correct for structural predicates (is_a, described_by)
        where "the fact is the same" means "don't duplicate", but wrong
        for ratings where re-evaluation is legitimate new information.
        See docs/link_author_plan.md discussion of CrowdTruth 2.0 and
        Davani et al. 2022 TACL on subjective rating semantics.

        Args:
            context:   the context entity that rated the memory
            predicate: 'rated_useful' or 'rated_irrelevant' -- other
                       predicates raise ValueError (use add_triple)
            memory:    the memory entity being rated
            confidence: 0.0-1.0 edge confidence (scaled from relevance)
            statement: optional; ratings are embedded only if present
            properties: {ts, relevance, reason, agent, ...}
            valid_from: when the rating was issued (ISO; defaults to now)

        Returns: the new triple id.
        """
        pred = _normalize_predicate(predicate)
        if pred not in self._RATING_PREDICATES:
            raise ValueError(
                f"add_rated_edge only accepts rating predicates "
                f"{sorted(self._RATING_PREDICATES)}, got {pred!r}. "
                f"Use add_triple for structural predicates."
            )

        sub_id = self._entity_id(context)
        obj_id = self._entity_id(memory)
        # Keep microseconds in the hash input so rapid-fire re-rates on the
        # same (ctx, mem, pred) land on distinct triple ids. Stripping to
        # seconds caused UNIQUE-constraint failures in supersede tests.
        now_full = datetime.now().isoformat()
        ended = now_full[:10]  # YYYY-MM-DD for consistency with invalidate()

        conn = self._conn()
        with conn:
            # Hard-reject phantom references (cold-start lock 2026-05-01).
            # Mirror add_triple's policy: rating edges write to existing
            # context + memory entities only. Pre-cold-start the lines
            # below silently auto-created missing endpoints; the cold-
            # start gate requires every entity to be minted via
            # mint_entity (with summary + identity check) before any
            # edge -- structural or rating -- can reference it.
            from .entity_gate import assert_entity_exists

            assert_entity_exists(sub_id, conn)
            assert_entity_exists(obj_id, conn)

            # Invalidate ANY current rating edge on this (ctx, memory)
            # pair, regardless of direction. One SQL pass covers both
            # rated_useful and rated_irrelevant predicates.
            conn.execute(
                "UPDATE triples SET valid_to = ? "
                "WHERE subject = ? AND object = ? "
                "AND predicate IN ('rated_useful', 'rated_irrelevant') "
                "AND valid_to IS NULL",
                (ended, sub_id, obj_id),
            )

            triple_id = (
                f"t_{sub_id}_{pred}_{obj_id}_"
                f"{hashlib.sha256(f'{valid_from}{now_full}'.encode()).hexdigest()[:12]}"
            )
            props_json = json.dumps(properties or {})
            conn.execute(
                """INSERT INTO triples (id, subject, predicate, object, valid_from, valid_to,
                                        confidence, source_file, creation_context_id, statement,
                                        properties)
                   VALUES (?, ?, ?, ?, ?, NULL, ?, NULL, ?, ?, ?)""",
                (
                    triple_id,
                    sub_id,
                    pred,
                    obj_id,
                    valid_from,
                    float(confidence),
                    sub_id,  # creation_context_id IS the rater context itself
                    statement,
                    props_json,
                ),
            )
        self._touch_entity(sub_id)
        self._touch_entity(obj_id)
        # Rating edges are not embedded (no statement for skip-list
        # predicates); no _index_triple_statement call.
        return triple_id

    # ════════════════════════════════════════════════════════════════
    # TRIPLE-SCOPED FEEDBACK (migration 018)
    # ════════════════════════════════════════════════════════════════
    # Triples have ids in a separate namespace from entities
    # (t_<sub>_<pred>_<obj>_<hash>), so the rated_useful /
    # rated_irrelevant / surfaced edges that live on
    # context → entity cannot target them -- add_triple would auto-
    # create a phantom entity via INSERT OR IGNORE. Triple feedback is
    # written natively into triple_context_feedback (migration 018)
    # with the same last-wins-across-directions contract add_rated_edge
    # uses on entity ratings. The partial unique index on
    # (context_id, triple_id) WHERE valid_to IS NULL enforces
    # at-most-one current row at the schema level. Channel D's
    # walk_rated_neighbourhood reads this table alongside edge-based
    # ratings and merges them into a single signed rated_scores map
    # keyed by object id (memory_id OR triple_id).

    _TRIPLE_FEEDBACK_KINDS = frozenset(("rated_useful", "rated_irrelevant", "surfaced"))

    def _record_triple_feedback(
        self,
        context_id: str,
        triple_id: str,
        kind: str,
        *,
        relevance: int = None,
        reason: str = "",
        rater_kind: str = "agent",
        rater_id: str = "",
        confidence: float = 1.0,
        valid_from: str = None,
    ):
        """Write a triple-scoped feedback row with last-wins supersede.

        Invalidates any current (valid_to IS NULL) row for the
        (context_id, triple_id) pair regardless of prior kind before
        inserting the new row. Same contract as add_rated_edge on
        entity-scope ratings; the partial unique index on
        triple_context_feedback enforces at-most-one current row.

        Public callers should go through ``record_feedback`` instead of
        calling this directly -- the dispatcher picks the right target
        namespace based on target_kind.
        """
        if kind not in self._TRIPLE_FEEDBACK_KINDS:
            raise ValueError(
                f"_record_triple_feedback only accepts "
                f"{sorted(self._TRIPLE_FEEDBACK_KINDS)}, got {kind!r}"
            )
        if rater_kind not in ("agent", "gate_llm", "haiku_auto"):
            raise ValueError(
                f"rater_kind must be 'agent', 'gate_llm', or 'haiku_auto', got {rater_kind!r}"
            )
        now_full = datetime.now().isoformat()
        ended = now_full[:19]  # second precision for valid_to
        conn = self._conn()
        with conn:
            # Supersede any current row for this (ctx, triple) pair --
            # direction-agnostic, same semantics as add_rated_edge.
            conn.execute(
                "UPDATE triple_context_feedback SET valid_to = ? "
                "WHERE context_id = ? AND triple_id = ? AND valid_to IS NULL",
                (ended, context_id, triple_id),
            )
            conn.execute(
                """INSERT INTO triple_context_feedback
                   (context_id, triple_id, kind, relevance, reason,
                    rater_kind, rater_id, confidence, valid_from, valid_to, ts)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)""",
                (
                    context_id,
                    triple_id,
                    kind,
                    int(relevance) if relevance is not None else None,
                    str(reason or ""),
                    rater_kind,
                    str(rater_id or ""),
                    float(confidence),
                    valid_from,
                    now_full,
                ),
            )

    def record_feedback(
        self,
        context_id: str,
        target_id: str,
        target_kind: str,
        *,
        relevance: int,
        reason: str = "",
        rater_kind: str = "agent",
        rater_id: str = "",
        confidence: float = 1.0,
        valid_from: str = None,
    ):
        """Unified feedback writer. Dispatches by target namespace.

        target_kind:
            'entity'  -- target_id refers to a row in entities
                        (records, concepts, classes, predicates,
                        literals). Writes a rated_useful or
                        rated_irrelevant edge via add_rated_edge.
            'triple'  -- target_id refers to a row in triples.
                        Writes a row in triple_context_feedback via
                        _record_triple_feedback (no phantom entity).

        relevance 1-5 maps to kind:
            1-2 → rated_irrelevant
            3-5 → rated_useful

        For ``surfaced`` retrieval-event edges (recall-only, no 1-5
        rating), call the lower-level writers directly: add_triple
        for entity targets, _record_triple_feedback(kind='surfaced')
        for triple targets.

        See add_rated_edge docstring for the four failure modes the
        supersede contract closes; the same contract applies here in
        both namespaces.
        """
        rel_int = int(relevance)
        if rel_int < 1 or rel_int > 5:
            raise ValueError(f"relevance must be 1-5, got {rel_int}")
        is_positive = rel_int >= 3
        if target_kind == "entity":
            pred = "rated_useful" if is_positive else "rated_irrelevant"
            props = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "relevance": rel_int,
                "reason": str(reason or ""),
                "agent": str(rater_id or ""),
                "rater_kind": str(rater_kind or "agent"),
            }
            self.add_rated_edge(
                context_id,
                pred,
                target_id,
                confidence=confidence,
                properties=props,
                valid_from=valid_from,
            )
        elif target_kind == "triple":
            kind = "rated_useful" if is_positive else "rated_irrelevant"
            self._record_triple_feedback(
                context_id,
                target_id,
                kind,
                relevance=rel_int,
                reason=reason,
                rater_kind=rater_kind,
                rater_id=rater_id,
                confidence=confidence,
                valid_from=valid_from,
            )
        else:
            raise ValueError(f"target_kind must be 'entity' or 'triple', got {target_kind!r}")

    def record_operation_rating(
        self,
        *,
        op_context_id: str,
        tool: str,
        quality: int,
        rater_kind: str = "agent",
        rater_id: str = "",
        reason: str = "",
    ):
        """Record a tool-invocation quality rating (v3.5.0).

        Quality >=4 writes a performed_well edge from the op-context
        to the operation entity; quality <=2 writes performed_poorly;
        quality=3 is neutral (no edge written).

        The op entity id is looked up from an executed_op edge whose
        subject is ``op_context_id``. The Haiku rater
        (mempalace.feedback_auto) is the primary caller; it walks
        contexts_touched_detail and emits one rating per per-op
        context. Best-effort: when the op entity can't be located,
        the call silently returns (the rater's batch is non-fatal).
        """
        if rater_kind not in ("agent", "gate_llm", "haiku_auto"):
            raise ValueError(
                f"rater_kind must be 'agent', 'gate_llm', or 'haiku_auto', got {rater_kind!r}"
            )
        q = int(quality)
        if q < 1 or q > 5:
            raise ValueError(f"quality must be 1..5, got {q}")
        if q == 3:
            return  # neutral, no edge

        pred = "performed_well" if q >= 4 else "performed_poorly"

        op_id = None
        try:
            cur = self._conn().execute(
                "SELECT object FROM triples "
                "WHERE subject = ? AND predicate = 'executed_op' "
                "AND valid_to IS NULL LIMIT 1",
                (op_context_id,),
            )
            row = cur.fetchone()
            if row:
                op_id = row[0]
        except Exception:
            return  # best-effort
        if not op_id:
            return

        statement = {
            "what": f"{op_context_id} {pred} {op_id}",
            "why": (f"{rater_kind} rated op quality {q}: {reason or '(no reason)'}")[:240],
        }
        try:
            self.add_triple(
                op_context_id,
                pred,
                op_id,
                statement=statement,
            )
        except Exception:
            return  # best-effort

    def get_triple_feedback(self, context_ids):
        """Return current triple feedback rows for the given contexts.

        Channel D's walk_rated_neighbourhood calls this once per walk
        with the active context plus its similar_to neighbourhood and
        merges the result into rated_scores/channel_D_list keyed by
        triple_id.

        Returns a list of dicts with keys:
        context_id, triple_id, kind, relevance, confidence, rater_kind.

        Empty input → empty list (no SQL emitted). Missing table →
        empty list (caller treats as no-feedback, graceful degrade
        for palaces that haven't applied migration 018 yet).
        """
        if not context_ids:
            return []
        conn = self._conn()
        placeholders = ",".join("?" for _ in context_ids)
        try:
            rows = conn.execute(
                "SELECT context_id, triple_id, kind, relevance, "
                "confidence, rater_kind "
                "FROM triple_context_feedback "
                f"WHERE valid_to IS NULL AND context_id IN ({placeholders})",
                tuple(context_ids),
            ).fetchall()
        except sqlite3.OperationalError:
            # Table doesn't exist yet (pre-018 palace); treat as empty.
            return []
        return [dict(r) for r in rows]

    # ════════════════════════════════════════════════════════════════
    # MEMORY-FLAGS + GARDENER RUNS (migrations 019 / 020)
    # ════════════════════════════════════════════════════════════════
    # The injection gate emits quality flags alongside its keep/drop
    # decisions -- duplicate pairs, contradictions, stale facts, orphan
    # memories, generic summaries, implied edges, unlinked entities.
    # Those land here as memory_flags rows. The out-of-session
    # memory_gardener process reads pending flags in batches,
    # investigates each via a Claude Code subprocess (Haiku is fine;
    # what matters is the tool access + reasoning), and acts: merge,
    # invalidate, link, rewrite, prune, propose edge (to link-author
    # queue), or defer. Every batch is logged to memory_gardener_runs
    # with per-action counters for audit.

    _MEMORY_FLAG_KINDS = frozenset(
        (
            "duplicate_pair",
            "contradiction_pair",
            "stale",
            "unlinked_entity",
            "orphan",
            "generic_summary",
            "edge_candidate",
            # S3a: operation-cluster flag emitted by declare_operation
            # when retrieve_past_operations surfaces >=3 same-tool
            # same-sign precedents. Gardener synthesises a template.
            "op_cluster_templatizable",
            # v3.10.0 (Adrian msg_176/178 2026-05-21): the bg quality pass
            # emits these ontology-review flags so the corpus self-heals.
            # is_a_review -- odd/missing/coarse is_a chain. kind_misclassification
            # -- wrong `kind` for the entity. class_id_improvement -- opaque
            # class id vs its description. Resolved by the gardener via
            # is_a_corrected / kind_corrected / class_renamed.
            "is_a_review",
            "kind_misclassification",
            "class_id_improvement",
        )
    )

    _FLAG_RESOLUTIONS = frozenset(
        (
            "merged",
            "invalidated",
            "linked",
            "edge_proposed",
            "summary_rewritten",
            "pruned",
            "deferred",
            "no_action",
            # S3b: gardener resolved an op_cluster_templatizable flag
            # by minting a template record + writing `templatizes`
            # edges back to the source operations.
            "templatized",
            # State-protocol v1 piece 6 (Adrian Option B 2026-05-03):
            # gardener resolved a state_init_needed flag by calling
            # state_schemas.materialize_default + record_state_revision
            # to seed the instance's first state payload. Mechanical
            # handler (no Haiku). Discovered missing by manual test
            # step 8 on 2026-05-03.
            "state_initialized",
            # v3.10.0 (Adrian msg_176/178 2026-05-21): resolutions for
            # the three ontology-review flags. is_a_corrected -- gardener
            # added/removed is_a edges to fix the chain. kind_corrected --
            # gardener changed the entity's kind. class_renamed -- gardener
            # renamed a poor class id (cascades via merge_entities).
            "is_a_corrected",
            "kind_corrected",
            "class_renamed",
        )
    )

    @staticmethod
    def _canonical_memory_key(memory_ids) -> str:
        """Canonical dedup key: sorted, joined member ids so pair
        flags are direction-agnostic and single-member flags hash
        deterministically."""
        if not isinstance(memory_ids, (list, tuple)):
            memory_ids = [memory_ids]
        cleaned = sorted(str(m) for m in memory_ids if m)
        return "|".join(cleaned)

    def _reflag_suppressed(
        self,
        conn,
        *,
        kind: str,
        mkey: str,
        reflag_cutoff: str,
        max_reflags: int,
    ) -> bool:
        """Reflag anti-oscillation decision (steps 1+4, 2026-05-23;
        widened to cross-context 2026-05-26 fix #2).

        Returns True if a flag for this (kind, memory_key) should be
        suppressed because prior RESOLVED rows -- across ALL context_ids
        for the same target -- indicate either:
          * circuit breaker -- already resolved >= max_reflags times
            (permanent; resolved rows persist), or
          * cooldown -- most recent resolution is at/after reflag_cutoff.

        v3.10.8 keyed this check on (kind, memory_key, context_id), but
        InjectionGate mints a fresh ctx_XXXX per retrieval frame so the
        brake never fired across frames. Real-corpus data: intent_py
        flagged 232x, memory_gardener 219x, knowledge_graph_py 134x --
        each via a different context_id, all sailing past the brake.
        Dropping context_id from the WHERE makes the brake do what its
        comments always claimed.

        Fails open (returns False) if the table is missing. Extracted
        from record_memory_flags to keep that method under the C901
        complexity ceiling.
        """
        try:
            grow = conn.execute(
                """SELECT COUNT(*) AS n, MAX(resolved_ts) AS last_resolved
                   FROM memory_flags
                   WHERE kind = ? AND memory_key = ?
                         AND resolved_ts IS NOT NULL""",
                (kind, mkey),
            ).fetchone()
        except sqlite3.OperationalError:
            return False
        if grow is None:
            return False
        prior_resolved = int(grow["n"] or 0)
        last_resolved = str(grow["last_resolved"] or "")
        if max_reflags > 0 and prior_resolved >= max_reflags:
            return True
        if reflag_cutoff and last_resolved and last_resolved >= reflag_cutoff:
            return True
        return False

    def _insert_or_bump_same_ctx(
        self,
        conn,
        *,
        kind: str,
        mids: list,
        mkey: str,
        detail: str,
        cid: str,
        now: str,
        rater_model: str,
    ) -> None:
        """INSERT a fresh memory_flags row; on unique-index collision
        with an existing same-ctx pending row, bump that row's
        attempted_count instead. Defensive backstop for the
        SELECT-then-INSERT race after _bump_pending_if_any cleared a
        pending row in a concurrent writer. Extracted from
        record_memory_flags to keep that method under the C901
        complexity ceiling.
        """
        try:
            conn.execute(
                """INSERT INTO memory_flags
                       (kind, memory_ids, memory_key, detail,
                        context_id, gate_run_ts, rater_model,
                        attempted_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 0)""",
                (
                    kind,
                    json.dumps(list(mids)),
                    mkey,
                    detail,
                    cid,
                    now,
                    rater_model,
                ),
            )
        except sqlite3.IntegrityError:
            conn.execute(
                """UPDATE memory_flags
                   SET attempted_count = attempted_count + 1,
                       last_attempt_ts = ?
                   WHERE kind = ? AND memory_key = ?
                         AND context_id = ?
                         AND resolved_ts IS NULL""",
                (now, kind, mkey, cid),
            )

    def _bump_pending_if_any(
        self,
        conn,
        *,
        kind: str,
        mkey: str,
        now: str,
    ) -> bool:
        """Cross-context pending UPSERT (fix #2, 2026-05-26).

        Bump attempted_count on any pending row for (kind, memory_key)
        regardless of context_id and return True; return False when no
        pending row exists.

        The legacy partial unique index keys on (kind, mkey, context_id)
        so different ctx_ids would otherwise insert as fresh
        duplicates -- intent_py picked up 232 flag rows lifetime, one
        per context_id the gate ever saw. Honouring the "one unresolved
        row per (kind, mkey)" dedup contract the record_memory_flags
        docstring promises requires bumping the existing pending row
        instead. Extracted from record_memory_flags to keep that method
        under the C901 complexity ceiling.
        """
        pending = conn.execute(
            """SELECT id FROM memory_flags
               WHERE kind = ? AND memory_key = ?
                     AND resolved_ts IS NULL
               LIMIT 1""",
            (kind, mkey),
        ).fetchone()
        if pending is None:
            return False
        conn.execute(
            """UPDATE memory_flags
               SET attempted_count = attempted_count + 1,
                   last_attempt_ts = ?
               WHERE id = ?""",
            (now, int(pending["id"])),
        )
        return True

    def record_memory_flags(self, flags: list, *, rater_model: str = "") -> int:
        """Persist a batch of gate-emitted flags.

        Each entry is a dict:
          {kind, memory_ids: [...], detail?, context_id?}

        Dedup contract: one unresolved row per
        (kind, memory_key, context_id). Re-observing the same issue
        from the same context bumps attempted_count on the existing
        pending row rather than inserting a duplicate. (attempted_count
        here reads as 'times observed before resolution'; the gardener
        bumps the same column on processing attempts -- same column,
        two related meanings, and the merge is intentional so a flag
        the gate re-asserts stays prioritised.)

        Settling-time guard (closes 2026-04-25 audit finding #15):
        flags whose target memory_ids include any entity created
        within the last MEMPALACE_FLAG_SETTLING_MIN minutes (default
        30) are dropped silently. New writes need a buffer to settle
        before the gardener starts second-guessing them -- without
        this, freshly-written records get re-flagged within minutes
        and the gardener chases its own tail.

        Reflag anti-oscillation guard (steps 1+4, 2026-05-23): before
        inserting, consult prior RESOLVED rows for the same
        (kind, memory_key, context_id). Skip the flag if it was resolved
        within MEMPALACE_FLAG_REFLAG_COOLDOWN_MIN minutes (default 1440;
        cooldown) or if it has already been resolved
        MEMPALACE_FLAG_MAX_REFLAGS times (default 5; permanent
        circuit-breaker). Either knob at 0 disables that brake. This
        closes the flag/resolve ping-pong the partial dedup index
        (WHERE resolved_ts IS NULL) would otherwise allow.

        Returns count of rows inserted OR bumped. Failures (bad kind,
        empty memory_ids, missing table) are skipped silently and
        do NOT abort the batch.
        """
        if not flags:
            return 0
        conn = self._conn()
        now = datetime.now().isoformat(timespec="seconds")
        written = 0

        # ── Settling-time pre-filter ──
        try:
            settling_min = int(os.environ.get("MEMPALACE_FLAG_SETTLING_MIN", "30") or 0)
        except (TypeError, ValueError):
            settling_min = 30
        if settling_min > 0:
            cutoff = (datetime.now() - timedelta(minutes=settling_min)).isoformat(
                timespec="seconds"
            )
            # Collect every memory_id referenced across the batch in one pass.
            all_ids: set[str] = set()
            for flag in flags:
                if isinstance(flag, dict):
                    for mid in flag.get("memory_ids") or []:
                        if mid:
                            all_ids.add(str(mid))
            young_ids: set[str] = set()
            if all_ids:
                # Look up created_at per id in a single IN-clause query.
                placeholders = ",".join("?" for _ in all_ids)
                try:
                    rows = conn.execute(
                        f"SELECT id, created_at FROM entities WHERE id IN ({placeholders})",
                        list(all_ids),
                    ).fetchall()
                    for r in rows:
                        ca = r["created_at"] if r else ""
                        # entities.created_at is "YYYY-MM-DD HH:MM:SS" or ISO
                        if ca and str(ca).replace(" ", "T") >= cutoff:
                            young_ids.add(r["id"])
                except sqlite3.OperationalError:
                    young_ids = set()
            if young_ids:
                flags = [
                    f
                    for f in flags
                    if isinstance(f, dict)
                    and not (set(str(m) for m in (f.get("memory_ids") or [])) & young_ids)
                ]
        if not flags:
            return 0

        # ── Reflag anti-oscillation guard config (steps 1+4, 2026-05-23) ──
        # Closes the flag/resolve ping-pong hole: because the dedup index
        # is partial (WHERE resolved_ts IS NULL), resolving a flag drops it
        # from the index, so the identical (kind, memory_key, context_id)
        # re-inserts as a fresh row at attempted_count=0 -- nothing damps it.
        # Two literature-backed brakes (see
        # record_ga_agent_reflag_cooldown_literature_evidence_2026_05_23):
        #   Step 1 -- COOLDOWN: skip a flag whose (kind, key, ctx) was
        #     resolved within the last MEMPALACE_FLAG_REFLAG_COOLDOWN_MIN
        #     minutes (default 1440 = 24h). Gives a fix time to settle.
        #   Step 4 -- CIRCUIT BREAKER: once the same (kind, key, ctx) has
        #     been resolved >= MEMPALACE_FLAG_MAX_REFLAGS times (default 5),
        #     suppress it permanently. The resolved rows persist, so the
        #     count -- and the suppression -- is permanent (a "wontfix"
        #     without a schema column).
        # Either knob set to 0 disables that brake (mirrors the settling knob).
        try:
            reflag_cooldown_min = int(
                os.environ.get("MEMPALACE_FLAG_REFLAG_COOLDOWN_MIN", "1440") or 0
            )
        except (TypeError, ValueError):
            reflag_cooldown_min = 1440
        try:
            max_reflags = int(os.environ.get("MEMPALACE_FLAG_MAX_REFLAGS", "5") or 0)
        except (TypeError, ValueError):
            max_reflags = 5
        reflag_cutoff = ""
        if reflag_cooldown_min > 0:
            reflag_cutoff = (datetime.now() - timedelta(minutes=reflag_cooldown_min)).isoformat(
                timespec="seconds"
            )

        try:
            with conn:
                for flag in flags:
                    if not isinstance(flag, dict):
                        continue
                    kind = flag.get("kind")
                    if kind not in self._MEMORY_FLAG_KINDS:
                        continue
                    mids = flag.get("memory_ids") or []
                    if not mids:
                        continue
                    mkey = self._canonical_memory_key(mids)
                    if not mkey:
                        continue
                    detail = str(flag.get("detail") or "")
                    cid = str(flag.get("context_id") or "")
                    # ── Reflag anti-oscillation guard (steps 1+4) ──
                    # Consult prior RESOLVED rows for (kind, memory_key)
                    # across ALL context_ids. v3.10.8 keyed this on
                    # (kind, mkey, cid) too but retrieval frames mint a
                    # fresh ctx_XXXX each call -- the brake never fired
                    # cross-frame. Widened 2026-05-26 (fix #2).
                    if (reflag_cooldown_min > 0 or max_reflags > 0) and (
                        self._reflag_suppressed(
                            conn,
                            kind=kind,
                            mkey=mkey,
                            reflag_cutoff=reflag_cutoff,
                            max_reflags=max_reflags,
                        )
                    ):
                        continue
                    # ── Cross-context pending UPSERT (fix #2, 2026-05-26) ──
                    # Bump any pending (kind, memory_key) row regardless
                    # of context_id; helper carries the rationale.
                    if self._bump_pending_if_any(conn, kind=kind, mkey=mkey, now=now):
                        written += 1
                        continue
                    self._insert_or_bump_same_ctx(
                        conn,
                        kind=kind,
                        mids=mids,
                        mkey=mkey,
                        detail=detail,
                        cid=cid,
                        now=now,
                        rater_model=rater_model,
                    )
                    written += 1
        except sqlite3.OperationalError:
            # Table doesn't exist (pre-019 palace); silent no-op.
            return 0
        return written

    def list_pending_flags(self, limit: int = 10) -> list:
        """Return up to `limit` unresolved flags, lowest attempted_count
        first so stuck retries don't starve new work. Used by the
        memory_gardener to build a batch for one Claude Code run."""
        conn = self._conn()
        try:
            rows = conn.execute(
                """SELECT id, kind, memory_ids, memory_key, detail,
                          context_id, gate_run_ts, attempted_count,
                          last_attempt_ts
                   FROM memory_flags
                   WHERE resolved_ts IS NULL
                     AND attempted_count < 3
                   ORDER BY attempted_count ASC, gate_run_ts DESC
                   LIMIT ?""",
                (int(limit),),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        out = []
        for r in rows:
            row = dict(r)
            try:
                row["memory_ids"] = json.loads(row["memory_ids"])
            except (TypeError, ValueError):
                row["memory_ids"] = []
            out.append(row)
        return out

    def count_pending_flags(self) -> int:
        """Count unresolved flags with attempted_count < 3. Used by
        finalize_intent to decide whether to trigger the gardener."""
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM memory_flags "
                "WHERE resolved_ts IS NULL AND attempted_count < 3"
            ).fetchone()
        except sqlite3.OperationalError:
            return 0
        return int(row["c"] if row else 0)

    def mark_flag_resolved(
        self,
        flag_id: int,
        resolution: str,
        *,
        note: str = "",
    ) -> bool:
        """Stamp a flag as resolved with an outcome code. Valid
        resolutions are in _FLAG_RESOLUTIONS. Returns True if a row
        was updated."""
        if resolution not in self._FLAG_RESOLUTIONS:
            raise ValueError(
                f"Unknown flag resolution {resolution!r}; valid: {sorted(self._FLAG_RESOLUTIONS)}"
            )
        conn = self._conn()
        now = datetime.now().isoformat(timespec="seconds")
        try:
            with conn:
                cur = conn.execute(
                    """UPDATE memory_flags
                       SET resolved_ts = ?, resolution = ?, resolution_note = ?,
                           attempted_count = attempted_count + 1,
                           last_attempt_ts = ?
                       WHERE id = ? AND resolved_ts IS NULL""",
                    (now, resolution, str(note), now, int(flag_id)),
                )
                return cur.rowcount > 0
        except sqlite3.OperationalError:
            return False

    def bump_flag_attempt(self, flag_id: int) -> bool:
        """Increment attempted_count without resolving -- used when the
        gardener decides to defer but may retry later. After
        attempted_count reaches 3 the flag is frozen (list_pending_flags
        filters it out) pending manual release."""
        conn = self._conn()
        now = datetime.now().isoformat(timespec="seconds")
        try:
            with conn:
                cur = conn.execute(
                    """UPDATE memory_flags
                       SET attempted_count = attempted_count + 1,
                           last_attempt_ts = ?
                       WHERE id = ? AND resolved_ts IS NULL""",
                    (now, int(flag_id)),
                )
                return cur.rowcount > 0
        except sqlite3.OperationalError:
            return False

    def gc_stale_gardener_runs(self, *, ttl_minutes: int = 60) -> int:
        """v3.10.15 (Adrian goal 2026-05-27): sweep stale in-flight
        memory_gardener_runs rows. Rows with completed_ts IS NULL whose
        started_ts is older than ttl_minutes are orphans from killed
        sessions / crashed gardener subprocesses -- the kernel-flock
        held by the live writer was released at process exit so any
        such row is by definition no longer being written to.

        Marks them completed_ts=now with subprocess_exit_code=-1 and
        errors='aborted: no completion within TTL'. Audit row stays;
        the gardener queries for in-flight rows stop returning them.

        Today's discovery: 131 rows older than 7 days; oldest 424h
        (started 2026-05-08). No GC path previously existed; running
        cleanup script cleared the backlog -- this method prevents
        recurrence by sweeping on every gardener spawn.

        Returns the number of rows swept. Safe under kernel-flock
        single-writer invariant; idempotent on subsequent calls."""
        if ttl_minutes <= 0:
            return 0
        conn = self._conn()
        cutoff = (datetime.now() - timedelta(minutes=int(ttl_minutes))).isoformat(
            timespec="seconds"
        )
        now = datetime.now().isoformat(timespec="seconds")
        try:
            with conn:
                cur = conn.execute(
                    """UPDATE memory_gardener_runs
                       SET completed_ts=?, subprocess_exit_code=-1,
                           errors='aborted: no completion within TTL ('||?||' minutes)'
                       WHERE completed_ts IS NULL AND started_ts < ?""",
                    (now, str(ttl_minutes), cutoff),
                )
                return int(cur.rowcount or 0)
        except sqlite3.OperationalError:
            return 0

    def start_gardener_run(self, *, gardener_model: str = "") -> int:
        """Insert a new memory_gardener_runs row and return its id.
        The gardener finishes the row later via finish_gardener_run."""
        conn = self._conn()
        now = datetime.now().isoformat(timespec="seconds")
        with conn:
            cur = conn.execute(
                """INSERT INTO memory_gardener_runs
                       (started_ts, gardener_model)
                   VALUES (?, ?)""",
                (now, gardener_model),
            )
            return int(cur.lastrowid)

    def finish_gardener_run(
        self,
        run_id: int,
        *,
        flag_ids: list | None = None,
        counters: dict | None = None,
        subprocess_exit_code: int | None = None,
        errors: str = "",
    ) -> None:
        """Complete a gardener run row with per-action counters and
        subprocess metadata. counters keys are merges, invalidations,
        links_created, edges_proposed, summary_rewrites, prunes,
        deferrals, no_action -- missing keys default to 0."""
        c = counters or {}
        conn = self._conn()
        now = datetime.now().isoformat(timespec="seconds")
        fid_json = json.dumps(list(flag_ids or []))
        with conn:
            conn.execute(
                """UPDATE memory_gardener_runs
                   SET completed_ts = ?,
                       flags_processed = ?,
                       flag_ids = ?,
                       merges = ?,
                       invalidations = ?,
                       links_created = ?,
                       edges_proposed = ?,
                       summary_rewrites = ?,
                       prunes = ?,
                       deferrals = ?,
                       no_action = ?,
                       subprocess_exit_code = ?,
                       errors = ?
                   WHERE id = ?""",
                (
                    now,
                    len(flag_ids or []),
                    fid_json,
                    int(c.get("merges", 0)),
                    int(c.get("invalidations", 0)),
                    int(c.get("links_created", 0)),
                    int(c.get("edges_proposed", 0)),
                    int(c.get("summary_rewrites", 0)),
                    int(c.get("prunes", 0)),
                    int(c.get("deferrals", 0)),
                    int(c.get("no_action", 0)),
                    subprocess_exit_code,
                    str(errors or ""),
                    int(run_id),
                ),
            )

    def get_entity(self, name: str):
        """Get entity details by name. Returns dict or None if not found."""
        eid = self._entity_id(name)
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM entities WHERE id = ? AND status = 'active'", (eid,)
        ).fetchone()
        if not row:
            return None
        # kind column may not exist in very old DBs -- fall back to type
        kind = "entity"
        try:
            kind = row["kind"] or "entity"
        except (IndexError, KeyError):
            pass
        # v3.7.42 FINDING #Z (Adrian pass 8 deep audit, 2026-05-19):
        # include created_at in the returned dict. The SQL column has
        # existed since the initial schema (migrations/001) but the
        # dict projection here omitted it. v3.7.40's _fetch_entity_
        # details_kg_fallback (kg_query date surface bridge) and
        # v3.7.41's tool_kg_list_declared (declared-entity date
        # surface) both read entity.get('created_at') expecting the
        # write-time stamp -- both got None silently for two ships
        # before the live-test caught it. Mocked unit tests in
        # test_kg_query_dates.py PASSED because the fake KG dict
        # carried created_at; real KG didn't. Adding it here unblocks
        # both prior fixes without further changes.
        created_at = ""
        try:
            created_at = row["created_at"] or ""
        except (IndexError, KeyError):
            pass
        return {
            "id": row["id"],
            "name": row["name"],
            "type": row["type"],
            "kind": kind,
            "content": row["content"] or "",
            "importance": row["importance"] or 3,
            "created_at": created_at,
            "last_touched": row["last_touched"] or "",
            "status": row["status"],
            "properties": json.loads(row["properties"]) if row["properties"] else {},
        }

    def list_entities(self, status: str = "active", kind: str = None):
        """List all entities with the given status, optionally filtered by kind.

        Args:
            status: 'active', 'merged', 'deprecated' (default 'active')
            kind: 'entity', 'predicate', 'class', 'literal' (default None = all)
        """
        conn = self._conn()
        if kind:
            rows = conn.execute(
                "SELECT * FROM entities WHERE status = ? AND kind = ? ORDER BY importance DESC, last_touched DESC",
                (status, kind),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM entities WHERE status = ? ORDER BY importance DESC, last_touched DESC",
                (status,),
            ).fetchall()
        results = []
        for row in rows:
            row_kind = "entity"
            try:
                row_kind = row["kind"] or "entity"
            except (IndexError, KeyError):
                pass
            results.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "type": row["type"],
                    "kind": row_kind,
                    "content": row["content"] or "",
                    "importance": row["importance"] or 3,
                    "last_touched": row["last_touched"] or "",
                }
            )
        return results

    def _refresh_body_view(self, entity_id: str) -> str:
        """Re-emit the ``{entity_id}__body`` Level-3 vec row for one
        entity to match its current ``entities.content`` value.

        v3.7.32 (Adrian directive 2026-05-18, FINDING #R fix). v3.7.29
        added inline __body emission to ``mint_entity`` (new-write
        path) but the UPDATE paths -- ``update_entity_content``,
        ``merge_entities``'s post-merge content rewrite, and any
        future caller mutating ``entities.content`` -- did not
        refresh the existing __body row. Symptom: 46 entities drifted
        (stored __body reflected old content) ~2.5h after the
        v3.7.29 retrofit. Workaround was a periodic force=True
        backfill. v3.7.32 wires this helper into every UPDATE site
        so the __body view stays consistent with content on every
        write.

        Behavior (idempotent):
          * content empty AND no __body row -> no-op.
          * content empty AND __body exists -> delete __body row
            (cleanup of stale vector when content cleared).
          * content == rendered_summary -> delete __body row if
            present (avoids duplicate L2/L3 vectors).
          * content distinct from rendered_summary -> embed +
            upsert __body row with view_kind='body' + view_index=-2
            metadata (matches backfill_l3_body_views shape exactly).

        Best-effort: embedder/vec-store failures are logged via the
        hook_errors path; the SQLite row is the source of truth and
        a subsequent ``backfill_l3_body_views`` run will recover.
        Returns a short status string ('upserted' / 'deleted' /
        'skipped_no_content' / 'skipped_duplicate_of_summary' /
        'skipped_no_embedder' / 'skipped_no_vectorstore' / 'error')
        for callers that want to log the action.
        """
        import json as _json  # noqa: PLC0415
        import os as _os  # noqa: PLC0415

        # 1. Read current content + rendered_summary from the row.
        row = (
            self._conn()
            .execute(
                "SELECT content, properties FROM entities WHERE id = ?",
                (entity_id,),
            )
            .fetchone()
        )
        if row is None:
            return "skipped_no_entity"
        content = (
            (row["content"] or "").strip()
            if isinstance(row, dict) or hasattr(row, "keys")
            else (row[0] or "").strip()
        )
        props_raw = row["properties"] if (isinstance(row, dict) or hasattr(row, "keys")) else row[1]
        rendered_summary = ""
        if props_raw:
            try:
                pd = _json.loads(props_raw) if isinstance(props_raw, str) else props_raw
                if isinstance(pd, dict):
                    sd = pd.get("summary")
                    if isinstance(sd, dict):
                        rendered_summary = serialize_summary_for_embedding(sd).strip()
            except Exception:
                rendered_summary = ""

        # 2. Open the vec store.
        try:
            from mempalace.vector_store import RECORDS_COLLECTION, get_vector_store

            palace_path = _os.path.dirname(_os.path.abspath(self.db_path))
            vs = get_vector_store(palace_path)
        except Exception:
            return "skipped_no_vectorstore"

        body_id = f"{entity_id}__body"

        # 3. Decide upsert vs delete based on distinctness.
        if not content or content == rendered_summary:
            # Cleanup case: __body should not exist for this entity.
            try:
                vs.delete(RECORDS_COLLECTION, ids=[body_id])
                return "deleted" if not content else "skipped_duplicate_of_summary"
            except Exception:
                return "skipped_no_content" if not content else "skipped_duplicate_of_summary"

        # 4. Upsert path: embed current content, write the row.
        try:
            from mempalace.embedder import get_default_embedder

            embedder = get_default_embedder()
            if embedder is None:
                return "skipped_no_embedder"
        except Exception:
            return "skipped_no_embedder"

        # Mirror the backfill helper's truncation ceiling: MiniLM-L6
        # has a 256-token cap; ~1800 chars is the empirical safe
        # ceiling (also used by _add_memory_internal).
        _EMBED_DOC_MAX_CHARS = 1800
        body_doc = content
        if len(body_doc) > _EMBED_DOC_MAX_CHARS:
            body_doc = body_doc[: _EMBED_DOC_MAX_CHARS - 1].rstrip() + "..."

        # Look up kind/name/importance from the row for metadata.
        meta_row = (
            self._conn()
            .execute(
                "SELECT name, kind, importance FROM entities WHERE id = ?",
                (entity_id,),
            )
            .fetchone()
        )
        # v3.7.38 FINDING #V: stamp date fields on body view refresh
        # too. v3.7.32 added this helper but its metadata only carried
        # the view-shape fields; without date_added the body view rows
        # produced by update_entity_content / merge_entities perpetuated
        # the same FINDING #U gap until v3.7.35's SQL fresh-fetch
        # bridged them on retrieval. Stamping at write time keeps the
        # bridge as the safety net for legacy rows only.
        _refresh_now = datetime.now().isoformat()
        meta = {
            "name": (meta_row["name"] if meta_row else entity_id) or entity_id,
            "kind": (meta_row["kind"] if meta_row else "entity") or "entity",
            "importance": int((meta_row["importance"] if meta_row else 3) or 3),
            "entity_id": entity_id,
            "view_kind": "body",
            "view_index": -2,
            "date_added": _refresh_now,
            "last_relevant_at": _refresh_now,
            "last_touched": _refresh_now,
        }
        try:
            emb = embedder([body_doc])
            if not emb:
                return "error"
            vs.upsert(
                RECORDS_COLLECTION,
                ids=[body_id],
                documents=[body_doc],
                metadatas=[meta],
                embeddings=[emb[0]],
            )
            return "upserted"
        except Exception:
            return "error"

    def update_entity_content(self, name: str, content: str, importance: int = None):
        """Update an entity's content (and optionally importance). Returns the entity.

        Canonical method as of migration 023 (2026-04-29). The legacy
        ``update_entity_description`` was removed; the rename is complete.

        v3.7.32 (FINDING #R fix 2026-05-18): every successful content
        mutation triggers ``_refresh_body_view(eid)`` so the Level-3
        ``{eid}__body`` vec row stays consistent with the new content
        without waiting for a force=True backfill. Idempotent + safe
        when the embedder or vec store is unavailable.
        """
        eid = self._entity_id(name)
        now = datetime.now().isoformat()
        conn = self._conn()
        with conn:
            if importance is not None:
                conn.execute(
                    "UPDATE entities SET content = ?, importance = ?, last_touched = ? WHERE id = ?",
                    (content, importance, now, eid),
                )
            else:
                conn.execute(
                    "UPDATE entities SET content = ?, last_touched = ? WHERE id = ?",
                    (content, now, eid),
                )
        # v3.7.32 FINDING #R: re-emit the Level-3 body view so the
        # stored vector reflects the new content instead of drifting
        # vs the SQLite source of truth.
        try:
            self._refresh_body_view(eid)
        except Exception:
            # Body-view refresh is best-effort; SQLite is the source
            # of truth and a backfill_l3_body_views(force=True) call
            # recovers if the inline refresh ever silently no-ops.
            pass
        return self.get_entity(name)

    def update_entity_properties(self, name: str, properties: dict):
        """Merge new properties into an entity's existing properties."""
        eid = self._entity_id(name)
        conn = self._conn()
        row = conn.execute("SELECT properties FROM entities WHERE id = ?", (eid,)).fetchone()
        if not row:
            return None
        existing = json.loads(row["properties"]) if row["properties"] else {}
        # Deep-merge `rules_profile` (mirrors tool_mutate.tool_kg_update_entity):
        # a partial update -- e.g. only tool_permissions, or only slots -- must
        # NOT drop the sibling sub-key. A plain top-level update replaced the
        # whole rules_profile, which is how intent-type classes lost their
        # slots (the recurring "has no slots defined" bug). Incoming sub-keys
        # win; absent sub-keys are preserved.
        for _pk, _pv in properties.items():
            _ev = existing.get(_pk)
            if _pk == "rules_profile" and isinstance(_ev, dict) and isinstance(_pv, dict):
                _merged = dict(_ev)
                _merged.update(_pv)
                existing[_pk] = _merged
            else:
                existing[_pk] = _pv
        now = datetime.now().isoformat()
        with conn:
            conn.execute(
                "UPDATE entities SET properties = ?, last_touched = ? WHERE id = ?",
                (json.dumps(existing), now, eid),
            )
        return self.get_entity(name)

    def entity_edge_count(self, name: str) -> int:
        """Count active edges (triples) involving an entity."""
        eid = self._entity_id(name)
        conn = self._conn()
        row = conn.execute(
            "SELECT COUNT(*) as n FROM triples WHERE (subject = ? OR object = ?) AND valid_to IS NULL",
            (eid, eid),
        ).fetchone()
        return row["n"] if row else 0

    # ── Query operations ──────────────────────────────────────────────────

    def query_entity(self, name: str, as_of: str = None, direction: str = "outgoing"):
        """
        Get all relationships for an entity.

        direction: "outgoing" (entity → ?), "incoming" (? → entity), "both"
        as_of: date string -- only return facts valid at that time
        """
        eid = self._entity_id(name)
        conn = self._conn()

        results = []

        if direction in ("outgoing", "both"):
            query = "SELECT t.*, e.name as obj_name FROM triples t JOIN entities e ON t.object = e.id WHERE t.subject = ?"
            params = [eid]
            if as_of:
                query += " AND (t.valid_from IS NULL OR t.valid_from <= ?) AND (t.valid_to IS NULL OR t.valid_to >= ?)"
                params.extend([as_of, as_of])
            for row in conn.execute(query, params).fetchall():
                fact = {
                    "direction": "outgoing",
                    "subject": name,
                    "predicate": row["predicate"],
                    "object": row["obj_name"],
                    "valid_from": row["valid_from"],
                    "valid_to": row["valid_to"],
                    "confidence": row["confidence"],
                    "current": row["valid_to"] is None,
                    # Added for Channel B triple emission: BFS walkers
                    # emit the traversed triple itself (not just the
                    # neighbour entity) into the fused ranking so
                    # triples get RRF cross-channel boost. Old callers
                    # that iterate known keys are unaffected -- these
                    # are additive.
                    "triple_id": row["id"],
                    "statement": row["statement"],
                }
                # render-time text fallback so
                # every kg_query fact row carries a natural-language
                # display string. When statement is absent the helper
                # synthesizes it from (subject, predicate, object).
                # Vocab lock 2026-05-01: rendered triple prose lives under
                # "statement_text" everywhere -- mirrors kg_add's response
                # echo and tool_read.py's triple-channel projection. The
                # underlying SQL column is still "statement" (raw stored
                # form); "statement_text" is the rendered form, possibly
                # synthesized when the column is null.
                fact["statement_text"] = _render_fact_display(fact)
                results.append(fact)

        if direction in ("incoming", "both"):
            query = "SELECT t.*, e.name as sub_name FROM triples t JOIN entities e ON t.subject = e.id WHERE t.object = ?"
            params = [eid]
            if as_of:
                query += " AND (t.valid_from IS NULL OR t.valid_from <= ?) AND (t.valid_to IS NULL OR t.valid_to >= ?)"
                params.extend([as_of, as_of])
            for row in conn.execute(query, params).fetchall():
                fact = {
                    "direction": "incoming",
                    "subject": row["sub_name"],
                    "predicate": row["predicate"],
                    "object": name,
                    "valid_from": row["valid_from"],
                    "valid_to": row["valid_to"],
                    "confidence": row["confidence"],
                    "current": row["valid_to"] is None,
                    "triple_id": row["id"],
                    "statement": row["statement"],
                }
                # Vocab lock 2026-05-01: rendered triple prose lives under
                # "statement_text" everywhere -- mirrors kg_add's response
                # echo and tool_read.py's triple-channel projection. The
                # underlying SQL column is still "statement" (raw stored
                # form); "statement_text" is the rendered form, possibly
                # synthesized when the column is null.
                fact["statement_text"] = _render_fact_display(fact)
                results.append(fact)

        return results

    # ── BM25-IDF keyword signals (P3 follow-up) ──
    def record_keyword_observations(self, keywords, *, recompute_idf: bool = True):
        """Bump freq for each keyword observed on a new record memory.

        Called by _add_memory_internal on record writes so the BM25-IDF
        table stays incrementally up to date. Recomputes idf for every
        keyword whose freq changed (cheap -- one log per bumped row).

        IDF formula (Robertson & Jones 1976; Robertson & Zaragoza 2009
        "Foundations of BM25 and Beyond"):

            idf(t) = log((N - freq(t) + 0.5) / (freq(t) + 0.5))

        where N is the total number of record-kind memories. Rare terms
        get large positive idf; dominant terms near N approach 0 or
        negative (the keyword channel clamps at min_idf=0.5 downstream).
        """
        import math

        if not keywords:
            return
        cleaned = list({k.strip() for k in keywords if isinstance(k, str) and k.strip()})
        if not cleaned:
            return
        conn = self._conn()
        now = datetime.now().isoformat()
        try:
            with conn:
                for kw in cleaned:
                    conn.execute(
                        """INSERT INTO keyword_idf (keyword, freq, idf, last_updated_ts)
                           VALUES (?, 1, 0.0, ?)
                           ON CONFLICT(keyword) DO UPDATE SET
                             freq = freq + 1,
                             last_updated_ts = excluded.last_updated_ts""",
                        (kw, now),
                    )
                if recompute_idf:
                    n_row = conn.execute(
                        "SELECT COUNT(*) FROM entities WHERE kind='record' AND status='active'"
                    ).fetchone()
                    total_n = int((n_row[0] if n_row else 0) or 0)
                    if total_n > 0:
                        for kw in cleaned:
                            f_row = conn.execute(
                                "SELECT freq FROM keyword_idf WHERE keyword=?", (kw,)
                            ).fetchone()
                            if not f_row:
                                continue
                            f = int(f_row[0] or 0)
                            # BM25 robust IDF (log stays positive by adding 1.0
                            # inside, so even dominant terms have a floor at 0).
                            idf = math.log(max(0.0, (total_n - f + 0.5) / (f + 0.5)) + 1.0)
                            conn.execute(
                                "UPDATE keyword_idf SET idf=? WHERE keyword=?",
                                (round(idf, 6), kw),
                            )
        except sqlite3.OperationalError:
            # keyword_idf table absent (pre-migration-016 DB) -- no-op.
            pass

    def get_keyword_idf(self, keywords) -> dict:
        """Return {keyword: idf} for each requested keyword (0.0 for unseen)."""
        if not keywords:
            return {}
        cleaned = list({k.strip() for k in keywords if isinstance(k, str) and k.strip()})
        if not cleaned:
            return {}
        conn = self._conn()
        result = {kw: 0.0 for kw in cleaned}
        try:
            placeholders = ",".join("?" for _ in cleaned)
            rows = conn.execute(
                f"SELECT keyword, idf FROM keyword_idf WHERE keyword IN ({placeholders})",
                cleaned,
            ).fetchall()
            for kw, idf in rows:
                try:
                    result[kw] = float(idf or 0.0)
                except (TypeError, ValueError):
                    continue
        except sqlite3.OperationalError:
            return result
        return result

    def recompute_keyword_idf_all(self):
        """Full recompute across every keyword in keyword_idf.

        O(rows). Call once after a bulk backfill, or in a maintenance
        path. For the per-write hot path, use record_keyword_observations
        which only recomputes the affected keywords.
        """
        import math

        conn = self._conn()
        try:
            n_row = conn.execute(
                "SELECT COUNT(*) FROM entities WHERE kind='record' AND status='active'"
            ).fetchone()
            total_n = int((n_row[0] if n_row else 0) or 0)
            if total_n <= 0:
                return
            rows = conn.execute("SELECT keyword, freq FROM keyword_idf").fetchall()
            updates = []
            for keyword, freq in rows:
                f = int(freq or 0)
                idf = math.log(max(0.0, (total_n - f + 0.5) / (f + 0.5)) + 1.0)
                updates.append((round(idf, 6), keyword))
            if updates:
                with conn:
                    conn.executemany("UPDATE keyword_idf SET idf=? WHERE keyword=?", updates)
        except sqlite3.OperationalError:
            return

    def triples_created_under(self, context_id: str) -> list:
        """Return triple_ids whose creation_context_id points at this context.

        Triples aren't materialised as entity rows (no kind='triple'
        entity), so a standard ``kg_query`` on a context won't return
        them via ``created_under`` edges -- there are none to triples.
        This is the triples-layer analogue of the memory/entity
        ``created_under`` edge walk: "which triples were written under
        this context."
        """
        if not context_id:
            return []
        conn = self._conn()
        rows = conn.execute(
            "SELECT id FROM triples WHERE creation_context_id=? "
            "AND (valid_to IS NULL OR valid_to='')",
            (context_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_entity_degree(self, entity_id: str) -> int:
        """Total in-degree + out-degree for an entity in the current triples.

        Used by Channel B's degree-dampening: mega-hub entities (like the
        agent's own id) would otherwise flood graph-channel results with
        their many neighbours. Each seed→memory contribution is weighted
        by ``1 / log(degree + 2)``, so a degree-50 hub contributes roughly
        a quarter of what a degree-2 specialist does.

        References:
          Hogan et al. "Knowledge Graphs." arXiv:2003.02320 (2021).
          West & Leskovec. "Human wayfinding in information networks."
            WWW 2012 -- inverse-log degree term is the standard dampening
            shape for random-walk over KGs.
          Bollacker et al. "Freebase." SIGMOD 2008 -- same dampening for
            popular entities.
        """
        if not entity_id:
            return 0
        eid = self._entity_id(entity_id)
        conn = self._conn()
        try:
            out_degree = conn.execute(
                "SELECT COUNT(*) FROM triples WHERE subject=? "
                "AND (valid_to IS NULL OR valid_to='')",
                (eid,),
            ).fetchone()[0]
            in_degree = conn.execute(
                "SELECT COUNT(*) FROM triples WHERE object=? AND (valid_to IS NULL OR valid_to='')",
                (eid,),
            ).fetchone()[0]
        except Exception:
            return 0
        return int(out_degree or 0) + int(in_degree or 0)

    def get_similar_contexts(self, context_id: str, hops: int = 4, decay: float = 0.85) -> list:
        """BFS ``similar_to`` neighbourhood of a context, with distance decay.

        Returns ``[(neighbour_context_id, accumulated_sim), …]`` sorted by
        accumulated_sim descending. 1-hop contributes ``sim``; 2-hop
        contributes ``sim * decay * parent_sim``; k-hop contributes
        ``sim * decay^(k-1) * product_of_parent_sims``. Early termination
        when a path's accumulated sim falls below 1e-4 (numerical noise
        floor; literature uses thresholds at this magnitude for
        convergence checks, e.g. PageRank power-iteration ||x_{k+1}-x_k||
        < 1e-6 is the canonical equivalent).

        Edge similarity is read from the ``confidence`` column (P1
        convention -- see ``context_lookup_or_create`` in mcp_server.py).

        DEFAULTS (literature-canonical, 2026-05-02 followup after Adrian's
        damping-factor literature audit):
          decay = 0.85 -- canonical PageRank teleport-complement (Brin &
            Page 1998), confirmed across Personalized PageRank / APPNP
            (Klicpera et al. 2019, ICLR) and Random Walk with Restart
            (Tong, Faloutsos, Pan 2006). Pre-2026-05-02 default was 0.5
            which was at the aggressive end of the literature spectrum
            (50% per-hop retention vs canonical 85%); previous 0.5 cut
            indirect contexts to 25% by hop 3 vs 72% under canonical 0.85.
          hops = 4 -- safety cap. PageRank uses K=10+ iterations to
            converge to a stationary distribution, but those are full
            graph-wide power-iteration steps. For a BFS walk from a
            single seed in a sparse similar_to graph, 4 hops captures
            the meaningful neighbourhood; the 0.85 decay handles
            diminishing returns naturally past that.

        Consumed by Channel D (retrieval, P2) to expand the context
        neighbourhood around the active context. Shipping the helper in
        P1 keeps the traversal unit-testable in isolation.
        """
        if not context_id or hops < 1:
            return []
        eid = self._entity_id(context_id)
        conn = self._conn()
        visited = {eid}
        # frontier: list of (current_context_id, accumulated_sim_so_far)
        frontier = [(eid, 1.0)]
        accumulated: dict = {}
        depth_decay = 1.0
        for depth in range(hops):
            if not frontier:
                break
            depth_decay *= decay if depth > 0 else 1.0
            next_frontier = []
            for cur_id, cur_sim in frontier:
                rows = conn.execute(
                    "SELECT object, confidence FROM triples "
                    "WHERE subject=? AND predicate='similar_to' "
                    "AND (valid_to IS NULL OR valid_to = '')",
                    (cur_id,),
                ).fetchall()
                for row in rows:
                    neighbour = row["object"]
                    if neighbour in visited:
                        continue
                    edge_sim = float(row["confidence"] or 0.0)
                    if edge_sim <= 0.0:
                        continue
                    contribution = cur_sim * edge_sim * depth_decay
                    if contribution < 1e-4:
                        continue
                    # Keep max contribution if the same neighbour is reached
                    # by multiple paths at different depths.
                    prev = accumulated.get(neighbour, 0.0)
                    if contribution > prev:
                        accumulated[neighbour] = contribution
                    visited.add(neighbour)
                    next_frontier.append((neighbour, contribution))
            frontier = next_frontier

        return sorted(accumulated.items(), key=lambda kv: kv[1], reverse=True)

    def query_relationship(self, predicate: str, as_of: str = None):
        """Get all triples with a given relationship type."""
        pred = _normalize_predicate(predicate)
        conn = self._conn()
        query = """
            SELECT t.*, s.name as sub_name, o.name as obj_name
            FROM triples t
            JOIN entities s ON t.subject = s.id
            JOIN entities o ON t.object = o.id
            WHERE t.predicate = ?
        """
        params = [pred]
        if as_of:
            query += " AND (t.valid_from IS NULL OR t.valid_from <= ?) AND (t.valid_to IS NULL OR t.valid_to >= ?)"
            params.extend([as_of, as_of])

        results = []
        for row in conn.execute(query, params).fetchall():
            results.append(
                {
                    "subject": row["sub_name"],
                    "predicate": pred,
                    "object": row["obj_name"],
                    "valid_from": row["valid_from"],
                    "valid_to": row["valid_to"],
                    "current": row["valid_to"] is None,
                }
            )
        return results

    def timeline(self, entity_name: str = None):
        """Get all facts in chronological order, optionally filtered by entity."""
        conn = self._conn()
        if entity_name:
            eid = self._entity_id(entity_name)
            rows = conn.execute(
                """
                SELECT t.*, s.name as sub_name, o.name as obj_name
                FROM triples t
                JOIN entities s ON t.subject = s.id
                JOIN entities o ON t.object = o.id
                WHERE (t.subject = ? OR t.object = ?)
                ORDER BY t.valid_from ASC NULLS LAST
                LIMIT 100
            """,
                (eid, eid),
            ).fetchall()
        else:
            rows = conn.execute("""
                SELECT t.*, s.name as sub_name, o.name as obj_name
                FROM triples t
                JOIN entities s ON t.subject = s.id
                JOIN entities o ON t.object = o.id
                ORDER BY t.valid_from ASC NULLS LAST
                LIMIT 100
            """).fetchall()

        return [
            {
                "subject": r["sub_name"],
                "predicate": r["predicate"],
                "object": r["obj_name"],
                "valid_from": r["valid_from"],
                "valid_to": r["valid_to"],
                "current": r["valid_to"] is None,
            }
            for r in rows
        ]

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self):
        conn = self._conn()
        entities = conn.execute("SELECT COUNT(*) as cnt FROM entities").fetchone()["cnt"]
        triples = conn.execute("SELECT COUNT(*) as cnt FROM triples").fetchone()["cnt"]
        current = conn.execute(
            "SELECT COUNT(*) as cnt FROM triples WHERE valid_to IS NULL"
        ).fetchone()["cnt"]
        expired = triples - current
        predicates = [
            r["predicate"]
            for r in conn.execute(
                "SELECT DISTINCT predicate FROM triples ORDER BY predicate"
            ).fetchall()
        ]
        return {
            "entities": entities,
            "triples": triples,
            "current_facts": current,
            "expired_facts": expired,
            "relationship_types": predicates,
        }

    # ── Seed from known facts ─────────────────────────────────────────────

    def seed_from_entity_facts(self, entity_facts: dict):
        """
        Seed the knowledge graph from fact_checker.py ENTITY_FACTS.
        This bootstraps the graph with known ground truth.

        Cold-start lock 2026-05-01: pre-declare every entity REFERENCED
        by a triple before writing the triple. Pre-cold-start, the
        loop below called add_triple(name, pred, target) where `target`
        was a name capitalized inline (parent / partner / sibling /
        owner / interest); add_triple's INSERT OR IGNORE phantom path
        silently created those targets with no kind, no summary. The
        gate's hard-reject (entity_gate.assert_entity_exists) closes
        that surface, so the seeder must declare its own targets.
        """
        # Pass 1: collect every name that will be referenced as a triple
        # endpoint and declare each as an entity so add_triple's
        # assert_entity_exists check passes. Idempotent via INSERT OR
        # REPLACE in add_entity.
        _all_names: dict[str, str] = {}  # name -> kind hint
        for key, facts in entity_facts.items():
            name = facts.get("full_name", key.capitalize())
            kind_hint = "animal" if facts.get("relationship") == "dog" else "entity"
            _all_names[name] = kind_hint
            for ref_field in ("parent", "partner", "sibling", "owner"):
                ref_val = facts.get(ref_field)
                if ref_val:
                    _all_names.setdefault(ref_val.capitalize(), "entity")
            for interest in facts.get("interests") or []:
                _all_names.setdefault(interest.capitalize(), "entity")
        for _ref_name, _ref_kind in _all_names.items():
            try:
                if not self.get_entity(self._entity_id(_ref_name)):
                    self.add_entity(
                        _ref_name,
                        kind=_ref_kind,
                        content=f"{_ref_name} (auto-declared by seed_from_entity_facts)",
                    )
            except Exception:
                pass

        for key, facts in entity_facts.items():
            name = facts.get("full_name", key.capitalize())
            self.add_entity(
                name,
                kind="entity",
                content=f"{name} ({facts.get('type', 'person')})",
                properties={
                    "gender": facts.get("gender", ""),
                    "birthday": facts.get("birthday", ""),
                },
            )

            # Relationships. Each add_triple supplies a statement on
            # non-skip predicates (TripleStatementRequired policy). The
            # sentences are derived from the known-fact dict at this
            # seed layer; this is still caller-written (by the
            # fact_checker author via seed_from_entity_facts) rather
            # than autogenerated at embed time.
            parent = facts.get("parent")
            if parent:
                self.add_triple(
                    name,
                    "child_of",
                    parent.capitalize(),
                    valid_from=facts.get("birthday"),
                    statement=f"{name} is the child of {parent.capitalize()}.",
                )

            partner = facts.get("partner")
            if partner:
                self.add_triple(
                    name,
                    "married_to",
                    partner.capitalize(),
                    statement=f"{name} is married to {partner.capitalize()}.",
                )

            relationship = facts.get("relationship", "")
            if relationship == "daughter":
                parent_name = facts.get("parent", "").capitalize() or name
                self.add_triple(
                    name,
                    "is_child_of",
                    parent_name,
                    valid_from=facts.get("birthday"),
                    statement=f"{name} is the child of {parent_name}.",
                )
            elif relationship == "husband":
                partner_name = facts.get("partner", name).capitalize()
                self.add_triple(
                    name,
                    "is_partner_of",
                    partner_name,
                    statement=f"{name} is the partner of {partner_name}.",
                )
            elif relationship == "brother":
                sibling_name = facts.get("sibling", name).capitalize()
                self.add_triple(
                    name,
                    "is_sibling_of",
                    sibling_name,
                    statement=f"{name} is a sibling of {sibling_name}.",
                )
            elif relationship == "dog":
                owner_name = facts.get("owner", name).capitalize()
                self.add_triple(
                    name,
                    "is_pet_of",
                    owner_name,
                    statement=f"{name} is a pet of {owner_name}.",
                )
                self.add_entity(name, "animal")

            # Interests
            for interest in facts.get("interests", []):
                interest_cap = interest.capitalize()
                self.add_triple(
                    name,
                    "loves",
                    interest_cap,
                    valid_from="2025-01-01",
                    statement=f"{name} loves {interest_cap}.",
                )
