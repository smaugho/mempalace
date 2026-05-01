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

    - ``what`` (required, ≥5 chars after strip): noun phrase naming
      the entity.
    - ``why`` (required, ≥15 chars after strip): purpose / role /
      claim clause -- not a name restatement.
    - ``scope`` (optional, ≤100 chars): temporal / domain qualifier.
    - The rendered prose form (``serialize_summary_for_embedding``)
      must fit within ``_SUMMARY_MAX_LEN`` (280 chars) so it stays
      a focused embedding view.

    Returns ``True`` on success. Raises ``SummaryStructureRequired``
    with a precise message naming the failing field plus the call
    site (``context_for_error``).

    Validation is intentionally STRUCTURAL only -- fields present,
    non-empty, length-bounded. No regex on prose, no role-verb
    detection, no em-dash heuristics. Stubs and name-restating
    placeholders are caught by the length floors on ``why``;
    deeper semantic quality is the gardener's job (Haiku-driven
    generic-summary flag pipeline).

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


def coerce_summary_for_persist(summary, *, context_for_error: str = "summary"):
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

    Strings raise -- see ``validate_summary`` for the migration path.
    """
    from .ascii_fold import fold_summary  # local import -- avoids circular import at module load

    folded = fold_summary(summary)
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


# ── Slice 1b 2026-04-28: render-time fact display ──
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
            return client.get_or_create_collection(
                TRIPLE_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
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

        # Seed canonical ontology on first run (no "thing" class yet)
        # Only for production palaces -- test KGs are empty by design
        if not os.environ.get("MEMPALACE_SKIP_SEED"):
            self.seed_ontology()

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
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def seed_ontology(self):
        """Seed canonical classes, predicates, and intent types. Idempotent.

        Called automatically on first run (empty entities table) or on demand.
        Uses add_entity + add_triple, so normalization and schema are consistent.
        """
        conn = self._conn()
        # Check if ontology already seeded (look for root class "thing")
        thing = conn.execute("SELECT id FROM entities WHERE id = 'thing'").fetchone()
        if thing:
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
            self.add_entity(
                name,
                kind="class",
                content=content,
                importance=imp,
                properties={"summary": summary},
            )
            if name != "thing":
                self.add_triple(name, "is_a", "thing")

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

            # Update target content if provided (already rendered prose).
            if summary:
                conn.execute(
                    "UPDATE entities SET content = ?, last_touched = ? WHERE id = ?",
                    (summary, now, target_id),
                )

            # Touch target
            conn.execute("UPDATE entities SET last_touched = ? WHERE id = ?", (now, target_id))

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
        if rater_kind not in ("agent", "gate_llm"):
            raise ValueError(f"rater_kind must be 'agent' or 'gate_llm', got {rater_kind!r}")
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
                        written += 1
                    except sqlite3.IntegrityError:
                        # Unique partial index fired: same unresolved
                        # (kind, mkey, ctx) already present -- bump
                        # the existing row's attempted_count instead.
                        conn.execute(
                            """UPDATE memory_flags
                               SET attempted_count = attempted_count + 1,
                                   last_attempt_ts = ?
                               WHERE kind = ? AND memory_key = ?
                                     AND context_id = ?
                                     AND resolved_ts IS NULL""",
                            (now, kind, mkey, cid),
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
        return {
            "id": row["id"],
            "name": row["name"],
            "type": row["type"],
            "kind": kind,
            "content": row["content"] or "",
            "importance": row["importance"] or 3,
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

    def update_entity_content(self, name: str, content: str, importance: int = None):
        """Update an entity's content (and optionally importance). Returns the entity.

        Canonical method as of migration 023 (2026-04-29). The legacy
        ``update_entity_description`` was removed; the rename is complete.
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
        return self.get_entity(name)

    def update_entity_properties(self, name: str, properties: dict):
        """Merge new properties into an entity's existing properties."""
        eid = self._entity_id(name)
        conn = self._conn()
        row = conn.execute("SELECT properties FROM entities WHERE id = ?", (eid,)).fetchone()
        if not row:
            return None
        existing = json.loads(row["properties"]) if row["properties"] else {}
        existing.update(properties)
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
                # Slice 1b 2026-04-28: render-time text fallback so
                # every kg_query fact row carries a natural-language
                # display string. When statement is absent the helper
                # synthesizes it from (subject, predicate, object).
                fact["text"] = _render_fact_display(fact)
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
                fact["text"] = _render_fact_display(fact)
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

    def get_similar_contexts(self, context_id: str, hops: int = 2, decay: float = 0.5) -> list:
        """BFS ``similar_to`` neighbourhood of a context, with distance decay.

        Returns ``[(neighbour_context_id, accumulated_sim), …]`` sorted by
        accumulated_sim descending. 1-hop contributes ``sim``; 2-hop
        contributes ``sim * decay * parent_sim``; 3-hop would contribute
        ``sim * decay² * parent_sim * grandparent_sim``. Early termination
        when a path's accumulated sim falls below 1e-4.

        Edge similarity is read from the ``confidence`` column (P1
        convention -- see ``context_lookup_or_create`` in mcp_server.py).

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
