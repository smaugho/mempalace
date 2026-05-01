"""Single chokepoint for entity creation -- the cold-start gate.

Adrian's design lock 2026-05-01: every entity row in SQLite must be
minted through ``mint_entity`` (or the ``record`` path, which goes
through ``_add_memory_internal`` and validates summary on its own).
Every triple/edge that references an entity must be guarded by
``assert_entity_exists`` if the caller didn't just mint it.

Background
----------
The pre-cold-start audit (2026-05-01) catalogued **12 distinct
entity-creation surfaces** that bypassed the summary contract:

  - 4 phantom sites: ``add_triple``/``add_rated_edge`` silently
    ``INSERT OR IGNORE INTO entities`` for any unknown subject/object
    name. Result: 1,780 entities in the live corpus with no ``is_a``
    edge, no kind, no summary.
  - 8 partial-bypass sites: ``tool_diary_write``,
    ``auto_declare_files``, ``finalize_intent`` group,
    ``declare_user_intents``, ``wake_up`` bootstrap,
    ``memory_gardener``, ``link_author``, plus seed/system bootstrap.

This module collapses all 12 onto one gate:

  * ``mint_entity(name, kind, summary={what,why,scope?}, queries=[...])``
    -- the single permitted entity-creation primitive. Handles
    summary validation, identity-collision detection, 3-level Chroma
    write (identity + abstract + probes).
  * ``assert_entity_exists(eid, conn)`` -- the hard-reject helper for
    sites that *reference* (don't create) entities. Replaces the
    phantom ``INSERT OR IGNORE`` pattern at ``add_triple`` /
    ``add_rated_edge``.

Three-level identity model (locked 2026-04-30)
----------------------------------------------
::

    Level 1  IDENTITY  : embed(what)                  -> equality (T_REUSE=0.92)
    Level 2  ABSTRACT  : embed(rendered prose form)   -> similarity contribution
    Level 4  PROBES    : embed(queries[i]) Nx         -> similarity primary + retrieval

Level 3 BODY (200-1000 word long-form) is intentionally **dropped** at
cold-start: cost (~$9 + 3-5h authoring + ongoing per-write latency)
exceeded value because no CE-rerank consumer exists for entities. Re-
introduce only when a concrete consumer ships.

Identity collision policy
-------------------------
Cosine on Level-1 identity embeddings:

  cosine ≥ T_REUSE_WHAT (0.92)
      Silent reuse. ``mint_entity`` returns the existing eid + ``True``
      for ``was_reused``.

  T_COLLISION_WARN (0.85) ≤ cosine < T_REUSE_WHAT (0.92)
      Register an ``entity_collision`` entry in ``_STATE.pending_conflicts``
      so the agent must call ``mempalace_resolve_conflicts``.
      ``collision_policy='strict'`` (agent-initiated paths) raises
      ``EntityCollisionError``; ``collision_policy='soft'`` (system-
      internal paths -- gardener, link_author, seed) logs the conflict
      and proceeds with creation so the bootstrap doesn't deadlock.

  cosine < T_COLLISION_WARN
      Proceed with new entity creation, no flag.

Stoplist on ``what``
--------------------
``what`` is the noun phrase used as the Level-1 identity embedding.
Generic phrases ("the project", "a thing") would create non-
discriminative identity vectors that cluster together at ~0.99 cosine,
defeating the gate. The conservative stoplist below catches the common
offenders; the gardener's Haiku-driven generic-summary flag pipeline
handles the long tail.

References
----------
* Khattab & Zaharia 2020 -- ColBERT (arXiv:2004.12832): late-interaction
  multi-view retrieval; precedent for separating identity from probe
  representations.
* Santhanam et al. 2022 -- ColBERTv2 / PLAID (arXiv:2205.09707):
  centroid+residual decomposition. Level-1 identity ≈ centroid.
* Karpukhin et al. 2020 -- DPR (arXiv:2004.04906): asymmetric Q vs D
  encoders. Inspires the Level-1 vs Level-4 split.
* Yu et al. 2025 -- Asymmetric Hierarchical Retrieval
  (arXiv:2509.16411): direct precedent for the 3-level layered model.
* Edge et al. 2024 -- GraphRAG (arXiv:2404.16130): community-level
  summaries as the navigation primitive.
* Anthropic 2024 -- Contextual Retrieval: WHAT+WHY context blocks lift
  retrieval F1 35-50%. Level-2 abstract operationalizes this.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Identity thresholds (cold-start lock 2026-05-01) ──────────────────
#
# Calibration provenance: T_REUSE=0.92 is the empirical lower bound at
# which embed(what) values still describe the same noun phrase across
# ASCII variants and minor stylistic differences. T_COLLISION_WARN=0.85
# is the upper bound where collisions are statistically suspicious but
# legitimate distinct entities still slip through (e.g. "InjectionGate"
# vs "InjectionGateConfig"). Both are tunable post-rollout if the
# observed rate of false reuses or false collisions is wrong.
T_REUSE_WHAT = 0.92
T_COLLISION_WARN = 0.85

# Discrimination floor on ``what`` length. ``validate_summary`` already
# enforces ≥5 chars structurally; this raises the bar for *identity*
# discrimination because 5-char nouns ("data", "thing") don't separate
# in embedding space. 8 chars admits "Adrian", "Warsaw", "ColBERT",
# "RetrAct"; rejects "data", "thing", "stuff".
_WHAT_MIN_DISCRIMINATIVE = 8

# Conservative stoplist of generic ``what`` values that produce non-
# discriminative identity embeddings. Compared case-insensitively after
# strip. NOT exhaustive -- the gardener's Haiku-driven generic-summary
# flag handles the long tail. Add specific repeat offenders as they
# surface in real corpora.
_WHAT_STOPLIST: frozenset[str] = frozenset(
    {
        # Pure generic
        "thing",
        "things",
        "data",
        "stuff",
        "item",
        "items",
        "entity",
        "object",
        "value",
        # Demonstrative-only
        "this",
        "that",
        "these",
        "those",
        "it",
        # Generic + article
        "the project",
        "the system",
        "the entity",
        "the memory",
        "the data",
        "the file",
        "the thing",
        "a thing",
        "an entity",
        "a project",
        "a system",
        # Placeholder strings
        "todo",
        "tbd",
        "n/a",
        "none",
        "null",
        "placeholder",
        "fixme",
        "xxx",
    }
)


# ── Exceptions ────────────────────────────────────────────────────────


class EntityGateError(Exception):
    """Base for all entity-gate errors. Subclasses carry structured
    context (similarity, conflict_id, etc.) so callers can branch on
    type and retrieve the metadata they need to build a clear error
    response or call ``resolve_conflicts``."""


class PhantomEntityRejected(EntityGateError):
    """Raised by ``assert_entity_exists`` when a triple/edge references
    an entity that has not been declared.

    Pre-cold-start, ``add_triple`` and ``add_rated_edge`` silently
    ``INSERT OR IGNORE`` the missing endpoint into the entities table.
    That created phantom rows with no kind, no summary, no class
    membership -- the root cause of the 1,780 no-is_a-edge entities
    counted in the live corpus on 2026-05-01.

    Post-cold-start, the phantom path is closed: callers must declare
    the entity via ``mempalace_kg_declare_entity`` (or any other
    ``mint_entity`` path) BEFORE writing edges that reference it.
    """


class EntityCollisionError(EntityGateError):
    """Raised by ``mint_entity`` when a new entity's ``what`` collides
    with an existing entity's identity embedding above
    ``T_COLLISION_WARN`` but below ``T_REUSE_WHAT``, and the caller
    specified ``collision_policy='strict'``.

    The conflict is registered in ``_STATE.pending_conflicts`` before
    the exception is raised so the agent can call
    ``mempalace_resolve_conflicts`` (action=merge / keep / skip) to
    disambiguate.

    Attributes:
        conflict_id: handle to look up in ``pending_conflicts``.
        existing_id: the eid that the new ``what`` collided with.
        similarity: the cosine score that triggered the collision.
    """

    def __init__(
        self,
        message: str,
        *,
        conflict_id: str,
        existing_id: str,
        similarity: float,
    ):
        super().__init__(message)
        self.conflict_id = conflict_id
        self.existing_id = existing_id
        self.similarity = float(similarity)


class WhatStoplistError(EntityGateError):
    """Raised when ``summary['what']`` matches the generic stoplist or
    falls below the discrimination floor. Such ``what`` values produce
    identity embeddings that cluster non-discriminatively (~0.99 cosine
    on synonyms), defeating the gate. Caller must supply a more
    specific noun phrase."""


# ── Public API ────────────────────────────────────────────────────────


def assert_entity_exists(eid: str, conn) -> None:
    """Hard-reject a phantom entity reference.

    Replaces the ``INSERT OR IGNORE INTO entities (id, name) VALUES (?, ?)``
    pattern at ``add_triple`` / ``add_rated_edge`` (lines 2076/2078 and
    2214/2218 in knowledge_graph.py pre-cold-start). Pre-cold-start the
    silent insert created phantom rows with no kind, no summary, no
    is_a edge -- the root cause of the 1,780 untyped entities counted
    on 2026-05-01.

    Parameters
    ----------
    eid : str
        The entity id (post-``_entity_id`` normalization) to verify.
    conn : sqlite3.Connection
        An active connection. Caller passes its own so the lookup
        runs inside the caller's transaction (matters for ``add_triple``
        which already holds a ``with conn:`` block).

    Raises
    ------
    PhantomEntityRejected
        If ``eid`` does not exist in the ``entities`` table. The
        error message instructs the caller to declare via
        ``mempalace_kg_declare_entity``.
    """
    cur = conn.execute("SELECT 1 FROM entities WHERE id = ? LIMIT 1", (eid,))
    if cur.fetchone() is None:
        raise PhantomEntityRejected(
            f"phantom entity reference: {eid!r} does not exist in the "
            f"entities table. Cold-start design lock 2026-05-01: "
            f"add_triple and add_rated_edge no longer auto-create "
            f"endpoints. Declare {eid!r} via mempalace_kg_declare_entity "
            f"(with summary={{what,why,scope?}} and queries=[...]) "
            f"before writing edges that reference it. If you intended "
            f"to write a structural edge during entity creation itself "
            f"(e.g. an is_a edge written inside the same transaction "
            f"that mints the entity), pass skip_existence_check=True."
        )


def mint_entity(
    name: str,
    *,
    kind: str = "entity",
    summary: dict,
    queries: list[str] | None = None,
    importance: int = 3,
    properties: dict | None = None,
    added_by: str | None = None,
    body_text: str | None = None,
    collision_policy: str = "strict",
) -> tuple[str, bool]:
    """Mint a new entity OR reuse an existing one if identity matches.

    The single permitted entity-creation primitive post-cold-start.
    Handles summary validation, stoplist check, identity-collision
    detection (3-tier: reuse / warn / new), SQLite write, and 3-level
    Chroma write (identity + abstract + probes).

    Parameters
    ----------
    name : str
        Human-readable entity name. Normalized to ``eid`` via
        ``knowledge_graph.normalize_entity_name``.
    kind : str
        One of ``entity``, ``class``, ``predicate``, ``literal``,
        ``context``, ``agent``. NOT ``record`` (records go through
        ``_add_memory_internal``) and NOT ``operation`` (ops go through
        ``_sync_entity_to_chromadb`` directly with their args_summary
        fingerprint).
    summary : dict
        ``{"what": str, "why": str, "scope": str?}``. Mandatory.
        Validated via ``coerce_summary_for_persist``.
    queries : list[str] | None
        Multi-view probe queries (Level 4). Each becomes one Chroma
        record at ``{eid}__v{i}``. Optional; entities without probe
        queries (some predicate / class entities) are valid.
    importance : int
        1-5 scale. Defaults to 3.
    properties : dict | None
        Additional structured metadata (file_path, etc.). Merged with
        ``{"summary": <dict>, "queries": <list>}`` before persistence.
    added_by : str | None
        Provenance: which agent initiated the creation. Stored on
        every Chroma record's metadata.
    body_text : str | None
        Long-form content (Level 3). Currently unused -- Level 3 was
        dropped at cold-start. Accepted in the signature so future
        consumers can pass it without breaking the API.
    collision_policy : {'strict', 'soft'}
        ``strict`` (default, agent-initiated paths): raise
        ``EntityCollisionError`` on T_COLLISION_WARN ≤ cos < T_REUSE.
        ``soft`` (system-internal paths -- gardener, link_author,
        seed_ontology): register conflict in pending_conflicts but
        proceed with creation so bootstrap doesn't deadlock.

    Returns
    -------
    (entity_id, was_reused) : tuple[str, bool]
        ``was_reused`` is True iff identity-cosine ≥ T_REUSE_WHAT
        matched an existing entity; False if a new entity was created
        (including the soft-collision path).

    Raises
    ------
    SummaryStructureRequired
        Bubbled from ``coerce_summary_for_persist``.
    WhatStoplistError
        ``summary['what']`` is too short or matches the stoplist.
    EntityCollisionError
        Identity collision and ``collision_policy='strict'``.
    """
    # Lazy imports keep entity_gate importable from knowledge_graph.py
    # without creating a cycle (mcp_server -> knowledge_graph ->
    # entity_gate -> mcp_server would deadlock at module load).
    from mempalace.knowledge_graph import (
        coerce_summary_for_persist,
        serialize_summary_for_embedding,
    )

    # 1. Validate + canonicalize summary. Raises SummaryStructureRequired
    #    on bad input; that exception is well-known to upstream callers
    #    and carries a precise migration message.
    summary_dict = coerce_summary_for_persist(
        summary, context_for_error=f"mint_entity({name!r}).summary"
    )
    what = summary_dict["what"]

    # 2. Stoplist + discrimination floor. Both are cheap pre-checks
    #    that fail fast before we touch SQLite or Chroma.
    _check_what_discriminative(what, name=name)

    # 3. Identity-collision check. Fresh palace (Chroma cold) returns
    #    None gracefully -- nothing to collide with, proceed to mint.
    existing_eid, similarity = _find_identity_match(what)
    if existing_eid is not None:
        if similarity >= T_REUSE_WHAT:
            logger.info(
                "mint_entity: silent reuse %s (existing=%s, cos=%.3f)",
                name,
                existing_eid,
                similarity,
            )
            return existing_eid, True
        if similarity >= T_COLLISION_WARN:
            conflict_id = _register_collision(
                new_name=name,
                new_what=what,
                existing_id=existing_eid,
                similarity=similarity,
            )
            if collision_policy == "strict":
                raise EntityCollisionError(
                    f"mint_entity({name!r}): 'what' collides with existing "
                    f"entity {existing_eid!r} at cos={similarity:.3f} "
                    f"(threshold T_COLLISION_WARN={T_COLLISION_WARN}). "
                    f"Conflict {conflict_id!r} registered. Resolve via "
                    f"mempalace_resolve_conflicts: merge (this is the same "
                    f"entity), keep (legitimately distinct), or skip (drop "
                    f"this mint).",
                    conflict_id=conflict_id,
                    existing_id=existing_eid,
                    similarity=similarity,
                )
            # soft policy: warn-and-proceed. The conflict stays in the
            # queue for the gardener / a later resolve_conflicts pass.
            logger.warning(
                "mint_entity: soft-collision %s vs %s (cos=%.3f, conf=%s)",
                name,
                existing_eid,
                similarity,
                conflict_id,
            )

    # 4. Persist via _create_entity (handles SQLite + abstract Chroma
    #    upsert via _sync_entity_to_chromadb). The abstract embedding
    #    text is the rendered summary prose.
    rendered_summary = serialize_summary_for_embedding(summary_dict)
    properties_full = dict(properties or {})
    properties_full["summary"] = summary_dict
    if queries:
        properties_full["queries"] = list(queries)

    from mempalace.mcp_server import _create_entity

    eid = _create_entity(
        name,
        kind=kind,
        content=body_text or rendered_summary,
        importance=importance,
        properties=properties_full,
        added_by=added_by,
        embed_text=rendered_summary,
    )

    # 5. Layer Level-1 identity + Level-4 probe records on top of the
    #    Level-2 abstract record that _create_entity just upserted.
    #    Best-effort: any Chroma write failure is non-fatal because
    #    SQLite is the source of truth and a backfill helper can
    #    re-embed later.
    _write_identity_and_probe_views(
        eid=eid,
        name=name,
        what=what,
        kind=kind,
        importance=importance,
        queries=queries or [],
        added_by=added_by or "",
    )

    logger.info("mint_entity: created %s (eid=%s, kind=%s)", name, eid, kind)
    return eid, False


# ── Internal helpers ──────────────────────────────────────────────────


def _check_what_discriminative(what: str, *, name: str) -> None:
    """Stoplist + minimum-length discrimination check.

    ``validate_summary`` already enforces ``what`` ≥5 chars. This
    raises the bar to the discrimination floor (8 chars) and rejects
    the conservative stoplist of generic phrases.
    """
    stripped = what.strip()
    folded = stripped.lower()
    if len(stripped) < _WHAT_MIN_DISCRIMINATIVE:
        raise WhatStoplistError(
            f"mint_entity({name!r}): 'what'={stripped!r} is below the "
            f"identity-discrimination floor ({_WHAT_MIN_DISCRIMINATIVE} "
            f"chars). Short generic nouns ('data', 'thing', 'value') do "
            f"not separate in embedding space and would create non-"
            f"discriminative identity vectors. Use a more specific noun "
            f"phrase that names *this particular* entity (e.g. "
            f"'InjectionGate', 'AdrianHomeOffice', 'ColBERTv2Centroid')."
        )
    if folded in _WHAT_STOPLIST:
        raise WhatStoplistError(
            f"mint_entity({name!r}): 'what'={stripped!r} matches the "
            f"generic-phrase stoplist. Such values produce non-"
            f"discriminative identity embeddings that cluster at ~0.99 "
            f"cosine, defeating the gate. Use a more specific noun "
            f"phrase. Example bad: 'the project'. Example good: "
            f"'mempalace pre-cold-restart audit'."
        )


def _find_identity_match(what: str) -> tuple[str | None, float]:
    """Query Chroma for the closest Level-1 identity match to ``what``.

    Returns ``(existing_eid, cosine_similarity)`` for the best match,
    or ``(None, 0.0)`` if no match was found, the collection is
    unavailable, or the query failed. Best-effort: any Chroma exception
    falls through as no-match so a cold-start gate without the
    collection still mints cleanly.

    The query filters on ``where={"view_kind": "identity"}`` so we
    compare identity embeddings to identity embeddings only -- not to
    abstract or probe vectors which have different distributions.
    """
    try:
        from mempalace.mcp_server import _get_entity_collection

        col = _get_entity_collection(create=False)
    except Exception as exc:
        logger.debug("identity match: collection unavailable (%s)", exc)
        return None, 0.0
    if col is None:
        return None, 0.0

    try:
        results = col.query(
            query_texts=[what],
            n_results=1,
            where={"view_kind": "identity"},
        )
    except Exception as exc:
        # Empty collection / cold start raises in some Chroma versions;
        # treat as no-match and proceed to mint.
        logger.debug("identity match: query failed (%s)", exc)
        return None, 0.0

    if not results or not results.get("ids") or not results["ids"][0]:
        return None, 0.0

    # Chroma returns distance (cosine distance = 1 - similarity).
    # _CHROMA_METADATA pins ``hnsw:space`` to ``cosine`` so this math
    # is reliable across the project.
    metas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]
    if not metas[0] or not distances[0]:
        return None, 0.0
    meta = metas[0][0] or {}
    eid = meta.get("entity_id")
    if not eid:
        return None, 0.0
    similarity = max(0.0, min(1.0, 1.0 - float(distances[0][0])))
    return eid, similarity


def _register_collision(
    *,
    new_name: str,
    new_what: str,
    existing_id: str,
    similarity: float,
) -> str:
    """Register an entity_collision conflict in pending_conflicts.

    Mirrors the duplicate-memory pattern at mcp_server.py:1052-1077:
    builds a conflict_entry dict, appends to ``_STATE.pending_conflicts``,
    persists the active intent state to disk so the conflict survives
    a process restart, and returns the conflict_id for the caller to
    embed in its error message.
    """
    try:
        from mempalace import intent
        from mempalace.mcp_server import _STATE

        if _STATE.pending_conflicts is None:
            _STATE.pending_conflicts = []
        conflict_id = f"conf_{len(_STATE.pending_conflicts) + 1}"
        entry: dict[str, Any] = {
            "id": conflict_id,
            "conflict_type": "entity_collision",
            "reason": (
                f"Identity collision (cos={similarity:.3f}, threshold "
                f"T_COLLISION_WARN={T_COLLISION_WARN}). New entity "
                f"{new_name!r} 'what'={new_what!r} matches existing "
                f"entity {existing_id!r}."
            ),
            "existing_id": existing_id,
            "new_name": new_name,
            "new_what": new_what,
            "similarity": float(similarity),
        }
        # Surface past resolution if any -- the gardener may have seen
        # this exact pair before (e.g. previous "merge into existing"
        # decision). Best-effort.
        try:
            past = _STATE.kg.get_past_conflict_resolution(existing_id, new_name, "entity_collision")
            if past:
                entry["past_resolution"] = past
        except Exception:
            pass
        _STATE.pending_conflicts.append(entry)
        try:
            intent._persist_active_intent()
        except Exception:
            pass
        return conflict_id
    except Exception as exc:
        # Last-resort fallback so the gate doesn't crash the caller
        # because of a state-shape regression. The collision still
        # raises if collision_policy='strict' -- the caller just gets
        # a synthetic conflict id.
        logger.warning("collision register fallback (%s)", exc)
        return "conf_unregistered"


def _write_identity_and_probe_views(
    *,
    eid: str,
    name: str,
    what: str,
    kind: str,
    importance: int,
    queries: list[str],
    added_by: str,
) -> None:
    """Layer Level-1 (identity) + Level-4 (probe) Chroma records on top
    of the Level-2 (abstract) record that ``_create_entity`` already
    wrote via ``_sync_entity_to_chromadb``.

    Layout post-cold-start
    ----------------------
    For an entity ``eid`` with N probe queries::

        {eid}                  -> Level-2 abstract: embed(rendered prose)
                                  (legacy single-doc id, written by
                                  _sync_entity_to_chromadb)
        {eid}__identity        -> Level-1 identity: embed(what)
        {eid}__v0 .. {eid}__vN-1 -> Level-4 probes: embed(queries[i])

    The ``view_kind`` metadata field discriminates the three layers so
    retrieval-side code can filter (``where={'view_kind': 'identity'}``
    for collision checks; ``where={'view_kind': 'probe'}`` for current
    multi-view max-sim semantics).

    Best-effort: Chroma write failures are logged at warning level but
    do not raise, because SQLite is the source of truth and a backfill
    helper can re-embed later.
    """
    from datetime import datetime as _dt

    try:
        from mempalace.mcp_server import _STATE, _get_entity_collection

        col = _get_entity_collection(create=True)
    except Exception as exc:
        logger.warning("3-level write: collection unavailable for %s (%s)", eid, exc)
        return
    if col is None:
        return

    now_iso = _dt.now().isoformat()
    base_meta = {
        "name": name,
        "kind": kind,
        "importance": importance,
        "last_touched": now_iso,
        "added_by": added_by or "",
        "entity_id": eid,
    }
    # Provenance auto-injection mirrors _sync_entity_to_chromadb so the
    # 3-level records carry the same session_id / intent_id stamps as
    # the abstract record they sit alongside.
    try:
        if _STATE.session_id:
            base_meta["session_id"] = str(_STATE.session_id)
        if _STATE.active_intent and _STATE.active_intent.get("intent_id"):
            base_meta["intent_id"] = str(_STATE.active_intent["intent_id"])
    except Exception:
        pass

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []

    # Level 1: identity. view_index=-1 marks it as the identity layer
    # so retrieval code can treat it differently from probe views
    # (negative index is unambiguous; existing code uses 0..N-1).
    ids.append(f"{eid}__identity")
    docs.append(what)
    metas.append({**base_meta, "view_kind": "identity", "view_index": -1})

    # Level 4: probes. Skip empty / non-string entries defensively;
    # callers should never pass empties but the gate must not blow up
    # on a stray None during seed bootstrap.
    for i, q in enumerate(queries):
        if not isinstance(q, str) or not q.strip():
            continue
        ids.append(f"{eid}__v{i}")
        docs.append(q.strip())
        metas.append({**base_meta, "view_kind": "probe", "view_index": i})

    try:
        col.upsert(ids=ids, documents=docs, metadatas=metas)
    except Exception as exc:
        logger.warning("3-level upsert failed for %s: %s", eid, exc)


__all__ = [
    "T_REUSE_WHAT",
    "T_COLLISION_WARN",
    "EntityGateError",
    "PhantomEntityRejected",
    "EntityCollisionError",
    "WhatStoplistError",
    "assert_entity_exists",
    "mint_entity",
]
