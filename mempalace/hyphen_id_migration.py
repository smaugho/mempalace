"""
hyphen_id_migration.py -- One-shot migration that renames legacy hyphenated
IDs to their canonical underscored form.

Background
----------
Before phase N1 the codebase had two normalizers: ``normalize_entity_name``
(emitted underscore) and ``_normalize_memory_slug`` (emitted hyphen). Record
IDs built with BOTH looked like ``record_ga_agent_some-slug-with-hyphens``.
Any later code path that re-normalized via ``normalize_entity_name`` flipped
the hyphens to underscores and then missed Chroma on ``col.get(ids=[...])``.

N1 collapsed the normalizers. This migration (N3) canonicalises every
legacy ID in place so downstream callsites no longer need a dual-lookup
dance. After it runs, the invariant ``stored_id == normalize_entity_name(id)``
holds for every ID in both Chroma and SQLite.

Idempotence + safety
--------------------
- Gated on ``ServerState.hyphen_ids_migrated`` so it runs once per process.
  Subsequent cold starts see the flag cleared but then examine the data,
  find nothing to migrate (all IDs already canonical), and exit quickly.
- Collision handling: if the target ID already exists as a distinct entity,
  we skip the rename and log the collision. The two rows stay independent;
  callers can merge them explicitly via ``kg_merge_entities`` later.
- Chroma writes round-trip embeddings via ``include=['embeddings']`` so
  we never re-compute cosine vectors. If Chroma omits embeddings on get
  (can happen for records written with older Chroma versions), we fall
  back to re-embed via upsert without ``embeddings=`` -- Chroma re-embeds
  from the document text. That's correct but slower; the fallback is rare.
- All DB writes happen inside a single SQLite transaction per table.
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger("mempalace.hyphen_id_migration")


# ==================== ID transformation ====================


def _split_view_suffix(chroma_id: str) -> tuple[str, str]:
    """Split an entity-collection Chroma ID into (entity_id, view_suffix).

    The entity collection stores multi-view records under
    ``<entity_id>__v<N>``. Legacy ``::view_N`` suffix is retired by
    ``_migrate_entity_views_schema`` and should not appear here, but we
    handle it defensively.

    Returns (entity_id, "") for IDs with no view suffix.
    """
    if "__v" in chroma_id:
        eid, sep, v = chroma_id.rpartition("__v")
        if v.isdigit():
            return eid, f"__v{v}"
    if "::view_" in chroma_id:
        eid, sep, v = chroma_id.rpartition("::view_")
        if v.isdigit():
            return eid, f"::view_{v}"
    return chroma_id, ""


def compute_target_id(old_id: str, normalize: Callable[[str], str]) -> str:
    """Return the canonical form of ``old_id``, preserving any view suffix."""
    eid, suffix = _split_view_suffix(old_id)
    return normalize(eid) + suffix


def id_needs_migration(old_id: str, normalize: Callable[[str], str]) -> bool:
    """True iff ``old_id`` would change under the canonical normalizer."""
    return compute_target_id(old_id, normalize) != old_id


# ==================== Chroma migration ====================


def migrate_chroma_collection(
    col,
    normalize: Callable[[str], str],
    *,
    update_metadata_entity_id: bool = False,
    batch_size: int = 200,
) -> dict:
    """Rename every hyphenated ID in ``col`` to its canonical form.

    Parameters
    ----------
    col:
        A ChromaDB collection handle.
    normalize:
        The canonical ID normalizer (``normalize_entity_name``).
    update_metadata_entity_id:
        True for the entity collection where records carry a
        ``metadata.entity_id`` field that must be rewritten in lockstep.
    batch_size:
        How many records to get/delete/upsert per round-trip.

    Returns a stats dict with counts of ``migrated`` and ``collisions``.
    """
    stats = {"scanned": 0, "migrated": 0, "collisions": 0, "errors": 0}
    if col is None:
        return stats

    # Full scan -- Chroma has no "where id LIKE" filter so we pull all IDs.
    try:
        got = col.get(include=["documents", "metadatas", "embeddings"])
    except Exception as exc:
        logger.warning(f"Chroma get() failed during hyphen-id migration: {exc}")
        stats["errors"] += 1
        return stats

    if not got or not got.get("ids"):
        return stats

    ids = got["ids"]
    docs = got.get("documents") or [None] * len(ids)
    metas = got.get("metadatas") or [None] * len(ids)
    embs = got.get("embeddings") or [None] * len(ids)
    stats["scanned"] = len(ids)

    # First pass: build a set of current IDs so we can detect collisions.
    existing_ids = set(ids)
    rename_plan: list[tuple[str, str, object, object, object]] = []  # (old, new, doc, meta, emb)
    for i, old in enumerate(ids):
        new = compute_target_id(old, normalize)
        if new == old:
            continue
        if new in existing_ids and new != old:
            logger.warning(
                f"Hyphen-id migration collision: '{old}' -> '{new}' but '{new}' "
                f"already exists. Skipping rename; merge candidate."
            )
            stats["collisions"] += 1
            continue
        existing_ids.add(new)  # reserve so later sibling views don't collide
        meta = dict(metas[i]) if isinstance(metas[i], dict) else (metas[i] or {})
        if update_metadata_entity_id and "entity_id" in meta:
            meta["entity_id"] = normalize(str(meta["entity_id"]))
        rename_plan.append((old, new, docs[i], meta, embs[i]))

    if not rename_plan:
        return stats

    # Second pass: apply in batches. Delete old IDs first so the upsert
    # under the new ID cannot collide within Chroma's own bookkeeping.
    for start in range(0, len(rename_plan), batch_size):
        chunk = rename_plan[start : start + batch_size]
        old_batch = [p[0] for p in chunk]
        new_batch = [p[1] for p in chunk]
        doc_batch = [p[2] for p in chunk]
        meta_batch = [p[3] for p in chunk]
        emb_batch = [p[4] for p in chunk]

        try:
            col.delete(ids=old_batch)
        except Exception as exc:
            logger.warning(f"Chroma delete() chunk failed: {exc}")
            stats["errors"] += 1
            continue

        upsert_kwargs: dict = {
            "ids": new_batch,
            "documents": doc_batch,
            "metadatas": meta_batch,
        }
        # Preserve embeddings verbatim when Chroma gave them to us. If any
        # slot is None, fall through and let Chroma re-embed from the doc.
        if all(e is not None for e in emb_batch):
            upsert_kwargs["embeddings"] = emb_batch

        try:
            col.upsert(**upsert_kwargs)
            stats["migrated"] += len(chunk)
        except Exception as exc:
            logger.warning(f"Chroma upsert() chunk failed: {exc}")
            stats["errors"] += 1

    return stats


# ==================== SQLite migration ====================


# Tables and columns that hold entity IDs. Order matters ONLY in that we
# want to discover the set of ID renames from `entities` first (the
# authoritative table), then cascade the remap across referring tables.
_SQLITE_ID_COLUMNS: list[tuple[str, list[str]]] = [
    ("entities", ["id", "merged_into"]),
    ("entity_aliases", ["alias", "canonical_id"]),
    ("entity_keywords", ["entity_id"]),
    ("triples", ["subject", "object"]),
    ("conflict_resolutions", ["existing_id", "new_id"]),
    ("edge_traversal_feedback", ["subject", "object"]),
    ("keyword_feedback", ["memory_id"]),
]


def _build_remap(conn, normalize: Callable[[str], str]) -> dict:
    """Scan every ID-bearing SQLite column; return a dict of old_id -> new_id.

    We gather IDs from the UNION of all referencing columns (not just
    ``entities.id``) because a triple or alias may reference an entity
    that was never inserted as a first-class row. Those foreign-keyed
    IDs still need the same canonicalisation.
    """
    remap: dict[str, str] = {}
    seen: set[str] = set()
    for table, cols in _SQLITE_ID_COLUMNS:
        for col in cols:
            try:
                for (val,) in conn.execute(f"SELECT {col} FROM {table}"):
                    if val is None or val in seen:
                        continue
                    seen.add(val)
                    new = normalize(val)
                    if new != val:
                        remap[val] = new
            except Exception as exc:
                logger.warning(f"SQLite scan of {table}.{col} failed: {exc}")
    return remap


def _collision_filter(conn, remap: dict) -> tuple[dict, int]:
    """Drop entries whose target already exists as a distinct entities.id.

    Returns (safe_remap, collisions).
    """
    safe: dict[str, str] = {}
    collisions = 0
    try:
        existing = {row[0] for row in conn.execute("SELECT id FROM entities")}
    except Exception:
        existing = set()
    for old, new in remap.items():
        if new in existing and new != old and old in existing:
            # Both the legacy hyphenated row AND the canonical underscored
            # row exist as independent entities. Don't silently merge --
            # leave both and let the agent merge explicitly.
            logger.warning(
                f"SQLite hyphen-id collision: '{old}' -> '{new}' but '{new}' "
                f"already exists as a distinct entity. Skipping."
            )
            collisions += 1
            continue
        safe[old] = new
    return safe, collisions


def migrate_sqlite(conn, normalize: Callable[[str], str]) -> dict:
    """Rename every hyphenated ID across every referencing SQLite table.

    Runs inside a single transaction so a mid-migration crash leaves the
    DB consistent. Returns stats dict.
    """
    stats = {"candidate_ids": 0, "migrated_ids": 0, "collisions": 0, "rows_touched": 0}

    remap = _build_remap(conn, normalize)
    stats["candidate_ids"] = len(remap)
    if not remap:
        return stats

    safe, collisions = _collision_filter(conn, remap)
    stats["collisions"] = collisions
    stats["migrated_ids"] = len(safe)
    if not safe:
        return stats

    cursor = conn.cursor()
    try:
        cursor.execute("BEGIN")
        for table, cols in _SQLITE_ID_COLUMNS:
            for col in cols:
                # UPDATE ... SET col = CASE col WHEN :old1 THEN :new1 ... END
                # is compact but harder to debug on failure. Loop instead;
                # each row is a single-PK UPDATE and the plan is trivial.
                for old, new in safe.items():
                    try:
                        cur = cursor.execute(
                            f"UPDATE {table} SET {col} = ? WHERE {col} = ?",
                            (new, old),
                        )
                        stats["rows_touched"] += cur.rowcount or 0
                    except Exception as exc:
                        # PK uniqueness violation on entities.id would mean
                        # our collision detection missed a case -- log and
                        # keep going so we don't abort the whole migration
                        # on one pathological row.
                        logger.warning(
                            f"SQLite UPDATE {table}.{col} '{old}' -> '{new}' failed: {exc}"
                        )
        cursor.execute("COMMIT")
    except Exception as exc:
        cursor.execute("ROLLBACK")
        logger.error(f"Hyphen-id SQLite migration failed, rolled back: {exc}")
        raise

    return stats


# ==================== Orchestrator ====================


def run_migration(
    state,
    *,
    chroma_record_col,
    chroma_entity_col,
    chroma_feedback_col,
    normalize: Callable[[str], str],
) -> dict:
    """Run the full hyphen-id migration exactly once per process.

    Gated on ``state.hyphen_ids_migrated``. Safe to call on every server
    startup -- a second invocation within the same process is a no-op.

    Returns a flat stats dict for logging.
    """
    if getattr(state, "hyphen_ids_migrated", False):
        return {"skipped": True}
    state.hyphen_ids_migrated = True

    combined: dict = {}
    # Chroma -- records collection (plain IDs, no view suffix).
    if chroma_record_col is not None:
        s = migrate_chroma_collection(chroma_record_col, normalize, update_metadata_entity_id=False)
        combined["chroma_records"] = s
    # Chroma -- entities collection (IDs carry __vN view suffix; metadata.entity_id).
    if chroma_entity_col is not None:
        s = migrate_chroma_collection(chroma_entity_col, normalize, update_metadata_entity_id=True)
        combined["chroma_entities"] = s
    # Chroma -- feedback contexts collection (plain IDs).
    if chroma_feedback_col is not None:
        s = migrate_chroma_collection(
            chroma_feedback_col, normalize, update_metadata_entity_id=False
        )
        combined["chroma_feedback"] = s
    # SQLite -- every table with an ID-bearing column.
    if state.kg is not None:
        try:
            conn = state.kg._conn()
            combined["sqlite"] = migrate_sqlite(conn, normalize)
        except Exception as exc:
            logger.warning(f"SQLite hyphen-id migration failed: {exc}")
            combined["sqlite"] = {"error": str(exc)}

    return combined
