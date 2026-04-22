"""
link_author.py — Analytical candidate accumulation for graph-link authoring.

This is the half of the link-author redesign that runs in-session at
finalize_intent time. It reads `active_intent.contexts_touched_detail`
(every reused + positively-rated context), accumulates Adamic-Adar
evidence on each unconnected entity pair that co-appeared inside those
contexts, and persists it to the `link_prediction_candidates` table.

Adamic-Adar weighting (Adamic & Adar 2003, Liben-Nowell & Kleinberg 2007):
for pair (A, B), add ``1 / log(|ctx.entities|)`` once per DISTINCT
context where both co-appear. Rare shared neighbours are stronger
evidence than common ones, so a context listing 2 entities contributes
more than one listing 30.

Distinct-context dedup (Option A): `link_prediction_sources` carries a
PK of (from_entity, to_entity, ctx_id). INSERT OR IGNORE means
re-observing the same context N times contributes exactly once — the
threshold reflects co-occurrence in DIFFERENT situations, not repeated
references to one.

The LLM jury pipeline (Opus designs → 3 Haiku jurors → Haiku
synthesis) that DRAINS this queue lives in Commit 3. Commit 2 only
accumulates evidence so the queue is observable (`SELECT * FROM
link_prediction_candidates ORDER BY score DESC`) as real work happens.
"""

from __future__ import annotations

from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def upsert_candidate(
    kg,
    from_entity: str,
    to_entity: str,
    weight: float,
    context_id: str,
) -> bool:
    """Accumulate one context's Adamic-Adar contribution to a pair's score.

    Canonicalises (from_entity, to_entity) so (A, B) and (B, A) hit the
    same row, then dedups by ctx_id: if this context has already
    contributed to this pair, does nothing and returns False.
    Otherwise inserts into `link_prediction_sources` (the dedup table),
    upserts into `link_prediction_candidates` adding `weight` to the
    score and incrementing `shared_context_count`, and returns True.

    Direct-edge short-circuit: if a 1-hop edge already exists between
    the pair (either direction, any predicate, `valid_to IS NULL`), the
    upsert is skipped. The jury's job is discovering NEW edges; the
    graph channel already surfaces directly-connected pairs together,
    so a candidate suggestion there would be pure noise.

    Raises nothing on the happy path. Any persistence error is
    propagated so finalize can record it — but the caller (see
    intent.tool_finalize_intent) wraps the whole upsert block in a
    try/except to keep finalize robust against DB failures.
    """
    if not from_entity or not to_entity:
        return False
    if from_entity == to_entity:
        return False
    if not context_id:
        return False

    # Canonical ordering keeps PK unique across symmetric pair perms.
    a, b = (from_entity, to_entity) if from_entity < to_entity else (to_entity, from_entity)

    conn = kg._conn()
    # Direct-edge skip. Cheap 1-row LIMIT query using the normal triples
    # table. Any exception bubbles up; the caller's try/except decides.
    row = conn.execute(
        "SELECT 1 FROM triples "
        "WHERE ((subject = ? AND object = ?) OR (subject = ? AND object = ?)) "
        "AND valid_to IS NULL "
        "LIMIT 1",
        (a, b, b, a),
    ).fetchone()
    if row is not None:
        return False

    now = _now_iso()
    # Dedup: atomic INSERT OR IGNORE on the sources table. If the
    # triple (pair, ctx_id) is already recorded, rowcount stays 0 and
    # we skip the candidate increment.
    cur = conn.execute(
        "INSERT OR IGNORE INTO link_prediction_sources "
        "(from_entity, to_entity, ctx_id, contributed_ts) "
        "VALUES (?, ?, ?, ?)",
        (a, b, context_id, now),
    )
    if cur.rowcount == 0:
        conn.commit()
        return False

    # Upsert the candidate row. SQLite 3.24+ supports ON CONFLICT DO
    # UPDATE. The candidate table's PK is (from_entity, to_entity),
    # so the conflict path handles both the "new pair" and the "nth
    # distinct context for existing pair" cases uniformly.
    conn.execute(
        "INSERT INTO link_prediction_candidates "
        "(from_entity, to_entity, score, shared_context_count, "
        " last_context_id, last_updated_ts) "
        "VALUES (?, ?, ?, 1, ?, ?) "
        "ON CONFLICT(from_entity, to_entity) DO UPDATE SET "
        "    score = score + excluded.score, "
        "    shared_context_count = shared_context_count + 1, "
        "    last_context_id = excluded.last_context_id, "
        "    last_updated_ts = excluded.last_updated_ts",
        (a, b, float(weight), context_id, now),
    )
    conn.commit()
    return True


def list_pending(kg, limit: int = 50, threshold: float = 1.5) -> list[dict]:
    """Return unprocessed candidates above `threshold`, ordered by score desc.

    Consumed by the `mempalace link-author process` CLI in Commit 3.
    Rows where `processed_ts IS NOT NULL` are excluded — rejections and
    acceptances stay in the table for audit but are not re-processed
    here.
    """
    conn = kg._conn()
    rows = conn.execute(
        "SELECT from_entity, to_entity, score, shared_context_count, "
        "       last_context_id, last_updated_ts "
        "FROM link_prediction_candidates "
        "WHERE processed_ts IS NULL AND score >= ? "
        "ORDER BY score DESC "
        "LIMIT ?",
        (float(threshold), int(limit)),
    ).fetchall()
    return [
        {
            "from_entity": r[0],
            "to_entity": r[1],
            "score": r[2],
            "shared_context_count": r[3],
            "last_context_id": r[4],
            "last_updated_ts": r[5],
        }
        for r in rows
    ]


def _dispatch_if_due(kg, interval_hours: int = 1) -> None:
    """Event-driven dispatcher stub (fully wired in Commit 3).

    Commit 3 will check NOW - last_run_ts >= interval_hours AND there
    are pending candidates, then spawn ``mempalace link-author process``
    as a detached subprocess (start_new_session=True on POSIX,
    DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP on Windows) so finalize
    doesn't block on LLM calls. The CLI doesn't exist yet in Commit 2;
    if finalize calls this now, it is a safe no-op. Any exception is
    caught by the finalize wrapper.
    """
    return None
