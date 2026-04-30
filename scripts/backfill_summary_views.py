"""Backfill: embed each entity's structured summary as view N+1 in Chroma.

Adrian's design lock 2026-04-30: "summary should be a view EVERYWHERE
wherever there is a summary." The new code path
(``_sync_entity_views_to_chromadb`` + ``context_lookup_or_create`` view
persistence) appends ``serialize_summary_for_embedding(summary)`` as view
N+1 on every WRITE. But existing entities written before that ship carry
only query views; their summary lives in ``properties.summary`` but is
not in the multi-view collection.

This script walks the live palace and re-upserts the multi-view records
for every entity that has a non-empty ``properties.summary`` dict, with
the rendered summary appended as view N+1 (carrying ``is_summary_view``
metadata flag so readers can distinguish summary-derived from query-
derived views).

Three classes of entities are backfilled:

  - kind='context'  → upsert into ``mempalace_context_views`` collection,
                      with metadata fields {context_id, view_index,
                      is_summary_view?} matching the live mint path at
                      mcp_server.context_lookup_or_create.
  - kind in entity/class/predicate/literal
                    → upsert into the entity views collection via
                      ``_sync_entity_views_to_chromadb`` with
                      ``summary_view`` kwarg.
  - kind='operation' → SKIPPED. Operations have a single args_summary
                      fingerprint that's already embedded; multi-view
                      doesn't apply (cf. arXiv 2512.18950).

Records (kind='record') are not touched here. Records use a separate
single-doc embed path that already includes summary in the embedded
text (Anthropic Contextual Retrieval pattern: ``summary\\n\\ncontent``).

Usage::

    python scripts/backfill_summary_views.py [--dry-run] [--limit N]

    --dry-run  count and log what would be backfilled, no Chroma writes.
    --limit N  cap entities processed (0 = all). Useful for staging.

Idempotent: re-running on already-backfilled entities re-upserts the same
ids; the view N+1 just gets re-written with the same content. Safe to run
in the middle of a session (no SQLite writes; only Chroma upserts).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

# ── Bootstrapping: make the in-tree mempalace package importable ──
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DB = Path(os.path.expanduser("~/.mempalace/palace/knowledge_graph.sqlite3"))


def _coerce_props(raw):
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _render_summary_inline(summary) -> str:
    """Local copy of serialize_summary_for_embedding semantics.

    Kept independent of mempalace import graph so the script can be
    reasoned about in isolation.
    """
    if not isinstance(summary, dict):
        return ""
    parts = []
    what = (summary.get("what") or "").strip()
    why = (summary.get("why") or "").strip()
    scope = (summary.get("scope") or "").strip()
    if what:
        parts.append(f"WHAT: {what}.")
    if why:
        parts.append(f"WHY: {why}.")
    if scope:
        parts.append(f"SCOPE: {scope}.")
    return " ".join(parts)


def _eligible_entities(conn: sqlite3.Connection, limit: int | None):
    cur = conn.cursor()
    sql = (
        "SELECT id, name, kind, importance, properties "
        "  FROM entities "
        " WHERE kind IN ('context','entity','class','predicate','literal') "
        "   AND properties IS NOT NULL "
        "   AND properties LIKE '%summary%'"
    )
    if limit is not None and limit > 0:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql)
    for row in cur.fetchall():
        eid, name, kind, importance, raw_props = row
        props = _coerce_props(raw_props)
        summary = props.get("summary")
        if not isinstance(summary, dict) or not summary:
            continue
        rendered = _render_summary_inline(summary)
        if not rendered:
            continue
        yield {
            "id": eid,
            "name": name,
            "kind": kind,
            "importance": int(importance or 3),
            "queries": props.get("queries") or [],
            "rendered_summary": rendered,
        }


def _backfill_context_views(records, dry_run: bool) -> tuple[int, int]:
    """Upsert summary view as view N+1 for each context entity."""
    if not records:
        return 0, 0
    if dry_run:
        return len(records), 0
    from mempalace.mcp_server import _get_context_views_collection

    col = _get_context_views_collection(create=True)
    if not col:
        print("  context-views collection unavailable; skipping context backfill")
        return 0, 0
    n_attempted = 0
    n_done = 0
    for rec in records:
        n_attempted += 1
        eid = rec["id"]
        queries = rec["queries"]
        rendered = rec["rendered_summary"]
        view_docs = list(queries)
        if rendered and rendered not in view_docs:
            view_docs.append(rendered)
        if len(view_docs) <= len(queries):
            # No new view to add (rendered was a duplicate of a query).
            continue
        summary_view_index = len(view_docs) - 1
        ids = [f"{eid}_v{i}" for i in range(len(view_docs))]
        metas = []
        for i in range(len(view_docs)):
            m = {"context_id": eid, "view_index": i}
            if i == summary_view_index:
                m["is_summary_view"] = True
            metas.append(m)
        try:
            col.upsert(ids=ids, documents=view_docs, metadatas=metas)
            n_done += 1
        except Exception as exc:
            print(f"  context backfill failed for {eid}: {exc!r}")
    return n_attempted, n_done


def _backfill_entity_views(records, dry_run: bool) -> tuple[int, int]:
    """Upsert summary view as view N+1 for each entity/class/predicate/literal."""
    if not records:
        return 0, 0
    if dry_run:
        return len(records), 0
    from mempalace.mcp_server import _sync_entity_views_to_chromadb

    n_attempted = 0
    n_done = 0
    for rec in records:
        n_attempted += 1
        try:
            _sync_entity_views_to_chromadb(
                rec["id"],
                rec["name"],
                rec["queries"],
                rec["kind"],
                rec["importance"],
                summary_view=rec["rendered_summary"],
            )
            n_done += 1
        except Exception as exc:
            print(f"  entity backfill failed for {rec['id']}: {exc!r}")
    return n_attempted, n_done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--db", type=str, default=str(DB))
    args = ap.parse_args()

    db_path = Path(os.path.expanduser(args.db))
    print(f"[1/4] Loading entities with summary from {db_path}", flush=True)
    conn = sqlite3.connect(db_path)
    eligible = list(_eligible_entities(conn, args.limit if args.limit > 0 else None))
    by_kind = {}
    for e in eligible:
        by_kind.setdefault(e["kind"], []).append(e)
    print(f"      {len(eligible)} entities have non-empty properties.summary")
    for k, v in sorted(by_kind.items()):
        print(f"        {k:>12}: {len(v)}")

    if args.dry_run:
        print("\n[2/4] --dry-run -- no Chroma writes")
        ctx_attempted, _ = _backfill_context_views(by_kind.get("context") or [], True)
        ent_records = [
            e for e in eligible if e["kind"] in ("entity", "class", "predicate", "literal")
        ]
        ent_attempted, _ = _backfill_entity_views(ent_records, True)
        print(f"      would attempt: {ctx_attempted} contexts + {ent_attempted} entities")
        return

    print("\n[2/4] Backfilling context views", flush=True)
    t0 = time.time()
    ctx_attempted, ctx_done = _backfill_context_views(by_kind.get("context") or [], False)
    print(f"      {ctx_done}/{ctx_attempted} done in {time.time() - t0:.1f}s")

    print("\n[3/4] Backfilling entity views", flush=True)
    t0 = time.time()
    ent_records = [e for e in eligible if e["kind"] in ("entity", "class", "predicate", "literal")]
    ent_attempted, ent_done = _backfill_entity_views(ent_records, False)
    print(f"      {ent_done}/{ent_attempted} done in {time.time() - t0:.1f}s")

    print("\n[4/4] Backfill complete", flush=True)
    print(f"      contexts:  {ctx_done}/{ctx_attempted}")
    print(f"      entities:  {ent_done}/{ent_attempted}")
    print(f"      total:     {ctx_done + ent_done}/{ctx_attempted + ent_attempted}")
    conn.close()


if __name__ == "__main__":
    main()
