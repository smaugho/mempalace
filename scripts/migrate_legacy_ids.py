#!/usr/bin/env python
"""Migrate legacy ids to new short formats.

Renumbers pre-slice ids (long ctx_<hex>_<ns>_<hex>, msg_<digest>_<ns>,
conf_<long>) to the new short ctx_<N> / msg_<sid_short>_<turn> / conf_<N>
shapes across every FK column in the mempalace SQLite DB. Per Adrian
2026-04-28: prose mentions inside record bodies are NOT touched.

Uniqueness: per-type counter starts at MAX(existing N in new format) + 1
so newly-allocated ids never collide with already-minted ones. Stable
sort over legacy id strings -> deterministic re-runs.
"""

from __future__ import annotations
import argparse
import os
import re
import sqlite3
import sys
from pathlib import Path

_ID_COLUMNS = [
    ("entities", "id"),
    ("entities", "name"),
    ("entities", "merged_into"),
    ("entities", "creation_context_id"),
    ("triples", "subject"),
    ("triples", "object"),
    ("triples", "creation_context_id"),
    ("entity_aliases", "alias"),
    ("entity_aliases", "canonical_id"),
    ("entity_keywords", "entity_id"),
    ("entity_keywords", "keyword"),
    ("conflict_resolutions", "conflict_id"),
    ("keyword_idf", "keyword"),
    ("link_prediction_candidates", "from_entity"),
    ("link_prediction_candidates", "last_context_id"),
    ("link_prediction_sources", "from_entity"),
    ("link_prediction_sources", "ctx_id"),
    ("memory_flags", "memory_key"),
    ("memory_flags", "context_id"),
]

_NEW_CTX = re.compile(r"^ctx_\d+$")
_NEW_MSG = re.compile(r"^msg_(\d+|[0-9a-f]{6}_\d+)$")
_NEW_CONF = re.compile(r"^conf_\d+$")


def _default_db():
    return Path(os.path.expanduser("~")) / ".mempalace" / "palace" / "knowledge_graph.sqlite3"


def _classify(value):
    if not isinstance(value, str):
        return ""
    if value.startswith("ctx_") and not _NEW_CTX.match(value):
        return "ctx"
    if value.startswith("msg_") and not _NEW_MSG.match(value):
        return "msg"
    if value.startswith("conf_") and not _NEW_CONF.match(value):
        return "conf"
    return ""


def _max_seq(conn, prefix):
    rows = conn.execute(f"SELECT id FROM entities WHERE id LIKE '{prefix}_%'").fetchall()
    max_n = 0
    plen = len(prefix) + 1
    for (rid,) in rows:
        suffix = rid[plen:]
        if suffix.isdigit():
            n = int(suffix)
            if n > max_n:
                max_n = n
    return max_n


def _collect(conn):
    out = {"ctx": set(), "msg": set(), "conf": set()}
    for table, col in _ID_COLUMNS:
        try:
            rows = conn.execute(f'SELECT DISTINCT "{col}" FROM "{table}"').fetchall()
        except sqlite3.OperationalError as exc:
            print(f"  [skip] {table}.{col}: {exc}", file=sys.stderr)
            continue
        for (val,) in rows:
            kind = _classify(val)
            if kind:
                out[kind].add(val)
    return out


def _build_map(conn, legacy_by_type):
    mapping = {}
    for prefix, legacy_set in legacy_by_type.items():
        if not legacy_set:
            continue
        counter = _max_seq(conn, prefix) + 1
        for legacy_id in sorted(legacy_set):
            mapping[legacy_id] = f"{prefix}_{counter}"
            counter += 1
    return mapping


def _stage(conn, mapping):
    conn.execute("DROP TABLE IF EXISTS _id_migration_map")
    conn.execute(
        "CREATE TABLE _id_migration_map (legacy_id TEXT PRIMARY KEY, new_id TEXT NOT NULL)"
    )
    conn.executemany("INSERT INTO _id_migration_map VALUES (?, ?)", list(mapping.items()))
    conn.execute("CREATE INDEX _id_migration_map_new_id ON _id_migration_map(new_id)")


def _apply(conn):
    counts = {}
    for table, col in _ID_COLUMNS:
        try:
            cur = conn.execute(
                f'UPDATE "{table}" SET "{col}" = (SELECT new_id FROM _id_migration_map WHERE legacy_id = "{table}"."{col}") WHERE "{col}" IN (SELECT legacy_id FROM _id_migration_map)'
            )
            counts[(table, col)] = cur.rowcount
        except sqlite3.OperationalError as exc:
            print(f"  [skip-update] {table}.{col}: {exc}", file=sys.stderr)
            counts[(table, col)] = -1
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(_default_db()))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    db = Path(args.db)
    if not db.is_file():
        print(f"db not found: {db}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA foreign_keys = OFF")

    print(f"[scan] {db}")
    legacy = _collect(conn)
    for k, v in legacy.items():
        print(f"  legacy {k}: {len(v)} distinct")

    mapping = _build_map(conn, legacy)
    print(f"  mapping size: {len(mapping)}")

    if not mapping:
        print("[done] nothing to migrate")
        conn.close()
        return 0

    by_type = {"ctx": [], "msg": [], "conf": []}
    for old, new in mapping.items():
        k = _classify(old)
        if k:
            by_type[k].append((old, new))
    for kind, samples in by_type.items():
        for old, new in samples[:3]:
            print(f"    sample {kind}: {old}  ->  {new}")

    conn.execute("BEGIN")
    try:
        _stage(conn, mapping)
        counts = _apply(conn)
        total = sum(c for c in counts.values() if c > 0)
        print()
        print("=== rows updated per column ===")
        for (table, col), n in counts.items():
            if n != 0:
                print(f"  {table}.{col}: {n}")
        print(f"  TOTAL: {total}")

        if args.dry_run:
            conn.execute("ROLLBACK")
            print("[dry-run] rolled back")
        else:
            conn.execute("DROP TABLE _id_migration_map")
            conn.execute("COMMIT")
            print("[committed]")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
