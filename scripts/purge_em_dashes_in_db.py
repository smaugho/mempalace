#!/usr/bin/env python
"""Purge U+2014 / U+2013 from the mempalace SQLite database.

Adrian directive 2026-04-28: "I don't want them AT ALL". The renderer
and codebase fixes prevent NEW writes from minting non-ASCII dashes;
this script scrubs EXISTING rows.

Strategy: walk every TEXT column in every table of the mempalace DB
and replace the dashes via SQL UPDATE. Idempotent: re-running on a
clean DB is a no-op.

Default DB path is ~/.mempalace/palace.sqlite3 ; override with --db.

The --dry-run flag prints per-table counts without writing.

Chroma collections are NOT touched by this script. Chroma stores the
rendered prose alongside the embedding; rewriting the document text
would desync the vector. Existing chroma documents with U+2014 will
refresh organically as records get touched (kg_update_entity, gardener
re-render passes). The SQLite scrub covers the source-of-truth surface
agents read via kg_query / kg_search result text.

Usage:
    python -X utf8 scripts/purge_em_dashes_in_db.py --dry-run
    python -X utf8 scripts/purge_em_dashes_in_db.py
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path


_EM = chr(0x2014)
_EN = chr(0x2013)


def _default_db_path() -> Path:
    home = Path(os.path.expanduser("~"))
    return home / ".mempalace" / "palace" / "knowledge_graph.sqlite3"


def _table_text_columns(conn: sqlite3.Connection, table: str) -> list[tuple[str, str]]:
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    out = []
    for cid, name, ctype, notnull, dflt, pk in rows:
        upper = (ctype or "").upper()
        if "TEXT" in upper or "CHAR" in upper or "CLOB" in upper or upper == "":
            out.append((name, ctype or "<unspecified>"))
    return out


def _list_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def _row_pk_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    pks = [name for cid, name, ctype, notnull, dflt, pk in rows if pk]
    return pks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(_default_db_path()))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    db = Path(args.db)
    if not db.is_file():
        print(f"db not found: {db}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("BEGIN")

    total_em_rows = 0
    total_en_rows = 0
    columns_touched = 0

    try:
        tables = _list_tables(conn)
        for table in tables:
            text_cols = _table_text_columns(conn, table)
            if not text_cols:
                continue
            for col, ctype in text_cols:
                # count rows containing either dash
                cnt_row = conn.execute(
                    f'SELECT COUNT(*) FROM "{table}" WHERE "{col}" LIKE ? OR "{col}" LIKE ?',
                    (f"%{_EM}%", f"%{_EN}%"),
                ).fetchone()
                cnt = cnt_row[0] if cnt_row else 0
                if cnt == 0:
                    continue
                em_rows = conn.execute(
                    f'SELECT COUNT(*) FROM "{table}" WHERE "{col}" LIKE ?',
                    (f"%{_EM}%",),
                ).fetchone()[0]
                en_rows = conn.execute(
                    f'SELECT COUNT(*) FROM "{table}" WHERE "{col}" LIKE ?',
                    (f"%{_EN}%",),
                ).fetchone()[0]
                total_em_rows += em_rows
                total_en_rows += en_rows
                columns_touched += 1
                print(f"  {table}.{col} ({ctype}): em_rows={em_rows} en_rows={en_rows}")
                if not args.dry_run:
                    conn.execute(
                        f'UPDATE "{table}" SET "{col}" = REPLACE(REPLACE("{col}", ?, ?), ?, ?) WHERE "{col}" LIKE ? OR "{col}" LIKE ?',
                        (_EM, " -- ", _EN, " - ", f"%{_EM}%", f"%{_EN}%"),
                    )

        if args.dry_run:
            conn.execute("ROLLBACK")
            mode = "dry-run"
        else:
            conn.execute("COMMIT")
            mode = "purged"
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()

    print()
    print(
        f"[{mode}] em_rows_total={total_em_rows} en_rows_total={total_en_rows} "
        f"columns_touched={columns_touched}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
