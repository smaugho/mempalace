"""023: Drop the legacy `description` column from entities -- final cleanup
of the description->content rename arc.

Phase 1 (022_add_content_column.py) added the `content` column and
backfilled from `description`. Phases 2 + 3a-3g migrated every production
caller to use `content=` for kwargs, `update_entity_content` for method
calls, and `["content"]` for dict-key reads. This migration finishes the
cycle by dropping the now-unused `description` column.

Adrian's directive 2026-04-29: clean break, no legacy alias. After this
migration plus the accompanying Python edits, the only surviving name is
`content`.

Rollback is intentionally a no-op. Restoring the dropped column would not
restore the data anyway -- the SQL DDL drops the data with the column.
Any rollback would have to re-run from a snapshot.

SQLite supports ALTER TABLE DROP COLUMN since 3.35.0 (2021); on older
SQLite the migration falls back to a no-op (we can't drop the column on
legacy SQLite, so we leave it in place and trust the Python layer to
ignore it).
"""

from yoyo import step

__depends__ = {"022_add_content_column"}


def _drop_description_column(conn):
    """Drop the description column from entities if it exists."""
    cursor = conn.cursor()
    existing = {row[1] for row in cursor.execute("PRAGMA table_info(entities)").fetchall()}

    if "description" not in existing:
        return  # already dropped or never existed

    try:
        cursor.execute("ALTER TABLE entities DROP COLUMN description")
    except Exception:
        # SQLite < 3.35.0 doesn't support DROP COLUMN. Leave the column in
        # place; Python code no longer references it after the accompanying
        # caller migration.
        pass


def _noop_rollback(conn):
    """Rollback is a no-op. Dropping a column drops the data; rollback cannot
    restore it."""
    pass


steps = [step(_drop_description_column, _noop_rollback)]
