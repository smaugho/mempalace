"""022: Add `content` column to entities -- step 1 of description->content rename.

Additive migration ONLY. Does NOT rename or drop the existing `description`
column. Both columns coexist after this migration; old code that reads or
writes `description` keeps working unchanged. Subsequent commits will flip
Python reads/writes to `content` while dual-writing both for safety; a
later cleanup migration (023) will drop `description`.

This shape -- ADD COLUMN + backfill, no drop -- is what makes the rename
shippable across many commits without the "yoyo auto-applies on import"
atomicity wall that reverted 5 prior attempts. See
record_ga_agent_result_rename_description_to_content_deferred_2026 in the
KG for the full architectural rationale.

Mirrors the idempotent PRAGMA-introspection idiom established in
002_entity_metadata_columns.py: read PRAGMA table_info(entities), only
ALTER if the column is missing. Runs cleanly on fresh databases (column
added) and existing palaces (column already present is a no-op).
Rollback is intentionally a no-op; keeping the column is safe.
"""

from yoyo import step

__depends__ = {"021_op_cluster_templatizable"}


def _add_content_column(conn):
    """Add `content` column if missing, then backfill from `description`."""
    cursor = conn.cursor()
    existing = {row[1] for row in cursor.execute("PRAGMA table_info(entities)").fetchall()}

    if "content" not in existing:
        cursor.execute("ALTER TABLE entities ADD COLUMN content TEXT DEFAULT ''")

    # Backfill: copy description into content where content is still empty
    # and description has a value. Idempotent -- re-running is a no-op
    # because the WHERE clause filters out already-populated rows.
    cursor.execute(
        "UPDATE entities SET content = description "
        "WHERE (content IS NULL OR content = '') "
        "AND description IS NOT NULL AND description != ''"
    )


def _noop_rollback(conn):
    """Rollback is a no-op. Dropping `content` would lose any writes that
    landed after subsequent commits flip code to write `content` directly,
    so we keep the column. Safe to leave behind."""
    pass


steps = [step(_add_content_column, _noop_rollback)]
