"""002: Entity metadata columns — description, importance, decay, status, kind.

Written as a Python migration (not .sql) because SQLite can't do
'ALTER TABLE ... ADD COLUMN IF NOT EXISTS'. We inspect PRAGMA table_info
and only add columns that are missing — so this runs cleanly on both
fresh databases (all columns added) and partial legacy databases
(some columns already present).
"""

from yoyo import step

__depends__ = {"001_initial_schema"}


def _add_missing_columns(conn):
    """Add each metadata column only if not already present."""
    cursor = conn.cursor()
    existing = {row[1] for row in cursor.execute("PRAGMA table_info(entities)").fetchall()}
    statements = [
        ("description", "ALTER TABLE entities ADD COLUMN description TEXT DEFAULT ''"),
        ("importance", "ALTER TABLE entities ADD COLUMN importance INTEGER DEFAULT 3"),
        ("last_touched", "ALTER TABLE entities ADD COLUMN last_touched TEXT DEFAULT ''"),
        ("status", "ALTER TABLE entities ADD COLUMN status TEXT DEFAULT 'active'"),
        ("merged_into", "ALTER TABLE entities ADD COLUMN merged_into TEXT DEFAULT NULL"),
        ("kind", "ALTER TABLE entities ADD COLUMN kind TEXT DEFAULT 'entity'"),
    ]
    for col, sql in statements:
        if col not in existing:
            cursor.execute(sql)

    # Index and backfill (idempotent)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_status ON entities(status)")
    cursor.execute(
        "UPDATE entities SET kind = 'predicate' "
        "WHERE type = 'predicate' AND (kind IS NULL OR kind = 'entity')"
    )


def _noop_rollback(conn):
    """Rollback intentionally drops nothing — keeping columns is safe."""
    pass


steps = [step(_add_missing_columns, _noop_rollback)]
