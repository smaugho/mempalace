"""SQL migrations for the mempalace knowledge graph.

Each .sql file is a versioned migration applied once and tracked by yoyo.
New schema changes must be added as a new migration file (NNN_description.sql)
-- never edit an already-applied migration.

Order: migrations apply in filename lexicographic order. Use three-digit
numeric prefixes (001_, 002_, ...) to preserve ordering.
"""

from pathlib import Path

MIGRATIONS_DIR = Path(__file__).parent

# yoyo scans this directory for migrations and will pick up __init__.py as a
# Python migration. Declaring an empty steps list makes it a no-op marker.
steps = []


def apply_migrations(db_path: str) -> None:
    """Apply all pending yoyo migrations to the given SQLite database.

    Idempotent: migrations already applied are skipped.
    """
    from yoyo import get_backend, read_migrations

    backend = get_backend(f"sqlite:///{db_path}")
    migrations = read_migrations(str(MIGRATIONS_DIR))
    with backend.lock():
        backend.apply_migrations(backend.to_apply(migrations))


def rollback_last(db_path: str) -> None:
    """Roll back the most recently applied migration. Testing/dev aid."""
    from yoyo import get_backend, read_migrations

    backend = get_backend(f"sqlite:///{db_path}")
    migrations = read_migrations(str(MIGRATIONS_DIR))
    with backend.lock():
        to_rollback = backend.to_rollback(migrations)
        if to_rollback:
            backend.rollback_migrations(to_rollback[-1:])


__all__ = ["apply_migrations", "rollback_last", "MIGRATIONS_DIR"]
