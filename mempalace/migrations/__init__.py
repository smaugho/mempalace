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


def _try_load_vec(connection) -> None:
    """Best-effort load of the sqlite-vec (vec0) extension on a connection.

    Pre-v3.2.0 environments may lack sqlite_vec; in that case the vec_palace
    cascade degrades to app-layer cleanup (same contract as
    knowledge_graph._conn). Never raises.
    """
    try:
        connection.enable_load_extension(True)
        try:
            import sqlite_vec  # noqa: PLC0415

            sqlite_vec.load(connection)
        finally:
            connection.enable_load_extension(False)
    except Exception:
        pass


def _ensure_vec_on_all_yoyo_connections(backend) -> None:
    """Make every connection yoyo creates load the vec0 extension.

    v3.10.1 (boot-crash fix 2026-05-21): the BEFORE DELETE trigger on
    vec_rowid_map cascades ``DELETE FROM vec_palace`` -- vec_palace is a vec0
    virtual table, so vec0 MUST be loaded on whatever connection runs the
    DELETE. Any migration that deletes vec_rowid_map rows (e.g. migration 030's
    memory_flags table rebuild, which cascades through FK enforcement) fires
    this trigger; without vec0 the migration aborts with 'no such module: vec0'
    and the entire MCP server fails to boot.

    yoyo runs each migration on a COPY of the backend (``apply_one`` ->
    ``with self.copy()``), and ``copy()`` builds a fresh backend of the same
    class with its own connection. So loading vec0 on the top-level connection
    alone is not enough -- the per-migration copy never sees it. Instead we
    wrap the backend class's ``init_connection`` hook, which yoyo calls for
    EVERY connection it constructs (initial backend, each copy, and after any
    rollback). Patch is idempotent and applied once per class.
    """
    cls = type(backend)
    if not getattr(cls, "_mempalace_vec_patched", False):
        orig_init_connection = cls.init_connection

        def init_connection(self, connection):  # noqa: ANN001
            orig_init_connection(self, connection)
            _try_load_vec(connection)

        cls.init_connection = init_connection
        cls._mempalace_vec_patched = True
    # The top-level backend's connection was built before the patch landed.
    _try_load_vec(getattr(backend, "_connection", None) or backend.connection)


def apply_migrations(db_path: str) -> None:
    """Apply all pending yoyo migrations to the given SQLite database.

    Idempotent: migrations already applied are skipped.
    """
    from yoyo import get_backend, read_migrations

    backend = get_backend(f"sqlite:///{db_path}")
    _ensure_vec_on_all_yoyo_connections(backend)
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
