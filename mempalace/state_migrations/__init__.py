"""State-schema migration runner -- Phase 6 lazy-migration-at-injection.

Adrian design lock 2026-05-03. When STATE_SCHEMAS[schema_id].version
bumps, the author also drops a migration file at
``mempalace/state_migrations/{schema_id}/v{N}_to_v{N+1}.py`` exporting::

    def migrate(payload: dict) -> dict:
        '''Transform a v{N} payload into a v{N+1} payload.'''
        ...

The injection-gate hook (mempalace/injection_gate.py:apply_gate) calls
``apply_pending_migrations`` for each kept entity whose latest revision
is at a version below the current schema. Migrations apply lazily:
dormant entities never trigger the cost; only entities surfaced to the
LLM and not gate-filtered get migrated. The migrated payload is
written as a NEW revision row (preserves audit trail; matches the
existing JTMS pattern of "every state change is a row"), with
schema_version stamped to the now-current value.

Discovery is import-based and lazy -- modules under this package are
only imported when needed for a specific schema_id + version step.
This means importing this package at server start has zero side
effects beyond registering the package itself.

Idempotency: migration authors must write idempotent transforms
(applying twice == applying once). The runner does NOT guard against
double-application -- if a migration is non-idempotent and the
runtime crashes mid-write, the half-applied state may survive. Future
hardening could wrap each migration step in a transaction, but v1
relies on author discipline. Tests for each migration should assert
``migrate(migrate(payload)) == migrate(payload)`` to catch
regressions.

Errors: missing migration files raise ImportError; the runner re-raises
so the caller (apply_gate) can fail-open and surface a flag to the
gardener instead of crashing the gate path.
"""

from __future__ import annotations

import importlib


class StateMigrationError(Exception):
    """Raised when a migration step fails or its module is missing."""


def apply_pending_migrations(
    payload: dict,
    schema_id: str,
    from_version: int,
    to_version: int,
) -> dict:
    """Walk the migration chain ``from_version`` -> ``to_version`` for the
    given ``schema_id`` and apply each step to ``payload`` in order.

    Returns the migrated payload. No-op when ``from_version >= to_version``
    (returns the input unchanged). Raises ``StateMigrationError`` when a
    step's migrate module is missing or its ``migrate(payload)`` call
    raises.

    This function is pure: it does not write to the database. The
    caller (typically ``KnowledgeGraph.migrate_state_for_entities``)
    is responsible for persisting the migrated payload as a new
    revision via ``record_state_revision`` with the current
    schema_version.
    """
    if from_version >= to_version:
        return payload
    if not isinstance(schema_id, str) or not schema_id:
        raise StateMigrationError(
            f"apply_pending_migrations: schema_id must be a non-empty string, got {schema_id!r}."
        )

    current = payload
    for v in range(int(from_version), int(to_version)):
        module_path = f"mempalace.state_migrations.{schema_id}.v{v}_to_v{v + 1}"
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise StateMigrationError(
                f"apply_pending_migrations: missing migration module "
                f"{module_path!r}; cannot migrate {schema_id} from "
                f"v{v} to v{v + 1}. Author must add the file with a "
                f"`migrate(payload: dict) -> dict` function."
            ) from exc
        migrate_fn = getattr(module, "migrate", None)
        if not callable(migrate_fn):
            raise StateMigrationError(
                f"apply_pending_migrations: module {module_path!r} "
                f"does not export a callable `migrate`."
            )
        try:
            current = migrate_fn(current)
        except Exception as exc:  # pragma: no cover - migration author's domain
            raise StateMigrationError(
                f"apply_pending_migrations: migrate step v{v}->v{v + 1} "
                f"for {schema_id} raised {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(current, dict):
            raise StateMigrationError(
                f"apply_pending_migrations: migrate step v{v}->v{v + 1} "
                f"for {schema_id} returned {type(current).__name__}, "
                f"expected dict."
            )
    return current


__all__ = ["StateMigrationError", "apply_pending_migrations"]
