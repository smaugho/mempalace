"""task_state v1 -> v2 migration.

v1 shape:
    {status: enum, assignee: str|null, due_date: str|null,
     blockers: [str], progress_pct: int}

v2 shape (Adrian 2026-05-04):
    {status: enum, subtodos: [{id, text, status, blocker?}],
     assignee: str|null, due_date: str|null, blocker: str|null}

Mapping:
- status, assignee, due_date pass through.
- blockers (list) collapses to a single string blocker (joined with
  '; ' if multiple). v2's per-task blocker is one free-form note.
- progress_pct is dropped -- v2 expresses progress via subtodos
  status, not a number.
- subtodos starts empty; agents populate when the task has
  compound work.

Idempotent: if payload already has 'subtodos', return as-is.
"""

from __future__ import annotations


def migrate(payload: dict) -> dict:
    if isinstance(payload, dict) and "subtodos" in payload:
        return payload

    payload = payload or {}
    blockers = payload.get("blockers") or []
    blocker = "; ".join(str(b) for b in blockers) if blockers else None

    return {
        "status": payload.get("status") or "open",
        "subtodos": [],
        "assignee": payload.get("assignee"),
        "due_date": payload.get("due_date"),
        "blocker": blocker,
    }
