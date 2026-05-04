"""project_state v1 -> v2 migration.

v1 shape:
    {current_phase: str, active_branches: [str], blockers: [str],
     recent_milestones: [{name, date}], open_questions: [str]}

v2 shape (Adrian 2026-05-04):
    {current_phase: str, active_branches: [str],
     open_todos: [{id, text, status, blocker?}],
     recent_milestones: [{name, date}]}

Mapping:
- current_phase, active_branches, recent_milestones pass through.
- open_questions becomes open_todos with status='pending'. v2 lets
  agents resolve them by flipping status to 'done' instead of
  removing from the list -- preserves history.
- v1 blockers (project-level) get prepended to open_todos as
  status='blocked' items so they stay visible in the todo list.

Idempotent: if payload already has 'open_todos', return as-is.
"""

from __future__ import annotations


def migrate(payload: dict) -> dict:
    if isinstance(payload, dict) and "open_todos" in payload:
        return payload

    payload = payload or {}
    blockers = payload.get("blockers") or []
    open_questions = payload.get("open_questions") or []

    open_todos: list[dict] = []
    next_id = 1
    for b in blockers:
        open_todos.append(
            {
                "id": f"todo{next_id}",
                "text": str(b),
                "status": "blocked",
                "blocker": str(b),
            }
        )
        next_id += 1
    for q in open_questions:
        open_todos.append(
            {
                "id": f"todo{next_id}",
                "text": str(q),
                "status": "pending",
                "blocker": None,
            }
        )
        next_id += 1

    return {
        "current_phase": payload.get("current_phase") or "",
        "active_branches": list(payload.get("active_branches") or []),
        "open_todos": open_todos,
        "recent_milestones": list(payload.get("recent_milestones") or [])[-5:],
    }
