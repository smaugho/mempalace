"""intent_state v1 -> v2 migration.

v1 shape:
    {current_step: str, progress_pct: int, blockers: [str],
     latest_observation: str, open_questions: [str]}

v2 shape (Adrian 2026-05-04):
    {todos: [{id, text, status, blocker?}],
     active_todo_id: str | null,
     latest_observation: str}

Mapping:
- current_step (if present) becomes the active todo. progress_pct
  drives its status: <100 -> in_progress, 100 -> done, missing -> pending.
- blockers entries get demoted to a single text-blob blocker on
  the active todo (v2 has per-todo blockers, not a separate list).
- open_questions become additional pending todos so the agent
  can continue tracking them in the v2 shape.
- latest_observation passes through unchanged.
- progress_pct is dropped.

Idempotent: if payload already looks v2 (has 'todos'), return as-is.
"""

from __future__ import annotations


def migrate(payload: dict) -> dict:
    if isinstance(payload, dict) and "todos" in payload:
        return payload

    payload = payload or {}
    current_step = (payload.get("current_step") or "").strip()
    progress_pct = payload.get("progress_pct")
    blockers = payload.get("blockers") or []
    open_questions = payload.get("open_questions") or []

    todos: list[dict] = []
    active_todo_id: str | None = None

    if current_step:
        if isinstance(progress_pct, int) and progress_pct >= 100:
            status = "done"
        elif isinstance(progress_pct, int) and progress_pct > 0:
            status = "in_progress"
        else:
            status = "pending"
        active_blocker = "; ".join(str(b) for b in blockers) if blockers else None
        todos.append(
            {
                "id": "t1",
                "text": current_step,
                "status": status,
                "blocker": active_blocker,
            }
        )
        active_todo_id = "t1"

    for i, q in enumerate(open_questions, start=2):
        todos.append(
            {
                "id": f"t{i}",
                "text": str(q),
                "status": "pending",
                "blocker": None,
            }
        )

    return {
        "todos": todos,
        "active_todo_id": active_todo_id,
        "latest_observation": payload.get("latest_observation") or "",
    }
