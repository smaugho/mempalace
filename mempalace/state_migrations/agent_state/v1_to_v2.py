"""agent_state v1 -> v2 migration.

v1 shape:
    {current_focus: str, active_intent_id: str|null,
     recent_findings: [str], pending_followups: [str]}

v2 shape (Adrian 2026-05-04):
    {current_focus: str, active_intent_id: str|null,
     pending_followups: [{id, text, status, blocker?}],
     recent_findings: [str]}

Mapping:
- current_focus, active_intent_id, recent_findings pass through.
- pending_followups goes from [str] to [{id, text, status='pending',
  blocker=None}] so v2's todo-shape patches work uniformly. Existing
  followups become 'pending' since v1 had no per-item status.

Idempotent: if pending_followups[0] is already a dict, return as-is.
"""

from __future__ import annotations


def migrate(payload: dict) -> dict:
    payload = payload or {}
    followups = payload.get("pending_followups") or []
    if followups and isinstance(followups[0], dict):
        return payload

    new_followups = [
        {"id": f"f{i + 1}", "text": str(t), "status": "pending", "blocker": None}
        for i, t in enumerate(followups)
    ]

    return {
        "current_focus": payload.get("current_focus") or "",
        "active_intent_id": payload.get("active_intent_id"),
        "pending_followups": new_followups,
        "recent_findings": list(payload.get("recent_findings") or [])[-5:],
    }
