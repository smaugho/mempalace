"""Integration tests for kg_search.include_user_messages (v3.9.0).

v3.9.0 (Adrian directive 2026-05-19): replaces the standalone
mempalace_recall_dialogue tool that shipped in v3.8.0. The recall-tier
functionality is now an opt-in flag on kg_search:

- Default include_user_messages=False -- strips kind=user_message rows
  from results (preserves the v3.7.43 FINDING #AA leak fix: bare
  user-turn text never appears as a memory in normal retrieval).

- include_user_messages=True -- additionally scans kind=user_message
  content for case-insensitive substring matches against any keyword
  in context.keywords; appends matching rows to the result list with
  source='memory' and meta.kind='user_message'.

These tests seed three synthetic user_message rows into an isolated
tmp KnowledgeGraph (via the seeded_kg fixture) and assert both branches.
"""

from __future__ import annotations

import pytest

from mempalace import mcp_server
from mempalace.tool_read import tool_kg_search

pytestmark = pytest.mark.integration


def _seed_user_messages(kg):
    """Insert three kind=user_message rows with distinct content."""
    from mempalace.hooks_cli import _make_user_message_id

    sid = "abc123xyz"
    msgs = [
        ("the user asks about widget refactor work", "2026-05-19T10:00:00"),
        ("final user message mentions WIDGET cleanup", "2026-05-19T10:05:00"),
        ("an unrelated turn about coffee preferences", "2026-05-19T10:10:00"),
    ]
    ids = []
    for turn_idx, (text, ts) in enumerate(msgs):
        mid = _make_user_message_id(sid, turn_idx, text)
        kg.add_entity(
            mid,
            kind="user_message",
            content=text,
            importance=3,
            session_id=sid,
            properties={"type": "user_message", "session_id": sid, "turn_idx": turn_idx},
        )
        with kg._conn() as c:
            c.execute("UPDATE entities SET created_at = ? WHERE id = ?", (ts, mid))
        ids.append(mid)
    return ids, sid


@pytest.fixture
def wired_state(seeded_kg, palace_path, monkeypatch):
    """seeded_kg + 3 user_message rows wired into mcp_server._STATE."""
    ids, sid = _seed_user_messages(seeded_kg)
    monkeypatch.setattr(mcp_server._STATE, "kg", seeded_kg)
    monkeypatch.setattr(mcp_server._STATE, "session_id", sid)

    # tool_kg_search reaches into _STATE.config.palace_path for the
    # vector store; stub a minimal config object so the call doesn't
    # NPE on the .config access. The vector store starts empty; the
    # side-channel SQL scan over kind=user_message does NOT depend
    # on any vector content.
    class _Cfg:
        pass

    _cfg = _Cfg()
    _cfg.palace_path = palace_path
    monkeypatch.setattr(mcp_server._STATE, "config", _cfg)
    return seeded_kg, ids


class TestDefaultFiltersUserMessages:
    def test_default_omits_user_messages_from_results(self, wired_state):
        """Default kg_search call must NOT surface any kind=user_message row,
        even when keywords obviously match user-turn content."""
        out = tool_kg_search(
            context={
                "queries": ["widget refactor work", "widget cleanup"],
                "keywords": ["widget", "refactor", "cleanup"],
            },
            agent="test_agent",
            limit=20,
        )
        assert "results" in out
        for r in out["results"]:
            kind = (r.get("meta") or {}).get("kind") or ""
            assert kind != "user_message", (
                f"v3.7.43 leak guard violated: kg_search returned user_message {r.get('id')}"
            )


class TestOptInSurfacesUserMessages:
    def test_opt_in_finds_user_messages_via_keyword(self, wired_state):
        """include_user_messages=True must surface kind=user_message rows
        whose content contains any keyword (case-insensitive)."""
        out = tool_kg_search(
            context={
                "queries": ["widget refactor work", "widget cleanup"],
                "keywords": ["widget", "refactor"],
            },
            agent="test_agent",
            include_user_messages=True,
            limit=20,
        )
        user_msg_hits = [
            r for r in out["results"] if (r.get("meta") or {}).get("kind") == "user_message"
        ]
        # Two of the three seeded turns contain 'widget' (case-insensitive).
        assert len(user_msg_hits) == 2
        texts = [(r.get("content") or "").lower() for r in user_msg_hits]
        assert all("widget" in t for t in texts)

    def test_opt_in_with_no_matching_keyword_returns_none(self, wired_state):
        """include_user_messages=True with a keyword absent from all user
        turns surfaces zero user_message rows (but doesn't error)."""
        out = tool_kg_search(
            context={
                "queries": ["something else entirely", "no match expected"],
                "keywords": ["nonexistent_token_xyz", "another_absent_word"],
            },
            agent="test_agent",
            include_user_messages=True,
            limit=20,
        )
        user_msg_hits = [
            r for r in out["results"] if (r.get("meta") or {}).get("kind") == "user_message"
        ]
        assert user_msg_hits == []

    def test_opt_in_case_insensitive(self, wired_state):
        """Case-insensitive SQL LIKE: lowercase keyword matches uppercase
        content and vice-versa."""
        out = tool_kg_search(
            context={
                "queries": ["coffee preferences", "coffee taste"],
                "keywords": ["COFFEE", "PREFERENCES"],
            },
            agent="test_agent",
            include_user_messages=True,
            limit=20,
        )
        user_msg_hits = [
            r for r in out["results"] if (r.get("meta") or {}).get("kind") == "user_message"
        ]
        assert len(user_msg_hits) == 1
        assert "coffee" in (user_msg_hits[0].get("content") or "").lower()
