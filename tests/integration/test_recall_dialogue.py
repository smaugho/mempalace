"""Integration tests for ``mempalace.bg_status.tool_recall_dialogue``.

v3.8.0 (Adrian directive 2026-05-19): the new MemGPT/Letta-pattern
recall-tier endpoint reads ``kind='user_message'`` rows out of the
entities table for time-ordered/grep/since access to raw user-turn
text. Tests seed a live SQLite KG with synthetic user_message rows
and assert the public return contract directly -- per the v3.7.42
FINDING #Z lesson, mocks against intent are insufficient; only a
live invocation locks the implementation.

Fixture pattern: build an isolated KnowledgeGraph in tmp, point
``mcp_server._STATE.kg`` + ``_STATE.session_id`` at it via
monkeypatch, then call tool_recall_dialogue and assert on the
returned dict shape.
"""

from __future__ import annotations

import pytest

from mempalace import bg_status
from mempalace import mcp_server

pytestmark = pytest.mark.integration


SID = "abc123xyz"  # Stable test session id; hash-prefix derived in helper.


def _seed_user_messages(kg, session_id=SID):
    """Insert three synthetic user_message rows with distinct timestamps.

    Mirrors the production path in intent.py::tool_declare_user_intents
    (kind='user_message', importance=3, properties carrying session_id +
    turn_idx + ts + added_by) but also passes session_id at the entities
    column level so the bg_status WHERE filter matches either branch.

    Returns the inserted message-id list in chronological insert order.
    """
    from mempalace.hooks_cli import _make_user_message_id

    msgs = [
        ("hello world first turn", "2026-05-19T10:00:00"),
        ("the user asks about widget refactor", "2026-05-19T10:05:00"),
        ("final message mentions WIDGET again", "2026-05-19T10:10:00"),
    ]
    ids = []
    for turn_idx, (text, ts) in enumerate(msgs):
        mid = _make_user_message_id(session_id, turn_idx, text)
        kg.add_entity(
            mid,
            kind="user_message",
            content=text,
            importance=3,
            session_id=session_id,
            properties={
                "type": "user_message",
                "session_id": session_id,
                "turn_idx": turn_idx,
                "ts": ts,
            },
        )
        # Force created_at to the synthetic ts so ORDER BY is deterministic.
        with kg._conn() as c:
            c.execute(
                "UPDATE entities SET created_at = ? WHERE id = ?",
                (ts, mid),
            )
        ids.append(mid)
    return ids


@pytest.fixture
def palace_with_messages(kg, monkeypatch):
    """Live KG with three user_message rows, wired into mcp_server._STATE."""
    ids = _seed_user_messages(kg)
    monkeypatch.setattr(mcp_server._STATE, "kg", kg)
    monkeypatch.setattr(mcp_server._STATE, "session_id", SID)
    return kg, ids


class TestReturnContract:
    def test_default_returns_all_three_chronologically(self, palace_with_messages):
        """Default call surfaces every seeded turn in chronological order."""
        _, ids = palace_with_messages
        out = bg_status.tool_recall_dialogue()
        assert out["session_id"] == SID
        assert out["count"] == 3
        assert out["total_matched"] == 3
        assert [t["id"] for t in out["transcript"]] == ids  # oldest -> newest

    def test_transcript_shape(self, palace_with_messages):
        """Each transcript entry exposes id / ts / text / session_id."""
        out = bg_status.tool_recall_dialogue()
        for entry in out["transcript"]:
            assert set(entry.keys()) == {"id", "ts", "text", "session_id"}
            assert entry["session_id"] == SID
            # v3.7.39 minute-precision trim: 'YYYY-MM-DDTHH:MM' (16 chars)
            assert len(entry["ts"]) == 16

    def test_last_n_clamps_slice(self, palace_with_messages):
        """last_n=1 returns only the most recent turn but reports
        total_matched=3 so the agent knows more history exists."""
        out = bg_status.tool_recall_dialogue(last_n=1)
        assert out["count"] == 1
        assert out["total_matched"] == 3
        assert out["transcript"][0]["text"] == "final message mentions WIDGET again"

    def test_last_n_clamped_to_bounds(self, palace_with_messages):
        """last_n outside [1, 200] silently clamps -- never errors."""
        out = bg_status.tool_recall_dialogue(last_n=0)
        assert out["count"] >= 1  # 0 clamped up to 1
        out = bg_status.tool_recall_dialogue(last_n=9999)
        assert out["count"] == 3  # 9999 clamped down, still <=3 rows exist


class TestFilters:
    def test_grep_case_insensitive_substring(self, palace_with_messages):
        """grep matches case-insensitively against content."""
        out = bg_status.tool_recall_dialogue(grep="widget")
        # 'widget refactor' + 'WIDGET again' both match
        assert out["count"] == 2
        assert out["total_matched"] == 2
        texts = [t["text"].lower() for t in out["transcript"]]
        assert all("widget" in t for t in texts)

    def test_grep_no_match_returns_empty(self, palace_with_messages):
        out = bg_status.tool_recall_dialogue(grep="nonexistent_token_xyz")
        assert out["count"] == 0
        assert out["total_matched"] == 0
        assert out["transcript"] == []

    def test_since_filter_cuts_old_turns(self, palace_with_messages):
        """since cutoff excludes earlier turns."""
        out = bg_status.tool_recall_dialogue(since="2026-05-19T10:05:00")
        # Turns at 10:05 and 10:10 survive; the 10:00 one is excluded.
        assert out["count"] == 2
        ts_values = [t["ts"] for t in out["transcript"]]
        assert ts_values == ["2026-05-19T10:05", "2026-05-19T10:10"]

    def test_explicit_session_id_overrides_default(self, palace_with_messages, kg):
        """Passing session_id='' skips the session filter (cross-session
        recall pattern -- documented in the tool's docstring)."""
        out = bg_status.tool_recall_dialogue(session_id="")
        assert out["count"] == 3  # still finds all three


class TestCarveOut:
    def test_no_session_returns_safe_default(self, kg, monkeypatch):
        """If _STATE.session_id is empty AND no session_id passed, the
        tool still runs (it just doesn't apply a session filter) and
        returns whatever user_messages exist."""
        _seed_user_messages(kg)
        monkeypatch.setattr(mcp_server._STATE, "kg", kg)
        monkeypatch.setattr(mcp_server._STATE, "session_id", "")
        out = bg_status.tool_recall_dialogue()
        assert out["session_id"] == ""
        assert out["count"] == 3
