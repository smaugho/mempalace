"""
test_subagent_sid_isolation.py — Isolation of subagent state from parent
state via the composite session-id scheme.

Claude Code subagents (dispatched via the Task tool) make tool calls
inside the SAME MCP session as the top-level conversation — the
PreToolUse hook payload's ``session_id`` is identical for both. Without
further disambiguation, a subagent's tool call could mutate the
parent's active-intent state and persist it to the parent's state
file, trapping the parent with pending state it never created. This
was the live failure mode on 2026-04-19.

Fix: the hook reads the ``agent_id`` field (present only when the
hook fires inside a subagent, unique per subagent invocation, stable
across every tool call in that invocation) and folds it into a
composite sid — ``<session>__sub_<agent_id>``. The server then
scopes its on-disk pending-state file and in-memory ``_STATE`` cache
by the composite sid, so subagent state lives in its own file and its
own cache slot.

These tests lock in:

  - The composite-sid format is deterministic for any (session_id,
    agent_id) pair.
  - The parent hook call emits the base sid; the subagent hook call
    emits a distinct composite sid.
  - Two DIFFERENT subagent invocations inside the same session emit
    DIFFERENT composite sids.
  - The server (via end-to-end round-trip) treats a composite sid like
    any other sid: pending state for a subagent's composite sid never
    contaminates the parent's base-sid state and vice versa.
"""

from __future__ import annotations

import io
import json
from unittest.mock import patch


from mempalace.hooks_cli import _effective_session_id, _sanitize_session_id, hook_pretooluse


# ═══════════════════════════════════════════════════════════════════════
#  _effective_session_id — pure function
# ═══════════════════════════════════════════════════════════════════════


class TestEffectiveSessionId:
    def test_no_agent_id_returns_base_session(self):
        assert _effective_session_id({"session_id": "abc123"}) == "abc123"

    def test_empty_agent_id_returns_base_session(self):
        assert _effective_session_id({"session_id": "abc123", "agent_id": ""}) == "abc123"

    def test_none_agent_id_returns_base_session(self):
        assert _effective_session_id({"session_id": "abc123", "agent_id": None}) == "abc123"

    def test_with_agent_id_returns_composite(self):
        assert (
            _effective_session_id({"session_id": "abc123", "agent_id": "explore-7"})
            == "abc123__sub_explore-7"
        )

    def test_composite_is_sanitizer_safe(self):
        """Composite sid must pass through _sanitize_session_id unchanged."""
        composite = _effective_session_id({"session_id": "abc123", "agent_id": "a-b_c"})
        assert composite == "abc123__sub_a-b_c"
        assert _sanitize_session_id(composite) == composite

    def test_different_agent_ids_same_session_produce_different_sids(self):
        """Two subagents in the same conversation must not collide."""
        a = _effective_session_id({"session_id": "sess", "agent_id": "alpha"})
        b = _effective_session_id({"session_id": "sess", "agent_id": "beta"})
        assert a != b
        assert a == "sess__sub_alpha"
        assert b == "sess__sub_beta"

    def test_same_agent_id_different_sessions_differ(self):
        """Agent-id reuse across sessions still maps to distinct composite sids."""
        assert _effective_session_id(
            {"session_id": "s1", "agent_id": "x"}
        ) != _effective_session_id({"session_id": "s2", "agent_id": "x"})

    def test_agent_id_without_session_still_scoped(self):
        """Defensive: if base session_id is missing for some reason, we still
        emit a usable subagent-scoped sid so state isn't silently written to
        'unknown' or 'default'."""
        result = _effective_session_id({"session_id": "", "agent_id": "a1"})
        assert result.startswith("sub_")
        assert "a1" in result

    def test_malicious_agent_id_characters_are_stripped(self):
        """Path-traversal attempts in agent_id must not leak into the sid."""
        result = _effective_session_id({"session_id": "sess", "agent_id": "../../etc/passwd"})
        # The sanitizer drops slashes and dots.
        assert "/" not in result
        assert ".." not in result
        assert result == "sess__sub_etcpasswd"

    def test_non_string_agent_id_coerced(self):
        """Defensive against unusual payload shapes (numeric agent ids etc.)."""
        result = _effective_session_id({"session_id": "sess", "agent_id": 42})
        assert result == "sess__sub_42"


# ═══════════════════════════════════════════════════════════════════════
#  hook_pretooluse — subagent vs parent payload
# ═══════════════════════════════════════════════════════════════════════


def _capture_hook_output(data, harness="claude"):
    """Run hook_pretooluse and return the parsed JSON response."""
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        hook_pretooluse(data, harness)
    raw = buf.getvalue().strip()
    if not raw:
        return None
    return json.loads(raw)


class TestHookPreToolUseSubagentInjection:
    """The hook must inject the composite sid into the MCP tool input when
    the payload identifies a subagent call; it must inject the base sid
    (current behavior) for parent calls."""

    def test_parent_call_injects_base_sid(self):
        payload = {
            "session_id": "parent-sess",
            "tool_name": "mcp__plugin_3_0_14_mempalace__mempalace_kg_stats",
            "tool_input": {},
            # no agent_id
        }
        out = _capture_hook_output(payload)
        assert out is not None
        updated = out["hookSpecificOutput"].get("updatedInput")
        assert updated is not None
        assert updated["sessionId"] == "parent-sess"

    def test_subagent_call_injects_composite_sid(self):
        payload = {
            "session_id": "parent-sess",
            "agent_id": "explore-7",
            "agent_type": "Explore",
            "tool_name": "mcp__plugin_3_0_14_mempalace__mempalace_kg_stats",
            "tool_input": {},
        }
        out = _capture_hook_output(payload)
        assert out is not None
        updated = out["hookSpecificOutput"].get("updatedInput")
        assert updated is not None
        assert updated["sessionId"] == "parent-sess__sub_explore-7"

    def test_two_subagents_in_same_session_get_different_sids(self):
        base = {
            "session_id": "parent-sess",
            "tool_name": "mcp__plugin_3_0_14_mempalace__mempalace_kg_stats",
            "tool_input": {},
        }
        out_a = _capture_hook_output({**base, "agent_id": "alpha"})
        out_b = _capture_hook_output({**base, "agent_id": "beta"})
        sid_a = out_a["hookSpecificOutput"]["updatedInput"]["sessionId"]
        sid_b = out_b["hookSpecificOutput"]["updatedInput"]["sessionId"]
        assert sid_a != sid_b
        assert sid_a == "parent-sess__sub_alpha"
        assert sid_b == "parent-sess__sub_beta"
