"""v3.7.44 fail-blocking UserPromptSubmit regression tests.

Background
----------
Pre-v3.7.44 ``_append_pending_user_message`` had a single try/except.
A transient disk failure (file lock, permission, disk-full burst)
returned False; the hook then continued FAIL-LOUD-WITHOUT-BLOCKING --
the user prompt went through unrecorded, no queue entry existed, the
PreToolUse gate never blocked, and the agent produced a permanent
orphan user_message with no fulfilling context.

Empirical impact: 97.5% coverage on Adrian's live palace (11 orphans
of 437 messages).

v3.7.44 fix:
  1. Retry the disk write up to 3 attempts with backoff.
  2. On terminal failure, write a sentinel file at
     ``~/.mempalace/hook_state/queue_write_failures_<sid>.json``
     containing the lost message JSON + error repr + timestamp.
  3. PreToolUse gate reads the sentinel and denies all non-tier-0
     tools until cleared, mirroring the pending-queue block.

Tests use real ``Path``s under tmp_path so the retry + sentinel
machinery exercises actual disk I/O.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.integration


def test_v3744_sentinel_path_helper_exists():
    """The new sentinel path helper must exist and produce a stable
    per-session path matching the documented scheme."""
    from mempalace.hooks_cli import _queue_write_failure_sentinel_path

    path = _queue_write_failure_sentinel_path("sess-abc-123")
    assert path is not None
    assert path.name.startswith("queue_write_failures_")
    assert path.name.endswith(".json")


def test_v3744_append_retries_three_times_then_sentinel(tmp_path, monkeypatch):
    """Three failing writes must trigger sentinel creation; the sentinel
    must contain the lost message JSON + error trace."""
    from mempalace import hooks_cli

    # Redirect STATE_DIR so both pending + sentinel land in tmp_path.
    monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)

    msg = {"id": "msg_test_v3744_0", "text": "hello", "turn_idx": 0, "ts": "2026-05-19T10:00:00"}
    sid = "sess_v3744_test"

    # Patch path.write_text to always raise -- simulates persistent
    # disk failure on every retry attempt.
    real_write = Path.write_text
    call_count = {"n": 0}

    def boom_write_text(self, *a, **kw):
        # Let sentinel write through; only fail on the pending file
        if "pending_user_messages" in self.name:
            call_count["n"] += 1
            raise PermissionError("simulated lock")
        return real_write(self, *a, **kw)

    with patch.object(Path, "write_text", boom_write_text):
        ok = hooks_cli._append_pending_user_message(sid, msg)

    assert ok is False, "append must return False after retry exhaustion"
    assert call_count["n"] == 3, f"expected 3 retry attempts; got {call_count['n']}"

    sentinel = hooks_cli._queue_write_failure_sentinel_path(sid)
    assert sentinel.is_file(), "sentinel file must be written on terminal failure"

    payload = json.loads(sentinel.read_text(encoding="utf-8"))
    assert payload["session_id"] == sid
    assert len(payload["failures"]) == 1
    failure = payload["failures"][0]
    assert failure["message"]["id"] == "msg_test_v3744_0"
    assert "PermissionError" in failure["error"]
    assert "ts_attempt" in failure


def test_v3744_append_recovers_on_second_attempt(tmp_path, monkeypatch):
    """Retry logic must transparently recover from a transient failure.
    No sentinel should be written when retry succeeds within 3 attempts."""
    from mempalace import hooks_cli

    monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)

    sid = "sess_v3744_recover"
    msg = {
        "id": "msg_test_v3744_recover_0",
        "text": "x",
        "turn_idx": 0,
        "ts": "2026-05-19T10:00:00",
    }

    real_write = Path.write_text
    call_count = {"n": 0}

    def flaky_write_text(self, *a, **kw):
        if "pending_user_messages" in self.name:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError("transient")
            # second attempt succeeds
        return real_write(self, *a, **kw)

    with patch.object(Path, "write_text", flaky_write_text):
        ok = hooks_cli._append_pending_user_message(sid, msg)

    assert ok is True, "append must return True after transient failure recovers"
    sentinel = hooks_cli._queue_write_failure_sentinel_path(sid)
    assert not sentinel.is_file(), "sentinel must NOT be written when retry recovers"

    # And the actual queue file should contain the message.
    queue = hooks_cli._pending_user_messages_path(sid)
    assert queue.is_file()
    data = json.loads(queue.read_text(encoding="utf-8"))
    assert any(m.get("id") == "msg_test_v3744_recover_0" for m in data["messages"])


def test_v3744_pretooluse_denies_on_sentinel_presence(tmp_path, monkeypatch, capsys):
    """When the sentinel file exists for a session, the PreToolUse gate
    must DENY tool calls (except tier-0 carve-outs)."""
    from mempalace import hooks_cli

    monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)

    sid = "sess_v3744_sentinel_block"
    # Plant a sentinel manually
    sentinel = hooks_cli._queue_write_failure_sentinel_path(sid)
    sentinel.write_text(
        json.dumps(
            {
                "session_id": sid,
                "failures": [
                    {
                        "message": {"id": "msg_lost_0", "text": "lost"},
                        "error": "PermissionError: locked",
                        "ts_attempt": "2026-05-19T10:00:00Z",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    # Drive the hook with a non-carve-out tool call
    hook_data = {
        "tool_name": "Bash",
        "tool_input": {"command": "ls"},
        "session_id": sid,
    }
    hooks_cli.hook_pretooluse(hook_data, harness="claude-code")
    out = capsys.readouterr().out
    # The output is JSON to stdout; parse and assert deny
    decision = json.loads(out)
    hsi = decision.get("hookSpecificOutput") or {}
    assert hsi.get("permissionDecision") == "deny", f"expected deny; got {hsi}"
    reason = hsi.get("permissionDecisionReason", "")
    assert "queue-write failure sentinel" in reason
    assert "msg_lost_0" not in reason  # specifics live in the file, not the reason


def test_v3744_pretooluse_allows_tier0_despite_sentinel(tmp_path, monkeypatch, capsys):
    """Tier-0 carve-outs (AskUserQuestion, ToolSearch, mempalace_wake_up,
    mempalace_declare_user_intents) must STILL be allowed even when the
    sentinel is present -- they're the only recovery paths."""
    from mempalace import hooks_cli

    monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)

    sid = "sess_v3744_tier0_allowed"
    sentinel = hooks_cli._queue_write_failure_sentinel_path(sid)
    sentinel.write_text(json.dumps({"session_id": sid, "failures": [{}]}), encoding="utf-8")

    for tool in ("AskUserQuestion", "ToolSearch"):
        hook_data = {"tool_name": tool, "tool_input": {}, "session_id": sid}
        hooks_cli.hook_pretooluse(hook_data, harness="claude-code")
        out = capsys.readouterr().out
        decision = json.loads(out)
        hsi = decision.get("hookSpecificOutput") or {}
        # tier-0 must NOT be denied by the sentinel check
        assert hsi.get("permissionDecision") != "deny", (
            f"tier-0 tool {tool} must not be denied by sentinel; got {hsi}"
        )
