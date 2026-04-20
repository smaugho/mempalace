import contextlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mempalace.hooks_cli import (
    LOCAL_CUE_ASSISTANT_MAX_CHARS,
    LOCAL_RETRIEVAL_MAX_CHARS,
    NEVER_STOP_BLOCK_REASON,
    REHYDRATION_MAX_CHARS,
    SAVE_INTERVAL,
    STOP_BLOCK_REASON,
    PRECOMPACT_WARNING_MESSAGE,
    USER_PROMPT_CUE_PREFIX,
    _extract_askuserquestion_texts,
    _extract_prompt_keywords,
    _format_retrieval_additional_context,
    _is_lazy_question,
    _is_local_retrieval_enabled,
    _is_non_iterative_mode,
    _maybe_build_local_retrieval_context,
    _maybe_deny_lazy_askuserquestion,
    _persist_accessed_memory_ids,
    _read_last_finalized_intent,
    _run_local_retrieval,
    _basename,
    _build_local_cue,
    _build_rehydration_payload,
    _check_permission,
    _count_human_messages,
    _log,
    _maybe_auto_ingest,
    _namespaced_args,
    _parent_path_tokens,
    _parse_harness_input,
    _read_last_assistant_message,
    _read_recent_trace,
    _sanitize_session_id,
    _summarize_tool_args,
    hook_stop,
    hook_pretooluse,
    hook_session_start,
    hook_precompact,
    hook_userpromptsubmit,
    run_hook,
)


# --- _sanitize_session_id ---


def test_sanitize_normal_id():
    assert _sanitize_session_id("abc-123_XYZ") == "abc-123_XYZ"


def test_sanitize_strips_dangerous_chars():
    assert _sanitize_session_id("../../etc/passwd") == "etcpasswd"


def test_sanitize_empty_returns_empty():
    """NO cross-agent fallback — empty or fully-stripped input returns "".

    The old behavior was ``"unknown"``, which every agent without a real
    sid ended up writing to a shared file — cross-contamination that
    caused the 2026-04-19 deadlocks. Empty-in-empty-out; callers decide
    whether to refuse the operation or log and skip.
    """
    assert _sanitize_session_id("") == ""
    assert _sanitize_session_id("!!!") == ""


# --- _count_human_messages ---


def _write_transcript(path: Path, entries: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def test_count_human_messages_basic(tmp_path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            {"message": {"role": "user", "content": "hello"}},
            {"message": {"role": "assistant", "content": "hi"}},
            {"message": {"role": "user", "content": "bye"}},
        ],
    )
    assert _count_human_messages(str(transcript)) == 2


def test_count_skips_command_messages(tmp_path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            {"message": {"role": "user", "content": "<command-message>status</command-message>"}},
            {"message": {"role": "user", "content": "real question"}},
        ],
    )
    assert _count_human_messages(str(transcript)) == 1


def test_count_handles_list_content(tmp_path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            {"message": {"role": "user", "content": [{"type": "text", "text": "hello"}]}},
            {
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "<command-message>x</command-message>"}],
                }
            },
        ],
    )
    assert _count_human_messages(str(transcript)) == 1


def test_count_missing_file():
    assert _count_human_messages("/nonexistent/path.jsonl") == 0


def test_count_empty_file(tmp_path):
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("")
    assert _count_human_messages(str(transcript)) == 0


def test_count_malformed_json_lines(tmp_path):
    transcript = tmp_path / "t.jsonl"
    transcript.write_text('not json\n{"message": {"role": "user", "content": "ok"}}\n')
    assert _count_human_messages(str(transcript)) == 1


# --- hook_stop ---


def _capture_hook_output(hook_fn, data, harness="claude-code", state_dir=None):
    """Run a hook and capture its JSON stdout output."""
    import io

    buf = io.StringIO()
    patches = [patch("mempalace.hooks_cli._output", side_effect=lambda d: buf.write(json.dumps(d)))]
    if state_dir:
        patches.append(patch("mempalace.hooks_cli.STATE_DIR", state_dir))
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        hook_fn(data, harness)
    return json.loads(buf.getvalue())


def _write_wrap_up_marker(state_dir: Path, session_id: str, outcome: str = "success"):
    """Write a wrap_up_session last-finalized marker for stop-hook proof-of-done tests."""
    state_dir.mkdir(parents=True, exist_ok=True)
    marker = state_dir / f"last_finalized_{session_id}.json"
    marker.write_text(
        json.dumps(
            {
                "intent_type": "wrap_up_session",
                "execution_entity": f"wrap_up_{session_id}",
                "outcome": outcome,
                "agent": "ga_agent",
                "ts": "2026-04-20T09:00:00",
            }
        ),
        encoding="utf-8",
    )


def test_stop_hook_passthrough_when_active(tmp_path):
    with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
        result = _capture_hook_output(
            hook_stop,
            {"session_id": "test", "stop_hook_active": True, "transcript_path": ""},
            state_dir=tmp_path,
        )
    assert result == {}


def test_stop_hook_passthrough_when_active_string(tmp_path):
    with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
        result = _capture_hook_output(
            hook_stop,
            {"session_id": "test", "stop_hook_active": "true", "transcript_path": ""},
            state_dir=tmp_path,
        )
    assert result == {}


def test_stop_hook_passthrough_below_interval(tmp_path):
    """With wrap-up proof present, below-interval exchanges still pass through."""
    _write_wrap_up_marker(tmp_path, "test")
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [{"message": {"role": "user", "content": f"msg {i}"}} for i in range(SAVE_INTERVAL - 1)],
    )
    result = _capture_hook_output(
        hook_stop,
        {"session_id": "test", "stop_hook_active": False, "transcript_path": str(transcript)},
        state_dir=tmp_path,
    )
    assert result == {}


def test_stop_hook_blocks_at_interval(tmp_path):
    """With wrap-up proof, the legacy save-interval block still fires at threshold."""
    _write_wrap_up_marker(tmp_path, "test")
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [{"message": {"role": "user", "content": f"msg {i}"}} for i in range(SAVE_INTERVAL)],
    )
    result = _capture_hook_output(
        hook_stop,
        {"session_id": "test", "stop_hook_active": False, "transcript_path": str(transcript)},
        state_dir=tmp_path,
    )
    assert result["decision"] == "block"
    assert result["reason"] == STOP_BLOCK_REASON


def test_stop_hook_tracks_save_point(tmp_path):
    _write_wrap_up_marker(tmp_path, "test")
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [{"message": {"role": "user", "content": f"msg {i}"}} for i in range(SAVE_INTERVAL)],
    )
    data = {"session_id": "test", "stop_hook_active": False, "transcript_path": str(transcript)}

    # First call blocks
    result = _capture_hook_output(hook_stop, data, state_dir=tmp_path)
    assert result["decision"] == "block"

    # Second call ALSO blocks — counter is NOT updated by stop hook.
    # Only diary_write updates the counter (dodge prevention).
    result = _capture_hook_output(hook_stop, data, state_dir=tmp_path)
    assert result["decision"] == "block"

    # Simulate diary_write updating counter via pending_save marker
    pending_file = tmp_path / "test_pending_save"
    assert pending_file.is_file(), "stop hook should write _pending_save marker"
    last_save_file = tmp_path / "test_last_save"
    last_save_file.write_text(pending_file.read_text(encoding="utf-8"), encoding="utf-8")
    pending_file.unlink()

    # NOW third call passes through (counter updated by diary_write)
    result = _capture_hook_output(hook_stop, data, state_dir=tmp_path)
    assert result == {}


# --- NEVER-STOP rule: non-iterative detection + wrap-up proof ---


def test_is_non_iterative_mode_default_false(monkeypatch):
    """Neither PAPERCLIP_RUN_ID nor AGENT_HOME set: iterative mode."""
    monkeypatch.delenv("PAPERCLIP_RUN_ID", raising=False)
    monkeypatch.delenv("AGENT_HOME", raising=False)
    assert _is_non_iterative_mode() is False


def test_is_non_iterative_mode_paperclip(monkeypatch):
    monkeypatch.setenv("PAPERCLIP_RUN_ID", "run-abc-123")
    monkeypatch.delenv("AGENT_HOME", raising=False)
    assert _is_non_iterative_mode() is True


def test_is_non_iterative_mode_agent_home(monkeypatch):
    monkeypatch.delenv("PAPERCLIP_RUN_ID", raising=False)
    monkeypatch.setenv("AGENT_HOME", "/path/to/agent")
    assert _is_non_iterative_mode() is True


def test_is_non_iterative_mode_empty_string_counts_as_unset(monkeypatch):
    """Empty/whitespace-only env vars should not trigger non-iterative mode."""
    monkeypatch.setenv("PAPERCLIP_RUN_ID", "")
    monkeypatch.setenv("AGENT_HOME", "   ")
    assert _is_non_iterative_mode() is False


def test_read_last_finalized_empty_sid(tmp_path):
    with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
        assert _read_last_finalized_intent("") == {}


def test_read_last_finalized_no_file(tmp_path):
    with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
        assert _read_last_finalized_intent("nonexistent_sid") == {}


def test_read_last_finalized_valid(tmp_path):
    _write_wrap_up_marker(tmp_path, "sess1", outcome="success")
    with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
        result = _read_last_finalized_intent("sess1")
    assert result["intent_type"] == "wrap_up_session"
    assert result["outcome"] == "success"


def test_read_last_finalized_corrupt_file(tmp_path):
    (tmp_path / "last_finalized_sess1.json").write_text("{not json", encoding="utf-8")
    with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
        assert _read_last_finalized_intent("sess1") == {}


def test_stop_hook_blocks_when_no_wrap_up_proof(tmp_path):
    """No last_finalized marker: never-stop rule blocks with NEVER_STOP_BLOCK_REASON."""
    transcript = tmp_path / "t.jsonl"
    _write_transcript(transcript, [{"message": {"role": "user", "content": "hi"}}])
    result = _capture_hook_output(
        hook_stop,
        {"session_id": "sess1", "stop_hook_active": False, "transcript_path": str(transcript)},
        state_dir=tmp_path,
    )
    assert result["decision"] == "block"
    assert result["reason"] == NEVER_STOP_BLOCK_REASON


def test_stop_hook_blocks_when_last_finalized_is_not_wrap_up(tmp_path):
    """Last finalized is something other than wrap_up_session: block."""
    marker = tmp_path / "last_finalized_sess1.json"
    marker.write_text(
        json.dumps({"intent_type": "ship_mempalace_feature", "outcome": "success"}),
        encoding="utf-8",
    )
    transcript = tmp_path / "t.jsonl"
    _write_transcript(transcript, [{"message": {"role": "user", "content": "hi"}}])
    result = _capture_hook_output(
        hook_stop,
        {"session_id": "sess1", "stop_hook_active": False, "transcript_path": str(transcript)},
        state_dir=tmp_path,
    )
    assert result["decision"] == "block"
    assert result["reason"] == NEVER_STOP_BLOCK_REASON


def test_stop_hook_blocks_when_wrap_up_outcome_failed(tmp_path):
    """wrap_up_session finalized but outcome != success: still block."""
    _write_wrap_up_marker(tmp_path, "sess1", outcome="failed")
    transcript = tmp_path / "t.jsonl"
    _write_transcript(transcript, [{"message": {"role": "user", "content": "hi"}}])
    result = _capture_hook_output(
        hook_stop,
        {"session_id": "sess1", "stop_hook_active": False, "transcript_path": str(transcript)},
        state_dir=tmp_path,
    )
    assert result["decision"] == "block"
    assert result["reason"] == NEVER_STOP_BLOCK_REASON


def test_stop_hook_passes_through_with_wrap_up_proof(tmp_path):
    """wrap_up_session success as last finalized: pass through."""
    _write_wrap_up_marker(tmp_path, "sess1")
    transcript = tmp_path / "t.jsonl"
    _write_transcript(transcript, [{"message": {"role": "user", "content": "one"}}])
    result = _capture_hook_output(
        hook_stop,
        {"session_id": "sess1", "stop_hook_active": False, "transcript_path": str(transcript)},
        state_dir=tmp_path,
    )
    assert result == {}


def test_stop_hook_non_iterative_mode_passes_through(tmp_path, monkeypatch):
    """PAPERCLIP_RUN_ID set: non-iterative, pass through even without wrap-up proof."""
    monkeypatch.setenv("PAPERCLIP_RUN_ID", "run-xyz")
    transcript = tmp_path / "t.jsonl"
    _write_transcript(transcript, [{"message": {"role": "user", "content": "hi"}}])
    result = _capture_hook_output(
        hook_stop,
        {"session_id": "sess1", "stop_hook_active": False, "transcript_path": str(transcript)},
        state_dir=tmp_path,
    )
    assert result == {}


def test_stop_hook_never_stop_precedes_save_interval(tmp_path):
    """Without wrap-up proof, NEVER_STOP fires with its own distinct reason,
    not the save-interval STOP_BLOCK_REASON."""
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [{"message": {"role": "user", "content": f"msg {i}"}} for i in range(SAVE_INTERVAL + 5)],
    )
    result = _capture_hook_output(
        hook_stop,
        {"session_id": "sess1", "stop_hook_active": False, "transcript_path": str(transcript)},
        state_dir=tmp_path,
    )
    assert result["decision"] == "block"
    assert result["reason"] == NEVER_STOP_BLOCK_REASON


# --- hook_session_start ---


def _write_active_intent(state_dir: Path, session_id: str, intent: dict):
    """Write a synthetic active_intent_{sid}.json for tests."""
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / f"active_intent_{session_id}.json"
    path.write_text(json.dumps(intent), encoding="utf-8")


def _write_trace(state_dir: Path, session_id: str, entries: list):
    """Write synthetic execution_trace_{sid}.jsonl for tests."""
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / f"execution_trace_{session_id}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _sample_intent():
    return {
        "intent_id": "intent_test_abc123",
        "intent_type": "ship_mempalace_feature",
        "description": "Implement Option A Phase 1 rehydration hook",
        "agent": "ga_agent",
        "slots": {"paths": ["D:/Flowsev/mempalace/**"], "commands": ["pytest"]},
        "budget": {"Read": 10, "Edit": 5, "Write": 2},
        "used": {"Read": 3, "Edit": 1},
        "injected_memory_ids": ["mem_a", "mem_b", "mem_c"],
        "accessed_memory_ids": ["mem_a", "mem_d"],
    }


def test_session_start_startup_passes_through(tmp_path):
    """source=startup or unspecified: pass-through, no rehydration."""
    result = _capture_hook_output(
        hook_session_start,
        {"session_id": "test", "source": "startup"},
        state_dir=tmp_path,
    )
    assert result == {}


def test_session_start_no_source_passes_through(tmp_path):
    """Missing source: pass-through."""
    result = _capture_hook_output(
        hook_session_start,
        {"session_id": "test"},
        state_dir=tmp_path,
    )
    assert result == {}


def test_session_start_compact_without_active_intent_passes_through(tmp_path):
    """compact source but no active intent: pass-through (nothing to rehydrate)."""
    result = _capture_hook_output(
        hook_session_start,
        {"session_id": "test", "source": "compact"},
        state_dir=tmp_path,
    )
    assert result == {}


def test_session_start_compact_with_active_intent_rehydrates(tmp_path):
    """compact + active intent: emit hookSpecificOutput.additionalContext."""
    _write_active_intent(tmp_path, "sess1", _sample_intent())
    result = _capture_hook_output(
        hook_session_start,
        {"session_id": "sess1", "source": "compact"},
        state_dir=tmp_path,
    )
    assert "hookSpecificOutput" in result
    hso = result["hookSpecificOutput"]
    assert hso["hookEventName"] == "SessionStart"
    ctx = hso["additionalContext"]
    assert "MemPalace rehydration" in ctx
    assert "ship_mempalace_feature" in ctx
    assert "Option A Phase 1" in ctx
    assert "mem_a" in ctx  # injected memory id
    assert "mem_d" in ctx  # accessed-only memory id
    assert "Read" in ctx  # budget line


def test_session_start_resume_with_active_intent_rehydrates(tmp_path):
    """resume source also triggers rehydration (same path as compact)."""
    _write_active_intent(tmp_path, "sess1", _sample_intent())
    result = _capture_hook_output(
        hook_session_start,
        {"session_id": "sess1", "source": "resume"},
        state_dir=tmp_path,
    )
    assert "hookSpecificOutput" in result
    assert "resume" in result["hookSpecificOutput"]["additionalContext"]


def test_session_start_clear_passes_through_even_with_intent(tmp_path):
    """source=clear: no rehydration — user explicitly cleared context."""
    _write_active_intent(tmp_path, "sess1", _sample_intent())
    result = _capture_hook_output(
        hook_session_start,
        {"session_id": "sess1", "source": "clear"},
        state_dir=tmp_path,
    )
    assert result == {}


def test_session_start_empty_session_id_passes_through(tmp_path):
    """Empty sid: skip silently — can't locate intent state."""
    result = _capture_hook_output(
        hook_session_start,
        {"session_id": "", "source": "compact"},
        state_dir=tmp_path,
    )
    assert result == {}


def test_session_start_fail_open_on_corrupt_intent(tmp_path):
    """Corrupt active_intent JSON must not block session entry."""
    (tmp_path / "active_intent_sess1.json").write_text("{not json", encoding="utf-8")
    result = _capture_hook_output(
        hook_session_start,
        {"session_id": "sess1", "source": "compact"},
        state_dir=tmp_path,
    )
    # _read_active_intent returns None on parse error; pass-through
    assert result == {}


def test_session_start_surfaces_error_on_unexpected_exception(tmp_path):
    """Post-silent-fail audit: exception must NOT be swallowed.
    Hook records + surfaces a visible MEMPALACE HOOK ERROR notice
    via additionalContext. Session entry is still non-blocking."""
    _write_active_intent(tmp_path, "sess1", _sample_intent())
    with patch(
        "mempalace.hooks_cli._build_rehydration_payload",
        side_effect=RuntimeError("boom"),
    ):
        result = _capture_hook_output(
            hook_session_start,
            {"session_id": "sess1", "source": "compact"},
            state_dir=tmp_path,
        )
    hso = result.get("hookSpecificOutput", {})
    assert hso, f"Expected visible error notice, got silent pass: {result}"
    body = hso.get("additionalContext", "")
    assert "MEMPALACE HOOK ERROR" in body
    assert "hook_session_start" in body


# --- _build_rehydration_payload ---


def test_rehydration_payload_minimal_intent():
    intent = {
        "intent_id": "x",
        "intent_type": "research",
        "description": "",
        "slots": {},
        "budget": {},
        "used": {},
        "injected_memory_ids": [],
        "accessed_memory_ids": [],
    }
    body = _build_rehydration_payload(intent, "", "compact")
    assert "MemPalace rehydration" in body
    assert "research" in body
    assert len(body) <= REHYDRATION_MAX_CHARS


def test_rehydration_payload_caps_memory_list():
    """More than REHYDRATION_MEMORY_CAP memories: list truncates with summary line."""
    intent = _sample_intent()
    intent["injected_memory_ids"] = [f"mem_{i}" for i in range(30)]
    intent["accessed_memory_ids"] = []
    body = _build_rehydration_payload(intent, "", "compact")
    assert "more)" in body  # the "... (N more)" summary line
    # Should not include all 30 raw ids — only the cap
    assert body.count("`mem_") <= 16  # 15 cap + some margin for other backticks


def test_rehydration_payload_hard_cap():
    """Payload capped at REHYDRATION_MAX_CHARS regardless of input size."""
    intent = _sample_intent()
    intent["description"] = "x" * 20000  # way oversized
    body = _build_rehydration_payload(intent, "", "compact")
    assert len(body) <= REHYDRATION_MAX_CHARS


def test_rehydration_payload_budget_remaining():
    """Remaining budget is correctly computed as limit - used."""
    intent = _sample_intent()  # Read: 10 budget, 3 used -> 7 remaining
    body = _build_rehydration_payload(intent, "", "compact")
    assert "Read: 7/10" in body


def test_rehydration_payload_dedupes_memory_ids():
    """Memory ids appearing in both injected and accessed are listed once."""
    intent = _sample_intent()
    # mem_a is in both injected and accessed; should appear once
    body = _build_rehydration_payload(intent, "", "compact")
    assert body.count("`mem_a`") == 1


# --- Option A Phase 2a: cue-builder helpers ---


def test_basename_forward_slash():
    assert _basename("src/auth.py") == "auth.py"


def test_basename_backslash():
    assert _basename("D:\\Flowsev\\mempalace\\intent.py") == "intent.py"


def test_basename_no_separator():
    assert _basename("auth.py") == "auth.py"


def test_basename_empty():
    assert _basename("") == ""


def test_parent_path_tokens_basic():
    assert _parent_path_tokens("src/auth/login.py") == ["src", "auth"]


def test_parent_path_tokens_depth():
    # Limits to trailing `depth` dirs
    assert _parent_path_tokens("a/b/c/d/e/f.py", depth=2) == ["d", "e"]


def test_parent_path_tokens_excludes_drive_letter():
    # `D:` contains a colon and should be filtered out
    tokens = _parent_path_tokens("D:/Flowsev/mempalace/intent.py")
    assert "D:" not in tokens
    assert tokens == ["flowsev", "mempalace"]


def test_parent_path_tokens_no_parents():
    # A bare filename has no parent dirs
    assert _parent_path_tokens("auth.py") == []


def test_parent_path_tokens_empty():
    assert _parent_path_tokens("") == []


# --- _namespaced_args ---


def test_namespaced_args_edit_file():
    kw = _namespaced_args(
        "Edit", {"file_path": "src/auth.py", "old_string": "x", "new_string": "y"}
    )
    assert "tool:Edit" in kw
    assert "file:auth.py" in kw
    assert "path:src" in kw


def test_namespaced_args_bash_command():
    kw = _namespaced_args("Bash", {"command": "pytest tests/test_auth.py -xvs"})
    assert "tool:Bash" in kw
    assert "command:pytest" in kw
    assert "flag:-x" in kw or "flag:-xvs" in kw  # depends on bashlex behaviour


def test_namespaced_args_bash_compound():
    """Compound command — should surface multiple leaf commands."""
    kw = _namespaced_args("Bash", {"command": "cd /tmp && pytest -q"})
    # bashlex decomposes; either cd or pytest (or both) should show
    commands = [t for t in kw if t.startswith("command:")]
    assert len(commands) >= 1
    assert any("pytest" in c for c in commands)


def test_namespaced_args_grep_pattern():
    kw = _namespaced_args("Grep", {"pattern": "accessed_memory_ids", "glob": "*.py"})
    assert "tool:Grep" in kw
    assert "pattern:accessed_memory_ids" in kw
    assert "glob:*.py" in kw


def test_namespaced_args_glob_pattern():
    kw = _namespaced_args("Glob", {"pattern": "**/*.py"})
    assert "tool:Glob" in kw
    assert "pattern:**/*.py" in kw


def test_namespaced_args_read_with_offset():
    kw = _namespaced_args("Read", {"file_path": "intent.py", "offset": 100, "limit": 50})
    assert "tool:Read" in kw
    assert "file:intent.py" in kw


def test_namespaced_args_unknown_tool():
    """Unknown tool: just emits tool:<name> token, no crash."""
    kw = _namespaced_args("SomeWeirdTool", {"foo": "bar"})
    assert kw == ["tool:SomeWeirdTool"]


def test_namespaced_args_dedupes():
    """Duplicate parents or commands only appear once."""
    kw = _namespaced_args("Edit", {"file_path": "src/src/file.py"})
    assert kw.count("path:src") == 1


def test_namespaced_args_caps_size():
    """Long command with many flags caps token count."""
    cmd = "pytest " + " ".join(f"--flag-{i}" for i in range(30))
    kw = _namespaced_args("Bash", {"command": cmd})
    assert len(kw) <= 10


def test_namespaced_args_invalid_input():
    """Non-dict tool_input shouldn't crash."""
    assert _namespaced_args("Edit", None) == ["tool:Edit"]
    assert _namespaced_args("Edit", "not a dict") == ["tool:Edit"]


# --- _read_last_assistant_message ---


def _write_jsonl(path: Path, entries: list):
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def test_read_last_assistant_message_basic(tmp_path):
    t = tmp_path / "transcript.jsonl"
    _write_jsonl(
        t,
        [
            {"message": {"role": "user", "content": "do X"}},
            {"message": {"role": "assistant", "content": "will edit auth.py"}},
        ],
    )
    assert _read_last_assistant_message(str(t)) == "will edit auth.py"


def test_read_last_assistant_message_picks_most_recent(tmp_path):
    t = tmp_path / "transcript.jsonl"
    _write_jsonl(
        t,
        [
            {"message": {"role": "assistant", "content": "older reply"}},
            {"message": {"role": "user", "content": "thanks"}},
            {"message": {"role": "assistant", "content": "newer reply"}},
        ],
    )
    assert _read_last_assistant_message(str(t)) == "newer reply"


def test_read_last_assistant_message_handles_list_content(tmp_path):
    t = tmp_path / "transcript.jsonl"
    _write_jsonl(
        t,
        [
            {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "mixed"},
                        {"type": "text", "text": "blocks"},
                    ],
                }
            }
        ],
    )
    assert _read_last_assistant_message(str(t)) == "mixed blocks"


def test_read_last_assistant_message_codex_format(tmp_path):
    t = tmp_path / "transcript.jsonl"
    _write_jsonl(
        t,
        [
            {"type": "event_msg", "payload": {"type": "user_message", "message": "hi"}},
            {"type": "event_msg", "payload": {"type": "assistant_message", "message": "hello"}},
        ],
    )
    assert _read_last_assistant_message(str(t)) == "hello"


def test_read_last_assistant_message_missing_file():
    assert _read_last_assistant_message("/nonexistent/path.jsonl") == ""


def test_read_last_assistant_message_empty_path():
    assert _read_last_assistant_message("") == ""


def test_read_last_assistant_message_no_assistant(tmp_path):
    t = tmp_path / "transcript.jsonl"
    _write_jsonl(t, [{"message": {"role": "user", "content": "only user"}}])
    assert _read_last_assistant_message(str(t)) == ""


def test_read_last_assistant_message_truncates(tmp_path):
    t = tmp_path / "transcript.jsonl"
    long_text = "x" * (LOCAL_CUE_ASSISTANT_MAX_CHARS + 200)
    _write_jsonl(t, [{"message": {"role": "assistant", "content": long_text}}])
    result = _read_last_assistant_message(str(t))
    assert len(result) == LOCAL_CUE_ASSISTANT_MAX_CHARS


def test_read_last_assistant_message_skips_malformed(tmp_path):
    t = tmp_path / "transcript.jsonl"
    t.write_text(
        'not json\n{"message": {"role": "assistant", "content": "ok"}}\n',
        encoding="utf-8",
    )
    assert _read_last_assistant_message(str(t)) == "ok"


# --- _summarize_tool_args ---


def test_summarize_edit():
    s = _summarize_tool_args(
        "Edit",
        {"file_path": "src/auth.py", "old_string": "def login", "new_string": "async def login"},
    )
    assert "Edit" in s
    assert "auth.py" in s
    assert "->" in s


def test_summarize_bash():
    s = _summarize_tool_args("Bash", {"command": "pytest tests/test_auth.py -xvs"})
    assert s.startswith("Bash")
    assert "pytest" in s


def test_summarize_grep_with_glob():
    s = _summarize_tool_args("Grep", {"pattern": "accessed_memory_ids", "glob": "*.py"})
    assert "accessed_memory_ids" in s
    assert "*.py" in s


def test_summarize_read_with_offset():
    s = _summarize_tool_args("Read", {"file_path": "intent.py", "offset": 1400, "limit": 100})
    assert "intent.py" in s
    assert "1400" in s


def test_summarize_caps_length():
    """Excessively long args are clipped."""
    s = _summarize_tool_args("Bash", {"command": "x" * 1000})
    assert len(s) <= 120 + len("Bash ")  # summary cap


def test_summarize_invalid_input():
    """Non-dict tool_input returns just the tool name."""
    assert _summarize_tool_args("Edit", None) == "Edit"


# --- _build_local_cue ---


def test_build_local_cue_basic(tmp_path):
    """Full integration: queries + keywords assembled end-to-end."""
    t = tmp_path / "transcript.jsonl"
    _write_jsonl(
        t,
        [
            {"message": {"role": "user", "content": "refactor login"}},
            {
                "message": {
                    "role": "assistant",
                    "content": "I will edit auth.py to make login async",
                }
            },
        ],
    )
    intent = {"_context_views": ["Implementing option A retrieval for auth rate limiter"]}
    cue = _build_local_cue(
        "Edit",
        {"file_path": "src/auth.py", "old_string": "def login", "new_string": "async def login"},
        str(t),
        intent,
    )
    assert "queries" in cue and "keywords" in cue
    assert len(cue["queries"]) == 3  # tool summary + assistant msg + activity view
    assert "Edit" in cue["queries"][0]
    assert "login async" in cue["queries"][1]
    assert "rate limiter" in cue["queries"][2]
    assert "tool:Edit" in cue["keywords"]
    assert "file:auth.py" in cue["keywords"]


def test_build_local_cue_no_transcript():
    """Missing transcript: queries[1] omitted, no crash."""
    cue = _build_local_cue("Read", {"file_path": "x.py"}, "", {})
    assert cue["queries"][0].startswith("Read")
    assert len(cue["queries"]) == 1  # only the tool summary


def test_build_local_cue_no_intent_context():
    """Missing active intent context views: queries[2] omitted."""
    cue = _build_local_cue("Bash", {"command": "pytest"}, "", None)
    # Only queries[0] (tool summary) — no assistant, no intent
    assert len(cue["queries"]) == 1


def test_build_local_cue_empty_context_views(tmp_path):
    """Intent with empty _context_views list: skipped, no crash."""
    t = tmp_path / "transcript.jsonl"
    _write_jsonl(t, [{"message": {"role": "assistant", "content": "reply"}}])
    cue = _build_local_cue("Read", {"file_path": "x.py"}, str(t), {"_context_views": []})
    # tool summary + assistant msg, no activity view
    assert len(cue["queries"]) == 2


# --- _read_recent_trace ---


def test_read_recent_trace_empty_sid(tmp_path):
    with patch("mempalace.hooks_cli._TRACE_DIR", tmp_path):
        assert _read_recent_trace("") == []


def test_read_recent_trace_no_file(tmp_path):
    with patch("mempalace.hooks_cli._TRACE_DIR", tmp_path):
        assert _read_recent_trace("nonexistent_sid") == []


def test_read_recent_trace_returns_last_n(tmp_path):
    entries = [
        {"ts": f"2026-04-20T{h:02d}:00:00", "tool": "Read", "target": f"f{h}.py"} for h in range(10)
    ]
    _write_trace(tmp_path, "sess1", entries)
    with patch("mempalace.hooks_cli._TRACE_DIR", tmp_path):
        result = _read_recent_trace("sess1", limit=3)
    assert len(result) == 3
    # Newest last — the limit returns the tail
    assert result[-1]["target"] == "f9.py"


def test_read_recent_trace_skips_malformed_lines(tmp_path):
    path = tmp_path / "execution_trace_sess1.jsonl"
    path.write_text(
        '{"ts": "t1", "tool": "Read", "target": "a"}\n'
        "garbage not json\n"
        '{"ts": "t2", "tool": "Edit", "target": "b"}\n',
        encoding="utf-8",
    )
    with patch("mempalace.hooks_cli._TRACE_DIR", tmp_path):
        result = _read_recent_trace("sess1", limit=10)
    assert len(result) == 2  # Malformed line skipped
    assert [e["target"] for e in result] == ["a", "b"]


# --- hook_precompact ---


def test_precompact_warns_without_blocking(tmp_path):
    """Precompact must NEVER block compaction — blocking risks losing the session
    when context fills up. It surfaces the save-everything instruction via
    systemMessage and lets compaction proceed.
    """
    result = _capture_hook_output(
        hook_precompact,
        {"session_id": "test"},
        state_dir=tmp_path,
    )
    assert "decision" not in result, "precompact must not set a decision (no blocking)"
    assert result.get("systemMessage") == PRECOMPACT_WARNING_MESSAGE


# --- _log ---


def test_log_writes_to_hook_log(tmp_path):
    with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
        _log("test message")
    log_path = tmp_path / "hook.log"
    assert log_path.is_file()
    content = log_path.read_text()
    assert "test message" in content


def test_log_oserror_is_silenced(tmp_path):
    """_log should not raise if the directory cannot be created."""
    with patch("mempalace.hooks_cli.STATE_DIR", Path("/nonexistent/deeply/nested/dir")):
        # Should not raise
        _log("this will fail silently")


# --- _maybe_auto_ingest ---


def test_maybe_auto_ingest_no_env(tmp_path):
    """Without MEMPAL_DIR set, does nothing."""
    with patch.dict("os.environ", {}, clear=True):
        with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
            _maybe_auto_ingest()  # should not raise


def test_maybe_auto_ingest_with_env(tmp_path):
    """With MEMPAL_DIR set to a valid directory, spawns subprocess."""
    mempal_dir = tmp_path / "project"
    mempal_dir.mkdir()
    with patch.dict("os.environ", {"MEMPAL_DIR": str(mempal_dir)}):
        with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
            with patch("mempalace.hooks_cli.subprocess.Popen") as mock_popen:
                _maybe_auto_ingest()
                mock_popen.assert_called_once()


def test_maybe_auto_ingest_oserror(tmp_path):
    """OSError during subprocess spawn is silenced."""
    mempal_dir = tmp_path / "project"
    mempal_dir.mkdir()
    with patch.dict("os.environ", {"MEMPAL_DIR": str(mempal_dir)}):
        with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
            with patch("mempalace.hooks_cli.subprocess.Popen", side_effect=OSError("fail")):
                _maybe_auto_ingest()  # should not raise


# --- _parse_harness_input ---


def test_parse_harness_input_unknown():
    """Unknown harness should sys.exit(1)."""
    with pytest.raises(SystemExit) as exc_info:
        _parse_harness_input({"session_id": "test"}, "unknown-harness")
    assert exc_info.value.code == 1


def test_parse_harness_input_valid():
    result = _parse_harness_input(
        {"session_id": "abc-123", "stop_hook_active": True, "transcript_path": "/tmp/t.jsonl"},
        "claude-code",
    )
    assert result["session_id"] == "abc-123"
    assert result["stop_hook_active"] is True


# --- hook_stop with OSError on write ---


def test_stop_hook_oserror_on_last_save_read(tmp_path):
    """When last_save_file has invalid content, falls back to 0."""
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [{"message": {"role": "user", "content": f"msg {i}"}} for i in range(SAVE_INTERVAL)],
    )
    # Write invalid content to last save file
    (tmp_path / "test_last_save").write_text("not_a_number")
    result = _capture_hook_output(
        hook_stop,
        {"session_id": "test", "stop_hook_active": False, "transcript_path": str(transcript)},
        state_dir=tmp_path,
    )
    assert result["decision"] == "block"


def test_stop_hook_oserror_on_write(tmp_path):
    """When write to last_save_file fails, hook still outputs correctly."""
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [{"message": {"role": "user", "content": f"msg {i}"}} for i in range(SAVE_INTERVAL)],
    )

    def bad_write_text(*args, **kwargs):
        raise OSError("disk full")

    with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
        with patch.object(Path, "write_text", bad_write_text):
            result = _capture_hook_output(
                hook_stop,
                {
                    "session_id": "test",
                    "stop_hook_active": False,
                    "transcript_path": str(transcript),
                },
                state_dir=tmp_path,
            )
    assert result["decision"] == "block"


# --- hook_precompact with MEMPAL_DIR ---


def test_precompact_with_mempal_dir(tmp_path):
    """Precompact runs subprocess.run when MEMPAL_DIR is set."""
    mempal_dir = tmp_path / "project"
    mempal_dir.mkdir()
    with patch.dict("os.environ", {"MEMPAL_DIR": str(mempal_dir)}):
        with patch("mempalace.hooks_cli.subprocess.run") as mock_run:
            result = _capture_hook_output(
                hook_precompact,
                {"session_id": "test"},
                state_dir=tmp_path,
            )
    assert "decision" not in result
    assert result.get("systemMessage") == PRECOMPACT_WARNING_MESSAGE
    mock_run.assert_called_once()


def test_precompact_with_mempal_dir_oserror(tmp_path):
    """Precompact handles OSError from subprocess gracefully."""
    mempal_dir = tmp_path / "project"
    mempal_dir.mkdir()
    with patch.dict("os.environ", {"MEMPAL_DIR": str(mempal_dir)}):
        with patch("mempalace.hooks_cli.subprocess.run", side_effect=OSError("fail")):
            result = _capture_hook_output(
                hook_precompact,
                {"session_id": "test"},
                state_dir=tmp_path,
            )
    assert "decision" not in result
    assert result.get("systemMessage") == PRECOMPACT_WARNING_MESSAGE


# --- run_hook ---


def test_run_hook_dispatches_session_start(tmp_path):
    """run_hook reads stdin JSON and dispatches to correct handler."""
    stdin_data = json.dumps({"session_id": "run-test"})
    with patch("sys.stdin", io.StringIO(stdin_data)):
        with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
            with patch("mempalace.hooks_cli._output") as mock_output:
                run_hook("session-start", "claude-code")
    mock_output.assert_called_once_with({})


def test_run_hook_dispatches_stop(tmp_path):
    _write_wrap_up_marker(tmp_path, "run-test")  # satisfy never-stop rule
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript, [{"message": {"role": "user", "content": f"msg {i}"}} for i in range(3)]
    )
    stdin_data = json.dumps(
        {
            "session_id": "run-test",
            "stop_hook_active": False,
            "transcript_path": str(transcript),
        }
    )
    with patch("sys.stdin", io.StringIO(stdin_data)):
        with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
            with patch("mempalace.hooks_cli._output") as mock_output:
                run_hook("stop", "claude-code")
    mock_output.assert_called_once_with({})


def test_run_hook_dispatches_precompact(tmp_path):
    stdin_data = json.dumps({"session_id": "run-test"})
    with patch("sys.stdin", io.StringIO(stdin_data)):
        with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
            with patch("mempalace.hooks_cli._output") as mock_output:
                run_hook("precompact", "claude-code")
    mock_output.assert_called_once()
    call_args = mock_output.call_args[0][0]
    assert "decision" not in call_args
    assert call_args.get("systemMessage") == PRECOMPACT_WARNING_MESSAGE


def test_run_hook_unknown_hook():
    stdin_data = json.dumps({"session_id": "test"})
    with patch("sys.stdin", io.StringIO(stdin_data)):
        with pytest.raises(SystemExit) as exc_info:
            run_hook("nonexistent", "claude-code")
        assert exc_info.value.code == 1


def test_run_hook_invalid_json(tmp_path):
    """Invalid stdin JSON should not crash — falls back to empty dict."""
    with patch("sys.stdin", io.StringIO("not valid json")):
        with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
            with patch("mempalace.hooks_cli._output") as mock_output:
                run_hook("session-start", "claude-code")
    mock_output.assert_called_once_with({})


# --- _check_permission scope matching ---


def test_check_permission_scope_matches_bare_parent():
    """Scope 'path/**' must match the bare parent directory too.

    fnmatch alone rejects 'd:/foo' against 'd:/foo/**' because the pattern's
    trailing '/' requires a '/' in the target. The hook strips a trailing
    /* or /** from the normalized scope and accepts the bare parent.
    """
    intent = {
        "intent_type": "test_type",
        "effective_permissions": [
            {"tool": "Grep", "scope": "D:/Flowsev/mempalace/**"},
        ],
    }
    ok, _ = _check_permission("Grep", {"path": "D:/Flowsev/mempalace"}, intent)
    assert ok, "bare parent of /** scope must be permitted"
    ok, _ = _check_permission("Grep", {"path": "D:/Flowsev/mempalace/sub/f.py"}, intent)
    assert ok, "descendants must still match"
    ok, _ = _check_permission("Grep", {"path": "C:/other"}, intent)
    assert not ok, "unrelated paths must still be denied"


# --- Phase 2b: PreToolUse local retrieval -----------------------------


def test_local_retrieval_enabled_default_true(monkeypatch):
    """On by default: no env var means retrieval fires."""
    monkeypatch.delenv("MEMPALACE_DISABLE_LOCAL_RETRIEVAL", raising=False)
    assert _is_local_retrieval_enabled() is True


def test_local_retrieval_disabled_via_opt_out_env(monkeypatch):
    """MEMPALACE_DISABLE_LOCAL_RETRIEVAL=1 (or truthy) opts out \u2014 break-glass."""
    for val in ("1", "true", "yes", "on", "TRUE", "Yes"):
        monkeypatch.setenv("MEMPALACE_DISABLE_LOCAL_RETRIEVAL", val)
        assert _is_local_retrieval_enabled() is False, f"{val!r} should disable"
    for val in ("0", "false", "no", "off", "", "  "):
        monkeypatch.setenv("MEMPALACE_DISABLE_LOCAL_RETRIEVAL", val)
        assert _is_local_retrieval_enabled() is True, f"{val!r} should leave enabled"


def test_format_retrieval_additional_context_empty():
    assert _format_retrieval_additional_context([]) == ""


def test_format_retrieval_additional_context_renders_entries():
    body = _format_retrieval_additional_context(
        [
            {"id": "mem_alpha", "preview": "pytest -x flag stops on first failure", "score": 0.9},
            {
                "id": "mem_beta",
                "preview": "rate-limit helper in src/auth.py has a quirk",
                "score": 0.8,
            },
        ]
    )
    assert "mem_alpha" in body
    assert "mem_beta" in body
    assert "Local retrieval" in body
    assert len(body) <= LOCAL_RETRIEVAL_MAX_CHARS


def test_format_retrieval_additional_context_caps_at_limit():
    """Many memories: trailing entries collapse into a '+ N more' line."""
    memories = [
        {"id": f"mem_{i:02d}", "preview": "x" * 300, "score": 1.0 - i * 0.01} for i in range(20)
    ]
    body = _format_retrieval_additional_context(memories)
    assert len(body) <= LOCAL_RETRIEVAL_MAX_CHARS
    assert "more omitted" in body


def test_run_local_retrieval_empty_queries_returns_empty():
    assert _run_local_retrieval({"queries": [], "keywords": []}, set(), top_k=3) == ([], None)
    assert _run_local_retrieval({"queries": None}, set(), top_k=3) == ([], None)


def test_maybe_build_local_retrieval_context_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMPALACE_DISABLE_LOCAL_RETRIEVAL", "1")
    intent = {"intent_id": "i1", "accessed_memory_ids": []}
    assert (
        _maybe_build_local_retrieval_context("Edit", {"file_path": "x.py"}, intent, "sess1", "")
        == ""
    )


def test_maybe_build_local_retrieval_context_skips_always_allowed(monkeypatch):
    monkeypatch.delenv("MEMPALACE_DISABLE_LOCAL_RETRIEVAL", raising=False)
    intent = {"intent_id": "i1", "accessed_memory_ids": []}
    for tool in ("TodoWrite", "Agent", "AskUserQuestion"):
        assert _maybe_build_local_retrieval_context(tool, {}, intent, "sess1", "") == "", (
            f"{tool} must be skipped"
        )


def test_maybe_build_local_retrieval_context_skips_mempalace_mcp(monkeypatch):
    monkeypatch.delenv("MEMPALACE_DISABLE_LOCAL_RETRIEVAL", raising=False)
    intent = {"intent_id": "i1", "accessed_memory_ids": []}
    assert (
        _maybe_build_local_retrieval_context(
            "mcp__plugin_3_0_14_mempalace__mempalace_kg_search",
            {},
            intent,
            "sess1",
            "",
        )
        == ""
    )


def test_maybe_build_local_retrieval_context_skips_no_intent(monkeypatch):
    monkeypatch.delenv("MEMPALACE_DISABLE_LOCAL_RETRIEVAL", raising=False)
    assert _maybe_build_local_retrieval_context("Edit", {}, None, "sess1", "") == ""


def test_maybe_build_local_retrieval_context_fires_when_enabled(monkeypatch, tmp_path):
    """When env=on and intent is present, the helper calls _run_local_retrieval
    and emits a formatted block with the returned memories."""
    monkeypatch.delenv("MEMPALACE_DISABLE_LOCAL_RETRIEVAL", raising=False)
    intent = {
        "intent_id": "i1",
        "accessed_memory_ids": [],
        "_context_views": ["Implement feature X"],
    }
    fake_hits = [
        {"id": "mem_fresh_1", "preview": "tip about foo.py", "score": 0.9},
        {"id": "mem_fresh_2", "preview": "gotcha about pytest -x", "score": 0.8},
    ]
    with patch("mempalace.hooks_cli._run_local_retrieval", return_value=(fake_hits, None)):
        with patch("mempalace.hooks_cli._persist_accessed_memory_ids") as mock_persist:
            with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
                body = _maybe_build_local_retrieval_context(
                    "Edit",
                    {"file_path": "foo.py"},
                    intent,
                    "sess1",
                    str(tmp_path / "transcript.jsonl"),
                )
    assert "mem_fresh_1" in body
    assert "mem_fresh_2" in body
    assert "Local retrieval" in body
    mock_persist.assert_called_once()


def test_maybe_build_local_retrieval_context_surfaces_error(monkeypatch, tmp_path):
    """Post-silent-fail audit: exception must surface as visible
    MEMPALACE HOOK ERROR notice in the returned additionalContext,
    not swallowed to empty."""
    monkeypatch.delenv("MEMPALACE_DISABLE_LOCAL_RETRIEVAL", raising=False)
    intent = {"intent_id": "i1", "accessed_memory_ids": []}
    with patch(
        "mempalace.hooks_cli._run_local_retrieval",
        side_effect=RuntimeError("simulated chroma failure"),
    ):
        body = _maybe_build_local_retrieval_context(
            "Edit",
            {"file_path": "foo.py"},
            intent,
            "sess1",
            "",
        )
    assert "MEMPALACE HOOK ERROR" in body
    assert "simulated chroma failure" in body


def test_persist_accessed_memory_ids_merges_without_duplicates(tmp_path):
    state_file = tmp_path / "active_intent_sess1.json"
    state_file.write_text(
        json.dumps(
            {
                "intent_id": "i1",
                "accessed_memory_ids": ["mem_existing"],
            }
        ),
        encoding="utf-8",
    )
    with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
        _persist_accessed_memory_ids(
            "sess1",
            {"intent_id": "i1"},
            ["mem_existing", "mem_new_a", "mem_new_b"],
        )
    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert set(data["accessed_memory_ids"]) == {"mem_existing", "mem_new_a", "mem_new_b"}


def test_persist_accessed_memory_ids_noop_when_file_missing(tmp_path):
    """No state file = nothing to update; do not create a phantom file."""
    with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
        _persist_accessed_memory_ids("sess_nofile", {"intent_id": "i1"}, ["mem_a"])
    assert not (tmp_path / "active_intent_sess_nofile.json").exists()


def test_hook_pretooluse_injects_additional_context_when_enabled(monkeypatch, tmp_path):
    """End-to-end: hook returns additionalContext alongside allow when retrieval fires."""
    monkeypatch.delenv("MEMPALACE_DISABLE_LOCAL_RETRIEVAL", raising=False)
    # Seed a valid active intent
    (tmp_path / "active_intent_sess1.json").write_text(
        json.dumps(
            {
                "intent_id": "i1",
                "intent_type": "ship_mempalace_feature",
                "slots": {"paths": ["D:/tmp/**"]},
                "effective_permissions": [
                    {"tool": "Read", "scope": "D:/tmp/**"},
                ],
                "accessed_memory_ids": [],
                "budget": {"Read": 10},
                "used": {},
                "injected_memory_ids": [],
            }
        ),
        encoding="utf-8",
    )
    fake_hits = [{"id": "mem_fresh", "preview": "something relevant", "score": 0.9}]
    with patch("mempalace.hooks_cli._run_local_retrieval", return_value=(fake_hits, None)):
        with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
            buf = io.StringIO()
            with patch(
                "mempalace.hooks_cli._output",
                side_effect=lambda d: buf.write(json.dumps(d)),
            ):
                hook_pretooluse(
                    {
                        "tool_name": "Read",
                        "tool_input": {"file_path": "D:/tmp/x.py"},
                        "session_id": "sess1",
                    },
                    "claude-code",
                )
    result = json.loads(buf.getvalue())
    hso = result["hookSpecificOutput"]
    assert hso["permissionDecision"] == "allow"
    assert "additionalContext" in hso
    assert "mem_fresh" in hso["additionalContext"]


# --- Phase 3b: lazy-question detector --------------------------------


LAZY_EXAMPLES = [
    "should I continue?",
    "Should I proceed with the next step?",
    "shall I keep going",
    "What's next?",
    "what is the next step?",
    "anything else to do?",
    "is there more to do here?",
    "ready to continue?",
    "Should we continue or stop?",
    "do you want me to continue",
    "am I done?",
    "Want me to keep going?",
    '  "should I proceed" ',  # wrapped in quotes and whitespace
]

NOT_LAZY_EXAMPLES = [
    "Which marker format for the never-stop rule \u2014 A, B, C, or D?",
    "The user is on a Windows machine; should I paths-normalize to forward slashes in the rehydration payload?",
    "Deleting this migration file is destructive \u2014 confirm to proceed?",
    "Do we want embeddings for lazy-question prototypes enabled by default?",
    "What exact regex should catch 'should I continue' patterns?",  # meta-question \u2014 not lazy
    "",
    None,
    "continuation of the previous discussion",  # word 'continue' appears mid-sentence
]


def test_is_lazy_question_catches_lazy_phrasings():
    for text in LAZY_EXAMPLES:
        lazy, pat = _is_lazy_question(text)
        assert lazy, f"Expected lazy: {text!r}"
        assert pat, "matched pattern source should be non-empty"


def test_is_lazy_question_leaves_specific_questions_alone():
    for text in NOT_LAZY_EXAMPLES:
        lazy, _ = _is_lazy_question(text)
        assert not lazy, f"Did not expect lazy: {text!r}"


def test_extract_askuserquestion_texts_handles_multi_question_payload():
    tool_input = {
        "questions": [
            {"question": "Which backend?", "header": "Backend", "options": []},
            {"question": "Should I continue?", "header": "Go", "options": []},
            {"not_a_dict": True},  # malformed entry \u2014 ignored
        ]
    }
    texts = _extract_askuserquestion_texts(tool_input)
    assert texts == ["Which backend?", "Should I continue?"]


def test_extract_askuserquestion_texts_empty_on_malformed():
    assert _extract_askuserquestion_texts(None) == []
    assert _extract_askuserquestion_texts({}) == []
    assert _extract_askuserquestion_texts({"questions": "not a list"}) == []


def test_maybe_deny_lazy_askuserquestion_non_matching_tool_returns_none():
    assert _maybe_deny_lazy_askuserquestion("Read", {"file_path": "x"}) is None


def test_maybe_deny_lazy_askuserquestion_genuine_question_returns_none():
    tool_input = {
        "questions": [
            {
                "question": "Which break-glass pattern: env var or sentinel file?",
                "header": "Bypass",
                "options": [],
            }
        ]
    }
    assert _maybe_deny_lazy_askuserquestion("AskUserQuestion", tool_input) is None


def test_maybe_deny_lazy_askuserquestion_denies_any_lazy_in_batch():
    """A single lazy question in a multi-question call denies the whole call."""
    tool_input = {
        "questions": [
            {"question": "Which library?", "header": "Lib", "options": []},
            {"question": "Should I continue?", "header": "Go", "options": []},
        ]
    }
    deny = _maybe_deny_lazy_askuserquestion("AskUserQuestion", tool_input)
    assert deny is not None
    hso = deny["hookSpecificOutput"]
    assert hso["hookEventName"] == "PreToolUse"
    assert hso["permissionDecision"] == "deny"
    assert "LAZY-QUESTION REJECTED" in hso["permissionDecisionReason"]


def test_askuserquestion_is_NOT_skipped_by_retrieval_helper(monkeypatch, tmp_path):
    """Carve-out: AskUserQuestion is always-allowed (skips permission check)
    but its content IS a rich cue, so _maybe_build_local_retrieval_context
    should fire retrieval for it rather than returning empty."""
    monkeypatch.delenv("MEMPALACE_DISABLE_LOCAL_RETRIEVAL", raising=False)
    intent = {
        "intent_id": "i1",
        "accessed_memory_ids": [],
        "_context_views": ["Discussing design choices"],
    }
    fake_hits = [
        {
            "id": "mem_prior_decision",
            "preview": "Prior marker-format decision: use AskUserQuestion",
            "score": 0.9,
        }
    ]
    tool_input = {
        "questions": [
            {
                "question": "Which marker format for never-stop rule?",
                "header": "Marker",
                "options": [],
            }
        ]
    }
    with patch("mempalace.hooks_cli._run_local_retrieval", return_value=(fake_hits, None)):
        with patch("mempalace.hooks_cli._persist_accessed_memory_ids"):
            with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
                body = _maybe_build_local_retrieval_context(
                    "AskUserQuestion",
                    tool_input,
                    intent,
                    "sess1",
                    "",
                )
    assert "mem_prior_decision" in body
    assert "Local retrieval" in body


def test_hook_pretooluse_askuserquestion_emits_additional_context(monkeypatch, tmp_path):
    """End-to-end: AskUserQuestion allow path emits additionalContext when
    retrieval fires. Confirms the carve-out wires through from the
    ALWAYS_ALLOWED_TOOLS branch all the way to hookSpecificOutput."""
    monkeypatch.delenv("MEMPALACE_DISABLE_LOCAL_RETRIEVAL", raising=False)
    (tmp_path / "active_intent_sess1.json").write_text(
        json.dumps(
            {
                "intent_id": "i1",
                "intent_type": "ship_mempalace_feature",
                "slots": {"paths": ["D:/tmp/**"]},
                "effective_permissions": [],
                "accessed_memory_ids": [],
                "budget": {},
                "used": {},
                "injected_memory_ids": [],
            }
        ),
        encoding="utf-8",
    )
    fake_hits = [{"id": "mem_hit", "preview": "relevant prior context", "score": 0.9}]
    buf = io.StringIO()
    with patch("mempalace.hooks_cli._run_local_retrieval", return_value=(fake_hits, None)):
        with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
            with patch(
                "mempalace.hooks_cli._output",
                side_effect=lambda d: buf.write(json.dumps(d)),
            ):
                hook_pretooluse(
                    {
                        "tool_name": "AskUserQuestion",
                        "tool_input": {
                            "questions": [
                                {
                                    "question": "Genuine design question about marker format?",
                                    "header": "Marker",
                                    "options": [],
                                }
                            ]
                        },
                        "session_id": "sess1",
                    },
                    "claude-code",
                )
    result = json.loads(buf.getvalue())
    hso = result["hookSpecificOutput"]
    assert hso["permissionDecision"] == "allow"
    assert "additionalContext" in hso
    assert "mem_hit" in hso["additionalContext"]


def test_hook_pretooluse_denies_lazy_askuserquestion(tmp_path):
    """End-to-end: AskUserQuestion with a lazy question gets denied by hook."""
    buf = io.StringIO()
    with patch(
        "mempalace.hooks_cli._output",
        side_effect=lambda d: buf.write(json.dumps(d)),
    ):
        with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
            hook_pretooluse(
                {
                    "tool_name": "AskUserQuestion",
                    "tool_input": {
                        "questions": [
                            {
                                "question": "should I continue?",
                                "header": "Go",
                                "options": [],
                            }
                        ]
                    },
                    "session_id": "sess1",
                },
                "claude-code",
            )
    result = json.loads(buf.getvalue())
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "LAZY" in result["hookSpecificOutput"]["permissionDecisionReason"]


def test_hook_pretooluse_allows_genuine_askuserquestion(tmp_path):
    """End-to-end: AskUserQuestion with a specific question passes through
    the always-allowed path as normal."""
    buf = io.StringIO()
    with patch(
        "mempalace.hooks_cli._output",
        side_effect=lambda d: buf.write(json.dumps(d)),
    ):
        with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
            hook_pretooluse(
                {
                    "tool_name": "AskUserQuestion",
                    "tool_input": {
                        "questions": [
                            {
                                "question": "Which library: FastAPI or Flask?",
                                "header": "Framework",
                                "options": [],
                            }
                        ]
                    },
                    "session_id": "sess1",
                },
                "claude-code",
            )
    result = json.loads(buf.getvalue())
    assert result["hookSpecificOutput"]["permissionDecision"] == "allow"


# --- Phase 5: UserPromptSubmit hook ----------------------------------


def test_extract_prompt_keywords_lowercases_and_filters_stopwords():
    kws = _extract_prompt_keywords("What should I do with the brute-force rate-limit auth code?")
    assert "brute-force" in kws
    assert "rate-limit" in kws
    assert "auth" in kws
    for stop in ("what", "should", "the", "with"):
        assert stop not in kws


def test_extract_prompt_keywords_dedupes_and_caps():
    text = " ".join(["token"] * 50)
    assert _extract_prompt_keywords(text) == ["token"]
    text2 = " ".join(f"word{i}" for i in range(30))
    kws = _extract_prompt_keywords(text2, limit=5)
    assert len(kws) == 5


def test_extract_prompt_keywords_empty_on_bad_input():
    assert _extract_prompt_keywords("") == []
    assert _extract_prompt_keywords(None) == []


def test_hook_userpromptsubmit_disabled_by_default(monkeypatch, tmp_path):
    """Env off: pass through with empty output regardless of intent state."""
    monkeypatch.setenv("MEMPALACE_DISABLE_LOCAL_RETRIEVAL", "1")
    _write_active_intent(tmp_path, "sess1", _sample_intent())
    result = _capture_hook_output(
        hook_userpromptsubmit,
        {
            "session_id": "sess1",
            "prompt": "fix the auth bug",
            "stop_hook_active": False,
            "transcript_path": "",
        },
        state_dir=tmp_path,
    )
    assert result == {}


def test_hook_userpromptsubmit_no_active_intent_passes_through(monkeypatch, tmp_path):
    monkeypatch.delenv("MEMPALACE_DISABLE_LOCAL_RETRIEVAL", raising=False)
    result = _capture_hook_output(
        hook_userpromptsubmit,
        {
            "session_id": "sess_no_intent",
            "prompt": "anything",
            "stop_hook_active": False,
            "transcript_path": "",
        },
        state_dir=tmp_path,
    )
    assert result == {}


def test_hook_userpromptsubmit_empty_prompt_passes_through(monkeypatch, tmp_path):
    monkeypatch.delenv("MEMPALACE_DISABLE_LOCAL_RETRIEVAL", raising=False)
    _write_active_intent(tmp_path, "sess1", _sample_intent())
    result = _capture_hook_output(
        hook_userpromptsubmit,
        {"session_id": "sess1", "prompt": "   ", "stop_hook_active": False, "transcript_path": ""},
        state_dir=tmp_path,
    )
    assert result == {}


def test_hook_userpromptsubmit_emits_additional_context_when_enabled(monkeypatch, tmp_path):
    monkeypatch.delenv("MEMPALACE_DISABLE_LOCAL_RETRIEVAL", raising=False)
    _write_active_intent(tmp_path, "sess1", _sample_intent())
    fake_hits = [
        {
            "id": "mem_prompt_hit",
            "preview": "prior session solved similar auth issue",
            "score": 0.95,
        },
    ]
    with patch("mempalace.hooks_cli._run_local_retrieval", return_value=(fake_hits, None)):
        with patch("mempalace.hooks_cli._persist_accessed_memory_ids") as mock_persist:
            result = _capture_hook_output(
                hook_userpromptsubmit,
                {
                    "session_id": "sess1",
                    "prompt": "Fix the brute-force auth bug we saw earlier",
                    "stop_hook_active": False,
                    "transcript_path": "",
                },
                state_dir=tmp_path,
            )
    hso = result["hookSpecificOutput"]
    assert hso["hookEventName"] == "UserPromptSubmit"
    assert "mem_prompt_hit" in hso["additionalContext"]
    mock_persist.assert_called_once()


def test_hook_userpromptsubmit_cue_prefixes_user_said(monkeypatch, tmp_path):
    """Verify role prefix 'User said: ' wraps the prompt in queries[0]."""
    monkeypatch.delenv("MEMPALACE_DISABLE_LOCAL_RETRIEVAL", raising=False)
    _write_active_intent(tmp_path, "sess1", _sample_intent())
    captured_cue = {}

    def _fake_retrieval(cue, accessed, top_k):
        captured_cue.update(cue)
        return ([], None)

    with patch("mempalace.hooks_cli._run_local_retrieval", side_effect=_fake_retrieval):
        _capture_hook_output(
            hook_userpromptsubmit,
            {
                "session_id": "sess1",
                "prompt": "please run the tests",
                "stop_hook_active": False,
                "transcript_path": "",
            },
            state_dir=tmp_path,
        )
    assert captured_cue["queries"][0].startswith(USER_PROMPT_CUE_PREFIX)
    assert "please run the tests" in captured_cue["queries"][0]


def test_hook_userpromptsubmit_surfaces_error_on_exception(monkeypatch, tmp_path):
    """Post-silent-fail audit: hook no longer swallows retrieval errors.
    Top-level exception must be recorded AND visible in additionalContext."""
    monkeypatch.delenv("MEMPALACE_DISABLE_LOCAL_RETRIEVAL", raising=False)
    _write_active_intent(tmp_path, "sess1", _sample_intent())
    with patch(
        "mempalace.hooks_cli._run_local_retrieval",
        side_effect=RuntimeError("simulated chroma failure"),
    ):
        result = _capture_hook_output(
            hook_userpromptsubmit,
            {
                "session_id": "sess1",
                "prompt": "anything",
                "stop_hook_active": False,
                "transcript_path": "",
            },
            state_dir=tmp_path,
        )
    # Expect visible error notice rather than silent pass-through
    hso = result.get("hookSpecificOutput", {})
    assert hso, f"Expected hookSpecificOutput with error notice, got {result}"
    body = hso.get("additionalContext", "")
    assert "MEMPALACE HOOK ERROR" in body or "hook_userpromptsubmit" in body


def test_hook_userpromptsubmit_skips_when_no_hits(monkeypatch, tmp_path):
    monkeypatch.delenv("MEMPALACE_DISABLE_LOCAL_RETRIEVAL", raising=False)
    _write_active_intent(tmp_path, "sess1", _sample_intent())
    with patch("mempalace.hooks_cli._run_local_retrieval", return_value=([], None)):
        result = _capture_hook_output(
            hook_userpromptsubmit,
            {
                "session_id": "sess1",
                "prompt": "anything",
                "stop_hook_active": False,
                "transcript_path": "",
            },
            state_dir=tmp_path,
        )
    assert result == {}


def test_run_hook_dispatches_userpromptsubmit(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMPALACE_DISABLE_LOCAL_RETRIEVAL", "1")
    stdin_data = json.dumps(
        {
            "session_id": "sess_ups",
            "prompt": "hello",
            "stop_hook_active": False,
            "transcript_path": "",
        }
    )
    with patch("sys.stdin", io.StringIO(stdin_data)):
        with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
            with patch("mempalace.hooks_cli._output") as mock_output:
                run_hook("userpromptsubmit", "claude-code")
    mock_output.assert_called_once_with({})


def test_hook_pretooluse_no_additional_context_when_disabled(monkeypatch, tmp_path):
    """Default (env off): allow path omits additionalContext entirely."""
    monkeypatch.setenv("MEMPALACE_DISABLE_LOCAL_RETRIEVAL", "1")
    (tmp_path / "active_intent_sess1.json").write_text(
        json.dumps(
            {
                "intent_id": "i1",
                "intent_type": "ship_mempalace_feature",
                "slots": {"paths": ["D:/tmp/**"]},
                "effective_permissions": [
                    {"tool": "Read", "scope": "D:/tmp/**"},
                ],
                "accessed_memory_ids": [],
                "budget": {"Read": 10},
                "used": {},
                "injected_memory_ids": [],
            }
        ),
        encoding="utf-8",
    )
    with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
        buf = io.StringIO()
        with patch(
            "mempalace.hooks_cli._output",
            side_effect=lambda d: buf.write(json.dumps(d)),
        ):
            hook_pretooluse(
                {
                    "tool_name": "Read",
                    "tool_input": {"file_path": "D:/tmp/x.py"},
                    "session_id": "sess1",
                },
                "claude-code",
            )
    result = json.loads(buf.getvalue())
    hso = result["hookSpecificOutput"]
    assert hso["permissionDecision"] == "allow"
    assert "additionalContext" not in hso
