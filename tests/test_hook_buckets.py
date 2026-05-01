"""Tests for the three-bucket carve-out gate (Slice C).

Three layers of test:

1. **Drift sentinel** -- the bucket basename frozensets in
   ``mempalace.hooks_cli`` are hardcoded for hook-subprocess speed, so
   they MUST stay in sync with each ``tool_*.py`` module's ``__all__``.
   These tests import the bucket modules and compare. If a handler
   moves bucket or a new one is added, this test breaks loudly.

2. **Pure-logic helpers** -- ``_mempalace_basename``, ``_bucket_of``,
   ``_is_user_intent_tier0``. Verify they accept/reject the right
   inputs for both bare and MCP-prefixed names.

3. **End-to-end pretooluse decision** -- run the hook handler with a
   minimal harness payload and assert allow/deny for representative
   tools, with and without active intent, with and without pending
   user_messages. Uses tmp_path to scope ``~/.mempalace/hook_state/``.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from mempalace import hooks_cli
from mempalace.tool_lifecycle import __all__ as LIFECYCLE_ALL
from mempalace.tool_mutate import __all__ as MUTATE_ALL
from mempalace.tool_read import __all__ as READ_ALL


# ── Shared helpers ────────────────────────────────────────────────────


def _bare_to_basename(name: str) -> str:
    """tool_kg_search → mempalace_kg_search."""
    assert name.startswith("tool_"), name
    return "mempalace_" + name[len("tool_") :]


def _mcp_id(basename: str, plugin_prefix: str = "plugin_3_1_11_mempalace") -> str:
    """Build an MCP-prefixed tool ID like the hook actually receives."""
    return f"mcp__{plugin_prefix}__{basename}"


# ── 1. Drift sentinel ─────────────────────────────────────────────────


class TestBucketDriftSentinel:
    """Bucket basename sets in hooks_cli must equal each module's __all__."""

    def test_lifecycle_bucket_matches_module_all(self):
        expected = {_bare_to_basename(n) for n in LIFECYCLE_ALL}
        assert hooks_cli._LIFECYCLE_BUCKET_BASENAMES == expected, (
            "tool_lifecycle.__all__ drifted from "
            "hooks_cli._LIFECYCLE_BUCKET_BASENAMES. Update both."
        )

    def test_read_bucket_matches_module_all(self):
        expected = {_bare_to_basename(n) for n in READ_ALL}
        assert hooks_cli._READ_BUCKET_BASENAMES == expected, (
            "tool_read.__all__ drifted from hooks_cli._READ_BUCKET_BASENAMES. Update both."
        )

    def test_mutate_bucket_matches_module_all(self):
        expected = {_bare_to_basename(n) for n in MUTATE_ALL}
        assert hooks_cli._MUTATE_BUCKET_BASENAMES == expected, (
            "tool_mutate.__all__ drifted from hooks_cli._MUTATE_BUCKET_BASENAMES. Update both."
        )

    def test_buckets_are_disjoint(self):
        lifecycle = hooks_cli._LIFECYCLE_BUCKET_BASENAMES
        read = hooks_cli._READ_BUCKET_BASENAMES
        mutate = hooks_cli._MUTATE_BUCKET_BASENAMES
        assert lifecycle & read == set(), f"lifecycle ∩ read = {lifecycle & read}"
        assert lifecycle & mutate == set(), f"lifecycle ∩ mutate = {lifecycle & mutate}"
        assert read & mutate == set(), f"read ∩ mutate = {read & mutate}"

    def test_user_intent_tier0_subset_of_lifecycle(self):
        # extend_feedback + declare_user_intents both live in lifecycle.
        assert hooks_cli._USER_INTENT_TIER0_BASENAMES.issubset(
            hooks_cli._LIFECYCLE_BUCKET_BASENAMES
        )

    def test_tier0_basenames_are_canonical(self):
        # Spec from Adrian 2026-04-27 (extended 2026-05-01): tier-0
        # is the closed set of mempalace tools allowed when a pending
        # user_message queue blocks the rest of the toolkit. Cold-start
        # lock 2026-05-01 added mempalace_wake_up to the carve-out so
        # fresh palaces don't deadlock at first user message
        # (declare_user_intents requires a declared agent; wake_up is
        # the only path that bootstraps the agent on a cold palace).
        assert hooks_cli._USER_INTENT_TIER0_BASENAMES == frozenset(
            {
                "mempalace_declare_user_intents",
                "mempalace_extend_feedback",
                "mempalace_wake_up",
            }
        )


# ── 2. Pure-logic helpers ─────────────────────────────────────────────


class TestMempalaceBasename:
    def test_extracts_basename_from_versioned_mcp_id(self):
        assert (
            hooks_cli._mempalace_basename("mcp__plugin_3_1_11_mempalace__mempalace_kg_search")
            == "mempalace_kg_search"
        )

    def test_extracts_basename_from_alternate_plugin_prefix(self):
        # Plugin prefixes vary by install (e.g. mcp__plugin_mempalace_mempalace__*).
        assert (
            hooks_cli._mempalace_basename(
                "mcp__plugin_mempalace_mempalace__mempalace_finalize_intent"
            )
            == "mempalace_finalize_intent"
        )

    def test_returns_empty_for_non_mcp_tool(self):
        assert hooks_cli._mempalace_basename("Bash") == ""
        assert hooks_cli._mempalace_basename("Read") == ""
        assert hooks_cli._mempalace_basename("AskUserQuestion") == ""

    def test_returns_empty_for_non_mempalace_mcp(self):
        assert hooks_cli._mempalace_basename("mcp__plugin_filesystem__read_file") == ""

    def test_returns_empty_for_empty_input(self):
        assert hooks_cli._mempalace_basename("") == ""


class TestBucketOf:
    @pytest.mark.parametrize(
        "basename",
        sorted(
            {
                "mempalace_active_intent",
                "mempalace_declare_intent",
                "mempalace_declare_user_intents",
                "mempalace_extend_feedback",
                "mempalace_extend_intent",
                "mempalace_finalize_intent",
                "mempalace_resolve_conflicts",
                "mempalace_wake_up",
            }
        ),
    )
    def test_lifecycle_bucket(self, basename):
        assert hooks_cli._bucket_of(_mcp_id(basename)) == "lifecycle"

    @pytest.mark.parametrize(
        "basename",
        sorted(
            {
                "mempalace_diary_read",
                "mempalace_kg_list_declared",
                "mempalace_kg_query",
                "mempalace_kg_search",
                "mempalace_kg_stats",
                "mempalace_kg_timeline",
            }
        ),
    )
    def test_read_bucket(self, basename):
        assert hooks_cli._bucket_of(_mcp_id(basename)) == "read"

    @pytest.mark.parametrize(
        "basename",
        sorted(
            {
                "mempalace_declare_operation",
                "mempalace_diary_write",
                "mempalace_kg_add",
                "mempalace_kg_add_batch",
                "mempalace_kg_declare_entity",
                "mempalace_kg_delete_entity",
                "mempalace_kg_invalidate",
                "mempalace_kg_merge_entities",
                "mempalace_kg_update_entity",
            }
        ),
    )
    def test_mutate_bucket(self, basename):
        assert hooks_cli._bucket_of(_mcp_id(basename)) == "mutate"

    def test_unknown_mempalace_tool_returns_empty(self):
        # Future / legacy un-bucketed mempalace tools fall back to "".
        assert hooks_cli._bucket_of("mcp__plugin_3_1_11_mempalace__mempalace_brand_new_tool") == ""

    def test_non_mempalace_returns_empty(self):
        assert hooks_cli._bucket_of("Bash") == ""
        assert hooks_cli._bucket_of("mcp__plugin_filesystem__read_file") == ""


class TestIsUserIntentTier0:
    def test_declare_user_intents_is_tier0(self):
        assert hooks_cli._is_user_intent_tier0(_mcp_id("mempalace_declare_user_intents"))

    def test_extend_feedback_is_tier0(self):
        assert hooks_cli._is_user_intent_tier0(_mcp_id("mempalace_extend_feedback"))

    def test_declare_intent_is_NOT_tier0(self):
        # Adrian 2026-04-27 spec: declare_intent does NOT bypass pending
        # user-message preemption.
        assert not hooks_cli._is_user_intent_tier0(_mcp_id("mempalace_declare_intent"))

    def test_finalize_intent_is_NOT_tier0(self):
        assert not hooks_cli._is_user_intent_tier0(_mcp_id("mempalace_finalize_intent"))

    def test_kg_search_is_NOT_tier0(self):
        # Reads are also blocked under preemption.
        assert not hooks_cli._is_user_intent_tier0(_mcp_id("mempalace_kg_search"))

    def test_kg_add_is_NOT_tier0(self):
        assert not hooks_cli._is_user_intent_tier0(_mcp_id("mempalace_kg_add"))

    def test_non_mempalace_is_NOT_tier0(self):
        assert not hooks_cli._is_user_intent_tier0("Bash")


# ── 3. End-to-end pretooluse decision ─────────────────────────────────


@pytest.fixture
def isolated_hook_state(tmp_path, monkeypatch):
    """Redirect ~/.mempalace/hook_state/ to a tmp dir for the test."""
    state_dir = tmp_path / "hook_state"
    state_dir.mkdir(parents=True)
    monkeypatch.setattr(hooks_cli, "STATE_DIR", state_dir)
    # Disable any local-retrieval network
    monkeypatch.setenv("MEMPALACE_DISABLE_LOCAL_RETRIEVAL", "1")
    return state_dir


def _run_pretooluse(tool_name: str, session_id: str, tool_input: dict | None = None) -> dict:
    """Invoke hook_pretooluse and parse the JSON it writes to stdout."""
    payload = {
        "session_id": session_id,
        "tool_name": tool_name,
        "tool_input": tool_input or {},
    }
    buf = io.StringIO()
    with redirect_stdout(buf):
        hooks_cli.hook_pretooluse(payload, harness="claude-code")
    output = buf.getvalue().strip()
    return json.loads(output) if output else {}


def _write_active_intent(state_dir: Path, session_id: str, intent: dict) -> None:
    safe_sid = hooks_cli._sanitize_session_id(session_id)
    (state_dir / f"active_intent_{safe_sid}.json").write_text(json.dumps(intent), encoding="utf-8")


def _write_pending_user_message(state_dir: Path, session_id: str) -> None:
    """Write a pending-user-message file matching _read_pending_user_messages's
    expected shape: {"session_id": <sid>, "messages": [{id, text, ...}]}."""
    safe_sid = hooks_cli._sanitize_session_id(session_id)
    (state_dir / f"pending_user_messages_{safe_sid}.json").write_text(
        json.dumps(
            {
                "session_id": session_id,
                "messages": [{"id": "msg_test_pending_001", "text": "hi"}],
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.usefixtures("isolated_hook_state")
class TestPretooluseBucketGate:
    def test_lifecycle_allowed_with_no_active_intent(self, isolated_hook_state):
        """Lifecycle tools bypass the intent gate."""
        result = _run_pretooluse(
            _mcp_id("mempalace_declare_intent"),
            session_id="sess_no_intent",
        )
        assert result["hookSpecificOutput"]["permissionDecision"] == "allow", result

    def test_read_allowed_with_no_active_intent(self, isolated_hook_state):
        """Read tools bypass the intent gate."""
        result = _run_pretooluse(
            _mcp_id("mempalace_kg_search"),
            session_id="sess_no_intent",
        )
        assert result["hookSpecificOutput"]["permissionDecision"] == "allow", result

    def test_mutate_denied_with_no_active_intent(self, isolated_hook_state):
        """Mutate tools require an active intent."""
        result = _run_pretooluse(
            _mcp_id("mempalace_kg_add"),
            session_id="sess_no_intent",
        )
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny", result
        reason = result["hookSpecificOutput"]["permissionDecisionReason"]
        assert "mutate" in reason.lower()
        assert "active intent" in reason.lower()

    def test_declare_operation_denied_with_no_active_intent(self, isolated_hook_state):
        """declare_operation lives in mutate bucket -- needs active intent."""
        result = _run_pretooluse(
            _mcp_id("mempalace_declare_operation"),
            session_id="sess_no_intent",
        )
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny", result

    def test_mutate_allowed_with_active_intent(self, isolated_hook_state):
        """Mutate tools allowed once an intent is active."""
        _write_active_intent(
            isolated_hook_state,
            "sess_with_intent",
            {
                "intent_id": "intent_test_001",
                "intent_type": "develop",
                "permitted_tools": ["Bash"],
            },
        )
        result = _run_pretooluse(
            _mcp_id("mempalace_kg_add"),
            session_id="sess_with_intent",
        )
        assert result["hookSpecificOutput"]["permissionDecision"] == "allow", result

    def test_pending_user_message_blocks_lifecycle_except_declare_user_intents(
        self, isolated_hook_state
    ):
        """Under preemption, declare_intent is blocked; declare_user_intents passes."""
        _write_pending_user_message(isolated_hook_state, "sess_pending")

        # declare_intent → blocked
        blocked = _run_pretooluse(
            _mcp_id("mempalace_declare_intent"),
            session_id="sess_pending",
        )
        assert blocked["hookSpecificOutput"]["permissionDecision"] == "deny"

        # declare_user_intents → allowed
        allowed = _run_pretooluse(
            _mcp_id("mempalace_declare_user_intents"),
            session_id="sess_pending",
        )
        assert allowed["hookSpecificOutput"]["permissionDecision"] == "allow"

    def test_pending_user_message_blocks_reads(self, isolated_hook_state):
        """Read tools also blocked under user-message preemption."""
        _write_pending_user_message(isolated_hook_state, "sess_pending")
        result = _run_pretooluse(
            _mcp_id("mempalace_kg_search"),
            session_id="sess_pending",
        )
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_pending_user_message_blocks_mutates(self, isolated_hook_state):
        """Mutate tools also blocked under user-message preemption."""
        _write_pending_user_message(isolated_hook_state, "sess_pending")
        result = _run_pretooluse(
            _mcp_id("mempalace_kg_add"),
            session_id="sess_pending",
        )
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_pending_user_message_allows_extend_feedback(self, isolated_hook_state):
        """extend_feedback is the second tier-0 carve-out."""
        _write_pending_user_message(isolated_hook_state, "sess_pending")
        result = _run_pretooluse(
            _mcp_id("mempalace_extend_feedback"),
            session_id="sess_pending",
        )
        assert result["hookSpecificOutput"]["permissionDecision"] == "allow"

    def test_pending_user_message_allows_askuserquestion(self, isolated_hook_state):
        """AskUserQuestion is always allowed (clarify path)."""
        _write_pending_user_message(isolated_hook_state, "sess_pending")
        result = _run_pretooluse(
            "AskUserQuestion",
            session_id="sess_pending",
            tool_input={"questions": [{"text": "What did you mean?"}]},
        )
        assert result["hookSpecificOutput"]["permissionDecision"] == "allow"

    def test_block_disabled_env_restores_old_behaviour(self, isolated_hook_state, monkeypatch):
        """MEMPALACE_USER_INTENT_BLOCK_DISABLED=1 rolls back to pre-Slice-B-2."""
        _write_pending_user_message(isolated_hook_state, "sess_pending")
        monkeypatch.setenv("MEMPALACE_USER_INTENT_BLOCK_DISABLED", "1")
        result = _run_pretooluse(
            _mcp_id("mempalace_kg_search"),
            session_id="sess_pending",
        )
        # With the block disabled, the read bucket bypasses unconditionally.
        assert result["hookSpecificOutput"]["permissionDecision"] == "allow"
