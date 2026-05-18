"""Unit tests for mempalace.conflict_resolver_auto (v3.7.19 slice 1).

Covers the observation-only bg Haiku conflict resolver:
- env-gate (MEMPALACE_CONFLICT_RESOLVER_AUTO_DISABLED=1) skips submit.
- submit_conflict rejects malformed input cleanly.
- _call_haiku falls back to action='abstain' when the SDK is missing
  OR when ANTHROPIC_API_KEY is missing.
- _call_haiku parses the resolve_conflict tool_use block on success.
- _log_result writes the expected telemetry row shape.

All tests mock the anthropic SDK and the telemetry writer so they run
hermetically (no real API calls, no real file I/O).
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from mempalace import conflict_resolver_auto as cra

pytestmark = pytest.mark.unit


# -- helpers ---------------------------------------------------------


def _make_conflict(conflict_id: str = "conf_1", **overrides) -> dict:
    base = {
        "id": conflict_id,
        "conflict_type": "entity_collision",
        "existing_id": "entity_alpha",
        "new_name": "entity_alpha_v2",
        "new_what": "alpha v2 noun phrase",
        "similarity": 0.93,
        "reason": "Identity collision (cos=0.930)",
    }
    base.update(overrides)
    return base


def _make_tool_use_resp(
    action="keep", reason="both valid", confidence=0.8, into="", merged_content=""
):
    block = types.SimpleNamespace(
        type="tool_use",
        name="resolve_conflict",
        input={
            "action": action,
            "reason": reason,
            "confidence": confidence,
            "into": into,
            "merged_content": merged_content,
        },
    )
    usage = types.SimpleNamespace(
        input_tokens=100,
        output_tokens=20,
        cache_read_input_tokens=80,
        cache_creation_input_tokens=0,
    )
    return types.SimpleNamespace(content=[block], usage=usage, stop_reason="end_turn")


# -- env-gate --------------------------------------------------------


class TestDisabledGate:
    def test_disabled_when_env_var_is_one(self, monkeypatch):
        monkeypatch.setenv("MEMPALACE_CONFLICT_RESOLVER_AUTO_DISABLED", "1")
        assert cra._disabled() is True

    def test_enabled_when_env_var_unset(self, monkeypatch):
        monkeypatch.delenv("MEMPALACE_CONFLICT_RESOLVER_AUTO_DISABLED", raising=False)
        assert cra._disabled() is False

    def test_enabled_when_env_var_is_zero(self, monkeypatch):
        monkeypatch.setenv("MEMPALACE_CONFLICT_RESOLVER_AUTO_DISABLED", "0")
        assert cra._disabled() is False

    def test_whitespace_one_disables(self, monkeypatch):
        monkeypatch.setenv("MEMPALACE_CONFLICT_RESOLVER_AUTO_DISABLED", "  1  ")
        assert cra._disabled() is True


# -- submit_conflict input validation --------------------------------


class TestSubmitConflict:
    def test_disabled_short_circuits_before_executor(self, monkeypatch):
        monkeypatch.setenv("MEMPALACE_CONFLICT_RESOLVER_AUTO_DISABLED", "1")
        mock_exec = MagicMock()
        monkeypatch.setattr(cra, "_get_executor", lambda: mock_exec)
        cra.submit_conflict(_make_conflict())
        mock_exec.submit.assert_not_called()

    def test_missing_id_short_circuits(self, monkeypatch):
        monkeypatch.delenv("MEMPALACE_CONFLICT_RESOLVER_AUTO_DISABLED", raising=False)
        mock_exec = MagicMock()
        monkeypatch.setattr(cra, "_get_executor", lambda: mock_exec)
        cra.submit_conflict({"conflict_type": "entity_collision"})  # no id
        mock_exec.submit.assert_not_called()

    def test_non_dict_short_circuits(self, monkeypatch):
        monkeypatch.delenv("MEMPALACE_CONFLICT_RESOLVER_AUTO_DISABLED", raising=False)
        mock_exec = MagicMock()
        monkeypatch.setattr(cra, "_get_executor", lambda: mock_exec)
        cra.submit_conflict("not a dict")  # type: ignore[arg-type]
        mock_exec.submit.assert_not_called()

    def test_valid_conflict_enqueues(self, monkeypatch):
        monkeypatch.delenv("MEMPALACE_CONFLICT_RESOLVER_AUTO_DISABLED", raising=False)
        mock_exec = MagicMock()
        monkeypatch.setattr(cra, "_get_executor", lambda: mock_exec)
        cra.submit_conflict(
            _make_conflict(), agent="ga_agent", intent_type="audit", session_id="sid_x"
        )
        mock_exec.submit.assert_called_once()
        args, _kwargs = mock_exec.submit.call_args
        # First positional arg is the worker function; second is the batch.
        assert args[0] is cra._run
        batch = args[1]
        assert batch.conflict["id"] == "conf_1"
        assert batch.agent == "ga_agent"
        assert batch.intent_type == "audit"
        assert batch.session_id == "sid_x"

    def test_executor_exception_does_not_propagate(self, monkeypatch):
        monkeypatch.delenv("MEMPALACE_CONFLICT_RESOLVER_AUTO_DISABLED", raising=False)
        mock_exec = MagicMock()
        mock_exec.submit.side_effect = RuntimeError("executor down")
        monkeypatch.setattr(cra, "_get_executor", lambda: mock_exec)
        # Must NOT raise -- the mint path can't be killed by resolver failure.
        cra.submit_conflict(_make_conflict())


# -- _call_haiku fallback paths --------------------------------------


class TestCallHaikuFallbacks:
    def test_missing_anthropic_sdk_returns_abstain(self, monkeypatch):
        # Set anthropic to None in sys.modules so the local `import anthropic`
        # inside _call_haiku raises ImportError. patch.dict restores cleanly
        # when the test exits IF we don't mutate sys.modules outside its
        # scope (an earlier draft did a sys.modules.pop BEFORE patch.dict;
        # that bypassed the restore and broke test_link_author_api_key
        # tests downstream because they import anthropic for real -- a
        # classic test-pollution leak).
        with patch.dict("sys.modules", {"anthropic": None}):
            batch = cra.ConflictResolverInput(conflict=_make_conflict())
            out = cra._call_haiku(batch)
        # Either ImportError-path or no-key-path can fire (depends on
        # whether the real SDK is installed). Both return abstain.
        assert out.recommended_action == "abstain"
        assert out.error is not None

    def test_missing_api_key_returns_abstain(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        # Stub _ensure_env_loaded so it doesn't repopulate the key.
        monkeypatch.setattr("mempalace.auto_author._ensure_env_loaded", lambda: None)
        batch = cra.ConflictResolverInput(conflict=_make_conflict())
        out = cra._call_haiku(batch)
        assert out.recommended_action == "abstain"
        assert out.error == "no_api_key"

    def test_haiku_exception_returns_abstain(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("simulated API down")
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            batch = cra.ConflictResolverInput(conflict=_make_conflict())
            out = cra._call_haiku(batch)
        assert out.recommended_action == "abstain"
        assert "simulated API down" in (out.error or "")

    def test_haiku_no_tool_use_block_returns_abstain(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        empty_resp = types.SimpleNamespace(content=[], usage=None, stop_reason="end_turn")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = empty_resp
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            batch = cra.ConflictResolverInput(conflict=_make_conflict())
            out = cra._call_haiku(batch)
        assert out.recommended_action == "abstain"
        assert out.error == "no_tool_use"


# -- _call_haiku success path ----------------------------------------


class TestCallHaikuSuccess:
    def test_keep_action_parses_cleanly(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_tool_use_resp(
            action="keep",
            reason="similarity 0.93 < 0.95; distinct semantics likely",
            confidence=0.75,
        )
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            batch = cra.ConflictResolverInput(conflict=_make_conflict())
            out = cra._call_haiku(batch)
        assert out.recommended_action == "keep"
        assert out.confidence == 0.75
        assert "distinct" in out.reason
        assert out.tokens_in == 100
        assert out.tokens_out == 20
        assert out.cache_read_input_tokens == 80
        assert out.error is None

    def test_merge_action_parses_into_and_merged_content(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_tool_use_resp(
            action="merge",
            reason="similarity 0.98; both carry distinct info",
            confidence=0.9,
            into="entity_alpha",
            merged_content="alpha v1 + v2 combined description here",
        )
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            batch = cra.ConflictResolverInput(conflict=_make_conflict(similarity=0.98))
            out = cra._call_haiku(batch)
        assert out.recommended_action == "merge"
        assert out.into == "entity_alpha"
        assert out.merged_content.startswith("alpha v1 + v2")


# -- _log_result telemetry row shape ---------------------------------


class TestLogResult:
    def test_writes_expected_row_shape(self, monkeypatch):
        captured = {}

        def fake_tel(filename, row):
            captured["filename"] = filename
            captured["row"] = row

        # Patch the lazy import inside _log_result so we don't have to
        # bootstrap the full mcp_server module.
        fake_mcp = types.SimpleNamespace(_telemetry_append_jsonl=fake_tel)
        with patch.dict("sys.modules", {"mempalace.mcp_server": fake_mcp}):
            batch = cra.ConflictResolverInput(
                conflict=_make_conflict(),
                agent="ga_agent",
                intent_type="audit_fix_workflow",
                session_id="sid_alpha",
            )
            result = cra.ConflictResolverResult(
                conflict_id="conf_1",
                recommended_action="keep",
                reason="distinct semantics",
                confidence=0.8,
                tokens_in=100,
                tokens_out=20,
                cache_read_input_tokens=80,
                elapsed_ms=1234.5,
            )
            # v3.7.20: _log_result now takes applied + apply_error too.
            cra._log_result(result, batch, applied=True, apply_error=None)

        assert captured["filename"] == "conflict_resolver_log.jsonl"
        row = captured["row"]
        assert row["conflict_id"] == "conf_1"
        assert row["recommended_action"] == "keep"
        assert row["confidence"] == 0.8
        assert row["agent"] == "ga_agent"
        assert row["intent_type"] == "audit_fix_workflow"
        assert row["session_id"] == "sid_alpha"
        assert row["conflict_type"] == "entity_collision"
        assert row["existing_id"] == "entity_alpha"
        # new_id falls back to new_name when new_id missing.
        assert row["new_id"] == "entity_alpha_v2"
        assert row["similarity"] == 0.93
        assert row["tokens_in"] == 100
        assert row["elapsed_ms"] == 1234.5
        # v3.7.20 invariants: active resolution; apply succeeded here.
        assert row["applied"] is True
        assert row["apply_error"] == ""
        assert row["slice"] == "v3.7.20-active"
        # The Haiku call itself didn't error -- the rename from 'error'
        # to 'haiku_error' separates Haiku failures from apply failures.
        assert row["haiku_error"] == ""

    def test_log_failure_swallowed(self, monkeypatch):
        def boom(*_a, **_k):
            raise IOError("disk full")

        fake_mcp = types.SimpleNamespace(_telemetry_append_jsonl=boom)
        with patch.dict("sys.modules", {"mempalace.mcp_server": fake_mcp}):
            batch = cra.ConflictResolverInput(conflict=_make_conflict())
            result = cra.ConflictResolverResult(
                conflict_id="conf_1",
                recommended_action="keep",
                reason="r",
                confidence=0.5,
            )
            # Must NOT raise -- telemetry failures never escape.
            cra._log_result(result, batch, applied=False, apply_error="test")


# ─────────────────────────────────────────────────────────────────────
# FINDING #S (v3.7.33 2026-05-18, Adrian's post-v3.7.32 audit):
# submit_conflict must early-return on two classes of false-positive
# conflicts: (1) view-suffix rows (__body / __identity / __vN added
# by v3.7.29) which are per-view vectors of an existing entity, not
# new records; (2) execution+result twin pairs deliberately created
# by finalize_intent as a pair (execution carries metadata + edges,
# result memory carries prose narrative). Pre-v3.7.33 11 __body
# conflicts + 10 twin merges leaked into the resolver log; one twin
# was actually MERGED (data corruption). These tests lock the
# filter so the regression cannot return.
# ─────────────────────────────────────────────────────────────────────


class TestFindingS_SubmitConflictFilters:
    """Lock the filter contracts so future regressions get caught."""

    def _captured_submits(self, monkeypatch):
        """Replace _get_executor with a capturer; return the list."""
        captured = []

        class _FakeExecutor:
            def submit(self, fn, batch):
                captured.append(batch)

        monkeypatch.setattr(cra, "_get_executor", lambda: _FakeExecutor())
        monkeypatch.setattr(cra, "_disabled", lambda: False)
        return captured

    def test_view_suffix_body_filtered(self, monkeypatch):
        """__body view rows must NOT be submitted as conflicts."""
        captured = self._captured_submits(monkeypatch)
        cra.submit_conflict(
            {
                "id": "conf_1",
                "conflict_type": "memory_duplicate",
                "existing_id": "some_record__body",
                "new_id": "another_record",
                "similarity": 1.0,
            }
        )
        assert captured == [], (
            "v3.7.33 regression: __body view row was submitted to resolver; "
            "must be filtered at submit_conflict boundary"
        )

    def test_view_suffix_identity_filtered(self, monkeypatch):
        """__identity view rows must NOT be submitted as conflicts."""
        captured = self._captured_submits(monkeypatch)
        cra.submit_conflict(
            {
                "id": "conf_1",
                "conflict_type": "memory_duplicate",
                "existing_id": "another_record",
                "new_id": "some_entity__identity",
                "similarity": 0.95,
            }
        )
        assert captured == []

    def test_view_suffix_vN_filtered(self, monkeypatch):
        """__v0/__v1/__vN probe view rows must NOT be submitted as conflicts."""
        captured = self._captured_submits(monkeypatch)
        cra.submit_conflict(
            {
                "id": "conf_1",
                "conflict_type": "memory_duplicate",
                "existing_id": "some_entity__v17",
                "new_id": "another_record",
                "similarity": 0.91,
            }
        )
        assert captured == []

    def test_twin_pair_execution_result_filtered(self, monkeypatch):
        """The finalize_intent twin pattern (execution + result memory)
        must NOT be submitted as a conflict. Pattern:
          existing_id = <base>
          new_id      = record_<agent>_result_<base>
        (or vice versa).
        """
        captured = self._captured_submits(monkeypatch)
        cra.submit_conflict(
            {
                "id": "conf_1",
                "conflict_type": "memory_duplicate",
                "existing_id": "wrap_ten_ship_arc_v3732_2026_05_18",
                "new_id": "record_ga_agent_result_wrap_ten_ship_arc_v3732_2026_05_18",
                "similarity": 0.92,
            }
        )
        assert captured == [], (
            "v3.7.33 regression: execution+result twin was submitted to "
            "resolver; must be filtered as a deliberate finalize_intent pair"
        )

    def test_twin_pair_inverse_direction_filtered(self, monkeypatch):
        """Same as above but the result memory is existing_id and the
        execution entity is new_id."""
        captured = self._captured_submits(monkeypatch)
        cra.submit_conflict(
            {
                "id": "conf_1",
                "conflict_type": "memory_duplicate",
                "existing_id": "record_ga_agent_result_foo",
                "new_id": "foo",
                "similarity": 0.88,
            }
        )
        assert captured == []

    def test_genuine_collision_still_submitted(self, monkeypatch):
        """A genuine record-vs-record collision (no view suffix, no twin
        pattern) MUST still be submitted -- the filter must not over-block."""
        captured = self._captured_submits(monkeypatch)
        cra.submit_conflict(
            {
                "id": "conf_1",
                "conflict_type": "memory_duplicate",
                "existing_id": "record_ga_agent_alpha",
                "new_id": "record_ga_agent_beta",
                "similarity": 0.93,
            }
        )
        assert len(captured) == 1, (
            "v3.7.33 over-block regression: a genuine record collision "
            "between two distinct records was filtered; resolver must still "
            "see it"
        )
