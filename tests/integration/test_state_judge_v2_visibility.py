"""v3.2.7 Phase 1: opt-in MEMPALACE_STATE_PROTOCOL=v2_visibility env flag.

Adrian directive 2026-05-12: the v0 state_judge gate blocks every op
when the judge flags an entity the agent didn't cover with state_deltas.
That's the "state_judge flagged entities you didn't cover" error Adrian
sees on every other op. The v2 redesign is multi-session work; Phase 1
is the smallest opt-in slice -- env-flag-on skips the v0 raise and
lets the op succeed while attaching state_changes_detected to the
response so the agent still sees what the judge flagged.

Tests gate the contract:
  * env flag OFF (default) -> v0 behavior preserved (block + return
    missing_state_deltas).
  * env flag ON -> op succeeds, response carries state_changes_detected.

These tests subclass _Slice12Fixture so they get the proven setup
(session_id, active_intent, ctx_test_op, context_lookup_or_create
monkeypatch, _persist_active_intent no-op). On top of that they
monkeypatch mempalace.injection_gate.run_state_judge to force a
deterministic flag, so the gate actually has something to react to.
"""

from __future__ import annotations

import os

import pytest

from tests.integration.test_operation_slots_slice12 import _Slice12Fixture


class _PhaseOneFixture(_Slice12Fixture):
    """Slice12 fixture + forced state_judge flag so the gate fires."""

    def setUp(self):
        super().setUp()
        # Force state_judge to flag ga_agent on every op. The real judge
        # lives at mempalace.injection_gate.run_state_judge; intent.py
        # imports it locally as _run_state_judge. In tests it fail-opens
        # (no API key). We patch the source module so both intent.py
        # call sites see our forced flag.
        from mempalace import injection_gate as _ig

        self._ig = _ig
        self._orig_run_state_judge = _ig.run_state_judge

        def _fake_judge(*args, **kwargs):
            changes = [
                {
                    "entity_id": "ga_agent",
                    "reason": "fake_judge: forced flag for v3.2.7 Phase 1 test",
                }
            ]
            report = {
                "elapsed_ms": 0.0,
                "detected_count": 1,
                "tokens": {
                    "input": 0,
                    "output": 0,
                    "cache_read": 0,
                    "cache_creation": 0,
                },
            }
            return changes, report

        _ig.run_state_judge = _fake_judge

    def tearDown(self):
        try:
            self._ig.run_state_judge = self._orig_run_state_judge
        except Exception:
            pass
        # Belt-and-suspenders: clear the env flag in case a test set it
        # and an assertion raised before the test's own cleanup ran.
        os.environ.pop("MEMPALACE_STATE_PROTOCOL", None)
        super().tearDown()


class TestPhase1EnvFlagOff(_PhaseOneFixture):
    """Default behavior: env flag absent -> v0 block on missing state_deltas."""

    def test_v0_default_blocks_on_missing_state_deltas(self):
        os.environ.pop("MEMPALACE_STATE_PROTOCOL", None)
        result = self._intent.tool_declare_operation(
            tool="Bash",
            args_summary="Bash {command}",
            context=self._ctx(),
            agent=self.agent,
            # state_deltas omitted; the fake judge flags ga_agent, so
            # the v0 gate must block.
        )
        self.assertFalse(
            result.get("success"),
            f"v0 default should block on missing_state_deltas; got {result}",
        )
        self.assertIn(
            "missing_state_deltas",
            result,
            f"expected missing_state_deltas key in failure response; got {result}",
        )
        self.assertIn("ga_agent", result.get("missing_state_deltas") or [])


class TestPhase1EnvFlagOn(_PhaseOneFixture):
    """v2_visibility opt-in: env flag set -> op succeeds + emits state_changes_detected."""

    def test_v2_visibility_skips_block_and_emits_changes(self):
        os.environ["MEMPALACE_STATE_PROTOCOL"] = "v2_visibility"
        try:
            result = self._intent.tool_declare_operation(
                tool="Bash",
                args_summary="Bash {command}",
                context=self._ctx(),
                agent=self.agent,
                # Same call shape as the v0 test; difference is the env flag.
            )
        finally:
            os.environ.pop("MEMPALACE_STATE_PROTOCOL", None)

        self.assertTrue(
            result.get("success"),
            f"v2_visibility should let the op succeed; got {result}",
        )
        self.assertNotIn(
            "missing_state_deltas",
            result,
            f"v2_visibility should NOT emit missing_state_deltas; got {result}",
        )
        # The judge's findings still surface so the agent can see what
        # was flagged.
        self.assertIn(
            "state_changes_detected",
            result,
            f"v2_visibility should surface state_changes_detected on success; got {result}",
        )
        flagged = result["state_changes_detected"]
        self.assertTrue(
            any((c.get("entity_id") == "ga_agent") for c in flagged),
            f"ga_agent should appear in state_changes_detected; got {flagged}",
        )

    def test_v2_visibility_is_case_insensitive_and_strip_tolerant(self):
        """MEMPALACE_STATE_PROTOCOL is parsed via .strip().lower()."""
        os.environ["MEMPALACE_STATE_PROTOCOL"] = "  V2_VISIBILITY  "
        try:
            result = self._intent.tool_declare_operation(
                tool="Bash",
                args_summary="Bash {command}",
                context=self._ctx(),
                agent=self.agent,
            )
        finally:
            os.environ.pop("MEMPALACE_STATE_PROTOCOL", None)
        self.assertTrue(result.get("success"), f"got {result}")

    def test_unknown_flag_value_keeps_v0_behavior(self):
        """Any value other than v2_visibility leaves v0 in effect."""
        os.environ["MEMPALACE_STATE_PROTOCOL"] = "v2_full"  # not yet implemented
        try:
            result = self._intent.tool_declare_operation(
                tool="Bash",
                args_summary="Bash {command}",
                context=self._ctx(),
                agent=self.agent,
            )
        finally:
            os.environ.pop("MEMPALACE_STATE_PROTOCOL", None)
        self.assertFalse(
            result.get("success"),
            f"unrecognized flag value should keep v0 block; got {result}",
        )
        self.assertIn("missing_state_deltas", result)


# ── v3.2.8 Phase 2: env-flag now also bypasses ──
# (a) declare_operation unchanged_violations raise
# (b) finalize_intent missing_state_deltas (via _all_complete)
# (c) extend_feedback missing_state_deltas


class TestPhase2UnchangedViolationsGate(_PhaseOneFixture):
    """declare_operation: unchanged_violations raise gated on v2_visibility."""

    def test_v0_default_blocks_unchanged_for_non_flagged_entity(self):
        """Sanity: with v0 default, status='unchanged' for non-flagged
        entity (task_alpha; judge only flags ga_agent) is rejected."""
        os.environ.pop("MEMPALACE_STATE_PROTOCOL", None)
        result = self._intent.tool_declare_operation(
            tool="Bash",
            args_summary="Bash {command}",
            context=self._ctx(),
            agent=self.agent,
            state_deltas=[
                {
                    "entity_id": "task_alpha",
                    "status": "unchanged",
                    "justification": "explicit override attempt",
                }
            ],
        )
        self.assertFalse(
            result.get("success"),
            f"v0 should reject unchanged-for-non-flagged; got {result}",
        )
        self.assertIn(
            "unchanged_violations",
            result,
            f"expected unchanged_violations on failure; got {result}",
        )

    def test_v2_visibility_skips_unchanged_violations_block(self):
        """v2_visibility: same call succeeds; agent doesn't get
        blocked on this bookkeeping rule either."""
        os.environ["MEMPALACE_STATE_PROTOCOL"] = "v2_visibility"
        try:
            result = self._intent.tool_declare_operation(
                tool="Bash",
                args_summary="Bash {command}",
                context=self._ctx(),
                agent=self.agent,
                state_deltas=[
                    {
                        "entity_id": "task_alpha",
                        "status": "unchanged",
                        "justification": "explicit override attempt",
                    }
                ],
            )
        finally:
            os.environ.pop("MEMPALACE_STATE_PROTOCOL", None)
        self.assertTrue(
            result.get("success"),
            f"v2_visibility should skip unchanged_violations block; got {result}",
        )
        self.assertNotIn(
            "unchanged_violations",
            result,
            f"v2_visibility should NOT emit unchanged_violations; got {result}",
        )


pytestmark = pytest.mark.integration
