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


@pytest.fixture(autouse=True)
def _force_foreground_state_judge(monkeypatch):
    """v3.7.4 Slice 3 compatibility: this file pins the foreground
    state_judge behavior (raises, auto-apply, state_changes_detected
    on same-op response). v3.7.4 moves judge to a bg worker by
    default (MEMPALACE_BG_STATE_JUDGE=1), which makes the same-op
    response carry NO state_changes_detected and the foreground
    coverage check short-circuit with an empty flagged set -- both
    intentional under the new contract. Forcing the env-off path
    here preserves the foreground-judge code path so the
    v2_visibility Phase 1 / Phase 2 / Phase 3 contracts stay locked.
    Bg path has dedicated tests in tests/unit/test_bg_state_judge.py.
    """
    monkeypatch.setenv("MEMPALACE_BG_STATE_JUDGE", "0")
    yield


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
    """v3.10.3 (Adrian directive 2026-05-22): the v0_strict opt-out and
    every state_deltas blocking raise were DELETED. Setting
    MEMPALACE_STATE_PROTOCOL=v0_strict must have NO effect -- the op
    proceeds and the judge's findings still surface. Guards that blocking
    can never be resurrected via env var."""

    def test_v0_strict_env_no_longer_blocks_on_missing_state_deltas(self):
        os.environ["MEMPALACE_STATE_PROTOCOL"] = "v0_strict"
        try:
            result = self._intent.tool_declare_operation(
                tool="Bash",
                args_summary="Bash {command}",
                context=self._ctx(),
                agent=self.agent,
                # state_deltas omitted; the fake judge flags ga_agent.
                # Pre-v3.10.3 this blocked; now it MUST proceed.
            )
        finally:
            os.environ.pop("MEMPALACE_STATE_PROTOCOL", None)
        self.assertTrue(
            result.get("success"),
            f"v0_strict must NOT block anymore; got {result}",
        )
        self.assertNotIn(
            "missing_state_deltas",
            result,
            f"missing_state_deltas blocking is removed; got {result}",
        )
        self.assertIn(
            "state_changes_detected",
            result,
            f"judge findings should still surface; got {result}",
        )


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

    def test_unknown_flag_value_keeps_v2_default(self):
        """v3.4.0 Phase 3 only the exact string 'v0_strict'
        opts back into the strict gates. Any other value (typo,
        legacy 'v2_full', empty, garbage) falls through to the v2
        default and the op succeeds with state_changes_detected."""
        os.environ["MEMPALACE_STATE_PROTOCOL"] = "v2_full"  # unrecognized
        try:
            result = self._intent.tool_declare_operation(
                tool="Bash",
                args_summary="Bash {command}",
                context=self._ctx(),
                agent=self.agent,
            )
        finally:
            os.environ.pop("MEMPALACE_STATE_PROTOCOL", None)
        self.assertTrue(
            result.get("success"),
            f"unrecognized flag value should fall through to v2 default; got {result}",
        )
        self.assertNotIn("missing_state_deltas", result)
        self.assertIn("state_changes_detected", result)


# ── v3.2.8 Phase 2: env-flag now also bypasses ──
# (a) declare_operation unchanged_violations raise
# (b) finalize_intent missing_state_deltas (via _all_complete)
# (c) extend_feedback missing_state_deltas


class TestPhase2UnchangedViolationsGate(_PhaseOneFixture):
    """declare_operation: unchanged_violations raise gated on v2_visibility."""

    def test_unchanged_for_non_flagged_never_blocks(self):
        """v3.10.3: status='unchanged' for a non-flagged entity used to be
        rejected (unchanged_violations). That raise is DELETED -- the op
        proceeds regardless of env, even with v0_strict set."""
        os.environ["MEMPALACE_STATE_PROTOCOL"] = "v0_strict"
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
            f"unchanged_violations blocking is removed; got {result}",
        )
        self.assertNotIn(
            "unchanged_violations",
            result,
            f"unchanged_violations key should never appear; got {result}",
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


# ── v3.2.9 Phase 3 judge auto-write on v2_visibility ──
# When the judge returns a change WITH schema_id + RFC 6902 patch
# AND the env flag is on AND the agent did not cover the entity via
# state_deltas, intent.py auto-applies the patch via
# record_state_revision with agent='state_judge'. The response's
# state_changes_detected entry carries applied=True + rev_id so the
# agent can see what was written.


class _PhaseThreeFixture(_Slice12Fixture):
    """Slice12 fixture + forced judge flag carrying a real patch."""

    def setUp(self):
        super().setUp()
        from mempalace import injection_gate as _ig

        self._ig = _ig
        self._orig_run_state_judge = _ig.run_state_judge

        def _fake_judge_with_patch(*args, **kwargs):
            # Patch the agent's current_focus -- ga_agent is_a agent
            # which carries agent_state schema (current_focus required).
            changes = [
                {
                    "entity_id": "ga_agent",
                    "reason": "fake_judge: current_focus stale",
                    "schema_id": "agent_state",
                    "patch": [
                        {
                            "op": "add",
                            "path": "/current_focus",
                            "value": "phase3_auto_apply_test",
                        }
                    ],
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

        _ig.run_state_judge = _fake_judge_with_patch

    def tearDown(self):
        try:
            self._ig.run_state_judge = self._orig_run_state_judge
        except Exception:
            pass
        os.environ.pop("MEMPALACE_STATE_PROTOCOL", None)
        super().tearDown()


class TestPhase3AutoApply(_PhaseThreeFixture):
    """v2_visibility ON + judge returns patch -> auto-write a revision."""

    def test_v2_visibility_auto_applies_judge_patch(self):
        os.environ["MEMPALACE_STATE_PROTOCOL"] = "v2_visibility"
        try:
            result = self._intent.tool_declare_operation(
                tool="Bash",
                args_summary="Bash {command}",
                context=self._ctx(),
                agent=self.agent,
            )
        finally:
            os.environ.pop("MEMPALACE_STATE_PROTOCOL", None)

        self.assertTrue(
            result.get("success"),
            f"v2_visibility should let op succeed; got {result}",
        )
        flagged = result.get("state_changes_detected") or []
        self.assertTrue(flagged, f"expected state_changes_detected entries; got {result}")
        ga_entry = next(
            (c for c in flagged if c.get("entity_id") == "ga_agent"),
            None,
        )
        self.assertIsNotNone(ga_entry, f"ga_agent missing from flagged; got {flagged}")
        self.assertTrue(
            ga_entry.get("applied"),
            f"ga_agent change should be applied=True; got {ga_entry}",
        )
        self.assertIn(
            "rev_id",
            ga_entry,
            f"applied change must carry rev_id; got {ga_entry}",
        )

    def test_auto_apply_fires_regardless_of_env(self):
        """v3.10.3: auto-apply is now UNCONDITIONAL. Even with the old
        v0_strict env set, the op proceeds and the judge's patch is
        auto-applied -- no env can disable it or block the op."""
        os.environ["MEMPALACE_STATE_PROTOCOL"] = "v0_strict"
        try:
            result = self._intent.tool_declare_operation(
                tool="Bash",
                args_summary="Bash {command}",
                context=self._ctx(),
                agent=self.agent,
            )
        finally:
            os.environ.pop("MEMPALACE_STATE_PROTOCOL", None)
        self.assertTrue(
            result.get("success"),
            f"v0_strict must not block; got {result}",
        )
        self.assertNotIn("missing_state_deltas", result)
        flagged = result.get("state_changes_detected") or []
        ga_entry = next(
            (c for c in flagged if c.get("entity_id") == "ga_agent"),
            None,
        )
        self.assertIsNotNone(ga_entry, f"ga_agent missing from flagged; got {flagged}")
        self.assertTrue(
            ga_entry.get("applied"),
            f"auto-apply must fire regardless of env; got {ga_entry}",
        )

    def test_v2_visibility_skips_auto_apply_when_agent_covered_entity(self):
        """When the agent provides state_deltas for the flagged entity,
        the judge auto-write is skipped (skip_reason='agent_covered')."""
        os.environ["MEMPALACE_STATE_PROTOCOL"] = "v2_visibility"
        try:
            result = self._intent.tool_declare_operation(
                tool="Bash",
                args_summary="Bash {command}",
                context=self._ctx(),
                agent=self.agent,
                state_deltas=[
                    {
                        "entity_id": "ga_agent",
                        "status": "changed",
                        "patch": [
                            {
                                "op": "add",
                                "path": "/current_focus",
                                "value": "agent_supplied_focus",
                            }
                        ],
                    }
                ],
            )
        finally:
            os.environ.pop("MEMPALACE_STATE_PROTOCOL", None)
        self.assertTrue(
            result.get("success"),
            f"agent_covered case should succeed; got {result}",
        )
        flagged = result.get("state_changes_detected") or []
        ga_entry = next(
            (c for c in flagged if c.get("entity_id") == "ga_agent"),
            None,
        )
        if ga_entry is not None:
            # The judge still flagged it; the auto-apply was skipped.
            self.assertFalse(
                ga_entry.get("applied", False),
                f"agent_covered must NOT auto-apply; got {ga_entry}",
            )
            if "skip_reason" in ga_entry:
                self.assertEqual(ga_entry["skip_reason"], "agent_covered")


pytestmark = pytest.mark.integration
