"""v3.3.0 Phase 3 Slice B: mempalace_challenge_state_change MCP.

Slice A (v3.2.9) gave the state_judge the ability to auto-write RFC
6902 patches to mempalace_state_revisions under
MEMPALACE_STATE_PROTOCOL=v2_visibility, attributed as
agent='state_judge'. Slice B (this module's surface under test)
closes the deferred-write protocol by letting the agent explicitly
challenge any specific revision -- either restoring prior state
(restore_prior=True, default) or filing an info-only challenge
(restore_prior=False).

These tests use the _Slice12Fixture to wire an isolated KG into the
mempalace singletons (same pattern v3.2.7+8+9 tests already use), then
seed a state_judge-attributed revision directly via
kg.record_state_revision so each test exercises the challenge handler
in isolation -- no Anthropic call, no env flag dance.

Test cases:
  1. Round-trip restore: agent challenges with restore_prior=True ->
     new revision restores prior state + challenge row stamped with
     retracted_rev_id + JTMS audit columns populated.
  2. Info-only: agent challenges with restore_prior=False -> judge
     revision stays current state; challenge row has null
     retracted_rev_id but the audit row exists.
  3. Missing rev: challenge against non-existent rev_id -> handler
     returns success=False with a clear error; no row written.
"""

from __future__ import annotations

import pytest

from tests.integration.test_operation_slots_slice12 import _Slice12Fixture


class _SliceBFixture(_Slice12Fixture):
    """Slice12 fixture + state-judge seeded revision against ga_agent."""

    def setUp(self):
        super().setUp()
        # Seed two revisions on ga_agent: a prior agent-written one,
        # then a judge auto-applied one on top. The Slice B challenge
        # targets the judge revision; restore_prior=True should rewind
        # to the agent's prior payload.
        self._prior_payload = {
            "current_focus": "agent_authored_focus",
        }
        self._prior_rev_id = self.kg.record_state_revision(
            "ga_agent",
            "agent_state",
            self._prior_payload,
            op_context_id="ctx_seed_agent",
            agent="ga_agent",
        )
        self._judge_payload = {
            "current_focus": "judge_overwrote_focus",
        }
        self._judge_rev_id = self.kg.record_state_revision(
            "ga_agent",
            "agent_state",
            self._judge_payload,
            op_context_id="ctx_seed_judge",
            agent="state_judge",
        )


class TestChallengeStateChangeRestore(_SliceBFixture):
    """restore_prior=True writes a new revision restoring prior state."""

    def test_round_trip_restore(self):
        from mempalace.tool_lifecycle import tool_challenge_state_change

        result = tool_challenge_state_change(
            rev_id=self._judge_rev_id,
            justification="judge over-eagerly stomped my focus",
            restore_prior=True,
            agent="ga_agent",
        )
        self.assertTrue(
            result.get("success"),
            f"restore round-trip should succeed; got {result}",
        )
        self.assertIn("challenge_id", result)
        restored = result.get("restored_rev_id")
        self.assertTrue(
            restored,
            f"restore_prior=True must produce a restored_rev_id; got {result}",
        )
        # The restored revision should carry the prior agent payload.
        latest = self.kg.latest_state_for_entity("ga_agent") or {}
        self.assertEqual(
            latest,
            self._prior_payload,
            f"latest_state should match prior payload; got {latest}",
        )

        # The challenge row must exist with retracted_rev_id pointing
        # at the restore revision.
        conn = self.kg._conn()
        row = conn.execute(
            "SELECT rev_id, agent, retracted_rev_id, justification "
            "FROM mempalace_state_revision_challenges "
            "WHERE challenge_id = ?",
            (result["challenge_id"],),
        ).fetchone()
        self.assertIsNotNone(row, "challenge row missing from table")
        self.assertEqual(row[0], self._judge_rev_id)
        self.assertEqual(row[1], "ga_agent")
        self.assertEqual(row[2], restored)
        self.assertIn("over-eagerly", row[3] or "")


class TestChallengeStateChangeInfoOnly(_SliceBFixture):
    """restore_prior=False leaves judge write in place but files audit."""

    def test_info_only_does_not_rewrite_state(self):
        from mempalace.tool_lifecycle import tool_challenge_state_change

        result = tool_challenge_state_change(
            rev_id=self._judge_rev_id,
            justification="info-only flag; not rolling back",
            restore_prior=False,
            agent="ga_agent",
        )
        self.assertTrue(result.get("success"), f"got {result}")
        self.assertIsNone(
            result.get("restored_rev_id"),
            f"info-only must not write a restore revision; got {result}",
        )
        # Judge payload is still the latest.
        latest = self.kg.latest_state_for_entity("ga_agent") or {}
        self.assertEqual(
            latest,
            self._judge_payload,
            f"info-only must leave judge state in place; got {latest}",
        )
        # Challenge row exists with retracted_rev_id null.
        conn = self.kg._conn()
        row = conn.execute(
            "SELECT retracted_rev_id FROM mempalace_state_revision_challenges "
            "WHERE challenge_id = ?",
            (result["challenge_id"],),
        ).fetchone()
        self.assertIsNotNone(row, "challenge row missing")
        self.assertIsNone(row[0], f"info-only retracted_rev_id must be null; got {row[0]}")


class TestChallengeStateChangeMissingRev(_SliceBFixture):
    """Challenging a non-existent rev_id returns success=False cleanly."""

    def test_missing_rev_rejected(self):
        from mempalace.tool_lifecycle import tool_challenge_state_change

        result = tool_challenge_state_change(
            rev_id="srv_nonexistent_999",
            justification="this revision does not exist",
            restore_prior=True,
            agent="ga_agent",
        )
        self.assertFalse(
            result.get("success"),
            f"missing rev must fail; got {result}",
        )
        self.assertIn("not found", (result.get("error") or "").lower())


pytestmark = pytest.mark.integration
