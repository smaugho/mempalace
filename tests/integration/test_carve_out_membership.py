"""v3.10.16 regression: ALWAYS_ALLOWED_TOOLS must include every
harness/infra tool that should bypass declare_operation.

Background
----------
Adrian (msg_6f0496_11, 2026-05-27) flagged ScheduleWakeup as an infra
tool that must be permitted alongside TodoWrite / Skill / Task / etc.
Audit also found EnterPlanMode missing (its sibling ExitPlanMode was
already in the carve-out -- asymmetric oversight).

This test locks the membership so future refactors can't quietly drop
a harness tool out of the carve-out, which would force agents to wrap
every ScheduleWakeup / EnterPlanMode call in a declare_operation that
serves no retrieval purpose.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_v31016_schedule_wakeup_is_carved_out():
    """v3.10.16: ScheduleWakeup must be in ALWAYS_ALLOWED_TOOLS."""
    from mempalace.hooks_cli import ALWAYS_ALLOWED_TOOLS

    assert "ScheduleWakeup" in ALWAYS_ALLOWED_TOOLS, (
        "v3.10.16: ScheduleWakeup is a harness self-resume scheduler "
        "(same category as TodoWrite / Task / Skill). It must be in "
        "ALWAYS_ALLOWED_TOOLS so agents don't have to wrap it in a "
        "declare_operation that serves no retrieval purpose."
    )


def test_v31016_enter_plan_mode_is_carved_out():
    """v3.10.16: EnterPlanMode must be in ALWAYS_ALLOWED_TOOLS
    (symmetric counterpart to ExitPlanMode)."""
    from mempalace.hooks_cli import ALWAYS_ALLOWED_TOOLS

    assert "EnterPlanMode" in ALWAYS_ALLOWED_TOOLS, (
        "v3.10.16: EnterPlanMode is the symmetric counterpart of "
        "ExitPlanMode (which is already carved out). Same harness/meta "
        "category. Missing previously was an oversight, not a design "
        "choice."
    )


def test_carve_out_baseline_membership_intact():
    """Anti-regression sentinel: the long-standing carve-out members
    must remain present. If this fails, somebody removed a harness
    tool from the carve-out and agents will see spurious gate errors."""
    from mempalace.hooks_cli import ALWAYS_ALLOWED_TOOLS

    expected_baseline = {
        # Built-in harness / meta tools
        "Agent",
        "Skill",
        "ToolSearch",
        "TaskCreate",
        "TaskUpdate",
        "TaskGet",
        "TaskList",
        "TaskOutput",
        "TaskStop",
        "TodoWrite",
        "ExitPlanMode",
        "AskUserQuestion",
        # v3.10.16 additions (also locked here so the pair stays in)
        "EnterPlanMode",
        "ScheduleWakeup",
    }
    missing = expected_baseline - ALWAYS_ALLOWED_TOOLS
    assert not missing, (
        f"Carve-out lost expected members: {sorted(missing)}. "
        "Every entry in expected_baseline is a harness / meta tool "
        "that must NOT require declare_operation."
    )
