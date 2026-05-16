"""Unit tests for mempalace.intent v3.7.4 Slice 3 background state_judge
helpers: per-sid pending buffer (_append, _drain, reset) + env-gate
(_bg_state_judge_enabled).

The actual bg spawn from apply_gate's parallel block is covered
end-to-end at the integration layer; this file pins the small helper
contract so a future refactor of the buffer layout can't silently
drop guarantees (per-sid isolation, empty-changes no-op, FIFO drain).
"""

from __future__ import annotations

import pytest

from mempalace import intent

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolate_buffer():
    """Every test gets a clean per-sid buffer so cross-test leakage is
    impossible. _reset_pending_state_updates is the dedicated escape
    hatch for fixtures."""
    intent._reset_pending_state_updates()
    yield
    intent._reset_pending_state_updates()


# ─────────────────────────────────────────────────────────────────────
# Buffer helpers
# ─────────────────────────────────────────────────────────────────────


class TestAppendDrain:
    def test_drain_empty_buffer_returns_empty_list(self):
        assert intent._drain_pending_state_updates("sid_a") == []

    def test_append_then_drain_returns_the_entry(self):
        intent._append_pending_state_updates(
            "sid_a",
            [{"entity_id": "ctx_x", "reason": "moved"}],
            {"elapsed_ms": 1234},
        )
        drained = intent._drain_pending_state_updates("sid_a")
        assert len(drained) == 1
        assert drained[0]["changes"][0]["entity_id"] == "ctx_x"
        assert drained[0]["report"]["elapsed_ms"] == 1234

    def test_empty_changes_is_no_op(self):
        intent._append_pending_state_updates("sid_a", [], {"elapsed_ms": 0})
        assert intent._drain_pending_state_updates("sid_a") == []

    def test_drain_pops_the_bucket(self):
        intent._append_pending_state_updates("sid_a", [{"entity_id": "ctx_x"}], None)
        first = intent._drain_pending_state_updates("sid_a")
        second = intent._drain_pending_state_updates("sid_a")
        assert len(first) == 1
        assert second == []

    def test_multiple_appends_drain_in_order(self):
        intent._append_pending_state_updates("sid_a", [{"entity_id": "e1"}], None)
        intent._append_pending_state_updates("sid_a", [{"entity_id": "e2"}], None)
        intent._append_pending_state_updates("sid_a", [{"entity_id": "e3"}], None)
        drained = intent._drain_pending_state_updates("sid_a")
        eids = [d["changes"][0]["entity_id"] for d in drained]
        assert eids == ["e1", "e2", "e3"]


class TestPerSidIsolation:
    def test_two_sids_drain_independently(self):
        intent._append_pending_state_updates("sid_a", [{"entity_id": "a"}], None)
        intent._append_pending_state_updates("sid_b", [{"entity_id": "b"}], None)
        a = intent._drain_pending_state_updates("sid_a")
        # sid_b's bucket survives the sid_a drain.
        b = intent._drain_pending_state_updates("sid_b")
        assert len(a) == 1
        assert a[0]["changes"][0]["entity_id"] == "a"
        assert len(b) == 1
        assert b[0]["changes"][0]["entity_id"] == "b"

    def test_empty_sid_is_its_own_bucket(self):
        intent._append_pending_state_updates("", [{"entity_id": "ghost"}], None)
        assert intent._drain_pending_state_updates("") == [
            {"changes": [{"entity_id": "ghost"}], "report": None}
        ]
        assert intent._drain_pending_state_updates("sid_a") == []


# ─────────────────────────────────────────────────────────────────────
# Env-gate
# ─────────────────────────────────────────────────────────────────────


class TestEnvGate:
    def test_default_enabled(self, monkeypatch):
        monkeypatch.delenv("MEMPALACE_BG_STATE_JUDGE", raising=False)
        assert intent._bg_state_judge_enabled() is True

    def test_explicit_zero_disables(self, monkeypatch):
        monkeypatch.setenv("MEMPALACE_BG_STATE_JUDGE", "0")
        assert intent._bg_state_judge_enabled() is False

    def test_explicit_one_enables(self, monkeypatch):
        monkeypatch.setenv("MEMPALACE_BG_STATE_JUDGE", "1")
        assert intent._bg_state_judge_enabled() is True

    def test_whitespace_zero_disables(self, monkeypatch):
        monkeypatch.setenv("MEMPALACE_BG_STATE_JUDGE", "  0  ")
        assert intent._bg_state_judge_enabled() is False

    def test_truthy_non_zero_string_enables(self, monkeypatch):
        # Any non-"0" string is enabled (the gate is permissive on the
        # default-on side; only explicit "0" turns it off).
        monkeypatch.setenv("MEMPALACE_BG_STATE_JUDGE", "true")
        assert intent._bg_state_judge_enabled() is True
