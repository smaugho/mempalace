"""Tests for the reflag anti-oscillation guard in
``KnowledgeGraph.record_memory_flags`` (steps 1+4, 2026-05-23).

The flag dedup index is partial (``WHERE resolved_ts IS NULL``), so
resolving a flag drops it from the index and the identical
``(kind, memory_key, context_id)`` re-inserts as a fresh row -- an
unbounded flag/resolve ping-pong. The guard adds two brakes consulted
before INSERT:

* Step 1 -- COOLDOWN: a key resolved within
  ``MEMPALACE_FLAG_REFLAG_COOLDOWN_MIN`` minutes is not re-flagged.
* Step 4 -- CIRCUIT BREAKER: once a key has been resolved
  ``MEMPALACE_FLAG_MAX_REFLAGS`` times it is suppressed permanently.

Each knob set to ``0`` disables that brake (mirrors the settling knob).
Settling is disabled in every test so the guard is exercised in
isolation.
"""

from __future__ import annotations

from mempalace.knowledge_graph import KnowledgeGraph

_FLAG = {"kind": "generic_summary", "memory_ids": ["mem-1"], "context_id": "ctx-1"}


def _kg(tmp_path):
    return KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))


def _resolve_only_pending(kg, resolution="summary_rewritten"):
    """Resolve the single pending flag; assert there was exactly one."""
    pending = kg.list_pending_flags()
    assert len(pending) == 1, f"expected 1 pending flag, got {len(pending)}"
    assert kg.mark_flag_resolved(pending[0]["id"], resolution)


def test_cooldown_blocks_reflag_after_recent_resolution(tmp_path, monkeypatch):
    """A flag resolved within the cooldown window is not re-flagged."""
    monkeypatch.setenv("MEMPALACE_FLAG_SETTLING_MIN", "0")
    monkeypatch.setenv("MEMPALACE_FLAG_REFLAG_COOLDOWN_MIN", "1440")
    monkeypatch.setenv("MEMPALACE_FLAG_MAX_REFLAGS", "5")
    kg = _kg(tmp_path)

    assert kg.record_memory_flags([dict(_FLAG)]) == 1
    _resolve_only_pending(kg)

    # Same (kind, key, ctx) just resolved -> suppressed by cooldown.
    assert kg.record_memory_flags([dict(_FLAG)]) == 0
    assert kg.count_pending_flags() == 0


def test_cooldown_disabled_allows_reflag(tmp_path, monkeypatch):
    """With the cooldown knob at 0 (and breaker high), reflag is allowed."""
    monkeypatch.setenv("MEMPALACE_FLAG_SETTLING_MIN", "0")
    monkeypatch.setenv("MEMPALACE_FLAG_REFLAG_COOLDOWN_MIN", "0")
    monkeypatch.setenv("MEMPALACE_FLAG_MAX_REFLAGS", "0")
    kg = _kg(tmp_path)

    assert kg.record_memory_flags([dict(_FLAG)]) == 1
    _resolve_only_pending(kg)

    # No brakes -> the identical flag re-inserts as a fresh pending row.
    assert kg.record_memory_flags([dict(_FLAG)]) == 1
    assert kg.count_pending_flags() == 1


def test_circuit_breaker_suppresses_after_max_reflags(tmp_path, monkeypatch):
    """After MAX_REFLAGS resolve cycles the key is permanently suppressed."""
    monkeypatch.setenv("MEMPALACE_FLAG_SETTLING_MIN", "0")
    # Disable cooldown so only the breaker is under test.
    monkeypatch.setenv("MEMPALACE_FLAG_REFLAG_COOLDOWN_MIN", "0")
    monkeypatch.setenv("MEMPALACE_FLAG_MAX_REFLAGS", "3")
    kg = _kg(tmp_path)

    # 3 full flag->resolve cycles are permitted.
    for _ in range(3):
        assert kg.record_memory_flags([dict(_FLAG)]) == 1
        _resolve_only_pending(kg, resolution="no_action")

    # 3 resolved rows now exist -> the 4th flag is suppressed forever.
    assert kg.record_memory_flags([dict(_FLAG)]) == 0
    assert kg.count_pending_flags() == 0


def test_guard_is_per_key_not_global(tmp_path, monkeypatch):
    """Suppressing one key must not block a different key."""
    monkeypatch.setenv("MEMPALACE_FLAG_SETTLING_MIN", "0")
    monkeypatch.setenv("MEMPALACE_FLAG_REFLAG_COOLDOWN_MIN", "1440")
    monkeypatch.setenv("MEMPALACE_FLAG_MAX_REFLAGS", "5")
    kg = _kg(tmp_path)

    assert kg.record_memory_flags([dict(_FLAG)]) == 1
    _resolve_only_pending(kg)
    # First key is now in cooldown...
    assert kg.record_memory_flags([dict(_FLAG)]) == 0

    # ...but a different memory_key is unaffected.
    other = {"kind": "generic_summary", "memory_ids": ["mem-2"], "context_id": "ctx-1"}
    assert kg.record_memory_flags([other]) == 1
    assert kg.count_pending_flags() == 1
