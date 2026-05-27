"""v3.10.15 regression: gardener spawn MUST sweep stale in-flight
memory_gardener_runs rows.

Background
----------
Today's gardener audit found 131 rows with completed_ts IS NULL older
than 7 days (oldest 424h, started 2026-05-08). These are orphans from
killed Claude Code sessions / OS reboots: the gardener subprocess died
mid-flag, kernel-flock released the lockfile at process exit, but the
in-flight memory_gardener_runs row stayed NULL forever.

v3.10.15 adds KnowledgeGraph.gc_stale_gardener_runs(ttl_minutes=60),
called from memory_gardener.process_batch right before
kg.start_gardener_run. The kernel-flock at gardener spawn guarantees
no live writer is touching old rows, so any row with completed_ts
IS NULL and started_ts < now - 60min is safe to mark
completed_ts=now, subprocess_exit_code=-1,
errors='aborted: no completion within TTL'.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_GARDENER_SRC = Path(__file__).parent.parent.parent / "mempalace" / "memory_gardener.py"
_KG_SRC = Path(__file__).parent.parent.parent / "mempalace" / "knowledge_graph.py"


def test_v31015_kg_has_gc_stale_gardener_runs_method():
    """KnowledgeGraph MUST expose gc_stale_gardener_runs."""
    content = _KG_SRC.read_text(encoding="utf-8")
    assert "def gc_stale_gardener_runs(" in content, (
        "v3.10.15: KnowledgeGraph must define gc_stale_gardener_runs to "
        "sweep stale in-flight memory_gardener_runs rows."
    )
    assert "WHERE completed_ts IS NULL AND started_ts < ?" in content, (
        "v3.10.15: sweep WHERE clause must scope to completed_ts IS NULL AND started_ts < cutoff."
    )


def test_v31015_process_batch_calls_gc_before_start_gardener_run():
    """memory_gardener.process_batch MUST call kg.gc_stale_gardener_runs
    BEFORE kg.start_gardener_run so the sweep happens under the
    kernel-flock single-writer invariant."""
    content = _GARDENER_SRC.read_text(encoding="utf-8")
    assert "kg.gc_stale_gardener_runs(ttl_minutes=60)" in content, (
        "v3.10.15: memory_gardener.process_batch must call "
        "kg.gc_stale_gardener_runs(ttl_minutes=60) before opening a "
        "new gardener run."
    )
    # Sweep call MUST appear before start_gardener_run in source order.
    sweep_idx = content.find("kg.gc_stale_gardener_runs(ttl_minutes=60)")
    start_idx = content.find("kg.start_gardener_run(gardener_model=model)")
    assert sweep_idx != -1 and start_idx != -1
    assert sweep_idx < start_idx, (
        "v3.10.15: gc_stale_gardener_runs must be called BEFORE "
        "start_gardener_run so stale rows are swept before the new "
        "run is opened."
    )


def test_v31015_gc_resolves_stale_runs_functional(tmp_path):
    """Functional regression: seed a stale in-flight row (started 6h
    ago, completed_ts NULL) plus a fresh in-flight row (started 5min
    ago). After gc_stale_gardener_runs(ttl_minutes=60) only the stale
    row gets swept."""
    from mempalace.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
    conn = kg._conn()

    stale_ts = (datetime.now() - timedelta(hours=6)).isoformat(timespec="seconds")
    fresh_ts = (datetime.now() - timedelta(minutes=5)).isoformat(timespec="seconds")
    with conn:
        conn.execute(
            "INSERT INTO memory_gardener_runs (started_ts, gardener_model) "
            "VALUES (?, 'test-stale')",
            (stale_ts,),
        )
        conn.execute(
            "INSERT INTO memory_gardener_runs (started_ts, gardener_model) "
            "VALUES (?, 'test-fresh')",
            (fresh_ts,),
        )

    # Pre-sweep both rows are in-flight.
    in_flight_pre = conn.execute(
        "SELECT COUNT(*) FROM memory_gardener_runs WHERE completed_ts IS NULL"
    ).fetchone()[0]
    assert in_flight_pre == 2

    swept = kg.gc_stale_gardener_runs(ttl_minutes=60)
    assert swept == 1, (
        f"v3.10.15: gc_stale_gardener_runs should sweep exactly 1 stale row, swept {swept}"
    )

    # Stale row now completed with subprocess_exit_code=-1 + TTL error.
    rows = list(
        conn.execute(
            "SELECT gardener_model, completed_ts, subprocess_exit_code, errors "
            "FROM memory_gardener_runs ORDER BY id"
        )
    )
    stale_row = next(r for r in rows if r[0] == "test-stale")
    fresh_row = next(r for r in rows if r[0] == "test-fresh")
    assert stale_row[1] is not None, "stale row must have completed_ts set"
    assert stale_row[2] == -1, f"stale row must carry exit_code=-1, got {stale_row[2]}"
    assert "aborted: no completion within TTL" in (stale_row[3] or ""), (
        f"stale row errors must mention TTL, got {stale_row[3]!r}"
    )
    assert fresh_row[1] is None, "fresh row must remain in-flight (completed_ts NULL)"


def test_v31015_gc_idempotent_on_already_swept(tmp_path):
    """Calling gc_stale_gardener_runs twice in a row must not touch
    rows already swept the first time."""
    from mempalace.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
    conn = kg._conn()
    stale_ts = (datetime.now() - timedelta(hours=6)).isoformat(timespec="seconds")
    with conn:
        conn.execute(
            "INSERT INTO memory_gardener_runs (started_ts, gardener_model) "
            "VALUES (?, 'test-stale')",
            (stale_ts,),
        )
    assert kg.gc_stale_gardener_runs(ttl_minutes=60) == 1
    # Second call: nothing left to sweep.
    assert kg.gc_stale_gardener_runs(ttl_minutes=60) == 0


def test_v31015_gc_disabled_when_ttl_zero(tmp_path):
    """ttl_minutes=0 disables the sweep (safety knob)."""
    from mempalace.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
    conn = kg._conn()
    stale_ts = (datetime.now() - timedelta(hours=6)).isoformat(timespec="seconds")
    with conn:
        conn.execute(
            "INSERT INTO memory_gardener_runs (started_ts, gardener_model) "
            "VALUES (?, 'test-stale')",
            (stale_ts,),
        )
    assert kg.gc_stale_gardener_runs(ttl_minutes=0) == 0
    in_flight = conn.execute(
        "SELECT COUNT(*) FROM memory_gardener_runs WHERE completed_ts IS NULL"
    ).fetchone()[0]
    assert in_flight == 1, "ttl=0 must be a no-op"
