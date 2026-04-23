"""
test_memory_gardener.py — End-to-end coverage for the gardener.

The gardener reads pending flags from memory_flags, invokes a Claude
Code subprocess to investigate, parses a structured JSON results
line, applies resolutions via kg.mark_flag_resolved, and logs the
run to memory_gardener_runs. We inject a fake subprocess_runner so
tests never shell out.

Scope:
  * build_flag_prompt shape (per-flag sections, counters).
  * parse_results_line pulls the LAST valid JSON line from stdout,
    ignores intermediate tool-use logs, handles malformed output.
  * process_batch end-to-end:
      - resolved flags land in memory_gardener_runs counters.
      - Unrecognized-in-response flags get attempted_count bumped.
      - Empty queue → early-return, no run row written.
      - Bad resolution codes filtered out.
      - Subprocess errors captured in run row.
  * maybe_trigger_from_finalize:
      - Below threshold: no-op, returns False.
      - At threshold: spawn attempted (we monkeypatch Popen).
"""

from __future__ import annotations

import json
from unittest.mock import patch


from mempalace.memory_gardener import (
    SubprocessResult,
    build_flag_prompt,
    maybe_trigger_from_finalize,
    parse_results_line,
    process_batch,
)


# ─────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────


class TestBuildFlagPrompt:
    def test_renders_each_flag_section(self):
        flags = [
            {"id": 1, "kind": "duplicate_pair", "memory_ids": ["a", "b"], "detail": "same fact"},
            {"id": 2, "kind": "orphan", "memory_ids": ["x"], "detail": "no entities"},
        ]
        prompt = build_flag_prompt(flags)
        assert "FLAG #1" in prompt and "FLAG #2" in prompt
        assert "duplicate_pair" in prompt
        assert "orphan" in prompt
        assert "same fact" in prompt
        assert '["a", "b"]' in prompt

    def test_includes_prior_attempts_when_bumped(self):
        flags = [
            {"id": 1, "kind": "stale", "memory_ids": ["x"], "detail": "d", "attempted_count": 2},
        ]
        prompt = build_flag_prompt(flags)
        assert "prior_attempts: 2" in prompt

    def test_missing_detail_renders_placeholder(self):
        flags = [{"id": 9, "kind": "stale", "memory_ids": ["x"], "detail": ""}]
        prompt = build_flag_prompt(flags)
        assert "(no detail provided)" in prompt


# ─────────────────────────────────────────────────────────────────────
# Result parser
# ─────────────────────────────────────────────────────────────────────


class TestParseResultsLine:
    def test_parses_trailing_json_line(self):
        stdout = (
            "some log line\n"
            '{"intermediate": true}\n'
            '{"results": [{"flag_id": 1, "resolution": "merged", "note": "ok"}]}\n'
        )
        results = parse_results_line(stdout)
        assert len(results) == 1
        assert results[0]["flag_id"] == 1

    def test_prefers_last_results_object_when_multiple(self):
        stdout = (
            '{"results": [{"flag_id": 1, "resolution": "deferred", "note": "draft"}]}\n'
            '{"results": [{"flag_id": 1, "resolution": "merged", "note": "final"}]}\n'
        )
        results = parse_results_line(stdout)
        assert results[0]["resolution"] == "merged"

    def test_whole_stdout_json_fallback(self):
        stdout = '{"results": [{"flag_id": 7, "resolution": "no_action", "note": "fp"}]}'
        results = parse_results_line(stdout)
        assert len(results) == 1

    def test_empty_stdout_returns_empty(self):
        assert parse_results_line("") == []

    def test_no_results_key_returns_empty(self):
        stdout = '{"something_else": 1}\n'
        assert parse_results_line(stdout) == []

    def test_malformed_json_returns_empty(self):
        stdout = "this is not json\n{bad\n"
        assert parse_results_line(stdout) == []


# ─────────────────────────────────────────────────────────────────────
# Batch processor — end-to-end with injected runner
# ─────────────────────────────────────────────────────────────────────


def _seed_flags(kg, count: int, kind: str = "orphan", base_ctx: str = "ctx_g"):
    for i in range(count):
        kg.record_memory_flags(
            [
                {
                    "kind": kind,
                    "memory_ids": [f"mem_{i}"],
                    "detail": f"flag {i}",
                    "context_id": f"{base_ctx}_{i}",
                }
            ]
        )


def _runner_with_results(results_by_flag: dict, *, exit_code: int = 0, stderr: str = ""):
    """Return a subprocess_runner that replies with the given
    flag_id → resolution mapping."""

    def runner(*, system_prompt, user_prompt, model):
        # Map the seeded flag ids from the user_prompt back to the ids
        # the test wants to target. Caller passes results keyed by
        # INDEX order 0..N-1; we translate here by scanning FLAG #<id>
        # occurrences in the prompt.
        flag_order = []
        for line in user_prompt.splitlines():
            if line.startswith("FLAG #"):
                try:
                    flag_order.append(int(line.split("#", 1)[1].strip()))
                except ValueError:
                    pass
        results = []
        for i, fid in enumerate(flag_order):
            if i in results_by_flag:
                results.append(
                    {
                        "flag_id": fid,
                        "resolution": results_by_flag[i],
                        "note": f"test note {i}",
                    }
                )
        body = json.dumps({"results": results})
        return SubprocessResult(stdout=body + "\n", stderr=stderr, exit_code=exit_code)

    return runner


class TestProcessBatch:
    def test_empty_queue_early_returns(self, kg):
        out = process_batch(kg, subprocess_runner=lambda **kw: SubprocessResult("", "", 0))
        assert out["run_id"] is None
        assert out["flag_ids"] == []

    def test_happy_path_all_resolved(self, kg):
        _seed_flags(kg, 3)
        runner = _runner_with_results({0: "merged", 1: "invalidated", 2: "linked"})
        out = process_batch(kg, batch_size=5, subprocess_runner=runner)
        assert out["run_id"] is not None
        assert len(out["flag_ids"]) == 3
        assert out["counters"]["merges"] == 1
        assert out["counters"]["invalidations"] == 1
        assert out["counters"]["links_created"] == 1
        # All flags resolved → pending queue is empty.
        assert kg.list_pending_flags() == []
        # Run row recorded.
        conn = kg._conn()
        row = conn.execute(
            "SELECT flags_processed, merges, invalidations, links_created, "
            "subprocess_exit_code FROM memory_gardener_runs WHERE id = ?",
            (out["run_id"],),
        ).fetchone()
        assert row["flags_processed"] == 3
        assert row["merges"] == 1
        assert row["subprocess_exit_code"] == 0

    def test_unresolved_flag_bumps_attempted_count(self, kg):
        """Subprocess returns 1 result for 3 flags → the 2 unresolved
        get attempted_count bumped so they don't live forever."""
        _seed_flags(kg, 3)
        runner = _runner_with_results({0: "merged"})
        process_batch(kg, batch_size=5, subprocess_runner=runner)
        # Pending-list remaining: the 2 unresolved flags, each with
        # attempted_count = 1.
        remaining = kg.list_pending_flags()
        assert len(remaining) == 2
        assert all(r["attempted_count"] == 1 for r in remaining)

    def test_bad_resolution_codes_filtered(self, kg):
        _seed_flags(kg, 2)
        runner = _runner_with_results({0: "made_up", 1: "merged"})
        out = process_batch(kg, batch_size=5, subprocess_runner=runner)
        # Only the merged one counts; the bad code is skipped.
        assert out["counters"].get("merges") == 1
        # Unresolved flag bumped.
        remaining = kg.list_pending_flags()
        assert len(remaining) == 1
        assert remaining[0]["attempted_count"] == 1

    def test_subprocess_error_recorded(self, kg):
        _seed_flags(kg, 1)
        runner = _runner_with_results({}, exit_code=127, stderr="claude not found")
        out = process_batch(kg, batch_size=5, subprocess_runner=runner)
        assert out["exit_code"] == 127
        assert "not found" in out["errors"]
        conn = kg._conn()
        row = conn.execute(
            "SELECT errors, subprocess_exit_code FROM memory_gardener_runs WHERE id = ?",
            (out["run_id"],),
        ).fetchone()
        assert row["subprocess_exit_code"] == 127
        assert "not found" in row["errors"]

    def test_batch_size_caps_pull(self, kg):
        _seed_flags(kg, 10)
        runner = _runner_with_results({i: "deferred" for i in range(10)})
        out = process_batch(kg, batch_size=3, subprocess_runner=runner)
        assert len(out["flag_ids"]) == 3


# ─────────────────────────────────────────────────────────────────────
# Finalize trigger
# ─────────────────────────────────────────────────────────────────────


class TestMaybeTriggerFromFinalize:
    def test_below_threshold_no_spawn(self, kg):
        _seed_flags(kg, 3)  # below default 5
        with patch("mempalace.memory_gardener.subprocess.Popen") as mp:
            spawned = maybe_trigger_from_finalize(kg)
        assert spawned is False
        assert mp.call_count == 0

    def test_at_threshold_spawns(self, kg):
        _seed_flags(kg, 5)  # exactly at threshold
        with patch("mempalace.memory_gardener.subprocess.Popen") as mp:
            spawned = maybe_trigger_from_finalize(kg)
        assert spawned is True
        assert mp.call_count == 1
        # Verify the spawned command targets the gardener CLI.
        call_args = mp.call_args[0][0]  # positional args list
        assert "gardener" in call_args
        assert "process" in call_args

    def test_kg_exception_returns_false(self):
        class BrokenKG:
            def count_pending_flags(self):
                raise RuntimeError("db locked")

        with patch("mempalace.memory_gardener.subprocess.Popen") as mp:
            spawned = maybe_trigger_from_finalize(BrokenKG())
        assert spawned is False
        assert mp.call_count == 0
