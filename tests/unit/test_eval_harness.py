"""Tests for mempalace.eval_harness -- P3 retrieval-quality reports.

The harness is a pure reader over the JSONL telemetry written by
tool_kg_search and tool_finalize_intent. Fixtures write synthetic
traces; assertions check the aggregated shapes.
"""

from __future__ import annotations

import pytest
import json

from mempalace import eval_harness


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_context_reuse_rate_with_mixed_reuse(tmp_path):
    log = tmp_path / "search_log.jsonl"
    _write_jsonl(
        log,
        [
            {"ts": "2026-04-22T00:00:00+00:00", "reused": True},
            {"ts": "2026-04-22T00:00:01+00:00", "reused": True},
            {"ts": "2026-04-22T00:00:02+00:00", "reused": False},
            {"ts": "2026-04-22T00:00:03+00:00", "reused": False},
            {"ts": "2026-04-22T00:00:04+00:00", "reused": False},
        ],
    )
    payload = eval_harness.context_reuse_rate(search_log_path=log)
    assert payload["total_searches"] == 5
    assert payload["reused"] == 2
    assert payload["rate"] == 0.4


def test_context_reuse_rate_empty_file(tmp_path):
    log = tmp_path / "search_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("")
    payload = eval_harness.context_reuse_rate(search_log_path=log)
    assert payload == {"total_searches": 0, "reused": 0, "rate": 0.0}


def test_context_reuse_rate_nonexistent_file(tmp_path):
    log = tmp_path / "nonexistent.jsonl"
    payload = eval_harness.context_reuse_rate(search_log_path=log)
    assert payload["total_searches"] == 0


def test_per_channel_contribution_aggregates(tmp_path):
    log = tmp_path / "search_log.jsonl"
    _write_jsonl(
        log,
        [
            {
                "ts": "2026-04-22T00:00:00+00:00",
                "per_channel_hits": {"cosine": 3, "context": 1},
            },
            {
                "ts": "2026-04-22T00:00:01+00:00",
                "per_channel_hits": {"cosine": 2, "graph": 1, "context": 2},
            },
            {
                "ts": "2026-04-22T00:00:02+00:00",
                "per_channel_hits": {"cosine": 1, "keyword": 1},
            },
        ],
    )
    payload = eval_harness.per_channel_contribution(search_log_path=log)
    assert payload["total_searches"] == 3
    assert payload["hits_per_channel"]["cosine"] == 6
    assert payload["hits_per_channel"]["graph"] == 1
    assert payload["hits_per_channel"]["keyword"] == 1
    assert payload["hits_per_channel"]["context"] == 3
    assert payload["calls_per_channel"]["cosine"] == 3
    assert payload["calls_per_channel"]["context"] == 2


def test_summary_report_end_to_end(tmp_path):
    search_log = tmp_path / "search_log.jsonl"
    finalize_log = tmp_path / "finalize_log.jsonl"
    _write_jsonl(
        search_log,
        [
            {
                "ts": "2026-04-22T00:00:00+00:00",
                "reused": True,
                "per_channel_hits": {"cosine": 2, "context": 1},
            },
            {
                "ts": "2026-04-22T00:00:01+00:00",
                "reused": False,
                "per_channel_hits": {"cosine": 3},
            },
        ],
    )
    _write_jsonl(
        finalize_log,
        [
            {
                "ts": "2026-04-22T00:00:02+00:00",
                "intent_id": "intent_a",
                "memories_rated": 4,
                "contexts_used": ["ctx_x", "ctx_y"],
            },
            {
                "ts": "2026-04-22T00:00:03+00:00",
                "intent_id": "intent_b",
                "memories_rated": 2,
                "contexts_used": ["ctx_x"],
            },
        ],
    )

    report = eval_harness.summary_report(search_log_path=search_log, finalize_log_path=finalize_log)
    assert report["reuse"]["rate"] == 0.5
    assert report["reuse"]["total_searches"] == 2
    assert report["channels"]["hits_per_channel"]["cosine"] == 5
    assert report["finalize"]["total_finalizes"] == 2
    assert report["finalize"]["memories_rated"] == 6
    assert report["finalize"]["unique_contexts_used"] == 2
    # ctx_x appears in both finalizes → top with count 2.
    top = dict(report["finalize"]["top_contexts"])
    assert top.get("ctx_x") == 2
    assert top.get("ctx_y") == 1


def test_format_report_smoke(tmp_path):
    report = eval_harness.summary_report(
        search_log_path=tmp_path / "nonexistent.jsonl",
        finalize_log_path=tmp_path / "nothing_either.jsonl",
    )
    text = eval_harness.format_report(report)
    assert "mempalace-eval report" in text
    assert "context reuse rate" in text


def test_malformed_lines_are_skipped(tmp_path):
    log = tmp_path / "search_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        '{"ts":"2026-04-22T00:00:00Z","reused":true}\n'
        "not valid json at all\n"
        '{"ts":"2026-04-22T00:00:01Z","reused":false}\n'
        "\n"
    )
    payload = eval_harness.context_reuse_rate(search_log_path=log)
    assert payload["total_searches"] == 2


pytestmark = pytest.mark.unit
