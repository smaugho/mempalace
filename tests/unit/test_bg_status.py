"""Unit tests for mempalace.bg_status.tool_bg_status.

v3.7.0 Slice 0 (Adrian directive 2026-05-16 -- Option 3 architecture
visibility requirement). The tool tails ~/.mempalace/hook_state/ telemetry
streams. Tests monkeypatch ``bg_status._telemetry_dir`` to a tmp dir so
the unit lane stays hermetic (no global filesystem touch).
"""

from __future__ import annotations

import json

import pytest

from mempalace import bg_status

pytestmark = pytest.mark.unit


@pytest.fixture
def telem_dir(tmp_path, monkeypatch):
    """Redirect bg_status._telemetry_dir at tmp_path."""
    monkeypatch.setattr(bg_status, "_telemetry_dir", lambda: tmp_path)
    return tmp_path


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ─────────────────────────────────────────────────────────────────────
# JSONL streams
# ─────────────────────────────────────────────────────────────────────


class TestJsonlStreams:
    def test_gate_log_tail_returns_most_recent(self, telem_dir):
        rows = [{"ts": f"t{i}", "elapsed_ms": i} for i in range(10)]
        _write_jsonl(telem_dir / "gate_log.jsonl", rows)
        out = bg_status.tool_bg_status(limit=3, streams=["gate_log"])
        gate = out["streams"]["gate_log"]
        assert gate["kind"] == "jsonl"
        assert gate["exists"] is True
        assert gate["size_bytes"] > 0
        # Most recent N entries (indices 7..9 of 10).
        assert [e["elapsed_ms"] for e in gate["entries"]] == [7, 8, 9]

    def test_missing_file_returns_empty_entries(self, telem_dir):
        out = bg_status.tool_bg_status(limit=3, streams=["gate_log"])
        gate = out["streams"]["gate_log"]
        assert gate["exists"] is False
        assert gate["size_bytes"] == 0
        assert gate["entries"] == []

    def test_malformed_line_surfaces_parse_error_sentinel(self, telem_dir):
        path = telem_dir / "retrieval_log.jsonl"
        path.write_text(
            '{"ok": 1}\nnot json at all\n{"ok": 2}\n',
            encoding="utf-8",
        )
        out = bg_status.tool_bg_status(limit=10, streams=["retrieval_log"])
        ents = out["streams"]["retrieval_log"]["entries"]
        assert ents[0] == {"ok": 1}
        assert "_parse_error" in ents[1]
        assert ents[1]["_raw"] == "not json at all"
        assert ents[2] == {"ok": 2}

    def test_blank_lines_skipped(self, telem_dir):
        path = telem_dir / "gate_log.jsonl"
        path.write_text(
            '{"i": 1}\n\n   \n{"i": 2}\n',
            encoding="utf-8",
        )
        out = bg_status.tool_bg_status(limit=10, streams=["gate_log"])
        ents = out["streams"]["gate_log"]["entries"]
        assert ents == [{"i": 1}, {"i": 2}]


# ─────────────────────────────────────────────────────────────────────
# Text streams (faulthandler)
# ─────────────────────────────────────────────────────────────────────


class TestTextStreams:
    def test_faulthandler_tail_returns_recent_lines(self, telem_dir):
        path = telem_dir / "faulthandler.log"
        path.write_text(
            "\n".join(f"line {i}" for i in range(10)) + "\n",
            encoding="utf-8",
        )
        out = bg_status.tool_bg_status(limit=4, streams=["faulthandler"])
        fh = out["streams"]["faulthandler"]
        assert fh["kind"] == "text"
        assert fh["exists"] is True
        assert fh["lines"] == ["line 6", "line 7", "line 8", "line 9"]

    def test_faulthandler_missing_returns_empty(self, telem_dir):
        out = bg_status.tool_bg_status(limit=4, streams=["faulthandler"])
        fh = out["streams"]["faulthandler"]
        assert fh["exists"] is False
        assert fh["lines"] == []


# ─────────────────────────────────────────────────────────────────────
# Args + clamping + discovery
# ─────────────────────────────────────────────────────────────────────


class TestArgs:
    def test_limit_clamped_to_max(self, telem_dir):
        rows = [{"i": i} for i in range(100)]
        _write_jsonl(telem_dir / "gate_log.jsonl", rows)
        out = bg_status.tool_bg_status(limit=200, streams=["gate_log"])
        assert out["limit"] == bg_status._MAX_LIMIT
        assert len(out["streams"]["gate_log"]["entries"]) == bg_status._MAX_LIMIT

    def test_limit_clamped_to_min(self, telem_dir):
        _write_jsonl(telem_dir / "gate_log.jsonl", [{"i": 0}, {"i": 1}])
        out = bg_status.tool_bg_status(limit=0, streams=["gate_log"])
        assert out["limit"] == 1
        assert len(out["streams"]["gate_log"]["entries"]) == 1

    def test_limit_non_integer_falls_back_to_default(self, telem_dir):
        _write_jsonl(
            telem_dir / "gate_log.jsonl",
            [{"i": i} for i in range(20)],
        )
        out = bg_status.tool_bg_status(limit="banana", streams=["gate_log"])
        assert out["limit"] == bg_status._DEFAULT_LIMIT

    def test_default_includes_all_known_streams(self, telem_dir):
        out = bg_status.tool_bg_status()
        for s in (
            "gate_log",
            "state_judge_log",
            "retrieval_log",
            "feedback_auto_log",
            "mcp_io_log",
            "search_log",
            "hook_errors",
            "faulthandler",
        ):
            assert s in out["streams"], s

    def test_unknown_stream_marked_unknown(self, telem_dir):
        out = bg_status.tool_bg_status(streams=["bogus_stream"])
        assert out["streams"]["bogus_stream"]["kind"] == "unknown"
        assert out["streams"]["bogus_stream"]["status"] == "unknown_stream"

    def test_streams_non_list_yields_empty(self, telem_dir):
        out = bg_status.tool_bg_status(streams="gate_log")  # string, not list
        # Non-list -> names=[]; only base_dir+limit+empty streams dict.
        assert out["streams"] == {}

    def test_base_dir_in_response(self, telem_dir):
        out = bg_status.tool_bg_status()
        assert out["base_dir"] == str(telem_dir)


# ─────────────────────────────────────────────────────────────────────
# MCP TOOLS dispatch wiring
# ─────────────────────────────────────────────────────────────────────


class TestMcpRegistration:
    def test_bg_status_registered_in_TOOLS(self):
        """mempalace_bg_status must surface via the MCP TOOLS dict so
        clients (and the list_tools advertisement) actually see it."""
        from mempalace import mcp_server

        assert "mempalace_bg_status" in mcp_server.TOOLS
        entry = mcp_server.TOOLS["mempalace_bg_status"]
        assert callable(entry["handler"])
        assert entry["input_schema"]["type"] == "object"
        props = entry["input_schema"]["properties"]
        assert "limit" in props
        assert "streams" in props
        assert props["limit"]["maximum"] == 50
