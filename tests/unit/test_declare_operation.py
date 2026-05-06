"""
test_declare_operation.py -- regression tests for the 2026-04-20
operation-level declaration feature.

Design summary (from intent.py tool_declare_operation docstring):
  - Agent calls mempalace_declare_operation(tool, queries, keywords)
    BEFORE each non-carve-out tool call.
  - Cue lands in active_intent['pending_operation_cues'] (list, supports
    parallel tool batches).
  - PreToolUse hook pops matching entry, uses it as retrieval cue (same
    _run_local_retrieval pipeline), persists shortened list.
  - Env MEMPALACE_REQUIRE_DECLARE_OPERATION flips the hook from "use cue
    if present, else auto-build" to "deny if no matching cue".
  - TTL OPERATION_CUE_TTL_SECONDS (300s) expires stale cues on consume.
  - Parallel-race wait: hook polls disk for up to
    OPERATION_CUE_WAIT_TIMEOUT_SECONDS (5s) before denying in strict mode.
"""

from __future__ import annotations

import pytest
import json


from mempalace import hooks_cli, intent as intent_mod


# ─────────────────────────────────────────────────────────────────────
# Pure-function tests -- no fixtures required
# ─────────────────────────────────────────────────────────────────────


class TestPruneExpiredCues:
    def test_empty_list_returns_empty(self):
        live, expired = hooks_cli._prune_expired_cues([])
        assert live == []
        assert expired == []

    def test_none_treated_as_empty(self):
        live, expired = hooks_cli._prune_expired_cues(None)
        assert live == []
        assert expired == []

    def test_fresh_cue_kept(self):
        from datetime import datetime

        now_iso = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        cues = [{"tool": "Read", "declared_at_ts": now_iso, "queries": ["x"]}]
        live, expired = hooks_cli._prune_expired_cues(cues)
        assert len(live) == 1
        assert expired == []

    def test_stale_cue_expired(self):
        stale = {
            "tool": "Read",
            "declared_at_ts": "2020-01-01T00:00:00Z",
            "queries": ["x"],
        }
        live, expired = hooks_cli._prune_expired_cues([stale])
        assert live == []
        assert len(expired) == 1

    def test_invalid_timestamp_kept_fail_open(self):
        cues = [{"tool": "Read", "declared_at_ts": "not-a-timestamp"}]
        live, expired = hooks_cli._prune_expired_cues(cues)
        assert len(live) == 1
        assert expired == []

    def test_missing_timestamp_kept_fail_open(self):
        cues = [{"tool": "Read", "queries": ["x"]}]
        live, expired = hooks_cli._prune_expired_cues(cues)
        assert len(live) == 1
        assert expired == []


class TestConsumePendingOperationCue:
    def _make_intent(self, cues):
        return {"pending_operation_cues": list(cues)}

    def _fresh(self, tool):
        from datetime import datetime

        return {
            "tool": tool,
            "queries": [f"query for {tool} a", f"query for {tool} b"],
            "keywords": [f"kw-{tool}-1", f"kw-{tool}-2"],
            "declared_at_ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }

    def test_no_cues_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)
        intent = self._make_intent([])
        cue, expired = hooks_cli._consume_pending_operation_cue("sid-test", intent, "Read")
        assert cue is None
        assert expired == 0

    def test_matching_tool_pops_first(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)
        state_path = tmp_path / "active_intent_sid-test.json"
        state_path.write_text(json.dumps({"intent_id": "x"}), encoding="utf-8")

        cues = [self._fresh("Read"), self._fresh("Grep"), self._fresh("Read")]
        intent = self._make_intent(cues)

        cue, expired = hooks_cli._consume_pending_operation_cue("sid-test", intent, "Read")
        assert cue is not None
        assert cue["queries"][0] == "query for Read a"
        assert expired == 0
        remaining = intent["pending_operation_cues"]
        assert [c["tool"] for c in remaining] == ["Grep", "Read"]

    def test_mismatched_tool_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)
        intent = self._make_intent([self._fresh("Grep")])
        cue, expired = hooks_cli._consume_pending_operation_cue("sid-test", intent, "Read")
        assert cue is None
        assert expired == 0

    def test_stale_cues_pruned_and_counted(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)
        state_path = tmp_path / "active_intent_sid-test.json"
        state_path.write_text(json.dumps({"intent_id": "x"}), encoding="utf-8")

        stale = {
            "tool": "Read",
            "declared_at_ts": "2020-01-01T00:00:00Z",
            "queries": ["x"],
            "keywords": ["y"],
        }
        fresh = self._fresh("Read")
        intent = self._make_intent([stale, fresh])

        cue, expired = hooks_cli._consume_pending_operation_cue("sid-test", intent, "Read")
        assert expired == 1
        assert cue is not None
        assert cue["queries"] == fresh["queries"]

    def test_persists_shortened_list_to_disk(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)
        state_path = tmp_path / "active_intent_sid-test.json"
        state_path.write_text(
            json.dumps({"intent_id": "x", "pending_operation_cues": []}),
            encoding="utf-8",
        )

        cues = [self._fresh("Read"), self._fresh("Grep")]
        intent = self._make_intent(cues)
        hooks_cli._consume_pending_operation_cue("sid-test", intent, "Read")

        on_disk = json.loads(state_path.read_text(encoding="utf-8"))
        assert [c["tool"] for c in on_disk["pending_operation_cues"]] == ["Grep"]


class TestWaitForMatchingPendingCue:
    def test_already_present_returns_immediately(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)
        state_path = tmp_path / "active_intent_sid-test.json"
        state_path.write_text(
            json.dumps(
                {
                    "intent_id": "x",
                    "pending_operation_cues": [{"tool": "Read", "queries": ["q"]}],
                }
            ),
            encoding="utf-8",
        )
        intent = {}
        found = hooks_cli._wait_for_matching_pending_cue("sid-test", "Read", intent)
        assert found is True
        assert intent.get("pending_operation_cues"), (
            "wait helper should mirror disk into intent dict for the caller"
        )

    def test_no_match_times_out_fast_when_shortcircuited(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hooks_cli, "OPERATION_CUE_WAIT_TIMEOUT_SECONDS", 0.4)
        monkeypatch.setattr(hooks_cli, "OPERATION_CUE_WAIT_POLL_INTERVAL_SECONDS", 0.1)
        monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)
        state_path = tmp_path / "active_intent_sid-test.json"
        state_path.write_text(
            json.dumps(
                {
                    "intent_id": "x",
                    "pending_operation_cues": [{"tool": "Grep", "queries": ["q"]}],
                }
            ),
            encoding="utf-8",
        )
        found = hooks_cli._wait_for_matching_pending_cue("sid-test", "Read", {})
        assert found is False


# ─────────────────────────────────────────────────────────────────────
# MCP-side handler tests
# ─────────────────────────────────────────────────────────────────────


def _patch_state(monkeypatch, tmp_path):
    """Minimal MCP-side state setup for tool_declare_operation tests."""
    from mempalace import mcp_server

    monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)

    monkeypatch.setattr(mcp_server, "_require_sid", lambda **kwargs: None)
    monkeypatch.setattr(mcp_server, "_require_agent", lambda *a, **k: None)
    # Slice 12 leak fix (Adrian directive 2026-05-05): the per-op
    # state-delta gate at intent.py:3590 now correctly fires when
    # state_deltas is None (the gate's coverage set no longer crashes
    # silently). These tests exercise declare_operation's CUE shape +
    # carve-outs, not the state-delta gate; turn the gate off here via
    # the documented escape hatch so the tests stay focused. monkeypatch
    # auto-restores the env on teardown.
    monkeypatch.setenv("MEMPALACE_STATE_DELTA_DISABLED", "1")
    mcp_server._STATE.active_intent = {
        "intent_id": "intent_test_x",
        "intent_type": "modify",
        "slots": {},
        "effective_permissions": [{"tool": "Read", "scope": "*"}],
        "accessed_memory_ids": set(),
    }
    mcp_server._STATE.session_id = "sid-op-test"
    monkeypatch.setattr(intent_mod, "_sync_from_disk", lambda: None)
    monkeypatch.setattr(intent_mod, "_persist_active_intent", lambda: None)
    monkeypatch.setattr(
        hooks_cli,
        "_run_local_retrieval",
        lambda cue, accessed, top_k: ([], None),
    )
    return mcp_server


class TestToolDeclareOperation:
    def test_rejects_empty_tool(self, monkeypatch, tmp_path):
        _patch_state(monkeypatch, tmp_path)
        result = intent_mod.tool_declare_operation(
            tool="",
            args_summary="OP {params}",
            context={
                "queries": ["a", "b"],
                "keywords": ["k1", "k2"],
                "entities": ["x"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="a",
        )
        assert result["success"] is False
        assert "tool" in result["error"].lower()

    def test_rejects_carve_out_tool(self, monkeypatch, tmp_path):
        _patch_state(monkeypatch, tmp_path)
        result = intent_mod.tool_declare_operation(
            tool="TodoWrite",
            args_summary="TodoWrite {params}",
            context={
                "queries": ["a", "b"],
                "keywords": ["k1", "k2"],
                "entities": ["x"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="a",
        )
        assert result["success"] is False
        assert "does not require" in result["error"]

    def test_rejects_mempalace_mcp_tool(self, monkeypatch, tmp_path):
        _patch_state(monkeypatch, tmp_path)
        result = intent_mod.tool_declare_operation(
            tool="mcp__plugin_mempalace__mempalace_kg_add",
            args_summary="mcp__plugin_mempalace__mempalace_kg_add {params}",
            context={
                "queries": ["a", "b"],
                "keywords": ["k1", "k2"],
                "entities": ["x"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="a",
        )
        assert result["success"] is False

    def test_rejects_short_queries(self, monkeypatch, tmp_path):
        _patch_state(monkeypatch, tmp_path)
        result = intent_mod.tool_declare_operation(
            tool="Read",
            args_summary="Read {params}",
            context={
                "queries": ["only one"],
                "keywords": ["k1", "k2"],
                "entities": ["x"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="a",
        )
        assert result["success"] is False
        assert "quer" in result["error"].lower()

    def test_rejects_short_keywords(self, monkeypatch, tmp_path):
        _patch_state(monkeypatch, tmp_path)
        result = intent_mod.tool_declare_operation(
            tool="Read",
            args_summary="Read {params}",
            context={
                "queries": ["a", "b"],
                "keywords": ["one"],
                "entities": ["x"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="a",
        )
        assert result["success"] is False
        assert "keyword" in result["error"].lower()

    def test_rejects_non_string_query(self, monkeypatch, tmp_path):
        _patch_state(monkeypatch, tmp_path)
        result = intent_mod.tool_declare_operation(
            tool="Read",
            args_summary="Read {params}",
            context={
                "queries": ["ok", 42],
                "keywords": ["k1", "k2"],
                "entities": ["x"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="a",
        )
        assert result["success"] is False

    def test_success_stores_cue(self, monkeypatch, tmp_path):
        mcp = _patch_state(monkeypatch, tmp_path)
        result = intent_mod.tool_declare_operation(
            tool="Read",
            args_summary="Read {params}",
            context={
                "queries": ["verify finalize contract", "check mandatory finalize"],
                "keywords": ["finalize_intent", "memory_feedback"],
                "entities": ["finalize_intent", "memory_feedback"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="ga_agent",
        )
        assert result["success"] is True
        cues = mcp._STATE.active_intent.get("pending_operation_cues")
        assert isinstance(cues, list)
        assert len(cues) == 1
        assert cues[0]["tool"] == "Read"
        assert cues[0]["queries"] == [
            "verify finalize contract",
            "check mandatory finalize",
        ]
        assert cues[0]["keywords"] == [
            "finalize_intent",
            "memory_feedback",
        ]
        assert cues[0].get("declared_at_ts"), "declared_at_ts drives TTL expiry"

    def test_multiple_declarations_append(self, monkeypatch, tmp_path):
        mcp = _patch_state(monkeypatch, tmp_path)
        intent_mod.tool_declare_operation(
            tool="Read",
            args_summary="Read {params}",
            context={
                "queries": ["a", "b"],
                "keywords": ["k1", "k2"],
                "entities": ["x"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="ga_agent",
        )
        intent_mod.tool_declare_operation(
            tool="Grep",
            args_summary="Grep {params}",
            context={
                "queries": ["c", "d"],
                "keywords": ["k3", "k4"],
                "entities": ["x"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="ga_agent",
        )
        cues = mcp._STATE.active_intent["pending_operation_cues"]
        assert [c["tool"] for c in cues] == ["Read", "Grep"]

    def test_no_active_intent_rejects(self, monkeypatch, tmp_path):
        mcp = _patch_state(monkeypatch, tmp_path)
        mcp._STATE.active_intent = None
        result = intent_mod.tool_declare_operation(
            tool="Read",
            args_summary="Read {params}",
            context={
                "queries": ["a", "b"],
                "keywords": ["k1", "k2"],
                "entities": ["x"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="ga_agent",
        )
        assert result["success"] is False
        assert "No active intent" in result["error"]


class TestDeclareOperationDedupFilter:
    """Cross-phase dedup: every memory surfaced so far in this intent
    must be excluded from the operation-time retrieval filter. That
    includes declare_intent surfaces (injected_memory_ids), prior
    declare_operation surfaces (accessed_memory_ids), and kg_search
    surfaces (also accessed_memory_ids). A regression here silently
    re-surfaces intent-time hits under narrower operation cues."""

    def _patch_capturing(self, monkeypatch, tmp_path):
        from mempalace import mcp_server

        monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)
        monkeypatch.setattr(mcp_server, "_require_sid", lambda **kwargs: None)
        monkeypatch.setattr(mcp_server, "_require_agent", lambda *a, **k: None)
        # Slice 12 leak fix (Adrian directive 2026-05-05): same as
        # _patch_state's setenv -- the per-op state-delta gate now
        # correctly fires when state_deltas is None. These dedup-filter
        # tests don't exercise gate B; turn it off via the documented
        # escape hatch so they stay focused.
        monkeypatch.setenv("MEMPALACE_STATE_DELTA_DISABLED", "1")
        mcp_server._STATE.active_intent = {
            "intent_id": "intent_test_dedup",
            "intent_type": "modify",
            "slots": {},
            "effective_permissions": [{"tool": "Read", "scope": "*"}],
            "accessed_memory_ids": set(),
            "injected_memory_ids": set(),
        }
        mcp_server._STATE.session_id = "sid-op-dedup"
        monkeypatch.setattr(intent_mod, "_sync_from_disk", lambda: None)
        monkeypatch.setattr(intent_mod, "_persist_active_intent", lambda: None)

        captured: dict = {}

        def fake_retrieval(cue, accessed, top_k):
            captured["accessed"] = set(accessed or [])
            return ([], None)

        monkeypatch.setattr(hooks_cli, "_run_local_retrieval", fake_retrieval)
        return mcp_server, captured

    def test_injected_memory_ids_included_in_accessed_filter(self, monkeypatch, tmp_path):
        """declare_operation MUST exclude declare_intent's surfaces.

        Regression: before the 2026-04-23 union fix, the dedup filter
        was built from accessed_memory_ids only, so memories surfaced
        by declare_intent (stored under injected_memory_ids) were
        re-surfaced by every subsequent declare_operation in the same
        intent.
        """
        mcp, captured = self._patch_capturing(monkeypatch, tmp_path)
        mcp._STATE.active_intent["injected_memory_ids"] = {
            "mem_from_intent_1",
            "mem_from_intent_2",
        }

        result = intent_mod.tool_declare_operation(
            tool="Read",
            args_summary="Read {params}",
            context={
                "queries": ["fresh operation cue", "different angle"],
                "keywords": ["kw1", "kw2"],
                "entities": ["x"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="ga_agent",
        )
        assert result["success"] is True, result
        assert "mem_from_intent_1" in captured["accessed"], (
            "declare_intent's surfaced memories must be in the dedup filter"
        )
        assert "mem_from_intent_2" in captured["accessed"]

    def test_accessed_and_injected_unioned_in_filter(self, monkeypatch, tmp_path):
        """Both lists must participate in the dedup filter."""
        mcp, captured = self._patch_capturing(monkeypatch, tmp_path)
        mcp._STATE.active_intent["injected_memory_ids"] = {"injected_only"}
        mcp._STATE.active_intent["accessed_memory_ids"] = {"accessed_only"}

        intent_mod.tool_declare_operation(
            tool="Read",
            args_summary="Read {params}",
            context={
                "queries": ["query one", "query two"],
                "keywords": ["k1", "k2"],
                "entities": ["x"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="ga_agent",
        )
        assert {"injected_only", "accessed_only"} <= captured["accessed"]

    def test_prior_operation_surfaces_filtered_next_operation(self, monkeypatch, tmp_path):
        """Op #1's hit must be in op #2's dedup filter."""
        mcp, _captured = self._patch_capturing(monkeypatch, tmp_path)

        accessed_captured = {"history": []}

        def fake_retrieval_seq(cue, accessed, top_k):
            accessed_captured["history"].append(set(accessed or []))
            if len(accessed_captured["history"]) == 1:
                return (
                    [{"id": "mem_op1_hit", "score": 0.9, "preview": "p"}],
                    None,
                )
            return ([], None)

        monkeypatch.setattr(hooks_cli, "_run_local_retrieval", fake_retrieval_seq)

        intent_mod.tool_declare_operation(
            tool="Read",
            args_summary="Read {params}",
            context={
                "queries": ["a", "b"],
                "keywords": ["k1", "k2"],
                "entities": ["x"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="ga_agent",
        )
        intent_mod.tool_declare_operation(
            tool="Grep",
            args_summary="Grep {params}",
            context={
                "queries": ["c", "d"],
                "keywords": ["k3", "k4"],
                "entities": ["x"],
                "summary": {
                    "what": "test fixture context",
                    "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                    "scope": "tests",
                },
            },
            agent="ga_agent",
        )

        assert "mem_op1_hit" in accessed_captured["history"][1], (
            "declare_operation #2 must filter out what #1 surfaced"
        )


pytestmark = pytest.mark.unit
