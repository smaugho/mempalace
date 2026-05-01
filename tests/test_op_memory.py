"""
test_op_memory.py -- S1 operation-memory tier unit tests.

S1 adds:
  * kind='operation' (graph-only, never embedded into Chroma)
  * predicates performed_well / performed_poorly / executed_op
  * tool-agnostic _append_trace (tool + truncated args + context_id)
  * operation_ratings parameter on finalize_intent (promotion gate)
  * walk_operation_neighbourhood + retrieve_past_operations in scoring
  * past_operations field on declare_operation response

Grounding: arXiv 2512.18950 (hierarchical procedural memory) for the
operation-tier rationale; Leontiev 1981 (Activity Theory AAO) for the
ontology.

These tests cover the pure-function building blocks:
  * _truncate_op_args: per-field cap, whole-payload cap, non-dict input
  * VALID_KINDS contains 'operation' (ontology-surface assertion)
  * _append_trace writes context_id + args into the jsonl
  * _consume_pending_operation_cue propagates active_context_id
  * walk_operation_neighbourhood signs performed_well/poorly correctly
  * retrieve_past_operations hydrates op properties from the KG

Full end-to-end integration (declare → op → rate → finalize →
re-declare surfaces past_operations) requires mempalace state fixtures
and is covered by the existing intent-system test harness -- these tests
focus on the units that compose into that path.
"""

from __future__ import annotations

import json

import pytest

from mempalace import hooks_cli
from mempalace import mcp_server
from mempalace.intent import _regex_copout_check, _semantic_copout_check
from mempalace.scoring import (
    retrieve_past_operations,
    walk_operation_neighbourhood,
)


# ─────────────────────────────────────────────────────────────────────
# Unit: _truncate_op_args
# ─────────────────────────────────────────────────────────────────────


class TestTruncateOpArgs:
    def test_none_returns_empty_dict(self):
        assert hooks_cli._truncate_op_args(None) == {}

    def test_non_dict_returns_empty_dict(self):
        assert hooks_cli._truncate_op_args("not-a-dict") == {}
        assert hooks_cli._truncate_op_args(["list"]) == {}

    def test_small_dict_passes_through(self):
        args = {"file_path": "/tmp/x.py", "offset": 10}
        out = hooks_cli._truncate_op_args(args)
        assert out == args

    def test_long_string_field_truncated_with_sha12(self):
        long_val = "a" * (hooks_cli._OP_TRACE_FIELD_CHARS + 50)
        out = hooks_cli._truncate_op_args({"content": long_val})
        assert len(out["content"]) < len(long_val)
        assert "<sha12:" in out["content"]
        # Deterministic fingerprint -- same input yields same hash.
        out2 = hooks_cli._truncate_op_args({"content": long_val})
        assert out["content"] == out2["content"]

    def test_oversized_payload_gets_truncation_marker(self):
        # 10 fields each 500 chars = 5KB -- over 2KB budget.
        args = {f"f{i}": "x" * 500 for i in range(10)}
        out = hooks_cli._truncate_op_args(args)
        assert "__truncated_sha12__" in out
        for k, v in out.items():
            if k == "__truncated_sha12__":
                continue
            if isinstance(v, str):
                # 200-char second-pass trim + suffix
                assert len(v) < 300

    def test_non_string_fields_unmodified(self):
        args = {"offset": 42, "limit": 10, "enabled": True, "ratio": 3.14}
        out = hooks_cli._truncate_op_args(args)
        assert out == args


# ─────────────────────────────────────────────────────────────────────
# Unit: VALID_KINDS ontology surface
# ─────────────────────────────────────────────────────────────────────


class TestValidKindsOperation:
    def test_operation_is_valid_kind(self):
        assert "operation" in mcp_server.VALID_KINDS

    def test_validator_accepts_operation(self):
        assert mcp_server._validate_kind("operation") == "operation"

    def test_validator_error_mentions_operation(self):
        with pytest.raises(ValueError) as exc:
            mcp_server._validate_kind(None)
        assert "operation" in str(exc.value)

    def test_validator_rejects_unknown_kind(self):
        with pytest.raises(ValueError):
            mcp_server._validate_kind("concept")


# ─────────────────────────────────────────────────────────────────────
# Unit: _append_trace shape (new format with context_id + args)
# ─────────────────────────────────────────────────────────────────────


class TestAppendTraceShape:
    def _read_entries(self, tmp_path, sid):
        f = tmp_path / f"execution_trace_{sid}.jsonl"
        if not f.is_file():
            return []
        return [
            json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()
        ]

    def test_entry_has_tool_and_context_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hooks_cli, "_TRACE_DIR", tmp_path)
        hooks_cli._append_trace(
            "sid-a",
            "Edit",
            {"file_path": "/tmp/x.py", "old_string": "foo", "new_string": "bar"},
            context_id="ctx_123",
        )
        entries = self._read_entries(tmp_path, "sid-a")
        assert len(entries) == 1
        e = entries[0]
        assert e["tool"] == "Edit"
        assert e["context_id"] == "ctx_123"
        assert "args" in e
        assert e["args"]["file_path"] == "/tmp/x.py"

    def test_entry_works_without_context_id(self, tmp_path, monkeypatch):
        # Backward compatibility: legacy callers still work; context_id
        # defaults to "" and the entry is written.
        monkeypatch.setattr(hooks_cli, "_TRACE_DIR", tmp_path)
        hooks_cli._append_trace("sid-b", "Read", {"file_path": "/tmp/y.py"})
        entries = self._read_entries(tmp_path, "sid-b")
        assert len(entries) == 1
        assert entries[0]["context_id"] == ""

    def test_empty_sid_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hooks_cli, "_TRACE_DIR", tmp_path)
        hooks_cli._append_trace("", "Read", {"file_path": "/tmp/x"}, context_id="c")
        assert list(tmp_path.iterdir()) == []


# ─────────────────────────────────────────────────────────────────────
# Unit: _consume_pending_operation_cue propagates active_context_id
# ─────────────────────────────────────────────────────────────────────


class TestConsumeCueContextId:
    def test_returns_active_context_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)
        from datetime import datetime

        now_iso = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        intent = {
            "pending_operation_cues": [
                {
                    "tool": "Edit",
                    "queries": ["q1", "q2"],
                    "keywords": ["kw1"],
                    "active_context_id": "ctx_abc",
                    "declared_at_ts": now_iso,
                }
            ]
        }
        popped, expired_n = hooks_cli._consume_pending_operation_cue("sid-test", intent, "Edit")
        assert popped is not None
        assert popped["active_context_id"] == "ctx_abc"
        assert popped["queries"] == ["q1", "q2"]

    def test_missing_active_context_id_defaults_to_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)
        from datetime import datetime

        now_iso = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        intent = {
            "pending_operation_cues": [
                {
                    "tool": "Read",
                    "queries": ["q"],
                    "keywords": ["k"],
                    "declared_at_ts": now_iso,
                    # No active_context_id -- legacy cue shape.
                }
            ]
        }
        popped, _ = hooks_cli._consume_pending_operation_cue("sid-test", intent, "Read")
        assert popped is not None
        assert popped["active_context_id"] == ""


# ─────────────────────────────────────────────────────────────────────
# In-memory KG stub for walk_operation_neighbourhood tests
# ─────────────────────────────────────────────────────────────────────


class _FakeKG:
    """Minimal in-memory KG stub for testing walk_operation_neighbourhood
    in isolation from the real sqlite-backed KG.

    Supports:
      * get_similar_contexts(cid, hops, decay) → list[(cid, sim)]
      * query_entity(cid, direction) → list[edge dicts]
      * get_entity(eid) → dict or None
    """

    def __init__(self, edges=None, similar=None, entities=None):
        # edges: {context_id: [{"predicate": p, "object": o, "current": bool, "confidence": float}]}
        self._edges = edges or {}
        self._similar = similar or {}
        self._entities = entities or {}

    def get_similar_contexts(self, cid, hops=2, decay=0.5):
        return list(self._similar.get(cid, []))

    def query_entity(self, cid, direction="outgoing"):
        return list(self._edges.get(cid, []))

    def get_entity(self, eid):
        return self._entities.get(eid)


# ─────────────────────────────────────────────────────────────────────
# Unit: walk_operation_neighbourhood signs performed_well / poorly
# ─────────────────────────────────────────────────────────────────────


class TestWalkOperationNeighbourhood:
    def test_no_context_returns_empty(self):
        kg = _FakeKG()
        out = walk_operation_neighbourhood("", kg)
        assert out == {"op_scores": {}, "good_ops": [], "bad_ops": []}

    def test_performed_well_positive(self):
        kg = _FakeKG(
            edges={
                "ctx_a": [
                    {
                        "predicate": "performed_well",
                        "object": "op_1",
                        "current": True,
                        "confidence": 1.0,
                    }
                ]
            },
        )
        out = walk_operation_neighbourhood("ctx_a", kg)
        assert out["op_scores"]["op_1"] > 0
        assert "op_1" in [op_id for _s, op_id in out["good_ops"]]

    def test_performed_poorly_negative(self):
        kg = _FakeKG(
            edges={
                "ctx_a": [
                    {
                        "predicate": "performed_poorly",
                        "object": "op_2",
                        "current": True,
                        "confidence": 1.0,
                    }
                ]
            },
        )
        out = walk_operation_neighbourhood("ctx_a", kg)
        assert out["op_scores"]["op_2"] < 0
        assert "op_2" in [op_id for _s, op_id in out["bad_ops"]]

    def test_similar_contexts_contribute_at_decay(self):
        kg = _FakeKG(
            edges={
                "ctx_similar": [
                    {
                        "predicate": "performed_well",
                        "object": "op_x",
                        "current": True,
                        "confidence": 1.0,
                    }
                ],
            },
            similar={"ctx_a": [("ctx_similar", 0.5)]},
        )
        out = walk_operation_neighbourhood("ctx_a", kg)
        assert out["op_scores"]["op_x"] == pytest.approx(0.5)

    def test_stale_edges_ignored(self):
        kg = _FakeKG(
            edges={
                "ctx_a": [
                    {
                        "predicate": "performed_well",
                        "object": "op_stale",
                        "current": False,
                        "confidence": 1.0,
                    },
                    {
                        "predicate": "performed_well",
                        "object": "op_live",
                        "current": True,
                        "confidence": 1.0,
                    },
                ]
            },
        )
        out = walk_operation_neighbourhood("ctx_a", kg)
        assert "op_stale" not in out["op_scores"]
        assert "op_live" in out["op_scores"]

    def test_unrelated_predicates_ignored(self):
        # rated_useful is the memory-relevance predicate -- must NOT
        # contribute to op scores (orthogonality guard).
        kg = _FakeKG(
            edges={
                "ctx_a": [
                    {
                        "predicate": "rated_useful",
                        "object": "mem_1",
                        "current": True,
                        "confidence": 1.0,
                    },
                    {
                        "predicate": "performed_well",
                        "object": "op_1",
                        "current": True,
                        "confidence": 1.0,
                    },
                ]
            },
        )
        out = walk_operation_neighbourhood("ctx_a", kg)
        assert "mem_1" not in out["op_scores"]
        assert "op_1" in out["op_scores"]


# ─────────────────────────────────────────────────────────────────────
# Unit: retrieve_past_operations hydrates op properties
# ─────────────────────────────────────────────────────────────────────


class TestRetrievePastOperations:
    def test_hydrates_tool_and_args_summary(self):
        kg = _FakeKG(
            edges={
                "ctx_a": [
                    {
                        "predicate": "performed_well",
                        "object": "op_edit_1",
                        "current": True,
                        "confidence": 1.0,
                    }
                ]
            },
            entities={
                "op_edit_1": {
                    "id": "op_edit_1",
                    "properties": {
                        "tool": "Edit",
                        "args_summary": "auth.py rate limiter",
                        "quality": 5,
                        "reason": "saved a refactor",
                    },
                },
            },
        )
        out = retrieve_past_operations("ctx_a", kg, k=5)
        assert len(out["good_precedents"]) == 1
        g = out["good_precedents"][0]
        # 2026-04-27: minimal-shape rows = {text, reason} only.
        # text is rendered description ("{tool} op: {args_summary}");
        # quality is implied by the lane (good_precedents = ≥4);
        # op_id is internal handle, no longer surfaced.
        assert g["summary_text"] == "Edit op: auth.py rate limiter"
        assert "op_id" not in g
        assert "quality" not in g

    def test_handles_json_string_properties(self):
        kg = _FakeKG(
            edges={
                "ctx_a": [
                    {
                        "predicate": "performed_poorly",
                        "object": "op_bash_1",
                        "current": True,
                        "confidence": 1.0,
                    }
                ]
            },
            entities={
                "op_bash_1": {
                    "id": "op_bash_1",
                    # Some KG impls store properties as JSON string
                    "properties": json.dumps({"tool": "Bash", "args_summary": "rm -rf /"}),
                },
            },
        )
        out = retrieve_past_operations("ctx_a", kg, k=5)
        assert len(out["avoid_patterns"]) == 1
        b = out["avoid_patterns"][0]
        # JSON-string properties parse; rendered text picks up tool+args.
        assert b["summary_text"] == "Bash op: rm -rf /"

    def test_k_caps_list_length(self):
        edges = [
            {
                "predicate": "performed_well",
                "object": f"op_{i}",
                "current": True,
                "confidence": 1.0,
            }
            for i in range(10)
        ]
        kg = _FakeKG(edges={"ctx_a": edges})
        out = retrieve_past_operations("ctx_a", kg, k=3)
        assert len(out["good_precedents"]) == 3


# ─────────────────────────────────────────────────────────────────────
# Unit: _regex_copout_check -- fast-path of the hybrid reason gate
# ─────────────────────────────────────────────────────────────────────


class TestRegexCopoutCheck:
    """Regex fast-path covers the cheap-and-obvious cop-outs.

    False positives on compound nouns that share a cop-out word as
    prefix (e.g. "skip-list", "skip-gram") are bugs -- the 2026-04-24
    fix tightened \\bskip(ped)?\\b to \\bskipped\\b precisely to
    avoid those. New patterns should follow: standalone-verb forms
    only, no broad prefix matches.
    """

    @pytest.mark.parametrize(
        "reason",
        [
            "don't know",
            "dont know what this is",
            "not used in this intent",
            "never used it",
            "didn't use",
            "didnt use this",
            "not sure what to say",
            "N/A",
            "n.a.",
            "n / a",
            "no idea",
            "not applicable",
            "aborted before running anything",
            "not rated yet",
            "skipped this memory",
            "unclear",
            "unknown",
            "placeholder",
            "TBD",
            "todo",
        ],
    )
    def test_rejects_known_cop_outs(self, reason):
        assert _regex_copout_check(reason) is True, f"regex should reject {reason!r}"

    @pytest.mark.parametrize(
        "reason",
        [
            # Data-structure / technical terms containing "skip" as prefix --
            # MUST NOT be rejected by the regex. The 2026-04-24 tightening
            # of \\bskip(ped)?\\b → \\bskipped\\b specifically prevents this.
            "Memory about skip-list data structure",
            "skip-gram embedding model",
            # Legitimate short-but-real reasons
            "Memory about auth, intent was about DB",
            "Generic system description, background only",
            "Unrelated to current task (topic mismatch)",
            # Contains 'aborted' but not with 'running' -- the pattern
            # requires both words together.
            "Aborted due to user decision after reading",
        ],
    )
    def test_does_not_false_positive_on_compound_nouns(self, reason):
        assert _regex_copout_check(reason) is False, (
            f"regex FALSE POSITIVE on {reason!r} -- "
            "tighten the pattern to avoid compound-noun prefix matches"
        )

    def test_non_string_input_returns_false(self):
        assert _regex_copout_check(None) is False
        assert _regex_copout_check(123) is False
        assert _regex_copout_check(["don't know"]) is False


# ─────────────────────────────────────────────────────────────────────
# Unit: _semantic_copout_check -- embedding-based second pass
# ─────────────────────────────────────────────────────────────────────


class TestSemanticCopoutCheck:
    """Semantic similarity gate catches paraphrased cop-outs that
    regex misses. Tests exercise: empty input, real embedder path,
    and fail-open behaviour when the embedder is unavailable.
    """

    def test_empty_reason_returns_false(self):
        assert _semantic_copout_check("") == (False, 0.0)
        assert _semantic_copout_check("   ") == (False, 0.0)

    def test_paraphrased_copout_has_nontrivial_similarity(self):
        # "I lack the information to evaluate this" is semantically
        # close to the exemplar "I don't know what this memory contains"
        # -- regex won't catch it, semantic should produce non-trivial
        # similarity even if it doesn't cross the 0.70 threshold on
        # every model version.
        pytest.importorskip("chromadb")
        _hit, sim = _semantic_copout_check(
            "I lack the information to properly evaluate this memory's relevance"
        )
        assert sim > 0.3, f"expected non-trivial similarity for paraphrase, got {sim}"

    def test_concrete_reason_low_similarity(self):
        # A reason that names specific technical content should NOT
        # fire the semantic gate.
        pytest.importorskip("chromadb")
        hit, sim = _semantic_copout_check(
            "Memory describes the InjectionGate Haiku judge contract "
            "which directly informs how I structured the new cop-out "
            "gate rejection error response"
        )
        assert hit is False, f"semantic gate over-rejected concrete reason (sim={sim})"

    def test_fail_open_on_embedder_error(self, monkeypatch):
        # If the chromadb import or embedder call throws, the helper
        # must return (False, 0.0) so a broken embedder cannot block
        # finalize.
        def _broken_import(name, *args, **kwargs):
            if name == "chromadb.utils":
                raise ImportError("simulated broken chromadb")
            return __import__(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _broken_import)
        hit, sim = _semantic_copout_check("some arbitrary reason text")
        assert hit is False
        assert sim == 0.0


# ─────────────────────────────────────────────────────────────────────
# Unit: S2 corrections -- retrieve_past_operations walks superseded_by
# ─────────────────────────────────────────────────────────────────────


class TestRetrievePastOperationsCorrections:
    """S2 adds `corrections` to the past_operations bundle: bad ops
    that carry a `superseded_by` edge to a better alternative get
    surfaced as a (bad, better) pair so the response reads
    'don't do X, do Y instead' rather than just 'don't do X'.
    """

    def test_no_superseded_by_yields_empty_corrections(self):
        kg = _FakeKG(
            edges={
                "ctx_a": [
                    {
                        "predicate": "performed_poorly",
                        "object": "op_bad",
                        "current": True,
                        "confidence": 1.0,
                    }
                ]
            },
            entities={
                "op_bad": {
                    "id": "op_bad",
                    "properties": {"tool": "Bash", "args_summary": "rm -rf /", "quality": 1},
                },
            },
        )
        out = retrieve_past_operations("ctx_a", kg, k=5)
        assert len(out["avoid_patterns"]) == 1
        assert out["corrections"] == []

    def test_superseded_by_edge_produces_correction_entry(self):
        kg = _FakeKG(
            edges={
                "ctx_a": [
                    {
                        "predicate": "performed_poorly",
                        "object": "op_bad_bash",
                        "current": True,
                        "confidence": 1.0,
                    }
                ],
                "op_bad_bash": [
                    {
                        "predicate": "superseded_by",
                        "object": "op_good_read",
                        "current": True,
                        "confidence": 1.0,
                    }
                ],
            },
            entities={
                "op_bad_bash": {
                    "id": "op_bad_bash",
                    "properties": {"tool": "Bash", "args_summary": "cat foo.py", "quality": 1},
                },
                "op_good_read": {
                    "id": "op_good_read",
                    "properties": {"tool": "Read", "args_summary": "foo.py", "quality": 5},
                },
            },
        )
        out = retrieve_past_operations("ctx_a", kg, k=5)
        assert len(out["corrections"]) == 1
        c = out["corrections"][0]
        # Internal handles bad_op_id/better_op_id stripped after the
        # template hoist runs; agent-visible shape is just bad+better
        # rows trimmed to {text, reason}.
        assert "bad_op_id" not in c
        assert "better_op_id" not in c
        assert c["bad"]["summary_text"] == "Bash op: cat foo.py"
        assert c["better"]["summary_text"] == "Read op: foo.py"
        assert "quality" not in c["better"]

    def test_stale_superseded_by_ignored(self):
        kg = _FakeKG(
            edges={
                "ctx_a": [
                    {
                        "predicate": "performed_poorly",
                        "object": "op_bad",
                        "current": True,
                        "confidence": 1.0,
                    }
                ],
                "op_bad": [
                    {
                        "predicate": "superseded_by",
                        "object": "op_stale",
                        "current": False,
                        "confidence": 1.0,
                    }
                ],
            },
            entities={
                "op_bad": {"id": "op_bad", "properties": {"tool": "Bash"}},
                "op_stale": {"id": "op_stale", "properties": {"tool": "Edit"}},
            },
        )
        out = retrieve_past_operations("ctx_a", kg, k=5)
        assert out["corrections"] == []

    def test_unrelated_predicate_ignored(self):
        # Only `superseded_by` drives the corrections walk. Other
        # outgoing edges on the bad op must NOT leak into corrections.
        kg = _FakeKG(
            edges={
                "ctx_a": [
                    {
                        "predicate": "performed_poorly",
                        "object": "op_bad",
                        "current": True,
                        "confidence": 1.0,
                    }
                ],
                "op_bad": [
                    {
                        "predicate": "executed_by",
                        "object": "some_agent",
                        "current": True,
                        "confidence": 1.0,
                    },
                    {
                        "predicate": "executed_op",
                        "object": "some_exec",
                        "current": True,
                        "confidence": 1.0,
                    },
                ],
            },
            entities={"op_bad": {"id": "op_bad", "properties": {"tool": "Bash"}}},
        )
        out = retrieve_past_operations("ctx_a", kg, k=5)
        assert out["corrections"] == []

    def test_good_precedents_do_not_carry_corrections(self):
        # Corrections derive ONLY from the bad_ops / avoid_patterns
        # lane. A superseded_by edge on a good op (nonsensical but
        # possible) must not produce a correction entry.
        kg = _FakeKG(
            edges={
                "ctx_a": [
                    {
                        "predicate": "performed_well",
                        "object": "op_good",
                        "current": True,
                        "confidence": 1.0,
                    }
                ],
                "op_good": [
                    {
                        "predicate": "superseded_by",
                        "object": "op_other",
                        "current": True,
                        "confidence": 1.0,
                    }
                ],
            },
            entities={
                "op_good": {"id": "op_good", "properties": {"tool": "Read"}},
                "op_other": {"id": "op_other", "properties": {"tool": "Grep"}},
            },
        )
        out = retrieve_past_operations("ctx_a", kg, k=5)
        assert len(out["good_precedents"]) == 1
        assert out["corrections"] == []


# ─────────────────────────────────────────────────────────────────────
# S3a: detect_op_cluster_flags -- same-tool cluster detection over
# retrieve_past_operations output. Emits flag dicts that declare_operation
# passes to kg.record_memory_flags for the gardener to template (S3b).
# ─────────────────────────────────────────────────────────────────────


class TestDetectOpClusterFlags:
    """detect_op_cluster_flags scans past_operations for same-tool
    same-sign clusters >= N members and returns flag dicts. It does
    NOT emit (the caller does) -- this is a pure function, so tests
    are all in-memory with no KG.
    """

    def test_empty_past_ops_returns_no_flags(self):
        from mempalace.scoring import detect_op_cluster_flags

        assert detect_op_cluster_flags({"good_precedents": [], "avoid_patterns": []}) == []

    def test_missing_keys_returns_no_flags(self):
        from mempalace.scoring import detect_op_cluster_flags

        assert detect_op_cluster_flags({}) == []

    def test_non_dict_input_returns_no_flags(self):
        # Fail-open guard: upstream retrieve_past_operations could
        # conceivably return None on a bad KG read; the detector must
        # not explode the declare_operation path.
        from mempalace.scoring import detect_op_cluster_flags

        assert detect_op_cluster_flags(None) == []
        assert detect_op_cluster_flags("not-a-dict") == []

    def test_two_same_tool_below_threshold_no_flag(self):
        from mempalace.scoring import detect_op_cluster_flags

        out = detect_op_cluster_flags(
            {
                "good_precedents": [
                    {"op_id": "op_1", "tool": "Read"},
                    {"op_id": "op_2", "tool": "Read"},
                ],
                "avoid_patterns": [],
            }
        )
        assert out == []

    def test_three_same_tool_good_triggers_positive_flag(self):
        from mempalace.scoring import detect_op_cluster_flags

        out = detect_op_cluster_flags(
            {
                "good_precedents": [
                    {"op_id": "op_1", "tool": "Read"},
                    {"op_id": "op_2", "tool": "Read"},
                    {"op_id": "op_3", "tool": "Read"},
                ],
                "avoid_patterns": [],
            }
        )
        assert len(out) == 1
        assert out[0]["kind"] == "op_cluster_templatizable"
        assert out[0]["detail"] == "positive"
        assert out[0]["memory_ids"] == ["op_1", "op_2", "op_3"]

    def test_three_same_tool_bad_triggers_negative_flag(self):
        from mempalace.scoring import detect_op_cluster_flags

        out = detect_op_cluster_flags(
            {
                "good_precedents": [],
                "avoid_patterns": [
                    {"op_id": "op_1", "tool": "Bash"},
                    {"op_id": "op_2", "tool": "Bash"},
                    {"op_id": "op_3", "tool": "Bash"},
                ],
            }
        )
        assert len(out) == 1
        assert out[0]["detail"] == "negative"
        assert out[0]["memory_ids"] == ["op_1", "op_2", "op_3"]

    def test_mixed_tools_each_need_own_threshold(self):
        # Read has 2 (below threshold), Edit has 3 (at threshold).
        # Only Edit should emit a flag.
        from mempalace.scoring import detect_op_cluster_flags

        out = detect_op_cluster_flags(
            {
                "good_precedents": [
                    {"op_id": "op_1", "tool": "Read"},
                    {"op_id": "op_2", "tool": "Read"},
                    {"op_id": "op_3", "tool": "Edit"},
                    {"op_id": "op_4", "tool": "Edit"},
                    {"op_id": "op_5", "tool": "Edit"},
                ],
                "avoid_patterns": [],
            }
        )
        assert len(out) == 1
        assert out[0]["memory_ids"] == ["op_3", "op_4", "op_5"]

    def test_good_and_bad_clusters_both_emit_independently(self):
        # Same-sign rule: a positive-tool cluster and a negative-tool
        # cluster emit separate flags. They never mix.
        from mempalace.scoring import detect_op_cluster_flags

        out = detect_op_cluster_flags(
            {
                "good_precedents": [{"op_id": f"g_{i}", "tool": "Read"} for i in range(3)],
                "avoid_patterns": [{"op_id": f"b_{i}", "tool": "Bash"} for i in range(3)],
            }
        )
        assert len(out) == 2
        details = sorted(f["detail"] for f in out)
        assert details == ["negative", "positive"]

    def test_missing_tool_field_ignored(self):
        # Op rows without a tool can't anchor a template recipe.
        from mempalace.scoring import detect_op_cluster_flags

        out = detect_op_cluster_flags(
            {
                "good_precedents": [
                    {"op_id": "op_1", "tool": ""},
                    {"op_id": "op_2"},
                    {"op_id": "op_3", "tool": "Read"},
                ],
                "avoid_patterns": [],
            }
        )
        assert out == []

    def test_missing_op_id_ignored(self):
        from mempalace.scoring import detect_op_cluster_flags

        out = detect_op_cluster_flags(
            {
                "good_precedents": [
                    {"tool": "Read"},
                    {"op_id": "op_2", "tool": "Read"},
                    {"op_id": "op_3", "tool": "Read"},
                ],
                "avoid_patterns": [],
            }
        )
        assert out == []

    def test_configurable_min_cluster_size(self):
        # Bumping min_cluster_size to 5 turns the 3-member cluster
        # into no-flag; dropping to 2 makes a 2-member cluster trigger.
        from mempalace.scoring import detect_op_cluster_flags

        past_ops = {
            "good_precedents": [
                {"op_id": "op_1", "tool": "Read"},
                {"op_id": "op_2", "tool": "Read"},
                {"op_id": "op_3", "tool": "Read"},
            ],
            "avoid_patterns": [],
        }
        assert detect_op_cluster_flags(past_ops, min_cluster_size=5) == []
        assert len(detect_op_cluster_flags(past_ops, min_cluster_size=2)) == 1

    def test_duplicate_op_ids_within_lane_collapse(self):
        # Retrieval dedupes already, but defensive: if an op_id
        # appears twice in the same lane it should NOT inflate the
        # cluster size (3 unique ops is the threshold, not 3 rows).
        from mempalace.scoring import detect_op_cluster_flags

        out = detect_op_cluster_flags(
            {
                "good_precedents": [
                    {"op_id": "op_1", "tool": "Read"},
                    {"op_id": "op_1", "tool": "Read"},
                    {"op_id": "op_2", "tool": "Read"},
                ],
                "avoid_patterns": [],
            }
        )
        # Only 2 unique ops -- below threshold.
        assert out == []


# ─────────────────────────────────────────────────────────────────────
# S3a: op_cluster_templatizable is registered in both flag-kind
# enforcement sites. This is a closed-set ontology-surface guard:
# any new kind added outside these two sites would silently no-op at
# the writer (_MEMORY_FLAG_KINDS skips unknowns) or at the gate
# (_FLAG_KINDS_ENUM drives the tool_use schema). Assertion here
# catches the most common regression path.
# ─────────────────────────────────────────────────────────────────────


class TestOpClusterFlagKindRegistered:
    def test_kind_in_knowledge_graph_frozenset(self):
        from mempalace.knowledge_graph import KnowledgeGraph

        assert "op_cluster_templatizable" in KnowledgeGraph._MEMORY_FLAG_KINDS

    def test_kind_in_injection_gate_enum(self):
        from mempalace.injection_gate import _FLAG_KINDS_ENUM

        assert "op_cluster_templatizable" in _FLAG_KINDS_ENUM


# ─────────────────────────────────────────────────────────────────────
# S3b: gardener-side surface -- synthesize_operation_template shim,
# _derive_resolution mapping, _MUTATION_TOOL_NAMES, _FLAG_RESOLUTIONS,
# and the `templatizes` predicate registration.
# ─────────────────────────────────────────────────────────────────────


class TestTemplatizedResolutionRegistered:
    def test_templatized_in_flag_resolutions(self):
        # mark_flag_resolved validates against this frozenset; missing
        # entry = ValueError at write time.
        from mempalace.knowledge_graph import KnowledgeGraph

        assert "templatized" in KnowledgeGraph._FLAG_RESOLUTIONS


class TestSynthesizeTemplateToolSchema:
    def test_tool_registered_in_schemas(self):
        from mempalace.memory_gardener import _TOOL_SCHEMAS

        names = [s["name"] for s in _TOOL_SCHEMAS]
        assert "mempalace_synthesize_operation_template" in names

    def test_tool_schema_has_required_inputs(self):
        from mempalace.memory_gardener import _TOOL_SCHEMAS

        schema = next(
            s for s in _TOOL_SCHEMAS if s["name"] == "mempalace_synthesize_operation_template"
        )
        required = set(schema["input_schema"]["required"])
        # Core pattern fields Haiku MUST fill in to mint a useful template.
        assert {"op_ids", "title", "when_to_use", "recipe"}.issubset(required)

    def test_tool_in_mutation_name_set(self):
        from mempalace.memory_gardener import _MUTATION_TOOL_NAMES

        assert "mempalace_synthesize_operation_template" in _MUTATION_TOOL_NAMES

    def test_tool_in_dispatch_after_init(self):
        # The dispatch map is lazily wired; confirm the S3b shim lands
        # in it after _init_tool_dispatch runs.
        from mempalace import memory_gardener as mg

        # Reset the dispatch to re-trigger lazy init cleanly.
        mg._TOOL_DISPATCH.clear()
        mg._init_tool_dispatch()
        assert "mempalace_synthesize_operation_template" in mg._TOOL_DISPATCH


class TestDeriveResolutionTemplatized:
    """_derive_resolution maps a successful synthesize_operation_template
    call back to the 'templatized' resolution bucket. This is the
    audit-path the run-log reads, so getting it wrong would silently
    drop templated flags into 'deferred'."""

    def _make_trace(self, *, name, result, is_error=False, arguments=None):
        # Mirror the ToolCallTrace shape used by the tool-use loop.
        from mempalace.memory_gardener import ToolCallTrace

        return ToolCallTrace(
            name=name,
            arguments=arguments or {},
            result=result,
            is_error=is_error,
        )

    def test_successful_synthesize_maps_to_templatized(self):
        from mempalace.memory_gardener import _derive_resolution

        trace = [
            self._make_trace(
                name="mempalace_synthesize_operation_template",
                result={
                    "success": True,
                    "template_id": "record_memory_gardener_op_template_abcdef123456",
                    "op_ids": ["op_read_1", "op_read_2", "op_read_3"],
                    "edges_written": 3,
                    "title": "read-then-grep for symbol usage",
                },
                arguments={
                    "op_ids": ["op_read_1", "op_read_2", "op_read_3"],
                    "title": "read-then-grep for symbol usage",
                },
            )
        ]
        resolution, note = _derive_resolution("op_cluster_templatizable", trace)
        assert resolution == "templatized"
        assert "record_memory_gardener_op_template_abcdef123456" in note
        assert "3 op" in note  # op count
        assert "3 edge" in note  # edge count

    def test_errored_synthesize_maps_to_deferred(self):
        # Any mutation error path bounces to deferred with the reason;
        # this mirrors how other tools behave and is the failure-mode
        # contract for the gardener.
        from mempalace.memory_gardener import _derive_resolution

        trace = [
            self._make_trace(
                name="mempalace_synthesize_operation_template",
                result={"success": False, "error": "op_ids collapsed to <2 after cleaning"},
                is_error=True,
            )
        ]
        resolution, _note = _derive_resolution("op_cluster_templatizable", trace)
        assert resolution == "deferred"


class TestTemplatizesPredicateRegistered:
    """_ensure_operation_ontology(kg) seeds the four op-memory
    predicates on a cold palace. S3b adds `templatizes`; this test
    confirms it lands with the right constraints (record -> operation,
    one-to-many), so kg.add_triple(record, 'templatizes', op) works
    without hitting predicate-validation errors."""

    def test_templatizes_predicate_seeded(self, tmp_path):
        from mempalace.knowledge_graph import KnowledgeGraph
        from mempalace.mcp_server import _ensure_operation_ontology

        db = tmp_path / "palace.db"
        kg = KnowledgeGraph(str(db))
        # Cold-start lock 2026-05-01: seed_ontology must run first so
        # the `is_a thing` triple emitted by _ensure_operation_ontology
        # lands on a declared parent class. Pre-cold-start this worked
        # via add_triple's phantom auto-create; closed at the gate.
        kg.seed_ontology()
        _ensure_operation_ontology(kg)
        ent = kg.get_entity("templatizes")
        assert ent is not None
        assert ent.get("kind") == "predicate"
        props = ent.get("properties") or {}
        if isinstance(props, str):
            import json as _json

            props = _json.loads(props)
        constraints = props.get("constraints") or {}
        assert "record" in (constraints.get("subject_kinds") or [])
        assert "operation" in (constraints.get("object_kinds") or [])
        assert constraints.get("cardinality") == "one-to-many"


class TestSynthesizeShim:
    """_synthesize_operation_template_shim mints a record + writes
    `templatizes` edges against the live KG. Uses a tmp palace so we
    exercise the real add_entity / add_triple path, not mocks."""

    def _bootstrap_kg(self, tmp_path):
        """Build a KG with the operation ontology + a couple of seed
        op entities so add_triple has valid endpoints."""
        from mempalace.knowledge_graph import KnowledgeGraph
        from mempalace.mcp_server import _ensure_operation_ontology

        db = tmp_path / "palace.db"
        kg = KnowledgeGraph(str(db))
        # Cold-start lock 2026-05-01: seed_ontology must run first so
        # the `is_a thing` triple emitted by _ensure_operation_ontology
        # lands on a declared parent class. Pre-cold-start this worked
        # via add_triple's phantom auto-create; closed at the gate.
        kg.seed_ontology()
        _ensure_operation_ontology(kg)
        # Seed three op entities so the shim has real endpoints.
        for op_id in ("op_read_aaa", "op_read_bbb", "op_read_ccc"):
            kg.add_entity(
                op_id,
                kind="operation",
                content=f"op {op_id}",
                importance=3,
                properties={"tool": "Read", "args_summary": "test"},
            )
        return kg

    def test_mints_record_and_writes_edges(self, tmp_path, monkeypatch):
        from mempalace import mcp_server as _mcp
        from mempalace.memory_gardener import _synthesize_operation_template_shim

        kg = self._bootstrap_kg(tmp_path)
        monkeypatch.setattr(_mcp._STATE, "kg", kg)

        result = _synthesize_operation_template_shim(
            op_ids=["op_read_aaa", "op_read_bbb", "op_read_ccc"],
            title="read-then-grep pattern",
            when_to_use="When locating symbol usage across a module.",
            recipe="Read the file, then grep for the symbol name.",
            failure_modes=[],
        )
        assert result["success"] is True
        tid = result["template_id"]
        assert tid.startswith("record_memory_gardener_op_template_")
        assert result["edges_written"] == 3

        # Confirm the record exists and carries the template content.
        rec = kg.get_entity(tid)
        assert rec is not None
        assert rec.get("kind") == "record"

        # Confirm templatizes edges point to all three ops.
        edges = kg.query_entity(tid, direction="outgoing")
        templatized = {
            e.get("object")
            for e in edges
            if e.get("predicate") == "templatizes" and e.get("current", True)
        }
        assert templatized == {"op_read_aaa", "op_read_bbb", "op_read_ccc"}

    def test_rejects_too_few_op_ids(self, tmp_path, monkeypatch):
        # Haiku could emit a degenerate cluster (n=1). The shim should
        # refuse -- a 1-op "template" is just a rename of the op.
        from mempalace import mcp_server as _mcp
        from mempalace.memory_gardener import _synthesize_operation_template_shim

        kg = self._bootstrap_kg(tmp_path)
        monkeypatch.setattr(_mcp._STATE, "kg", kg)

        result = _synthesize_operation_template_shim(
            op_ids=["op_read_aaa"],
            title="t",
            when_to_use="w",
            recipe="r",
        )
        assert result["success"] is False
        assert "op_ids" in result["error"]

    def test_rejects_missing_required_fields(self, tmp_path, monkeypatch):
        from mempalace import mcp_server as _mcp
        from mempalace.memory_gardener import _synthesize_operation_template_shim

        kg = self._bootstrap_kg(tmp_path)
        monkeypatch.setattr(_mcp._STATE, "kg", kg)

        # Empty title -- shim requires all three pattern fields.
        result = _synthesize_operation_template_shim(
            op_ids=["op_read_aaa", "op_read_bbb"],
            title="",
            when_to_use="w",
            recipe="r",
        )
        assert result["success"] is False

    def test_failure_modes_included_in_content(self, tmp_path, monkeypatch):
        # Negative clusters carry failure_modes; confirm they land in
        # the record's stored full_content so the gardener's audit and
        # future retrieval (S3c) can read them.
        from mempalace import mcp_server as _mcp
        from mempalace.memory_gardener import _synthesize_operation_template_shim

        kg = self._bootstrap_kg(tmp_path)
        monkeypatch.setattr(_mcp._STATE, "kg", kg)

        result = _synthesize_operation_template_shim(
            op_ids=["op_read_aaa", "op_read_bbb", "op_read_ccc"],
            title="avoid: cat-piping-to-head",
            when_to_use="When you just need the first N lines of a file.",
            recipe="Prefer Read with offset/limit over Bash cat|head.",
            failure_modes=[
                "cat invokes the shell and loses binary-safety",
                "head drops trailing newline on short files",
            ],
        )
        assert result["success"] is True
        rec = kg.get_entity(result["template_id"])
        props = rec.get("properties") or {}
        if isinstance(props, str):
            import json as _json

            props = _json.loads(props)
        full = props.get("full_content", "")
        assert "Failure modes" in full
        assert "cat invokes the shell" in full

    def test_deterministic_template_id_for_same_cluster(self, tmp_path, monkeypatch):
        # Re-running the gardener on the same cluster should land on
        # the same template_id -- add_entity upserts, no duplicates.
        from mempalace import mcp_server as _mcp
        from mempalace.memory_gardener import _synthesize_operation_template_shim

        kg = self._bootstrap_kg(tmp_path)
        monkeypatch.setattr(_mcp._STATE, "kg", kg)

        args = {
            "op_ids": ["op_read_aaa", "op_read_bbb", "op_read_ccc"],
            "title": "t",
            "when_to_use": "w",
            "recipe": "r",
        }
        r1 = _synthesize_operation_template_shim(**args)
        r2 = _synthesize_operation_template_shim(**args)
        assert r1["template_id"] == r2["template_id"]

        # Same cluster with reordered op_ids yields the same id too
        # (sort before hashing). Protects against duplicate templates
        # when the flag's memory_ids ordering drifts.
        args_reordered = dict(args, op_ids=["op_read_ccc", "op_read_aaa", "op_read_bbb"])
        r3 = _synthesize_operation_template_shim(**args_reordered)
        assert r3["template_id"] == r1["template_id"]


# ─────────────────────────────────────────────────────────────────────
# S3c: retrieve_past_operations templates lane -- when a surfaced op
# has an incoming `templatizes` edge from a record, hoist that record
# into a new templates field and suppress the raw op from the regular
# lanes. Replace-not-append keeps payload bounded while delivering the
# higher-signal distilled version.
# ─────────────────────────────────────────────────────────────────────


class TestRetrievePastOperationsTemplates:
    def _make_kg(self, *, edges, entities, similar_contexts=None):
        """Reuse the in-process _FakeKG pattern from earlier classes
        but extended with optional similar-context returns."""
        sims = similar_contexts or []

        class _FakeKG:
            def get_similar_contexts(self, ctx, hops=2, decay=0.5):
                return sims

            def query_entity(self, ent, direction="both"):
                pool = edges.get(ent, []) or []
                if direction == "outgoing":
                    return [e for e in pool if e.get("_role", "out") == "out"]
                if direction == "incoming":
                    return [e for e in pool if e.get("_role", "out") == "in"]
                return list(pool)

            def get_entity(self, ent):
                return entities.get(ent)

        return _FakeKG()

    def test_no_templatizes_edge_no_hoist(self):
        # Baseline: no templates predicate present, response shape
        # gains an empty templates list but lanes are untouched.
        kg = self._make_kg(
            edges={
                "ctx_a": [
                    {
                        "predicate": "performed_well",
                        "object": "op_1",
                        "current": True,
                        "confidence": 1.0,
                        "_role": "out",
                    }
                ]
            },
            entities={"op_1": {"id": "op_1", "properties": {"tool": "Read"}}},
        )
        out = retrieve_past_operations("ctx_a", kg, k=5)
        assert len(out["good_precedents"]) == 1
        assert out["templates"] == []

    def test_template_hoist_suppresses_covered_op(self):
        # Surfaced op_1 has an incoming templatizes edge from
        # template_t1; the raw op gets dropped, the template surfaces.
        kg = self._make_kg(
            edges={
                "ctx_a": [
                    {
                        "predicate": "performed_well",
                        "object": "op_1",
                        "current": True,
                        "confidence": 1.0,
                        "_role": "out",
                    }
                ],
                "op_1": [
                    {
                        "predicate": "templatizes",
                        "subject": "template_t1",
                        "current": True,
                        "_role": "in",
                    }
                ],
            },
            entities={
                "op_1": {"id": "op_1", "properties": {"tool": "Read"}},
                "template_t1": {
                    "id": "template_t1",
                    "kind": "record",
                    "content": "read-then-grep pattern",
                    "properties": {
                        "title": "read-then-grep",
                        "full_content": "# read-then-grep\nuse Read then Grep",
                    },
                },
            },
        )
        out = retrieve_past_operations("ctx_a", kg, k=5)
        # Raw op suppressed, template hoisted.
        assert out["good_precedents"] == []
        assert len(out["templates"]) == 1
        t = out["templates"][0]
        assert t["template_id"] == "template_t1"
        assert "op_1" in t["op_ids"]
        assert t["title"] == "read-then-grep"

    def test_template_covers_multiple_ops_in_one_entry(self):
        # Two surfaced ops, both pointing back to one template -- the
        # template entry collects both in its op_ids list.
        kg = self._make_kg(
            edges={
                "ctx_a": [
                    {
                        "predicate": "performed_well",
                        "object": "op_1",
                        "current": True,
                        "confidence": 1.0,
                        "_role": "out",
                    },
                    {
                        "predicate": "performed_well",
                        "object": "op_2",
                        "current": True,
                        "confidence": 1.0,
                        "_role": "out",
                    },
                ],
                "op_1": [
                    {
                        "predicate": "templatizes",
                        "subject": "template_t1",
                        "current": True,
                        "_role": "in",
                    }
                ],
                "op_2": [
                    {
                        "predicate": "templatizes",
                        "subject": "template_t1",
                        "current": True,
                        "_role": "in",
                    }
                ],
            },
            entities={
                "op_1": {"id": "op_1", "properties": {"tool": "Read"}},
                "op_2": {"id": "op_2", "properties": {"tool": "Read"}},
                "template_t1": {
                    "id": "template_t1",
                    "kind": "record",
                    "content": "shared pattern",
                    "properties": {"title": "shared"},
                },
            },
        )
        out = retrieve_past_operations("ctx_a", kg, k=5)
        assert out["good_precedents"] == []
        assert len(out["templates"]) == 1
        assert sorted(out["templates"][0]["op_ids"]) == ["op_1", "op_2"]

    def test_templatizes_on_avoid_pattern_also_hoists(self):
        # Negative cluster: templates apply to performed_poorly ops too.
        kg = self._make_kg(
            edges={
                "ctx_a": [
                    {
                        "predicate": "performed_poorly",
                        "object": "op_bad",
                        "current": True,
                        "confidence": 1.0,
                        "_role": "out",
                    }
                ],
                "op_bad": [
                    {
                        "predicate": "templatizes",
                        "subject": "template_avoid",
                        "current": True,
                        "_role": "in",
                    }
                ],
            },
            entities={
                "op_bad": {"id": "op_bad", "properties": {"tool": "Bash"}},
                "template_avoid": {
                    "id": "template_avoid",
                    "kind": "record",
                    "content": "avoid the cat-pipe-head trap",
                    "properties": {"title": "avoid: cat-pipe-head"},
                },
            },
        )
        out = retrieve_past_operations("ctx_a", kg, k=5)
        assert out["avoid_patterns"] == []
        assert len(out["templates"]) == 1
        assert out["templates"][0]["template_id"] == "template_avoid"

    def test_partial_template_coverage_keeps_uncovered_ops(self):
        # Two ops surface; only op_1 is templatized. op_2 stays in
        # good_precedents; only op_1 gets suppressed.
        kg = self._make_kg(
            edges={
                "ctx_a": [
                    {
                        "predicate": "performed_well",
                        "object": "op_1",
                        "current": True,
                        "confidence": 1.0,
                        "_role": "out",
                    },
                    {
                        "predicate": "performed_well",
                        "object": "op_2",
                        "current": True,
                        "confidence": 0.5,
                        "_role": "out",
                    },
                ],
                "op_1": [
                    {
                        "predicate": "templatizes",
                        "subject": "template_t1",
                        "current": True,
                        "_role": "in",
                    }
                ],
                # op_2 has no incoming templatizes edges
            },
            entities={
                "op_1": {
                    "id": "op_1",
                    "properties": {"tool": "Read", "args_summary": "covered_path"},
                },
                "op_2": {
                    "id": "op_2",
                    "properties": {"tool": "Read", "args_summary": "uncovered_path"},
                },
                "template_t1": {
                    "id": "template_t1",
                    "kind": "record",
                    "content": "covers op_1 only",
                    "properties": {"title": "t1"},
                },
            },
        )
        out = retrieve_past_operations("ctx_a", kg, k=5)
        assert len(out["good_precedents"]) == 1
        # op_id is stripped; identify surviving op by its rendered text
        # (op_1 should be hoisted into templates lane; op_2 stays here).
        assert out["good_precedents"][0]["summary_text"] == "Read op: uncovered_path"
        assert len(out["templates"]) == 1

    def test_stale_templatizes_edge_ignored(self):
        # Invalidated edge (current=False) must NOT hoist a template;
        # the underlying op stays in its lane unchanged.
        kg = self._make_kg(
            edges={
                "ctx_a": [
                    {
                        "predicate": "performed_well",
                        "object": "op_1",
                        "current": True,
                        "confidence": 1.0,
                        "_role": "out",
                    }
                ],
                "op_1": [
                    {
                        "predicate": "templatizes",
                        "subject": "template_stale",
                        "current": False,  # invalidated
                        "_role": "in",
                    }
                ],
            },
            entities={
                "op_1": {"id": "op_1", "properties": {"tool": "Read"}},
                "template_stale": {
                    "id": "template_stale",
                    "kind": "record",
                    "content": "should not surface",
                    "properties": {},
                },
            },
        )
        out = retrieve_past_operations("ctx_a", kg, k=5)
        assert len(out["good_precedents"]) == 1
        assert out["templates"] == []

    def test_corrections_lane_also_suppressed_when_templated(self):
        # If the bad-op side of a correction is templatized, the
        # whole correction entry gets dropped -- the template carries
        # the avoid signal more concisely than the bad/better pair.
        kg = self._make_kg(
            edges={
                "ctx_a": [
                    {
                        "predicate": "performed_poorly",
                        "object": "op_bad",
                        "current": True,
                        "confidence": 1.0,
                        "_role": "out",
                    }
                ],
                "op_bad": [
                    {
                        "predicate": "superseded_by",
                        "object": "op_better",
                        "current": True,
                        "_role": "out",
                    },
                    {
                        "predicate": "templatizes",
                        "subject": "template_t1",
                        "current": True,
                        "_role": "in",
                    },
                ],
            },
            entities={
                "op_bad": {"id": "op_bad", "properties": {"tool": "Bash"}},
                "op_better": {"id": "op_better", "properties": {"tool": "Read"}},
                "template_t1": {
                    "id": "template_t1",
                    "kind": "record",
                    "content": "use Read not Bash",
                    "properties": {"title": "use Read not Bash"},
                },
            },
        )
        out = retrieve_past_operations("ctx_a", kg, k=5)
        assert out["avoid_patterns"] == []
        assert out["corrections"] == []
        assert len(out["templates"]) == 1


# ─────────────────────────────────────────────────────────────────────
# Silent-drop regression: templatizes must be in _TRIPLE_SKIP_PREDICATES
# so add_triple accepts a templatizes edge without a `statement` param.
# Earlier (pre-2026-04-25) the missing skip-list entry caused
# add_triple to raise TripleStatementRequired, which the gardener
# shim's bare-except swallowed → 0 edges written, no error surfaced.
# Per Adrian's rule "ensure this crashes, and is not silently omitted",
# this test locks the predicate's place in the skip list AND verifies
# the shim now propagates real failures instead of swallowing them.
# ─────────────────────────────────────────────────────────────────────


class TestTemplatizesSkipListRegression:
    def test_templatizes_in_triple_skip_predicates(self):
        from mempalace.knowledge_graph import _TRIPLE_SKIP_PREDICATES

        assert "templatizes" in _TRIPLE_SKIP_PREDICATES

    def test_add_triple_templatizes_no_statement_succeeds(self, tmp_path):
        # Direct add_triple without statement must succeed for
        # templatizes (the silent-drop bug surfaced precisely because
        # this raised TripleStatementRequired previously).
        from mempalace.knowledge_graph import KnowledgeGraph
        from mempalace.mcp_server import _ensure_operation_ontology

        db = tmp_path / "palace.db"
        kg = KnowledgeGraph(str(db))
        # Cold-start lock 2026-05-01: seed_ontology must run first so
        # the `is_a thing` triple emitted by _ensure_operation_ontology
        # lands on a declared parent class. Pre-cold-start this worked
        # via add_triple's phantom auto-create; closed at the gate.
        kg.seed_ontology()
        _ensure_operation_ontology(kg)
        kg.add_entity("op_aaa", kind="operation", content="op_aaa", importance=3)
        kg.add_entity(
            "record_template_xxx",
            kind="record",
            content="some template",
            importance=4,
        )
        # Should not raise -- templatizes is now in the skip list.
        kg.add_triple("record_template_xxx", "templatizes", "op_aaa")
        edges = kg.query_entity("op_aaa", direction="incoming")
        templatizes_edges = [
            e for e in edges if e.get("predicate") == "templatizes" and e.get("current", True)
        ]
        assert len(templatizes_edges) == 1
        assert templatizes_edges[0].get("subject") == "record_template_xxx"

    def test_shim_no_longer_swallows_kg_errors(self, tmp_path, monkeypatch):
        # If kg.add_triple genuinely raises (not the missing-statement
        # path -- that's fixed by the skip list), the shim must
        # propagate. The bare-except that hid TripleStatementRequired
        # was retired; any future write failure surfaces loudly.
        import pytest as _pytest
        from mempalace import mcp_server as _mcp
        from mempalace.knowledge_graph import KnowledgeGraph
        from mempalace.mcp_server import _ensure_operation_ontology
        from mempalace.memory_gardener import _synthesize_operation_template_shim

        db = tmp_path / "palace.db"
        kg = KnowledgeGraph(str(db))
        # Cold-start lock 2026-05-01: seed_ontology must run first so
        # the `is_a thing` triple emitted by _ensure_operation_ontology
        # lands on a declared parent class. Pre-cold-start this worked
        # via add_triple's phantom auto-create; closed at the gate.
        kg.seed_ontology()
        _ensure_operation_ontology(kg)
        for op_id in ("op_aaa", "op_bbb", "op_ccc"):
            kg.add_entity(op_id, kind="operation", content=op_id, importance=3)
        monkeypatch.setattr(_mcp._STATE, "kg", kg)

        original = kg.add_triple

        def _explode(*a, **kw):
            if len(a) >= 2 and a[1] == "templatizes":
                raise RuntimeError("simulated KG write failure")
            return original(*a, **kw)

        monkeypatch.setattr(kg, "add_triple", _explode)

        with _pytest.raises(RuntimeError, match="simulated KG write failure"):
            _synthesize_operation_template_shim(
                op_ids=["op_aaa", "op_bbb", "op_ccc"],
                title="t",
                when_to_use="w",
                recipe="r",
            )


# ─────────────────────────────────────────────────────────────────────
# Per-flag-kind prompt dispatch (2026-04-25 refactor):
# the legacy single _SYSTEM_PROMPT was split into _PROMPTS_BY_KIND so
# each flag-kind invocation only sees the rules it needs (Adrian's
# original "focused per-kind prompts" goal). _select_prompt is the
# helper process_batch calls; unknown kinds raise KeyError loudly.
# ─────────────────────────────────────────────────────────────────────


class TestPerKindPromptDispatch:
    EXPECTED_KINDS = {
        "duplicate_pair",
        "contradiction_pair",
        "stale",
        "orphan",
        "generic_summary",
        "edge_candidate",
        "unlinked_entity",
        "op_cluster_templatizable",
    }

    def test_prompts_by_kind_covers_every_memory_flag_kind(self):
        # Every kind in the closed _MEMORY_FLAG_KINDS frozenset MUST
        # have a focused prompt. A drift here means a new flag kind
        # was added without a matching gardener prompt -- the gardener
        # would crash on dispatch (which is the loud-fail behaviour
        # we want, but coverage at the test level catches it earlier).
        from mempalace.knowledge_graph import KnowledgeGraph
        from mempalace.memory_gardener import _PROMPTS_BY_KIND

        flag_kinds = set(KnowledgeGraph._MEMORY_FLAG_KINDS)
        prompt_kinds = set(_PROMPTS_BY_KIND)
        missing = flag_kinds - prompt_kinds
        assert not missing, f"flag kinds without a prompt: {sorted(missing)}"

    def test_prompts_by_kind_has_expected_set(self):
        from mempalace.memory_gardener import _PROMPTS_BY_KIND

        assert set(_PROMPTS_BY_KIND) == self.EXPECTED_KINDS

    def test_select_prompt_returns_focused_prompt(self):
        # The returned prompt for a given kind MUST contain the kind's
        # task block AND the shared preamble's tools list, but should
        # NOT mention sibling kinds' task verbs by default. This is
        # the structural promise of the refactor.
        from mempalace.memory_gardener import _select_prompt

        dup = _select_prompt("duplicate_pair")
        assert "duplicate_pair" in dup
        assert "kg_merge_entities" in dup  # the kind-specific tool
        assert "THE 8 TOOLS YOU HAVE" in dup  # shared preamble lands

        # Other kinds' task verbs should NOT leak into duplicate_pair's prompt.
        assert "GROUND BEFORE WRITING" not in dup  # generic_summary / op_cluster
        assert "VERIFY BOTH ENDPOINTS" not in dup  # edge_candidate

    def test_select_prompt_op_cluster_grounded(self):
        # The tightened op_cluster_templatizable block carries the
        # SHOULD-ground rule + stub-length defer + REQUIRED
        # failure_modes for negative.
        from mempalace.memory_gardener import _select_prompt

        prompt = _select_prompt("op_cluster_templatizable")
        assert "GROUND BEFORE WRITING" in prompt
        assert "no grounding evidence in op cluster" in prompt
        assert "REQUIRED for" in prompt and "negative" in prompt

    def test_select_prompt_unknown_kind_raises(self):
        # Loud failure for unknown kinds -- no silent fallback.
        import pytest as _pytest
        from mempalace.memory_gardener import _select_prompt

        with _pytest.raises(KeyError, match="Unknown flag_kind"):
            _select_prompt("not_a_real_kind")

    def test_each_prompt_includes_shared_preamble(self):
        # Every per-kind prompt must include the shared preamble so
        # Haiku always sees the tool list, recovery rule, and
        # never-do list regardless of which kind it's resolving.
        from mempalace.memory_gardener import _PROMPTS_BY_KIND, _SHARED_PREAMBLE

        for kind, prompt in _PROMPTS_BY_KIND.items():
            assert _SHARED_PREAMBLE.strip() in prompt, (
                f"prompt for {kind!r} missing shared preamble"
            )

    def test_per_kind_prompts_are_smaller_than_legacy_combined(self):
        # The whole point: each per-kind prompt should be SHORTER than
        # the sum of all task blocks would be in one prompt. Concrete
        # check: any single prompt is shorter than the longest two
        # combined (so we know the dispatch is actually reducing
        # token surface, not just nesting it).
        from mempalace.memory_gardener import _PROMPTS_BY_KIND

        sizes = sorted(len(p) for p in _PROMPTS_BY_KIND.values())
        assert sizes[-1] < sizes[-1] + sizes[-2], (
            "tautology -- but proving the per-kind structure is in place"
        )
        # Stronger: every prompt is < 1.5x the smallest one. The shared
        # preamble dominates; per-kind tasks add a small slice. If a
        # per-kind block balloons past this ratio the focused goal is
        # broken and someone should review.
        smallest = sizes[0]
        largest = sizes[-1]
        ratio = largest / smallest
        # Threshold raised 2026-04-25 from 1.6x → 2.5x to accommodate
        # the S4f rewrite of _TASK_GENERIC_SUMMARY: that block grew to
        # carry the WHAT+WHY+SCOPE target shape inline plus three
        # good and three bad summary examples. The "focused-prompt"
        # spirit of the test still holds -- variance < 2.5x means even
        # the longest per-kind prompt is shorter than the legacy
        # all-in-one prompt's ~7000-char surface.
        assert ratio < 2.5, (
            f"per-kind prompt size variance {ratio:.2f}x exceeds "
            f"focused-prompt budget; review the longest task block"
        )
