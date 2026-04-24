"""
test_op_memory.py — S1 operation-memory tier unit tests.

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
and is covered by the existing intent-system test harness — these tests
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
        # Deterministic fingerprint — same input yields same hash.
        out2 = hooks_cli._truncate_op_args({"content": long_val})
        assert out["content"] == out2["content"]

    def test_oversized_payload_gets_truncation_marker(self):
        # 10 fields each 500 chars = 5KB — over 2KB budget.
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
                    # No active_context_id — legacy cue shape.
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
        # rated_useful is the memory-relevance predicate — must NOT
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
        assert g["op_id"] == "op_edit_1"
        assert g["tool"] == "Edit"
        assert g["args_summary"] == "auth.py rate limiter"
        assert g["quality"] == 5

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
        assert b["tool"] == "Bash"

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
# Unit: _regex_copout_check — fast-path of the hybrid reason gate
# ─────────────────────────────────────────────────────────────────────


class TestRegexCopoutCheck:
    """Regex fast-path covers the cheap-and-obvious cop-outs.

    False positives on compound nouns that share a cop-out word as
    prefix (e.g. "skip-list", "skip-gram") are bugs — the 2026-04-24
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
            # Data-structure / technical terms containing "skip" as prefix —
            # MUST NOT be rejected by the regex. The 2026-04-24 tightening
            # of \\bskip(ped)?\\b → \\bskipped\\b specifically prevents this.
            "Memory about skip-list data structure",
            "skip-gram embedding model",
            # Legitimate short-but-real reasons
            "Memory about auth, intent was about DB",
            "Generic system description, background only",
            "Unrelated to current task (topic mismatch)",
            # Contains 'aborted' but not with 'running' — the pattern
            # requires both words together.
            "Aborted due to user decision after reading",
        ],
    )
    def test_does_not_false_positive_on_compound_nouns(self, reason):
        assert _regex_copout_check(reason) is False, (
            f"regex FALSE POSITIVE on {reason!r} — "
            "tighten the pattern to avoid compound-noun prefix matches"
        )

    def test_non_string_input_returns_false(self):
        assert _regex_copout_check(None) is False
        assert _regex_copout_check(123) is False
        assert _regex_copout_check(["don't know"]) is False


# ─────────────────────────────────────────────────────────────────────
# Unit: _semantic_copout_check — embedding-based second pass
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
        # — regex won't catch it, semantic should produce non-trivial
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
# Unit: S2 corrections — retrieve_past_operations walks superseded_by
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
        assert c["bad_op_id"] == "op_bad_bash"
        assert c["better_op_id"] == "op_good_read"
        assert c["bad"]["tool"] == "Bash"
        assert c["better"]["tool"] == "Read"
        assert c["better"]["quality"] == 5

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
