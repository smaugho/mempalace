"""
test_injection_gate.py -- InjectionGate end-to-end coverage.

Scope:
  * Prompt composition: session frame, parent frame, primary context,
    per-source rendering for memory/entity/triple items.
  * Filter dispatch: mock Anthropic client returns structured
    tool-use decisions; kept/dropped split matches those decisions.
  * Fail-open paths: missing SDK/key, malformed response, all-drops.
  * Skip-small-k: gate bypasses for K below min_items.
  * persist_drops: dropped items become rated_irrelevant via
    KnowledgeGraph.record_feedback with rater_kind='gate_llm'.
  * Triple drops land in triple_context_feedback (NOT as edges).
  * cwd anchor guard: project emitted only when an anchor file is
    present in cwd.

The gate never contacts the real API here -- a fake Anthropic client
is injected via ``InjectionGate(_client=...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from mempalace.injection_gate import (
    _PROJECT_ANCHORS,
    GateDecision,
    GateItem,
    InjectionGate,
    _detect_project_anchor,
    build_prompt,
    build_session_frame,
    persist_drops,
)


# ─────────────────────────────────────────────────────────────────────
# Fake Anthropic client
# ─────────────────────────────────────────────────────────────────────


@dataclass
class _FakeToolUseBlock:
    type: str
    name: str
    input: dict


@dataclass
class _FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 50
    # Prompt-caching fields mirror Anthropic's real usage shape.
    # cache_creation_input_tokens: tokens written to cache on a miss.
    # cache_read_input_tokens: tokens served from cache on a hit.
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class _FakeResponse:
    content: list
    usage: _FakeUsage


class _FakeMessages:
    def __init__(self, decisions_per_call):
        self._decisions_per_call = decisions_per_call
        self._call_count = 0
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        payload = self._decisions_per_call[self._call_count]
        self._call_count += 1
        if payload is _RAISE:
            raise RuntimeError("simulated API error")
        if payload is _EMPTY_TOOL_BLOCK:
            return _FakeResponse(
                content=[_FakeToolUseBlock(type="tool_use", name="gate_decisions", input={})],
                usage=_FakeUsage(),
            )
        if payload is _NO_TOOL_BLOCK:
            return _FakeResponse(content=[], usage=_FakeUsage())
        # payload may be a bare decisions list (pre-flags tests) or a
        # dict {"decisions": [...], "flags": [...]} for flag tests.
        tool_input = payload if isinstance(payload, dict) else {"decisions": payload}
        return _FakeResponse(
            content=[
                _FakeToolUseBlock(
                    type="tool_use",
                    name="gate_decisions",
                    input=tool_input,
                )
            ],
            usage=_FakeUsage(),
        )


class _FakeClient:
    def __init__(self, decisions_per_call):
        self.messages = _FakeMessages(decisions_per_call)


_RAISE = object()
_EMPTY_TOOL_BLOCK = object()
_NO_TOOL_BLOCK = object()


# ─────────────────────────────────────────────────────────────────────
# Sample data helpers
# ─────────────────────────────────────────────────────────────────────


def _sample_items(n=5):
    items = []
    for i in range(n):
        items.append(
            GateItem(
                id=f"mem_{i}",
                source="memory",
                text=f"memory content {i}",
                channel="A",
                rank=i + 1,
                score=0.8 - i * 0.1,
                extra={"summary": f"summary {i}"},
            )
        )
    return items


def _sample_mixed_items():
    return [
        GateItem(
            id="record_a",
            source="memory",
            text="Decision: use JWT for auth.",
            channel="A",
            rank=1,
            score=0.9,
            extra={"summary": "JWT decision"},
        ),
        GateItem(
            id="ent_auth_service",
            source="entity",
            text="Authentication service for user logins.",
            channel="B",
            rank=2,
            score=0.7,
            extra={"name": "auth_service", "kind": "entity"},
        ),
        GateItem(
            id="t_alice_knows_bob_abc",
            source="triple",
            text="Alice knows Bob.",
            channel="A",
            rank=3,
            score=0.6,
            extra={
                "subject": "alice",
                "predicate": "knows",
                "object": "bob",
                "confidence": 1.0,
            },
        ),
    ]


# ─────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────


class TestBuildPrompt:
    def test_renders_primary_context(self):
        prompt = build_prompt(
            primary_context={
                "source": "declare_intent",
                "queries": ["auth hardening", "jwt expiry"],
                "keywords": ["auth", "jwt"],
                "entities": ["AuthService"],
            },
            items=_sample_items(1),
        )
        assert "PRIMARY CONTEXT" in prompt
        assert "auth hardening" in prompt
        assert "AuthService" in prompt
        assert "declare_intent" in prompt

    def test_renders_session_frame_when_present(self):
        prompt = build_prompt(
            primary_context={"queries": ["x", "y"]},
            items=_sample_items(1),
            session_frame={"agent": "ga_agent", "project": "mempalace"},
        )
        assert "SESSION FRAME" in prompt
        assert "ga_agent" in prompt
        assert "mempalace" in prompt

    def test_omits_session_frame_when_empty(self):
        prompt = build_prompt(
            primary_context={"queries": ["x", "y"]},
            items=_sample_items(1),
            session_frame={},
        )
        assert "SESSION FRAME" not in prompt

    def test_renders_parent_frame_for_nested_retrieval(self):
        prompt = build_prompt(
            primary_context={"queries": ["op-specific cue"]},
            items=_sample_items(1),
            parent_intent={
                "intent_type": "research",
                "subject": "retrieval_score_cutoff",
                "query": "what does the literature say",
            },
        )
        assert "PARENT FRAME" in prompt
        assert "research" in prompt
        assert "retrieval_score_cutoff" in prompt

    def test_renders_triple_items_with_spo(self):
        items = _sample_mixed_items()
        prompt = build_prompt(
            primary_context={"queries": ["a", "b"]},
            items=items,
        )
        assert "statement: Alice knows Bob." in prompt
        assert "subject: alice" in prompt
        assert "predicate: knows" in prompt
        assert "object: bob" in prompt

    def test_renders_entity_items_with_name_and_kind(self):
        items = _sample_mixed_items()
        prompt = build_prompt(
            primary_context={"queries": ["a", "b"]},
            items=items,
        )
        assert "name: auth_service" in prompt
        assert "kind: entity" in prompt
        assert "Authentication service" in prompt

    def test_renders_memory_items_with_summary(self):
        items = _sample_mixed_items()
        prompt = build_prompt(
            primary_context={"queries": ["a", "b"]},
            items=items,
        )
        assert "summary: JWT decision" in prompt
        assert "Decision: use JWT" in prompt

    def test_includes_order_instruction(self):
        prompt = build_prompt(
            primary_context={"queries": ["x", "y"]},
            items=_sample_items(2),
        )
        assert "SAME ORDER" in prompt
        assert "exactly once" in prompt

    def test_truncates_very_long_items(self):
        huge = "x" * 20000
        items = [
            GateItem(
                id="mem_huge",
                source="memory",
                text=huge,
                channel="A",
                rank=1,
                score=0.5,
                extra={},
            )
        ]
        prompt = build_prompt(primary_context={"queries": ["a", "b"]}, items=items)
        assert "[truncated]" in prompt
        # Prompt must not contain the full 20000 xs.
        assert len(prompt) < 20000


# ─────────────────────────────────────────────────────────────────────
# Filter dispatch
# ─────────────────────────────────────────────────────────────────────


class TestFilterDispatch:
    def test_all_keep(self):
        items = _sample_items(5)
        client = _FakeClient(
            [[{"id": it.id, "action": "keep", "reasoning": "on topic"} for it in items]]
        )
        gate = InjectionGate(_client=client)
        result = gate.filter(
            primary_context={"queries": ["a", "b"], "keywords": ["x", "y"]},
            items=items,
        )
        assert len(result.kept) == 5
        assert result.dropped == []
        assert result.gate_status == {"state": "ok"}

    def test_system_prompt_sent_as_cacheable_block(self):
        """Caching contract: system arrives as a list of content blocks and
        the last block carries cache_control. Without this shape, Anthropic
        serves everything as fresh input (90% more expensive, slower)."""
        items = _sample_items(3)
        client = _FakeClient([[{"id": it.id, "action": "keep", "reasoning": "ok"} for it in items]])
        gate = InjectionGate(_client=client)
        gate.filter(
            primary_context={"queries": ["a", "b"], "keywords": ["x", "y"]},
            items=items,
        )
        kwargs = client.messages.last_kwargs
        assert kwargs is not None
        system = kwargs.get("system")
        assert isinstance(system, list), "system must be a list of content blocks for caching"
        assert system, "system block list must not be empty"
        last = system[-1]
        assert last.get("type") == "text"
        assert isinstance(last.get("text"), str) and last["text"]
        assert last.get("cache_control") == {"type": "ephemeral"}

    def test_tool_schema_marked_cacheable(self):
        """Caching contract: the gate_decisions tool carries cache_control so
        its 400-ish-token schema is cached alongside the system prompt."""
        items = _sample_items(3)
        client = _FakeClient([[{"id": it.id, "action": "keep", "reasoning": "ok"} for it in items]])
        gate = InjectionGate(_client=client)
        gate.filter(
            primary_context={"queries": ["a", "b"], "keywords": ["x", "y"]},
            items=items,
        )
        kwargs = client.messages.last_kwargs
        tools = kwargs.get("tools")
        assert isinstance(tools, list) and len(tools) == 1
        tool = tools[0]
        assert tool.get("name") == "gate_decisions"
        assert tool.get("cache_control") == {"type": "ephemeral"}
        # Tool schema fields still present under the cache_control wrapper.
        assert "input_schema" in tool
        assert "description" in tool

    def test_cache_usage_threads_through_to_result_and_log(self):
        """Cache-hit/miss counts from Anthropic's usage object must land on
        GateResult so apply_gate can emit them to gate_log.jsonl."""
        items = _sample_items(3)
        decisions = [{"id": it.id, "action": "keep", "reasoning": "ok"} for it in items]
        client = _FakeClient([decisions])
        # Replace the default usage on the one response the fake will emit.
        # Simulating a cache HIT: high cache_read, zero cache_creation.
        hit_usage = _FakeUsage(
            input_tokens=4200,
            output_tokens=180,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=1800,
        )
        real_create = client.messages.create

        def wrapped_create(**kwargs):
            resp = real_create(**kwargs)
            resp.usage = hit_usage
            return resp

        client.messages.create = wrapped_create
        gate = InjectionGate(_client=client)
        result = gate.filter(
            primary_context={"queries": ["a", "b"], "keywords": ["x", "y"]},
            items=items,
        )
        assert result.cache_read_input_tokens == 1800
        assert result.cache_creation_input_tokens == 0
        assert result.judge_tokens_in == 4200
        assert result.judge_tokens_out == 180

    def test_mixed_keep_and_drop(self):
        items = _sample_items(5)
        decisions = []
        for i, it in enumerate(items):
            decisions.append(
                {
                    "id": it.id,
                    "action": "drop" if i % 2 == 1 else "keep",
                    "reasoning": f"decision {i}",
                }
            )
        client = _FakeClient([decisions])
        gate = InjectionGate(_client=client)
        result = gate.filter(
            primary_context={"queries": ["a", "b"], "keywords": ["x", "y"]},
            items=items,
        )
        assert len(result.kept) == 3
        assert len(result.dropped) == 2
        kept_ids = {it.id for it in result.kept}
        dropped_ids = {it.id for (it, _) in result.dropped}
        assert kept_ids == {"mem_0", "mem_2", "mem_4"}
        assert dropped_ids == {"mem_1", "mem_3"}

    def test_missing_decision_defaults_to_keep(self):
        """Defensive: if judge skips an id, item passes through."""
        items = _sample_items(3)
        # Only rate items[0]; items[1] and [2] missing.
        decisions = [{"id": items[0].id, "action": "drop", "reasoning": "off topic"}]
        client = _FakeClient([decisions])
        gate = InjectionGate(_client=client)
        result = gate.filter(
            primary_context={"queries": ["a", "b"], "keywords": ["x", "y"]},
            items=items,
        )
        # 1 dropped, 2 default-kept.
        assert len(result.dropped) == 1
        assert len(result.kept) == 2

    def test_hallucinated_id_is_ignored(self):
        items = _sample_items(3)
        decisions = [
            {"id": items[0].id, "action": "keep", "reasoning": "ok"},
            {"id": "NOT_A_REAL_ID", "action": "drop", "reasoning": "hallucinated"},
            {"id": items[1].id, "action": "drop", "reasoning": "off topic"},
        ]
        client = _FakeClient([decisions])
        gate = InjectionGate(_client=client)
        result = gate.filter(
            primary_context={"queries": ["a", "b"], "keywords": ["x", "y"]},
            items=items,
        )
        # items[2] is missing → default keep; NOT_A_REAL_ID is dropped.
        assert len(result.kept) == 2
        assert len(result.dropped) == 1
        assert result.dropped[0][0].id == items[1].id


# ─────────────────────────────────────────────────────────────────────
# Fail-open paths
# ─────────────────────────────────────────────────────────────────────


class TestFailOpen:
    def test_no_client_skipped_quietly(self):
        """No API key / SDK: state is 'skipped_no_client', NOT
        'degraded'. Happy-path callers treat this as silent opt-out
        and emit no gate_status in their response."""
        gate = InjectionGate(_client=None)
        # Force client init to return None by bypassing lazy init.
        gate._client_initialized = True
        result = gate.filter(
            primary_context={"queries": ["a", "b"], "keywords": ["x", "y"]},
            items=_sample_items(5),
        )
        assert len(result.kept) == 5
        assert result.dropped == []
        assert result.gate_status["state"] == "skipped_no_client"

    def test_api_error_triggers_retry_then_fail_open(self):
        client = _FakeClient([_RAISE, _RAISE])  # both attempts fail
        gate = InjectionGate(_client=client, max_retries=2)
        result = gate.filter(
            primary_context={"queries": ["a", "b"], "keywords": ["x", "y"]},
            items=_sample_items(5),
        )
        assert len(result.kept) == 5
        assert result.gate_status["state"] == "degraded"
        assert "judge_failed_after" in result.gate_status["reason"]

    def test_api_error_then_success_returns_filtered(self):
        items = _sample_items(3)
        good = [
            {"id": items[0].id, "action": "keep", "reasoning": "x"},
            {"id": items[1].id, "action": "drop", "reasoning": "y"},
            {"id": items[2].id, "action": "keep", "reasoning": "z"},
        ]
        client = _FakeClient([_RAISE, good])  # first fails, second works
        gate = InjectionGate(_client=client, max_retries=2)
        result = gate.filter(
            primary_context={"queries": ["a", "b"], "keywords": ["x", "y"]},
            items=items,
        )
        assert result.gate_status["state"] == "ok"
        assert len(result.kept) == 2
        assert len(result.dropped) == 1

    def test_malformed_tool_block_fails_open(self):
        client = _FakeClient([_EMPTY_TOOL_BLOCK, _EMPTY_TOOL_BLOCK])
        gate = InjectionGate(_client=client)
        result = gate.filter(
            primary_context={"queries": ["a", "b"], "keywords": ["x", "y"]},
            items=_sample_items(5),
        )
        assert result.gate_status["state"] == "degraded"
        assert len(result.kept) == 5

    def test_no_tool_block_fails_open(self):
        client = _FakeClient([_NO_TOOL_BLOCK, _NO_TOOL_BLOCK])
        gate = InjectionGate(_client=client)
        result = gate.filter(
            primary_context={"queries": ["a", "b"], "keywords": ["x", "y"]},
            items=_sample_items(5),
        )
        assert result.gate_status["state"] == "degraded"


# ─────────────────────────────────────────────────────────────────────
# K-based skip conditions
# ─────────────────────────────────────────────────────────────────────


class TestSkipSmallK:
    def test_empty_items_skipped(self):
        gate = InjectionGate(_client=_FakeClient([]))
        result = gate.filter(primary_context={"queries": ["a", "b"]}, items=[])
        assert result.kept == []
        assert result.gate_status["state"] == "skipped_empty"

    def test_k_below_min_is_passthrough(self):
        items = _sample_items(2)
        client = _FakeClient([])  # should not be called
        gate = InjectionGate(_client=client, min_items=3)
        result = gate.filter(
            primary_context={"queries": ["a", "b"], "keywords": ["x", "y"]},
            items=items,
        )
        assert len(result.kept) == 2
        assert result.gate_status["state"] == "skipped_small_k"
        assert client.messages.last_kwargs is None


# ─────────────────────────────────────────────────────────────────────
# persist_drops -- end-to-end writeback through record_feedback.
# ─────────────────────────────────────────────────────────────────────


class TestPersistDrops:
    def test_persists_entity_drop_as_rated_irrelevant_edge(self, kg):
        # Declare the context entity so add_rated_edge has a real row
        # to write against.
        kg.add_entity("ctx_gate_a", kind="entity", content="gate ctx")
        kg.add_entity("mem_x", kind="record", content="a memory")

        items = [
            GateItem(
                id="mem_x",
                source="memory",
                text="content",
                channel="A",
                rank=1,
                score=0.5,
                extra={"summary": "s"},
            )
        ]
        decisions = [GateDecision(id="mem_x", action="drop", reasoning="off-topic")]
        dropped = [(items[0], decisions[0])]

        n = persist_drops(kg, context_id="ctx_gate_a", dropped=dropped)
        assert n == 1

        conn = kg._conn()
        sub = kg._entity_id("ctx_gate_a")
        obj = kg._entity_id("mem_x")
        row = conn.execute(
            "SELECT predicate FROM triples WHERE subject = ? AND object = ? AND valid_to IS NULL",
            (sub, obj),
        ).fetchone()
        assert row is not None
        assert row["predicate"] == "rated_irrelevant"

    def test_persists_triple_drop_in_triple_feedback_table(self, kg):
        """Dropping a triple item must NOT create a phantom entity;
        the feedback lands in triple_context_feedback with
        rater_kind='gate_llm'."""
        kg.add_entity("alice", kind="entity", content="Alice")
        kg.add_entity("bob", kind="entity", content="Bob")
        tid = kg.add_triple("alice", "knows", "bob", statement="Alice knows Bob.")

        items = [
            GateItem(
                id=tid,
                source="triple",
                text="Alice knows Bob.",
                channel="A",
                rank=1,
                score=0.5,
                extra={"subject": "alice", "predicate": "knows", "object": "bob"},
            )
        ]
        decisions = [GateDecision(id=tid, action="drop", reasoning="off-topic")]
        dropped = [(items[0], decisions[0])]

        n = persist_drops(
            kg, context_id="ctx_gate_b", dropped=dropped, rater_id="claude-haiku-gate"
        )
        assert n == 1

        conn = kg._conn()
        rows = conn.execute(
            "SELECT kind, rater_kind, rater_id, reason "
            "FROM triple_context_feedback "
            "WHERE context_id = ? AND triple_id = ? AND valid_to IS NULL",
            ("ctx_gate_b", tid),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["kind"] == "rated_irrelevant"
        assert rows[0]["rater_kind"] == "gate_llm"
        assert rows[0]["rater_id"] == "claude-haiku-gate"
        assert "[gate]" in rows[0]["reason"]

        # Ensure NO phantom entity with triple_id-like name was created.
        phantom = conn.execute("SELECT 1 FROM entities WHERE id = ?", (tid,)).fetchone()
        assert phantom is None, "phantom entity leaked from triple drop"

    def test_empty_dropped_list_is_noop(self, kg):
        assert persist_drops(kg, context_id="ctx", dropped=[]) == 0

    def test_missing_context_id_is_noop(self, kg):
        items = [
            GateItem(
                id="mem_x",
                source="memory",
                text="c",
                channel="A",
                rank=1,
                score=0.5,
                extra={},
            )
        ]
        dropped = [(items[0], GateDecision(id="mem_x", action="drop", reasoning="x"))]
        assert persist_drops(kg, context_id="", dropped=dropped) == 0


# ─────────────────────────────────────────────────────────────────────
# cwd anchor guard
# ─────────────────────────────────────────────────────────────────────


class TestCwdAnchorGuard:
    def test_anchor_present_returns_project_name(self, tmp_dir):
        p = Path(tmp_dir) / "myproj"
        p.mkdir()
        (p / "pyproject.toml").touch()
        assert _detect_project_anchor(str(p)) == "myproj"

    def test_no_anchor_returns_none(self, tmp_dir):
        p = Path(tmp_dir) / "notaproject"
        p.mkdir()
        # No anchor file.
        assert _detect_project_anchor(str(p)) is None

    def test_missing_directory_returns_none(self):
        assert _detect_project_anchor("/nonexistent/path/xyz") is None

    def test_none_input_returns_none(self):
        assert _detect_project_anchor(None) is None

    def test_all_anchors_recognised(self, tmp_dir):
        for anchor in _PROJECT_ANCHORS:
            sub = Path(tmp_dir) / f"proj_{anchor.replace('.', '_')}"
            sub.mkdir(exist_ok=True)
            target = sub / anchor
            # .git is usually a directory; the rest are files.
            if anchor == ".git":
                target.mkdir(exist_ok=True)
            else:
                target.touch()
            assert _detect_project_anchor(str(sub)) == sub.name, (
                f"anchor {anchor!r} should be recognised"
            )

    def test_build_session_frame_includes_project_when_anchor_present(self, tmp_dir):
        p = Path(tmp_dir) / "mempalace"
        p.mkdir()
        (p / "pyproject.toml").touch()
        frame = build_session_frame(agent="ga_agent", cwd=str(p))
        assert frame == {"agent": "ga_agent", "project": "mempalace"}

    def test_build_session_frame_omits_project_without_anchor(self, tmp_dir):
        p = Path(tmp_dir) / "random"
        p.mkdir()
        frame = build_session_frame(agent="ga_agent", cwd=str(p))
        assert frame == {"agent": "ga_agent"}
        assert "project" not in frame


# ─────────────────────────────────────────────────────────────────────
# Flag emission, parsing, and persistence (Phase A of memory_gardener)
# ─────────────────────────────────────────────────────────────────────


def _decisions_for(items):
    """Boilerplate: one keep decision per item."""
    return [{"id": it.id, "action": "keep", "reasoning": "ok"} for it in items]


class TestFlagParsing:
    def test_absent_flags_key_parses_as_empty(self):
        """Old-shape responses (decisions only, no flags key) must
        keep working -- flags list defaults to empty."""
        items = _sample_items(3)
        client = _FakeClient([_decisions_for(items)])
        gate = InjectionGate(_client=client)
        result = gate.filter(
            primary_context={"queries": ["a", "b"], "keywords": ["x", "y"]},
            items=items,
        )
        assert result.flags == []

    def test_well_formed_flags_flow_through(self):
        items = _sample_items(3)
        flags = [
            {
                "kind": "duplicate_pair",
                "memory_ids": [items[0].id, items[1].id],
                "detail": "same fact stated twice",
            },
            {
                "kind": "orphan",
                "memory_ids": [items[2].id],
                "detail": "no entities attached",
            },
        ]
        client = _FakeClient([{"decisions": _decisions_for(items), "flags": flags}])
        gate = InjectionGate(_client=client)
        result = gate.filter(
            primary_context={"queries": ["a", "b"], "keywords": ["x", "y"]},
            items=items,
        )
        assert len(result.flags) == 2
        kinds = {f["kind"] for f in result.flags}
        assert kinds == {"duplicate_pair", "orphan"}

    def test_malformed_flag_entries_dropped(self):
        items = _sample_items(3)
        flags = [
            {"kind": "bogus_kind", "memory_ids": [items[0].id], "detail": "x"},
            {"kind": "orphan", "memory_ids": "not-a-list", "detail": "x"},
            {"kind": "orphan", "memory_ids": [], "detail": "x"},
            "totally not a dict",
            {"kind": "stale", "memory_ids": [items[0].id], "detail": "real one"},
        ]
        client = _FakeClient([{"decisions": _decisions_for(items), "flags": flags}])
        gate = InjectionGate(_client=client)
        result = gate.filter(
            primary_context={"queries": ["a", "b"], "keywords": ["x", "y"]},
            items=items,
        )
        assert len(result.flags) == 1
        assert result.flags[0]["kind"] == "stale"

    def test_flags_are_optional_on_api_errors_too(self):
        """When the judge fails and we fail-open, flags list is empty."""
        items = _sample_items(3)
        client = _FakeClient([_RAISE, _RAISE])
        gate = InjectionGate(_client=client, max_retries=2)
        result = gate.filter(
            primary_context={"queries": ["a", "b"], "keywords": ["x", "y"]},
            items=items,
        )
        assert result.flags == []


class TestKnowledgeGraphFlagStore:
    """Direct coverage of the memory_flags / memory_gardener_runs
    tables and their read/write methods on KnowledgeGraph."""

    def test_migration_019_creates_memory_flags(self, kg):
        conn = kg._conn()
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_flags'"
        ).fetchone()
        assert row is not None

    def test_migration_020_creates_gardener_runs(self, kg):
        conn = kg._conn()
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_gardener_runs'"
        ).fetchone()
        assert row is not None

    def test_record_memory_flags_inserts_row(self, kg):
        n = kg.record_memory_flags(
            [
                {
                    "kind": "duplicate_pair",
                    "memory_ids": ["rec_a", "rec_b"],
                    "detail": "same fact",
                    "context_id": "ctx_gf",
                }
            ]
        )
        assert n == 1
        rows = kg.list_pending_flags()
        assert len(rows) == 1
        row = rows[0]
        assert row["kind"] == "duplicate_pair"
        assert row["memory_ids"] == ["rec_a", "rec_b"]
        assert row["context_id"] == "ctx_gf"

    def test_record_skips_bad_kinds(self, kg):
        n = kg.record_memory_flags(
            [
                {"kind": "bogus", "memory_ids": ["a"], "context_id": "c"},
                {"kind": "stale", "memory_ids": ["b"], "context_id": "c"},
            ]
        )
        assert n == 1

    def test_dedup_bumps_attempted_count(self, kg):
        flag = {
            "kind": "duplicate_pair",
            "memory_ids": ["rec_a", "rec_b"],
            "context_id": "ctx_dup",
        }
        kg.record_memory_flags([flag])
        kg.record_memory_flags([flag])
        kg.record_memory_flags([flag])
        rows = kg.list_pending_flags()
        # Three observations collapsed into one row with bumped counter.
        assert len(rows) == 1
        # First write = 0, two re-observations = +2.
        assert rows[0]["attempted_count"] == 2

    def test_pair_order_agnostic_dedup(self, kg):
        """duplicate_pair(A, B) and duplicate_pair(B, A) collapse."""
        kg.record_memory_flags(
            [{"kind": "duplicate_pair", "memory_ids": ["a", "b"], "context_id": "c"}]
        )
        kg.record_memory_flags(
            [{"kind": "duplicate_pair", "memory_ids": ["b", "a"], "context_id": "c"}]
        )
        rows = kg.list_pending_flags()
        assert len(rows) == 1

    def test_count_pending_flags(self, kg):
        assert kg.count_pending_flags() == 0
        kg.record_memory_flags([{"kind": "orphan", "memory_ids": ["x"], "context_id": "c"}])
        assert kg.count_pending_flags() == 1

    def test_mark_flag_resolved(self, kg):
        kg.record_memory_flags([{"kind": "orphan", "memory_ids": ["x"], "context_id": "c"}])
        row = kg.list_pending_flags()[0]
        ok = kg.mark_flag_resolved(row["id"], "pruned", note="low signal")
        assert ok is True
        # Now unresolved list drops it.
        assert kg.list_pending_flags() == []
        # Direct SELECT verifies resolution was recorded.
        conn = kg._conn()
        r = conn.execute(
            "SELECT resolution, resolution_note FROM memory_flags WHERE id = ?",
            (row["id"],),
        ).fetchone()
        assert r["resolution"] == "pruned"
        assert r["resolution_note"] == "low signal"

    def test_mark_flag_rejects_unknown_resolution(self, kg):
        kg.record_memory_flags([{"kind": "orphan", "memory_ids": ["x"], "context_id": "c"}])
        row = kg.list_pending_flags()[0]
        with pytest.raises(ValueError):
            kg.mark_flag_resolved(row["id"], "made_up_outcome")

    def test_attempted_count_ceiling_excludes_from_pending(self, kg):
        flag = {"kind": "orphan", "memory_ids": ["x"], "context_id": "c"}
        kg.record_memory_flags([flag])
        row = kg.list_pending_flags()[0]
        # Force attempted_count to 3.
        kg.bump_flag_attempt(row["id"])
        kg.bump_flag_attempt(row["id"])
        kg.bump_flag_attempt(row["id"])
        # Pending list now filters it out (frozen at attempted_count >= 3).
        assert kg.list_pending_flags() == []

    def test_gardener_run_lifecycle(self, kg):
        run_id = kg.start_gardener_run(gardener_model="claude-haiku-4-5")
        kg.finish_gardener_run(
            run_id,
            flag_ids=[1, 2, 3],
            counters={
                "merges": 1,
                "invalidations": 0,
                "links_created": 2,
                "edges_proposed": 1,
                "summary_rewrites": 0,
                "prunes": 0,
                "deferrals": 1,
                "no_action": 0,
            },
            subprocess_exit_code=0,
            errors="",
        )
        conn = kg._conn()
        row = conn.execute("SELECT * FROM memory_gardener_runs WHERE id = ?", (run_id,)).fetchone()
        assert row is not None
        assert row["flags_processed"] == 3
        assert row["merges"] == 1
        assert row["links_created"] == 2
        assert row["edges_proposed"] == 1
        assert row["deferrals"] == 1
        assert row["subprocess_exit_code"] == 0
        assert row["completed_ts"] is not None


class TestBgQualityPass:
    """v3.7.2 Slice 2 (Adrian directive 2026-05-16, Option 3
    architecture): after the lean foreground gate (Slice 1) returns,
    apply_gate spawns a background quality pass on _BG_QUALITY_EXECUTOR
    that runs a SECOND Haiku call against the full prompt + full
    schema and writes the resulting flag rows asynchronously. These
    tests monkeypatch the executor's submit() to call the closure
    synchronously so effects are observable inline.
    """

    def test_bg_pass_writes_flags(self, kg, monkeypatch):
        # Foreground response: lean schema (decisions only, all keep).
        # Background response: full schema (decisions + flags).
        items = _sample_items(3)
        fg_decisions = _decisions_for(items)
        bg_flags = [
            {
                "kind": "duplicate_pair",
                "memory_ids": [items[0].id, items[1].id],
                "detail": "two records narrate the same ship event",
            }
        ]
        client = _FakeClient(
            [
                {"decisions": fg_decisions},  # foreground lean
                {"decisions": fg_decisions, "flags": bg_flags},  # bg full
            ]
        )
        gate = InjectionGate(_client=client)

        # Run the executor's submitted callable synchronously so the
        # bg pass's effects are observable when apply_gate returns.
        from mempalace import injection_gate as _ig

        def _sync_submit(fn, *args, **kwargs):
            fn(*args, **kwargs)
            return None  # apply_gate ignores the return value

        monkeypatch.setattr(_ig._BG_QUALITY_EXECUTOR, "submit", _sync_submit)

        memories = [{"id": it.id, "text": it.text} for it in items]
        filtered, status, _gate_report = _ig.apply_gate(
            memories=memories,
            combined_meta={},
            primary_context={"queries": ["a", "b"], "keywords": ["x", "y"]},
            context_id="ctx_bg_writes",
            kg=kg,
            agent="test_agent",
            gate=gate,
        )
        assert status is None
        # Foreground returned all items as kept.
        assert {m["id"] for m in filtered} == {it.id for it in items}
        # BG pass wrote the duplicate_pair flag with the right context_id.
        rows = kg.list_pending_flags()
        assert len(rows) == 1
        assert rows[0]["kind"] == "duplicate_pair"
        assert rows[0]["context_id"] == "ctx_bg_writes"
        # FakeClient saw two messages.create calls (1 fg + 1 bg).
        assert client.messages._call_count == 2

    def test_bg_pass_failure_isolated(self, kg, monkeypatch):
        # Foreground response succeeds; bg pass raises.
        items = _sample_items(3)
        fg_decisions = _decisions_for(items)
        client = _FakeClient(
            [
                {"decisions": fg_decisions},  # foreground succeeds
                _RAISE,  # bg pass raises -- must be caught silently
            ]
        )
        gate = InjectionGate(_client=client)

        from mempalace import injection_gate as _ig

        def _sync_submit(fn, *args, **kwargs):
            fn(*args, **kwargs)
            return None

        monkeypatch.setattr(_ig._BG_QUALITY_EXECUTOR, "submit", _sync_submit)

        memories = [{"id": it.id, "text": it.text} for it in items]
        filtered, status, _gate_report = _ig.apply_gate(
            memories=memories,
            combined_meta={},
            primary_context={"queries": ["a", "b"], "keywords": ["x", "y"]},
            context_id="ctx_bg_fails",
            kg=kg,
            agent="test_agent",
            gate=gate,
        )
        # Foreground still returns normally.
        assert status is None
        assert {m["id"] for m in filtered} == {it.id for it in items}
        # No flags written -- bg pass raised before record_memory_flags.
        assert kg.list_pending_flags() == []

    def test_bg_pass_skipped_when_env_disabled(self, kg, monkeypatch):
        # MEMPALACE_BG_QUALITY=0 -> executor.submit NOT called.
        items = _sample_items(3)
        fg_decisions = _decisions_for(items)
        client = _FakeClient([{"decisions": fg_decisions}])
        gate = InjectionGate(_client=client)

        from mempalace import injection_gate as _ig

        submit_calls = []

        def _spy_submit(fn, *args, **kwargs):  # pragma: no cover -- asserted not called
            submit_calls.append(fn)
            return None

        monkeypatch.setattr(_ig._BG_QUALITY_EXECUTOR, "submit", _spy_submit)
        monkeypatch.setenv("MEMPALACE_BG_QUALITY", "0")

        memories = [{"id": it.id, "text": it.text} for it in items]
        filtered, status, _gate_report = _ig.apply_gate(
            memories=memories,
            combined_meta={},
            primary_context={"queries": ["a", "b"], "keywords": ["x", "y"]},
            context_id="ctx_bg_disabled",
            kg=kg,
            agent="test_agent",
            gate=gate,
        )
        assert status is None
        assert submit_calls == []
        # Only the foreground call happened.
        assert client.messages._call_count == 1


class TestApplyGatePersistsFlags:
    """apply_gate should write emitted flags to memory_flags, scoped
    to the active context, with rater_model = gate.model."""

    def test_flags_persisted_after_filter(self, kg):
        items = _sample_items(3)
        flags = [
            {
                "kind": "contradiction_pair",
                "memory_ids": [items[0].id, items[1].id],
                "detail": "two dates conflict",
            }
        ]
        client = _FakeClient([{"decisions": _decisions_for(items), "flags": flags}])
        gate = InjectionGate(_client=client)

        from mempalace.injection_gate import apply_gate

        memories = [{"id": it.id, "text": it.text} for it in items]
        # apply_gate returns (filtered, status, gate_report) -- the
        # third element is per-call telemetry (timings, sizes, etc.)
        # added 2026-05; tests that only care about filter behaviour
        # bind it to _gate_report and ignore the contents.
        filtered, status, _gate_report = apply_gate(
            memories=memories,
            combined_meta={},
            primary_context={"queries": ["a", "b"], "keywords": ["x", "y"]},
            context_id="ctx_apply",
            kg=kg,
            agent="test_agent",
            gate=gate,
        )
        assert status is None  # 'ok' is happy-path, no status surfaced
        rows = kg.list_pending_flags()
        assert len(rows) == 1
        assert rows[0]["kind"] == "contradiction_pair"
        assert rows[0]["context_id"] == "ctx_apply"
        assert rows[0]["memory_ids"] == [items[0].id, items[1].id]


# ─────────────────────────────────────────────────────────────────────
# UTF-16 lone-surrogate regression
# ─────────────────────────────────────────────────────────────────────
#
# Background: lone UTF-16 low-surrogates (U+D800-U+DFFF) slip into memory
# text from upstream pipelines that decoded bytes with errors='surrogate-
# escape' or read Windows wide-char APIs. Two downstream sinks reject
# them:
#   * chroma's ONNX tokenizer raises "TextInputSequence must be str"
#   * Anthropic's HTTP client raises UnicodeEncodeError at JSON-serialize
#
# Both crashes surfaced as the injection gate fail-opening and dumping
# K=20 unfiltered memories per turn (see palace entity
# gate_degraded_judge_failed_with_unicode_encode_erro).
#
# Defense in depth: strip on the write side in config.sanitize_content
# so NEW records can't carry the contamination; scrub on the read side
# in injection_gate.apply_gate so legacy records written before the
# write-side fix still flow through cleanly. These tests lock both.


class TestSanitizeContentStripsSurrogates:
    """Write-side contract: config.sanitize_content folds lone UTF-16
    surrogates to empty before returning. Fold-to-empty (not replace
    with '?') so clean prose stays clean."""

    def test_lone_low_surrogate_folded_to_empty(self):
        from mempalace.config import sanitize_content

        tainted = "hello \udc9d world"
        assert sanitize_content(tainted) == "hello  world"

    def test_lone_high_surrogate_folded_to_empty(self):
        from mempalace.config import sanitize_content

        tainted = "prefix \ud83d suffix"
        assert sanitize_content(tainted) == "prefix  suffix"

    def test_surrogate_range_boundaries_folded(self):
        from mempalace.config import sanitize_content

        tainted = "a\ud800b\udfffc"
        assert sanitize_content(tainted) == "abc"

    def test_clean_ascii_is_no_op(self):
        from mempalace.config import sanitize_content

        clean = "plain ASCII content with no surrogates"
        assert sanitize_content(clean) == clean

    def test_strip_runs_before_punct_normalize(self):
        """Surrogate strip + em-dash normalize must compose: a string
        mixing both kinds of contamination ends up ASCII-clean with no
        surrogates. Order matters because _normalize_punct assumes
        already-sanitized UTF-8."""
        from mempalace.config import sanitize_content

        tainted = "em\u2014dash and \udc9d surrogate"
        out = sanitize_content(tainted)
        assert "\udc9d" not in out
        assert out == "em--dash and  surrogate"

    def test_sanitized_output_encodes_cleanly(self):
        """Historically the raw form raised UnicodeEncodeError at
        chroma/Anthropic; sanitized form must encode without error."""
        from mempalace.config import sanitize_content

        tainted = "payload \udc9d continues"
        with pytest.raises(UnicodeEncodeError):
            tainted.encode("utf-8")
        # Sanitized form encodes fine.
        sanitize_content(tainted).encode("utf-8")


class TestApplyGateScrubsSurrogates:
    """Read-side contract: apply_gate scrubs lone UTF-16 surrogates from
    both GateItem.text (from doc) and every string value in extras
    (name/summary/description/statement/spo) before the judge prompt is
    built. Legacy records written before the write-side fix still work."""

    def test_gate_inlet_scrubs_text_field(self, kg):
        from mempalace.injection_gate import apply_gate

        tainted = "hello \udc9d world"
        memories = [{"id": f"mem_{i}", "text": tainted} for i in range(3)]
        # combined_meta carries surrogate contamination in every string-valued
        # field the gate renderer reaches.
        combined_meta = {
            m["id"]: {
                "source": "memory",
                "doc": tainted,
                "meta": {
                    "summary": f"summary \udc9d {i}",
                    "name": f"name \udc9d {i}",
                    "content": f"desc \udc9d {i}",
                },
            }
            for i, m in enumerate(memories)
        }

        client = _FakeClient(
            [[{"id": m["id"], "action": "keep", "reasoning": "ok"} for m in memories]]
        )
        gate = InjectionGate(_client=client)
        apply_gate(
            memories=memories,
            combined_meta=combined_meta,
            primary_context={"queries": ["a", "b"], "keywords": ["x", "y"]},
            context_id="ctx_surr",
            kg=kg,
            agent="test_agent",
            gate=gate,
        )
        # The judge prompt is the user-message content. It must contain
        # no surrogate codepoints -- otherwise Anthropic's HTTP layer
        # would have raised UnicodeEncodeError before the call landed.
        kwargs = client.messages.last_kwargs
        assert kwargs is not None
        prompt = kwargs["messages"][0]["content"]
        assert not any(0xD800 <= ord(c) <= 0xDFFF for c in prompt), (
            "gate inlet must strip lone UTF-16 surrogates from text and extras"
        )
        # Sanity: the prompt does still contain the clean substrings
        # around where surrogates used to live.
        assert "hello" in prompt and "world" in prompt

    def test_gate_inlet_scrub_preserves_clean_text(self, kg):
        """No-op for ASCII / valid UTF-8: the scrubber must not corrupt
        clean content (CJK, emoji, accented chars all survive)."""
        from mempalace.injection_gate import apply_gate

        clean = "Café naïve -- 日本語 🎉"
        memories = [{"id": f"mem_{i}", "text": clean} for i in range(3)]
        combined_meta = {
            m["id"]: {"source": "memory", "doc": clean, "meta": {"summary": clean}}
            for m in memories
        }
        client = _FakeClient(
            [[{"id": m["id"], "action": "keep", "reasoning": "ok"} for m in memories]]
        )
        gate = InjectionGate(_client=client)
        apply_gate(
            memories=memories,
            combined_meta=combined_meta,
            primary_context={"queries": ["a", "b"], "keywords": ["x", "y"]},
            context_id="ctx_clean",
            kg=kg,
            agent="test_agent",
            gate=gate,
        )
        prompt = client.messages.last_kwargs["messages"][0]["content"]
        # Every non-surrogate character in the original should still appear.
        for ch in "Café naïve 日本語 🎉":
            assert ch in prompt


# ─────────────────────────────────────────────────────────────────────
# v3.4.3 regression tests for v3.4.2 behavior fixes
# ─────────────────────────────────────────────────────────────────────


def test_passthrough_gate_report_omits_tokens_when_no_llm_call():
    """v3.4.2 (Adrian post-reinstall 2026-05-13): gate_report's `tokens`
    block is OMITTED when no Haiku call happened (passthrough path:
    gate disabled, empty memories, or get_gate() failure). Pre-fix the
    block was emitted with all-zero values, which read as broken
    telemetry to anyone scanning the response.

    This test exercises the empty-memories short-circuit: apply_gate
    returns the passthrough gate_report which must NOT contain a
    'tokens' key.
    """
    from mempalace.injection_gate import apply_gate

    # Empty memories -> _report_passthrough() path -> no LLM call.
    _filtered, _status, gate_report = apply_gate(
        memories=[],
        combined_meta={},
        primary_context={"queries": ["q"], "keywords": ["k"]},
        context_id="ctx_v343_test",
        kg=None,
        agent="test_agent",
        gate=None,
    )
    assert gate_report is not None, "gate_report should not be None on empty memories; got None"
    assert "tokens" not in gate_report, (
        f"gate_report must OMIT the tokens block when no LLM call happened; "
        f"got gate_report={gate_report!r}"
    )
    # Sanity: the basic shape still ships.
    assert gate_report["input_count"] == 0
    assert gate_report["output_count"] == 0
    assert "elapsed_ms" in gate_report


def test_judge_system_prompt_requires_patch_when_concrete():
    """v3.4.2 (Adrian post-reinstall 2026-05-13): the judge system
    prompt was rewritten from MAY-include-patch-when-confident to
    REQUIRED-when-the-fix-is-concrete, with worked examples 1-3
    updated to emit schema_id + patch in their Correct output. The
    pre-v3.4.2 wording let the LLM emit reason-only entries (which
    LLMs default to because the examples taught them that shape),
    defeating Phase 3 auto-apply in practice.

    This test inspects the run_state_judge source so a future refactor
    that reverts the wording (drops 'REQUIRED', re-adds MAY language,
    or removes patches from the worked examples) gets caught in CI
    instead of silently breaking production.
    """
    import inspect

    import mempalace.injection_gate as _ig

    src = inspect.getsource(_ig.run_state_judge)

    # The tightened patch instruction MUST be in the prompt. Pre-v3.4.2
    # said "You MAY also include the entity's schema_id plus an RFC
    # 6902 'patch'" -- post-v3.4.2 says "The patch is REQUIRED whenever
    # the fix is concrete". The system prompt is built from
    # concatenated string literals so inspect.getsource preserves the
    # source-line breaks between literals; assert on substrings that
    # fit inside a single literal, not the resolved-string phrase.
    assert "REQUIRED whenever the " in src, (
        "judge prompt must declare patch REQUIRED-when-concrete; "
        "the 'REQUIRED whenever the' fragment is missing -- v3.4.2 "
        "wording regressed"
    )
    assert "fix is concrete" in src, (
        "judge prompt must reference the 'fix is concrete' condition "
        "that gates required-patch emission -- v3.4.2 wording regressed"
    )

    # Worked examples must demonstrate the patch field. LLMs follow
    # examples literally; reverting any of the examples back to
    # reason-only breaks auto-apply even with the right rule text.
    # Each Correct-output block in examples 1-3 must include a
    # `"patch": [` opener (case-sensitive, single-line substring).
    example_patch_blocks = src.count('"patch": [')
    assert example_patch_blocks >= 3, (
        f"judge prompt worked examples must show 'patch': [ at least 3 "
        f"times (examples 1-3 each); got {example_patch_blocks} -- "
        f"worked examples regressed"
    )

    # The legacy MAY-also-include phrasing must NOT come back. The
    # pre-v3.4.2 fragment that uniquely identifies the lazy wording
    # was the verbatim string "You MAY also " (only appeared in this
    # specific prompt clause). Catching the fragment is enough to
    # block a sloppy revert.
    assert "You MAY also " not in src, (
        "legacy 'You MAY also' phrasing came back in run_state_judge "
        "system prompt -- v3.4.2 fix regressed"
    )


class TestRunStateJudgeNoOpPatchFilter:
    """v3.7.17 (Adrian directive 2026-05-17): the judge LLM frequently
    emits RFC 6902 patches whose values already match the entity's
    current_state at the same path (e.g. /todos/0/status -> 'done' when
    the entry is already 'done'). Those produce zero behavioural change
    but still trigger record_state_revision writes and
    state_changes_detected entries. run_state_judge now filters those
    no-op patches at the boundary using the entity_states current_state
    map it already receives -- if applying the patch yields a payload
    equal to current_state, the patch field is dropped and the change
    carries skip_reason='no_op_patch' instead.

    These tests mock the Anthropic client to return canned tool_use
    blocks so we exercise the filter without spending tokens."""

    def _make_resp(self, changes_payload):
        import types

        block = types.SimpleNamespace(
            type="tool_use",
            name="report_state_changes",
            input={"changes": changes_payload},
        )
        return types.SimpleNamespace(content=[block], usage=None)

    def _make_gate_with_canned_resp(self, monkeypatch, resp):
        import mempalace.injection_gate as _ig

        gate = _ig.InjectionGate()
        # Bypass real client init; return a stub that yields our canned
        # response on messages.create.
        from unittest.mock import MagicMock

        client = MagicMock()
        client.messages.create.return_value = resp
        monkeypatch.setattr(gate, "_get_client", lambda: client)
        return gate

    def test_noop_replace_patch_is_filtered(self, monkeypatch):
        import mempalace.injection_gate as _ig

        resp = self._make_resp(
            [
                {
                    "entity_id": "task_demo",
                    "reason": "task is now done",
                    "schema_id": "task_state",
                    "patch": [
                        {"op": "replace", "path": "/status", "value": "done"},
                    ],
                }
            ]
        )
        gate = self._make_gate_with_canned_resp(monkeypatch, resp)
        # current_state ALREADY has /status='done'; the patch is a no-op.
        entity_states = [
            {
                "entity_id": "task_demo",
                "state_schema_id": "task_state",
                "current_state": {"status": "done"},
            }
        ]
        changes, _report = _ig.run_state_judge(
            transcript_text="<test transcript>",
            entity_states=entity_states,
            agent="ga_agent",
            gate=gate,
        )
        assert len(changes) == 1
        # The change still surfaces (reason is useful telemetry) but
        # the patch field is dropped + skip_reason is recorded.
        assert "patch" not in changes[0]
        assert changes[0].get("skip_reason") == "no_op_patch"
        assert changes[0]["reason"] == "task is now done"

    def test_real_change_patch_is_preserved(self, monkeypatch):
        import mempalace.injection_gate as _ig

        resp = self._make_resp(
            [
                {
                    "entity_id": "task_demo",
                    "reason": "task moves from in_progress to done",
                    "schema_id": "task_state",
                    "patch": [
                        {"op": "replace", "path": "/status", "value": "done"},
                    ],
                }
            ]
        )
        gate = self._make_gate_with_canned_resp(monkeypatch, resp)
        # current_state has /status='in_progress'; the patch IS a real change.
        entity_states = [
            {
                "entity_id": "task_demo",
                "state_schema_id": "task_state",
                "current_state": {"status": "in_progress"},
            }
        ]
        changes, _report = _ig.run_state_judge(
            transcript_text="<test transcript>",
            entity_states=entity_states,
            agent="ga_agent",
            gate=gate,
        )
        assert len(changes) == 1
        # Real change: patch is preserved + no skip_reason.
        assert changes[0].get("patch") == [
            {"op": "replace", "path": "/status", "value": "done"},
        ]
        assert "skip_reason" not in changes[0]

    def test_missing_current_state_passes_through(self, monkeypatch):
        # When entity_states does NOT carry a current_state for the
        # entity the judge flagged, we cannot decide whether the patch
        # is a no-op. Pass through verbatim and let the apply site
        # surface any error.
        import mempalace.injection_gate as _ig

        resp = self._make_resp(
            [
                {
                    "entity_id": "task_unknown",
                    "reason": "judge flagged an entity not in the followed set",
                    "schema_id": "task_state",
                    "patch": [
                        {"op": "replace", "path": "/status", "value": "done"},
                    ],
                }
            ]
        )
        gate = self._make_gate_with_canned_resp(monkeypatch, resp)
        entity_states = [
            {
                "entity_id": "task_other",
                "state_schema_id": "task_state",
                "current_state": {"status": "open"},
            }
        ]
        changes, _report = _ig.run_state_judge(
            transcript_text="<test transcript>",
            entity_states=entity_states,
            agent="ga_agent",
            gate=gate,
        )
        assert len(changes) == 1
        assert changes[0].get("patch") == [
            {"op": "replace", "path": "/status", "value": "done"},
        ]
        assert "skip_reason" not in changes[0]


pytestmark = pytest.mark.integration
