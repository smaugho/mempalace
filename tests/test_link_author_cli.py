"""
test_link_author_cli.py -- End-to-end tests for the LLM jury pipeline.

All Anthropic SDK calls are mocked via a fake AsyncAnthropic client
whose ``messages.create`` returns scripted JSON per stage. Tests cover:

  - Happy path: unanimous edge → persisted + kg_add fired.
  - Jury disagreement (2-1 predicate mismatch) → uncertain; no edge.
  - All-no_edge jury → rejection verdict; no edge.
  - New-predicate proposal with ontologist+1 consensus → predicate created.
  - New-predicate proposal blocked by near-duplicate → existing used.
  - Stage 1 Opus failure → jury_design_failed (no fallback personas).
  - Opus malformed JSON → same failure mode.
  - Domain-hint batching: N candidates share one design call.
  - Synthesis schema-violating "edge" (no predicate / no statement) → uncertain.

See docs/link_author_plan.md §2.5 + §2.5.1.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from mempalace import link_author


# ─────────────────────────────────────────────────────────────────────
# Fake anthropic client infrastructure
# ─────────────────────────────────────────────────────────────────────


def _text_block(text: str):
    """Build a fake SDK content block (has a `.text` attribute)."""
    b = MagicMock()
    b.text = text
    return b


def _fake_response(text: str):
    """Build a fake Message with a single text content block."""
    r = MagicMock()
    r.content = [_text_block(text)]
    return r


class ScriptedClient:
    """Async anthropic-shaped client whose create() returns canned payloads.

    ``script`` is a list of either:
      - strings: returned as the next response text, one per call.
      - exceptions: raised when hit.

    Raises IndexError if the script runs dry -- that's a test-authoring
    bug and should fail loudly rather than silently cycle.
    """

    def __init__(self, script):
        self._script = list(script)
        self._calls: list[dict] = []
        self.messages = self  # shim so client.messages.create works

    async def create(self, **kwargs):
        self._calls.append(kwargs)
        if not self._script:
            raise IndexError("ScriptedClient script exhausted")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return _fake_response(item)

    @property
    def calls(self):
        return self._calls


def _cfg():
    return {
        "api_key_env": "ANTHROPIC_API_KEY",
        "jury_design_model": "claude-opus-4-5",
        "jury_execution_model": "claude-haiku-4-5",
        "synthesis_model": "claude-haiku-4-5",
        "design_max_tokens": 1024,
        "juror_max_tokens": 512,
        "synthesis_max_tokens": 512,
        "batch_design_by_domain_similarity": True,
        "batch_domain_cosine_threshold": 0.9,
        "threshold": 1.5,
        "max_per_run": 50,
        "interval_hours": 1,
        "retry_uncertain_next_run": True,
        "rejection_cooldown_days": 30,
        "escalate_uncertain_to": None,
    }


def _personas_json():
    return json.dumps(
        [
            {
                "role": "ontologist",
                "persona_prompt": "You are a knowledge-graph ontologist.",
            },
            {
                "role": "skeptic",
                "persona_prompt": "You are a skeptic challenging evidence.",
            },
            {
                "role": "senior software engineer",
                "persona_prompt": "You are a senior engineer familiar with SaaS APIs.",
            },
        ]
    )


def _juror(role, verdict, predicate=None, statement=None, proposal=None, reason="x"):
    return json.dumps(
        {
            "role": role,
            "verdict": verdict,
            "confidence": 0.8,
            "predicate_choice": predicate,
            "propose_new_predicate": proposal,
            "statement": statement,
            "reason": reason,
        }
    )


def _synthesis(verdict, predicate=None, statement=None, new_pred=None, agreement="majority"):
    return json.dumps(
        {
            "verdict": verdict,
            "predicate": predicate,
            "new_predicate_proposed": new_pred,
            "statement": statement,
            "reason": "synthesis",
            "juror_agreement": agreement,
        }
    )


# ─────────────────────────────────────────────────────────────────────
# KG + candidate fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def seeded_candidate(kg):
    """Seed one pending candidate A-B with a context, 2 entities, and the
    predicate vocabulary Stage 2 will need to choose from."""
    kg.add_entity(
        "alice_service",
        kind="entity",
        content="the user-facing auth service",
        importance=3,
    )
    kg.add_entity(
        "bob_store",
        kind="entity",
        content="the backing identity datastore",
        importance=3,
    )
    kg.add_entity(
        "depends_on",
        kind="predicate",
        content="subject depends on object for runtime correctness",
        importance=4,
        properties={
            "constraints": {
                "subject_kinds": ["entity"],
                "object_kinds": ["entity"],
                "cardinality": "many-to-many",
            }
        },
    )
    # Seed a context so the domain hint has something to render.
    kg.add_entity(
        "ctx-demo",
        kind="context",
        content="demo context for link-author tests",
        importance=3,
        properties={
            "queries": ["auth flow review", "identity lookups"],
            "keywords": ["auth", "identity", "saas"],
            "entities": ["alice_service", "bob_store"],
        },
    )
    # Plant a candidate + source row so list_pending sees this pair.
    assert link_author.upsert_candidate(kg, "alice_service", "bob_store", 2.0, "ctx-demo")
    return {"from_entity": "alice_service", "to_entity": "bob_store"}


# ─────────────────────────────────────────────────────────────────────
# Happy path -- unanimous edge, existing predicate
# ─────────────────────────────────────────────────────────────────────


class TestHappyPath:
    def test_unanimous_edge_creates_triple(self, kg, seeded_candidate):
        client = ScriptedClient(
            [
                _personas_json(),  # Stage 1
                # 3 juror verdicts -- all agree on depends_on
                _juror(
                    "ontologist",
                    "edge",
                    predicate="depends_on",
                    statement="alice_service depends on bob_store.",
                ),
                _juror(
                    "skeptic",
                    "edge",
                    predicate="depends_on",
                    statement="alice_service depends on bob_store.",
                ),
                _juror(
                    "senior software engineer",
                    "edge",
                    predicate="depends_on",
                    statement="alice_service depends on bob_store.",
                ),
                # Stage 3 synthesis
                _synthesis(
                    "edge",
                    predicate="depends_on",
                    statement="alice_service depends on bob_store.",
                    agreement="unanimous",
                ),
            ]
        )
        result = asyncio.run(link_author.author_candidate(kg, _cfg(), client, seeded_candidate))
        assert result["llm_verdict"] == "edge"
        assert result["llm_predicate"] == "depends_on"
        assert "depends on" in result["llm_statement"].lower()
        # Five Anthropic calls total: 1 design + 3 jurors + 1 synthesis.
        assert len(client.calls) == 5
        # Synthesis input should carry the juror verdicts as JSON in
        # the user message.
        synth_user = client.calls[-1]["messages"][0]["content"]
        assert "depends_on" in synth_user


# ─────────────────────────────────────────────────────────────────────
# Jury disagreement paths
# ─────────────────────────────────────────────────────────────────────


class TestJuryDisagreement:
    def test_predicate_mismatch_returns_uncertain(self, kg, seeded_candidate):
        """All 3 say 'edge' but disagree on predicate. Synthesis marks
        uncertain → no edge authored."""
        client = ScriptedClient(
            [
                _personas_json(),
                _juror("ontologist", "edge", predicate="depends_on", statement="a depends on b."),
                _juror("skeptic", "edge", predicate="calls", statement="a calls b."),
                _juror("senior software engineer", "edge", predicate="uses", statement="a uses b."),
                _synthesis("uncertain", predicate=None, statement=None, agreement="split"),
            ]
        )
        result = asyncio.run(link_author.author_candidate(kg, _cfg(), client, seeded_candidate))
        assert result["llm_verdict"] == "uncertain"
        assert result["llm_predicate"] is None
        assert result["llm_statement"] is None

    def test_all_no_edge_verdict_is_no_edge(self, kg, seeded_candidate):
        client = ScriptedClient(
            [
                _personas_json(),
                _juror("ontologist", "no_edge", reason="evidence too weak"),
                _juror("skeptic", "no_edge", reason="nothing here"),
                _juror("senior software engineer", "no_edge"),
                _synthesis("no_edge"),
            ]
        )
        result = asyncio.run(link_author.author_candidate(kg, _cfg(), client, seeded_candidate))
        assert result["llm_verdict"] == "no_edge"
        assert result["llm_predicate"] is None

    def test_schema_violating_edge_demoted_to_uncertain(self, kg, seeded_candidate):
        """Synthesis says verdict='edge' but forgets to include a
        predicate -- we must NOT author a bad edge. Demote to uncertain."""
        client = ScriptedClient(
            [
                _personas_json(),
                _juror("ontologist", "edge", predicate="depends_on", statement="ok."),
                _juror("skeptic", "edge", predicate="depends_on", statement="ok."),
                _juror("senior software engineer", "edge", predicate="depends_on", statement="ok."),
                # Synthesis returns 'edge' but predicate is missing
                _synthesis("edge", predicate=None, statement=None),
            ]
        )
        result = asyncio.run(link_author.author_candidate(kg, _cfg(), client, seeded_candidate))
        assert result["llm_verdict"] == "uncertain"
        assert "predicate" in (result["llm_reason"] or "").lower()


# ─────────────────────────────────────────────────────────────────────
# New-predicate proposal paths
# ─────────────────────────────────────────────────────────────────────


class TestNewPredicateProposal:
    def test_consensus_creates_new_predicate(self, kg, seeded_candidate):
        """Ontologist + one other juror proposed a new predicate 'wraps'.
        Synthesis accepts it → predicate is created and used."""
        proposal = {
            "name": "wraps",
            "content": "Subject provides a higher-level API over object",
            "subject_kinds": ["entity"],
            "object_kinds": ["entity"],
            "cardinality": "many-to-one",
        }
        client = ScriptedClient(
            [
                _personas_json(),
                _juror(
                    "ontologist",
                    "edge",
                    proposal=proposal,
                    statement="alice_service wraps bob_store.",
                ),
                _juror(
                    "skeptic", "edge", proposal=proposal, statement="alice_service wraps bob_store."
                ),
                _juror(
                    "senior software engineer",
                    "edge",
                    proposal=proposal,
                    statement="alice_service wraps bob_store.",
                ),
                _synthesis(
                    "edge",
                    predicate=None,
                    statement="alice_service wraps bob_store.",
                    new_pred=proposal,
                ),
            ]
        )
        result = asyncio.run(link_author.author_candidate(kg, _cfg(), client, seeded_candidate))
        assert result["llm_verdict"] == "edge"
        assert result["llm_predicate"] == "wraps"
        # Predicate entity now lives in KG.
        assert kg.get_entity("wraps") is not None

    def test_near_duplicate_proposal_reuses_existing(self, kg, seeded_candidate):
        """Proposal description near-duplicates 'depends_on' (cosine ≥
        0.75). Near-duplicate check should reuse existing."""
        proposal = {
            "name": "requires_at_runtime",
            # Same semantic as depends_on -- near-duplicate by description.
            "content": "subject depends on object for runtime correctness",
            "subject_kinds": ["entity"],
            "object_kinds": ["entity"],
            "cardinality": "many-to-many",
        }
        client = ScriptedClient(
            [
                _personas_json(),
                _juror("ontologist", "edge", proposal=proposal, statement="x."),
                _juror("skeptic", "edge", proposal=proposal, statement="x."),
                _juror("senior software engineer", "edge", proposal=proposal, statement="x."),
                _synthesis("edge", predicate=None, statement="x.", new_pred=proposal),
            ]
        )
        result = asyncio.run(link_author.author_candidate(kg, _cfg(), client, seeded_candidate))
        assert result["llm_verdict"] == "edge"
        # Existing predicate reused; new one NOT created.
        assert result["llm_predicate"] == "depends_on"
        assert kg.get_entity("requires_at_runtime") is None


# ─────────────────────────────────────────────────────────────────────
# Stage 1 failure modes (plan §2.5 removal-discipline)
# ─────────────────────────────────────────────────────────────────────


class TestJuryDesignFailure:
    def test_opus_api_error_marks_jury_design_failed(self, kg, seeded_candidate):
        """Opus raises an APIError → candidate marked jury_design_failed,
        NO fallback personas used, NO downstream calls made."""
        import anthropic

        client = ScriptedClient(
            [
                anthropic.APIError("opus down", request=MagicMock(), body=None),
                # If the pipeline tried to run jurors anyway, it would
                # hit IndexError here (script exhausted).
            ]
        )
        result = asyncio.run(link_author.author_candidate(kg, _cfg(), client, seeded_candidate))
        assert result["llm_verdict"] == "jury_design_failed"
        assert result["llm_predicate"] is None
        # Exactly 1 call (the failing Opus one) -- no juror / synthesis.
        assert len(client.calls) == 1

    def test_opus_malformed_json_marks_jury_design_failed(self, kg, seeded_candidate):
        client = ScriptedClient(
            [
                "this is not JSON at all, Opus went rogue",
            ]
        )
        result = asyncio.run(link_author.author_candidate(kg, _cfg(), client, seeded_candidate))
        assert result["llm_verdict"] == "jury_design_failed"

    def test_opus_wrong_persona_count_marks_jury_design_failed(self, kg, seeded_candidate):
        """Stage 1 must return EXACTLY 3 personas. Two or four = failure."""
        two_personas = json.dumps(
            [
                {"role": "ontologist", "persona_prompt": "..."},
                {"role": "skeptic", "persona_prompt": "..."},
            ]
        )
        client = ScriptedClient([two_personas])
        result = asyncio.run(link_author.author_candidate(kg, _cfg(), client, seeded_candidate))
        assert result["llm_verdict"] == "jury_design_failed"


# ─────────────────────────────────────────────────────────────────────
# Batching -- multiple candidates share one design call
# ─────────────────────────────────────────────────────────────────────


class TestBatching:
    def test_single_cluster_shares_one_design_call(self):
        """Three candidates with identical domain-hint vectors cluster
        into one group -- only ONE design call is made."""
        cand_a = {"from_entity": "a1", "to_entity": "a2"}
        cand_b = {"from_entity": "b1", "to_entity": "b2"}
        cand_c = {"from_entity": "c1", "to_entity": "c2"}

        # All three hints share the same vector → one cluster.
        same_vec = [1.0, 0.0, 0.0, 0.0]
        items = [
            (cand_a, "hint_a", same_vec),
            (cand_b, "hint_b", list(same_vec)),
            (cand_c, "hint_c", list(same_vec)),
        ]
        clusters = link_author._cluster_candidates_by_domain(items, threshold=0.9)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_distant_vectors_form_separate_clusters(self):
        """Two candidates with orthogonal vectors → two clusters."""
        items = [
            ({"from_entity": "a"}, "h1", [1.0, 0.0, 0.0]),
            ({"from_entity": "b"}, "h2", [0.0, 1.0, 0.0]),
        ]
        clusters = link_author._cluster_candidates_by_domain(items, threshold=0.9)
        assert len(clusters) == 2

    def test_missing_vector_falls_through_as_singleton(self):
        """A candidate whose hint failed to embed (vec=None) never joins
        a cluster -- becomes its own singleton. Pipeline stays correct."""
        items = [
            ({"from_entity": "a"}, "h1", [1.0, 0.0]),
            ({"from_entity": "b"}, "h2", None),
        ]
        clusters = link_author._cluster_candidates_by_domain(items, threshold=0.9)
        assert len(clusters) == 2
        assert clusters[1][0][0]["from_entity"] == "b"


# ─────────────────────────────────────────────────────────────────────
# Parallelism -- jury executes via asyncio.gather
# ─────────────────────────────────────────────────────────────────────


class TestJuryParallelism:
    def test_three_jurors_run_concurrently(self, kg, seeded_candidate):
        """The 3 juror calls dispatch via asyncio.gather -- all 3 are
        'in flight' before any completes. We prove this by having each
        juror block on an event that only releases after count == 3."""
        personas = [
            {"role": "ontologist", "persona_prompt": "x"},
            {"role": "skeptic", "persona_prompt": "y"},
            {"role": "expert", "persona_prompt": "z"},
        ]
        in_flight = 0
        peak = 0
        gate = asyncio.Event()

        class ConcurrentClient:
            def __init__(self):
                self.messages = self

            async def create(self, **kwargs):
                nonlocal in_flight, peak
                in_flight += 1
                peak = max(peak, in_flight)
                if peak == 3:
                    gate.set()
                await gate.wait()
                in_flight -= 1
                return _fake_response(
                    _juror(
                        kwargs["system"][:32] if kwargs.get("system") else "?",
                        "no_edge",
                    )
                )

        client = ConcurrentClient()
        entity_ctx = {
            "a": {"id": "a", "kind": "entity", "content": "", "keywords": []},
            "b": {"id": "b", "kind": "entity", "content": "", "keywords": []},
            "shared_blob": "",
            "n_shared": 0,
        }

        asyncio.run(link_author._run_jury(client, _cfg(), personas, entity_ctx, []))
        assert peak == 3, f"jury calls should have reached 3 concurrent, peaked at {peak}"


# ─────────────────────────────────────────────────────────────────────
# Process orchestration -- end-to-end via the CLI entry
# ─────────────────────────────────────────────────────────────────────


class TestProcessOrchestration:
    def test_dry_run_does_not_persist_verdicts(self, kg, seeded_candidate, monkeypatch):
        """With --dry-run, processed_ts stays NULL. The candidate
        remains in the pending queue for the next real run."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test1234567890")

        import anthropic

        # Build a scripted client and factor it out of AsyncAnthropic so
        # both _validate_api_key's ping and the main pipeline use it.
        scripted = ScriptedClient(
            [
                "pong",  # _ping_anthropic
                _personas_json(),
                _juror("ontologist", "no_edge"),
                _juror("skeptic", "no_edge"),
                _juror("senior software engineer", "no_edge"),
                _synthesis("no_edge"),
            ]
        )
        monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda *a, **kw: scripted)

        summary = link_author.process(kg, _cfg(), dry_run=True)
        assert summary["candidates_processed"] == 1
        assert summary["edges_rejected"] == 1
        # processed_ts is still NULL -- next run would pick it up again.
        rows = link_author.list_pending(kg, limit=10, threshold=0.0)
        assert len(rows) == 1

    def test_real_run_persists_verdict_and_removes_from_pending(
        self, kg, seeded_candidate, monkeypatch
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test1234567890")
        import anthropic

        scripted = ScriptedClient(
            [
                "pong",
                _personas_json(),
                _juror("ontologist", "no_edge"),
                _juror("skeptic", "no_edge"),
                _juror("senior software engineer", "no_edge"),
                _synthesis("no_edge"),
            ]
        )
        monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda *a, **kw: scripted)

        summary = link_author.process(kg, _cfg(), dry_run=False)
        assert summary["edges_rejected"] == 1
        # Pending list should now exclude this candidate (processed_ts set).
        rows = link_author.list_pending(kg, limit=10, threshold=0.0)
        assert rows == []
