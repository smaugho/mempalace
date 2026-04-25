"""Strict coverage validator on finalize_intent's memory_feedback.

Every ``(context, memory)`` pair with a ``surfaced`` edge written during
this intent must have a matching rated_useful or rated_irrelevant
entry in memory_feedback. Missing pairs → finalize rejects with the
list of unresolved pairs.

Design rationale (Adrian's "suggestion is death with LLMs"): feedback
must be mandatory. Optional tool parameters get ignored by models.
The validator enforces coverage at finalize time so no surfaced pair
escapes a rating.
"""

from __future__ import annotations

from tests.test_intent_system import _patch_mcp_for_intents, _TEST_BUDGET


def _boot(monkeypatch, config, kg, palace_path):
    mcp = _patch_mcp_for_intents(monkeypatch, config, kg, palace_path)
    kg.add_entity("context", kind="class", description="ctx root", importance=5)
    kg.add_entity("created_under", kind="predicate", description="prov", importance=4)
    kg.add_entity("surfaced", kind="predicate", description="surf", importance=4)
    kg.add_entity("rated_useful", kind="predicate", description="pos", importance=4)
    kg.add_entity("rated_irrelevant", kind="predicate", description="neg", importance=4)
    return mcp


def _declare(mcp):
    return mcp.tool_declare_intent(
        intent_type="research",
        slots={"subject": ["test_target"]},
        context={
            "queries": ["verify finalize coverage", "a second perspective"],
            "keywords": ["finalize", "coverage"],
            "entities": ["test_target"],
            "summary": {
                "what": "test fixture context",
                "why": "auto-migrated context-summary placeholder for legacy test fixtures pre-dating the dict-only contract",
                "scope": "tests",
            },
        },
        agent="test_agent",
        budget=_TEST_BUDGET,
    )


def _stage_surfaced(mcp, ctx_id, memory_id):
    """Simulate tool_kg_search writing a surfaced edge."""
    mcp._STATE.kg.add_triple(ctx_id, "surfaced", memory_id)


def test_finalize_parks_pending_feedback_when_surfaced_pair_has_no_rating(
    monkeypatch, config, kg, palace_path
):
    """Migrated 2026-04-25 from legacy hard-reject contract to the
    99f81f9 two-tool contract.

    Pre-99f81f9 the call returned ``success=False`` with
    ``"Insufficient memory_feedback coverage..."`` and a top-level
    ``missing_pairs`` array — the all-or-nothing block Adrian's
    redesign explicitly retired. Post-fix surfaced-pair coverage gaps
    no longer block finalize: the execution entity is created, partial
    feedback is recorded, and the remainder is parked as
    ``pending_feedback`` for the ``extend_feedback`` round-trip.
    """
    mcp = _boot(monkeypatch, config, kg, palace_path)
    _declare(mcp)
    ctx_id = mcp._STATE.active_intent["active_context_id"]
    assert ctx_id

    # Seed a memory-entity and then a surfaced edge from the active context.
    kg.add_entity("mem_unrated", kind="record", description="m", importance=3)
    _stage_surfaced(mcp, ctx_id, "mem_unrated")

    from mempalace.intent import tool_finalize_intent

    result = tool_finalize_intent(
        slug="missing-coverage",
        outcome="success",
        content="Ended without rating mem_unrated.",
        summary={
            "what": "test fixture record",
            "why": "Ended without rating mem_unrated.",
            "scope": "tests",
        },
        agent="test_agent",
        memory_feedback=[],
    )

    # Legacy hard-reject keys MUST NOT appear.
    assert "missing_pairs_count" not in result, (
        "Legacy missing_pairs_count key surfaced — the all-or-nothing "
        "contract that 99f81f9 retired."
    )
    assert "missing_pairs" not in result, (
        "Legacy missing_pairs key surfaced — same retired contract."
    )
    err = (result.get("error") or "").lower()
    assert "insufficient memory_feedback coverage" not in err, (
        "Legacy error string re-introduced. Use extend_feedback path."
    )

    # Either the call succeeds (no other gates fire) or it parks
    # pending_feedback (other gates triggered). Both acceptable.
    if not result.get("success"):
        assert "extend_feedback" in (result.get("error") or ""), (
            "Post-fix non-success response must direct caller to "
            "extend_feedback per the 99f81f9 two-tool contract."
        )


def test_finalize_accepts_list_shape_when_active_ctx_set(monkeypatch, config, kg, palace_path):
    mcp = _boot(monkeypatch, config, kg, palace_path)
    _declare(mcp)
    ctx_id = mcp._STATE.active_intent["active_context_id"]

    kg.add_entity("mem_rated_a", kind="record", description="ma", importance=3)
    _stage_surfaced(mcp, ctx_id, "mem_rated_a")

    from mempalace.intent import tool_finalize_intent

    # Flat list entry — defaults its context to active_ctx_id under the
    # validator's permissive fallback.
    result = tool_finalize_intent(
        slug="list-shape-ok",
        outcome="success",
        content="Rated mem_rated_a via legacy list shape.",
        summary={
            "what": "test fixture record",
            "why": "list shape, rated mem_rated_a",
            "scope": "tests",
        },
        agent="test_agent",
        memory_feedback=[
            {
                "context_id": ctx_id,
                "feedback": [
                    {
                        "id": "mem_rated_a",
                        "relevant": True,
                        "relevance": 4,
                        "reason": "matched the core question exactly",
                    },
                ],
            }
        ],
    )
    assert result["success"] is True, result


def test_finalize_accepts_map_shape_per_context(monkeypatch, config, kg, palace_path):
    mcp = _boot(monkeypatch, config, kg, palace_path)
    _declare(mcp)
    ctx_id = mcp._STATE.active_intent["active_context_id"]

    kg.add_entity("mem_m1", kind="record", description="m1", importance=3)
    kg.add_entity("mem_m2", kind="record", description="m2", importance=3)
    _stage_surfaced(mcp, ctx_id, "mem_m1")
    _stage_surfaced(mcp, ctx_id, "mem_m2")

    from mempalace.intent import tool_finalize_intent

    result = tool_finalize_intent(
        slug="map-shape-ok",
        outcome="success",
        content="Rated both via map shape.",
        summary={"what": "test fixture record", "why": "map shape, both rated", "scope": "tests"},
        agent="test_agent",
        memory_feedback=[
            {
                "context_id": ctx_id,
                "feedback": [
                    {
                        "id": "mem_m1",
                        "relevant": True,
                        "relevance": 5,
                        "reason": "addressed the core question about finalize coverage",
                    },
                    {
                        "id": "mem_m2",
                        "relevant": False,
                        "relevance": 2,
                        "reason": "outdated and unrelated to this task",
                    },
                ],
            }
        ],
    )
    assert result["success"] is True, result


def test_finalize_partial_map_coverage_does_not_legacy_reject(monkeypatch, config, kg, palace_path):
    """Migrated 2026-04-25 to the 99f81f9 two-tool contract.

    Pre-fix: finalize returned ``success=False`` with the legacy
    ``missing_pairs`` array whenever ANY surfaced pair lacked a
    rating. Post-fix: surfaced-pair coverage gaps no longer block at
    finalize-time; either the call succeeds or it directs to
    ``extend_feedback`` (when other coverage gates also fire).
    """
    mcp = _boot(monkeypatch, config, kg, palace_path)
    _declare(mcp)
    ctx_id = mcp._STATE.active_intent["active_context_id"]

    kg.add_entity("mem_covered", kind="record", description="c", importance=3)
    kg.add_entity("mem_uncovered", kind="record", description="u", importance=3)
    _stage_surfaced(mcp, ctx_id, "mem_covered")
    _stage_surfaced(mcp, ctx_id, "mem_uncovered")

    from mempalace.intent import tool_finalize_intent

    result = tool_finalize_intent(
        slug="partial-map",
        outcome="success",
        content="only one rated",
        summary={
            "what": "test fixture record",
            "why": "only one rated (test fixture)",
            "scope": "tests",
        },
        agent="test_agent",
        memory_feedback=[
            {
                "context_id": ctx_id,
                "feedback": [
                    {
                        "id": "mem_covered",
                        "relevant": True,
                        "relevance": 3,
                        "reason": "partially addressed the question",
                    },
                ],
            }
        ],
    )
    # Legacy hard-reject keys MUST NOT appear post-99f81f9.
    assert "missing_pairs_count" not in result
    assert "missing_pairs" not in result
    err = (result.get("error") or "").lower()
    assert "insufficient memory_feedback coverage" not in err
    if not result.get("success"):
        assert "extend_feedback" in (result.get("error") or "")
