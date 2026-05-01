"""Regression locks for the 2026-05-02 channel-separation fix.

Discovery (record_ga_agent_channel_violation_saturation): pre-fix,
``tool_declare_intent`` auto-appended slot-entity content + intent_id
literal into ``_views`` before passing them to
``context_lookup_or_create``. Two consequences Adrian flagged:

  1. ``properties.queries`` on the persisted context contained strings
     the caller never typed (the entity content[:200], the intent_id
     literal). Channel-A/B contract violation.
  2. ``similar_to`` confidence saturated at 1.0 across every context
     sharing a slot entity, because max-of-max picked up the byte-
     identical entity-content view.

Fix: drop the auto-append; replace with explicit ``anchored_by`` graph
edges (Channel B). Add a one-shot migration that strips polluted views
from existing contexts and backfills the missing edges. Auto-run from
``KnowledgeGraph.__init__`` so existing palaces self-heal.

These tests lock the four behaviors:

  A. ``properties.queries`` on a freshly-minted context contains ONLY
     the caller queries (no entity content, no intent_id literal).
  B. ``anchored_by`` edges are written from the context to each entity
     in ``context.entities`` (BFS-walkable both directions).
  C. Two contexts sharing slot entities but with distinct caller
     queries do NOT produce ``similar_to`` confidence == 1.0.
  D. ``migrate_strip_polluted_context_views`` cleans pre-fix contexts
     idempotently and restores ``anchored_by`` edges.
"""

from __future__ import annotations


def _setup_state(monkeypatch, config, kg):
    from mempalace import mcp_server

    monkeypatch.setattr(mcp_server._STATE, "kg", kg)
    monkeypatch.setattr(mcp_server._STATE, "config", config)
    monkeypatch.setattr(mcp_server._STATE, "session_id", "test-channel-sep")
    monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
    monkeypatch.setattr(mcp_server._STATE, "client_cache", None)
    monkeypatch.setattr(mcp_server._STATE, "collection_cache", None)
    mcp_server._get_entity_collection(create=True)


def _seed_agent(kg, name: str = "ga_agent") -> None:
    """Seed an agent entity with content long enough that the legacy
    auto-append would have shoved its content[:200] into views."""
    from mempalace.entity_gate import mint_entity

    kg.add_entity("agent", kind="class", content="Agent class for is_a")
    mint_entity(
        name,
        kind="entity",
        summary={
            "what": f"{name} test fixture for channel-separation",
            "why": "long-content agent so the legacy auto-append would have polluted views with this entity content[:200] string",
            "scope": "test_channel_separation_2026_05_02",
        },
        queries=[f"who is the {name} test fixture", "channel separation fixture"],
        importance=3,
    )
    kg.add_triple(name, "is_a", "agent")


# A. caller queries preserved, no structural pollution


def test_context_properties_queries_contains_only_caller_queries(
    monkeypatch, config, palace_path, kg
):
    """After context_lookup_or_create mints a fresh context, its
    properties.queries must contain ONLY the queries the caller passed
    -- not the entity content[:200], not any intent_id literal."""
    from mempalace import mcp_server
    import json

    _setup_state(monkeypatch, config, kg)
    kg.seed_ontology()
    _seed_agent(kg)

    caller_queries = [
        "investigate context-channel separation",
        "verify Channel A views stay clean",
    ]
    cid, reused, _ = mcp_server.context_lookup_or_create(
        queries=caller_queries,
        keywords=["audit", "channel-sep"],
        entities=["ga_agent"],
        agent="ga_agent",
        summary={
            "what": "channel-separation regression check",
            "why": "verify properties.queries does not include entity content or intent_id",
            "scope": "regression test",
        },
    )
    assert cid, "context must mint"
    assert reused is False
    ent = kg.get_entity(cid)
    assert ent is not None
    props = ent.get("properties") or {}
    if isinstance(props, str):
        props = json.loads(props)
    stored_queries = props.get("queries") or []
    agent_ent = kg.get_entity("ga_agent")
    agent_content_prefix = (agent_ent.get("content") or "")[:200]
    for q in stored_queries:
        assert q != agent_content_prefix, (
            f"properties.queries leaked the agent content prefix {q!r} -- "
            f"channel-separation violation regressed."
        )
        assert not q.startswith("intent_"), (
            f"properties.queries contains an intent_id literal {q!r} -- "
            f"channel-separation violation regressed."
        )


# B. anchored_by edges written


def test_anchored_by_edges_written_for_each_entity(monkeypatch, config, palace_path, kg):
    """Every entity in context.entities must have a ``ctx -> anchored_by
    -> entity`` edge so Channel B BFS can walk to the context from any
    of its anchors."""
    from mempalace import mcp_server

    _setup_state(monkeypatch, config, kg)
    kg.seed_ontology()
    _seed_agent(kg)

    cid, _r, _m = mcp_server.context_lookup_or_create(
        queries=["audit anchored_by edges", "verify Channel B reachability"],
        keywords=["anchored-by", "channel-b"],
        entities=["ga_agent"],
        agent="ga_agent",
        summary={
            "what": "anchored_by edge regression",
            "why": "verify Channel B BFS reachability from anchor entities to context",
            "scope": "regression test",
        },
    )
    assert cid

    edges = kg.query_entity(cid, direction="outgoing")
    anchored_targets = {
        e["object"] for e in edges if e.get("predicate") == "anchored_by" and e.get("current", True)
    }
    assert "ga_agent" in anchored_targets, (
        f"anchored_by edge from {cid!r} to ga_agent missing; "
        f"got anchored targets: {anchored_targets}"
    )

    incoming = kg.query_entity("ga_agent", direction="incoming")
    incoming_anchors = {
        e["subject"]
        for e in incoming
        if e.get("predicate") == "anchored_by" and e.get("current", True)
    }
    assert cid in incoming_anchors, (
        "Channel B BFS should be able to reach the context from the entity"
    )


# C. no similar_to saturation when only structural anchors match


def test_similar_to_does_not_saturate_on_shared_entities(monkeypatch, config, palace_path, kg):
    """Two contexts sharing a slot entity but with DIFFERENT caller
    queries must not produce a similar_to confidence == 1.0."""
    from mempalace import mcp_server

    _setup_state(monkeypatch, config, kg)
    kg.seed_ontology()
    _seed_agent(kg)

    cid_a, _, _ = mcp_server.context_lookup_or_create(
        queries=[
            "investigate database connection pooling tuning for postgres",
            "raise pool size from 20 to 50 to clear queue backlog",
        ],
        keywords=["postgres", "pool-size"],
        entities=["ga_agent"],
        agent="ga_agent",
        summary={
            "what": "postgres pool tuning context",
            "why": "discriminate from any other ga_agent-anchored context to test channel-separation",
            "scope": "regression test A",
        },
    )
    cid_b, _, _ = mcp_server.context_lookup_or_create(
        queries=[
            "audit user-message tier coverage gap in finalize_intent",
            "extend_feedback rejects diary memory ids without entities row",
        ],
        keywords=["user-message", "extend-feedback"],
        entities=["ga_agent"],
        agent="ga_agent",
        summary={
            "what": "user-message tier coverage gap",
            "why": "different topic from postgres pool tuning; only the ga_agent anchor is shared",
            "scope": "regression test B",
        },
    )
    assert cid_a and cid_b
    assert cid_a != cid_b, "two distinct contexts should mint, not reuse"

    edges = kg.query_entity(cid_b, direction="outgoing")
    similar_to_a = [
        e for e in edges if e.get("predicate") == "similar_to" and e.get("object") == cid_a
    ]
    if similar_to_a:
        conf = similar_to_a[0].get("confidence", 0.0)
        assert conf < 0.999, (
            f"similar_to confidence between distinct-topic contexts saturated "
            f"at {conf} -- channel-separation regressed."
        )


# D. migration helper strips and backfills idempotently


def test_migration_strips_polluted_views_and_backfills_edges(monkeypatch, config, palace_path, kg):
    """Simulate a pre-fix context: write directly to the entities table
    with polluted properties.queries. Run the migration and verify
    cleanup + edge backfill + idempotency."""
    import json

    _setup_state(monkeypatch, config, kg)
    kg.seed_ontology()
    _seed_agent(kg)

    agent_ent = kg.get_entity("ga_agent")
    agent_content_prefix = (agent_ent.get("content") or "")[:200]
    assert agent_content_prefix

    cid = "ctx_test_polluted_pre_fix"
    polluted_queries = [
        "the actual semantic query the caller typed",
        "another real perspective",
        agent_content_prefix,
        "intent_modify_test_legacy",
    ]
    polluted_props = {
        "queries": polluted_queries,
        "keywords": ["legacy", "polluted"],
        "entities": ["ga_agent"],
        "agent": "ga_agent",
    }
    kg.add_entity(
        cid,
        kind="context",
        content="legacy context",
        importance=3,
        properties=polluted_props,
    )

    pre_edges = kg.query_entity(cid, direction="outgoing")
    assert not any(e.get("predicate") == "anchored_by" for e in pre_edges)

    result = kg.migrate_strip_polluted_context_views()
    assert result["considered"] >= 1
    assert result["stripped_views"] >= 2
    assert result["added_anchored_by_edges"] >= 1

    cleaned_ent = kg.get_entity(cid)
    cleaned_props = cleaned_ent.get("properties") or {}
    if isinstance(cleaned_props, str):
        cleaned_props = json.loads(cleaned_props)
    cleaned_queries = cleaned_props.get("queries") or []
    assert agent_content_prefix not in cleaned_queries
    assert "intent_modify_test_legacy" not in cleaned_queries
    assert "the actual semantic query the caller typed" in cleaned_queries
    assert "another real perspective" in cleaned_queries

    post_edges = kg.query_entity(cid, direction="outgoing")
    anchored = {
        e["object"]
        for e in post_edges
        if e.get("predicate") == "anchored_by" and e.get("current", True)
    }
    assert "ga_agent" in anchored

    second = kg.migrate_strip_polluted_context_views()
    assert second["stripped_views"] == 0, (
        "second migration run must find no further structural strings to strip"
    )
