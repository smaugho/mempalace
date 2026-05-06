"""Rocchio-style enrichment of reused contexts at finalize_intent.

When an existing context entity is reused (MaxSim >= T_reuse during
declare_intent) AND the intent finishes with net-positive feedback
(mean relevance >= 4.0 among rated entries), we merge the caller's
NEW queries / keywords / entities into the context so future lookups
land on it more easily.

Reference: Rocchio 1971 (Manning/Raghavan/Schutze IR book Ch.9) -- the
query-reformulation shift toward the centroid of relevant results.
Here the shift is applied to the context ENTITY's stored views,
keywords, and related entities.

Cap: 20 views per context (LRU eviction on overflow).
"""

from __future__ import annotations


import pytest


def _get_ctx_props(kg, cid):
    ent = kg.get_entity(cid)
    assert ent is not None
    props = ent.get("properties", {}) or {}
    import json as _json

    if isinstance(props, str):
        props = _json.loads(props)
    return props


def test_rocchio_merges_novel_queries(monkeypatch, config, kg, palace_path):
    from tests.integration.test_mcp_server import _patch_mcp_server

    _patch_mcp_server(monkeypatch, config, kg)
    kg.seed_ontology()

    from mempalace import mcp_server

    # Create a context entity the usual way so it's properly shaped.
    cid, reused, _ = mcp_server.context_lookup_or_create(
        queries=["find the auth refresh flow", "locate jwt session teardown"],
        keywords=["auth", "jwt"],
        entities=[],
        agent="test_agent",
    )
    assert cid
    assert reused is False

    # Enrich with novel queries.
    stats = mcp_server.rocchio_enrich_context(
        cid,
        new_queries=[
            "find the auth refresh flow",  # duplicate -- must be dropped
            "resolve expired jwt sessions in the cache",  # novel
        ],
        new_keywords=["auth", "session-teardown"],  # one novel
        new_entities=["AuthCache"],  # novel
    )
    assert stats["added_queries"] == 1
    assert stats["added_keywords"] == 1
    assert stats["added_entities"] == 1
    assert stats["evicted_views"] == 0

    props = _get_ctx_props(kg, cid)
    queries = props.get("queries", [])
    assert "resolve expired jwt sessions in the cache" in queries
    assert queries.count("find the auth refresh flow") == 1, "no duplicate"
    assert "session-teardown" in props.get("keywords", [])
    # Entities are stored in canonical normalize_entity_name form --
    # "AuthCache" becomes "auth_cache".
    assert "auth_cache" in props.get("entities", [])


def test_rocchio_lru_caps_at_20_views(monkeypatch, config, kg, palace_path):
    from tests.integration.test_mcp_server import _patch_mcp_server

    _patch_mcp_server(monkeypatch, config, kg)
    kg.seed_ontology()

    from mempalace import mcp_server

    # Seed with 10 queries (well under the cap). Each query is a
    # distinct semantic topic so the new MaxSim-based dedup (threshold
    # 0.85) doesn't collapse them -- the test is about LRU, not dedup.
    topics = [
        "authentication and jwt session flow",
        "postgres database schema migrations",
        "kubernetes deployment rolling update strategy",
        "react frontend state management patterns",
        "monorepo build cache invalidation logic",
        "distributed tracing with opentelemetry",
        "golang goroutine scheduling internals",
        "typescript generic variance narrowing",
        "rust borrow checker lifetime elision",
        "elasticsearch query DSL tuning",
    ]
    cid, _, _ = mcp_server.context_lookup_or_create(
        queries=topics[:6],  # lookup_or_create caps input at 6 in practice
        keywords=["seed", "keyword"],
        entities=[],
        agent="test_agent",
    )
    # Pad up to 10 via explicit enrichment. Again distinct topics.
    stats = mcp_server.rocchio_enrich_context(cid, new_queries=topics[6:])
    props = _get_ctx_props(kg, cid)
    assert len(props["queries"]) == 10
    assert stats["evicted_views"] == 0

    # Now add 15 more distinct-topic queries -- that's 10 + 15 = 25,
    # should evict 5.
    pad_queries = [
        "python asyncio event loop internals",
        "nginx reverse proxy TLS termination",
        "redis sorted set leaderboard patterns",
        "docker multi-stage build caching",
        "aws s3 bucket lifecycle policies",
        "graphql schema stitching federation",
        "webpack code splitting dynamic imports",
        "rabbitmq exchange binding topologies",
        "prometheus alertmanager routing trees",
        "clickhouse merge tree engine tuning",
        "envoy proxy xDS configuration",
        "cassandra consistency levels tradeoffs",
        "terraform provider plugin protocol",
        "consul service mesh health checks",
        "grpc bidirectional streaming backpressure",
    ]
    stats = mcp_server.rocchio_enrich_context(cid, new_queries=pad_queries)
    assert stats["added_queries"] == 15
    assert stats["evicted_views"] == 5

    props = _get_ctx_props(kg, cid)
    assert len(props["queries"]) == 20
    # The 5 oldest (topics[0..4]) should be evicted.
    for q in topics[:5]:
        assert q not in props["queries"]
    # The survivors (topics[5..9] + pad_queries) should all be present.
    for q in topics[5:10]:
        assert q in props["queries"]
    for q in pad_queries:
        assert q in props["queries"]


def test_rocchio_short_circuits_when_nothing_novel(monkeypatch, config, kg, palace_path):
    from tests.integration.test_mcp_server import _patch_mcp_server

    _patch_mcp_server(monkeypatch, config, kg)
    kg.seed_ontology()

    from mempalace import mcp_server

    cid, _, _ = mcp_server.context_lookup_or_create(
        queries=["first", "second"],
        keywords=["kw1", "kw2"],
        entities=["e1"],
        agent="test_agent",
    )
    stats = mcp_server.rocchio_enrich_context(
        cid,
        new_queries=["first", "second"],  # both duplicates
        new_keywords=["kw1", "KW2"],  # duplicates (case-insensitive match)
        new_entities=["e1"],  # duplicate
    )
    assert stats == {
        "added_queries": 0,
        "added_keywords": 0,
        "added_entities": 0,
        "evicted_views": 0,
        "dedup_dropped_queries": 0,
    }


def test_rocchio_refuses_non_context_entity(monkeypatch, config, kg, palace_path):
    from tests.integration.test_mcp_server import _patch_mcp_server

    _patch_mcp_server(monkeypatch, config, kg)
    kg.seed_ontology()
    kg.add_entity("not_a_context", kind="entity", content="n", importance=3)

    from mempalace import mcp_server

    stats = mcp_server.rocchio_enrich_context(
        "not_a_context", new_queries=["anything"], new_keywords=[], new_entities=[]
    )
    # No-op -- kind != "context".
    assert stats["added_queries"] == 0
    assert stats["added_keywords"] == 0
    assert stats["added_entities"] == 0


pytestmark = pytest.mark.integration
