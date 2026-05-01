"""Regression test: memory-preview rendering is canonical across all retrieval paths.

Adrian's render-divergence audit 2026-05-01: pre-fix, the same entity
surfaced with different ``text`` previews depending on which retrieval
path returned it.

  * ``mempalace_declare_user_intents`` returned the queries[0] string
    (the matched Chroma view's document) when the probe view outranked
    the abstract record.
  * ``mempalace_declare_intent`` returned the rendered summary prose
    only when the abstract record (id=``{eid}``, document=rendered
    prose) outranked the probes -- same latent bug under different
    rerank orderings.
  * ``mempalace_declare_operation`` shared the
    ``hooks_cli._run_local_retrieval`` path so it had the same bug
    as declare_user_intents.

Adrian's question that surfaced this: *"is not the memory rendering
exactly same method, reused everywhere?"* -- the answer pre-fix was
NO. Two parallel paths read from different sources (Chroma view
metadata vs reranker output), neither reading the canonical
``entities.properties.summary`` from SQLite.

Fix
---
``scoring.render_memory_preview(logical_id, kg, fallback_text)`` --
single source of truth that reads ``entities.properties.summary`` from
SQLite, renders via ``serialize_summary_for_embedding``, falls through
to ``entities.content``, then to ``fallback_text``. Both retrieval
paths now route through this helper.

These tests lock the contract.
"""

from __future__ import annotations


def test_render_memory_preview_returns_summary_prose(tmp_path):
    """The helper reads entities.properties.summary and renders the
    canonical ``what -- why; scope`` prose form. This is the path the
    fix relies on: every entity created via mint_entity carries this
    summary in properties, so the helper finds it."""
    from mempalace.knowledge_graph import KnowledgeGraph, serialize_summary_for_embedding
    from mempalace.scoring import render_memory_preview

    db = tmp_path / "render.sqlite3"
    kg = KnowledgeGraph(db_path=str(db))
    kg.seed_ontology()

    summary = {
        "what": "RenderTestEntity -- regression test fixture",
        "why": "exists to verify render_memory_preview returns the canonical rendered summary prose, not the matched view's document",
        "scope": "test_render_consistency.py 2026-05-01",
    }
    kg.add_entity(
        "render_test_entity",
        kind="entity",
        content=serialize_summary_for_embedding(summary),
        properties={"summary": summary},
    )

    rendered = render_memory_preview("render_test_entity", kg, fallback_text="FALLBACK")
    assert rendered == serialize_summary_for_embedding(summary)
    # Must NOT have leaked into fallback even though fallback was provided
    assert "FALLBACK" not in rendered


def test_render_memory_preview_falls_through_to_content_when_no_summary(tmp_path):
    """When properties.summary is missing (legacy or partially-migrated
    data), the helper falls through to entities.content -- not to
    fallback_text -- because content is still canonical."""
    from mempalace.knowledge_graph import KnowledgeGraph
    from mempalace.scoring import render_memory_preview

    db = tmp_path / "render.sqlite3"
    kg = KnowledgeGraph(db_path=str(db))
    kg.seed_ontology()
    kg.add_entity(
        "no_summary_entity",
        kind="entity",
        content="legacy content prose with no summary dict",
        properties={},
    )

    rendered = render_memory_preview("no_summary_entity", kg, fallback_text="FALLBACK")
    assert rendered == "legacy content prose with no summary dict"
    assert "FALLBACK" not in rendered


def test_render_memory_preview_uses_fallback_when_entity_missing(tmp_path):
    """Last-resort fallback fires only when the SQLite lookup returns
    nothing (entity was never minted, or was deleted). The matched-view
    document the caller passes as ``fallback_text`` is the right
    semantic last-resort because at least it represents what the
    retrieval channel found."""
    from mempalace.knowledge_graph import KnowledgeGraph
    from mempalace.scoring import render_memory_preview

    db = tmp_path / "render.sqlite3"
    kg = KnowledgeGraph(db_path=str(db))
    kg.seed_ontology()

    rendered = render_memory_preview(
        "never_minted",
        kg,
        fallback_text="last-resort matched-view doc",
    )
    assert rendered == "last-resort matched-view doc"


def test_render_memory_preview_handles_empty_logical_id(tmp_path):
    """Empty logical_id bypasses SQLite and returns fallback. The helper
    must not crash -- some retrieval edge cases produce empty ids."""
    from mempalace.knowledge_graph import KnowledgeGraph
    from mempalace.scoring import render_memory_preview

    db = tmp_path / "render.sqlite3"
    kg = KnowledgeGraph(db_path=str(db))
    kg.seed_ontology()

    rendered = render_memory_preview("", kg, fallback_text="fb")
    assert rendered == "fb"


def test_render_memory_preview_handles_string_properties_json(tmp_path):
    """Some palaces store properties as JSON-string in SQLite (legacy /
    serialised through paths that didn't decode). The helper must coerce
    or fall through cleanly -- not crash on non-dict properties."""
    import json as _json

    from mempalace.knowledge_graph import KnowledgeGraph
    from mempalace.scoring import render_memory_preview

    db = tmp_path / "render.sqlite3"
    kg = KnowledgeGraph(db_path=str(db))
    kg.seed_ontology()

    summary = {
        "what": "JsonStringProps -- regression fixture",
        "why": "verify render_memory_preview decodes string-form properties JSON cleanly",
        "scope": "edge case",
    }
    # Manually insert with stringified properties to simulate legacy
    # serialisation path.
    conn = kg._conn()
    conn.execute(
        "INSERT INTO entities (id, name, kind, properties, content, importance, last_touched, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'active')",
        (
            "json_string_props",
            "JsonStringProps",
            "entity",
            _json.dumps({"summary": summary}),
            "fallback content",
            3,
            "2026-05-01T00:00:00",
        ),
    )
    conn.commit()

    rendered = render_memory_preview("json_string_props", kg)
    # The summary dict is decoded from string-form properties and rendered.
    assert "JsonStringProps -- regression fixture" in rendered
    # Not the fallback content.
    assert "fallback content" not in rendered


# ── Cross-path consistency: all 3 retrieval paths render the same entity identically ──
#
# This is the integration-level lock. Pre-fix, the SAME entity surfaced
# with three different ``text`` strings depending on which path returned
# it. Post-fix, they MUST be identical.


def test_cross_path_render_consistency_for_minted_entity(monkeypatch, config, palace_path, kg):
    """Mint an entity via the gate, then verify ``render_memory_preview``
    returns the same canonical prose regardless of which logical_id /
    Chroma view ranked highest. This locks the contract: there is no
    per-path divergence at the rendering layer."""
    from mempalace import mcp_server
    from mempalace.entity_gate import mint_entity
    from mempalace.knowledge_graph import serialize_summary_for_embedding
    from mempalace.scoring import render_memory_preview

    monkeypatch.setattr(mcp_server._STATE, "kg", kg)
    monkeypatch.setattr(mcp_server._STATE, "config", config)
    monkeypatch.setattr(mcp_server._STATE, "session_id", "test-render-consistency")
    monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
    monkeypatch.setattr(mcp_server._STATE, "client_cache", None)
    monkeypatch.setattr(mcp_server._STATE, "collection_cache", None)
    kg.seed_ontology()

    summary = {
        "what": "ConsistencyTest entity",
        "why": "regression: same entity must render the same canonical preview no matter which retrieval path or matched view found it",
        "scope": "cold-start render lock",
    }
    eid, _was_reused = mint_entity(
        "consistency_test_entity",
        kind="entity",
        summary=summary,
        queries=[
            "what is the consistency test entity",
            "describe the regression test fixture",
        ],
        importance=3,
    )
    expected = serialize_summary_for_embedding(summary)

    # The helper must return ``expected`` regardless of the per-view
    # `fallback_text` supplied -- it reads SQLite, not the matched view.
    for fallback in (
        "what is the consistency test entity",  # probe-view doc
        "ConsistencyTest entity",  # identity-view doc
        expected,  # abstract-record doc
        "",  # missing
    ):
        rendered = render_memory_preview(eid, kg, fallback_text=fallback)
        assert rendered == expected, (
            f"render divergence for fallback={fallback!r}: got={rendered!r}, expected={expected!r}"
        )
