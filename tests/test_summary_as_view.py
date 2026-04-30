"""Regression guard: rendered summary lands as view N+1 with is_summary_view flag.

Adrian's design lock 2026-04-30: "summary should be a view EVERYWHERE
wherever there is a summary." Today's multi-view embedding stores one
Chroma record per query view; the structured summary ({what, why,
scope?}) gets serialized via ``serialize_summary_for_embedding`` and
appended as view N+1 with metadata flag ``is_summary_view=True`` so
multi_view_max_sim sees it on equal footing with query views and
readers can distinguish summary-derived from query-derived views.

These tests lock the contract at the two write paths:
  - ``context_lookup_or_create`` mint branch -> mempalace_context_views
  - ``_sync_entity_views_to_chromadb`` -> entity multi-view collection

Plus the omission case: when no summary is provided, no extra view is
appended and no metadata flag is written (token-diet default).
"""

from __future__ import annotations

from tests.test_mcp_server import _patch_mcp_server


def _boot(monkeypatch, config, palace_path, kg):
    _patch_mcp_server(monkeypatch, config, kg)
    from mempalace import mcp_server

    kg.seed_ontology()
    return mcp_server


def _ctx_view_records(mcp_server, ctx_id: str) -> list:
    """Return all multi-view records for the given context_id.

    Each record is a dict {id, document, metadata} -- collected by
    iterating Chroma's get(where=...) result so the test can introspect
    metadata without depending on the Chroma client's exact return shape.
    """
    col = mcp_server._get_context_views_collection(create=False)
    if col is None:
        return []
    raw = col.get(where={"context_id": ctx_id}, include=["documents", "metadatas"])
    ids = raw.get("ids") or []
    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []
    out = []
    for i, _id in enumerate(ids):
        out.append(
            {
                "id": _id,
                "document": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
            }
        )
    return out


def test_context_mint_appends_summary_as_view_with_flag(monkeypatch, config, palace_path, kg):
    """Mint a context with summary -> N+1 records, last one carries is_summary_view."""
    mcp_server = _boot(monkeypatch, config, palace_path, kg)
    queries = ["audit injection gate behavior", "verify rated_irrelevant filter"]
    summary = {
        "what": "injection-gate audit",
        "why": "verify rated_irrelevant filter excludes drops from retrieval",
    }
    cid, reused, _ = mcp_server.context_lookup_or_create(
        queries=queries,
        keywords=["audit", "gate"],
        entities=[],
        agent="test_agent",
        summary=summary,
    )
    assert reused is False
    records = _ctx_view_records(mcp_server, cid)
    # N=2 query views + 1 summary view = 3 records expected.
    assert len(records) == len(queries) + 1, (
        f"expected {len(queries) + 1} view records (queries + summary), got {len(records)}"
    )
    # The last view (highest view_index) carries is_summary_view=True; the
    # earlier ones do not.
    by_index = sorted(records, key=lambda r: int(r["metadata"].get("view_index", -1)))
    summary_record = by_index[-1]
    assert summary_record["metadata"].get("is_summary_view") is True, (
        f"expected last view to carry is_summary_view=True, "
        f"got metadata={summary_record['metadata']!r}"
    )
    # Earlier views must NOT carry the flag.
    for r in by_index[:-1]:
        assert "is_summary_view" not in r["metadata"], (
            f"query view {r['metadata'].get('view_index')} unexpectedly "
            f"carries is_summary_view; metadata={r['metadata']!r}"
        )
    # The summary view document is the rendered prose, not a raw query.
    assert summary_record["document"] not in queries, (
        f"summary view document should be the rendered summary prose, "
        f"got document={summary_record['document']!r} which is a query view"
    )
    # Summary view document must contain both the "what" and "why"
    # content from the structured summary so it carries discriminative
    # signal beyond what the query views already embed. The exact
    # rendered shape is owned by serialize_summary_for_embedding (kept
    # opaque here so the test doesn't break on cosmetic format tweaks).
    assert summary["what"] in summary_record["document"], (
        f"summary view document should contain the structured 'what'; "
        f"summary={summary!r}, document={summary_record['document']!r}"
    )
    assert summary["why"] in summary_record["document"], (
        f"summary view document should contain the structured 'why'; "
        f"summary={summary!r}, document={summary_record['document']!r}"
    )


def test_context_mint_without_summary_emits_only_query_views(monkeypatch, config, palace_path, kg):
    """No summary -> exactly N query views, no is_summary_view flag anywhere."""
    mcp_server = _boot(monkeypatch, config, palace_path, kg)
    queries = ["query one for the gate audit", "query two same topic"]
    cid, reused, _ = mcp_server.context_lookup_or_create(
        queries=queries,
        keywords=["gate"],
        entities=[],
        agent="test_agent",
        # summary deliberately omitted
    )
    assert reused is False
    records = _ctx_view_records(mcp_server, cid)
    assert len(records) == len(queries), (
        f"expected {len(queries)} view records (no summary), got {len(records)}"
    )
    for r in records:
        assert "is_summary_view" not in r["metadata"], (
            f"unexpected is_summary_view flag on query view "
            f"index {r['metadata'].get('view_index')}; metadata={r['metadata']!r}"
        )


def test_sync_entity_views_summary_view_kwarg(monkeypatch, config, palace_path, kg):
    """The helper appends summary_view as view N+1 with is_summary_view metadata."""
    mcp_server = _boot(monkeypatch, config, palace_path, kg)

    # Create a synthetic entity row in SQLite so the helper has something
    # to associate Chroma records with. We don't go through
    # tool_kg_declare_entity here; we exercise the helper directly so the
    # test is laser-focused on the views shape.
    entity_id = "test_entity_summary_view_target"
    queries = ["what is X", "why does X matter"]
    rendered_summary = "WHAT: synthetic entity. WHY: regression-guard test fixture."

    mcp_server._STATE.kg.add_entity(
        entity_id,
        kind="entity",
        content="synthetic test entity for summary-as-view regression",
        importance=3,
        properties={"queries": queries, "summary": {"what": "synthetic entity", "why": "fixture"}},
    )
    mcp_server._sync_entity_views_to_chromadb(
        entity_id,
        entity_id,
        queries,
        "entity",
        3,
        added_by="test_agent",
        summary_view=rendered_summary,
    )
    ecol = mcp_server._get_entity_collection(create=False)
    assert ecol is not None
    raw = ecol.get(where={"entity_id": entity_id}, include=["documents", "metadatas"])
    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []
    assert len(docs) == len(queries) + 1, (
        f"expected {len(queries) + 1} multi-view records (queries + summary), got {len(docs)}"
    )
    # Find which record carries is_summary_view=True.
    flagged = [(i, m) for i, m in enumerate(metas) if m.get("is_summary_view") is True]
    assert len(flagged) == 1, (
        f"expected exactly 1 record with is_summary_view=True, got {len(flagged)}; "
        f"all metadatas={metas!r}"
    )
    flagged_idx = flagged[0][0]
    assert docs[flagged_idx] == rendered_summary, (
        f"summary view document mismatch: expected {rendered_summary!r}, got {docs[flagged_idx]!r}"
    )


def test_sync_entity_views_omits_summary_view_when_none(monkeypatch, config, palace_path, kg):
    """No summary_view kwarg -> no is_summary_view flag on any record."""
    mcp_server = _boot(monkeypatch, config, palace_path, kg)
    entity_id = "test_entity_no_summary_view"
    queries = ["alpha view", "beta view"]
    mcp_server._STATE.kg.add_entity(
        entity_id,
        kind="entity",
        content="synthetic test entity without summary view",
        importance=3,
        properties={"queries": queries},
    )
    mcp_server._sync_entity_views_to_chromadb(
        entity_id,
        entity_id,
        queries,
        "entity",
        3,
        added_by="test_agent",
        # summary_view omitted on purpose
    )
    ecol = mcp_server._get_entity_collection(create=False)
    assert ecol is not None
    raw = ecol.get(where={"entity_id": entity_id}, include=["documents", "metadatas"])
    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []
    assert len(docs) == len(queries), (
        f"expected exactly {len(queries)} records (no summary), got {len(docs)}"
    )
    for m in metas:
        assert "is_summary_view" not in m, (
            f"unexpected is_summary_view flag on a query-only record; metadata={m!r}"
        )
