"""Regression tests for summary-first indexing.

Grounded in Anthropic Contextual Retrieval (2024) and Chen et al.
"Dense X Retrieval" (2023). Neither paper gates by length; every chunk
gets the treatment.

Contract enforced at the record-write boundary
(``mcp_server._add_memory_internal``):
  - ``summary`` is ALWAYS required and is a structured dict
    ``{what, why, scope?}`` (Adrian's design lock 2026-04-25).
  - The rendered prose form (``serialize_summary_for_embedding``) is
    capped at 280 chars.
  - When summary is provided, the embedded document is
    ``f"{summary_prose}\n\n{content}"`` (Anthropic CR prepend) and
    ``metadata['summary']`` carries the verbatim rendered prose for
    injection-time display.
"""

from __future__ import annotations

from tests.test_mcp_server import _patch_mcp_server, _get_collection


def _setup(monkeypatch, config, palace_path, kg):
    _patch_mcp_server(monkeypatch, config, kg)
    _client, _col = _get_collection(palace_path, create=True)
    del _client

    from mempalace import mcp_server

    kg.add_entity("agent", kind="class", description="root agent class", importance=5)
    kg.add_entity("test_agent", kind="entity", description="harness agent", importance=3)
    kg.add_triple("test_agent", "is_a", "agent")
    kg.add_entity("host_entity", kind="entity", description="host for link", importance=3)
    return mcp_server


def test_any_record_without_summary_rejected(monkeypatch, config, palace_path, kg):
    """Summary is always required - long content with no summary is rejected."""
    mcp_server = _setup(monkeypatch, config, palace_path, kg)

    long_body = "x" * 500
    result = mcp_server._add_memory_internal(
        content=long_body,
        slug="long-without-summary",
        added_by="test_agent",
        content_type="fact",
        importance=3,
        entity="host_entity",
    )

    assert result["success"] is False
    assert "summary" in result["error"].lower()
    assert "required" in result["error"].lower()


def test_short_record_also_requires_summary(monkeypatch, config, palace_path, kg):
    """Short content also requires summary."""
    mcp_server = _setup(monkeypatch, config, palace_path, kg)

    result = mcp_server._add_memory_internal(
        content="short fact",
        slug="short-without-summary",
        added_by="test_agent",
        content_type="fact",
        importance=3,
        entity="host_entity",
    )

    assert result["success"] is False
    assert "summary" in result["error"].lower()


def test_record_with_valid_summary_stored_with_cr_prepend(monkeypatch, config, palace_path, kg):
    """Valid dict summary -> record created; doc starts with prose summary then CR newlines then body."""
    mcp_server = _setup(monkeypatch, config, palace_path, kg)

    from mempalace.knowledge_graph import serialize_summary_for_embedding

    summary = {
        "what": "long-record distillation",
        "why": "anchor retrieval-time display with WHAT+WHY for the long-content record",
        "scope": "tests",
    }
    summary_prose = serialize_summary_for_embedding(summary)
    body = "content body " * 50
    result = mcp_server._add_memory_internal(
        content=body,
        slug="long-with-summary",
        added_by="test_agent",
        content_type="fact",
        importance=3,
        entity="host_entity",
        summary=summary,
    )

    assert result["success"] is True, result
    memory_id = result["memory_id"]

    col = mcp_server._get_collection(create=False)
    got = col.get(ids=[memory_id], include=["documents", "metadatas"])
    assert got["ids"] == [memory_id]
    doc = got["documents"][0]
    cr_sep = chr(10) + chr(10)
    assert doc.startswith(summary_prose + cr_sep), doc[:200]
    assert body in doc
    assert got["metadatas"][0].get("summary") == summary_prose


def test_short_record_with_different_angle_summary(monkeypatch, config, palace_path, kg):
    """Short content + summary rephrase from different angle = two retrieval views."""
    mcp_server = _setup(monkeypatch, config, palace_path, kg)

    from mempalace.knowledge_graph import serialize_summary_for_embedding

    content = "DB credentials stored in .env.prod, not in the repo"
    summary = {
        "what": "production secrets handling",
        "why": "live in the .env.prod file and are deliberately not committed to version control",
        "scope": "tests",
    }
    summary_prose = serialize_summary_for_embedding(summary)
    result = mcp_server._add_memory_internal(
        content=content,
        slug="db-credentials-reframe",
        added_by="test_agent",
        content_type="fact",
        importance=3,
        entity="host_entity",
        summary=summary,
    )

    assert result["success"] is True, result
    memory_id = result["memory_id"]
    col = mcp_server._get_collection(create=False)
    got = col.get(ids=[memory_id], include=["documents", "metadatas"])
    doc = got["documents"][0]
    cr_sep = chr(10) + chr(10)
    assert doc.startswith(summary_prose + cr_sep)
    assert content in doc
    assert got["metadatas"][0].get("summary") == summary_prose


def test_oversize_summary_rejected(monkeypatch, config, palace_path, kg):
    """Rendered prose form capped at 280 chars; longer dicts rejected."""
    mcp_server = _setup(monkeypatch, config, palace_path, kg)

    too_long_summary = {
        "what": "oversize fixture",
        "why": ("x" * 290),
        "scope": "tests",
    }
    result = mcp_server._add_memory_internal(
        content="some short body",
        slug="summary-too-long",
        added_by="test_agent",
        content_type="fact",
        importance=3,
        entity="host_entity",
        summary=too_long_summary,
    )

    assert result["success"] is False
    assert "280" in result["error"]
    assert "summary" in result["error"].lower()
