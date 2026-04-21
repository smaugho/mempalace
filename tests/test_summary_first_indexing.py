"""Regression tests for summary-first indexing.

Grounded in Anthropic Contextual Retrieval
(https://www.anthropic.com/news/contextual-retrieval, 2024 — prepended
per-chunk context reduces retrieval failure ~49%) and Chen et al.
"Dense X Retrieval" (arXiv:2312.06648, 2023 — proposition-granularity
embeddings outperform passage-granularity on open-domain QA). Neither
paper gates by length; every chunk gets the treatment.

Contract enforced at the record-write boundary
(``mcp_server._add_memory_internal``):
  - ``summary`` is ALWAYS required — regardless of content length. For
    long content the summary distills WHAT/WHY; for short content the
    summary should rephrase from a different angle so the summary+content
    pair yields two distinct cosine views of the same semantic.
  - ``summary`` has a hard cap of 280 chars. ``content`` is free-length.
  - When summary is provided, the embedded document is
    ``f"{summary}\\n\\n{content}"`` (Anthropic CR prepend) and
    ``metadata['summary']`` carries the verbatim summary for
    injection-time display.
"""

from __future__ import annotations

from tests.test_mcp_server import _patch_mcp_server, _get_collection


def _setup(monkeypatch, config, palace_path, kg):
    _patch_mcp_server(monkeypatch, config, kg)
    _client, _col = _get_collection(palace_path, create=True)
    del _client

    from mempalace import mcp_server

    # Seed a declared agent so _add_memory_internal does not reject on
    # the is_a agent check.
    kg.add_entity("agent", kind="class", description="root agent class", importance=5)
    kg.add_entity("test_agent", kind="entity", description="harness agent", importance=3)
    kg.add_triple("test_agent", "is_a", "agent")
    kg.add_entity("host_entity", kind="entity", description="host for link", importance=3)
    return mcp_server


def test_any_record_without_summary_rejected(monkeypatch, config, palace_path, kg):
    """Summary is always required — a long content with no summary is rejected."""
    mcp_server = _setup(monkeypatch, config, palace_path, kg)

    long_body = "x" * 500  # any length; rule applies universally
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
    """Short content is not exempt — summary is required there too. The
    summary should rephrase from a different angle, but any non-empty
    ≤280-char string satisfies the gate mechanically."""
    mcp_server = _setup(monkeypatch, config, palace_path, kg)

    result = mcp_server._add_memory_internal(
        content="short fact",
        slug="short-without-summary",
        added_by="test_agent",
        content_type="fact",
        importance=3,
        entity="host_entity",
        # summary omitted — should be rejected
    )

    assert result["success"] is False
    assert "summary" in result["error"].lower()


def test_record_with_valid_summary_stored_with_cr_prepend(monkeypatch, config, palace_path, kg):
    """Valid summary → record created, embedded document is
    ``summary\\n\\ncontent`` (Anthropic CR prepend), metadata carries the
    summary verbatim."""
    mcp_server = _setup(monkeypatch, config, palace_path, kg)

    summary = "Long-record distillation for retrieval-time display."
    body = "content body " * 50  # free-length content
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
    assert doc.startswith(summary + "\n\n"), doc[:120]
    assert body in doc
    assert got["metadatas"][0].get("summary") == summary


def test_short_record_with_different_angle_summary(monkeypatch, config, palace_path, kg):
    """Short content + summary rephrasing from a different angle produces
    two distinct retrieval views. The gate still requires summary; CR
    prepend still applies."""
    mcp_server = _setup(monkeypatch, config, palace_path, kg)

    content = "DB credentials stored in .env.prod, not in the repo"
    # Summary rephrases using different vocabulary — "secrets",
    # "production", "not committed" — broadening the retrieval surface.
    summary = (
        "Production secrets live in the .env.prod file and are deliberately "
        "not committed to version control."
    )
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
    # Both the original content AND the reframe summary live in the
    # embedded document — the pair is what drives the two-view CR gain.
    assert doc.startswith(summary + "\n\n")
    assert content in doc
    assert got["metadatas"][0].get("summary") == summary


def test_oversize_summary_rejected(monkeypatch, config, palace_path, kg):
    """summary capped at 280 chars; longer values rejected with one clear error."""
    mcp_server = _setup(monkeypatch, config, palace_path, kg)

    too_long_summary = "x" * 290
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
