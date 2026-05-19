"""Integration tests for v3.7.40 FINDING #W -- kg_query.details surface
must include date_added + last_relevant_at trimmed to minute precision.

Background
----------
Pre-v3.7.40 the ``_ENTITY_DETAIL_META_KEYS`` whitelist in
``mcp_server.py`` allowed only ``("kind", "summary", "importance",
"content_type")`` through to ``_fetch_entity_details`` callers. The
v3.7.34/v3.7.38 datetime contract surfaced dates on declare_intent /
declare_user_intents / kg_search projections via ``_project_memory``,
but ``mempalace_kg_query`` -- the single-entity lookup primary tool --
bypassed that path entirely. So agents calling ``kg_query(entity='X')``
got back a details payload with NO time signal, despite v3.7.38
writers stamping ``date_added`` in vec metadata since.

v3.7.40 closes the gap:
  - extends ``_ENTITY_DETAIL_META_KEYS`` with date_added + last_relevant_at
  - bridges ``entities.created_at`` -> meta.date_added in the SQL
    fallback path (`_fetch_entity_details_kg_fallback`)
  - applies the v3.7.39 minute-precision trim to surfaced date fields
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_v3740_whitelist_contains_date_fields():
    """The kg_query details whitelist must allow date_added and
    last_relevant_at through. Pre-v3.7.40 these were silently dropped."""
    from mempalace.mcp_server import _ENTITY_DETAIL_META_KEYS

    assert "date_added" in _ENTITY_DETAIL_META_KEYS, (
        "v3.7.40 FINDING #W: kg_query.details whitelist must surface "
        "date_added so agents see when entities were filed"
    )
    assert "last_relevant_at" in _ENTITY_DETAIL_META_KEYS, (
        "v3.7.40 FINDING #W: kg_query.details whitelist must surface "
        "last_relevant_at (touch-on-use clock)"
    )


def test_v3740_kg_fallback_bridges_created_at_to_date_added(monkeypatch):
    """The SQL fallback path must alias entities.created_at to
    meta.date_added so kg_query.details is uniform between vec-backed
    and SQL-only entity lookups."""
    from mempalace import mcp_server

    class _FakeKG:
        def get_entity(self, eid):
            return {
                "name": "ent_x",
                "kind": "entity",
                "importance": 3,
                "content": "some content",
                "created_at": "2026-05-19T08:00:00.000000",
                "last_touched": "2026-05-19T09:30:00.000000",
                "properties": {},
            }

    class _FakeState:
        kg = _FakeKG()
        session_id = None
        active_intent = None

    monkeypatch.setattr(mcp_server, "_STATE", _FakeState())
    result = mcp_server._fetch_entity_details_kg_fallback("ent_x")
    assert result is not None
    meta, content = result
    assert meta["date_added"] == "2026-05-19T08:00:00.000000", (
        "v3.7.40 fallback: created_at must be surfaced as date_added "
        "(trim happens at _fetch_entity_details output, not here)"
    )
    assert meta["last_relevant_at"] == "2026-05-19T09:30:00.000000"
    assert content == "some content"


def test_v3740_kg_fallback_skips_null_dates(monkeypatch):
    """If created_at is NULL (very old entity row), the fallback must
    not synthesize empty-string dates -- leave the field absent so the
    agent sees no date rather than a misleading blank."""
    from mempalace import mcp_server

    class _FakeKG:
        def get_entity(self, eid):
            return {
                "name": "ent_x",
                "kind": "entity",
                "importance": 3,
                "content": "c",
                "created_at": None,
                "last_touched": None,
                "properties": {},
            }

    class _FakeState:
        kg = _FakeKG()
        session_id = None
        active_intent = None

    monkeypatch.setattr(mcp_server, "_STATE", _FakeState())
    meta, _content = mcp_server._fetch_entity_details_kg_fallback("ent_x")
    assert "date_added" not in meta
    assert "last_relevant_at" not in meta
