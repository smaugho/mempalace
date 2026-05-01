"""Regression tests for the cold-start entity gate.

Adrian's design lock 2026-05-01: ``mempalace/entity_gate.py`` is the
single chokepoint for entity creation. Every entity row in SQLite must
flow through ``mint_entity`` (or the ``record`` path which validates
summaries on its own); every triple/edge that references an entity
must be guarded by ``assert_entity_exists`` if the caller didn't just
mint it.

These tests lock the gate's contract:

* ``mint_entity`` validates summary, rejects generic 'what' values,
  detects identity collisions on the Level-1 embedding layer, and
  reuses (silently) above ``T_REUSE_WHAT``.
* ``assert_entity_exists`` hard-rejects phantom entity references
  (the pre-cold-start ``INSERT OR IGNORE`` pattern that created
  1,780 untyped entities in the live corpus).
* The 4 phantom sites in ``knowledge_graph.py`` (``add_triple`` x2 +
  ``add_rated_edge`` x2) raise ``PhantomEntityRejected`` when called
  with undeclared subjects/objects.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mempalace.entity_gate import (
    T_COLLISION_WARN,
    T_REUSE_WHAT,
    PhantomEntityRejected,
    WhatStoplistError,
    assert_entity_exists,
)
from mempalace.knowledge_graph import SummaryStructureRequired


# ── Pure-unit tests: no Chroma needed ─────────────────────────────────


def test_thresholds_relate_correctly():
    """Sanity: T_COLLISION_WARN must be strictly below T_REUSE_WHAT.

    The collision-warn band lives between the two; if they overlap or
    invert, the policy logic in mint_entity is meaningless."""
    assert 0.0 < T_COLLISION_WARN < T_REUSE_WHAT < 1.0


def test_assert_entity_exists_passes_when_present(tmp_path: Path):
    """assert_entity_exists is a no-op when the eid is already in
    the entities table."""
    db = tmp_path / "tiny.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE entities (id TEXT PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO entities (id, name) VALUES (?, ?)", ("foo", "Foo"))
    conn.commit()
    # Should not raise.
    assert_entity_exists("foo", conn)


def test_assert_entity_exists_rejects_phantom(tmp_path: Path):
    """assert_entity_exists raises PhantomEntityRejected for an unknown
    eid -- the cold-start replacement for the silent INSERT OR IGNORE
    auto-create."""
    db = tmp_path / "tiny.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE entities (id TEXT PRIMARY KEY, name TEXT)")
    conn.commit()
    with pytest.raises(PhantomEntityRejected) as exc_info:
        assert_entity_exists("nonexistent", conn)
    msg = str(exc_info.value)
    # The error must point the caller at the migration path so a stale
    # caller can fix itself without spelunking through code.
    assert "mempalace_kg_declare_entity" in msg
    assert "nonexistent" in msg


# ── Integration tests: exercise mint_entity end-to-end ────────────────


def _patch_mcp_server(monkeypatch, config, kg):
    """Mirror tests/test_mcp_server.py boot pattern so mint_entity finds
    a wired-up _STATE."""
    from mempalace import mcp_server

    monkeypatch.setattr(mcp_server._STATE, "kg", kg)
    monkeypatch.setattr(mcp_server._STATE, "config", config)
    monkeypatch.setattr(mcp_server._STATE, "session_id", "test_session")
    monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
    monkeypatch.setattr(mcp_server._STATE, "pending_conflicts", None)
    monkeypatch.setattr(mcp_server._STATE, "client_cache", None)
    monkeypatch.setattr(mcp_server._STATE, "collection_cache", None)
    return mcp_server


def test_mint_entity_creates_new_entity(monkeypatch, config, palace_path, kg):
    """Happy path: a fresh palace + valid summary creates a new entity
    with the right SQLite row + 3-level Chroma layout."""
    _patch_mcp_server(monkeypatch, config, kg)
    from mempalace.entity_gate import mint_entity

    eid, was_reused = mint_entity(
        "InjectionGate",
        kind="entity",
        summary={
            "what": "InjectionGate component",
            "why": "filters retrieved memories before injection via Haiku tool-use",
            "scope": "one instance per palace process",
        },
        queries=[
            "what filters retrieved memories",
            "how does the injection gate decide what to drop",
        ],
        added_by="test_agent",
    )
    assert was_reused is False
    assert eid  # non-empty
    # SQLite row exists with the right kind + summary preserved in properties.
    conn = kg._conn()
    row = conn.execute("SELECT id, kind, properties FROM entities WHERE id = ?", (eid,)).fetchone()
    assert row is not None
    assert row["kind"] == "entity"
    import json as _json

    props = _json.loads(row["properties"] or "{}")
    assert "summary" in props
    assert props["summary"]["what"] == "InjectionGate component"
    assert "queries" in props
    assert len(props["queries"]) == 2


def test_mint_entity_rejects_string_summary(monkeypatch, config, palace_path, kg):
    """Strings are rejected at the summary contract layer -- the dict-only
    lock from 2026-04-25 must hold through mint_entity."""
    _patch_mcp_server(monkeypatch, config, kg)
    from mempalace.entity_gate import mint_entity

    with pytest.raises(SummaryStructureRequired):
        mint_entity(
            "BadEntity",
            kind="entity",
            summary="legacy prose summary instead of a dict",  # type: ignore[arg-type]
        )


def test_mint_entity_discrimination_floor_after_summary(monkeypatch, config, palace_path, kg):
    """A 'what' that passes validate_summary (>=5 chars) but is below
    the gate's discrimination floor (>=8 chars) raises WhatStoplistError."""
    _patch_mcp_server(monkeypatch, config, kg)
    from mempalace.entity_gate import mint_entity

    with pytest.raises(WhatStoplistError) as exc_info:
        mint_entity(
            "AlmostShort",
            kind="entity",
            summary={
                "what": "foobar",  # 6 chars -- passes summary but not gate
                "why": "this is plenty long enough to satisfy the why floor",
            },
        )
    assert "discrimination floor" in str(exc_info.value)


def test_mint_entity_rejects_stoplist_what(monkeypatch, config, palace_path, kg):
    """'what' values matching the generic-phrase stoplist raise
    WhatStoplistError -- they would create non-discriminative identity
    embeddings."""
    _patch_mcp_server(monkeypatch, config, kg)
    from mempalace.entity_gate import mint_entity

    with pytest.raises(WhatStoplistError) as exc_info:
        mint_entity(
            "StoplistWhat",
            kind="entity",
            summary={
                "what": "the project",  # canonical generic phrase
                "why": "this why is long enough to pass the structural floor",
            },
        )
    assert "stoplist" in str(exc_info.value).lower()


def test_mint_entity_silent_reuse_on_identical_what(monkeypatch, config, palace_path, kg):
    """Two mint_entity calls with the same 'what' should produce the same
    eid on the second call (silent reuse path; cosine = 1.0 >= T_REUSE_WHAT).
    The second call returns ``was_reused=True``."""
    _patch_mcp_server(monkeypatch, config, kg)
    from mempalace.entity_gate import mint_entity

    summary = {
        "what": "AdrianHomeOffice setup",
        "why": "primary workstation Adrian uses for mempalace development",
    }
    eid1, reused1 = mint_entity("AdrianHomeOffice", kind="entity", summary=summary)
    assert reused1 is False

    eid2, reused2 = mint_entity(
        "AdrianHomeOffice2",
        kind="entity",
        summary=dict(summary),  # same what
    )
    # Identical 'what' embeds at cosine 1.0 -- well above T_REUSE_WHAT.
    assert reused2 is True
    assert eid2 == eid1


# ── Phantom-reject regression tests on knowledge_graph.add_triple/add_rated_edge ──


def test_add_triple_rejects_undeclared_subject(kg):
    """add_triple with an undeclared subject raises PhantomEntityRejected
    after the cold-start refactor at knowledge_graph.py:2072+."""
    kg.add_entity("known_obj", kind="entity", content="known")
    with pytest.raises(PhantomEntityRejected):
        kg.add_triple(
            "phantom_subject",
            "relates_to",
            "known_obj",
            statement="phantom_subject relates to known_obj",
        )


def test_add_triple_rejects_undeclared_object(kg):
    """add_triple with an undeclared object raises PhantomEntityRejected."""
    kg.add_entity("known_sub", kind="entity", content="known")
    with pytest.raises(PhantomEntityRejected):
        kg.add_triple(
            "known_sub",
            "relates_to",
            "phantom_object",
            statement="known_sub relates to phantom_object",
        )


def test_add_triple_passes_when_both_declared(kg):
    """When both endpoints exist, add_triple passes the gate and writes
    the edge as before."""
    kg.add_entity("alice_test", kind="entity", content="Alice")
    kg.add_entity("bob_test", kind="entity", content="Bob")
    tid = kg.add_triple(
        "alice_test",
        "knows",
        "bob_test",
        statement="alice_test knows bob_test from work",
    )
    assert tid  # non-empty triple id


def test_add_rated_edge_rejects_undeclared_endpoints(kg):
    """add_rated_edge mirrors add_triple's hard-reject policy."""
    kg.add_entity("ctx_known", kind="context", content="known context")
    with pytest.raises(PhantomEntityRejected):
        kg.add_rated_edge(
            "ctx_known",
            "rated_useful",
            "phantom_memory",
            confidence=0.8,
        )
    # And the reverse direction.
    kg.add_entity("mem_known", kind="record", content="known memory")
    with pytest.raises(PhantomEntityRejected):
        kg.add_rated_edge(
            "phantom_context",
            "rated_useful",
            "mem_known",
            confidence=0.8,
        )


def test_add_rated_edge_passes_when_both_declared(kg):
    """When both context + memory exist, add_rated_edge writes the rating."""
    kg.add_entity("ctx_x", kind="context", content="ctx x")
    kg.add_entity("mem_x", kind="record", content="mem x")
    tid = kg.add_rated_edge(
        "ctx_x",
        "rated_useful",
        "mem_x",
        confidence=0.7,
    )
    assert tid
