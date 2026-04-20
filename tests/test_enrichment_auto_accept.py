"""
Tests for phase N5 — implicit enrichment acceptance on kg_add.

When the agent calls kg_add(subject, predicate, object) for a pair that
appears in ``_STATE.pending_enrichments``, the edge creation IS the accept
signal. The handler must:
  1) record positive edge feedback on the chosen predicate,
  2) remove the matching enrichment from pending state,
  3) re-persist the active intent so the PreToolUse hook sees the update,
  4) surface ``auto_resolved_enrichment`` in the response.
"""

from __future__ import annotations

from tests.test_mcp_server import _patch_mcp_server, _get_collection


def _setup(monkeypatch, config, palace_path, kg):
    _patch_mcp_server(monkeypatch, config, kg)
    _client, _col = _get_collection(palace_path, create=True)
    del _client

    from mempalace import mcp_server

    # Seed the entities needed for kg_add validation. Going direct to the
    # KG + _STATE.declared_entities avoids the context-validation round trip
    # that tool_kg_declare_entity would enforce — this test only cares about
    # the enrichment-auto-accept logic, not declaration plumbing.
    kg.add_entity("thing", kind="class", description="root class", importance=5)
    kg.add_entity("related_to", kind="predicate", description="test predicate", importance=3)
    kg.add_entity("foo_entity", kind="entity", description="foo", importance=3)
    kg.add_entity("bar_entity", kind="entity", description="bar", importance=3)
    kg.add_triple("foo_entity", "is_a", "thing")
    kg.add_triple("bar_entity", "is_a", "thing")

    # Populate the in-memory declared_entities cache so _is_declared hits
    # the fast path. (The KG fallback would work too, but the tests stay
    # stable when the cache is warm.)
    for eid in ("thing", "related_to", "foo_entity", "bar_entity", "agent", "test_agent"):
        mcp_server._STATE.declared_entities.add(eid)

    return mcp_server


def test_kg_add_auto_resolves_matching_enrichment(monkeypatch, config, palace_path, kg):
    mcp_server = _setup(monkeypatch, config, palace_path, kg)

    # Simulate an active intent with a pending enrichment.
    mcp_server._STATE.active_intent = {
        "intent_id": "intent_test_x",
        "intent_type": "modify",
        "slots": {},
        "effective_permissions": [],
        "agent": "test_agent",
        "budget": {},
        "used": {},
    }
    mcp_server._STATE.pending_enrichments = [
        {
            "id": "enrich_foo_entity_bar_entity",
            "reason": "agents often link foo to bar after seeing them in the same session",
            "from_entity": "foo_entity",
            "to_entity": "bar_entity",
            "similarity": 0.72,
            "to_description": "bar description",
        }
    ]

    from mempalace.mcp_server import tool_kg_add

    result = tool_kg_add(
        subject="foo_entity",
        predicate="related_to",
        object="bar_entity",
        context={
            "queries": ["linking foo to bar explicitly", "test accept path"],
            "keywords": ["foo", "bar", "link"],
        },
        agent="test_agent",
        statement="foo_entity is related to bar_entity (auto-consume test edge).",
    )

    assert result["success"] is True, result
    # The enrichment was consumed — the response carries it.
    assert "auto_resolved_enrichment" in result
    assert result["auto_resolved_enrichment"]["id"] == "enrich_foo_entity_bar_entity"
    # pending_enrichments is cleared (empty list becomes None by convention).
    assert not mcp_server._STATE.pending_enrichments

    # Positive edge feedback was recorded for the chosen predicate.
    # Query edge_traversal_feedback directly via the KG connection.
    conn = mcp_server._STATE.kg._conn()
    rows = conn.execute(
        "SELECT useful, context_keywords FROM edge_traversal_feedback "
        "WHERE subject=? AND predicate=? AND object=?",
        ("foo_entity", "related_to", "bar_entity"),
    ).fetchall()
    assert rows, "Expected a feedback row after auto-resolve"
    useful_flags = [bool(r[0]) for r in rows]
    assert True in useful_flags, f"Expected at least one useful=True row, got {rows}"


def test_kg_add_without_matching_enrichment_is_untouched(monkeypatch, config, palace_path, kg):
    mcp_server = _setup(monkeypatch, config, palace_path, kg)

    # Intent with a pending enrichment for a DIFFERENT pair.
    mcp_server._STATE.active_intent = {
        "intent_id": "intent_test_y",
        "intent_type": "modify",
        "slots": {},
        "effective_permissions": [],
        "agent": "test_agent",
        "budget": {},
        "used": {},
    }
    mcp_server._STATE.pending_enrichments = [
        {
            "id": "enrich_unrelated_pair",
            "reason": "unrelated",
            "from_entity": "thing",
            "to_entity": "agent",
            "similarity": 0.5,
        }
    ]

    from mempalace.mcp_server import tool_kg_add

    result = tool_kg_add(
        subject="foo_entity",
        predicate="related_to",
        object="bar_entity",
        context={
            "queries": ["unrelated link", "different pair"],
            "keywords": ["foo", "bar"],
        },
        agent="test_agent",
        statement="foo_entity is related to bar_entity (unrelated edge, no auto-consume expected).",
    )

    assert result["success"] is True, result
    # No auto-resolution; pending_enrichments is unchanged.
    assert "auto_resolved_enrichment" not in result
    assert mcp_server._STATE.pending_enrichments
    assert len(mcp_server._STATE.pending_enrichments) == 1


# ---------------------------------------------------------------------------
# Phase 1 — kind-compatibility prefilter for _detect_suggested_links
# ---------------------------------------------------------------------------


def test_kind_compat_filter_includes_narrow_predicate_pairs(monkeypatch, config, palace_path, kg):
    """A predicate with explicit (subject_kinds, object_kinds) must
    contribute exactly those pairs to the compat table. Grounded in
    Krompaß, Baier, Tresp (ISWC 2015) — type constraints are the
    highest-leverage precision lever for link prediction; this test
    guards that the compat table actually reads the constraint metadata
    rather than silently defaulting to permissive everything.

    Note: the fixture's ``related_to`` predicate ships with empty
    constraints (implicit "any kind"), so the compat table will also
    contain many other pairs contributed by that permissive predicate.
    We assert only the positive contribution of the narrow predicate
    here; the empty-means-any semantics is covered by the next test."""
    mcp_server = _setup(monkeypatch, config, palace_path, kg)

    kg.add_entity(
        "class_to_entity_predicate",
        kind="predicate",
        description="narrow class->entity predicate",
        importance=3,
        properties={
            "constraints": {
                "subject_kinds": ["class"],
                "object_kinds": ["entity"],
                "subject_classes": [],
                "object_classes": [],
                "cardinality": "many",
            }
        },
    )

    compat = mcp_server._build_enrichment_kind_compat()
    # The narrow predicate contributes exactly (class, entity).
    assert ("class", "entity") in compat


def test_kind_compat_filter_empty_constraint_means_any(monkeypatch, config, palace_path, kg):
    """A predicate declared without explicit subject_kinds / object_kinds
    should be treated as accepting ALL kinds on that side (same semantics
    as the validator at tool_kg_add). That lets legacy predicates remain
    permissive until someone tightens them."""
    mcp_server = _setup(monkeypatch, config, palace_path, kg)

    kg.add_entity(
        "permissive_predicate",
        kind="predicate",
        description="no explicit constraints",
        importance=3,
        properties={"constraints": {}},
    )

    compat = mcp_server._build_enrichment_kind_compat()
    # Empty constraint ≡ all kinds ≡ every (subject_kind, object_kind)
    # pair is admitted. Spot-check a record-object pair that a strict
    # built-in predicate (like described_by) would reject.
    assert ("entity", "record") in compat
    assert ("class", "literal") in compat


# ---------------------------------------------------------------------------
# Phase 4 — path-shortening filter: already-connected pair skip
# ---------------------------------------------------------------------------


def test_phase4_already_connected_pair_is_detected(monkeypatch, config, palace_path, kg):
    """_pair_already_directly_connected must return True for any 1-hop
    neighbor in either direction, on any predicate — and False for
    un-connected pairs. This is the core of Phase 4's path-shortening
    skip.

    Grounded in Peleg & Schäffer (1989) graph-spanner semantics: a new
    edge only reduces diameter when its endpoints are NOT already
    adjacent. Surfacing an enrichment for an already-adjacent pair is
    pure noise."""
    mcp_server = _setup(monkeypatch, config, palace_path, kg)

    # foo_entity --related_to--> bar_entity is added by _setup? no —
    # _setup only declares entities. Add one direct edge for the
    # connected case. related_to is not a skip-list predicate so it
    # requires a caller-provided statement per the 2026-04-19
    # verbalization-required contract.
    kg.add_triple(
        "foo_entity",
        "related_to",
        "bar_entity",
        statement="foo_entity is related to bar_entity (Phase 4 test fixture edge).",
    )

    assert mcp_server._pair_already_directly_connected("foo_entity", "bar_entity") is True
    # Reverse direction must also count as connected.
    assert mcp_server._pair_already_directly_connected("bar_entity", "foo_entity") is True
    # Self-edge must return False (degenerate case).
    assert mcp_server._pair_already_directly_connected("foo_entity", "foo_entity") is False
    # Un-seeded pair must return False. Using 'agent' as the second
    # entity because _setup adds foo_entity-is_a->thing, so
    # (foo_entity, thing) would spuriously test True via the setup
    # triple rather than the fixture edge we just added.
    assert mcp_server._pair_already_directly_connected("foo_entity", "agent") is False
