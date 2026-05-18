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


# ── Seed-ontology regression: every seed entity carries a summary dict ──


def test_wake_up_requires_context_on_fresh_palace(monkeypatch, config, palace_path, kg):
    """Cold-start lock 2026-05-01 (Adrian's wake_up analysis):
    ``_bootstrap_agent_if_missing`` must hard-reject when called for a
    new agent without a real ``context`` dict. No template fallback,
    no silent degradation -- the agent must introduce themselves with
    a real ``{what, why, scope?}`` summary on first wake_up.
    """
    from mempalace import mcp_server
    from mempalace.mcp_server import (
        AgentBootstrapContextRequired,
        _bootstrap_agent_if_missing,
    )

    monkeypatch.setattr(mcp_server._STATE, "kg", kg)
    monkeypatch.setattr(mcp_server._STATE, "config", config)
    monkeypatch.setattr(mcp_server._STATE, "session_id", "test-bootstrap")
    monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
    monkeypatch.setattr(mcp_server._STATE, "client_cache", None)
    monkeypatch.setattr(mcp_server._STATE, "collection_cache", None)
    kg.seed_ontology()

    # Missing context: hard-fail.
    with pytest.raises(AgentBootstrapContextRequired):
        _bootstrap_agent_if_missing("ga_agent_alpha")

    # Empty context: hard-fail (validate_context rejects).
    with pytest.raises(AgentBootstrapContextRequired):
        _bootstrap_agent_if_missing("ga_agent_beta", context={})

    # Context with summary but missing queries/keywords/entities: hard-fail.
    # The shared validate_context contract requires all four fields;
    # wake_up uses the same validator as every other write tool.
    with pytest.raises(AgentBootstrapContextRequired):
        _bootstrap_agent_if_missing(
            "ga_agent_gamma",
            context={
                "summary": {
                    "what": "ga_agent_gamma -- partial context",
                    "why": "missing queries+keywords+entities should still fail",
                },
            },
        )

    # Adrian's wake_up exception 2026-05-01: entities is OPTIONAL at
    # wake_up only (entities_min=0). Other write tools keep entities
    # mandatory. So queries+keywords+summary without entities SUCCEEDS
    # at wake_up, even though the same shape would fail at
    # kg_declare_entity. This test documents the carve-out.
    _bootstrap_agent_if_missing(
        "ga_agent_no_entities",
        context={
            "queries": ["who", "what role this agent plays"],
            "keywords": ["GA", "Adrian"],
            "summary": {
                "what": "ga_agent_no_entities -- carve-out test",
                "why": "first wake_up has zero accumulated graph associations so entities=[] is allowed",
            },
        },
    )
    no_ent = kg.get_entity("ga_agent_no_entities")
    assert no_ent is not None
    assert no_ent.get("kind") == "entity"

    # Complete Context (with entities): also succeeds.
    real_context = {
        "queries": [
            "who is this agent",
            "what work does it do",
            "what runtime does it operate in",
        ],
        "keywords": ["GA", "Adrian", "mempalace"],
        "entities": ["agent"],
        "summary": {
            "what": "ga_agent_delta -- mempalace dev companion",
            "why": "general-purpose Claude session that audits + ships mempalace internals on Adrian's Windows workstation",
            "scope": "Adrian's home office; Opus long-context sessions",
        },
    }
    _bootstrap_agent_if_missing("ga_agent_delta", context=real_context)
    ent = kg.get_entity("ga_agent_delta")
    assert ent is not None
    assert ent.get("kind") == "entity"


def test_wake_up_accepts_json_stringified_context(monkeypatch, config, palace_path, kg):
    """Cold-start lock 2026-05-01 (Adrian's MCP-transport deadlock fix):
    some MCP clients JSON-stringify object params on the wire even when
    the schema declares ``type: object``. ``_bootstrap_agent_if_missing``
    must coerce the string back into a dict before its isinstance check
    fires -- otherwise the only entry point for fresh-palace bootstrap
    deadlocks (wake_up rejects, declare_user_intents redirects to
    wake_up, hook gate blocks every other path).
    """
    import json as _json

    from mempalace import mcp_server
    from mempalace.mcp_server import _bootstrap_agent_if_missing

    monkeypatch.setattr(mcp_server._STATE, "kg", kg)
    monkeypatch.setattr(mcp_server._STATE, "config", config)
    monkeypatch.setattr(mcp_server._STATE, "session_id", "test-stringified")
    monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
    monkeypatch.setattr(mcp_server._STATE, "client_cache", None)
    monkeypatch.setattr(mcp_server._STATE, "collection_cache", None)
    kg.seed_ontology()

    real_context = {
        "queries": [
            "who is the GA agent",
            "what work does GA do for Adrian",
            "GA agent runtime details",
        ],
        "keywords": ["GA", "Adrian", "mempalace"],
        "summary": {
            "what": "ga_agent_stringified -- mempalace dev companion (transport fix test)",
            "why": "regression guard for MCP transport JSON-stringifying object params on the wire",
            "scope": "test-stringified session",
        },
    }
    # Pass the context as a JSON STRING -- exactly what some MCP
    # transports produce on the wire. The bootstrap must coerce.
    stringified = _json.dumps(real_context)
    _bootstrap_agent_if_missing("ga_agent_stringified", context=stringified)

    ent = kg.get_entity("ga_agent_stringified")
    assert ent is not None
    assert ent.get("kind") == "entity"


def test_wake_up_rejects_unparseable_context_string(monkeypatch, config, palace_path, kg):
    """The transport-coerce path must NOT swallow real bad input. A
    string that fails json.loads falls through to the canonical
    isinstance(dict) check, which raises with the required-shape
    message so the caller sees a clear error -- not a JSON parse trace.
    """
    from mempalace import mcp_server
    from mempalace.mcp_server import (
        AgentBootstrapContextRequired,
        _bootstrap_agent_if_missing,
    )

    monkeypatch.setattr(mcp_server._STATE, "kg", kg)
    monkeypatch.setattr(mcp_server._STATE, "config", config)
    monkeypatch.setattr(mcp_server._STATE, "session_id", "test-bad-string")
    monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
    monkeypatch.setattr(mcp_server._STATE, "client_cache", None)
    monkeypatch.setattr(mcp_server._STATE, "collection_cache", None)
    kg.seed_ontology()

    with pytest.raises(AgentBootstrapContextRequired) as exc_info:
        _bootstrap_agent_if_missing("ga_agent_bad_string", context="not valid json {{{")
    # The canonical Context-shape message (not a JSON parse trace).
    assert "describes THIS AGENT" in str(exc_info.value)


def test_wake_up_idempotent_on_existing_agent(monkeypatch, config, palace_path, kg):
    """Re-wake_up of an existing agent ignores ``context`` (zero-friction
    re-boot). Only the FIRST wake_up of a given agent name needs real
    context; subsequent sessions read-back the existing entity."""
    from mempalace import mcp_server
    from mempalace.mcp_server import _bootstrap_agent_if_missing

    monkeypatch.setattr(mcp_server._STATE, "kg", kg)
    monkeypatch.setattr(mcp_server._STATE, "config", config)
    monkeypatch.setattr(mcp_server._STATE, "session_id", "test-rebootstrap")
    monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
    monkeypatch.setattr(mcp_server._STATE, "client_cache", None)
    monkeypatch.setattr(mcp_server._STATE, "collection_cache", None)
    kg.seed_ontology()

    real_context = {
        "queries": ["who is this agent", "what work does it do"],
        "keywords": ["GA", "Adrian"],
        "entities": ["agent"],
        "summary": {
            "what": "ga_agent_repeat -- mempalace dev companion",
            "why": "general-purpose Claude session for ontology refactor today",
            "scope": "test-rebootstrap session",
        },
    }
    _bootstrap_agent_if_missing("ga_agent_repeat", context=real_context)
    # Second call without context must NOT raise -- agent already exists.
    _bootstrap_agent_if_missing("ga_agent_repeat")
    # Confirm the SQLite row is still there and unchanged kind.
    ent = kg.get_entity("ga_agent_repeat")
    assert ent is not None
    assert ent.get("kind") == "entity"


def test_wake_up_distinct_agents_do_not_collide_at_identity_layer(
    monkeypatch, config, palace_path, kg
):
    """Cold-start lock regression guard: two agents with DIFFERENT real
    identity contexts must not silently merge at the gate's identity
    layer. The pre-fix template ``f"{agent} (mempalace agent)"``
    produced ~0.95 cosine across agent identities and the gate's
    T_REUSE_WHAT=0.92 silently reused the first agent's eid for the
    second. Real distinctive whats keep them separate."""
    from mempalace import mcp_server
    from mempalace.mcp_server import _bootstrap_agent_if_missing

    monkeypatch.setattr(mcp_server._STATE, "kg", kg)
    monkeypatch.setattr(mcp_server._STATE, "config", config)
    monkeypatch.setattr(mcp_server._STATE, "session_id", "test-distinct-agents")
    monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
    monkeypatch.setattr(mcp_server._STATE, "client_cache", None)
    monkeypatch.setattr(mcp_server._STATE, "collection_cache", None)
    kg.seed_ontology()

    ga_context = {
        "queries": [
            "who is the GA agent",
            "what mempalace work does GA do",
            "GA agent runtime details",
        ],
        "keywords": ["GA", "Adrian", "mempalace"],
        "entities": ["agent"],
        "summary": {
            "what": "ga_agent_unique -- mempalace dev general-purpose",
            "why": "general-purpose Claude session that audits + ships mempalace internals on Adrian's Windows workstation",
            "scope": "Adrian's home office; Opus long-context sessions",
        },
    }
    tl_context = {
        "queries": [
            "who is the technical lead agent",
            "what paperclip work does TL do",
            "TL specialist coordination",
        ],
        "keywords": ["TL", "paperclip", "specialist"],
        "entities": ["agent"],
        "summary": {
            "what": "tl_agent_unique -- paperclip technical lead role",
            "why": "technical lead specialist coordinating multi-agent work on the paperclip pipeline; reviews PRs and unblocks specialists",
            "scope": "paperclip runtime; ephemeral specialist sessions",
        },
    }
    _bootstrap_agent_if_missing("ga_agent_unique", context=ga_context)
    _bootstrap_agent_if_missing("tl_agent_unique", context=tl_context)

    ga_ent = kg.get_entity("ga_agent_unique")
    tl_ent = kg.get_entity("tl_agent_unique")
    assert ga_ent is not None and tl_ent is not None
    assert ga_ent["id"] != tl_ent["id"], (
        "agents collided at the identity layer -- the gate silently reused "
        f"one eid for both. ga={ga_ent['id']!r}, tl={tl_ent['id']!r}"
    )


def test_operation_entities_skip_summary_contract(tmp_path):
    """Cold-start lock 2026-05-01 (Adrian's op-summary analysis):
    operations are graph-only entities identified by their args_summary
    fingerprint; their parent context's summary carries the WHY. The
    summary contract (every entity carries {what,why,scope?}) does NOT
    apply to kind='operation' -- mirrors the kind='user_message' carve-
    out at _sync_entity_to_chromadb. Op rows persist with NO
    properties.summary; retrieval finds them only via graph traversal
    (executed_op / performed_well / performed_poorly).
    """
    import json as _json

    from mempalace.knowledge_graph import KnowledgeGraph

    db = tmp_path / "ops.sqlite3"
    kg = KnowledgeGraph(db_path=str(db))
    kg.seed_ontology()
    # Mint an op directly via add_entity (matches what finalize_intent
    # does for op-promotion). NO summary in properties -- the carve-out
    # contract.
    op_id = "op_read_test_aaa"
    kg.add_entity(
        op_id,
        kind="operation",
        content="Read op: {file_path}",
        importance=2,
        properties={
            "tool": "Read",
            "args_summary": "{file_path}",
            "context_id": "ctx_test_aaa",
            "quality": 4,
            "reason": "fast lookup",
        },
    )
    ent = kg.get_entity(op_id)
    assert ent is not None
    assert ent["kind"] == "operation"
    props = ent.get("properties") or {}
    if isinstance(props, str):
        props = _json.loads(props)
    # The carve-out contract: ops do NOT carry a summary dict.
    assert "summary" not in props, (
        f"operation entity unexpectedly carries summary -- the carve-out "
        f"contract requires kind='operation' to skip the summary "
        f"requirement (args_summary fingerprint IS the identity, parent "
        f"context's summary carries the WHY). Got properties keys: "
        f"{sorted(props.keys())!r}"
    )
    # The args_summary fingerprint IS persisted (it's the identity).
    assert props.get("args_summary") == "{file_path}"


def test_seed_ontology_persists_structured_summary_on_every_entity(tmp_path):
    """Cold-start invariant lockbox: after seed_ontology + seed_all run,
    every entity row carries a ``properties.summary`` dict that passes
    ``coerce_summary_for_persist``.

    This is the regression guard for the 12-site bypass-surface audit:
    pre-cold-start the bootstrap entities slipped through the summary
    contract because seed_ontology called kg.add_entity directly. The
    cold-start design lock (2026-05-01) routes every seed entry through
    ``_build_seed_summary`` so the gardener / kg_query / future linter
    can introspect bootstrap entities on the same axes as agent-declared
    ones.
    """
    import json as _json
    import sqlite3 as _sqlite3

    from mempalace.knowledge_graph import KnowledgeGraph, coerce_summary_for_persist
    from mempalace.seed import seed_all

    db = tmp_path / "seed.sqlite3"
    kg = KnowledgeGraph(db_path=str(db))
    # _init_db calls seed_ontology when MEMPALACE_SKIP_SEED is unset; force
    # the run here regardless so the assertion is unambiguous.
    kg.seed_ontology()
    seed_all(kg)

    conn = _sqlite3.connect(str(db))
    rows = conn.execute("SELECT id, kind, properties FROM entities").fetchall()
    assert rows, "expected seed_ontology to produce at least one entity"

    missing: list[tuple[str, str]] = []
    invalid: list[tuple[str, str, str]] = []
    for eid, kind, props_json in rows:
        try:
            props = _json.loads(props_json or "{}")
        except _json.JSONDecodeError:
            props = {}
        summary = props.get("summary")
        if not summary:
            missing.append((eid, kind))
            continue
        try:
            coerce_summary_for_persist(summary, context_for_error=f"seed_regression({eid!r})")
        except Exception as exc:
            invalid.append((eid, kind, repr(exc)))
    conn.close()

    assert not missing, (
        f"seed entities without properties.summary (cold-start invariant violated): {missing!r}"
    )
    assert not invalid, (
        f"seed entities with invalid summary (failed coerce_summary_for_persist): {invalid!r}"
    )


# ─────────────────────────────────────────────────────────────────────
# v3.7.29 regression (Adrian directive 2026-05-18): Level-3 BODY view
# reinstatement. The 2026-04 refactor had dropped L3 entirely on the
# theory that MiniLM-L6's 256-token ceiling made long-content embeds
# noisy. Adrian re-locked the design: content MUST be a stored view
# in the multi-view system alongside L1 identity + L2 summary + L4
# probes. mint_entity now emits {eid}__body whenever body_text is
# non-empty AND distinct from BOTH 'what' and 'rendered_summary'.
# These tests lock the contract so a future refactor cannot silently
# drop the view again.
# ─────────────────────────────────────────────────────────────────────


def _body_view_present(palace_path, eid):
    """Return (present, metadata) for the {eid}__body row in records."""
    from mempalace.vector_store import RECORDS_COLLECTION, get_vector_store

    vs = get_vector_store(palace_path)
    body_id = f"{eid}__body"
    got = vs.get(RECORDS_COLLECTION, ids=[body_id], include=["metadatas"])
    if got.ids and got.ids[0] == body_id:
        meta = got.metadatas[0] if got.metadatas else None
        return True, meta
    return False, None


def test_v3729_mint_entity_emits_body_view_when_content_distinct(
    monkeypatch, config, palace_path, kg
):
    """body_text distinct from what AND rendered_summary -> __body view
    exists with view_kind='body' + view_index=-2."""
    _patch_mcp_server(monkeypatch, config, kg)
    from mempalace.entity_gate import mint_entity

    eid, _was_reused = mint_entity(
        "DistinctBodyEntity",
        kind="entity",
        summary={
            "what": "DistinctBodyEntity",
            "why": "carries a body_text that adds information beyond the summary",
            "scope": "v3.7.29 regression",
        },
        queries=["probe one", "probe two"],
        added_by="test_agent",
        body_text=(
            "This is the long-form body content. It elaborates "
            "on the entity's purpose with details not present in "
            "the rendered summary prose. The L3 body view must "
            "embed THIS text, not the summary."
        ),
    )

    present, meta = _body_view_present(palace_path, eid)
    assert present, (
        "v3.7.29 regression: " + str(eid) + "__body view not found in vec "
        "store; mint_entity must emit L3 body view when body_text "
        "is distinct from rendered_summary"
    )
    assert meta is not None
    assert meta.get("view_kind") == "body", "body view metadata missing view_kind='body': " + repr(
        meta
    )
    assert meta.get("view_index") == -2, "body view metadata missing view_index=-2: " + repr(meta)
    assert meta.get("entity_id") == eid


def test_v3729_mint_entity_skips_body_view_when_body_equals_summary(
    monkeypatch, config, palace_path, kg
):
    """body_text equal to rendered_summary -> no __body view (avoids
    duplicate L2/L3 vector)."""
    _patch_mcp_server(monkeypatch, config, kg)
    from mempalace.entity_gate import mint_entity
    from mempalace.knowledge_graph import serialize_summary_for_embedding

    summary = {
        "what": "DuplicateBodyEntity",
        "why": "body_text is the same as rendered_summary, so no L3 view should be added",
        "scope": "v3.7.29 regression",
    }
    rendered = serialize_summary_for_embedding(summary)
    eid, _was_reused = mint_entity(
        "DuplicateBodyEntity",
        kind="entity",
        summary=summary,
        queries=["probe one"],
        added_by="test_agent",
        body_text=rendered,
    )

    present, _meta = _body_view_present(palace_path, eid)
    assert not present, (
        "v3.7.29 regression: " + str(eid) + "__body view exists despite "
        "body_text == rendered_summary; the helper distinctness "
        "check should have skipped the write"
    )


def test_v3729_mint_entity_skips_body_view_when_no_body_text(monkeypatch, config, palace_path, kg):
    """No body_text provided -> no __body view (most non-record
    entities take this path; their content IS the rendered summary)."""
    _patch_mcp_server(monkeypatch, config, kg)
    from mempalace.entity_gate import mint_entity

    eid, _was_reused = mint_entity(
        "NoBodyEntity",
        kind="entity",
        summary={
            "what": "NoBodyEntity",
            "why": "no body_text passed; only L1+L2+L4 views should exist",
            "scope": "v3.7.29 regression",
        },
        queries=["probe one"],
        added_by="test_agent",
    )

    present, _meta = _body_view_present(palace_path, eid)
    assert not present, (
        "v3.7.29 regression: " + str(eid) + "__body view exists despite no "
        "body_text arg; mint_entity defaults body_text to None and "
        "skips L3 emission when caller did not supply distinct body"
    )


# ─────────────────────────────────────────────────────────────────────
# v3.7.36 FINDING #U writer-side complement: mint_entity must stamp
# date_added + last_relevant_at on every fresh vec write, mirroring the
# kind=record write path. Pre-v3.7.36 only last_touched was set; the
# 99.7%-of-corpus gap that v3.7.35 had to bridge via SQL fresh-fetch
# at read time. With v3.7.36 the writer stops perpetuating the gap;
# the v3.7.35 bridge becomes the legacy-data safety net only.
# ─────────────────────────────────────────────────────────────────────


def _identity_view_meta(palace_path, eid):
    """Return the {eid}__identity vec row metadata dict (the v3.7.36
    fields land on every view; identity is the canonical read site)."""
    from mempalace.vector_store import RECORDS_COLLECTION, get_vector_store

    vs = get_vector_store(palace_path)
    identity_id = f"{eid}__identity"
    got = vs.get(RECORDS_COLLECTION, ids=[identity_id], include=["metadatas"])
    if got.ids and got.ids[0] == identity_id and got.metadatas:
        return got.metadatas[0] or {}
    return {}


def test_v3736_mint_entity_stamps_date_added_in_vec_metadata(monkeypatch, config, palace_path, kg):
    """v3.7.36 FINDING #U writer fix: every fresh entity write MUST
    carry date_added in vec metadata so the v3.7.35 fresh-fetch bridge
    isn't load-bearing forever. The field flows through to the agent
    via the v3.7.34 _project_memory hoist (extras['metadata']
    -> top-level entry['date_added'])."""
    _patch_mcp_server(monkeypatch, config, kg)
    from mempalace.entity_gate import mint_entity

    eid, _was_reused = mint_entity(
        "V3736DateStampedEntity",
        kind="entity",
        summary={
            "what": "V3736DateStampedEntity",
            "why": "verify writer stamps date_added on identity view metadata",
            "scope": "v3.7.36 regression",
        },
        queries=["probe one"],
        added_by="test_agent",
    )
    meta = _identity_view_meta(palace_path, eid)
    assert "date_added" in meta, (
        "v3.7.36 FINDING #U writer fix: identity view metadata must "
        "carry date_added so _project_memory can hoist it for the "
        "agent without depending on the v3.7.35 SQL bridge. "
        f"actual keys: {sorted(meta.keys())}"
    )
    assert meta["date_added"], "date_added must be a non-empty ISO string"


def test_v3736_mint_entity_stamps_last_relevant_at_in_vec_metadata(
    monkeypatch, config, palace_path, kg
):
    """Same contract for last_relevant_at -- the touch-on-use clock
    field that scoring.hybrid_score reads as last_relevant_iso."""
    _patch_mcp_server(monkeypatch, config, kg)
    from mempalace.entity_gate import mint_entity

    eid, _was_reused = mint_entity(
        "V3736RelevantAtEntity",
        kind="entity",
        summary={
            "what": "V3736RelevantAtEntity",
            "why": "verify writer stamps last_relevant_at on identity view metadata",
            "scope": "v3.7.36 regression",
        },
        queries=["probe one"],
        added_by="test_agent",
    )
    meta = _identity_view_meta(palace_path, eid)
    assert "last_relevant_at" in meta, (
        "v3.7.36 FINDING #U writer fix: identity view metadata must "
        "carry last_relevant_at so hybrid_score's recency-bonus axis "
        "has a real value on fresh entity writes. "
        f"actual keys: {sorted(meta.keys())}"
    )
    # At write time, last_relevant_at equals date_added (both = now_iso);
    # touch-on-use updates entities.last_touched later but vec stays the
    # write-time snapshot.
    assert meta["last_relevant_at"] == meta.get("date_added"), (
        "v3.7.36: at write time, last_relevant_at and date_added are "
        "both set to now_iso so the recency clock starts equal to the "
        "creation clock"
    )


pytestmark = pytest.mark.integration
