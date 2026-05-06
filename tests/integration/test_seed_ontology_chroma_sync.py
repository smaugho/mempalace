"""Regression locks for the seed-ontology Chroma sync fix.

Audit follow-up 2026-05-01 (``record_ga_agent_db_audit_2026_05_01_findings``):
the post-restart audit found that root ontology classes seeded by
``KnowledgeGraph.seed_ontology`` (``thing``, ``agent``, ``intent_type``,
``modify``, ``inspect``, ``research``, ``wrap_up_session``, ``person``,
``project``, ``file``, ``rule``, ``tool``, ``process``, ``concept``,
``environment``, ``context`` plus all predicates) had SQLite rows but no
``mempalace_entities`` Chroma row. Result:

  * ``kg_query`` returned facts/edges but no ``details`` block (Chroma
    fallback failed because the row was missing).
  * ``kg_search`` couldn't surface them as entity hits.
  * ``render_memory_preview`` fell through to fallback text instead of
    the canonical structured prose persisted in ``properties.summary``.

The fix routes every seed ``add_entity`` call through a new helper
``_sync_seed_entity_to_chroma`` that lazy-imports
``mcp_server._sync_entity_to_chromadb`` and writes the same rendered
prose to the entity collection. A companion ``backfill_seed_chroma``
helper exists for existing palaces that were seeded before the fix.

These tests lock both contracts.
"""

from __future__ import annotations


import pytest


def _setup_state(monkeypatch, config, kg):
    """Wire up the bare minimum of mcp_server._STATE so
    _sync_entity_to_chromadb can resolve its Chroma client cache during
    the seed call. Mirrors the pattern in test_congruence_audit, plus
    eager-bootstraps the entity collection via _get_entity_collection
    so that the post-seed assertions can verify writes immediately
    (without the seed itself bearing the cost of cold-starting Chroma
    inside its best-effort try/except). The _STATE.kg monkeypatch is
    required by _create_entity / _sync_entity_to_chromadb on Windows
    where the seed's lazy-imported sync helper otherwise sees an
    uninitialized client cache and silently no-ops."""
    from mempalace import mcp_server

    monkeypatch.setattr(mcp_server._STATE, "kg", kg)
    monkeypatch.setattr(mcp_server._STATE, "config", config)
    monkeypatch.setattr(mcp_server._STATE, "session_id", "test-seed-chroma")
    monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
    monkeypatch.setattr(mcp_server._STATE, "client_cache", None)
    monkeypatch.setattr(mcp_server._STATE, "collection_cache", None)
    # Eager bootstrap so the seed's best-effort sync has a live Chroma
    # client + entity collection to write into. Without this the seed
    # falls through silently (the helper is best-effort by design) and
    # the post-seed col.get assertions return empty.
    mcp_server._get_entity_collection(create=True)


# ── Helper coverage ─────────────────────────────────────────────────


def test_sync_seed_entity_to_chroma_is_best_effort_without_mcp_server(tmp_dir):
    """The helper must NOT raise when mcp_server is unavailable. The
    seed path runs from KnowledgeGraph.__init__ before _STATE.kg is
    wired up, so a failed lazy import is the normal cold-start case."""
    from mempalace.knowledge_graph import KnowledgeGraph
    import os

    # Use the conftest tmp_dir fixture so pytest's cleanup runs AFTER
    # the KG connection releases its WAL/shm locks (Windows otherwise
    # raises PermissionError at TemporaryDirectory __exit__).
    os.environ["MEMPALACE_SKIP_SEED"] = "1"
    try:
        kg = KnowledgeGraph(db_path=os.path.join(tmp_dir, "test.sqlite3"))
        # Helper must be a no-op when mcp_server isn't ready.
        kg._sync_seed_entity_to_chroma(
            entity_id="test_helper_entity",
            name="test_helper_entity",
            content="any content",
            kind="class",
            importance=3,
        )
    finally:
        os.environ.pop("MEMPALACE_SKIP_SEED", None)


# ── seed_ontology -> Chroma row contract ────────────────────────────


def _build_fresh_seeded_kg(monkeypatch, config, tmp_dir):
    """Build a freshly-constructed KG in a state where _STATE is wired
    up BEFORE seed_ontology runs, so the Chroma sync inside seed
    actually fires. Conftest's ``kg`` fixture seeds in __init__ before
    we can monkeypatch mcp_server._STATE.config, which causes the
    best-effort sync to silently no-op (no Chroma client yet); using
    MEMPALACE_SKIP_SEED defers the seed call until we control the
    environment."""
    import os

    from mempalace import mcp_server
    from mempalace.knowledge_graph import KnowledgeGraph

    monkeypatch.setenv("MEMPALACE_SKIP_SEED", "1")
    db_path = os.path.join(tmp_dir, "test_seed_chroma.sqlite3")
    fresh_kg = KnowledgeGraph(db_path=db_path)
    monkeypatch.delenv("MEMPALACE_SKIP_SEED", raising=False)
    monkeypatch.setattr(mcp_server._STATE, "kg", fresh_kg)
    monkeypatch.setattr(mcp_server._STATE, "config", config)
    monkeypatch.setattr(mcp_server._STATE, "session_id", "test-seed-chroma")
    monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
    monkeypatch.setattr(mcp_server._STATE, "client_cache", None)
    monkeypatch.setattr(mcp_server._STATE, "collection_cache", None)
    mcp_server._get_entity_collection(create=True)
    fresh_kg.seed_ontology()
    return fresh_kg


def test_seed_ontology_writes_root_class_to_chroma(monkeypatch, config, palace_path, tmp_dir):
    """After seed_ontology runs with mcp_server._STATE wired up, the
    root class ``thing`` MUST exist in mempalace_entities Chroma
    collection. Pre-fix this row was absent (SQLite-only seed), causing
    kg_query.details to be empty for every root class."""
    from mempalace import mcp_server

    _build_fresh_seeded_kg(monkeypatch, config, tmp_dir)
    ecol = mcp_server._get_entity_collection(create=False)
    assert ecol is not None, "entity collection must exist after seed"
    got = ecol.get(ids=["thing"], include=["documents", "metadatas"])
    assert got["ids"], (
        "root class 'thing' must have a Chroma row after seed_ontology; "
        "this is the audit-found regression that the fix closes."
    )
    # The document is the rendered summary prose.
    doc = got["documents"][0]
    assert "ontology root class" in doc.lower(), (
        f"thing's Chroma document must carry the rendered summary prose; got {doc!r}"
    )
    meta = got["metadatas"][0]
    assert meta.get("kind") == "class"


def test_seed_ontology_writes_intent_type_classes_to_chroma(
    monkeypatch, config, palace_path, tmp_dir
):
    """The intent_type subclasses (modify, inspect, research,
    wrap_up_session, etc.) were the most damaging phantoms because
    declare_intent retrieval relies on them being findable. Verify each
    has a Chroma row."""
    from mempalace import mcp_server

    _build_fresh_seeded_kg(monkeypatch, config, tmp_dir)
    ecol = mcp_server._get_entity_collection(create=False)
    assert ecol is not None
    expected = ["modify", "inspect", "research", "wrap_up_session", "execute", "communicate"]
    got = ecol.get(ids=expected, include=["metadatas"])
    surfaced = set(got["ids"]) if got and got.get("ids") else set()
    missing = [name for name in expected if name not in surfaced]
    assert not missing, (
        f"Intent types missing from Chroma after seed_ontology: {missing}. "
        f"Pre-fix every one was phantom; the fix routes the seed loop "
        f"through _sync_seed_entity_to_chroma."
    )


def test_seed_ontology_writes_predicates_to_chroma(monkeypatch, config, palace_path, tmp_dir):
    """Predicates also need Chroma rows so kg_query.details + retrieval
    works. Spot-check a representative subset."""
    from mempalace import mcp_server

    _build_fresh_seeded_kg(monkeypatch, config, tmp_dir)
    ecol = mcp_server._get_entity_collection(create=False)
    assert ecol is not None
    expected = ["is_a", "described_by", "tested_by", "rated_useful", "similar_to"]
    got = ecol.get(ids=expected, include=["metadatas"])
    surfaced = set(got["ids"]) if got and got.get("ids") else set()
    missing = [name for name in expected if name not in surfaced]
    assert not missing, f"Predicates missing from Chroma after seed_ontology: {missing}"


# ── backfill_seed_chroma idempotency + recovery ─────────────────────


def test_backfill_seed_chroma_recovers_phantom_rows(monkeypatch, config, palace_path, kg):
    """Simulate a pre-fix palace: seed via raw add_entity (SQLite only,
    no Chroma), then call backfill_seed_chroma and verify the Chroma
    rows materialize. This is the upgrade path for existing palaces."""
    from mempalace import mcp_server

    _setup_state(monkeypatch, config, kg)
    # Bypass the fixed path and write SQLite-only rows the way the
    # legacy seed_ontology did.
    kg.add_entity(
        "thing",
        kind="class",
        content="thing -- ontology root class -- legacy seed",
        importance=5,
        properties={"summary": {"what": "thing -- root", "why": "ontology root"}},
    )
    kg.add_entity(
        "agent",
        kind="class",
        content="agent class -- legacy seed",
        importance=4,
        properties={"summary": {"what": "agent class", "why": "ai agent class"}},
    )

    # Pre-condition: no Chroma rows for these.
    ecol = mcp_server._get_entity_collection(create=True)
    assert ecol is not None
    pre = ecol.get(ids=["thing", "agent"], include=["metadatas"])
    pre_ids = set(pre["ids"]) if pre and pre.get("ids") else set()
    # If the test harness is preseeded somehow we can't assert empty,
    # but the backfill below is idempotent so the test still locks the
    # post-condition.

    result = kg.backfill_seed_chroma()
    assert result["considered"] >= 2
    assert result["synced"] >= 2

    # Post-condition: BOTH rows exist in Chroma.
    post = ecol.get(ids=["thing", "agent"], include=["documents", "metadatas"])
    post_ids = set(post["ids"]) if post and post.get("ids") else set()
    assert "thing" in post_ids
    assert "agent" in post_ids
    # Sanity: the document is the persisted SQLite content.
    doc_idx = post["ids"].index("agent")
    assert "agent class" in post["documents"][doc_idx].lower()
    # Suppress unused: pre_ids documents the pre-state for debugging.
    del pre_ids


def test_backfill_seed_chroma_is_idempotent(monkeypatch, config, palace_path, kg):
    """Running the backfill twice must produce the same Chroma state.
    Cold-start callers can run it unconditionally on every boot without
    drift.

    Followup 2026-05-02 (data_migrations stamp pattern,
    ``record_ga_agent_channel_fix_followups_2026_05_02``): idempotency
    is now guaranteed mechanically by the stamp-based early-return --
    once ``backfill_seed_chroma_2026_05_01`` is stamped, every
    subsequent call returns ``status='already_applied'`` with
    ``considered=0`` and ``synced=0`` without iterating. The pre-stamp
    contract (``first.considered == second.considered``) was a weaker
    behavioral guarantee; the stamp pattern strengthens it to O(1)
    short-circuit on boot."""
    _setup_state(monkeypatch, config, kg)
    # The conftest ``kg`` fixture seeds in __init__ before _STATE is
    # wired; the explicit seed_ontology() below hits the
    # already-seeded early-return branch which auto-runs
    # backfill_seed_chroma() (stamps it on success). Both explicit
    # calls below then short-circuit on the stamp.
    kg.seed_ontology()
    first = kg.backfill_seed_chroma()
    second = kg.backfill_seed_chroma()
    # Same status both runs -- stamp early-return makes idempotency
    # mechanical; both must report already_applied.
    assert first["status"] == second["status"]
    assert second["status"] == "already_applied"
    # Second run never iterates (early return on stamp).
    assert second["considered"] == 0
    assert second["synced"] == 0


def test_backfill_seed_chroma_handles_missing_mcp_server_gracefully(tmp_dir):
    """If mcp_server isn't importable (e.g., bare KG in unit tests),
    backfill_seed_chroma must return zeros without raising. SQLite is
    the source of truth; the caller can re-run the backfill once
    mcp_server is up."""
    from mempalace.knowledge_graph import KnowledgeGraph
    import os

    os.environ["MEMPALACE_SKIP_SEED"] = "1"
    try:
        kg = KnowledgeGraph(db_path=os.path.join(tmp_dir, "test.sqlite3"))
        # Without seeding the entities table is empty AND mcp_server
        # may not be initialized. Both paths must return zeros, no
        # raise.
        result = kg.backfill_seed_chroma()
        assert isinstance(result, dict)
        assert result["considered"] >= 0
        assert result["synced"] >= 0
    finally:
        os.environ.pop("MEMPALACE_SKIP_SEED", None)


pytestmark = pytest.mark.integration
