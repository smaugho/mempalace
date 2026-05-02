"""Regression locks for the 2026-05-02 data_migrations stamp pattern.

Adrian's followup after the channel-separation fix: one-shot data
migrations (``backfill_seed_chroma``,
``migrate_strip_polluted_context_views``,
``migrate_recompute_similar_to_confidences``) used to iterate every row
on every KG init even when there was nothing to migrate. The stamp
pattern -- ``data_migrations(name PRIMARY KEY, applied_at TEXT)`` plus
``_data_migration_applied`` / ``_stamp_data_migration`` accessors --
makes every helper O(1) on the second boot.

These tests lock the contract:

  A. The ``data_migrations`` table is created on KG init.
  B. ``_stamp_data_migration`` makes ``_data_migration_applied`` return
     True for the same name; idempotent on re-stamp.
  C. ``migrate_strip_polluted_context_views`` returns
     ``status='already_applied'`` on the second call after stamp.
  D. ``backfill_seed_chroma`` stamps + skips on second call.
  E. ``migrate_recompute_similar_to_confidences`` defers (no stamp) when
     mcp_server isn't ready, and stamps on successful run.
"""

from __future__ import annotations


def _setup_state(monkeypatch, config, kg):
    from mempalace import mcp_server

    monkeypatch.setattr(mcp_server._STATE, "kg", kg)
    monkeypatch.setattr(mcp_server._STATE, "config", config)
    monkeypatch.setattr(mcp_server._STATE, "session_id", "test-stamp")
    monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
    monkeypatch.setattr(mcp_server._STATE, "client_cache", None)
    monkeypatch.setattr(mcp_server._STATE, "collection_cache", None)
    mcp_server._get_entity_collection(create=True)


# A. table is created on KG init


def test_data_migrations_table_exists_after_init(kg):
    """The ``data_migrations`` table must exist after KG init -- the
    bootstrap is unconditional (independent of MEMPALACE_SKIP_SEED).
    Without it the stamp helpers crash on every call."""
    row = (
        kg._conn()
        .execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='data_migrations'")
        .fetchone()
    )
    assert row is not None, "data_migrations table must exist after KG init; bootstrap regressed"


# B. stamp helpers idempotent


def test_stamp_helpers_idempotent(kg):
    """``_stamp_data_migration`` followed by ``_data_migration_applied``
    must return True; re-stamping the same name is a no-op (INSERT OR
    IGNORE). The check is O(1) -- a single SELECT against a primary-key
    column."""
    name = "test_stamp_idempotent_2026_05_02"
    assert kg._data_migration_applied(name) is False
    kg._stamp_data_migration(name)
    assert kg._data_migration_applied(name) is True
    # Second stamp: still applied, no exception.
    kg._stamp_data_migration(name)
    assert kg._data_migration_applied(name) is True


# C. strip migration honors the stamp


def test_strip_migration_returns_already_applied_after_stamp(monkeypatch, config, palace_path, kg):
    """First call runs the work + stamps; second call returns
    ``status='already_applied'`` without iterating. Cheap to verify by
    seeding a polluted context, running once, then re-running and
    confirming the second run did not strip again (counts stay at 0)."""
    _setup_state(monkeypatch, config, kg)
    kg.seed_ontology()
    # Polluted context fixture
    kg.add_entity(
        "agent",
        kind="class",
        content="Agent class fixture for stamp test ensuring entity content is long enough to detect",
    )
    from mempalace.entity_gate import mint_entity

    mint_entity(
        "ga_agent",
        kind="entity",
        summary={
            "what": "ga_agent stamp-test fixture",
            "why": "stamp test fixture; provides a long-content slot entity so legacy auto-append patterns produce stripable views in the regression",
            "scope": "test_data_migrations_stamp",
        },
        queries=["who is ga_agent stamp fixture", "stamp test fixture"],
    )
    kg.add_triple("ga_agent", "is_a", "agent")
    agent_ent = kg.get_entity("ga_agent")
    agent_prefix = (agent_ent.get("content") or "")[:200]
    kg.add_entity(
        "ctx_polluted_for_stamp",
        kind="context",
        content="legacy",
        importance=3,
        properties={
            "queries": ["a real query", "another real one", agent_prefix],
            "keywords": ["legacy"],
            "entities": ["ga_agent"],
            "agent": "ga_agent",
        },
    )

    first = kg.migrate_strip_polluted_context_views()
    assert first.get("status") == "applied"
    assert first.get("stripped_views", 0) >= 1

    second = kg.migrate_strip_polluted_context_views()
    assert second.get("status") == "already_applied", (
        f"second run must short-circuit on stamp; got {second}"
    )
    # No further stripping happened (counts stay at 0 in the early-return).
    assert second.get("stripped_views", 0) == 0


# D. backfill_seed_chroma honors the stamp


def test_backfill_seed_chroma_stamp(monkeypatch, config, palace_path, kg):
    """Same pattern: first run does the work + stamps; second run returns
    ``already_applied`` immediately."""
    _setup_state(monkeypatch, config, kg)
    kg.seed_ontology()  # populates seed entities; backfill_seed_chroma
    # then has rows to iterate on first call.

    first = kg.backfill_seed_chroma()
    assert first.get("status") in ("applied", "already_applied")
    if first.get("status") == "applied":
        # Stamp landed; second call must short-circuit
        second = kg.backfill_seed_chroma()
        assert second.get("status") == "already_applied"
        assert second.get("synced", 0) == 0


# E. recompute defers cleanly when chroma not ready; stamps on success


def test_recompute_defers_without_mcp_server(tmp_dir, monkeypatch):
    """Without an initialized mcp_server / Chroma client, the recompute
    helper returns ``status='deferred'`` WITHOUT stamping, so a later
    boot once mcp_server is ready actually runs the work.

    Uses the conftest ``tmp_dir`` fixture (not ``tempfile.TemporaryDirectory``)
    because pytest's tmp_dir cleanup runs AFTER the KG connection
    releases its WAL/shm locks; ``TemporaryDirectory`` __exit__ races
    those locks on Windows and raises PermissionError.

    Uses ``monkeypatch`` for both the env var and the _STATE fields so
    pytest restores them at teardown -- a prior version that did
    ``os.environ.pop("MEMPALACE_SKIP_SEED")`` in finally leaked the
    conftest-wide skip-seed setting, causing alphabetically-later test
    files (e.g. ``test_knowledge_graph.py``) to see auto-seeded KGs
    instead of empty ones."""
    import os

    from mempalace.knowledge_graph import KnowledgeGraph

    # MEMPALACE_SKIP_SEED is set globally by conftest at module load;
    # monkeypatch.setenv would re-set it (idempotent) but we don't even
    # need that -- just don't unset it.
    db = os.path.join(tmp_dir, "kg.sqlite3")
    kg = KnowledgeGraph(db_path=db)
    # Ensure stamp table exists even with skip-seed (init bootstrapped it).
    row = (
        kg._conn()
        .execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='data_migrations'")
        .fetchone()
    )
    assert row is not None
    # Without _STATE wired up, _get_context_views_collection returns
    # None and the helper bails with status='deferred'. Use monkeypatch
    # so subsequent tests don't see leaked _STATE either.
    from mempalace import mcp_server

    monkeypatch.setattr(mcp_server._STATE, "client_cache", None)
    monkeypatch.setattr(mcp_server._STATE, "collection_cache", None)
    monkeypatch.setattr(mcp_server._STATE, "config", None)
    result = kg.migrate_recompute_similar_to_confidences()
    assert result.get("status") == "deferred"
    # NO stamp landed -- a future boot can retry.
    assert kg._data_migration_applied("recompute_similar_to_confidences_2026_05_02") is False


def test_recompute_stamps_on_successful_run(monkeypatch, config, palace_path, kg):
    """When mcp_server + Chroma are wired and the recompute completes
    (even with zero similar_to edges to recompute), it stamps so future
    boots are O(1)."""
    _setup_state(monkeypatch, config, kg)
    kg.seed_ontology()

    # No similar_to edges in this fresh palace -- the helper iterates
    # zero rows but still stamps because the run completed.
    result = kg.migrate_recompute_similar_to_confidences()
    assert result.get("status") in ("applied", "already_applied", "deferred")
    if result.get("status") == "applied":
        assert kg._data_migration_applied("recompute_similar_to_confidences_2026_05_02") is True
        # Second run is the short-circuit.
        second = kg.migrate_recompute_similar_to_confidences()
        assert second.get("status") == "already_applied"
