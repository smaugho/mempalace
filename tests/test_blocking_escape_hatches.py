"""
test_blocking_escape_hatches.py — Invariants for every mempalace blocking
mechanism. These tests exist because the "enrichment loop" bug of 2026-04-19
made declare_intent permanently unblockable: rejected enrichments kept
re-surfacing on every session-id switch because their persistence path was
inconsistent (finalize_intent populated in-memory pending state AFTER the
state file was unlinked, and _save_session_state cached them into
session_state which _restore_session_state then resurrected).

Contract these tests lock in:

  1. Any state that can block a public tool has a code path that clears it.
  2. Clearing it is durable across session-id switches and MCP restarts.
  3. Orphan state (pending references to deleted entities) is auto-pruned.
  4. Persistence is symmetric: persist then load must round-trip.

If any future change breaks one of these, the bug will be caught here
rather than in a live session where the only recourse is rm -rf the
state file.
"""

from __future__ import annotations

import json
from pathlib import Path

from mempalace import intent
from mempalace import mcp_server
from mempalace.knowledge_graph import KnowledgeGraph


def _make_state(tmp_path: Path, sid: str = "test-sid-a"):
    """Spin up a disposable ServerState pointing at tmp_path for disk I/O."""
    import mempalace.mcp_server as m

    # Disposable KG so entity-existence checks are real, not mocked.
    kg_path = tmp_path / "kg.sqlite3"
    kg = KnowledgeGraph(str(kg_path))
    m._STATE.kg = kg
    m._STATE.active_intent = None
    m._STATE.pending_conflicts = None
    m._STATE.pending_enrichments = None
    m._STATE.session_state = {}
    m._STATE.session_id = sid
    m._STATE.declared_entities = set()
    m._INTENT_STATE_DIR = tmp_path
    return m._STATE


def _state_file(tmp_path: Path, sid: str) -> Path:
    return tmp_path / f"active_intent_{sid}.json"


# ==================== Persistence contract ====================


class TestPersistContract:
    """_persist_active_intent must be SYMMETRIC and COMPLETE.

    - active_intent present: write the full state block.
    - no active_intent but pending state: write a placeholder so pending
      state survives past finalize_intent.
    - no active_intent and no pending: unlink the file.
    """

    def test_persist_intent_only_writes_full_block(self, tmp_path):
        state = _make_state(tmp_path)
        state.active_intent = {
            "intent_id": "intent_test_1",
            "intent_type": "modify",
            "slots": {},
            "effective_permissions": [],
        }
        state.pending_conflicts = None
        state.pending_enrichments = None

        intent._persist_active_intent()
        p = _state_file(tmp_path, state.session_id)
        assert p.is_file(), "file must exist after persist with active intent"
        data = json.loads(p.read_text())
        assert data["intent_id"] == "intent_test_1"
        assert data["pending_conflicts"] == []
        assert data["pending_enrichments"] == []

    def test_persist_no_intent_but_pending_enrichments_writes_placeholder(self, tmp_path):
        """THE bug that trapped Adrian. Pending state set AFTER active_intent
        becomes None (this is what finalize_intent does) used to disappear
        on the second _persist call because it hit the unlink branch."""
        state = _make_state(tmp_path)
        state.active_intent = None
        state.pending_enrichments = [
            {"id": "e1", "from_entity": "A", "to_entity": "B", "reason": "r"}
        ]

        intent._persist_active_intent()
        p = _state_file(tmp_path, state.session_id)
        assert p.is_file(), "file must persist pending state even without active intent"
        data = json.loads(p.read_text())
        assert data.get("intent_id") == "", "placeholder file has no real intent"
        assert data["pending_enrichments"][0]["id"] == "e1"

    def test_persist_no_intent_but_pending_conflicts_writes_placeholder(self, tmp_path):
        state = _make_state(tmp_path)
        state.active_intent = None
        state.pending_conflicts = [{"id": "c1", "conflict_type": "edge_contradiction"}]
        state.pending_enrichments = None

        intent._persist_active_intent()
        p = _state_file(tmp_path, state.session_id)
        assert p.is_file()
        data = json.loads(p.read_text())
        assert data["pending_conflicts"][0]["id"] == "c1"
        assert data["pending_enrichments"] == []

    def test_persist_no_intent_no_pending_unlinks_file(self, tmp_path):
        state = _make_state(tmp_path)
        # Seed a stale file to verify it is removed.
        _state_file(tmp_path, state.session_id).write_text('{"intent_id": "stale"}')

        state.active_intent = None
        state.pending_conflicts = None
        state.pending_enrichments = None

        intent._persist_active_intent()
        assert not _state_file(tmp_path, state.session_id).exists()

    def test_persist_round_trip_preserves_pending(self, tmp_path):
        state = _make_state(tmp_path)
        state.active_intent = None
        state.pending_enrichments = [
            {"id": "e1", "from_entity": "A", "to_entity": "B"},
            {"id": "e2", "from_entity": "C", "to_entity": "D"},
        ]
        intent._persist_active_intent()

        loaded = mcp_server._load_pending_enrichments_from_disk(state.session_id)
        assert len(loaded) == 2
        assert loaded[0]["id"] == "e1"


# ==================== Session-cache contract ====================


class TestSessionCacheContract:
    """_save_session_state + _restore_session_state MUST NOT cache pending
    state. Pending state lives on disk only. This prevents the exact bug
    where resolve_enrichments cleared in-memory AND disk, but the next
    session-id switch resurrected the old pending from session_state[sid].
    """

    def test_save_does_not_cache_pending_conflicts(self, tmp_path):
        state = _make_state(tmp_path)
        state.pending_conflicts = [{"id": "c1"}]
        state.pending_enrichments = [{"id": "e1"}]

        mcp_server._save_session_state()

        cached = state.session_state[state.session_id]
        assert "pending_conflicts" not in cached, (
            "pending_conflicts must NOT be cached in session_state; disk is authoritative"
        )
        assert "pending_enrichments" not in cached

    def test_restore_after_clear_does_not_resurrect(self, tmp_path):
        """Exact regression: enrichment cleared via resolve, then session
        switch fires _restore, old enrichment came back from session_state."""
        state = _make_state(tmp_path)
        # Populate + save a snapshot with pending set
        state.pending_enrichments = [{"id": "e1", "from_entity": "A", "to_entity": "B"}]
        mcp_server._save_session_state()

        # Caller "resolved" the enrichment — clear in-memory AND disk
        state.pending_enrichments = None
        intent._persist_active_intent()

        # Now a session switch happens: restore should NOT resurrect.
        mcp_server._restore_session_state(state.session_id)

        assert state.pending_enrichments is None, (
            "restore resurrected cleared pending from stale cache"
        )

    def test_restore_reads_pending_from_disk(self, tmp_path):
        """If the disk file has pending state (legitimate post-finalize case),
        restore MUST pick it up."""
        state = _make_state(tmp_path)
        state.pending_enrichments = [{"id": "e1", "from_entity": "A", "to_entity": "B"}]
        intent._persist_active_intent()
        state.pending_enrichments = None  # simulate fresh process

        mcp_server._restore_session_state(state.session_id)
        assert state.pending_enrichments
        assert state.pending_enrichments[0]["id"] == "e1"


# ==================== Orphan auto-prune ====================


class TestOrphanPrune:
    """An enrichment whose from_entity or to_entity no longer exists in the
    KG has no valid structural resolution. declare_intent must auto-prune
    rather than block the agent forever. This was the Adrian 2026-04-19
    blocker: a record he had deleted kept appearing as enrichment source.

    NOTE: these tests exercise the prune *primitive* in isolation (the
    KG-lookup + list-rebuild pattern). The REAL tool_declare_intent
    integration is covered in tests/test_blocking_round_trips.py
    ::TestOrphanPruneReal. Keeping both so a primitive regression is
    localisable without having to spin up the full mcp_env.
    """

    def test_declare_intent_prunes_orphan_enrichment_with_missing_from_entity(self, tmp_path):
        state = _make_state(tmp_path)
        # Seed one real entity so the to_entity exists, but from_entity is orphan.
        state.kg.add_entity("real_target", kind="entity", description="real", importance=3)

        state.pending_enrichments = [
            {
                "id": "e_orphan",
                "from_entity": "ghost_that_was_deleted",
                "to_entity": "real_target",
                "reason": "stale proposal",
            }
        ]
        intent._persist_active_intent()

        # Import the declare_intent pending check inline (full declare_intent
        # requires a lot of setup; we drive just the orphan-prune path).
        # The production path is: declare_intent reads _STATE.pending_enrichments,
        # prunes orphans, re-persists, and only blocks if any remain.
        # Re-implementing the prune here mirrors the production code exactly.
        pending = state.pending_enrichments
        pruned = []
        for enr in pending:
            fe = enr.get("from_entity") or ""
            te = enr.get("to_entity") or ""
            if state.kg.get_entity(fe) and state.kg.get_entity(te):
                pruned.append(enr)
        state.pending_enrichments = pruned or None

        assert state.pending_enrichments is None, (
            "orphan enrichment should be auto-pruned, not left to block"
        )

    def test_declare_intent_prunes_orphan_enrichment_with_missing_to_entity(self, tmp_path):
        state = _make_state(tmp_path)
        state.kg.add_entity("real_source", kind="entity", description="real", importance=3)
        state.pending_enrichments = [
            {
                "id": "e_orphan",
                "from_entity": "real_source",
                "to_entity": "ghost_gone",
                "reason": "stale",
            }
        ]
        # Mirror of the production prune loop.
        pruned = [
            e
            for e in state.pending_enrichments
            if state.kg.get_entity(e["from_entity"]) and state.kg.get_entity(e["to_entity"])
        ]
        state.pending_enrichments = pruned or None
        assert state.pending_enrichments is None

    def test_prune_keeps_valid_enrichments(self, tmp_path):
        state = _make_state(tmp_path)
        state.kg.add_entity("a", kind="entity", description="a", importance=3)
        state.kg.add_entity("b", kind="entity", description="b", importance=3)
        state.kg.add_entity("c", kind="entity", description="c", importance=3)

        state.pending_enrichments = [
            {"id": "valid1", "from_entity": "a", "to_entity": "b"},
            {"id": "orphan", "from_entity": "a", "to_entity": "nonexistent"},
            {"id": "valid2", "from_entity": "b", "to_entity": "c"},
        ]
        pruned = [
            e
            for e in state.pending_enrichments
            if state.kg.get_entity(e["from_entity"]) and state.kg.get_entity(e["to_entity"])
        ]
        assert {e["id"] for e in pruned} == {"valid1", "valid2"}


# ==================== resolve flow durability ====================


class TestResolveFlow:
    def test_resolve_enrichments_clears_disk_file(self, tmp_path):
        """After resolve_enrichments clears in-memory pending, the disk
        file must ALSO be cleared — otherwise a restart would re-block."""
        state = _make_state(tmp_path)
        state.pending_enrichments = [{"id": "e1", "from_entity": "a", "to_entity": "b"}]
        intent._persist_active_intent()
        assert _state_file(tmp_path, state.session_id).is_file()

        # Simulate resolve_enrichments's clear + persist.
        state.pending_enrichments = None
        intent._persist_active_intent()

        # Disk file should be gone (no intent + no pending).
        assert not _state_file(tmp_path, state.session_id).exists()

    def test_resolve_conflicts_clears_disk_file(self, tmp_path):
        state = _make_state(tmp_path)
        state.pending_conflicts = [{"id": "c1"}]
        intent._persist_active_intent()
        assert _state_file(tmp_path, state.session_id).is_file()

        state.pending_conflicts = None
        intent._persist_active_intent()
        assert not _state_file(tmp_path, state.session_id).exists()

    def test_partial_resolve_keeps_remaining_pending_on_disk(self, tmp_path):
        """Resolving SOME enrichments but not all must leave the remaining
        ones on disk, not silently drop them."""
        state = _make_state(tmp_path)
        state.pending_enrichments = [
            {"id": "e1", "from_entity": "a", "to_entity": "b"},
            {"id": "e2", "from_entity": "c", "to_entity": "d"},
        ]
        intent._persist_active_intent()

        # Partial resolve: keep only e2.
        state.pending_enrichments = [state.pending_enrichments[1]]
        intent._persist_active_intent()

        loaded = mcp_server._load_pending_enrichments_from_disk(state.session_id)
        assert len(loaded) == 1
        assert loaded[0]["id"] == "e2"


# ==================== Cross-session isolation ====================


class TestCrossSession:
    def test_pending_scoped_to_session_id(self, tmp_path):
        """Two sessions A and B must not see each other's pending state."""
        state = _make_state(tmp_path, sid="session-a")
        state.pending_enrichments = [{"id": "e_a", "from_entity": "x", "to_entity": "y"}]
        intent._persist_active_intent()

        # Switch to session B — fresh in-memory, disk scoped by sid.
        state.session_id = "session-b"
        state.pending_enrichments = None
        mcp_server._restore_session_state("session-b")

        assert state.pending_enrichments is None, (
            "session B must not see session A's pending enrichments"
        )

        # Switch back to A — pending must still be there.
        state.session_id = "session-a"
        mcp_server._restore_session_state("session-a")
        assert state.pending_enrichments
        assert state.pending_enrichments[0]["id"] == "e_a"


# ==================== Server-restart resilience ====================


def _simulate_server_restart(tmp_path: Path, sid: str):
    """Force the ServerState back to its fresh-construction defaults.

    A server restart in production is: new Python process, new import of
    mempalace.mcp_server, new ServerState dataclass, fresh KG handle.
    The KG and Chroma live on disk, so they reconnect automatically; the
    MCP hook-state directory stays on disk too. The ONLY thing that
    disappears is the in-memory `_STATE` dict.

    This helper reproduces that by clearing every in-memory field AND
    re-reading the KG handle from the same sqlite file. The disk state
    (palace/knowledge_graph.sqlite3 + hook_state/active_intent_*.json)
    survives unchanged, just like a real restart.
    """
    kg_path = tmp_path / "kg.sqlite3"
    fresh_kg = KnowledgeGraph(str(kg_path))
    mcp_server._STATE.kg = fresh_kg
    mcp_server._STATE.active_intent = None
    mcp_server._STATE.pending_conflicts = None
    mcp_server._STATE.pending_enrichments = None
    mcp_server._STATE.session_state = {}
    mcp_server._STATE.declared_entities = set()
    mcp_server._STATE.session_id = sid
    mcp_server._INTENT_STATE_DIR = tmp_path
    return mcp_server._STATE


class TestServerRestartResilience:
    """Exhaustive server-restart scenarios. The user's rule: after a restart
    with cleared memory, NO blocking condition must survive that cannot be
    unblocked by the normal MCP API. If a disk state file makes the server
    unusable, that's a bug.
    """

    def test_restart_with_clean_disk_has_no_pending(self, tmp_path):
        _make_state(tmp_path, sid="s1")
        # No file written.
        state = _simulate_server_restart(tmp_path, sid="s1")
        mcp_server._restore_session_state("s1")
        assert state.pending_conflicts is None
        assert state.pending_enrichments is None
        assert state.active_intent is None

    def test_restart_with_stale_pending_enrichments_loads_them(self, tmp_path):
        """Disk had legitimately-persisted pending enrichments (post-finalize).
        A restart MUST load them so resolve_enrichments can see them."""
        state = _make_state(tmp_path, sid="s2")
        state.pending_enrichments = [{"id": "e1", "from_entity": "alpha", "to_entity": "beta"}]
        state.kg.add_entity("alpha", kind="entity", description="a", importance=3)
        state.kg.add_entity("beta", kind="entity", description="b", importance=3)
        intent._persist_active_intent()

        state = _simulate_server_restart(tmp_path, sid="s2")
        # Re-seed the entities since fresh KG was re-opened from disk; they
        # persist in the sqlite file so should still exist.
        assert state.kg.get_entity("alpha"), "KG must survive restart"
        assert state.kg.get_entity("beta")

        mcp_server._restore_session_state("s2")
        assert state.pending_enrichments
        assert state.pending_enrichments[0]["id"] == "e1"

    def test_restart_with_orphan_enrichments_auto_prunes_on_declare(self, tmp_path):
        """Disk has a pending enrichment pointing at an entity that no longer
        exists. After restart, the declare_intent orphan-prune must drop it
        so the agent is not trapped."""
        state = _make_state(tmp_path, sid="s3")
        # Seed only ONE entity so from_entity 'ghost' is orphan.
        state.kg.add_entity("anchor", kind="entity", description="a", importance=3)
        state.pending_enrichments = [
            {
                "id": "e_ghost",
                "from_entity": "ghost_was_deleted",
                "to_entity": "anchor",
                "reason": "stale",
            }
        ]
        intent._persist_active_intent()

        state = _simulate_server_restart(tmp_path, sid="s3")
        mcp_server._restore_session_state("s3")
        assert state.pending_enrichments, "disk load must still produce the enrichment"
        # Now the orphan-prune kicks in (production path in declare_intent).
        pruned = []
        for enr in state.pending_enrichments:
            fe_ok = bool(state.kg.get_entity(enr.get("from_entity") or ""))
            te_ok = bool(state.kg.get_entity(enr.get("to_entity") or ""))
            if fe_ok and te_ok:
                pruned.append(enr)
        state.pending_enrichments = pruned or None
        intent._persist_active_intent()

        assert state.pending_enrichments is None, (
            "orphan-prune must drop the ghost-pointing enrichment"
        )
        assert not _state_file(tmp_path, "s3").exists(), (
            "after orphan-prune leaves no pending state, disk file must be unlinked"
        )

    def test_restart_after_resolve_leaves_no_block(self, tmp_path):
        """Agent resolved enrichments; server restarts; must find NO pending."""
        state = _make_state(tmp_path, sid="s4")
        state.pending_enrichments = [{"id": "e1", "from_entity": "x", "to_entity": "y"}]
        intent._persist_active_intent()

        # resolve clears
        state.pending_enrichments = None
        intent._persist_active_intent()

        # Restart.
        state = _simulate_server_restart(tmp_path, sid="s4")
        mcp_server._restore_session_state("s4")
        assert state.pending_enrichments is None
        assert state.pending_conflicts is None
        assert state.active_intent is None

    def test_restart_with_corrupted_state_file_does_not_crash(self, tmp_path):
        """Defensive: a corrupted state file (invalid JSON) must NOT prevent
        the server from starting — it should load as if the file were empty."""
        _make_state(tmp_path, sid="s5")
        _state_file(tmp_path, "s5").write_text("{ this is not valid json")

        state = _simulate_server_restart(tmp_path, sid="s5")
        # _load_pending_*_from_disk swallows JSON errors and returns [].
        assert mcp_server._load_pending_conflicts_from_disk("s5") == []
        assert mcp_server._load_pending_enrichments_from_disk("s5") == []
        mcp_server._restore_session_state("s5")
        assert state.pending_enrichments is None

    def test_restart_with_pending_conflicts_only_loads_them(self, tmp_path):
        """Mirror of the enrichment-restart test, for conflicts."""
        state = _make_state(tmp_path, sid="s6")
        state.pending_conflicts = [{"id": "c1", "conflict_type": "edge_contradiction"}]
        intent._persist_active_intent()

        state = _simulate_server_restart(tmp_path, sid="s6")
        mcp_server._restore_session_state("s6")
        assert state.pending_conflicts
        assert state.pending_conflicts[0]["id"] == "c1"

    def test_multiple_session_ids_all_isolate_across_restart(self, tmp_path):
        """Each session's disk file is independent. After restart, switching
        between sessions must see each its own pending state, not the
        other's."""
        # Session A with one enrichment
        state = _make_state(tmp_path, sid="sess-a")
        state.pending_enrichments = [{"id": "e_a", "from_entity": "x", "to_entity": "y"}]
        intent._persist_active_intent()

        # Session B with a different enrichment
        state.session_id = "sess-b"
        state.pending_enrichments = [{"id": "e_b", "from_entity": "m", "to_entity": "n"}]
        intent._persist_active_intent()

        # Restart. Fresh memory, both disk files survive.
        _simulate_server_restart(tmp_path, sid="sess-a")
        mcp_server._restore_session_state("sess-a")
        assert mcp_server._STATE.pending_enrichments[0]["id"] == "e_a"

        mcp_server._STATE.session_id = "sess-b"
        mcp_server._restore_session_state("sess-b")
        assert mcp_server._STATE.pending_enrichments[0]["id"] == "e_b"


# ==================== Hook tilde expansion ====================


class TestHookTildeExpansion:
    """hooks_cli._normalize_win_path must expand ``~`` so scope patterns
    declared as ``~/.mempalace/**`` match absolute paths the tool receives.
    Without this, declaring a reasonable scope for user-home files silently
    rejected every Read/Grep against the actual file path.
    """

    def test_tilde_expands_to_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path).replace("\\", "/"))
        monkeypatch.setenv("USERPROFILE", str(tmp_path).replace("\\", "/"))
        from mempalace.hooks_cli import _normalize_win_path

        out = _normalize_win_path("~/.mempalace/hook_state/hook.log")
        assert "~" not in out
        # HOME path should be inside the normalized result
        assert str(tmp_path).replace("\\", "/").lower() in out.lower()

    def test_absolute_path_still_normalizes(self):
        from mempalace.hooks_cli import _normalize_win_path

        assert _normalize_win_path("D:\\Flowsev\\test.py") == "d:/flowsev/test.py"
        assert _normalize_win_path("/d/flowsev/test.py") == "d:/flowsev/test.py"

    def test_empty_input_returns_empty(self):
        from mempalace.hooks_cli import _normalize_win_path

        assert _normalize_win_path("") == ""
        assert _normalize_win_path(None) == ""  # defensive
