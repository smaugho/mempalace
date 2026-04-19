"""
test_blocking_round_trips.py — End-to-end tool-call round-trips that lock
in the anti-deadlock contract at the **public API surface**.

The sibling test file ``test_blocking_escape_hatches.py`` pins the
primitives (``_persist_active_intent``, ``_load_pending_*_from_disk``,
``_restore_session_state``). Those tests manipulate ``_STATE`` directly
and never go through ``tool_resolve_enrichments`` / ``tool_declare_intent``.

Every deadlock the user hit in practice had the same shape: the
primitives behaved correctly in isolation, but the *round-trip through
the tool entry points* produced a different answer than a naive chain of
``persist → restore`` does. This file closes that gap.

Round-trips covered:
  1. resolve_enrichments(...) → declare_intent(...) — gate actually opens
  2. resolve_enrichments(unknown ids while pending exists) — must NOT silently succeed
  3. resolve_enrichments(empty actions while pending exists) — error, not fake success
  4. resolve_enrichments(any) on empty state — benign no-op, disk clean after
  5. Partial resolve → declare still blocks on the remaining → resolve remaining → declare ok
  6. Double-resolve idempotency — second call is a no-op, not a regression
  7. Orphan pending triggers auto-prune via REAL tool_declare_intent
  8. Session-switch round-trip with pending-only state
  9. kg_declare_entity → resolve all suggested enrichments → declare succeeds
 10. Corrupted state file → tool_resolve_enrichments doesn't crash
 11. Hyphen/underscore id-shape collision in pending enrichments
 12. Rapid seed→resolve→declare alternation (20 iterations, no state leak)
 13. Conflicts: resolve → declare round-trip
 14. resolve with disk-only pending (in-memory empty) — loads from disk and
     resolves, then the NEXT declare_intent is unblocked. This is the exact
     failure mode Adrian hit on 2026-04-19.

These tests call the ACTUAL ``tool_*`` functions so any regression in the
tool's response envelope, action-matching, or persist call ordering is
surfaced here rather than in a live session.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Default budget for tests — generous to avoid budget-related failures.
_TEST_BUDGET = {"Read": 20, "Edit": 20, "Bash": 20, "Grep": 20, "Glob": 20, "Write": 20}


def _seed_intent_types(kg, palace_path):
    """Minimal intent type hierarchy: research + agent + thing."""
    import chromadb

    client = chromadb.PersistentClient(path=palace_path)
    ecol = client.get_or_create_collection("mempalace_entities")

    kg.add_entity(
        "intent_type", kind="class", description="Root class for all intent types", importance=5
    )
    kg.add_entity("thing", kind="class", description="Root class for all entities", importance=5)
    kg.add_entity("agent", kind="class", description="An AI agent", importance=5)
    kg.add_entity("test_agent", kind="entity", description="Test agent", importance=3)
    kg.add_triple("test_agent", "is_a", "agent")

    research_props = {
        "rules_profile": {
            "slots": {"subject": {"classes": ["thing"], "required": True, "multiple": True}},
            "tool_permissions": [
                {"tool": "Read", "scope": "*"},
                {"tool": "Grep", "scope": "*"},
                {"tool": "Glob", "scope": "*"},
            ],
        }
    }
    kg.add_entity(
        "research",
        kind="class",
        description="Exploratory research",
        importance=5,
        properties=research_props,
    )
    kg.add_triple("research", "is_a", "intent_type")

    kg.add_entity("test_target", kind="entity", description="A test target", importance=3)
    kg.add_triple("test_target", "is_a", "thing")

    # Predicates used by the tooling
    for pred in (
        "is_a",
        "suggested_link",
        "found_useful",
        "found_irrelevant",
        "relates_to",
        "described_by",
        "executed_by",
        "targeted",
    ):
        kg.add_entity(pred, kind="predicate", description=f"predicate {pred}", importance=4)

    ecol.upsert(
        ids=[
            "intent_type",
            "thing",
            "agent",
            "test_agent",
            "research",
            "test_target",
        ],
        documents=[
            "Root class for all intent types",
            "Root class for all entities",
            "An AI agent",
            "Test agent",
            "Exploratory research",
            "A test target",
        ],
        metadatas=[
            {"name": "intent_type", "kind": "class", "importance": 5},
            {"name": "thing", "kind": "class", "importance": 5},
            {"name": "agent", "kind": "class", "importance": 5},
            {"name": "test_agent", "kind": "entity", "importance": 3, "added_by": "test_agent"},
            {"name": "research", "kind": "class", "importance": 5},
            {"name": "test_target", "kind": "entity", "importance": 3},
        ],
    )
    del client


@pytest.fixture
def mcp_env(monkeypatch, tmp_path):
    """Patched mcp_server with an isolated palace + KG + session-scoped state dir.

    Yields the mcp_server module; tests call ``mcp.tool_*`` directly.
    """
    from mempalace import mcp_server
    from mempalace.config import MempalaceConfig
    from mempalace.knowledge_graph import KnowledgeGraph

    palace_path = tmp_path / "palace"
    palace_path.mkdir()
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(json.dumps({"palace_path": str(palace_path)}))
    cfg = MempalaceConfig(config_dir=str(cfg_dir))

    kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
    _seed_intent_types(kg, str(palace_path))

    monkeypatch.setattr(mcp_server._STATE, "config", cfg)
    monkeypatch.setattr(mcp_server._STATE, "kg", kg)
    monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
    monkeypatch.setattr(mcp_server._STATE, "pending_conflicts", None)
    monkeypatch.setattr(mcp_server._STATE, "pending_enrichments", None)
    monkeypatch.setattr(mcp_server._STATE, "session_id", "rt-session")
    monkeypatch.setattr(mcp_server._STATE, "declared_entities", set())

    state_dir = tmp_path / "hook_state"
    state_dir.mkdir()
    monkeypatch.setattr(mcp_server, "_INTENT_STATE_DIR", state_dir)

    return mcp_server


def _state_file(mcp, sid: str = None) -> Path:
    sid = sid or mcp._STATE.session_id
    return mcp._INTENT_STATE_DIR / f"active_intent_{sid}.json"


def _seed_pending_enrichment(mcp, ident: str, fe: str = "test_target", te: str = "test_agent"):
    """Seed a pending enrichment both in memory AND on disk (realistic shape)."""
    from mempalace import intent

    entry = {
        "id": ident,
        "from_entity": fe,
        "to_entity": te,
        "reason": "seeded-for-test",
        "similarity": 0.6,
    }
    mcp._STATE.pending_enrichments = list(mcp._STATE.pending_enrichments or []) + [entry]
    intent._persist_active_intent()
    return entry


def _declare_research(mcp, *, subject="test_target"):
    """Helper: call tool_declare_intent with the 'research' intent type."""
    return mcp.tool_declare_intent(
        intent_type="research",
        slots={"subject": [subject]},
        context={
            "queries": ["round-trip smoke", "contract test"],
            "keywords": ["test", "round-trip"],
        },
        agent="test_agent",
        budget=_TEST_BUDGET,
    )


# ═══════════════════════════════════════════════════════════════════════
#  1. Happy-path round-trip
# ═══════════════════════════════════════════════════════════════════════


class TestResolveThenDeclare:
    def test_resolve_all_then_declare_unblocks(self, mcp_env):
        """seed pending → resolve all with action='done' → declare_intent succeeds.

        This is the single most important anti-deadlock property: a successful
        resolve_enrichments must leave the system in a state where the NEXT
        declare_intent is not blocked.
        """
        mcp = mcp_env
        e = _seed_pending_enrichment(mcp, "e1")

        # Baseline: declare is blocked.
        blocked = _declare_research(mcp)
        assert blocked["success"] is False
        assert "pending_enrichments" in blocked

        # Resolve with a proper action.
        resolved = mcp.tool_resolve_enrichments(
            actions=[
                {
                    "id": e["id"],
                    "action": "reject",
                    "reason": "Not actually a valid link for this test scenario.",
                }
            ],
            agent="test_agent",
        )
        assert resolved["success"] is True
        assert len(resolved["resolved"]) == 1

        # The gate MUST be open now.
        after = _declare_research(mcp)
        assert after["success"] is True, (
            "declare_intent remained blocked after a successful resolve — "
            "the exact class of bug this test exists to prevent"
        )
        # And the disk file must be cleared.
        assert (
            not _state_file(mcp).exists()
            or _state_file(mcp).stat().st_size == 0
            or json.loads(_state_file(mcp).read_text())["intent_id"] != ""
        )


# ═══════════════════════════════════════════════════════════════════════
#  2. Resolve with unknown IDs while pending exists — must not silently succeed
# ═══════════════════════════════════════════════════════════════════════


class TestResolveUnknownIds:
    def test_unknown_ids_while_pending_exists_returns_error(self, mcp_env):
        """If pending enrichments exist and the caller passes IDs that don't
        match any of them, the tool MUST return an error — not the
        misleading 'No pending enrichments' that masks the real state.

        This was THE bug on 2026-04-19: resolve returned a success envelope,
        declare_intent then re-blocked on the same items that 'resolved'
        claimed to have cleared.
        """
        mcp = mcp_env
        _seed_pending_enrichment(mcp, "e_real")

        result = mcp.tool_resolve_enrichments(
            actions=[
                {
                    "id": "e_ghost_nonexistent",
                    "action": "reject",
                    "reason": "Pretending to resolve a thing that isn't there.",
                }
            ],
            agent="test_agent",
        )
        # Must be an error OR at minimum must NOT claim global success.
        assert result.get("success") is False, (
            f"resolve claimed success for an ID that doesn't match pending. Response: {result!r}"
        )
        # Pending must still be on disk — next declare still blocks on e_real.
        blocked = _declare_research(mcp)
        assert blocked["success"] is False
        pending_ids = [p.get("id") for p in blocked.get("pending_enrichments", [])]
        assert "e_real" in pending_ids


# ═══════════════════════════════════════════════════════════════════════
#  3. Empty actions while pending exists
# ═══════════════════════════════════════════════════════════════════════


class TestResolveEmptyActions:
    def test_empty_actions_with_pending_returns_error(self, mcp_env):
        mcp = mcp_env
        _seed_pending_enrichment(mcp, "e1")

        result = mcp.tool_resolve_enrichments(actions=[], agent="test_agent")
        assert result["success"] is False
        assert "pending" in result or "Must provide" in result.get("error", "")

    def test_none_actions_with_pending_returns_error(self, mcp_env):
        mcp = mcp_env
        _seed_pending_enrichment(mcp, "e1")

        result = mcp.tool_resolve_enrichments(actions=None, agent="test_agent")
        assert result["success"] is False

    def test_empty_actions_with_no_pending_is_benign(self, mcp_env):
        mcp = mcp_env
        result = mcp.tool_resolve_enrichments(actions=[], agent="test_agent")
        assert result["success"] is True
        assert "No pending" in result.get("message", "")


# ═══════════════════════════════════════════════════════════════════════
#  4. Disk-only pending (in-memory empty) — Adrian's exact failure mode
# ═══════════════════════════════════════════════════════════════════════


class TestDiskOnlyPending:
    def test_resolve_loads_from_disk_when_memory_empty(self, mcp_env):
        """Simulate: process came up with in-memory state empty but disk
        still has pending from a prior session. resolve_enrichments must
        load from disk, match the IDs, and actually clear them.
        """
        mcp = mcp_env
        from mempalace import intent

        # Write disk-only pending, leave memory empty.
        mcp._STATE.pending_enrichments = [
            {
                "id": "disk_e1",
                "from_entity": "test_target",
                "to_entity": "test_agent",
                "reason": "disk-only seed",
            }
        ]
        intent._persist_active_intent()
        # Simulate a fresh process by clearing memory but leaving disk.
        mcp._STATE.pending_enrichments = None

        # Confirm disk has it.
        loaded = mcp._load_pending_enrichments_from_disk()
        assert len(loaded) == 1 and loaded[0]["id"] == "disk_e1"

        result = mcp.tool_resolve_enrichments(
            actions=[
                {
                    "id": "disk_e1",
                    "action": "reject",
                    "reason": "Disk-only pending should still be resolvable via the tool.",
                }
            ],
            agent="test_agent",
        )
        assert result["success"] is True

        # Both memory AND disk must now show no pending.
        assert mcp._STATE.pending_enrichments in (None, [])
        assert mcp._load_pending_enrichments_from_disk() == []

        # And declare_intent is unblocked.
        after = _declare_research(mcp)
        assert after["success"] is True


# ═══════════════════════════════════════════════════════════════════════
#  5. Partial resolve then declare still blocks on remainder
# ═══════════════════════════════════════════════════════════════════════


class TestPartialResolve:
    def test_partial_blocks_on_remainder_then_full_unblocks(self, mcp_env):
        mcp = mcp_env
        _seed_pending_enrichment(mcp, "e_one")
        _seed_pending_enrichment(mcp, "e_two")

        # Attempt: resolve only one, miss the other — must error (not silently
        # half-clear).
        partial = mcp.tool_resolve_enrichments(
            actions=[
                {
                    "id": "e_one",
                    "action": "reject",
                    "reason": "Resolving exactly one out of two deliberately.",
                }
            ],
            agent="test_agent",
        )
        assert partial["success"] is False
        assert "unresolved" in partial
        # The partially-done item MUST still be pending.
        assert mcp._STATE.pending_enrichments and len(mcp._STATE.pending_enrichments) == 2

        # Now resolve both.
        both = mcp.tool_resolve_enrichments(
            actions=[
                {"id": "e_one", "action": "reject", "reason": "First one, rejected properly."},
                {"id": "e_two", "action": "reject", "reason": "Second one, rejected properly."},
            ],
            agent="test_agent",
        )
        assert both["success"] is True
        # Declare is unblocked.
        assert _declare_research(mcp)["success"] is True


# ═══════════════════════════════════════════════════════════════════════
#  6. Double-resolve idempotency
# ═══════════════════════════════════════════════════════════════════════


class TestIdempotency:
    def test_resolve_twice_with_same_actions_is_safe(self, mcp_env):
        mcp = mcp_env
        _seed_pending_enrichment(mcp, "e1")

        first = mcp.tool_resolve_enrichments(
            actions=[
                {"id": "e1", "action": "reject", "reason": "First resolve of e1 — legitimate."}
            ],
            agent="test_agent",
        )
        assert first["success"] is True

        # Second call with same actions — nothing pending, should be clean.
        second = mcp.tool_resolve_enrichments(
            actions=[
                {
                    "id": "e1",
                    "action": "reject",
                    "reason": "Second resolve of e1 — idempotent retry.",
                }
            ],
            agent="test_agent",
        )
        # Either success with "No pending enrichments" or success with empty resolved.
        assert second.get("success") is True

        # And declare STILL works.
        assert _declare_research(mcp)["success"] is True

    def test_declare_resolve_declare_resolve_converges(self, mcp_env):
        """Alternating declare+finalize and resolve cycles never accumulate
        state drift.
        """
        mcp = mcp_env
        for i in range(5):
            _seed_pending_enrichment(mcp, f"cycle_e_{i}")
            # Declare is blocked.
            blocked = _declare_research(mcp)
            assert blocked["success"] is False
            # Resolve.
            result = mcp.tool_resolve_enrichments(
                actions=[
                    {
                        "id": f"cycle_e_{i}",
                        "action": "reject",
                        "reason": f"Idempotent cycle iteration {i} — resolved correctly.",
                    }
                ],
                agent="test_agent",
            )
            assert result["success"] is True
            # Declare succeeds.
            ok = _declare_research(mcp)
            assert ok["success"] is True, f"iteration {i} left the gate locked"
            # Finalize to clean up for next loop.
            mcp._STATE.active_intent = None
            mcp._STATE.pending_conflicts = None
            mcp._STATE.pending_enrichments = None


# ═══════════════════════════════════════════════════════════════════════
#  7. Orphan auto-prune via REAL tool_declare_intent
# ═══════════════════════════════════════════════════════════════════════


class TestOrphanPruneReal:
    """The original test file reimplements the prune logic inline; this
    exercises the production code path instead."""

    def test_tool_declare_intent_prunes_orphan_enrichment(self, mcp_env):
        mcp = mcp_env
        from mempalace import intent

        # Seed a pending enrichment whose from_entity does NOT exist in the KG.
        mcp._STATE.pending_enrichments = [
            {
                "id": "e_ghost",
                "from_entity": "ghost_deleted_entity",
                "to_entity": "test_target",
                "reason": "orphan",
            }
        ]
        intent._persist_active_intent()
        assert _state_file(mcp).is_file()

        # declare_intent should AUTO-PRUNE the orphan and succeed.
        result = _declare_research(mcp)
        assert result["success"] is True, (
            f"orphan pending should have been auto-pruned; got: {result!r}"
        )
        # Pending state (enrichment-side) now empty.
        assert not mcp._STATE.pending_enrichments

    def test_tool_declare_intent_keeps_valid_enrichment_blocks(self, mcp_env):
        """Orphan-prune must NOT drop a legitimate pending enrichment."""
        mcp = mcp_env
        _seed_pending_enrichment(mcp, "e_valid", fe="test_target", te="test_agent")

        result = _declare_research(mcp)
        assert result["success"] is False, "declare should still block on valid pending enrichment"
        assert any(p["id"] == "e_valid" for p in result.get("pending_enrichments", []))


# ═══════════════════════════════════════════════════════════════════════
#  8. Session-switch round-trip with pending-only state
# ═══════════════════════════════════════════════════════════════════════


class TestSessionSwitchRoundTrip:
    def test_sid_switch_preserves_other_sid_pending(self, mcp_env):
        """sid A has pending, switch to sid B (no pending), back to A — A's
        pending still resolvable via the tool."""
        mcp = mcp_env
        # Seed pending under sid A.
        mcp._STATE.session_id = "sid-a"
        _seed_pending_enrichment(mcp, "a1")
        assert _state_file(mcp, "sid-a").is_file()

        # Switch to sid B.
        mcp._save_session_state()
        mcp._STATE.session_id = "sid-b"
        mcp._restore_session_state("sid-b")
        assert not mcp._STATE.pending_enrichments, "sid B must not see sid A's pending"
        # declare on sid B is unblocked.
        ok_b = _declare_research(mcp)
        assert ok_b["success"] is True

        # Switch back to sid A — pending must still be there.
        mcp._save_session_state()
        mcp._STATE.active_intent = None
        mcp._STATE.session_id = "sid-a"
        mcp._restore_session_state("sid-a")
        assert mcp._STATE.pending_enrichments, "sid A pending vanished during switch"
        # And it resolves cleanly.
        result = mcp.tool_resolve_enrichments(
            actions=[
                {
                    "id": "a1",
                    "action": "reject",
                    "reason": "Resolving sid-a's pending after round-trip switch.",
                }
            ],
            agent="test_agent",
        )
        assert result["success"] is True
        # declare_intent on sid A is now unblocked.
        assert _declare_research(mcp)["success"] is True


# ═══════════════════════════════════════════════════════════════════════
#  9. kg_declare_entity → suggested links → resolve → declare
# ═══════════════════════════════════════════════════════════════════════


class TestDeclareEntitySuggestedLinksRoundTrip:
    def test_declare_entity_enrichments_do_not_deadlock(self, mcp_env):
        """Real-world flow: declare an entity that has semantic neighbours,
        suggested_links get proposed, agent resolves them, declare_intent
        proceeds unblocked. No fabricated pending lists."""
        mcp = mcp_env
        # Seed a neighbour with high semantic overlap to provoke suggested_links.
        # kg_declare_entity runs suggested-link detection on the entities collection.
        # Seed one "similar" entity first.
        mcp.tool_kg_declare_entity(
            name="feedback_system_seed",
            kind="entity",
            description=(
                "MemPalace feedback subsystem: recording, collection, application, learning"
            ),
            added_by="test_agent",
            importance=3,
            context={"queries": ["feedback subsystem"], "keywords": ["feedback"]},
        )

        # Now declare a new entity with a close description.
        mcp.tool_kg_declare_entity(
            name="feedback_system_probe",
            kind="entity",
            description=(
                "MemPalace feedback subsystem: recording, collection, application, learning"
            ),
            added_by="test_agent",
            importance=3,
            context={"queries": ["feedback subsystem"], "keywords": ["feedback"]},
        )
        # This may or may not produce pending enrichments (depends on
        # similarity threshold + collection state). If it does, resolve them.
        pending = mcp._STATE.pending_enrichments or []
        if pending:
            actions = [
                {
                    "id": p["id"],
                    "action": "reject",
                    "reason": f"Probe entity's enrichment {p['id']} rejected for the round-trip test.",
                }
                for p in pending
            ]
            resolved = mcp.tool_resolve_enrichments(actions=actions, agent="test_agent")
            assert resolved["success"] is True

        # declare_intent must proceed without being trapped.
        result = _declare_research(mcp)
        assert result["success"] is True, (
            f"kg_declare_entity enrichments left the gate locked; response: {result!r}"
        )


# ═══════════════════════════════════════════════════════════════════════
# 10. Corrupted state file
# ═══════════════════════════════════════════════════════════════════════


class TestCorruptedState:
    def test_corrupted_file_does_not_crash_resolve(self, mcp_env):
        mcp = mcp_env
        _state_file(mcp).write_text("{ corrupt json ")

        result = mcp.tool_resolve_enrichments(actions=[], agent="test_agent")
        # Must not raise, must not block — returns benign "no pending".
        assert result["success"] is True

    def test_corrupted_file_does_not_crash_declare(self, mcp_env):
        mcp = mcp_env
        _state_file(mcp).write_text("{ also corrupt")
        result = _declare_research(mcp)
        assert result["success"] is True


# ═══════════════════════════════════════════════════════════════════════
# 11. Hyphen / underscore id-shape collision
# ═══════════════════════════════════════════════════════════════════════


class TestIdShapeCollision:
    def test_hyphenated_to_entity_is_pruned_if_only_underscore_exists(self, mcp_env):
        """A pending enrichment referencing a hyphenated entity-id that the
        KG only stores in underscored form must be treated as orphan and
        pruned on declare_intent. The inverse (KG has hyphen, enrichment has
        underscore) applies symmetrically."""
        mcp = mcp_env
        from mempalace import intent

        # Seed pending with a hyphenated to_entity that doesn't exist.
        mcp._STATE.pending_enrichments = [
            {
                "id": "e_hyphen",
                "from_entity": "test_target",
                "to_entity": "record-with-hyphens-in-id",
                "reason": "hyphen/underscore collision",
            }
        ]
        intent._persist_active_intent()

        result = _declare_research(mcp)
        # Either the orphan-prune drops it (declare succeeds) or the tool
        # returns a specific error pointing at the mismatch — both are
        # acceptable. A silent, permanent block is NOT.
        if result["success"] is False:
            pending = result.get("pending_enrichments", [])
            # If it's still blocking, the tool owes the caller a way to resolve.
            # The agent must be able to call resolve with e_hyphen and have it
            # succeed without KG entity existence.
            assert any(p["id"] == "e_hyphen" for p in pending)
            res = mcp.tool_resolve_enrichments(
                actions=[
                    {
                        "id": "e_hyphen",
                        "action": "reject",
                        "reason": "Hyphen/underscore id-shape collision — entity does not exist.",
                    }
                ],
                agent="test_agent",
            )
            assert res["success"] is True
            # And now declare proceeds.
            assert _declare_research(mcp)["success"] is True
        else:
            # Orphan-prune dropped it silently — also acceptable.
            assert not mcp._STATE.pending_enrichments


# ═══════════════════════════════════════════════════════════════════════
# 12. Rapid alternation stress
# ═══════════════════════════════════════════════════════════════════════


class TestStressAlternation:
    @pytest.mark.parametrize("iterations", [20])
    def test_no_state_leak_across_many_cycles(self, mcp_env, iterations):
        mcp = mcp_env
        for i in range(iterations):
            _seed_pending_enrichment(mcp, f"stress_{i}")
            assert _declare_research(mcp)["success"] is False
            result = mcp.tool_resolve_enrichments(
                actions=[
                    {
                        "id": f"stress_{i}",
                        "action": "reject",
                        "reason": f"Stress iteration {i} — rejection of seeded enrichment.",
                    }
                ],
                agent="test_agent",
            )
            assert result["success"] is True
            assert _declare_research(mcp)["success"] is True
            mcp._STATE.active_intent = None


# ═══════════════════════════════════════════════════════════════════════
# 13. Conflicts round-trip
# ═══════════════════════════════════════════════════════════════════════


class TestConflictsRoundTrip:
    def test_resolve_conflicts_then_declare_unblocks(self, mcp_env):
        mcp = mcp_env
        from mempalace import intent

        mcp._STATE.pending_conflicts = [
            {
                "id": "conflict_demo",
                "conflict_type": "edge_contradiction",
                "reason": "demo conflict",
                "existing_id": "existing_edge",
                "new_id": "new_edge",
            }
        ]
        intent._persist_active_intent()

        # Baseline: blocked.
        blocked = _declare_research(mcp)
        assert blocked["success"] is False

        # Resolve.
        result = mcp.tool_resolve_conflicts(
            actions=[
                {
                    "id": "conflict_demo",
                    "action": "keep",
                    "reason": "Keeping both — test round-trip.",
                }
            ],
            agent="test_agent",
        )
        assert result["success"] is True

        # Unblocked.
        assert _declare_research(mcp)["success"] is True


# ═══════════════════════════════════════════════════════════════════════
# 14. Response-envelope integrity
# ═══════════════════════════════════════════════════════════════════════


class TestResponseEnvelope:
    def test_resolve_response_does_not_lie_about_pending_when_state_is_stale(self, mcp_env):
        """If the disk state file for the current sid has pending items
        but _STATE.pending_enrichments is None (fresh process), the tool
        MUST load from disk and treat that state as pending. Returning
        'No pending enrichments' while disk has pending is a lie that
        caused the 2026-04-19 deadlock."""
        mcp = mcp_env
        from mempalace import intent

        # Set pending + persist.
        mcp._STATE.pending_enrichments = [
            {"id": "disk_truth", "from_entity": "test_target", "to_entity": "test_agent"}
        ]
        intent._persist_active_intent()

        # Clear in-memory only; disk remains authoritative.
        mcp._STATE.pending_enrichments = None

        # Call resolve with NO actions — tool should surface the pending from disk,
        # not fabricate success.
        result = mcp.tool_resolve_enrichments(actions=[], agent="test_agent")
        assert result["success"] is False, (
            f"resolve claimed success while disk still has pending; response: {result!r}"
        )

    def test_declare_intent_surfaces_disk_pending_after_restart(self, mcp_env):
        """After a 'restart' (memory cleared, disk intact), declare_intent
        must block on the disk-resident pending state."""
        mcp = mcp_env
        from mempalace import intent

        mcp._STATE.pending_enrichments = [
            {"id": "e_disk", "from_entity": "test_target", "to_entity": "test_agent"}
        ]
        intent._persist_active_intent()
        mcp._STATE.pending_enrichments = None  # "restart"

        result = _declare_research(mcp)
        assert result["success"] is False
        assert any(p["id"] == "e_disk" for p in result.get("pending_enrichments", []))


# ═══════════════════════════════════════════════════════════════════════
# 15. Session-id oscillation (parent + subagent interleaving)
# ═══════════════════════════════════════════════════════════════════════


class TestSessionIdOscillation:
    """The 2026-04-19 deadlock root cause: subagent tool calls and parent
    tool calls arrive at the MCP server with different sessionId values.
    Each switch reloads pending state from the OTHER session's disk file,
    which looks like state randomly appearing/disappearing to the agent.

    Every tool must be sid-correct: resolve_enrichments clears ONLY the
    current sid's pending, never the other's. declare_intent sees ONLY
    the current sid's pending."""

    def test_oscillating_sids_do_not_cross_contaminate(self, mcp_env):
        mcp = mcp_env

        # Parent sid P seeds its own pending.
        mcp._STATE.session_id = "sid-parent"
        _seed_pending_enrichment(mcp, "p1")
        # Simulate switch: Claude-in-Chrome dispatches a subagent tool call.
        mcp._save_session_state()
        mcp._STATE.session_id = "sid-sub"
        mcp._restore_session_state("sid-sub")
        assert mcp._STATE.pending_enrichments is None, "subagent saw parent pending"

        # Subagent seeds its own pending.
        _seed_pending_enrichment(mcp, "s1")
        # Switch back to parent.
        mcp._save_session_state()
        mcp._STATE.session_id = "sid-parent"
        mcp._restore_session_state("sid-parent")
        assert mcp._STATE.pending_enrichments
        assert mcp._STATE.pending_enrichments[0]["id"] == "p1"
        # Parent resolves its own only.
        result = mcp.tool_resolve_enrichments(
            actions=[
                {
                    "id": "p1",
                    "action": "reject",
                    "reason": "Parent resolving its own sid's pending, not subagent's.",
                }
            ],
            agent="test_agent",
        )
        assert result["success"] is True

        # Subagent's pending must STILL be there on sid-sub.
        mcp._save_session_state()
        mcp._STATE.session_id = "sid-sub"
        mcp._restore_session_state("sid-sub")
        assert mcp._STATE.pending_enrichments
        assert mcp._STATE.pending_enrichments[0]["id"] == "s1"

    def test_resolve_on_empty_sid_does_not_touch_other_sid(self, mcp_env):
        """Key contract: resolve_enrichments on sid-B (empty) must not
        clear sid-A's disk file. Sid isolation is absolute."""
        mcp = mcp_env

        mcp._STATE.session_id = "sid-a"
        _seed_pending_enrichment(mcp, "a1")
        a_file = _state_file(mcp, "sid-a")
        assert a_file.is_file()

        # Switch to empty sid-b.
        mcp._save_session_state()
        mcp._STATE.session_id = "sid-b"
        mcp._restore_session_state("sid-b")
        # Attempt a resolve that finds nothing.
        result = mcp.tool_resolve_enrichments(actions=[], agent="test_agent")
        assert result["success"] is True

        # Sid-A's disk file MUST still exist with its pending.
        assert a_file.is_file(), "sid-b's empty resolve touched sid-a's disk file"
        data = json.loads(a_file.read_text())
        assert data.get("pending_enrichments"), "sid-a's pending got dropped"
        assert data["pending_enrichments"][0]["id"] == "a1"

    def test_declare_on_sid_b_is_unblocked_when_only_sid_a_has_pending(self, mcp_env):
        """Even if sid-A has a mountain of pending, sid-B's declare_intent
        must not be blocked by A's state. Cross-sid contamination is the
        bug this rules out."""
        mcp = mcp_env
        mcp._STATE.session_id = "sid-a"
        for i in range(5):
            _seed_pending_enrichment(mcp, f"a_load_{i}")

        mcp._save_session_state()
        mcp._STATE.session_id = "sid-b"
        mcp._restore_session_state("sid-b")

        result = _declare_research(mcp)
        assert result["success"] is True, (
            "sid-B's declare was blocked by sid-A's pending — cross-contamination bug"
        )

    def test_rapid_oscillation_preserves_each_sids_state(self, mcp_env):
        """Stress: alternate sid-a ↔ sid-b 10 times; each sid's state must
        remain intact throughout."""
        mcp = mcp_env
        # Seed both.
        mcp._STATE.session_id = "sid-a"
        _seed_pending_enrichment(mcp, "a_stress")
        mcp._save_session_state()
        mcp._STATE.session_id = "sid-b"
        mcp._restore_session_state("sid-b")
        _seed_pending_enrichment(mcp, "b_stress")

        for i in range(10):
            target = "sid-a" if i % 2 == 0 else "sid-b"
            expected_id = "a_stress" if target == "sid-a" else "b_stress"
            mcp._save_session_state()
            mcp._STATE.session_id = target
            mcp._restore_session_state(target)
            assert mcp._STATE.pending_enrichments, f"iter {i}: lost pending on {target}"
            assert mcp._STATE.pending_enrichments[0]["id"] == expected_id


# ═══════════════════════════════════════════════════════════════════════
# 16. Full MCP dispatch with injected sessionId (end-to-end)
# ═══════════════════════════════════════════════════════════════════════


class TestMCPDispatchSessionId:
    """The production sid-switch path runs inside _handle_request when a
    tool call arrives with sessionId in its arguments. These tests
    exercise THAT exact code path rather than calling tool_* functions
    directly. A regression in the sid-injection middleware would only
    show up here."""

    def test_dispatch_with_differing_sid_switches_state(self, mcp_env):
        mcp = mcp_env

        # Seed pending for sid-alpha via direct state + persist.
        mcp._STATE.session_id = "sid-alpha"
        _seed_pending_enrichment(mcp, "alpha_1")

        # Dispatch a tool call that injects sessionId='sid-beta'.
        # Any tool works for the switch side-effect; use tool_active_intent
        # because it requires no arguments.
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "mempalace_active_intent",
                "arguments": {"sessionId": "sid-beta"},
            },
        }
        mcp.handle_request(req)

        # After this, _STATE.session_id must be sid-beta, and pending
        # must reflect sid-beta's disk (empty).
        assert mcp._STATE.session_id == "sid-beta"
        assert mcp._STATE.pending_enrichments is None

        # sid-alpha's disk file is untouched.
        alpha_file = _state_file(mcp, "sid-alpha")
        assert alpha_file.is_file()
        data = json.loads(alpha_file.read_text())
        assert data["pending_enrichments"][0]["id"] == "alpha_1"
