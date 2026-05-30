"""Regression tests for the rules_profile shallow-merge clobber (2026-05-26).

Three production incidents (2026-05-09, -12, -28) wiped intent-type class
``rules_profile`` data, surfacing to agents as "intent type X has no slots
defined" and forcing intent-type hopping.

ROOT CAUSE: a partial ``rules_profile`` update merged with a top-level
``dict.update`` REPLACED the whole rules_profile, dropping the sibling
sub-key. The protocol's own gate-block remedy tells agents to write a
partial rules_profile (just ``tool_permissions``, or just ``slots``), so the
shallow merge silently destroyed the other half.

FIX: both write paths -- ``tool_mutate.tool_kg_update_entity`` (agent-facing)
and ``knowledge_graph.update_entity_properties`` (internal) -- now DEEP-merge
the rules_profile sub-dict: incoming sub-keys win, absent sub-keys are
preserved.

(The separate finalize-collision corruption -- an execution entity
overwriting a same-named class row -- is prevented by the exec_id
class-collision guard in ``finalize_intent``.)
"""

import json

from mempalace.knowledge_graph import KnowledgeGraph


def _rules_profile(kg: KnowledgeGraph, name: str) -> dict:
    ent = kg.get_entity(name)
    props = ent.get("properties") or {}
    if isinstance(props, str):
        props = json.loads(props)
    return props.get("rules_profile") or {}


def test_partial_update_tool_permissions_preserves_slots(tmp_path):
    """Updating ONLY tool_permissions must keep the existing slots."""
    kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
    kg.seed_ontology()

    before = _rules_profile(kg, "modify")
    assert set(before["slots"]) == {"files", "paths"}
    new_perms = before["tool_permissions"] + [{"tool": "Bash", "scope": "{commands}"}]

    kg.update_entity_properties("modify", {"rules_profile": {"tool_permissions": new_perms}})

    after = _rules_profile(kg, "modify")
    assert set(after["slots"]) == {"files", "paths"}, "slots were clobbered"
    assert len(after["tool_permissions"]) == len(before["tool_permissions"]) + 1


def test_partial_update_slots_preserves_tool_permissions(tmp_path):
    """Updating ONLY slots must keep the existing tool_permissions."""
    kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
    kg.seed_ontology()

    before = _rules_profile(kg, "execute")
    n_perms = len(before["tool_permissions"])
    new_slots = dict(before["slots"])
    new_slots["extra"] = {"raw": True, "required": False, "multiple": True}

    kg.update_entity_properties("execute", {"rules_profile": {"slots": new_slots}})

    after = _rules_profile(kg, "execute")
    assert len(after["tool_permissions"]) == n_perms, "tool_permissions were clobbered"
    assert "extra" in after["slots"]


def test_non_rules_profile_keys_unaffected(tmp_path):
    """A non-rules_profile property update must not touch rules_profile."""
    kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
    kg.seed_ontology()

    before = _rules_profile(kg, "inspect")
    kg.update_entity_properties("inspect", {"some_marker": 123})

    after = _rules_profile(kg, "inspect")
    assert after == before
    ent = kg.get_entity("inspect")
    props = ent["properties"]
    if isinstance(props, str):
        props = json.loads(props)
    assert props.get("some_marker") == 123


def test_tool_layer_partial_update_preserves_slots(tmp_path, monkeypatch):
    """End-to-end through the agent-facing kg_update_entity tool: a partial
    rules_profile update (the protocol's gate-block remedy) preserves slots."""
    monkeypatch.setenv("MEMPALACE_SKIP_SEED", "1")
    from mempalace import mcp_server as _mcp
    from mempalace.tool_mutate import tool_kg_update_entity

    kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
    kg.seed_ontology()
    # Declare a test agent (tool_kg_update_entity validates the agent exists).
    kg.add_entity("test_agent", kind="entity")
    kg.add_triple("test_agent", "is_a", "agent")
    monkeypatch.setattr(_mcp._STATE, "kg", kg, raising=False)
    monkeypatch.setattr(_mcp._STATE, "session_id", "test_sid_rpp", raising=False)

    before = _rules_profile(kg, "modify")
    new_perms = before["tool_permissions"] + [{"tool": "Bash", "scope": "{commands}"}]
    res = tool_kg_update_entity(
        entity="modify",
        agent="test_agent",
        properties={"rules_profile": {"tool_permissions": new_perms}},
    )
    assert res.get("success") is not False, res

    after = _rules_profile(kg, "modify")
    assert set(after["slots"]) == {"files", "paths"}, "slots clobbered via tool layer"
    assert len(after["tool_permissions"]) == len(before["tool_permissions"]) + 1


def test_class_protection_guard_blocks_entity_clobber(tmp_path):
    """add_entity must REFUSE to overwrite an existing kind='class' row with a
    non-class write -- the storage-layer backstop that makes the intent-type
    corruption impossible regardless of caller (an execution finalized with a
    slug == the type name, a gardener kind-flip, etc.)."""
    kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
    kg.add_entity(
        "myclass",
        kind="class",
        content="a class",
        importance=4,
        properties={"rules_profile": {"slots": {"x": {"raw": True}}}},
    )
    # Simulate the finalize-collision: a kind='entity' write at the same id.
    kg.add_entity(
        "myclass",
        kind="entity",
        content="an execution",
        importance=3,
        properties={"outcome": "partial", "finalized_at": "2026-05-28T20:50:04"},
    )
    ent = kg.get_entity("myclass")
    assert ent["kind"] == "class", "class was clobbered to entity"
    assert _rules_profile(kg, "myclass").get("slots"), "rules_profile was wiped"


def test_class_protection_guard_allows_legit_class_update(tmp_path):
    """The guard must NOT block a legitimate class re-write (kind='class') --
    e.g. seed re-runs or kg_declare_entity updating a class."""
    kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
    kg.add_entity(
        "myclass",
        kind="class",
        content="a class",
        importance=4,
        properties={"rules_profile": {"slots": {"x": {"raw": True}}}},
    )
    kg.add_entity(
        "myclass",
        kind="class",
        content="updated class",
        importance=4,
        properties={
            "rules_profile": {
                "slots": {"x": {"raw": True}},
                "tool_permissions": [{"tool": "Bash", "scope": "*"}],
            }
        },
    )
    assert kg.get_entity("myclass")["kind"] == "class"
    assert _rules_profile(kg, "myclass").get("tool_permissions"), "legit update blocked"


# ── Additive (recursive) merge: a PARTIAL fragment must blend in, not replace ──
# These cover the residual clobber the 2026-05-26 sibling-only fix missed:
# sending JUST the new tool_permissions entry (or JUST a new slot) used to
# REPLACE the whole list/dict, zeroing the class's tools. The gate-block
# message promises "Tools are ADDITIVE"; the merge must honour that.


def test_partial_tool_permissions_fragment_preserves_existing_tools(tmp_path):
    """Sending ONLY a new tool_permissions entry (a fragment, not the full
    list) must PRESERVE existing perms and ADD the new one."""
    kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
    kg.seed_ontology()

    before = _rules_profile(kg, "modify")
    n_before = len(before["tool_permissions"])
    assert n_before > 0

    # Fragment only -- NOT before["tool_permissions"] + [...].
    kg.update_entity_properties(
        "modify",
        {"rules_profile": {"tool_permissions": [{"tool": "Bash", "scope": "git status"}]}},
    )

    after = _rules_profile(kg, "modify")
    perms = after["tool_permissions"]
    assert len(perms) == n_before + 1, "existing tools dropped by a partial update"
    assert {"tool": "Bash", "scope": "git status"} in perms
    for p in before["tool_permissions"]:
        assert p in perms, "a pre-existing permission was lost"
    assert set(after["slots"]) == {"files", "paths"}


def test_partial_single_slot_preserves_existing_slots(tmp_path):
    """Sending ONLY a new slot (not the full slots dict) must preserve the
    existing slots and add the new one (recursive dict merge)."""
    kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
    kg.seed_ontology()

    before = _rules_profile(kg, "modify")
    assert set(before["slots"]) == {"files", "paths"}

    kg.update_entity_properties(
        "modify",
        {"rules_profile": {"slots": {"extra": {"raw": True, "required": False, "multiple": True}}}},
    )

    after = _rules_profile(kg, "modify")
    assert set(after["slots"]) == {"files", "paths", "extra"}, "existing slots dropped"
    assert len(after["tool_permissions"]) == len(before["tool_permissions"])


def test_additive_merge_dedups_identical_perm(tmp_path):
    """Re-sending an already-present permission is a no-op (union dedup)."""
    kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
    kg.seed_ontology()

    before = _rules_profile(kg, "modify")
    existing_perm = before["tool_permissions"][0]
    kg.update_entity_properties("modify", {"rules_profile": {"tool_permissions": [existing_perm]}})
    after = _rules_profile(kg, "modify")
    assert len(after["tool_permissions"]) == len(before["tool_permissions"]), "dup added"


def test_tool_layer_partial_fragment_preserves_tools(tmp_path, monkeypatch):
    """Agent-facing tool path: sending ONLY a new tool_permissions entry
    preserves existing tools -- the exact scenario that zeroed a custom
    class when an agent 'added a tool' via the gate-block remedy."""
    monkeypatch.setenv("MEMPALACE_SKIP_SEED", "1")
    from mempalace import mcp_server as _mcp
    from mempalace.tool_mutate import tool_kg_update_entity

    kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
    kg.seed_ontology()
    kg.add_entity("test_agent", kind="entity")
    kg.add_triple("test_agent", "is_a", "agent")
    monkeypatch.setattr(_mcp._STATE, "kg", kg, raising=False)
    monkeypatch.setattr(_mcp._STATE, "session_id", "test_sid_frag", raising=False)

    before = _rules_profile(kg, "modify")
    n_before = len(before["tool_permissions"])
    res = tool_kg_update_entity(
        entity="modify",
        agent="test_agent",
        properties={"rules_profile": {"tool_permissions": [{"tool": "Bash", "scope": "gh"}]}},
    )
    assert res.get("success") is not False, res

    after = _rules_profile(kg, "modify")
    assert len(after["tool_permissions"]) == n_before + 1, "tool layer dropped existing tools"
    assert {"tool": "Bash", "scope": "gh"} in after["tool_permissions"]
    assert set(after["slots"]) == {"files", "paths"}
