"""Regression tests for the default-OFF state keeper feature flag.

Adrian directive 2026-05-28: MEMPALACE_STATE_KEEPER_ENABLED gates the
ENTIRE state-protocol surface. When unset (default) the agent must see
NO trace of state anywhere -- not in the wake_up protocol, not in the
tools/list schemas, not in any tool/param description. These tests lock
that in and catch prose drift (a new state aside added to a description
will fail test_tools_list_off_has_zero_state_tokens loudly).
"""

import mempalace.mcp_server as m
from mempalace.state_schemas import mentions_state, state_keeper_enabled

pytestmark = __import__("pytest").mark.integration


def _all_strings(obj):
    """Yield every string (dict keys + values, list items) reachable in obj."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                yield k
            yield from _all_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _all_strings(v)


def test_mentions_state_detects_surface_tokens():
    for tok in (
        "state_schema",
        "state_changed_by",
        "state_deltas",
        "state_updatable",
        "state-bearing",
        "state-protocol",
        "state keeper",
        "state_judge",
        "initial_state",
        "initial_intent_state",
        "current_state",
        "wake_up.schemas",
        "STATE_SCHEMAS",
        "intent_state",
        "agent_state",
        "task_state",
        "project_state",
        "challenge_state_change",
    ):
        assert mentions_state(tok), f"should flag {tok!r}"


def test_mentions_state_ignores_unrelated_words():
    # The bare word "state" must NOT trip the detector (avoid false
    # positives on ordinary prose / enum values).
    for tok in ("statement", "state of the art", "restate", "status", "estate", "stateless"):
        assert not mentions_state(tok), f"should NOT flag {tok!r}"


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("MEMPALACE_STATE_KEEPER_ENABLED", raising=False)
    assert state_keeper_enabled() is False


def test_tools_list_off_has_zero_state_tokens(monkeypatch):
    monkeypatch.delenv("MEMPALACE_STATE_KEEPER_ENABLED", raising=False)
    filtered = m._filter_tools_for_state(m.TOOLS)

    # State-only tool is gone entirely.
    assert "mempalace_challenge_state_change" not in filtered

    # No string anywhere in the advertised tool list names state.
    leaks = []
    for name, spec in filtered.items():
        payload = {
            "name": name,
            "description": spec.get("description"),
            "input_schema": spec.get("input_schema"),
        }
        for s in _all_strings(payload):
            if mentions_state(s):
                leaks.append((name, s[:140]))
    assert not leaks, f"state tokens leaked in tools/list when keeper off: {leaks}"


def test_tools_list_on_exposes_state(monkeypatch):
    monkeypatch.setenv("MEMPALACE_STATE_KEEPER_ENABLED", "1")
    filtered = m._filter_tools_for_state(m.TOOLS)
    assert "mempalace_challenge_state_change" in filtered
    do_props = filtered["mempalace_declare_operation"]["input_schema"]["properties"]
    assert "state_deltas" in do_props
    fi_props = filtered["mempalace_finalize_intent"]["input_schema"]["properties"]
    assert "state_deltas" in fi_props


def test_original_tools_dict_untouched(monkeypatch):
    # Filtering must deep-copy: the global TOOLS keeps its state surface so
    # flipping the env var on (server restart) restores everything.
    monkeypatch.delenv("MEMPALACE_STATE_KEEPER_ENABLED", raising=False)
    m._filter_tools_for_state(m.TOOLS)
    assert "mempalace_challenge_state_change" in m.TOOLS
    assert "state_deltas" in m.TOOLS["mempalace_declare_operation"]["input_schema"]["properties"]


def test_build_protocol_gated(monkeypatch):
    monkeypatch.delenv("MEMPALACE_STATE_KEEPER_ENABLED", raising=False)
    off = m.build_protocol()
    assert "STATE-PROTOCOL v1" not in off
    assert "IMPLICIT ACTIVE SET" not in off
    # non-state sections survive
    assert "USER-INTENT TIER" in off
    assert "WHEN RECEIVING INJECTED MEMORIES" in off

    monkeypatch.setenv("MEMPALACE_STATE_KEEPER_ENABLED", "1")
    on = m.build_protocol()
    assert "STATE-PROTOCOL v1" in on
    # full constant is always the complete text (back-compat)
    assert "STATE-PROTOCOL v1" in m.PALACE_PROTOCOL


def test_declared_block_off_drops_state(monkeypatch):
    monkeypatch.delenv("MEMPALACE_STATE_KEEPER_ENABLED", raising=False)
    import mempalace.tool_lifecycle as tl

    declared = {
        "predicates": "is_a, state_changed_by, depends_on",
        "classes": "agent, state_schema, person, task",
        "intent_types": "inspect<execute | research<inspect",
        "entities": "dspot_infra[5], state_keeper_feature_flag[4], person[4]",
        "count": 3,
    }
    out = tl._scrub_declared_for_state(declared)

    # state entries gone
    assert "state_changed_by" not in out["predicates"]
    assert "state_schema" not in out["classes"]
    assert "state_keeper_feature_flag" not in out["entities"]
    # non-state entries preserved
    assert "is_a" in out["predicates"] and "depends_on" in out["predicates"]
    assert "agent" in out["classes"] and "person" in out["classes"]
    # zero state tokens anywhere in the scrubbed block
    for k in ("predicates", "classes", "intent_types", "entities"):
        assert not mentions_state(out[k]), (k, out[k])


def test_declared_block_on_unchanged(monkeypatch):
    monkeypatch.setenv("MEMPALACE_STATE_KEEPER_ENABLED", "1")
    import mempalace.tool_lifecycle as tl

    declared = {
        "predicates": "is_a, state_changed_by",
        "classes": "agent, state_schema",
        "intent_types": "inspect",
        "entities": "x[1]",
        "count": 1,
    }
    assert tl._scrub_declared_for_state(declared) == declared
