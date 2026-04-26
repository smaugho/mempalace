"""Drift detection between gardener _TOOL_SCHEMAS and mcp_server TOOLS.

The gardener exposes 8 tools to Haiku (raw Anthropic SDK loop, NOT MCP). Each
tool the gardener exposes that ALSO exists in the canonical MCP server TOOLS
dict must have a structurally-compatible input_schema, otherwise Haiku is
told one thing and the server enforces another (the kg_update_entity
description string-vs-dict bite that bit us 2026-04-26).

These tests fail loudly on:
  - a property type mismatch (Type B drift — Haiku sends the wrong shape)
  - a required field present in canonical but missing in gardener AND not in
    the shim auto-inject allowlist (Type B drift — Haiku omits a required field)
  - object-shaped properties whose nested `required` set diverges

Type A drifts (gardener missing OPTIONAL canonical fields) are reported as
warnings only — they cap Haiku's expressiveness but don't break it.

Two gardener tools have no canonical counterpart (propose_edge_candidate,
synthesize_operation_template) — they live as shims in memory_gardener.
For those we validate against the shim's Python signature instead.
"""

from __future__ import annotations

import inspect

import pytest

from mempalace import memory_gardener
from mempalace import mcp_server


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _gardener_schemas() -> dict[str, dict]:
    """Map of gardener tool name -> input_schema dict."""
    return {t["name"]: t["input_schema"] for t in memory_gardener._TOOL_SCHEMAS}


def _canonical_schemas() -> dict[str, dict]:
    """Map of canonical mcp_server tool name -> input_schema dict.

    mcp_server.TOOLS is the source of truth — it pairs each schema with its
    handler so the schema cannot drift from what the handler actually accepts
    without somebody noticing.
    """
    return {name: spec["input_schema"] for name, spec in mcp_server.TOOLS.items()}


def _shim_only_tools() -> set[str]:
    """Gardener-only tools that have no canonical counterpart.

    These dispatch to a Python shim inside memory_gardener (not via MCP), so
    canonical comparison is N/A — we validate against the shim signature in
    test_shim_signatures_match.
    """
    return {
        "mempalace_propose_edge_candidate",
        "mempalace_synthesize_operation_template",
    }


# Fields the gardener shim auto-injects on the way to the MCP handler. If a
# canonical-required field appears here, the gardener can omit it from its
# input_schema without breaking the call (the shim adds it before dispatch).
#
# Keep this list TIGHT — every entry is a place where Haiku does NOT see
# what the server demands, and we'd rather expose the field to Haiku than
# rely on a shim. Audit this when adding any new auto-injection.
_SHIM_AUTO_INJECTED: dict[str, set[str]] = {
    # kg_search canonical requires "agent"; gardener shim hard-codes agent.
    "mempalace_kg_search": {"agent"},
}


# ─────────────────────────────────────────────────────────────────────────
# Test 1: every gardener tool either has a canonical counterpart or is a shim
# ─────────────────────────────────────────────────────────────────────────


def test_every_gardener_tool_classified():
    """Every gardener-exposed tool must be either canonical-backed or a known shim.

    A new tool added to _TOOL_SCHEMAS must be either:
      (a) registered in mcp_server.TOOLS (canonical-backed), OR
      (b) added to _shim_only_tools() (shim-backed)
    Anything else is a structural mistake that should fail CI immediately.
    """
    gardener = set(_gardener_schemas())
    canonical = set(_canonical_schemas())
    shim_only = _shim_only_tools()
    unclassified = gardener - canonical - shim_only
    assert not unclassified, (
        f"gardener exposes tools that are neither canonical-backed nor "
        f"declared shim-only: {sorted(unclassified)}. Either add the tool "
        f"to mcp_server.TOOLS or add its name to _shim_only_tools()."
    )


# ─────────────────────────────────────────────────────────────────────────
# Test 2: every gardener-declared property has a matching canonical property
# of the same shape (for tools that have a canonical counterpart)
# ─────────────────────────────────────────────────────────────────────────


def _property_shape(prop: dict) -> tuple:
    """Reduce a property dict to a comparable shape tuple.

    For object properties, includes the nested `required` set so dict-shape
    drift (the kg_update_entity bite) is caught structurally.
    """
    t = prop.get("type")
    if t == "object":
        nested_required = tuple(sorted(prop.get("required") or []))
        nested_props = tuple(sorted((prop.get("properties") or {}).keys()))
        return ("object", nested_required, nested_props)
    if t == "array":
        item_type = (prop.get("items") or {}).get("type")
        return ("array", item_type)
    return (t,)


@pytest.mark.parametrize(
    "tool_name",
    sorted(set(_gardener_schemas()) & set(_canonical_schemas())),
)
def test_gardener_property_types_match_canonical(tool_name):
    """Every property the gardener declares must match the canonical type.

    This is the test that would have caught the kg_update_entity dict-vs-string
    bug at PR time. Missing canonical-required fields are a separate failure
    mode caught in test_gardener_required_subset_of_canonical.
    """
    g_schema = _gardener_schemas()[tool_name]
    c_schema = _canonical_schemas()[tool_name]

    g_props = g_schema.get("properties") or {}
    c_props = c_schema.get("properties") or {}

    drifts: list[str] = []
    for name, g_prop in g_props.items():
        if name not in c_props:
            drifts.append(f"  {name}: gardener declares but canonical has no such field")
            continue
        g_shape = _property_shape(g_prop)
        c_shape = _property_shape(c_props[name])
        if g_shape != c_shape:
            drifts.append(f"  {name}: gardener={g_shape!r} vs canonical={c_shape!r}")

    assert not drifts, (
        f"Schema drift on {tool_name}:\n"
        + "\n".join(drifts)
        + f"\nCanonical lives at mempalace.mcp_server.TOOLS[{tool_name!r}]; "
        f"gardener lives at mempalace.memory_gardener._TOOL_SCHEMAS."
    )


# ─────────────────────────────────────────────────────────────────────────
# Test 3: every canonical-required field is either gardener-required or
# auto-injected by a known shim
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "tool_name",
    sorted(set(_gardener_schemas()) & set(_canonical_schemas())),
)
def test_gardener_covers_canonical_required(tool_name):
    """Canonical-required fields must be either gardener-required or in the
    auto-inject allowlist. Otherwise Haiku omits them and the server rejects."""
    g_schema = _gardener_schemas()[tool_name]
    c_schema = _canonical_schemas()[tool_name]

    g_required = set(g_schema.get("required") or [])
    c_required = set(c_schema.get("required") or [])
    auto_injected = _SHIM_AUTO_INJECTED.get(tool_name, set())

    missing = c_required - g_required - auto_injected
    assert not missing, (
        f"Canonical schema for {tool_name} requires {sorted(missing)} "
        f"but gardener neither requires them nor auto-injects via shim. "
        f"Add to gardener input_schema.required, or add to "
        f"_SHIM_AUTO_INJECTED if a shim sets them before dispatch."
    )


# ─────────────────────────────────────────────────────────────────────────
# Test 4: shim-only tools' schemas match the shim Python signature
# ─────────────────────────────────────────────────────────────────────────


_SHIM_FUNCS = {
    "mempalace_propose_edge_candidate": "_propose_edge_candidate_shim",
    "mempalace_synthesize_operation_template": "_synthesize_operation_template_shim",
}


@pytest.mark.parametrize("tool_name", sorted(_shim_only_tools()))
def test_shim_signatures_match(tool_name):
    """Shim Python signature must accept every gardener-required field.

    For shim-only tools (no canonical counterpart), the only authority on what
    the gardener can call is the Python function the shim dispatches to. If
    the gardener schema lists a required field the shim doesn't accept, Haiku
    will pass it and the call will TypeError at runtime.
    """
    schema = _gardener_schemas()[tool_name]
    func_name = _SHIM_FUNCS[tool_name]
    func = getattr(memory_gardener, func_name, None)
    assert func is not None, (
        f"Shim {func_name} not found in memory_gardener — drift between "
        f"_SHIM_FUNCS allowlist and the module."
    )
    sig = inspect.signature(func)
    sig_params = set(sig.parameters)

    # Every property gardener declares must be a parameter the shim accepts
    # (or be **kwargs-absorbed). If the shim has **kwargs, we trust it.
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_kw:
        return  # **kwargs absorbs any field

    g_props = set((schema.get("properties") or {}).keys())
    unaccepted = g_props - sig_params
    assert not unaccepted, (
        f"{tool_name}: gardener schema declares fields the shim does not "
        f"accept: {sorted(unaccepted)}. Either fix the schema or update "
        f"the shim signature."
    )


# ─────────────────────────────────────────────────────────────────────────
# Test 5: explicit regression for the 2026-04-26 bite
# ─────────────────────────────────────────────────────────────────────────


def test_kg_update_entity_description_is_dict_not_string():
    """Regression for the dict-vs-string drift that broke generic_summary.

    On 2026-04-26 we discovered the gardener was telling Haiku to pass
    description as a string while the server validate_summary required a
    dict {what, why, scope?}. Many generic_summary flags deferred with
    'legacy string form is no longer accepted on writes'. Lock this fix
    so it can't silently regress.
    """
    schema = _gardener_schemas()["mempalace_kg_update_entity"]
    desc = (schema.get("properties") or {}).get("description")
    assert desc is not None, "kg_update_entity must declare a description field"
    assert desc.get("type") == "object", (
        f"kg_update_entity.description must be object (dict), got "
        f"{desc.get('type')!r}. Server demands dict {{what, why, scope?}}; "
        f"strings are rejected by validate_summary."
    )
    nested_required = set(desc.get("required") or [])
    assert {"what", "why"}.issubset(nested_required), (
        f"description object must require 'what' and 'why', got "
        f"required={sorted(nested_required)!r}"
    )
