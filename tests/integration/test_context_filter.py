"""v3.10.2 regression: kind='context' must not surface as a memory
through retrieval projection.

Background
----------
Adrian (msg_c96c8a_182, 2026-05-21) flagged that kind='context'
entities were surfacing in retrieval results -- e.g. ctx_10141 with a
bare ``summary_text`` of ``"workflows_temporal_ts file and its role in
temporal scheduling"`` (a queries[0] string, no why).

A context is a GROUPING node: it groups memories via ``created_under``
edges and is embedded only so MaxSim can reuse/link it. It is not a
memory. Contexts reached the projection loops as cosine / Channel-B
hits and leaked into the agent-facing memories list -- the same
graph-glue-leak bug class as the class/predicate skip and the v3.7.43
FINDING #AA user_message skip.

v3.10.2 adds 'context' to the kind-skip filter at all three projection
sites:
  - intent.py:2791  (declare_intent / declare_operation rerank loop)
  - intent.py:5365  (declare_intent / declare_user_intents projection)
  - tool_read.py     (kg_search defensive filter)

Note: kg_query (exact entity-ID lookup) intentionally does NOT filter --
if you ask for a context by id you should get it back.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).parent.parent.parent / "mempalace"


def test_intent_rerank_filter_skips_context():
    """The declare_intent / declare_operation rerank skip tuple MUST
    include 'context'. Source: intent.py:2791."""
    content = (_ROOT / "intent.py").read_text(encoding="utf-8")
    target = '_r_kind in ("class", "predicate", "user_message", "context"):'
    assert target in content, (
        "v3.10.2: intent.py rerank loop must skip kind=context alongside "
        "class/predicate/user_message. Dropping it leaks context grouping "
        "nodes (with bare queries[0] summary_text) into the memories list."
    )


def test_declare_user_intents_projection_skips_context():
    """The declare_intent / declare_user_intents projection loop MUST
    skip 'context' too. Source: intent.py:5365."""
    content = (_ROOT / "intent.py").read_text(encoding="utf-8")
    target = 'if _h_kind in ("user_message", "context"):'
    assert target in content, (
        "v3.10.2: declare_user_intents/_run_local_retrieval projection "
        "must skip kind=context. Same leak class as the rerank filter."
    )


def test_kg_search_filter_skips_context():
    """The kg_search defensive filter MUST exclude 'context'.
    Source: tool_read.py."""
    content = (_ROOT / "tool_read.py").read_text(encoding="utf-8")
    target = 'not in ("user_message", "context")'
    assert target in content, (
        "v3.10.2: kg_search top-projection filter must strip kind=context "
        "alongside user_message. Contexts are grouping nodes, not memories."
    )
