"""v3.10.10 regression: apply_gate must skip graph-glue kinds BEFORE
building the GateItem list AND drop generic_summary flags whose
target is kind=record.

Background
----------
Adrian (msg_6f0496_5, 2026-05-27) noted that v3.10.2 (commit ba8524a)
added kind='context' to three DOWNSTREAM projection filters
(intent.py:2798 rerank, intent.py:5294 declare_user_intents,
tool_read.py:572 kg_search top-projection) -- but those run AFTER the
InjectionGate Haiku rater. apply_gate (injection_gate.py:1613-1820)
iterates input ``memories`` with zero kind-filter and pays Haiku to
rate every item, including contexts/classes/predicates/operations/
literals/user_messages/state_schemas. Corpus before v3.10.10:
8,013 of 11,930 generic_summary flags ever (67%) targeted kinds the
gardener cannot or should not rewrite.

v3.10.10 closes the leak at the source:
  * ``_GATE_SKIP_KINDS`` frozenset (graph-glue / metadata kinds) is
    consulted in the items-build loop; matching kinds are dropped
    BEFORE construction of the GateItem.
  * After Haiku returns, generic_summary flags whose target is
    kind=record are filtered out before the record_memory_flags call.
    Records are kept in the gate input pool because other flag kinds
    (duplicate_pair, stale, unlinked_entity) on records are still
    valid -- only the structurally-unfixable generic_summary on records
    is dropped (tool_mutate.py:1747-1755 rejects in-place summary
    updates on records).

Mirrors the source-pinning regression style of
``test_context_filter.py`` + ``test_user_message_filter.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_GATE_SRC = Path(__file__).parent.parent.parent / "mempalace" / "injection_gate.py"


def _src() -> str:
    return _GATE_SRC.read_text(encoding="utf-8")


def test_gate_skip_kinds_constant_exists_with_expected_members():
    """``_GATE_SKIP_KINDS`` MUST be defined and contain every graph-glue
    kind the v3.10.10 audit identified. Dropping a member here lets the
    rater pay Haiku tokens for that kind again."""
    content = _src()
    assert "_GATE_SKIP_KINDS = frozenset({" in content, (
        "v3.10.10: injection_gate.py must define _GATE_SKIP_KINDS "
        "as a frozenset of graph-glue kinds to skip at apply_gate input."
    )
    # Each member MUST appear; their order in the frozenset literal
    # is irrelevant but presence is mandatory.
    for kind in (
        '"context"',
        '"user_message"',
        '"class"',
        '"predicate"',
        '"operation"',
        '"state_schema"',
        '"literal"',
    ):
        assert kind in content, (
            f"v3.10.10: _GATE_SKIP_KINDS must include {kind}. "
            "Dropping it leaks that kind to the Haiku rater."
        )


def test_apply_gate_consults_skip_kinds_before_items_build():
    """apply_gate MUST short-circuit (``continue``) when the input
    memory's raw_meta.kind is in ``_GATE_SKIP_KINDS``. The check has to
    happen INSIDE the items-build loop and BEFORE ``items.append`` so
    Haiku never sees the item."""
    content = _src()
    target = "if _raw_kind in _GATE_SKIP_KINDS:"
    assert target in content, (
        "v3.10.10: apply_gate items-build loop must check "
        "_raw_kind against _GATE_SKIP_KINDS and continue before "
        "appending a GateItem. Without this the rater still pays "
        "Haiku for graph-glue kinds."
    )


def test_apply_gate_builds_id_to_kind_map_for_record_post_filter():
    """apply_gate MUST populate an ``id_to_kind`` map alongside the
    items list so the post-Haiku flag-emit step can drop
    generic_summary flags whose target is kind=record."""
    content = _src()
    assert "id_to_kind: dict[str, str] = {}" in content, (
        "v3.10.10: apply_gate must build id_to_kind map for the record-generic_summary post-filter."
    )
    assert "id_to_kind[str(mid)] = _raw_kind" in content, (
        "v3.10.10: id_to_kind must be populated per surviving item."
    )


def test_apply_gate_drops_generic_summary_flags_on_records():
    """The flag-emit step MUST filter out generic_summary flags whose
    target memory_ids include a kind=record. tool_mutate.py rejects
    in-place summary rewrites on records, so emitting these flags is
    pure wasted gardener spend."""
    content = _src()
    # The filter comprehension references id_to_kind + record kind.
    assert 'f.get("kind") == "generic_summary"' in content, (
        "v3.10.10: apply_gate flag-emit must check flag kind == "
        "generic_summary in the record-target filter."
    )
    assert 'id_to_kind.get(str(_mid)) == "record"' in content, (
        "v3.10.10: apply_gate flag-emit must skip generic_summary "
        "flags whose target id maps to kind=record via id_to_kind."
    )
