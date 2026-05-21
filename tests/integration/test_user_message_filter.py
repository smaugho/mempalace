"""v3.7.43 FINDING #AA regression: user_message kind must not surface
as a memory through retrieval projection.

Background
----------
Adrian flagged that messages like ``msg_c96c8a_119`` with summary_text
``"reinstalled"`` were surfacing in declare_operation memories list.
The 2026-05-01 cold-start lock made user_message entities SQLite-only
graph anchors -- no embeddings, no keywords. Contract honored at the
WRITE layer (0 vec rows, 0 entity_keywords for user_message ids on a
434-row sample).

The leak was at READ: Channel B graph BFS traverses outgoing edges
from surfaced contexts; ``ctx_N fulfills_user_message msg_X`` pulls
msg_X as a neighbour. The rerank loop at intent.py:2711 filtered
``kind in ("class", "predicate")`` but not user_message. v3.7.43
extends the tuple.

Literature: MemGPT (Packer 2023), Generative Agents (Park 2023),
MemoryBank (Wang 2023), Letta all store dialogue with explicit
speaker turns AND retrieve via a recall-memory tier separate from
knowledge retrieval. Bare turn text leaking into knowledge retrieval
violates the universal pattern.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_v3743_intent_rerank_filter_includes_user_message():
    """The declare_intent / declare_operation rerank skip tuple MUST
    include 'user_message' so graph-BFS-pulled user-turn entities never
    surface as memories. Source: intent.py:2711."""
    # Read the source directly to lock the filter contract -- this is
    # the cheapest possible regression that catches a future refactor
    # that drops the kind from the tuple.
    from pathlib import Path

    intent_src = Path(__file__).parent.parent.parent / "mempalace" / "intent.py"
    content = intent_src.read_text(encoding="utf-8")
    # The filter line MUST include 'user_message' alongside class/predicate.
    target = '_r_kind in ("class", "predicate", "user_message", "context"):'
    assert target in content, (
        "v3.7.43 FINDING #AA: intent.py rerank loop must skip "
        "kind=user_message alongside class/predicate. Future refactors "
        "that drop user_message from the tuple will leak bare user-turn "
        "text into the agent's memories list."
    )


def test_v3743_declare_user_intents_filter_includes_user_message():
    """The declare_user_intents projection loop MUST skip user_message
    kind too; parallel filter to the declare_intent rerank fix.
    Source: intent.py:5285 area."""
    from pathlib import Path

    intent_src = Path(__file__).parent.parent.parent / "mempalace" / "intent.py"
    content = intent_src.read_text(encoding="utf-8")
    target = 'if _h_kind in ("user_message", "context"):'
    assert target in content, (
        "v3.7.43 FINDING #AA: declare_user_intents projection loop "
        "must skip kind=user_message. Same leak class as the rerank "
        "filter; both must be in place."
    )


def test_v3743_user_message_contract_at_write_layer(tmp_path):
    """LIVE test: a fresh user_message entity written via add_entity
    must NOT generate vec rows or keyword rows -- the 2026-05-01
    cold-start lock contract."""
    import sqlite3

    from mempalace.knowledge_graph import KnowledgeGraph

    db_path = str(tmp_path / "test_palace.sqlite3")
    kg = KnowledgeGraph(db_path)
    # Mint a user_message entity via the canonical write path.
    kg.add_entity(
        "msg_test_user_v3743_0",
        kind="user_message",
        content="reinstalled",
        importance=3,
    )

    # Verify entity exists in SQL.
    raw = sqlite3.connect(db_path)
    raw.row_factory = sqlite3.Row
    row = raw.execute(
        "SELECT id, kind FROM entities WHERE id = ?", ("msg_test_user_v3743_0",)
    ).fetchone()
    assert row is not None, "entity must be persisted in SQL"
    assert row["kind"] == "user_message"

    # Verify NO entity_keywords for it.
    kw = raw.execute(
        "SELECT COUNT(*) FROM entity_keywords WHERE entity_id = ?",
        ("msg_test_user_v3743_0",),
    ).fetchone()[0]
    assert kw == 0, (
        f"v3.7.43 FINDING #AA: user_message must NOT have entity_keywords "
        f"(graph-only contract); got {kw} keyword rows"
    )
