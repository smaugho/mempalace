"""Tests for mempalace.layers -- Layer0, Layer1, MemoryStack.

Layer2 and Layer3 removed. Deep search is handled by kg_search
(via scoring.multi_channel_search).
"""

import pytest

import os
from unittest.mock import MagicMock, patch

from mempalace.layers import Layer0, Layer1, MemoryStack


# ── Layer0 -- with identity file ─────────────────────────────────────────


def test_layer0_reads_identity_file(tmp_path):
    identity_file = tmp_path / "identity.txt"
    identity_file.write_text("I am Atlas, a personal AI assistant for Alice.")
    layer = Layer0(identity_path=str(identity_file))
    text = layer.render()
    assert "Atlas" in text
    assert "Alice" in text


def test_layer0_caches_text(tmp_path):
    identity_file = tmp_path / "identity.txt"
    identity_file.write_text("Hello world")
    layer = Layer0(identity_path=str(identity_file))
    first = layer.render()
    identity_file.write_text("Changed content")
    second = layer.render()
    assert first == second
    assert second == "Hello world"


def test_layer0_missing_file_returns_default(tmp_path):
    missing = str(tmp_path / "nonexistent.txt")
    layer = Layer0(identity_path=missing)
    text = layer.render()
    assert "No identity configured" in text
    assert "identity.txt" in text


def test_layer0_token_estimate(tmp_path):
    identity_file = tmp_path / "identity.txt"
    content = "A" * 400
    identity_file.write_text(content)
    layer = Layer0(identity_path=str(identity_file))
    estimate = layer.token_estimate()
    assert estimate == 100


def test_layer0_token_estimate_empty(tmp_path):
    identity_file = tmp_path / "identity.txt"
    identity_file.write_text("")
    layer = Layer0(identity_path=str(identity_file))
    assert layer.token_estimate() == 0


def test_layer0_strips_whitespace(tmp_path):
    identity_file = tmp_path / "identity.txt"
    identity_file.write_text("  Hello world  \n\n")
    layer = Layer0(identity_path=str(identity_file))
    text = layer.render()
    assert text == "Hello world"


def test_layer0_default_path():
    layer = Layer0()
    expected = os.path.expanduser("~/.mempalace/identity.txt")
    assert layer.path == expected


# ── Layer1 -- mocked chromadb ────────────────────────────────────────────


def _mock_vs_for_layer(docs, metas, monkeypatch=None):
    """Return a mock VectorStore whose .get(...) returns a GetResult.

    Post-Tier-1 (commit 58b6509): layers.py routes through
    mempalace.vector_store.get_vector_store and calls vs.get(...) which
    returns GetResult dataclass with .ids/.documents/.metadatas attrs
    (was raw chromadb dict). The first batch returns data, the second
    returns empty to terminate pagination just like the original.
    """
    from mempalace.vector_store import GetResult

    mock_vs = MagicMock()
    mock_vs.get.side_effect = [
        GetResult(
            ids=[f"id{i}" for i in range(len(docs))],
            documents=list(docs),
            metadatas=list(metas),
        ),
        GetResult(ids=[]),
    ]
    return mock_vs


def test_layer1_no_palace():
    """Layer1 returns helpful message when no palace exists."""
    with patch("mempalace.layers.MempalaceConfig") as mock_cfg:
        mock_cfg.return_value.palace_path = "/nonexistent/palace"
        layer = Layer1(palace_path="/nonexistent/palace")
    result = layer.generate()
    assert "No palace found" in result or "No memories" in result or "No entries" in result


def test_layer1_generates_essential_story():
    docs = [
        "Important memory about project decisions",
        "Key architectural choice for the backend",
    ]
    metas = [
        {"content_type": "event", "source_file": "meeting.txt", "importance": 5},
        {"content_type": "fact", "source_file": "design.txt", "importance": 4},
    ]
    mock_vs = _mock_vs_for_layer(docs, metas)

    with (
        patch("mempalace.layers.MempalaceConfig") as mock_cfg,
        patch("mempalace.layers.get_vector_store", return_value=mock_vs),
    ):
        mock_cfg.return_value.palace_path = "/fake"
        layer = Layer1(palace_path="/fake")
        result = layer.generate()

    assert "ESSENTIAL STORY" in result
    assert "project decisions" in result


def test_layer1_empty_palace():
    from mempalace.vector_store import GetResult

    mock_vs = MagicMock()
    mock_vs.get.return_value = GetResult(ids=[])

    with (
        patch("mempalace.layers.MempalaceConfig") as mock_cfg,
        patch("mempalace.layers.get_vector_store", return_value=mock_vs),
    ):
        mock_cfg.return_value.palace_path = "/fake"
        layer = Layer1(palace_path="/fake")
        result = layer.generate()

    assert "No memories" in result or "No entries" in result


def test_layer1_with_agent_filter():
    docs = ["Memory about project X"]
    metas = [{"content_type": "fact", "source_file": "x.txt", "importance": 3}]
    mock_vs = _mock_vs_for_layer(docs, metas)

    with (
        patch("mempalace.layers.MempalaceConfig") as mock_cfg,
        patch("mempalace.layers.get_vector_store", return_value=mock_vs),
    ):
        mock_cfg.return_value.palace_path = "/fake"
        layer = Layer1(palace_path="/fake", agent="project_x")
        result = layer.generate()

    assert "ESSENTIAL STORY" in result
    # Verify agent filter was passed through vs.get(where=...)
    call_kwargs = mock_vs.get.call_args_list[0][1]
    assert call_kwargs.get("where") == {"added_by": "project_x"}


def test_layer1_truncates_long_snippets():
    docs = ["A" * 300]
    metas = [{"content_type": "fact", "source_file": "long.txt"}]
    mock_vs = _mock_vs_for_layer(docs, metas)

    with (
        patch("mempalace.layers.MempalaceConfig") as mock_cfg,
        patch("mempalace.layers.get_vector_store", return_value=mock_vs),
    ):
        mock_cfg.return_value.palace_path = "/fake"
        layer = Layer1(palace_path="/fake")
        result = layer.generate()

    assert "..." in result


def test_layer1_respects_max_chars():
    """L1 stops adding entries once MAX_CHARS is reached."""
    docs = [f"Memory number {i} with substantial content padding here" for i in range(30)]
    metas = [
        {"content_type": "fact", "source_file": f"f{i}.txt", "importance": 5} for i in range(30)
    ]
    mock_vs = _mock_vs_for_layer(docs, metas)

    with (
        patch("mempalace.layers.MempalaceConfig") as mock_cfg,
        patch("mempalace.layers.get_vector_store", return_value=mock_vs),
    ):
        mock_cfg.return_value.palace_path = "/fake"
        layer = Layer1(palace_path="/fake")
        layer.MAX_CHARS = 200  # Very low cap to trigger truncation
        result = layer.generate()

    assert "more in search" in result


def test_layer1_importance_from_various_keys():
    """Layer1 tries importance, emotional_weight, weight keys."""
    docs = ["mem1", "mem2", "mem3"]
    metas = [
        {"content_type": "fact", "emotional_weight": 5},
        {"content_type": "fact", "weight": 1},
        {"content_type": "fact"},  # no weight key, defaults to 3
    ]
    mock_vs = _mock_vs_for_layer(docs, metas)

    with (
        patch("mempalace.layers.MempalaceConfig") as mock_cfg,
        patch("mempalace.layers.get_vector_store", return_value=mock_vs),
    ):
        mock_cfg.return_value.palace_path = "/fake"
        layer = Layer1(palace_path="/fake")
        result = layer.generate()

    assert "ESSENTIAL STORY" in result


def test_layer1_batch_exception_breaks():
    """If vs.get raises on a batch, loop breaks gracefully."""
    from mempalace.vector_store import GetResult

    mock_vs = MagicMock()
    mock_vs.get.side_effect = [
        GetResult(ids=["id0"], documents=["doc1"], metadatas=[{"content_type": "fact"}]),
        RuntimeError("batch error"),
    ]

    with (
        patch("mempalace.layers.MempalaceConfig") as mock_cfg,
        patch("mempalace.layers.get_vector_store", return_value=mock_vs),
    ):
        mock_cfg.return_value.palace_path = "/fake"
        layer = Layer1(palace_path="/fake")
        result = layer.generate()

    assert "ESSENTIAL STORY" in result


# Layer2 and Layer3 tests removed: both classes deleted.
# Deep search is handled by kg_search (scoring.multi_channel_search).


# ── MemoryStack ─────────────────────────────────────────────────────────


def test_memory_stack_wake_up(tmp_path):
    identity_file = tmp_path / "identity.txt"
    identity_file.write_text("I am Atlas.")

    with patch("mempalace.layers.MempalaceConfig") as mock_cfg:
        mock_cfg.return_value.palace_path = "/nonexistent"
        stack = MemoryStack(
            palace_path="/nonexistent",
            identity_path=str(identity_file),
        )
        result = stack.wake_up()

    assert "Atlas" in result
    # L1 will say no palace found
    assert "No palace" in result or "No memories" in result or "No entries" in result


def test_memory_stack_wake_up_with_agent(tmp_path):
    identity_file = tmp_path / "identity.txt"
    identity_file.write_text("I am Atlas.")

    with patch("mempalace.layers.MempalaceConfig") as mock_cfg:
        mock_cfg.return_value.palace_path = "/nonexistent"
        stack = MemoryStack(
            palace_path="/nonexistent",
            identity_path=str(identity_file),
        )
        result = stack.wake_up(agent="my_agent")

    assert stack.l1.agent == "my_agent"
    assert "Atlas" in result


def test_memory_stack_search_returns_removed_message(tmp_path):
    """stack.search delegates to _Layer3Removed stub."""
    identity_file = tmp_path / "identity.txt"
    identity_file.write_text("I am Atlas.")

    with patch("mempalace.layers.MempalaceConfig") as mock_cfg:
        mock_cfg.return_value.palace_path = "/nonexistent"
        stack = MemoryStack(
            palace_path="/nonexistent",
            identity_path=str(identity_file),
        )
        result = stack.search("test query")

    assert "removed" in result.lower() or "kg_search" in result


def test_memory_stack_status(tmp_path):
    identity_file = tmp_path / "identity.txt"
    identity_file.write_text("I am Atlas.")

    with patch("mempalace.layers.MempalaceConfig") as mock_cfg:
        mock_cfg.return_value.palace_path = "/nonexistent"
        stack = MemoryStack(
            palace_path="/nonexistent",
            identity_path=str(identity_file),
        )
        result = stack.status()

    assert result["palace_path"] == "/nonexistent"
    assert result["total_records"] == 0
    assert "L0_identity" in result
    assert "L1_essential" in result


def test_memory_stack_status_with_palace(tmp_path):
    identity_file = tmp_path / "identity.txt"
    identity_file.write_text("I am Atlas.")

    mock_vs = MagicMock()
    mock_vs.count.return_value = 42

    with (
        patch("mempalace.layers.MempalaceConfig") as mock_cfg,
        patch("mempalace.layers.get_vector_store", return_value=mock_vs),
    ):
        mock_cfg.return_value.palace_path = "/fake"
        stack = MemoryStack(
            palace_path="/fake",
            identity_path=str(identity_file),
        )
        result = stack.status()

    assert result["total_records"] == 42
    assert result["L0_identity"]["exists"] is True


# ─────────────────────────────────────────────────────────────────────
# FINDING #B + #D regression (v3.7.24, 2026-05-18): wake_up rendered
# "## L0 -- IDENTITY\nNo identity configured" + "## L1 -- No entries
# yet." on every populated palace for ~6 weeks. Two latent bugs masked
# by silent excepts:
#
#   FINDING #B (L1 empty): Layer1.generate passed ``offset=offset`` to
#     ``vs.get`` after the chromadb -> sqlite_vec swap; sqlite-vec
#     get() did not accept offset; TypeError silently swallowed by
#     ``except Exception: pass`` -- L1 was empty for everyone.
#
#   FINDING #D (L0 empty): Layer0._load_from_kg looked up
#     ``entity['description']`` (key did not exist; entities table
#     stores 'content') AND queried vs.get with the canonical KG id
#     instead of the per-view-suffixed ``__v0`` storage id; both
#     KeyError + missing-id swallowed by the outer bare except --
#     "No identity configured" on every agent w/ described_by edges.
#
# These tests exercise the FULL path against a real SqliteVecVectorStore
# + real KnowledgeGraph so any future signature drift surfaces in CI.
# ─────────────────────────────────────────────────────────────────────


def _seed_palace_with_agent_and_records(palace_path):
    """Seed a fresh palace with an agent entity, described_by edges,
    and matching records stored with ``__v0`` view suffix.

    Returns (kg, vs, agent_id).
    """
    from mempalace.knowledge_graph import KnowledgeGraph
    from mempalace.sqlite_vec_store import SqliteVecVectorStore
    from mempalace.vector_store import RECORDS_COLLECTION
    from mempalace.embedder import get_default_embedder

    os.makedirs(palace_path, exist_ok=True)
    kg = KnowledgeGraph(db_path=os.path.join(palace_path, "knowledge_graph.sqlite3"))
    vs = SqliteVecVectorStore(palace_path)

    agent_id = "ga_agent_test"
    kg.add_entity(
        agent_id,
        kind="entity",
        importance=4,
        content="ga_agent_test description: identity body lives here.",
    )

    # Three described_by memories; store under the __v0 suffix the
    # multi-view embedder uses for the primary content view.
    embedder = get_default_embedder()
    rec_ids = [
        "record_ga_agent_test_first_rule",
        "record_ga_agent_test_second_rule",
        "record_ga_agent_test_third_rule",
    ]
    docs = [
        "FIRST_RULE_BODY -- a key directive that drives agent behaviour.",
        "SECOND_RULE_BODY -- a sibling directive complementing the first.",
        "THIRD_RULE_BODY -- the third leg of the identity tripod.",
    ]
    for rid, doc in zip(rec_ids, docs):
        emb = embedder([doc])[0]  # Embedder is callable: embedder(texts)
        # Declare the record entity FIRST (cold-start gate 2026-05-01:
        # add_triple rejects phantom subject/object ids).
        kg.add_entity(
            rid,
            kind="entity",
            importance=4,
            content=doc,
        )
        vs.add(
            RECORDS_COLLECTION,
            ids=[rid + "__v0"],
            documents=[doc],
            metadatas=[
                {
                    "kind": "memory",
                    "added_by": agent_id,
                    "importance": 4,
                    "view_index": 0,
                }
            ],
            embeddings=[emb],
        )
        kg.add_triple(agent_id, "described_by", rid, statement="test")

    return kg, vs, agent_id


def test_layer0_renders_kg_identity_with_view_suffix_lookup(tmp_path):
    """FINDING #D regression: L0 must find described_by records when
    KG stores canonical id ``record_*`` and vec store keys them as
    ``record_*__v0`` (multi-view storage suffix)."""
    palace = str(tmp_path / "palace")
    _, _, agent_id = _seed_palace_with_agent_and_records(palace)

    from mempalace.layers import Layer0

    l0 = Layer0(
        identity_path=str(tmp_path / "missing.txt"),
        palace_path=palace,
        agent=agent_id,
    )
    text = l0.render()
    assert "No identity configured" not in text, (
        "L0 fell through to default text; __v0 lookup or description fallback regressed"
    )
    assert agent_id in text, "L0 should name the agent entity"
    assert "FIRST_RULE_BODY" in text or "SECOND_RULE_BODY" in text or "THIRD_RULE_BODY" in text, (
        "L0 should embed described_by record bodies"
    )


def test_layer1_paginates_via_offset_kwarg(tmp_path):
    """FINDING #B regression: Layer1.generate must paginate via
    ``vs.get(offset=...)`` without TypeError; with seeded records the
    output must NOT be ``## L1 -- No entries yet.``."""
    palace = str(tmp_path / "palace")
    _, _, agent_id = _seed_palace_with_agent_and_records(palace)

    from mempalace.layers import Layer1

    l1 = Layer1(palace_path=palace, agent=agent_id)
    out = l1.generate()
    assert out != "## L1 -- No entries yet.", (
        "Layer1 returned the empty sentinel on a seeded palace -- "
        "offset kwarg or silent-except regression"
    )
    assert "## L1 -- ESSENTIAL STORY" in out
    assert "FIRST_RULE_BODY" in out or "SECOND_RULE_BODY" in out or "THIRD_RULE_BODY" in out


def test_memory_stack_wake_up_end_to_end_against_real_palace(tmp_path):
    """FINDING #B + #D combined: full MemoryStack.wake_up against a
    real sqlite_vec palace must render both L0 + L1 with content."""
    palace = str(tmp_path / "palace")
    _, _, agent_id = _seed_palace_with_agent_and_records(palace)

    from mempalace.layers import MemoryStack

    stack = MemoryStack(
        palace_path=palace,
        identity_path=str(tmp_path / "missing.txt"),
    )
    text = stack.wake_up(agent=agent_id)
    assert "No identity configured" not in text
    assert "No entries yet" not in text
    assert "L0 -- IDENTITY" in text
    assert "L1 -- ESSENTIAL STORY" in text


pytestmark = pytest.mark.integration
