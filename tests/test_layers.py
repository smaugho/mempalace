"""Tests for mempalace.layers — Layer0, Layer1, Layer2, MemoryStack.

Layer3 removed in kg_search (via scoring.multi_channel_search) IS the
real deep search. The old Layer3 was single-query cosine against records only.
"""

import os
from unittest.mock import MagicMock, patch

from mempalace.layers import Layer0, Layer1, Layer2, MemoryStack


# ── Layer0 — with identity file ─────────────────────────────────────────


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


# ── Layer1 — mocked chromadb ────────────────────────────────────────────


def _mock_chromadb_for_layer(docs, metas, monkeypatch=None):
    """Return a mock PersistentClient whose collection.get returns docs/metas."""
    mock_col = MagicMock()
    # First batch returns data, second batch returns empty (end of pagination)
    mock_col.get.side_effect = [
        {"documents": docs, "metadatas": metas},
        {"documents": [], "metadatas": []},
    ]
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_col
    return mock_client


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
        {"room": "decisions", "source_file": "meeting.txt", "importance": 5},
        {"room": "architecture", "source_file": "design.txt", "importance": 4},
    ]
    mock_client = _mock_chromadb_for_layer(docs, metas)

    with (
        patch("mempalace.layers.MempalaceConfig") as mock_cfg,
        patch("mempalace.layers.chromadb.PersistentClient", return_value=mock_client),
    ):
        mock_cfg.return_value.palace_path = "/fake"
        layer = Layer1(palace_path="/fake")
        result = layer.generate()

    assert "ESSENTIAL STORY" in result
    assert "project decisions" in result


def test_layer1_empty_palace():
    mock_col = MagicMock()
    mock_col.get.return_value = {"documents": [], "metadatas": []}
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_col

    with (
        patch("mempalace.layers.MempalaceConfig") as mock_cfg,
        patch("mempalace.layers.chromadb.PersistentClient", return_value=mock_client),
    ):
        mock_cfg.return_value.palace_path = "/fake"
        layer = Layer1(palace_path="/fake")
        result = layer.generate()

    assert "No memories" in result or "No entries" in result


def test_layer1_with_wing_filter():
    docs = ["Memory about project X"]
    metas = [{"room": "general", "source_file": "x.txt", "importance": 3}]
    mock_client = _mock_chromadb_for_layer(docs, metas)

    with (
        patch("mempalace.layers.MempalaceConfig") as mock_cfg,
        patch("mempalace.layers.chromadb.PersistentClient", return_value=mock_client),
    ):
        mock_cfg.return_value.palace_path = "/fake"
        layer = Layer1(palace_path="/fake", wing="project_x")
        result = layer.generate()

    assert "ESSENTIAL STORY" in result
    # Verify wing filter was passed
    call_kwargs = mock_client.get_collection.return_value.get.call_args_list[0][1]
    assert call_kwargs.get("where") == {"wing": "project_x"}


def test_layer1_truncates_long_snippets():
    docs = ["A" * 300]
    metas = [{"room": "general", "source_file": "long.txt"}]
    mock_client = _mock_chromadb_for_layer(docs, metas)

    with (
        patch("mempalace.layers.MempalaceConfig") as mock_cfg,
        patch("mempalace.layers.chromadb.PersistentClient", return_value=mock_client),
    ):
        mock_cfg.return_value.palace_path = "/fake"
        layer = Layer1(palace_path="/fake")
        result = layer.generate()

    assert "..." in result


def test_layer1_respects_max_chars():
    """L1 stops adding entries once MAX_CHARS is reached."""
    docs = [f"Memory number {i} with substantial content padding here" for i in range(30)]
    metas = [{"room": "general", "source_file": f"f{i}.txt", "importance": 5} for i in range(30)]
    mock_client = _mock_chromadb_for_layer(docs, metas)

    with (
        patch("mempalace.layers.MempalaceConfig") as mock_cfg,
        patch("mempalace.layers.chromadb.PersistentClient", return_value=mock_client),
    ):
        mock_cfg.return_value.palace_path = "/fake"
        layer = Layer1(palace_path="/fake")
        layer.MAX_CHARS = 200  # Very low cap to trigger truncation
        result = layer.generate()

    assert "more in L3 search" in result


def test_layer1_importance_from_various_keys():
    """Layer1 tries importance, emotional_weight, weight keys."""
    docs = ["mem1", "mem2", "mem3"]
    metas = [
        {"room": "r", "emotional_weight": 5},
        {"room": "r", "weight": 1},
        {"room": "r"},  # no weight key, defaults to 3
    ]
    mock_client = _mock_chromadb_for_layer(docs, metas)

    with (
        patch("mempalace.layers.MempalaceConfig") as mock_cfg,
        patch("mempalace.layers.chromadb.PersistentClient", return_value=mock_client),
    ):
        mock_cfg.return_value.palace_path = "/fake"
        layer = Layer1(palace_path="/fake")
        result = layer.generate()

    assert "ESSENTIAL STORY" in result


def test_layer1_batch_exception_breaks():
    """If col.get raises on a batch, loop breaks gracefully."""
    mock_col = MagicMock()
    mock_col.get.side_effect = [
        {"documents": ["doc1"], "metadatas": [{"room": "r"}]},
        RuntimeError("batch error"),
    ]
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_col

    with (
        patch("mempalace.layers.MempalaceConfig") as mock_cfg,
        patch("mempalace.layers.chromadb.PersistentClient", return_value=mock_client),
    ):
        mock_cfg.return_value.palace_path = "/fake"
        layer = Layer1(palace_path="/fake")
        result = layer.generate()

    assert "ESSENTIAL STORY" in result


# ── Layer2 — mocked chromadb ────────────────────────────────────────────


def test_layer2_no_palace():
    with patch("mempalace.layers.MempalaceConfig") as mock_cfg:
        mock_cfg.return_value.palace_path = "/nonexistent/palace"
        layer = Layer2(palace_path="/nonexistent/palace")
    result = layer.retrieve(wing="test")
    assert "No palace found" in result


def test_layer2_retrieve_with_wing():
    mock_col = MagicMock()
    mock_col.get.return_value = {
        "documents": ["Some memory about the project"],
        "metadatas": [{"room": "backend", "source_file": "notes.txt"}],
    }
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_col

    with (
        patch("mempalace.layers.MempalaceConfig") as mock_cfg,
        patch("mempalace.layers.chromadb.PersistentClient", return_value=mock_client),
    ):
        mock_cfg.return_value.palace_path = "/fake"
        layer = Layer2(palace_path="/fake")
        result = layer.retrieve(wing="project")

    assert "ON-DEMAND" in result
    assert "memory about the project" in result


def test_layer2_retrieve_with_room():
    mock_col = MagicMock()
    mock_col.get.return_value = {
        "documents": ["Backend architecture notes"],
        "metadatas": [{"room": "architecture", "source_file": "arch.txt"}],
    }
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_col

    with (
        patch("mempalace.layers.MempalaceConfig") as mock_cfg,
        patch("mempalace.layers.chromadb.PersistentClient", return_value=mock_client),
    ):
        mock_cfg.return_value.palace_path = "/fake"
        layer = Layer2(palace_path="/fake")
        result = layer.retrieve(room="architecture")

    assert "ON-DEMAND" in result


def test_layer2_retrieve_wing_and_room():
    mock_col = MagicMock()
    mock_col.get.return_value = {
        "documents": ["Filtered result"],
        "metadatas": [{"room": "backend", "source_file": "x.txt"}],
    }
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_col

    with (
        patch("mempalace.layers.MempalaceConfig") as mock_cfg,
        patch("mempalace.layers.chromadb.PersistentClient", return_value=mock_client),
    ):
        mock_cfg.return_value.palace_path = "/fake"
        layer = Layer2(palace_path="/fake")
        result = layer.retrieve(wing="proj", room="backend")

    assert "ON-DEMAND" in result
    call_kwargs = mock_col.get.call_args[1]
    assert "$and" in call_kwargs.get("where", {})


def test_layer2_retrieve_empty():
    mock_col = MagicMock()
    mock_col.get.return_value = {"documents": [], "metadatas": []}
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_col

    with (
        patch("mempalace.layers.MempalaceConfig") as mock_cfg,
        patch("mempalace.layers.chromadb.PersistentClient", return_value=mock_client),
    ):
        mock_cfg.return_value.palace_path = "/fake"
        layer = Layer2(palace_path="/fake")
        result = layer.retrieve(wing="missing")

    assert "No memories found" in result


def test_layer2_retrieve_no_filter():
    mock_col = MagicMock()
    mock_col.get.return_value = {"documents": [], "metadatas": []}
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_col

    with (
        patch("mempalace.layers.MempalaceConfig") as mock_cfg,
        patch("mempalace.layers.chromadb.PersistentClient", return_value=mock_client),
    ):
        mock_cfg.return_value.palace_path = "/fake"
        layer = Layer2(palace_path="/fake")
        layer.retrieve()

    # No where filter should be passed
    call_kwargs = mock_col.get.call_args[1]
    assert "where" not in call_kwargs


def test_layer2_retrieve_error():
    mock_col = MagicMock()
    mock_col.get.side_effect = RuntimeError("db error")
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_col

    with (
        patch("mempalace.layers.MempalaceConfig") as mock_cfg,
        patch("mempalace.layers.chromadb.PersistentClient", return_value=mock_client),
    ):
        mock_cfg.return_value.palace_path = "/fake"
        layer = Layer2(palace_path="/fake")
        result = layer.retrieve(wing="test")

    assert "Retrieval error" in result


def test_layer2_truncates_long_snippets():
    mock_col = MagicMock()
    mock_col.get.return_value = {
        "documents": ["B" * 400],
        "metadatas": [{"room": "r", "source_file": "s.txt"}],
    }
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_col

    with (
        patch("mempalace.layers.MempalaceConfig") as mock_cfg,
        patch("mempalace.layers.chromadb.PersistentClient", return_value=mock_client),
    ):
        mock_cfg.return_value.palace_path = "/fake"
        layer = Layer2(palace_path="/fake")
        result = layer.retrieve(wing="test")

    assert "..." in result


# Layer3 tests removed: Layer3 class deleted. kg_search IS the real
# deep search, via scoring.multi_channel_search against both collections.


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


def test_memory_stack_wake_up_with_wing(tmp_path):
    identity_file = tmp_path / "identity.txt"
    identity_file.write_text("I am Atlas.")

    with patch("mempalace.layers.MempalaceConfig") as mock_cfg:
        mock_cfg.return_value.palace_path = "/nonexistent"
        stack = MemoryStack(
            palace_path="/nonexistent",
            identity_path=str(identity_file),
        )
        result = stack.wake_up(wing="my_project")

    assert stack.l1.wing == "my_project"
    assert "Atlas" in result


def test_memory_stack_recall(tmp_path):
    identity_file = tmp_path / "identity.txt"
    identity_file.write_text("I am Atlas.")

    with patch("mempalace.layers.MempalaceConfig") as mock_cfg:
        mock_cfg.return_value.palace_path = "/nonexistent"
        stack = MemoryStack(
            palace_path="/nonexistent",
            identity_path=str(identity_file),
        )
        result = stack.recall(wing="test")

    assert "No palace found" in result


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
    assert result["total_drawers"] == 0
    assert "L0_identity" in result
    assert "L1_essential" in result
    assert "L2_on_demand" in result
    assert "L3_deep_search" in result


def test_memory_stack_status_with_palace(tmp_path):
    identity_file = tmp_path / "identity.txt"
    identity_file.write_text("I am Atlas.")

    mock_col = MagicMock()
    mock_col.count.return_value = 42
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_col

    with (
        patch("mempalace.layers.MempalaceConfig") as mock_cfg,
        patch("mempalace.layers.chromadb.PersistentClient", return_value=mock_client),
    ):
        mock_cfg.return_value.palace_path = "/fake"
        stack = MemoryStack(
            palace_path="/fake",
            identity_path=str(identity_file),
        )
        result = stack.status()

    assert result["total_drawers"] == 42
    assert result["L0_identity"]["exists"] is True
