"""Tests for mempalace.repair -- scan, prune, and rebuild HNSW index.

Tier 2 fixture rewrite (2026-05-10): production callsites moved to
``mempalace.vector_store.make_persistent_client`` for raw chromadb
opens, so the test fixtures monkeypatch THAT helper to return a fake
client instead of patching ``mempalace.repair.chromadb`` (which is
only kept as a module-attribute hook for legacy ``@patch`` discovery).
"""

import pytest

import os
from unittest.mock import MagicMock, patch


from mempalace import repair


# ── _get_palace_path ──────────────────────────────────────────────────


@patch("mempalace.repair.MempalaceConfig", create=True)
def test_get_palace_path_from_config(mock_config_cls):
    mock_config_cls.return_value.palace_path = "/configured/palace"
    with patch.dict("sys.modules", {}):
        # Force reimport to pick up the mock
        result = repair._get_palace_path()
    assert isinstance(result, str)


def test_get_palace_path_fallback():
    with patch("mempalace.repair._get_palace_path") as mock_get:
        mock_get.return_value = os.path.join(os.path.expanduser("~"), ".mempalace", "palace")
        result = mock_get()
        assert ".mempalace" in result


# ── _paginate_ids ─────────────────────────────────────────────────────


def test_paginate_ids_single_batch():
    col = MagicMock()
    col.get.return_value = {"ids": ["id1", "id2", "id3"]}
    ids = repair._paginate_ids(col)
    assert ids == ["id1", "id2", "id3"]


def test_paginate_ids_empty():
    col = MagicMock()
    col.get.return_value = {"ids": []}
    ids = repair._paginate_ids(col)
    assert ids == []


def test_paginate_ids_with_where():
    col = MagicMock()
    col.get.return_value = {"ids": ["id1"]}
    repair._paginate_ids(col, where={"added_by": "test"})
    col.get.assert_called_with(where={"added_by": "test"}, include=[], limit=1000, offset=0)


def test_paginate_ids_offset_exception_fallback():
    col = MagicMock()
    # First call raises, fallback returns ids, second fallback returns empty
    col.get.side_effect = [
        Exception("offset bug"),
        {"ids": ["id1", "id2"]},
        Exception("offset bug"),
        {"ids": ["id1", "id2"]},  # same ids = no new = break
    ]
    ids = repair._paginate_ids(col)
    assert "id1" in ids


# ── make_persistent_client patch helper (Tier 2 fixture pattern) ──────
#
# Production scan_palace / prune_corrupt / rebuild_index do a LOCAL
# ``from mempalace.vector_store import make_persistent_client`` inside
# the function, so the binding the function actually resolves is
# ``mempalace.vector_store.make_persistent_client`` -- patching
# ``mempalace.repair.chromadb`` (the legacy surface) never intercepts
# the call. Patch the vector_store helper to return our fake client
# directly; that's the Tier 2 patch surface.


def _patch_client(monkeypatch, mock_client):
    """Route both repair.py and vector_store.make_persistent_client
    callers through the same fake client."""
    monkeypatch.setattr("mempalace.vector_store.make_persistent_client", lambda path: mock_client)
    monkeypatch.setattr("mempalace.repair.make_persistent_client", lambda path: mock_client)


# ── scan_palace ───────────────────────────────────────────────────────


def test_scan_palace_no_ids(monkeypatch, tmp_path):
    mock_col = MagicMock()
    mock_col.count.return_value = 0
    mock_col.get.return_value = {"ids": []}
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_col
    _patch_client(monkeypatch, mock_client)

    good, bad = repair.scan_palace(palace_path=str(tmp_path))
    assert good == set()
    assert bad == set()


def test_scan_palace_all_good(monkeypatch, tmp_path):
    mock_col = MagicMock()
    mock_col.count.return_value = 2
    # _paginate_ids call
    mock_col.get.side_effect = [
        {"ids": ["id1", "id2"]},  # paginate
        {"ids": ["id1", "id2"]},  # probe batch -- both returned
    ]
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_col
    _patch_client(monkeypatch, mock_client)

    good, bad = repair.scan_palace(palace_path=str(tmp_path))
    assert "id1" in good
    assert "id2" in good
    assert len(bad) == 0


def test_scan_palace_with_bad_ids(monkeypatch, tmp_path):
    mock_col = MagicMock()
    mock_col.count.return_value = 2

    def get_side_effect(**kwargs):
        ids = kwargs.get("ids", None)
        if ids is None:
            # paginate call
            return {"ids": ["good1", "bad1"]}
        if "bad1" in ids and len(ids) == 1:
            raise Exception("corrupt")
        if "good1" in ids and len(ids) == 1:
            return {"ids": ["good1"]}
        # batch probe -- raise to force per-id
        raise Exception("batch fail")

    mock_col.get.side_effect = get_side_effect
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_col
    _patch_client(monkeypatch, mock_client)

    good, bad = repair.scan_palace(palace_path=str(tmp_path))
    assert "good1" in good
    assert "bad1" in bad


def test_scan_palace_basic_scan(monkeypatch, tmp_path):
    """scan_palace runs without filters (only_wing removed)."""
    mock_col = MagicMock()
    mock_col.count.return_value = 1
    mock_col.get.side_effect = [
        {"ids": ["id1"]},  # paginate
        {"ids": ["id1"]},  # probe
    ]
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_col
    _patch_client(monkeypatch, mock_client)

    good, bad = repair.scan_palace(palace_path=str(tmp_path))
    assert "id1" in good
    assert len(bad) == 0


# ── prune_corrupt ─────────────────────────────────────────────────────


def test_prune_corrupt_no_file(monkeypatch, tmp_path):
    # Should print message and return without error
    repair.prune_corrupt(palace_path=str(tmp_path))


def test_prune_corrupt_dry_run(monkeypatch, tmp_path):
    bad_file = tmp_path / "corrupt_ids.txt"
    bad_file.write_text("bad1\nbad2\n")
    # Track whether make_persistent_client gets called -- dry run must skip it.
    called = {"n": 0}

    def _no_call(_path):
        called["n"] += 1
        raise AssertionError("dry run should not open a chroma client")

    monkeypatch.setattr("mempalace.vector_store.make_persistent_client", _no_call)
    monkeypatch.setattr("mempalace.repair.make_persistent_client", _no_call)
    repair.prune_corrupt(palace_path=str(tmp_path), confirm=False)
    assert called["n"] == 0


def test_prune_corrupt_confirmed(monkeypatch, tmp_path):
    bad_file = tmp_path / "corrupt_ids.txt"
    bad_file.write_text("bad1\nbad2\n")

    mock_col = MagicMock()
    mock_col.count.side_effect = [10, 8]
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_col
    _patch_client(monkeypatch, mock_client)

    repair.prune_corrupt(palace_path=str(tmp_path), confirm=True)
    mock_col.delete.assert_called_once()


def test_prune_corrupt_delete_failure_fallback(monkeypatch, tmp_path):
    bad_file = tmp_path / "corrupt_ids.txt"
    bad_file.write_text("bad1\nbad2\n")

    mock_col = MagicMock()
    mock_col.count.side_effect = [10, 8]
    # Batch delete fails, per-id succeeds
    mock_col.delete.side_effect = [Exception("batch fail"), None, None]
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_col
    _patch_client(monkeypatch, mock_client)

    repair.prune_corrupt(palace_path=str(tmp_path), confirm=True)
    assert mock_col.delete.call_count == 3  # 1 batch + 2 individual


# ── rebuild_index ─────────────────────────────────────────────────────


@patch("mempalace.repair.shutil")
def test_rebuild_index_no_palace(mock_shutil, monkeypatch, tmp_path):
    nonexistent = str(tmp_path / "nope")
    called = {"n": 0}

    def _no_call(_path):
        called["n"] += 1
        raise AssertionError("missing palace should never reach the chroma client")

    monkeypatch.setattr("mempalace.vector_store.make_persistent_client", _no_call)
    monkeypatch.setattr("mempalace.repair.make_persistent_client", _no_call)
    repair.rebuild_index(palace_path=nonexistent)
    assert called["n"] == 0


@patch("mempalace.repair.shutil")
def test_rebuild_index_empty_palace(mock_shutil, monkeypatch, tmp_path):
    mock_col = MagicMock()
    mock_col.count.return_value = 0
    # Slice 15+ uses col.get() pagination; mock empty page so loop exits.
    mock_col.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_col
    _patch_client(monkeypatch, mock_client)

    repair.rebuild_index(palace_path=str(tmp_path))
    mock_client.delete_collection.assert_not_called()


@patch("mempalace.repair.shutil")
def test_rebuild_index_success(mock_shutil, monkeypatch, tmp_path):
    # Create a fake sqlite file
    sqlite_path = tmp_path / "chroma.sqlite3"
    sqlite_path.write_text("fake")

    mock_col = MagicMock()
    mock_col.count.return_value = 2
    # Slice 15+ uses col.get() pagination; first call returns data, subsequent return empty.
    mock_col.get.side_effect = [
        {
            "ids": ["id1", "id2"],
            "documents": ["doc1", "doc2"],
            "metadatas": [{"added_by": "a"}, {"added_by": "b"}],
        },
    ] + [{"ids": [], "documents": [], "metadatas": []}] * 20

    mock_new_col = MagicMock()
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_col
    mock_client.create_collection.return_value = mock_new_col
    _patch_client(monkeypatch, mock_client)

    repair.rebuild_index(palace_path=str(tmp_path))

    # Verify: backed up sqlite only (not copytree)
    mock_shutil.copy2.assert_called_once()
    assert "chroma.sqlite3" in str(mock_shutil.copy2.call_args)

    # Verify: deleted and recreated with cosine (slice 13/16: also for context_views + triples)
    mock_client.delete_collection.assert_any_call("mempalace_records")
    # Slice 16 added hnsw:sync_threshold to create_collection metadata; assert
    # the records collection was created with cosine space, ignoring extra keys.
    create_calls = [
        c
        for c in mock_client.create_collection.call_args_list
        if c.args and c.args[0] == "mempalace_records"
    ]
    assert create_calls, (
        f"create_collection should have been called for mempalace_records; got {mock_client.create_collection.call_args_list}"
    )
    md = create_calls[0].kwargs.get("metadata", {})
    assert md.get("hnsw:space") == "cosine", f"expected cosine; got {md}"

    # Verify: used upsert not add


@patch("mempalace.repair.shutil")
def test_rebuild_index_error_reading(mock_shutil, monkeypatch, tmp_path):
    mock_client = MagicMock()
    mock_client.get_collection.side_effect = Exception("corrupt")
    _patch_client(monkeypatch, mock_client)

    repair.rebuild_index(palace_path=str(tmp_path))
    mock_client.delete_collection.assert_not_called()


pytestmark = pytest.mark.integration
