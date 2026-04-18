"""Tests for mempalace.cli — the main CLI dispatcher."""

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mempalace.cli import (
    cmd_hook,
    cmd_init,
    cmd_instructions,
    cmd_mine,
    cmd_repair,
    cmd_search,
    cmd_split,
    cmd_status,
    cmd_wakeup,
    main,
)


# ── cmd_status ─────────────────────────────────────────────────────────


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_status_default_palace(mock_config_cls):
    mock_config_cls.return_value.palace_path = "/fake/palace"
    args = argparse.Namespace(palace=None)
    mock_miner = MagicMock()
    with patch.dict("sys.modules", {"mempalace.miner": mock_miner}):
        cmd_status(args)
        mock_miner.status.assert_called_once_with(palace_path="/fake/palace")


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_status_custom_palace(mock_config_cls):
    args = argparse.Namespace(palace="~/my_palace")
    mock_miner = MagicMock()
    with patch.dict("sys.modules", {"mempalace.miner": mock_miner}):
        cmd_status(args)
        import os

        expected = os.path.expanduser("~/my_palace")
        mock_miner.status.assert_called_once_with(palace_path=expected)


# ── cmd_search ─────────────────────────────────────────────────────────


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_search_calls_search(mock_config_cls):
    mock_config_cls.return_value.palace_path = "/fake/palace"
    args = argparse.Namespace(palace=None, query="test query", agent="myagent", results=3)
    with patch("mempalace.searcher.search") as mock_search:
        cmd_search(args)
        mock_search.assert_called_once_with(
            query="test query",
            palace_path="/fake/palace",
            added_by="myagent",
            n_results=3,
        )


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_search_error_exits(mock_config_cls):
    mock_config_cls.return_value.palace_path = "/fake/palace"
    args = argparse.Namespace(palace=None, query="q", agent=None, results=5)
    from mempalace.searcher import SearchError

    with patch("mempalace.searcher.search", side_effect=SearchError("fail")):
        with pytest.raises(SystemExit) as exc_info:
            cmd_search(args)
        assert exc_info.value.code == 1


# ── cmd_instructions ───────────────────────────────────────────────────


def test_cmd_instructions_calls_run_instructions():
    args = argparse.Namespace(name="help")
    with patch("mempalace.instructions_cli.run_instructions") as mock_run:
        cmd_instructions(args)
        mock_run.assert_called_once_with(name="help")


# ── cmd_hook ───────────────────────────────────────────────────────────


def test_cmd_hook_calls_run_hook():
    args = argparse.Namespace(hook="session-start", harness="claude-code")
    with patch("mempalace.hooks_cli.run_hook") as mock_run:
        cmd_hook(args)
        mock_run.assert_called_once_with(hook_name="session-start", harness="claude-code")


# ── cmd_init ───────────────────────────────────────────────────────────


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_init_no_entities(mock_config_cls, tmp_path):
    args = argparse.Namespace(dir=str(tmp_path), yes=True)
    with patch("mempalace.entity_detector.scan_for_detection", return_value=[]):
        cmd_init(args)
        mock_config_cls.return_value.init.assert_called_once()


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_init_with_entities(mock_config_cls, tmp_path):
    fake_files = [tmp_path / "a.txt"]
    detected = {"people": [{"name": "Alice"}], "projects": [], "uncertain": []}
    confirmed = {"people": ["Alice"], "projects": []}
    args = argparse.Namespace(dir=str(tmp_path), yes=True)
    with (
        patch("mempalace.entity_detector.scan_for_detection", return_value=fake_files),
        patch("mempalace.entity_detector.detect_entities", return_value=detected),
        patch("mempalace.entity_detector.confirm_entities", return_value=confirmed),
        patch("builtins.open", MagicMock()),
    ):
        cmd_init(args)


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_init_with_entities_zero_total(mock_config_cls, tmp_path, capsys):
    """When entities detected but total is 0, prints 'No entities' message."""
    fake_files = [tmp_path / "a.txt"]
    detected = {"people": [], "projects": [], "uncertain": []}
    args = argparse.Namespace(dir=str(tmp_path), yes=False)
    with (
        patch("mempalace.entity_detector.scan_for_detection", return_value=fake_files),
        patch("mempalace.entity_detector.detect_entities", return_value=detected),
    ):
        cmd_init(args)
    out = capsys.readouterr().out
    assert "No entities detected" in out


# ── cmd_mine ───────────────────────────────────────────────────────────


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_mine_projects_mode(mock_config_cls):
    mock_config_cls.return_value.palace_path = "/fake/palace"
    args = argparse.Namespace(
        dir="/src",
        palace=None,
        mode="projects",
        agent="mempalace",
        limit=0,
        dry_run=False,
        no_gitignore=False,
        include_ignored=[],
        extract="exchange",
    )
    with patch("mempalace.miner.mine") as mock_mine:
        cmd_mine(args)
        mock_mine.assert_called_once_with(
            project_dir="/src",
            palace_path="/fake/palace",
            agent="mempalace",
            limit=0,
            dry_run=False,
            respect_gitignore=True,
            include_ignored=[],
        )


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_mine_convos_mode(mock_config_cls):
    mock_config_cls.return_value.palace_path = "/fake/palace"
    args = argparse.Namespace(
        dir="/chats",
        palace=None,
        mode="convos",
        agent="me",
        limit=10,
        dry_run=True,
        no_gitignore=False,
        include_ignored=[],
        extract="general",
    )
    with patch("mempalace.convo_miner.mine_convos") as mock_mine:
        cmd_mine(args)
        mock_mine.assert_called_once_with(
            convo_dir="/chats",
            palace_path="/fake/palace",
            agent="me",
            limit=10,
            dry_run=True,
            extract_mode="general",
        )


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_mine_include_ignored_comma_split(mock_config_cls):
    mock_config_cls.return_value.palace_path = "/fake/palace"
    args = argparse.Namespace(
        dir="/src",
        palace=None,
        mode="projects",
        agent="mempalace",
        limit=0,
        dry_run=False,
        no_gitignore=False,
        include_ignored=["a.txt,b.txt", "c.txt"],
        extract="exchange",
    )
    with patch("mempalace.miner.mine") as mock_mine:
        cmd_mine(args)
        mock_mine.assert_called_once()
        call_kwargs = mock_mine.call_args[1]
        assert call_kwargs["include_ignored"] == ["a.txt", "b.txt", "c.txt"]


# ── cmd_wakeup ─────────────────────────────────────────────────────────


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_wakeup(mock_config_cls, capsys):
    mock_config_cls.return_value.palace_path = "/fake/palace"
    args = argparse.Namespace(palace=None)
    mock_stack = MagicMock()
    mock_stack.wake_up.return_value = "Hello world context"
    with patch("mempalace.layers.MemoryStack", return_value=mock_stack):
        cmd_wakeup(args)
    out = capsys.readouterr().out
    assert "Hello world context" in out
    assert "tokens" in out


# ── cmd_split ──────────────────────────────────────────────────────────


def test_cmd_split_basic():
    args = argparse.Namespace(dir="/chats", output_dir=None, dry_run=False, min_sessions=2)
    with patch("mempalace.split_mega_files.main") as mock_main:
        cmd_split(args)
        mock_main.assert_called_once()


def test_cmd_split_all_options():
    args = argparse.Namespace(dir="/chats", output_dir="/out", dry_run=True, min_sessions=5)
    with patch("mempalace.split_mega_files.main") as mock_main:
        cmd_split(args)
        mock_main.assert_called_once()
    # sys.argv should be restored
    assert sys.argv[0] != "mempalace split"


# ── main() argparse dispatch ──────────────────────────────────────────


def test_main_no_args_prints_help(capsys):
    with patch("sys.argv", ["mempalace"]):
        main()
    out = capsys.readouterr().out
    assert "MemPalace" in out


def test_main_status_dispatches():
    with (
        patch("sys.argv", ["mempalace", "status"]),
        patch("mempalace.cli.cmd_status") as mock_cmd,
    ):
        main()
        mock_cmd.assert_called_once()


def test_main_search_dispatches():
    with (
        patch("sys.argv", ["mempalace", "search", "my query"]),
        patch("mempalace.cli.cmd_search") as mock_cmd,
    ):
        main()
        mock_cmd.assert_called_once()


def test_main_init_dispatches():
    with (
        patch("sys.argv", ["mempalace", "init", "/some/dir"]),
        patch("mempalace.cli.cmd_init") as mock_cmd,
    ):
        main()
        mock_cmd.assert_called_once()


def test_main_mine_dispatches():
    with (
        patch("sys.argv", ["mempalace", "mine", "/some/dir"]),
        patch("mempalace.cli.cmd_mine") as mock_cmd,
    ):
        main()
        mock_cmd.assert_called_once()


def test_main_wakeup_dispatches():
    with (
        patch("sys.argv", ["mempalace", "wake-up"]),
        patch("mempalace.cli.cmd_wakeup") as mock_cmd,
    ):
        main()
        mock_cmd.assert_called_once()


def test_main_split_dispatches():
    with (
        patch("sys.argv", ["mempalace", "split", "/chats"]),
        patch("mempalace.cli.cmd_split") as mock_cmd,
    ):
        main()
        mock_cmd.assert_called_once()


def test_mcp_command_prints_setup_guidance(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["mempalace", "mcp"])

    main()

    captured = capsys.readouterr()
    assert "MemPalace MCP quick setup:" in captured.out
    assert "claude mcp add mempalace -- python -m mempalace.mcp_server" in captured.out
    assert "\nOptional custom palace:\n" in captured.out
    assert "python -m mempalace.mcp_server --palace /path/to/palace" in captured.out
    assert "[--palace /path/to/palace]" not in captured.out
    assert captured.err == ""


def test_mcp_command_uses_custom_palace_path_when_provided(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["mempalace", "--palace", "~/tmp/my palace", "mcp"])

    main()

    captured = capsys.readouterr()
    expanded = str(Path("~/tmp/my palace").expanduser())

    assert "python -m mempalace.mcp_server --palace" in captured.out
    assert expanded in captured.out
    assert "Optional custom palace:" not in captured.out
    assert "[--palace /path/to/palace]" not in captured.out
    assert captured.err == ""


def test_main_hook_no_subcommand_prints_help(capsys):
    with patch("sys.argv", ["mempalace", "hook"]):
        main()
    out = capsys.readouterr().out
    assert "hook" in out.lower() or "run" in out.lower()


def test_main_hook_run_dispatches():
    with (
        patch(
            "sys.argv",
            ["mempalace", "hook", "run", "--hook", "session-start", "--harness", "claude-code"],
        ),
        patch("mempalace.cli.cmd_hook") as mock_cmd,
    ):
        main()
        mock_cmd.assert_called_once()


def test_main_instructions_no_subcommand_prints_help(capsys):
    with patch("sys.argv", ["mempalace", "instructions"]):
        main()
    out = capsys.readouterr().out
    assert "instructions" in out.lower() or "init" in out.lower()


def test_main_instructions_dispatches():
    with (
        patch("sys.argv", ["mempalace", "instructions", "help"]),
        patch("mempalace.cli.cmd_instructions") as mock_cmd,
    ):
        main()
        mock_cmd.assert_called_once()


def test_main_repair_dispatches():
    with (
        patch("sys.argv", ["mempalace", "repair"]),
        patch("mempalace.cli.cmd_repair") as mock_cmd,
    ):
        main()
        mock_cmd.assert_called_once()


# ── cmd_repair ─────────────────────────────────────────────────────────


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_repair_no_palace(mock_config_cls, tmp_path, capsys):
    mock_config_cls.return_value.palace_path = str(tmp_path / "nonexistent")
    args = argparse.Namespace(palace=None)
    mock_chromadb = MagicMock()
    with patch.dict("sys.modules", {"chromadb": mock_chromadb}):
        cmd_repair(args)
    out = capsys.readouterr().out
    assert "No palace found" in out


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_repair_error_reading(mock_config_cls, tmp_path, capsys):
    palace_dir = tmp_path / "palace"
    palace_dir.mkdir()
    mock_config_cls.return_value.palace_path = str(palace_dir)
    args = argparse.Namespace(palace=None)
    mock_chromadb = MagicMock()
    mock_client = MagicMock()
    mock_client.get_collection.side_effect = Exception("corrupt db")
    mock_chromadb.PersistentClient.return_value = mock_client
    with patch.dict("sys.modules", {"chromadb": mock_chromadb}):
        cmd_repair(args)
    out = capsys.readouterr().out
    assert "Error reading palace" in out


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_repair_zero_drawers(mock_config_cls, tmp_path, capsys):
    palace_dir = tmp_path / "palace"
    palace_dir.mkdir()
    mock_config_cls.return_value.palace_path = str(palace_dir)
    args = argparse.Namespace(palace=None)
    mock_chromadb = MagicMock()
    mock_col = MagicMock()
    mock_col.count.return_value = 0
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_col
    mock_chromadb.PersistentClient.return_value = mock_client
    with patch.dict("sys.modules", {"chromadb": mock_chromadb}):
        cmd_repair(args)
    out = capsys.readouterr().out
    assert "Nothing to repair" in out


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_repair_success(mock_config_cls, tmp_path, capsys):
    palace_dir = tmp_path / "palace"
    palace_dir.mkdir()
    mock_config_cls.return_value.palace_path = str(palace_dir)
    args = argparse.Namespace(palace=None)
    mock_chromadb = MagicMock()
    mock_col = MagicMock()
    mock_col.count.return_value = 2
    mock_col.get.return_value = {
        "ids": ["id1", "id2"],
        "documents": ["doc1", "doc2"],
        "metadatas": [{"added_by": "a"}, {"added_by": "b"}],
    }
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_col
    mock_new_col = MagicMock()
    mock_client.create_collection.return_value = mock_new_col
    mock_chromadb.PersistentClient.return_value = mock_client
    with patch.dict("sys.modules", {"chromadb": mock_chromadb}):
        cmd_repair(args)
    out = capsys.readouterr().out
    assert "Repair complete" in out
    assert "2 memories rebuilt" in out


def test_cmd_repair_trailing_slash_does_not_recurse():
    """Repair with trailing slash should put backup outside palace dir (#395)."""
    import os

    args = argparse.Namespace(palace="/tmp/fake_palace/")
    with patch("mempalace.cli.os.path.isdir", return_value=False):
        cmd_repair(args)
    # Verify the rstrip logic: palace_path should not end with separator
    palace_path = os.path.expanduser(args.palace).rstrip(os.sep)
    backup_path = palace_path + ".backup"
    assert not backup_path.startswith(palace_path + os.sep)
