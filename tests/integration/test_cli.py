"""Tests for mempalace.cli -- the main CLI dispatcher."""

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
#
# cmd_repair was retired 2026-05-12 (chromadb removed): the
# chromadb HNSW rebuild path no longer applies under sqlite_vec.
# The remaining tests assert the retirement banner is printed so
# operators with stale muscle memory see a clear message instead
# of a silent no-op.


def test_cmd_repair_prints_retirement_banner(capsys):
    args = argparse.Namespace(palace=None)
    cmd_repair(args)
    out = capsys.readouterr().out
    assert "retired" in out
    assert "re-mine" in out


# ── cmd_backfill_vectors ───────────────────────────────────────────────


def test_cmd_backfill_vectors_rebuilds_vec_palace(tmp_path):
    """End-to-end: seed an entity into a fresh palace, then run
    `mempalace backfill-vectors` and confirm the entity now has a row
    in vec_palace. This is the user-facing recovery path for palaces
    upgraded from chromadb."""
    import argparse
    from mempalace.cli import cmd_backfill_vectors
    from mempalace.knowledge_graph import KnowledgeGraph
    from mempalace.vector_store import (
        RECORDS_COLLECTION,
        get_vector_store,
        reset_singletons,
    )

    palace = tmp_path / "palace"
    palace.mkdir()
    db_path = str(palace / "knowledge_graph.sqlite3")
    kg = KnowledgeGraph(db_path=db_path)
    # Seed via direct add_entity so we exercise the "exists in SQLite,
    # missing from vec_palace" path that real chromadb-upgrade palaces
    # have.
    kg.add_entity(
        "backfill_target_alpha",
        kind="entity",
        content="alpha entity content for backfill regression test",
        importance=3,
    )

    # Wrest the per-test palace path away from the production
    # MempalaceConfig singleton so cmd_backfill_vectors sees ours.
    args = argparse.Namespace(
        palace=str(palace),
        dry_run=False,
        force=False,
        json=True,
    )

    reset_singletons()
    cmd_backfill_vectors(args)

    # After backfill, vec_palace should contain the seeded entity.
    vs = get_vector_store(str(palace))
    got = vs.get(RECORDS_COLLECTION, ids=["backfill_target_alpha"])
    assert got.ids == ["backfill_target_alpha"], (
        f"expected backfill to write the seeded entity to vec_palace; got ids={got.ids!r}"
    )
    reset_singletons()


def test_cmd_backfill_vectors_dry_run_writes_nothing(tmp_path):
    """`--dry-run` walks + counts but writes nothing to vec_palace."""
    import argparse
    from mempalace.cli import cmd_backfill_vectors
    from mempalace.knowledge_graph import KnowledgeGraph
    from mempalace.vector_store import (
        RECORDS_COLLECTION,
        get_vector_store,
        reset_singletons,
    )

    palace = tmp_path / "palace_dry"
    palace.mkdir()
    db_path = str(palace / "knowledge_graph.sqlite3")
    kg = KnowledgeGraph(db_path=db_path)
    kg.add_entity(
        "backfill_dry_target",
        kind="entity",
        content="dry run target -- must NOT land in vec_palace",
        importance=3,
    )

    args = argparse.Namespace(
        palace=str(palace),
        dry_run=True,
        force=False,
        json=True,
    )

    reset_singletons()
    cmd_backfill_vectors(args)

    vs = get_vector_store(str(palace))
    got = vs.get(RECORDS_COLLECTION, ids=["backfill_dry_target"])
    assert got.ids == [], f"expected dry-run to write nothing; got ids={got.ids!r}"
    reset_singletons()


def test_cmd_backfill_vectors_writes_multi_view_and_triples(tmp_path):
    """v3.2.2 dual-method backfill writes (1) single-row,
    (2) multi-view ``{eid}__v{i}``, (3) context-view ``{cid}_v{i}``,
    AND (4) triple-statement rows. Regression guard: v3.2.1 shipped
    only (1) which left Channel A multi-view + Channel D + triple
    cosine cold against pre-v3.2.0 palaces.
    """
    import argparse
    import json as _json

    from mempalace.cli import cmd_backfill_vectors
    from mempalace.knowledge_graph import KnowledgeGraph
    from mempalace.vector_store import (
        CONTEXT_VIEWS_COLLECTION,
        RECORDS_COLLECTION,
        TRIPLES_COLLECTION,
        get_vector_store,
        reset_singletons,
    )

    palace = tmp_path / "palace_full"
    palace.mkdir()
    db_path = str(palace / "knowledge_graph.sqlite3")
    kg = KnowledgeGraph(db_path=db_path)

    # Seed an entity bound to a fabricated creation context whose
    # properties.queries list mimics what context_lookup_or_create
    # persists. Backfill should fan that out into __v0, __v1 rows
    # plus the trailing summary view.
    ctx_id = "ctx_backfill_test_001"
    kg.add_entity(
        ctx_id,
        kind="context",
        content="context for backfill multi-view test",
        importance=3,
    )
    # Patch the context entity to carry queries in its properties JSON
    # (mirrors mcp_server.context_lookup_or_create's write site).
    # Reuse kg._conn() so we share the existing write lock instead of
    # racing a second sqlite connection against it.
    props = {
        "queries": [
            "first cosine perspective on the topic",
            "second cosine perspective on the topic",
        ]
    }
    conn = kg._conn()
    conn.execute(
        "UPDATE entities SET properties=? WHERE id=?",
        (_json.dumps(props), ctx_id),
    )
    conn.commit()

    # The entity-under-test points at the context above so backfill
    # picks up its queries[].
    eid = "backfill_multiview_target"
    kg.add_entity(
        eid,
        kind="entity",
        content="multi-view backfill target content",
        importance=3,
    )
    kg.add_entity(
        "object_of_triple_for_backfill",
        kind="entity",
        content="object of triple for backfill",
        importance=3,
    )
    # Patch creation_context_id + seed a triple, again on the kg conn.
    conn = kg._conn()
    conn.execute(
        "UPDATE entities SET creation_context_id=? WHERE id=?",
        (ctx_id, eid),
    )
    conn.execute(
        "INSERT INTO triples (id, subject, predicate, object, "
        "statement, confidence, valid_from) VALUES (?,?,?,?,?,?,?)",
        (
            "triple_backfill_001",
            eid,
            "related_to",
            "object_of_triple_for_backfill",
            "backfill target is related to its triple object",
            1.0,
            "2026-05-12",
        ),
    )
    conn.commit()

    args = argparse.Namespace(
        palace=str(palace),
        dry_run=False,
        force=True,  # ignore any prior stamp
        json=True,
    )

    reset_singletons()
    cmd_backfill_vectors(args)

    vs = get_vector_store(str(palace))

    # ── Path 1: single-row content ────────────────────────────────────
    got_single = vs.get(RECORDS_COLLECTION, ids=[eid])
    assert got_single.ids == [eid], (
        f"expected single-row content to land in mempalace_records; got ids={got_single.ids!r}"
    )

    # ── Path 2: multi-view rows ───────────────────────────────────────
    view_ids = [f"{eid}__v0", f"{eid}__v1", f"{eid}__v2"]
    got_multi = vs.get(RECORDS_COLLECTION, ids=view_ids)
    assert eid + "__v0" in got_multi.ids and eid + "__v1" in got_multi.ids, (
        "expected multi-view rows {eid}__v0 and {eid}__v1 to land in "
        f"mempalace_records; got ids={got_multi.ids!r}"
    )

    # ── Path 3: context-view rows ─────────────────────────────────────
    ctx_view_ids = [f"{ctx_id}_v0", f"{ctx_id}_v1"]
    got_ctx = vs.get(CONTEXT_VIEWS_COLLECTION, ids=ctx_view_ids)
    assert ctx_id + "_v0" in got_ctx.ids and ctx_id + "_v1" in got_ctx.ids, (
        "expected context-view rows {cid}_v0 and {cid}_v1 to land in "
        f"mempalace_context_views; got ids={got_ctx.ids!r}"
    )

    # ── Path 4: triple statement row ──────────────────────────────────
    got_triple = vs.get(TRIPLES_COLLECTION, ids=["triple_backfill_001"])
    assert got_triple.ids == ["triple_backfill_001"], (
        f"expected triple statement to land in mempalace_triples; got ids={got_triple.ids!r}"
    )

    reset_singletons()


pytestmark = pytest.mark.integration
