#!/usr/bin/env python3
"""
MemPalace -- Give your AI a memory. No API key required.

Two ways to ingest:
  Projects:      mempalace mine ~/projects/my_app          (code, docs, notes)
  Conversations: mempalace mine ~/chats/ --mode convos     (Claude, ChatGPT, Slack)

Same palace. Same search. Different ingest strategies.

Commands:
    mempalace init <dir>                  Detect entities from folder structure
    mempalace split <dir>                 Split concatenated mega-files into per-session files
    mempalace mine <dir>                  Mine project files (default)
    mempalace mine <dir> --mode convos    Mine conversation exports
    mempalace search "query"              Find anything, exact words
    mempalace mcp                         Show MCP setup command
    mempalace wake-up                     Show L0 + L1 wake-up context
    mempalace status                      Show what's been filed

Examples:
    mempalace init ~/projects/my_app
    mempalace mine ~/projects/my_app
    mempalace mine ~/chats/claude-sessions --mode convos
    mempalace search "why did we switch to GraphQL"
    mempalace search "pricing discussion"
"""

import os
import sys
import shlex
import argparse
from pathlib import Path

from .config import MempalaceConfig


def cmd_init(args):
    import json
    from pathlib import Path
    from .entity_detector import scan_for_detection, detect_entities, confirm_entities

    # Auto-detect people and projects from file content
    print(f"\n  Scanning for entities in: {args.dir}")
    files = scan_for_detection(args.dir)
    if files:
        print(f"  Reading {len(files)} files...")
        detected = detect_entities(files)
        total = len(detected["people"]) + len(detected["projects"]) + len(detected["uncertain"])
        if total > 0:
            confirmed = confirm_entities(detected, yes=getattr(args, "yes", False))
            # Save confirmed entities to <project>/entities.json for the miner
            if confirmed["people"] or confirmed["projects"]:
                entities_path = Path(args.dir).expanduser().resolve() / "entities.json"
                with open(entities_path, "w") as f:
                    json.dump(confirmed, f, indent=2)
                print(f"  Entities saved: {entities_path}")
        else:
            print("  No entities detected.")

    MempalaceConfig().init()


def cmd_mine(args):
    palace_path = os.path.expanduser(args.palace) if args.palace else MempalaceConfig().palace_path
    include_ignored = []
    for raw in args.include_ignored or []:
        include_ignored.extend(part.strip() for part in raw.split(",") if part.strip())

    if args.mode == "convos":
        from .convo_miner import mine_convos

        mine_convos(
            convo_dir=args.dir,
            palace_path=palace_path,
            agent=args.agent,
            limit=args.limit,
            dry_run=args.dry_run,
            extract_mode=args.extract,
        )
    else:
        from .miner import mine

        mine(
            project_dir=args.dir,
            palace_path=palace_path,
            agent=args.agent,
            limit=args.limit,
            dry_run=args.dry_run,
            respect_gitignore=not args.no_gitignore,
            include_ignored=include_ignored,
        )


def cmd_search(args):
    from .searcher import search, SearchError

    palace_path = os.path.expanduser(args.palace) if args.palace else MempalaceConfig().palace_path
    try:
        search(
            query=args.query,
            palace_path=palace_path,
            added_by=args.agent,
            n_results=args.results,
        )
    except SearchError:
        sys.exit(1)


def cmd_wakeup(args):
    """Show L0 (identity) + L1 (essential story) -- the wake-up context."""
    from .layers import MemoryStack

    palace_path = os.path.expanduser(args.palace) if args.palace else MempalaceConfig().palace_path
    stack = MemoryStack(palace_path=palace_path)

    text = stack.wake_up()
    tokens = len(text) // 4
    print(f"Wake-up text (~{tokens} tokens):")
    print("=" * 50)
    print(text)


def cmd_clear_pending(args):
    """Clear stuck pending_user_messages -- in-band recovery for poisoned queues.

    The user-intent gate refuses every non-mempalace tool until the
    pending queue drains. If the queue is corrupt (e.g. lone UTF-16
    surrogates from a pre-slice-9 stdin encoding hiccup, or any
    schema-mismatched entry that crashes mint), the gate hard-locks
    every session that reads it -- agents can't drain through the very
    tool that crashes. This command is the eject button: delete the
    file, the gate sees an empty queue, the next user prompt mints
    cleanly.

    Selection rules:
      --session-id <sid>  : clear that specific session's queue
      --all               : clear every pending_user_messages_*.json
      (neither)           : clear the MOST-RECENTLY-MODIFIED queue
                            (typical recovery scenario: the agent that
                            just got stuck is the latest writer)
    """
    from . import hooks_cli as _hc

    state_dir = _hc.STATE_DIR
    candidates = list(state_dir.glob("pending_user_messages_*.json")) if state_dir.is_dir() else []

    if not candidates:
        print(f"No pending_user_messages_*.json files in {state_dir}.")
        return

    if getattr(args, "all", False):
        targets = candidates
    elif args.session_id:
        from .hooks_cli import _sanitize_session_id

        safe_sid = _sanitize_session_id(args.session_id)
        path = state_dir / f"pending_user_messages_{safe_sid}.json"
        if not path.is_file():
            print(f"No pending queue for session_id='{args.session_id}' (looked at {path}).")
            return
        targets = [path]
    else:
        # Most-recent path
        targets = [max(candidates, key=lambda p: p.stat().st_mtime)]

    cleared = 0
    for path in targets:
        try:
            count = 0
            try:
                import json as _json

                data = _json.loads(path.read_text(encoding="utf-8"))
                count = len(data.get("messages") or [])
            except Exception:
                pass
            path.unlink()
            print(f"  cleared: {path.name} (had {count} pending message(s))")
            cleared += 1
        except Exception as e:
            print(f"  FAILED to clear {path.name}: {e}")

    print(f"\nDone. {cleared} queue file(s) cleared.")
    if cleared:
        print(
            "Next user prompt in any of these sessions will mint cleanly "
            "and the user-intent gate will release."
        )


def cmd_split(args):
    """Split concatenated transcript mega-files into per-session files."""
    from .split_mega_files import main as split_main
    import sys

    # Rebuild argv for split_mega_files argparse
    argv = ["--source", args.dir]
    if args.output_dir:
        argv += ["--output-dir", args.output_dir]
    if args.dry_run:
        argv.append("--dry-run")
    if args.min_sessions != 2:
        argv += ["--min-sessions", str(args.min_sessions)]

    old_argv = sys.argv
    sys.argv = ["mempalace split"] + argv
    try:
        split_main()
    finally:
        sys.argv = old_argv


def cmd_migrate(args):
    """Migrate palace from a different ChromaDB version."""
    from .migrate import migrate

    palace_path = os.path.expanduser(args.palace) if args.palace else MempalaceConfig().palace_path
    migrate(palace_path=palace_path, dry_run=args.dry_run)


def cmd_status(args):
    from .miner import status

    palace_path = os.path.expanduser(args.palace) if args.palace else MempalaceConfig().palace_path
    status(palace_path=palace_path)


def cmd_doctor(args):
    """Audit chroma <-> SQLite consistency for the entity collection.

    Reports counts on each side and the first N orphans (ids present in
    one store but not the other). Closes 2026-04-25 audit finding #3:
    drift between the two stores can leak through tool_kg_delete_entity
    (chroma-only delete pre-fix) and through declare paths that fail on
    one side. Run periodically to catch silent drift.
    """
    import chromadb
    import json as _json

    palace_path = os.path.expanduser(args.palace) if args.palace else MempalaceConfig().palace_path
    if not os.path.isdir(palace_path):
        print(f"\n  No palace found at {palace_path}")
        raise SystemExit(2)

    from .knowledge_graph import KnowledgeGraph

    db_path = os.path.join(palace_path, "knowledge_graph.sqlite3")
    kg = KnowledgeGraph(db_path=db_path)
    conn = kg._conn()
    try:
        sql_rows = conn.execute(
            "SELECT id, status FROM entities WHERE status IN ('active','merged')"
        ).fetchall()
    except Exception as exc:
        print(f"  SQLite query failed: {exc}")
        raise SystemExit(2)
    sql_ids = {r["id"] for r in sql_rows}
    sql_active = {r["id"] for r in sql_rows if r["status"] == "active"}

    # Chroma side: skip the records collection; collect ids from every
    # other collection (the entity collection name has changed across
    # versions, so list_collections + filter is more robust than a
    # hardcoded name).
    # Collect ids ONLY from the entity collection. The palace also
    # holds mempalace_records (for kind=record content), mempalace_triples
    # (triple-context-feedback), and mempalace_context_views (context
    # accretion); none of those should be compared against the entities
    # SQLite table.
    chroma_ids: set[str] = set()
    chroma_collection_name = "mempalace_entities"
    try:
        client = chromadb.PersistentClient(path=palace_path)
        try:
            col = client.get_collection(chroma_collection_name)
        except Exception as exc:
            print(f"  Chroma collection '{chroma_collection_name}' not found: {exc}")
            print("  (this palace may use a different name; edit cmd_doctor to target it)")
            raise SystemExit(2)
        try:
            got = col.get(include=[])
            ids = got.get("ids") if isinstance(got, dict) else None
            if ids:
                chroma_ids.update(ids)
        except Exception as exc:
            print(f"  Chroma get failed: {exc}")
            raise SystemExit(2)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"  Chroma open failed: {exc}")
        raise SystemExit(2)

    sql_only = sorted(sql_active - chroma_ids)
    chroma_only = sorted(chroma_ids - sql_ids)

    payload = {
        "palace_path": palace_path,
        "chroma_collection": chroma_collection_name,
        "counts": {
            "sql_total_active_or_merged": len(sql_ids),
            "sql_active": len(sql_active),
            "chroma_entities": len(chroma_ids),
            "sql_active_only_orphans": len(sql_only),
            "chroma_only_orphans": len(chroma_only),
        },
        "samples": {
            "sql_active_only": sql_only[: args.max_orphans],
            "chroma_only": chroma_only[: args.max_orphans],
        },
    }

    if args.json:
        print(_json.dumps(payload, indent=2))
        return

    print(f"\n{'=' * 55}")
    print("  MemPalace Doctor - entity store drift")
    print(f"{'=' * 55}\n")
    print(f"  Palace: {palace_path}")
    print(f"  Chroma collection: {chroma_collection_name}\n")
    c = payload["counts"]
    print(f"  SQL entities (active+merged): {c['sql_total_active_or_merged']}")
    print(f"  SQL entities (active only):   {c['sql_active']}")
    print(f"  Chroma entities:              {c['chroma_entities']}")
    print(f"  Active-only-in-SQL orphans:   {c['sql_active_only_orphans']}")
    print(f"  Only-in-Chroma orphans:       {c['chroma_only_orphans']}")
    if sql_only:
        print(f"\n  SQL-active-only sample (first {args.max_orphans}):")
        for x in sql_only[: args.max_orphans]:
            print(f"    - {x}")
    if chroma_only:
        print(f"\n  Chroma-only sample (first {args.max_orphans}):")
        for x in chroma_only[: args.max_orphans]:
            print(f"    - {x}")
    print()


def cmd_repair(args):
    """Rebuild palace vector index from SQLite metadata."""
    import chromadb
    import shutil

    palace_path = os.path.expanduser(args.palace) if args.palace else MempalaceConfig().palace_path

    if not os.path.isdir(palace_path):
        print(f"\n  No palace found at {palace_path}")
        return

    print(f"\n{'=' * 55}")
    print("  MemPalace Repair")
    print(f"{'=' * 55}\n")
    print(f"  Palace: {palace_path}")

    # Try to read existing memories
    try:
        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_records")
        total = col.count()
        print(f"  Memories found: {total}")
    except Exception as e:
        print(f"  Error reading palace: {e}")
        print("  Cannot recover -- palace may need to be re-mined from source files.")
        return

    if total == 0:
        print("  Nothing to repair.")
        return

    # Extract all memories in batches
    print("\n  Extracting memories...")
    batch_size = 5000
    all_ids = []
    all_docs = []
    all_metas = []
    offset = 0
    while offset < total:
        batch = col.get(limit=batch_size, offset=offset, include=["documents", "metadatas"])
        all_ids.extend(batch["ids"])
        all_docs.extend(batch["documents"])
        all_metas.extend(batch["metadatas"])
        offset += batch_size
    print(f"  Extracted {len(all_ids)} memories")

    # Backup and rebuild
    palace_path = palace_path.rstrip(os.sep)
    backup_path = palace_path + ".backup"
    if os.path.exists(backup_path):
        shutil.rmtree(backup_path)
    print(f"  Backing up to {backup_path}...")
    shutil.copytree(palace_path, backup_path)

    print("  Rebuilding collection...")
    client.delete_collection("mempalace_records")
    # Pin cosine on rebuild -- matches the rest of the palace.
    new_col = client.create_collection("mempalace_records", metadata={"hnsw:space": "cosine"})

    filed = 0
    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i : i + batch_size]
        batch_docs = all_docs[i : i + batch_size]
        batch_metas = all_metas[i : i + batch_size]
        new_col.add(documents=batch_docs, ids=batch_ids, metadatas=batch_metas)
        filed += len(batch_ids)
        print(f"  Re-filed {filed}/{len(all_ids)} memories...")

    print(f"\n  Repair complete. {filed} memories rebuilt.")
    print(f"  Backup saved at {backup_path}")
    print(f"\n{'=' * 55}\n")


def cmd_hook(args):
    """Run hook logic: reads JSON from stdin, outputs JSON to stdout."""
    from .hooks_cli import run_hook

    run_hook(hook_name=args.hook, harness=args.harness)


def cmd_instructions(args):
    """Output skill instructions to stdout."""
    from .instructions_cli import run_instructions

    run_instructions(name=args.name)


def cmd_eval(args):
    """mempalace eval -- P3 retrieval-quality reports from JSONL telemetry.

    Reads the search_log.jsonl + finalize_log.jsonl traces written by
    tool_kg_search and tool_finalize_intent, and summarises:
      - context reuse rate (ColBERT-MaxSim hit rate against the
        T_reuse=0.90 threshold from context_lookup_or_create)
      - per-channel contribution to top-K (cosine / graph / keyword / context)
      - finalize-time stats (memories rated, contexts used)

    See mempalace/eval_harness.py for the underlying implementations.
    """
    import json as _json
    from . import eval_harness

    days = getattr(args, "days", None)
    if getattr(args, "reuse_rate", False) and not getattr(args, "report", False):
        payload = eval_harness.context_reuse_rate(days=days)
        if getattr(args, "json", False):
            print(_json.dumps(payload, indent=2))
        else:
            print(
                f"context reuse rate (window: {'all' if not days else f'{days}d'}): "
                f"{payload['rate']:.2%} ({payload['reused']}/{payload['total_searches']})"
            )
        return
    report = eval_harness.summary_report(days=days)
    if getattr(args, "json", False):
        print(_json.dumps(report, indent=2))
    else:
        print(eval_harness.format_report(report))


def cmd_linkauthor_process(args):
    """Drain the link-author candidate queue via the Opus/Haiku jury.

    Reads the palace from config, validates the API key, iterates
    pending candidates up to ``--max`` above ``--threshold``, and
    persists verdicts. Exit codes are documented in
    docs/link_author_scheduling.md.
    """
    import json as _json

    from .knowledge_graph import KnowledgeGraph
    from . import link_author

    cfg = MempalaceConfig()
    palace_path = args.palace or cfg.palace_path
    kg = KnowledgeGraph(db_path=str(Path(palace_path) / "knowledge_graph.sqlite3"))
    la_cfg = dict(cfg.link_author)
    la_cfg["palace_path"] = palace_path

    batch_design = not args.no_batch_design
    try:
        summary = link_author.process(
            kg,
            la_cfg,
            max_per_run=args.max,
            threshold=args.threshold,
            dry_run=args.dry_run,
            batch_design=batch_design,
        )
    except SystemExit:
        # _validate_api_key exits with 2 or 3 on its own -- propagate.
        raise
    print(_json.dumps(summary, indent=2, default=str))


def cmd_linkauthor_status(args):
    """Report pending / recent-run / (optional) new-predicate state."""
    import json as _json

    from .knowledge_graph import KnowledgeGraph
    from . import link_author

    cfg = MempalaceConfig()
    palace_path = args.palace or cfg.palace_path
    kg = KnowledgeGraph(db_path=str(Path(palace_path) / "knowledge_graph.sqlite3"))
    snap = link_author.status(
        kg,
        recent=args.recent,
        new_predicates=args.new_predicates,
    )
    print(_json.dumps(snap, indent=2, default=str))


def cmd_mcp(args):
    """Show how to wire MemPalace into MCP-capable hosts."""
    base_server_cmd = "python -m mempalace.mcp_server"

    if args.palace:
        resolved_palace = str(Path(args.palace).expanduser())
        server_cmd = f"{base_server_cmd} --palace {shlex.quote(resolved_palace)}"
    else:
        server_cmd = base_server_cmd

    print("MemPalace MCP quick setup:")
    print(f"  claude mcp add mempalace -- {server_cmd}")
    print("\nRun the server directly:")
    print(f"  {server_cmd}")

    if not args.palace:
        print("\nOptional custom palace:")
        print(f"  claude mcp add mempalace -- {base_server_cmd} --palace /path/to/palace")
        print(f"  {base_server_cmd} --palace /path/to/palace")


def main():
    parser = argparse.ArgumentParser(
        description="MemPalace -- Give your AI a memory. No API key required.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--palace",
        default=None,
        help="Where the palace lives (default: from ~/.mempalace/config.json or ~/.mempalace/palace)",
    )

    sub = parser.add_subparsers(dest="command")

    # init
    p_init = sub.add_parser("init", help="Detect entities from your project files")
    p_init.add_argument("dir", help="Project directory to set up")
    p_init.add_argument(
        "--yes", action="store_true", help="Auto-accept all detected entities (non-interactive)"
    )

    # mine
    p_mine = sub.add_parser("mine", help="Mine files into the palace")
    p_mine.add_argument("dir", help="Directory to mine")
    p_mine.add_argument(
        "--mode",
        choices=["projects", "convos"],
        default="projects",
        help="Ingest mode: 'projects' for code/docs (default), 'convos' for chat exports",
    )
    p_mine.add_argument(
        "--no-gitignore",
        action="store_true",
        help="Don't respect .gitignore files when scanning project files",
    )
    p_mine.add_argument(
        "--include-ignored",
        action="append",
        default=[],
        help="Always scan these project-relative paths even if ignored; repeat or pass comma-separated paths",
    )
    p_mine.add_argument(
        "--agent",
        default="mempalace",
        help="Your name -- recorded on every memory (default: mempalace)",
    )
    p_mine.add_argument("--limit", type=int, default=0, help="Max files to process (0 = all)")
    p_mine.add_argument(
        "--dry-run", action="store_true", help="Show what would be filed without filing"
    )
    p_mine.add_argument(
        "--extract",
        choices=["exchange", "general"],
        default="exchange",
        help="Extraction strategy for convos mode: 'exchange' (default) or 'general' (5 memory types)",
    )

    # search
    p_search = sub.add_parser("search", help="Find anything, exact words")
    p_search.add_argument("query", help="What to search for")
    p_search.add_argument("--agent", default=None, help="Limit to memories by a specific agent")
    p_search.add_argument("--results", type=int, default=5, help="Number of results")

    # wake-up
    sub.add_parser("wake-up", help="Show L0 + L1 wake-up context (~600-900 tokens)")

    # clear-pending -- in-band recovery for poisoned pending_user_messages
    p_clear = sub.add_parser(
        "clear-pending",
        help="Clear stuck pending_user_messages for a session (recovery from a poisoned queue).",
    )
    p_clear.add_argument(
        "--session-id",
        default=None,
        help=(
            "Session id to clear. If omitted, clears the most-recent "
            "pending_user_messages_*.json file in the hook_state dir. "
            "Use this to unstick an agent whose queue was poisoned by a "
            "pre-slice-9 surrogate-encoding bug or any other corruption."
        ),
    )
    p_clear.add_argument(
        "--all",
        action="store_true",
        help="Clear pending queues for ALL sessions (use with care).",
    )

    # split
    p_split = sub.add_parser(
        "split",
        help="Split concatenated transcript mega-files into per-session files (run before mine)",
    )
    p_split.add_argument("dir", help="Directory containing transcript files")
    p_split.add_argument(
        "--output-dir",
        default=None,
        help="Write split files here (default: same directory as source files)",
    )
    p_split.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be split without writing files",
    )
    p_split.add_argument(
        "--min-sessions",
        type=int,
        default=2,
        help="Only split files containing at least N sessions (default: 2)",
    )

    # hook
    p_hook = sub.add_parser(
        "hook",
        help="Run hook logic (reads JSON from stdin, outputs JSON to stdout)",
    )
    hook_sub = p_hook.add_subparsers(dest="hook_action")
    p_hook_run = hook_sub.add_parser("run", help="Execute a hook")
    p_hook_run.add_argument(
        "--hook",
        required=True,
        # Accept BOTH hyphenated and un-hyphenated forms for every hook
        # the dispatcher in hooks_cli.run_hook actually handles. The
        # plugin shell wrappers under .claude-plugin/hooks/ pass the
        # un-hyphenated forms (sessionstart, userpromptsubmit), so any
        # name missing here makes argparse exit 2 and Claude Code reads
        # exit 2 as a BLOCKING error -- UserPromptSubmit blocks user
        # prompts entirely; SessionStart blocks session start. Pre-fix
        # only `pretooluse`, `precompact`, `stop`, and the hyphenated
        # `session-start` worked, leaving `userpromptsubmit` and
        # `sessionstart` failing every invocation. The dispatcher
        # already mapped both forms; the gap was only argparse choices.
        choices=[
            "session-start",
            "sessionstart",
            "stop",
            "precompact",
            "pretooluse",
            "userpromptsubmit",
            "user-prompt-submit",
        ],
        help="Hook name to run",
    )
    p_hook_run.add_argument(
        "--harness",
        required=True,
        choices=["claude-code", "codex"],
        help="Harness type (determines stdin JSON format)",
    )

    # instructions
    p_instructions = sub.add_parser(
        "instructions",
        help="Output skill instructions to stdout",
    )
    instructions_sub = p_instructions.add_subparsers(dest="instructions_name")
    for instr_name in ["init", "search", "mine", "help", "status"]:
        instructions_sub.add_parser(instr_name, help=f"Output {instr_name} instructions")

    # repair
    sub.add_parser(
        "repair",
        help="Rebuild palace vector index from stored data (fixes segfaults after corruption)",
    )

    # mcp
    sub.add_parser(
        "mcp",
        help="Show MCP setup command for connecting MemPalace to your AI client",
    )

    # eval -- P3 retrieval-quality reports over the JSONL telemetry.
    p_eval = sub.add_parser(
        "eval",
        help="Report retrieval-quality stats from hook_state JSONL telemetry",
    )
    p_eval.add_argument(
        "--report",
        action="store_true",
        help="Print the full summary report (reuse rate + per-channel + finalize stats)",
    )
    p_eval.add_argument(
        "--reuse-rate",
        action="store_true",
        help="Print just the context reuse rate over the window",
    )
    p_eval.add_argument(
        "--days",
        type=int,
        default=None,
        help="Lookback window in days (default: all time)",
    )
    p_eval.add_argument(
        "--json",
        action="store_true",
        help="Emit the raw JSON payload instead of the pretty-printed report",
    )

    # status
    # migrate
    p_migrate = sub.add_parser(
        "migrate",
        help="Migrate palace from a different ChromaDB version (fixes 3.0.0 → 3.1.0 upgrade)",
    )
    p_migrate.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without changing anything",
    )

    sub.add_parser("status", help="Show what's been filed")

    p_doctor = sub.add_parser(
        "doctor",
        help="Audit chroma <-> SQLite consistency for the entity collection",
    )
    p_doctor.add_argument(
        "--max-orphans",
        type=int,
        default=20,
        help="Max orphan ids to print per side (default 20).",
    )
    p_doctor.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of prose.",
    )

    # link-author -- graph-link authoring pipeline
    p_linkauthor = sub.add_parser(
        "link-author",
        help="Drain the link-author candidate queue via the Opus/Haiku jury",
    )
    linkauthor_sub = p_linkauthor.add_subparsers(dest="linkauthor_action")

    p_la_process = linkauthor_sub.add_parser(
        "process",
        help=(
            "Validate API key, iterate pending candidates, author accepted "
            "edges. Exit 0 = ok, 2 = bad/missing key, 3 = API unreachable."
        ),
    )
    p_la_process.add_argument(
        "--max",
        type=int,
        default=None,
        help="Max candidates to process in this run (default from config.link_author.max_per_run)",
    )
    p_la_process.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum candidate score to consider (default from config.link_author.threshold)",
    )
    p_la_process.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate + compute verdicts but don't write edges or verdicts back",
    )
    p_la_process.add_argument(
        "--no-batch-design",
        action="store_true",
        help="Disable Stage-1 batching (one Opus call per candidate instead of per cluster)",
    )

    p_la_status = linkauthor_sub.add_parser(
        "status",
        help="Summary of pending / recently-processed candidates and runs",
    )
    p_la_status.add_argument(
        "--recent",
        type=int,
        default=5,
        help="Number of most-recent link_author_runs rows to show (default: 5)",
    )
    p_la_status.add_argument(
        "--new-predicates",
        action="store_true",
        help="Also include the contents of new_predicates.jsonl (all recent predicate creations)",
    )

    # gardener -- memory-gardener background process for flag resolution.
    p_gardener = sub.add_parser(
        "gardener",
        help="Process injection-gate quality flags via Claude Code",
    )
    gardener_sub = p_gardener.add_subparsers(dest="gardener_action")
    p_gardener_process = gardener_sub.add_parser(
        "process",
        help="Drain up to --max-batches batches of pending flags",
    )
    p_gardener_process.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Max flags per Claude Code subprocess invocation (default: 5)",
    )
    p_gardener_process.add_argument(
        "--max-batches",
        type=int,
        default=1,
        help="Max batches to process in this run (default: 1)",
    )
    p_gardener_process.add_argument(
        "--model",
        default=None,
        help="Override the gardener model (default: claude-haiku-4-5)",
    )
    p_gardener_process.add_argument(
        "--flag-id",
        type=int,
        default=None,
        help="Target a specific flag by id (skips the queue; runs one batch).",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Handle two-level subcommands
    if args.command == "hook":
        if not getattr(args, "hook_action", None):
            p_hook.print_help()
            return
        cmd_hook(args)
        return

    if args.command == "instructions":
        name = getattr(args, "instructions_name", None)
        if not name:
            p_instructions.print_help()
            return
        args.name = name
        cmd_instructions(args)
        return

    if args.command == "link-author":
        action = getattr(args, "linkauthor_action", None)
        if not action:
            p_linkauthor.print_help()
            return
        if action == "process":
            cmd_linkauthor_process(args)
        elif action == "status":
            cmd_linkauthor_status(args)
        return

    if args.command == "gardener":
        action = getattr(args, "gardener_action", None)
        if not action:
            p_gardener.print_help()
            return
        if action == "process":
            from . import memory_gardener as _mg

            code = _mg.cli_process(args)
            if code:
                raise SystemExit(code)
        return

    dispatch = {
        "init": cmd_init,
        "mine": cmd_mine,
        "split": cmd_split,
        "search": cmd_search,
        "mcp": cmd_mcp,
        "wake-up": cmd_wakeup,
        "repair": cmd_repair,
        "migrate": cmd_migrate,
        "status": cmd_status,
        "eval": cmd_eval,
        "doctor": cmd_doctor,
        "clear-pending": cmd_clear_pending,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
