"""
Hook logic for MemPalace — Python implementation of session-start, stop, and precompact hooks.

Reads JSON from stdin, outputs JSON to stdout.
Supported hooks: session-start, stop, precompact
Supported harnesses: claude-code, codex (extensible to cursor, gemini, etc.)
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SAVE_INTERVAL = 15
STATE_DIR = Path.home() / ".mempalace" / "hook_state"

STOP_BLOCK_REASON = (
    "AUTO-SAVE checkpoint. "
    "First, finalize the active intent if one exists (mempalace_finalize_intent). "
    "Then persist new knowledge from this session: "
    "(1) Decisions, rules, discoveries, gotchas as memories + KG triples (twin pattern). "
    "(2) Changed facts via kg_invalidate + kg_add. "
    "(3) New entities via kg_declare_entity. "
    "Then call diary_write — readable prose, delta-only (what changed since last entry), "
    "focused on decisions/status/big picture. Do NOT repeat commits, gotchas, or features "
    "already captured by finalize_intent. "
    "THEN KEEP WORKING. "
    "There is no 'wrapping the session' while work is pending. "
    "Do NOT offer to continue 'in a later session', do NOT summarize progress and stop, "
    "do NOT ask 'should I keep going?' or 'want me to pick this up next time?'. "
    "If the TodoWrite list has pending items, DO THEM — 100%. "
    "The user does not care about session boundaries or context limits — finish the work. "
    "Only pause when a tool call genuinely needs the user's answer (ambiguous requirement, "
    "missing credential, destructive action requiring consent). Everything else is your job "
    "to complete without asking."
)

PRECOMPACT_BLOCK_REASON = (
    "COMPACTION IMMINENT. "
    "First, finalize the active intent if one exists (mempalace_finalize_intent). "
    "Then persist ALL new knowledge before context is lost: "
    "(1) Decisions, rules, discoveries, gotchas as memories + KG triples (twin pattern). "
    "(2) Changed facts via kg_invalidate + kg_add. "
    "(3) New entities via kg_declare_entity. "
    "(4) Then diary_write — readable prose, delta-only, focused on decisions/status/big picture. "
    "Do NOT repeat what finalize_intent already captured. "
    "Be thorough on DECISIONS and PENDING ITEMS — after compaction, detailed context will be lost. "
    "THEN KEEP WORKING on the next pending task. "
    "Compaction is not a stopping point. Do NOT summarize and wait — continue execution "
    "so the compacted summary includes real progress, not a report of halted state."
)


def _sanitize_session_id(session_id: str) -> str:
    """Strip path-traversal characters. Returns "" for empty input.

    NO FALLBACK TO 'unknown' OR 'default'. A missing or fully-stripped
    session_id is a real error upstream — callers MUST decide whether to
    skip the operation, refuse, or log. We do NOT quietly substitute a
    shared name that every agent writes into (that caused the 2026-04-19
    cross-agent contamination deadlock).
    """
    if not session_id:
        return ""
    return re.sub(r"[^a-zA-Z0-9_-]", "", session_id)


def _effective_session_id(data: dict) -> str:
    """Return the session-id to use for intent-state scoping.

    Claude Code subagents (dispatched via the Task tool) make tool calls
    inside the SAME MCP session as the top-level conversation — same
    `session_id` in the hook payload. Without further disambiguation,
    subagent tool calls would mutate the parent's active-intent and
    pending-state files, causing the parent to see "phantom" pending
    enrichments left behind by a subagent's kg_declare_entity call.

    When the hook fires inside a subagent, the payload carries an extra
    `agent_id` field (unique per subagent invocation, stable across every
    tool call inside that invocation). We combine it with the base
    session_id to produce a composite sid so the subagent gets its OWN
    intent-state file, its own pending queue, and its own trace log.

    Format: ``<base_sid>__sub_<agent_id>`` when agent_id is present,
    else just ``<base_sid>``. The ``__sub_`` separator survives the
    sanitizer (which keeps `_`) and makes subagent sids visually
    distinguishable in logs and on disk.

    No `agent_id` in payload = top-level tool call = base sid. This is
    the common case; subagents are opt-in via the Task tool.
    """
    base = str(data.get("session_id", ""))
    agent_id = data.get("agent_id") or ""
    if not agent_id:
        return base
    # Sanitize each half independently so a malicious agent_id can't
    # smuggle path characters into the composite.
    safe_base = _sanitize_session_id(base) if base else ""
    safe_agent = _sanitize_session_id(str(agent_id))
    if not safe_base:
        return f"sub_{safe_agent}"
    return f"{safe_base}__sub_{safe_agent}"


_TRACE_DIR = Path(os.path.expanduser("~/.mempalace/hook_state"))


def _append_trace(session_id: str, tool_name: str, tool_input: dict):
    """Append a tool call to the execution trace for the current session.

    Lightweight: just tool name + abbreviated target + timestamp.
    Read by finalize_intent to create the trace memory.

    No-op when session_id is empty. NO FALLBACK to a shared default
    trace file — that would merge every agent's trace and make them
    unreadable.
    """
    safe_sid = _sanitize_session_id(session_id)
    if not safe_sid:
        return  # No session → no trace. Loud-by-absence.
    try:
        _TRACE_DIR.mkdir(parents=True, exist_ok=True)
        trace_file = _TRACE_DIR / f"execution_trace_{safe_sid}.jsonl"

        # Abbreviate target
        target = ""
        if tool_name in ("Edit", "Write", "Read"):
            target = tool_input.get("file_path") or ""
            # Keep just filename, not full path
            if "/" in target:
                target = target.rsplit("/", 1)[-1]
            elif "\\" in target:
                target = target.rsplit("\\", 1)[-1]
        elif tool_name == "Bash":
            cmd = tool_input.get("command") or ""
            target = cmd[:60]
        elif tool_name in ("Grep", "Glob"):
            target = tool_input.get("pattern") or tool_input.get("path") or ""
            target = target[:60]

        entry = json.dumps(
            {
                "ts": datetime.now().isoformat()[:19],
                "tool": tool_name,
                "target": target[:80],
            }
        )

        with open(trace_file, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except Exception:
        pass  # Non-fatal — trace is best-effort


def _count_human_messages(transcript_path: str) -> int:
    """Count human messages in a JSONL transcript, skipping command-messages."""
    path = Path(transcript_path).expanduser()
    if not path.is_file():
        return 0
    count = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    msg = entry.get("message", {})
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            if "<command-message>" in content:
                                continue
                        elif isinstance(content, list):
                            text = " ".join(
                                b.get("text", "") for b in content if isinstance(b, dict)
                            )
                            if "<command-message>" in text:
                                continue
                        count += 1
                    # Also handle Codex CLI transcript format
                    # {"type": "event_msg", "payload": {"type": "user_message", "message": "..."}}
                    elif entry.get("type") == "event_msg":
                        payload = entry.get("payload", {})
                        if isinstance(payload, dict) and payload.get("type") == "user_message":
                            msg_text = payload.get("message", "")
                            if isinstance(msg_text, str) and "<command-message>" not in msg_text:
                                count += 1
                except (json.JSONDecodeError, AttributeError):
                    pass
    except OSError:
        return 0
    return count


def _log(message: str):
    """Append to hook state log file. Forces UTF-8 so non-ASCII
    characters in reasons / markers don't crash the hook on Windows
    (whose default cp1252 encoding would raise UnicodeEncodeError)."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        log_path = STATE_DIR / "hook.log"
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
    except (OSError, UnicodeError):
        pass


def _output(data: dict):
    """Print JSON to stdout with consistent formatting (pretty-printed)."""
    print(json.dumps(data, indent=2, ensure_ascii=False))


def _maybe_auto_ingest():
    """If MEMPAL_DIR is set and exists, run mempalace mine in background."""
    mempal_dir = os.environ.get("MEMPAL_DIR", "")
    if mempal_dir and os.path.isdir(mempal_dir):
        try:
            log_path = STATE_DIR / "hook.log"
            with open(log_path, "a") as log_f:
                subprocess.Popen(
                    [sys.executable, "-m", "mempalace", "mine", mempal_dir],
                    stdout=log_f,
                    stderr=log_f,
                )
        except OSError:
            pass


SUPPORTED_HARNESSES = {"claude-code", "codex"}


def _parse_harness_input(data: dict, harness: str) -> dict:
    """Parse stdin JSON according to the harness type."""
    if harness not in SUPPORTED_HARNESSES:
        print(f"Unknown harness: {harness}", file=sys.stderr)
        sys.exit(1)
    return {
        # No 'unknown' default. Empty session_id propagates as empty — callers
        # decide what to do rather than all funneling to a shared 'unknown'
        # file that mixes agents.
        "session_id": _sanitize_session_id(str(data.get("session_id", ""))),
        "stop_hook_active": data.get("stop_hook_active", False),
        "transcript_path": str(data.get("transcript_path", "")),
    }


def hook_stop(data: dict, harness: str):
    """Stop hook: block every N messages for auto-save."""
    parsed = _parse_harness_input(data, harness)
    session_id = parsed["session_id"]
    stop_hook_active = parsed["stop_hook_active"]
    transcript_path = parsed["transcript_path"]

    # If already in a save cycle, let through (infinite-loop prevention)
    if str(stop_hook_active).lower() in ("true", "1", "yes"):
        _output({})
        return

    # Check for active unfinalised intent — always block with save reminder
    intent = _read_active_intent(session_id)
    if intent and intent.get("intent_id"):
        intent_type = intent.get("intent_type", "unknown")
        intent_desc = intent.get("description", "")[:80]
        _log(f"Stop BLOCK: active intent '{intent_type}' not finalized")
        _output(
            {
                "decision": "block",
                "reason": (
                    f"Active intent '{intent_type}' not finalized: {intent_desc}. "
                    f"Call mempalace_finalize_intent FIRST, then persist knowledge "
                    f"(memories + KG triples), then diary_write. "
                    f"Do NOT stop without finalizing — your work will be lost."
                ),
            }
        )
        return

    # Count human messages
    exchange_count = _count_human_messages(transcript_path)

    # Track last save point
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    last_save_file = STATE_DIR / f"{session_id}_last_save"
    last_save = 0
    if last_save_file.is_file():
        try:
            last_save = int(last_save_file.read_text().strip())
        except (ValueError, OSError):
            last_save = 0

    since_last = exchange_count - last_save

    _log(f"Session {session_id}: {exchange_count} exchanges, {since_last} since last save")

    if since_last >= SAVE_INTERVAL and exchange_count > 0:
        # Write the pending save marker — diary_write reads this to update
        # the last_save counter. Counter is NOT updated here to prevent
        # the dodge where agents ignore the save prompt and it never fires again.
        try:
            pending_save_file = STATE_DIR / f"{session_id}_pending_save"
            pending_save_file.write_text(str(exchange_count), encoding="utf-8")
        except OSError:
            pass

        _log(f"TRIGGERING SAVE at exchange {exchange_count}")

        # Optional: auto-ingest if MEMPAL_DIR is set
        _maybe_auto_ingest()

        _output({"decision": "block", "reason": STOP_BLOCK_REASON})
    else:
        _output({})


def hook_session_start(data: dict, harness: str):
    """Session start hook: initialize session tracking state."""
    parsed = _parse_harness_input(data, harness)
    session_id = parsed["session_id"]

    _log(f"SESSION START for session {session_id}")

    # Initialize session state directory
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    # Pass through — no blocking on session start
    _output({})


def hook_precompact(data: dict, harness: str):
    """Precompact hook: always block with comprehensive save instruction."""
    parsed = _parse_harness_input(data, harness)
    session_id = parsed["session_id"]

    _log(f"PRE-COMPACT triggered for session {session_id}")

    # Optional: auto-ingest synchronously before compaction (so memories land first)
    mempal_dir = os.environ.get("MEMPAL_DIR", "")
    if mempal_dir and os.path.isdir(mempal_dir):
        try:
            log_path = STATE_DIR / "hook.log"
            with open(log_path, "a") as log_f:
                subprocess.run(
                    [sys.executable, "-m", "mempalace", "mine", mempal_dir],
                    stdout=log_f,
                    stderr=log_f,
                    timeout=60,
                )
        except OSError:
            pass

    # Always block -- compaction = save everything
    _output({"decision": "block", "reason": PRECOMPACT_BLOCK_REASON})


INTENT_STATE_DIR = STATE_DIR

# Tools that are always allowed regardless of active intent
ALWAYS_ALLOWED_TOOLS = {
    # All mempalace MCP tools
    "mcp__plugin_mempalace_mempalace__mempalace_wake_up",
    # mempalace_add_drawer merged into mempalace_kg_declare_entity
    "mcp__plugin_mempalace_mempalace__mempalace_kg_delete_entity",
    "mcp__plugin_mempalace_mempalace__mempalace_kg_add",
    "mcp__plugin_mempalace_mempalace__mempalace_kg_add_batch",
    "mcp__plugin_mempalace_mempalace__mempalace_kg_query",
    "mcp__plugin_mempalace_mempalace__mempalace_kg_search",
    "mcp__plugin_mempalace_mempalace__mempalace_kg_declare_entity",
    "mcp__plugin_mempalace_mempalace__mempalace_kg_merge_entities",
    "mcp__plugin_mempalace_mempalace__mempalace_kg_invalidate",
    "mcp__plugin_mempalace_mempalace__mempalace_kg_timeline",
    "mcp__plugin_mempalace_mempalace__mempalace_kg_stats",
    "mcp__plugin_mempalace_mempalace__mempalace_kg_list_declared",
    "mcp__plugin_mempalace_mempalace__mempalace_kg_update_entity",  # replaces 3 legacy update tools
    "mcp__plugin_mempalace_mempalace__mempalace_declare_intent",
    "mcp__plugin_mempalace_mempalace__mempalace_active_intent",
    "mcp__plugin_mempalace_mempalace__mempalace_finalize_intent",
    "mcp__plugin_mempalace_mempalace__mempalace_extend_intent",
    "mcp__plugin_mempalace_mempalace__mempalace_resolve_conflicts",
    "mcp__plugin_mempalace_mempalace__mempalace_diary_write",
    "mcp__plugin_mempalace_mempalace__mempalace_diary_read",
    "mcp__plugin_mempalace_mempalace__mempalace_traverse",
    # mempalace_update_drawer_metadata merged into mempalace_kg_update_entity
    # Claude Code built-in tools that are safe/meta
    "Agent",
    "Skill",
    "ToolSearch",
    "TaskCreate",
    "TaskUpdate",
    "TaskGet",
    "TaskList",
    "TaskOutput",
    "TaskStop",
    "TodoWrite",
    "ExitPlanMode",
}


def _read_active_intent(session_id: str = None):
    """Read active intent from the session-scoped state file.

    Returns the stored dict or None. NO FALLBACK to a shared
    ``active_intent_default.json`` — that file is a cross-agent
    contamination vector and is forbidden by policy.

    If ``session_id`` is empty, returns None (which the caller surfaces
    as "no active intent declared", which is the correct guidance:
    declare one).
    """
    safe_sid = _sanitize_session_id(session_id or "")
    if not safe_sid:
        return None

    # Resolve the intent-state directory dynamically. STATE_DIR is patched
    # by tests via unittest.mock.patch; the module-level INTENT_STATE_DIR
    # alias captured the unpatched Path at import time, so reading through
    # it would bypass the test isolation.
    base_dir = STATE_DIR
    path = base_dir / f"active_intent_{safe_sid}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not data.get("intent_id"):
        return None
    return data


def _normalize_win_path(p: str) -> str:
    """Normalize a path-like string for the hook's scope matcher.

    - Expand ``~`` / ``~user`` against HOME so a scope like ``~/.mempalace/**``
      matches the absolute path the tool actually receives. Tests:
      test_blocking_escape_hatches.test_scope_tilde_expands_to_home.
    - Lowercase + forward-slash so Windows and POSIX paths compare equal.
    - Convert Git Bash mount format ``/d/foo`` to ``d:/foo``.
    """
    if not p:
        return ""
    # ~ expansion FIRST, before any normalization that would lose the tilde.
    p = os.path.expanduser(p)
    p = p.replace("\\", "/").lower()
    # Git Bash mount format: /d/foo -> d:/foo
    m = re.match(r"^/([a-z])/(.*)$", p)
    if m:
        p = f"{m.group(1)}:/{m.group(2)}"
    return p


def _parse_bash_commands(command: str) -> list:
    """Extract individual command keywords from a Bash command string.

    Uses bashlex (full Bash AST parser) to properly handle compound
    commands, pipes, subshells, process substitution, and all Bash
    syntax. Falls back to shlex (stdlib tokenizer) if bashlex fails,
    and to raw string splitting as last resort.

    Returns a list of command keywords (e.g. ['cd', 'git', 'pytest']).
    Each keyword is the first word of a CommandNode in the AST — the
    actual program being invoked.
    """
    # Primary: bashlex AST parser (handles pipes, subshells, etc.)
    try:
        import bashlex

        commands = []

        def _walk(node):
            """Recursively walk the AST and collect command keywords."""
            kind = node.kind
            if kind == "command" and hasattr(node, "parts"):
                # First WordNode in a CommandNode is the command keyword
                for part in node.parts:
                    if part.kind == "word":
                        commands.append(part.word)
                        break
            # Recurse into children (list, pipeline, compound, etc.)
            if hasattr(node, "parts"):
                for child in node.parts:
                    if hasattr(child, "kind"):
                        _walk(child)
            if hasattr(node, "list"):
                for child in node.list:
                    if hasattr(child, "kind"):
                        _walk(child)

        for tree in bashlex.parse(command):
            _walk(tree)
        return commands

    except Exception:
        pass

    # Fallback: shlex tokenizer (handles quoting but not structure)
    try:
        import shlex

        tokens = shlex.split(command, posix=True)
        if not tokens:
            return []
        operators = {"&&", "||", ";", "|"}
        commands = []
        expect_command = True
        for token in tokens:
            if token in operators:
                expect_command = True
                continue
            if expect_command:
                commands.append(token)
                expect_command = False
        return commands
    except Exception:
        pass

    # Last resort: raw first word
    return [command.split()[0]] if command.strip() else []


def _check_permission(tool_name: str, tool_input: dict, intent: dict) -> tuple:  # noqa: C901
    """Check if tool_name is permitted by the active intent.

    Returns (permitted: bool, reason: str).

    For Bash commands: uses shlex-based parsing to extract individual
    command keywords from compound commands (pipes, chains). Each keyword
    is checked against the declared command scopes independently. This
    catches cases where 'cd /tmp && rm -rf /' would be permitted by a
    scope that only allows 'cd' — now 'rm' is also checked.
    """
    permissions = intent.get("effective_permissions", [])

    # Extract the primary target from tool_input
    target = None
    if tool_name in ("Edit", "Write", "Read"):
        target = tool_input.get("file_path", "")
    elif tool_name == "Bash":
        target = tool_input.get("command", "")
    elif tool_name in ("Grep", "Glob"):
        # Only `path` is a filesystem target. `pattern` is a regex (Grep) or
        # glob expression (Glob) — those are content/name matchers, not paths,
        # so they must NOT be compared against path scopes. When `path` is
        # omitted the tool defaults to cwd, which is already inside the
        # declared path scope; leaving target empty lets the scope check
        # take the "no target to check" branch below and permit the call.
        target = tool_input.get("path", "")

    import fnmatch

    # Normalize target path (handles /d/ vs D:/ on Windows Git Bash)
    norm_target = _normalize_win_path(target) if target else ""

    # For Bash: parse compound commands and check EACH keyword against scopes.
    # A compound command like "cd D:/foo && git add bar" must match scopes
    # for BOTH 'cd' and 'git add'. If ANY parsed command keyword doesn't
    # match any Bash scope, the whole command is denied.
    if tool_name == "Bash" and target:
        parsed_commands = _parse_bash_commands(target)
        bash_scopes = [
            _normalize_win_path(p.get("scope", "*")) for p in permissions if p["tool"] == "Bash"
        ]
        if any(s == "*" for s in bash_scopes):
            return True, f"Bash is unrestricted in intent '{intent['intent_type']}'"

        for cmd_keyword in parsed_commands:
            norm_cmd = _normalize_win_path(cmd_keyword)
            matched = False
            for scope in bash_scopes:
                if scope in norm_target or fnmatch.fnmatch(norm_target, scope):
                    matched = True
                    break
                # Also check just the command keyword against the scope
                if scope in norm_cmd or fnmatch.fnmatch(norm_cmd, scope):
                    matched = True
                    break
            if not matched:
                # Fall through to the general denial path below
                break
        else:
            # All parsed commands matched at least one scope
            if parsed_commands:
                return True, f"Bash commands {parsed_commands} all match declared scopes"

    for perm in permissions:
        perm_tool = perm["tool"]
        # Support wildcard tool patterns (e.g. mcp__playwright__*)
        if perm_tool == tool_name or ("*" in perm_tool and fnmatch.fnmatch(tool_name, perm_tool)):
            scope = perm.get("scope", "*")
            if scope == "*":
                return True, f"{tool_name} is unrestricted in intent '{intent['intent_type']}'"
            # Normalize scope (same /d/ vs D:/ handling)
            norm_scope = _normalize_win_path(scope)
            # A scope like "d:/foo/**" should match "d:/foo" itself AND any
            # descendant. fnmatch alone rejects the bare parent because the
            # trailing "/" in the pattern requires "/" in the target.
            parent_scope = re.sub(r"/\*+$", "", norm_scope)
            # Scoped — check if target matches scope
            if norm_target and (
                norm_scope in norm_target  # direct substring (file path in full path)
                or fnmatch.fnmatch(norm_target, norm_scope)  # glob pattern
                or norm_target == parent_scope  # bare parent of a /** scope
            ):
                return True, f"{tool_name} permitted on '{target}' (matches scope '{scope}')"
            elif not norm_target:
                return True, f"{tool_name} is scoped (no target to check)"
            # Keep checking other permissions for this tool
            continue

    permitted_tools = sorted(set(p["tool"] for p in permissions))

    # Build helpful error with intent hierarchy (if available in state file)
    hierarchy = intent.get("intent_hierarchy", [])
    matching_types = [h for h in hierarchy if tool_name in h.get("tools", [])]

    error_parts = [
        f"Tool '{tool_name}' not permitted by active intent '{intent['intent_type']}'.",
        f"Permitted tools: {permitted_tools}.",
        "",
    ]

    # Score: Context-rank (pre-computed at declare time via 3-channel
    # kg_search) first, fallback to importance + agent affinity for
    # ties and unranked candidates. Hooks stay dep-free — the ranking
    # is already baked into intent_hierarchy by declare_intent.
    current_agent = intent.get("agent", "")

    def _score_intent_type(t):
        imp = t.get("importance", 3)
        agent_boost = 2 if (current_agent and t.get("added_by") == current_agent) else 0
        # Context-rank bonus: ranked types get a big boost, ordered by
        # rank (rank 0 > rank 1 > …). Unranked → 0.
        rank = t.get("context_rank")
        if rank is not None:
            context_boost = max(0.0, 20.0 - float(rank))
        else:
            context_boost = 0.0
        return float(imp + agent_boost + context_boost)

    # Adaptive-K with gap detection. Inlined to avoid importing scoring.py
    # in this hot subprocess path. Same logic as scoring.adaptive_k: sorted-descending
    # scores, find the largest relative gap, cut there. Falls back to max_k when
    # scores are uniform (no clear cliff).
    def _adaptive_k(scores, max_k, min_k=1, gap_threshold=0.25):
        if not scores:
            return 0
        if len(scores) == 1:
            return 1
        sorted_scores = sorted(scores, reverse=True)
        max_k = max(min_k, min(max_k, len(sorted_scores)))
        best_k = max_k
        biggest_rel_gap = 0.0
        for i in range(min_k, max_k):
            top = sorted_scores[i - 1]
            nxt = sorted_scores[i]
            denom = abs(top) if abs(top) > 1e-9 else 1.0
            rel_gap = (top - nxt) / denom
            if rel_gap > biggest_rel_gap and rel_gap >= gap_threshold:
                biggest_rel_gap = rel_gap
                best_k = i
        return best_k

    if matching_types:
        matching_types.sort(key=_score_intent_type, reverse=True)
        scores = [_score_intent_type(t) for t in matching_types]
        # Cap at 10 for the matching-types list (was fixed 5).
        k = _adaptive_k(scores, max_k=10, min_k=3)
        shown = matching_types[:k]
        error_parts.append(f"Intent types that already permit '{tool_name}' (Context-ranked):")
        for m in shown:
            error_parts.append(f"  - {m['id']} (is_a {m['parent']})")
        if len(matching_types) > k:
            error_parts.append(f"  ... and {len(matching_types) - k} more")
        error_parts.append("")
    else:
        # No existing type grants this tool — fall back to the four
        # canonical parents for new-type declaration. We intentionally
        # do NOT dump the full 48-type hierarchy: that list grows
        # without bound and the canonical parents are the right
        # picker for 99% of cases.
        error_parts.append(
            f"No declared intent type currently permits '{tool_name}'. "
            "Declare a new one, inheriting from one of the canonical parents:"
        )
        error_parts.append("  - inspect   (Read, Grep, Glob)")
        error_parts.append("  - modify    (Edit, Write, Read, Grep, Glob)")
        error_parts.append("  - execute   (Bash, Read, Grep, Glob)")
        error_parts.append("  - communicate (Bash, WebFetch, WebSearch, Read, Grep, Glob)")
        error_parts.append("")

    # Build creation guide with scoped example
    parts = tool_name.split("__")
    if len(parts) >= 3:
        wildcard = "__".join(parts[:2]) + "__*"
        tool_example = f'{{"tool": "{wildcard}", "scope": "<pattern>"}}'
    else:
        tool_example = f'{{"tool": "{tool_name}", "scope": "<pattern>"}}'

    error_parts.extend(
        [
            f"To create a NEW intent type that includes '{tool_name}':",
            "  Pick the CLOSEST parent from the list above.",
            "  Tools are ADDITIVE — only specify what the parent DOESN'T have.",
            "  Use wildcards for MCP tool groups (e.g. mcp__playwright__*).",
            "  Scope must be specific (file patterns, command patterns).",
            '  "*" scope requires user approval (user_approved_star_scope=true).',
            "",
            "1. kg_declare_entity(",
            '     name="<your_type>", kind="class", importance=4,',
            '     added_by="<your_agent>",',
            '     context={"queries": ["<what this action does>", "<another angle>"],',
            '              "keywords": ["<term1>", "<term2>"]},',
            '     properties={"rules_profile": {',
            '       "slots": {"subject": {"classes": ["thing"], "required": true}},',
            f'       "tool_permissions": [{tool_example}]',
            "     }}",
            "   )",
            '2. kg_add(subject="<your_type>", predicate="is_a", object="<parent>",',
            '          context={"queries": [...], "keywords": [...]})',
            '3. mempalace_declare_intent(intent_type="<your_type>", slots={...},',
            '       context={"queries": [...], "keywords": [...]},',
            '       agent="<your_agent>", budget={"Read": 5, "Edit": 3})',
        ]
    )

    return False, "\n".join(error_parts)


#
# ╔═══════════════════════════════════════════════════════════════════╗
# ║  BREAK-GLASS HOOK BYPASS                                          ║
# ║                                                                   ║
# ║  If the file  ~/.mempalace/HOOK_BYPASS_USER_ONLY  exists, every   ║
# ║  permissionDecision returned by the hook is forced to "allow"     ║
# ║  regardless of what the normal gate logic would have decided.     ║
# ║  The gate still RUNS and the would-have-been reason is still      ║
# ║  composed and logged, so the user can observe what was bypassed.  ║
# ║                                                                   ║
# ║  THE BYPASS FILE MUST ONLY EVER BE CREATED OR DELETED BY THE      ║
# ║  HUMAN USER — NEVER BY AN AGENT / SUBAGENT / TOOL CALL. Creating  ║
# ║  it disables every permission control in this codebase and is     ║
# ║  intended exclusively for recovering from a wedged mempalace      ║
# ║  state where the agent can't even Edit a source file to fix the   ║
# ║  bug that's causing the wedge. Agent authors: DO NOT write code   ║
# ║  that touches this path. DO NOT ask to touch it. If you see it on ║
# ║  disk, treat it as a sign the user is mid-debug and LEAVE IT      ║
# ║  ALONE.                                                           ║
# ╚═══════════════════════════════════════════════════════════════════╝
_BYPASS_FILE = Path(os.path.expanduser("~/.mempalace/HOOK_BYPASS_USER_ONLY"))


def _bypass_active() -> bool:
    """True iff the user-created bypass file exists."""
    try:
        return _BYPASS_FILE.is_file()
    except OSError:
        return False


def _apply_bypass_if_active(response: dict, denied_reason: str = "") -> dict:
    """If the bypass file exists, downgrade any ``deny`` decision to
    ``allow`` while stamping a loud indicator into the response and the
    log so the user sees the bypass firing in real time.

    Returns the (possibly mutated) response unchanged when no bypass is
    active OR when the decision was already ``allow``.
    """
    hso = response.get("hookSpecificOutput") or {}
    if hso.get("permissionDecision") != "deny":
        return response
    if not _bypass_active():
        return response
    tool = hso.get("toolName") or ""
    _log(
        "[!] HOOK BYPASS ACTIVE (user-created "
        "~/.mempalace/HOOK_BYPASS_USER_ONLY) -- would have DENIED "
        f"{tool}: {denied_reason[:200]}"
    )
    hso["permissionDecision"] = "allow"
    hso["permissionDecisionReason"] = (
        "[!] HOOK BYPASS ACTIVE -- ~/.mempalace/HOOK_BYPASS_USER_ONLY exists "
        "(user-created). This tool would normally have been DENIED. "
        f"Original deny reason: {denied_reason[:500]}"
    )
    response["hookSpecificOutput"] = hso
    return response


def hook_pretooluse(data: dict, harness: str):
    """PreToolUse hook: enforce intent-based tool permissions."""
    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})
    # Effective sid folds in the `agent_id` field that Claude Code adds to
    # subagent tool calls, so subagent state is isolated from parent state
    # on disk. See _effective_session_id for the rationale and format.
    session_id = _effective_session_id(data)

    # For mempalace MCP tools: always allow + inject sessionId via updatedInput
    # Match any plugin ID pattern — versioned IDs vary by install
    # (e.g. mcp__plugin_mempalace_mempalace__*, mcp__plugin_3_0_14_mempalace__*)
    is_mempalace_mcp = tool_name.startswith("mcp__") and "__mempalace_" in tool_name
    if tool_name in ALWAYS_ALLOWED_TOOLS or is_mempalace_mcp:
        response = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
            },
        }
        # Inject sessionId into MCP tool input so the server knows the session.
        # The composite sid (when the call is from a subagent) means the server
        # scopes _STATE + on-disk pending/active-intent to the subagent instance.
        if is_mempalace_mcp and session_id:
            updated = dict(tool_input) if tool_input else {}
            updated["sessionId"] = session_id
            response["hookSpecificOutput"]["updatedInput"] = updated
        _output(response)
        return

    # For other always-allowed tools (Agent, Task*, Skill, etc.) — pass through
    if tool_name in ALWAYS_ALLOWED_TOOLS:
        _output(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                }
            }
        )
        return

    # Read active intent from session-scoped state file
    intent = _read_active_intent(session_id)

    if not intent:
        # No active intent — deny with guidance
        _log(f"PreToolUse DENY {tool_name}: no active intent")
        reason = (
            f"No active intent declared. You must call mempalace_declare_intent "
            f"before using '{tool_name}'. Example: "
            "mempalace_declare_intent(intent_type='modify', "
            'slots={"files": ["target_file"]}, '
            'context={"queries": ["<what you plan to do>", "<another angle>"], '
            '"keywords": ["<term1>", "<term2>"]}, '
            "agent='<your_agent>', "
            'budget={"Read": 5, "Edit": 3})'
        )
        _output(
            _apply_bypass_if_active(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": reason,
                    }
                },
                denied_reason=reason,
            )
        )
        return

    # Check for pending conflicts — block non-mempalace tools until resolved
    pending_conflicts = intent.get("pending_conflicts", [])
    if pending_conflicts:
        _log(f"PreToolUse DENY {tool_name}: {len(pending_conflicts)} pending conflicts")
        reason = (
            f"{len(pending_conflicts)} conflicts pending. You MUST resolve ALL "
            f"conflicts before continuing. Call mempalace_resolve_conflicts with "
            f"actions for each conflict: invalidate, merge, keep, or skip."
        )
        _output(
            _apply_bypass_if_active(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": reason,
                    }
                },
                denied_reason=reason,
            )
        )
        return

    # Check for pending enrichments — block non-mempalace tools until resolved
    pending_enrichments = intent.get("pending_enrichments", [])
    if pending_enrichments:
        _log(f"PreToolUse DENY {tool_name}: {len(pending_enrichments)} pending enrichments")
        reason = (
            f"{len(pending_enrichments)} graph enrichment tasks pending. "
            f"For each: call kg_add(subject, predicate, object) to create the "
            f"edge with a predicate you choose, then call "
            f"mempalace_resolve_enrichments to mark done. Or reject with reason."
        )
        _output(
            _apply_bypass_if_active(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": reason,
                    }
                },
                denied_reason=reason,
            )
        )
        return

    permitted, reason = _check_permission(tool_name, tool_input, intent)

    if permitted:
        # Check budget — deny if exhausted for this tool
        budget = intent.get("budget", {})
        used = intent.get("used", {})
        if budget:
            tool_budget = budget.get(tool_name, 0)
            tool_used = used.get(tool_name, 0)
            if tool_budget == 0:
                # Tool not in budget — deny
                _log(f"PreToolUse DENY {tool_name}: not in budget")
                not_in_budget_reason = (
                    f"Tool '{tool_name}' not in declared budget. "
                    f"Current budget: {budget}. "
                    f"Either extend (mempalace_extend_intent) or "
                    f"finalize and redeclare with this tool in the budget."
                )
                _output(
                    _apply_bypass_if_active(
                        {
                            "hookSpecificOutput": {
                                "hookEventName": "PreToolUse",
                                "permissionDecision": "deny",
                                "permissionDecisionReason": not_in_budget_reason,
                            }
                        },
                        denied_reason=not_in_budget_reason,
                    )
                )
                return
            if tool_used >= tool_budget:
                remaining = {k: budget.get(k, 0) - used.get(k, 0) for k in budget}
                _log(f"PreToolUse DENY {tool_name}: budget exhausted ({tool_used}/{tool_budget})")
                budget_reason = (
                    f"Budget exhausted for '{tool_name}': used {tool_used}/{tool_budget}. "
                    f"Remaining budget: {remaining}. "
                    f"If you are STILL working on the SAME task, extend: "
                    f"mempalace_extend_intent(budget={{'{tool_name}': N}}). "
                    f"If you are switching to a DIFFERENT task, you MUST finalize "
                    f"first (mempalace_finalize_intent) then declare a new intent."
                )
                _output(
                    _apply_bypass_if_active(
                        {
                            "hookSpecificOutput": {
                                "hookEventName": "PreToolUse",
                                "permissionDecision": "deny",
                                "permissionDecisionReason": budget_reason,
                            }
                        },
                        denied_reason=budget_reason,
                    )
                )
                return
            # Budget OK — increment used count and persist
            used[tool_name] = tool_used + 1
            intent["used"] = used
            try:
                state_path = (
                    INTENT_STATE_DIR / f"active_intent_{_sanitize_session_id(session_id)}.json"
                )
                if state_path.is_file():
                    state_path.write_text(json.dumps(intent, indent=2), encoding="utf-8")
            except Exception:
                pass  # Non-fatal

        _log(f"PreToolUse ALLOW {tool_name}: {reason}")
        # Accumulate execution trace for finalize_intent
        _append_trace(session_id, tool_name, tool_input)
        _output(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                }
            }
        )
    else:
        _log(f"PreToolUse DENY {tool_name}: {reason}")
        _output(
            _apply_bypass_if_active(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": reason,
                    }
                },
                denied_reason=reason,
            )
        )


def run_hook(hook_name: str, harness: str):
    """Main entry point: read stdin JSON, dispatch to hook handler."""
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        _log("WARNING: Failed to parse stdin JSON, proceeding with empty data")
        data = {}

    hooks = {
        "session-start": hook_session_start,
        "stop": hook_stop,
        "precompact": hook_precompact,
        "pretooluse": hook_pretooluse,
    }

    handler = hooks.get(hook_name)
    if handler is None:
        print(f"Unknown hook: {hook_name}", file=sys.stderr)
        sys.exit(1)

    handler(data, harness)
