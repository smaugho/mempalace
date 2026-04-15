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
    "(1) Decisions, rules, discoveries, gotchas as drawers + KG triples (twin pattern). "
    "(2) Changed facts via kg_invalidate + kg_add. "
    "(3) New entities via kg_declare_entity. "
    "Then call diary_write — readable prose, delta-only (what changed since last entry), "
    "focused on decisions/status/big picture. Do NOT repeat commits, gotchas, or features "
    "already captured by finalize_intent. "
    "Continue working after saving — do NOT ask 'what's next?' or pause for permission. "
    "If there are pending tasks, keep going. Only pause for genuine blockers that need user input."
)

PRECOMPACT_BLOCK_REASON = (
    "COMPACTION IMMINENT. "
    "First, finalize the active intent if one exists (mempalace_finalize_intent). "
    "Then persist ALL new knowledge before context is lost: "
    "(1) Decisions, rules, discoveries, gotchas as drawers + KG triples (twin pattern). "
    "(2) Changed facts via kg_invalidate + kg_add. "
    "(3) New entities via kg_declare_entity. "
    "(4) Then diary_write — readable prose, delta-only, focused on decisions/status/big picture. "
    "Do NOT repeat what finalize_intent already captured. "
    "Be thorough on DECISIONS and PENDING ITEMS \u2014 after compaction, detailed context will be lost."
)


def _sanitize_session_id(session_id: str) -> str:
    """Only allow alnum, dash, underscore to prevent path traversal."""
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", session_id)
    return sanitized or "unknown"


_TRACE_DIR = Path(os.path.expanduser("~/.mempalace/hook_state"))


def _append_trace(session_id: str, tool_name: str, tool_input: dict):
    """Append a tool call to the execution trace for the current session.

    Lightweight: just tool name + abbreviated target + timestamp.
    Read by finalize_intent to create the trace drawer.
    """
    try:
        _TRACE_DIR.mkdir(parents=True, exist_ok=True)
        safe_sid = _sanitize_session_id(session_id) if session_id else "default"
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
    """Append to hook state log file."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        log_path = STATE_DIR / "hook.log"
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(log_path, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
    except OSError:
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
        "session_id": _sanitize_session_id(str(data.get("session_id", "unknown"))),
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
                    f"(drawers + KG triples), then diary_write. "
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
        # Update last save point
        try:
            last_save_file.write_text(str(exchange_count), encoding="utf-8")
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
    "mcp__plugin_mempalace_mempalace__mempalace_search",
    "mcp__plugin_mempalace_mempalace__mempalace_status",
    "mcp__plugin_mempalace_mempalace__mempalace_add_drawer",
    "mcp__plugin_mempalace_mempalace__mempalace_delete_drawer",
    "mcp__plugin_mempalace_mempalace__mempalace_check_duplicate",
    "mcp__plugin_mempalace_mempalace__mempalace_kg_add",
    "mcp__plugin_mempalace_mempalace__mempalace_kg_query",
    "mcp__plugin_mempalace_mempalace__mempalace_kg_search",
    "mcp__plugin_mempalace_mempalace__mempalace_kg_declare_entity",
    "mcp__plugin_mempalace_mempalace__mempalace_kg_merge_entities",
    "mcp__plugin_mempalace_mempalace__mempalace_kg_entity_info",
    "mcp__plugin_mempalace_mempalace__mempalace_kg_invalidate",
    "mcp__plugin_mempalace_mempalace__mempalace_kg_timeline",
    "mcp__plugin_mempalace_mempalace__mempalace_kg_stats",
    "mcp__plugin_mempalace_mempalace__mempalace_kg_list_declared",
    "mcp__plugin_mempalace_mempalace__mempalace_kg_update_entity_description",
    "mcp__plugin_mempalace_mempalace__mempalace_kg_update_predicate_constraints",
    "mcp__plugin_mempalace_mempalace__mempalace_declare_intent",
    "mcp__plugin_mempalace_mempalace__mempalace_active_intent",
    "mcp__plugin_mempalace_mempalace__mempalace_check_tool_permission",
    "mcp__plugin_mempalace_mempalace__mempalace_diary_write",
    "mcp__plugin_mempalace_mempalace__mempalace_diary_read",
    "mcp__plugin_mempalace_mempalace__mempalace_list_wings",
    "mcp__plugin_mempalace_mempalace__mempalace_list_rooms",
    "mcp__plugin_mempalace_mempalace__mempalace_get_taxonomy",
    "mcp__plugin_mempalace_mempalace__mempalace_get_aaak_spec",
    "mcp__plugin_mempalace_mempalace__mempalace_traverse",
    "mcp__plugin_mempalace_mempalace__mempalace_find_tunnels",
    "mcp__plugin_mempalace_mempalace__mempalace_graph_stats",
    "mcp__plugin_mempalace_mempalace__mempalace_update_drawer_metadata",
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
}


def _read_active_intent(session_id: str = None):
    """Read active intent from session-scoped state file. Returns dict or None.

    IMPORTANT: Only reads the session-specific file. No fallback to default.json
    to prevent cross-session intent leakage between agents.
    """
    if not session_id:
        return None  # No session = no intent, never fall back to default

    path = INTENT_STATE_DIR / f"active_intent_{_sanitize_session_id(session_id)}.json"
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("intent_id"):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return None


def _normalize_win_path(p: str) -> str:
    """Normalize Windows path formats: /d/foo -> d:/foo, D:\\foo -> d:/foo."""
    p = p.replace("\\", "/").lower()
    # Git Bash mount format: /d/foo -> d:/foo
    m = re.match(r"^/([a-z])/(.*)$", p)
    if m:
        p = f"{m.group(1)}:/{m.group(2)}"
    return p


def _check_permission(tool_name: str, tool_input: dict, intent: dict) -> tuple:
    """Check if tool_name is permitted by the active intent.

    Returns (permitted: bool, reason: str).
    """
    permissions = intent.get("effective_permissions", [])

    # Extract the primary target from tool_input
    target = None
    if tool_name in ("Edit", "Write", "Read"):
        target = tool_input.get("file_path", "")
    elif tool_name == "Bash":
        target = tool_input.get("command", "")
    elif tool_name in ("Grep", "Glob"):
        target = tool_input.get("path", "") or tool_input.get("pattern", "")

    import fnmatch

    # Normalize target path (handles /d/ vs D:/ on Windows Git Bash)
    norm_target = _normalize_win_path(target) if target else ""

    for perm in permissions:
        perm_tool = perm["tool"]
        # Support wildcard tool patterns (e.g. mcp__playwright__*)
        if perm_tool == tool_name or ("*" in perm_tool and fnmatch.fnmatch(tool_name, perm_tool)):
            scope = perm.get("scope", "*")
            if scope == "*":
                return True, f"{tool_name} is unrestricted in intent '{intent['intent_type']}'"
            # Normalize scope (same /d/ vs D:/ handling)
            norm_scope = _normalize_win_path(scope)
            # Scoped — check if target matches scope
            if norm_target and (
                norm_scope in norm_target  # direct substring (file path in full path)
                or fnmatch.fnmatch(norm_target, norm_scope)  # glob pattern
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

    # Rank by agent affinity + importance (same scoring logic as everywhere else)
    current_agent = intent.get("agent", "")

    def _rank_intent_type(t):
        imp = t.get("importance", 3)
        agent_boost = 2 if (current_agent and t.get("added_by") == current_agent) else 0
        return -(imp + agent_boost)  # negative for ascending sort

    if matching_types:
        matching_types.sort(key=_rank_intent_type)
        shown = matching_types[:5]
        error_parts.append(f"Existing intent types that already permit '{tool_name}':")
        for m in shown:
            error_parts.append(f"  - {m['id']} (is_a {m['parent']})")
        if len(matching_types) > 5:
            error_parts.append(f"  ... and {len(matching_types) - 5} more")
        error_parts.append("")

    if hierarchy:
        hierarchy.sort(key=_rank_intent_type)
        shown = hierarchy[:10]
        error_parts.append(f"Intent types (top {len(shown)} of {len(hierarchy)}):")
        for h in shown:
            tools_str = ", ".join(h.get("tools", [])) or "inherits from parent"
            error_parts.append(f"  - {h['id']} (is_a {h['parent']}): {tools_str}")
        if len(hierarchy) > 10:
            error_parts.append(
                f"  ... and {len(hierarchy) - 10} more (use mempalace_kg_search to find specific types)"
            )
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
            "  Pick the CLOSEST parent from the hierarchy above.",
            "  Tools are ADDITIVE — only specify what the parent DOESN'T have.",
            "  Use wildcards for MCP tool groups (e.g. mcp__playwright__*).",
            "  Scope must be specific (file patterns, command patterns).",
            '  "*" scope requires user approval (user_approved_star_scope=true).',
            "",
            "1. kg_declare_entity(",
            '     name="<your_type>", kind="class", importance=4,',
            '     description="<what this action does>",',
            '     properties={"rules_profile": {',
            '       "slots": {"subject": {"classes": ["thing"], "required": true}},',
            f'       "tool_permissions": [{tool_example}]',
            "     }}",
            "   )",
            '2. kg_add(subject="<your_type>", predicate="is_a", object="<parent>")',
            '3. mempalace_declare_intent(intent_type="<your_type>", slots={...})',
        ]
    )

    return False, "\n".join(error_parts)


def hook_pretooluse(data: dict, harness: str):
    """PreToolUse hook: enforce intent-based tool permissions."""
    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})
    session_id = data.get("session_id", "")

    # For mempalace MCP tools: always allow + inject sessionId via updatedInput
    if tool_name in ALWAYS_ALLOWED_TOOLS or tool_name.startswith("mcp__plugin_mempalace"):
        response = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
            },
        }
        # Inject sessionId into MCP tool input so the server knows the session
        if tool_name.startswith("mcp__plugin_mempalace") and session_id:
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
        _output(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"No active intent declared. You must call mempalace_declare_intent "
                        f"before using '{tool_name}'. Example: "
                        f'mempalace_declare_intent(intent_type=\'modify\', slots={{"files": ["target_file"]}}, '
                        f"description='what you plan to do')"
                    ),
                }
            }
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
                _output(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": (
                                f"Tool '{tool_name}' not in declared budget. "
                                f"Current budget: {budget}. "
                                f"Either extend (mempalace_extend_intent) or "
                                f"finalize and redeclare with this tool in the budget."
                            ),
                        }
                    }
                )
                return
            if tool_used >= tool_budget:
                remaining = {k: budget.get(k, 0) - used.get(k, 0) for k in budget}
                _log(f"PreToolUse DENY {tool_name}: budget exhausted ({tool_used}/{tool_budget})")
                _output(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": (
                                f"Budget exhausted for '{tool_name}': used {tool_used}/{tool_budget}. "
                                f"Remaining budget: {remaining}. "
                                f"Call mempalace_extend_intent(budget={{'{tool_name}': N}}) to add more, "
                                f"or finalize this intent and redeclare."
                            ),
                        }
                    }
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
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
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
