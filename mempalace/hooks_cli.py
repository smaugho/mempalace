"""
Hook logic for MemPalace — Python implementation of session-start, stop, and precompact hooks.

Reads JSON from stdin, outputs JSON to stdout.
Supported hooks: session-start / sessionstart, stop, precompact, pretooluse, userpromptsubmit
Supported harnesses: claude-code, codex (extensible to cursor, gemini, etc.)

Context-as-entity note (P1 of docs/context_as_entity_redesign_plan.md):
hooks_cli is downstream of declare_operation. It NEVER emits its own
context entity — the three emit sites are tool_declare_intent,
tool_declare_operation, and tool_kg_search. The PreToolUse hook pops a
pending_operation_cue whose active_context_id was minted by
declare_operation; all hook-side retrieval inherits that id. Writers
triggered by the subsequent tool call read active_context_id from the
active_intent dict (see _active_context_id in mcp_server.py).
"""

import hashlib
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

NEVER_STOP_BLOCK_REASON = (
    "NO-STOP RULE: you cannot stop this session until a `wrap_up_session` "
    "intent has been declared, executed, and finalized with outcome=success "
    "as the LAST finalized intent in this session.\n\n"
    "To stop cleanly, do this RIGHT NOW:\n"
    "  1. Ensure TodoWrite is empty (every item completed) and no intent is active.\n"
    "  2. Declare: `mempalace_declare_intent(intent_type='wrap_up_session', "
    "slots={'subject': ['<session_topic_entity>']}, ...)` — use an existing "
    "concept entity or declare one.\n"
    "  3. Run at least 2 `mempalace_kg_search` calls against pending-work "
    "patterns (e.g. 'pending', 'TODO', 'unresolved', 'not yet', 'next step') "
    "to verify nothing is left. Inspect the hits.\n"
    "  4. `mempalace_finalize_intent(slug='wrap-up-<topic>-<date>', "
    "outcome='success', summary='<analysis findings>', memory_feedback=[...])`.\n\n"
    "If you actually have more work, DO IT instead of trying to stop. "
    "If you have a real blocker that needs user input, use `AskUserQuestion` "
    "(pauses for input, does not end the session). "
    "Questions like 'should I continue?' / 'what next?' / 'anything else?' "
    "are NOT valid blockers \u2014 they are you being lazy. Solve priorities instead."
)


PRECOMPACT_WARNING_MESSAGE = (
    "COMPACTION IMMINENT — this message is a WARNING, not a block. "
    "Compaction will proceed regardless. Before context is lost, persist what matters: "
    "(1) Finalize the active intent if one exists (mempalace_finalize_intent). "
    "(2) Decisions, rules, discoveries, gotchas as memories + KG triples (twin pattern). "
    "(3) Changed facts via kg_invalidate + kg_add. "
    "(4) New entities via kg_declare_entity. "
    "(5) Then diary_write — readable prose, delta-only, focused on decisions/status/big picture. "
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
    pending-state files, causing the parent to see phantom pending
    state left behind by a subagent's tool call.

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


# S1 execution-trace constants — tool-agnostic trace entries store
# truncated tool_input + the operation's context_id, so finalize_intent
# can promote rated entries to kind='operation' entities and attach
# them to the right context via performed_well / performed_poorly.
# Reference: arXiv 2512.18950 (Operation tier of hierarchical
# procedural memory); Leontiev 1981 (Activity Theory AAO).
#
# Budget discipline: per-field char cap + whole-payload byte cap.
# Overflow collapses to a truncated prefix plus sha12 fingerprint so
# identical-in-full args collide deterministically across sessions
# (useful for gardener-side clustering in S3).
_OP_TRACE_FIELD_CHARS = 400
_OP_TRACE_PAYLOAD_BYTES = 2048


def _truncate_op_args(tool_input):
    """Tool-agnostic truncation of a tool_input dict for the exec trace.

    Each string field capped at _OP_TRACE_FIELD_CHARS; long strings get
    '<sha12:HASH>' appended so collisions are deterministic. Whole
    payload capped at _OP_TRACE_PAYLOAD_BYTES; oversize payloads get a
    second-pass trim (200 chars/field) plus a top-level
    '__truncated_sha12__' marker carrying a hash of the full payload.

    Works for any tool (Bash, Edit, Write, Read, Grep, Glob, MCP,
    future) without per-tool branches.
    """
    if not isinstance(tool_input, dict):
        return {}
    out = {}
    for k, v in tool_input.items():
        if isinstance(v, str) and len(v) > _OP_TRACE_FIELD_CHARS:
            h = hashlib.sha256(v.encode("utf-8", errors="replace")).hexdigest()[:12]
            out[k] = v[:_OP_TRACE_FIELD_CHARS] + f"<sha12:{h}>"
        else:
            out[k] = v
    try:
        js = json.dumps(out, default=str)
    except Exception:
        return {"__truncated_sha12__": "serialize_failed"}
    if len(js.encode("utf-8")) <= _OP_TRACE_PAYLOAD_BYTES:
        return out
    full_h = hashlib.sha256(js.encode("utf-8", errors="replace")).hexdigest()[:12]
    trimmed = {}
    for k, v in out.items():
        if isinstance(v, str) and len(v) > 200:
            trimmed[k] = v[:200] + f"<sha12:{full_h}>"
        else:
            trimmed[k] = v
    trimmed["__truncated_sha12__"] = full_h
    return trimmed


def _append_trace(
    session_id: str,
    tool_name: str,
    tool_input: dict,
    context_id: str = "",
):
    """Append a tool call to the execution trace for the current session.

    Tool-agnostic: stores {ts, tool, context_id, args} where `args` is
    the full tool_input run through _truncate_op_args. Read by
    finalize_intent to promote rated entries into kind='operation'
    entities attached to `context_id` via performed_well /
    performed_poorly edges.

    `context_id` should be the active_context_id from the matching
    pending_operation_cue (minted by declare_operation). Missing
    context_id means the operation can still be traced but won't be
    promotable at finalize (provenance loss, not a crash).

    No-op when session_id is empty. NO FALLBACK to a shared default
    trace file — that would merge every agent's trace and make them
    unreadable.
    """
    safe_sid = _sanitize_session_id(session_id)
    if not safe_sid:
        return  # No session, no trace. Loud-by-absence.
    try:
        _TRACE_DIR.mkdir(parents=True, exist_ok=True)
        trace_file = _TRACE_DIR / f"execution_trace_{safe_sid}.jsonl"
        entry = json.dumps(
            {
                "ts": datetime.now().isoformat()[:19],
                "tool": tool_name,
                "context_id": context_id or "",
                "args": _truncate_op_args(tool_input),
            }
        )
        with open(trace_file, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except Exception as _e:
        # Trace-append is best-effort but not silent. Record so a
        # broken trace path is visible next turn.
        _record_hook_error("_append_trace", _e)


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


# ── No-silent-failure infrastructure ─────────────────────────────────
# Every exception in hook-layer code MUST be recorded here AND surfaced
# to the agent on its next turn. The prior pattern (_log + return empty)
# hid real bugs \u2014 the Phase 2b Config-vs-MempalaceConfig import error
# failed on every tool call for days without visibility. No more.

HOOK_ERROR_LOG_NAME = "hook_errors.jsonl"
HOOK_ERROR_RETAIN = 200  # keep the last N lines to bound file growth
HOOK_ERROR_SURFACE_LIMIT = 5  # how many recent errors to surface in payloads


def _hook_error_log_path() -> Path:
    return STATE_DIR / HOOK_ERROR_LOG_NAME


def _record_hook_error(where: str, err: BaseException) -> dict:
    """Persist a structured error record AND return a notice dict.

    The caller is expected to surface the returned notice to the agent
    via additionalContext or systemMessage \u2014 do NOT just silently log.
    On return, the caller has the information needed to emit a
    "[!] Mempalace hook error: ..." block.
    """
    notice = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "where": where,
        "error_type": type(err).__name__,
        "message": str(err)[:500],
    }
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        path = _hook_error_log_path()
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(notice) + "\n")
        # Trim to last HOOK_ERROR_RETAIN lines so the file doesn't grow
        # unboundedly on a broken plugin.
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) > HOOK_ERROR_RETAIN:
                with open(path, "w", encoding="utf-8") as f:
                    f.writelines(lines[-HOOK_ERROR_RETAIN:])
        except OSError:
            pass
    except (OSError, UnicodeError):
        # Even the recorder itself cannot be allowed to raise.
        pass
    _log(f"HOOK_ERROR {where}: {notice['error_type']}: {notice['message']}")
    return notice


def _recent_hook_errors(limit: int = HOOK_ERROR_SURFACE_LIMIT) -> list:
    """Read the tail of the hook-error log. Returns [] when the file is
    missing or unreadable. Used by SessionStart rehydration and the
    mempalace_hook_health tool to surface recent failures."""
    path = _hook_error_log_path()
    if not path.is_file():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []
    out = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _format_hook_error_notice(notice: dict) -> str:
    """Format a single error record as a short agent-readable string."""
    if not notice:
        return ""
    where = notice.get("where", "?")
    etype = notice.get("error_type", "?")
    msg = notice.get("message", "?")
    ts = notice.get("ts", "")
    return (
        f"[!] MEMPALACE HOOK ERROR at {where}: {etype}: {msg}"
        f" (ts={ts}). This was silently swallowed by fail-open; see "
        f"~/.mempalace/hook_state/{HOOK_ERROR_LOG_NAME}. Investigate and fix."
    )


def _format_recent_errors_block(errors: list) -> str:
    """Markdown block of recent hook errors for rehydration payloads."""
    if not errors:
        return ""
    lines = ["## [!] Recent hook errors (last {})".format(len(errors))]
    for e in errors:
        lines.append(
            f"- [{e.get('ts', '')}] {e.get('where', '?')}: "
            f"{e.get('error_type', '?')}: {e.get('message', '')[:160]}"
        )
    lines.append("")
    lines.append(
        "These failed silently and returned empty. Fix the underlying "
        f"cause; see ~/.mempalace/hook_state/{HOOK_ERROR_LOG_NAME} for full log."
    )
    return "\n".join(lines)


def _output(data: dict):
    """Print JSON to stdout with consistent formatting.

    Uses ensure_ascii=True so non-ASCII characters (emoji, smart quotes,
    etc.) serialize as escaped ``\\uXXXX`` forms. This prevents Windows
    console encoding crashes (cp1252/charmap can't encode chars like
    U+26A0). The JSON parser on the Claude Code side decodes the escapes
    back to the original characters, so visible content is unchanged.

    Before this fix, any error notice containing an emoji caused the
    hook's print() to crash with UnicodeEncodeError, and that error was
    recorded as ANOTHER hook error with the same emoji, looping on each
    subsequent hook invocation.
    """
    print(json.dumps(data, indent=2, ensure_ascii=True))


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
        except OSError as _e:
            _record_hook_error("_maybe_auto_ingest", _e)


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


def _is_non_iterative_mode() -> bool:
    """Detect whether this process is running a non-iterative specialist agent.

    Non-iterative agents (paperclip/flowsev workflow runs) cannot ask the user
    questions mid-run and must be permitted to stop cleanly. They are signaled
    via ``PAPERCLIP_RUN_ID`` or ``AGENT_HOME`` env vars per the CLAUDE.md
    paperclip rules.

    Returns True if ANY of the signals is set to a non-empty string.
    """
    for var in ("PAPERCLIP_RUN_ID", "AGENT_HOME"):
        if os.environ.get(var, "").strip():
            return True
    return False


def _read_last_finalized_intent(session_id: str) -> dict:
    """Read the last-finalized marker file for this session, or {} if absent.

    The marker is written by ``tool_finalize_intent`` at the end of each
    successful finalize. It contains ``intent_type``, ``outcome``,
    ``execution_entity``, ``agent``, ``ts``. The Stop hook reads it
    synchronously; any read error returns an empty dict (fail-open on
    missing/corrupt marker \u2014 caller treats that as "no proof").
    """
    safe_sid = _sanitize_session_id(session_id or "")
    if not safe_sid:
        return {}
    path = STATE_DIR / f"last_finalized_{safe_sid}.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as _e:
        # Corrupt marker is a real bug \u2014 record and surface rather than
        # silently returning empty (which would spuriously trip never-stop).
        _record_hook_error("_read_last_finalized_intent", _e)
        return {}
    if not isinstance(data, dict):
        _record_hook_error(
            "_read_last_finalized_intent",
            TypeError(f"expected dict, got {type(data).__name__}"),
        )
        return {}
    return data


def hook_stop(data: dict, harness: str):
    """Stop hook: enforce the no-stop rule for iterative agents.

    Iterative main-agent sessions cannot stop until a ``wrap_up_session`` intent
    has been finalized with outcome=success as the LAST finalized intent. The
    goal is to prevent lazy end-of-turn stops ("nothing else to do?") without
    proof-of-analysis. Claude Code's ``stop_hook_active`` loop-protection still
    caps us at ONE block per stop cycle, so the block message is rich enough
    to give the agent a full next-turn plan.

    Pass-through cases (the hook does NOT block):
      - ``stop_hook_active=true``: mandatory safety to avoid Claude Code's
        infinite-loop scenario.
      - Non-iterative mode (PAPERCLIP_RUN_ID / AGENT_HOME env set).
      - User break-glass bypass file active.
      - Wrap-up proof present (last finalized intent is wrap_up_session
        with outcome=success).

    The existing save-interval logic is kept as a secondary check: if wrap-up
    proof is present but the save counter has exceeded SAVE_INTERVAL, we still
    fire a save block. In practice this is now rare because the wrap-up flow
    itself persists everything.
    """
    parsed = _parse_harness_input(data, harness)
    session_id = parsed["session_id"]
    stop_hook_active = parsed["stop_hook_active"]
    transcript_path = parsed["transcript_path"]

    # If already in a save cycle, let through (infinite-loop prevention)
    if str(stop_hook_active).lower() in ("true", "1", "yes"):
        _output({})
        return

    # Non-iterative specialists must be allowed to stop.
    if _is_non_iterative_mode():
        _output({})
        return

    # User-only break-glass bypass overrides ALL iterative blocks.
    if _bypass_active():
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

    # NEVER-STOP rule: require wrap_up_session proof.
    # Iterative main-agent cannot stop until the LAST finalized intent in this
    # session is a wrap_up_session with outcome=success. Prevents lazy stops.
    last_finalized = _read_last_finalized_intent(session_id)
    if (
        not last_finalized
        or last_finalized.get("intent_type") != "wrap_up_session"
        or last_finalized.get("outcome") != "success"
    ):
        last_type = (last_finalized or {}).get("intent_type", "<none>")
        last_outcome = (last_finalized or {}).get("outcome", "<none>")
        _log(
            f"Stop BLOCK: no wrap_up_session proof for {session_id} "
            f"(last={last_type}/{last_outcome})"
        )
        _output({"decision": "block", "reason": NEVER_STOP_BLOCK_REASON})
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


REHYDRATION_MAX_CHARS = 6000  # ~2k tokens safety cap
REHYDRATION_MEMORY_CAP = 15
REHYDRATION_TRACE_CAP = 8


def _read_recent_trace(session_id: str, limit: int = REHYDRATION_TRACE_CAP) -> list:
    """Read the last N entries from the execution trace for this session.

    Returns a list of dicts (newest last). Empty list on any error — this
    is best-effort recovery information, never fatal.
    """
    safe_sid = _sanitize_session_id(session_id)
    if not safe_sid:
        return []
    trace_file = _TRACE_DIR / f"execution_trace_{safe_sid}.jsonl"
    if not trace_file.is_file():
        return []
    try:
        with open(trace_file, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as _e:
        _record_hook_error("_read_recent_trace", _e)
        return []
    recent = lines[-limit:]
    out = []
    bad_lines = 0
    for line in recent:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            bad_lines += 1
    if bad_lines:
        _record_hook_error(
            "_read_recent_trace",
            ValueError(f"{bad_lines} malformed JSONL lines in trace file"),
        )
    return out


def _build_rehydration_payload(intent: dict, session_id: str, source: str) -> str:
    """Build a rehydration block summarising the active intent for the model.

    Called from hook_session_start on source in {compact, resume}. Emits a
    readable markdown block with intent description, scope, remaining
    budget, memory pointers (id + presence, not full content — pointer
    style), and the last few trace entries so the model can pick up where
    compaction left off.

    Output is capped at REHYDRATION_MAX_CHARS (~2k tokens) to avoid
    dominating the freshly-compacted context window.
    """
    lines = []
    lines.append(f"## MemPalace rehydration — source: {source}")
    lines.append("")
    lines.append(
        "Context was compacted/resumed. The active intent below survived; "
        "continue the work from this checkpoint."
    )
    lines.append("")

    intent_type = intent.get("intent_type", "unknown")
    lines.append(f"**Active intent:** `{intent_type}`")
    desc = (intent.get("description") or "").strip()
    if desc:
        lines.append(f"**Goal:** {desc[:400]}")
    agent = intent.get("agent") or ""
    if agent:
        lines.append(f"**Agent:** {agent}")

    slots = intent.get("slots") or {}
    if slots:
        lines.append("")
        lines.append("**Scope (slots):**")
        for k, v in slots.items():
            lines.append(f"- {k}: {v}")

    budget = intent.get("budget") or {}
    used = intent.get("used") or {}
    if budget:
        lines.append("")
        lines.append("**Remaining budget (tool → remaining/total):**")
        for tool, limit in budget.items():
            spent = used.get(tool, 0)
            remaining = max(0, limit - spent)
            lines.append(f"- {tool}: {remaining}/{limit}")

    # Memory pointers — id + presence only, not full body
    injected = list(intent.get("injected_memory_ids") or [])
    accessed = list(intent.get("accessed_memory_ids") or [])
    # Dedupe while preserving order (injected first, then accessed-but-not-injected)
    seen = set()
    all_ids = []
    for mid in injected + accessed:
        if mid and mid not in seen:
            seen.add(mid)
            all_ids.append(mid)
    if all_ids:
        lines.append("")
        lines.append(f"**Memories surfaced during this intent ({len(all_ids)}):**")
        for mid in all_ids[:REHYDRATION_MEMORY_CAP]:
            lines.append(f"- `{mid}`")
        if len(all_ids) > REHYDRATION_MEMORY_CAP:
            lines.append(f"- ... ({len(all_ids) - REHYDRATION_MEMORY_CAP} more)")
        lines.append("")
        lines.append(
            "To re-read a specific memory, call "
            "`mempalace_kg_query(entity='<memory_id>')` or "
            "`mempalace_kg_search` with relevant keywords."
        )

    trace_entries = _read_recent_trace(session_id, limit=REHYDRATION_TRACE_CAP)
    if trace_entries:
        lines.append("")
        lines.append(f"**Recent actions (last {len(trace_entries)}):**")
        for e in trace_entries:
            ts = e.get("ts", "")
            tool = e.get("tool", "")
            target = e.get("target", "")
            lines.append(f"- [{ts}] {tool}: {target}")

    lines.append("")
    lines.append(
        "Continue the intent. If it is genuinely done, call "
        "`mempalace_finalize_intent` with the appropriate outcome. "
        "Do not re-declare without finalizing first."
    )

    body = "\n".join(lines)
    if len(body) > REHYDRATION_MAX_CHARS:
        body = body[: REHYDRATION_MAX_CHARS - 100] + "\n... (truncated)"
    return body


def hook_session_start(data: dict, harness: str):
    """SessionStart hook: on compact/resume, rehydrate active intent context.

    Claude Code fires SessionStart with ``source`` in
    {startup, resume, clear, compact}. On ``compact`` and ``resume`` — the
    two cases where the model just lost its prior context — we re-inject
    the active intent's description, scope, budget, memory pointers, and
    recent trace via ``hookSpecificOutput.additionalContext``. This pairs
    with the precompact systemMessage save-warning to close the
    compaction-recovery loop.

    On ``startup``/``clear`` or when no active intent exists: pass through
    with a recent-errors summary if any hook errors are pending \u2014 that's
    the first thing the freshly-booted agent should see.

    FAIL-LOUD-WITHOUT-BLOCKING: never blocks session entry, but any
    unexpected exception is recorded and surfaced via additionalContext.
    """
    try:
        parsed = _parse_harness_input(data, harness)
        session_id = parsed["session_id"]
        source = str(data.get("source", "")).lower()

        _log(f"SESSION START session={session_id} source={source or 'unspecified'}")

        STATE_DIR.mkdir(parents=True, exist_ok=True)

        # Surface recent hook errors regardless of source \u2014 the start of
        # any new session is the perfect moment to show accumulated silent
        # failures so the agent can address them.
        recent_errors = _recent_hook_errors()
        errors_block = _format_recent_errors_block(recent_errors) if recent_errors else ""

        if source not in ("compact", "resume"):
            if errors_block:
                _output(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "SessionStart",
                            "additionalContext": errors_block,
                        }
                    }
                )
            else:
                _output({})
            return

        if not session_id:
            if errors_block:
                _output(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "SessionStart",
                            "additionalContext": errors_block,
                        }
                    }
                )
            else:
                _output({})
            return

        intent = _read_active_intent(session_id)
        if not intent:
            if errors_block:
                _output(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "SessionStart",
                            "additionalContext": errors_block,
                        }
                    }
                )
            else:
                _output({})
            return

        payload = _build_rehydration_payload(intent, session_id, source)
        # Always append recent errors block to the rehydration payload.
        if errors_block:
            payload = (payload + "\n\n" + errors_block) if payload else errors_block
        if not payload:
            _output({})
            return

        _output(
            {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": payload,
                }
            }
        )
    except Exception as e:
        # Top-level SessionStart failure: record + surface via
        # additionalContext. Never swallow.
        notice = _record_hook_error("hook_session_start", e)
        try:
            _output(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "SessionStart",
                        "additionalContext": _format_hook_error_notice(notice),
                    }
                }
            )
        except Exception:
            _output({})


# ── UserPromptSubmit uses this; PreToolUse does not retrieve anymore ──
LOCAL_CUE_ASSISTANT_MAX_CHARS = 400


# ── Phase 3b: AskUserQuestion lazy-question detector ────────────────
#
# When the never-stop rule is active, the agent's escape hatch is
# AskUserQuestion. But not every question is a real blocker: "should
# I continue?", "what's next?", "anything else?" are laziness patterns
# that dodge the responsibility to decide and keep working.
#
# Layer 1 (shipped here): deterministic regex against a small, curated
# pattern set. Rasa-style first pass \u2014 ~100% precision on explicit
# phrasings, zero cost, easy to extend. Layer 2 (deferred): semantic
# similarity against a lazy-prototype set for creative rewordings.

# Compiled as module-level tuple so the hook pays the compile cost
# exactly once per process. Patterns are start-anchored (with optional
# leading whitespace / punctuation) and case-insensitive. Keep this set
# small; precision > recall on the regex layer.
_LAZY_QUESTION_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        # should/shall/can/may I <continuation-verb>
        r"^\s*[\"'\(]*\s*(?:should|shall|can|may|could|would)\s+(?:i|we)\s+"
        r"(?:continue|proceed|keep\s+going|go\s+on|move\s+on|move\s+forward|start|begin)\b",
        # what's next / what next / what is the next step
        r"^\s*[\"'\(]*\s*(?:what'?s?|what\s+is)\s+"
        r"(?:next|the\s+next|our\s+next|my\s+next|next\s+step|the\s+next\s+step)\b",
        # is/are there more/anything/additional
        r"^\s*[\"'\(]*\s*(?:is|are)\s+there\s+"
        r"(?:more|anything|any|additional|other)\s+(?:to|i\s+should|we\s+need|work)\b",
        # any/anything more/else
        r"^\s*[\"'\(]*\s*(?:any|anything)\s+(?:more|else)\b",
        # ready to continue/proceed/finish
        r"^\s*[\"'\(]*\s*(?:are\s+you|are\s+we|am\s+i)?\s*ready\s+to\s+"
        r"(?:continue|proceed|move|finish|stop)\b",
        # should we continue / keep going
        r"^\s*[\"'\(]*\s*should\s+we\s+(?:continue|proceed|keep\s+going|stop|move\s+on)\b",
        # do you want me to continue
        r"^\s*[\"'\(]*\s*(?:do|should)\s+(?:you\s+want|we\s+want)\s+me\s+to\s+"
        r"(?:continue|proceed|keep\s+going|stop|move\s+on)\b",
        # am i/are we done?
        r"^\s*[\"'\(]*\s*(?:am\s+i|are\s+we|is\s+this)\s+done\b",
        # want me to keep going / proceed / continue
        r"^\s*[\"'\(]*\s*want\s+me\s+to\s+"
        r"(?:continue|proceed|keep\s+going|go\s+on|move\s+on)\b",
    )
)


def _is_lazy_question(question_text: str) -> tuple:
    """Return (is_lazy, matched_pattern_source). Matches if any compiled
    pattern in ``_LAZY_QUESTION_PATTERNS`` hits the text. Case-insensitive,
    whitespace-tolerant, leading-punct-tolerant. Returns (False, None) for
    any non-string / empty input \u2014 callers should short-circuit there."""
    if not question_text or not isinstance(question_text, str):
        return (False, None)
    for pat in _LAZY_QUESTION_PATTERNS:
        if pat.search(question_text):
            return (True, pat.pattern)
    return (False, None)


def _extract_askuserquestion_texts(tool_input: dict) -> list:
    """Extract all question strings from an AskUserQuestion tool_input.

    AskUserQuestion shape: {"questions": [{"question": str, ...}, ...]}.
    Defensive: any missing/malformed entry yields an empty text rather
    than raising; callers treat empty as non-lazy.
    """
    out = []
    if not isinstance(tool_input, dict):
        return out
    questions = tool_input.get("questions")
    if not isinstance(questions, list):
        return out
    for q in questions:
        if isinstance(q, dict):
            text = q.get("question")
            if isinstance(text, str):
                out.append(text)
    return out


def _maybe_deny_lazy_askuserquestion(tool_name: str, tool_input: dict):
    """If ``tool_name`` is AskUserQuestion and ANY question matches a
    lazy-question regex, return a PreToolUse deny response. Otherwise
    return None (caller continues normal flow)."""
    if tool_name != "AskUserQuestion":
        return None
    texts = _extract_askuserquestion_texts(tool_input)
    if not texts:
        return None
    for text in texts:
        lazy, pattern = _is_lazy_question(text)
        if lazy:
            reason = (
                "LAZY-QUESTION REJECTED by never-stop rule. Your question "
                f"({text[:80]!r}) matches a should-I-continue / what's-next / "
                "anything-else pattern. These are NOT valid blockers \u2014 they "
                "are you dodging the responsibility to decide and keep working. "
                "Instead: (1) review your TodoWrite list, (2) if work remains, "
                "DO IT; (3) if you are genuinely done, run the wrap_up_session "
                "intent cycle (>=2 kg_search against pending-work patterns, "
                "then finalize); (4) only use AskUserQuestion for SPECIFIC "
                "blockers (ambiguous requirement, missing credential, destructive "
                "action needing consent)."
            )
            _log(f"PreToolUse DENY AskUserQuestion: lazy pattern {pattern!r} matched {text[:60]!r}")
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
    return None


# ── Phase 2b: PreToolUse local retrieval ────────────────────────────
#
# On by default. The hook stays dep-free at import time \u2014 Chroma and
# scoring modules are lazy-imported inside _run_local_retrieval so the
# common permission-only path pays no startup cost. A fail-open timeout
# wraps the actual retrieval so any slowness / error yields an empty
# additionalContext rather than blocking the tool call. The historic
# env gate (MEMPALACE_LOCAL_RETRIEVAL) was hedging; it defeated the
# purpose of Option A (memories must surface automatically) and has
# been removed. An opt-OUT env (MEMPALACE_DISABLE_LOCAL_RETRIEVAL=1)
# is kept as a break-glass for incident response.

ENV_OPT_OUT = "MEMPALACE_DISABLE_LOCAL_RETRIEVAL"
LOCAL_RETRIEVAL_TOP_K = 3
LOCAL_RETRIEVAL_MAX_CHARS = 1200  # ~400 tokens cap for additionalContext
LOCAL_RETRIEVAL_TIMEOUT_SEC = 10.0  # fresh python subprocess pays full cold-start (ONNX model load ~1-2s + chromadb init + search); architectural fix = hook delegate to long-running MCP server, but until then this covers real observed Windows cold-start latency
LOCAL_RETRIEVAL_MEMORY_PREVIEW_CHARS = 260


def _is_local_retrieval_enabled() -> bool:
    """Return True unless the user has explicitly opted out via the
    ``MEMPALACE_DISABLE_LOCAL_RETRIEVAL`` env var. The opt-out is a
    break-glass for incident response \u2014 the default is ON so Option A's
    per-tool-call retrieval fires automatically.
    """
    val = os.environ.get(ENV_OPT_OUT, "").strip().lower()
    return val not in ("1", "true", "yes", "on")


def _format_retrieval_additional_context(memories: list) -> tuple[str, list[str]]:
    """Render a list of retrieved-memory dicts as markdown additionalContext.

    Each memory dict carries keys: id, preview, score. Every retrieved
    memory is rendered — TOP_K (see LOCAL_RETRIEVAL_TOP_K) already bounds
    the size; there is no char-budget cap. Previously, an outer
    LOCAL_RETRIEVAL_MAX_CHARS budget would silently omit overflow items
    with a "... N more omitted" line while still adding their ids to
    ``accessed_memory_ids``. That produced a perpetual "coverage 0% at
    finalize_intent" failure mode because the agent was asked to rate
    ids it had never seen. Retrieved memories MUST always be shown so
    the agent can observe (and later rate) every id in accessed.

    Returns ``(markdown, rendered_ids)`` — ``rendered_ids`` is always the
    full list of input ids. The tuple shape is kept so the call site
    persists exactly the ids that rendered (same as the full input now,
    but the contract survives future formatter changes).
    """
    if not memories:
        return "", []
    lines = ["## Local retrieval (operation-level memories)"]
    lines.append("")
    lines.append(
        "These were surfaced by a per-tool-call cue, separate from the "
        "activity-level memories injected at declare_intent time. Skim "
        "before acting."
    )
    lines.append("")
    rendered_ids: list[str] = []
    for m in memories:
        mid = m.get("id", "")
        preview = (m.get("preview") or "").strip().replace("\n", " ")
        if len(preview) > LOCAL_RETRIEVAL_MEMORY_PREVIEW_CHARS:
            preview = preview[: LOCAL_RETRIEVAL_MEMORY_PREVIEW_CHARS - 1] + "\u2026"
        lines.append(f"- `{mid}`: {preview}")
        if mid:
            rendered_ids.append(mid)
    return "\n".join(lines), rendered_ids


def _run_local_retrieval(cue: dict, accessed_memory_ids, top_k: int) -> tuple:
    """Execute multi_channel_search and return ``(hits, error_notice)``.

    Returns:
        (list, None) on success \u2014 list is up to ``top_k`` dicts
            {id, preview, score}, possibly empty if no hits.
        ([], notice_dict) on failure \u2014 notice is the dict from
            ``_record_hook_error`` so the caller can format it into
            additionalContext. The caller MUST surface the notice, not
            swallow it.

    Lazy-imports mempalace.palace and mempalace.scoring so the common
    permission-only hook path stays free of Chroma startup cost.
    """
    queries = [q for q in (cue or {}).get("queries") or [] if q]
    keywords = [k for k in (cue or {}).get("keywords") or [] if k]
    if not queries:
        return ([], None)
    try:
        import time as _time

        _t_start = _time.monotonic()
        _deadline = _t_start + LOCAL_RETRIEVAL_TIMEOUT_SEC

        from .config import MempalaceConfig as _Config
        from .palace import get_collection as _get_collection
        from .scoring import multi_channel_search as _mcs

        _t_after_imports = _time.monotonic()

        _cfg = _Config()
        palace_path = _cfg.palace_path
        collection_name = _cfg.collection_name
        _t_after_config = _time.monotonic()

        _log(
            f"RETRIEVAL_TIMING imports={(_t_after_imports - _t_start) * 1000:.0f}ms "
            f"config={(_t_after_config - _t_after_imports) * 1000:.0f}ms "
            f"palace_path={palace_path!r} collection={collection_name!r}"
        )

        if _time.monotonic() > _deadline:
            return (
                [],
                _record_hook_error(
                    "_run_local_retrieval",
                    TimeoutError("exceeded LOCAL_RETRIEVAL_TIMEOUT_SEC before collection open"),
                ),
            )
        col = _get_collection(palace_path, collection_name)
        _t_after_collection = _time.monotonic()
        _log(
            f"RETRIEVAL_TIMING get_collection={(_t_after_collection - _t_after_config) * 1000:.0f}ms"
        )

        if col is None:
            return (
                [],
                _record_hook_error(
                    "_run_local_retrieval",
                    RuntimeError("get_collection returned None"),
                ),
            )
        if _time.monotonic() > _deadline:
            return (
                [],
                _record_hook_error(
                    "_run_local_retrieval",
                    TimeoutError("exceeded LOCAL_RETRIEVAL_TIMEOUT_SEC after collection open"),
                ),
            )

        pipe = _mcs(
            col,
            list(queries),
            keywords=list(keywords),
            kg=None,  # skip graph channel in hook; keeps latency predictable
            fetch_limit_per_view=20,
            include_graph=False,
        )
        _t_after_search = _time.monotonic()
        rrf_scores_preview = pipe.get("rrf_scores") or {}
        _log(
            f"RETRIEVAL_TIMING multi_channel_search={(_t_after_search - _t_after_collection) * 1000:.0f}ms "
            f"total_since_start={(_t_after_search - _t_start) * 1000:.0f}ms "
            f"rrf_hits={len(rrf_scores_preview)} "
            f"queries={[str(q)[:40] for q in queries]!r} "
            f"keywords={keywords!r} "
            f"accessed_count={len(accessed_memory_ids or [])}"
        )

        if _time.monotonic() > _deadline:
            return (
                [],
                _record_hook_error(
                    "_run_local_retrieval",
                    TimeoutError("exceeded LOCAL_RETRIEVAL_TIMEOUT_SEC during search"),
                ),
            )

        ranked_lists = pipe.get("ranked_lists") or {}
        seen_meta = pipe.get("seen_meta") or {}
        if not ranked_lists and not pipe.get("rrf_scores"):
            return ([], None)  # genuinely no hits is NOT an error

        # ── Route through the canonical two-stage pipeline ──
        # scoring.two_stage_retrieve encapsulates RRF → hybrid_score rerank
        # → adaptive_k cutoff. declare_intent and kg_search share the same
        # helper so the three tools produce hits on the same scale and
        # with the same semantics.
        from .scoring import two_stage_retrieve as _two_stage

        reranked, _rrf_full, _candidate_map = _two_stage(
            ranked_lists,
            seen_meta,
            agent="",
            session_id="",
            intent_type_id="",
            context_feedback={},
            rerank_top_m=50,
            max_k=top_k,
            min_k=1,
        )

        # Project to the shape hooks_cli's consumers expect; collapse
        # multi-view physical ids (e.g. '{id}__v2') back to their logical
        # memory id, keep best-hybrid_score per logical id.
        accessed_set = set(accessed_memory_ids or [])
        logical_best: dict = {}
        for entry in reranked:
            meta = entry.get("meta") or {}
            logical_id = meta.get("entity_id") or entry["id"]
            if logical_id in accessed_set:
                continue
            # Summary-first preview: prefer meta.summary, else the doc
            # text the channel surfaced (capped downstream by
            # intent._shorten_preview at the tool boundary).
            summary_val = (meta.get("summary") or "").strip()
            preview = summary_val or (entry.get("text") or "").strip()
            score = float(entry["hybrid_score"])
            if score > logical_best.get(logical_id, (float("-inf"),))[0]:
                logical_best[logical_id] = (score, preview)

        ranked = [
            {"id": lid, "score": score, "preview": preview}
            for lid, (score, preview) in logical_best.items()
        ]
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return (ranked[: max(0, top_k)], None)
    except Exception as _e:
        return ([], _record_hook_error("_run_local_retrieval", _e))


def _persist_accessed_memory_ids(session_id: str, intent: dict, new_ids: list):
    """Write the updated accessed_memory_ids set back to the session-scoped
    active_intent state file so finalize_intent sees the new ids and the
    feedback-coverage check covers them. On failure, records a visible
    hook error via _record_hook_error (no longer silent).
    """
    if not session_id or not intent or not new_ids:
        return
    safe_sid = _sanitize_session_id(session_id)
    if not safe_sid:
        return
    try:
        state_path = STATE_DIR / f"active_intent_{safe_sid}.json"
        if not state_path.is_file():
            return
        data = json.loads(state_path.read_text(encoding="utf-8"))
        existing = set(data.get("accessed_memory_ids", []) or [])
        updated = existing | set(new_ids)
        if updated != existing:
            data["accessed_memory_ids"] = sorted(updated)
            state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as _e:
        _record_hook_error("_persist_accessed_memory_ids", _e)


# ─────────────────────────────────────────────────────────────────────
# Slice B (user-intent tier): pending user-message persistence
# ─────────────────────────────────────────────────────────────────────
#
# Lives in a per-session JSON file decoupled from active_intent state
# because user messages may arrive BEFORE any intent is declared (and a
# declare_user_intents call is always the first thing an agent does on
# a fresh session). File shape:
#
#   pending_user_messages_<session_id>.json
#   {
#     "session_id": "<sid>",
#     "messages": [
#       {"id": "msg_<sha12>_<ts_ns>",
#        "text": "<raw user prompt>",
#        "turn_idx": 5,
#        "ts": "2026-04-26T03:00:00Z"},
#       ...
#     ]
#   }
#
# IDs are deterministic (sha12 over session+turn+text + nanosecond
# tiebreak) so the agent's additionalContext block can name them, and
# mempalace_declare_user_intents can validate covered IDs against the
# pending list before minting record entities for each.


def _make_user_message_id(session_id: str, turn_idx: int, text: str) -> str:
    """Deterministic id for a user message turn.

    SHA12 over (session_id, turn_idx, text) gives a stable id the agent
    can name in additionalContext; the nanosecond suffix prevents
    collisions when two distinct turns happen to hash the same prefix
    (rare but cheap to defend against)."""
    import hashlib
    import time

    payload = f"{session_id}\t{turn_idx}\t{text}".encode("utf-8", errors="replace")
    digest = hashlib.sha256(payload).hexdigest()[:12]
    ns = time.time_ns()
    return f"msg_{digest}_{ns}"


def _pending_user_messages_path(session_id: str):
    """Per-session JSON path for the pending user-message queue."""
    safe_sid = _sanitize_session_id(session_id)
    if not safe_sid:
        return None
    return STATE_DIR / f"pending_user_messages_{safe_sid}.json"


def _read_pending_user_messages(session_id: str) -> list:
    """Read the pending user-message list for this session.

    Returns [] when the file is absent, malformed, or the session id is
    empty. Never raises — callers treat absence and corruption identically
    (both mean "nothing pending")."""
    path = _pending_user_messages_path(session_id)
    if path is None or not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        msgs = data.get("messages") or []
        return [m for m in msgs if isinstance(m, dict) and m.get("id")]
    except Exception as _e:
        _record_hook_error("_read_pending_user_messages", _e)
        return []


def _append_pending_user_message(session_id: str, message: dict) -> bool:
    """Append a user-message entry to the pending queue. Idempotent on
    duplicate ids (silent skip). Returns True on successful append, False
    on any failure (the failure is recorded via _record_hook_error)."""
    path = _pending_user_messages_path(session_id)
    if path is None:
        return False
    if not isinstance(message, dict) or not message.get("id"):
        return False
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = {"session_id": session_id, "messages": []}
        msgs = data.get("messages") or []
        if any(m.get("id") == message["id"] for m in msgs):
            return True  # idempotent
        msgs.append(message)
        data["session_id"] = session_id
        data["messages"] = msgs
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return True
    except Exception as _e:
        _record_hook_error("_append_pending_user_message", _e)
        return False


def _clear_pending_user_messages(session_id: str) -> int:
    """Drain the pending queue for this session. Returns the count cleared.
    Atomic delete (rename to temp + unlink) so concurrent reads never see
    a half-written file."""
    path = _pending_user_messages_path(session_id)
    if path is None or not path.is_file():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        n = len(data.get("messages") or [])
        path.unlink()
        return n
    except Exception as _e:
        _record_hook_error("_clear_pending_user_messages", _e)
        return 0


OPERATION_CUE_TTL_SECONDS = 300  # 5 min: generous for parallel-batch work;
# forgotten cues don't accumulate past this.
OPERATION_CUE_WAIT_TIMEOUT_SECONDS = 5.0  # total parallel-batch wait budget.
OPERATION_CUE_WAIT_POLL_INTERVAL_SECONDS = 0.25


def _parse_iso_utc(ts: str):
    """Parse an ISO-8601 UTC timestamp (trailing 'Z'). Returns naive UTC
    datetime or None on any parse failure."""
    if not ts or not isinstance(ts, str):
        return None
    try:
        import datetime as _dt_mod

        # Handle both '2026-04-20T12:34:56Z' and '2026-04-20T12:34:56+00:00'
        clean = ts.rstrip("Z")
        return _dt_mod.datetime.fromisoformat(clean)
    except (ValueError, AttributeError):
        return None


def _prune_expired_cues(cues: list) -> tuple:
    """Drop cues older than OPERATION_CUE_TTL_SECONDS. Returns
    ``(live_cues, expired_cues)``. Pure function — callers persist.
    """
    if not cues or not isinstance(cues, list):
        return [], []
    import datetime as _dt_mod

    now = _dt_mod.datetime.utcnow()
    live, expired = [], []
    for c in cues:
        if not isinstance(c, dict):
            continue
        ts = _parse_iso_utc(c.get("declared_at_ts", ""))
        if ts is None:
            # No/invalid timestamp → treat as live (fail-open; we never
            # want to over-aggressively drop the agent's just-written cue
            # because of a clock-format mismatch).
            live.append(c)
            continue
        age = (now - ts).total_seconds()
        if age > OPERATION_CUE_TTL_SECONDS:
            expired.append(c)
        else:
            live.append(c)
    return live, expired


def _read_pending_operation_cues_from_disk(session_id: str):
    """Read pending_operation_cues directly from the active_intent state
    file. Used by the strict-mode wait-loop so we re-check disk freshly
    each poll without going through the full _read_active_intent parse
    path (which is fine either way, kept thin here for the poll loop).
    """
    safe_sid = _sanitize_session_id(session_id or "")
    if not safe_sid:
        return []
    path = STATE_DIR / f"active_intent_{safe_sid}.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    cues = data.get("pending_operation_cues") or []
    return cues if isinstance(cues, list) else []


def _persist_pending_operation_cues(session_id: str, cues: list):
    """Write the updated pending_operation_cues list back to disk. Used
    by the hook after consuming/expiring entries. Last-writer-wins on the
    full list — good enough given each mutation window is microseconds."""
    safe_sid = _sanitize_session_id(session_id or "")
    if not safe_sid:
        return
    path = STATE_DIR / f"active_intent_{safe_sid}.json"
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if not isinstance(data, dict):
        return
    data["pending_operation_cues"] = cues or []
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as _e:
        _record_hook_error("_persist_pending_operation_cues", _e)


def _consume_pending_operation_cue(session_id: str, intent: dict, tool_name: str):
    """Pop the first pending_operation_cues entry matching ``tool_name``
    off the active_intent state (TTL-pruning stale entries in the same
    pass) and persist the shortened list so the NEXT tool call sees the
    updated state. Returns ``(cue_dict_or_None, expired_count)``.

    The parallel-batch design (2026-04-20): when Claude Code dispatches
    multiple tool uses in one assistant message, each real tool call's
    hook subprocess pops its own matching cue independently. Last-writer-
    wins on the list — the mutation window is microseconds so races are
    rare and the effect of losing a cue to a race is "falls back to
    legacy cue-from-args" (or strict-mode deny), never silent
    misattribution.

    No-match returns (None, expired_count). The strict hook branch
    handles "missing" / "mismatched" / "only stale were present" reporting.
    """
    cues = intent.get("pending_operation_cues") if intent else None
    if not isinstance(cues, list):
        return None, 0
    live, expired = _prune_expired_cues(cues)
    # Find first matching live entry by tool name
    popped = None
    remaining = []
    for c in live:
        if popped is None and isinstance(c, dict) and c.get("tool") == tool_name:
            popped = c
        else:
            remaining.append(c)
    if popped is None and not expired:
        return None, 0
    # Persist the pruned + popped state back to disk.
    try:
        _persist_pending_operation_cues(session_id, remaining)
    except Exception as _e:
        _record_hook_error("_consume_pending_operation_cue.persist", _e)
    intent["pending_operation_cues"] = remaining
    if popped is None:
        return None, len(expired)
    return (
        {
            "queries": list(popped.get("queries") or []),
            "keywords": list(popped.get("keywords") or []),
            # S1: propagate the operation's context entity id so
            # _append_trace can tag the trace entry. Used at
            # finalize_intent to attach the promoted operation node
            # via performed_well / performed_poorly edges.
            "active_context_id": str(popped.get("active_context_id") or ""),
        },
        len(expired),
    )


def _wait_for_matching_pending_cue(
    session_id: str,
    tool_name: str,
    intent: dict,
):
    """Strict-mode parallel-race helper. When the agent emits declare_
    operation + the real tool call in a single assistant message, the
    hook for the real tool may fire before the MCP server finishes
    persisting the cue. Poll disk up to OPERATION_CUE_WAIT_TIMEOUT for a
    matching cue before giving up. Re-reads intent in place if a match
    arrives. Returns True if a matching cue was observed (caller should
    re-consume), False if the timeout elapsed with no match.
    """
    import time as _time

    deadline = _time.monotonic() + OPERATION_CUE_WAIT_TIMEOUT_SECONDS
    while _time.monotonic() < deadline:
        cues = _read_pending_operation_cues_from_disk(session_id)
        if cues:
            # Hydrate intent dict so _consume_pending_operation_cue sees
            # the updated list when the caller invokes it.
            intent["pending_operation_cues"] = cues
            if any(isinstance(c, dict) and c.get("tool") == tool_name for c in cues):
                return True
        _time.sleep(OPERATION_CUE_WAIT_POLL_INTERVAL_SECONDS)
    return False


USER_PROMPT_CUE_PREFIX = "User said: "
USER_PROMPT_CUE_MAX_CHARS = 400


def _extract_prompt_keywords(prompt_text: str, limit: int = 8) -> list:
    """Extract a small, deduped keyword list from a free-form user prompt.

    Lowercase, alphanumeric-plus-hyphen tokens, drop a tiny stopword set,
    cap at ``limit`` unique tokens (preserving order of first occurrence).
    Channel C (keyword) is a secondary signal; we don't try to be clever
    here \u2014 cosine on the prefixed query does the main lift.
    """
    if not prompt_text or not isinstance(prompt_text, str):
        return []
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", prompt_text.lower())
    stop = {
        "the",
        "and",
        "for",
        "with",
        "you",
        "but",
        "not",
        "are",
        "was",
        "this",
        "that",
        "from",
        "have",
        "has",
        "had",
        "what",
        "when",
        "why",
        "how",
        "who",
        "which",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "there",
        "where",
        "about",
        "their",
        "they",
        "your",
        "our",
        "ours",
        "one",
        "two",
        "all",
        "any",
    }
    seen = set()
    out = []
    for tok in tokens:
        if tok in stop or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= limit:
            break
    return out


def hook_userpromptsubmit(data: dict, harness: str):
    """UserPromptSubmit hook (Slice B-2): persist the user prompt to the
    per-session pending_user_messages queue and surface the pending ids
    + ``mempalace_declare_user_intents`` pointer as additionalContext.

    Replaces the legacy local-retrieval path with the user-intent tier
    flow. The hook no longer runs ``multi_channel_search`` directly;
    retrieval-per-context now happens inside
    ``mempalace_declare_user_intents`` (Slice B-1) which the agent must
    call in response to this additionalContext block. PreToolUse blocks
    every non-allowed tool until the pending queue is cleared by a
    successful ``declare_user_intents`` invocation.

    Env-var escapes (any TRUE turns this hook into a no-op):
      * ``MEMPALACE_USER_INTENT_DISABLED=1`` - explicit Slice B-2 escape.
      * ``MEMPALACE_DISABLE_LOCAL_RETRIEVAL=1`` - legacy escape from the
        pre-Slice-B world; honoured here so existing operator overrides
        still work.

    FAIL-LOUD-WITHOUT-BLOCKING: never blocks the user's prompt; any
    exception is recorded via ``_record_hook_error`` AND surfaced via
    additionalContext on THIS turn. No silent swallow.
    """
    if (
        os.environ.get("MEMPALACE_USER_INTENT_DISABLED", "").strip().lower()
        in ("1", "true", "yes", "on")
        or not _is_local_retrieval_enabled()
    ):
        _output({})
        return

    error_notices = []
    try:
        parsed = _parse_harness_input(data, harness)
        session_id = parsed["session_id"]
        prompt_text = str(data.get("prompt", "")).strip()

        if not session_id or not prompt_text:
            _output({})
            return

        # Compute deterministic id; turn_idx = current pending depth so
        # consecutive turns have distinct ids even when the agent has
        # not yet drained earlier ones.
        existing_pending = _read_pending_user_messages(session_id)
        turn_idx = len(existing_pending)
        from datetime import datetime as _dt
        from datetime import timezone as _tz

        ts = _dt.now(_tz.utc).isoformat(timespec="seconds")
        msg_id = _make_user_message_id(session_id, turn_idx, prompt_text)
        msg = {
            "id": msg_id,
            "text": prompt_text,
            "turn_idx": turn_idx,
            "ts": ts,
        }
        if not _append_pending_user_message(session_id, msg):
            error_notices.append(
                {
                    "fn": "_append_pending_user_message",
                    "error": "append failed; see hook_errors.jsonl for details",
                }
            )

        # Re-read so the additionalContext lists every pending id this
        # session, not just the freshly-appended one. That way an agent
        # who got the prior turn's instruction but did not act sees the
        # full backlog now.
        all_pending = _read_pending_user_messages(session_id)
        pending_ids = [m["id"] for m in all_pending if m.get("id")]

        body_lines = [
            f"## User-intent tier: {len(pending_ids)} pending user_message(s)",
            "",
            "Before any other tool call, call `mempalace_declare_user_intents`",
            "to declare a context per user-intent in the message(s) below. The",
            "tool validates that ALL pending ids are covered (union of",
            "context.user_message_ids across declared contexts). If a message",
            "has no actionable intent (ack, 'thanks', etc.), declare it under",
            "`no_intent=true` AFTER asking the user via AskUserQuestion to",
            "confirm - `no_intent_clarified_with_user=true` is mandatory in",
            "that case. Only AskUserQuestion and mempalace_* tools are allowed",
            "until the pending queue is cleared.",
            "",
            "Pending ids (cover all of these):",
        ]
        for mid in pending_ids:
            body_lines.append(f"  - `{mid}`")

        for n in error_notices:
            body_lines.append("")
            body_lines.append(_format_hook_error_notice(n))

        body = "\n".join(body_lines)
        _output(
            {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": body,
                }
            }
        )
    except Exception as e:
        # Top-level unexpected failure: record and surface via
        # additionalContext so the agent sees the bug immediately.
        notice = _record_hook_error("hook_userpromptsubmit", e)
        try:
            _output(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "UserPromptSubmit",
                        "additionalContext": _format_hook_error_notice(notice),
                    }
                }
            )
        except Exception:
            # Absolute-last-resort: even the output path failed. Pass
            # through rather than crash the session \u2014 the error is
            # already persisted to hook_errors.jsonl.
            _output({})


def hook_precompact(data: dict, harness: str):
    """Precompact hook: surface a save-everything warning but DO NOT block.

    Blocking compaction risks losing the whole session when context fills up,
    so the hook must never prevent compaction from running. It emits the save
    instructions via `systemMessage` (cross-harness non-blocking field) and
    returns an empty decision so compaction proceeds.
    """
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

    _output({"systemMessage": PRECOMPACT_WARNING_MESSAGE})


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
    "AskUserQuestion",  # meta-tool: asks user for input, same category as TodoWrite
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

# Hard-block guard: any non-always-allowed tool whose tool_input carries
# a filesystem reference to the break-glass bypass file is denied
# unconditionally by PreToolUse. Match scope is path-prefixed (a path
# separator immediately before the filename); prose mentions in
# docstrings / TodoWrite content / banner comments are deliberately
# allowed so documentation remains editable. The break-glass bypass
# itself cannot soften this deny — the whole point is that ONLY the
# human user, operating at the OS terminal, can ever create or delete
# this file. Agents (including Claude) must not be able to touch it via
# any tool, even when every other permission would otherwise allow it.
_BYPASS_FILE_NAME = "HOOK_BYPASS_USER_ONLY"


# Path-prefixed regex so prose mentions of the filename (docstrings,
# TodoWrite reminders, this comment) are not mis-detected. Built via
# string concatenation so this module source itself does not contain the
# literal filename in a location that would defeat "cannot edit the
# hook" self-reference cycles.
_BYPASS_PATH_RE = re.compile(r"[\\/]HOOK_BYPASS_" + r"USER_ONLY\b")


def _references_bypass_file(value) -> bool:
    """Recursively scan a tool_input value for a filesystem reference
    to the bypass file.

    Returns True if any string value (at any nesting depth inside dicts
    and lists) contains a path-like reference to the bypass file —
    i.e. the filename appears immediately after a path separator
    (``/`` or ``\\``). This catches Read/Edit/Write ``file_path``
    targets, Bash commands that touch or redirect to the file,
    filesystem-MCP ``path`` args, Grep searches targeting the file by
    path, etc. Bare prose mentions of the filename (docstrings,
    TodoWrite reminder content, comments in source files) are ignored
    so that documentation and meta-content can refer to the file by
    name without tripping the guard.
    """
    if isinstance(value, str):
        return bool(_BYPASS_PATH_RE.search(value))
    if isinstance(value, dict):
        for v in value.values():
            if _references_bypass_file(v):
                return True
        return False
    if isinstance(value, (list, tuple, set)):
        for v in value:
            if _references_bypass_file(v):
                return True
        return False
    return False


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

    # Phase 3b: AskUserQuestion lazy-pattern vet BEFORE always-allowed short
    # -circuit. The tool stays always-allowed when the question is genuine;
    # only lazy "should I continue / what's next" variants get denied.
    lazy_deny = _maybe_deny_lazy_askuserquestion(tool_name, tool_input)
    if lazy_deny is not None:
        _output(
            _apply_bypass_if_active(
                lazy_deny, denied_reason=lazy_deny["hookSpecificOutput"]["permissionDecisionReason"]
            )
        )
        return

    # For mempalace MCP tools: always allow + inject sessionId via updatedInput
    # Match any plugin ID pattern — versioned IDs vary by install
    # (e.g. mcp__plugin_mempalace_mempalace__*, mcp__plugin_3_0_14_mempalace__*).
    # AskUserQuestion is deliberately excluded from this short-circuit even
    # though it's in ALWAYS_ALLOWED_TOOLS \u2014 the carve-out branch below fires
    # local retrieval on its question text before emitting allow.
    # ── Slice B-2: user-intent tier block-check ─────────────────────────
    # If pending user_message ids exist for this session, deny every
    # tool EXCEPT AskUserQuestion (clarify path) and mempalace_*
    # (declare-and-clear path + memory lookup path). Catches the rest
    # of ALWAYS_ALLOWED (TodoWrite / Skill / Agent / Task* /
    # ExitPlanMode) so the agent cannot dodge the user-intent
    # declaration via "harmless" side calls.
    #
    # Env-var escape: MEMPALACE_USER_INTENT_BLOCK_DISABLED=1 disables
    # the block entirely (rolls back to pre-Slice-B-2 behaviour).
    _is_mempalace_mcp_for_block = tool_name.startswith("mcp__") and "__mempalace_" in tool_name
    _block_disabled = os.environ.get(
        "MEMPALACE_USER_INTENT_BLOCK_DISABLED", ""
    ).strip().lower() in ("1", "true", "yes", "on")
    if not _block_disabled and tool_name != "AskUserQuestion" and not _is_mempalace_mcp_for_block:
        _pending = _read_pending_user_messages(session_id)
        if _pending:
            _pending_ids = [m["id"] for m in _pending if m.get("id")]
            _block_reason = (
                f"BLOCKED: {len(_pending_ids)} pending user_message(s) "
                f"require mempalace_declare_user_intents before any other "
                f"tool call. Pending ids: {_pending_ids}. Only "
                f"AskUserQuestion and mempalace_* tools are allowed until "
                f"you declare. If a covered message has no actionable "
                f"intent, declare it under no_intent=true (with "
                f"no_intent_clarified_with_user=true after asking the "
                f"user via AskUserQuestion)."
            )
            _log(f"PreToolUse DENY {tool_name}: {len(_pending_ids)} pending user_messages")
            _output(
                _apply_bypass_if_active(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": _block_reason,
                        }
                    },
                    denied_reason=_block_reason,
                )
            )
            return

    is_mempalace_mcp = tool_name.startswith("mcp__") and "__mempalace_" in tool_name
    if (tool_name in ALWAYS_ALLOWED_TOOLS and tool_name != "AskUserQuestion") or is_mempalace_mcp:
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

    # For always-allowed tools (Agent, Task*, Skill, TodoWrite,
    # AskUserQuestion, ExitPlanMode, etc.) — pass through unconditionally.
    # Per the 2026-04-21 cue-quality redesign, no retrieval happens at the
    # hook anymore; memories only flow through the MCP tool
    # mempalace_declare_operation (called BEFORE the real tool). If the
    # agent wants memories before an AskUserQuestion, they declare_operation
    # for it explicitly. Keeps the hook a pure permission gate.
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

    # ── HARD BLOCK: bypass-file references ───────────────────────────────
    # Any non-always-allowed tool whose tool_input mentions the break-glass
    # filename (HOOK_BYPASS_USER_ONLY) is denied unconditionally — no intent
    # permission, no scope match, no _apply_bypass_if_active downgrade can
    # reach this file. The bypass file is a USER-only escape hatch and must
    # be managed exclusively from the OS terminal. Agents must never touch
    # it via Read/Edit/Write/Bash/filesystem-MCP/etc. See
    # `record_ga_agent_learning_hook_bypass_user_only` for policy context.
    if _references_bypass_file(tool_input):
        _log(f"PreToolUse HARD-DENY {tool_name}: references bypass file ({_BYPASS_FILE_NAME})")
        _output(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"HARD BLOCK: '{_BYPASS_FILE_NAME}' is a USER-ONLY "
                        "break-glass file. No agent tool is permitted to "
                        "read, write, or otherwise touch this file as a "
                        "filesystem path. Only the human user, operating "
                        "at the OS terminal, may create or delete it. "
                        "This deny is not downgraded by the break-glass "
                        "bypass — the file's whole purpose is to remain "
                        "out of agent reach. (Prose mentions by name in "
                        "docs / TodoWrite / comments are allowed; only "
                        "path-form references trigger this block.)"
                    ),
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

        # MANDATORY declare_operation gate (2026-04-21 cue-quality
        # redesign, simplified). Every non-carve-out tool call MUST be
        # preceded by mempalace_declare_operation for the matching tool
        # name. No env flag, no opt-out, no auto-build fallback — Adrian's
        # design law: "nothing optional survives in an AI tool contract".
        #
        # The retrieval itself happened at declare_operation time (MCP
        # side); its memories reached the agent in that tool's response.
        # The hook no longer runs retrieval or emits additionalContext —
        # duplicating the surface would just be post-dedup noise. The
        # hook's only job on this path is to:
        #   1. Verify a matching cue exists (parallel-batch wait-loop
        #      handles the race where declare + real-tool arrive in the
        #      same assistant message).
        #   2. Consume it (pop + TTL-prune + disk write-back) so the
        #      next call requires a fresh declare.
        #   3. Allow the tool call.
        # Missing or mismatched cue → hard deny with recipe.
        #
        # Carve-outs that skip this gate entirely: ALWAYS_ALLOWED_TOOLS
        # (TodoWrite, Skill, Agent, ToolSearch, AskUserQuestion, Task*,
        # ExitPlanMode) and all mempalace_* MCP tools — handled in the
        # earlier branches above and never reach here.
        cues = intent.get("pending_operation_cues") or []
        has_match = any(isinstance(c, dict) and c.get("tool") == tool_name for c in cues)
        if not has_match:
            # Parallel-race wait: give the MCP server up to
            # OPERATION_CUE_WAIT_TIMEOUT_SECONDS to persist a matching
            # cue the agent declared in the same assistant message.
            has_match = _wait_for_matching_pending_cue(session_id, tool_name, intent)
        if not has_match:
            cues = intent.get("pending_operation_cues") or []
            other_tools = sorted(
                {c.get("tool") for c in cues if isinstance(c, dict) and c.get("tool")}
            )
            if other_tools:
                detail = (
                    f"Pending operation cue(s) are for {other_tools!r} "
                    f"but you tried to call '{tool_name}'. Declare a "
                    f"fresh operation for '{tool_name}' before calling it."
                )
            else:
                detail = (
                    f"No pending_operation_cue for tool '{tool_name}'. "
                    f"Call mempalace_declare_operation(tool='{tool_name}', "
                    "queries=[...2-5...], keywords=[...2-5...], "
                    "agent='<your_agent>') FIRST, then retry this tool "
                    "call. Parallel batches: emit all declares + tool "
                    "calls in the same assistant message; the hook waits "
                    f"up to {OPERATION_CUE_WAIT_TIMEOUT_SECONDS}s for "
                    "the matching declare to land."
                )
            deny_reason = (
                "Every non-carve-out tool call must be preceded by "
                "mempalace_declare_operation so the retrieval cue matches "
                "your actual intention (not the shape of the tool call). " + detail
            )
            _log(f"PreToolUse DENY {tool_name}: {deny_reason}")
            _output(
                _apply_bypass_if_active(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": deny_reason,
                        }
                    },
                    denied_reason=deny_reason,
                )
            )
            return

        # Consume the matching cue: pop + TTL-prune + disk write-back.
        # The returned cue dict carries active_context_id (minted by
        # declare_operation) — we plumb it into _append_trace so
        # finalize_intent can promote rated entries to kind='operation'
        # entities attached to the right context.
        _popped_cue, _expired_n = _consume_pending_operation_cue(session_id, intent, tool_name)
        _op_ctx_id = (_popped_cue or {}).get("active_context_id", "")

        # Accumulate execution trace for finalize_intent.
        _append_trace(session_id, tool_name, tool_input, context_id=_op_ctx_id)

        # Plain allow. No retrieval, no additionalContext — memories
        # already landed at declare_operation time.
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
        "sessionstart": hook_session_start,  # matches pretooluse/precompact no-hyphen convention
        "stop": hook_stop,
        "precompact": hook_precompact,
        "pretooluse": hook_pretooluse,
        "userpromptsubmit": hook_userpromptsubmit,
        "user-prompt-submit": hook_userpromptsubmit,
    }

    handler = hooks.get(hook_name)
    if handler is None:
        print(f"Unknown hook: {hook_name}", file=sys.stderr)
        sys.exit(1)

    handler(data, harness)
