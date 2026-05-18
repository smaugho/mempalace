"""Background subsystem status tool.

Surfaces the tail of every telemetry stream the mempalace background
subsystems write to ``~/.mempalace/hook_state/``, so an agent can
diagnose gate / state_judge / retrieval / feedback_auto / mcp_io /
search / hook_errors health and the C-level faulthandler text log
without shelling out to ``tail``/``cat``.

v3.7.0 Slice 0 (Adrian directive 2026-05-16, Option 3 architecture
visibility requirement). This is the smallest piece of Phase A and
the building block the lean-gate refactor + background subsystems
will lean on for observability. Pure additive read-only tool; zero
risk to existing paths.

The on-disk filenames + base dir mirror what
``mcp_server._telemetry_append_jsonl`` and the faulthandler bootstrap
in ``mcp_server`` write to. Adding a new stream is one line in
``_JSONL_STREAMS`` (or ``_TEXT_STREAMS`` for non-jsonl files).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Canonical JSONL stream registry. Keys are short stream names returned
# in the response body; values are the on-disk basename inside
# ``~/.mempalace/hook_state/``. To add a new stream the writer side
# appends to, add one entry here -- no other change required.
_JSONL_STREAMS: dict[str, str] = {
    "gate_log": "gate_log.jsonl",
    "state_judge_log": "state_judge_log.jsonl",
    "retrieval_log": "retrieval_log.jsonl",
    "feedback_auto_log": "feedback_auto_log.jsonl",
    "mcp_io_log": "mcp_io_log.jsonl",
    "search_log": "search_log.jsonl",
    "hook_errors": "hook_errors.jsonl",
    # v3.7.2 Slice 2 (Adrian directive 2026-05-16): background quality
    # pass telemetry. Each row records the wall-clock cost of one bg
    # Haiku call and the count of flag rows written to the gardener
    # feed. n_flags=0 means the bg pass ran but Haiku found nothing
    # to flag; n_flags<0 never happens (sentinel for "skipped").
    "bg_quality_log": "bg_quality_log.jsonl",
    # v3.7.14 (Adrian directive 2026-05-17): wrapper-layer phase
    # timing -- one row per tools/call inside handle_request. Captures
    # sid_switch_ms, coerce_ms, handler_ms, serialize_ms, wrapper_ms
    # (sum incl. emit), outcome ('ok' | 'unknown_tool' |
    # 'handler_exception'), tool name, result_bytes. Pre-v3.7.14 the
    # mcp_io_log only had outer handle_ms with no inner breakdown --
    # the 14s residual between handle_ms and gate+retrieval+judge was
    # a black box. Cross-reference: handler_ms here == handle_ms in
    # mcp_io_log minus serialize+sid+coerce; subtract gate_log +
    # retrieval_log + state_judge_log to attribute the remaining time
    # to declare_operation overhead, past_operations retrieval, context
    # creation/lookup, etc.
    "wrapper_log": "wrapper_log.jsonl",
    # v3.7.19 Slice 1 (Adrian directive 2026-05-17): background Haiku
    # conflict resolver telemetry. One row per pending conflict the
    # bg resolver processed. Fields: conflict_id, conflict_type,
    # existing_id, new_id, similarity, recommended_action (invalidate
    # | merge | keep | skip | abstain), reason, confidence (0.0-1.0),
    # tokens_in/out, elapsed_ms, applied (False this slice --
    # observation only; future slices flip to True when resolver
    # persists). Read this stream to audit Haiku quality before
    # subsequent slices give the resolver write authority.
    "conflict_resolver_log": "conflict_resolver_log.jsonl",
}

# Text streams (not one JSON per line). Tailed as raw lines.
_TEXT_STREAMS: dict[str, str] = {
    "faulthandler": "faulthandler.log",
}

_DEFAULT_LIMIT = 5
_MAX_LIMIT = 50


def _telemetry_dir() -> Path:
    """Resolve the telemetry base dir.

    Mirrors ``mcp_server._telemetry_append_jsonl``'s writer:
    ``~/.mempalace/hook_state``. Tests monkeypatch this function to
    redirect to a tmp dir.
    """
    return Path(os.path.expanduser("~/.mempalace/hook_state"))


def _tail_jsonl(path: Path, limit: int) -> list:
    """Return the last ``limit`` parsed JSON dicts from a JSONL file.

    Soft-fail on parse errors: a malformed line becomes
    ``{'_raw': <line>, '_parse_error': <str>}`` so the tail does not
    drop entries silently. Missing file -> empty list.
    """
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        return [{"_read_error": str(e)}]
    tail = lines[-limit:] if limit > 0 else []
    out: list = []
    for raw in tail:
        line = raw.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception as e:
            out.append({"_raw": line, "_parse_error": str(e)})
    return out


def _tail_text(path: Path, limit: int) -> list:
    """Return the last ``limit`` non-empty lines from a text file."""
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        return [f"[read error] {e}"]
    nonempty = [ln.rstrip("\n") for ln in lines if ln.strip()]
    return nonempty[-limit:] if limit > 0 else []


def _file_meta(path: Path) -> dict:
    """Return ``{exists, size_bytes, mtime_iso}`` for a telemetry file."""
    if not path.exists():
        return {"exists": False, "size_bytes": 0, "mtime_iso": ""}
    try:
        st = path.stat()
        mtime_iso = (
            datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
        return {
            "exists": True,
            "size_bytes": int(st.st_size),
            "mtime_iso": mtime_iso,
        }
    except Exception as e:
        return {
            "exists": True,
            "size_bytes": 0,
            "mtime_iso": "",
            "stat_error": str(e),
        }


def tool_bg_status(
    limit: int = _DEFAULT_LIMIT,
    streams: Optional[list] = None,
) -> dict:
    """Surface the tail of every telemetry stream the mempalace background
    subsystems write to. Read-only diagnostic.

    Args:
        limit: Number of entries (or lines, for text streams) per stream.
            Clamped to ``[1, 50]``. Default 5. Pass higher when you need
            more historical context for a stuck-call diagnosis.
        streams: Optional list of stream names to include. Default = all
            known streams. Unknown names are returned with
            ``kind='unknown', status='unknown_stream'`` so the caller
            sees the typo rather than silent omission.

    Returns:
        ``{base_dir, limit, streams: {<name>: {kind, path, exists,
        size_bytes, mtime_iso, entries|lines}}}``.

        For ``kind='jsonl'`` streams, each ``entries[i]`` is a parsed
        dict (or a parse-error sentinel ``{_raw, _parse_error}``). For
        ``kind='text'`` streams, ``lines[i]`` is the raw line minus its
        trailing newline.
    """
    try:
        limit_i = int(limit) if limit is not None else _DEFAULT_LIMIT
    except (TypeError, ValueError):
        limit_i = _DEFAULT_LIMIT
    if limit_i < 1:
        limit_i = 1
    if limit_i > _MAX_LIMIT:
        limit_i = _MAX_LIMIT

    base = _telemetry_dir()
    known = {**_JSONL_STREAMS, **_TEXT_STREAMS}
    if streams is None:
        names = list(known.keys())
    elif isinstance(streams, list):
        names = [str(s) for s in streams]
    else:
        names = []

    out: dict = {
        "base_dir": str(base),
        "limit": limit_i,
        "streams": {},
    }

    for name in names:
        if name in _JSONL_STREAMS:
            path = base / _JSONL_STREAMS[name]
            meta = _file_meta(path)
            entry: dict = {
                "kind": "jsonl",
                "path": str(path),
            }
            entry.update(meta)
            entry["entries"] = _tail_jsonl(path, limit_i) if meta.get("exists") else []
            out["streams"][name] = entry
        elif name in _TEXT_STREAMS:
            path = base / _TEXT_STREAMS[name]
            meta = _file_meta(path)
            entry = {
                "kind": "text",
                "path": str(path),
            }
            entry.update(meta)
            entry["lines"] = _tail_text(path, limit_i) if meta.get("exists") else []
            out["streams"][name] = entry
        else:
            out["streams"][name] = {
                "kind": "unknown",
                "status": "unknown_stream",
            }
    return out


def tool_pending_user_intents():
    """Return the pending user-message queue for the active session.

    v3.7.31 (Adrian directive 2026-05-18): restart-recovery endpoint.
    The UserPromptSubmit hook is currently the ONLY surface that
    tells the agent which user messages still need a
    ``declare_user_intents`` coverage call. Across an MCP server
    restart, a context compaction, or any other event that resets
    the agent's in-context state, the hook's additionalContext blob
    is lost and the agent has no way to discover what is pending.
    The state IS persisted on disk -- the hook itself writes to
    ``~/.mempalace/hook_state/pending_user_messages_<session_id>.json``
    via ``hooks_cli._append_pending_user_message`` -- but until
    v3.7.31 no MCP read endpoint exposed it. This handler closes
    that gap by wrapping ``_read_pending_user_messages`` and
    returning the queue in a shape the agent can act on directly.

    Returns
    -------
    dict
        ``{session_id, count, pending}`` where ``pending`` is a list
        of ``{id, text, received_at, ...}`` dicts (whatever the hook
        persisted; the agent should treat unknown keys as opaque).
        Empty list when nothing is pending (or when the session-id
        file is absent / malformed -- the underlying helper treats
        absence and corruption identically since both mean "no work
        to cover").

    Carve-out
    ---------
    Read-only diagnostic in the same bucket as ``tool_bg_status``:
    no intent required, safe to call at any point in the session,
    including before the first ``declare_intent``. Idempotent. Sits
    here rather than in tool_lifecycle.py because the drift sentinel
    test (test_hook_buckets.py) enforces a 1:1 mapping between
    tool_lifecycle.__all__ and _LIFECYCLE_BUCKET_BASENAMES; this
    tool is read-bucket, not lifecycle-bucket, so it lives next to
    its bucket-mate ``tool_bg_status``.
    """
    from mempalace.mcp_server import _STATE
    from mempalace import hooks_cli as _hc

    sid = _STATE.session_id or ""
    try:
        pending = _hc._read_pending_user_messages(sid) if sid else []
    except Exception as _e:
        return {
            "session_id": sid,
            "count": 0,
            "pending": [],
            "error": f"{type(_e).__name__}: {_e}",
        }
    return {
        "session_id": sid,
        "count": len(pending),
        "pending": pending,
    }
