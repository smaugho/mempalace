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
