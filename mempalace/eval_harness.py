"""
Eval harness for the context-as-entity retrieval redesign (P3).

Reads the JSONL traces written by:
  - tool_kg_search  → `~/.mempalace/hook_state/search_log.jsonl`
  - tool_finalize_intent → `~/.mempalace/hook_state/finalize_log.jsonl`

and computes a small set of retrieval-quality metrics that the
``mempalace eval`` CLI subcommand exposes.

Context reuse rate is the headline metric — it tells you whether the
ColBERT-style MaxSim lookup in ``context_lookup_or_create`` is actually
recognising repeat retrieval contexts, or if T_reuse=0.90 is too tight
for the current corpus. Per-channel RRF contribution reports which
channels fired how often in the top-K; it grounds channel-weight
tuning in observed data (Bruch/Gai/Ingber 2023 ACM TOIS).

JSONL format:
  search: {ts, active_context_id, reused, per_channel_hits, top_k, ...}
  finalize: {ts, intent_id, contexts_used, memories_rated, reuse_at_start}

All functions are pure readers — the harness never mutates state and
can be safely pointed at a stale file or a live one.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path


def default_search_log_path() -> Path:
    return Path.home() / ".mempalace" / "hook_state" / "search_log.jsonl"


def default_finalize_log_path() -> Path:
    return Path.home() / ".mempalace" / "hook_state" / "finalize_log.jsonl"


def _iter_jsonl(path):
    if path is None:
        return
    p = Path(path)
    if not p.exists():
        return
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return


def _parse_ts(s):
    if not s:
        return None
    try:
        # Accept both "...Z" and "+00:00" forms.
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _within_window(ts, days):
    dt = _parse_ts(ts)
    if dt is None:
        return True  # don't drop untimestamped rows from a quick summary
    if days is None or days <= 0:
        return True
    cutoff = datetime.now(dt.tzinfo or timezone.utc) - timedelta(days=days)
    return dt >= cutoff


def context_reuse_rate(search_log_path=None, days=None) -> dict:
    """Fraction of tool_kg_search calls that reused an existing context
    entity (MaxSim ≥ 0.90) over the lookback window."""
    path = search_log_path or default_search_log_path()
    total = 0
    reused = 0
    for row in _iter_jsonl(path):
        if not _within_window(row.get("ts"), days):
            continue
        total += 1
        if bool(row.get("reused")):
            reused += 1
    rate = (reused / total) if total else 0.0
    return {"total_searches": total, "reused": reused, "rate": round(rate, 4)}


def per_channel_contribution(search_log_path=None, days=None) -> dict:
    """How often each channel (cosine / graph / keyword / context)
    contributed to the top-K of a search, averaged across the window.

    Each search log row is expected to carry ``per_channel_hits`` — a
    dict ``{channel: hits_in_top_k}``. Missing rows are skipped.
    """
    path = search_log_path or default_search_log_path()
    counts: Counter = Counter()
    calls = Counter()
    totals = 0
    for row in _iter_jsonl(path):
        if not _within_window(row.get("ts"), days):
            continue
        per = row.get("per_channel_hits") or {}
        if not isinstance(per, dict):
            continue
        totals += 1
        for ch, hits in per.items():
            try:
                counts[ch] += int(hits or 0)
            except (TypeError, ValueError):
                continue
            if hits:
                calls[ch] += 1
    return {
        "total_searches": totals,
        "hits_per_channel": dict(counts),
        "calls_per_channel": dict(calls),
    }


def summary_report(search_log_path=None, finalize_log_path=None, days=None) -> dict:
    """One-shot report combining reuse rate + channel contribution +
    finalize stats. Shape is the ``mempalace eval --report`` payload."""
    search_log_path = search_log_path or default_search_log_path()
    finalize_log_path = finalize_log_path or default_finalize_log_path()

    reuse = context_reuse_rate(search_log_path=search_log_path, days=days)
    channels = per_channel_contribution(search_log_path=search_log_path, days=days)

    # Finalize telemetry.
    finalizes = 0
    memories_rated = 0
    contexts_used = Counter()
    for row in _iter_jsonl(finalize_log_path):
        if not _within_window(row.get("ts"), days):
            continue
        finalizes += 1
        try:
            memories_rated += int(row.get("memories_rated") or 0)
        except (TypeError, ValueError):
            pass
        for cid in row.get("contexts_used") or []:
            contexts_used[cid] += 1

    return {
        "window_days": days,
        "reuse": reuse,
        "channels": channels,
        "finalize": {
            "total_finalizes": finalizes,
            "memories_rated": memories_rated,
            "unique_contexts_used": len(contexts_used),
            "top_contexts": contexts_used.most_common(5),
        },
    }


def format_report(report: dict) -> str:
    """Pretty-print helper for the CLI."""
    lines = []
    win = report.get("window_days")
    lines.append(f"# mempalace-eval report (window: {'all' if not win else f'{win}d'})")
    r = report.get("reuse") or {}
    lines.append(
        f"context reuse rate: {r.get('rate', 0.0):.2%} "
        f"({r.get('reused', 0)}/{r.get('total_searches', 0)} searches)"
    )
    ch = report.get("channels") or {}
    lines.append(f"channels ({ch.get('total_searches', 0)} searches):")
    hits = ch.get("hits_per_channel") or {}
    calls = ch.get("calls_per_channel") or {}
    for name in ("cosine", "graph", "keyword", "context"):
        lines.append(f"  {name:<8} hits={hits.get(name, 0):<6} in {calls.get(name, 0)} searches")
    fin = report.get("finalize") or {}
    lines.append(
        f"finalizes: {fin.get('total_finalizes', 0)}, "
        f"memories rated: {fin.get('memories_rated', 0)}, "
        f"unique contexts: {fin.get('unique_contexts_used', 0)}"
    )
    top = fin.get("top_contexts") or []
    if top:
        lines.append("top contexts:")
        for cid, n in top:
            lines.append(f"  {cid}  used {n}x")
    return "\n".join(lines)
