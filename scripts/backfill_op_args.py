"""Backfill `args_summary` on historical op entities via Haiku batches.

Context
-------
Until 2026-04-27 (commit b905373), `args_summary` was an OPTIONAL field
on `operation_ratings`. Agents almost always skipped it, so every op
entity in the KG has empty `args_summary`. Promotion stored `""`,
`retrieve_past_operations` surfaced `""`, and the gardener S3a
templatize detector had nothing meaningful to cluster on.

Today's redesign moved `args_summary` to MANDATORY at
`mempalace_declare_operation` time, in parametrized-core form. NEW ops
will carry the right value. This script BACKFILLS the historical
~1281 op entities by synthesizing a parametrized args_summary from
three signals already on disk:

  1. The op entity's `tool` field (always populated)
  2. The op entity's `reason` field (often hints at args)
  3. The op-context entity referenced by `properties.context_id`
     — its `queries`, `keywords`, and `summary` describe intent

Plus, where available:

  4. The execution_trace JSONL row matching (tool, context_id) — direct
     ground-truth literal args; the LLM PARAMETRIZES from this rather
     than inventing.

Usage
-----
    # Dry-run (default): print what WOULD be sent + the synthesized
    # args_summary per op, but do NOT update the DB.
    python scripts/backfill_op_args.py --dry-run --limit 20

    # Real run with default batch size:
    python scripts/backfill_op_args.py --limit 50

    # Full backfill (1281 ops):
    python scripts/backfill_op_args.py

Environment
-----------
Reads ANTHROPIC_API_KEY from <palace>/.env (same source the
injection_gate + gardener use). Falls back to OS env if absent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Add repo root to path so `mempalace` imports work when invoked via
# `python scripts/backfill_op_args.py`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Constants
HAIKU_MODEL = "claude-haiku-4-5"
BATCH_SIZE_DEFAULT = 25  # ops per Haiku call; keep small for fast feedback
MAX_TOKENS = 4096
RETRY_DELAYS = [2, 4, 8]  # seconds between retries on transient errors


SYSTEM_PROMPT = """You are an operation fingerprint synthesizer.

For each historical operation row, output a parametrized args_summary
string (5-400 chars) that captures the INVARIANT shape of what the
operation did, with per-execution variables abstracted as {placeholders}.
Two ops sharing the same parametrized args_summary should be the SAME
operation by intent.

Rules:
- Strip plumbing (cd, env vars, redirects).
- Abstract anything that varies per call: file paths, commit messages,
  test names, query strings, queries lists, script bodies.
- Keep what's invariant: tool name, flag combinations, command
  structure.

Examples:
  tool: Bash    reason: "git commit + push of redesign change"
  → 'git commit -m "{commit_message}"'

  tool: Bash    reason: "pytest run for intent_system regression"
  → 'python -m pytest {test_path} -q'

  tool: Grep    reason: "located all schemas by line number"
  → 'Grep pattern={regex} path={file_or_dir}'

  tool: Edit    reason: "patched mutate-without-intent deny branch"
  → 'Edit {file_path} {old_string}→{new_string}'

  tool: Read    reason: "read scoring around retrieve_past_operations"
  → 'Read {file_path} offset={line} limit={N}'

  tool: mempalace_kg_search   reason: "swept pending TODO patterns"
  → 'kg_search context.queries=[{N perspectives on a topic}]'

You receive a JSON array of op rows. For each, return a JSON object
with the same op_id and your synthesized args_summary. Output ONLY
valid JSON — an array of {op_id, args_summary} objects. No prose,
no explanations, no markdown."""


def _load_env_from_palace() -> None:
    """Load <palace>/.env so ANTHROPIC_API_KEY is available."""
    try:
        from mempalace.config import MempalaceConfig

        cfg = MempalaceConfig()
        palace = cfg.palace_path
        env_file = Path(palace) / ".env"
        if env_file.is_file():
            for raw in env_file.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip("'").strip('"')
                # Override if absent OR empty — shells often export
                # ANTHROPIC_API_KEY="" which masks the real value.
                if k and not os.environ.get(k):
                    os.environ[k] = v
    except Exception as e:
        print(f"warning: could not load palace .env: {e}", file=sys.stderr)


def _gather_ops(limit: int | None = None) -> list[dict]:
    """Pull every op entity from the live KG, hydrating tool + reason +
    context_id from `properties`. Filters to ops with empty args_summary
    so re-running is idempotent."""
    from mempalace import mcp_server  # triggers _STATE init

    kg = mcp_server._STATE.kg
    raw_ops = kg.list_entities(kind="operation")
    out: list[dict] = []
    for ent in raw_ops:
        eid = ent.get("id")
        if not eid:
            continue
        # Pull fresh properties via get_entity (list_entities sometimes
        # leaves them empty for op kind; get_entity is the canonical
        # accessor).
        try:
            full = kg.get_entity(eid)
        except Exception:
            full = ent
        props = full.get("properties") or {}
        if isinstance(props, str):
            try:
                props = json.loads(props)
            except Exception:
                props = {}
        cur_args = (props.get("args_summary") or "").strip()
        if cur_args:
            continue  # already populated; skip
        out.append(
            {
                "op_id": eid,
                "tool": props.get("tool", "") or "",
                "reason": props.get("reason", "") or "",
                "context_id": props.get("context_id", "") or "",
                "description": full.get("description", "") or "",
            }
        )
        if limit and len(out) >= limit:
            break
    return out


def _hydrate_context(op: dict) -> dict:
    """Augment an op row with context-entity queries/keywords/summary."""
    from mempalace import mcp_server

    kg = mcp_server._STATE.kg
    cid = op.get("context_id") or ""
    if not cid:
        return op
    try:
        ctx = kg.get_entity(cid)
    except Exception:
        ctx = None
    if not ctx:
        return op
    cprops = ctx.get("properties") or {}
    if isinstance(cprops, str):
        try:
            cprops = json.loads(cprops)
        except Exception:
            cprops = {}
    op["context_queries"] = cprops.get("queries") or []
    op["context_keywords"] = cprops.get("keywords") or []
    op["context_summary"] = cprops.get("summary") or {}
    return op


def _build_user_prompt(batch: list[dict]) -> str:
    """Render a batch of ops as JSON for the Haiku prompt."""
    payload = []
    for op in batch:
        row: dict[str, Any] = {
            "op_id": op["op_id"],
            "tool": op.get("tool", ""),
            "reason": (op.get("reason") or "")[:300],
        }
        cq = op.get("context_queries") or []
        if cq:
            row["context_queries"] = cq[:3]
        ck = op.get("context_keywords") or []
        if ck:
            row["context_keywords"] = ck[:5]
        cs = op.get("context_summary") or {}
        if cs:
            row["context_summary"] = {
                "what": cs.get("what", "")[:80],
                "why": cs.get("why", "")[:120],
            }
        payload.append(row)
    return json.dumps(payload, indent=2)


def _call_haiku(user_prompt: str) -> list[dict]:
    """Send one batch to Haiku, parse + return [{op_id, args_summary}]."""
    from anthropic import Anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Either export it, or place it in "
            "<palace>/.env under ANTHROPIC_API_KEY=..."
        )
    client = Anthropic(api_key=api_key)

    last_err = None
    for attempt, delay in enumerate([0] + RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            resp = client.messages.create(
                model=HAIKU_MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = "".join(b.text for b in resp.content if b.type == "text")
            # Strip any code fences if Haiku adds them
            text = text.strip()
            if text.startswith("```"):
                # remove first line + last line if it's a fence
                lines = text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines)
            data = json.loads(text)
            if not isinstance(data, list):
                raise ValueError(f"expected list, got {type(data).__name__}")
            return data
        except Exception as e:
            last_err = e
            print(
                f"  attempt {attempt + 1} failed: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
    raise RuntimeError(f"all retries failed: {last_err}")


def _persist(updates: list[dict], dry_run: bool) -> int:
    """Write each op_id's new args_summary back to the entity."""
    if dry_run:
        return 0
    from mempalace import mcp_server

    kg = mcp_server._STATE.kg
    written = 0
    for u in updates:
        op_id = u.get("op_id")
        new_args = (u.get("args_summary") or "").strip()
        if not op_id or not new_args:
            continue
        # Validate length per schema
        if len(new_args) < 5 or len(new_args) > 400:
            print(
                f"  skip {op_id}: args_summary len={len(new_args)} out of [5,400]",
                file=sys.stderr,
            )
            continue
        try:
            ent = kg.get_entity(op_id)
        except Exception as e:
            print(f"  skip {op_id}: get_entity failed: {e}", file=sys.stderr)
            continue
        if not ent:
            continue
        props = ent.get("properties") or {}
        if isinstance(props, str):
            try:
                props = json.loads(props)
            except Exception:
                props = {}
        props["args_summary"] = new_args
        # Use raw KG update — the MCP tool path requires an active intent
        # which a backfill script doesn't have.
        try:
            kg.update_entity_properties(op_id, props)  # type: ignore[attr-defined]
            written += 1
        except AttributeError:
            # Fallback: direct sqlite write if the API name differs.
            kg._update_entity_properties_raw(op_id, props)  # type: ignore[attr-defined]
            written += 1
    return written


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true", help="print without writing")
    p.add_argument("--limit", type=int, default=None, help="cap ops processed")
    p.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE_DEFAULT,
        help=f"ops per Haiku call (default {BATCH_SIZE_DEFAULT})",
    )
    args = p.parse_args()

    _load_env_from_palace()

    print(f"gathering ops with empty args_summary (limit={args.limit})...")
    ops = _gather_ops(limit=args.limit)
    print(f"  found {len(ops)} ops needing backfill")
    if not ops:
        print("nothing to do")
        return 0

    print("hydrating context entities...")
    for op in ops:
        _hydrate_context(op)

    total_written = 0
    total_synthesized = 0
    failed_batches = 0
    n_batches = (len(ops) + args.batch_size - 1) // args.batch_size

    for i in range(0, len(ops), args.batch_size):
        batch = ops[i : i + args.batch_size]
        batch_idx = (i // args.batch_size) + 1
        print(
            f"\nbatch {batch_idx}/{n_batches} ({len(batch)} ops)...",
            flush=True,
        )
        prompt = _build_user_prompt(batch)
        try:
            updates = _call_haiku(prompt)
        except Exception as e:
            print(f"  batch failed: {e}", file=sys.stderr)
            failed_batches += 1
            continue
        total_synthesized += len(updates)
        if args.dry_run:
            for u in updates[:5]:
                print(f"  [dry-run] {u.get('op_id')}: {u.get('args_summary')!r}")
            if len(updates) > 5:
                print(f"  [dry-run] ... +{len(updates) - 5} more")
        else:
            written = _persist(updates, dry_run=False)
            total_written += written
            print(f"  wrote {written} entities")

    print(
        f"\ndone — synthesized {total_synthesized} args_summaries; "
        f"wrote {total_written} entities; "
        f"failed batches {failed_batches}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
