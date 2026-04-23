"""
memory_gardener.py — Out-of-session corpus-refinement process.

The injection gate reads K memories at once and, alongside keep/drop
decisions, emits quality flags (duplicate_pair, contradiction_pair,
stale, unlinked_entity, orphan, generic_summary, edge_candidate).
The gardener processes those flags asynchronously, investigating each
via a Claude Code subprocess and acting on the palace — merge
duplicates, invalidate contradictions, create missing entity links,
rewrite generic summaries, prune orphans, or push edge_candidate
findings into the existing link_prediction_candidates queue so the
link-author jury decides whether to author the edge.

Why Claude Code and not a bare Haiku call (like link-author):
the gardener must read full memory content, query the KG for related
entities, sometimes read project files to verify a claim, and make
multi-step decisions (merge or invalidate? link or prune?). That's
Claude Code's strength — tool use + exploration + chain-of-thought.
Haiku-alone would give shallow, often-wrong verdicts. Running it as
``claude code`` with a narrow agent prompt and tool allowlist is the
right shape. The underlying MODEL can still be Haiku (fast, cheap);
what matters is the tooling envelope.

Tool allowlist (what the gardener subprocess may use):
  - Bash (investigation — git state, file existence, pwd, etc.)
  - Read, Grep, Glob (read-only palace + project exploration)
  - mempalace_* (all — the gardener's whole job is modifying
    memories/entities/triples via these)
NO Edit / Write on project source files. The gardener refines
memory, not code.

Trigger: every ``finalize_intent`` checks
``kg.count_pending_flags()``. If ≥ ``_FINALIZE_TRIGGER_THRESHOLD``
(default 5), a detached gardener subprocess is spawned so the
current finalize isn't blocked on gardener latency. Same pattern
link-author uses — see mempalace/link_author.py.

Batch size: each gardener invocation processes up to 5 flags per
Claude Code subprocess. One subprocess per batch, not per flag —
gives the model joint context across related flags and reduces
subprocess-spawn overhead.

Logging: every batch writes one row in ``memory_gardener_runs``
with per-action counters (merges, invalidations, links_created,
edges_proposed, summary_rewrites, prunes, deferrals, no_action) +
subprocess exit code + any errors.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 5
_FINALIZE_TRIGGER_THRESHOLD = 5
_DEFAULT_GARDENER_MODEL = "claude-haiku-4-5"


def _resolve_claude_bin() -> str:
    """Return the absolute path to the ``claude`` CLI, or the bare
    name as a last-resort fallback.

    Windows issue (found live 2026-04-23): nvm / global npm installs
    Node CLIs as ``claude.cmd`` — a .cmd shim, not a .exe — and
    subprocess.Popen without ``shell=True`` looks for ``claude.exe``
    and gets FileNotFoundError → exit 127. shutil.which honours
    PATHEXT (so .cmd / .bat / .exe all resolve) and returns the
    full path, which Popen accepts on every platform.

    Resolution order:
      1. ``MEMPALACE_CLAUDE_BIN`` env override (explicit operator
         choice — trust it as-is).
      2. ``shutil.which("claude")`` — picks up .cmd on Windows,
         the bare binary on POSIX.
      3. Bare "claude" — last resort; Popen will fail loud and
         the 127 lands in memory_gardener_runs.errors.
    """
    override = os.environ.get("MEMPALACE_CLAUDE_BIN")
    if override:
        return override
    resolved = shutil.which("claude")
    if resolved:
        return resolved
    return "claude"


_CLAUDE_CLI_BIN = _resolve_claude_bin()


# ═══════════════════════════════════════════════════════════════════
# Prompt builder — what the Claude Code subprocess sees
# ═══════════════════════════════════════════════════════════════════


_SYSTEM_PROMPT = """\
You are the memory_gardener for a MemPalace instance — a background
corpus-refinement agent. The injection gate has flagged quality
issues in the palace by examining K memories jointly during a
retrieval call. Your job is to investigate each flag and ACT using
the mempalace_* tools. You run as Claude Code with a narrow tool
allowlist: Bash, Read, Grep, Glob, and all mempalace_* MCP tools.
You must NOT Edit or Write project source files. Your scope is
memory/entity/triple curation ONLY.

For each flag:
  1. Hydrate — read each referenced memory in full via
     mempalace_kg_query(entity=<id>). Look at content, not just
     the summary. Query related entities if the content implies
     them.
  2. Decide — pick ONE of the resolutions below per flag.
  3. Act — use the appropriate mempalace_* tool.
  4. Report — emit one JSON line per flag in your FINAL output.

Resolutions (must match the outcome exactly):
  - "merged"            duplicate_pair: combine via
                         mempalace_kg_merge_entities OR
                         mempalace_resolve_conflicts with a
                         merged_content that preserves ALL unique
                         detail from both sides.
  - "invalidated"       contradiction_pair / stale: pick which side
                         is stale and call
                         mempalace_kg_invalidate on it.
  - "linked"             unlinked_entity: declare the entity if
                         missing (mempalace_kg_declare_entity) and
                         add a ``mentioned_in`` edge
                         (mempalace_kg_add).
  - "edge_proposed"     edge_candidate: push into
                         link_prediction_candidates via
                         mempalace_kg_add_batch with a
                         candidate-shaped edge, OR write directly
                         to the table if the API exposes it. The
                         link-author jury decides whether to
                         author the edge; you do NOT author it
                         directly.
  - "summary_rewritten" generic_summary: produce a better summary
                         (faithful to content, different angle
                         than content, ≤280 chars) and call
                         mempalace_kg_update_entity.
  - "pruned"             orphan: if investigation confirms the
                         memory has no value in its current form,
                         invalidate it.
  - "deferred"           Investigation inconclusive or requires
                         human judgment. Leave a rationale; the
                         flag's attempted_count is bumped, and it
                         will reappear in the queue up to 3 times.
  - "no_action"          Re-examined and determined the flag was
                         itself a false positive — nothing to do.

OUTPUT FORMAT — your FINAL output (after any tool use) must be
a single JSON object on its own line with this shape:

  {"results": [
     {"flag_id": <int>, "resolution": "<resolution_code>",
      "note": "<one sentence for audit>"},
     …
   ]}

Exactly one entry per input flag, resolution from the closed set
above. The gardener harness parses that line and writes it to
memory_gardener_runs for audit.
"""


def build_flag_prompt(flags: list, *, agent_name: str = "memory_gardener") -> str:
    """Compose the per-batch prompt appended to the system prompt."""
    lines = [
        f"You are {agent_name}. Here are {len(flags)} pending flag(s) "
        "to investigate and act on. Work each flag independently. "
        "After acting, emit the final JSON results line described "
        "in the system prompt."
    ]
    for f in flags:
        lines.append("")
        lines.append(f"FLAG #{f.get('id', '?')}")
        lines.append(f"  kind: {f.get('kind')}")
        mids = f.get("memory_ids") or []
        lines.append(f"  memory_ids: {json.dumps(mids)}")
        lines.append(f"  detail: {f.get('detail') or '(no detail provided)'}")
        ctx = f.get("context_id") or "(none)"
        lines.append(f"  gate_context_id: {ctx}")
        if f.get("attempted_count"):
            lines.append(
                f"  prior_attempts: {f['attempted_count']} (consider why earlier attempts deferred)"
            )
    lines.append("")
    lines.append(
        "Investigate via mempalace_kg_query for each memory id, "
        "then act. When done, emit the results JSON line."
    )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Subprocess runner — default invokes `claude` CLI
# ═══════════════════════════════════════════════════════════════════


@dataclass
class SubprocessResult:
    stdout: str
    stderr: str
    exit_code: int


def _default_subprocess_runner(
    *, system_prompt: str, user_prompt: str, model: str
) -> SubprocessResult:
    """Invoke the Claude Code CLI with a narrow agent config.

    Expects the `claude` binary on PATH (override via
    MEMPALACE_CLAUDE_BIN). The CLI is called with --print (or
    equivalent non-interactive flag) so the subprocess exits after
    one turn. Tool allowlist is enforced via the agent config we
    ship to the subprocess.

    The shape of the actual `claude` CLI invocation differs across
    versions; the default runner here uses the most widely-supported
    flags. Operators who need different flags can inject their own
    runner into process_batch via the ``subprocess_runner``
    parameter.
    """
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    try:
        result = subprocess.run(
            [
                _CLAUDE_CLI_BIN,
                "--print",
                "--model",
                model,
                "--allowed-tools",
                "Bash,Read,Grep,Glob,mcp__*__mempalace_*",
                full_prompt,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        return SubprocessResult(
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            exit_code=int(result.returncode or 0),
        )
    except subprocess.TimeoutExpired as exc:
        return SubprocessResult(
            stdout=exc.stdout.decode("utf-8", "replace") if exc.stdout else "",
            stderr=f"timeout after 600s: {exc}",
            exit_code=124,
        )
    except FileNotFoundError as exc:
        return SubprocessResult(
            stdout="",
            stderr=f"claude CLI not found ({_CLAUDE_CLI_BIN}): {exc}",
            exit_code=127,
        )


# ═══════════════════════════════════════════════════════════════════
# Result parser
# ═══════════════════════════════════════════════════════════════════


_VALID_RESOLUTIONS = frozenset(
    (
        "merged",
        "invalidated",
        "linked",
        "edge_proposed",
        "summary_rewritten",
        "pruned",
        "deferred",
        "no_action",
    )
)


def parse_results_line(stdout: str) -> list[dict]:
    """Extract the last JSON object with a ``results`` key from stdout.

    The Claude Code subprocess may emit intermediate tool-use logs
    before the final JSON line. We scan backward so the canonical
    results object (the LAST one printed) wins over any partial
    drafts. Returns [] on parse failure (harness logs to errors).
    """
    if not stdout:
        return []
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        results = obj.get("results")
        if isinstance(results, list):
            return results
    # Fallback: try the whole stdout as one JSON blob.
    try:
        obj = json.loads(stdout)
        if isinstance(obj, dict) and isinstance(obj.get("results"), list):
            return obj["results"]
    except json.JSONDecodeError:
        pass
    return []


# ═══════════════════════════════════════════════════════════════════
# Batch processor — the public entry point
# ═══════════════════════════════════════════════════════════════════


_RESOLUTION_TO_COUNTER = {
    "merged": "merges",
    "invalidated": "invalidations",
    "linked": "links_created",
    "edge_proposed": "edges_proposed",
    "summary_rewritten": "summary_rewrites",
    "pruned": "prunes",
    "deferred": "deferrals",
    "no_action": "no_action",
}


def process_batch(
    kg,
    *,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    model: str = _DEFAULT_GARDENER_MODEL,
    subprocess_runner: Callable[..., SubprocessResult] | None = None,
) -> dict:
    """Run ONE gardener batch: read pending flags, invoke the Claude
    Code subprocess, apply resolutions, log the run.

    Returns a summary dict with ``run_id``, ``flag_ids``,
    ``results``, and ``counters`` — useful for tests and the CLI
    status subcommand.

    subprocess_runner is injectable so tests run with a canned
    response; in production it defaults to invoking the ``claude``
    CLI.
    """
    runner = subprocess_runner or _default_subprocess_runner

    flags = kg.list_pending_flags(limit=batch_size)
    if not flags:
        return {
            "run_id": None,
            "flag_ids": [],
            "results": [],
            "counters": {},
            "note": "no pending flags",
        }

    run_id = kg.start_gardener_run(gardener_model=model)
    flag_ids = [int(f["id"]) for f in flags]

    user_prompt = build_flag_prompt(flags, agent_name="memory_gardener")
    sub = runner(system_prompt=_SYSTEM_PROMPT, user_prompt=user_prompt, model=model)

    results = parse_results_line(sub.stdout)
    counters: dict = {}
    errors = sub.stderr if sub.exit_code != 0 else ""

    resolved_ids: set[int] = set()
    for entry in results:
        if not isinstance(entry, dict):
            continue
        fid = entry.get("flag_id")
        resolution = entry.get("resolution")
        note = str(entry.get("note") or "")
        if not isinstance(fid, int) or fid not in flag_ids:
            continue
        if resolution not in _VALID_RESOLUTIONS:
            continue
        try:
            kg.mark_flag_resolved(fid, resolution, note=note)
        except Exception as exc:
            log.info(
                "memory_gardener: mark_flag_resolved failed for %s: %s",
                fid,
                exc,
            )
            continue
        counter = _RESOLUTION_TO_COUNTER.get(resolution)
        if counter:
            counters[counter] = counters.get(counter, 0) + 1
        resolved_ids.add(fid)

    # Any flag the subprocess didn't resolve: bump attempted_count
    # so a stuck flag reaches the ceiling eventually and stops
    # clogging the queue.
    for fid in flag_ids:
        if fid in resolved_ids:
            continue
        try:
            kg.bump_flag_attempt(fid)
        except Exception:
            pass

    kg.finish_gardener_run(
        run_id,
        flag_ids=flag_ids,
        counters=counters,
        subprocess_exit_code=sub.exit_code,
        errors=errors,
    )

    return {
        "run_id": run_id,
        "flag_ids": flag_ids,
        "results": results,
        "counters": counters,
        "exit_code": sub.exit_code,
        "errors": errors,
    }


# ═══════════════════════════════════════════════════════════════════
# Finalize-triggered detached spawn
# ═══════════════════════════════════════════════════════════════════


def maybe_trigger_from_finalize(kg) -> bool:
    """Called at the tail of finalize_intent. If the pending-flag
    count has reached the trigger threshold, spawn a detached
    gardener subprocess (via the CLI) so finalize doesn't block
    on gardener latency. Returns True if a subprocess was spawned.

    Mirrors the pattern link-author uses — see
    docs/link_author_scheduling.md §2.
    """
    try:
        pending = kg.count_pending_flags()
    except Exception:
        return False
    if pending < _FINALIZE_TRIGGER_THRESHOLD:
        return False
    # Note: we deliberately do NOT load ANTHROPIC_API_KEY here. The
    # claude CLI the subprocess invokes authenticates via its OAuth
    # subscription (stored per-user by the CLI itself), not via
    # ANTHROPIC_API_KEY. Only the gate (direct anthropic SDK) needs
    # the env-loaded key; the gardener inherits no secret material.
    try:
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mempalace",
                "gardener",
                "process",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except Exception as exc:
        log.info("memory_gardener: detached spawn failed: %s", exc)
        return False


# ═══════════════════════════════════════════════════════════════════
# CLI entry — wired in cli.py
# ═══════════════════════════════════════════════════════════════════


def cli_process(args) -> int:
    """Handler for ``mempalace gardener process``.

    Drains up to ``--max-batches`` batches (default 1) from the
    pending-flag queue. Each batch is one Claude Code subprocess.
    Exits non-zero if ANY batch reports a non-zero subprocess exit
    code so the caller (cron / finalize trigger) surfaces failures.

    The claude CLI the subprocess invokes authenticates via its
    OAuth subscription — no ANTHROPIC_API_KEY needed, nothing for
    us to inject into the subprocess environment.
    """
    from . import mcp_server  # lazy to avoid import cycles at module load

    kg = mcp_server._STATE.kg
    if kg is None:
        print("[memory_gardener] KnowledgeGraph not initialised", file=sys.stderr)
        return 2
    max_batches = int(getattr(args, "max_batches", 1) or 1)
    batch_size = int(getattr(args, "batch_size", _DEFAULT_BATCH_SIZE) or _DEFAULT_BATCH_SIZE)
    model = getattr(args, "model", None) or _DEFAULT_GARDENER_MODEL
    any_failed = False
    for i in range(max_batches):
        result = process_batch(kg, batch_size=batch_size, model=model)
        if not result.get("flag_ids"):
            break
        exit_code = result.get("exit_code") or 0
        if exit_code != 0:
            any_failed = True
        print(
            f"[memory_gardener] batch {i + 1}: "
            f"run_id={result['run_id']} "
            f"flags={len(result['flag_ids'])} "
            f"counters={result['counters']} "
            f"exit={exit_code}"
        )
    return 3 if any_failed else 0
