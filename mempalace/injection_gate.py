"""
injection_gate.py — Post-retrieval relevance gate.

The gate sits between the retriever and the main agent's working
context. It decides per-item keep/drop: the RRF-ranked list coming
out of retrieval is unchanged; only what the GENERATOR sees is
filtered. Dropped items never enter accessed_memory_ids — the gate
writes their rated_irrelevant feedback itself via KnowledgeGraph
.record_feedback with rater_kind='gate_llm'.

Research grounding
------------------
* Empirical distraction effect: Zhou et al. ACL 2025
  (arXiv:2505.06914) — irrelevant retrievals measurably degrade
  generation; stronger retrievers produce MORE distracting
  irrelevants because they look semantically plausible.
* Self-RAG (Asai et al. ICLR 2024) — ISREL reflection tokens gate
  each retrieved passage post-retrieval. This gate is the
  palace-shaped equivalent.
* Adaptive-RAG (Jeong et al. 2024) — classifier decides whether to
  retrieve; here we always retrieve but filter injection.

Design principles
-----------------
* Bias-to-keep: if an item relates to the primary context in any
  way, KEEP. Drop only when clearly unrelated (different project /
  domain / thread). Stated twice in the prompt because LLM judges
  default to being discriminating.
* Structured output: forced tool-use via ``tool_choice`` eliminates
  JSON-parse failures.
* Fail-open: any API or parse failure returns ``gate_status.state =
  'degraded'`` and passes all items through unfiltered. The main
  agent is instructed (via the session-start protocol) to surface
  degraded gates to the user.
* Project disambiguation via cwd anchor: cwd is included in the
  session frame ONLY when the directory contains a project anchor
  file (pyproject.toml / package.json / .git / …). Otherwise we
  omit it to avoid leaking misleading project tags.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

# Directory anchor files used for project-root detection. cwd is
# injected into the session frame only when at least one of these is
# present in the process's current working directory. Keep this list
# in sync with the cwd-anchor decision recorded in the wrap-up memory
# wrap_up_injection_gate_design_decisions_2026_04_23.
_PROJECT_ANCHORS = (
    "pyproject.toml",
    "package.json",
    ".git",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "Gemfile",
    "CMakeLists.txt",
    "composer.json",
    "mix.exs",
    "requirements.txt",
    "setup.py",
)

# Hard cap on per-item content rendered into the prompt. Haiku 4.5
# has 200K context; K × 5K memories fits comfortably. The cap is
# defensive — a single runaway memory shouldn't blow the budget.
_MAX_ITEM_CHARS = 6000

# Default model. The runtime can override via MempalaceConfig or the
# ``MEMPALACE_GATE_MODEL`` env var.
_DEFAULT_MODEL = "claude-haiku-4-5"


# ═══════════════════════════════════════════════════════════════════
# Data shapes
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class GateItem:
    """One retrieval candidate the gate must judge.

    Shape mirrors what the three retrieval sites (declare_intent,
    declare_operation, kg_search) naturally have on hand: an id, the
    source namespace (memory/entity/triple), the document/statement
    text, the channel that produced it (A/B/C/D), rank, and score.
    ``extra`` carries per-source detail the prompt renderer uses
    (e.g. triple subject/predicate/object, entity kind + description,
    memory summary).
    """

    id: str
    source: Literal["memory", "entity", "triple"]
    text: str
    channel: str  # "A" (cosine), "B" (graph), "C" (keyword), "D" (context)
    rank: int
    score: float
    extra: dict = field(default_factory=dict)


@dataclass
class GateDecision:
    id: str
    action: Literal["keep", "drop"]
    reasoning: str
    proposed_summary: str | None = None


@dataclass
class GateResult:
    kept: list[GateItem]
    dropped: list[tuple[GateItem, GateDecision]]
    gate_status: dict
    judge_tokens_in: int = 0
    judge_tokens_out: int = 0
    # Quality-issue flags the judge emitted across the K items
    # together. Each is a dict: {kind, memory_ids, detail}.
    # Persisted by apply_gate to the memory_flags table for the
    # memory_gardener to investigate later.
    flags: list[dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# Prompt builder
# ═══════════════════════════════════════════════════════════════════


_SYSTEM_PROMPT = (
    "You are the relevance gate AND quality inspector for a memory "
    "palace. Two jobs per call, both important.\n\n"
    "JOB 1 — KEEP / DROP. For each retrieved item, decide INJECT "
    "(keep) or SUPPRESS (drop). Be GENEROUS toward keep: if an item "
    "relates to the primary context in any way — shares a topic, "
    "touches a mentioned entity, records a prior thread on the same "
    "question, informs a tangential decision — mark it KEEP. Mark "
    "DROP only if the item is clearly from a different project / "
    "domain / thread and would add noise without signal. A low-"
    "importance but on-topic item is KEEP. Project mismatch is a "
    "strong drop signal. Importance alone is never a keep signal.\n\n"
    "Channel provenance is informative. Channel D (context-walk) "
    "items are already upvoted by past behaviour on this or similar "
    "contexts — lean toward keep there even if content looks "
    "tangential. Channel A (cosine) items matched the primary "
    "context's text; Channel B (graph) items are neighbours of "
    "seed entities; Channel C (keyword) hits are exact-term matches.\n\n"
    "Before emitting each decision, write one sentence explaining "
    "the item's relation (or non-relation) to the PRIMARY CONTEXT. "
    "If you keep an item whose current summary is generic while "
    "its content is specific, propose a better summary (≤280 chars, "
    "faithful to the content, written from a different angle than "
    "the content itself).\n\n"
    "Repeat the rule to yourself: BIAS TO KEEP. DROP only when the "
    "item is clearly unrelated.\n\n"
    "JOB 2 — FLAG QUALITY ISSUES. This is VERY IMPORTANT and you "
    "must attend to it every call. You are looking at K memories "
    "together — a rare joint vantage point a single-memory process "
    "never has. Use it. Emit flags when you see:\n"
    "  • duplicate_pair — two items state the same fact.\n"
    "  • contradiction_pair — two items contradict each other on a "
    "specific claim (dates, identities, outcomes, …).\n"
    "  • stale — an item's facts look outdated relative to the "
    "context (superseded decisions, renamed entities, …).\n"
    "  • unlinked_entity — an item clearly mentions a person, "
    "project, location, file, or system that does NOT appear in "
    "the primary context's entities list and is probably missing a "
    "KG link.\n"
    "  • orphan — an item lists no entities in its meta and seems "
    "to describe something concrete — probably lost its entity "
    "links and needs re-anchoring.\n"
    "  • generic_summary — the summary is boilerplate or template "
    "while the content is specific and informative. ALSO flag when: "
    "(a) the item's description starts with '[AUTO' or contains "
    "'needs refinement' — self-identifying placeholder from an "
    "auto-mint path, (b) the description is a bare 'File: <path>' "
    "stub with no WHAT/WHY, (c) the description just restates the "
    "entity's name with no added meaning. You may also propose a "
    "better summary per-item as described above — prefer proposing "
    "a rewrite over flagging when you can.\n"
    "  • edge_candidate — the content strongly implies a factual "
    "relationship between two named entities that the KG probably "
    "doesn't have (e.g. 'A replaces B', 'A depends on B', 'A was "
    "built by B'). Include the two entity ids in memory_ids and the "
    "suggested predicate in detail. Do NOT author edges here — the "
    "link-author jury owns that decision; you only flag the "
    "candidate for it.\n\n"
    "Flags are OPTIONAL — if nothing stands out, return flags: []. "
    "But do not skip this job. A perfect call catches every issue "
    "a human operator would have caught reading the K items side-"
    "by-side. Under-flagging is a failure mode; over-flagging is "
    "recoverable (the memory_gardener investigates each and can "
    "defer). When in doubt, flag."
)


def _detect_project_anchor(cwd: str | None) -> str | None:
    """Return project name (cwd basename) only when an anchor file
    exists in cwd. Otherwise return None.

    This is the only check we perform before injecting cwd-based
    project info into the gate prompt. If the anchor check fails we
    omit the project tag silently rather than inject a potentially
    wrong one.
    """
    if not cwd:
        return None
    try:
        p = Path(cwd)
        if not p.is_dir():
            return None
        for anchor in _PROJECT_ANCHORS:
            if (p / anchor).exists():
                return p.name
    except OSError:
        return None
    return None


def _render_item(item: GateItem) -> str:
    """Render one retrieved item for the judge prompt."""
    text = (item.text or "").strip()
    if len(text) > _MAX_ITEM_CHARS:
        text = text[:_MAX_ITEM_CHARS] + " …[truncated]"

    lines = [
        f"  [{item.rank}] id={item.id}",
        f"      source: {item.source}   channel: {item.channel}   "
        f"rank: {item.rank}   score: {item.score:.3f}",
    ]

    if item.source == "triple":
        subj = item.extra.get("subject", "?")
        pred = item.extra.get("predicate", "?")
        obj = item.extra.get("object", "?")
        conf = item.extra.get("confidence", 1.0)
        lines.append(f"      statement: {text}")
        lines.append(
            f"      subject: {subj}   predicate: {pred}   object: {obj}   confidence: {conf}"
        )
    elif item.source == "entity":
        name = item.extra.get("name", item.id)
        kind = item.extra.get("kind", "entity")
        lines.append(f"      name: {name}   kind: {kind}")
        if text:
            lines.append(f"      description: {text}")
    else:  # memory
        summary = item.extra.get("summary") or ""
        if summary:
            lines.append(f"      summary: {summary}")
        if text:
            lines.append(f"      content: {text}")
    return "\n".join(lines)


def build_prompt(
    *,
    primary_context: dict,
    items: list[GateItem],
    parent_intent: dict | None = None,
    session_frame: dict | None = None,
) -> str:
    """Compose the user-message body for the judge.

    primary_context: {queries: [...], keywords: [...], entities: [...]}
    parent_intent (optional): {intent_type, subject, queries[0]}
    session_frame (optional): {agent, project, recent_intents: [...]}
    """
    parts = []

    if session_frame:
        frame_lines = ["SESSION FRAME"]
        if session_frame.get("agent"):
            frame_lines.append(f"  agent: {session_frame['agent']}")
        if session_frame.get("project"):
            frame_lines.append(f"  project (cwd-inferred): {session_frame['project']}")
        if len(frame_lines) > 1:
            parts.append("\n".join(frame_lines))

    if parent_intent:
        p = parent_intent
        parent_lines = ["PARENT FRAME (enclosing intent)"]
        if p.get("intent_type"):
            parent_lines.append(f"  intent_type: {p['intent_type']}")
        if p.get("subject"):
            parent_lines.append(f"  subject: {p['subject']}")
        if p.get("query"):
            parent_lines.append(f"  query: {p['query']}")
        parts.append("\n".join(parent_lines))

    pc_lines = ["PRIMARY CONTEXT (this retrieval's context — judge against THIS)"]
    if primary_context.get("source"):
        pc_lines.append(f"  source: {primary_context['source']}")
    for q in (primary_context.get("queries") or [])[:5]:
        pc_lines.append(f"  query: {q}")
    if primary_context.get("keywords"):
        pc_lines.append("  keywords: [" + ", ".join(primary_context["keywords"][:8]) + "]")
    if primary_context.get("entities"):
        pc_lines.append("  entities: [" + ", ".join(primary_context["entities"][:10]) + "]")
    parts.append("\n".join(pc_lines))

    retrieved_header = f"RETRIEVED ITEMS (K={len(items)})"
    parts.append(retrieved_header + "\n" + "\n".join(_render_item(it) for it in items))

    parts.append(
        "Emit one decision per item, in the SAME ORDER, via the "
        "gate_decisions tool. Every id must appear exactly once."
    )
    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════
# Tool schema — forced structured output
# ═══════════════════════════════════════════════════════════════════


_FLAG_KINDS_ENUM = [
    "duplicate_pair",
    "contradiction_pair",
    "stale",
    "unlinked_entity",
    "orphan",
    "generic_summary",
    "edge_candidate",
]


GATE_DECISIONS_TOOL = {
    "name": "gate_decisions",
    "description": (
        "Emit a keep/drop decision for every input item, in input "
        "order. Exactly one entry per item id. ALSO emit a flags "
        "array capturing any quality issues visible across the K "
        "items together — this is the second job of the call and "
        "is VERY IMPORTANT. Empty flags is allowed when nothing "
        "stands out, but under-flagging is a failure mode."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "reasoning": {
                            "type": "string",
                            "description": (
                                "One sentence: how does this item relate "
                                "to the primary context, or why doesn't it?"
                            ),
                        },
                        "action": {"enum": ["keep", "drop"]},
                        "proposed_summary": {
                            "type": "string",
                            "description": (
                                "OPTIONAL. Set only if action=keep AND "
                                "the item's current summary is generic "
                                "while its content is specific. ≤280 "
                                "chars, faithful to content, different "
                                "angle than the content."
                            ),
                        },
                    },
                    "required": ["id", "reasoning", "action"],
                },
            },
            "flags": {
                "type": "array",
                "description": (
                    "Quality issues visible across the K items. Empty "
                    "array if nothing stands out. See the system "
                    "prompt's JOB 2 section for the flag taxonomy."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"enum": _FLAG_KINDS_ENUM},
                        "memory_ids": {
                            "type": "array",
                            "description": (
                                "Item ids this flag is about. Pair kinds "
                                "(duplicate_pair, contradiction_pair, "
                                "edge_candidate) carry exactly 2 ids in "
                                "subject→object order for edge_candidate. "
                                "Single-memory kinds carry 1 id."
                            ),
                            "items": {"type": "string"},
                        },
                        "detail": {
                            "type": "string",
                            "description": (
                                "One sentence explaining the issue. For "
                                "edge_candidate, include the proposed "
                                "predicate (e.g. 'depends_on', "
                                "'replaces', 'built_by')."
                            ),
                        },
                    },
                    "required": ["kind", "memory_ids", "detail"],
                },
            },
        },
        "required": ["decisions"],
    },
}


# ═══════════════════════════════════════════════════════════════════
# Gate runtime
# ═══════════════════════════════════════════════════════════════════


class InjectionGate:
    """Relevance gate wrapper. One instance per palace process.

    Usage::

        gate = InjectionGate()
        result = gate.filter(
            primary_context={...},
            items=[GateItem(...), ...],
            parent_intent={...} or None,
        )
        # result.kept flows to the main agent
        # result.dropped is written back as rated_irrelevant by the
        # caller (via kg.record_feedback(rater_kind='gate_llm'))

    Safety: construction never contacts the API. The client is
    built lazily on first filter() call. If the SDK or key is
    missing, filter() returns GateResult with gate_status='degraded'
    and all items in kept (fail-open).
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_retries: int = 2,
        min_items: int = 3,
        _client=None,  # test injection
    ):
        self.model = model or os.environ.get("MEMPALACE_GATE_MODEL") or _DEFAULT_MODEL
        self.api_key_env = api_key_env
        self.max_retries = max_retries
        self.min_items = min_items
        self._client = _client
        self._client_initialized = _client is not None

    # ── Lazy client init ──

    def _get_client(self):
        if self._client_initialized:
            return self._client
        self._client_initialized = True
        try:
            import anthropic
        except ImportError as exc:
            log.warning("injection_gate: anthropic SDK not available: %s", exc)
            self._client = None
            return None
        # Load the palace .env if the key isn't already in the process
        # environment. Link-author's CLI does this at process start,
        # but the MCP server that hosts the gate does not — without
        # this call the gate would be permanently key-blind despite
        # the operator having set the key in the documented place.
        # Mirrors mempalace.link_author._load_env with override=True
        # so a stale shell var can't shadow the .env.
        _ensure_palace_env_loaded(self.api_key_env)
        key = os.environ.get(self.api_key_env) or ""
        if not key.strip():
            log.info("injection_gate: %s not set, gate will fail-open", self.api_key_env)
            self._client = None
            return None
        try:
            self._client = anthropic.Anthropic(api_key=key)
        except Exception as exc:
            log.warning("injection_gate: client construction failed: %s", exc)
            self._client = None
        return self._client

    # ── Public API ──

    def filter(
        self,
        *,
        primary_context: dict,
        items: list[GateItem],
        parent_intent: dict | None = None,
        session_frame: dict | None = None,
    ) -> GateResult:
        """Filter retrieved items. See module docstring for semantics."""
        # K=0: pass-through, no API call.
        if not items:
            return GateResult(
                kept=[],
                dropped=[],
                gate_status={"state": "skipped_empty"},
            )

        # K below min_items: not worth the latency; pass all through.
        # This matches the design decision that very small K doesn't
        # benefit from gating (the agent can eyeball two items).
        if len(items) < self.min_items:
            return GateResult(
                kept=list(items),
                dropped=[],
                gate_status={
                    "state": "skipped_small_k",
                    "k": len(items),
                    "min_k": self.min_items,
                },
            )

        client = self._get_client()
        if client is None:
            # Distinguish "no key configured" (operator chose not to
            # run the gate — NOT a runtime failure) from "runtime
            # degradation" (network, timeout, malformed response).
            # Happy-path callers treat this as a silent pass-through
            # and do NOT inject gate_status into their response.
            return GateResult(
                kept=list(items),
                dropped=[],
                gate_status={
                    "state": "skipped_no_client",
                    "reason": "anthropic_sdk_or_key_missing",
                },
            )

        prompt = build_prompt(
            primary_context=primary_context,
            items=items,
            parent_intent=parent_intent,
            session_frame=session_frame,
        )

        # Forced tool-use: Anthropic guarantees the response uses the
        # named tool, so the decisions arrive as structured arguments
        # — no free-text JSON parsing.
        last_err = None
        parsed: tuple[dict[str, GateDecision], list[dict]] | None = None
        tokens_in = 0
        tokens_out = 0
        for attempt in range(self.max_retries):
            try:
                resp = client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=_SYSTEM_PROMPT,
                    tools=[GATE_DECISIONS_TOOL],
                    tool_choice={"type": "tool", "name": "gate_decisions"},
                    messages=[{"role": "user", "content": prompt}],
                )
                usage = getattr(resp, "usage", None)
                if usage:
                    tokens_in = int(getattr(usage, "input_tokens", 0) or 0)
                    tokens_out = int(getattr(usage, "output_tokens", 0) or 0)
                parsed = _extract_decisions(resp, {it.id for it in items})
                if parsed is not None:
                    break
                last_err = "missing_decisions_in_tool_call"
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                log.info("injection_gate attempt %d failed: %s", attempt + 1, last_err)

        if parsed is None:
            return self._fail_open(
                items,
                reason=f"judge_failed_after_{self.max_retries}_attempts: {last_err}",
                instruction=(
                    "Relevance gate failed this turn. All items injected "
                    "unfiltered. Surface this to the user — retrieval "
                    "quality may be reduced. Consider whether to proceed "
                    "or abort, and note the failure in your response."
                ),
            )

        decisions_by_id, flags = parsed

        # Route items by decision. Missing decisions fail-open for
        # that item (kept with a synthetic decision noting the miss).
        kept: list[GateItem] = []
        dropped: list[tuple[GateItem, GateDecision]] = []
        for item in items:
            dec = decisions_by_id.get(item.id)
            if dec is None:
                kept.append(item)
                continue
            if dec.action == "drop":
                dropped.append((item, dec))
            else:
                kept.append(item)

        return GateResult(
            kept=kept,
            dropped=dropped,
            gate_status={"state": "ok"},
            judge_tokens_in=tokens_in,
            judge_tokens_out=tokens_out,
            flags=flags,
        )

    # ── Helpers ──

    def _fail_open(self, items: list[GateItem], *, reason: str, instruction: str) -> GateResult:
        return GateResult(
            kept=list(items),
            dropped=[],
            gate_status={
                "state": "degraded",
                "reason": reason,
                "fallback": f"all {len(items)} items injected unfiltered",
                "agent_instruction": instruction,
            },
        )


# ═══════════════════════════════════════════════════════════════════
# Tool-call parser
# ═══════════════════════════════════════════════════════════════════


def _extract_decisions(
    resp, known_ids: set[str]
) -> tuple[dict[str, GateDecision], list[dict]] | None:
    """Parse a forced tool-use response into (decisions_by_id, flags).

    Returns None if the expected tool_use block is absent or malformed
    — the caller interprets that as "retry then fail-open".

    Unknown ids (hallucinated by the model) are dropped. Duplicate ids
    keep the first decision. Flags are filtered to valid kinds + at
    least one known memory id per flag; malformed entries are dropped
    silently so a bad flag can't sink the whole response.
    """
    blocks = getattr(resp, "content", None) or []
    for block in blocks:
        if getattr(block, "type", None) != "tool_use":
            continue
        if getattr(block, "name", None) != "gate_decisions":
            continue
        inp = getattr(block, "input", None) or {}
        raw_decisions = inp.get("decisions") if isinstance(inp, dict) else None
        if not isinstance(raw_decisions, list):
            return None
        by_id: dict[str, GateDecision] = {}
        for d in raw_decisions:
            if not isinstance(d, dict):
                continue
            did = d.get("id")
            action = d.get("action")
            reasoning = d.get("reasoning") or ""
            proposed = d.get("proposed_summary")
            if not isinstance(did, str) or did not in known_ids or action not in ("keep", "drop"):
                continue
            if did in by_id:
                continue
            by_id[did] = GateDecision(
                id=did,
                action=action,
                reasoning=str(reasoning),
                proposed_summary=str(proposed) if isinstance(proposed, str) else None,
            )

        # Flags are optional. An absent key is not malformed — it's
        # "judge had nothing to flag this call".
        raw_flags = inp.get("flags") if isinstance(inp, dict) else None
        flags: list[dict] = []
        if isinstance(raw_flags, list):
            for f in raw_flags:
                if not isinstance(f, dict):
                    continue
                kind = f.get("kind")
                if kind not in _FLAG_KINDS_ENUM:
                    continue
                mids = f.get("memory_ids")
                if not isinstance(mids, list):
                    continue
                cleaned_ids = [str(m) for m in mids if isinstance(m, str) and m]
                if not cleaned_ids:
                    continue
                flags.append(
                    {
                        "kind": kind,
                        "memory_ids": cleaned_ids,
                        "detail": str(f.get("detail") or ""),
                    }
                )
        return by_id, flags
    return None


# ═══════════════════════════════════════════════════════════════════
# Caller-side helper: write dropped items back as rated_irrelevant.
# ═══════════════════════════════════════════════════════════════════


def persist_drops(
    kg,
    *,
    context_id: str,
    dropped: list[tuple[GateItem, GateDecision]],
    rater_id: str = "claude-haiku-gate",
) -> int:
    """Write rated_irrelevant feedback for every dropped item.

    Uses KnowledgeGraph.record_feedback so both entity-scope and
    triple-scope feedback go through the unified dispatcher —
    entity-target drops become rated_irrelevant edges on context →
    entity; triple-target drops become rows in
    triple_context_feedback. No phantom entities.

    relevance=2 (rated_irrelevant, non-misleading noise) mirrors the
    user's guidance that the gate is bias-to-keep; a gate DROP is
    "on-topic enough to have surfaced but still not useful for this
    context" — squarely relevance=2 rather than =1 (misleading).

    Returns the number of successful writes. Failures are logged but
    non-fatal; the caller should treat this as best-effort.
    """
    if not dropped or not context_id:
        return 0
    n = 0
    for item, dec in dropped:
        target_kind = "triple" if item.source == "triple" else "entity"
        try:
            kg.record_feedback(
                context_id,
                item.id,
                target_kind,
                relevance=2,
                reason=f"[gate] {dec.reasoning[:400]}",
                rater_kind="gate_llm",
                rater_id=rater_id,
            )
            n += 1
        except Exception as exc:  # pragma: no cover — best-effort
            log.info(
                "persist_drops: record_feedback failed for %s: %s",
                item.id,
                exc,
            )
    return n


# ═══════════════════════════════════════════════════════════════════
# Session-frame helper
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
# Palace .env loader — shared with memory_gardener for env parity
# ═══════════════════════════════════════════════════════════════════


_PALACE_ENV_LOADED = False


def _ensure_palace_env_loaded(api_key_env: str = "ANTHROPIC_API_KEY") -> None:
    """Load <palace>/.env into os.environ the first time any gate /
    gardener code path needs an API key.

    Link-author's CLI calls ``_load_env`` at process start because its
    entry point is a dedicated subcommand. The gate instead runs inside
    the long-lived MCP server process, which never had a documented
    place to load the .env — so the key was invisible even though the
    file was present. This helper closes that gap exactly once per
    process, at the moment the first client is constructed. override
    is True so a stale / empty shell var can't shadow the file.

    No-op when the key is already in os.environ (shell-set), when the
    palace path can't be resolved, when the .env file is absent, or
    when python-dotenv is missing (hard dep but defensive).
    """
    global _PALACE_ENV_LOADED
    if _PALACE_ENV_LOADED:
        return
    _PALACE_ENV_LOADED = True
    if os.environ.get(api_key_env, "").strip():
        # Already in process env — shell-set, parent-inherited, or a
        # prior load. Nothing to do.
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        log.info("injection_gate: python-dotenv missing; skipping .env load")
        return
    try:
        from .config import MempalaceConfig

        cfg = MempalaceConfig()
        palace = getattr(cfg, "palace_path", None)
    except Exception as exc:
        log.info("injection_gate: MempalaceConfig unavailable: %s", exc)
        palace = None
    if not palace:
        return
    from pathlib import Path

    target = Path(palace) / ".env"
    if not target.is_file():
        log.info("injection_gate: no .env at %s; skipping", target)
        return
    try:
        load_dotenv(str(target), override=True)
        log.info("injection_gate: palace .env loaded from %s", target)
    except Exception as exc:
        log.info("injection_gate: load_dotenv failed: %s", exc)


def build_session_frame(
    *,
    agent: str | None,
    cwd: str | None = None,
) -> dict:
    """Assemble the optional session-frame block for the gate prompt.

    Only includes cwd-derived ``project`` when an anchor file is
    present in cwd (prevents misleading project tags from sessions
    launched outside a project root).
    """
    frame: dict = {}
    if agent:
        frame["agent"] = agent
    project = _detect_project_anchor(cwd)
    if project:
        frame["project"] = project
    return frame


# ═══════════════════════════════════════════════════════════════════
# Process-wide singleton + wiring helper
# ═══════════════════════════════════════════════════════════════════


_GATE_SINGLETON: InjectionGate | None = None


def get_gate() -> InjectionGate:
    """Lazy process-wide InjectionGate.

    One gate per process so the Anthropic client (held on the gate
    instance) isn't rebuilt per retrieval call. Tests inject their
    own by constructing InjectionGate(_client=...) directly.
    """
    global _GATE_SINGLETON
    if _GATE_SINGLETON is None:
        _GATE_SINGLETON = InjectionGate()
    return _GATE_SINGLETON


def _gate_disabled() -> bool:
    """Opt-out via env: MEMPALACE_GATE_DISABLED=1 turns the gate off
    entirely (apply_gate becomes a pass-through). Used in tests and
    by operators who want to roll back without a code change."""
    return os.environ.get("MEMPALACE_GATE_DISABLED", "").strip() in (
        "1",
        "true",
        "True",
        "yes",
    )


def apply_gate(
    *,
    memories: list[dict],
    combined_meta: dict | None,
    primary_context: dict,
    context_id: str,
    kg,
    agent: str | None,
    parent_intent: dict | None = None,
    gate: InjectionGate | None = None,
    default_channel: str = "A",
) -> tuple[list[dict], dict | None]:
    """Run the injection gate on a built retrieval list.

    Shared by declare_intent, declare_operation, and kg_search so the
    wiring is one call per site. Input ``memories`` is the list the
    caller was about to return (dicts with ``id`` and ``text``;
    optional ``hybrid_score`` / ``score``). Output is the filtered
    list plus an optional ``gate_status`` dict to surface in the
    response ONLY when the state is not ``ok``/``skipped_empty``/
    ``skipped_small_k`` (callers check the second return for None).

    Dropped items are persisted via record_feedback with
    rater_kind='gate_llm' — entity drops become rated_irrelevant
    edges, triple drops land in triple_context_feedback, no phantom
    entities.

    Fail-open: any exception is caught and the original ``memories``
    list passes through unchanged; callers never see the gate kill
    their payload on a bug in this module.
    """
    if _gate_disabled() or not memories:
        return memories, None
    try:
        gate = gate or get_gate()
    except Exception as exc:  # pragma: no cover — defensive
        log.info("apply_gate: get_gate failed: %s", exc)
        return memories, None

    items: list[GateItem] = []
    for i, m in enumerate(memories):
        mid = m.get("id")
        if not mid:
            continue
        meta_entry = (combined_meta or {}).get(mid, {}) or {}
        source = meta_entry.get("source") or ("triple" if str(mid).startswith("t_") else "memory")
        if source not in ("memory", "entity", "triple"):
            source = "memory"
        extras = {}
        raw_meta = meta_entry.get("meta") or {}
        if isinstance(raw_meta, dict):
            extras = dict(raw_meta)
        doc = meta_entry.get("doc") or m.get("text") or ""
        score = m.get("hybrid_score")
        if score is None:
            score = m.get("score") or meta_entry.get("similarity") or 0.0
        items.append(
            GateItem(
                id=str(mid),
                source=source,  # type: ignore[arg-type]
                text=str(doc or ""),
                channel=default_channel,
                rank=i + 1,
                score=float(score or 0.0),
                extra=extras,
            )
        )

    frame = build_session_frame(agent=agent, cwd=os.getcwd())
    try:
        result = gate.filter(
            primary_context=primary_context,
            items=items,
            parent_intent=parent_intent,
            session_frame=frame,
        )
    except Exception as exc:  # pragma: no cover — defensive
        log.info("apply_gate: filter failed: %s", exc)
        return memories, None

    if result.dropped and context_id:
        try:
            persist_drops(kg, context_id=context_id, dropped=result.dropped)
        except Exception as exc:  # pragma: no cover — best-effort
            log.info("apply_gate: persist_drops failed: %s", exc)

    # Persist quality flags for the memory_gardener background process.
    # Scoped to the active context so re-observing the same issue in
    # the same context bumps the existing row instead of duplicating.
    # Best-effort: a flag-write failure must not prevent returning
    # kept items.
    if result.flags and context_id:
        try:
            enriched = [{**f, "context_id": context_id} for f in result.flags]
            kg.record_memory_flags(enriched, rater_model=getattr(gate, "model", "") or "")
        except Exception as exc:  # pragma: no cover — best-effort
            log.info("apply_gate: record_memory_flags failed: %s", exc)

    kept_ids = {it.id for it in result.kept}
    filtered = [m for m in memories if str(m.get("id")) in kept_ids]

    state = result.gate_status.get("state")
    # Only surface gate_status on non-happy-path outcomes. The default
    # path should add zero extra tokens to the response. "skipped_*"
    # states are all happy-path: empty input, K below threshold, or
    # no API key configured (operator-chosen opt-out). Only "degraded"
    # (actual runtime failure) reaches the main agent — that's the
    # signal worth surfacing.
    if state in (
        "ok",
        "skipped_empty",
        "skipped_small_k",
        "skipped_no_client",
    ):
        return filtered, None
    return filtered, result.gate_status
