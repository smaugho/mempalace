#!/usr/bin/env python3
"""
mempalace/intent.py -- Intent declaration, active-intent tracking, and finalization.

Extracted from mcp_server.py. Uses a module-reference pattern to access
mcp_server globals without circular imports.
"""

from __future__ import annotations


import concurrent.futures
import hashlib
import json
import math
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from .knowledge_graph import normalize_entity_name

# Module reference (set by init())
_mcp = None


# ═══════════════════════════════════════════════════════════════════
# v3.7.4 Slice 3 (Adrian directive 2026-05-16, Option 3 architecture):
# Background state_judge subsystem.
#
# After Slice 1 (v3.7.1) shrank the foreground apply_gate to ~2-3s,
# run_state_judge became the dominant foreground cost (~6-9s with the
# full padded prompt). Moving judge to a daemon worker thread cuts the
# foreground critical path to ~2-3s matching the gate.
#
# MCP push-notifications cannot inject content into the agent's
# context (spec gap), so the bg judge's state_changes_detected are
# buffered per-session and surfaced on the NEXT declare_operation
# response as ``state_updates_since_last_op[]``. One-op lag is
# acceptable because the judge's auto-apply path
# (record_state_revision with agent='state_judge') still lands the
# writes in the KG; the surfacing is purely for agent visibility.
#
# Per-session buffer is dict[sid -> list[{changes, report}]] under a
# Lock so foreground reads (drain) and bg writes (append) don't race.
# Single-worker executor bounds Anthropic rate-limit pressure.
# Disable via MEMPALACE_BG_STATE_JUDGE=0 to keep judge in the
# foreground parallel block (back-compat).
# ═══════════════════════════════════════════════════════════════════
_BG_STATE_JUDGE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="mempalace-bg-state-judge",
)
_PENDING_STATE_UPDATES_BY_SID: dict = {}
_PENDING_STATE_UPDATES_LOCK = threading.Lock()


def _append_pending_state_updates(sid: str, changes: list, report) -> None:
    """Append a bg state_judge result to the per-session buffer.

    No-op when ``changes`` is empty (silence means no update to
    surface; the buffer would otherwise accumulate empty entries on
    every quiet op). Thread-safe via _PENDING_STATE_UPDATES_LOCK.
    """
    if not changes:
        return
    with _PENDING_STATE_UPDATES_LOCK:
        bucket = _PENDING_STATE_UPDATES_BY_SID.setdefault(sid or "", [])
        bucket.append({"changes": changes, "report": report})


def _drain_pending_state_updates(sid: str) -> list:
    """Pop all buffered bg state_judge results for this session.

    Returns a flat list of {changes, report} entries -- one per bg
    pass since the last drain. Empty list when there's nothing
    pending. Thread-safe via _PENDING_STATE_UPDATES_LOCK.
    """
    with _PENDING_STATE_UPDATES_LOCK:
        return _PENDING_STATE_UPDATES_BY_SID.pop(sid or "", [])


def _reset_pending_state_updates() -> None:
    """Test-only: clear the entire per-sid buffer. Used by fixtures
    to isolate bg state-judge tests from cross-session leakage."""
    with _PENDING_STATE_UPDATES_LOCK:
        _PENDING_STATE_UPDATES_BY_SID.clear()


def _bg_state_judge_enabled() -> bool:
    """Read MEMPALACE_BG_STATE_JUDGE at call time so tests can flip
    the flag with monkeypatch.setenv without reload."""
    return os.environ.get("MEMPALACE_BG_STATE_JUDGE", "1").strip() != "0"


def _apply_judge_changes_to_state(
    changes: list,
    op_context_id: str = "",
    delta_covered: set | None = None,
    session_id: str | None = None,
) -> list:
    """Persist state_judge findings as actual revisions.

    v3.7.23 fix (FINDING #10, Adrian directive 2026-05-18): pre-v3.7.4
    the foreground auto-apply at tool_declare_operation:4533 was the
    only path that wrote judge findings via record_state_revision. The
    v3.7.4 Slice 3 bg-state_judge refactor moved the judge call off
    the critical path but DID NOT move the auto-apply with it -- the
    bg worker only appended findings to the per-sid pending buffer
    (drained on next op as state_updates_since_last_op). Result: with
    bg-judge enabled (the default), the foreground judge future was
    None, _judge_changes_perop stayed empty, the 4533 block had
    nothing to apply, and the bg findings went to the buffer for
    agent visibility but NEVER reached mempalace_state_revisions. Two
    days of judge calls (180+ entries in state_judge_log.jsonl)
    produced zero new ga_agent state revisions. Active_intent showed
    a stale current_focus that never updated despite the judge
    proposing patches on every op.

    This helper centralizes the apply logic so both call sites
    (foreground 4533 + bg _run_bg_judge) can use it. Mirrors the
    v3.7.20 conflict_resolver_auto pattern: bg worker does Haiku +
    apply + log.

    Returns the same ``changes`` list with per-entry
    ``applied`` / ``rev_id`` / ``error`` / ``skip_reason`` annotated
    in-place so callers can surface them to the agent unchanged.

    Args:
        changes: list of judge change dicts from
            injection_gate.run_state_judge. Each dict carries
            entity_id, schema_id, reason, optional patch (RFC 6902 op
            list). Flag-only entries (no patch) are skipped silently.
        op_context_id: the active context id at apply time -- becomes
            the op_context_id column on the new revision row (JTMS
            justification). Empty string for bg-only applies that
            have no enclosing op context.
        delta_covered: set of entity ids the agent ALREADY covered
            this op via explicit state_deltas. Entities in this set
            are skipped to avoid stomping the agent's writes; the
            change entry gets ``applied=False, skip_reason='agent_covered'``.
            Pass None for bg-only applies (no agent context).
        session_id: the session id to stamp on the new revision row.
            When None the row's session_id stays NULL (visible to all
            sessions under the scope-aware read; correct for global
            schemas and acceptable for session schemas via the
            ``session_id = ? OR session_id IS NULL`` SQL refilter).

    Best-effort: per-entry exceptions land on the change dict's
    ``error`` field; helper never raises.
    """
    if not changes:
        return changes
    try:
        import jsonpatch as _jp
    except Exception:
        _jp = None
    delta_set = delta_covered or set()
    for _change in changes:
        if not isinstance(_change, dict):
            continue
        _eid = (_change.get("entity_id") or "").strip()
        _sid = (_change.get("schema_id") or "").strip()
        _patch = _change.get("patch")
        if not _eid or not _sid or not isinstance(_patch, list) or not _patch:
            # Flag-only change (no fix proposed) -- agent sees the
            # 'reason' but nothing is auto-applied.
            continue
        if _eid in delta_set:
            _change["applied"] = False
            _change["skip_reason"] = "agent_covered"
            continue
        if _jp is None:
            _change["applied"] = False
            _change["error"] = "jsonpatch_unavailable"
            continue
        try:
            _current = _mcp._STATE.kg.latest_state_for_entity(_eid, session_id=session_id) or {}
            _new = _jp.apply_patch(_current, _patch)
            _rev_id = _mcp._STATE.kg.record_state_revision(
                _eid,
                _sid,
                _new,
                op_context_id=op_context_id,
                agent="state_judge",
                session_id=session_id,
            )
            _change["applied"] = True
            _change["rev_id"] = _rev_id
        except Exception as _exc:
            _change["applied"] = False
            _change["error"] = f"{type(_exc).__name__}: {_exc}"
    return changes


# ── Cop-out reason detection -- semantic-similarity side of the hybrid gate ──
# Regex catches the obvious literal forms ("don't know", "N/A", "never used")
# but agents can evade with rephrasings ("lack of information to evaluate",
# "cannot determine relevance", "unclear to me whether this was useful") that
# are semantically identical cop-outs. Second-pass embedding check catches
# those by measuring cosine similarity against a small set of exemplars that
# span the cop-out intent space. Threshold 0.70 tuned conservatively -- better
# to let a borderline reason through than to reject a genuine short-but-real
# rating.
#
# Exemplars seeded from reasons I (the agent) caught myself writing during
# the 2026-04-24 session plus typical paraphrases. Extend as patterns are
# discovered post-deploy; cache invalidates at process restart.
_COPOUT_EXEMPLARS = [
    "I don't know what this memory contains, cannot evaluate",
    "I never used this memory, did not engage with it",
    "Not relevant, skipped without reading the content",
    "Aborted before running any verification, no rating available",
    "Placeholder rating, unclear whether the memory was useful",
    "Cannot determine relevance, no idea what this is about",
    "Did not fetch the memory content, N/A for this intent",
    "Unable to assess, insufficient context to rate fairly",
    "Skipped evaluation, moving past without reading",
]
_COPOUT_EMB_CACHE: list | None = None  # lazy-populated on first _semantic_copout_check call
_COPOUT_SIM_THRESHOLD = 0.70


# Regex side of the hybrid gate. Promoted to module level (2026-04-24)
# so tests can import and assert pattern behaviour directly without
# calling the full finalize pipeline. Patterns are narrow and literal
# by design -- only cheap-and-obvious cop-outs here; semantic similarity
# handles paraphrased evasions. Adding a pattern that false-positives
# on compound nouns (e.g. \bskip(ped)?\b hitting "skip-list") is a
# bug -- prefer standalone-verb forms like \bskipped\b.
_LOW_QUALITY_REASON_PATTERNS = [
    r"\bdon'?t know\b",
    r"\bnot used\b",
    r"\bnever used\b",
    r"\bdidn'?t use\b",
    r"\bnot sure\b",
    r"\bn\.?\s*/?\s*a\b",
    r"\bno idea\b",
    r"\bnot applicable\b",
    r"\baborted\b.*\brunning\b",
    r"\bnot rated\b",
    # Narrow to the verb "skipped" only -- standalone "skip" matches
    # legitimate data-structure terms (skip-list, skip-gram, etc.)
    # which ARE valid content references in rating reasons.
    r"\bskipped\b",
    r"^\s*(unclear|unknown|placeholder|tbd|todo)\s*$",
]
_LOW_QUALITY_RE = re.compile("|".join(_LOW_QUALITY_REASON_PATTERNS), re.IGNORECASE)


def _regex_copout_check(reason_text: str) -> bool:
    """Regex fast-path of the hybrid cop-out gate. Returns True when the
    reason matches any literal-cop-out pattern. Tests import this to
    assert pattern coverage without calling the full finalize pipeline.
    """
    if not isinstance(reason_text, str):
        return False
    return bool(_LOW_QUALITY_RE.search(reason_text))


def _semantic_copout_check(reason_text: str) -> tuple[bool, float]:
    """Hybrid-gate second pass: cosine similarity of reason vs cop-out exemplars.

    Returns ``(is_copout, max_similarity)``. Caches exemplar embeddings so the
    per-call cost is one embedder forward (reason only). Fail-open: any
    exception yields ``(False, 0.0)`` so a broken embedder can't block
    finalize -- the regex gate still catches the obvious cases.
    """
    global _COPOUT_EMB_CACHE
    text = (reason_text or "").strip()
    if not text:
        return False, 0.0
    try:
        from mempalace.embedder import get_default_embedder

        efunc = get_default_embedder()
        if efunc is None:
            return False, 0.0
        if _COPOUT_EMB_CACHE is None:
            vecs = efunc(_COPOUT_EXEMPLARS)
            _COPOUT_EMB_CACHE = [list(float(x) for x in v) for v in vecs]
        reason_vec = list(float(x) for x in efunc([text])[0])
        na = math.sqrt(sum(x * x for x in reason_vec))
        if na == 0:
            return False, 0.0
        max_sim = 0.0
        for ex_vec in _COPOUT_EMB_CACHE:
            nb = math.sqrt(sum(x * x for x in ex_vec))
            if nb == 0:
                continue
            dot = sum(x * y for x, y in zip(reason_vec, ex_vec))
            sim = dot / (na * nb)
            if sim > max_sim:
                max_sim = sim
        return max_sim >= _COPOUT_SIM_THRESHOLD, max_sim
    except Exception:
        # Fail-open: embedder down must not block finalize.
        return False, 0.0


# ── Debug-return overlay (on by default; set env var to "0" to suppress) ──
# DEBUG_RETURN_SCORES: attach the fused retrieval score (``hybrid_score``,
#   the post-RRF fused score that ranks the returned memories) to every
#   item in the ``memories`` list of declare_intent / declare_operation /
#   kg_search. Debug-only; callers that serialize the payload should
#   treat this field as optional.
# DEBUG_RETURN_CONTEXT: attach a top-level ``context: {id, queries}``
#   block to declare_intent / declare_operation / kg_search responses so
#   callers can see which context entity minted/reused for the call and
#   the exact queries that seeded retrieval.
# MEMORY_PREVIEW_MAX_CHARS: safety cap applied to every per-memory preview
#   returned in the three tools above. Summary-first records fit easily;
#   legacy records without the summary\n\ncontent split no longer leak the
#   full content into the injection payload.
def _env_flag_on(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


DEBUG_RETURN_SCORES = _env_flag_on("MEMPALACE_DEBUG_RETURN_SCORES", False)
DEBUG_RETURN_CONTEXT = _env_flag_on("MEMPALACE_DEBUG_RETURN_CONTEXT", True)
MEMORY_PREVIEW_MAX_CHARS = 400


def _shorten_preview(text):
    """Summary-first + length cap for a single memory preview.

    Splits on the first blank line so summary-first records (written as
    ``summary\\n\\ncontent``) render only the ≤280-char distilled summary.
    Then caps at ``MEMORY_PREVIEW_MAX_CHARS`` as a safety net for legacy
    records that pre-date summary-first indexing.
    """
    if not isinstance(text, str):
        return text
    if "\n\n" in text:
        text = text.split("\n\n", 1)[0]
    if len(text) > MEMORY_PREVIEW_MAX_CHARS:
        text = text[: MEMORY_PREVIEW_MAX_CHARS - 1].rstrip() + "\u2026"
    return text


# v3.5.9 (Adrian directive 2026-05-16): surface memory CONTENT alongside
# the summary so the agent can read full bodies without a kg_query
# roundtrip. Cap with an explicit "trimmed" marker. Set to 0 via env
# to disable content entirely (token-budget escape hatch).
try:
    MEMORY_CONTENT_MAX_CHARS = int(os.environ.get("MEMPALACE_MEMORY_CONTENT_MAX_CHARS", "2000"))
except (TypeError, ValueError):
    MEMORY_CONTENT_MAX_CHARS = 2000

# v3.7.30 (Adrian directive 2026-05-18): similarity-dedup gate now
# uses MiniLM-L6 cosine on embedded vectors (replacing the v3.7.3
# difflib.SequenceMatcher char-overlap measure). When cosine(summary,
# content) >= MEMORY_CONTENT_DEDUP_THRESHOLD, blank the content and
# mark the entry with content_redundant=True so the agent sees WHY
# content is missing. Set threshold to 0 via env to disable dedup
# entirely (always surface content).
#
# Default 0.85 -- empirically calibrated on MiniLM-L6-v2 (the embedder
# used everywhere in mempalace). Live distribution on real surface
# strings:
#   IDENTICAL   summary == content                            -> cos 1.00
#   PARAPHRASE  "X shipped" vs "X has been shipped"           -> cos 0.89
#   LOOSE       "Adrian is X" vs "Adrian is the X who Y"      -> cos 0.81
#   DISTINCT    summary vs real elaboration                   -> cos 0.34
#   UNRELATED   different topics                              -> cos -0.01
# 0.85 catches IDENTICAL + PARAPHRASE (the "restate" cases the v3.7.3
# gate was designed for) while skipping LOOSE paraphrase that genuinely
# adds new framing or detail. Operators can override via
# MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD env var. The pre-v3.7.30
# difflib default of 0.75 was a different measurement (Ratcliff-
# Obershelp char overlap, not semantic cosine) and is no longer the
# right threshold; this is the source of the FINDING #L false-positive
# cascade Adrian flagged.
try:
    MEMORY_CONTENT_DEDUP_THRESHOLD = float(
        os.environ.get("MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD", "0.85")
    )
except (TypeError, ValueError):
    MEMORY_CONTENT_DEDUP_THRESHOLD = 0.85


def _render_class_path(kg, entity_id, kind):
    """v3.9.6 (Adrian msg_171/172 2026-05-21): render the
    ``(kind) ancestor -> ancestor`` class-chain signature -- the single
    source of truth so every retrieval surface renders it identically.

    - kind in {entity, class}: walk is_a ancestors transitively (BFS,
      deduped across branches, root ``thing`` omitted) and join with
      `` -> ``. Multiple is_a parents are listed in BFS order.
        e.g. version_py is_a file -> ``(entity) file``
             mempalace is_a concept, single_sqlite_file_architecture, ...
                       -> ``(entity) concept -> single_sqlite_file_architecture -> ...``
    - other kinds (record / predicate / literal): just ``(kind)``.

    Bounded (<=12 hops) + indexed (kg.is_a_parents is one indexed query
    per node). Fail-open to ``(kind)`` on any error so a render hiccup
    never breaks retrieval.
    """
    if not kind:
        return ""
    base = f"({kind})"
    if kind not in ("entity", "class") or kg is None or not entity_id:
        return base
    try:
        ancestors = []
        seen = {entity_id}
        frontier = [entity_id]
        hops = 0
        while frontier and hops < 12:
            hops += 1
            nxt = []
            for node in frontier:
                for obj in kg.is_a_parents(node) or []:
                    if not obj or obj == "thing" or obj in seen:
                        continue
                    seen.add(obj)
                    ancestors.append(obj)
                    nxt.append(obj)
            frontier = nxt
        if ancestors:
            return base + " " + " -> ".join(ancestors)
        return base
    except Exception:
        return base


def _project_memory(memory_id, raw_text, extras=None):
    """v3.7.9 canonical memory-projection helper. Builds the standard
    surface dict for a single memory hit:

        {id, summary_text,
         [content],            -- present iff helper returned non-empty content
         [content_trimmed],    -- present iff content was capped at MAX_CHARS
         [content_redundant],  -- present iff content was suppressed as near-
                                  duplicate of the summary (v3.7.3 dedup)
         **extras}             -- merged in last (e.g. hybrid_score, source,
                                  similarity, added_by, content_type, ...)

    Adrian's directive 2026-05-17: BEFORE v3.7.9 the entry-build was
    duplicated inline at 4 sites (declare_intent / declare_user_intents /
    kg_search top / kg_search lean forward) and SILENTLY OMITTED at 2
    more (intent._attach_context_rank + searcher record return), so the
    SAME memory could surface with different shape depending on which
    retrieval path returned it. Centralizing via this single helper
    eliminates the divergence and makes future shape changes (new
    field, new trim semantics) one-line patches.

    Args:
        memory_id: the memory id (str).
        raw_text:  full rendered preview prose (the "summary\\n\\ncontent"
                   shape from kg.render_memory_preview, or any string
                   the caller has on hand). Empty/None is tolerated.
        extras:    optional dict merged onto the output AFTER the canonical
                   keys, so callers can attach per-call fields without
                   forking the helper.

    Returns the entry dict.
    """
    _summary, _content, _trimmed, _redundant = _split_for_surface(raw_text or "")
    entry = {"id": memory_id, "summary_text": _summary}
    if _content:
        entry["content"] = _content
        if _trimmed:
            entry["content_trimmed"] = True
    elif _redundant:
        entry["content_redundant"] = True
    if extras:
        entry.update(extras)
    # v3.7.34 (Adrian msg_138 2026-05-18): surface date_added +
    # last_relevant_at so the agent can see WHEN a memory was filed /
    # last used. Literature (Ebbinghaus 1885; Wickelgren 1974; Park
    # et al. Generative Agents 2023; Wang et al. MemoryBank 2023;
    # LangChain TimeWeightedVectorStoreRetriever) all converge on the
    # principle that retrieval AND reasoning should be time-aware,
    # not just ranking. Pre-v3.7.34 mempalace had Wickelgren-Wixted
    # power-law decay wired into hybrid_score (scoring.py:246) but
    # the agent itself never saw the dates -- so it could not reason
    # "this is six months old, double-check before trusting" or
    # "this fact superseded an older one filed last week." The fields
    # are hoisted from extras (top-level) or extras['metadata']
    # (the vec metadata sub-dict shape searcher.py already produces);
    # both paths fall through silently if absent so legacy callers
    # that don't supply dates just omit the field rather than carrying
    # a null. date_added is the write-time stamp; last_relevant_at is
    # the touch-on-use stamp (reset post-gate by injection_gate; see
    # FINDING #T fix in scoring.two_stage_retrieve for the freshness
    # plumbing).
    _meta = (extras or {}).get("metadata") or {}
    if "date_added" not in entry:
        _date_added = _meta.get("date_added")
        if _date_added:
            entry["date_added"] = _date_added
    if "last_relevant_at" not in entry:
        _last_relevant = _meta.get("last_relevant_at")
        if _last_relevant:
            entry["last_relevant_at"] = _last_relevant
    # v3.7.39 (Adrian msg_c96c8a_143 2026-05-19): trim surfaced dates
    # to minute precision. "we don't need the milliseconds, up to the
    # minutes TBH is enough, not even the seconds, though seconds
    # could be left." Minutes = 16-char prefix of any ISO 8601 form
    # (YYYY-MM-DDTHH:MM or YYYY-MM-DD HH:MM -- both space- and T-
    # separated work since the position of the colon-minute boundary
    # is identical). Saves ~10 chars per date * 2 dates per memory
    # on retrieval-heavy responses. Trimming at SURFACE only --
    # underlying SQL + vec storage keep full microsecond precision
    # for accurate decay scoring; the agent simply doesn't see it.
    for _key in ("date_added", "last_relevant_at"):
        _val = entry.get(_key)
        if isinstance(_val, str) and len(_val) >= 16:
            entry[_key] = _val[:16]
    # v3.9.6 (Adrian msg_c96c8a_171/172 2026-05-21): surface the
    # rendered `class_path` signature -- "(kind) ancestor -> ancestor",
    # is_a chain with root `thing` omitted -- so every result tells the
    # agent WHAT it is at a glance, identically across all surfaces.
    # Supersedes v3.9.5's bare `kind` field (the kind is now embedded in
    # the parens, so a separate `kind` key would be redundant verbosity).
    # `kind` comes from caller extras (entity branch sets it) or the vec
    # metadata; the is_a walk reads the kg singleton (one indexed query
    # per hop, bounded). Fail-open: no kg / error -> just "(kind)".
    _kind = entry.get("kind") or _meta.get("kind")
    if _kind:
        _kg = None
        try:
            from mempalace.mcp_server import _STATE as _S

            _kg = _S.kg
        except Exception:
            _kg = None
        _sig = _render_class_path(_kg, memory_id, _kind)
        if _sig:
            entry["class_path"] = _sig
    # Drop any bare `kind` carried in via extras -- class_path replaces it.
    entry.pop("kind", None)
    # v3.7.37 verbosity fix (Adrian msg_c96c8a_141 2026-05-19): strip
    # the raw vec metadata dict from the agent-visible surface. The
    # v3.7.34 plumbing started passing extras['metadata'] = vec_meta
    # through this helper so the date_added / last_relevant_at hoist
    # above could read from it -- but extras.update(entry) at line 428
    # also dumped the WHOLE meta blob (session_id, intent_id,
    # content_type, view_index, added_by, etc.) into the agent's
    # response. Adrian reported the entries were "too verbose."
    # Once the date fields are hoisted, nothing downstream reads
    # entry['metadata']; the only consumer of the metadata sub-dict
    # is this helper itself (line 448 reads from extras directly,
    # never from entry). Pop unconditionally so the surface stays
    # lean. searcher.py path (which sets extras['metadata'] = meta
    # for "re-ranking" purposes per its 2026-05 comment) is covered
    # by this same pop -- that re-ranking path also never reads
    # entry['metadata'] post-projection, so dropping it is safe.
    entry.pop("metadata", None)
    return entry


def _split_for_surface(text):
    """Split a raw memory preview into (summary, content, content_trimmed, content_redundant).

    Used at the 4 retrieval projection sites (declare_intent / declare_
    operation memories list + declare_user_intents memories list +
    kg_search top + kg_search lean projection) to surface BOTH the
    summary AND a length-capped slice of the content body in one
    payload. Pre-v3.5.9 only the summary surfaced; the agent had to
    fire a follow-up kg_query for every retrieved item to read the
    content, which doubled call counts on every retrieval-heavy path.

    v3.7.3 adds a similarity-dedup gate: when the difflib
    SequenceMatcher ratio between summary and content is >=
    MEMORY_CONTENT_DEDUP_THRESHOLD (default 0.75), the content is
    blanked and content_redundant=True is set so the caller can
    surface a 'content suppressed -- near-duplicate of summary'
    marker to the agent without wasting tokens on the restate.

    Returns:
      summary  -- shortened per _shorten_preview.
      content  -- content side of the split, capped at
                  MEMORY_CONTENT_MAX_CHARS; "" when the record has no
                  separable content body, when env-flag=0, OR when the
                  similarity-dedup gate fired.
      content_trimmed -- True when content was actually capped; the
                  surfaced string already carries a ...[trimmed at N
                  chars; kg_query for full body] marker.
      content_redundant -- True when content was suppressed by the
                  similarity-dedup gate (>=
                  MEMORY_CONTENT_DEDUP_THRESHOLD similarity to
                  summary). When True, content is "" -- callers
                  surface a marker so the agent knows the suppression
                  was intentional, not a missing-content bug.
    """
    if not isinstance(text, str) or MEMORY_CONTENT_MAX_CHARS <= 0:
        return _shorten_preview(text), "", False, False
    if "\n\n" in text:
        summary_part, content_part = text.split("\n\n", 1)
    else:
        # FINDING #L (v3.7.25 2026-05-18, Adrian): when the rendered
        # preview has no "summary\n\ncontent" split -- which is the
        # case for EVERY entity-kind row whose render comes from
        # serialize_summary_for_embedding (intent_py, layers_py,
        # phantom_entity_pattern, every described_by-only record,
        # etc.) -- the content body is genuinely empty (the summary
        # IS the full information). Previously this branch aliased
        # content_part = text, so the SequenceMatcher dedup below
        # compared the text against itself, got ratio 1.0, and
        # flagged content_redundant=True on EVERY such hit. The
        # symptom: Adrian saw "content_redundant: true" on virtually
        # every retrieved memory across every session. Fix: when
        # there's no separable content body, surface the summary
        # and report content="" + content_redundant=False (there is
        # no suppressed body to flag).
        return _shorten_preview(text), "", False, False
    summary_out = _shorten_preview(summary_part)
    content_out = (content_part or "").strip()

    # v3.7.30 (Adrian directive 2026-05-18): cosine on MiniLM-L6
    # embeddings replaces the v3.7.3 difflib.SequenceMatcher gate. The
    # character-overlap measure was wrong for the job -- summary and
    # content with the same anchor phrases ("v3.7.X", "Adrian", date,
    # entity names) score above 0.75 on char-overlap even when the
    # content elaborates with genuinely new information. The cosine
    # measure compares the SEMANTIC content of the two strings:
    # paraphrase/restate scores high, elaboration scores lower.
    #
    # Threshold convention: cosine 0.92 is the default (analogous to
    # T_REUSE_WHAT for identity collisions); the rendered-prose form
    # of summary and a near-verbatim restate of summary in content
    # land at ~0.95-0.99 cosine, while genuine elaboration drops to
    # 0.6-0.85 typical. Operators can override via
    # MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD (interpreted as cosine
    # threshold post-v3.7.30; same env var, new measurement).
    #
    # On-demand cost: two MiniLM-L6 embedder calls per surfaced hit
    # (summary + content, each <=400 / <=2000 chars => well under the
    # 256-token cap). Typical embed ~5ms each on ONNX runtime; ~10ms
    # total per hit. With ~10 hits per surfacing call this adds ~100ms
    # vs the 12-16s median surfacing latency: marginal. NO FALLBACK to
    # difflib (Adrian: "no fallback needed, the vectors comparison
    # should be used"); embedder failure leaves redundant=False so
    # content surfaces (fail-open on the gate, not fail-closed --
    # better to show duplicate text than silently suppress real body
    # content).
    redundant = False
    if content_out and summary_out and MEMORY_CONTENT_DEDUP_THRESHOLD > 0.0:
        try:
            from mempalace.embedder import get_default_embedder

            _embedder = get_default_embedder()
            if _embedder is not None:
                vecs = _embedder([summary_out, content_out])
                if vecs and len(vecs) == 2:
                    import math as _math

                    a = vecs[0]
                    b = vecs[1]
                    dot = sum(x * y for x, y in zip(a, b))
                    na = _math.sqrt(sum(x * x for x in a))
                    nb = _math.sqrt(sum(y * y for y in b))
                    cos = dot / (na * nb) if (na > 0 and nb > 0) else 0.0
                    if cos >= MEMORY_CONTENT_DEDUP_THRESHOLD:
                        content_out = ""
                        redundant = True
        except Exception:
            # Embedder hiccup must not kill the surface path; fail-open
            # so the agent at least sees the content rather than losing
            # the body to a silent embedder error.
            pass

    trimmed = False
    if content_out and len(content_out) > MEMORY_CONTENT_MAX_CHARS:
        content_out = (
            content_out[: MEMORY_CONTENT_MAX_CHARS - 1].rstrip()
            + f" \u2026[trimmed at {MEMORY_CONTENT_MAX_CHARS} chars; "
            + "kg_query for full body]"
        )
        trimmed = True
    return summary_out, content_out, trimmed, redundant


# \u2500\u2500 Summary co-render helpers (2026-04-28) \u2500\u2500
# Used by finalize_intent's error renderers (missing_injected / missing_accessed)
# so the model sees what each missing reference is ABOUT alongside the bare id.
# Without this, callers had to do a follow-up kg_query per missing id just to
# rate it. The principle: "ids are pointers, summaries are meaning - never
# surface a pointer without its meaning, except in structured fields where the
# caller can look it up themselves." Renderer co-render is the cheapest way to
# honour that without bloating storage.
_SUMMARY_PREVIEW_MAX_CHARS = 80


def _short_summary_for_id(memory_id: str, max_len: int = _SUMMARY_PREVIEW_MAX_CHARS):
    """Resolve a short human-readable summary for a memory id.

    Reads ``summary.what`` directly -- the structured summary contract
    (Adrian design lock 2026-04-25) guarantees ``what`` is present and
    validated >=5 chars on every entity write. Length-capped with
    ellipsis.

    Adrian directive 2026-05-06 fix: the old implementation read
    ``content`` and split the first sentence, which returned None for
    kind=context entities (contexts are identity-only by their
    queries+keywords fingerprint -- ``content`` is often empty even
    though ``summary.what`` is fully populated). The missing_state_deltas
    response then surfaced ``what: null`` despite the entity having a
    real summary. ``summary.what`` is the canonical headline; reading
    it directly bypasses the content-emptiness gap.

    Falls back to the legacy first-sentence-of-content path for
    extreme edge cases where summary is missing (pre-contract entities
    or write-side bug). Returns None only when both paths come up
    empty.
    """
    if not memory_id or _mcp is None:
        return None
    try:
        details = _mcp._fetch_entity_details(memory_id)
    except Exception:
        return None
    if not details:
        return None
    # Primary path: structured summary.what (Adrian design lock 2026-04-25).
    summary = details.get("summary")
    if isinstance(summary, dict):
        what = summary.get("what")
        if isinstance(what, str) and what.strip():
            what = what.strip()
            if len(what) > max_len:
                what = what[: max_len - 1].rstrip() + "\u2026"
            return what
    # Fallback path: first sentence of content. Defense for pre-contract
    # entities or any future write that bypasses the validator.
    content = details.get("content") or ""
    if not content:
        return None
    first = content.split(". ", 1)[0]
    if len(first) > max_len:
        first = first[: max_len - 1].rstrip() + "\u2026"
    return first


def _enrich_ids_with_summaries(ids):
    """Map a list of memory_ids -> list of {"id": <id>, "what": <summary?>}.

    Unknown ids get ``what: None`` so the caller can still tell the id
    is missing AND that no summary was resolvable. Used by finalize_intent
    error rendering at the missing_injected / missing_accessed sites.
    """
    return [{"id": mid, "what": _short_summary_for_id(mid)} for mid in ids]


def init(mcp_module):
    """Wire this module to mcp_server so we can access its globals/functions."""
    global _mcp
    _mcp = mcp_module


# ==================== INTENT DECLARATION ====================
# Note: _STATE (ServerState instance) and _INTENT_STATE_DIR live in mcp_server.py.
# We access active-intent state exclusively via _mcp._STATE.active_intent and the
# hook-state directory via _mcp._INTENT_STATE_DIR.


def _intent_state_path() -> Optional[Path]:
    """Session-scoped active-intent state file path, or None if no sid.

    Returns None when ``_STATE.session_id`` is empty. The caller MUST
    treat None as "no persist / no read" rather than substituting a
    shared default filename. A shared ``active_intent_default.json``
    was the cross-agent contamination vector behind the 2026-04-19
    deadlock -- it collected every agent's pending state into one file,
    so one agent's resolve could be reloaded as another agent's block.
    """
    sid = _mcp._STATE.session_id
    if not sid:
        return None
    return _mcp._INTENT_STATE_DIR / f"active_intent_{sid}.json"


def _build_intent_hierarchy(context: dict = None) -> list:
    """Build a list of all intent types with their tools and is_a parent.

    Walks the KG to find all entities that is_a intent_type (directly or
    transitively). Returns a list of dicts with id, parent, tools,
    importance, added_by -- plus context_rank / context_score when a
    Context is supplied.

    Context-ranked hierarchy. When the caller passes the active
    intent's Context (queries + keywords), we re-use the SAME 3-channel
    pipeline the rest of the palace uses (scoring.multi_channel_search
    against the entity collection with kind='class') to rank intent
    types by semantic similarity to what the agent is actually doing.
    The ranking is baked into the hierarchy entries and persisted to the
    session state file, so the PreToolUse hook -- which must stay
    dep-free (no ChromaDB, no Torch) -- reads a pre-sorted list with
    zero retrieval work at hook time.
    """

    hierarchy = []
    # Find all entities in the KG that might be intent types
    ecol = _mcp._get_entity_collection(create=False)
    if not ecol:
        return hierarchy

    try:
        all_entities = ecol.get(include=["metadatas"])
        if not all_entities or not all_entities["ids"]:
            return hierarchy
    except Exception:
        return hierarchy

    # Post-P5.2 the entity collection stores multi-view records keyed by
    # '{entity_id}__v{N}' -- dedupe by logical entity_id from metadata so
    # we only walk each class once.
    seen_logical = set()
    for i, raw_id in enumerate(all_entities["ids"]):
        meta = all_entities["metadatas"][i] or {}
        if meta.get("kind") != "class":
            continue
        eid = meta.get("entity_id") or raw_id
        if eid in seen_logical:
            continue
        seen_logical.add(eid)

        # Check if this class is-a intent_type (direct or via parent)
        edges = _mcp._STATE.kg.query_entity(eid, direction="outgoing")
        parent_id = None
        for e in edges:
            if e["predicate"] == "is_a" and e["current"]:
                obj = normalize_entity_name(e["object"])
                if obj == "intent_type":
                    parent_id = "intent-type"
                    break
                # Check if parent is itself an intent type
                parent_edges = _mcp._STATE.kg.query_entity(obj, direction="outgoing")
                for pe in parent_edges:
                    if pe["predicate"] == "is_a" and pe["current"]:
                        if normalize_entity_name(pe["object"]) == "intent_type":
                            parent_id = obj
                            break
                if parent_id:
                    break

        if not parent_id:
            continue

        # Get tool permissions via hierarchy resolution
        _, tools = _resolve_intent_profile(eid)
        tool_names = sorted(set(t["tool"] for t in tools)) if tools else []

        importance = meta.get("importance", 3)
        added_by = meta.get("added_by", "")
        hierarchy.append(
            {
                "id": eid,
                "parent": parent_id,
                "tools": tool_names,
                "importance": importance,
                "added_by": added_by,
            }
        )

    # Optional Context-based rank. Uses the same 3-channel
    # pipeline as kg_search / declare_intent memory injection.
    if context:
        _attach_context_rank(hierarchy, context, ecol)

    # Sort: context_rank first (when present; None last), then importance
    # desc, then top-level before children, finally by id for stability.
    hierarchy.sort(
        key=lambda x: (
            x.get("context_rank") if x.get("context_rank") is not None else 10**6,
            -x.get("importance", 3),
            0 if x["parent"] == "intent-type" else 1,
            x["id"],
        )
    )
    return hierarchy


def _attach_context_rank(hierarchy: list, context: dict, ecol) -> None:
    """Attach context_rank + context_score to each hierarchy entry in-place.

    Reuses scoring.multi_channel_search against the entity collection
    filtered to kind='class'. Maps physical Chroma ids (post-P5.2
    '{eid}__v{N}') back to logical entity_ids via metadata.entity_id
    and keeps the max RRF score per logical id.
    """
    queries = context.get("queries") or []
    keywords = context.get("keywords") or []
    if not queries:
        return
    try:
        from . import scoring as _scoring
        from .vector_store import RECORDS_COLLECTION, get_vector_store

        _vs = get_vector_store(_mcp._STATE.config.palace_path)
        pipe = _scoring.multi_channel_search(
            _vs,
            RECORDS_COLLECTION,
            list(queries),
            keywords=list(keywords),
            kg=_mcp._STATE.kg,
            kind="class",
            fetch_limit_per_view=50,
            include_graph=False,
            caller="intent_hierarchy_rank",
        )
    except Exception:
        return
    rrf_scores = pipe.get("rrf_scores") or {}
    seen_meta = pipe.get("seen_meta") or {}

    logical_scores: dict = {}
    for phys_id, score in rrf_scores.items():
        entry = seen_meta.get(phys_id) or {}
        meta = entry.get("meta") or {}
        logical_id = meta.get("entity_id") or phys_id
        if score > logical_scores.get(logical_id, float("-inf")):
            logical_scores[logical_id] = score

    ranked_ids = sorted(logical_scores.keys(), key=lambda k: -logical_scores[k])
    id_to_rank = {eid: i for i, eid in enumerate(ranked_ids)}

    for entry in hierarchy:
        if entry["id"] in id_to_rank:
            entry["context_rank"] = id_to_rank[entry["id"]]
            entry["context_score"] = round(float(logical_scores[entry["id"]]), 6)


def _build_intent_hierarchy_safe(context: dict = None) -> list:
    """Safe wrapper -- never crashes, returns [] on any error."""
    try:
        return _build_intent_hierarchy(context)
    except Exception:
        return []


def _sync_from_disk():
    """Reload active intent state from disk.

    Two cases:
      - Normal sync: in-memory intent matches disk intent \u2014 merge back
        ``used`` and ``budget`` because the PreToolUse hook may have
        bumped them out-of-process.
      - Cold hydration: in-memory state is empty but disk has a valid
        intent. Restore the WHOLE intent record plus pending conflicts.
        Handles MCP-server restart, plugin reinstall,
        or any path that clears ``_STATE`` while an on-disk session is
        still active. Before this hydration path, a restart mid-session
        would leave ``finalize_intent`` returning "No active intent" even
        though the disk file still carried the intent \u2014 the wedge that
        bit the 2026-04-20 wrap_up_session cycle.

    Any read/parse error is non-fatal; the caller falls back to "no
    intent" which is the correct loud-by-absence behavior.
    """
    try:
        state_file = _intent_state_path()
        if state_file is None or not state_file.is_file():
            return
        # JSONDecodeError tolerance 2026-04-30: a partial-write race between
        # the MCP server and the PreToolUse hook subprocess can leave the
        # state file with two concatenated JSON documents (the second
        # write appended instead of replacing). Surfaces as
        # "Extra data: line N column M (char K)" and the surrounding
        # try/except silently swallows it -- leaving the agent without
        # intent state. Recover by parsing just the valid prefix via
        # raw_decode and atomically rewriting so the corruption doesn't
        # recur on the next read.
        raw_text = state_file.read_text(encoding="utf-8")
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as _je:
            if "Extra data" not in str(_je):
                raise
            data, _end_idx = json.JSONDecoder().raw_decode(raw_text.lstrip())
            try:
                tmp = state_file.with_suffix(state_file.suffix + ".tmp")
                tmp.write_text(json.dumps(data), encoding="utf-8")
                os.replace(str(tmp), str(state_file))
            except Exception:
                pass
        if not data.get("intent_id"):
            # Disk has only pending state (no intent) \u2014 restore pending
            # conflicts so the agent resolves them before the next declare.
            pending_c = data.get("pending_conflicts") or []
            if pending_c and not _mcp._STATE.pending_conflicts:
                _mcp._STATE.pending_conflicts = pending_c
            return

        if _mcp._STATE.active_intent:
            # Same-intent sync path \u2014 just refresh used/budget + cues.
            if data["intent_id"] == _mcp._STATE.active_intent["intent_id"]:
                _mcp._STATE.active_intent["used"] = data.get("used", {})
                _mcp._STATE.active_intent["budget"] = data.get("budget", {})
                # pending_operation_cues may have been mutated by the hook
                # (entries consumed / TTL-expired) between our last write
                # and this sync; mirror disk truth as the single source.
                _mcp._STATE.active_intent["pending_operation_cues"] = (
                    data.get("pending_operation_cues") or []
                )
                # 2026-04-29: refresh op_args_by_ctx_tool so the lookup at
                # finalize promotion sees args declared in any earlier turn
                # of the same intent. Disk is authoritative since the hook
                # subprocess may have written between our last sync.
                _mcp._STATE.active_intent["op_args_by_ctx_tool"] = dict(
                    data.get("op_args_by_ctx_tool") or {}
                )
            return

        # Cold-hydration path: memory empty, disk has a live intent. Rebuild
        # the full active_intent dict so the next finalize can find it.
        _mcp._STATE.active_intent = {
            "intent_id": data["intent_id"],
            "intent_type": data.get("intent_type", ""),
            "slots": data.get("slots", {}),
            "effective_permissions": data.get("effective_permissions", []),
            "content": data.get("content", ""),
            "agent": data.get("agent", ""),
            "injected_memory_ids": set(data.get("injected_memory_ids", []) or []),
            "injected_by_context": dict(data.get("injected_by_context", {}) or {}),
            "accessed_memory_ids": set(data.get("accessed_memory_ids", []) or []),
            "budget": data.get("budget", {}),
            "used": data.get("used", {}),
            "intent_hierarchy": data.get("intent_hierarchy", []),
            "active_context_id": data.get("active_context_id", "") or "",
            "contexts_touched": list(data.get("contexts_touched") or []),
            "contexts_touched_detail": list(data.get("contexts_touched_detail") or []),
            # 2026-04-29: cold-hydration must also restore op_args_by_ctx_tool
            # so finalize promotion can recover args_summary for ops declared
            # before a server restart. Without this, every cold-restored
            # intent loses its op-args store and finalizes with empty args.
            "op_args_by_ctx_tool": dict(data.get("op_args_by_ctx_tool") or {}),
        }
        # Preserve pending_operation_cues across MCP restart so agents
        # who declared operations just before the restart don't lose
        # their cues when the server re-hydrates from disk.
        _mcp._STATE.active_intent["pending_operation_cues"] = (
            data.get("pending_operation_cues") or []
        )
        pending_c = data.get("pending_conflicts") or []
        if pending_c and not _mcp._STATE.pending_conflicts:
            _mcp._STATE.pending_conflicts = pending_c
    except Exception as _e:
        # NEVER silent: a failed sync means the in-memory state diverges
        # from disk (stale active_intent, missed hook-updated budget, etc).
        # Record so the next SessionStart surfaces the sync failure.
        try:
            from . import hooks_cli as _hc

            _hc._record_hook_error("_sync_from_disk", _e)
        except Exception:
            pass


def _persist_active_intent():
    """Write the session-scoped state file for the PreToolUse hook.

    Contract:
      - An active_intent without pending state → write intent block, no pending keys.
      - No active_intent but pending conflicts → write a "no-intent"
        state file with just the pending conflicts list. The PreToolUse
        hook does not gate tools on this case (no intent = no permissions),
        but declare_intent reads this pending list on its next call so
        conflicts are never lost just because the intent was finalized
        before the agent resolved them.
      - No active_intent AND no pending state → unlink the file.
    """
    state_file = _intent_state_path()
    if state_file is None:
        # No session_id → no per-agent state file → refuse to persist.
        # Writing to active_intent_default.json would cross-contaminate
        # every agent sharing this MCP server.
        try:
            log_path = _mcp._INTENT_STATE_DIR / "hook.log"
            _mcp._INTENT_STATE_DIR.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"PERSIST_SKIP: _STATE.session_id is empty; refusing to "
                    f"write active_intent_default.json (cross-agent risk). "
                    f"active_intent={bool(_mcp._STATE.active_intent)} "
                    f"pending_conflicts={bool(_mcp._STATE.pending_conflicts)}\n"
                )
        except OSError:
            pass
        return
    try:
        _mcp._INTENT_STATE_DIR.mkdir(parents=True, exist_ok=True)
        has_intent = bool(_mcp._STATE.active_intent)
        has_pending = bool(_mcp._STATE.pending_conflicts)

        if not has_intent and not has_pending:
            # Fully clean state -- nothing to persist.
            if state_file.exists():
                state_file.unlink()
            return

        if has_intent:
            cached_hierarchy = _mcp._STATE.active_intent.get("intent_hierarchy")
            if cached_hierarchy is None:
                cached_hierarchy = _build_intent_hierarchy_safe()
            state = {
                "intent_id": _mcp._STATE.active_intent["intent_id"],
                "intent_type": _mcp._STATE.active_intent["intent_type"],
                "slots": _mcp._STATE.active_intent["slots"],
                "effective_permissions": _mcp._STATE.active_intent["effective_permissions"],
                "content": _mcp._STATE.active_intent.get("content", ""),
                "agent": _mcp._STATE.active_intent.get("agent", ""),
                "session_id": _mcp._STATE.session_id,
                "intent_hierarchy": cached_hierarchy,
                "injected_memory_ids": list(
                    _mcp._STATE.active_intent.get("injected_memory_ids", set())
                ),
                "injected_by_context": dict(
                    _mcp._STATE.active_intent.get("injected_by_context", {})
                ),
                "accessed_memory_ids": list(
                    _mcp._STATE.active_intent.get("accessed_memory_ids", set())
                ),
                "budget": _mcp._STATE.active_intent.get("budget", {}),
                "used": _mcp._STATE.active_intent.get("used", {}),
                "pending_conflicts": _mcp._STATE.pending_conflicts or [],
                # pending_operation_cues (2026-04-20): list of agent-declared
                # operation cues from mempalace_declare_operation, consumed
                # by the PreToolUse hook subprocess. List form supports
                # Claude Code's parallel tool dispatch (N declares in one
                # message, N tool calls follow -- each consumes its own cue
                # by tool-name match). Hook pops first matching entry on
                # consume, writes shortened list back. Entries carry
                # declared_at_ts; the hook expires stale entries on consume
                # (see OPERATION_CUE_TTL_SECONDS in hooks_cli.py).
                "pending_operation_cues": _mcp._STATE.active_intent.get("pending_operation_cues")
                or [],
                # P1 context-as-entity: the context entity id active for
                # this intent. Writers (kg_declare_entity, kg_add,
                # _add_memory_internal) read it from active_intent to
                # emit `created_under` edges on every write.
                "active_context_id": _mcp._STATE.active_intent.get("active_context_id", "") or "",
                "contexts_touched": list(_mcp._STATE.active_intent.get("contexts_touched") or []),
                "contexts_touched_detail": list(
                    _mcp._STATE.active_intent.get("contexts_touched_detail") or []
                ),
                # 2026-04-29: persist op_args_by_ctx_tool across turn
                # boundaries. This dict maps "{context_id}|{tool}" →
                # parametrized args_summary, populated by
                # tool_declare_operation and consumed by finalize_intent
                # promotion. Without persistence, every op declared in a
                # prior turn lands with empty args_summary at finalize
                # because the in-memory state was lost. Audit on
                # 2026-04-29 found 224 post-mandate ops leaked this way.
                "op_args_by_ctx_tool": dict(
                    _mcp._STATE.active_intent.get("op_args_by_ctx_tool") or {}
                ),
            }
        else:
            # No active intent but pending state must outlive the finalize.
            # Write a minimal placeholder file that the hook ignores (no
            # intent_id key) but the declare_intent pending-check reads via
            # _load_pending_*_from_disk.
            state = {
                "intent_id": "",
                "session_id": _mcp._STATE.session_id,
                "pending_conflicts": _mcp._STATE.pending_conflicts or [],
            }
        # Atomic rename pattern (cold-start lock 2026-05-01): the prior
        # write_text path was non-atomic on Windows, which let a race
        # between the MCP server and the PreToolUse hook subprocess
        # produce concatenated JSON in the state file (the trigger for
        # _sync_from_disk's "Extra data: line N column M" recovery).
        # Two writers calling write_text concurrently can interleave
        # because Windows ReplaceFile-on-truncate isn't guaranteed
        # atomic across cross-process opens. Writing to a sibling .tmp
        # file then os.replace flips the inode (POSIX) / does an
        # atomic rename (Windows MoveFileEx with MOVEFILE_REPLACE_EXISTING)
        # so readers always see either the old complete state or the
        # new complete state -- never a partial overlay.
        tmp_path = state_file.with_suffix(state_file.suffix + ".tmp")
        try:
            tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            os.replace(str(tmp_path), str(state_file))
        finally:
            # Best-effort cleanup if the rename raised mid-operation.
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
    except OSError as _e:
        # NEVER silent: record to hook_errors.jsonl so the next SessionStart
        # (or any hook output) surfaces the failure. A silent persist loss
        # means the hook + the server disagree about active_intent forever.
        try:
            from . import hooks_cli as _hc

            _hc._record_hook_error("_persist_active_intent", _e)
        except Exception:
            # Absolute last resort: even the recorder cannot raise or we'd
            # lose the whole tool call. Swallow but leave a log breadcrumb
            # via the module-level log file already written above.
            pass


def _resolve_intent_profile(intent_type_id: str):
    """Walk is-a hierarchy to resolve effective slots and tool_permissions.

    Returns (slots, tool_permissions) where:
    - slots: merged from child to parent (child wins on conflict)
    - tool_permissions: ADDITIVE -- child tools are merged with parent tools.
      Child can only ADD tools, not remove parent tools. This prevents
      overreach: a child of inspect can add WebFetch but can't drop Read.
    """

    visited = set()
    current = intent_type_id
    merged_slots = {}
    merged_tools = []  # Additive: collect from all levels
    seen_tools = set()  # Deduplicate by tool name

    # Walk upward through is-a chain (max 5 hops)
    for _ in range(5):
        if current in visited:
            break
        visited.add(current)

        entity = _mcp._STATE.kg.get_entity(current)
        if not entity:
            break

        props = entity.get("properties", {})
        if isinstance(props, str):
            import json as _json

            try:
                props = _json.loads(props)
            except Exception:
                props = {}

        profile = props.get("rules_profile", {})

        # Slots: merge (child wins, so only add parent slots not already defined)
        for slot_name, slot_def in profile.get("slots", {}).items():
            if slot_name not in merged_slots:
                merged_slots[slot_name] = slot_def

        # Tool permissions: ADDITIVE -- collect from all levels, child + parent
        for perm in profile.get("tool_permissions", []):
            tool_key = perm.get("tool", "")
            if tool_key not in seen_tools:
                seen_tools.add(tool_key)
                merged_tools.append(perm)

        # Walk to parent via is-a -- prefer intent hierarchy over universal "thing"
        edges = _mcp._STATE.kg.query_entity(current, direction="outgoing")
        parent = None
        for e in edges:
            if e["predicate"] == "is_a" and e["current"]:
                parent_id = normalize_entity_name(e["object"])
                # Stop at the root intent_type class
                if parent_id == "intent_type":
                    break
                # Skip universal base class -- not part of intent hierarchy
                if parent_id == "thing":
                    continue
                parent_entity = _mcp._STATE.kg.get_entity(parent_id)
                if parent_entity and parent_entity.get("kind") == "class":
                    parent = parent_id
                    break
        if not parent:
            break
        current = parent

    return merged_slots, merged_tools


def _resolve_operation_profile(tool: str):
    """Look up the operation_class for a given tool and return its slot schema.

    (Adrian directive 2026-05-05): operation_class entities are
    classes with ``is_a operation`` carrying ``properties.rules_profile.tool``
    (the tool name they apply to) and ``properties.rules_profile.slots``
    (the slot schema, same shape as intent_type rules_profile.slots).

    Returns ``slots_dict`` (possibly empty) for the FIRST class found
    whose ``rules_profile.tool`` matches the given tool. When multiple
    classes match, the most-specific one (deepest in the is_a chain)
    wins; when none match, returns ``{}`` -- declare_operation falls
    back to the prior no-slot behaviour for back-compat.

    The resolver does NOT walk the is_a hierarchy upward (operations
    don't inherit slot schemas the way intent_types do -- a tool either
    has a class registered for it or it doesn't). This keeps the lookup
    O(N) over operation_class entities, which is bounded by the small
    set of tools the agent actually uses.
    """
    if not tool:
        return {}

    try:
        all_classes = _mcp._STATE.kg.list_entities(status="active", kind="class")
    except Exception:
        return {}

    matching = []
    for e in all_classes:
        # Cheap filter: must be is_a operation (subclass of the seeded
        # operation root class). Skip classes that aren't operation
        # subclasses to avoid scanning every intent_type's properties.
        try:
            edges = _mcp._STATE.kg.query_entity(e["id"], direction="outgoing")
        except Exception:
            continue
        is_op_class = any(
            edge["predicate"] == "is_a"
            and edge["current"]
            and normalize_entity_name(edge["object"]) == "operation"
            for edge in edges
        )
        if not is_op_class:
            continue

        # list_entities() does NOT include `properties` in the returned
        # dict (only id/name/type/kind/content/importance/last_touched);
        # fetch the full record via get_entity so we can read the
        # rules_profile schema. fixture-debug 2026-05-05.
        try:
            full = _mcp._STATE.kg.get_entity(e["id"])
        except Exception:
            full = None
        if not full:
            continue
        props = full.get("properties", {}) or {}
        if isinstance(props, str):
            import json as _json

            try:
                props = _json.loads(props)
            except Exception:
                props = {}
        profile = props.get("rules_profile", {}) or {}
        tool_for_class = profile.get("tool")
        if not tool_for_class:
            continue
        if tool_for_class.strip() == tool.strip():
            matching.append((e["id"], profile.get("slots", {}) or {}))

    if not matching:
        return {}
    # First match wins -- multiple classes for the same tool would be
    # an ontology bug; surface this to the gardener via a one-shot
    # log, but don't error out the operation.
    return matching[0][1]


def _is_intent_type(entity_id: str) -> bool:
    """Check if an entity is-a intent_type (direct or inherited)."""

    edges = _mcp._STATE.kg.query_entity(entity_id, direction="outgoing")
    for e in edges:
        if e["predicate"] == "is_a" and e["current"]:
            obj = normalize_entity_name(e["object"])
            if obj == "intent_type":
                return True
            # Check parent (one level -- e.g., edit_file is-a modify is-a intent_type)
            parent_edges = _mcp._STATE.kg.query_entity(obj, direction="outgoing")
            for pe in parent_edges:
                if pe["predicate"] == "is_a" and pe["current"]:
                    if normalize_entity_name(pe["object"]) == "intent_type":
                        return True
    return False


def tool_declare_intent(  # noqa: C901
    intent_type: str,
    slots: dict,
    context: dict = None,  # mandatory unified Context
    agent: str = None,
    budget: dict = None,
    cause_id: str = None,  # required (user-ctx | Task | 'autonomous')
    initial_intent_state: dict = None,  # eager-init rev0 payload (slice 11 required)
):
    """Declare what you intend to do BEFORE doing it. Returns permissions + context.

    budget: MANDATORY dict of tool_name -> max_calls. E.g. {"Read": 5, "Edit": 3}.
            Must cover all tools you plan to use. Budget is tracked by the hook --
            when exhausted, the tool is blocked until you extend (mempalace_extend_intent)
            or finalize and redeclare. Keep budgets tight -- inflated budgets waste context.

    One active intent at a time -- declaring a new intent expires the previous.
    mempalace_* tools are always allowed (not gated by intent).

    Args:
        intent_type: A declared intent type entity (is-a intent_type).
            Built-in types: inspect, modify, execute, communicate.
            Domain-specific: edit_file, write_tests, deploy, run_tests, etc.
            Declare new types via kg_declare_entity with is-a <parent_intent_type>.

        slots: Named slots filled with entity names. Each intent type defines
            expected slots with class constraints. Example:
            For edit_file:  {"files": ["auth.test.ts", "auth.utils.ts"]}
            For deploy:     {"target": ["flowsev_repository"], "environment": ["staging"]}
            For inspect:    {"subject": ["paperclip_server"]}

            Slot definitions are stored in the intent type's rules_profile.slots.
            Each slot has: classes (accepted entity classes), required (bool),
            multiple (bool -- accepts list vs single entity).

        context: MANDATORY Context fingerprint for this intent.
            {
              "queries":  list[str]   2-5 perspectives on what you're about to do
              "keywords": list[str]   2-5 caller-provided exact terms
              "entities": list[str]   0+ related/seed entity ids (defaults to slot
                                      entities when omitted -- they ARE the entities
                                      this intent is about)
            }
            Each query becomes a separate cosine view for multi-view retrieval;
            keywords drive the keyword channel (no auto-extraction); entities
            seed Channel B graph BFS. The Context's view vectors are persisted
            so future feedback applies via MaxSim. Example:
            context={
              "queries": ["Editing auth rate limiter",
                          "Security hardening against brute force",
                          "Adding tests for login endpoint"],
              "keywords": ["auth", "rate-limit", "brute-force", "login"],
              "entities": ["LoginService", "AuthRateLimiter"]
            }

    Returns:
        permissions: Which tools are allowed and their scope (scoped to slots or unrestricted).
        memories: Relevant injected memories (multi-view retrieved using the Context).
        previous_expired: ID of the previous active intent if one was replaced.
    """

    from .scoring import validate_context as _validate_context

    clean_context, ctx_err = _validate_context(
        context,
        require_summary=True,
        summary_context_for_error="declare_intent.context.summary",
    )
    if ctx_err:
        return ctx_err
    _description_views = clean_context["queries"]
    _context_keywords = clean_context["keywords"]
    _context_entities = clean_context["entities"]
    # Render context.summary to the canonical description prose. The
    # queries[0] auto-derive that used to live here was retired
    # (Adrian's design lock 2026-04-25) -- same principle as
    # tool_kg_declare_entity: no auto-derive of summary fields.
    from .knowledge_graph import serialize_summary_for_embedding as _serialize_summary

    description = _serialize_summary(clean_context["summary"])

    # fail-fast agent validation. Unified with finalize_intent and
    # every other write entry point: undeclared agents are rejected at
    # the boundary instead of causing silent downstream failures.
    sid_err = _mcp._require_sid(action="declare_intent")
    if sid_err:
        return sid_err
    agent_err = _mcp._require_agent(agent, action="declare_intent")
    if agent_err:
        return agent_err

    # ── Check for pending conflicts ──
    # Disk is source of truth -- reload from disk if memory is empty (MCP restart scenario)
    # v3.7.20 (Adrian directive 2026-05-17): pending_conflicts blocking
    # gate removed. Conflicts are resolved by Haiku in the background
    # via mempalace/conflict_resolver_auto.py; declare_intent never
    # blocks on them. mempalace_bg_status surfaces the audit trail via
    # conflict_resolver_log.jsonl for operators who want to see what
    # Haiku decided.

    # ── Validate intent_type ──
    try:
        intent_type = _mcp.sanitize_name(intent_type, "intent_type")
    except ValueError as e:
        return {"success": False, "error": str(e)}

    intent_id = normalize_entity_name(intent_type)

    if not _mcp._is_declared(intent_id):
        return {
            "success": False,
            "error": (
                f"Intent type '{intent_id}' not declared in this session. "
                f"Specific intent types are preferred over broad ones -- they carry domain-specific "
                f"rules (must, requires, has_gotcha) that broad types don't. "
                f"Create it now:\n"
                f"  1. "
                + _mcp._declare_entity_recipe(
                    intent_type,
                    kind="class",
                    hint="what this action does, when to use it",
                    extra_properties=(
                        "{'rules_profile': {'slots': {...}, 'tool_permissions': [...]}}"
                    ),
                )
                + "\n"
                f"  2. kg_add(subject='{intent_type}', predicate='is_a', "
                f"object='<parent>', context={{'queries': [...], 'keywords': [...]}}) "
                f"-- where parent is the broad type it inherits from "
                f"(inspect, modify, execute, or communicate)\n"
                f"  3. Then retry declare_intent with this type.\n"
                f"This is a one-time cost -- once created, the type persists across sessions "
                f"and accumulates rules that will be surfaced on every future use."
            ),
        }

    if not _is_intent_type(intent_id):
        return {
            "success": False,
            "error": (
                f"'{intent_id}' exists but is not an intent type (missing is_a edge to the hierarchy). "
                f"Link it to the parent it inherits from:\n"
                f"  kg_add(subject='{intent_id}', predicate='is_a', object='<parent>')\n"
                f"Where parent is the broad type it specializes "
                f"(inspect, modify, execute, or communicate). "
                f"The type will then inherit its parent's permissions and slots, "
                f"and you can attach domain-specific rules to it."
            ),
        }

    # ── Auto-narrow: use description to find best-fit child intent type ──
    narrowed_from = None
    subtypes = []
    child_scores = []
    # Only kind=class -- execution instances (kind=entity) are NOT subtypes
    all_entities = _mcp._STATE.kg.list_entities(status="active", kind="class")
    for e in all_entities:
        e_edges = _mcp._STATE.kg.query_entity(e["id"], direction="outgoing")
        for edge in e_edges:
            if edge["predicate"] == "is_a" and edge["current"]:
                parent_id = normalize_entity_name(edge["object"])
                if parent_id == intent_id:
                    subtypes.append(
                        {
                            "id": e["id"],
                            "content": e.get("content", ""),
                        }
                    )
                    break

    if subtypes and description.strip():
        ecol = _mcp._get_entity_collection(create=False)
        if ecol:
            try:
                child_id_set = {s["id"] for s in subtypes}
                count = ecol.count()
                if count > 0:
                    results = ecol.query(
                        query_texts=[description],
                        n_results=min(count, 50),
                        include=["documents", "metadatas", "distances"],
                    )
                    # Collect distances for parent and children
                    parent_dist = None
                    child_scores = []  # (id, distance, description)
                    if results["ids"] and results["ids"][0]:
                        for i, eid in enumerate(results["ids"][0]):
                            dist = results["distances"][0][i]
                            if eid == intent_id:
                                parent_dist = dist
                            elif eid in child_id_set:
                                child_scores.append(
                                    {
                                        "id": eid,
                                        "distance": dist,
                                        "content": results["documents"][0][i],
                                    }
                                )
                    # Auto-narrow: if a child is closer than the parent, it's
                    # a better fit for the agent's description. Use it.
                    # But only if the child's slots are compatible with what was provided.
                    if parent_dist is not None and child_scores:
                        child_scores.sort(key=lambda c: c["distance"])
                        better = [c for c in child_scores if c["distance"] < parent_dist]
                        # Filter out children whose required slots don't match
                        compatible = []
                        for candidate in better:
                            child_slots, _ = _resolve_intent_profile(candidate["id"])
                            if not child_slots:
                                continue
                            # Check: all required child slots must be present in provided slots
                            missing = [
                                s
                                for s, d in child_slots.items()
                                if d.get("required", False) and s not in slots
                            ]
                            if not missing:
                                compatible.append(candidate)
                        if len(compatible) == 1:
                            narrowed_from = intent_id
                            intent_id = compatible[0]["id"]
                            _mcp._STATE.declared_entities.add(intent_id)
                        elif len(compatible) > 1:
                            # Multiple children beat the parent -- disambiguate
                            return {
                                "success": False,
                                "error": (
                                    f"Description matches multiple subtypes of '{intent_id}' "
                                    f"better than '{intent_id}' itself. "
                                    f"Pick the most appropriate one and declare it directly."
                                ),
                                "matching_subtypes": [
                                    {"id": c["id"], "content": c["content"][:120]}
                                    for c in compatible
                                ],
                            }
            except Exception:
                child_scores = []  # Non-fatal -- narrowing is best-effort

    # ── Resolve effective profile via inheritance ──
    effective_slots, effective_permissions = _resolve_intent_profile(intent_id)

    if not effective_slots:
        return {
            "success": False,
            "error": (
                f"Intent type '{intent_id}' has no slots defined in its rules_profile. "
                f"Update its properties to include rules_profile.slots. Example: "
                f'{{"slots": {{"files": {{"classes": ["file"], "required": true, "multiple": true}}}}}}'
            ),
        }

    # ── Validate slots ──
    if not isinstance(slots, dict):
        # Kind-aware error: tell the writer (a) what type the validator
        # actually received (so they can spot MCP transport stringification
        # of dict args), and (b) per-slot example values that match each
        # slot's KIND -- raw glob/command strings, file paths, or
        # pre-declared entity names. The legacy template used
        # ["entity_name"] for every slot regardless of kind, which
        # actively misled callers into passing entity ids for raw `paths`
        # or `commands` slots.
        def _slot_example(name, sd):
            if sd.get("raw", False):
                if name == "paths":
                    return '["D:/Flowsev/repo/**"]'
                if name == "commands":
                    return '["pytest", "git status"]'
                return '["raw_string"]'
            classes = sd.get("classes", ["thing"])
            if "file" in classes:
                return '["src/auth.py"]'
            return '["my_entity_name"]'

        def _slot_legend(name, sd):
            if sd.get("raw", False):
                return f"{name}=raw string"
            classes = sd.get("classes", ["thing"])
            if "file" in classes:
                return f"{name}=file path (auto-declared)"
            return f"{name}=pre-declared entity (classes: {classes})"

        legend = ", ".join(_slot_legend(k, v) for k, v in effective_slots.items())
        example_body = ", ".join(
            f'"{k}": {_slot_example(k, v)}' for k, v in effective_slots.items()
        )
        return {
            "success": False,
            "error": (
                f"slots must be a JSON object/dict, received "
                f"{type(slots).__name__}. "
                f"(If you passed a dict and still see this, the MCP "
                f"transport may have stringified it -- re-fetch the "
                f"declare_intent schema via ToolSearch and retry.) "
                f"Expected slots for '{intent_id}': {legend}. "
                f"Example: {{{example_body}}}"
            ),
        }

    slot_errors = []
    resolved_slots = {}  # slot_name -> list of normalized entity IDs

    # Check required slots are present
    for slot_name, slot_def in effective_slots.items():
        if slot_def.get("required", False) and slot_name not in slots:
            slot_errors.append(
                f"Required slot '{slot_name}' not provided. "
                f"Accepted classes: {slot_def.get('classes', ['thing'])}."
            )

    # Check provided slots are valid
    for slot_name, slot_values in slots.items():
        if slot_name not in effective_slots:
            slot_errors.append(
                f"Unknown slot '{slot_name}'. Valid slots: {list(effective_slots.keys())}."
            )
            continue

        slot_def = effective_slots[slot_name]

        # Normalize to list
        if isinstance(slot_values, str):
            slot_values = [slot_values]
        if not isinstance(slot_values, list):
            slot_errors.append(f"Slot '{slot_name}' must be a string or list of strings.")
            continue

        # Check multiple
        if not slot_def.get("multiple", False) and len(slot_values) > 1:
            slot_errors.append(
                f"Slot '{slot_name}' accepts only one entity (multiple=false), got {len(slot_values)}."
            )
            continue

        # Raw slots: accept strings as-is, no entity declaration needed
        # Used for command patterns, URLs, etc.
        if slot_def.get("raw", False):
            normalized_values = [{"id": val, "raw": val} for val in slot_values]
            resolved_slots[slot_name] = normalized_values
            continue

        # Validate each entity in slot
        normalized_values = []
        allowed_classes = slot_def.get("classes", ["thing"])
        is_file_slot = "file" in allowed_classes

        for val in slot_values:
            # For file slots: use basename for entity name, keep raw path for scoping
            if is_file_slot:
                file_basename = os.path.basename(val)
                val_id = normalize_entity_name(file_basename)
            else:
                val_id = normalize_entity_name(val)

            # ── design-lock 2026-04-28 ──
            # This block is the SOLE entity auto-naming surface in mempalace.
            # Per the 2026-04-26/28 id-design discussion (Adrian): every
            # entity must carry a model/user-authored name; the only
            # exception is file entities whose name is the basename of an
            # already-known file path (deterministic, structural, not a
            # noun-phrase guess from prose). Other paths confirmed
            # auto-naming-free by the 2026-04-28 codebase audit:
            #   - kg_add / kg_add_batch: reject undeclared subject/object
            #     with a structured "declare first" error (tool_mutate.py).
            #   - Mining (miner.py): pure 800-char chunking, no entity
            #     extraction.
            #   - Gardener (link_author.py): operates on existing entity
            #     names; the LLM jury authors edges, not entity names.
            #   - Intent slot resolution: rejects unknown entity names
            #     except file slots routed through THIS block.
            #   - sanitize_name (config.py): validates shape via
            #     _SAFE_NAME_RE without auto-trimming; raises ValueError
            #     on bad shape rather than silently mangling.
            # Adding any new auto-naming codepath violates the design
            # lock -- route through kg_declare_entity with a model-authored
            # name instead, OR document the new structural-derivation rule
            # here alongside the file-basename case.
            # (Adrian directive 2026-05-05): the auto-
            # declare path was retired. Pre-fix it minted file entities
            # with a placeholder {what, why, scope} stub at slot-
            # validation time and immediately flagged them for the
            # gardener -- but those stub entities polluted retrieval
            # (cosine on stub prose, no real signal) and the gardener
            # only got around to refining them on later passes.
            # Forcing the agent to call mempalace_kg_declare_entity
            # first means file entities carry a real caller-authored
            # summary from creation; one-time friction the first time
            # a file is referenced amortises across reuse (file
            # entities live forever; agents stop re-paying after a
            # few sessions). The fall-through reject below catches
            # un-declared file slots with a kg_declare_entity-first
            # hint same as any other slot.
            if not _mcp._is_declared(val_id):
                if is_file_slot:
                    slot_errors.append(
                        f"File entity '{val_id}' in slot '{slot_name}' not "
                        f"declared Files no longer auto-declare "
                        f"-- call mempalace_kg_declare_entity(name='{file_basename}', "
                        f"kind='entity', is_a='file', summary={{...}}, "
                        f"context={{...}}, added_by='<agent>') first. The one-"
                        f"time declaration cost amortises across reuse: file "
                        f"entities are reusable across intents and sessions, "
                        f"and a real caller-authored summary keeps retrieval "
                        f"clean. The earlier auto-declare path minted stubs "
                        f"that polluted Channel A cosine until the gardener "
                        f"got around to rewriting them."
                    )
                else:
                    slot_errors.append(
                        f"Entity '{val_id}' in slot '{slot_name}' not declared. "
                        f"Call kg_declare_entity first."
                    )
                continue

            # Check class constraint via is-a + inheritance
            if "thing" not in allowed_classes:
                entity_classes = [
                    e["object"]
                    for e in _mcp._STATE.kg.query_entity(val_id, direction="outgoing")
                    if e["predicate"] == "is_a" and e["current"]
                ]
                if entity_classes:
                    from .knowledge_graph import normalize_entity_name as _norm

                    norm_classes = [_norm(c) for c in entity_classes]
                    norm_allowed = [_norm(c) for c in allowed_classes]

                    def _check_subclass(classes, allowed, depth=5):
                        if any(c in allowed for c in classes):
                            return True
                        visited = set(classes)
                        frontier = list(classes)
                        for _ in range(depth):
                            nxt = []
                            for cls in frontier:
                                for e in _mcp._STATE.kg.query_entity(cls, direction="outgoing"):
                                    if e["predicate"] == "is_a" and e["current"]:
                                        p = _norm(e["object"])
                                        if p in allowed:
                                            return True
                                        if p not in visited:
                                            visited.add(p)
                                            nxt.append(p)
                            frontier = nxt
                            if not frontier:
                                break
                        return False

                    if not _check_subclass(norm_classes, norm_allowed):
                        slot_errors.append(
                            f"Entity '{val_id}' in slot '{slot_name}' is-a {entity_classes}, "
                            f"but slot requires classes {allowed_classes}."
                        )
                        continue

            normalized_values.append({"id": val_id, "raw": val})
        resolved_slots[slot_name] = normalized_values

    if slot_errors:
        return {
            "success": False,
            "error": "Slot validation failed for declare_intent.",
            "slot_issues": slot_errors,
            "expected_slots": {
                name: {
                    "classes": d.get("classes", ["thing"]),
                    "required": d.get("required", False),
                    "multiple": d.get("multiple", False),
                }
                for name, d in effective_slots.items()
            },
        }

    # ── Build permissions ──
    # Flatten resolved_slots for return (id only) and keep raw paths for scoping
    flat_slots = {}  # slot_name -> [entity_id, ...]
    raw_paths = {}  # slot_name -> [raw_value, ...]
    all_slot_entities = []
    raw_slot_names = set()
    for slot_name, entries in resolved_slots.items():
        flat_slots[slot_name] = [e["id"] for e in entries]
        raw_paths[slot_name] = [e["raw"] for e in entries]
        # Check if this is a raw slot (commands, etc.) -- don't add to entity list
        slot_def = effective_slots.get(slot_name, {})
        if slot_def.get("raw", False):
            raw_slot_names.add(slot_name)
        else:
            all_slot_entities.extend(flat_slots[slot_name])

    def _resolve_file_path(entity_id):
        """Resolve actual file path for a file entity.

        Checks entity properties for 'file_path', then falls back to
        extracting the path from the description (format: 'path/to/file.py -- ...')
        """
        entity = _mcp._STATE.kg.get_entity(entity_id)
        if not entity:
            return None
        # Check properties first
        props = entity.get("properties", {})
        if isinstance(props, str):
            import json as _json

            try:
                props = _json.loads(props)
            except Exception:
                props = {}
        fp = props.get("file_path")
        if fp:
            return fp
        # Fall back to description -- extract path from known formats
        desc = entity.get("content", "")
        # Format: "File: /path/to/file.ext" or "File: /path/to/file.ext (new)"
        if desc.startswith("File: "):
            candidate = desc[6:].split("(")[0].strip()
            if "/" in candidate or "\\" in candidate:
                return candidate
        # Format: "path/to/file.py -- description text"
        for sep in (" -- ", " - ", " - "):
            if sep in desc:
                candidate = desc.split(sep, 1)[0].strip()
                if (
                    "/" in candidate
                    or "\\" in candidate
                    or candidate.endswith((".py", ".ts", ".js", ".json"))
                ):
                    return candidate
        return None

    permissions = []
    for slot_name, entity_ids in flat_slots.items():
        raws = raw_paths.get(slot_name, entity_ids)
        # Check if this slot contains file entities -- resolve actual paths
        slot_def = effective_slots.get(slot_name, {})
        slot_classes = slot_def.get("classes", [])
        is_file_slot = "file" in slot_classes
        for perm in effective_permissions:
            scope = perm.get("scope", "*")
            if f"{{{slot_name}}}" in scope:
                for raw_val, entity_id in zip(raws, entity_ids):
                    resolved_scope = raw_val
                    if is_file_slot:
                        file_path = _resolve_file_path(entity_id)
                        if not file_path:
                            return {
                                "success": False,
                                "error": (
                                    f"File entity '{entity_id}' has no file_path configured. "
                                    f"Either re-declare it with properties={{'file_path': "
                                    f"'path/to/file.ext'}} using "
                                    + _mcp._declare_entity_recipe(
                                        entity_id,
                                        kind="entity",
                                        hint=f"file entity {entity_id}",
                                        extra_properties="{'file_path': 'path/to/file.ext'}",
                                    )
                                    + ", or update it via kg_update_entity(entity='"
                                    + entity_id
                                    + "', properties={'file_path': 'path/to/file.ext'})."
                                ),
                            }
                        resolved_scope = file_path
                    permissions.append(
                        {
                            "tool": perm["tool"],
                            "scope": scope.replace(f"{{{slot_name}}}", resolved_scope),
                            "slot": slot_name,
                            "entity": entity_id,
                        }
                    )
            elif scope == "*":
                if not any(p["tool"] == perm["tool"] and p["scope"] == "*" for p in permissions):
                    permissions.append({"tool": perm["tool"], "scope": "*"})
            else:
                if not any(p["tool"] == perm["tool"] and p["scope"] == scope for p in permissions):
                    permissions.append({"tool": perm["tool"], "scope": scope})

    # ── Validate budget (after permissions so slot/type errors come first) ──
    if not budget or not isinstance(budget, dict):
        return {
            "success": False,
            "error": (
                "budget is MANDATORY. Provide a dict of tool_name -> max_calls. "
                'Example: budget={"Read": 5, "Edit": 3, "Bash": 2}. '
                "Keep budgets tight -- estimate the minimum calls needed for this task."
            ),
        }
    # Validate budget: only keep tools that are actually permitted
    permitted_tool_names = {p["tool"] for p in permissions}
    validated_budget = {}
    for tool_name, count in budget.items():
        if tool_name not in permitted_tool_names:
            continue  # Silently ignore -- permission check blocks anyway
        try:
            n = int(count)
            if n < 1:
                return {
                    "success": False,
                    "error": f"Budget for '{tool_name}' must be >= 1, got {n}",
                }
            validated_budget[tool_name] = n
        except (TypeError, ValueError):
            return {
                "success": False,
                "error": f"Budget for '{tool_name}' must be int, got {count!r}",
            }
    if not validated_budget:
        return {
            "success": False,
            "error": (
                f"Budget has no permitted tools. Permitted: {sorted(permitted_tool_names)}. "
                f"Budget must include at least one of these."
            ),
        }

    # ── Collect context via 3-channel retrieval ──
    context = {"memories": []}

    # ── 3-channel retrieval: cosine + graph + keyword → RRF merge ──

    # ── Context-scoped relevance feedback (signed, confidence-graded) ──
    # The signal is read from rated_useful / rated_irrelevant edges on the
    # active context PLUS its 1-2 hop similar_to neighbourhood
    # (lookup_context_feedback). finalize_intent stores
    # confidence = relevance/5.0 on each rated_* edge, so the mapping is:
    #
    #   relevance 5 useful      → confidence 1.0 → boost +1.0
    #   relevance 1 useful      → confidence 0.2 → boost +0.2
    #   no feedback             → 0.0 (neutral)
    #   relevance 1 irrelevant  → confidence 0.2 → penalty -0.2
    #   relevance 5 irrelevant  → confidence 1.0 → penalty -1.0
    #
    # The dict is populated AFTER _views is built and context_lookup_or_create
    # has minted / reused an active_context_id (below). Until then _relevance_boost
    # returns 0 (no signal) -- retrieval runs AFTER the populate step anyway.
    _context_feedback: dict = {}

    def _relevance_boost(memory_id):
        """Return continuous relevance signal from context feedback.

        Returns float in [-1.0, +1.0]. Feeds hybrid_score as the signed
        relevance_feedback term -- rated_irrelevant memories drop below
        neutral, rated_useful rise above.
        """
        return _context_feedback.get(memory_id, 0.0)

    def _preview(entity_id_or_memory):
        """Get text preview for any ID -- memory content or entity description."""
        if entity_id_or_memory.startswith(("record_", "diary_")):
            try:
                col = _mcp._get_collection(create=False)
                if col:
                    d = col.get(ids=[entity_id_or_memory], include=["documents"])
                    if d and d["documents"] and d["documents"][0]:
                        return d["documents"][0][:150].replace("\n", " ")
            except Exception:
                pass
        else:
            try:
                ent = _mcp._STATE.kg.get_entity(entity_id_or_memory)
                if ent and ent.get("content"):
                    return ent["content"][:150].replace("\n", " ")
            except Exception:
                pass
        return ""

    already_seen_ids = set()  # dedup across all channels

    # ── Build multi-view queries from caller's context.queries ONLY ──
    # Channel-separation lock 2026-05-02 (audit
    # record_ga_agent_channel_violation_saturation): the prior auto-build
    # appended intent_id literal + first 200 chars of each slot entity's
    # content to _views, mixing Channel B inputs (entity content) into
    # Channel A (cosine views). Two consequences:
    #   (1) properties.queries on the persisted context contained strings
    #       the caller never typed -- data corruption.
    #   (2) max-of-max similar_to saturated at 1.0 across every context
    #       sharing any slot entity, because the entity content[:200]
    #       string is byte-identical at every emit site.
    # Channel B reachability (find this context by walking from a slot
    # entity) is now provided by an explicit anchored_by graph edge
    # written inside context_lookup_or_create (Channel B BFS), not by
    # smuggling entity text into the cosine view set.
    _views = list(_description_views)  # caller's queries -- Channel A semantic
    if not _views and description:
        _views.append(description)
    _views = list(dict.fromkeys(_views))[:6]
    if not _views:
        # Last-resort fallback: caller passed no queries at all and there
        # is no description. Use the literal intent_id so context_lookup_
        # or_create has SOMETHING to embed; the resulting context will
        # still be findable via the anchored_by edges.
        _views = [intent_id or "unknown"]

    # ── Context as first-class entity ──
    # Mint or reuse a kind="context" entity BEFORE the retrieval loops
    # so _relevance_boost can read rated_* edges scoped to this context's
    # similar_to neighbourhood. declare_intent is an emit site; other
    # writers (kg_declare_entity, _add_memory_internal) will reference
    # this id via created_under.
    _active_context_id = ""
    _active_context_reused = False
    try:
        _cid, _reused, _cms = _mcp.context_lookup_or_create(
            queries=_views,
            keywords=_context_keywords,
            entities=_context_entities,
            agent=agent or "",
            summary=clean_context.get("summary"),
        )
        _active_context_id = _cid or ""
        _active_context_reused = bool(_reused)
    except Exception:
        _active_context_id = ""

    # ── eager-init intent_state rev0 ────────────────────
    # State-protocol v3 (Adrian directive 2026-05-04). The activity-
    # intent's context entity IS the intent instance -- it should
    # carry intent_state from the moment it's minted, not retrofitted
    # later by a gardener that never fires in practice.
    #
    # Validates initial_intent_state against intent_state json_schema
    # via record_state_revision (hardening already does the
    # jsonschema.validate). Defaults to {"todos": []} when omitted --
    # the minimum payload satisfying intent_state.required = ["todos"].
    # When _active_context_id is empty (mint failed) we skip silently;
    # state-protocol degrades to no-rev0 rather than blocking the
    # whole declare_intent on a substrate hiccup.
    #
    # Reused contexts skip rev0: the prior intent that minted this
    # context already wrote its initial state, and subsequent intents
    # land deltas via state_deltas at finalize, not new rev0s. (A
    # reused context with no prior revision is theoretically possible
    # if the prior intent crashed pre-rev0; the gardener will retrofit
    # via the state_init_needed flag in that edge case.)
    if _active_context_id and not _active_context_reused:
        # (Adrian directive 2026-05-04 after observing agents
        # skip the field): initial_intent_state is now MANDATORY. Reject
        # at the boundary instead of silently defaulting to {todos: []}.
        # The MCP schema also lists initial_intent_state in required[], so
        # this handler-side check is defense-in-depth for callers that
        # bypass the MCP transport (direct Python imports, tests).
        if not isinstance(initial_intent_state, dict):
            return {
                "success": False,
                "error": (
                    "declare_intent.initial_intent_state is MANDATORY "
                    " Pass a dict matching state_schemas."
                    "STATE_SCHEMAS['intent_state'].json_schema -- "
                    "minimum {todos: []} satisfies the schema, but "
                    "pre-populate with the ACTUAL todos for this intent "
                    "so subsequent declare_operation calls can patch "
                    "individual items via /todos/N/status RFC 6902 paths. "
                    "See wake_up.schemas.intent_state for the full shape."
                ),
            }
        _rev0_payload = initial_intent_state
        try:
            _mcp._STATE.kg.record_state_revision(
                entity_id=_active_context_id,
                schema_id="intent_state",
                payload=_rev0_payload,
                op_context_id="",  # rev0 is anchored at declare time, not by an op
                agent=agent or "",
                session_id=_mcp._STATE.session_id or None,
            )
        except ValueError as _ve:
            # Schema validation failed -- surface a clear error so the
            # agent knows their initial_intent_state shape is wrong
            # rather than silently dropping the rev0 write.
            return {
                "success": False,
                "error": (
                    f"declare_intent.initial_intent_state failed schema "
                    f"validation: {_ve}. See wake_up.schemas.intent_state "
                    f"for the required shape (required: 'todos'; minimum "
                    f"payload {{'todos': []}})."
                ),
            }
        except Exception:
            # Substrate-level failures (table missing, transient SQLite
            # error) shouldn't block the whole declare_intent. Log via
            # the system's existing telemetry and proceed; the gardener
            # retrofit path remains available as a fallback.
            pass

    # ── cause_id validation + caused_by edge ─────────────
    # Optional parent-cause linkage: when cause_id is provided, validate
    # it is either (a) a kind='context' entity with at least one
    # fulfills_user_message outgoing edge (i.e. a user-context minted by
    # mempalace_declare_user_intents earlier this turn), or (b) a
    # kind='entity' entity with an is_a Task edge (paperclip / scheduled
    # path). On success, write a caused_by edge from this activity-
    # intent's context to the cause. Telemetry: stash on active_intent
    # so finalize_intent can apply the user-context
    # feedback coverage rule scoped to this cause.
    _resolved_cause_id = ""
    _resolved_cause_kind = ""  # "user_context" or "task" or "autonomous"
    # first-rater snapshot defaults -- populated only on the
    # cause_kind=='user_context' path below. For Task or no-cause cases
    # they stay at their first-rater=True / no-exemption defaults so the
    # active_intent dict reads them safely.
    _user_ctx_first_rater = True
    _user_ctx_exempt_ids: list = []

    # (Adrian directive 2026-05-04): cause_id is now
    # MANDATORY. Three accepted forms: a user-context id, a Task entity
    # id, or the literal string 'autonomous' for intents with no
    # parent. Reject empty/missing -- the earlier back-compat optional
    # was the same trap as slice-2's initial_intent_state silent
    # default: agents skip without thinking. The 'autonomous' escape
    # forces the agent to acknowledge no parent rather than silently
    # leaving cause_id blank. The MCP schema also has cause_id in
    # required[]; this handler check is defense-in-depth.
    _cid_raw = (cause_id or "").strip() if isinstance(cause_id, str) else ""
    if not _cid_raw:
        return {
            "success": False,
            "error": (
                "declare_intent.cause_id is MANDATORY. "
                "Pass one of:\n"
                "  - A user-context entity id "
                "(contexts[*].ctx_id from mempalace_declare_user_intents) "
                "when this intent fulfils a user prompt.\n"
                "  - A Task entity id (kind='entity', is_a Task) when "
                "this intent fulfils a long-running task.\n"
                "  - The literal string 'autonomous' when this intent "
                "has no parent (background gardener pass, scheduled "
                "audit, agent-initiated reflection). The handler "
                "writes no caused_by edge but the explicit value "
                "forces you to acknowledge no parent rather than "
                "silently skipping."
            ),
        }
    if _cid_raw == "autonomous":
        # v3.6.0 Slice A (Adrian directive 2026-05-16): sub-agent
        # sessions cannot self-declare cause_id="autonomous". The
        # parent agent dispatched them; their work always inherits a
        # parent cause. Allowing the "autonomous" magic string from
        # sub-agents drops the causal attribution chain and lets
        # gardener / link-author work float free of the user message
        # that triggered the whole flow. Reject with a directive that
        # tells the parent how to fix.
        _sid_for_subagent_check = _mcp._STATE.session_id or ""
        if "__sub_" in _sid_for_subagent_check:
            return {
                "success": False,
                "error": (
                    "SUB-AGENT PROTOCOL VIOLATION: cause_id='autonomous' "
                    "is rejected for sub-agents. Sub-agents inherit their "
                    "work from the parent agent; they cannot declare "
                    "themselves autonomous.\n\n"
                    "How to fix in the PARENT agent:\n"
                    "  1. Declare a Task entity that lays out the work:\n"
                    "     mempalace_kg_declare_entity(\n"
                    "       kind='entity', is_a='Task',\n"
                    "       name='task_<descriptive_slug>',\n"
                    "       added_by='<parent_agent>',\n"
                    "       importance=4,\n"
                    "       context={ ... what+why+scope of the task ... })\n"
                    "  2. Re-dispatch this sub-agent. Prefix the sub-agent "
                    "prompt with 'task_id=task_<descriptive_slug>' as the "
                    "first line; the sub-agent reads its parent task id "
                    "from that line and passes the 'task_<slug>' string as "
                    "cause_id in its FIRST mempalace_declare_intent call "
                    "(replacing 'autonomous').\n\n"
                    "Why: causal attribution must chain through the Task "
                    "so the sub-agent's intents trace back to the user "
                    "message that triggered the parent."
                ),
                "error_kind": "subagent_autonomous_rejected",
            }
        # Explicit no-parent escape. No edge written; record the
        # cause_kind so finalize_intent + telemetry can distinguish
        # autonomous intents from user-driven / task-driven ones.
        _resolved_cause_kind = "autonomous"
        _resolved_cause_id = ""
    elif cause_id and isinstance(cause_id, str) and cause_id.strip():
        _cid_clean = cause_id.strip()
        try:
            _cause_ent = _mcp._STATE.kg.get_entity(_cid_clean)
        except Exception:
            _cause_ent = None
        if not _cause_ent:
            return {
                "success": False,
                "error": (
                    f"cause_id={_cid_clean!r} does not resolve to any entity. "
                    "Pass either a user-context id (returned by "
                    "mempalace_declare_user_intents.contexts[*].ctx_id) "
                    "or a Task entity id (kind='entity', is_a Task)."
                ),
            }
        _cause_kind = _cause_ent.get("kind")
        try:
            _cause_edges = _mcp._STATE.kg.query_entity(_cid_clean, direction="outgoing")
        except Exception:
            _cause_edges = []
        _is_a_targets = {
            e.get("object")
            for e in _cause_edges
            if e.get("predicate") == "is_a" and e.get("current", True)
        }
        _has_fulfills = any(
            e.get("predicate") == "fulfills_user_message" and e.get("current", True)
            for e in _cause_edges
        )
        if _cause_kind == "context" and _has_fulfills:
            _resolved_cause_kind = "user_context"
        elif _cause_kind == "entity" and "Task" in _is_a_targets:
            _resolved_cause_kind = "task"
        else:
            return {
                "success": False,
                "error": (
                    f"cause_id={_cid_clean!r} is not a valid parent cause. "
                    f"Got kind={_cause_kind!r}; expected either kind='context' "
                    f"with at least one fulfills_user_message edge (a user-"
                    f"context from declare_user_intents), or kind='entity' "
                    f"with is_a Task (a Task entity). is_a targets: "
                    f"{sorted(_is_a_targets)}; has_fulfills_user_message: "
                    f"{_has_fulfills}."
                ),
            }
        _resolved_cause_id = _cid_clean

        # v3.6.1 (Adrian directive 2026-05-16, follow-on to Slice A):
        # Sub-agent sessions MUST anchor to a Task entity. user-context
        # cause_ids belong to the PARENT session that received the user
        # message; sub-agents cannot inherit them directly (the parent's
        # intent does that). The Slice A "autonomous" rejection above
        # closed the magic-word loophole; this closes the second
        # loophole where a sub-agent could pass the parent's user-context
        # ctx_id and bypass Task-entity attribution. The Task entity is
        # the agreed bridge between user-tier and sub-agent-tier intents.
        _sid_for_subagent_check = _mcp._STATE.session_id or ""
        if "__sub_" in _sid_for_subagent_check and _resolved_cause_kind != "task":
            return {
                "success": False,
                "error": (
                    "SUB-AGENT PROTOCOL VIOLATION: sub-agent declare_intent "
                    f"cause_id={_cid_clean!r} resolves to "
                    f"cause_kind={_resolved_cause_kind!r} but sub-agents "
                    "MUST anchor to a Task entity (kind='entity', is_a Task). "
                    "user-context cause_ids belong to the parent session "
                    "that received the user message.\n\n"
                    "How to fix in the PARENT agent:\n"
                    "  1. Declare a Task entity that lays out the work:\n"
                    "     mempalace_kg_declare_entity(\n"
                    "       kind='entity', is_a='Task',\n"
                    "       name='task_<descriptive_slug>',\n"
                    "       added_by='<parent_agent>',\n"
                    "       importance=4,\n"
                    "       context={ ... what+why+scope of the task ... })\n"
                    "  2. Re-dispatch this sub-agent. Prefix the sub-agent "
                    "prompt with 'task_id=task_<descriptive_slug>' as the "
                    "first line; the sub-agent reads its parent task id "
                    "from that line and passes the 'task_<slug>' string as "
                    "cause_id in its FIRST mempalace_declare_intent call.\n\n"
                    "Why: the user-context surfaced for the parent's "
                    "session is the parent's anchor; sub-agents must "
                    "anchor to the Task that the parent scoped for them, "
                    "so causal attribution chains through "
                    "user_message -> parent intent -> Task -> sub-agent "
                    "intent."
                ),
                "error_kind": "subagent_non_task_cause_rejected",
            }

        # snapshot first-rater state for cause_kind='user_context'.
        # The FIRST agent intent that finalizes against a given user-context
        # carries full feedback coverage of the user-context's surfaced
        # memories. Subsequent intents with the same cause_id inherit the
        # coverage and are exempt from re-rating those exact memories. We
        # snapshot AT DECLARE TIME (not finalize) so the rating contract is
        # established when the agent commits to the intent -- stable across
        # any later finalize / extend_feedback path.
        _user_ctx_first_rater = True
        _user_ctx_exempt_ids: list = []
        if _resolved_cause_kind == "user_context":
            _sid_for_rated = _mcp._STATE.session_id or ""
            _rated_set = _rated_user_contexts_for(_sid_for_rated)
            if _resolved_cause_id in _rated_set:
                _user_ctx_first_rater = False
                # Look up which memories the user-context surfaced via
                # `surfaced` outgoing edges. Soft-fail: a stale or empty
                # user-context just means no exemptions, which is the
                # safe default (full coverage required).
                try:
                    _ctx_edges = _mcp._STATE.kg.query_entity(
                        _resolved_cause_id,
                        direction="outgoing",
                    )
                    _user_ctx_exempt_ids = sorted(
                        {
                            e.get("object")
                            for e in _ctx_edges
                            if e.get("predicate") == "surfaced"
                            and e.get("current", True)
                            and e.get("object")
                        }
                    )
                except Exception:
                    _user_ctx_exempt_ids = []

        # Write the caused_by edge. The predicate is non-skip-list so
        # add_triple requires a natural-language statement (per the
        # 2026-04-19 lock that retired autogenerated verbalisations).
        # Build a short verbalisation of the parent linkage. Soft-fail
        # at edge level so a transient kg issue doesn't prevent intent
        # declaration entirely; cause_id persists on active_intent
        # regardless so finalize can apply its coverage rule.
        if _active_context_id:
            _caused_by_statement = (
                f"This activity-intent context ({_active_context_id}) "
                f"was caused by {_resolved_cause_kind.replace('_', ' ')} "
                f"{_resolved_cause_id} per the user-intent tier "
                f"hierarchy."
            )
            try:
                _mcp._STATE.kg.add_triple(
                    _active_context_id,
                    "caused_by",
                    _resolved_cause_id,
                    statement=_caused_by_statement,
                )
            except Exception:
                pass

    # Pre-compute the intent-level emit entry; merged into
    # active_intent.contexts_touched_detail right after the dict is
    # built (see below). Rocchio enrichment at finalize will iterate
    # every entry in that detail list, not just this one, so operation
    # and search contexts also qualify for enrichment when reused +
    # net-positive.
    _intent_emit_entry = {
        "ctx_id": _active_context_id,
        "reused": _active_context_reused,
        "scope": "intent",
        "queries": list(_views),
        "keywords": list(_context_keywords),
        "entities": list(_context_entities),
        # v3.5.0: ids surfaced under this intent-level emit, used by
        # mempalace.feedback_auto at finalize-time to ship the
        # intent_memories Haiku-rater batch. Back-filled below once
        # `already_injected` is populated by the retrieval pipeline.
        "surfaced_ids": [],
    }

    # ══════════════════════════════════════════════════════════════
    # CHANNELS A+C: Unified retrieval -- BOTH collections.
    # Uses the SAME scoring.multi_channel_search as kg_search. Each
    # collection runs Channels A (multi-view cosine) and C (keyword
    # overlap) internally; results merge into a shared RRF pot with
    # Channel B (graph BFS, below). Entity AND record candidates
    # compete head-to-head for injection -- rules, concepts, gotchas,
    # past executions, and prose records all surface by relevance.
    #
    # This replaces the pre-P6.6 split where entities were queried
    # for _entity_sim (thrown away as candidates) and only records
    # became Channel A results. Now EVERYTHING is a candidate.
    # ══════════════════════════════════════════════════════════════
    from . import scoring as _scoring

    _channel_a_lists = {}  # unified: "record_cosine_0", "entity_cosine_0", etc.
    _combined_meta = {}  # mid -> {"meta": {...}, "doc": "...", "similarity": float}
    _entity_sim = {}  # entity_id -> max similarity (still needed by Channel B)

    # Share ONE walk of the context neighbourhood across all three
    # collection pipes (record / entity / triple) AND _relevance_boost.
    # The walker returns two aggregates:
    #   - rated_scores: per-memory signed float for hybrid_score's W_REL
    #     (consumed via _relevance_boost / _context_feedback below).
    #   - channel_D_list: ranked list for Channel D (passed via
    #     rated_walk kwarg into multi_channel_search).
    _rated_walk = (
        _scoring.walk_rated_neighbourhood(_active_context_id, _mcp._STATE.kg)
        if _active_context_id
        else {"rated_scores": {}, "channel_D_list": [], "contributing_contexts": {}}
    )
    _context_feedback = _rated_walk.get("rated_scores") or {}
    # Step 2 of similar_context_id flag (record_ga_agent_similar_context_id_
    # flag_design_2026_04_30): surface neighbour contributions per memory so
    # the agent can monitor which similar_to neighbours contributed to each
    # injected memory. Default-on; the active context is excluded from the
    # map by walk_rated_neighbourhood itself.
    _contributing_contexts = _rated_walk.get("contributing_contexts") or {}

    # Shared VectorStore handle for all three collection pipes
    # (record / entity / triple). Tier 2 migration: scoring functions
    # take (vs, collection_name) instead of raw chromadb Collection.
    from .vector_store import (
        CONTEXT_VIEWS_COLLECTION as _CV_NAME,  # noqa: F401  (kept for downstream callers)
        RECORDS_COLLECTION as _RECORDS_NAME,
        TRIPLES_COLLECTION as _TRIPLES_NAME,
        get_vector_store as _get_vs,
    )

    _vs = _get_vs(_mcp._STATE.config.palace_path)

    # Record collection (prose records -- the old "memory" collection)
    try:
        if _vs.count(_RECORDS_NAME) >= 0:
            record_pipe = _scoring.multi_channel_search(
                _vs,
                _RECORDS_NAME,
                _views,
                keywords=_context_keywords,
                kg=_mcp._STATE.kg,
                fetch_limit_per_view=50,
                include_graph=False,
                active_context_id=_active_context_id,
                rated_walk=_rated_walk,
                caller="declare_intent:records",
            )
            for name, lst in record_pipe.get("ranked_lists", {}).items():
                _channel_a_lists[f"record_{name}"] = lst
            for mid, info in record_pipe.get("seen_meta", {}).items():
                _combined_meta[mid] = {**info, "source": "record"}
    except Exception:
        pass

    # Entity collection (structured entities -- rules, concepts, past execs)
    # Post-M1 lives in the same RECORDS_COLLECTION discriminated by metadata.kind.
    try:
        if _vs.count(_RECORDS_NAME) >= 0:
            entity_pipe = _scoring.multi_channel_search(
                _vs,
                _RECORDS_NAME,
                _views,
                keywords=_context_keywords,
                kg=_mcp._STATE.kg,
                fetch_limit_per_view=50,
                include_graph=False,
                active_context_id=_active_context_id,
                rated_walk=_rated_walk,
                caller="declare_intent:entities",
            )
            for name, lst in entity_pipe.get("ranked_lists", {}).items():
                _channel_a_lists[f"entity_{name}"] = lst
            # Build _entity_sim from entity pipe's seen_meta (Channel B needs it)
            for mid, info in entity_pipe.get("seen_meta", {}).items():
                meta = info.get("meta") or {}
                logical_id = meta.get("entity_id") or mid
                sim = info.get("similarity", 0.0)
                _entity_sim[logical_id] = max(_entity_sim.get(logical_id, 0.0), sim)
                _combined_meta[mid] = {**info, "source": "entity"}
    except Exception:
        pass

    # Triple verbalization collection -- surfaces structured (subject, predicate,
    # object) facts as first-class injected context. Without this, declare_intent
    # only sees prose memories and entity descriptions; triples like
    # (adrian, lives_in, warsaw) only contribute to the BFS Channel B if an
    # entity in the walk happens to attach them.
    try:
        if _vs.count(_TRIPLES_NAME) > 0:
            triple_pipe = _scoring.multi_channel_search(
                _vs,
                _TRIPLES_NAME,
                _views,
                keywords=_context_keywords,
                kg=_mcp._STATE.kg,
                fetch_limit_per_view=50,
                include_graph=False,
                active_context_id=_active_context_id,
                rated_walk=_rated_walk,
                caller="declare_intent:triples",
            )
            for name, lst in triple_pipe.get("ranked_lists", {}).items():
                _channel_a_lists[f"triple_{name}"] = lst
            for mid, info in triple_pipe.get("seen_meta", {}).items():
                _combined_meta[mid] = {**info, "source": "triple"}
    except Exception:
        pass

    # ══════════════════════════════════════════════════════════════
    # CHANNEL B: Graph -- BFS from slot entities + intent type
    # Subsumes old sources 1 (KG edges), 2 (intent rules),
    # 4 (past executions), 5 (graph memories).
    #
    # Graph-seed derivation strategy (P5.9 doc):
    # This is the CONTROLLED-BFS variant, complementing the autonomous
    # top-cosine-seeds strategy in scoring.multi_channel_search. The
    # intent declaration already NAMES the entities it's about (via
    # slots + context.entities), so we anchor the walk on those rather
    # than guessing from semantic similarity. Two modes are intentional:
    #   - declare_intent → controlled BFS (caller knows the anchors)
    #   - kg_search      → autonomous BFS (caller doesn't always know)
    # ══════════════════════════════════════════════════════════════
    GRAPH_BUDGET = 30
    _MAX_HOPS = 3
    _MIN_EDGE_USEFULNESS = -0.5
    _GRAPH_SIM = {1: 0.5, 2: 0.3, 3: 0.1}
    _graph_memories = {}  # memory_id -> distance (for hop-shortening in finalize)
    _graph_entities = {}  # entity_id -> distance
    _channel_b_list = []
    _past_exec_ids = []  # for promotion check
    try:
        # BFS seeds: slot entities + intent type
        # Channel B seeds: slot entities + intent type + caller-provided
        # context.entities (explicit graph anchors). Slots stay as the
        # default backbone; context.entities augments them.
        bfs_seeds = list(all_slot_entities)
        for cent in _context_entities or []:
            cent_id = normalize_entity_name(cent)
            if cent_id and cent_id not in bfs_seeds:
                bfs_seeds.append(cent_id)
        if intent_id and intent_id not in bfs_seeds:
            bfs_seeds.append(intent_id)
        bfs_queue = [(eid, 0) for eid in bfs_seeds]
        visited = set(bfs_seeds)
        items_explored = 0

        while bfs_queue and items_explored < GRAPH_BUDGET:
            current_id, distance = bfs_queue.pop(0)
            if distance >= _MAX_HOPS:
                continue

            edges = _mcp._STATE.kg.query_entity(current_id, direction="both")
            for e in edges:
                if items_explored >= GRAPH_BUDGET:
                    break
                if not e.get("current", True):
                    continue
                pred = e["predicate"]
                subj = e["subject"]
                obj = e["object"]
                # Skip OUTGOING is_a (don't walk up type hierarchy)
                # Allow INCOMING is_a (find instances: past executions is_a intent_type)
                if pred == "is_a" and subj == current_id:
                    continue

                # Edge-usefulness gating RETIRED (P2). The old
                # edge_traversal_feedback table was dropped in migration
                # 015; the signal it provided is now expressed by
                # context --rated_useful--> memory edges consumed by
                # Channel D at retrieval time. Keeping BFS unfiltered
                # here lets every current edge contribute; the final
                # hybrid-score reranker still applies the signed W_REL
                # term so rated-irrelevant memories sink.

                other = obj if subj == current_id else subj
                if other in visited:
                    continue
                visited.add(other)
                items_explored += 1

                new_dist = distance + 1
                graph_sim = _GRAPH_SIM.get(new_dist, 0.1)

                # ── Channel B score = pure graph-walk signal ──
                # Two-stage retrieval (Nogueira/Cho 2019; Bruch 2023): each
                # channel ranks by its own natural signal, RRF fuses the
                # ranks, and the post-RRF reranker applies the feature-rich
                # hybrid_score. Mixing importance/decay/relevance_feedback
                # into the channel rank would double-count those terms once
                # the post-fusion rerank runs (it applies hybrid_score over
                # every RRF winner). Keep this channel honest by scoring
                # only (distance-based graph_sim, cosine overlap, log-degree
                # dampening). The reranker handles the rest.
                try:
                    _deg = len(_mcp._STATE.kg.query_entity(other, direction="both") or [])
                except Exception:
                    _deg = 0
                import math as _math

                _degree_damp = 1.0 / _math.log(_deg + 2)

                if other.startswith(("record_", "diary_")):
                    _graph_memories.setdefault(other, new_dist)
                    try:
                        col = _mcp._get_collection(create=False)
                        if col:
                            d = col.get(ids=[other], include=["documents", "metadatas"])
                            if d and d["ids"]:
                                score = graph_sim * _degree_damp
                                snippet = (d["documents"][0] or "")[:150].replace("\n", " ")
                                _channel_b_list.append((score, snippet, other))
                    except Exception:
                        pass
                else:
                    _graph_entities.setdefault(other, new_dist)
                    # Track past executions (instances of intent type via is_a)
                    if pred == "is_a" and obj == current_id:
                        _past_exec_ids.append(other)
                    preview = _preview(other)
                    if preview:
                        arrow = "->" if subj == current_id else "<-"
                        text = f'{arrow} {pred} {arrow} {other}: "{preview}"'
                        # effective_sim = max(graph_sim, cosine_sim) keeps
                        # the channel aware of entity-level cosine overlap
                        # without pulling in the reranker's feature space.
                        cosine_sim = _entity_sim.get(other, 0.0)
                        effective_sim = max(graph_sim, cosine_sim)
                        score = effective_sim * _degree_damp
                        _channel_b_list.append((score, text, other))
                    # Continue BFS from entities (not memories)
                    if new_dist < _MAX_HOPS:
                        bfs_queue.append((other, new_dist))

                # Channel B triple emission: emit the traversed edge
                # itself (not just the neighbour entity) so triples get
                # RRF cross-channel boost. Without this, triples only
                # surface via Channel A cosine over mempalace_triples and
                # never accumulate rank contributions from multiple
                # channels the way memories/entities do. Skip-list
                # predicates (schema glue, feedback topology) are
                # excluded -- same filter as _index_triple_statement uses
                # at embed time, for the same reason (low-signal text).
                from .knowledge_graph import _TRIPLE_SKIP_PREDICATES

                triple_id = e.get("triple_id")
                statement = e.get("statement")
                if triple_id and statement and pred not in _TRIPLE_SKIP_PREDICATES:
                    triple_text = (statement or "")[:200].replace("\n", " ")
                    _channel_b_list.append((graph_sim * _degree_damp, triple_text, triple_id))
                    _combined_meta[triple_id] = {
                        "meta": {
                            "subject": subj,
                            "predicate": pred,
                            "object": obj,
                            "confidence": e.get("confidence", 1.0),
                        },
                        "doc": triple_text,
                        "similarity": 0.0,
                        "source": "triple",
                    }
    except Exception:
        pass  # Non-fatal

    # Channel C (keyword) is now built INTO multi_channel_search -- no
    # separate keyword pass needed. The record_pipe and entity_pipe above
    # already include keyword-ranked lists when _context_keywords is non-empty.

    # ══════════════════════════════════════════════════════════════
    # RRF MERGE -- unified across A (cosine) + B (graph) + C (keyword)
    # All channels from both collections compete head-to-head.
    # ══════════════════════════════════════════════════════════════
    all_rrf_lists = dict(_channel_a_lists)
    if _channel_b_list:
        all_rrf_lists["graph"] = _channel_b_list

    # ══════════════════════════════════════════════════════════════
    # Canonical two-stage pipeline (rrf → hybrid_score rerank →
    # adaptive_k) centralised in scoring.two_stage_retrieve. Every
    # context-creating tool routes through this helper so declare_intent
    # / declare_operation / kg_search produce results on the same scale
    # with the same semantics.
    # ══════════════════════════════════════════════════════════════
    from .scoring import two_stage_retrieve as _two_stage

    reranked, rrf_scores, candidate_map = _two_stage(
        all_rrf_lists,
        _combined_meta,
        agent=agent or "",
        session_id=_mcp._STATE.session_id or "",
        intent_type_id=intent_id or "",
        context_feedback=_context_feedback,
        rerank_top_m=50,
        max_k=20,
        min_k=3,
        # v3.7.34 FINDING #T fix: hand kg in so the rerank loop can
        # batch-fetch fresh entities.last_touched and apply touch-on-use
        # as the decay clock. See scoring.two_stage_retrieve for the
        # full rationale.
        kg=_mcp._STATE.kg,
    )

    already_injected = set()
    # Cold-start lock 2026-05-01 (Adrian's render-divergence fix): preview
    # comes from scoring.render_memory_preview reading SQLite
    # entities.properties.summary as the single source of truth, not from
    # the reranker's per-view r["text"] which is the matched Chroma view's
    # document. Pre-fix, when a probe view (eid__vN) outranked the abstract
    # record (eid), declare_intent leaked the probe-query string into the
    # preview -- same latent bug as hooks_cli._run_local_retrieval, just
    # masked by the abstract record usually winning the rerank.
    from .scoring import render_memory_preview as _render_memory_preview

    for r in reranked:
        memory_id = r["id"]
        # Channel-separation lock 2026-05-02: skip the just-minted active
        # context entity itself if it surfaces as its own retrieval hit.
        # Pre-channel-fix the polluted entity-content view in the active
        # context's _views list happened to push self-similarity below
        # the rerank cutoff for free; post-fix the views are tighter so
        # the active context's own stored views match its own query views
        # at cosine 1.0 and the context shows up as its own top hit.
        # That makes finalize_intent's coverage rule require feedback on
        # an id the agent never saw as a memory, breaking the contract.
        if _active_context_id and memory_id == _active_context_id:
            continue
        # Kind filter (Adrian directive 2026-05-12, post-chromadb-removal):
        # classes (intent types, ontology roots) and predicates (KG edge
        # labels) are GLUE entities, not retrievable memories the agent
        # needs to rate at finalize time. Under chromadb's lazy
        # embeddings_queue these never reached the HNSW index for
        # sub-100-row test corpora, so they were invisible to retrieval
        # and the issue was latent. sqlite_vec indexes writes
        # immediately, so the filter has to be explicit at the surface
        # boundary -- otherwise finalize_intent demands feedback on
        # every is_a / found_useful / intent_type id, none of which is
        # an actual memory.
        _r_kind = ((_combined_meta.get(memory_id) or {}).get("meta") or {}).get("kind", "")
        # v3.7.43 FINDING #AA (Adrian msg_c96c8a_146+147 2026-05-19):
        # add user_message to the skip list. Cold-start lock 2026-05-01
        # makes user_message entities SQLite-only graph anchors -- not
        # memories. They reach this rerank loop via Channel B graph BFS
        # along fulfills_user_message edges from surfaced contexts;
        # without this filter, bare user-turn text (e.g. "reinstalled")
        # surfaces as a memory with no conversational context. That's
        # noise during knowledge retrieval AND wrong per the literature
        # (MemGPT, Generative Agents, MemoryBank, Letta all store
        # dialogue with explicit speaker turns; bare strings without
        # context never qualify as "memories"). Same filter pattern as
        # the existing class/predicate skip (other graph-glue kinds).
        if _r_kind in ("class", "predicate", "user_message"):
            continue
        # v3.7.9 (Adrian directive 2026-05-17): pass the FULL rendered
        # preview to the canonical _project_memory helper so the entry
        # carries content + content_trimmed + content_redundant where
        # applicable. Pre-v3.7.9 this site emitted only summary_text
        # (the _shorten_preview head), silently dropping the content
        # body and creating shape divergence vs declare_intent's
        # main memories projection at line ~3990. Now both paths emit
        # identical entries -- one helper, one shape.
        raw_preview = _render_memory_preview(
            memory_id, _mcp._STATE.kg, fallback_text=r.get("text") or ""
        )
        already_seen_ids.add(memory_id)
        already_injected.add(memory_id)
        extras = {}
        if DEBUG_RETURN_SCORES:
            # hybrid_score = scoring.hybrid_score output after the post-RRF
            # rerank. Uniform across declare_intent / declare_operation /
            # kg_search -- same function, same scale (0.3-0.8).
            extras["hybrid_score"] = round(float(r["hybrid_score"]), 6)
        # v3.7.34: surface date_added + last_relevant_at via the
        # _project_memory hoist. The vec meta is already on
        # _combined_meta from the rerank pipeline, so the lookup is
        # zero-cost.
        _r_meta_for_proj = (_combined_meta.get(memory_id) or {}).get("meta") or {}
        if _r_meta_for_proj:
            extras["metadata"] = _r_meta_for_proj
        entry = _project_memory(memory_id, raw_preview, extras=extras)
        # Per-memory similar_context_ids are added by
        # scoring.render_similar_contexts_block below in one pass with
        # the top-level similar_contexts builder; no inline duplication.
        context["memories"].append(entry)

    # Build past_exec_candidates for promotion check from graph-discovered executions
    past_exec_candidates = []
    for eid in _past_exec_ids:
        rrf_score = rrf_scores.get(eid, 0.0)
        text, _ = candidate_map.get(eid, ("", ""))
        past_exec_candidates.append((rrf_score, text, "graph", eid))

    # ── Mandatory type promotion check: 3+ similar executions ──
    PROMOTION_COUNT = 3
    BASE_THRESHOLD = 0.7
    if len(past_exec_candidates) >= PROMOTION_COUNT:
        parent_threshold = BASE_THRESHOLD
        try:
            type_entity = _mcp._STATE.kg.get_entity(intent_id)
            if type_entity:
                props = type_entity.get("properties", {})
                if isinstance(props, str):
                    props = json.loads(props)
                parent_threshold = props.get("promoted_at_similarity", BASE_THRESHOLD)
        except Exception:
            pass

        if parent_threshold < 1.0:
            # Use score as similarity proxy for promotion check
            high_sim = [c for c in past_exec_candidates if c[0] > parent_threshold]
            if len(high_sim) >= PROMOTION_COUNT:
                avg_sim = sum(c[0] for c in high_sim) / len(high_sim)
                exec_list = "\n".join(f"  - {c[3]}: {c[1][:100]}" for c in high_sim[:5])
                return {
                    "success": False,
                    "error": (
                        f"Intent type '{intent_id}' has {len(high_sim)} similar past executions "
                        f"above threshold {parent_threshold:.2f}. You MUST either:\n\n"
                        f"(a) Create a specific intent type (set promoted_at_similarity={avg_sim:.3f}):\n"
                        f"    "
                        + _mcp._declare_entity_recipe(
                            "<specific-type>",
                            kind="class",
                            hint="what this action does",
                            extra_properties=(
                                f"{{'promoted_at_similarity': {avg_sim:.3f}, "
                                f"'rules_profile': {{...}}}}"
                            ),
                        )
                        + "\n"
                        f"    kg_add(subject='<specific-type>', predicate='is_a', object='{intent_id}', "
                        f"context={{'queries': [...], 'keywords': [...]}})\n"
                        f"    Then re-declare with the specific type.\n\n"
                        f"(b) Disambiguate existing executions (if they're actually different):\n"
                        f"    kg_update_entity(entity='<exec_id>', summary={{'what': ..., 'why': ...}}, "
                        f"context={{'queries': ['<new meaning>', '<angle 2>'], "
                        f"'keywords': ['<term1>', '<term2>']}})\n\n"
                        f"Similar executions (avg similarity {avg_sim:.3f}):\n{exec_list}"
                    ),
                    "similar_executions": [
                        {"id": c[3], "summary_text": c[1][:100]} for c in high_sim[:5]
                    ],
                    "promotion_threshold": parent_threshold,
                    "suggested_promoted_at_similarity": round(avg_sim, 3),
                }

    # ── Hard fail if previous intent not finalized ──
    if _mcp._STATE.active_intent:
        prev_id = _mcp._STATE.active_intent.get("intent_id")
        prev_type = _mcp._STATE.active_intent.get("intent_type", "unknown")
        prev_desc = _mcp._STATE.active_intent.get("content", "")
        return {
            "success": False,
            "error": (
                f"Active intent '{prev_type}' ({prev_id}) has not been finalized. "
                f"You MUST call mempalace_finalize_intent before declaring a new intent. "
                f"Only the agent knows how to properly summarize what happened.\n\n"
                f"Call: mempalace_finalize_intent(\n"
                f"  slug='<descriptive-slug>',\n"
                f"  outcome='success' | 'partial' | 'failed' | 'abandoned',\n"
                f"  content='<full narrative body -- what happened in detail>',\n"
                f"  summary='<≤280-char distilled one-liner of the outcome>',\n"
                f"  agent='<your_agent_name>'\n"
                f")\n\n"
                f"Previous intent: {prev_type} -- {prev_desc[:100]}"
            ),
            "active_intent": prev_id,
        }

    intent_hash = hashlib.md5(
        f"{intent_id}:{description}:{datetime.now().isoformat()}".encode()
    ).hexdigest()[:12]
    new_intent_id = f"intent_{intent_id}_{intent_hash}"

    # bake a Context-ranked intent_hierarchy ONCE here so the
    # PreToolUse hook has a pre-sorted list and never needs to retrieve.
    # Uses the same 3-channel pipeline as kg_search -- no reinvented
    # similarity math.
    context_for_ranking = {
        "queries": list(_description_views),
        "keywords": list(_context_keywords),
    }
    ranked_hierarchy = _build_intent_hierarchy_safe(context_for_ranking)

    # _memory_scoring_snapshot retired (P3 polish): the weight-learning
    # feedback path now reads signals directly from the sim + rel
    # seen_meta + _context_feedback at finalize time rather than a
    # separate snapshot dict on active_intent. Cleaner -- fewer
    # persistent fields to maintain.
    #
    # _active_context_id was minted earlier (before the retrieval loops
    # so _relevance_boost could consume context-scoped feedback). No
    # second call here.

    # Parallel map to injected_memory_ids that preserves which context
    # surfaced each id. Still persisted across MCP restarts; in v3.5.0
    # the async-Haiku rater (mempalace.feedback_auto) consumes the
    # equivalent surfaced_ids list on contexts_touched_detail to ship
    # one rater batch per retrieval site.
    _injected_by_context = (
        {_active_context_id: sorted(already_injected)}
        if (_active_context_id and already_injected)
        else {}
    )

    # v3.5.0: back-fill the intent emit entry's surfaced_ids now that
    # `already_injected` has been populated by the retrieval pipeline.
    # feedback_auto.submit_finalize_feedback at finalize reads
    # contexts_touched_detail[*].surfaced_ids to ship per-context
    # Haiku-rater batches.
    if _active_context_id and already_injected:
        _intent_emit_entry["surfaced_ids"] = sorted(already_injected)

    _mcp._STATE.active_intent = {
        "intent_id": new_intent_id,
        "intent_type": intent_id,
        "slots": flat_slots,
        "effective_permissions": permissions,
        "injected_memory_ids": already_injected,
        "injected_by_context": _injected_by_context,
        "accessed_memory_ids": set(),
        "_graph_memories_snapshot": dict(_graph_memories),  # distance map for hop-shortening
        "content": description,
        "_context_views": _views,  # multi-view query strings for context vector storage
        "active_context_id": _active_context_id,  # P1 context-as-entity
        # State-protocol follow-up #3 (Adrian directive
        # 2026-05-06): persistent intent-level context id, never
        # overwritten by declare_operation. ``active_context_id`` is
        # used for KG-write attribution and gets clobbered with the
        # most-recent op-ctx on every declare_operation; gate B reads
        # this field instead so it stays pointed at the intent's
        # primary ctx for the whole intent's lifetime. Returned in
        # declare_intent's response so agents know the id to put in
        # state_deltas.
        "intent_context_id": _active_context_id,
        # Every context id touched during this intent (intent-level +
        # any operation/search emits). Enumerated at finalize to build
        # the strict coverage set: every (ctx, memory) surfaced pair
        # must have a rated_* edge or finalize is rejected.
        "contexts_touched": [_active_context_id] if _active_context_id else [],
        # Per-emit detail list -- one entry per context emit during the
        # intent's lifecycle. Finalize iterates this to run Rocchio
        # enrichment independently per reused context. Initialised with
        # the intent-level emit; declare_operation + kg_search append
        # their own entries via _record_context_emit.
        "contexts_touched_detail": ([_intent_emit_entry] if _active_context_id else []),
        "agent": agent or "",
        "budget": validated_budget,
        "used": {},  # tool_name -> count, incremented by hook
        "intent_hierarchy": ranked_hierarchy,  # cached, context-ranked
        # parent-cause linkage. cause_id is the validated
        # entity id from declare_intent (user-context OR Task entity)
        # written to active_intent so finalize_intent can apply the
        # user-context feedback coverage rule scoped to that cause.
        # cause_kind is "user_context" / "task" / "" (none).
        "cause_id": _resolved_cause_id,
        "cause_kind": _resolved_cause_kind,
        # first-rater snapshot for cause_kind='user_context'.
        # user_context_first_rater is True iff this intent IS the first
        # one in this session to finalize against that user-context.
        # When False, user_context_exempt_ids enumerates the memory ids
        # that were surfaced under cause_id at declare-user-intents time
        # -- finalize subtracts them from injected/accessed coverage so
        # subsequent intents inherit the prior intent's ratings rather
        # than repeating them.
        "user_context_first_rater": bool(_user_ctx_first_rater),
        "user_context_exempt_ids": list(_user_ctx_exempt_ids),
    }

    # Persist to state file for PreToolUse hook (runs in separate process)
    _persist_active_intent()

    _mcp._wal_log(
        "declare_intent",
        {
            "intent_id": new_intent_id,
            "intent_type": intent_id,
            "slots": flat_slots,
            "content": description[:200],
        },
    )

    # feedback_reminder removed 2026-04-21: rules live in wake_up protocol.

    # Ranked subtype suggestions -- top 3 that score well AND have required tools
    ranked_suggestions = []
    needed_tools = set(validated_budget.keys()) if validated_budget else set()
    if not narrowed_from and subtypes and description.strip():
        try:
            for cs in sorted(child_scores, key=lambda c: c["distance"])[:10]:
                sim = round(1 - cs["distance"], 3)
                if sim <= 0.1:
                    continue
                # Check if this subtype has the tools we need
                if needed_tools:
                    _, sub_tools = _resolve_intent_profile(cs["id"])
                    sub_tool_names = {t["tool"] for t in sub_tools} if sub_tools else set()
                    if not needed_tools.issubset(sub_tool_names):
                        continue
                ranked_suggestions.append(
                    {
                        "id": cs["id"],
                        "similarity": sim,
                        "content": (cs.get("content") or "")[:100],
                    }
                )
                if len(ranked_suggestions) >= 3:
                    break
        except Exception:
            pass

    # ── Injection-stage gate ──
    # Filter the composed memories list via the Haiku-backed relevance
    # gate before returning to the main agent. Dropped items are
    # persisted as rated_irrelevant feedback (rater_kind='gate_llm')
    # on the active context via kg.record_feedback -- entity drops
    # become rated_* edges; triple drops land in
    # triple_context_feedback. No phantom entities. Fail-open: any
    # gate exception passes memories through unchanged.
    _gate_status = None
    _gate_report = None
    try:
        from .injection_gate import apply_gate as _apply_gate

        _gated, _gate_status, _gate_report = _apply_gate(
            memories=context["memories"],
            combined_meta=_combined_meta,
            primary_context={
                "source": "declare_intent",
                "queries": list(_description_views),
                "keywords": list(_context_keywords),
                "entities": list(_context_entities or []),
            },
            context_id=_active_context_id or "",
            kg=_mcp._STATE.kg,
            agent=agent,
            parent_intent=None,  # declare_intent IS the root frame
        )
        context["memories"] = _gated
    except Exception:
        # Any wiring bug must not kill the declare_intent path.
        pass

    # State-protocol v1 (Adrian 2026-05-03): enrich any
    # state-bearing surfaced memories with current_state +
    # state_schema_id parallel to declare_operation /
    # declare_user_intents -- agents need the current value to author
    # meaningful JSON Patches.
    _di_schemas: dict = {}
    try:
        _di_schemas = _enrich_memories_with_state(context["memories"], _mcp._STATE.kg) or {}
    except Exception:
        pass
    if _di_schemas:
        context["schemas"] = _di_schemas

    # ── Fix: re-derive injected_memory_ids from POST-GATE memories ──
    # already_injected was populated pre-gate and contained every item
    # that the retrieval pipeline surfaced. The injection gate then
    # filtered context["memories"], dropping items the agent never
    # actually saw. Persisting the PRE-gate set made finalize demand
    # feedback on gate-dropped items the agent couldn't possibly rate,
    # producing a perpetual `coverage 0%` failure mode (same pattern
    # hook_userpromptsubmit already fixed at hooks_cli.py:1518-1521;
    # also parallels _persist_accessed_memory_ids's rendered-only
    # contract).
    #
    # We rebuild from the filtered list so the persisted set contains
    # exactly what the agent saw in the response. Gate-dropped items
    # still get their rated_irrelevant edges from apply_gate, so the
    # retrieval-learning signal is preserved -- we just don't demand
    # agent re-rating of what it never received.
    already_injected = {m["id"] for m in context["memories"] if m.get("id")}

    # Step 2 of similar_context_id flag (default-on): build top-level
    # similar_contexts list of Context objects (queries/keywords/summary
    # + link_score) for each unique neighbour cid that contributed
    # weight to ANY surviving memory. Mirrors the existing context-block
    # reuse rendering. Helper also annotates each memory entry with
    # similar_context_ids in place; field omitted when no neighbours
    # contributed (token-diet).
    from . import scoring as _scoring_render

    _similar_contexts_block = _scoring_render.render_similar_contexts_block(
        context["memories"],
        _contributing_contexts,
        _rated_walk.get("neighbourhood_weights") or {},
        _mcp._STATE.kg,
    )

    # Token-diet response: we deliberately DON'T echo `intent_type`,
    # `slots`, or `budget` -- the caller just sent them, and the intent_id
    # itself carries the type (intent_{type}_{hash}). Anyone who genuinely
    # needs the normalized slot values or remaining budget should call
    # mempalace_active_intent, which is the single source of truth for
    # reconstructing the declaration. Keeping the return lean saves ~100
    # tokens per declare on typical intents and prevents tests from
    # coupling to server-side echoes.
    # follow-up (Adrian directive 2026-05-06): permissions
    # echo dropped from the response. The agent already knows the
    # intent_type and slots it sent; the server-computed permissions
    # list is fully inferable from those plus the intent_type's
    # tool_permissions registry (which lives in declared.intent_types
    # from wake_up). Echoing it back was noise on every declare. Same
    # lesson as the state_deltas_hint drop (f899e65) and the
    # intent_type/slots/budget non-echo above: the response should
    # carry only what the agent COULDN'T have computed itself --
    # intent_id, intent_context_id, retrieved memories, schemas.
    # Anyone who genuinely needs the resolved permissions can call
    # mempalace_active_intent (which still surfaces them).
    result = {
        "success": True,
        "intent_id": new_intent_id,
        # follow-up #3 (Adrian directive 2026-05-06): the
        # persistent intent-level context id, surfaced unconditionally
        # so the agent can reference it in state_deltas across the
        # intent's lifetime. ``active_context_id`` exists too but it
        # rotates per declare_operation; this one stays put.
        **({"intent_context_id": _active_context_id} if _active_context_id else {}),
        "memories": context["memories"],
        **({"schemas": context["schemas"]} if context.get("schemas") else {}),
    }
    # similar_contexts visibility is centralised in
    # scoring.render_similar_contexts_block (hide-by-default + opt-in
    # via MEMPALACE_SHOW_SIMILAR_CONTEXTS=1; Adrian directive
    # 2026-05-04). The helper returns [] when hidden, so an empty
    # block here means "do not emit".
    if _similar_contexts_block:
        result["similar_contexts"] = _similar_contexts_block
    if _gate_status is not None:
        result["gate_status"] = _gate_status
    # v3.7.10 (Adrian directive 2026-05-17): gate_report no longer
    # attached inline. Same telemetry now available via
    # mempalace_bg_status(streams=["gate_log"]) which tails
    # ~/.mempalace/hook_state/gate_log.jsonl -- where apply_gate
    # already writes per-call rows (input/output counts + ms +
    # tokens + cache_creation/read). Removing the inline copy saves
    # ~150-250 bytes per declare_intent response.
    if DEBUG_RETURN_CONTEXT:
        # Token-diet 2026-04-24: non-reused contexts collapse to the
        # literal string "new" -- the caller just sent the cue, no
        # need to echo it back. On reuse we return the stored id +
        # the queries we retrieved under (often different from what
        # the caller sent), so the agent can see what matched. The
        # shape of the `context` field itself signals reuse: string
        # "new" = fresh mint; object = reused.
        if _active_context_reused:
            result["context"] = {
                "id": _active_context_id,
                "queries": list(_description_views),
            }
        else:
            result["context"] = "new"
    if narrowed_from:
        result["narrowed_from"] = narrowed_from
    if ranked_suggestions:
        result["better_intent_types"] = ranked_suggestions
    return result


def tool_active_intent():
    """Return the current active intent, or null if none declared.

    Shows: intent type, permissions, budget remaining, the full state-
    schema catalog, and the current state payloads for the active
    intent's context entity + the agent (state-protocol v3, Adrian
    directive 2026-05-04). Agents that lost context (mid-session
    truncation, tool restart) can call this to recover both the shape
    contract (schemas, same as wake_up) and the live values without a
    fresh wake_up round trip.
    """
    _sync_from_disk()
    if not _mcp._STATE.active_intent:
        return {
            "active": False,
            "message": "No active intent. Call mempalace_declare_intent before acting.",
        }
    perms = _mcp._STATE.active_intent["effective_permissions"]
    budget = _mcp._STATE.active_intent.get("budget", {})
    used = _mcp._STATE.active_intent.get("used", {})
    remaining = {k: budget.get(k, 0) - used.get(k, 0) for k in budget}
    result = {
        "active": True,
        "intent_id": _mcp._STATE.active_intent["intent_id"],
        "intent_type": _mcp._STATE.active_intent["intent_type"],
        "slots": _mcp._STATE.active_intent.get("slots", {}),
        "permissions": [f"{p['tool']}({p.get('scope', '*')})" for p in perms],
        "budget_remaining": remaining,
    }

    # ── v3 follow-on: schemas catalog + current states for recovery ──
    # Mirrors wake_up.schemas (full STATE_SCHEMAS registry) so an agent
    # that lost context after wake_up can re-fetch the shape contract
    # without re-bootstrapping. Also surfaces the live state payloads
    # for the implicit-active-set (active intent's context entity +
    # the agent), so deltas can be authored against real current
    # values rather than guessed shapes.
    try:
        from .state_schemas import STATE_SCHEMAS as _SS

        result["schemas"] = {sid: dict(sdef) for sid, sdef in _SS.items()}
    except Exception:
        pass

    states: dict = {}
    # 1. The active intent's context entity carries intent_state
    # (rev0 written eagerly by declare_intent slice 2; later deltas
    # land via state_deltas at finalize / declare_operation).
    #
    # FINDING #M (v3.7.26 2026-05-18, Adrian's second-pass audit):
    # use ``intent_context_id`` (stable for the whole intent lifetime)
    # NOT ``active_context_id`` (rotates with each declare_operation
    # for KG-write attribution). The earlier choice of
    # active_context_id surfaced the WRONG intent_state payload
    # whenever a non-mempalace tool had been declared since the
    # intent started -- callers like Adrian's session saw stale
    # ctx_11714 "push_v350" todos from days-old intents leaking
    # through. Fall back to active_context_id only when
    # intent_context_id is missing (back-compat for active_intent
    # entries minted before the 2026-05-06 split).
    _active_ctx = (
        _mcp._STATE.active_intent.get("intent_context_id")
        or _mcp._STATE.active_intent.get("active_context_id")
        or ""
    )
    _sess_id = _mcp._STATE.session_id or None
    if _active_ctx:
        try:
            # Phase D: scope-aware read. intent_state is
            # session-scoped, so passing session_id ensures we only
            # see this session's revisions even if a prior session
            # wrote a different intent_state on the same context id.
            _intent_state = _mcp._STATE.kg.latest_state_for_entity(_active_ctx, session_id=_sess_id)
            if _intent_state is not None:
                states["intent_state"] = {
                    "entity_id": _active_ctx,
                    "payload": _intent_state,
                }
        except Exception:
            pass
    # 2. The agent's own state. Note: agent_state eager-init lands in
    # slice 5b (implicit-active-set); until then this read may return
    # None on agents that predate the rollout -- harmless, surfaces
    # the absence so callers know to seed. agent_state is also
    # session-scoped per v2 schema, so we pass session_id here too.
    _agent_id = _mcp._STATE.active_intent.get("agent") or ""
    if _agent_id:
        try:
            _agent_state = _mcp._STATE.kg.latest_state_for_entity(_agent_id, session_id=_sess_id)
            if _agent_state is not None:
                states["agent_state"] = {
                    "entity_id": _agent_id,
                    "payload": _agent_state,
                }
        except Exception:
            pass
    if states:
        result["states"] = states

    # (Adrian directive 2026-05-04): surface pending
    # conflict ids in active_intent so an agent that lost context
    # mid-session (e.g. wake_up dropped, MCP restart) can still
    # discover what's blocking PreToolUse and call resolve_conflicts
    # with the right ids. Mirrors the wake_up surfacing in slice 7.
    try:
        _pending = _mcp._STATE.pending_conflicts or []
        if _pending:
            result["pending_conflicts"] = [
                {
                    k: v
                    for k, v in c.items()
                    if k
                    in (
                        "id",
                        "conflict_type",
                        "reason",
                        "existing_id",
                        "existing_preview",
                        "new_id",
                        "similarity",
                        "past_resolution",
                    )
                }
                for c in _pending
                if isinstance(c, dict)
            ]
    except Exception:
        pass

    return result


def tool_extend_intent(budget: dict, agent: str = None):
    """Extend the active intent's tool budget without redeclaring.

    Use when your budget is exhausted but you're still working on the same task.
    Adds the specified counts to the existing budget.

    Args:
        budget: Dict of tool_name -> additional_calls. E.g. {"Read": 3, "Edit": 2}.
        agent: Your agent name (for logging).
    """
    sid_err = _mcp._require_sid(action="extend_intent")
    if sid_err:
        return sid_err
    _sync_from_disk()
    if not _mcp._STATE.active_intent:
        return {"success": False, "error": "No active intent to extend."}

    if not budget or not isinstance(budget, dict):
        return {"success": False, "error": "budget must be a dict of tool_name -> count."}

    current_budget = _mcp._STATE.active_intent.get("budget", {})

    for tool_name, count in budget.items():
        try:
            n = int(count)
            if n < 1:
                return {"success": False, "error": f"Extension for '{tool_name}' must be >= 1"}
            current_budget[tool_name] = current_budget.get(tool_name, 0) + n
        except (TypeError, ValueError):
            return {
                "success": False,
                "error": f"Extension for '{tool_name}' must be int, got {count!r}",
            }

    _mcp._STATE.active_intent["budget"] = current_budget
    _persist_active_intent()  # Sync to disk for hook

    used = _mcp._STATE.active_intent.get("used", {})
    remaining = {k: current_budget.get(k, 0) - used.get(k, 0) for k in current_budget}

    return {
        "success": True,
        "budget": current_budget,
        "used": used,
        "remaining": remaining,
    }


# ───────────────────────────────────────────────────────────────────────
# Operation-level declaration (2026-04-20)
# ───────────────────────────────────────────────────────────────────────
# Per-tool-call cue declared explicitly by the agent, replacing the
# auto-built cue-from-tool-args that the PreToolUse hook historically
# used. Motivation: 2026-04-20 empirical audit showed ~58% of surfaced
# memories during a normal working session were pure noise, driven by
# the fact that generic cues like "run pytest" / "edit test_file" / "read
# line range" have no topic anchor -- nearest-neighbor returns whatever
# past traces also ran pytest / edited tests / read files, regardless of
# topic. Agent-declared queries+keywords raise cue specificity to the
# same bar as declare_intent's Context fingerprint and align with AAO
# (Activity-Action-Operation) hierarchy: the intent is the Activity, the
# tool call is the Operation, and this is where the Operation cue lives.
#
# Retrieval reuses hooks_cli._run_local_retrieval (same multi-view cosine
# + keyword channel + RRF + dedup pipeline the PreToolUse hook already
# uses -- no new scoring code). The hook is then responsible for consuming
# the pending_operation_cue and emitting the injected memories as
# additionalContext; see hooks_cli.hook_pretooluse for the consumer.
#
# Enforcement is gated by env MEMPALACE_REQUIRE_DECLARE_OPERATION. When
# off (default during rollout), missing cues fall back to the legacy
# auto-build path so existing sessions don't break. When on, missing
# cues cause the hook to deny the tool call with a recipe. Flip this on
# only after telemetry shows the agent reliably declares.

MIN_OP_QUERIES = 2
MAX_OP_QUERIES = 5
MIN_OP_KEYWORDS = 2
MAX_OP_KEYWORDS = 5
# Mandatory under link-author: every operation lists the entities it
# touches (files it'll read, services it reasons about, agents involved,
# etc.). Capped to keep abuse + the candidate-upsert fanout bounded.
MIN_OP_ENTITIES = 1
MAX_OP_ENTITIES = 10
OP_CUE_TOP_K = 5  # same cap as PreToolUse retrieval today


def _record_op_recall_diagnostic(op_context_id: str, populated: bool) -> None:
    """2026-04-26: track op-recall hit/miss per declare_operation call.

    Added per Adrian's directive after the ops-recall audit
    (``audit_operations_recall_end_to_end_2026_04_26``). The
    ``past_operations`` bucket is silently omitted from the response
    when the walker comes back with nothing, which masks two distinct
    failure modes the operator needs to tell apart:

      (a) graph genuinely has no rated ops in this context's
          ``similar_to`` neighbourhood -- fine, will warm up;
      (b) the walker keeps returning empty across many calls because
          contexts aren't getting ``similar_to`` edges
          (T_similar=0.70, MaxSim averaged across views, only top-1
          candidate gets the edge -- see the audit memo).

    Without a counter, (b) is invisible. Per-call DEBUG log plus a
    session-level counter on ``_STATE.session_state`` lets operators
    query "how often was past_operations empty / populated this
    session" without spamming production logs. Fire-and-forget:
    every branch swallows exceptions so a metrics failure cannot
    break the declare_operation response.
    """
    key = "op_recall_populated_count" if populated else "op_recall_empty_count"
    try:
        ss = _mcp._STATE.session_state
        ss[key] = int(ss.get(key, 0)) + 1
    except Exception:
        pass
    if not populated:
        try:
            import logging as _ops_log

            _ops_log.getLogger(__name__).debug(
                "past_operations empty for ctx=%s (no good/avoid in similar_to neighbourhood)",
                op_context_id,
            )
        except Exception:
            pass


def _emit_op_cluster_flags(past_ops: dict, op_context_id: str, kg) -> None:
    """S3a: detect same-tool same-sign clusters in past_operations and
    persist them as ``op_cluster_templatizable`` memory_flags rows.

    Split out of ``tool_declare_operation`` to keep that function below
    the ruff C901 complexity budget and to give the emission path its
    own testable seam. Fire-and-forget: swallows every exception so
    retrieval errors never break the declare_operation response.
    """
    try:
        from .scoring import detect_op_cluster_flags as _detect_clusters

        flags = _detect_clusters(past_ops)
        if not flags:
            return
        for flag in flags:
            flag["context_id"] = op_context_id
        kg.record_memory_flags(flags)
    except Exception:
        # S3a is advisory -- a failure here must not propagate. The
        # gardener simply won't get this cluster flag; it will fire
        # again next time the same cluster re-surfaces.
        pass


def _enrich_memories_with_state(memories: list, kg) -> dict:
    """State-protocol v1 (Adrian 2026-05-03): for each surfaced
    memory whose entity is_a a state-bearing class, attach the entity's
    current_state + state_schema_id to the memory dict in place. Without
    this enrichment agents have no way to author meaningful state_deltas
    -- they cannot diff against an unknown current state. Cost is one
    is_a lookup + one state_schema_id check + one latest_state read per
    memory; small in practice because most retrievals surface 5-10 items.

    Per-memory carries the lean state_schema_id string + current_state
    only. The schema definitions themselves (json_schema +
    slot_descriptions + parent_schema_id) come back as the RETURN value:
    a dict {schema_id -> schema_def} containing only the schemas
    referenced by this memory list. Callers attach this dict to their
    top-level response under the key "schemas" so each schema is sent
    at most once per response, not once per memory (Adrian token-budget
    directive 2026-05-03).

    Mutates the memories list in place. Failures are silent (per-memory
    try/except) so a bug in one row never breaks the response. Returns
    an empty dict when no state-bearing memory is enriched.
    """
    if not memories or kg is None:
        return {}
    try:
        conn = kg._conn()
    except Exception:
        return {}
    # Discover state-bearing classes once (small set: Task, agent,
    # intent_type today). LIKE search on properties JSON is acceptable
    # because the table is small and this runs at most once per
    # declare_operation / declare_intent call.
    try:
        rows = conn.execute(
            "SELECT name, properties FROM entities WHERE kind='class' "
            "AND properties LIKE '%\"state_updatable\": true%'"
        ).fetchall()
    except Exception:
        return {}
    if not rows:
        return {}
    import json as _json

    state_class_to_schema: dict = {}
    for _row in rows:
        try:
            _props = _json.loads(_row[1] or "{}")
            _sid = _props.get("state_schema_id") if isinstance(_props, dict) else None
            if isinstance(_sid, str) and _sid:
                state_class_to_schema[_row[0]] = _sid
        except Exception:
            continue
    if not state_class_to_schema:
        return {}
    # The triples table stores normalized (lowercase) entity ids on both
    # sides, so the SQL JOIN must compare normalized class names too --
    # raw "Task" never matches normalized "task". Build a normalized->
    # schema_id map; preserve raw->schema for direct-match fallback.
    norm_class_to_schema = {kg._entity_id(name): sid for name, sid in state_class_to_schema.items()}
    norm_class_names = list(norm_class_to_schema.keys())
    placeholders = ",".join("?" * len(norm_class_names))
    referenced: set = set()
    for entry in memories:
        eid = entry.get("id")
        if not eid:
            continue
        try:
            norm = kg._entity_id(str(eid))
            # (Adrian directive 2026-05-05): the prior
            # direct-class match branch was retired. Classes themselves
            # carry no instance state, so tagging them with
            # state_schema_id + current_state=null on every memory
            # surface was pure noise -- the schemas catalog already
            # lives in wake_up.schemas, so the per-memory enrichment
            # for classes added zero information AND distracted agents
            # by suggesting they should author state_deltas for
            # surfaced classes (the bug slice 5 + slice 5-followon
            # already fixed in coverage; this fix removes the same
            # confusion from the read surface). Only kind=entity
            # instances get enrichment now -- they have real
            # current_state to compare against and a real schema_id
            # to author patches against. The is_a walk below catches
            # all instances that are state-bearing.
            # Walk is_a edges to find the entity's class. Single hop;
            # transitive class chains are rare in practice and would
            # need a recursive CTE; v1 keeps it simple.
            row = conn.execute(
                "SELECT object FROM triples "
                "WHERE subject=? AND predicate='is_a' "
                f"AND object IN ({placeholders}) "
                "AND valid_to IS NULL LIMIT 1",
                (norm, *norm_class_names),
            ).fetchone()
            if not row:
                continue
            schema_id = norm_class_to_schema.get(row[0])
            if not schema_id:
                continue
            # only enrich kind='entity' instances; classes
            # that happen to have an is_a edge to a state-bearing class
            # (e.g. 'inspect' is_a 'intent_type') are themselves classes
            # and carry no instance state. Same kind-filter as the slice
            # 5 follow-on coverage block; the two enforcement axes share
            # one rule so class entries stay clean across both write
            # (coverage) and read (enrichment) paths.
            try:
                _kind_row = conn.execute(
                    "SELECT kind FROM entities WHERE id=? LIMIT 1", (norm,)
                ).fetchone()
                if not _kind_row or (_kind_row[0] or "") != "entity":
                    continue
            except Exception:
                continue
            entry["state_schema_id"] = schema_id
            referenced.add(schema_id)
            try:
                cur = kg.latest_state_for_entity(eid)
                # current_state may be None when no revisions exist yet;
                # surface explicitly so agents know to declare a delta
                # (or rely on retrofit gardener default).
                entry["current_state"] = cur
            except Exception:
                pass
        except Exception:
            continue
    # Build the referenced-schemas dict for the caller to attach to its
    # top-level response. Each schema (json_schema + slot_descriptions +
    # parent_schema_id) is included exactly once even if N memories
    # share it, keeping per-response tokens bounded by the schema count
    # (4 in v1) regardless of K (memories surfaced).
    schemas_out: dict = {}
    if referenced:
        try:
            from . import state_schemas as _ss

            for sid in referenced:
                _sd = _ss.STATE_SCHEMAS.get(sid)
                if _sd:
                    schemas_out[sid] = {
                        "json_schema": _sd.get("json_schema"),
                        "slot_descriptions": _sd.get("slot_descriptions"),
                        "parent_schema_id": _sd.get("parent_schema_id"),
                    }
        except Exception:
            pass
    return schemas_out


def tool_declare_operation(  # noqa: C901
    tool: str,
    args_summary: str = None,
    context: dict = None,
    agent: str = None,
    state_deltas: list = None,
    slots: dict = None,  # slice 12: per-tool operation_class slot schema
):
    """Declare the operation (tool call) you are about to perform.

    Mandatory pre-step for every non-carve-out tool call under the
    2026-04-20 cue-quality redesign. The cue you provide drives the
    same retrieval pipeline the PreToolUse hook uses today; memories are
    returned here and the hook also surfaces them as additionalContext
    when the real tool call fires (one-turn lag, identical to today).

    Unified Context shape (same as declare_intent / kg_search / kg_add /
    kg_declare_entity / kg_add_batch -- ONE shape for every emit site):

        context = {
          "queries":  [2-5 natural-language perspectives],
          "keywords": [2-5 exact domain terms],
          "entities": [1-10 entity ids the operation touches],
        }

    Args:
        tool: Name of the tool you are about to call (e.g. 'Read', 'Grep',
              'Bash', 'Edit'). Must be permitted under the active intent.
        context: Mandatory unified Context dict. See shape above. Validated
                 by ``scoring.validate_context`` -- same validator every
                 other emit site uses, same error messages, same bounds.
        agent: Your agent name.

    Returns:
        {"success": true, "memories": [...], "feedback_reminder": "..."}
        on success. Surfaced memory ids land in accessed_memory_ids and
        on the contexts_touched_detail entry for this op so the
        post-finalize async-Haiku rater (mempalace.feedback_auto) can
        ship a per-op rating batch for them.

    Carve-outs: mempalace_* tools and the ALWAYS_ALLOWED set in
    hooks_cli (TodoWrite, Skill, Agent, ToolSearch, AskUserQuestion,
    Task*, ExitPlanMode) do NOT need declare_operation -- they skip
    retrieval entirely. Attempting to declare an operation for one of
    those returns an informative error.
    """
    sid_err = _mcp._require_sid(action="declare_operation")
    if sid_err:
        return sid_err
    _sync_from_disk()
    if not _mcp._STATE.active_intent:
        return {
            "success": False,
            "error": (
                "No active intent. Call mempalace_declare_intent first. "
                "Operation-level declarations live under an Activity-level "
                "intent -- you cannot declare an operation with no intent."
            ),
        }

    agent_err = _mcp._require_agent(agent, action="declare_operation")
    if agent_err:
        return agent_err

    # ── Validate tool name ──
    if not isinstance(tool, str) or not tool.strip():
        return {"success": False, "error": "tool must be a non-empty string."}
    tool = tool.strip()

    # Carve-outs: mempalace_* and ALWAYS_ALLOWED skip retrieval, so
    # declaring an operation for them is a no-op at best and confusing
    # at worst. Teach the agent directly.
    try:
        from . import hooks_cli as _hc_mod

        always_allowed = _hc_mod.ALWAYS_ALLOWED_TOOLS
    except Exception:
        always_allowed = set()
    is_mempalace_mcp = tool.startswith("mcp__") and "__mempalace_" in tool
    if tool in always_allowed or is_mempalace_mcp:
        return {
            "success": False,
            "error": (
                f"Tool '{tool}' does not require declare_operation. "
                "mempalace_* tools and ALWAYS_ALLOWED tools (TodoWrite, "
                "Skill, Agent, ToolSearch, AskUserQuestion, Task*, "
                "ExitPlanMode) skip PreToolUse retrieval -- just call "
                "them directly."
            ),
        }

    # ── Validate args_summary (mandatory, parametrized-core form) ──
    # 2026-04-27 redesign: args_summary moved from optional rating-side
    # field to mandatory declare-time field. Two ops sharing the same
    # parametrized args_summary cluster as the SAME operation in the
    # past_operations neighbourhood walk and the gardener S3a templatize
    # detector -- so the fingerprint must capture INTENT, not literal
    # text. See the schema description for parametrization examples.
    if not isinstance(args_summary, str) or not args_summary.strip():
        return {
            "success": False,
            "error": (
                "args_summary is required (string, 5-400 chars). It is the "
                "PARAMETRIZED CORE of the operation -- invariant shape with "
                "per-execution variables abstracted as {placeholders}. "
                "Examples:\n"
                " Bad: 'git commit -m \"feat: ship gate\"'\n"
                "  Good: 'git commit -m \"{commit_message}\"'\n"
                "  Bad:  'python -m pytest tests/test_intent.py -q'\n"
                "  Good: 'python -m pytest {test_path} -q'\n"
                "Strip plumbing (cd, env vars, redirects). Two ops with "
                "the same args_summary string cluster as the same operation."
            ),
        }
    args_summary = args_summary.strip()
    if len(args_summary) < 5:
        return {
            "success": False,
            "error": (
                f"args_summary too short ({len(args_summary)} chars; "
                f"minimum 5). It must be a parametrized-core fingerprint, "
                f"not a one-word label."
            ),
        }
    if len(args_summary) > 400:
        return {
            "success": False,
            "error": (
                f"args_summary too long ({len(args_summary)} chars; "
                f"maximum 400). Compress to the parametrized core; long "
                f"literal strings defeat the cluster-matching purpose."
            ),
        }

    # ── Validate Context -- same shared validator every emit site uses ──
    # Bounds (MIN_OP_QUERIES etc.) are passed explicitly so module-level
    # constants stay the authoritative source-of-truth the schema + tests
    # can reference. Matches declare_intent / kg_search / kg_add / etc.
    from .scoring import validate_context as _validate_context

    clean_context, ctx_err = _validate_context(
        context,
        queries_min=MIN_OP_QUERIES,
        queries_max=MAX_OP_QUERIES,
        keywords_min=MIN_OP_KEYWORDS,
        keywords_max=MAX_OP_KEYWORDS,
        entities_min=MIN_OP_ENTITIES,
        entities_max=MAX_OP_ENTITIES,
        require_summary=True,
        summary_context_for_error="declare_operation.context.summary",
    )
    if ctx_err:
        return ctx_err
    queries = clean_context["queries"]
    keywords = clean_context["keywords"]
    entities = clean_context["entities"]

    # ── (Adrian directive 2026-05-05): operation slot validation ──
    # If an operation_class is registered for this tool (a kind='class'
    # entity is_a operation, with properties.rules_profile.tool == this
    # tool), enforce its slot schema the same way declare_intent does
    # for intent_types. The resolved slot entities get attached to the
    # pending_operation_cue so the operation entity (minted later by
    # promotion) carries entity-level provenance, not just an
    # args_summary fingerprint. Tools without a registered class skip
    # this block (back-compat -- existing test suite predates slice 12).
    op_slot_schema = _resolve_operation_profile(tool)
    resolved_op_slots: dict = {}
    if op_slot_schema:
        if slots is None:
            slots = {}
        if not isinstance(slots, dict):
            return {
                "success": False,
                "error": (
                    "slots must be a JSON object/dict for an operation "
                    "with a registered operation_class. "
                    f"tool '{tool}' expects: {list(op_slot_schema.keys())}."
                ),
            }
        slot_errors = []
        # Required-slot presence check
        for sname, sdef in op_slot_schema.items():
            if sdef.get("required", False) and sname not in slots:
                slot_errors.append(
                    f"Required slot '{sname}' not provided. "
                    f"Accepted classes: {sdef.get('classes', ['thing'])}."
                )
        # Per-slot validation
        for sname, svals in slots.items():
            if sname not in op_slot_schema:
                slot_errors.append(
                    f"Unknown slot '{sname}'. Valid slots: {list(op_slot_schema.keys())}."
                )
                continue
            sdef = op_slot_schema[sname]
            if isinstance(svals, str):
                svals = [svals]
            if not isinstance(svals, list):
                slot_errors.append(f"Slot '{sname}' must be a string or list of strings.")
                continue
            # design lock (Adrian): operation slots default to
            # multiple=false. Most operations touch one entity at a
            # time; an op needing two files is two separate ops.
            if not sdef.get("multiple", False) and len(svals) > 1:
                slot_errors.append(
                    f"Slot '{sname}' accepts only one entity (multiple=false), got {len(svals)}."
                )
                continue
            if sdef.get("raw", False):
                resolved_op_slots[sname] = list(svals)
                continue
            allowed_classes = sdef.get("classes", ["thing"])
            is_file_slot = "file" in allowed_classes
            normalized = []
            for val in svals:
                if is_file_slot:
                    val_id = normalize_entity_name(os.path.basename(val))
                else:
                    val_id = normalize_entity_name(val)
                if not _mcp._is_declared(val_id):
                    slot_errors.append(
                        f"Entity '{val_id}' in slot '{sname}' not declared. "
                        f"Call mempalace_kg_declare_entity first "
                        f"(file slots: pass kind='entity', is_a='file')."
                    )
                    continue
                # Class constraint -- mirror declare_intent's check
                if "thing" not in allowed_classes:
                    try:
                        edges = _mcp._STATE.kg.query_entity(val_id, direction="outgoing")
                    except Exception:
                        edges = []
                    entity_classes = [
                        e["object"] for e in edges if e["predicate"] == "is_a" and e["current"]
                    ]
                    if entity_classes:
                        norm_ec = [normalize_entity_name(c) for c in entity_classes]
                        norm_allowed = [normalize_entity_name(c) for c in allowed_classes]
                        if not any(c in norm_allowed for c in norm_ec):
                            slot_errors.append(
                                f"Entity '{val_id}' in slot '{sname}' is-a "
                                f"{entity_classes}, but slot requires "
                                f"classes {allowed_classes}."
                            )
                            continue
                normalized.append(val_id)
            if normalized:
                resolved_op_slots[sname] = normalized
        if slot_errors:
            return {
                "success": False,
                "error": "Slot validation failed for declare_operation.",
                "slot_issues": slot_errors,
                "expected_slots": {
                    name: {
                        "classes": d.get("classes", ["thing"]),
                        "required": d.get("required", False),
                        "multiple": d.get("multiple", False),
                    }
                    for name, d in op_slot_schema.items()
                },
            }

    # ── Run retrieval via the SAME pipeline the hook uses today ──
    # _run_local_retrieval handles lazy Chroma import, dedup against
    # accessed_memory_ids, top-K cap, timeout, fail-loud error recording.
    # Reusing it keeps scoring.multi_channel_search the single source of
    # truth for cue → ranked memories.
    from . import hooks_cli as _hc

    cue = {"queries": [q.strip() for q in queries], "keywords": [k.strip() for k in keywords]}
    # Dedup filter: every memory surfaced so far in this intent must be
    # excluded from operation-time retrieval. Two lists carry those ids:
    # accessed_memory_ids (populated by declare_operation and kg_search)
    # and injected_memory_ids (populated by declare_intent). The finalize
    # coverage validator treats them separately so the two must remain
    # distinct for rating purposes, but for "already shown" they are the
    # same signal and must be unioned here. Without the union,
    # declare_operation re-surfaces whatever declare_intent already showed.
    accessed = set(_mcp._STATE.active_intent.get("accessed_memory_ids") or []) | set(
        _mcp._STATE.active_intent.get("injected_memory_ids") or []
    )
    try:
        hits, notice = _hc._run_local_retrieval(cue, accessed, OP_CUE_TOP_K)
    except Exception as _e:
        hits, notice = [], {"fn": "_run_local_retrieval", "error": repr(_e)}

    # ── Context as first-class entity (P1) ──
    # declare_operation is an emit site. A fresh operation cue gets its
    # own context entity: future operations whose cue is MaxSim-similar
    # reuse it. The stored context is the "operation flavour" of the
    # active intent's context -- they may be similar (both pertain to the
    # same task) but diverge enough to merit their own accretion.
    # The returned id becomes this operation's active_context_id for any
    # writes that happen during the triggered tool call. We stash it on
    # the pending cue so the hook can later advertise it.
    _op_context_id = ""
    _op_context_reused = False
    try:
        _cid, _reused, _ms = _mcp.context_lookup_or_create(
            queries=cue["queries"],
            keywords=cue["keywords"],
            entities=entities,
            agent=agent or _mcp._STATE.active_intent.get("agent", ""),
            summary=clean_context.get("summary"),
        )
        _op_context_id = _cid or ""
        _op_context_reused = bool(_reused)
    except Exception:
        _op_context_id = ""
    # Most-recent-emit precedence: a declare_operation supersedes the
    # intent-level context for any writes that fire between now and the
    # next emit (intent switch, next operation, kg_search).
    if _op_context_id:
        _mcp._STATE.active_intent["active_context_id"] = _op_context_id
        _mcp._record_context_emit(
            _op_context_id,
            reused=_op_context_reused,
            scope="operation",
            queries=cue["queries"],
            keywords=cue["keywords"],
            entities=entities,
            surfaced_ids=[h.get("id") for h in hits if h.get("id")],
        )

    # ── Persist pending_operation_cues (append) + accessed_memory_ids ──
    # The hook pops the first matching-tool entry on the next real tool
    # call, uses it as the retrieval cue (replacing the legacy heuristic
    # cue build), then writes the shortened list back. List form supports
    # parallel tool dispatch: agent can declare N operations in one
    # message and the subsequent N tool calls each consume their own cue.
    # Each cue carries declared_at_ts; the hook expires entries older than
    # OPERATION_CUE_TTL_SECONDS on consume so a forgotten declaration
    # doesn't poison future tool calls indefinitely.
    new_cue = {
        "tool": tool,
        "args_summary": args_summary,
        "queries": cue["queries"],
        "keywords": cue["keywords"],
        "declared_at_ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "surfaced_ids": [h.get("id") for h in hits if h.get("id")],
        # P1: context entity id minted for this operation cue. Writers
        # that fire while this cue is the most-recent one use it as
        # active_context_id.
        "active_context_id": _op_context_id,
        # (Adrian directive 2026-05-05): per-slot entity ids
        # validated against the tool's operation_class. Empty dict when
        # the tool has no operation_class registered (back-compat). The
        # gardener / promotion path reads this to anchor the operation
        # entity to the files/entities it actually touched.
        "resolved_slots": resolved_op_slots,
    }
    # 2026-04-27 redesign: persist args_summary on the active intent
    # under (context_id, tool) so the finalize_intent / extend_feedback
    # promotion path can fetch the parametrized-core fingerprint instead
    # of the (now-removed) rating-side field. Last-write-wins per
    # (ctx_id, tool); rare in practice because two ops with the same
    # context_id are usually the same operation by design.
    if _op_context_id:
        _op_args_store = _mcp._STATE.active_intent.get("op_args_by_ctx_tool")
        if not isinstance(_op_args_store, dict):
            _op_args_store = {}
            _mcp._STATE.active_intent["op_args_by_ctx_tool"] = _op_args_store
        _op_args_store[f"{_op_context_id}|{tool}"] = args_summary
    existing_cues = _mcp._STATE.active_intent.get("pending_operation_cues") or []
    if not isinstance(existing_cues, list):
        existing_cues = []
    _mcp._STATE.active_intent["pending_operation_cues"] = existing_cues + [new_cue]
    # Single-list design (2026-04-23 decision): every memory surfaced
    # by a declare_operation is added to accessed_memory_ids. It now
    # participates both in within-intent dedup (filter on retrieval)
    # and finalize coverage (the agent must rate it). Sessions with
    # many operation cues will demand many ratings at finalize; that
    # is expected. Background-jury rating is tracked as a separate
    # TODO for later consideration.
    # State-protocol v1 (Adrian Option B 2026-05-03): accept
    # caller-provided state_deltas list. Each entry is a dict
    # {entity_id, status, patch?, justification?} where status is
    # 'changed' (with JSON Patch list in `patch`), 'unchanged', or
    # 'irrelevant'. Persistence on active_intent: state_deltas_by_op
    # accumulates per-op deltas under the operation context_id;
    # irrelevant_state_set accumulates instances marked irrelevant
    # (their coverage requirement is relieved for the rest of the
    # intent per the v2 design lock irrelevant-relief rule).
    # state_deltas_entity_set accumulates entity_ids that have at least
    # one delta entry (any status) so the finalize coverage check can
    # subtract them from expected_state_ids.
    # enforcement (separate edit at intent.py:4165-4234)
    # rejects ops missing state_deltas for surfaced state-bearing
    # memories. The kill-switch env MEMPALACE_STATE_DELTA_DISABLED=1
    # disables the enforcement layer; this plumbing layer always runs
    # so deltas can be observed in tests + telemetry even with kill
    # switch on.
    # gate B leak fix (Adrian directive 2026-05-05): ALWAYS
    # initialize _validated_deltas / _delta_entity_set / _new_irrelevant
    # even when the caller omitted state_deltas. Pre-fix these only
    # existed inside the `if state_deltas is not None` branch, so a
    # caller passing state_deltas=None landed in the per-op gate with
    # _delta_entity_set undefined; the gate's surrounding try/except
    # caught the NameError silently and skipped enforcement entirely.
    # Adrian's call-out tonight ('how were you ABLE to do it??') was
    # exactly this leak: I called declare_operation many times without
    # state_deltas and was never blocked, because the gate crashed and
    # was swallowed before it could fail-loud.
    _validated_deltas: list = []
    _delta_entity_set: set = set()
    _new_irrelevant: set = set()
    if state_deltas is not None:
        if not isinstance(state_deltas, list):
            return {
                "success": False,
                "error": (
                    f"state_deltas must be a list of dicts; got "
                    f"{type(state_deltas).__name__}. Each entry: "
                    "{entity_id: str, status: 'changed'|'unchanged', "
                    "patch?: list[JSONPatchOp] (RFC 6902, required iff "
                    "status=='changed'), justification?: str}."
                ),
            }
        for _i, _d in enumerate(state_deltas):
            if not isinstance(_d, dict):
                return {
                    "success": False,
                    "error": f"state_deltas[{_i}] must be a dict; got {type(_d).__name__}.",
                }
            _eid = (_d.get("entity_id") or "").strip()
            _status = (_d.get("status") or "").strip()
            if not _eid:
                return {
                    "success": False,
                    "error": f"state_deltas[{_i}].entity_id is required.",
                }
            if _status not in ("changed", "unchanged"):
                return {
                    "success": False,
                    "error": (
                        f"state_deltas[{_i}].status must be 'changed' "
                        f"or 'unchanged'; got {_status!r}."
                    ),
                }
            # conflict rejection (Adrian corner-case audit
            # 2026-05-03): refuse 'irrelevant' after a prior 'changed'
            # delta in this intent. The 'changed' delta wrote a durable
            # state revision; marking it irrelevant now would contradict
            # history (revision exists but coverage no longer requires
            # it). Other transitions are allowed -- agents may
            # legitimately change their mind once. irrelevant->changed
            # also clears the entity from irrelevant_state_set so
            # coverage requires it again.
            _status_map = _mcp._STATE.active_intent.get("state_delta_status_per_entity")
            if not isinstance(_status_map, dict):
                _status_map = {}
            _prior_status = _status_map.get(_eid)
            if _prior_status == "changed" and _status == "irrelevant":
                return {
                    "success": False,
                    "error": (
                        f"state_deltas[{_i}] for entity_id={_eid!r}: "
                        f"cannot mark 'irrelevant' after a prior "
                        f"'changed' delta in this intent. The 'changed' "
                        f"delta wrote a durable state revision; marking "
                        f"it irrelevant now would contradict history. "
                        f"Mark 'changed' again for further revisions, "
                        f"or leave the entity in coverage."
                    ),
                }
            _status_map[_eid] = _status
            _mcp._STATE.active_intent["state_delta_status_per_entity"] = _status_map
            if _prior_status == "irrelevant" and _status != "irrelevant":
                # Transitioning out of 'irrelevant'; remove from the
                # irrelevant set so coverage requires this entity again.
                _irr_clear = _mcp._STATE.active_intent.get("irrelevant_state_set")
                if isinstance(_irr_clear, set):
                    _irr_clear.discard(_eid)
                elif isinstance(_irr_clear, (list, tuple)):
                    _mcp._STATE.active_intent["irrelevant_state_set"] = {
                        _x for _x in _irr_clear if _x != _eid
                    }
            if _status == "changed":
                _patch = _d.get("patch")
                if not isinstance(_patch, list) or not _patch:
                    return {
                        "success": False,
                        "error": (
                            f"state_deltas[{_i}].patch is required as a "
                            "non-empty JSON Patch list (RFC 6902) when "
                            "status='changed'."
                        ),
                    }
            # Adrian directive 2026-05-11: justification on
            # status='unchanged' is ALLOWED and REQUIRED when overriding
            # a state_judge flag. Earlier (follow-up #2 2026-05-05)
            # this was hard-failed to stop boilerplate spam, but the
            # ban contradicted the judge-override error message which
            # explicitly tells agents to use unchanged+justification.
            # Result: agents that genuinely overrode the judge had no
            # legal path -- they fell back to bare 'unchanged' which
            # erased the audit trail. Field is optional for routine
            # acks; gate hook + gardener can flag boilerplate spam at
            # post-hoc analysis time rather than blocking writes.
            _justification_in = _d.get("justification")
            _validated_deltas.append(
                {
                    "entity_id": _eid,
                    "status": _status,
                    "patch": _d.get("patch"),
                    "justification": _justification_in,
                }
            )
            _delta_entity_set.add(_eid)
            if _status == "irrelevant":
                _new_irrelevant.add(_eid)
            # (Adrian 2026-05-03): on status=changed, apply
            # the RFC 6902 patch to the entity's latest state and write
            # a new revision via kg.record_state_revision. The
            # op_context_id ties the JTMS column to the operation that
            # caused the change. Schema validation + durable write live
            # here so deltas don't sit in active_intent state without
            # ever becoming history.
            if _status == "changed" and _mcp._STATE.kg is not None:
                try:
                    import jsonpatch as _jp

                    _current = _mcp._STATE.kg.latest_state_for_entity(_eid) or {}
                    _new_payload = _jp.apply_patch(_current, _d.get("patch"))
                except ImportError:
                    return {
                        "success": False,
                        "error": (
                            "jsonpatch library required for state_deltas "
                            "with status='changed'. pip install jsonpatch."
                        ),
                    }
                except Exception as _patch_err:
                    return {
                        "success": False,
                        "error": (
                            f"state_deltas[{_i}].patch failed to apply to "
                            f"current state of {_eid!r}: {_patch_err}. "
                            "Patch must be valid RFC 6902 ops against the "
                            "schema's slot dict."
                        ),
                    }
                # Schema validation: best-effort. Resolve schema_id via
                # the entity's class state_schema_id property; if any
                # resolution step fails, fall through to write without
                # validation -- the durable record is still valuable.
                try:
                    from . import state_schemas as _ss

                    # Walk is_a edge to find class; read class.properties.
                    _conn_v = _mcp._STATE.kg._conn()
                    _cls_row = _conn_v.execute(
                        "SELECT object FROM triples WHERE subject=? "
                        "AND predicate='is_a' AND valid_to IS NULL LIMIT 1",
                        (normalize_entity_name(_eid),),
                    ).fetchone()
                    _schema_id = None
                    if _cls_row:
                        _props_row = _conn_v.execute(
                            "SELECT properties FROM entities WHERE name=?",
                            (_cls_row[0],),
                        ).fetchone()
                        if _props_row and _props_row[0]:
                            try:
                                _props = json.loads(_props_row[0])
                                _schema_id = (
                                    _props.get("state_schema_id")
                                    if isinstance(_props, dict)
                                    else None
                                )
                            except Exception:
                                pass
                    if _schema_id and _schema_id in _ss.STATE_SCHEMAS:
                        try:
                            import jsonschema as _js

                            _js.validate(
                                _new_payload,
                                _ss.STATE_SCHEMAS[_schema_id]["json_schema"],
                            )
                        except ImportError:
                            pass  # jsonschema optional in v1
                        except Exception as _val_err:
                            return {
                                "success": False,
                                "error": (
                                    f"state_deltas[{_i}] resulting payload "
                                    f"failed JSON Schema validation against "
                                    f"{_schema_id}: {_val_err}"
                                ),
                            }
                except Exception:
                    pass
                # Durable write: schema_id from class properties (or
                # empty string if we couldn't resolve it -- the row
                # still records the entity + payload + op_context_id
                # which is enough for retraction sweeps).
                try:
                    _mcp._STATE.kg.record_state_revision(
                        entity_id=_eid,
                        schema_id=_schema_id or "",
                        payload=_new_payload,
                        op_context_id=_op_context_id or "",
                        agent=agent or "",
                    )
                except Exception:  # pragma: no cover -- defensive
                    pass
        # Persist onto active_intent.
        _delta_log = _mcp._STATE.active_intent.get("state_deltas_by_op")
        if not isinstance(_delta_log, list):
            _delta_log = []
        _delta_log.append(
            {
                "op_context_id": _op_context_id or "",
                "tool": tool,
                "deltas": _validated_deltas,
            }
        )
        _mcp._STATE.active_intent["state_deltas_by_op"] = _delta_log
        _delta_set = _mcp._STATE.active_intent.get("state_deltas_entity_set")
        if not isinstance(_delta_set, set):
            _delta_set = set(_delta_set or [])
        _delta_set.update(_delta_entity_set)
        _mcp._STATE.active_intent["state_deltas_entity_set"] = _delta_set
        if _new_irrelevant:
            _irr_set = _mcp._STATE.active_intent.get("irrelevant_state_set")
            if not isinstance(_irr_set, set):
                _irr_set = set(_irr_set or [])
            _irr_set.update(_new_irrelevant)
            _mcp._STATE.active_intent["irrelevant_state_set"] = _irr_set

    _new_op_ids = [h.get("id") for h in hits if h.get("id")]
    if _new_op_ids:
        # v3.5.0 (2026-05-14): no more pending_feedback limbo gate --
        # ids always land in accessed_memory_ids; the async-Haiku rater
        # (mempalace.feedback_auto) rates them post-finalize.
        _acc_set = _mcp._STATE.active_intent.get("accessed_memory_ids")
        if not isinstance(_acc_set, set):
            _acc_set = set(_acc_set or [])
        _acc_set.update(_new_op_ids)
        _mcp._STATE.active_intent["accessed_memory_ids"] = _acc_set

    # State-protocol v2 Phase C (Adrian 2026-05-04): per-op coverage
    # enforcement. Every state-bearing entity (Task/agent/intent_type
    # INSTANCE, not the class itself) surfaced by THIS declare_operation
    # call must be covered in the same call's state_deltas. The v1
    # design deferred coverage to finalize_intent, which let agents
    # accumulate uncovered entities across an intent and scramble at
    # finalize. v2 says: block here, force the agent to engage with
    # state at the moment of surface.
    #
    # Filtered to instances only (subject IN _norm_ids, predicate='is_a',
    # object IN state_classes) -- the class itself surfacing as an
    # accessed memory does NOT mean state is required (classes have no
    # state, only instances do). entities.status='active' filter as
    # defense-in-depth so deleted entities can't slip through.
    # Build memories list early so the parallel block below can pass
    # it to apply_gate. We build a NEW list (preserving the canonical
    # post-retrieval shape from line 3914 area) and let the later
    # build site become a no-op when _parallel_kicked is True.
    memories = []
    for h in hits:
        # v3.7.9: routed through the canonical _project_memory helper
        # so the entry shape matches every other memory-emission site
        # (declare_user_intents / kg_search / _attach_context_rank /
        # searcher). Pre-v3.7.9 this block hand-rolled the same
        # {summary_text, content, content_trimmed, content_redundant}
        # build inline -- one of 4 duplicated copies that diverged
        # over time.
        extras = {}
        if DEBUG_RETURN_SCORES:
            extras["hybrid_score"] = round(float(h.get("score", 0.0) or 0.0), 6)
        # v3.7.34: pass vec metadata sub-dict so _project_memory can
        # hoist date_added + last_relevant_at to the agent. hooks_cli.
        # _run_local_retrieval attaches h['meta'] post-v3.7.34.
        _h_meta = h.get("meta") or {}
        if _h_meta:
            extras["metadata"] = _h_meta
        entry = _project_memory(h["id"], (h.get("preview") or "").strip(), extras=extras)
        memories.append(entry)
    # State enrichment happens here too, before the parallel block,
    # so apply_gate sees the enriched memories. Result dict uses
    # _op_schemas downstream.
    _op_schemas: dict = {}
    try:
        _op_schemas = _enrich_memories_with_state(memories, _mcp._STATE.kg) or {}
    except Exception:
        pass

    # v3.5.0 (2026-05-14): pending_feedback limbo state retired; the
    # finalize-time apply_gate gate is always live now.
    _is_finalizing_now = False
    _state_delta_kill_switch_op = bool(os.environ.get("MEMPALACE_STATE_DELTA_DISABLED"))
    # v3.4.0 Phase 3 (Adrian directive 2026-05-13): the v2
    # deferred-write protocol is now the DEFAULT. v3.2.7-3.3.0 shipped
    # it under an opt-IN env flag (MEMPALACE_STATE_PROTOCOL=v2_visibility);
    # v3.4.0 flips that to opt-OUT. Set MEMPALACE_STATE_PROTOCOL=v0_strict
    # to bring back the original strict gates (missing_state_deltas raise,
    # unchanged_violations raise, _all_complete gating, extend_feedback
    # coverage requirement) for one release as a back-compat escape
    # hatch. The legacy v2_visibility value is also recognised as a
    # no-op for callers that haven't unset their env yet.
    _v0_strict = os.environ.get("MEMPALACE_STATE_PROTOCOL", "").strip().lower() == "v0_strict"
    # `_v2_visibility` retained as the inverted alias so the existing
    # gate-site conditions read naturally (`if X and not _v2_visibility:`
    # blocks under v0_strict; auto-apply fires when _v2_visibility is
    # True which is now the default).
    _v2_visibility = not _v0_strict
    # Hoisted so the success path can attach state_judge_report to
    # the response dict regardless of which branch fired below.
    _judge_report_perop = None
    _judge_changes_perop: list = []
    # ── follow-up (Adrian directive 2026-05-07): parallel
    # state-judge + apply_gate execution. Both calls hit Haiku; both
    # are I/O-bound; ThreadPoolExecutor with 2 workers fires them
    # concurrently. Total wall time = max(judge, gate) instead of
    # sum. We pre-compute all inputs here (cheap dict construction +
    # SQLite latest_state lookups), submit both, and gather. The
    # judge result feeds the gate B coverage check below; the
    # apply_gate result is used in place of the original sequential
    # call site (which becomes a no-op).
    _gate_status = None
    _gate_report = None
    _parallel_kicked = False
    if not _is_finalizing_now and not _state_delta_kill_switch_op and _mcp._STATE.kg is not None:
        try:
            # apply_gate inputs
            _op_combined_meta = {
                h["id"]: {
                    "source": ("triple" if str(h.get("id", "")).startswith("t_") else "memory"),
                    "doc": (h.get("preview") or "").strip(),
                    "similarity": float(h.get("score", 0.0) or 0.0),
                }
                for h in hits
                if h.get("id")
            }
            try:
                _ai_for_gate = _mcp._STATE.active_intent or {}
                _parent_intent = {
                    "intent_type": _ai_for_gate.get("intent_type"),
                    "subject": ", ".join(
                        (_ai_for_gate.get("slots", {}) or {}).get("subject", []) or []
                    ),
                    "query": (_ai_for_gate.get("description_views") or [""])[0],
                }
            except Exception:
                _parent_intent = None
            _primary_context_for_gate = {
                "source": "declare_operation",
                "queries": list(cue["queries"]),
                "keywords": list(cue["keywords"]),
                "entities": list(entities or []),
            }

            # judge inputs
            _agent_id_raw = _mcp._STATE.active_intent.get("agent") or ""
            _ctx_id_raw = (
                _mcp._STATE.active_intent.get("intent_context_id")
                or _mcp._STATE.active_intent.get("active_context_id")
                or ""
            )
            _followed: list = []
            if _agent_id_raw:
                try:
                    _ag_state = _mcp._STATE.kg.latest_state_for_entity(_agent_id_raw)
                except Exception:
                    _ag_state = None
                _followed.append(
                    {
                        "entity_id": _agent_id_raw,
                        "state_schema_id": "agent_state",
                        "current_state": _ag_state or {},
                    }
                )
            if _ctx_id_raw:
                try:
                    _cx_state = _mcp._STATE.kg.latest_state_for_entity(_ctx_id_raw)
                except Exception:
                    _cx_state = None
                _followed.append(
                    {
                        "entity_id": _ctx_id_raw,
                        "state_schema_id": "intent_state",
                        "current_state": _cx_state or {},
                    }
                )
            _intent_type_for_judge = _mcp._STATE.active_intent.get("intent_type") or "?"
            _intent_slots_for_judge = _mcp._STATE.active_intent.get("slots") or {}
            _transcript_perop = (
                f"intent_type: {_intent_type_for_judge}\n"
                f"slots: {_intent_slots_for_judge}\n"
                f"this op: tool={tool}, args_summary={args_summary!r}\n"
                f"cue.queries: {cue.get('queries', [])}\n"
            )

            from concurrent.futures import ThreadPoolExecutor as _TPE

            from .injection_gate import apply_gate as _apply_gate
            from .injection_gate import run_state_judge as _run_state_judge

            # v3.5.5 hang fix (Adrian directive 2026-05-15): cap each
            # future.result() at MEMPALACE_HAIKU_RESULT_TIMEOUT_SEC
            # (default 90s, slightly above the SDK's per-request
            # MEMPALACE_HAIKU_TIMEOUT_SEC=60 so the SDK fails first).
            # Pre-fix: agents stuck in declare_intent for hours when
            # Anthropic API stalled because both .result() calls had
            # NO timeout and would block indefinitely. On TimeoutError
            # we cancel the future and fail-open (gate keeps all
            # memories; judge returns empty changes) so the agent's
            # critical path never gets pinned by an upstream stall.
            try:
                _result_timeout_s = float(
                    os.environ.get("MEMPALACE_HAIKU_RESULT_TIMEOUT_SEC", "90")
                )
            except (TypeError, ValueError):
                _result_timeout_s = 90.0
            from concurrent.futures import TimeoutError as _FutTimeout

            # v3.7.4 Slice 3 (Adrian directive 2026-05-16, Option 3):
            # When MEMPALACE_BG_STATE_JUDGE=1 (default), the judge no
            # longer runs in the foreground parallel block. It is
            # spawned on _BG_STATE_JUDGE_EXECUTOR after the gate
            # result is bound; its findings land in the per-sid buffer
            # via _append_pending_state_updates and are surfaced on
            # the NEXT declare_operation response as
            # state_updates_since_last_op[]. _judge_changes_perop and
            # _judge_report_perop stay at their defaults ([] and None)
            # so the existing same-op coverage check at line ~4170
            # naturally short-circuits with no flagged entities (i.e.
            # the coverage check becomes a no-op when judge is bg,
            # equivalent to v2_visibility's soften path).
            #
            # When MEMPALACE_BG_STATE_JUDGE=0, the legacy foreground
            # parallel path runs as it did pre-v3.7.4 (back-compat).
            _bg_judge_active = _bg_state_judge_enabled()
            with _TPE(max_workers=2) as _executor:
                _gate_future = _executor.submit(
                    _apply_gate,
                    memories=memories,
                    combined_meta=_op_combined_meta,
                    primary_context=_primary_context_for_gate,
                    context_id=_op_context_id or "",
                    kg=_mcp._STATE.kg,
                    agent=agent,
                    parent_intent=_parent_intent,
                )
                _judge_future = None
                if not _bg_judge_active:
                    _judge_future = _executor.submit(
                        _run_state_judge,
                        transcript_text=_transcript_perop,
                        entity_states=_followed,
                        agent=agent,
                    )
                try:
                    _g_filtered, _gate_status, _gate_report = _gate_future.result(
                        timeout=_result_timeout_s
                    )
                    memories = _g_filtered
                except _FutTimeout:
                    # Hang detected: gate took longer than the wall-time
                    # budget. Cancel + fail-open so the agent proceeds
                    # with the unfiltered memory pool. Logged via
                    # state_judge_log telemetry side-channel (search
                    # 'gate_timeout' in feedback_auto telemetry path).
                    _gate_future.cancel()
                    try:
                        from .mcp_server import _telemetry_append_jsonl as _tel

                        _tel(
                            "gate_log.jsonl",
                            {
                                "event": "gate_timeout",
                                "timeout_s": _result_timeout_s,
                                "agent": agent or "",
                            },
                        )
                    except Exception:
                        pass
                except Exception:
                    # apply_gate fail-opens internally; this catches anything
                    # the executor itself raises. Memories pass through.
                    pass
                if _judge_future is not None:
                    try:
                        _judge_changes_perop, _judge_report_perop = _judge_future.result(
                            timeout=_result_timeout_s
                        )
                    except _FutTimeout:
                        _judge_future.cancel()
                        _judge_changes_perop = []
                        _judge_report_perop = None
                        try:
                            from .mcp_server import _telemetry_append_jsonl as _tel

                            _tel(
                                "state_judge_log.jsonl",
                                {
                                    "event": "judge_timeout",
                                    "timeout_s": _result_timeout_s,
                                    "agent": agent or "",
                                },
                            )
                        except Exception:
                            pass
                    except Exception:
                        _judge_changes_perop = []
                        _judge_report_perop = None
            _parallel_kicked = True

            # v3.7.4 Slice 3 bg spawn: after the foreground gate has
            # bound its result, fire the state_judge off to the bg
            # executor. Snapshot captures every input by value so a
            # later op cannot mutate them mid-run. The closure NEVER
            # raises; failures fall through silently (the bg log
            # telemetry can be added in a follow-up). On success, the
            # judge's (changes, report) tuple is appended to the
            # per-sid pending buffer via _append_pending_state_updates.
            if _bg_judge_active:
                _bg_sid = _mcp._STATE.session_id or ""
                _bg_transcript = _transcript_perop
                _bg_followed = list(_followed)
                _bg_agent = agent

                def _run_bg_judge():
                    try:
                        from .injection_gate import run_state_judge as _bg_run_judge

                        _bg_changes, _bg_report = _bg_run_judge(
                            transcript_text=_bg_transcript,
                            entity_states=_bg_followed,
                            agent=_bg_agent,
                        )
                    except Exception:
                        _bg_changes = []
                        _bg_report = None
                    # v3.7.23 FINDING #10 fix (Adrian directive 2026-05-18):
                    # the bg path must persist judge findings via
                    # record_state_revision, not just buffer them for
                    # next-op surfacing. Pre-fix the auto-apply lived
                    # only in the foreground branch at the
                    # state_changes_detected attach site, which never
                    # ran under bg-default because the foreground
                    # judge_future was None. Result: bg judge filled
                    # the pending buffer with findings + state_judge_log
                    # rows but landed zero rev_id writes; state never
                    # updated. Apply now mirrors the v3.7.20
                    # conflict_resolver_auto pattern: bg worker does
                    # Haiku + apply + log.
                    try:
                        _apply_judge_changes_to_state(
                            _bg_changes,
                            op_context_id="",
                            delta_covered=None,
                            session_id=_bg_sid or None,
                        )
                    except Exception:
                        pass
                    try:
                        _append_pending_state_updates(_bg_sid, _bg_changes, _bg_report)
                    except Exception:
                        pass

                try:
                    _BG_STATE_JUDGE_EXECUTOR.submit(_run_bg_judge)
                except Exception:
                    # Spawn failure must not affect foreground.
                    pass
        except Exception:
            # ThreadPoolExecutor or input-build failure: fall back to
            # the original sequential apply_gate site below; gate B
            # block will use the empty _judge_changes_perop.
            _parallel_kicked = False
    if not _is_finalizing_now and not _state_delta_kill_switch_op and _mcp._STATE.kg is not None:
        try:
            # Adrian directive 2026-05-11 (judge-gated coverage): the
            # prior coverage rule demanded state_deltas for ALL
            # state-updatable entities surfaced this op (agent +
            # intent_context + every is_a state-bearing instance in
            # accessed memories). Result: agents were forced to ack
            # every entity per op, even when nothing moved. The "ack"
            # path became the lazy escape ("unchanged" without thought)
            # which defeats the whole point.
            #
            # New rule: state_deltas required ONLY for entities the
            # state_judge flagged this op. Silence on a non-flagged
            # entity means "nothing happened" -- no ack needed.
            # `unchanged` becomes exclusively a judge-override (agent
            # disagrees with the flag); justification is REQUIRED on
            # every unchanged.
            #
            # The surfaced-instances accumulation is GONE -- if an
            # instance was surfaced but didn't move, the judge will
            # say so (empty changes) and no coverage is demanded.
            _judge_flagged_perop: set = set()
            for _change in _judge_changes_perop:
                _flagged_id = (_change.get("entity_id") or "").strip()
                if _flagged_id:
                    _judge_flagged_perop.add(normalize_entity_name(_flagged_id))
            # Validate `unchanged` deltas: each must reference an
            # entity in the judge-flagged set (override-only) and
            # must carry a justification (audit trail).
            _unchanged_violations: list = []
            for _vd in _validated_deltas:
                if _vd.get("status") != "unchanged":
                    continue
                _vd_eid = normalize_entity_name(_vd.get("entity_id") or "")
                if _vd_eid not in _judge_flagged_perop:
                    _unchanged_violations.append(
                        {
                            "entity_id": _vd.get("entity_id"),
                            "reason": (
                                "status='unchanged' is only valid when "
                                "overriding a state_judge flag. This "
                                "entity was not flagged by the judge "
                                "this op -- omit the entry entirely."
                            ),
                        }
                    )
                    continue
                if not (_vd.get("justification") or "").strip():
                    _unchanged_violations.append(
                        {
                            "entity_id": _vd.get("entity_id"),
                            "reason": (
                                "status='unchanged' requires a "
                                "justification explaining why the judge "
                                "was wrong (audit trail for the override)."
                            ),
                        }
                    )
            if _unchanged_violations and not _v2_visibility:
                _resp_block = {
                    "success": False,
                    "error": (
                        "state_deltas 'unchanged' validation failed -- "
                        "see unchanged_violations for the entries."
                    ),
                    "unchanged_violations": _unchanged_violations,
                }
                if _judge_changes_perop:
                    _resp_block["state_changes_detected"] = _judge_changes_perop
                # v3.7.10: state_judge_report removed -- tail
                # mempalace_bg_status(streams=["state_judge_log"]).
                return _resp_block
            elif _unchanged_violations and _v2_visibility:
                # v3.2.8 Phase 2 (Adrian directive 2026-05-13): opt-in
                # skip of the unchanged-violations raise too. Same
                # rationale as Phase 1 missing_state_deltas -- agents
                # under v2_visibility shouldn't be blocked on per-op
                # state_deltas bookkeeping; the judge's findings still
                # surface in state_changes_detected on success.
                try:  # pragma: no cover - logging is best-effort
                    import logging as _v2_log  # noqa: PLC0415

                    _v2_log.getLogger(__name__).info(
                        "state_deltas unchanged_violations (v2_visibility opt-in; op proceeds): %s",
                        [v.get("entity_id") for v in _unchanged_violations],
                    )
                except Exception:
                    pass
            _covered_perop = {normalize_entity_name(eid) for eid in _delta_entity_set}
            _missing_perop = _judge_flagged_perop - _covered_perop
            if _missing_perop and not _v2_visibility:
                _resp_block = {
                    "success": False,
                    "error": (
                        "state_judge flagged entities you didn't cover. "
                        "For each missing entity, provide state_deltas "
                        "with status='changed' + RFC 6902 patch (or, "
                        "if you disagree with the judge, "
                        "status='unchanged' + justification explaining "
                        "why)."
                    ),
                    "missing_state_deltas": sorted(_missing_perop),
                }
                if _judge_changes_perop:
                    _resp_block["state_changes_detected"] = _judge_changes_perop
                # v3.7.10: state_judge_report removed -- tail
                # mempalace_bg_status(streams=["state_judge_log"]).
                return _resp_block
            elif _missing_perop and _v2_visibility:
                # v3.2.7 Phase 1: opt-in env flag MEMPALACE_STATE_PROTOCOL=
                # v2_visibility skips the v0 raise and lets the op
                # succeed. The agent still sees what the judge detected
                # via state_changes_detected attached to the success
                # response below. No auto-patch generation yet; future
                # phases add that + the challenge MCP.
                try:  # pragma: no cover - logging is best-effort
                    import logging as _v2_log  # noqa: PLC0415

                    _v2_log.getLogger(__name__).info(
                        "state_judge missing_state_deltas (v2_visibility opt-in; op proceeds): %s",
                        sorted(_missing_perop),
                    )
                except Exception:
                    pass
        except Exception:  # pragma: no cover - defensive; never block on bug here
            pass
    _persist_active_intent()

    # ── Build response ──
    # Rules (mandatory-coverage, fetch-full-via-kg_query, declare-gate)
    # live in the wake_up protocol -- we no longer repeat them in every
    # operation response. See wake_up's protocol string for the contract.
    #
    # Step 3 of similar_context_id flag (default-on, parity with
    # declare_intent + kg_search): walk the rated-neighbourhood of the
    # operation's context (or fallback to the active intent's context)
    # to surface which similar_to neighbours contributed weight to each
    # retrieved item. declare_operation does not normally walk -- the
    # cosine retrieval here is op-cue-only -- so we do a fresh walk
    # purely to populate the monitoring fields.
    _op_walk_ctx = _op_context_id or (
        _mcp._STATE.active_intent.get("active_context_id") if _mcp._STATE.active_intent else ""
    )
    try:
        from . import scoring as _scoring_op

        _op_rated_walk = (
            _scoring_op.walk_rated_neighbourhood(_op_walk_ctx, _mcp._STATE.kg)
            if _op_walk_ctx
            else {"contributing_contexts": {}}
        )
    except Exception:
        _op_rated_walk = {"contributing_contexts": {}}
    _op_contributing_contexts = _op_rated_walk.get("contributing_contexts") or {}

    # follow-up 2026-05-07: memories list + enrichment
    # are now built above (before the parallel apply_gate +
    # state_judge block) so apply_gate has its input ready
    # synchronously. Keeping a comment here as a sign-post.

    # Step 3 of similar_context_id flag (default-on): shared helper
    # annotates memories in place with similar_context_ids and returns
    # the top-level similar_contexts list with link_score. Same shape
    # as declare_intent + kg_search + declare_user_intents.
    from . import scoring as _scoring_render

    _op_similar_contexts = _scoring_render.render_similar_contexts_block(
        memories,
        _op_contributing_contexts,
        _op_rated_walk.get("neighbourhood_weights") or {},
        _mcp._STATE.kg,
    )

    # ── Injection-stage gate ──
    # Same wiring as declare_intent: filter memories via the Haiku
    # relevance gate, persist drops as rated_irrelevant feedback
    # (rater_kind='gate_llm'), fail-open on any bug. Parent frame =
    # the active intent (this operation is nested under it).
    # Sequential apply_gate fallback (only when the parallel block
    # above didn't fire -- finalizing path, kill-switch on, KG None,
    # or executor crash). When _parallel_kicked is True, the gate
    # already ran upstream and we skip this site to avoid a double
    # Haiku call.
    if not _parallel_kicked:
        try:
            from .injection_gate import apply_gate as _apply_gate

            _op_combined_meta = {
                h["id"]: {
                    "source": ("triple" if str(h.get("id", "")).startswith("t_") else "memory"),
                    "doc": (h.get("preview") or "").strip(),
                    "similarity": float(h.get("score", 0.0) or 0.0),
                }
                for h in hits
                if h.get("id")
            }
            _parent_intent = None
            try:
                ai = _mcp._STATE.active_intent or {}
                _parent_intent = {
                    "intent_type": ai.get("intent_type"),
                    "subject": ", ".join((ai.get("slots", {}) or {}).get("subject", []) or []),
                    "query": (ai.get("description_views") or [""])[0],
                }
            except Exception:
                _parent_intent = None

            _gated, _gate_status, _gate_report = _apply_gate(
                memories=memories,
                combined_meta=_op_combined_meta,
                primary_context={
                    "source": "declare_operation",
                    "queries": list(cue["queries"]),
                    "keywords": list(cue["keywords"]),
                    "entities": list(entities or []),
                },
                context_id=_op_context_id or "",
                kg=_mcp._STATE.kg,
                agent=agent,
                parent_intent=_parent_intent,
            )
            memories = _gated
        except Exception:
            _gate_report = None

    result = {"success": True, "memories": memories}

    # v3.7.4 Slice 3 (Adrian directive 2026-05-16): drain the per-sid
    # pending state-updates buffer and surface bg judge findings from
    # PRIOR ops on this response. One-op lag is acceptable per the
    # architecture review: judge auto-applies its writes inline via
    # record_state_revision(agent='state_judge'), so the surfacing is
    # purely for agent visibility. When the buffer is empty (judge
    # ran but flagged nothing, or bg mode is off), the field is
    # omitted entirely so quiet ops stay lean.
    try:
        _pending_updates = _drain_pending_state_updates(_mcp._STATE.session_id or "")
        if _pending_updates:
            result["state_updates_since_last_op"] = _pending_updates
    except Exception:
        # Drain failure must not break the op response.
        pass
    if _op_schemas:
        result["schemas"] = _op_schemas
    # similar_contexts visibility centralised in scoring helper
    # (hide-by-default + opt-in via MEMPALACE_SHOW_SIMILAR_CONTEXTS=1;
    # Adrian directive 2026-05-04). Empty block here means hidden.
    if _op_similar_contexts:
        result["similar_contexts"] = _op_similar_contexts
    if _gate_status is not None:
        result["gate_status"] = _gate_status
    # v3.7.10 (Adrian directive 2026-05-17): gate_report +
    # state_judge_report removed from inline response. Same
    # telemetry available via mempalace_bg_status(streams=
    # ["gate_log", "state_judge_log"]) which tails the per-call
    # rows apply_gate + run_state_judge already write to
    # ~/.mempalace/hook_state/. Removing the inline copies saves
    # ~300-500 bytes per declare_operation response and ends the
    # per-response token tax on every gate-triggering op.
    # v3.2.9 Phase 3 (Adrian directive 2026-05-13): when the
    # v2_visibility env flag is on AND the judge supplied an RFC 6902
    # patch + schema_id for a flagged entity AND the agent did NOT
    # cover that entity via state_deltas this op, auto-apply the
    # patch via record_state_revision with agent='state_judge'. This
    # is the deferred-write half of the v2 design: judge becomes
    # capable of moving state without agent ack, and the per-change
    # 'applied'/'rev_id'/'error' attribution surfaces on the response
    # so the agent (and future challenge_state_change MCP) can see
    # exactly what was written.
    #
    # Why gated on v2_visibility: v0 default still REQUIRES agent
    # state_deltas coverage on every flag (gate raises above); there
    # is nothing to auto-apply under v0 because the op already
    # failed. Auto-apply is meaningful only when v2_visibility lets
    # the op succeed despite missing agent coverage.
    if _v2_visibility and _judge_changes_perop:
        # v3.7.23 (FINDING #10 fix): delegate to centralized helper so
        # the foreground apply path stays in lockstep with the bg
        # worker (mempalace/intent.py:_run_bg_judge). Pre-v3.7.23 this
        # site held an inline copy of the logic; the bg refactor in
        # v3.7.4 introduced a silent gap where the bg path lacked the
        # equivalent. Consolidating both call sites on one helper
        # makes the drift impossible by construction.
        _apply_judge_changes_to_state(
            _judge_changes_perop,
            op_context_id=(_op_context_id or ""),
            delta_covered=(_mcp._STATE.active_intent.get("state_deltas_entity_set") or set()),
            session_id=(_mcp._STATE.session_id or None),
        )

    # v3.2.7 Phase 1 (Adrian directive 2026-05-12): attach
    # state_changes_detected to the success response too -- not just
    # to the failure responses. This is opt-in via env flag
    # MEMPALACE_STATE_PROTOCOL=v2_visibility but also useful in v0
    # mode for ops that DID cover their flagged entities (so the
    # agent sees what the judge flagged + what they patched). When
    # the v2 flag IS set, this is the only place the agent learns
    # the judge fired at all -- the v0 raise was skipped above.
    # v3.2.9 Phase 3 when v2_visibility is on, the entries
    # ALSO carry 'applied'/'rev_id'/'error' per-change attribution
    # so the agent can see what the judge auto-wrote.
    if _judge_changes_perop:
        result["state_changes_detected"] = _judge_changes_perop

    # ── S1: past_operations -- op-tier retrieval ──
    # Orthogonal to memories (Channels A-D). Walks performed_well /
    # performed_poorly edges from the current operation's context
    # neighbourhood, returning good precedents + cautionary patterns.
    # Not filtered through the memory gate; rendered in its own slot
    # so op-tier noise can't pollute the memory retrieval signal. Cf.
    # arXiv 2512.18950 (Operation tier), Leontiev 1981 AAO.
    if _op_context_id:
        try:
            from .scoring import retrieve_past_operations as _retrieve_ops
            from .vector_store import RECORDS_COLLECTION, get_vector_store

            # 2026-04-27: pass args_summary + op-Chroma collection so
            # retrieve_past_operations can populate the args_precedents
            # lane via cosine recall + BGE-reranker rerank, surfacing
            # ops with similar parametrized fingerprint regardless of
            # context. Tool filter eliminates cross-tool false matches.
            # Tier-3 (2026-05-10): route through VectorStore.query
            # against RECORDS_COLLECTION (post-M1 collection that
            # absorbed the legacy mempalace_entities collection;
            # kind='operation' is the metadata discriminator).
            _vs = get_vector_store(_mcp._STATE.config.palace_path)
            _past_ops = _retrieve_ops(
                _op_context_id,
                _mcp._STATE.kg,
                k=5,
                current_args_summary=args_summary,
                vs=_vs,
                op_collection_name=RECORDS_COLLECTION,
                current_tool=tool,
            )
            _has_good = bool(_past_ops.get("good_precedents"))
            _has_bad = bool(_past_ops.get("avoid_patterns"))
            _has_args = bool(_past_ops.get("args_precedents"))
            # Only attach when there is something to say -- keeps the
            # response lean when the graph has no op history yet.
            if _has_good or _has_bad or _has_args:
                result["past_operations"] = _past_ops

            # 2026-04-26 diagnostic -- see _record_op_recall_diagnostic
            # for the rationale and full doc.
            _record_op_recall_diagnostic(_op_context_id, _has_good or _has_bad)

            # S3a: piggyback-flag same-tool same-sign clusters for the
            # gardener to templatize (S3b) and for retrieval to later
            # hoist as reusable patterns (S3c). Helper is
            # fire-and-forget; see _emit_op_cluster_flags.
            _emit_op_cluster_flags(_past_ops, _op_context_id, _mcp._STATE.kg)
        except Exception:
            # Fail-silent: op retrieval is a nice-to-have, not load-bearing.
            pass

    if DEBUG_RETURN_CONTEXT:
        # Token-diet 2026-04-24: non-reused collapses to "new"; reused
        # returns {id, queries}. See tool_declare_intent for the full
        # rationale. Shape-as-signal: string "new" = fresh, object = reused.
        if _op_context_reused:
            result["context"] = {
                "id": _op_context_id,
                "queries": list(cue["queries"]),
            }
        else:
            result["context"] = "new"
    if notice:
        # Fail-loud: retrieval error surfaces to agent, not silent.
        result["retrieval_notice"] = notice
    return result


# ═══════════════════════════════════════════════════════════════════
# (user-intent tier): tool_declare_user_intents
# ═══════════════════════════════════════════════════════════════════
#
# Top-tier (Motive / Strategy in Leontiev 1981) declaration. The agent
# calls this AFTER each user message (or batch of messages) to declare
# what the user is asking for, BEFORE proceeding to declare any
# activity-intent. The tool:
#
#   1. Reads pending_user_messages from session state (written by
#      UserPromptSubmit hook).
#   2. Validates that union(context.user_message_ids for context in
#      contexts) covers every pending user_message_id -- no message
#      falls through unnoticed.
#   3. Mints a kind='record' user_message entity for each pending
#      message (content = raw prompt text). Links each user-context
#      to its referenced messages via fulfills_user_message edges.
#   4. Calls context_lookup_or_create per declared context (MaxSim
#      reuse, similar_to graph wiring -- same path as declare_intent /
#      declare_operation / kg_search).
#   5. Runs retrieval per context, dedup'd against accessed/injected
#      memory ids accumulated in this session, returns top-K.
# 6. Clears the pending queue so the PreToolUse block
#      releases.
#
# Grounding: STITCH (arXiv:2601.10702) for the structured-intent-tuple
# pattern, Agent-Sentry (arXiv:2603.22868) for the forced-cause-linkage
# discipline, BDI (Rao & Georgeff 1995) for the hierarchical-cause
# invariant. See diary_ga_agent_user_intent_tier_design_locked_2026_04_24
# for the full design narrative.
#
# scope: tool ships and works end-to-end (agent can call it,
# validates pending coverage, mints records, returns memories per
# context). wires the PreToolUse block + UserPromptSubmit
# rewrite that produces the pending entries. adds optional
# cause_id on declare_intent + finalize coverage rule.


# Per-context bounds for the user-intent tier. Mirrors MIN_OP_QUERIES
# / MAX_OP_QUERIES style -- kept module-level so schema + tests share
# the source-of-truth.
MIN_USER_INTENT_QUERIES = 1  # one perspective per intent is enough
MAX_USER_INTENT_QUERIES = 5
MIN_USER_INTENT_KEYWORDS = 2
MAX_USER_INTENT_KEYWORDS = 5
MIN_USER_INTENT_ENTITIES = 1
MAX_USER_INTENT_ENTITIES = 10
USER_INTENT_TOP_K = 5  # memories per context


# session-scoped first-rater set. Maps session_id → set of
# user-context entity ids whose surfaced memories have been claimed by
# the first agent intent finalize in this session. v3.5.0 (2026-05-14)
# removed the synchronous coverage gate -- the async-Haiku rater rates
# memories post-finalize regardless -- but this set is still surfaced
# to declare_intent so downstream signals can distinguish "first-rater
# intent" from "inheriting intent" without re-rating. In-memory only --
# survives only as long as the MCP server process. Tests reset by
# reassigning to a fresh dict (see _reset_rated_user_contexts).
_RATED_USER_CONTEXTS: dict = {}


def _rated_user_contexts_for(sid: str) -> set:
    """Get-or-create the rated_user_contexts set for the given session."""
    if not isinstance(sid, str):
        sid = ""
    bucket = _RATED_USER_CONTEXTS.get(sid)
    if bucket is None:
        bucket = set()
        _RATED_USER_CONTEXTS[sid] = bucket
    return bucket


def _reset_rated_user_contexts() -> None:
    """Drop all session buckets. Used by tests to isolate state."""
    _RATED_USER_CONTEXTS.clear()


def tool_declare_user_intents(  # noqa: C901
    contexts: list = None,
    agent: str = None,
):
    """Declare the user-intent contexts that cover the pending user
    messages for this session. Top tier of the activity hierarchy
    (Motive/Strategy); activity-intents declared via declare_intent
    later this turn link upward via cause_id.

    Args:
        contexts: list of dicts, one per user-intent. Each dict carries:
            - context: {queries, keywords, entities, summary} --
                same unified Context shape as every other emit site.
            - user_message_ids: list[str] -- pending message ids this
                user-intent covers. Union across all contexts MUST
                equal the pending set (no message left unattributed).
            - time_window: {start, end} optional ISO dates for soft
                date-range boost in retrieval (same semantics as
                kg_search.time_window).
            - no_intent: bool default False -- set TRUE to declare that
                a covered user message has no actionable intent (ack,
                "thanks", clarifying question already answered, etc.).
                When TRUE, no_intent_clarified_with_user MUST be a
                truthful bool (set TRUE only if the agent actually
                asked the user via AskUserQuestion to confirm).
        agent: caller's agent name. Required.

    Returns:
        {
          "success": True,
          "contexts": [
            {"ctx_id": "...", "reused": bool, "memories": [...]},
            ...
          ],
          "cleared_pending_count": N,
        }

    Validation rejects (success=False with explicit error) on:
        * Empty contexts list.
        * Any context missing user_message_ids.
        * Any user_message_id not in the pending queue.
        * Pending queue not fully covered by union of user_message_ids.
        * Standard Context validator failures (per-context queries /
          keywords / entities / summary bounds).
        * no_intent=True without no_intent_clarified_with_user=True.
    """
    sid_err = _mcp._require_sid(action="declare_user_intents")
    if sid_err:
        return sid_err
    agent_err = _mcp._require_agent(agent, action="declare_user_intents")
    if agent_err:
        return agent_err

    contexts, _err = _coerce_list_param("contexts", contexts)
    if _err:
        return _err
    if not contexts:
        return {
            "success": False,
            "error": (
                "contexts is required and must be a non-empty list. "
                "At least one user-intent context per call. Each context "
                "carries {context: {queries, keywords, entities, summary}, "
                "user_message_ids: [...], time_window?: {...}, "
                "no_intent?: bool, no_intent_clarified_with_user?: bool}."
            ),
        }

    # ── Read pending user-messages for this session ──
    from . import hooks_cli as _hc

    sid = _mcp._STATE.session_id or ""
    pending_msgs = _hc._read_pending_user_messages(sid)
    pending_ids = {m["id"] for m in pending_msgs}

    # ── Per-context shape validation + collect referenced ids ──
    from .scoring import validate_context as _validate_context

    cleaned_contexts = []
    referenced_ids = set()
    for i, c in enumerate(contexts):
        if not isinstance(c, dict):
            return {
                "success": False,
                "error": (
                    f"contexts[{i}] must be a dict with keys "
                    "'context', 'user_message_ids', and optional "
                    "'time_window' / 'no_intent' / "
                    "'no_intent_clarified_with_user'."
                ),
            }
        raw_ctx = c.get("context")
        clean_ctx, ctx_err = _validate_context(
            raw_ctx,
            queries_min=MIN_USER_INTENT_QUERIES,
            queries_max=MAX_USER_INTENT_QUERIES,
            keywords_min=MIN_USER_INTENT_KEYWORDS,
            keywords_max=MAX_USER_INTENT_KEYWORDS,
            entities_min=MIN_USER_INTENT_ENTITIES,
            entities_max=MAX_USER_INTENT_ENTITIES,
            require_summary=True,
            summary_context_for_error=f"declare_user_intents.contexts[{i}].context.summary",
        )
        if ctx_err:
            return ctx_err

        umids = c.get("user_message_ids")
        if not isinstance(umids, list) or not umids:
            return {
                "success": False,
                "error": (
                    f"contexts[{i}].user_message_ids is required (non-empty list). "
                    "Reference at least one pending user_message id this "
                    "user-intent covers. See additionalContext from the "
                    "UserPromptSubmit hook for pending ids."
                ),
            }
        for mid in umids:
            if not isinstance(mid, str) or not mid.strip():
                return {
                    "success": False,
                    "error": f"contexts[{i}].user_message_ids contains a non-string entry.",
                }
            if mid not in pending_ids:
                return {
                    "success": False,
                    "error": (
                        f"contexts[{i}].user_message_ids references {mid!r} which is "
                        f"not in the pending user_message queue for this session. "
                        f"Pending ids: {sorted(pending_ids)}"
                    ),
                }
            referenced_ids.add(mid)

        no_intent = bool(c.get("no_intent", False))
        if no_intent:
            confirmed = bool(c.get("no_intent_clarified_with_user", False))
            if not confirmed:
                return {
                    "success": False,
                    "error": (
                        f"contexts[{i}].no_intent=True requires "
                        "no_intent_clarified_with_user=True -- the agent must "
                        "have actually asked the user (via AskUserQuestion) "
                        "to confirm the message has no actionable intent. "
                        "Self-asserting no_intent without proof is rejected."
                    ),
                }

        time_window = c.get("time_window")
        if time_window is not None and not isinstance(time_window, dict):
            return {
                "success": False,
                "error": f"contexts[{i}].time_window must be a dict {{start, end}} or omitted.",
            }

        cleaned_contexts.append(
            {
                "clean_ctx": clean_ctx,
                "user_message_ids": list(umids),
                "time_window": time_window,
                "no_intent": no_intent,
            }
        )

    # ── Coverage check: every pending id must be referenced by ≥1 context ──
    if pending_ids and pending_ids - referenced_ids:
        missing = sorted(pending_ids - referenced_ids)
        return {
            "success": False,
            "error": (
                f"Pending user_message ids not covered by any declared context: "
                f"{missing}. Every pending message must appear in at least one "
                f"context.user_message_ids. If a message has no actionable intent, "
                f"declare it under a no_intent=True context (with "
                f"no_intent_clarified_with_user=True after asking the user)."
            ),
            "missing_user_message_ids": missing,
        }

    # ── Mint user_message records for each pending entry ──
    # Idempotent: if the record already exists (re-run scenario), skip.
    minted_user_message_ids = []
    for m in pending_msgs:
        mid = m["id"]
        if not mid:
            continue
        existing = None
        try:
            existing = _mcp._STATE.kg.get_entity(mid)
        except Exception:
            existing = None
        if existing:
            minted_user_message_ids.append(mid)
            continue
        try:
            # Cold-start lock 2026-05-01 (Adrian's user-message analysis):
            # user_messages get their own kind, not 'record'. They're
            # SQLite-only by design -- literal user text is value, not
            # identity, so we skip the summary contract AND the Chroma
            # write. Retrieval naturally filters them out (not embedded
            # -> not searchable). The user-context that fulfills them
            # carries the searchable identity via its own {what, why,
            # scope} summary. _sync_entity_to_chromadb has the matching
            # carve-out so any future caller that routes through
            # _create_entity still skips Chroma for this kind.
            # Defense-in-depth surrogate sanitization at the mint
            # boundary. The hooks_cli read+write paths already sanitize
            # but a corrupt JSONL written by a pre-fix server may still
            # be on disk; this ensures the mint never crashes on legacy
            # data. Idempotent on already-clean strings.
            try:
                from . import hooks_cli as _hc

                _safe_text = _hc._sanitize_utf8(m.get("text") or "")[:500]
            except Exception:
                _safe_text = (m.get("text") or "")[:500]
            _mcp._STATE.kg.add_entity(
                mid,
                kind="user_message",
                content=_safe_text,
                importance=3,
                properties={
                    "type": "user_message",
                    "session_id": sid,
                    "turn_idx": int(m.get("turn_idx") or 0),
                    "ts": m.get("ts") or "",
                    "added_by": agent or "",
                },
            )
            minted_user_message_ids.append(mid)
        except Exception as _mint_err:
            return {
                "success": False,
                "error": f"Failed to mint user_message record {mid!r}: {_mint_err!r}",
            }

    # ── For each context: lookup_or_create + fulfills_user_message edges + retrieval ──
    response_contexts = []
    new_injected_ids = []
    for entry in cleaned_contexts:
        clean_ctx = entry["clean_ctx"]
        ctx_id = ""
        reused = False
        try:
            ctx_id, reused, _ms = _mcp.context_lookup_or_create(
                queries=clean_ctx["queries"],
                keywords=clean_ctx["keywords"],
                entities=clean_ctx["entities"],
                agent=agent,
                summary=clean_ctx.get("summary"),
            )
        except Exception:
            ctx_id = ""
            reused = False

        # Wire user_message → user-context coverage edges. fulfills_user_message
        # is the predicate cause_id validator reads to identify
        # "user-tier" contexts. The predicate is non-skip-list so a natural
        # -language statement is required (2026-04-19 lock that retired
        # autogenerated verbalisations). Soft-fail at edge level so a
        # transient kg/seeder issue does not prevent the context creation.
        if ctx_id:
            for um_id in entry["user_message_ids"]:
                _ful_statement = (
                    f"User-context {ctx_id} fulfils user_message {um_id} "
                    f"by declaring an intent that covers this user prompt."
                )
                try:
                    _mcp._STATE.kg.add_triple(
                        ctx_id,
                        "fulfills_user_message",
                        um_id,
                        statement=_ful_statement,
                    )
                except Exception:
                    pass

        # Retrieval per context -- same pipeline as declare_operation.
        cue = {
            "queries": list(clean_ctx["queries"]),
            "keywords": list(clean_ctx["keywords"]),
        }
        accessed = (
            set(_mcp._STATE.active_intent.get("accessed_memory_ids") or [])
            | set(_mcp._STATE.active_intent.get("injected_memory_ids") or [])
            if _mcp._STATE.active_intent
            else set()
        )
        try:
            hits, _notice = _hc._run_local_retrieval(cue, accessed, USER_INTENT_TOP_K)
        except Exception:
            hits = []
        memories = []
        for h in hits:
            mid = h.get("id")
            if not mid:
                continue
            # v3.7.43 FINDING #AA: skip user_message kind here too. The
            # _run_local_retrieval pipeline routes graph BFS neighbors
            # through the same rerank, and contexts surface their
            # fulfills_user_message edges as Channel B neighbors --
            # user_messages would otherwise surface to the agent as
            # bare turn text. See intent.py:2711 fix for the parallel
            # filter in the declare_intent/declare_operation path.
            _h_kind = (h.get("meta") or {}).get("kind", "")
            if _h_kind == "user_message":
                continue
            new_injected_ids.append(mid)
            # v3.7.9: centralized via _project_memory helper.
            # v3.7.34: pass vec metadata sub-dict so _project_memory
            # can hoist date_added + last_relevant_at to the agent.
            _ui_extras = {}
            _ui_h_meta = h.get("meta") or {}
            if _ui_h_meta:
                _ui_extras["metadata"] = _ui_h_meta
            memories.append(
                _project_memory(mid, (h.get("preview") or "").strip(), extras=_ui_extras or None)
            )

        # State-protocol v1 (Adrian 2026-05-03): enrich
        # state-bearing surfaced memories with current_state +
        # state_schema_id parallel to declare_operation / declare_intent.
        # Capture the schemas dict so each per-context block can carry
        # its referenced schemas at the per-context level.
        _ui_schemas: dict = {}
        try:
            _ui_schemas = _enrich_memories_with_state(memories, _mcp._STATE.kg) or {}
        except Exception:
            pass

        # Step 4 of similar_context_id flag (default-on): the user-intent
        # context just minted/reused has its own rated-neighbourhood --
        # walk it and render similar_contexts so the agent sees which
        # similar_to neighbours of THIS user-message context contributed
        # to the retrieved memories. Same shape as the other 3 surfaces.
        from . import scoring as _scoring_render

        _user_rated_walk = (
            _scoring_render.walk_rated_neighbourhood(ctx_id, _mcp._STATE.kg)
            if ctx_id
            else {"contributing_contexts": {}, "neighbourhood_weights": {}}
        )
        _user_similar_contexts = _scoring_render.render_similar_contexts_block(
            memories,
            _user_rated_walk.get("contributing_contexts") or {},
            _user_rated_walk.get("neighbourhood_weights") or {},
            _mcp._STATE.kg,
        )

        block = {"ctx_id": ctx_id, "reused": bool(reused)}
        # Token-diet: echo queries/keywords/entities only on reuse, mirroring
        # declare_intent / declare_operation convention.
        if reused:
            block["queries"] = list(clean_ctx["queries"])
            block["keywords"] = list(clean_ctx["keywords"])
            block["entities"] = list(clean_ctx["entities"])
        if memories:
            block["memories"] = memories
        if _ui_schemas:
            block["schemas"] = _ui_schemas
        # similar_contexts visibility centralised in scoring helper
        # (hide-by-default + opt-in via MEMPALACE_SHOW_SIMILAR_CONTEXTS=1;
        # Adrian directive 2026-05-04).
        if _user_similar_contexts:
            block["similar_contexts"] = _user_similar_contexts
        if entry["no_intent"]:
            block["no_intent"] = True
        response_contexts.append(block)

    # ── Persist injected ids to active_intent (if any) ──
    # When no active_intent exists yet (early in the session), we still
    # cleared pending; the next declare_intent will inherit retrieval
    # via its own context. This matches the "user-tier sits ABOVE
    # activity" design -- user contexts can exist without an activity.
    if new_injected_ids and _mcp._STATE.active_intent:
        _inj = _mcp._STATE.active_intent.get("injected_memory_ids")
        if not isinstance(_inj, set):
            _inj = set(_inj or [])
        _inj.update(new_injected_ids)
        _mcp._STATE.active_intent["injected_memory_ids"] = _inj
        try:
            _persist_active_intent()
        except Exception:
            pass

    # ── Clear ONLY the declared ids from the pending queue ──
    # Adrian's spec 2026-04-29: per-id removal, not bulk-drain. Messages
    # still pending after this call remain in the queue and surface in
    # the next UserPromptSubmit / PreToolUse gate check. Preserves
    # next_turn_idx so future ids stay monotonic.
    cleared_n = _hc._remove_pending_user_messages(sid, referenced_ids)

    return {
        "success": True,
        "contexts": response_contexts,
        "cleared_pending_count": cleared_n,
        # 2026-05-04 (Adrian token-diet): minted_user_message_ids was
        # echoing the input user_message_ids back to the caller -- pure
        # waste since the caller just sent them. Field dropped.
    }


# Single-field relevance → (relevant, confidence) mapping.
#
# v3.5.0 (2026-05-14): the agent-side _derive_feedback_pair helper +
# _RELEVANCE_MAPPING constant lived here. They are retired -- the
# async-Haiku rater in mempalace.feedback_auto rates retrieved memories
# post-finalize via kg.record_feedback (rater_kind='haiku_auto') and the
# raw 1-5 signal lands without an in-process derive.


def _coerce_list_param(name: str, val):
    """Normalize an MCP list-shaped param, guarding against stringified JSON.

    Mirrors the guard already used by ``tool_resolve_conflicts``. Some MCP
    transports (and the Opus planner under load) serialize a top-level
    array argument as a JSON string. A naive ``for item in val`` then walks
    characters and emits one bogus error per char -- the same bug that
    could balloon a response to ~61k chars of per-char entries.

    Returns ``(coerced, err_response)``. If ``err_response`` is not None the
    caller must return it unmodified. ``None`` and real lists pass through
    untouched.
    """
    if val is None or isinstance(val, list):
        return val, None
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
        except Exception:
            return None, {
                "success": False,
                "error": (
                    f"`{name}` arrived as an unparseable JSON string. "
                    f"Pass a JSON array of objects, not a string."
                ),
            }
        if not isinstance(parsed, list):
            return None, {
                "success": False,
                "error": (
                    f"`{name}` parsed from string must be a list, got {type(parsed).__name__}."
                ),
            }
        return parsed, None
    return None, {
        "success": False,
        "error": f"`{name}` must be a list, got {type(val).__name__}.",
    }


def tool_finalize_intent(  # noqa: C901
    slug: str,
    outcome: str,
    content: str,
    summary: str,
    agent: str,
    key_actions: list = None,
    gotchas: list = None,
    learnings: list = None,
    promote_gotchas_to_type: bool = False,
    state_deltas: list = None,
):
    """Finalize the active intent -- capture what happened as structured memory.

    MUST be called before declaring a new intent or exiting the session.
    Creates an execution entity (kind=entity, is_a intent_type) with
    relationships linking it to the agent, targets, result memory, gotchas,
    and execution trace.

    v3.5.0 (2026-05-14): the agent-side ``memory_feedback`` +
    ``operation_ratings`` params and their 100%-coverage gates are GONE.
    Rating retrieved memories + tool-call quality is now performed
    asynchronously by ``mempalace.feedback_auto`` (Haiku rater) after
    finalize returns -- the agent no longer constructs ratings inline.
    Coverage friction is replaced by post-hoc fire-and-forget rater
    jobs that fill in rated_useful / rated_irrelevant +
    performed_well / performed_poorly edges out of band.

    VOCABULARY -- uniform across every record-write boundary in mempalace:
      ``content`` = full narrative body. FREE LENGTH -- as detailed as needed.
        Stored verbatim.
      ``summary`` = ≤280-char distillation / reframe. ALWAYS required (no
        length threshold on content -- every record gets a summary). For
        long content the summary distills the WHAT/WHY; for short content
        the summary should REPHRASE the same fact from a different angle
        (different keywords / framing) so the summary+content pair yields
        two distinct cosine views of the same semantic -- real retrieval
        gain, not redundancy. Anthropic Contextual Retrieval (2024)
        prepends the summary to the content before embedding (single CR
        vector); the summary is also what injection-time previews display.
        No prefix slicing. No auto-derivation. The caller produces it.

    Args:
        slug: Human-readable ID for this execution (e.g. 'edit-auth-rate-limiter-2026-04-14')
        outcome: 'success', 'partial', 'failed', or 'abandoned'
        content: Full outcome narrative -- the body of the result memory. Any
            length. Becomes the embedded document (with summary prepended).
        summary: ≤280-char distilled one-liner of the outcome (or a
            different-angle rephrase when content is short). Shown in
            injections and prepended to content for embedding.
        agent: Agent entity name (e.g. 'technical_lead_agent')
        key_actions: Abbreviated tool+params list (optional -- auto-filled from trace if omitted)
        gotchas: List of gotchas discovered during execution. Each entry
            is ``{summary: {what, why, scope?}, content: str}`` --
            structured-anchor + verbatim body. Strings are rejected
            (no auto-derive; Adrian's design lock 2026-04-28).
        learnings: List of lessons worth remembering. Each entry is
            ``{summary: {what, why, scope?}, content: str}`` -- same
            strict dict shape. Strings are rejected.
        promote_gotchas_to_type: Also link gotchas to the intent type (not just execution)
        state_deltas: State-protocol v1 / v2 delta declarations -- see
            the per-op state_deltas contract on declare_operation. Each
            entry is {entity_id, status, schema_id?, patch?, justification?}.
            Unchanged from v3.4.x; gated by the state_judge at finalize.
    """

    # Sid check FIRST -- an empty sid means the tool call came in without
    # hook-injected sessionId, which makes every downstream state op a
    # potential cross-agent contamination risk. Fail loud at the boundary.
    sid_err = _mcp._require_sid(action="finalize_intent")
    if sid_err:
        return sid_err

    # v3.5.0 (2026-05-14): the pending_feedback limbo state is gone --
    # finalize no longer parks; it submits async-Haiku rater batches
    # via mempalace.feedback_auto and returns immediately. Older
    # _sync_from_disk() still runs lower down where it is needed.

    # ── Pending user-intent gate (Adrian's spec 2026-04-29) ──
    # Refuse to finalize if the session still has user_message ids in the
    # pending queue that haven't been declared via mempalace_declare_user_intents.
    # The agent must surface those messages and either declare an activity
    # intent for each or mark them no_intent (with confirmation) before
    # closing the current activity. Without this gate, finalize silently
    # drops user prompts the agent never addressed -- exactly the failure
    # mode that motivated the user-intent tier in the first place.
    try:
        from . import hooks_cli as _hc

        sid_for_check = _mcp._STATE.session_id or ""
        if sid_for_check:
            _pending_now = _hc._read_pending_user_messages(sid_for_check)
            _pending_ids_now = sorted(
                m.get("id") for m in (_pending_now or []) if isinstance(m, dict) and m.get("id")
            )
            if _pending_ids_now:
                return {
                    "success": False,
                    "error": (
                        f"finalize_intent refuses to close while user_message ids "
                        f"remain undeclared in this session's pending queue. "
                        f"Pending ids: {_pending_ids_now}. Call "
                        f"mempalace_declare_user_intents first to declare an "
                        f"intent context for each (or no_intent=True with "
                        f"no_intent_clarified_with_user=True after asking the "
                        f"user via AskUserQuestion). Adrian's design 2026-04-29: "
                        f"every user prompt must be acknowledged before any "
                        f"activity intent finalizes."
                    ),
                    "pending_user_message_ids": _pending_ids_now,
                }
    except Exception:
        # Fail-open on read errors -- the pending file is best-effort
        # state; a corrupt or missing file shouldn't block legitimate
        # finalize calls. The hook layer already records read errors via
        # _record_hook_error.
        pass

    # ── Summary-first gate: strict validation at the boundary ──
    # Mirrors _add_memory_internal's ≤280-char rule. Enforced HERE (not
    # only inside the downstream result_memory upsert) because the old
    # behaviour collected the downstream rejection into `errors` and
    # returned success=True -- so a 299-char summary would finalize the
    # intent, create the execution entity, but leave no result memory,
    # letting the caller assume everything was fine. Every method that
    # accepts a summary rejects over-length up front and fails the call.
    # Keep this in lockstep with _add_memory_internal so the two rules
    # never drift.
    # Dict-only contract (Adrian's design lock 2026-04-25): summary
    # is a structured {what, why, scope?} dict. coerce_summary_for_persist
    # validates and serialize_summary_for_embedding renders the prose form
    # downstream code reads as the human-facing one-liner. Keep this in
    # lockstep with _add_memory_internal so the two contracts never drift.
    if summary is None:
        return {
            "success": False,
            "error": (
                f"`summary` is required. Pass a dict {{what, why, scope?}}; "
                f"the rendered prose form is capped at "
                f"{_mcp._RECORD_SUMMARY_MAX_LEN} chars."
            ),
        }
    try:
        from .knowledge_graph import (
            SummaryStructureRequired as _SSR,
            coerce_summary_for_persist as _coerce_summary,
            serialize_summary_for_embedding as _ser_summary,
        )

        _summary_dict = _coerce_summary(
            summary,
            context_for_error="finalize_intent.summary",
        )
    except _SSR as _vs_err:
        return {"success": False, "error": str(_vs_err)}
    _summary_clean = _ser_summary(_summary_dict).strip()
    summary = _summary_clean  # downstream reads the rendered prose form
    if len(_summary_clean) > _mcp._RECORD_SUMMARY_MAX_LEN:
        return {
            "success": False,
            "error": (
                f"`summary` is {len(_summary_clean)} chars; maximum is "
                f"{_mcp._RECORD_SUMMARY_MAX_LEN}. Distill further -- one "
                f"sentence, names the WHAT and WHY, no filler."
            ),
        }
    summary = _summary_clean

    # v3.5.0 (2026-05-14): the agent-side memory_feedback shape coercion
    # / list-vs-dict validation block lived here. It is retired -- ratings
    # are now produced asynchronously by mempalace.feedback_auto's Haiku
    # rater after this function returns. No agent input is accepted for
    # memory_feedback or operation_ratings any more.

    gotchas, _pe = _coerce_list_param("gotchas", gotchas)
    if _pe:
        return _pe
    learnings, _pe = _coerce_list_param("learnings", learnings)
    if _pe:
        return _pe
    key_actions, _pe = _coerce_list_param("key_actions", key_actions)
    if _pe:
        return _pe

    _sync_from_disk()
    if not _mcp._STATE.active_intent:
        return {"success": False, "error": "No active intent to finalize."}

    # v3.5.0 (2026-05-14): pending_feedback re-entrance gate retired --
    # finalize is now atomic, no limbo state for extend_feedback to close.

    # fail-fast agent validation. Before P6.1 an undeclared agent
    # would silently break result/trace/learning memory creation deep
    # inside _add_memory_internal; now we reject upfront with the same
    # recipe the hook teaches.
    agent_err = _mcp._require_agent(agent, action="finalize_intent")
    if agent_err:
        return agent_err

    intent_type = _mcp._STATE.active_intent["intent_type"]
    intent_desc = _mcp._STATE.active_intent.get("content", "")
    slot_entities = []
    for slot_name, slot_vals in _mcp._STATE.active_intent.get("slots", {}).items():
        if isinstance(slot_vals, list):
            slot_entities.extend(slot_vals)
        elif isinstance(slot_vals, str):
            slot_entities.append(slot_vals)

    # Normalize slug
    exec_id = normalize_entity_name(slug)
    if not exec_id:
        return {"success": False, "error": "slug normalizes to empty."}

    # v3.5.0 (2026-05-14): the cop-out rejection helpers, per-entry
    # partial-accept gates, FINALIZE_COVERAGE_IN/MISS diagnostics,
    # first-rater user-context exemption, feedback_ids computation,
    # injected-memories coverage gate + missing_by_context builder,
    # _pending_missing_* sentinel vars, and the all-complete branching
    # at the bottom of finalize are all gone. The async-Haiku rater
    # (mempalace.feedback_auto) fills rated_useful / rated_irrelevant
    # + performed_well / performed_poorly out of band after this
    # function returns.

    # State-protocol v1 (Adrian rule-5 closure 2026-05-03):
    # finalize_intent accepts state_deltas to honor the v2 design-lock
    # rule "finalize re-declares all surfaced state so mid-intent
    # learning can correct." Same shape + semantics as
    # declare_operation / extend_feedback: each delta is
    # {entity_id, status: changed|unchanged|irrelevant, schema_id?,
    # patch?}. Validated + persisted onto active_intent's
    # state_deltas_entity_set / irrelevant_state_set. status=changed
    # applies the RFC 6902 patch and writes a state_revision (with
    # empty op_context_id since no operation cue is mid-flight at
    # finalize). Errors return immediately so the agent sees what to
    # fix; the coverage block below then re-reads these sets.
    if state_deltas is not None:
        if not isinstance(state_deltas, list):
            return {
                "success": False,
                "error": (
                    f"state_deltas must be a list of dicts; got {type(state_deltas).__name__}."
                ),
            }
        for _i, _d in enumerate(state_deltas):
            if not isinstance(_d, dict):
                return {
                    "success": False,
                    "error": f"state_deltas[{_i}] must be a dict.",
                }
            _eid = (_d.get("entity_id") or "").strip()
            _status = (_d.get("status") or "").strip()
            if not _eid or _status not in ("changed", "unchanged"):
                return {
                    "success": False,
                    "error": (
                        f"state_deltas[{_i}] requires entity_id + "
                        "status in {{changed,unchanged}}. State-protocol "
                        "v2 (Adrian 2026-05-04) removed 'irrelevant'."
                    ),
                }
            # conflict rejection (Adrian 2026-05-03): refuse
            # 'irrelevant' after a prior 'changed' in this intent.
            # See declare_operation block for the full rationale.
            _status_map = _mcp._STATE.active_intent.get("state_delta_status_per_entity")
            if not isinstance(_status_map, dict):
                _status_map = {}
            _prior_status = _status_map.get(_eid)
            if _prior_status == "changed" and _status == "irrelevant":
                return {
                    "success": False,
                    "error": (
                        f"state_deltas[{_i}] for entity_id={_eid!r}: "
                        f"cannot mark 'irrelevant' after a prior "
                        f"'changed' delta in this intent."
                    ),
                }
            _status_map[_eid] = _status
            _mcp._STATE.active_intent["state_delta_status_per_entity"] = _status_map
            if _prior_status == "irrelevant" and _status != "irrelevant":
                _irr_clear = _mcp._STATE.active_intent.get("irrelevant_state_set")
                if isinstance(_irr_clear, set):
                    _irr_clear.discard(_eid)
                elif isinstance(_irr_clear, (list, tuple)):
                    _mcp._STATE.active_intent["irrelevant_state_set"] = {
                        _x for _x in _irr_clear if _x != _eid
                    }
            _delta_set = _mcp._STATE.active_intent.get("state_deltas_entity_set")
            if not isinstance(_delta_set, set):
                _delta_set = set(_delta_set or [])
            _delta_set.add(_eid)
            _mcp._STATE.active_intent["state_deltas_entity_set"] = _delta_set
            if _status == "irrelevant":
                _irr_set = _mcp._STATE.active_intent.get("irrelevant_state_set")
                if not isinstance(_irr_set, set):
                    _irr_set = set(_irr_set or [])
                _irr_set.add(_eid)
                _mcp._STATE.active_intent["irrelevant_state_set"] = _irr_set
            if _status == "changed" and _mcp._STATE.kg is not None:
                _patch = _d.get("patch")
                if not isinstance(_patch, list) or not _patch:
                    return {
                        "success": False,
                        "error": (
                            f"state_deltas[{_i}].patch required as "
                            "non-empty RFC 6902 list when status=changed."
                        ),
                    }
                try:
                    import jsonpatch as _jp

                    _current = _mcp._STATE.kg.latest_state_for_entity(_eid) or {}
                    _new_payload = _jp.apply_patch(_current, _patch)
                    _mcp._STATE.kg.record_state_revision(
                        entity_id=_eid,
                        schema_id=(_d.get("schema_id") or ""),
                        payload=_new_payload,
                        op_context_id="",
                        agent=agent or "",
                    )
                except ImportError:
                    return {
                        "success": False,
                        "error": ("jsonpatch lib required for status=changed."),
                    }
                except Exception as _err:  # pragma: no cover - defensive
                    return {
                        "success": False,
                        "error": (f"state_deltas[{_i}] patch apply failed: {_err}"),
                    }
        _persist_active_intent()

    # State-protocol v1 (Adrian Option B 2026-05-03): require
    # state_deltas coverage for surfaced state-bearing entities. An
    # entity is state-bearing when its is_a class carries
    # state_updatable=True (Task, agent, intent_type today; the set is
    # discovered via SQL rather than hardcoded so future seed additions
    # work automatically). Per the v2 design lock irrelevant-relief
    # rule, marking an entity 'irrelevant' in state_deltas removes it
    # from the expected set for the rest of the intent. The kill-switch
    # env MEMPALACE_STATE_DELTA_DISABLED=1 disables this entire block;
    # the per-op plumbing (state_deltas accumulation in declare_operation)
    # still runs so deltas can be observed in tests + telemetry.
    _pending_missing_state_deltas: list = []
    _state_delta_kill_switch = bool(os.environ.get("MEMPALACE_STATE_DELTA_DISABLED"))
    if not _state_delta_kill_switch:
        try:
            _delta_set = _mcp._STATE.active_intent.get("state_deltas_entity_set") or set()
            if not isinstance(_delta_set, set):
                _delta_set = set(_delta_set or [])
            _irr_set = _mcp._STATE.active_intent.get("irrelevant_state_set") or set()
            if not isinstance(_irr_set, set):
                _irr_set = set(_irr_set or [])
            # Adrian directive 2026-05-11 (judge-gated coverage):
            # finalize coverage requires state_deltas ONLY for entities
            # the state_judge flags at finalize time. The prior surfaced-
            # instances scan + agent/ctx always-cover are GONE -- if
            # nothing moved, the judge says nothing and no coverage is
            # demanded. `unchanged` is exclusively a judge-override.
            _state_bearing_accessed = set()
            _judge_changes_finalize: list = []
            _judge_report_finalize = None
            try:
                _agent_id_for_judge = _mcp._STATE.active_intent.get("agent") or ""
                _ctx_id_for_judge = (
                    _mcp._STATE.active_intent.get("intent_context_id")
                    or _mcp._STATE.active_intent.get("active_context_id")
                    or ""
                )
                _followed_finalize: list = []
                if _agent_id_for_judge and _mcp._STATE.kg is not None:
                    try:
                        _ag_state = _mcp._STATE.kg.latest_state_for_entity(_agent_id_for_judge)
                    except Exception:
                        _ag_state = None
                    _followed_finalize.append(
                        {
                            "entity_id": _agent_id_for_judge,
                            "state_schema_id": "agent_state",
                            "current_state": _ag_state or {},
                        }
                    )
                if _ctx_id_for_judge and _mcp._STATE.kg is not None:
                    try:
                        _cx_state = _mcp._STATE.kg.latest_state_for_entity(_ctx_id_for_judge)
                    except Exception:
                        _cx_state = None
                    _followed_finalize.append(
                        {
                            "entity_id": _ctx_id_for_judge,
                            "state_schema_id": "intent_state",
                            "current_state": _cx_state or {},
                        }
                    )
                _it_type = _mcp._STATE.active_intent.get("intent_type") or "?"
                _it_slug = slug or "?"
                _transcript_finalize = (
                    f"finalizing intent_type: {_it_type}\n"
                    f"slug: {_it_slug}\n"
                    f"outcome: {outcome}\n"
                    f"summary.what: {(summary or {}).get('what', '?')}\n"
                    f"summary.why: {(summary or {}).get('why', '?')}\n"
                    f"content (first 1000 chars): {(content or '')[:1000]}\n"
                )
                from .injection_gate import run_state_judge as _run_state_judge

                _judge_changes_finalize, _judge_report_finalize = _run_state_judge(
                    transcript_text=_transcript_finalize,
                    entity_states=_followed_finalize,
                    agent=agent,
                )
                for _change in _judge_changes_finalize:
                    _flagged_id = (_change.get("entity_id") or "").strip()
                    if _flagged_id:
                        _state_bearing_accessed.add(normalize_entity_name(_flagged_id))
            except Exception:
                # fail-open
                _judge_changes_finalize = []
                _judge_report_finalize = None
            # Map back from normalized to raw id for the missing list.
            _expected = _state_bearing_accessed - _irr_set
            _covered_norm = {normalize_entity_name(eid) for eid in _delta_set}
            _missing = _expected - _covered_norm
            if _missing:
                _pending_missing_state_deltas = _enrich_ids_with_summaries(sorted(_missing))
        except Exception:  # pragma: no cover - defensive; never block finalize on a bug here
            pass

    # v3.5.0 (2026-05-14): the accessed-memories 100%-coverage gate
    # is retired. Surfaced ids land on contexts_touched_detail and
    # accessed_memory_ids; the async-Haiku rater (mempalace.feedback_auto)
    # rates them post-finalize via rated_useful / rated_irrelevant edges.

    # ── Read execution trace from hook state file ──
    trace_entries = []
    if not _mcp._STATE.session_id:
        # No sid means we never had a private trace file. Skipping is
        # correct -- falling back to execution_trace_default.jsonl would
        # pull another agent's trace into THIS agent's finalize.
        trace_file = None
    else:
        trace_file = _mcp._INTENT_STATE_DIR / f"execution_trace_{_mcp._STATE.session_id}.jsonl"
    try:
        if trace_file and trace_file.exists():
            with open(trace_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        trace_entries.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        pass
            # Clear trace file after reading
            trace_file.write_text("", encoding="utf-8")
    except Exception:
        pass

    # Auto-fill key_actions from trace if not provided
    if not key_actions and trace_entries:
        key_actions = [f"{e['tool']} {e.get('target', '')}".strip() for e in trace_entries[-20:]]

    # v3.5.0 (2026-05-14): operation_ratings coverage gate is retired
    # along with the agent-side ratings parameter. The async-Haiku rater
    # (mempalace.feedback_auto) emits performed_well / performed_poorly
    # edges from observation post-finalize, no agent input needed.

    # ── Create execution entity ──
    # Full description stored in SQLite (for display)
    # Execution-entity description shows the distilled summary directly --
    # summary is already ≤280 chars by construction, no slicing needed.
    exec_description = f"{intent_desc or intent_type}: {summary}"
    # Embedding uses description-only (no summary) so similar intents cluster
    embed_description = intent_desc or intent_type
    # Cold-start lock 2026-05-01: persist a structured {what, why, scope?}
    # summary on the execution entity. The string `summary` arg is the
    # caller's distilled outcome-line (≤280 chars); it becomes the WHY.
    # WHAT is the intent + slug pair (specific noun phrase, discriminative
    # by construction so the gate's identity layer separates similar
    # intent runs). SCOPE pins the outcome and date so retrieval can
    # filter "successful executions of intent X in the last week".
    _exec_what = f"{intent_type or 'intent'} execution: {slug}"[:200]
    if len(_exec_what) < 8:  # extremely short slugs need padding
        _exec_what = f"{_exec_what} (intent execution entity)"
    _exec_summary_dict = {
        "what": _exec_what,
        "why": (summary or exec_description)[:240],
        "scope": (
            f"outcome={outcome}; agent={agent}; finalized={datetime.now().strftime('%Y-%m-%d')}"
        )[:100],
    }
    # Cold-start lock 2026-05-01 (no back-compat): single-shot validation.
    # If the rendered prose exceeds the 280-char budget, the caller's
    # finalize_intent.summary is too long for the structured summary
    # contract -- surface that as an error so the caller fixes it. No
    # multi-tier retry; no progressive degradation.
    from .knowledge_graph import (
        SummaryStructureRequired,
        coerce_summary_for_persist,
    )

    try:
        _exec_summary_dict = coerce_summary_for_persist(
            _exec_summary_dict,
            context_for_error=f"finalize_intent({slug!r}).execution_summary",
        )
    except SummaryStructureRequired as exc:
        return {
            "success": False,
            "error": (
                f"Cannot persist structured execution summary for intent "
                f"{slug!r}: {exc!s}. Trim the `summary` argument so the "
                f"rendered ``what -- why; scope`` form fits within "
                f"280 chars after ASCII-fold."
            ),
        }
    try:
        _mcp._create_entity(
            exec_id,
            kind="entity",
            content=exec_description,
            importance=3,
            properties={
                "outcome": outcome,
                "agent": agent,
                "added_by": agent,
                "intent_type": intent_type,
                "finalized_at": datetime.now().isoformat(),
                "summary": _exec_summary_dict,
            },
            added_by=agent,
            embed_text=embed_description,  # description-only, no summary
        )
    except Exception as e:
        return {"success": False, "error": f"Failed to create execution entity: {e}"}

    # ── KG relationships ──
    edges_created = []
    # Shared errors list -- initialized here so the S1 operation-rating
    # promotion (which runs before the result-memory block) can append
    # without a NameError. The result-memory block below re-uses this
    # same list, so duplicate init is avoided.
    errors: list = []

    # is_a → intent type (entity is_a class = instantiation)
    try:
        _mcp._STATE.kg.add_triple(exec_id, "is_a", intent_type)
        edges_created.append(f"{exec_id} is_a {intent_type}")
    except Exception:
        pass

    # executed_by → agent
    try:
        _mcp._STATE.kg.add_triple(exec_id, "executed_by", agent)
        edges_created.append(f"{exec_id} executed_by {agent}")
    except Exception:
        pass

    # targeted → slot entities
    for target in slot_entities:
        try:
            target_id = normalize_entity_name(target)
            _mcp._STATE.kg.add_triple(exec_id, "targeted", target_id)
            edges_created.append(f"{exec_id} targeted {target_id}")
        except Exception:
            pass

    # outcome as has_value -- unskipped 2026-04-25 (see _TRIPLE_SKIP_PREDICATES
    # comment for rationale). The statement verbalises the value pair so the
    # triple becomes a first-class search target ("intent X concluded with
    # outcome success/partial/failed/abandoned"), which is exactly the lookup
    # future agents make when auditing past intents by outcome.
    #
    # Cold-start lock 2026-05-01 (Adrian's congruence audit): the
    # outcome literal MUST exist as an entities-table row before the
    # has_value edge can land -- add_triple no longer phantom-creates
    # missing endpoints. Idempotent upsert (kind='literal') so repeat
    # finalizes don't drift the row.
    try:
        _mcp._STATE.kg.add_entity(
            outcome,
            kind="literal",
            content=f"intent outcome value: {outcome}",
            importance=3,
        )
    except Exception:
        pass
    try:
        _hv_stmt = f"Intent execution {exec_id} concluded with outcome {outcome}"
        _mcp._STATE.kg.add_triple(exec_id, "has_value", outcome, statement=_hv_stmt)
        edges_created.append(f"{exec_id} has_value {outcome}")
    except Exception:
        pass

    # ── caused_by edge from execution entity to parent cause ──
    # When declare_intent stashed cause_id / cause_kind on active_intent
    # (path: optional parent linkage to a user-context or Task
    # entity), the execution entity inherits that linkage so future audits
    # can trace activity-tier executions back to the user message that
    # provoked them. caused_by is non-skip-list so add_triple requires a
    # natural-language statement (per the 2026-04-19 lock that retired
    # autogenerated verbalisations). Soft-fail at edge level so a transient
    # kg issue does not prevent finalization itself.
    _cause_id_for_edge = _mcp._STATE.active_intent.get("cause_id") or ""
    _cause_kind_for_edge = _mcp._STATE.active_intent.get("cause_kind") or ""
    if _cause_id_for_edge:
        _cb_stmt = (
            f"Execution {exec_id} was caused by "
            f"{_cause_kind_for_edge.replace('_', ' ') or 'parent context'} "
            f"{_cause_id_for_edge} per the user-intent tier hierarchy."
        )
        try:
            _mcp._STATE.kg.add_triple(
                exec_id,
                "caused_by",
                _cause_id_for_edge,
                statement=_cb_stmt,
            )
            edges_created.append(f"{exec_id} caused_by {_cause_id_for_edge}")
        except Exception:
            pass

    # v3.5.0 (2026-05-14): the S1 agent-rated op-promotion loop is gone.
    # mempalace.feedback_auto submits a Haiku rater batch per operation
    # at the end of this function; kg.record_operation_rating writes the
    # performed_well / performed_poorly edges out-of-band.

    # ── Result memory (summary) ──
    # silent-failure surface: when _add_memory_internal rejects the
    # call (e.g. agent not declared, duplicate slug), we used to swallow
    # the error and return result_memory=null with no indication. Now
    # every failure is appended to `errors` and surfaced in the response.
    # (errors list initialized above, shared with S1 operation promotion.)
    result_memory_id = None
    try:
        # Result memory: the body is the agent's `content` wrapped with an
        # intent/outcome header; the ≤280-char distilled summary is the
        # agent's `summary` verbatim. No slicing, no auto-derivation -- the
        # summary-first contract requires the caller to have produced a
        # real distillation, and we honor it here.
        _result_body = f"## {intent_type}: {intent_desc}\n\n**Outcome:** {outcome}\n\n{content}"
        result = _mcp._add_memory_internal(
            content=_result_body,
            slug=f"result-{exec_id}",
            added_by=agent,
            content_type="event",
            importance=3,
            entity=exec_id,
            predicate="resulted_in",
            summary=_summary_dict,
        )
        if result.get("success"):
            result_memory_id = result.get("memory_id")
            edges_created.append(f"{exec_id} resulted_in {result_memory_id}")
        else:
            errors.append({"kind": "result_memory", "error": result.get("error", "unknown")})
    except Exception as e:
        errors.append({"kind": "result_memory", "error": f"exception: {e}"})

    # ── Trace memory ── (retired 2026-04-22)
    # Traces used to be filed as ``record_ga_agent_trace_<slug>`` prose
    # memories with importance=2 and a count-of-tool-calls summary. They
    # polluted retrieval -- every finalize added another "Trace of X: N
    # tool call(s)" hit competing with actual prose. The same information
    # is already available without the embedded memory:
    #   - execution_trace_<sid>.jsonl on disk (the raw tool-call log,
    #     cleared after finalize reads it)
    #   - key_actions on the execution entity (distilled tool+target
    #     list, auto-filled from the trace above)
    #   - edges on the execution entity (executed_by, targeted, is_a)
    # If you ever need the blow-by-blow, read the JSONL file between
    # finalizes -- do not re-introduce a prose memory for it.

    # ── Gotchas ──
    # Strict dict-only contract (Adrian's design lock 2026-04-28):
    # each gotcha is {summary: {what, why, scope?}, content: str}.
    # The summary is rendered to prose for the entity description
    # (validated at source -- no auto-derive of summary from content),
    # and the content is preserved verbatim in entity.properties for
    # full retrieval. Strings are rejected with a migration error.
    if gotchas:
        try:
            from .knowledge_graph import (
                SummaryStructureRequired,
                coerce_summary_for_persist,
                serialize_summary_for_embedding,
            )
        except Exception:
            coerce_summary_for_persist = None
            serialize_summary_for_embedding = None
            SummaryStructureRequired = Exception
        for i, gotcha in enumerate(gotchas):
            try:
                if not isinstance(gotcha, dict):
                    errors.append(
                        {
                            "kind": "gotcha_entity",
                            "index": i,
                            "error": (
                                "gotcha must be "
                                "dict{summary: {what, why, scope?}, "
                                "content: str}; got "
                                f"{type(gotcha).__name__}. "
                                "Strings are rejected -- Adrian's design "
                                "lock 2026-04-28 forbids auto-derive of "
                                "summary from content."
                            ),
                        }
                    )
                    continue
                _g_content = str(gotcha.get("content") or "").strip()
                _g_summary = gotcha.get("summary")
                if not _g_content:
                    errors.append(
                        {
                            "kind": "gotcha_entity",
                            "index": i,
                            "error": (
                                f"gotcha[{i}].content is empty. Provide the verbatim gotcha body."
                            ),
                        }
                    )
                    continue
                if not isinstance(_g_summary, dict):
                    errors.append(
                        {
                            "kind": "gotcha_entity",
                            "index": i,
                            "error": (
                                f"gotcha[{i}].summary must be a dict "
                                f"{{what, why, scope?}}; got "
                                f"{type(_g_summary).__name__}. No "
                                f"auto-derive -- caller authors the "
                                f"WHAT and WHY."
                            ),
                        }
                    )
                    continue
                # Validate the summary dict via the shared gate so we
                # surface the same error messages the rest of the
                # write surface uses (field-level + 280-char rendered
                # cap).
                try:
                    _g_summary_dict = (
                        coerce_summary_for_persist(
                            _g_summary,
                            context_for_error=f"finalize.gotchas[{i}].summary",
                        )
                        if coerce_summary_for_persist
                        else _g_summary
                    )
                except SummaryStructureRequired as _se:
                    errors.append(
                        {
                            "kind": "gotcha_entity",
                            "index": i,
                            "error": str(_se),
                        }
                    )
                    continue
                _g_prose = (
                    serialize_summary_for_embedding(_g_summary_dict)
                    if serialize_summary_for_embedding
                    else str(_g_summary_dict)
                )
                if len(_g_prose) > _mcp._RECORD_SUMMARY_MAX_LEN:
                    errors.append(
                        {
                            "kind": "gotcha_entity",
                            "index": i,
                            "error": (
                                f"gotcha[{i}].summary rendered prose is "
                                f"{len(_g_prose)} chars; maximum is "
                                f"{_mcp._RECORD_SUMMARY_MAX_LEN}. Trim "
                                f"'why' or 'scope' so the prose form "
                                f"fits the embedding budget."
                            ),
                        }
                    )
                    continue
                # Entity name derives from summary.what (first 50 chars
                # normalised) -- this is the structured anchor of the
                # gotcha. content is verbatim narrative; we store it in
                # properties so future retrieval can pull the full body
                # without losing it to the 280-char prose cap.
                _g_what = str(_g_summary_dict.get("what") or "").strip()
                gotcha_id = normalize_entity_name(_g_what[:50])
                if not gotcha_id:
                    errors.append(
                        {
                            "kind": "gotcha_entity",
                            "index": i,
                            "error": (
                                f"gotcha[{i}].summary.what normalises "
                                f"to empty; provide a meaningful WHAT."
                            ),
                        }
                    )
                    continue
                existing = _mcp._STATE.kg.get_entity(gotcha_id)
                if not existing:
                    _mcp._create_entity(
                        gotcha_id,
                        kind="entity",
                        content=_g_prose,
                        importance=3,
                        properties={
                            "summary": _g_summary_dict,
                            "content": _g_content,
                        },
                        added_by=agent,
                    )
                # has_gotcha is NOT a skip predicate; we need a real
                # statement so the edge is searchable. The rendered
                # summary prose serves as the natural sentence.
                gotcha_sentence = f"Execution {exec_id} ran into this gotcha: {_g_prose}"
                _mcp._STATE.kg.add_triple(
                    exec_id,
                    "has_gotcha",
                    gotcha_id,
                    statement=gotcha_sentence,
                )
                edges_created.append(f"{exec_id} has_gotcha {gotcha_id}")
                if promote_gotchas_to_type:
                    type_sentence = (
                        f"Intent type '{intent_type}' has a recurring gotcha: {_g_prose}"
                    )
                    _mcp._STATE.kg.add_triple(
                        intent_type,
                        "has_gotcha",
                        gotcha_id,
                        statement=type_sentence,
                    )
                    edges_created.append(f"{intent_type} has_gotcha {gotcha_id}")
            except Exception as e:
                errors.append(
                    {
                        "kind": "gotcha_entity",
                        "index": i,
                        "error": f"exception: {e}",
                    }
                )

    # ── Learnings ──
    # Strict dict-only contract (Adrian's design lock 2026-04-28):
    # each learning is {summary: {what, why, scope?}, content: str}.
    # No string fallback -- strings used to be accepted with an
    # auto-derived summary dict (what="learning N of <type>",
    # why=<the string>, scope=<exec_id>), which violated the
    # no-auto-derive rule and overflowed the 280-char rendered cap
    # whenever the string was long. Caller now passes the structured
    # summary upfront; we forward it verbatim to _add_memory_internal.
    if learnings:
        for i, learning in enumerate(learnings):
            try:
                if not isinstance(learning, dict):
                    errors.append(
                        {
                            "kind": "learning_memory",
                            "index": i,
                            "error": (
                                "learning must be "
                                "dict{summary: {what, why, scope?}, "
                                "content: str}; got "
                                f"{type(learning).__name__}. "
                                "Strings are rejected -- Adrian's design "
                                "lock 2026-04-28 forbids auto-derive of "
                                "summary from content."
                            ),
                        }
                    )
                    continue
                _l_content = str(learning.get("content") or "").strip()
                _l_summary = learning.get("summary")
                if not _l_content:
                    errors.append(
                        {
                            "kind": "learning_memory",
                            "index": i,
                            "error": (
                                f"learning[{i}].content is empty. Provide the verbatim lesson body."
                            ),
                        }
                    )
                    continue
                if not isinstance(_l_summary, dict):
                    errors.append(
                        {
                            "kind": "learning_memory",
                            "index": i,
                            "error": (
                                f"learning[{i}].summary must be a dict "
                                f"{{what, why, scope?}}; got "
                                f"{type(_l_summary).__name__}. No "
                                f"auto-derive -- caller authors the "
                                f"WHAT and WHY."
                            ),
                        }
                    )
                    continue
                # Caller-provided summary dict passes through verbatim;
                # _add_memory_internal validates it via
                # coerce_summary_for_persist + the 280-char rendered
                # prose cap. Failures surface as record errors with
                # a clear field-level message -- no auto-derive at any
                # layer between caller and gate.
                learning_result = _mcp._add_memory_internal(
                    content=_l_content,
                    slug=f"learning-{exec_id}-{i}",
                    added_by=agent,
                    content_type="discovery",
                    importance=4,
                    entity=exec_id,
                    predicate="evidenced_by",
                    summary=_l_summary,
                )
                if not learning_result.get("success"):
                    errors.append(
                        {
                            "kind": "learning_memory",
                            "index": i,
                            "error": learning_result.get("error", "unknown"),
                        }
                    )
            except Exception as e:
                errors.append({"kind": "learning_memory", "index": i, "error": f"exception: {e}"})

    # v3.5.0 (2026-05-14): the strict coverage validator + the
    # agent-rated memory_feedback writer loop are retired. Memory
    # ratings are now emitted by mempalace.feedback_auto's Haiku rater
    # post-finalize -- the rater consumes contexts_touched_detail's
    # per-emit surfaced_ids and writes rated_useful / rated_irrelevant
    # via kg.record_feedback(rater_kind='haiku_auto').
    _contexts_touched = list(_mcp._STATE.active_intent.get("contexts_touched") or [])
    _active_ctx_id = _mcp._STATE.active_intent.get("active_context_id", "") or ""

    # v3.5.0 (2026-05-14): the agent-rated memory_feedback writer loop
    # is retired. mempalace.feedback_auto's Haiku rater writes the
    # rated_useful / rated_irrelevant edges + last_relevant_at decay
    # reset post-finalize via kg.record_feedback(rater_kind='haiku_auto').
    feedback_count = 0

    # v3.5.0 (2026-05-14): the Link-prediction Adamic-Adar candidate
    # accumulator is retired. It depended on the synchronous per-context
    # memory_feedback signal; the async Haiku rater writes ratings
    # out-of-band so this loop has nothing to filter on. If the
    # link-author candidate channel matters in v3.6+, re-introduce it
    # inside mempalace.feedback_auto's persist path (after Haiku rates).

    # ── Finalize-triggered background dispatch (stub in Commit 2) ──
    # Default interval is 1 hour; operators can opt into aggressive
    # mode via MEMPALACE_LINK_AUTHOR_AGGRESSIVE=1 to dispatch on
    # every finalize that has pending candidates. Closes 2026-04-25
    # audit finding #14 (jury throughput bottleneck once the gardener
    # max_batches=10 change starts pumping candidates faster).
    try:
        import os as _os

        from . import link_author as _la  # noqa: F811

        _aggressive = _os.environ.get("MEMPALACE_LINK_AUTHOR_AGGRESSIVE", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        _la._dispatch_if_due(_mcp._STATE.kg, interval_hours=0 if _aggressive else 1)
    except Exception:
        pass

    # v3.5.0 (2026-05-14): Rocchio enrichment, per-channel scoring-
    # feedback, and weight-learning ride on the synchronous agent
    # memory_feedback signal that is now retired. The async Haiku
    # rater (mempalace.feedback_auto) writes rated_useful /
    # rated_irrelevant edges out-of-band; future versions can revive
    # these post-rate hooks inside feedback_auto's persist path if
    # the experimental data warrants it.

    # Feedback context vectors (store_feedback_context), edge-traversal
    # feedback (record_edge_feedback), and keyword-suppression feedback
    # (record_keyword_suppression / reset_keyword_suppression) are all
    # RETIRED in the P3 polish sweep. Their signals now live on:
    #   - rated_useful / rated_irrelevant edges → Channel D retrieval +
    #     hybrid_score's signed W_REL term.
    #   - keyword_idf table → BM25-IDF dampens dominant keywords
    #     channel-wide (replacing per-memory suppression).
    #   - created_under / similar_to context edges → Channel D's
    #     neighbourhood expansion replaces edge-usefulness gating.
    # This block used to reach for all three retired APIs.

    # v3.5.0 (2026-05-14): the two-tool / extend_feedback parking
    # branch is gone. Finalize is atomic now -- coverage gates retired,
    # async-Haiku rater fills ratings out-of-band. The state_judge's
    # findings still surface on the success response below.

    # ── register cause_id in rated_user_contexts ──
    # The intent finalized successfully under cause_kind='user_context';
    # add cause_id to the session-scoped rated set so the NEXT agent
    # intent declared with the same cause_id inherits the coverage and
    # skips the user-context-surfaced memories. Skip on cause_kind=='task'
    # (Task entities have no surfaced-memory inheritance contract) and
    # on no-cause intents. Read AT FINALIZE because active_intent gets
    # cleared next.
    _final_cause_id = _mcp._STATE.active_intent.get("cause_id") or ""
    _final_cause_kind = _mcp._STATE.active_intent.get("cause_kind") or ""
    if _final_cause_kind == "user_context" and _final_cause_id:
        try:
            _rated_user_contexts_for(
                _mcp._STATE.session_id or "",
            ).add(_final_cause_id)
        except Exception:
            pass

    # ── v3.5.0: async-Haiku rater fire-and-forget ──
    # Submits per-emit Haiku-rater batches that fill rated_useful /
    # rated_irrelevant + performed_well / performed_poorly edges out of
    # band. Failures land in feedback_auto_log.jsonl; never raises here.
    # Plumbed BEFORE deactivation so the active_intent state is still
    # populated when we read contexts_touched_detail.
    _auto_submitted = 0
    try:
        from . import feedback_auto as _fb_auto

        _ai_for_auto = _mcp._STATE.active_intent or {}
        _intent_context_id = _ai_for_auto.get("intent_context_id") or ""
        # Render intent_context prose from the active intent's content
        # field (which carries "what -- why" already). When empty, the
        # rater handles "(none provided)" gracefully.
        _intent_context_prose = str(_ai_for_auto.get("content") or "")[:280]

        # Walk contexts_touched_detail and partition per scope. The
        # intent-level entry's surfaced_ids feed intent_memories; each
        # operation entry pairs with its pending_operation_cue (tool +
        # args_summary lookup via op_args_by_ctx_tool); each search
        # entry ships a search batch.
        _detail = _ai_for_auto.get("contexts_touched_detail") or []
        _op_args_store = _ai_for_auto.get("op_args_by_ctx_tool") or {}
        _intent_memories: list[dict] = []
        _op_batches: list[dict] = []
        _search_batches: list[dict] = []

        # pending_operation_cues records (tool, args_summary, ctx_id)
        # per declare_operation call. Build a lookup so detail-entry
        # operation scopes can resolve their tool quickly.
        _cue_tool_by_ctx: dict[str, str] = {}
        for _cue in _ai_for_auto.get("pending_operation_cues") or []:
            if not isinstance(_cue, dict):
                continue
            _cctx = _cue.get("active_context_id") or ""
            _ctool = _cue.get("tool") or ""
            if _cctx and _ctool and _cctx not in _cue_tool_by_ctx:
                _cue_tool_by_ctx[_cctx] = _ctool

        for _e in _detail:
            if not isinstance(_e, dict):
                continue
            _ids = _e.get("surfaced_ids") or []
            if not _ids:
                continue
            _scope = _e.get("scope") or ""
            _ctx = _e.get("ctx_id") or ""
            _mem_dicts = [{"id": _mid, "source": "memory"} for _mid in _ids if _mid]
            if _scope == "intent":
                _intent_memories.extend(_mem_dicts)
            elif _scope == "operation" and _ctx:
                _tool = _cue_tool_by_ctx.get(_ctx, "")
                _args = str(_op_args_store.get(f"{_ctx}|{_tool}", "")) if _tool else ""
                _op_batches.append(
                    {
                        "context_id": _ctx,
                        "tool": _tool,
                        "args_summary": _args,
                        "memories": _mem_dicts,
                    }
                )
            elif _scope == "search" and _ctx:
                _search_batches.append({"context_id": _ctx, "memories": _mem_dicts})

        _auto_submitted = _fb_auto.submit_finalize_feedback(
            intent_exec_id=exec_id,
            agent=agent or "",
            intent_context_prose=_intent_context_prose,
            intent_context_id=_intent_context_id,
            intent_memories=_intent_memories,
            op_batches=_op_batches,
            search_batches=_search_batches,
            kg=_mcp._STATE.kg,
        )
    except Exception:
        _auto_submitted = 0

    # ── Deactivate intent ──
    _mcp._STATE.active_intent = None
    _persist_active_intent()

    # ── Write last-finalized marker for Stop-hook proof-of-done check ──
    # The never-stop rule requires the Stop hook to see that the LAST finalized
    # intent in this session was a wrap_up_session with outcome=success before
    # allowing a stop. Writing a session-scoped marker here gives the dep-free
    # hook a file to read without needing SQLite or Chroma. Best-effort -- any
    # error is non-fatal to the finalize itself.
    try:
        sid = _mcp._STATE.session_id or ""
        if sid:
            marker_path = _mcp._INTENT_STATE_DIR / f"last_finalized_{sid}.json"
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text(
                json.dumps(
                    {
                        "intent_type": intent_type,
                        "execution_entity": exec_id,
                        "outcome": outcome,
                        "agent": agent,
                        "ts": datetime.now().isoformat(),
                    }
                ),
                encoding="utf-8",
            )
    except Exception as _e:
        # NEVER silent: a failed marker write means the Stop hook's
        # never-stop rule will block the next stop attempt forever (reads
        # missing/stale marker as "no wrap-up proof"). Record for
        # SessionStart to surface.
        try:
            from . import hooks_cli as _hc

            _hc._record_hook_error("tool_finalize_intent.last_finalized_marker", _e)
        except Exception:
            pass

    result = {
        "success": True,
        "execution_entity": exec_id,
        "outcome": outcome,
        "edges_created_count": len(edges_created),
        "trace_entries": len(trace_entries),
        "result_memory": result_memory_id,
        "feedback_count": feedback_count,
        # v3.5.0: number of async-Haiku rater batches submitted; rated
        # edges land asynchronously via mempalace.feedback_auto. 0 when
        # the env flag MEMPALACE_FEEDBACK_AUTO_DISABLED=1 is set or no
        # memories were surfaced this intent.
        "auto_feedback_submitted": _auto_submitted,
    }

    # ── Memory-gardener detached spawn ──
    # If the injection gate has accumulated enough quality flags on
    # memory_flags, kick off a gardener subprocess so the finalize
    # caller isn't blocked on Claude Code latency. Mirrors the
    # link-author finalize-triggered detached pattern. Fail-silent:
    # a spawn failure must not block finalize.
    try:
        from . import memory_gardener as _mg

        _mg.maybe_trigger_from_finalize(_mcp._STATE.kg)
    except Exception:
        pass

    # ── P3 telemetry: finalize trace for mempalace-eval ──
    try:
        from datetime import timezone as _tz

        contexts_used = sorted(set(_contexts_touched or []))
        if _active_ctx_id and _active_ctx_id not in contexts_used:
            contexts_used.append(_active_ctx_id)
        _mcp._telemetry_append_jsonl(
            "finalize_log.jsonl",
            {
                "ts": datetime.now(_tz.utc).isoformat(timespec="seconds"),
                "intent_id": exec_id,
                "contexts_used": contexts_used,
                "memories_rated": feedback_count,
                "outcome": outcome,
                "agent": agent or "",
            },
        )
    except Exception:
        pass
    if errors:
        result["errors"] = errors
        result["warning"] = (
            f"{len(errors)} side-memory creation(s) failed silently before "
            "see 'errors' for details. The execution entity itself was created and "
            "feedback/gotchas were recorded; only the filed memories were affected."
        )
    return result
