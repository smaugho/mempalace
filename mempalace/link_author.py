"""
link_author.py — Two-layer link-authoring pipeline.

Layer 1 (analytical, in-session, Commit 2)
==========================================

``upsert_candidate`` / ``list_pending`` accumulate Adamic-Adar evidence
on entity pairs that co-appear inside reused, positively-rated
contexts. Adamic-Adar weighting (Adamic & Adar 2003, Liben-Nowell &
Kleinberg 2007): for pair (A, B), add ``1 / log(|ctx.entities|)`` once
per DISTINCT context where both co-appear. Distinct-context dedup via
``link_prediction_sources`` (Option A per plan §2.2): re-observing the
same context N times contributes exactly once.

Layer 2 (LLM-authored, out-of-session, Commit 3)
=================================================

Per candidate, a three-stage Anthropic-API pipeline decides whether a
real edge exists, which predicate fits, and what the statement says:

  Stage 1 (Opus):     designs 3 domain-appropriate juror personas
                      from a "domain hint" built from the candidate's
                      top shared contexts.
  Stage 2 (Haiku×3):  runs each persona in parallel via asyncio.gather
                      against the candidate's entity/context payload.
  Stage 3 (Haiku):    synthesises the 3 juror verdicts into a final
                      decision (edge / no_edge / uncertain), picks the
                      predicate, and writes the statement.

Only LLM-accepted edges land in the KG (via ``kg.add_triple``). The
analytical layer is a filter, not an author.

Research grounding: Liben-Nowell & Kleinberg 2007 JASIST (topology
predicts links 30× better than random, Adamic-Adar beats common
neighbours); Shu et al. 2024 arXiv:2403.07311 (LLM predicate selection
beats classical); Du et al. 2024 (jury debate improves factuality);
Graphiti 2024 (LLM-only edge authoring in agent KGs).

Single-vendor Anthropic. No Claude-CLI subprocess. No OpenAI. No other
providers. Embeddings for domain-hint clustering reuse ChromaDB's
built-in all-MiniLM-L6-v2. See docs/link_author_plan.md §2.9.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Layer 1 — in-session accumulator (Commit 2)
# ═══════════════════════════════════════════════════════════════════════


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def upsert_candidate(
    kg,
    from_entity: str,
    to_entity: str,
    weight: float,
    context_id: str,
) -> bool:
    """Accumulate one context's Adamic-Adar contribution to a pair's score.

    Canonicalises (from_entity, to_entity) so (A, B) and (B, A) hit the
    same row, then dedups by ctx_id: if this context has already
    contributed to this pair, does nothing and returns False.
    Otherwise inserts into ``link_prediction_sources`` (the dedup table),
    upserts into ``link_prediction_candidates`` adding ``weight`` to the
    score and incrementing ``shared_context_count``, and returns True.

    Direct-edge short-circuit: if a 1-hop edge already exists between
    the pair (either direction, any predicate, ``valid_to IS NULL``), the
    upsert is skipped. The jury's job is discovering NEW edges; the
    graph channel already surfaces directly-connected pairs together,
    so a candidate suggestion there would be pure noise.

    Raises nothing on the happy path. Any persistence error is
    propagated so finalize can record it — but the caller (see
    intent.tool_finalize_intent) wraps the whole upsert block in a
    try/except to keep finalize robust against DB failures.
    """
    if not from_entity or not to_entity:
        return False
    if from_entity == to_entity:
        return False
    if not context_id:
        return False

    # Canonical ordering keeps PK unique across symmetric pair perms.
    a, b = (from_entity, to_entity) if from_entity < to_entity else (to_entity, from_entity)

    conn = kg._conn()
    # Direct-edge skip. Cheap 1-row LIMIT query using the normal triples
    # table. Any exception bubbles up; the caller's try/except decides.
    row = conn.execute(
        "SELECT 1 FROM triples "
        "WHERE ((subject = ? AND object = ?) OR (subject = ? AND object = ?)) "
        "AND valid_to IS NULL "
        "LIMIT 1",
        (a, b, b, a),
    ).fetchone()
    if row is not None:
        return False

    now = _now_iso()
    # Dedup: atomic INSERT OR IGNORE on the sources table. If the
    # triple (pair, ctx_id) is already recorded, rowcount stays 0 and
    # we skip the candidate increment.
    cur = conn.execute(
        "INSERT OR IGNORE INTO link_prediction_sources "
        "(from_entity, to_entity, ctx_id, contributed_ts) "
        "VALUES (?, ?, ?, ?)",
        (a, b, context_id, now),
    )
    if cur.rowcount == 0:
        conn.commit()
        return False

    # Upsert the candidate row. SQLite 3.24+ supports ON CONFLICT DO
    # UPDATE. The candidate table's PK is (from_entity, to_entity),
    # so the conflict path handles both the "new pair" and the "nth
    # distinct context for existing pair" cases uniformly.
    conn.execute(
        "INSERT INTO link_prediction_candidates "
        "(from_entity, to_entity, score, shared_context_count, "
        " last_context_id, last_updated_ts) "
        "VALUES (?, ?, ?, 1, ?, ?) "
        "ON CONFLICT(from_entity, to_entity) DO UPDATE SET "
        "    score = score + excluded.score, "
        "    shared_context_count = shared_context_count + 1, "
        "    last_context_id = excluded.last_context_id, "
        "    last_updated_ts = excluded.last_updated_ts",
        (a, b, float(weight), context_id, now),
    )
    conn.commit()
    return True


def list_pending(kg, limit: int = 50, threshold: float = 1.5) -> list[dict]:
    """Return unprocessed candidates above ``threshold``, ordered by score desc.

    Rows where ``processed_ts IS NOT NULL`` are excluded — rejections
    and acceptances stay in the table for audit but are not re-processed
    here. Use ``status`` CLI subcommand to see recently processed rows.
    """
    conn = kg._conn()
    rows = conn.execute(
        "SELECT from_entity, to_entity, score, shared_context_count, "
        "       last_context_id, last_updated_ts "
        "FROM link_prediction_candidates "
        "WHERE processed_ts IS NULL AND score >= ? "
        "ORDER BY score DESC "
        "LIMIT ?",
        (float(threshold), int(limit)),
    ).fetchall()
    return [
        {
            "from_entity": r[0],
            "to_entity": r[1],
            "score": r[2],
            "shared_context_count": r[3],
            "last_context_id": r[4],
            "last_updated_ts": r[5],
        }
        for r in rows
    ]


# ═══════════════════════════════════════════════════════════════════════
# Layer 2 — LLM jury pipeline (Commit 3)
# ═══════════════════════════════════════════════════════════════════════

# Exit codes for CLI (documented in docs/link_author_scheduling.md so
# cron/systemd logs can distinguish failure modes without parsing stderr).
EXIT_OK = 0
EXIT_BAD_KEY = 2
EXIT_API_DOWN = 3

NEW_PREDICATES_LOG = Path(os.path.expanduser("~/.mempalace/hook_state/new_predicates.jsonl"))


# ─────────────────────────────────────────────────────────────────────
# Environment + API-key validation (plan §2.8)
# ─────────────────────────────────────────────────────────────────────


def _load_env(palace_path: str | None = None, dotenv_path: str | None = None) -> None:
    """Load ``<palace>/.env`` (or an explicit path) into ``os.environ``.

    No-op when the file doesn't exist — the key may come from the shell
    environment instead. Called once at ``process`` startup BEFORE any
    API client is constructed.

    Never logs the key value. Logs only "loaded N variables" with a
    count so the operator sees something happened without leaking the
    secret.
    """
    # python-dotenv is a hard dependency per pyproject.toml. Import at
    # call time so the module imports clean even if the optional dep is
    # somehow missing — the _validate_api_key step would fail loud in
    # that case anyway.
    try:
        from dotenv import load_dotenv
    except ImportError:
        log.warning("python-dotenv not installed; skipping .env load")
        return

    target = Path(dotenv_path) if dotenv_path else Path(palace_path or "") / ".env"
    if not target.is_file():
        log.info("no .env at %s; relying on shell environment", target)
        return
    # override=True: the .env file is the documented source of truth
    # for mempalace's ANTHROPIC_API_KEY. If the shell happens to have
    # an empty or stale ANTHROPIC_API_KEY set (from a profile script,
    # a prior `set` command, or an earlier failed run), we want the
    # file value to win rather than silently fail with "not set".
    # The silent-shadow failure mode caused a live panic 2026-04-22 —
    # see record_ga_agent_env_key_shell_shadowing_diagnostic.
    load_dotenv(str(target), override=True)
    log.info(".env loaded from %s", target)


def _mask_key(key: str) -> str:
    """Return a mask-safe representation of an API key — never the raw value.

    Shows length + last-4 characters. Used in log lines so operators can
    verify a key is present + sane without leaking it to logs/stdout.
    """
    if not key:
        return "<empty>"
    return f"len={len(key)} ends=...{key[-4:]}"


def _build_client(cfg: dict) -> tuple[Any, str]:
    """Sync half of API-key validation: present + format + construct client.

    Returns ``(client, key)``. Raises ``SystemExit(EXIT_BAD_KEY)`` with
    an actionable stderr message on missing / malformed key or missing
    SDK. The network ping lives in ``_ping_and_translate``; splitting
    like this lets callers in async contexts skip the nested
    ``asyncio.run`` that would otherwise trip ``_process_async``.
    """
    env_var = cfg.get("api_key_env", "ANTHROPIC_API_KEY")
    key = os.environ.get(env_var) or ""

    # 1. Present.
    if not key.strip():
        print(
            f"[link-author] {env_var} not set. Create a key at "
            "https://console.anthropic.com/settings/keys and add to "
            "<palace>/.env (ANTHROPIC_API_KEY=sk-ant-...). See "
            "docs/link_author_scheduling.md.",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_BAD_KEY)

    # 2. Format.
    if not key.startswith("sk-ant-"):
        print(
            f"[link-author] {env_var} does not look like a valid "
            "Anthropic key (should start with 'sk-ant-'). Check "
            "<palace>/.env.",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_BAD_KEY)

    # SDK available? Imported here so the module is importable without
    # the optional dep (useful for any test path that never touches
    # the LLM flow).
    try:
        import anthropic
    except ImportError as exc:
        print(
            f"[link-author] anthropic SDK not installed: {exc}. Run: pip install anthropic>=0.39.0",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_BAD_KEY) from exc

    client = anthropic.AsyncAnthropic(api_key=key)
    return client, key


async def _ping_and_translate(client, cfg: dict, key: str) -> None:
    """Async ping + translate anthropic errors to ``SystemExit`` codes.

    Auth / key shape problems → exit 2. Network / 5xx / anything else
    the API raises post-retry → exit 3. These are the two exit codes
    the scheduling doc advertises for cron/systemd log classification.
    """
    import anthropic

    env_var = cfg.get("api_key_env", "ANTHROPIC_API_KEY")
    try:
        await _ping_anthropic(client, cfg)
    except anthropic.AuthenticationError as exc:
        print(
            f"[link-author] {env_var} rejected by API (invalid or "
            "revoked). Generate a new key at "
            "https://console.anthropic.com and update <palace>/.env.",
            file=sys.stderr,
        )
        log.info("auth failure for key (%s): %s", _mask_key(key), exc)
        raise SystemExit(EXIT_BAD_KEY) from exc
    except anthropic.APIConnectionError as exc:
        print(
            "[link-author] Anthropic API unreachable. Check network and try again.",
            file=sys.stderr,
        )
        log.info("connection error on ping: %s", exc)
        raise SystemExit(EXIT_API_DOWN) from exc
    except anthropic.APIError as exc:
        # Anything 5xx or otherwise transient after SDK retries.
        print(
            f"[link-author] Anthropic API error during ping: {exc}. Try again later.",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_API_DOWN) from exc


def _validate_api_key(cfg: dict) -> Any:
    """Sync entry point: present + format + ping. Raises SystemExit.

    Called from the CLI (``cmd_linkauthor_process`` path) and from
    tests that verify all three checks end-to-end. ``_process_async``
    does NOT call this — it calls ``_build_client`` + awaits
    ``_ping_and_translate`` directly to avoid a nested ``asyncio.run``.
    """
    client, key = _build_client(cfg)
    asyncio.run(_ping_and_translate(client, cfg, key))
    log.info("api key validated (%s)", _mask_key(key))
    return client


async def _ping_anthropic(client, cfg: dict) -> None:
    """One-token ping to verify the key + connectivity. Raises on error."""
    model = cfg.get("jury_execution_model", "claude-haiku-4-5")
    await client.messages.create(
        model=model,
        max_tokens=1,
        messages=[{"role": "user", "content": "ping"}],
    )


# ─────────────────────────────────────────────────────────────────────
# Domain-hint building + batching (plan §2.5 Stage 1 input)
# ─────────────────────────────────────────────────────────────────────


def _build_domain_hint(kg, candidate: dict, max_contexts: int = 5) -> str:
    """Concatenate top-N shared contexts into a ~500-token domain blob.

    The blob is Opus's input in Stage 1 — it's what the model uses to
    design jurors tailored to this candidate's domain. Fields per
    plan §2.5: queries[:2], keywords[:5], dominant entity kinds.

    Contexts are selected by recency (``last_updated_ts``) via
    ``link_prediction_sources``, then read from the contexts table. If
    a context is missing (deleted, corrupt), it's silently skipped.
    """
    conn = kg._conn()
    a, b = candidate["from_entity"], candidate["to_entity"]
    rows = conn.execute(
        "SELECT ctx_id FROM link_prediction_sources "
        "WHERE from_entity = ? AND to_entity = ? "
        "ORDER BY contributed_ts DESC LIMIT ?",
        (a, b, int(max_contexts)),
    ).fetchall()
    ctx_ids = [r[0] for r in rows]

    parts: list[str] = []
    for cid in ctx_ids:
        try:
            ctx = kg.get_entity(cid)
        except Exception:
            continue
        if not ctx:
            continue
        props = ctx.get("properties") or {}
        if isinstance(props, str):
            try:
                props = json.loads(props)
            except Exception:
                props = {}
        qs = list(props.get("queries") or [])[:2]
        kws = list(props.get("keywords") or [])[:5]
        ents = list(props.get("entities") or [])
        # Dominant entity kinds over the context's entities.
        kinds: dict[str, int] = {}
        for eid in ents:
            try:
                ent = kg.get_entity(eid)
            except Exception:
                ent = None
            if not ent:
                continue
            k = ent.get("kind") or "entity"
            kinds[k] = kinds.get(k, 0) + 1
        kind_str = ", ".join(f"{k}×{v}" for k, v in sorted(kinds.items())) or "—"
        parts.append(f"- queries: {qs}\n  keywords: {kws}\n  entity_kinds: {kind_str}")
    if not parts:
        return f"(no shared context metadata for pair {a!r}—{b!r})"
    return f"pair: {a} — {b}\nshared contexts (most recent first):\n" + "\n".join(parts)


def _embed_domain_hint(palace_path: str, hint: str) -> list[float] | None:
    """Embed a string via ChromaDB's local embedding function.

    Uses the SAME model that mempalace uses everywhere else
    (all-MiniLM-L6-v2 via chromadb's default). Returns ``None`` on any
    failure — clustering becomes a pass-through (no batching) in that
    case so the pipeline still works.

    No OpenAI, no external embedding API — see plan §2.9.
    """
    try:
        from chromadb.utils import embedding_functions as ef
    except ImportError:
        return None
    try:
        # Instantiating the function is cheap after the model is cached
        # by chromadb's session. On a cold start it triggers a ~79 MB
        # ONNX model download; that's a one-time cost shared with the
        # rest of mempalace.
        efunc = ef.DefaultEmbeddingFunction()
        if efunc is None:
            return None
        vecs = efunc([hint])
        if not vecs:
            return None
        v = vecs[0]
        # chromadb returns a numpy array; coerce to plain list for JSON-
        # friendliness in logs/tests.
        return list(float(x) for x in v)
    except Exception as exc:
        log.info("domain-hint embed failed: %s", exc)
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _cluster_candidates_by_domain(
    candidates_with_hints: list[tuple[dict, str, list[float] | None]],
    threshold: float = 0.9,
) -> list[list[tuple[dict, str, list[float] | None]]]:
    """Greedy single-link clustering on domain-hint embeddings.

    Candidates with cosine ≥ threshold on their domain-hint vectors
    share a Stage-1 Opus call (one persona set serves the cluster).
    Missing embeddings fall through as singleton clusters so the
    pipeline still makes progress.

    Deterministic ordering (by input index) keeps batch contents
    reproducible across runs — important for tests and for diffing
    telemetry run-over-run.
    """
    clusters: list[list[tuple[dict, str, list[float] | None]]] = []
    for item in candidates_with_hints:
        _, _, vec = item
        placed = False
        if vec is not None:
            for cluster in clusters:
                # Compare against cluster centroid (first member's vec).
                _, _, c_vec = cluster[0]
                if c_vec is not None and _cosine(vec, c_vec) >= threshold:
                    cluster.append(item)
                    placed = True
                    break
        if not placed:
            clusters.append([item])
    return clusters


# ─────────────────────────────────────────────────────────────────────
# Stage 1 — Opus designs the jury (plan §2.5)
# ─────────────────────────────────────────────────────────────────────


_STAGE1_PROMPT = """You are designing a jury of 3 to assess whether two entities in a knowledge graph share a real relationship.

Here is the domain they co-occurred in:
{domain_hint}

Design 3 juror personas tailored to this domain. ALWAYS include:
  - an "ontologist" (picks the most-fitting predicate; domain-agnostic)
  - a "skeptic" (challenges the evidence; domain-agnostic)
The THIRD juror must be a domain expert appropriate for this domain
(e.g. senior software engineer, paralegal, accountant, marketing
strategist, security analyst, clinical researcher — whatever fits).

Return STRICT JSON, no prose, no markdown fences:
[
  {{"role": "ontologist", "persona_prompt": "You are ..."}},
  {{"role": "skeptic",    "persona_prompt": "You are ..."}},
  {{"role": "<expert>",   "persona_prompt": "You are ..."}}
]"""


async def _design_jury(client, cfg: dict, domain_hint: str) -> list[dict]:
    """Stage 1: one Opus call returning [{role, persona_prompt}, ×3].

    Raises on any failure (API error, malformed JSON, wrong shape) so
    ``author_candidate`` marks the candidate ``jury_design_failed`` and
    retries next run. NO fallback to hard-coded personas — a poorly-
    designed jury authoring real edges is worse than no jury running.
    """
    model = cfg["jury_design_model"]
    text = await _call_anthropic(
        client,
        model=model,
        max_tokens=cfg["design_max_tokens"],
        system=None,
        user=_STAGE1_PROMPT.format(domain_hint=domain_hint),
    )
    personas = _parse_json_response(text, schema_name="jury_design")
    if not isinstance(personas, list) or len(personas) != 3:
        raise ValueError(
            f"jury_design must return a list of 3 personas, got "
            f"{type(personas).__name__} len={len(personas) if hasattr(personas, '__len__') else '?'}"
        )
    for p in personas:
        if not isinstance(p, dict) or "role" not in p or "persona_prompt" not in p:
            raise ValueError(f"jury_design persona malformed: {p!r}")
    return personas


# ─────────────────────────────────────────────────────────────────────
# Stage 2 — Haiku jurors run in parallel (plan §2.5)
# ─────────────────────────────────────────────────────────────────────


_STAGE2_USER_TEMPLATE = """Assess whether entities A and B share a real relationship that SHOULD be recorded as a graph edge.

ENTITY A ({a_kind}) [id={a_id}]:
{a_desc}
keywords: {a_keywords}

ENTITY B ({b_kind}) [id={b_id}]:
{b_desc}
keywords: {b_keywords}

SHARED CONTEXTS (top {n_shared}, most relevant first):
{shared_blob}

AVAILABLE PREDICATES (kind-compatible with this pair):
{predicates_blob}

Return STRICT JSON matching this schema (no prose, no markdown fences):
{{
  "role": "<your role from the persona prompt>",
  "verdict": "edge" | "no_edge" | "uncertain",
  "confidence": 0.0-1.0,
  "predicate_choice": "<existing predicate name>" | null,
  "propose_new_predicate": null | {{
    "name": "snake_case_name",
    "description": "one-sentence semantic",
    "subject_kinds": ["entity"|"class"|"predicate"|"literal"|"record"],
    "object_kinds":  ["entity"|"class"|"predicate"|"literal"|"record"],
    "cardinality": "many-to-many"|"many-to-one"|"one-to-many"|"one-to-one"
  }},
  "statement": "A declarative sentence naming A + B." | null,
  "reason": "short explanation"
}}

Rules:
- Set verdict="edge" only if the relationship is specific enough to
  name with ONE predicate. If you'd write multiple edges, you're
  over-reaching — return "no_edge".
- predicate_choice must come from AVAILABLE PREDICATES, OR you must
  populate propose_new_predicate. Not both. Not neither (when verdict
  is "edge").
- statement is mandatory when verdict is "edge". Use the entity names
  (not ids) in natural English.
- Never invent a relationship just to produce an answer. "no_edge"
  and "uncertain" are valid outputs when the evidence is weak.
"""


async def _run_juror(
    client,
    cfg: dict,
    persona: dict,
    entity_ctx: dict,
    predicates: list[dict],
) -> dict:
    """Stage 2: one Haiku juror with an Opus-designed persona prompt.

    Returns the parsed verdict dict. On a malformed-JSON or API error,
    returns a fallback ``uncertain`` verdict so the synthesis step can
    still run — retries of individual juror calls happen inside
    ``_call_anthropic`` (SDK-level).
    """
    a = entity_ctx["a"]
    b = entity_ctx["b"]
    shared_blob = entity_ctx["shared_blob"]
    predicates_blob = (
        "\n".join(
            f"- {p['name']}: {p.get('description', '')} "
            f"(subject_kinds={p.get('subject_kinds', 'any')}, "
            f"object_kinds={p.get('object_kinds', 'any')}, "
            f"cardinality={p.get('cardinality', 'many-to-many')})"
            for p in predicates
        )
        or "(no predicates currently match this kind pair — propose a new one if you accept)"
    )

    user = _STAGE2_USER_TEMPLATE.format(
        a_kind=a.get("kind", "entity"),
        a_id=a["id"],
        a_desc=a.get("description", "")[:500] or "(no description)",
        a_keywords=", ".join(a.get("keywords", [])[:10]) or "—",
        b_kind=b.get("kind", "entity"),
        b_id=b["id"],
        b_desc=b.get("description", "")[:500] or "(no description)",
        b_keywords=", ".join(b.get("keywords", [])[:10]) or "—",
        n_shared=entity_ctx.get("n_shared", 0),
        shared_blob=shared_blob,
        predicates_blob=predicates_blob,
    )
    try:
        text = await _call_anthropic(
            client,
            model=cfg["jury_execution_model"],
            max_tokens=cfg["juror_max_tokens"],
            system=persona["persona_prompt"],
            user=user,
        )
        verdict = _parse_json_response(text, schema_name="juror")
        if not isinstance(verdict, dict):
            raise ValueError(f"juror returned non-dict: {type(verdict).__name__}")
        verdict.setdefault("role", persona.get("role", "?"))
        return verdict
    except Exception as exc:
        log.warning("juror %s failed: %s", persona.get("role"), exc)
        return {
            "role": persona.get("role", "?"),
            "verdict": "uncertain",
            "confidence": 0.0,
            "predicate_choice": None,
            "propose_new_predicate": None,
            "statement": None,
            "reason": f"juror failure: {exc}",
        }


async def _run_jury(client, cfg: dict, personas, entity_ctx, predicates) -> list[dict]:
    """Three Haiku juror calls dispatched in parallel via asyncio.gather.

    ``gather(*)`` gives us concurrent latency: total wall time for the
    jury ≈ max(t_juror) instead of sum. With three ~2s Haiku calls, this
    is the 6s → 2s savings per candidate that makes a 50-candidate run
    realistic under a 1-hour dispatch interval.
    """
    return await asyncio.gather(
        *[_run_juror(client, cfg, p, entity_ctx, predicates) for p in personas]
    )


# ─────────────────────────────────────────────────────────────────────
# Stage 3 — Haiku synthesis (plan §2.5.1)
# ─────────────────────────────────────────────────────────────────────


_STAGE3_PROMPT = """Three jurors have returned verdicts on whether two entities share a real relationship. Your job is to produce the final decision.

JUROR VERDICTS (JSON):
{jurors_blob}

Consensus rules:
- All 3 "edge" with same predicate     → accept, use best statement.
- All 3 "edge", different predicates   → pick the best based on their reasoning; mark "uncertain" if you can't.
- 2 "edge" + 1 "no_edge", predicate agreement → accept.
- 2 "edge" + 1 "no_edge", predicate split     → "uncertain".
- 2 "no_edge" + 1 "edge"               → reject (no_edge).
- All 3 "no_edge"                      → reject.
- Any schema-violating juror counts as "uncertain".

New-predicate consensus: a new predicate may ONLY be created if the
ontologist AND at least one other juror propose the SAME name with
broadly-compatible semantics. Otherwise reject the proposal; do NOT
invent a fallback.

Return STRICT JSON, no prose, no markdown fences:
{{
  "verdict": "edge" | "no_edge" | "uncertain",
  "predicate": "<name>" | null,
  "new_predicate_proposed": null | {{
    "name": "...", "description": "...",
    "subject_kinds": [...], "object_kinds": [...],
    "cardinality": "..."
  }},
  "statement": "A declarative sentence naming the two entities." | null,
  "reason": "jury synthesis reason — one or two sentences",
  "juror_agreement": "unanimous" | "majority" | "split"
}}"""


async def _synthesise(client, cfg: dict, juror_verdicts: list[dict]) -> dict:
    """Stage 3: one Haiku call turning 3 juror dicts into the final verdict."""
    text = await _call_anthropic(
        client,
        model=cfg["synthesis_model"],
        max_tokens=cfg["synthesis_max_tokens"],
        system=None,
        user=_STAGE3_PROMPT.format(jurors_blob=json.dumps(juror_verdicts, indent=2)),
    )
    return _parse_json_response(text, schema_name="synthesis")


# ─────────────────────────────────────────────────────────────────────
# SDK coupling — single point of failure for retries + parsing
# ─────────────────────────────────────────────────────────────────────


async def _call_anthropic(
    client,
    *,
    model: str,
    max_tokens: int,
    system: str | None,
    user: str,
) -> str:
    """Thin async wrapper around ``client.messages.create``.

    All three stages go through here so SDK-level retries (429, 5xx)
    and response-extraction are centralised. Returns the assistant
    message text — whatever the caller asked for is up to them to
    parse via ``_parse_json_response``.
    """
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user}],
    }
    if system:
        kwargs["system"] = system
    resp = await client.messages.create(**kwargs)
    # The SDK's Message.content is a list of content blocks. We only
    # request text, so pulling .text from the first block is correct.
    parts = getattr(resp, "content", []) or []
    if not parts:
        return ""
    first = parts[0]
    return getattr(first, "text", "") or ""


def _parse_json_response(text: str, schema_name: str) -> Any:
    """Strip optional markdown fences and parse JSON. Raises on failure."""
    if not text or not text.strip():
        raise ValueError(f"{schema_name}: empty response")
    s = text.strip()
    # Tolerate ```json ... ``` fences even though we asked the model not
    # to use them — some models slip occasionally.
    if s.startswith("```"):
        # Drop opening fence (``` or ```json) and closing fence.
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{schema_name}: malformed JSON — {exc}") from exc


# ─────────────────────────────────────────────────────────────────────
# Predicate creation (plan §2.6)
# ─────────────────────────────────────────────────────────────────────


def _maybe_create_predicate(kg, proposal: dict) -> str | None:
    """Near-duplicate check + creation. Returns the final predicate name.

    Near-duplicate rule (plan §2.6): cosine on predicate DESCRIPTIONS
    ≥ 0.75 means "use existing, don't create new". Keeps the predicate
    space from exploding with semantically-identical synonyms.

    Logs creations to ``~/.mempalace/hook_state/new_predicates.jsonl``
    so the operator can periodically audit the predicate-space shape.
    """
    if not isinstance(proposal, dict):
        return None
    name = (proposal.get("name") or "").strip()
    description = (proposal.get("description") or "").strip()
    if not name or not description:
        return None

    # Cosine on predicate descriptions. Reuse ChromaDB's local embedder;
    # if it fails for any reason we fall back to "no near-duplicate" and
    # create the predicate — matching the fail-open rule elsewhere.
    try:
        from chromadb.utils import embedding_functions as ef

        efunc = ef.DefaultEmbeddingFunction()
        proposal_vec = efunc([description])[0] if efunc else None
    except Exception:
        proposal_vec = None

    if proposal_vec is not None:
        try:
            existing = kg.list_entities(status="active", kind="predicate")
        except Exception:
            existing = []
        best_name, best_sim = None, 0.0
        for p in existing:
            pid = p.get("id") or p.get("name")
            pdesc = (p.get("description") or "").strip()
            if not pid or not pdesc:
                continue
            try:
                vec = efunc([pdesc])[0]
            except Exception:
                continue
            sim = _cosine(list(proposal_vec), list(vec))
            if sim > best_sim:
                best_sim, best_name = sim, pid
        if best_name and best_sim >= 0.75:
            log.info(
                "predicate proposal %r near-duplicates existing %r (sim=%.3f); using existing",
                name,
                best_name,
                best_sim,
            )
            return best_name

    constraints = {
        "subject_kinds": list(proposal.get("subject_kinds") or []),
        "object_kinds": list(proposal.get("object_kinds") or []),
        "cardinality": proposal.get("cardinality") or "many-to-many",
    }
    try:
        kg.add_entity(
            name,
            kind="predicate",
            description=description,
            importance=3,
            properties={"constraints": constraints},
        )
    except Exception as exc:
        log.warning("predicate creation failed: %r: %s", name, exc)
        return None

    _append_new_predicate_log(
        {
            "ts": _now_iso(),
            "name": name,
            "description": description,
            "constraints": constraints,
        }
    )
    return name


def _append_new_predicate_log(record: dict) -> None:
    """Append one JSONL line. Swallows IO errors — auditing is best-effort."""
    try:
        NEW_PREDICATES_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(NEW_PREDICATES_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


# ─────────────────────────────────────────────────────────────────────
# Orchestration — author one candidate, process a run
# ─────────────────────────────────────────────────────────────────────


def _entity_payload(kg, entity_id: str) -> dict:
    """Build the compact entity blob Stage 2 jurors consume."""
    try:
        ent = kg.get_entity(entity_id) or {}
    except Exception:
        ent = {}
    props = ent.get("properties") or {}
    if isinstance(props, str):
        try:
            props = json.loads(props)
        except Exception:
            props = {}
    keywords = list(props.get("keywords") or [])
    return {
        "id": entity_id,
        "kind": ent.get("kind") or "entity",
        "description": (ent.get("description") or "")[:500],
        "keywords": keywords[:10],
    }


def _compatible_predicates(kg, a_kind: str, b_kind: str, limit: int = 12) -> list[dict]:
    """Return declared predicates whose constraints allow (a_kind, b_kind).

    Empty ``subject_kinds`` / ``object_kinds`` means "any" (matches the
    validator semantics at tool_kg_add). This is the SAME rule used by
    the retired suggester's kind-compat prefilter, relocated here as a
    Stage-2 input feeder only.
    """
    try:
        preds = kg.list_entities(status="active", kind="predicate")
    except Exception:
        return []
    _ALL = {"entity", "class", "predicate", "literal", "record"}
    out: list[dict] = []
    for p in preds:
        pid = p.get("id") or p.get("name")
        if not pid:
            continue
        props = p.get("properties") or {}
        if isinstance(props, str):
            try:
                props = json.loads(props)
            except Exception:
                props = {}
        c = props.get("constraints") or {}
        sub_kinds = set(c.get("subject_kinds") or _ALL)
        obj_kinds = set(c.get("object_kinds") or _ALL)
        if a_kind in sub_kinds and b_kind in obj_kinds:
            out.append(
                {
                    "name": pid,
                    "description": p.get("description", ""),
                    "subject_kinds": list(sub_kinds) if c.get("subject_kinds") else "any",
                    "object_kinds": list(obj_kinds) if c.get("object_kinds") else "any",
                    "cardinality": c.get("cardinality", "many-to-many"),
                }
            )
        if len(out) >= limit:
            break
    return out


def _shared_contexts_blob(kg, candidate: dict, max_n: int = 5) -> str:
    """Rendered top-N shared contexts for the juror prompt."""
    conn = kg._conn()
    a, b = candidate["from_entity"], candidate["to_entity"]
    rows = conn.execute(
        "SELECT ctx_id FROM link_prediction_sources "
        "WHERE from_entity = ? AND to_entity = ? "
        "ORDER BY contributed_ts DESC LIMIT ?",
        (a, b, int(max_n)),
    ).fetchall()
    parts: list[str] = []
    for (cid,) in rows:
        try:
            ctx = kg.get_entity(cid)
        except Exception:
            continue
        if not ctx:
            continue
        props = ctx.get("properties") or {}
        if isinstance(props, str):
            try:
                props = json.loads(props)
            except Exception:
                props = {}
        qs = list(props.get("queries") or [])[:2]
        kws = list(props.get("keywords") or [])[:5]
        parts.append(f"- queries={qs} keywords={kws}")
    return "\n".join(parts) or "(no shared context metadata available)"


async def author_candidate(
    kg,
    cfg: dict,
    client,
    candidate: dict,
    personas: list[dict] | None = None,
) -> dict:
    """Full Stage 1-3 pipeline for ONE candidate.

    Caller passes pre-designed personas when they're batched; otherwise
    Stage 1 runs per candidate. Returns a verdict dict persisted back
    onto ``link_prediction_candidates`` by ``process``.

    Never raises — every failure mode surfaces as a ``llm_verdict`` value
    in the returned dict (``jury_design_failed`` / ``uncertain`` /
    ``no_edge`` / ``edge``). That way ``process`` can batch-commit
    verdicts without worrying about partial failure.
    """
    out: dict = {
        "candidate": candidate,
        "llm_verdict": None,
        "llm_predicate": None,
        "llm_statement": None,
        "llm_reason": None,
        "llm_jury_personas": None,
        "llm_jury_design_model": cfg["jury_design_model"],
        "llm_jury_exec_model": cfg["jury_execution_model"],
    }

    # Stage 1 (only if not batched).
    if personas is None:
        try:
            hint = _build_domain_hint(kg, candidate)
            personas = await _design_jury(client, cfg, hint)
        except Exception as exc:
            log.warning(
                "jury design failed for %s — %s: %s",
                (candidate["from_entity"], candidate["to_entity"]),
                type(exc).__name__,
                exc,
            )
            out["llm_verdict"] = "jury_design_failed"
            out["llm_reason"] = f"design: {exc}"
            return out

    out["llm_jury_personas"] = personas

    # Stage 2 inputs.
    a_payload = _entity_payload(kg, candidate["from_entity"])
    b_payload = _entity_payload(kg, candidate["to_entity"])
    predicates = _compatible_predicates(kg, a_payload["kind"], b_payload["kind"])
    shared_blob = _shared_contexts_blob(kg, candidate)
    entity_ctx = {
        "a": a_payload,
        "b": b_payload,
        "shared_blob": shared_blob,
        "n_shared": shared_blob.count("- queries=") or candidate.get("shared_context_count", 0),
    }

    # Stage 2.
    try:
        juror_verdicts = await _run_jury(client, cfg, personas, entity_ctx, predicates)
    except Exception as exc:
        log.warning("jury execution failed: %s", exc)
        out["llm_verdict"] = "uncertain"
        out["llm_reason"] = f"execution: {exc}"
        return out

    # Stage 3.
    try:
        final = await _synthesise(client, cfg, juror_verdicts)
    except Exception as exc:
        log.warning("synthesis failed: %s", exc)
        out["llm_verdict"] = "uncertain"
        out["llm_reason"] = f"synthesis: {exc}"
        return out

    verdict = final.get("verdict") or "uncertain"
    out["llm_verdict"] = verdict
    out["llm_reason"] = final.get("reason")

    if verdict == "edge":
        # Predicate resolution: existing, or create-from-proposal.
        proposal = final.get("new_predicate_proposed") or None
        predicate_name = final.get("predicate") or None
        if proposal:
            created = _maybe_create_predicate(kg, proposal)
            if created:
                predicate_name = created
        statement = final.get("statement") or None
        if not predicate_name or not statement:
            # Schema-violating "edge" verdict — demote to uncertain.
            out["llm_verdict"] = "uncertain"
            out["llm_reason"] = (
                f"synthesis marked 'edge' but predicate={predicate_name!r} "
                f"statement={bool(statement)}"
            )
            return out
        out["llm_predicate"] = predicate_name
        out["llm_statement"] = statement

    return out


async def _process_async(
    kg,
    cfg: dict,
    *,
    max_per_run: int,
    threshold: float,
    dry_run: bool,
    batch_design: bool,
) -> dict:
    """The actual async body of ``process``. Returns a run-summary dict."""
    # Split validation avoids nesting asyncio.run inside this coroutine.
    client, key = _build_client(cfg)
    await _ping_and_translate(client, cfg, key)
    log.info("api key validated (%s)", _mask_key(key))

    # Record run start BEFORE any API call so _dispatch_if_due's
    # concurrency gate sees us as active and doesn't spawn a second
    # process that tramples our candidates.
    started_ts = _now_iso()
    run_id = None
    if not dry_run:
        conn = kg._conn()
        cur = conn.execute(
            "INSERT INTO link_author_runs (started_ts, design_model, exec_model) VALUES (?, ?, ?)",
            (started_ts, cfg["jury_design_model"], cfg["jury_execution_model"]),
        )
        run_id = cur.lastrowid
        conn.commit()

    candidates = list_pending(kg, limit=max_per_run, threshold=threshold)
    summary = {
        "run_id": run_id,
        "started_ts": started_ts,
        "candidates_found": len(candidates),
        "candidates_processed": 0,
        "edges_created": 0,
        "edges_rejected": 0,
        "uncertain": 0,
        "jury_design_failures": 0,
        "design_calls": 0,
        "new_predicates_created": 0,
    }
    if not candidates:
        summary["completed_ts"] = _now_iso()
        if run_id is not None:
            _finalize_run_row(kg, run_id, summary)
        return summary

    # Batch design: build hints + embeddings + cluster.
    with_hints: list[tuple[dict, str, list[float] | None]] = []
    for c in candidates:
        hint = _build_domain_hint(kg, c)
        vec = None
        if batch_design:
            vec = _embed_domain_hint(cfg.get("palace_path", ""), hint)
        with_hints.append((c, hint, vec))
    if batch_design:
        clusters = _cluster_candidates_by_domain(
            with_hints, threshold=cfg["batch_domain_cosine_threshold"]
        )
    else:
        clusters = [[item] for item in with_hints]

    # Per-cluster: Stage 1 once, Stage 2-3 per candidate.
    for cluster in clusters:
        cluster_hint = cluster[0][1]  # first member's hint is the representative
        try:
            personas = await _design_jury(client, cfg, cluster_hint)
            summary["design_calls"] += 1
        except Exception as exc:
            log.warning("cluster jury design failed: %s", exc)
            summary["jury_design_failures"] += 1
            for cand, _, _ in cluster:
                _persist_verdict(
                    kg,
                    cand,
                    {
                        "llm_verdict": "jury_design_failed",
                        "llm_reason": f"design: {exc}",
                        "llm_jury_personas": None,
                        "llm_jury_design_model": cfg["jury_design_model"],
                        "llm_jury_exec_model": cfg["jury_execution_model"],
                        "llm_predicate": None,
                        "llm_statement": None,
                    },
                    dry_run=dry_run,
                )
            continue

        for cand, _, _ in cluster:
            result = await author_candidate(kg, cfg, client, cand, personas=personas)
            summary["candidates_processed"] += 1
            v = result.get("llm_verdict")
            if v == "edge":
                if not dry_run:
                    _persist_edge(kg, cand, result)
                summary["edges_created"] += 1
            elif v == "no_edge":
                summary["edges_rejected"] += 1
            elif v == "jury_design_failed":
                summary["jury_design_failures"] += 1
            else:
                summary["uncertain"] += 1
            if result.get("llm_predicate") and _predicate_is_new(kg, result):
                summary["new_predicates_created"] += 1
            _persist_verdict(kg, cand, result, dry_run=dry_run)

    summary["completed_ts"] = _now_iso()
    if run_id is not None:
        _finalize_run_row(kg, run_id, summary)
    return summary


def _predicate_is_new(kg, result: dict) -> bool:
    """Heuristic: did this result carry a newly-created predicate?

    We can't perfectly distinguish "I just created this predicate" from
    "the jury picked an existing predicate whose description matched
    the proposal" post-hoc. For telemetry, we count it as new when the
    jury's final included a proposal. Close enough for a run summary.
    """
    return bool(
        result.get("llm_predicate")
        and result.get("candidate", {}).get("_had_new_predicate_proposal")
    )


def _persist_edge(kg, candidate: dict, result: dict) -> None:
    """Create the authored edge via kg.add_triple. Best-effort."""
    try:
        kg.add_triple(
            candidate["from_entity"],
            result["llm_predicate"],
            candidate["to_entity"],
            statement=result["llm_statement"],
            confidence=0.9,  # LLM-authored; calibration lands post-reinstall.
        )
    except Exception as exc:
        # Don't lose the verdict just because the edge-write failed; the
        # verdict row still records the synthesis outcome.
        log.warning(
            "add_triple failed for %s --[%s]--> %s: %s",
            candidate["from_entity"],
            result["llm_predicate"],
            candidate["to_entity"],
            exc,
        )


def _persist_verdict(kg, candidate: dict, result: dict, *, dry_run: bool) -> None:
    """Write the LLM outcome back onto link_prediction_candidates."""
    if dry_run:
        return
    try:
        conn = kg._conn()
        conn.execute(
            "UPDATE link_prediction_candidates SET "
            "  processed_ts = ?, "
            "  llm_verdict = ?, "
            "  llm_predicate = ?, "
            "  llm_statement = ?, "
            "  llm_reason = ?, "
            "  llm_jury_personas = ?, "
            "  llm_jury_design_model = ?, "
            "  llm_jury_exec_model = ? "
            "WHERE from_entity = ? AND to_entity = ?",
            (
                _now_iso(),
                result.get("llm_verdict"),
                result.get("llm_predicate"),
                result.get("llm_statement"),
                result.get("llm_reason"),
                json.dumps(result.get("llm_jury_personas") or [], ensure_ascii=False)
                if result.get("llm_jury_personas") is not None
                else None,
                result.get("llm_jury_design_model"),
                result.get("llm_jury_exec_model"),
                candidate["from_entity"],
                candidate["to_entity"],
            ),
        )
        conn.commit()
    except Exception as exc:
        log.warning("persist_verdict failed: %s", exc)


def _finalize_run_row(kg, run_id: int, summary: dict) -> None:
    """Write the run summary back onto link_author_runs."""
    try:
        conn = kg._conn()
        conn.execute(
            "UPDATE link_author_runs SET "
            "  completed_ts = ?, "
            "  candidates_processed = ?, "
            "  edges_created = ?, "
            "  edges_rejected = ?, "
            "  new_predicates_created = ?, "
            "  design_calls = ?, "
            "  jury_design_failures = ? "
            "WHERE id = ?",
            (
                summary.get("completed_ts") or _now_iso(),
                summary.get("candidates_processed", 0),
                summary.get("edges_created", 0),
                summary.get("edges_rejected", 0),
                summary.get("new_predicates_created", 0),
                summary.get("design_calls", 0),
                summary.get("jury_design_failures", 0),
                run_id,
            ),
        )
        conn.commit()
    except Exception as exc:
        log.warning("finalize_run_row failed: %s", exc)


def process(
    kg,
    cfg: dict,
    *,
    max_per_run: int | None = None,
    threshold: float | None = None,
    dry_run: bool = False,
    batch_design: bool | None = None,
) -> dict:
    """Synchronous CLI entry point — drives the whole async pipeline.

    Resolves defaults from cfg, then runs the async body via
    ``asyncio.run``. Exit codes (when a SystemExit propagates): 0 ok,
    2 bad key, 3 API down. The CLI handler catches and formats.
    """
    max_per_run = max_per_run if max_per_run is not None else cfg.get("max_per_run", 50)
    threshold = threshold if threshold is not None else cfg.get("threshold", 1.5)
    batch_design = (
        batch_design
        if batch_design is not None
        else cfg.get("batch_design_by_domain_similarity", True)
    )
    _load_env(palace_path=cfg.get("palace_path"), dotenv_path=cfg.get("dotenv_path"))
    return asyncio.run(
        _process_async(
            kg,
            cfg,
            max_per_run=max_per_run,
            threshold=threshold,
            dry_run=dry_run,
            batch_design=batch_design,
        )
    )


# ─────────────────────────────────────────────────────────────────────
# Status reporting — agent/user-facing summary
# ─────────────────────────────────────────────────────────────────────


def status(kg, *, recent: int = 5, new_predicates: bool = False) -> dict:
    """Return a status snapshot of the link-author pipeline."""
    conn = kg._conn()
    pending = conn.execute(
        "SELECT COUNT(*), COALESCE(MAX(score), 0) "
        "FROM link_prediction_candidates WHERE processed_ts IS NULL"
    ).fetchone()
    processed = conn.execute(
        "SELECT llm_verdict, COUNT(*) FROM link_prediction_candidates "
        "WHERE processed_ts IS NOT NULL GROUP BY llm_verdict"
    ).fetchall()
    runs = conn.execute(
        "SELECT started_ts, completed_ts, candidates_processed, "
        "       edges_created, edges_rejected, new_predicates_created, "
        "       design_calls, jury_design_failures "
        "FROM link_author_runs ORDER BY started_ts DESC LIMIT ?",
        (int(recent),),
    ).fetchall()

    snapshot = {
        "pending": {
            "count": pending[0] if pending else 0,
            "max_score": float(pending[1]) if pending else 0.0,
        },
        "processed_by_verdict": {v or "(null)": n for v, n in processed},
        "recent_runs": [
            {
                "started_ts": r[0],
                "completed_ts": r[1],
                "candidates_processed": r[2],
                "edges_created": r[3],
                "edges_rejected": r[4],
                "new_predicates_created": r[5],
                "design_calls": r[6],
                "jury_design_failures": r[7],
            }
            for r in runs
        ],
    }
    if new_predicates and NEW_PREDICATES_LOG.is_file():
        try:
            lines = NEW_PREDICATES_LOG.read_text(encoding="utf-8").splitlines()
            snapshot["new_predicates"] = [json.loads(line) for line in lines if line.strip()]
        except Exception:
            snapshot["new_predicates"] = []
    return snapshot


# ─────────────────────────────────────────────────────────────────────
# Finalize-triggered background dispatcher (plan §2.7)
# ─────────────────────────────────────────────────────────────────────


def _dispatch_if_due(kg, interval_hours: int = 1) -> None:
    """Event-driven dispatcher — spawn ``mempalace link-author process`` detached.

    Contract (plan §2.7):
      1. Check that at least ``interval_hours`` have passed since the
         most recent ``link_author_runs.started_ts``. If no prior run,
         always eligible.
      2. Check there is at least one pending candidate.
      3. If both, spawn a fire-and-forget detached subprocess so the
         finalize call doesn't block on LLM work.
      4. BEFORE the fork, write a new ``link_author_runs`` row with
         ``started_ts`` set — that way a concurrent finalize won't
         spawn a second process (the cadence gate trips).

    Non-blocking. Any exception is swallowed; the caller wraps this in
    try/except as well. Inheritance: the detached child inherits the
    parent's environment, so ``ANTHROPIC_API_KEY`` flows through as
    long as the MCP server was started with it loaded (the plugin
    launcher sources ``.env``). See docs/link_author_scheduling.md.
    """
    try:
        conn = kg._conn()

        # Cadence gate.
        row = conn.execute(
            "SELECT started_ts FROM link_author_runs ORDER BY started_ts DESC LIMIT 1"
        ).fetchone()
        if row and row[0]:
            try:
                last_started = datetime.fromisoformat(row[0].replace("Z", "+00:00"))
            except Exception:
                last_started = None
            if last_started is not None:
                now = datetime.now(timezone.utc)
                if (now - last_started).total_seconds() < int(interval_hours) * 3600:
                    return

        # Pending gate.
        pending = conn.execute(
            "SELECT 1 FROM link_prediction_candidates WHERE processed_ts IS NULL LIMIT 1"
        ).fetchone()
        if pending is None:
            return

        # Pre-fork marker so two concurrent finalizes don't double-spawn.
        conn.execute(
            "INSERT INTO link_author_runs (started_ts, design_model, exec_model) VALUES (?, ?, ?)",
            (_now_iso(), "<pending>", "<pending>"),
        )
        conn.commit()

        # Detached subprocess — platform-aware.
        cmd = [sys.executable, "-m", "mempalace", "link-author", "process"]
        if os.name == "nt":
            # Windows: DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP.
            DETACHED_PROCESS = 0x00000008  # noqa: N806
            CREATE_NEW_PROCESS_GROUP = 0x00000200  # noqa: N806
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                close_fds=True,
            )
        else:
            # POSIX: start_new_session detaches from the controlling TTY.
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
    except Exception as exc:
        log.info("_dispatch_if_due skipped: %s", exc)
