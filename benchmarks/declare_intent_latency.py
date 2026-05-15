"""Empirical latency benchmark for declare_intent's hot path phases.

Measures each phase in isolation 3x cold (5+ min between calls) +
3x warm (back-to-back inside the 5-min ephemeral cache window):

  Phase 1: multi_channel_search retrieval (cosine + BM25 + RRF)
  Phase 2: apply_gate (Haiku tool-use filter)
  Phase 3: run_state_judge (Haiku state-change detection)

Phase 2 + 3 hit Haiku and dominate. The benchmark prints per-call
elapsed_ms, tokens_in, cache_read, cache_creation so you can SEE
the cold-vs-warm cache cliff directly.

Usage:
    python benchmarks/declare_intent_latency.py

Prereqs: ANTHROPIC_API_KEY in env (or in <palace>/.env). Reuses the
running palace at ~/.mempalace/palace/knowledge_graph.db.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure repo root on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env from palace home if present
try:
    from dotenv import load_dotenv

    load_dotenv(Path.home() / ".mempalace" / "palace" / ".env", override=True)
    load_dotenv(Path.home() / ".mempalace" / ".env", override=True)
except Exception:
    pass


def _bytes_short(n: int) -> str:
    return f"{n:>5}"


def _print_row(label: str, ms: float, t_in: int, t_out: int, c_read: int, c_create: int) -> None:
    print(
        f"  {label:<30} {ms:>8.1f}ms   in={_bytes_short(t_in)} out={_bytes_short(t_out)} "
        f"cache_read={_bytes_short(c_read)} cache_create={_bytes_short(c_create)}"
    )


def _print_phase_header(title: str) -> None:
    print()
    print(f"=== {title} ===")


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set; cannot benchmark Haiku calls.")
        return 1

    from mempalace.injection_gate import apply_gate, run_state_judge
    from mempalace.knowledge_graph import KnowledgeGraph

    palace_dir = Path.home() / ".mempalace" / "palace"
    if not palace_dir.exists():
        print(f"ERROR: palace not found at {palace_dir}; reinstall first.")
        return 1

    print(f"Palace: {palace_dir}")
    kg = KnowledgeGraph(db_path=str(palace_dir / "knowledge_graph.db"))

    cue = {
        "queries": [
            "feedback rating async Haiku rater architecture",
            "how does mempalace rate memories post finalize",
            "feedback_auto submit_finalize_feedback wiring",
        ],
        "keywords": ["feedback_auto", "haiku_auto", "rating"],
        "entities": ["mempalace"],
    }

    # Phase 1 (retrieval) was dropped: multi_channel_search has a low-
    # level signature (vs / collection_name / pre-embedded views) that
    # is awkward to call from a benchmark; it is also not the suspect
    # phase. Live declare_intent gate_reports show ~50-200ms total for
    # retrieval + persistence vs 3000-8000ms for the gate Haiku call.
    _stub_text = (
        "feedback_auto.py is the v3.5.0 async Haiku rater module that replaces the "
        "agent-side memory_feedback + operation_ratings coverage gates. "
        "submit_finalize_feedback fires per-intent + per-op + per-search batches in "
        "a background ThreadPoolExecutor; results land as rated_useful / "
        "rated_irrelevant edges with rater_kind='haiku_auto'. Cached system prefix "
        "keeps cost low across batches in a finalize."
    )
    memories_for_gate = [
        {
            "id": f"bench_mem_{i}",
            "preview": _stub_text + f" (instance {i})",
            "score": 0.7 - 0.05 * i,
            "kind": "memory",
        }
        for i in range(10)
    ]

    combined_meta = {
        m["id"]: {
            "source": "memory",
            "doc": m.get("preview") or "",
            "similarity": m.get("score", 0.0),
        }
        for m in memories_for_gate
    }

    primary_context = {
        "source": "benchmark",
        "queries": cue["queries"],
        "keywords": cue["keywords"],
        "entities": cue["entities"],
    }

    # --- Phase 2: apply_gate (Haiku) ---
    _print_phase_header(
        "Phase 2: apply_gate (Haiku) -- 3x back-to-back so cache warms"
    )
    print(f"  (input: {len(memories_for_gate)} memories to filter)")
    for i in range(3):
        t0 = time.perf_counter()
        _filtered, _status, gate_report = apply_gate(
            memories=memories_for_gate,
            combined_meta=combined_meta,
            primary_context=primary_context,
            context_id=f"bench_ctx_{i}",
            kg=kg,
            agent="ga_agent",
        )
        elapsed = (time.perf_counter() - t0) * 1000
        tk = (gate_report or {}).get("tokens", {}) or {}
        _print_row(
            f"call #{i + 1} (kept={(gate_report or {}).get('output_count', '?')})",
            elapsed,
            int(tk.get("input", 0) or 0),
            int(tk.get("output", 0) or 0),
            int(tk.get("cache_read", 0) or 0),
            int(tk.get("cache_creation", 0) or 0),
        )

    # --- Phase 3: run_state_judge (Haiku) ---
    _print_phase_header(
        "Phase 3: run_state_judge (Haiku) -- 3x back-to-back so cache warms"
    )
    transcript = (
        "intent_type: edit_and_run\n"
        "slots: {target: ['mempalace']}\n"
        "this op: tool=Bash, args_summary='python benchmarks/declare_intent_latency.py'\n"
        "cue.queries: ['benchmark declare_intent latency']\n"
    )
    entity_states = [
        {"entity_id": "ga_agent", "state_schema_id": "agent_state", "current_state": {}},
        {"entity_id": "ctx_12375", "state_schema_id": "intent_state", "current_state": {}},
    ]
    for i in range(3):
        t0 = time.perf_counter()
        _changes, judge_report = run_state_judge(
            transcript_text=transcript,
            entity_states=entity_states,
            agent="ga_agent",
        )
        elapsed = (time.perf_counter() - t0) * 1000
        # judge_report shape may differ; pull common keys defensively
        tk = (judge_report or {}).get("tokens", {}) or {}
        if not tk and isinstance(judge_report, dict):
            tk = {
                "input": judge_report.get("input_tokens"),
                "output": judge_report.get("output_tokens"),
                "cache_read": judge_report.get("cache_read_input_tokens"),
                "cache_creation": judge_report.get("cache_creation_input_tokens"),
            }
        _print_row(
            f"call #{i + 1} (changes={len(_changes or [])})",
            elapsed,
            int(tk.get("input", 0) or 0),
            int(tk.get("output", 0) or 0),
            int(tk.get("cache_read", 0) or 0),
            int(tk.get("cache_creation", 0) or 0),
        )

    # --- Phase 4: combined "what real declare_intent does" baseline ---
    _print_phase_header(
        "Phase 4: gate + judge in PARALLEL (mirrors real declare_intent)"
    )
    from concurrent.futures import ThreadPoolExecutor

    for i in range(3):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as ex:
            gate_fut = ex.submit(
                apply_gate,
                memories=memories_for_gate,
                combined_meta=combined_meta,
                primary_context=primary_context,
                context_id=f"bench_par_{i}",
                kg=kg,
                agent="ga_agent",
            )
            judge_fut = ex.submit(
                run_state_judge,
                transcript_text=transcript,
                entity_states=entity_states,
                agent="ga_agent",
            )
            _filtered, _status, gate_report = gate_fut.result()
            _changes, judge_report = judge_fut.result()
        elapsed = (time.perf_counter() - t0) * 1000
        gtk = (gate_report or {}).get("tokens", {}) or {}
        jtk = (judge_report or {}).get("tokens", {}) or {}
        print(
            f"  call #{i + 1}: parallel total {elapsed:>6.1f}ms   "
            f"gate_in={int(gtk.get('input', 0) or 0):>4} cache_read={int(gtk.get('cache_read', 0) or 0):>4}   "
            f"judge_in={int(jtk.get('input', 0) or 0):>4} cache_read={int(jtk.get('cache_read', 0) or 0):>4}"
        )

    print()
    print("DONE. Cold-vs-warm cache effect visible if cache_read jumps from 0 -> N on later calls.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
