# ruff: noqa
# Microbenchmark harness -- terse style (multi-statement lines, one-letter
# locals) is intentional; excluded from project lint/format. Not a pytest
# target -- invoke via `python tests/timing_harness_injection.py`.
from __future__ import annotations
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.pop("MEMPALACE_GATE_DISABLED", None)

from mempalace.injection_gate import (
    GateItem,
    InjectionGate,
    apply_gate,
    build_prompt,
    _extract_decisions,
)


class _FakeToolBlock:
    def __init__(self, ids, drop_every=0):
        self.type = "tool_use"
        self.name = "gate_decisions"
        decs = []
        for i, _id in enumerate(ids):
            action = "drop" if drop_every and ((i + 1) % drop_every == 0) else "keep"
            decs.append({"id": _id, "action": action, "reasoning": "test"})
        self.input = {"decisions": decs, "flags": []}


class _FakeResp:
    def __init__(self, ids, drop_every=0):
        self.content = [_FakeToolBlock(ids, drop_every=drop_every)]
        self.usage = SimpleNamespace(input_tokens=100 * len(ids), output_tokens=30 * len(ids))


class FakeClient:
    def __init__(self, *, delay_s=0.0, fail_n=0, drop_every=0):
        self.delay_s = delay_s
        self.fail_n = fail_n
        self.drop_every = drop_every
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.calls <= self.fail_n:
            raise RuntimeError("synthetic API failure")
        user = kwargs["messages"][0]["content"]
        ids = [
            line.split("id=", 1)[1].strip()
            for line in user.splitlines()
            if "id=" in line and line.strip().startswith("[")
        ]
        return _FakeResp(ids, drop_every=self.drop_every)


def _items(n):
    return [
        GateItem(
            id=f"m{i}",
            source="memory",
            text=f"content #{i} " * 20,
            channel="A",
            rank=i + 1,
            score=0.9 - 0.01 * i,
            extra={"summary": f"summary {i}"},
        )
        for i in range(n)
    ]


PRIMARY = {
    "source": "declare_intent",
    "queries": ["latency", "profile"],
    "keywords": ["gate", "latency"],
    "entities": ["injection_stage_gate"],
}


def _tm(f, rep=1):
    runs = []
    out = None
    for _ in range(rep):
        t0 = time.perf_counter()
        out = f()
        runs.append((time.perf_counter() - t0) * 1000.0)
    return round(statistics.median(runs), 3), round(min(runs), 3), round(max(runs), 3), out


def build_bench():
    r = []
    for n in (3, 5, 10, 25, 50):
        its = _items(n)
        med, lo, hi, _ = _tm(lambda: build_prompt(primary_context=PRIMARY, items=its), rep=50)
        r.append((n, med, lo, hi))
    return r


def parse_bench():
    r = []
    for n in (3, 5, 10, 25, 50):
        ids = [f"m{i}" for i in range(n)]
        resp = _FakeResp(ids, drop_every=3)
        known = set(ids)
        med, lo, hi, _ = _tm(lambda: _extract_decisions(resp, known), rep=200)
        r.append((n, med, lo, hi))
    return r


def short_bench():
    r = []
    gate = InjectionGate(_client=None)
    for lab, its in (("K=0 empty", []), ("K=2 small_k", _items(2)), ("K=5 no_client", _items(5))):
        med, lo, hi, res = _tm(
            lambda its=its: gate.filter(primary_context=PRIMARY, items=its), rep=10
        )
        r.append((lab, res.gate_status.get("state"), med, lo, hi))
    return r


def fake_ok_bench():
    r = []
    for n, dly in ((3, 0), (5, 0), (10, 0), (25, 0), (5, 50), (5, 200)):
        c = FakeClient(delay_s=dly / 1000.0, drop_every=4)
        g = InjectionGate(_client=c)
        its = _items(n)
        med, lo, hi, res = _tm(lambda: g.filter(primary_context=PRIMARY, items=its), rep=5)
        t = res.timings
        r.append(
            (
                f"K={n} d={dly}ms",
                res.gate_status.get("state"),
                med,
                t.get("prompt_ms"),
                t.get("llm_ms"),
                t.get("parse_ms"),
                t.get("attempts"),
                len(res.kept),
                len(res.dropped),
            )
        )
    return r


def retry_bench():
    c = FakeClient(fail_n=1, drop_every=0)
    g = InjectionGate(_client=c, max_retries=2)
    its = _items(5)
    t0 = time.perf_counter()
    res = g.filter(primary_context=PRIMARY, items=its)
    return {
        "state": res.gate_status.get("state"),
        "total_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        "timings": res.timings,
        "kept": len(res.kept),
        "dropped": len(res.dropped),
    }


def degraded_bench():
    c = FakeClient(fail_n=99)
    g = InjectionGate(_client=c, max_retries=2)
    its = _items(5)
    t0 = time.perf_counter()
    res = g.filter(primary_context=PRIMARY, items=its)
    return {
        "state": res.gate_status.get("state"),
        "total_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        "timings": res.timings,
        "kept": len(res.kept),
    }


class _StubKG:
    def record_feedback(self, *a, **kw):
        pass

    def record_memory_flags(self, *a, **kw):
        pass


def apply_bench():
    r = []
    for n in (3, 5, 10, 25):
        c = FakeClient(drop_every=4)
        g = InjectionGate(_client=c)
        mems = [{"id": f"m{i}", "text": f"c{i}"} for i in range(n)]
        cm = {
            f"m{i}": {"source": "memory", "meta": {"summary": f"s{i}"}, "doc": f"c{i}"}
            for i in range(n)
        }
        med, lo, hi, _ = _tm(
            lambda: apply_gate(
                memories=mems,
                combined_meta=cm,
                primary_context=PRIMARY,
                context_id="ctx_t",
                kg=_StubKG(),
                agent="ga_agent",
                gate=g,
            ),
            rep=5,
        )
        r.append((n, med, lo, hi))
    return r


def log_stats():
    p = Path.home() / ".mempalace" / "hook_state" / "gate_log.jsonl"
    if not p.exists():
        return {"error": "not found"}
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return {"error": "empty"}
    by = {}
    for rr in rows:
        by.setdefault(rr.get("state", "?"), []).append(rr)

    def _s(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        if not xs:
            return {}
        return {
            "min": round(min(xs), 2),
            "med": round(statistics.median(xs), 2),
            "max": round(max(xs), 2),
        }

    out = {"count": len(rows), "by_state": {}}
    for st, rs in by.items():
        out["by_state"][st] = {
            "n": len(rs),
            "apply_total_ms": _s([r.get("apply_total_ms", 0) for r in rs]),
            "llm_ms": _s([(r.get("timings") or {}).get("llm_ms", 0) or 0 for r in rs]),
            "prompt_ms": _s([(r.get("timings") or {}).get("prompt_ms", 0) or 0 for r in rs]),
            "parse_ms": _s([(r.get("timings") or {}).get("parse_ms", 0) or 0 for r in rs]),
            "n_items": _s([r.get("n_items", 0) for r in rs]),
            "n_dropped": _s([r.get("n_dropped", 0) for r in rs]),
            "n_flags": _s([r.get("n_flags", 0) for r in rs]),
            "tokens_in": _s([r.get("tokens_in", 0) for r in rs]),
            "tokens_out": _s([r.get("tokens_out", 0) for r in rs]),
        }
    return out


def tbl(title, hdrs, rows):
    print("")
    print("=== " + title + " ===")
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(hdrs)]
    fmt = " | ".join("{:<" + str(x) + "}" for x in w)
    print(fmt.format(*hdrs))
    print("-+-".join("-" * x for x in w))
    for r in rows:
        print(fmt.format(*[str(c) for c in r]))


t0 = time.perf_counter()
tbl("A. build_prompt (ms)", ("n_items", "med", "min", "max"), build_bench())
tbl("B. _extract_decisions parse (ms)", ("n_items", "med", "min", "max"), parse_bench())
tbl("C. filter() short-circuits (ms)", ("scenario", "state", "med", "min", "max"), short_bench())
tbl(
    "D. filter() fake_ok full path (ms)",
    ("scenario", "state", "total", "prompt", "llm", "parse", "att", "kept", "drop"),
    fake_ok_bench(),
)
print("")
print("=== E. filter() retry-then-success ===")
print(json.dumps(retry_bench(), indent=2))
print("")
print("=== F. filter() degraded (all attempts fail) ===")
print(json.dumps(degraded_bench(), indent=2))
tbl("G. apply_gate() wall-clock (ms)", ("n_items", "med", "min", "max"), apply_bench())
print("")
print("=== H. gate_log.jsonl aggregate stats (REAL Haiku runs) ===")
print(json.dumps(log_stats(), indent=2))
print("")
print("harness wall-clock: %.2fs" % (time.perf_counter() - t0))
