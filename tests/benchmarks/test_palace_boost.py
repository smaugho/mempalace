"""
Agent-scoped retrieval validation — does added_by filtering help?

Quantifies the retrieval improvement from agent-scoped filtering.
Uses planted needles to measure recall with and without filtering
at different scales.
"""

import time

import pytest

from tests.benchmarks.data_generator import PalaceDataGenerator
from tests.benchmarks.report import record_metric


@pytest.mark.benchmark
class TestFilteredVsUnfilteredRecall:
    """Quantify scoping boost: recall improvement from added_by filtering."""

    SIZES = [1_000, 2_500, 5_000]

    @pytest.mark.parametrize("n_drawers", SIZES)
    def test_scoping_boost_recall(self, n_drawers, tmp_path, bench_scale):
        """Compare recall@5 with/without added_by filter at increasing scale."""
        gen = PalaceDataGenerator(seed=42, scale=bench_scale)
        palace_path = str(tmp_path / "palace")
        _, _, needle_info = gen.populate_palace_directly(
            palace_path, n_drawers=n_drawers, include_needles=True
        )

        from mempalace.searcher import search_memories

        n_queries = min(10, len(needle_info))
        unfiltered_hits = 0
        filtered_hits = 0

        for needle in needle_info[:n_queries]:
            # Unfiltered search
            result = search_memories(needle["query"], palace_path=palace_path, n_results=5)
            texts = [h["text"] for h in result.get("results", [])]
            if any("NEEDLE_" in t for t in texts[:5]):
                unfiltered_hits += 1

            # Agent-filtered search
            result = search_memories(
                needle["query"], palace_path=palace_path, added_by=needle["added_by"], n_results=5
            )
            texts = [h["text"] for h in result.get("results", [])]
            if any("NEEDLE_" in t for t in texts[:5]):
                filtered_hits += 1

        recall_none = unfiltered_hits / max(n_queries, 1)
        recall_filtered = filtered_hits / max(n_queries, 1)

        boost = recall_filtered - recall_none

        record_metric("scoping_boost", f"recall_unfiltered_at_{n_drawers}", round(recall_none, 3))
        record_metric("scoping_boost", f"recall_filtered_at_{n_drawers}", round(recall_filtered, 3))
        record_metric("scoping_boost", f"boost_at_{n_drawers}", round(boost, 3))


@pytest.mark.benchmark
class TestFilterLatencyBenefit:
    """Does filtering reduce query latency by narrowing the search space?"""

    def test_filter_speedup(self, tmp_path, bench_scale):
        """Compare latency: no filter vs added_by filter."""
        gen = PalaceDataGenerator(seed=42, scale=bench_scale)
        palace_path = str(tmp_path / "palace")
        gen.populate_palace_directly(palace_path, n_drawers=5_000, include_needles=False)

        from mempalace.searcher import search_memories

        agent = gen.agents[0]
        query = "authentication middleware optimization"
        n_runs = 10

        # No filter
        latencies_none = []
        for _ in range(n_runs):
            start = time.perf_counter()
            search_memories(query, palace_path=palace_path, n_results=5)
            latencies_none.append((time.perf_counter() - start) * 1000)

        # Agent filter
        latencies_filtered = []
        for _ in range(n_runs):
            start = time.perf_counter()
            search_memories(query, palace_path=palace_path, added_by=agent, n_results=5)
            latencies_filtered.append((time.perf_counter() - start) * 1000)

        avg_none = sum(latencies_none) / len(latencies_none)
        avg_filtered = sum(latencies_filtered) / len(latencies_filtered)

        record_metric("filter_latency", "avg_unfiltered_ms", round(avg_none, 1))
        record_metric("filter_latency", "avg_filtered_ms", round(avg_filtered, 1))
        if avg_none > 0:
            record_metric(
                "filter_latency", "speedup_pct", round((1 - avg_filtered / avg_none) * 100, 1)
            )


@pytest.mark.benchmark
class TestBoostAtIncreasingScale:
    """Does the scoping boost increase as the palace grows?"""

    def test_boost_scaling(self, tmp_path, bench_scale):
        """Measure agent-filtered recall improvement at multiple sizes."""
        sizes = [500, 1_000, 2_500]
        boosts = []

        for size in sizes:
            gen = PalaceDataGenerator(seed=42, scale=bench_scale)
            palace_path = str(tmp_path / f"palace_{size}")
            _, _, needle_info = gen.populate_palace_directly(
                palace_path, n_drawers=size, include_needles=True
            )

            from mempalace.searcher import search_memories

            n_queries = min(8, len(needle_info))
            unfiltered_hits = 0
            filtered_hits = 0

            for needle in needle_info[:n_queries]:
                result = search_memories(needle["query"], palace_path=palace_path, n_results=5)
                if any("NEEDLE_" in h["text"] for h in result.get("results", [])[:5]):
                    unfiltered_hits += 1

                result = search_memories(
                    needle["query"], palace_path=palace_path, added_by=needle["added_by"], n_results=5
                )
                if any("NEEDLE_" in h["text"] for h in result.get("results", [])[:5]):
                    filtered_hits += 1

            recall_none = unfiltered_hits / max(n_queries, 1)
            recall_filtered = filtered_hits / max(n_queries, 1)
            boost = recall_filtered - recall_none
            boosts.append({"size": size, "boost": boost})

        record_metric("boost_scaling", "boosts_by_size", boosts)
        # Check if boost increases with scale (the hypothesis)
        if len(boosts) >= 2:
            trend_positive = boosts[-1]["boost"] >= boosts[0]["boost"]
            record_metric("boost_scaling", "trend_positive", trend_positive)
