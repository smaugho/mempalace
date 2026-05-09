"""
test_searcher.py -- Tests for both search() (CLI) and search_memories() (API).

Uses the real ChromaDB fixtures from conftest.py for integration tests,
plus mock-based tests for error paths.
"""

from unittest.mock import MagicMock, patch

import pytest

from mempalace.searcher import SearchError, search, search_memories


# ── search_memories (API) ──────────────────────────────────────────────


class TestSearchMemories:
    def test_basic_search(self, palace_path, seeded_collection):
        result = search_memories("JWT authentication", palace_path)
        assert "results" in result
        assert len(result["results"]) > 0
        assert result["query"] == "JWT authentication"

    def test_added_by_filter(self, palace_path, seeded_collection):
        result = search_memories("planning", palace_path, added_by="miner")
        assert all(r["added_by"] == "miner" for r in result["results"])

    def test_n_results_limit(self, palace_path, seeded_collection):
        result = search_memories("code", palace_path, n_results=2)
        assert len(result["results"]) <= 2

    def test_no_palace_returns_error(self, tmp_path):
        result = search_memories("anything", str(tmp_path / "missing"))
        assert "error" in result

    def test_result_fields(self, palace_path, seeded_collection):
        result = search_memories("authentication", palace_path)
        hit = result["results"][0]
        # Vocab lock 2026-05-01: rendered memory preview lives under
        # the canonical "summary_text" key (was "text" pre-rename).
        assert "summary_text" in hit
        assert "added_by" in hit
        assert "content_type" in hit
        assert "source_file" in hit
        assert "similarity" in hit
        assert isinstance(hit["similarity"], float)

    def test_search_memories_query_error(self):
        """search_memories returns error dict when query raises.

        Post-Tier-1 (commit 58b6509): searcher.py routes through
        VectorStore, not chromadb.PersistentClient directly. The
        mock now intercepts get_vector_store and returns a degraded
        QueryResult so the searcher's qres.is_degraded branch fires
        the same "no palace found" / error response shape the original
        test was exercising.
        """
        from mempalace.vector_store import QueryResult

        mock_vs = MagicMock()
        mock_vs.query.return_value = QueryResult.empty(
            n_query_texts=1, reason="unavailable: query failed"
        )
        with patch("mempalace.searcher.get_vector_store", return_value=mock_vs):
            result = search_memories("test", "/fake/path")
        assert "error" in result
        assert "No palace found" in result["error"]

    def test_search_memories_filters_in_result(self, palace_path, seeded_collection):
        result = search_memories("test", palace_path, added_by="miner")
        assert result["filters"]["added_by"] == "miner"


# ── search() (CLI print function) ─────────────────────────────────────


class TestSearchCLI:
    def test_search_prints_results(self, palace_path, seeded_collection, capsys):
        search("JWT authentication", palace_path)
        captured = capsys.readouterr()
        assert "JWT" in captured.out or "authentication" in captured.out

    def test_search_with_added_by_filter(self, palace_path, seeded_collection, capsys):
        search("planning", palace_path, added_by="miner")
        captured = capsys.readouterr()
        assert "Results for" in captured.out

    def test_search_no_palace_raises(self, tmp_path):
        with pytest.raises(SearchError, match="No palace found"):
            search("anything", str(tmp_path / "missing"))

    def test_search_no_results(self, palace_path, collection, capsys):
        """Empty collection returns no results message."""
        # collection is empty (no seeded data)
        result = search("xyzzy_nonexistent_query", palace_path, n_results=1)
        captured = capsys.readouterr()
        # Either prints "No results" or returns None
        assert result is None or "No results" in captured.out

    def test_search_query_error_raises(self):
        """search raises SearchError when palace is unavailable.

        Post-Tier-1: searcher.py routes through VectorStore. A degraded
        QueryResult with "unavailable" reason maps to the SearchError
        the CLI raises (see searcher.search line 47-50).
        """
        from mempalace.vector_store import QueryResult

        mock_vs = MagicMock()
        mock_vs.query.return_value = QueryResult.empty(
            n_query_texts=1, reason="unavailable: collection missing"
        )
        with patch("mempalace.searcher.get_vector_store", return_value=mock_vs):
            with pytest.raises(SearchError, match="No palace found"):
                search("test", "/fake/path")

    def test_search_n_results(self, palace_path, seeded_collection, capsys):
        search("code", palace_path, n_results=1)
        captured = capsys.readouterr()
        # Should have output with at least one result block
        assert "[1]" in captured.out


pytestmark = pytest.mark.integration
