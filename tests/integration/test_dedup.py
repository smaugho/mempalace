"""Tests for mempalace.dedup -- near-duplicate memory detection and removal.

Post-Tier-1 (commit 58b6509): dedup.py routes through VectorStore. The
helpers `get_source_groups` and `dedup_source_group` now take a `vs`
arg (was raw chromadb collection); `vs.get(...)` returns a GetResult
dataclass with .ids/.documents/.metadatas attrs (was a raw dict);
`vs.query(...)` returns a QueryResult with .ids/.distances/.is_degraded
(was a raw dict). The show_stats / dedup_palace entry points patch
`mempalace.dedup.get_vector_store` (was `mempalace.dedup.chromadb`).
"""

import pytest

from unittest.mock import MagicMock, patch


from mempalace import dedup
from mempalace.vector_store import GetResult, QueryResult


def _meta_for(srcs):
    return [{"source_file": s} for s in srcs]


# ── get_source_groups ─────────────────────────────────────────────────


def test_get_source_groups_basic():
    vs = MagicMock()
    vs.count.return_value = 5
    vs.get.side_effect = [
        GetResult(
            ids=["d1", "d2", "d3", "d4", "d5"],
            metadatas=_meta_for(["a.txt"] * 5),
        ),
        GetResult(ids=[]),
    ]
    groups = dedup.get_source_groups(vs, min_count=5)
    assert "a.txt" in groups
    assert len(groups["a.txt"]) == 5


def test_get_source_groups_below_min():
    vs = MagicMock()
    vs.count.return_value = 2
    vs.get.side_effect = [
        GetResult(
            ids=["d1", "d2"],
            metadatas=_meta_for(["a.txt", "a.txt"]),
        ),
        GetResult(ids=[]),
    ]
    groups = dedup.get_source_groups(vs, min_count=5)
    assert len(groups) == 0


def test_get_source_groups_source_filter():
    vs = MagicMock()
    vs.count.return_value = 6
    vs.get.side_effect = [
        GetResult(
            ids=["d1", "d2", "d3", "d4", "d5", "d6"],
            metadatas=_meta_for(["project_a.txt"] * 5 + ["other.txt"]),
        ),
        GetResult(ids=[]),
    ]
    groups = dedup.get_source_groups(vs, min_count=5, source_pattern="project_a")
    assert "project_a.txt" in groups
    assert "other.txt" not in groups


def test_get_source_groups_agent_filter():
    vs = MagicMock()
    vs.count.return_value = 5
    vs.get.side_effect = [
        GetResult(
            ids=["d1", "d2", "d3", "d4", "d5"],
            metadatas=_meta_for(["a.txt"] * 5),
        ),
        GetResult(ids=[]),
    ]
    dedup.get_source_groups(vs, min_count=5, agent="my_agent")
    # Verify where filter was passed through vs.get(where=...)
    first_call = vs.get.call_args_list[0]
    assert first_call.kwargs.get("where") == {"added_by": "my_agent"}


def test_get_source_groups_missing_source_file():
    vs = MagicMock()
    vs.count.return_value = 5
    vs.get.side_effect = [
        GetResult(
            ids=["d1", "d2", "d3", "d4", "d5"],
            metadatas=[{}, {}, {}, {}, {}],
        ),
        GetResult(ids=[]),
    ]
    groups = dedup.get_source_groups(vs, min_count=5)
    assert "unknown" in groups


# ── dedup_source_group ────────────────────────────────────────────────


def test_dedup_source_group_all_unique():
    vs = MagicMock()
    vs.get.return_value = GetResult(
        ids=["d1", "d2"],
        documents=["long document one content here", "different document two here"],
        metadatas=[{"added_by": "a"}, {"added_by": "a"}],
    )
    vs.query.return_value = QueryResult(
        ids=[["d1"]],
        distances=[[0.8]],  # far apart = unique
    )
    kept, deleted = dedup.dedup_source_group(vs, ["d1", "d2"], threshold=0.15, dry_run=True)
    assert len(kept) == 2
    assert len(deleted) == 0


def test_dedup_source_group_with_duplicate():
    vs = MagicMock()
    vs.get.return_value = GetResult(
        ids=["d1", "d2"],
        documents=[
            "long document content that is fairly long",
            "long document content that is fairly long",
        ],
        metadatas=[{"added_by": "a"}, {"added_by": "a"}],
    )
    vs.query.return_value = QueryResult(
        ids=[["d1"]],
        distances=[[0.05]],  # very close = duplicate
    )
    kept, deleted = dedup.dedup_source_group(vs, ["d1", "d2"], threshold=0.15, dry_run=True)
    assert len(kept) == 1
    assert len(deleted) == 1


def test_dedup_source_group_short_docs_deleted():
    vs = MagicMock()
    vs.get.return_value = GetResult(
        ids=["d1", "d2"],
        documents=["long enough document to keep in the palace", "tiny"],
        metadatas=[{"added_by": "a"}, {"added_by": "a"}],
    )
    kept, deleted = dedup.dedup_source_group(vs, ["d1", "d2"], threshold=0.15, dry_run=True)
    assert "d2" in deleted  # too short


def test_dedup_source_group_empty_doc_deleted():
    vs = MagicMock()
    vs.get.return_value = GetResult(
        ids=["d1", "d2"],
        documents=["real document content here that is long enough", None],
        metadatas=[{"added_by": "a"}, {"added_by": "a"}],
    )
    kept, deleted = dedup.dedup_source_group(vs, ["d1", "d2"], threshold=0.15, dry_run=True)
    assert "d2" in deleted


def test_dedup_source_group_live_deletes():
    vs = MagicMock()
    vs.get.return_value = GetResult(
        ids=["d1", "d2"],
        documents=["long document content here enough", "long document content here enough"],
        metadatas=[{"added_by": "a"}, {"added_by": "a"}],
    )
    vs.query.return_value = QueryResult(
        ids=[["d1"]],
        distances=[[0.05]],
    )
    kept, deleted = dedup.dedup_source_group(vs, ["d1", "d2"], threshold=0.15, dry_run=False)
    vs.delete.assert_called_once()


def test_dedup_source_group_query_failure_keeps():
    """Post-Tier-1: vs.query returns a degraded QueryResult on internal
    failure (rather than raising). dedup's fail-open branch checks
    qres.is_degraded and keeps the row -- same behaviour as the
    pre-Tier-1 try/except wrapping a raising col.query.
    """
    vs = MagicMock()
    vs.get.return_value = GetResult(
        ids=["d1", "d2"],
        documents=[
            "long document one content here enough",
            "long document two content here enough",
        ],
        metadatas=[{"added_by": "a"}, {"added_by": "a"}],
    )
    vs.query.return_value = QueryResult.empty(n_query_texts=1, reason="failed: query failed")
    kept, deleted = dedup.dedup_source_group(vs, ["d1", "d2"], threshold=0.15, dry_run=True)
    assert len(kept) == 2  # both kept on degraded query


# ── show_stats ────────────────────────────────────────────────────────


@patch("mempalace.dedup.get_vector_store")
def test_show_stats(mock_get_vs, tmp_path):
    mock_vs = MagicMock()
    mock_vs.count.return_value = 5
    mock_vs.get.side_effect = [
        GetResult(
            ids=["d1", "d2", "d3", "d4", "d5"],
            metadatas=_meta_for(["a.txt"] * 5),
        ),
        GetResult(ids=[]),
    ]
    mock_get_vs.return_value = mock_vs

    dedup.show_stats(palace_path=str(tmp_path))  # should not raise


# ── dedup_palace ──────────────────────────────────────────────────────


@patch("mempalace.dedup.dedup_source_group")
@patch("mempalace.dedup.get_source_groups")
@patch("mempalace.dedup.get_vector_store")
def test_dedup_palace_dry_run(mock_get_vs, mock_groups, mock_dedup_group, tmp_path):
    mock_vs = MagicMock()
    mock_vs.count.return_value = 10
    mock_get_vs.return_value = mock_vs

    mock_groups.return_value = {"a.txt": ["d1", "d2", "d3", "d4", "d5"]}
    mock_dedup_group.return_value = (["d1", "d2", "d3"], ["d4", "d5"])

    dedup.dedup_palace(palace_path=str(tmp_path), dry_run=True)
    mock_dedup_group.assert_called_once()


@patch("mempalace.dedup.dedup_source_group")
@patch("mempalace.dedup.get_source_groups")
@patch("mempalace.dedup.get_vector_store")
def test_dedup_palace_with_agent(mock_get_vs, mock_groups, mock_dedup_group, tmp_path):
    mock_vs = MagicMock()
    mock_vs.count.return_value = 10
    mock_get_vs.return_value = mock_vs

    mock_groups.return_value = {}
    dedup.dedup_palace(palace_path=str(tmp_path), agent="test_agent", dry_run=True)
    mock_groups.assert_called_once_with(mock_vs, 5, None, agent="test_agent")


@patch("mempalace.dedup.dedup_source_group")
@patch("mempalace.dedup.get_source_groups")
@patch("mempalace.dedup.get_vector_store")
def test_dedup_palace_no_groups(mock_get_vs, mock_groups, mock_dedup_group, tmp_path):
    mock_vs = MagicMock()
    mock_vs.count.return_value = 3
    mock_get_vs.return_value = mock_vs

    mock_groups.return_value = {}
    dedup.dedup_palace(palace_path=str(tmp_path), dry_run=True)
    mock_dedup_group.assert_not_called()


pytestmark = pytest.mark.integration
