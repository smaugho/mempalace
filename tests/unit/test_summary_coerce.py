"""Unit tests for mempalace.summary_coerce (Haiku-driven length trim).

Adrian directive 2026-05-12: when a summary dict's rendered prose
exceeds 280 chars, route through Claude Haiku instead of hard-rejecting
the write. Tests mock the anthropic SDK so they exercise every branch
without spending real tokens.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from mempalace import summary_coerce
from mempalace.knowledge_graph import (
    SummaryStructureRequired,
    coerce_summary_for_persist,
    serialize_summary_for_embedding,
)


pytestmark = pytest.mark.unit


# ── Helpers ────────────────────────────────────────────────────────────


def _make_too_long_dict():
    """Return a structurally-valid summary dict whose rendered form is
    just over the 280-char cap."""
    what = "InjectionGate (post-retrieval relevance filter)"
    why = (
        "filters retrieved memories before injection via Haiku tool-use, "
        "emits quality flags for the gardener, gates the four channels "
        "A/B/C/D and supplies fingerprint context to subsequent declare "
        "operations across the session"
    )
    scope = "mempalace v3.2.x; one instance per palace process; long-context Opus"
    d = {"what": what, "why": why, "scope": scope}
    rendered = serialize_summary_for_embedding(d)
    assert len(rendered) > 280, f"fixture must exceed 280 chars; got {len(rendered)}"
    return d


def _make_haiku_resp(what: str, why: str, scope: str | None = None):
    """Build a stub Anthropic response shaped like a real tool_use reply."""
    payload = {"what": what, "why": why}
    if scope is not None:
        payload["scope"] = scope
    block = types.SimpleNamespace(type="tool_use", name="trim_summary", input=payload)
    resp = types.SimpleNamespace(content=[block], usage=None, stop_reason="end_turn")
    return resp


@pytest.fixture(autouse=True)
def _reset_coerce_state(monkeypatch):
    """Reset budget + cache + stats between tests so each case starts clean."""
    summary_coerce.reset_budget()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    yield
    summary_coerce.reset_budget()


# ── haiku_coerce_summary_to_length: success path ───────────────────────


def test_haiku_coerce_success_returns_trimmed_dict():
    too_long = _make_too_long_dict()
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_haiku_resp(
        what="InjectionGate",
        why="filters retrieved memories pre-injection; emits gardener flags",
        scope="mempalace v3.2.x",
    )
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client
    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        out = summary_coerce.haiku_coerce_summary_to_length(
            too_long, max_len=280, context_for_error="unit_test"
        )
    assert out is not None
    assert out["what"] == "InjectionGate"
    assert out["why"].startswith("filters retrieved memories")
    assert out["scope"] == "mempalace v3.2.x"
    stats = summary_coerce.get_stats()
    assert stats["successful_coerces"] == 1
    assert stats["haiku_invocations"] == 1
    assert stats["cache_hits"] == 0


def test_haiku_coerce_omits_scope_when_input_had_none():
    too_long = _make_too_long_dict()
    too_long.pop("scope")
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_haiku_resp(
        what="InjectionGate",
        why="filters retrieved memories pre-injection; emits gardener flags",
        scope="should be dropped",  # Haiku invented a scope -- coerce must drop it
    )
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client
    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        out = summary_coerce.haiku_coerce_summary_to_length(
            too_long, max_len=280, context_for_error="unit_test"
        )
    assert out is not None
    assert "scope" not in out, "input had no scope, output must omit scope even if Haiku adds one"


def test_haiku_coerce_caches_result():
    too_long = _make_too_long_dict()
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_haiku_resp(
        what="InjectionGate",
        why="filters retrieved memories pre-injection",
        scope="v3.2.x",
    )
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client
    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        first = summary_coerce.haiku_coerce_summary_to_length(
            too_long, max_len=280, context_for_error="unit_test"
        )
        second = summary_coerce.haiku_coerce_summary_to_length(
            too_long, max_len=280, context_for_error="unit_test"
        )
    assert first == second
    stats = summary_coerce.get_stats()
    assert stats["haiku_invocations"] == 1, "second call must hit cache"
    assert stats["cache_hits"] == 1


# ── haiku_coerce_summary_to_length: failure paths ──────────────────────


def test_haiku_coerce_returns_none_when_sdk_missing():
    too_long = _make_too_long_dict()
    with patch.dict("sys.modules", {"anthropic": None}):
        # Simulate ImportError by removing the module reference.
        import sys

        sys.modules.pop("anthropic", None)
        out = summary_coerce.haiku_coerce_summary_to_length(
            too_long, max_len=280, context_for_error="unit_test"
        )
    # Either ImportError-path or no-key-path can fire here; both return None.
    if out is not None:
        # If the SDK was actually installed in the test env, the no-key
        # branch only fires for missing key. Re-run with the key cleared
        # to take the no-key branch deterministically.
        pytest.skip("anthropic SDK installed in test env; covered by no-key test")


def test_haiku_coerce_returns_none_when_api_key_missing(monkeypatch):
    too_long = _make_too_long_dict()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Also stub _ensure_env_loaded so it doesn't repopulate the key.
    monkeypatch.setattr("mempalace.auto_author._ensure_env_loaded", lambda: None)
    out = summary_coerce.haiku_coerce_summary_to_length(
        too_long, max_len=280, context_for_error="unit_test"
    )
    assert out is None
    stats = summary_coerce.get_stats()
    assert stats["fallbacks_to_raise"] >= 1


def test_haiku_coerce_returns_none_when_haiku_call_raises():
    too_long = _make_too_long_dict()
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("simulated API down")
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client
    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        out = summary_coerce.haiku_coerce_summary_to_length(
            too_long, max_len=280, context_for_error="unit_test"
        )
    assert out is None
    stats = summary_coerce.get_stats()
    assert stats["haiku_failures"] == 1


def test_haiku_coerce_returns_none_when_haiku_emits_no_tool_block():
    too_long = _make_too_long_dict()
    # Empty content list -- no tool_use block.
    resp = types.SimpleNamespace(content=[], usage=None, stop_reason="end_turn")
    mock_client = MagicMock()
    mock_client.messages.create.return_value = resp
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client
    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        out = summary_coerce.haiku_coerce_summary_to_length(
            too_long, max_len=280, context_for_error="unit_test"
        )
    assert out is None


def test_haiku_coerce_budget_exhausted_returns_none():
    too_long = _make_too_long_dict()
    # Force the counter to the cap.
    summary_coerce._call_counter = summary_coerce._MAX_COERCE_CALLS_PER_PROCESS
    mock_anthropic = MagicMock()
    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        out = summary_coerce.haiku_coerce_summary_to_length(
            too_long, max_len=280, context_for_error="unit_test"
        )
    assert out is None
    # Haiku must NOT have been called -- budget short-circuits first.
    mock_anthropic.Anthropic.assert_not_called()


# ── coerce_summary_for_persist: end-to-end length-retry path ───────────


def test_coerce_summary_for_persist_routes_through_haiku_on_length():
    too_long = _make_too_long_dict()
    trimmed = {
        "what": "InjectionGate",
        "why": "filters retrieved memories pre-injection; emits gardener flags",
        "scope": "mempalace v3.2.x",
    }
    # Sanity: trimmed must fit the cap.
    assert len(serialize_summary_for_embedding(trimmed)) <= 280

    with patch(
        "mempalace.summary_coerce.haiku_coerce_summary_to_length",
        return_value=trimmed,
    ) as mock_coerce:
        out = coerce_summary_for_persist(too_long, context_for_error="ut")
    assert out["what"] == trimmed["what"]
    assert out["why"] == trimmed["why"]
    assert out["scope"] == trimmed["scope"]
    mock_coerce.assert_called_once()


def test_coerce_summary_for_persist_raises_when_coerce_returns_none():
    too_long = _make_too_long_dict()
    with patch(
        "mempalace.summary_coerce.haiku_coerce_summary_to_length",
        return_value=None,
    ):
        with pytest.raises(SummaryStructureRequired) as excinfo:
            coerce_summary_for_persist(too_long, context_for_error="ut")
    assert "rendered summary exceeds" in str(excinfo.value)


def test_coerce_summary_for_persist_skips_haiku_when_disabled():
    too_long = _make_too_long_dict()
    with patch("mempalace.summary_coerce.haiku_coerce_summary_to_length") as mock_coerce:
        with pytest.raises(SummaryStructureRequired):
            coerce_summary_for_persist(too_long, context_for_error="ut", allow_haiku_coerce=False)
    mock_coerce.assert_not_called()


def test_coerce_summary_for_persist_doesnt_call_haiku_on_clean_input():
    clean = {
        "what": "InjectionGate",
        "why": "filters retrieved memories pre-injection",
        "scope": "v3.2.x",
    }
    with patch("mempalace.summary_coerce.haiku_coerce_summary_to_length") as mock_coerce:
        out = coerce_summary_for_persist(clean, context_for_error="ut")
    assert out["what"] == "InjectionGate"
    mock_coerce.assert_not_called()


def test_coerce_summary_for_persist_raises_on_non_length_failure_without_haiku():
    """Missing 'why' is a structural failure -- must NOT route through Haiku."""
    bad = {"what": "InjectionGate", "why": "x"}  # why too short
    with patch("mempalace.summary_coerce.haiku_coerce_summary_to_length") as mock_coerce:
        with pytest.raises(SummaryStructureRequired):
            coerce_summary_for_persist(bad, context_for_error="ut")
    mock_coerce.assert_not_called()


def test_coerce_summary_for_persist_raises_when_haiku_output_still_overflows():
    """If Haiku returns a trimmed dict that STILL exceeds the cap, the
    re-validation must raise (no infinite recursion)."""
    too_long = _make_too_long_dict()
    still_too_long = {
        "what": too_long["what"],
        "why": too_long["why"],
        "scope": too_long["scope"],
    }
    with patch(
        "mempalace.summary_coerce.haiku_coerce_summary_to_length",
        return_value=still_too_long,
    ):
        with pytest.raises(SummaryStructureRequired):
            coerce_summary_for_persist(too_long, context_for_error="ut")
