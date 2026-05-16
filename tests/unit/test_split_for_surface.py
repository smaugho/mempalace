"""Unit tests for mempalace.intent._split_for_surface.

Locks two behaviors:
- v3.5.9: surface (summary, content) split with cap+trim marker.
- v3.7.3: similarity-dedup gate (Adrian directive 2026-05-16) -- when
  difflib.SequenceMatcher ratio(summary, content) >=
  MEMORY_CONTENT_DEDUP_THRESHOLD (default 0.75), content is blanked
  and content_redundant=True is returned so callers can surface the
  suppression intentionally.

Tests use ``importlib.reload`` to re-read env vars cleanly per case
without leaking module-level constants across tests.
"""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.unit


def _reload_intent(monkeypatch, **env):
    """Reload mempalace.intent with the given env vars applied so
    module-level constants (MEMORY_CONTENT_MAX_CHARS,
    MEMORY_CONTENT_DEDUP_THRESHOLD) pick up the new values."""
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    from mempalace import intent

    return importlib.reload(intent)


# ─────────────────────────────────────────────────────────────────────
# v3.5.9: summary/content split + cap+trim marker
# ─────────────────────────────────────────────────────────────────────


class TestSplitBasic:
    def test_summary_only_text_yields_empty_content(self, monkeypatch):
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000",
            MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD="0",  # disable dedup
        )
        # No "\n\n" separator -- the whole text becomes both summary
        # AND content (legacy-record fallback). With dedup off, the
        # full text surfaces as both halves.
        s, c, trim, redun = intent._split_for_surface("just a single line")
        assert s == "just a single line"
        assert c == "just a single line"
        assert trim is False
        assert redun is False

    def test_summary_plus_content_split(self, monkeypatch):
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000",
            MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD="0",
        )
        text = "Short summary line\n\nLong content body with details."
        s, c, trim, redun = intent._split_for_surface(text)
        assert s == "Short summary line"
        assert c == "Long content body with details."
        assert trim is False
        assert redun is False

    def test_content_cap_marker_added(self, monkeypatch):
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_MAX_CHARS="40",
            MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD="0",
        )
        text = "Short summary line\n\n" + ("X" * 100)
        s, c, trim, redun = intent._split_for_surface(text)
        assert s == "Short summary line"
        assert trim is True
        assert "trimmed at 40 chars" in c
        assert redun is False

    def test_max_chars_zero_disables_content_entirely(self, monkeypatch):
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_MAX_CHARS="0",
        )
        s, c, trim, redun = intent._split_for_surface("summary\n\ncontent")
        # Whole-text path through _shorten_preview returns summary only.
        assert c == ""
        assert trim is False
        assert redun is False


# ─────────────────────────────────────────────────────────────────────
# v3.7.3: similarity-dedup gate
# ─────────────────────────────────────────────────────────────────────


class TestSimilarityDedup:
    def test_verbatim_restate_is_suppressed(self, monkeypatch):
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000",
            MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD="0.75",
        )
        # Summary and content are identical -- ratio() == 1.0, well
        # above 0.75 threshold.
        text = "v3.7.3 ship shipped\n\nv3.7.3 ship shipped"
        s, c, trim, redun = intent._split_for_surface(text)
        assert s == "v3.7.3 ship shipped"
        assert c == ""
        assert trim is False
        assert redun is True

    def test_light_paraphrase_is_suppressed(self, monkeypatch):
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000",
            MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD="0.75",
        )
        # 90%+ character overlap -- light edit, should be suppressed.
        text = "Adrian shipped v3.7.3 today\n\nAdrian shipped v3.7.3 today."  # only added period
        s, c, trim, redun = intent._split_for_surface(text)
        assert redun is True
        assert c == ""

    def test_distinct_content_is_kept(self, monkeypatch):
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000",
            MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD="0.75",
        )
        # Summary names the topic; content provides substantive
        # elaboration with different vocabulary. Low SequenceMatcher
        # ratio -- not suppressed.
        text = (
            "v3.7.3 similarity-dedup gate\n\n"
            "Implementation: difflib.SequenceMatcher computes ratio "
            "between summary lowercase and content lowercase; ratio "
            ">= MEMORY_CONTENT_DEDUP_THRESHOLD triggers blanking. "
            "Default 0.75 catches verbatim and light-paraphrase "
            "restates without filtering genuine elaboration. "
            "Caller dicts gain content_redundant=True when fired."
        )
        s, c, trim, redun = intent._split_for_surface(text)
        assert redun is False
        assert c.startswith("Implementation: difflib")
        assert trim is False

    def test_threshold_zero_disables_dedup_completely(self, monkeypatch):
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000",
            MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD="0",
        )
        # Even with identical summary + content, threshold=0 disables
        # the gate so content surfaces verbatim.
        text = "identical\n\nidentical"
        s, c, trim, redun = intent._split_for_surface(text)
        assert redun is False
        assert c == "identical"

    def test_threshold_one_only_blanks_verbatim_match(self, monkeypatch):
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000",
            MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD="1.0",
        )
        # Exact match -- ratio 1.0 >= 1.0 -- suppressed.
        s, c, trim, redun = intent._split_for_surface("same\n\nsame")
        assert redun is True
        # Light edit -- ratio < 1.0 -- kept.
        s, c, trim, redun = intent._split_for_surface("Same line\n\nSame line!")
        assert redun is False
        assert c == "Same line!"


# ─────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_non_string_input_short_circuits(self, monkeypatch):
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000",
        )
        # Bytes/None/dict all fall through _shorten_preview unchanged.
        s, c, trim, redun = intent._split_for_surface(None)
        assert s is None
        assert c == ""
        assert trim is False
        assert redun is False

    def test_dedup_threshold_env_invalid_falls_back_to_default(self, monkeypatch):
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD="not-a-number",
        )
        assert intent.MEMORY_CONTENT_DEDUP_THRESHOLD == 0.75
