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
        # FINDING #L (v3.7.25 2026-05-18, Adrian): no "\n\n" separator
        # means the preview has no separable content body -- the summary
        # IS the full information. Content must be "" and
        # content_redundant must be False (there is no suppressed body
        # to flag). Pre-v3.7.25 this branch aliased content=text which,
        # combined with the SequenceMatcher dedup gate, flagged
        # content_redundant=True on EVERY entity-kind preview (because
        # ratio(text, text) == 1.0 > 0.75 default threshold). Adrian
        # observed every retrieval marked redundant; this test locks
        # the fix.
        s, c, trim, redun = intent._split_for_surface("just a single line")
        assert s == "just a single line"
        assert c == ""
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


# ─────────────────────────────────────────────────────────────────────
# FINDING #L (v3.7.25, 2026-05-18, Adrian's observation): legacy
# fallback in _split_for_surface used to alias ``content_part = text``
# when the preview had no ``\n\n`` separator. The SequenceMatcher
# dedup gate then trivially scored ratio(text, text) = 1.0 against
# the default 0.75 threshold and flagged EVERY entity-kind preview as
# content_redundant=True. Adrian noticed "almost everything is marked
# as content redundant" -- root cause was this false-positive cascade,
# NOT the dedup gate threshold and NOT the encoder choice (it's
# difflib.SequenceMatcher, neither biencoder nor cross-encoder, per
# Adrian's clarifying question).
# ─────────────────────────────────────────────────────────────────────


class TestFindingL_NoFalsePositiveOnLegacyFallback:
    """Lock the FINDING #L fix: previews without ``\n\n`` separator
    (entity-kind hits via serialize_summary_for_embedding) must NOT
    surface as content_redundant=True at the default threshold."""

    def test_entity_kind_preview_at_default_threshold_not_flagged(self, monkeypatch):
        """The exact symptom: a single-line entity preview (no \n\n)
        at the default 0.75 threshold must not flag redundant."""
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000",
            # Use the default threshold deliberately -- pre-fix, ANY
            # threshold below 1.0 inclusive flagged every entity-kind
            # preview as redundant because ratio(text, text) = 1.0.
            MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD="0.75",
        )
        # Realistic entity-kind preview shape (one-line summary,
        # no separable content body -- emitted by
        # render_memory_preview via serialize_summary_for_embedding).
        text = (
            "mempalace/intent.py (intent module + sub-agent enforcement) "
            "-- owns declare_intent, sub-agent cause_id rules, "
            "state-judge, intent lifecycle; mempalace core"
        )
        s, c, trim, redun = intent._split_for_surface(text)
        assert s, "summary should be the shortened preview text"
        assert c == "", "no separable content body -- content must be empty"
        assert trim is False
        assert redun is False, (
            "FINDING #L regression: entity-kind preview must NOT be "
            "flagged content_redundant on the legacy-fallback path"
        )

    def test_true_dedup_still_works_when_separator_present(self, monkeypatch):
        """Confirm the fix didn't break the actual dedup gate: when
        a real ``summary\\n\\ncontent`` preview HAS the separator AND
        the two halves are duplicate, the gate must still fire."""
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000",
            MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD="0.75",
        )
        text = "v3.7.25 ship\n\nv3.7.25 ship"
        s, c, trim, redun = intent._split_for_surface(text)
        assert s == "v3.7.25 ship"
        assert c == ""
        assert redun is True, (
            "real summary==content duplicate must still flag redundant; "
            "the FINDING #L fix only suppresses the legacy-fallback "
            "false positive, not the genuine dedup case"
        )

    def test_real_elaboration_not_flagged(self, monkeypatch):
        """When summary and content are distinct elaborations (not
        verbatim restates), the gate must not flag -- otherwise we
        suppress legitimate body content."""
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000",
            MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD="0.75",
        )
        text = (
            "FINDING L summary\n\n"
            "Detailed write-up of why the legacy fallback aliased "
            "content_part to the same text, how the SequenceMatcher "
            "dedup gate trivially scored 1.0 on identical strings, "
            "and the user-visible symptom of every retrieval marking "
            "content_redundant for six weeks of session output."
        )
        s, c, trim, redun = intent._split_for_surface(text)
        assert s == "FINDING L summary"
        assert c, "real elaboration content must be surfaced"
        assert redun is False
