"""
test_relevance_single_field.py — API cutover: one signed ``relevance`` 1-5.

After 2026-04-22 the agent-facing memory_feedback entry shape is
``{id, relevance, reason, ...}`` — the legacy ``relevant`` bool is
derived server-side from the integer. These tests lock in the mapping
(plan §Rating rubric follow-up), the back-compat override path, and
downstream behaviour preservation (decay reset, Rocchio bucket).

Grounding: CrowdTruth 2.0 (Aroyo & Welty) — preserving disagreement
is signal; Davani et al. 2022 TACL — relevance is subjective, no
ground truth, full dynamic range matters more than scalar collapse.
"""

from __future__ import annotations

import pytest

from mempalace.intent import _RELEVANCE_MAPPING, _derive_feedback_pair


# ─────────────────────────────────────────────────────────────────────
# The exact mapping
# ─────────────────────────────────────────────────────────────────────


class TestMapping:
    @pytest.mark.parametrize(
        "score,expected_relevant,expected_conf",
        [
            (1, False, 1.0),
            (2, False, 0.5),
            (3, True, 0.2),
            (4, True, 0.8),
            (5, True, 1.0),
        ],
    )
    def test_each_level_maps_correctly(self, score, expected_relevant, expected_conf):
        _, relevant, confidence = _derive_feedback_pair({"relevance": score})
        assert relevant is expected_relevant
        assert confidence == pytest.approx(expected_conf)

    def test_mapping_dict_contract(self):
        """The module-level _RELEVANCE_MAPPING IS the public contract;
        downstream writers (finalize feedback loop) and tests both read
        from it. Drift here is a drift there."""
        assert _RELEVANCE_MAPPING == {
            1: (False, 1.0),
            2: (False, 0.5),
            3: (True, 0.2),
            4: (True, 0.8),
            5: (True, 1.0),
        }

    def test_signed_range_covers_full_minus_one_to_plus_one(self):
        """Signal = confidence × sign. The mapping must span [-1.0, +1.0]
        with a weak-positive floor at value 3 to preserve the pre-cutover
        signal geometry that Channel D + W_REL were tuned against."""
        signed = []
        for score in range(1, 6):
            _, relevant, conf = _derive_feedback_pair({"relevance": score})
            signed.append(conf * (1 if relevant else -1))
        # Monotonic strictly increasing.
        assert signed == sorted(signed)
        # Range endpoints.
        assert signed[0] == -1.0
        assert signed[-1] == +1.0
        # Weak-positive floor at 3 (NOT 0.0 — preserves "related
        # context, didn't change what I did" as a real signal).
        assert signed[2] > 0.0
        assert signed[2] < 0.5


# ─────────────────────────────────────────────────────────────────────
# Default / coercion behaviour
# ─────────────────────────────────────────────────────────────────────


class TestDefaults:
    def test_missing_relevance_defaults_to_three(self):
        score, relevant, conf = _derive_feedback_pair({})
        assert score == 3
        assert relevant is True
        assert conf == pytest.approx(0.2)

    def test_non_integer_defaults_to_three(self):
        score, relevant, conf = _derive_feedback_pair({"relevance": "pretty good"})
        assert score == 3

    def test_out_of_range_low_coerces_to_three(self):
        """Value 0 is a policy violation (scale is 1-5). Coerce to 3
        rather than silently clamping — matches the "default when
        unsure" anchor."""
        score, _, _ = _derive_feedback_pair({"relevance": 0})
        assert score == 3

    def test_out_of_range_high_coerces_to_three(self):
        score, _, _ = _derive_feedback_pair({"relevance": 7})
        assert score == 3

    def test_float_truncated_to_int(self):
        """Agents sending 4.7 (half-thoughts) get treated as 4, not 5."""
        score, _, _ = _derive_feedback_pair({"relevance": 4.7})
        assert score == 4


# ─────────────────────────────────────────────────────────────────────
# No back-compat: `relevant` bool is dead, only `relevance` matters
# ─────────────────────────────────────────────────────────────────────


class TestNoBackCompatRelevantOverride:
    def test_explicit_relevant_is_ignored(self):
        """Any ``relevant`` key in the feedback entry is silently ignored.
        Sign is derived exclusively from the integer — single-field API.
        Callers can no longer shape the signal via the legacy bool."""
        # Pass conflicting explicit value; helper must ignore it.
        score, relevant, conf = _derive_feedback_pair({"relevance": 1, "relevant": True})
        # Integer 1 → (False, 1.0) per mapping; explicit True does NOT win.
        assert relevant is False
        assert conf == pytest.approx(1.0)
        assert score == 1

    def test_relevant_none_is_ignored(self):
        """Same rule: None, True, False, any value — all ignored."""
        _, relevant, _ = _derive_feedback_pair({"relevance": 4, "relevant": None})
        # Integer 4 → (True, 0.8) per mapping; None doesn't flip it.
        assert relevant is True
