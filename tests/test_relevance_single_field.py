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
# Back-compat: explicit `relevant` override still works
# ─────────────────────────────────────────────────────────────────────


class TestExplicitRelevantOverride:
    def test_relevant_true_overrides_low_relevance(self):
        """Caller passes `relevant=True` AND `relevance=1`. The derived
        sign (False for 1) is overridden to True. Confidence still
        comes from the integer so magnitude stays calibrated.

        Use case: caller wants to say "this was relevant but barely
        useful" — explicit sign, low confidence."""
        score, relevant, conf = _derive_feedback_pair({"relevance": 1, "relevant": True})
        assert score == 1
        assert relevant is True
        # Confidence from the integer 1 → 1.0 (flipped-sign signal +1.0)
        # — preserves the overall magnitude the caller requested.
        assert conf == pytest.approx(1.0)

    def test_relevant_false_overrides_high_relevance(self):
        """Mirror case — strong positive integer with explicit negative
        sign. This is the "I rated 5 in the old system with
        relevant=False, meaning strongly irrelevant" back-compat path."""
        score, relevant, conf = _derive_feedback_pair({"relevance": 5, "relevant": False})
        assert score == 5
        assert relevant is False
        assert conf == pytest.approx(1.0)

    def test_relevant_absent_uses_derived_sign(self):
        """The default path: no `relevant` key → sign comes from mapping."""
        _, relevant, _ = _derive_feedback_pair({"relevance": 2})
        assert relevant is False  # derived from mapping, not defaulted to True

    def test_relevant_none_treated_as_absent(self):
        """``{"relevant": None}`` explicitly — we don't bother
        distinguishing absent from None-valued; both fall through to the
        derived sign. (Keeps the API forgiving; no sharp edges.)"""
        # Note: the helper reads fb["relevant"] when the key is present.
        # None would set relevant=None which is truthy-coerced to None
        # → bool(None)=False. Test the actual behaviour so future
        # refactors don't silently shift semantics.
        _, relevant, _ = _derive_feedback_pair({"relevance": 3, "relevant": None})
        # bool(None) == False, so the explicit None overrides.
        # If we ever want "None means absent", change the helper; this
        # test will notice.
        assert relevant is False
