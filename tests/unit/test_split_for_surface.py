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
        # v3.7.30 (Adrian directive 2026-05-18): default bumped from
        # 0.75 (difflib char-overlap) to 0.85 (cosine on MiniLM-L6
        # embeddings). The measurement changed; the threshold had to
        # be re-calibrated for the cosine distribution. See intent.py
        # MEMORY_CONTENT_DEDUP_THRESHOLD docstring for the empirical
        # cosine distribution used to pick 0.85.
        assert intent.MEMORY_CONTENT_DEDUP_THRESHOLD == 0.85


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


# ─────────────────────────────────────────────────────────────────────
# v3.7.34: _project_memory hoists date_added + last_relevant_at so the
# agent can reason about WHEN memories were filed / last used. Pre-fix
# the entry was {id, summary_text, content?} with no datetime, so the
# decay applied at ranking time was invisible to the agent's own
# reasoning loop. See Adrian msg_138 2026-05-18.
# ─────────────────────────────────────────────────────────────────────


class TestProjectMemoryDateSurface:
    def test_hoists_date_added_from_metadata_dict(self, monkeypatch):
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000",
            MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD="0",
        )
        entry = intent._project_memory(
            "rec_x",
            "summary here\n\ncontent body",
            extras={"metadata": {"date_added": "2026-05-18T10:00:00"}},
        )
        # v3.7.39: surface trims to minute precision; assert trimmed.
        assert entry["date_added"] == "2026-05-18T10:00", (
            "v3.7.34 hoist + v3.7.39 trim: date_added inside "
            "extras['metadata'] must be hoisted AND trimmed to minutes"
        )

    def test_hoists_last_relevant_at_from_metadata_dict(self, monkeypatch):
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000",
            MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD="0",
        )
        entry = intent._project_memory(
            "rec_x",
            "summary",
            extras={
                "metadata": {
                    "date_added": "2026-04-01T00:00:00",
                    "last_relevant_at": "2026-05-17T12:00:00",
                }
            },
        )
        # v3.7.39: trimmed to minutes.
        assert entry["last_relevant_at"] == "2026-05-17T12:00"
        assert entry["date_added"] == "2026-04-01T00:00"

    def test_top_level_extras_win_over_metadata(self, monkeypatch):
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000",
            MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD="0",
        )
        # Caller may pass date_added directly via extras (extras.update
        # is applied before the hoist, so caller-supplied wins).
        entry = intent._project_memory(
            "rec_x",
            "summary",
            extras={
                "date_added": "2026-05-18T10:00:00",
                "metadata": {"date_added": "1900-01-01T00:00:00"},
            },
        )
        # v3.7.39: top-level still wins, trim is applied to the winner.
        assert entry["date_added"] == "2026-05-18T10:00"

    def test_omits_field_when_absent(self, monkeypatch):
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000",
            MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD="0",
        )
        # No metadata at all -- entry must not carry a None date_added,
        # callers should see absence as "unknown" rather than "null".
        entry = intent._project_memory("rec_x", "summary")
        assert "date_added" not in entry
        assert "last_relevant_at" not in entry

    def test_no_extras_legacy_callers_safe(self, monkeypatch):
        """Calling _project_memory with extras=None (legacy two-arg
        form) must still work cleanly post-v3.7.34 hoist."""
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000",
            MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD="0",
        )
        entry = intent._project_memory("rec_y", "summary body\n\nlong content here")
        assert entry["id"] == "rec_y"
        assert entry["summary_text"] == "summary body"
        assert entry["content"] == "long content here"


# ─────────────────────────────────────────────────────────────────────
# v3.9.6 (Adrian msg_c96c8a_171/172 2026-05-21): _project_memory surfaces
# a rendered `class_path` signature -- "(kind) ancestor -> ancestor"
# (is_a chain, root 'thing' omitted) -- uniformly, superseding v3.9.5's
# bare `kind` field. record/predicate/literal render just "(kind)".
# ─────────────────────────────────────────────────────────────────────


class TestProjectMemoryClassPath:
    def test_record_renders_bare_kind(self, monkeypatch):
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000",
            MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD="0",
        )
        # kind=record -> no is_a walk, deterministic without a kg.
        entry = intent._project_memory(
            "rec_x", "summary here", extras={"metadata": {"kind": "record"}}
        )
        assert entry["class_path"] == "(record)"
        # bare kind is replaced by class_path, not surfaced separately.
        assert "kind" not in entry

    def test_class_path_omitted_when_no_kind(self, monkeypatch):
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000",
            MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD="0",
        )
        entry = intent._project_memory("rec_x", "summary", extras={"metadata": {}})
        assert "class_path" not in entry
        assert "kind" not in entry

    def test_top_level_kind_wins_over_metadata(self, monkeypatch):
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000",
            MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD="0",
        )
        # top-level kind (applied via extras.update before the build) wins.
        entry = intent._project_memory(
            "rec_x", "summary", extras={"kind": "record", "metadata": {"kind": "literal"}}
        )
        assert entry["class_path"] == "(record)"

    def test_class_path_does_not_leak_metadata(self, monkeypatch):
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000",
            MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD="0",
        )
        entry = intent._project_memory(
            "rec_x", "summary", extras={"metadata": {"kind": "record", "session_id": "abc"}}
        )
        assert entry["class_path"] == "(record)"
        assert "metadata" not in entry
        assert "session_id" not in entry


# ─────────────────────────────────────────────────────────────────────
# v3.9.6 _render_class_path: the single source of truth for the
# "(kind) ancestor -> ancestor" signature. is_a chain walked transitively
# (BFS, deduped, cycle-safe), root 'thing' omitted, multiple parents
# joined with ' -> '. Fail-open to "(kind)".
# ─────────────────────────────────────────────────────────────────────


class TestRenderClassPath:
    class _FakeKG:
        def __init__(self, parents):
            self._p = parents

        def is_a_parents(self, eid):
            return self._p.get(eid, [])

    def test_entity_chain_omits_thing(self, monkeypatch):
        intent = _reload_intent(monkeypatch, MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000")
        kg = self._FakeKG({"version_py": ["file"], "file": ["thing"]})
        assert intent._render_class_path(kg, "version_py", "entity") == "(entity) file"

    def test_class_with_only_thing_is_bare(self, monkeypatch):
        intent = _reload_intent(monkeypatch, MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000")
        kg = self._FakeKG({"file": ["thing"]})
        assert intent._render_class_path(kg, "file", "class") == "(class)"

    def test_multi_parent_joined_with_arrows(self, monkeypatch):
        intent = _reload_intent(monkeypatch, MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000")
        kg = self._FakeKG({"mempalace": ["concept", "arch", "model"]})
        assert (
            intent._render_class_path(kg, "mempalace", "entity")
            == "(entity) concept -> arch -> model"
        )

    def test_record_kind_no_walk(self, monkeypatch):
        intent = _reload_intent(monkeypatch, MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000")
        kg = self._FakeKG({"r": ["x"]})  # parents present but record never walks
        assert intent._render_class_path(kg, "r", "record") == "(record)"

    def test_none_kg_fail_open(self, monkeypatch):
        intent = _reload_intent(monkeypatch, MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000")
        assert intent._render_class_path(None, "x", "entity") == "(entity)"

    def test_cycle_safe(self, monkeypatch):
        intent = _reload_intent(monkeypatch, MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000")
        kg = self._FakeKG({"a": ["b"], "b": ["a"]})  # a<->b cycle
        assert intent._render_class_path(kg, "a", "entity") == "(entity) b"

    def test_empty_kind_returns_empty(self, monkeypatch):
        intent = _reload_intent(monkeypatch, MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000")
        assert intent._render_class_path(None, "x", "") == ""


# ─────────────────────────────────────────────────────────────────────
# v3.7.37 verbosity fix (Adrian msg_c96c8a_141 2026-05-19): the
# _project_memory helper must NOT leak the raw vec metadata dict into
# the agent-visible entry. v3.7.34 plumbing pushed extras['metadata']
# through so the helper could hoist date_added/last_relevant_at; but
# entry.update(extras) also dumped the full meta blob (session_id,
# intent_id, content_type, view_index, etc.) onto the surface. Lock
# the strip so future refactors don't silently re-leak it.
# ─────────────────────────────────────────────────────────────────────


class TestVerbosityFix:
    def test_metadata_key_stripped_after_hoist(self, monkeypatch):
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000",
            MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD="0",
        )
        entry = intent._project_memory(
            "rec_x",
            "summary",
            extras={
                "metadata": {
                    "date_added": "2026-05-19T08:00:00",
                    "last_relevant_at": "2026-05-19T08:00:00",
                    "session_id": "sess_xyz",
                    "intent_id": "intent_abc",
                    "added_by": "ga_agent",
                    "view_index": 0,
                    "kind": "record",
                    "importance": 3,
                }
            },
        )
        # Dates hoisted to top-level for the agent; v3.7.39 trims to minutes.
        assert entry["date_added"] == "2026-05-19T08:00"
        assert entry["last_relevant_at"] == "2026-05-19T08:00"
        # v3.7.37: the verbose metadata blob must NOT reach the agent.
        assert "metadata" not in entry, (
            "v3.7.37: agent surface must not carry the full vec meta "
            "dict; only the hoisted date fields. "
            f"actual keys: {sorted(entry.keys())}"
        )
        # Pre-v3.7.37 leak signatures: ensure these vec-internal fields
        # are NOT on the entry. They were never agent-relevant.
        for leaked in ("session_id", "intent_id", "view_index"):
            assert leaked not in entry, (
                f"v3.7.37: '{leaked}' must not leak from vec meta to agent surface"
            )

    def test_strip_safe_when_no_metadata_in_extras(self, monkeypatch):
        """The unconditional pop must be safe when extras has no
        'metadata' key (i.e. when caller never plumbed it)."""
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000",
            MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD="0",
        )
        entry = intent._project_memory(
            "rec_z",
            "summary",
            extras={"hybrid_score": 0.642, "source": "vector"},
        )
        assert entry["id"] == "rec_z"
        assert entry["hybrid_score"] == 0.642
        assert entry["source"] == "vector"
        assert "metadata" not in entry  # never was, still isn't


# ─────────────────────────────────────────────────────────────────────
# v3.7.39 (Adrian msg_c96c8a_143 2026-05-19): trim surfaced dates to
# minute precision. "we don't need the milliseconds, up to the minutes
# TBH is enough, not even the seconds, though seconds could be left."
# 16-char prefix covers both ISO 8601 'T' and space separators.
# ─────────────────────────────────────────────────────────────────────


class TestDateTrim:
    def test_microsecond_iso_trimmed_to_minutes(self, monkeypatch):
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000",
            MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD="0",
        )
        entry = intent._project_memory(
            "rec_x",
            "summary",
            extras={
                "metadata": {
                    "date_added": "2026-05-19T01:41:03.207700",
                    "last_relevant_at": "2026-05-19T01:41:03.265998",
                }
            },
        )
        assert entry["date_added"] == "2026-05-19T01:41", (
            "v3.7.39: microsecond ISO must trim to YYYY-MM-DDTHH:MM"
        )
        assert entry["last_relevant_at"] == "2026-05-19T01:41"

    def test_space_separated_iso_trimmed_to_minutes(self, monkeypatch):
        """Space-separated form (older writes used this) also trims at
        position 16 since YYYY-MM-DD HH:MM is the same length as
        YYYY-MM-DDTHH:MM."""
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000",
            MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD="0",
        )
        entry = intent._project_memory(
            "rec_x",
            "summary",
            extras={
                "metadata": {
                    "date_added": "2026-05-18 11:09:36",
                }
            },
        )
        assert entry["date_added"] == "2026-05-18 11:09"

    def test_already_minute_precision_passthrough(self, monkeypatch):
        """If the writer already produced minute precision, the trim
        is a no-op (slice[:16] == original)."""
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000",
            MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD="0",
        )
        entry = intent._project_memory(
            "rec_x",
            "summary",
            extras={"metadata": {"date_added": "2026-05-19T01:41"}},
        )
        assert entry["date_added"] == "2026-05-19T01:41"

    def test_short_or_malformed_left_alone(self, monkeypatch):
        """Strings shorter than 16 chars are left as-is rather than
        producing a meaningless truncation."""
        intent = _reload_intent(
            monkeypatch,
            MEMPALACE_MEMORY_CONTENT_MAX_CHARS="2000",
            MEMPALACE_MEMORY_CONTENT_DEDUP_THRESHOLD="0",
        )
        entry = intent._project_memory(
            "rec_x",
            "summary",
            extras={"metadata": {"date_added": "2026-05-19"}},
        )
        # date-only is 10 chars; pass through.
        assert entry["date_added"] == "2026-05-19"
