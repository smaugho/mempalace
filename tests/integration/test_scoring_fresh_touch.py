"""Unit tests for scoring.two_stage_retrieve fresh-touch override
(v3.7.34 FINDING #T fix).

Background
----------
Before v3.7.34, scoring.hybrid_score's date_anchor (decay clock) and
last_relevant_iso (recency bonus) were both read from vec metadata, but
vec metadata is only written at insert time -- it is never updated when
``kg.touch_entities(kept_ids)`` bumps the SQL ``entities.last_touched``
column post-injection-gate. So touch-on-use, despite being documented as
the canonical decay-reset signal (Adrian directive 2026-05-04,
knowledge_graph.py:3477-3483), was silently dropped before it could
influence the next retrieval's ranking.

v3.7.34 closes the loop: when ``two_stage_retrieve`` is handed a kg
handle, it batch-fetches the FRESH ``entities.last_touched`` value for
every top-M rerank candidate and merges it into ``meta`` so both
``date_anchor`` and ``last_relevant`` use the touch-on-use timestamp.

The tests below mock a minimal kg with a ``_conn().execute(...)`` shape
matching the real KG so the fresh-fetch path is exercised without
depending on a populated palace. The mock also lets us control the
returned rows so we can assert the override semantics precisely.
"""

from __future__ import annotations

import pytest

from mempalace.scoring import two_stage_retrieve

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------
# Mock KG -- minimal duck-typed surface matching the real kg._conn()
# .execute(...).fetchall() pattern that two_stage_retrieve uses for the
# batched last_touched lookup.
# ---------------------------------------------------------------------


class _FakeRow(dict):
    """sqlite3.Row stand-in: behaves like both a dict (real Row supports
    keyed access) and an object with __getitem__ by key."""


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows
        self.captured_sql = None
        self.captured_params = None

    def execute(self, sql, params=()):
        self.captured_sql = sql
        self.captured_params = params
        return _FakeCursor(self._rows)


class _FakeKG:
    def __init__(self, rows):
        self._fake_conn = _FakeConn(rows)

    def _conn(self):
        return self._fake_conn


# ---------------------------------------------------------------------
# Helpers for constructing the two_stage_retrieve inputs
# ---------------------------------------------------------------------


def _ranked_lists(ids):
    """Build a single-channel ranked_lists dict; rank N -> 1 / (N + 1)."""
    return {"channel_a": [(1.0 / (i + 1), f"text for {mid}", mid) for i, mid in enumerate(ids)]}


def _seen_meta(ids, *, last_touched=None, last_relevant_at=None):
    """Build seen_meta where each id carries fixed stale dates so we can
    verify the fresh-fetch override semantics."""
    return {
        mid: {
            "meta": {
                "importance": 3,
                "added_by": "ga_agent",
                "date_added": "2020-01-01T00:00:00",
                "last_touched": last_touched or "2020-01-02T00:00:00",
                "last_relevant_at": last_relevant_at or "2020-01-03T00:00:00",
            },
            "doc": f"doc for {mid}",
            "similarity": 0.5,
            "source": "vector",
        }
        for mid in ids
    }


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


class TestFreshTouchOverride:
    def test_no_kg_falls_back_to_meta(self):
        """Pre-v3.7.34 behavior must still work for callers that do not
        supply a kg handle (tests, legacy code paths)."""
        ids = ["rec_a", "rec_b"]
        reranked, _rrf, _cm = two_stage_retrieve(
            _ranked_lists(ids),
            _seen_meta(ids),
            agent="ga_agent",
            kg=None,
        )
        # Without kg, no error and the rerank still produces results.
        # The decay clock falls back to meta.last_touched as before.
        assert len(reranked) == 2
        for entry in reranked:
            # Sanity: hybrid_score is a finite float in the expected
            # bounded range (sim weight dominates here).
            assert isinstance(entry["hybrid_score"], float)

    def test_kg_fresh_touch_is_queried(self):
        """When kg is supplied, the SQL fetch is invoked with the top-M
        ids as parameters."""
        ids = ["rec_a", "rec_b"]
        kg = _FakeKG(rows=[])  # empty rows -> falls back to meta values
        two_stage_retrieve(
            _ranked_lists(ids),
            _seen_meta(ids),
            agent="ga_agent",
            kg=kg,
        )
        sql = kg._fake_conn.captured_sql or ""
        # v3.7.35 FINDING #U expanded the SELECT to also pull created_at
        # so kind=entity rows can surface a date_added via the bridge.
        assert "SELECT id, last_touched, created_at FROM entities" in sql
        assert "IN (" in sql
        # Both rerank candidates must appear in the parameter tuple.
        assert set(kg._fake_conn.captured_params or ()) == set(ids)

    def test_fresh_touch_beats_stale_meta(self):
        """When the SQL fetch returns a fresher last_touched than what
        vec meta carries, the fresh value MUST win as the decay anchor
        -- this is the whole point of FINDING #T's fix."""
        ids = ["rec_fresh", "rec_stale_only"]
        # rec_fresh gets a very recent SQL last_touched; rec_stale_only
        # gets no fresh row, falling back to the year-2020 meta value.
        fresh_rows = [
            _FakeRow(id="rec_fresh", last_touched="2026-05-18T12:00:00"),
        ]
        kg = _FakeKG(rows=fresh_rows)
        reranked, _rrf, _cm = two_stage_retrieve(
            _ranked_lists(ids),
            _seen_meta(ids),
            agent="ga_agent",
            kg=kg,
        )
        scores = {e["id"]: e["hybrid_score"] for e in reranked}
        # Identical channel position + identical other meta means the
        # ONLY differentiator between the two entries is the decay
        # axis. The fresh-touched entry must outrank (higher
        # hybrid_score) the one stuck with year-2020 meta.
        assert scores["rec_fresh"] > scores["rec_stale_only"], (
            f"v3.7.34 FINDING #T: fresh last_touched must override "
            f"stale vec meta as the decay anchor. "
            f"scores={scores}"
        )

    def test_fresh_touch_handles_sql_failure_gracefully(self):
        """If the SQL fetch raises, the rerank must still complete using
        vec meta -- never block retrieval on a freshness fetch."""

        class _ExplodingKG:
            def _conn(self):
                raise RuntimeError("simulated DB outage")

        ids = ["rec_a"]
        reranked, _rrf, _cm = two_stage_retrieve(
            _ranked_lists(ids),
            _seen_meta(ids),
            agent="ga_agent",
            kg=_ExplodingKG(),
        )
        # No raise, rerank produced the candidate using stale meta.
        assert len(reranked) == 1
        assert reranked[0]["id"] == "rec_a"


# ---------------------------------------------------------------------
# v3.7.35 FINDING #U: the fresh-fetch ALSO bridges entities.created_at
# into meta.date_added so kind=entity memories (which never carry
# date_added in vec metadata -- 99.7% of the live corpus on Adrian's
# palace post-audit 2026-05-18) finally surface a creation timestamp
# to the agent. The mutation happens on meta in place so the same dict
# reference flows through reranked[i]['meta'] -> hooks_cli ranked-dict
# -> _project_memory extras['metadata'] hoist -> agent.
# ---------------------------------------------------------------------


def _seen_meta_entity_shape(ids):
    """Build seen_meta matching the kind=entity vec metadata shape --
    minimal fields, no date_added / last_touched / last_relevant_at.
    This is the 99.7% case from the FINDING #U live-corpus probe."""
    return {
        mid: {
            "meta": {
                "importance": 3,
                "added_by": "ga_agent",
                "kind": "entity",
                "name": mid,
            },
            "doc": f"doc for {mid}",
            "similarity": 0.5,
            "source": "vector",
        }
        for mid in ids
    }


class TestFINDINGU_FreshFetchMetaMutation:
    def test_created_at_hoisted_into_meta_date_added(self):
        """Entity-shape vec meta has NO date_added; SQL has created_at.
        After two_stage_retrieve runs with kg, meta.date_added must be
        populated from SQL.created_at so downstream projection (via
        _project_memory extras['metadata'] hoist) surfaces it."""
        ids = ["ent_x"]
        kg = _FakeKG(
            rows=[
                _FakeRow(
                    id="ent_x",
                    last_touched="2026-05-18T20:00:00",
                    created_at="2026-04-01T08:30:00",
                )
            ]
        )
        seen = _seen_meta_entity_shape(ids)
        reranked, _rrf, _cm = two_stage_retrieve(
            _ranked_lists(ids),
            seen,
            agent="ga_agent",
            kg=kg,
        )
        # The reranked dict carries the SAME meta reference as seen_meta;
        # the mutation must surface there.
        assert len(reranked) == 1
        rmeta = reranked[0]["meta"]
        assert rmeta["date_added"] == "2026-04-01T08:30:00", (
            "v3.7.35 FINDING #U: SQL.created_at must be aliased to "
            "meta.date_added so entity-shape memories surface a "
            "creation timestamp to the agent"
        )
        assert rmeta["last_touched"] == "2026-05-18T20:00:00"
        assert rmeta["last_relevant_at"] == "2026-05-18T20:00:00", (
            "v3.7.35: last_relevant_at must fall back to last_touched "
            "for entity rows that never had a touch-on-use clock set"
        )

    def test_existing_meta_date_added_not_clobbered(self):
        """If meta ALREADY has date_added (e.g. kind=record write path
        set it at insert time), the SQL value must NOT overwrite it."""
        ids = ["rec_y"]
        kg = _FakeKG(
            rows=[
                _FakeRow(
                    id="rec_y",
                    last_touched="2026-05-18T20:00:00",
                    created_at="2026-04-01T08:30:00",
                )
            ]
        )
        seen = {
            "rec_y": {
                "meta": {
                    "importance": 3,
                    "added_by": "ga_agent",
                    "kind": "record",
                    "date_added": "2026-05-10T12:00:00",  # writer-supplied
                    "last_touched": "2026-05-11T12:00:00",
                    "last_relevant_at": "2026-05-12T12:00:00",
                },
                "doc": "doc",
                "similarity": 0.5,
                "source": "vector",
            }
        }
        reranked, _rrf, _cm = two_stage_retrieve(
            _ranked_lists(ids),
            seen,
            agent="ga_agent",
            kg=kg,
        )
        rmeta = reranked[0]["meta"]
        # Writer-supplied values win.
        assert rmeta["date_added"] == "2026-05-10T12:00:00"
        assert rmeta["last_touched"] == "2026-05-11T12:00:00"
        assert rmeta["last_relevant_at"] == "2026-05-12T12:00:00"

    def test_no_kg_leaves_entity_meta_dateless(self):
        """Without kg (legacy callers / tests), the entity-shape meta
        passes through unmutated -- date_added stays absent. Back-compat
        contract: don't synthesize dates we can't source."""
        ids = ["ent_x"]
        seen = _seen_meta_entity_shape(ids)
        reranked, _rrf, _cm = two_stage_retrieve(
            _ranked_lists(ids),
            seen,
            agent="ga_agent",
            kg=None,
        )
        rmeta = reranked[0]["meta"]
        assert "date_added" not in rmeta
        assert "last_touched" not in rmeta
        assert "last_relevant_at" not in rmeta

    def test_sql_returns_null_created_at_skipped(self):
        """If SQL returns NULL for created_at on a row (older entity
        predating the column populate), don't inject empty string --
        leave the field absent so the agent sees no date rather than
        a misleading empty string."""
        ids = ["ent_x"]
        kg = _FakeKG(
            rows=[
                _FakeRow(
                    id="ent_x",
                    last_touched="2026-05-18T20:00:00",
                    created_at=None,
                )
            ]
        )
        seen = _seen_meta_entity_shape(ids)
        reranked, _rrf, _cm = two_stage_retrieve(
            _ranked_lists(ids),
            seen,
            agent="ga_agent",
            kg=kg,
        )
        rmeta = reranked[0]["meta"]
        # last_touched still bridges; date_added stays absent.
        assert rmeta["last_touched"] == "2026-05-18T20:00:00"
        assert "date_added" not in rmeta
