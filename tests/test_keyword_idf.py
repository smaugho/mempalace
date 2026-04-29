"""BM25-robust IDF weighting on Channel C (the keyword channel).

P3 follow-up -- the ``keyword_idf`` table (migration 016) is now
maintained incrementally by ``_add_memory_internal``, and
``_build_keyword_channel`` scores matches via summed IDF instead of
the legacy ``overlap_ratio`` heuristic.

Scope clarification: this is the IDF component of BM25 applied standalone.
Full BM25's TF saturation (k1) and length normalisation (b) contribute
nothing here because the channel treats each keyword-entity match as
binary -- TF ≡ 1, document length irrelevant. For this channel's input
shape, "IDF-alone" ranks identically to "full BM25".

References:
  - Robertson & Jones 1976 JASIS
  - Gao/Lu/Lin "Which BM25 Do You Mean?" 2020 -- no statistically
    significant diffs between BM25 variants; what matters is that IDF
    is there at all.
"""

from __future__ import annotations

import math


def test_record_keyword_observations_bumps_freq_and_idf(kg):
    kg.seed_ontology()
    # Seed two record memories so N > 0 for the IDF recompute.
    kg.add_entity("mem_a", kind="record", content="a", importance=3)
    kg.add_entity("mem_b", kind="record", content="b", importance=3)

    kg.record_keyword_observations(["auth", "jwt"])
    kg.record_keyword_observations(["auth"])  # auth seen twice, jwt once

    idfs = kg.get_keyword_idf(["auth", "jwt", "never-seen"])
    assert idfs["never-seen"] == 0.0
    # Both are recorded -- freq(auth)=2, freq(jwt)=1, with N=2.
    # idf = log((N - freq + 0.5) / (freq + 0.5) + 1)
    expected_auth = math.log(max(0.0, (2 - 2 + 0.5) / (2 + 0.5)) + 1.0)
    expected_jwt = math.log(max(0.0, (2 - 1 + 0.5) / (1 + 0.5)) + 1.0)
    assert abs(idfs["auth"] - expected_auth) < 1e-3
    assert abs(idfs["jwt"] - expected_jwt) < 1e-3
    # Rare term > common term (jwt matched 1 record, auth matched both).
    assert idfs["jwt"] > idfs["auth"]


def test_get_keyword_idf_empty_input_returns_empty(kg):
    kg.seed_ontology()
    assert kg.get_keyword_idf([]) == {}
    assert kg.get_keyword_idf(None) == {}


def test_recompute_keyword_idf_all_updates_everything(kg):
    kg.seed_ontology()
    kg.add_entity("mem_x", kind="record", content="x", importance=3)
    kg.add_entity("mem_y", kind="record", content="y", importance=3)
    kg.add_entity("mem_z", kind="record", content="z", importance=3)
    # Seed freqs without letting the per-write recompute land them,
    # simulating a bulk backfill.
    kg.record_keyword_observations(["alpha"], recompute_idf=False)
    kg.record_keyword_observations(["alpha"], recompute_idf=False)
    kg.record_keyword_observations(["beta"], recompute_idf=False)

    # Before recompute, idf defaults to 0.
    pre = kg.get_keyword_idf(["alpha", "beta"])
    assert pre["alpha"] == 0.0
    assert pre["beta"] == 0.0

    kg.recompute_keyword_idf_all()

    post = kg.get_keyword_idf(["alpha", "beta"])
    assert post["alpha"] > 0.0
    assert post["beta"] > 0.0
    # beta is rarer (1 record vs 2), so it wins.
    assert post["beta"] > post["alpha"]


def test_keyword_channel_idf_weighted_ranking_matches_rarity(monkeypatch, config, kg, palace_path):
    """End-to-end: a rare keyword outscores a common one in the channel output."""
    from tests.test_mcp_server import _patch_mcp_server, _get_collection

    _patch_mcp_server(monkeypatch, config, kg)
    _c, col = _get_collection(palace_path, create=True)

    from mempalace.scoring import _build_keyword_channel

    # Four memories, two keywords -- one rare, one common.
    # "jwt" appears in one memory (rare); "auth" appears in all four.
    kg.add_entity("mem_rare", kind="record", content="r", importance=3)
    kg.add_entity_keywords("mem_rare", ["auth", "jwt"])
    kg.record_keyword_observations(["auth", "jwt"])
    for i in range(3):
        eid = f"mem_common_{i}"
        kg.add_entity(eid, kind="record", content=f"c{i}", importance=3)
        kg.add_entity_keywords(eid, ["auth"])
        kg.record_keyword_observations(["auth"])

    # Seed docs into the Chroma collection so keyword_lookup can fetch.
    for eid in ["mem_rare", "mem_common_0", "mem_common_1", "mem_common_2"]:
        col.upsert(
            ids=[eid],
            documents=[f"doc for {eid}"],
            metadatas=[{"name": eid, "kind": "record", "importance": 3}],
        )

    seen_meta: dict = {}
    ranked = _build_keyword_channel(
        col,
        ["auth", "jwt"],
        kg=kg,
        added_by=None,
        kind_filter=None,
        seen_meta=seen_meta,
    )
    # Best score must be mem_rare because it picks up both auth-IDF + jwt-IDF
    # while the common-only memories only get auth-IDF.
    assert ranked, "keyword channel should surface something"
    ranked.sort(key=lambda x: x[0], reverse=True)
    top_score, _doc, top_id = ranked[0]
    assert top_id == "mem_rare", (
        f"expected mem_rare at top (rare keyword gives higher IDF), got {ranked[:3]}"
    )
    # And the common memories' score should be strictly less than mem_rare's.
    other_scores = [s for s, _d, mid in ranked if mid != "mem_rare"]
    assert all(s < top_score for s in other_scores)


def test_keyword_channel_cold_start_falls_back_to_uniform(monkeypatch, config, kg, palace_path):
    """When keyword_idf is empty, every keyword gets uniform weight."""
    from tests.test_mcp_server import _patch_mcp_server, _get_collection

    _patch_mcp_server(monkeypatch, config, kg)
    _c, col = _get_collection(palace_path, create=True)

    from mempalace.scoring import _build_keyword_channel

    # Two memories -- one matches two keywords, the other matches one.
    # With no IDF data, both keywords score uniformly, so mem_2match wins
    # on count alone.
    kg.add_entity("mem_2match", kind="record", content="a", importance=3)
    kg.add_entity_keywords("mem_2match", ["alpha", "beta"])
    kg.add_entity("mem_1match", kind="record", content="b", importance=3)
    kg.add_entity_keywords("mem_1match", ["alpha"])
    for eid in ("mem_2match", "mem_1match"):
        col.upsert(
            ids=[eid],
            documents=[f"doc {eid}"],
            metadatas=[{"name": eid, "kind": "record", "importance": 3}],
        )
    # Intentionally do NOT call record_keyword_observations -- cold-start.

    seen_meta: dict = {}
    ranked = _build_keyword_channel(
        col,
        ["alpha", "beta"],
        kg=kg,
        added_by=None,
        kind_filter=None,
        seen_meta=seen_meta,
    )
    assert ranked
    ranked.sort(key=lambda x: x[0], reverse=True)
    assert ranked[0][2] == "mem_2match"
