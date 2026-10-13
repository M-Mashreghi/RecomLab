from datetime import UTC, datetime

import pytest

from recomlab.candidates import CandidateRequest, PopularityCandidates
from recomlab.candidates.fusion import fuse
from recomlab.candidates.item_item import ItemItemCandidates
from recomlab.datasets.labels import LabelPolicy
from recomlab.domain.eligibility import EligibilityFilter
from recomlab.evaluation.beyond_accuracy import intra_list_diversity
from recomlab.evaluation.offline import holdout_truth
from recomlab.features import PointInTimeFeatures
from recomlab.rankers import RankCandidate, RankedList, RankRequest, RetrievalScoreRanker
from recomlab.rankers.heuristic import HeuristicRanker
from recomlab.reranking import CalibrationReranker, ColdStartPolicy, MMRDiversifier, rerank_report

CATS = {"a": ("x",), "b": ("x",), "c": ("y",), "d": ("z",)}


def ranked(items):
    req = RankRequest("u", tuple(RankCandidate(i, s) for i, s in items.items()), k=len(items))
    return RetrievalScoreRanker().rank(req)


BASE = ranked({"a": 1.0, "b": 0.9, "c": 0.5, "d": 0.1})


def test_mmr_diversifies_without_adding_items():
    out = MMRDiversifier(CATS, lam=0.5).rerank(BASE)
    assert out.item_ids[0] == "a" and out.item_ids[1] != "b"
    assert set(out.item_ids) == set(BASE.item_ids)
    assert MMRDiversifier(CATS, lam=1.0).rerank(BASE).item_ids == BASE.item_ids
    assert intra_list_diversity(out.item_ids, CATS, 2) > intra_list_diversity(
        BASE.item_ids, CATS, 2
    )
    assert out.metadata["reranker"] == "mmr"
    with pytest.raises(ValueError):
        MMRDiversifier(CATS, lam=2.0)


def test_calibration_matches_profile():
    out = CalibrationReranker(CATS, lam=0.9).rerank(BASE, {"x": 0.5, "y": 0.25, "z": 0.25}, k=2)
    assert set(out.item_ids) <= set(BASE.item_ids) and len(out.item_ids) == 2
    assert out.item_ids[0] == "a" and out.item_ids[1] in {"c", "d"}
    skipped = CalibrationReranker(CATS).rerank(BASE, {}, k=3)
    assert skipped.item_ids == BASE.item_ids[:3]
    assert skipped.metadata["calibration_skipped"] == "no_profile"


@pytest.fixture
def ctx(tiny_ds, tiny_split):
    as_of = datetime(2024, 1, 14, tzinfo=UTC)
    feats = PointInTimeFeatures(tiny_split.train, tiny_ds.items, as_of)
    elig = EligibilityFilter(tiny_ds.items, tiny_split.train)
    return feats, elig


def test_reranking_never_adds_items_outside_candidates(tiny_ds, tiny_split, ctx):
    feats, elig = ctx
    gens = [
        PopularityCandidates().fit(tiny_split.train, tiny_ds.items),
        ItemItemCandidates().fit(tiny_split.train, tiny_ds.items),
    ]
    cats = {i.item_id: i.categories for i in tiny_ds.items}
    for user in ("u1", "u2", "u4"):
        fused = fuse(user, {g.name: g.generate(CandidateRequest(user, 4)) for g in gens}, elig)
        base = HeuristicRanker(feats).rank(RankRequest.from_fused(fused, 4))
        for out in (
            MMRDiversifier(cats, 0.3).rerank(base),
            CalibrationReranker(cats).rerank(base, feats.user_category_profile(user)),
        ):
            assert set(out.item_ids) <= set(fused.item_ids)
            assert all(elig.is_eligible(user, i) for i in out.item_ids)


def test_cold_start_non_empty_and_deterministic(tiny_ds, ctx):
    feats, elig = ctx
    empty = RankedList("u7", (), "heuristic", "v")
    out1 = ColdStartPolicy().apply(empty, 3, elig, feats)
    out2 = ColdStartPolicy().apply(empty, 3, elig, feats)
    assert len(out1.item_ids) == 3 and out1 == out2
    assert set(out1.metadata["fallbacks"].values()) == {"popularity"}
    assert all(elig.is_eligible("u7", i) for i in out1.item_ids)
    # Cold item slot: i10 and i11 have no training interactions; ties broken by id for u7.
    out = ColdStartPolicy(cold_item_slots=1).apply(empty, 3, elig, feats)
    assert out.item_ids[-1] == "i10"
    assert out.metadata["fallbacks"]["i10"] == "cold_item_content"
    assert "i12" not in ColdStartPolicy().apply(empty, 12, elig, feats).item_ids  # unavailable


def test_before_after_report(tiny_split):
    truth = holdout_truth(tiny_split, "test", LabelPolicy())
    user = next(iter(truth))
    before = {user: ["a", "b", "c", *truth[user]]}
    after = {user: [*truth[user], "a", "c", "b"]}
    report = rerank_report(before, after, truth, CATS, k=3)
    assert report["after"]["ndcg@3"] > report["before"]["ndcg@3"]
    assert report["delta"]["ndcg@3"] == pytest.approx(
        report["after"]["ndcg@3"] - report["before"]["ndcg@3"]
    )
    assert set(report["before"]) == {"ndcg@3", "ild@3", "distinct_items@3"}
