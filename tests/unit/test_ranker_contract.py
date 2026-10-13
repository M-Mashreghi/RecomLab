import dataclasses
import json

import pytest

from recomlab.candidates import CandidateRequest, PopularityCandidates
from recomlab.candidates.fusion import fuse
from recomlab.domain.eligibility import EligibilityFilter
from recomlab.rankers import (
    RankCandidate,
    RankedList,
    Ranker,
    RankerContractError,
    RankRequest,
    RetrievalScoreRanker,
)


class FixedRanker(Ranker):
    name = "fixed"

    def __init__(self, scores):
        self.scores = scores

    def _score(self, request):
        return {i: (s, {"bias": s}) for i, s in self.scores.items() if i in request.item_ids}


class GreedyRanker(Ranker):
    name = "greedy"

    def _score(self, request):
        return {"outside": (99.0, {}), **{c.item_id: (0.0, {}) for c in request.candidates}}


def test_ordering_and_deterministic_ties():
    req = RankRequest.from_items("u", ["c", "a", "b", "d"], k=4)
    out = FixedRanker({"a": 1.0, "b": 2.0, "c": 1.0, "d": 0.5}).rank(req)
    assert out.item_ids == ("b", "a", "c", "d")
    assert [i.rank for i in out.items] == [1, 2, 3, 4]
    assert FixedRanker({"a": 1.0, "c": 1.0}).rank(
        RankRequest.from_items("u", ["c", "a"], k=2)
    ).item_ids == ("a", "c")


def test_cannot_exceed_candidate_set():
    with pytest.raises(RankerContractError):
        GreedyRanker().rank(RankRequest.from_items("u", ["a", "b"], k=5))
    out = FixedRanker({"a": 1.0}).rank(RankRequest.from_items("u", ["a", "b"], k=5))
    assert out.item_ids == ("a",)
    assert out.metadata["n_unscored"] == 1
    k_limited = FixedRanker({"a": 1.0, "b": 2.0}).rank(RankRequest.from_items("u", ["a", "b"], 1))
    assert k_limited.item_ids == ("b",)


def test_request_validation():
    with pytest.raises(RankerContractError):
        RankRequest.from_items("u", ["a", "a"], k=2)
    with pytest.raises(RankerContractError):
        RankRequest.from_items("u", ["a"], k=0)
    labelish = {f.name for f in dataclasses.fields(RankRequest)}
    assert labelish == {"user_id", "candidates", "k", "request_id"}
    assert {f.name for f in dataclasses.fields(RankCandidate)} == {
        "item_id",
        "retrieval_score",
        "sources",
    }


def test_serializable_score_and_rank_metadata(tiny_ds, tiny_split):
    gen = PopularityCandidates().fit(tiny_split.train, tiny_ds.items)
    elig = EligibilityFilter(tiny_ds.items, tiny_split.train)
    fused = fuse("u1", {"popularity": gen.generate(CandidateRequest("u1", 5))}, elig)
    out = RetrievalScoreRanker().rank(RankRequest.from_fused(fused, k=3, request_id="r1"))
    data = json.loads(out.to_json())
    assert data["model_name"] == "retrieval_score" and data["request_id"] == "r1"
    assert set(data["items"][0]) == {"item_id", "raw_score", "final_score", "rank", "contributions"}
    assert RankedList.from_dict(data) == dataclasses.replace(out, metadata=data["metadata"])
    # Ranker output is independent of presentation: no explanation/text fields.
    assert not {"explanation", "text"} & set(data) and not {"explanation", "text"} & set(
        data["items"][0]
    )
    assert all(isinstance(i.final_score, float) for i in out.items)
