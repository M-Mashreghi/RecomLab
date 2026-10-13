import math
from datetime import UTC, datetime

import pytest

from recomlab.domain.schemas import Interaction, Item
from recomlab.features import FutureDataError, PointInTimeFeatures
from recomlab.rankers import RankCandidate, RankRequest
from recomlab.rankers.heuristic import HeuristicRanker, HeuristicWeights

AS_OF = datetime(2024, 1, 10, tzinfo=UTC)
ITEMS = [Item("a", categories=("x",)), Item("b", categories=("y",)), Item("c", categories=("x",))]


def _ev(n, user, item, day):
    return Interaction(f"e{n}", user, item, "click", f"2024-01-{day:02d}T00:00:00Z")  # type: ignore[arg-type]


EVENTS = [
    _ev(1, "u", "a", 1),
    _ev(2, "v", "b", 9),
    _ev(3, "w", "b", 9),
    _ev(4, "z", "b", 9),
]  # popularity: a=1, b=3, c=0; u profile = {x: 1.0}
REQ = RankRequest(
    "u", (RankCandidate("b", 0.2), RankCandidate("c", 0.6), RankCandidate("a", 1.0)), k=3
)


@pytest.fixture
def feats():
    return PointInTimeFeatures(EVENTS, ITEMS, AS_OF, recency_half_life_days=1.0)


def test_hand_computed_ranking(feats):
    w = HeuristicWeights({"retrieval": 0.5, "popularity": 0.25, "content": 0.25})
    out = HeuristicRanker(feats, w).rank(REQ)
    lp = math.log1p(3)
    expected = {
        "a": 0.5 * 1.0 + 0.25 * math.log1p(1) / lp + 0.25 * 1.0,
        "b": 0.5 * 0.0 + 0.25 * 1.0 + 0.0,
        "c": 0.5 * 0.5 + 0.0 + 0.25 * 1.0,
    }
    assert {i.item_id: i.raw_score for i in out.items} == pytest.approx(expected)
    assert out.item_ids == ("a", "c", "b")


def test_weight_changes_are_predictable(feats):
    only_pop = HeuristicRanker(feats, HeuristicWeights({"popularity": 1.0})).rank(REQ)
    assert only_pop.item_ids == ("b", "a", "c")
    only_retrieval = HeuristicRanker(feats, HeuristicWeights({"retrieval": 1.0})).rank(REQ)
    assert only_retrieval.item_ids == ("a", "c", "b")
    only_recency = HeuristicRanker(feats, HeuristicWeights({"recency": 1.0})).rank(REQ)
    assert only_recency.item_ids[0] == "b"  # most recent events


def test_contributions_sum_to_score(feats):
    out = HeuristicRanker(feats).rank(REQ)
    for item in out.items:
        assert sum(item.contributions.values()) == pytest.approx(item.raw_score)
        assert set(item.contributions) == {"retrieval", "popularity", "recency", "content"}


@pytest.mark.leakage
def test_no_post_event_features():
    future = [*EVENTS, _ev(9, "q", "c", 20), _ev(10, "r", "c", 21)]
    with_future = PointInTimeFeatures(future, ITEMS, AS_OF, recency_half_life_days=1.0)
    base = PointInTimeFeatures(EVENTS, ITEMS, AS_OF, recency_half_life_days=1.0)
    assert HeuristicRanker(with_future).rank(REQ).items == HeuristicRanker(base).rank(REQ).items
    with_future.events.append(_ev(11, "q", "c", 25))
    with pytest.raises(FutureDataError):
        HeuristicRanker(with_future)


def test_weights_stored_for_manifest(feats):
    r = HeuristicRanker(feats, HeuristicWeights({"content": 1.0}))
    assert r.params()["weights"] == {
        "retrieval": 0.0,
        "popularity": 0.0,
        "recency": 0.0,
        "content": 1.0,
    }
    assert r.params()["features"]["as_of"] == AS_OF.isoformat()
    with pytest.raises(ValueError):
        HeuristicWeights({"magic": 1.0})
