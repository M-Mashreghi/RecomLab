from collections import Counter

import pytest

from recomlab.models.als import ALSParams, ALSRecommender
from recomlab.rankers import RankRequest
from recomlab.rankers.collaborative import CollaborativeRanker

FAST = ALSParams(factors=4, iterations=8, seed=0)


@pytest.fixture
def setup(tiny_ds, tiny_split):
    model = ALSRecommender(FAST).fit(tiny_split.train, tiny_ds.items)
    pop = Counter(e.item_id for e in tiny_split.train)
    return model, CollaborativeRanker(model, pop), pop


def test_known_score_ordering(setup):
    model, ranker, _ = setup
    items = ["i05", "i08", "i09", "i10", "i11"]
    out = ranker.rank(RankRequest.from_items("u1", items, k=5))
    expected = sorted(model.score("u1", items).items(), key=lambda kv: (-kv[1], kv[0]))
    assert out.item_ids == tuple(i for i, _ in expected)
    assert out.metadata["fallback"] is None
    assert out.metadata["model_version"].startswith("als-")
    assert out.items[0].contributions["collaborative"] == pytest.approx(out.items[0].raw_score)


def test_unknown_user_is_deterministic_popularity(setup, tiny_ds):
    _, ranker, pop = setup
    items = ["i01", "i02", "i11", "i06"]
    a = ranker.rank(RankRequest.from_items("u7", items, k=4))
    b = ranker.rank(RankRequest.from_items("u7", list(reversed(items)), k=4))
    assert a.item_ids == b.item_ids
    assert a.metadata["fallback"] == "unknown_user"
    expected = sorted(items, key=lambda i: (-pop.get(i, 0), i))
    assert list(a.item_ids) == expected


def test_unknown_items_ranked_below_known(setup):
    _, ranker, _ = setup
    out = ranker.rank(RankRequest.from_items("u1", ["new_b", "i09", "new_a", "i10"], k=4))
    assert set(out.item_ids[:2]) == {"i09", "i10"}
    assert out.item_ids[2:] == ("new_a", "new_b")
    assert out.metadata["fallback"] == "unknown_items:2"


@pytest.mark.leakage
def test_scores_only_candidates_never_catalog(setup):
    model, ranker, _ = setup
    calls = []
    original = model.score

    def spy(user, items):
        calls.append(tuple(items))
        return original(user, items)

    model.score = spy  # type: ignore[method-assign]
    ranker.rank(RankRequest.from_items("u2", ["i06", "i07"], k=2))
    assert calls == [("i06", "i07")]


def test_batch_matches_single_and_latency(setup):
    _, ranker, _ = setup
    reqs = [RankRequest.from_items(u, ["i06", "i07", "i08", "i11"], 3) for u in ("u1", "u2", "u7")]
    batch = ranker.rank_batch(reqs)
    for r in reqs:
        assert batch[r.user_id].items == ranker.rank(r).items
        assert batch[r.user_id].metadata["latency_ms"] >= 0


def test_requires_trained_model():
    with pytest.raises(ValueError):
        CollaborativeRanker(ALSRecommender(), {})
