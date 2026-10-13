from datetime import UTC, datetime

import pytest

from recomlab.datasets import FixtureAdapter
from recomlab.datasets.negative_sampling import TrainingNegativeSampler
from recomlab.domain.schemas import Interaction


@pytest.fixture
def tiny(tiny_dir):
    return FixtureAdapter().load(tiny_dir)


@pytest.mark.leakage
@pytest.mark.parametrize("strategy", ["uniform", "popularity"])
def test_no_negative_is_a_known_training_positive(tiny, strategy):
    horizon = datetime(2024, 1, 8, tzinfo=UTC)
    negs = TrainingNegativeSampler(strategy, seed=3).sample(
        tiny.interactions, tiny.item_ids, 4, horizon=horizon
    )
    known = {(e.user_id, e.item_id) for e in tiny.interactions if e.event_time <= horizon}
    assert negs.samples
    assert not {(s.user_id, s.item_id) for s in negs.samples} & known
    assert all(s.synthetic for s in negs.samples)
    assert negs.purpose == "training_only"


@pytest.mark.leakage
def test_post_horizon_events_do_not_influence_sampling(tiny):
    horizon = datetime(2024, 1, 8, tzinfo=UTC)
    sampler = TrainingNegativeSampler("popularity", seed=1)
    train_only = [e for e in tiny.interactions if e.event_time <= horizon]
    a = sampler.sample(train_only, tiny.item_ids, 3, horizon=horizon)
    b = sampler.sample(tiny.interactions, tiny.item_ids, 3, horizon=horizon)
    assert a.samples == b.samples


@pytest.mark.parametrize("strategy", ["uniform", "popularity"])
def test_deterministic_for_fixed_seed(tiny, strategy):
    s = TrainingNegativeSampler(strategy, seed=7)
    a = s.sample(tiny.interactions, tiny.item_ids, 3)
    b = s.sample(list(reversed(tiny.interactions)), list(reversed(tiny.item_ids)), 3)
    assert a.samples == b.samples
    c = TrainingNegativeSampler(strategy, seed=8).sample(tiny.interactions, tiny.item_ids, 3)
    assert c.samples != a.samples


def test_sparse_users_and_small_catalogs():
    def ev(eid, user, item):
        return Interaction(eid, user, item, "click", "2024-01-01T00:00:00Z")  # type: ignore[arg-type]

    events = [ev("e1", "dense", "a"), ev("e2", "dense", "b"), ev("e3", "sparse", "a")]
    negs = TrainingNegativeSampler("uniform", seed=0).sample(
        events, ["a", "b", "c"], 5, users=["dense", "sparse", "cold"]
    )
    by_user = negs.by_user()
    assert by_user["dense"] == ["c"]
    assert by_user["sparse"] == ["b", "c"]
    assert by_user["cold"] == ["a", "b", "c"]
    assert set(negs.exhausted_users) == {"dense", "sparse", "cold"}
    full = TrainingNegativeSampler("uniform").sample(
        [ev("e1", "u", "a"), ev("e2", "u", "b")], ["a", "b"], 1
    )
    assert full.samples == () and full.exhausted_users == ("u",)


def test_invalid_strategy():
    with pytest.raises(ValueError):
        TrainingNegativeSampler("hard")
