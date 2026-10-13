import pytest

from recomlab.datasets.labels import LabelPolicy
from recomlab.domain.eligibility import EligibilityFilter, EligibilityPolicy
from recomlab.experiments import ExperimentRegistry
from recomlab.experiments.runner import run_offline_experiment
from recomlab.models import NotFittedError, PopularityRecommender, RandomRecommender


def test_popularity_deterministic_topk(tiny_ds, tiny_split):
    model = PopularityRecommender().fit(tiny_split.train, tiny_ds.items)
    elig = EligibilityFilter(tiny_ds.items, tiny_split.train)
    recs = model.recommend("u7", 4, elig)
    assert [r.item_id for r in recs] == [r.item_id for r in model.recommend("u7", 4, elig)]
    scores = [r.score for r in recs]
    assert scores == sorted(scores, reverse=True)
    # ties broken by item id
    for a, b in zip(recs, recs[1:], strict=False):
        assert (a.score > b.score) or (a.item_id < b.item_id)
    assert "i12" not in {r.item_id for r in model.recommend("u7", 12, elig)}  # unavailable


@pytest.mark.leakage
def test_popularity_counts_exclude_validation_and_test(tiny_ds, tiny_split):
    model = PopularityRecommender().fit(tiny_split.train, tiny_ds.items)
    train_counts: dict[str, int] = {}
    for e in tiny_split.train:
        train_counts[e.item_id] = train_counts.get(e.item_id, 0) + 1
    assert model.counts == train_counts
    held_ids = {e.event_id for e in (*tiny_split.validation, *tiny_split.test)}
    assert held_ids  # sanity: there is held-out data that must be ignored
    full = PopularityRecommender().fit(tiny_ds.interactions, tiny_ds.items)
    assert full.counts != model.counts


def test_random_seeded_and_order_independent(tiny_ds, tiny_split):
    elig = EligibilityFilter(tiny_ds.items, tiny_split.train)
    m1 = RandomRecommender(seed=5).fit(tiny_split.train, tiny_ds.items)
    m2 = RandomRecommender(seed=5).fit(tiny_split.train, tiny_ds.items)
    m2.recommend("u2", 3, elig)  # different call order
    assert m1.recommend("u1", 3, elig) == m2.recommend("u1", 3, elig)
    assert m1.recommend("u1", 5, elig) != RandomRecommender(seed=6).fit(
        tiny_split.train, tiny_ds.items
    ).recommend("u1", 5, elig)


def test_eligibility_filters(tiny_ds, tiny_split):
    model = PopularityRecommender().fit(tiny_split.train, tiny_ds.items)
    consumed = {e.item_id for e in tiny_split.train if e.user_id == "u1"}
    elig = EligibilityFilter(tiny_ds.items, tiny_split.train)
    assert not {r.item_id for r in model.recommend("u1", 12, elig)} & consumed
    keep = EligibilityFilter(
        tiny_ds.items,
        tiny_split.train,
        EligibilityPolicy(exclude_consumed=False, excluded_items=frozenset({"i01"})),
    )
    ids = {r.item_id for r in model.recommend("u1", 12, keep)}
    assert consumed - {"i01"} <= ids and "i01" not in ids


def test_unfitted_models_raise(tiny_ds):
    elig = EligibilityFilter(tiny_ds.items, [])
    with pytest.raises(NotFittedError):
        PopularityRecommender().recommend("u1", 1, elig)
    with pytest.raises(NotFittedError):
        RandomRecommender().recommend("u1", 1, elig)


@pytest.mark.integration
def test_baselines_saved_to_registry_with_coverage_and_latency(
    tiny_ds, tiny_split, tiny_ref, tmp_path
):
    reg = ExperimentRegistry(tmp_path / "runs.db")
    results = {}
    for model in (PopularityRecommender(), RandomRecommender(seed=1)):
        manifest, result = run_offline_experiment(
            experiment_id="baselines",
            model=model,
            dataset=tiny_ds,
            dataset_ref=tiny_ref,
            split=tiny_split,
            label_policy=LabelPolicy(),
            ks=(3, 5),
            seeds={"model": 1},
            registry=reg,
        )
        results[model.name] = (manifest, result)
        stored = reg.get_run(manifest.run_id)
        assert stored.metrics["catalog_coverage@3"] > 0
        assert set(stored.metrics["latency_ms"]) == {"mean", "p50", "max"}
        assert "recall@5" in stored.metrics and "ndcg@3" in stored.metrics
        assert stored.evaluation["mode"] == "full"
    rows = reg.compare_runs([m.run_id for m, _ in results.values()], ["recall@5"])
    assert [r["model"] for r in rows] == ["popularity@1", "random@1"]
    # deterministic metrics on rerun (latency excluded)
    again, _ = run_offline_experiment(
        experiment_id="baselines",
        model=PopularityRecommender(),
        dataset=tiny_ds,
        dataset_ref=tiny_ref,
        split=tiny_split,
        label_policy=LabelPolicy(),
        ks=(3, 5),
        seeds={"model": 1},
    )
    first = {k: v for k, v in results["popularity"][0].metrics.items() if k != "latency_ms"}
    assert first == {k: v for k, v in again.metrics.items() if k != "latency_ms"}
