import json

import pytest

from recomlab.datasets.labels import LabelPolicy
from recomlab.domain.schemas import Interaction
from recomlab.evaluation.slices import (
    SliceConfig,
    item_popularity_buckets,
    slice_report,
    user_slices,
)
from recomlab.experiments.runner import run_offline_experiment
from recomlab.models import PopularityRecommender

pytestmark = pytest.mark.metrics


def _ev(n, user, item):
    return Interaction(f"e{n}", user, item, "click", "2024-01-01T00:00:00Z")  # type: ignore[arg-type]


HIST = [_ev(n, "dense", f"i{n % 4}") for n in range(10)] + [_ev(20, "sparse", "i0")]


def test_slice_memberships_deterministic():
    cfg = SliceConfig(sparse_max=3, dense_min=10)
    s1 = user_slices(HIST, ["dense", "sparse", "new"], cfg)
    s2 = user_slices(list(reversed(HIST)), ["new", "sparse", "dense"], cfg)
    assert s1 == s2
    assert s1["dense"] == {"coldness": "warm", "activity": "dense"}
    assert s1["sparse"] == {"coldness": "warm", "activity": "sparse"}
    assert s1["new"] == {"coldness": "cold", "activity": "none"}
    buckets = item_popularity_buckets(HIST, ["i0", "i1", "i2", "i3", "i9"], cfg)
    # counts: i0=4, i1=3, i2=2, i3=2 -> head {i0} (top 20%), tail {i2, i3} (bottom 50%), mid {i1}
    assert buckets["i9"] == "cold"
    assert buckets["i0"] == "head"
    assert list(buckets.values()).count("tail") == 2
    assert buckets == item_popularity_buckets(
        list(reversed(HIST)), ["i9", "i3", "i2", "i1", "i0"], cfg
    )


def test_counts_reconcile_with_population(tiny_ds, tiny_split, tiny_ref, tmp_path):
    cfg = SliceConfig(sparse_max=3, dense_min=5, user_attributes=("signup_cohort",))
    manifest, result = run_offline_experiment(
        experiment_id="slices",
        model=PopularityRecommender(),
        dataset=tiny_ds,
        dataset_ref=tiny_ref,
        split=tiny_split,
        label_policy=LabelPolicy(),
        ks=(3,),
    )
    history = tiny_split.train_until("test")
    slices = user_slices(history, list(result.per_user), cfg, tiny_ds.users)
    buckets = item_popularity_buckets(history, [i.item_id for i in tiny_ds.items], cfg)
    from recomlab.evaluation.offline import holdout_truth

    truth = holdout_truth(tiny_split, "test", LabelPolicy())
    report = slice_report(result.per_user, slices, result.rankings, truth, buckets, 3, cfg)
    assert report.population == result.n_users_evaluated
    assert report.reconciles()
    assert set(report.user_dimensions) == {"coldness", "activity", "attr:signup_cohort"}
    total_relevant = sum(len(v) for v in truth.values())
    assert sum(b["n_relevant"] for b in report.item_buckets.values()) == total_relevant
    # Leaderboard link: the slice report id is stored with the run.
    rid = report.write_json(tmp_path / "slices.json")
    assert json.loads((tmp_path / "slices.json").read_text())["report_id"] == rid
    linked = type(manifest)(**{**manifest.__dict__, "artifacts": {"slice_report": rid}})
    assert linked.artifacts["slice_report"].startswith("slices-")


def test_invalid_config_and_missing_membership():
    with pytest.raises(ValueError):
        SliceConfig(sparse_max=5, dense_min=2)
    with pytest.raises(ValueError):
        SliceConfig(head_fraction=0.7, tail_fraction=0.5)
    with pytest.raises(ValueError):
        slice_report({"u": {"m": 1.0}}, {}, {}, {}, {}, 3, SliceConfig())
