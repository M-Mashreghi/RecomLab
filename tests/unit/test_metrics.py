import math

import pytest

from recomlab.evaluation.metrics import (
    MetricConfig,
    UndefinedMetricError,
    aggregate,
    average_precision_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from recomlab.experiments import RunManifest

pytestmark = pytest.mark.metrics

RANKED = ["a", "x", "b", "y", "c"]
REL = {"a": 1.0, "b": 1.0, "z": 1.0}  # z never retrieved


def test_hand_computed_binary_metrics():
    assert precision_at_k(RANKED, REL, 3) == pytest.approx(2 / 3)
    assert precision_at_k(RANKED, REL, 5) == pytest.approx(2 / 5)
    assert recall_at_k(RANKED, REL, 3) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, REL, 1) == pytest.approx(1 / 3)
    assert hit_rate_at_k(RANKED, REL, 1) == 1.0
    assert hit_rate_at_k(["x"], REL, 1) == 0.0
    # AP@3 = (1/1 + 2/3) / min(3, 3)
    assert average_precision_at_k(RANKED, REL, 3) == pytest.approx((1 + 2 / 3) / 3)
    # AP@2 = (1/1) / min(3, 2)
    assert average_precision_at_k(RANKED, REL, 2) == pytest.approx(0.5)
    assert reciprocal_rank_at_k(["x", "y", "b"], REL, 5) == pytest.approx(1 / 3)
    assert reciprocal_rank_at_k(["x", "y", "b"], REL, 2) == 0.0


def test_hand_computed_ndcg_binary_and_graded():
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKED, REL, 3) == pytest.approx(dcg / idcg)
    graded = {"a": 1.0, "b": 3.0}
    lin = (1 / 1 + 3 / 2) / (3 / 1 + 1 / math.log2(3))
    assert ndcg_at_k(RANKED, graded, 3, "linear") == pytest.approx(lin)
    exp = (1 / 1 + 7 / 2) / (7 / 1 + 1 / math.log2(3))
    assert ndcg_at_k(RANKED, graded, 3, "exponential") == pytest.approx(exp)
    assert ndcg_at_k(["b", "a"], graded, 2) == pytest.approx(1.0)


def test_edge_cases():
    for fn in (precision_at_k, recall_at_k, ndcg_at_k, average_precision_at_k, hit_rate_at_k):
        with pytest.raises(UndefinedMetricError):
            fn(RANKED, {}, 3)
        with pytest.raises(ValueError):
            fn(RANKED, REL, 0)
    assert precision_at_k([], REL, 3) == 0.0
    assert ndcg_at_k([], REL, 3) == 0.0
    assert recall_at_k(["a", "b", "z"], REL, 10) == 1.0  # k larger than list
    assert precision_at_k(["a"], REL, 10) == pytest.approx(0.1)


def test_aggregation_handles_users_without_positives():
    cfg = MetricConfig(ks=(2,), metrics=("recall", "hit_rate"))
    rankings = {"u1": ["a", "b"], "u2": ["x"], "u4": ["q"]}
    truth = {"u1": {"a": 1.0, "c": 1.0}, "u2": {"x": 1.0}, "u3": {"a": 1.0}, "u4": {}}
    agg = aggregate(rankings, truth, cfg)
    assert agg.users_without_positives == ["u4"]
    assert agg.users_without_ranking == ["u3"]  # scored as an empty list
    assert agg.n_users == 3
    assert agg.per_user["u3"] == {"recall@2": 0.0, "hit_rate@2": 0.0}
    assert agg.means["recall@2"] == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    assert agg.to_dict()["n_users_without_positives"] == 1


def test_metric_config_stored_in_manifest():
    cfg = MetricConfig(ks=(5,), gain="exponential")
    manifest = RunManifest(
        experiment_id="m",
        dataset={"dataset_id": "tiny", "version": "1"},
        split={"split_id": "s"},
        model={"name": "m", "version": "1"},
        metric_config=cfg.to_dict(),
        seeds={},
        created_at="2024-01-01T00:00:00+00:00",
    )
    stored = RunManifest.from_dict(manifest.to_dict()).metric_config
    assert stored["gain"] == "exponential" and stored["ks"] == [5]
    with pytest.raises(ValueError):
        MetricConfig(metrics=("auc",))
    with pytest.raises(ValueError):
        MetricConfig(gain="cubic")
