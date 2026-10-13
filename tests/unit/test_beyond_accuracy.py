import math

import pytest

from recomlab.evaluation.beyond_accuracy import (
    beyond_accuracy,
    calibration_kl,
    gini,
    intra_list_diversity,
    list_category_distribution,
)

pytestmark = pytest.mark.metrics

CATS = {"a": ("x",), "b": ("x",), "c": ("y",), "d": ("x", "y"), "e": ()}
CATALOG = ["a", "b", "c", "d", "e"]
POP = {"a": 4, "b": 2, "c": 1, "d": 0, "e": 0}  # training users per item; 4 train users
RANKINGS = {"u1": ["a", "c"], "u2": ["a", "b"], "u3": []}


def report(**kw):
    args = {
        "catalog": CATALOG,
        "train_item_users": POP,
        "n_train_users": 4,
        "categories": CATS,
        "profiles": {"u1": {"x": 0.5, "y": 0.5}, "u2": {"x": 1.0}},
    }
    args.update(kw)
    return beyond_accuracy(RANKINGS, 2, **args)


def test_deterministic_fixture_values():
    r = report()
    v = r.values
    assert v["catalog_coverage"] == pytest.approx(3 / 5)
    assert v["user_coverage"] == pytest.approx(2 / 3)
    assert v["ild"] == pytest.approx((1.0 + 0.0) / 2)
    info = [-math.log2(5 / 5), -math.log2(2 / 5), -math.log2(5 / 5), -math.log2(3 / 5)]
    assert v["novelty"] == pytest.approx(sum(info) / 4)
    assert v["gini"] == pytest.approx(gini([2, 1, 1, 0, 0]))
    assert v["head_share"] == pytest.approx(2 / 4)  # head = top 20% = {a}
    kl_u1 = calibration_kl({"x": 0.5, "y": 0.5}, {"x": 0.5, "y": 0.5})
    kl_u2 = calibration_kl({"x": 1.0}, {"x": 1.0})
    assert v["calibration_kl"] == pytest.approx((kl_u1 + kl_u2) / 2) == pytest.approx(0.0)
    assert report().to_dict() == r.to_dict()


def test_helpers():
    assert gini([1, 1, 1]) == pytest.approx(0.0)
    assert gini([0, 0, 3]) == pytest.approx(2 / 3)
    assert gini([]) is None and gini([0, 0]) is None
    assert intra_list_diversity(["a"], CATS, 5) is None
    assert list_category_distribution(["d"], CATS) == {"x": 0.5, "y": 0.5}
    assert calibration_kl({"x": 1.0}, {"y": 1.0}, alpha=0.01) == pytest.approx(math.log(100))


def test_undefined_metrics_are_unavailable_not_zero():
    r = report(categories=None, profiles=None)
    assert r.values["ild"] is None and r.values["calibration_kl"] is None
    assert r.unavailable["ild"] == "no category metadata"
    empty = beyond_accuracy(
        {"u": []}, 3, catalog=CATALOG, train_item_users=POP, n_train_users=4, categories=CATS
    )
    for name in ("novelty", "gini", "head_share", "ild", "calibration_kl"):
        assert empty.values[name] is None, name
        assert name in empty.unavailable
    assert empty.values["user_coverage"] == 0.0  # defined: zero users served
    assert "unavailable" in empty.to_dict()
