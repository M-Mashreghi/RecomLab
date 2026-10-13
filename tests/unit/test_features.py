import math
from datetime import UTC, datetime

import pytest

from recomlab.datasets import FixtureAdapter
from recomlab.datasets.splits import SplitConfig, make_split
from recomlab.domain.schemas import Interaction
from recomlab.experiments import RunManifest
from recomlab.features import (
    FEATURE_VERSION,
    FutureDataError,
    PointInTimeFeatures,
    assert_point_in_time,
)

AS_OF = datetime(2024, 1, 5, tzinfo=UTC)


@pytest.fixture
def ds(tiny_dir):
    return FixtureAdapter().load(tiny_dir)


def test_hand_built_point_in_time_counts(ds):
    f = PointInTimeFeatures(ds.interactions, ds.items, AS_OF)
    # Before Jan 5: i01 rated by u1 (1st), u2 (1st); i02 by u1, u3, u2; i03 by u3 (3rd), u1 (4th).
    assert f.item_features("i01")["popularity"] == 2.0
    assert f.item_features("i02")["popularity"] == 3.0
    assert f.item_features("i03")["popularity"] == 2.0
    assert f.item_features("i11")["popularity"] == 0.0
    u1 = f.user_features("u1")
    assert u1["history_length"] == 3.0
    assert u1["mean_rating"] == pytest.approx((5 + 4 + 5) / 3)
    assert u1["days_since_last_event"] == pytest.approx(15 / 24)  # Jan 4 09:00 -> Jan 5 00:00
    assert f.user_features("u7") == {
        "history_length": 0.0,
        "distinct_items": 0.0,
        "days_since_last_event": None,
        "days_active": None,
        "mean_rating": None,
    }
    assert f.user_category_profile("u1") == pytest.approx(
        {"action": 3 / 5, "scifi": 1 / 5, "thriller": 1 / 5}
    )
    pair = f.pair_features("u1", "i05")
    assert pair == {"category_affinity": pytest.approx((3 / 5 + 1 / 5) / 2), "seen_before": 0.0}
    assert f.pair_features("u1", "i01")["seen_before"] == 1.0


def test_recency_weighted_popularity(ds):
    f = PointInTimeFeatures(ds.interactions, ds.items, AS_OF, recency_half_life_days=1.0)
    # i01 events at Jan 1 09:00 and 10:00 -> ages 3.625 and 3.583 days.
    expected = 2 ** -(3 + 15 / 24) + 2 ** -(3 + 14 / 24)
    assert f.item_features("i01")["recent_popularity"] == pytest.approx(expected)
    assert math.isclose(f.item_features("i11")["recent_popularity"], 0.0)


@pytest.mark.leakage
def test_future_data_injection_does_not_change_features(ds):
    base = PointInTimeFeatures(ds.interactions, ds.items, AS_OF)
    future = [
        Interaction(f"f{n}", "u1", "i11", "purchase", "2024-01-20T00:00:00Z")  # type: ignore[arg-type]
        for n in range(5)
    ]
    injected = PointInTimeFeatures([*ds.interactions, *future], ds.items, AS_OF)
    assert injected.item_features("i11") == base.item_features("i11")
    assert injected.user_features("u1") == base.user_features("u1")
    assert_point_in_time(injected)
    with pytest.raises(FutureDataError):
        PointInTimeFeatures([*ds.interactions, *future], ds.items, AS_OF, strict=True)


@pytest.mark.leakage
def test_leakage_check_fails_when_future_data_used(ds):
    f = PointInTimeFeatures(ds.interactions, ds.items, AS_OF)
    f.events.append(
        Interaction("x", "u1", "i11", "click", "2024-01-30T00:00:00Z")  # type: ignore[arg-type]
    )
    with pytest.raises(FutureDataError):
        assert_point_in_time(f)


@pytest.mark.leakage
def test_validation_and_test_events_excluded_from_training_aggregates(ds):
    split = make_split(
        ds.interactions,
        SplitConfig(
            protocol="global_time",
            validation_cutoff="2024-01-08T00:00:00Z",
            test_cutoff="2024-01-11T00:00:00Z",
        ),
    )
    as_of = datetime(2024, 1, 8, tzinfo=UTC)
    from_all = PointInTimeFeatures(ds.interactions, ds.items, as_of)
    from_train = PointInTimeFeatures(split.train, ds.items, as_of)
    assert from_all.item_popularity() == from_train.item_popularity()


def test_feature_manifest_stored_with_runs(ds):
    f = PointInTimeFeatures(ds.interactions, ds.items, AS_OF)
    manifest = RunManifest(
        experiment_id="feat",
        dataset={"dataset_id": "tiny", "version": "1"},
        split={"split_id": "s"},
        model={"name": "m", "version": "1"},
        metric_config={},
        seeds={"global": 0},
        features=f.manifest.to_dict(),
        created_at="2024-01-01T00:00:00+00:00",
    )
    restored = RunManifest.from_dict(manifest.to_dict())
    assert restored.features["version"] == FEATURE_VERSION
    assert restored.features["n_events_used"] == 12
    assert restored.features["max_event_time_used"] < restored.features["as_of"]


def test_naive_as_of_rejected(ds):
    with pytest.raises(ValueError):
        PointInTimeFeatures(ds.interactions, ds.items, datetime(2024, 1, 5))
