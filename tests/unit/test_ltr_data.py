from datetime import UTC, datetime

import numpy as np
import pytest

from recomlab.datasets.labels import LabelPolicy
from recomlab.datasets.splits import LeakageError, SplitConfig, make_split
from recomlab.rankers.ltr_data import (
    LTR_FEATURE_SCHEMA,
    LTRDataset,
    build_ltr_dataset,
    inner_training_cutoff,
)

GLOBAL = SplitConfig(
    protocol="global_time",
    validation_cutoff="2024-01-08T00:00:00Z",
    test_cutoff="2024-01-11T00:00:00Z",
)


@pytest.fixture
def split(tiny_ds):
    return make_split(tiny_ds.interactions, GLOBAL)


def _train_dataset(split, tiny_ds):
    cutoff = inner_training_cutoff(split.train, 0.7)
    hist = [e for e in split.train if e.event_time < cutoff]
    target = [e for e in split.train if e.event_time >= cutoff]
    return build_ltr_dataset(
        hist,
        target,
        tiny_ds.items,
        cutoff,
        LabelPolicy(),
        forbidden_event_ids=[e.event_id for e in (*split.validation, *split.test)],
    )


def test_group_sizes_reconcile_with_candidates(split, tiny_ds):
    ds = _train_dataset(split, tiny_ds)
    assert ds.group_sizes and sum(ds.group_sizes) == ds.features.shape[0] == len(ds.labels)
    assert ds.features.shape[1] == len(ds.feature_names)
    for _qid, sl in ds.groups():
        assert len(set(ds.item_ids[sl])) == sl.stop - sl.start  # unique items per group
    assert ds.labels.max() == 1.0  # at least one positive in the data
    assert ds.manifest()["schema_version"] == LTR_FEATURE_SCHEMA
    assert ds.manifest()["n_groups"] == len(ds.query_ids)


@pytest.mark.leakage
def test_feature_label_timestamp_ordering_enforced(split, tiny_ds):
    as_of = datetime(2024, 1, 5, tzinfo=UTC)
    with pytest.raises(LeakageError):  # history after as_of
        build_ltr_dataset(split.train, split.validation, tiny_ds.items, as_of, LabelPolicy())
    with pytest.raises(LeakageError):  # target before as_of
        build_ltr_dataset(
            [],
            [e for e in split.train if e.event_time < as_of],
            tiny_ds.items,
            as_of,
            LabelPolicy(),
        )


@pytest.mark.leakage
def test_no_validation_or_test_target_in_training_rows(split, tiny_ds):
    ds = _train_dataset(split, tiny_ds)
    held_pairs = {(e.user_id, e.item_id) for e in (*split.validation, *split.test)}
    train_target_pairs = set()
    for qid, sl in ds.groups():
        for item, label in zip(ds.item_ids[sl], ds.labels[sl], strict=True):
            if label > 0:
                train_target_pairs.add((qid, item))
    assert not train_target_pairs & held_pairs
    cutoff = datetime.fromisoformat(ds.as_of)
    with pytest.raises(LeakageError):
        build_ltr_dataset(
            [e for e in split.train if e.event_time < cutoff],
            [*[e for e in split.train if e.event_time >= cutoff], *split.validation],
            tiny_ds.items,
            cutoff,
            LabelPolicy(),
            forbidden_event_ids=[e.event_id for e in split.validation],
        )


def test_validation_groups_use_train_history_only(split, tiny_ds):
    as_of = datetime(2024, 1, 8, tzinfo=UTC)
    ds = build_ltr_dataset(split.train, split.validation, tiny_ds.items, as_of, LabelPolicy())
    assert ds.as_of == as_of.isoformat()
    assert all(np.isfinite(ds.features).ravel())


def test_dataset_validation():
    with pytest.raises(ValueError):
        LTRDataset(np.zeros((2, 1)), np.zeros(2), (3,), ("u",), ("a", "b"), ("f",), "t", "l")
