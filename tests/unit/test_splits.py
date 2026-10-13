import dataclasses
import json

import pytest

from recomlab.datasets import FixtureAdapter
from recomlab.datasets.splits import (
    LeakageError,
    Split,
    SplitConfig,
    SplitError,
    assert_no_leakage,
    load_split_ids,
    make_split,
)
from recomlab.domain.schemas import Interaction

pytestmark = pytest.mark.leakage

GLOBAL = SplitConfig(
    protocol="global_time",
    validation_cutoff="2024-01-08T00:00:00Z",
    test_cutoff="2024-01-11T00:00:00Z",
)


@pytest.fixture
def events(tiny_dir):
    return FixtureAdapter().load(tiny_dir).interactions


def test_leave_last_n_temporal_order(events):
    split = make_split(events, SplitConfig())
    for user in {e.user_id for e in split.test}:
        train_t = max(e.event_time for e in split.train if e.user_id == user)
        val_t = [e.event_time for e in split.validation if e.user_id == user]
        test_t = min(e.event_time for e in split.test if e.user_id == user)
        assert train_t < test_t
        assert all(train_t < t < test_t for t in val_t)
    assert split.excluded_users == {"u6": "insufficient_history"}
    assert all(e.user_id != "u6" for e in (*split.validation, *split.test))
    # u1: last event (i06, Jan 11) is test; the one before (i05, Jan 9) is validation.
    assert [e.event_id for e in split.test if e.user_id == "u1"] == ["e029"]


def test_global_time_boundaries(events):
    split = make_split(events, GLOBAL)
    assert max(e.event_time for e in split.train) < min(e.event_time for e in split.validation)
    assert max(e.event_time for e in split.validation) < min(e.event_time for e in split.test)
    assert split.boundaries["test_cutoff"].startswith("2024-01-11")
    assert split.excluded_users.get("u6") == "no_training_history"


def test_same_dataset_and_config_same_split_id(events):
    a = make_split(events, SplitConfig())
    b = make_split(list(reversed(events)), SplitConfig())
    assert a.split_id == b.split_id and a.test == b.test
    assert make_split(events, SplitConfig(n_test=2)).split_id != a.split_id
    assert make_split(events[:-1], SplitConfig()).split_id != a.split_id


def test_no_test_event_in_train_or_validation(events):
    for cfg in (SplitConfig(), GLOBAL, SplitConfig(protocol="random", seed=1)):
        split = make_split(events, cfg)
        test_ids = {e.event_id for e in split.test}
        assert not test_ids & {e.event_id for e in (*split.train, *split.validation)}
        total = len(split.train) + len(split.validation) + len(split.test)
        assert total + len(split.dropped_events) == len(events)


def test_repeat_interactions_dropped_not_moved(events):
    # u3 rates i03 on Jan 3 then watches it on Jan 9: Jan 9 watch cannot be a held-out item.
    split = make_split(events, GLOBAL)
    assert split.dropped_events
    assert all(r == "repeat_interaction" for r in split.dropped_events.values())
    train_ids = {e.event_id for e in split.train}
    assert not set(split.dropped_events) & train_ids
    seen = {(e.user_id, e.item_id) for e in split.train}
    assert not any((e.user_id, e.item_id) in seen for e in split.test)


def test_random_split_is_labeled_diagnostic(events):
    split = make_split(events, SplitConfig(protocol="random", seed=3))
    assert split.is_diagnostic and "DIAGNOSTIC" in split.label
    assert split.manifest()["diagnostic"] is True
    assert make_split(events, SplitConfig(protocol="random", seed=3)).test == split.test
    assert not make_split(events, SplitConfig()).is_diagnostic


def test_injected_leak_detected(events):
    split = make_split(events, SplitConfig())
    leaked = dataclasses.replace(split, train=(*split.train, split.test[0]))
    with pytest.raises(LeakageError):
        assert_no_leakage(leaked)
    late = Interaction(
        "late",
        "u1",
        "i11",
        "click",
        "2024-02-01T00:00:00Z",  # type: ignore[arg-type]
    )
    with pytest.raises(LeakageError):
        assert_no_leakage(dataclasses.replace(split, train=(*split.train, late)))


def test_persisted_boundaries_and_ids(events, tmp_path):
    split = make_split(events, SplitConfig())
    split.save(tmp_path / "split.json")
    ids = load_split_ids(tmp_path / "split.json")
    assert ids["test"] == [e.event_id for e in split.test]
    data = json.loads((tmp_path / "split.json").read_text())
    assert data["split_id"] == split.split_id and "per_user" in data["boundaries"]
    assert isinstance(split, Split)


def test_history_visibility(events):
    split = make_split(events, SplitConfig())
    assert split.train_until("validation") == split.train
    assert len(split.train_until("test")) == len(split.train) + len(split.validation)


def test_invalid_configs():
    with pytest.raises(SplitError):
        SplitConfig(protocol="kfold")
    with pytest.raises(SplitError):
        SplitConfig(protocol="global_time")
    with pytest.raises(SplitError):
        SplitConfig(
            protocol="global_time",
            validation_cutoff="2024-02-01T00:00:00Z",
            test_cutoff="2024-01-01T00:00:00Z",
        )
    dup = Interaction("x", "u", "i", "click", "2024-01-01T00:00:00Z")  # type: ignore[arg-type]
    with pytest.raises(SplitError):
        make_split([dup, dup], SplitConfig())
