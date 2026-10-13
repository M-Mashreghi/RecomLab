import pytest

from recomlab.datasets.labels import (
    LabelPolicy,
    LabelPolicyError,
    derive_labels,
    label_policy_from_manifest,
    relevance_by_user,
)
from recomlab.domain.schemas import Interaction
from recomlab.experiments import RunManifest


def ev(eid, etype, value, item="i1"):
    return Interaction.from_dict(
        {
            "event_id": eid,
            "user_id": "u1",
            "item_id": item,
            "event_type": etype,
            "value": value,
            "event_time": "2024-01-01T00:00:00Z",
        }
    )


BOUNDARY = [
    ev("r1", "rating", "4", "a"),
    ev("r2", "rating", "3.99", "b"),
    ev("r3", "rating", "5", "c"),
    ev("w1", "watch", "0.8", "d"),
    ev("w2", "watch", "0.79", "e"),
    ev("w3", "watch", "", "f"),
    ev("p1", "purchase", "", "g"),
    ev("c1", "click", "", "h"),
]


def test_binary_boundaries():
    labels = {lab.event_id: lab for lab in derive_labels(BOUNDARY, LabelPolicy())}
    assert [labels[k].positive for k in ["r1", "r2", "r3", "w1", "w2", "w3", "p1", "c1"]] == [
        True,
        False,
        True,
        True,
        False,
        False,
        True,
        False,
    ]
    assert labels["r2"].explicit_negative and not labels["c1"].explicit_negative
    assert {lab.relevance for lab in labels.values()} == {0.0, 1.0}


def test_graded_relevance():
    policy = LabelPolicy(relevance_mode="graded", positive_event_types=("rating", "purchase"))
    labels = {lab.event_id: lab for lab in derive_labels(BOUNDARY, policy)}
    assert labels["r1"].relevance == 1.0
    assert labels["r3"].relevance == 2.0
    assert labels["p1"].relevance == 3.0
    assert labels["w1"].relevance == 0.0  # watch not a positive type in this policy


def test_raw_events_untouched():
    before = [e.to_dict() for e in BOUNDARY]
    derive_labels(BOUNDARY, LabelPolicy(relevance_mode="graded"))
    assert [e.to_dict() for e in BOUNDARY] == before


def test_policy_change_changes_label_version():
    base = LabelPolicy()
    assert base.label_version == LabelPolicy().label_version
    assert LabelPolicy(rating_positive_threshold=3.5).label_version != base.label_version
    assert LabelPolicy(relevance_mode="graded").label_version != base.label_version
    assert (
        LabelPolicy(positive_event_types=("watch", "rating")).label_version
        == LabelPolicy(positive_event_types=("rating", "watch")).label_version
    )


def test_policy_stored_and_read_from_manifest():
    policy = LabelPolicy(rating_positive_threshold=3.5, relevance_mode="graded")
    manifest = RunManifest(
        experiment_id="x",
        dataset={"dataset_id": "tiny", "version": "1"},
        split={"split_id": "s"},
        model={"name": "m", "version": "1"},
        metric_config={},
        seeds={"global": 1},
        label_policy=policy.to_dict(),
        created_at="2024-01-01T00:00:00+00:00",
    )
    restored = RunManifest.from_dict(manifest.to_dict())
    assert label_policy_from_manifest(restored.label_policy) == policy
    assert restored.label_policy["label_version"] == policy.label_version
    with pytest.raises(LabelPolicyError):
        label_policy_from_manifest({})
    tampered = {**policy.to_dict(), "label_version": "labels-x"}
    with pytest.raises(LabelPolicyError):
        label_policy_from_manifest(tampered)


def test_invalid_policies_rejected():
    with pytest.raises(LabelPolicyError):
        LabelPolicy(relevance_mode="fuzzy")
    with pytest.raises(LabelPolicyError):
        LabelPolicy(positive_event_types=("teleport",))
    with pytest.raises(LabelPolicyError):
        LabelPolicy(watch_positive_threshold=2)


def test_relevance_by_user_takes_max(tiny_dir):
    from recomlab.datasets import FixtureAdapter

    ds = FixtureAdapter().load(tiny_dir)
    rel = relevance_by_user(derive_labels(ds.interactions, LabelPolicy(relevance_mode="graded")))
    # u3 rated i03 5 (grade 2) and watched 95% (grade 2): max stays 2.
    assert rel["u3"]["i03"] == 2.0
    assert "i06" not in rel["u1"]  # rated 2: explicit negative
    assert "i07" not in rel["u1"]  # click is not positive by default
