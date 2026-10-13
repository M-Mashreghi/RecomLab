from datetime import timedelta

import pytest

from recomlab.domain.exposure import (
    EXPOSURE_SCHEMA_VERSION,
    ExposureAvailability,
    FeedbackEvent,
    FeedbackType,
    Impression,
    LinkStatus,
    assess_exposure_log,
    link_feedback,
)
from recomlab.domain.schemas import SchemaValidationError


def imp(iid="imp1", user="u1", item="i1", t="2024-01-01T10:00:00Z", propensity=None, pos=1):
    return Impression.from_dict(
        {
            "impression_id": iid,
            "request_id": "r1",
            "user_id": user,
            "item_id": item,
            "position": pos,
            "model_id": "pop-v1",
            "shown_at": t,
            "propensity": propensity,
        }
    )


def fb(fid="f1", user="u1", item="i1", t="2024-01-01T10:05:00Z", impression_id=None, **extra):
    raw = {
        "feedback_id": fid,
        "user_id": user,
        "item_id": item,
        "feedback_type": "click",
        "event_time": t,
        "impression_id": impression_id,
    }
    raw.update(extra)
    return FeedbackEvent.from_dict(raw)


def test_schema_is_versioned():
    assert EXPOSURE_SCHEMA_VERSION == "1.0"


def test_impression_validation():
    assert imp(propensity="0.25").propensity == 0.25
    assert imp().propensity is None
    for bad in ("0", "1.5", "-0.1"):
        with pytest.raises(SchemaValidationError) as err:
            imp(propensity=bad)
        assert err.value.code == "value_out_of_range"
    with pytest.raises(SchemaValidationError):
        imp(pos=0)
    with pytest.raises(SchemaValidationError):
        imp(t="2024-01-01T10:00:00")


def test_round_trips():
    i = imp(propensity=0.5)
    assert Impression.from_dict(i.to_dict()) == i
    f = fb(impression_id="imp1")
    assert FeedbackEvent.from_dict(f.to_dict()) == f
    assert f.feedback_type is FeedbackType.CLICK


def test_explicit_linkage_verified():
    imps = [imp()]
    links = link_feedback(
        imps,
        [
            fb("f1", impression_id="imp1"),
            fb("f2", impression_id="nope"),
            fb("f3", item="i2", impression_id="imp1"),
            fb("f4", t="2024-01-01T09:00:00Z", impression_id="imp1"),
        ],
    )
    assert [link.status for link in links] == [
        LinkStatus.LINKED,
        LinkStatus.INVALID,
        LinkStatus.INVALID,
        LinkStatus.INVALID,
    ]
    assert [link.reason for link in links[1:]] == [
        "unknown_impression",
        "user_item_mismatch",
        "feedback_before_exposure",
    ]
    assert links[0].impression == imps[0]


def test_inference_off_by_default_and_windowed():
    imps = [imp("a", t="2024-01-01T09:00:00Z"), imp("b", t="2024-01-01T10:00:00Z")]
    feedback = [fb("f1"), fb("f2", t="2024-01-03T00:00:00Z")]
    assert {link.status for link in link_feedback(imps, feedback)} == {LinkStatus.UNLINKED}
    links = link_feedback(imps, feedback, infer_window=timedelta(hours=1))
    assert links[0].status is LinkStatus.INFERRED and links[0].impression.impression_id == "b"
    assert links[1].status is LinkStatus.UNLINKED


def test_missing_propensity_is_explicit_not_guessed():
    assert assess_exposure_log(None).availability is ExposureAvailability.UNAVAILABLE
    partial = assess_exposure_log([imp("a", propensity=0.5), imp("b")])
    assert partial.availability is ExposureAvailability.IMPRESSIONS_ONLY
    assert partial.n_missing_propensity == 1
    assert not partial.exposure_aware_evaluation_allowed
    full = assess_exposure_log([imp("a", propensity=0.5)])
    assert full.exposure_aware_evaluation_allowed


@pytest.mark.parametrize("field", ["comment", "free_text", "email"])
def test_free_text_and_pii_rejected(field):
    with pytest.raises(SchemaValidationError) as err:
        fb(**{field: "great movie, mail me"})
    assert err.value.code == "free_text_not_allowed"
