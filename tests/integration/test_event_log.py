from datetime import UTC, datetime

import pytest

from recomlab.domain.exposure import (
    ExposureAvailability,
    FeedbackEvent,
    Impression,
    LinkStatus,
    assess_exposure_log,
)
from recomlab.registry.model_registry import ModelRegistry
from recomlab.services.event_log import ConflictingEventError, EventLog
from recomlab.services.recommendation import RecommendationRequest, RecommendationService

pytestmark = pytest.mark.integration
SHOWN = datetime(2024, 2, 1, 12, tzinfo=UTC)


@pytest.fixture
def served(tmp_path, bundle_factory, serving_context):
    items, history, as_of = serving_context
    reg = ModelRegistry(tmp_path / "registry")
    svc = RecommendationService(reg, items, history, as_of)
    svc.activate(bundle_factory(reg).bundle_id)
    resp = svc.recommend(RecommendationRequest("u2", 3, trace_id="req1"))
    return resp, EventLog(tmp_path / "events.db")


def test_duplicate_events_are_idempotent(served):
    resp, log = served
    first = log.log_response(resp, SHOWN)
    again = log.log_response(resp, SHOWN)
    assert (first.inserted, first.duplicates) == (3, 0)
    assert (again.inserted, again.duplicates) == (0, 3)
    assert len(log.impressions("req1")) == 3
    fb = FeedbackEvent.from_dict(
        {
            "feedback_id": "f1",
            "user_id": "u2",
            "item_id": resp.items[0].item_id,
            "feedback_type": "click",
            "event_time": "2024-02-01T12:05:00Z",
            "impression_id": "req1:1",
        }
    )
    assert log.log_feedback([fb, fb]).duplicates == 1
    changed = FeedbackEvent.from_dict({**fb.to_dict(), "feedback_type": "like"})
    with pytest.raises(ConflictingEventError):
        log.log_feedback([changed])
    moved = Impression.from_dict({**log.impressions("req1")[0].to_dict(), "position": 9})
    with pytest.raises(ConflictingEventError):
        log.log_impressions([moved])


def test_feedback_linkage_is_queryable(served):
    resp, log = served
    log.log_response(resp, SHOWN)
    log.log_feedback(
        [
            FeedbackEvent.from_dict(
                {
                    "feedback_id": "f1",
                    "user_id": "u2",
                    "item_id": resp.items[1].item_id,
                    "feedback_type": "click",
                    "event_time": "2024-02-01T12:01:00Z",
                    "impression_id": "req1:2",
                }
            ),
            FeedbackEvent.from_dict(
                {
                    "feedback_id": "f2",
                    "user_id": "u2",
                    "item_id": "i11",
                    "feedback_type": "purchase",
                    "event_time": "2024-02-01T12:02:00Z",
                }
            ),
        ]
    )
    links = {link.feedback.feedback_id: link for link in log.linked_feedback()}
    assert links["f1"].status is LinkStatus.LINKED
    assert links["f1"].impression.position == 2
    assert links["f2"].status is LinkStatus.UNLINKED
    per_request = log.feedback_for_request("req1")
    assert [link.feedback.feedback_id for link in per_request] == ["f1"]
    events = log.as_interactions()
    assert {e.event_id for e in events} == {"fb:f1", "fb:f2"}


def test_exposure_conforms_to_contract(served):
    resp, log = served
    log.log_response(resp, SHOWN)
    stored = log.impressions()
    assert all(isinstance(i, Impression) for i in stored)
    assert [i.position for i in stored] == [1, 2, 3]
    assert {i.model_id for i in stored} == {resp.bundle_id}
    assert assess_exposure_log(stored).availability is ExposureAvailability.IMPRESSIONS_ONLY
    log2 = EventLog(":memory:")
    log2.log_response(resp, SHOWN, propensities=[0.9, 0.5, 0.2])
    assert assess_exposure_log(log2.impressions()).exposure_aware_evaluation_allowed
    with pytest.raises(ValueError):
        log2.log_response(resp, SHOWN, propensities=[0.5])
