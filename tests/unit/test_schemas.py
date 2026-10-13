from datetime import UTC, datetime, timedelta, timezone

import pytest

from recomlab.domain.schemas import (
    EventType,
    FeedbackKind,
    Interaction,
    Item,
    SchemaValidationError,
    User,
    find_duplicate_events,
    sort_key,
)
from recomlab.ingestion.csv_io import load_interactions, load_items, load_users


def _ev(**overrides):
    raw = {
        "event_id": "e1",
        "user_id": "u1",
        "item_id": "i1",
        "event_type": "rating",
        "value": "4",
        "event_time": "2024-01-01T10:00:00Z",
    }
    raw.update(overrides)
    return Interaction.from_dict(raw)


def test_tiny_fixture_validates(tiny_dir):
    users = load_users(tiny_dir / "users.csv")
    items = load_items(tiny_dir / "items.csv")
    events = load_interactions(tiny_dir / "interactions.csv")
    assert len(users) == 7 and len(items) == 12 and len(events) == 33
    assert all(e.event_time.tzinfo is UTC for e in events)
    assert {e.kind for e in events} == {FeedbackKind.EXPLICIT, FeedbackKind.IMPLICIT}
    assert not [i for i in items if i.item_id == "i12"][0].available
    assert items[0].categories == ("action", "scifi")


def test_explicit_rating_example():
    ev = _ev()
    assert ev.event_type is EventType.RATING
    assert ev.kind is FeedbackKind.EXPLICIT
    assert ev.value == 4.0


@pytest.mark.parametrize(
    ("etype", "value", "expected"),
    [("click", "", None), ("watch", "0.5", 0.5), ("purchase", "2", 2.0), ("like", None, None)],
)
def test_implicit_examples_keep_semantics(etype, value, expected):
    ev = _ev(event_type=etype, value=value)
    assert ev.kind is FeedbackKind.IMPLICIT
    assert ev.value == expected


def test_timestamps_normalized_to_utc():
    ev = _ev(event_time="2024-01-01T12:00:00+02:00")
    assert ev.event_time == datetime(2024, 1, 1, 10, tzinfo=UTC)
    local = datetime(2024, 1, 1, 12, tzinfo=timezone(timedelta(hours=2)))
    assert _ev(event_time=local).event_time == ev.event_time


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"event_id": ""}, "missing_id"),
        ({"user_id": None}, "missing_id"),
        ({"item_id": "bad id"}, "invalid_id"),
        ({"event_type": "teleport"}, "invalid_event_type"),
        ({"event_time": "2024-01-01T10:00:00"}, "naive_timestamp"),
        ({"event_time": "yesterday"}, "invalid_timestamp"),
        ({"event_time": ""}, "missing_timestamp"),
        ({"value": ""}, "missing_value"),
        ({"value": "abc"}, "invalid_value"),
        ({"value": "nan"}, "invalid_value"),
        ({"value": "11"}, "value_out_of_range"),
        ({"event_type": "watch", "value": "1.5"}, "value_out_of_range"),
        ({"event_type": "purchase", "value": "-1"}, "value_out_of_range"),
        ({"session_id": "has space"}, "invalid_id"),
    ],
)
def test_invalid_interactions_fail_predictably(overrides, code):
    with pytest.raises(SchemaValidationError) as err:
        _ev(**overrides)
    assert err.value.code == code


def test_user_and_item_validation():
    assert User.from_dict({"user_id": "u1", "cohort": "a"}).attributes == {"cohort": "a"}
    with pytest.raises(SchemaValidationError) as err:
        User.from_dict({"user_id": "u1", "email": "x@example.com"})
    assert err.value.code == "pii_attribute"
    with pytest.raises(SchemaValidationError):
        Item.from_dict({"item_id": "i1", "available": "maybe"})
    item = Item.from_dict({"item_id": "i1", "categories": "a| |b", "available": ""})
    assert item.categories == ("a", "b") and item.available


def test_duplicate_policy_exact_vs_conflicting():
    a = _ev()
    b = _ev()
    c = _ev(value="2")
    exact, conflict = find_duplicate_events([a, b, c])
    assert exact == [b]
    assert conflict == [c]


def test_sort_key_is_time_then_id():
    early = _ev(event_id="e9", event_time="2024-01-01T00:00:00Z")
    tie_a = _ev(event_id="e1", event_time="2024-01-02T00:00:00Z")
    tie_b = _ev(event_id="e2", event_time="2024-01-02T00:00:00Z")
    assert sorted([tie_b, early, tie_a], key=sort_key) == [early, tie_a, tie_b]


def test_round_trip_to_dict():
    ev = _ev(session_id="s1")
    assert Interaction.from_dict(ev.to_dict()) == ev
