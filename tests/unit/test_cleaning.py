from recomlab.domain.schemas import Item, User
from recomlab.ingestion.cleaning import CleaningPolicy, clean_interactions
from recomlab.ingestion.csv_io import load_items, load_users, read_rows

USERS = [User("u1"), User("u2")]
ITEMS = [Item("i1"), Item("i2")]


def row(eid, user="u1", item="i1", etype="rating", value="4", t="2024-01-01T00:00:00Z"):
    return {
        "event_id": eid,
        "user_id": user,
        "item_id": item,
        "event_type": etype,
        "value": value,
        "event_time": t,
    }


BAD_ROWS = [
    row("e1"),  # ok
    row("e2", t="2024-01-01T00:00:00"),  # schema:naive_timestamp
    row("e3", etype="teleport"),  # schema:invalid_event_type
    row("e1"),  # duplicate_exact
    row("e4", value="3"),  # conflicting duplicate pair ...
    row("e4", value="5"),  # ... both dropped
    row("e5", user="u9"),  # unknown_user
    row("e6", item="i9"),  # unknown_item
    row("e7", t="2023-12-01T00:00:00Z"),  # out_of_time_window
    row("e8", user="u2", item="i2", etype="click", value=""),  # ok
    row("", user="u2"),  # schema:missing_id
]


def test_known_bad_fixtures_yield_expected_reason_codes():
    result = clean_interactions(
        BAD_ROWS, USERS, ITEMS, CleaningPolicy(min_time="2024-01-01T00:00:00Z")
    )
    reasons = {(r.row_index, r.reason) for r in result.report.rejections}
    assert reasons == {
        (1, "schema:naive_timestamp"),
        (2, "schema:invalid_event_type"),
        (3, "duplicate_exact"),
        (4, "duplicate_conflict"),
        (5, "duplicate_conflict"),
        (6, "unknown_user"),
        (7, "unknown_item"),
        (8, "out_of_time_window"),
        (10, "schema:missing_id"),
    }
    assert [e.event_id for e in result.interactions] == ["e1", "e8"]


def test_report_totals_reconcile():
    result = clean_interactions(BAD_ROWS, USERS, ITEMS)
    rep = result.report
    assert rep.reconciles()
    assert rep.n_input == len(BAD_ROWS)
    assert rep.n_output + sum(rep.reason_counts.values()) == rep.n_input
    assert [s.rule for s in rep.steps] == [
        "schema",
        "duplicates",
        "known_users",
        "known_items",
        "time_window",
    ]
    assert rep.to_dict()["reason_counts"]["duplicate_conflict"] == 2


def test_no_silent_removal_without_reason():
    result = clean_interactions(BAD_ROWS, USERS, ITEMS)
    removed = len(BAD_ROWS) - len(result.interactions)
    assert removed == len(result.report.rejections)
    assert all(r.reason for r in result.report.rejections)


def test_repeated_runs_identical_and_order_independent(tiny_dir):
    rows = list(read_rows(tiny_dir / "interactions.csv"))
    users = load_users(tiny_dir / "users.csv")
    items = load_items(tiny_dir / "items.csv")
    a = clean_interactions(rows, users, items)
    b = clean_interactions(rows, users, items)
    c = clean_interactions(list(reversed(rows)), users, items)
    assert a.checksum == b.checksum == c.checksum
    assert a.report.to_dict() == b.report.to_dict()
    assert a.report.n_output == 33 and a.report.reconciles()


def test_policy_id_changes_with_policy():
    assert CleaningPolicy().policy_id != CleaningPolicy(min_time="2024-01-01T00:00:00Z").policy_id
    assert CleaningPolicy().policy_id == CleaningPolicy().policy_id


def test_referential_checks_can_be_disabled():
    result = clean_interactions(
        [row("e1", user="u9")],
        USERS,
        ITEMS,
        CleaningPolicy(require_known_users=False),
    )
    assert len(result.interactions) == 1 and result.report.reconciles()
