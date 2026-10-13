import math

import pytest

from recomlab.candidates import CandidateRequest, validate_candidate_set
from recomlab.candidates.item_item import ItemItemCandidates
from recomlab.domain.schemas import Interaction, Item


def _ev(n, user, item, day):
    return Interaction(  # type: ignore[arg-type]
        f"e{n}", user, item, "click", f"2024-01-{day:02d}T00:00:00Z"
    )


EVENTS = [
    _ev(1, "u1", "a", 1),
    _ev(2, "u1", "b", 2),
    _ev(3, "u2", "a", 1),
    _ev(4, "u2", "b", 2),
    _ev(5, "u2", "c", 3),
    _ev(6, "u3", "c", 1),
    _ev(7, "u3", "d", 2),
    _ev(8, "u4", "a", 5),
]
ITEMS = [Item("a", categories=("x",)), Item("b", categories=("x",)), Item("c"), Item("d")]


def test_known_neighbors_hand_checked():
    gen = ItemItemCandidates().fit(EVENTS, ITEMS)
    # users(a)={u1,u2,u4}, users(b)={u1,u2}, users(c)={u2,u3}, users(d)={u3}
    assert dict(gen.neighbors("a")) == pytest.approx({"b": 2 / math.sqrt(6), "c": 1 / math.sqrt(6)})
    assert gen.neighbors("d") == [("c", pytest.approx(1 / math.sqrt(2)))]


def test_candidates_aggregate_history_with_evidence():
    gen = ItemItemCandidates(decay=0.5).fit(EVENTS, ITEMS)
    cs = gen.generate(CandidateRequest("u4", 5))
    validate_candidate_set(cs)
    assert cs.item_ids == ("b", "c")
    top = cs.candidates[0]
    assert top.evidence["seed_items"] == [{"item_id": "a", "similarity": round(2 / 6**0.5, 6)}]
    cs1 = gen.generate(CandidateRequest("u1", 5))
    # u1 history: a (older), b (recent, weight 1); c gets b->c + 0.5 * a->c
    expected_c = 1 / math.sqrt(4) + 0.5 * (1 / math.sqrt(6))
    assert cs1.candidates[0].item_id == "c"
    assert cs1.candidates[0].score == pytest.approx(expected_c)
    assert "a" not in cs1.item_ids and "b" not in cs1.item_ids


def test_cold_user_fallback_is_explicit():
    gen = ItemItemCandidates().fit(EVENTS, ITEMS)
    cs = gen.generate(CandidateRequest("new_user", 5))
    assert cs.is_empty
    assert cs.metadata["fallback_required"] is True
    assert cs.metadata["reason"] == "cold_user_no_history"


@pytest.mark.leakage
def test_no_future_interactions_used(tiny_ds, tiny_split):
    train_gen = ItemItemCandidates().fit(tiny_split.train, tiny_ds.items)
    full_gen = ItemItemCandidates().fit(tiny_ds.interactions, tiny_ds.items)
    assert train_gen.neighbors("i01") != full_gen.neighbors("i01")
    held = {(e.user_id, e.item_id) for e in tiny_split.test}
    for user in {u for u, _ in held}:
        assert all(i not in train_gen.user_history[user] for u, i in held if u == user)


def test_metadata_similarity_blend():
    gen = ItemItemCandidates(metadata_weight=1.0).fit(EVENTS, ITEMS)
    assert gen.neighbors("a") == [("b", 1.0)]
    with pytest.raises(ValueError):
        ItemItemCandidates(metadata_weight=2.0)


def test_tiny_fixture_cluster_neighbors(tiny_ds, tiny_split):
    gen = ItemItemCandidates().fit(tiny_split.train, tiny_ds.items)
    top_action = [i for i, _ in gen.neighbors("i02")[:3]]
    assert set(top_action) <= {"i01", "i03", "i04", "i05"}
    top_drama = [i for i, _ in gen.neighbors("i09")[:2]]
    assert set(top_drama) <= {"i06", "i07", "i08", "i10"}
