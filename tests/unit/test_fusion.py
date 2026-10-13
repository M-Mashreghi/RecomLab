import pytest

from recomlab.candidates import (
    Candidate,
    CandidateRequest,
    CandidateSet,
    PopularityCandidates,
    generate_all,
)
from recomlab.candidates.fusion import FusionConfig, fuse
from recomlab.candidates.item_item import ItemItemCandidates
from recomlab.domain.eligibility import EligibilityFilter, EligibilityPolicy
from recomlab.domain.schemas import Interaction, Item

ITEMS = [Item("a"), Item("b"), Item("c"), Item("d"), Item("x", available=False)]
HISTORY = [Interaction("e1", "u", "d", "click", "2024-01-01T00:00:00Z")]  # type: ignore[arg-type]


def cset(source, scored, user="u"):
    req = CandidateRequest(user, 10)
    cands = tuple(Candidate(i, source, s, r + 1) for r, (i, s) in enumerate(scored))
    return CandidateSet(req, source, f"{source}-1", cands)


SETS = {
    "s1": cset("s1", [("a", 0.9), ("b", 0.5), ("x", 0.4)]),
    "s2": cset("s2", [("b", 3.0), ("c", 2.0), ("d", 1.0)]),
}


def test_rrf_hand_computed_and_provenance():
    elig = EligibilityFilter(ITEMS, HISTORY)
    out = fuse("u", SETS, elig, FusionConfig(method="rrf", rrf_k=1))
    # b: 1/(1+2) + 1/(1+1) = 0.8333; a: 1/2 = 0.5; c: 1/3
    assert out.item_ids == ("b", "a", "c")
    assert out.candidates[0].score == pytest.approx(1 / 3 + 1 / 2)
    assert set(out.candidates[0].sources) == {"s1", "s2"}
    assert out.candidates[0].sources["s2"]["rank"] == 1
    assert out.items_from_source("s1") == {"a", "b"}
    assert out.counts == {"pre_dedup": 6, "post_dedup": 5, "post_filter": 3, "final": 3}
    assert out.removed == {"not_eligible_catalog": 1, "consumed": 1}
    assert out.source_versions == {"s1": "s1-1", "s2": "s2-1"}


def test_weighted_score_fusion():
    elig = EligibilityFilter(ITEMS, [], EligibilityPolicy(require_available=False))
    out = fuse("u", SETS, elig, FusionConfig(method="weighted_score", weights={"s1": 2.0}))
    # s1 minmax: a=1, b=0.2, x=0; s2 minmax: b=1, c=0.5, d=0
    expected = {"a": 2.0, "b": 2 * 0.2 + 1.0, "c": 0.5, "x": 0.0, "d": 0.0}
    assert {c.item_id: c.score for c in out.candidates} == pytest.approx(expected)
    assert out.item_ids[:3] == ("a", "b", "c")
    assert out.item_ids[3:] == ("d", "x")  # zero-score tie broken by id


def test_deterministic_and_order_independent():
    elig = EligibilityFilter(ITEMS, HISTORY)
    a = fuse("u", SETS, elig)
    b = fuse("u", dict(reversed(list(SETS.items()))), elig)
    assert a == b


def test_filters_cannot_reintroduce_duplicates_and_budget(tiny_ds, tiny_split):
    gens = [
        PopularityCandidates().fit(tiny_split.train, tiny_ds.items),
        ItemItemCandidates().fit(tiny_split.train, tiny_ds.items),
    ]
    sets = generate_all(gens, CandidateRequest("u2", 12))
    elig = EligibilityFilter(tiny_ds.items, tiny_split.train)
    out = fuse("u2", sets, elig, FusionConfig(budget=4))
    assert len(out.item_ids) == len(set(out.item_ids)) == 4
    assert all(elig.is_eligible("u2", i) for i in out.item_ids)
    assert "i12" not in out.item_ids
    for c in out.candidates:
        assert set(c.sources) <= {"popularity", "item_item"} and c.sources


def test_invalid_inputs():
    with pytest.raises(ValueError):
        FusionConfig(method="borda")
    with pytest.raises(ValueError):
        FusionConfig(weights={"s1": -1})
    with pytest.raises(ValueError):
        fuse("other", SETS, EligibilityFilter(ITEMS, []))
