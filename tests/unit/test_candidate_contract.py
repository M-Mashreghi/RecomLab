import dataclasses
import inspect
from datetime import UTC, datetime

import pytest

from recomlab.candidates import (
    Candidate,
    CandidateContractError,
    CandidateGenerator,
    CandidateRequest,
    PopularityCandidates,
    generate_all,
    validate_candidate_set,
)
from recomlab.models import NotFittedError


class DuplicateEmitter(CandidateGenerator):
    """Emits scores via a dict, so duplicates are impossible by construction."""

    name = "dup"

    def _fit(self, history, items):
        pass

    def _score(self, request):
        return {"i01": 1.0, "i02": 1.0, "i03": 0.5}, {}, {"note": "x"}


def test_budget_enforced_and_contract_valid(tiny_ds, tiny_split):
    gen = PopularityCandidates().fit(tiny_split.train, tiny_ds.items)
    for budget in (1, 3, 50):
        cs = gen.generate(CandidateRequest("u1", budget))
        validate_candidate_set(cs)
        assert len(cs.candidates) == min(budget, len(tiny_ds.items))
    cs = gen.generate(CandidateRequest("u1", 3))
    assert cs.metadata["truncated"] and cs.metadata["budget"] == 3
    assert cs.source == "popularity" and cs.source_version == "popularity-1"
    assert all(c.evidence["train_count"] >= 0 for c in cs.candidates)


def test_invalid_budget_rejected():
    for bad in (0, -1, 2.5):
        with pytest.raises(CandidateContractError):
            CandidateRequest("u1", bad)  # type: ignore[arg-type]


def test_unique_items_and_deterministic_ties(tiny_ds):
    gen = DuplicateEmitter().fit([], tiny_ds.items)
    cs = gen.generate(CandidateRequest("u1", 5, exclude_items=frozenset({"i03"})))
    assert cs.item_ids == ("i01", "i02")
    assert [c.rank for c in cs.candidates] == [1, 2]
    validate_candidate_set(cs)


def test_validator_catches_violations(tiny_ds):
    cs = DuplicateEmitter().fit([], tiny_ds.items).generate(CandidateRequest("u1", 2))
    dup = dataclasses.replace(cs, candidates=(cs.candidates[0], cs.candidates[0]))
    with pytest.raises(CandidateContractError):
        validate_candidate_set(dup)
    over = dataclasses.replace(cs, request=CandidateRequest("u1", 1))
    with pytest.raises(CandidateContractError):
        validate_candidate_set(over)
    mixed = dataclasses.replace(
        cs, candidates=(cs.candidates[0], Candidate("i02", "other", 1.0, 2))
    )
    with pytest.raises(CandidateContractError):
        validate_candidate_set(mixed)


@pytest.mark.leakage
def test_generators_never_see_labels_or_future(tiny_ds, tiny_split):
    params = inspect.signature(CandidateGenerator.generate).parameters
    assert list(params) == ["self", "request"]
    assert {f.name for f in dataclasses.fields(CandidateRequest)} == {
        "user_id",
        "budget",
        "exclude_items",
        "request_id",
    }
    with pytest.raises(ValueError):
        PopularityCandidates().fit(
            tiny_ds.interactions, tiny_ds.items, as_of=datetime(2024, 1, 5, tzinfo=UTC)
        )


def test_multiple_generators_for_fusion(tiny_ds, tiny_split):
    gens = [
        PopularityCandidates().fit(tiny_split.train, tiny_ds.items),
        DuplicateEmitter().fit([], tiny_ds.items),
    ]
    sets = generate_all(gens, CandidateRequest("u1", 4))
    assert set(sets) == {"popularity", "dup"}
    with pytest.raises(CandidateContractError):
        generate_all([gens[0], gens[0]], CandidateRequest("u1", 4))
    with pytest.raises(NotFittedError):
        PopularityCandidates().generate(CandidateRequest("u1", 4))
