import json

import pytest

from recomlab.candidates import Candidate, CandidateRequest, CandidateSet, PopularityCandidates
from recomlab.candidates.fusion import FusionConfig, fuse
from recomlab.candidates.item_item import ItemItemCandidates
from recomlab.datasets.labels import LabelPolicy
from recomlab.domain.eligibility import EligibilityFilter
from recomlab.domain.schemas import Interaction
from recomlab.evaluation.offline import holdout_truth
from recomlab.evaluation.retrieval import (
    activity_segments,
    failure_breakdown,
    retrieval_diagnostics,
)
from recomlab.experiments import RunManifest


def cset(user, items, source="s"):
    req = CandidateRequest(user, 3)
    return CandidateSet(
        req,
        source,
        f"{source}-1",
        tuple(Candidate(i, source, 1.0, n + 1) for n, i in enumerate(items)),
    )


CANDS = {"u1": cset("u1", ["a", "b", "c"]), "u2": cset("u2", ["d"]), "u3": cset("u3", [])}
TRUTH = {"u1": {"a": 1.0, "z": 1.0}, "u2": {"d": 1.0}, "u3": {"a": 1.0}}
SEGMENTS = {"u1": "warm", "u2": "sparse", "u3": "cold"}


@pytest.mark.metrics
def test_hand_checkable_candidate_recall():
    d = retrieval_diagnostics(CANDS, TRUTH, SEGMENTS, candidate_budget=3)
    assert d.per_user_recall == {"u1": 0.5, "u2": 1.0, "u3": 0.0}
    assert d.recall == pytest.approx(0.5)
    assert d.hit_rate == pytest.approx(2 / 3)
    assert d.empty_candidate_rate == pytest.approx(1 / 3)
    assert d.mean_candidate_count == pytest.approx(4 / 3)
    assert d.candidate_count == {"min": 0.0, "median": 1, "max": 3}
    assert d.by_segment["cold"]["empty_candidate_rate"] == 1.0
    assert d.by_segment["warm"]["recall"] == 0.5
    assert d.by_source["s"]["recall"] == pytest.approx(0.5)


@pytest.mark.metrics
def test_missing_from_candidates_vs_ranking_misses():
    rankings = {"u1": ["b", "a"], "u2": ["d"], "u3": []}
    assert failure_breakdown(CANDS, rankings, TRUTH, k=1) == {
        "hit": 1,  # u2:d
        "ranking_failure": 1,  # u1:a is a candidate but ranked 2nd
        "retrieval_failure": 2,  # u1:z and u3:a never retrieved
    }


def test_activity_segments():
    ev = [
        Interaction(f"e{n}", "w", "i", "click", "2024-01-01T00:00:00Z")  # type: ignore[arg-type]
        for n in range(3)
    ]
    ev.append(Interaction("s1", "s", "i", "click", "2024-01-01T00:00:00Z"))  # type: ignore[arg-type]
    assert activity_segments(ev, ["w", "s", "c"]) == {"w": "warm", "s": "sparse", "c": "cold"}


@pytest.mark.integration
def test_diagnostics_visible_in_experiment_output(tiny_ds, tiny_split, tmp_path):
    history = tiny_split.train_until("test")
    gens = [
        PopularityCandidates().fit(history, tiny_ds.items),
        ItemItemCandidates().fit(history, tiny_ds.items),
    ]
    truth = holdout_truth(tiny_split, "test", LabelPolicy())
    elig = EligibilityFilter(tiny_ds.items, history)
    fused = {}
    for user in truth:
        sets = {g.name: g.generate(CandidateRequest(user, 5)) for g in gens}
        fused[user] = fuse(user, sets, elig, FusionConfig(budget=5))
    segments = activity_segments(history, truth)
    diag = retrieval_diagnostics(fused, truth, segments, candidate_budget=5)
    assert set(diag.by_source) == {"popularity", "item_item"}
    assert 0.0 <= diag.recall <= 1.0
    diag.write_json(tmp_path / "retrieval.json")
    assert json.loads((tmp_path / "retrieval.json").read_text())["recall@5"] == diag.recall
    manifest = RunManifest(
        experiment_id="retrieval",
        dataset={"dataset_id": "tiny", "version": "1"},
        split={"split_id": tiny_split.split_id},
        model={"name": "fusion", "version": "1"},
        metric_config={"candidate_budget": 5},
        seeds={},
        metrics={"retrieval": diag.to_dict()},
        created_at="2024-01-01T00:00:00+00:00",
    )
    assert RunManifest.from_dict(manifest.to_dict()).metrics["retrieval"]["n_users"] == len(truth)


def test_users_without_candidates_rejected():
    with pytest.raises(ValueError):
        retrieval_diagnostics({}, TRUTH, SEGMENTS, 3)
