from collections import Counter
from datetime import UTC, datetime

import numpy as np
import pytest

from recomlab.datasets.labels import LabelPolicy
from recomlab.datasets.splits import SplitConfig, make_split
from recomlab.evaluation.metrics import ndcg_at_k
from recomlab.evaluation.offline import holdout_truth
from recomlab.features import PointInTimeFeatures
from recomlab.models.als import ALSParams, ALSRecommender
from recomlab.rankers import RankRequest
from recomlab.rankers.collaborative import CollaborativeRanker
from recomlab.rankers.heuristic import HeuristicRanker
from recomlab.rankers.ltr import (
    HeldOutTuningError,
    LTRParams,
    LTRRanker,
    group_ndcg,
    train_ltr,
    tune_on_validation,
)
from recomlab.rankers.ltr_data import (
    LTRDataset,
    build_ltr_dataset,
    default_candidate_factory,
    inner_training_cutoff,
)

GLOBAL = SplitConfig(
    protocol="global_time",
    validation_cutoff="2024-01-08T00:00:00Z",
    test_cutoff="2024-01-11T00:00:00Z",
)
GRID = [LTRParams(0.1, 0.01, 100), LTRParams(0.5, 0.0, 100), LTRParams(0.05, 0.1, 50)]


@pytest.fixture
def data(tiny_ds):
    split = make_split(tiny_ds.interactions, GLOBAL)
    cutoff = inner_training_cutoff(split.train, 0.6)
    train_ds = build_ltr_dataset(
        [e for e in split.train if e.event_time < cutoff],
        [e for e in split.train if e.event_time >= cutoff],
        tiny_ds.items,
        cutoff,
        LabelPolicy(),
        partition="train",
        forbidden_event_ids=[e.event_id for e in (*split.validation, *split.test)],
    )
    val_as_of = datetime(2024, 1, 8, tzinfo=UTC)
    val_ds = build_ltr_dataset(
        split.train,
        split.validation,
        tiny_ds.items,
        val_as_of,
        LabelPolicy(),
        partition="validation",
    )
    return split, train_ds, val_ds, val_as_of


def test_training_and_evaluation_smoke(data):
    _, train_ds, val_ds, _ = data
    model = train_ltr(train_ds, LTRParams(epochs=50))
    assert model.training_pairs > 0
    assert model.weights.shape == (len(train_ds.feature_names),)
    scores = model.score_matrix(val_ds.features)
    assert np.isfinite(scores).all()
    assert 0.0 <= group_ndcg(val_ds, scores, 5) <= 1.0
    again = train_ltr(train_ds, LTRParams(epochs=50))
    assert np.array_equal(model.weights, again.weights)  # deterministic


def test_tuning_uses_validation_and_refuses_test(data):
    _, train_ds, val_ds, _ = data
    model, trials = tune_on_validation(train_ds, val_ds, GRID, k=5)
    assert len(trials) == len(GRID)
    best = max(t["val_ndcg@5"] for t in trials)
    assert model.validation["ndcg@5"] == best
    persisted = model.to_dict()
    assert persisted["params"] in [t["params"] for t in trials]
    assert sum(persisted["feature_importance"].values()) == pytest.approx(1.0)
    test_tagged = LTRDataset(**{**val_ds.__dict__, "partition": "test"})
    with pytest.raises(HeldOutTuningError):
        tune_on_validation(train_ds, test_tagged, GRID)
    with pytest.raises(HeldOutTuningError):
        train_ltr(test_tagged)


@pytest.mark.integration
def test_compared_with_heuristic_and_collaborative(data, tiny_ds):
    split, train_ds, val_ds, val_as_of = data
    model, _ = tune_on_validation(train_ds, val_ds, GRID, k=5)
    feats = PointInTimeFeatures(split.train, tiny_ds.items, val_as_of)
    als = ALSRecommender(ALSParams(factors=4, iterations=8)).fit(split.train, tiny_ds.items)
    rankers = {
        "ltr": LTRRanker(model, feats),
        "heuristic": HeuristicRanker(feats),
        "collaborative": CollaborativeRanker(als, Counter(e.item_id for e in split.train)),
    }
    candidates_for = default_candidate_factory()(split.train, tiny_ds.items)
    truth = holdout_truth(split, "validation", LabelPolicy())
    results = {}
    for name, ranker in rankers.items():
        vals = []
        for user, rel in sorted(truth.items()):
            req = RankRequest.from_fused(candidates_for(user), k=5)
            vals.append(ndcg_at_k(list(ranker.rank(req).item_ids), rel, 5))
        results[name] = float(np.mean(vals))
    assert set(results) == {"ltr", "heuristic", "collaborative"}
    assert all(0.0 <= v <= 1.0 for v in results.values())
    # The LTR ranker reproduces the dataset scores for the same candidates.
    user = val_ds.query_ids[0]
    out = rankers["ltr"].rank(RankRequest.from_fused(candidates_for(user), k=50))
    sl = next(s for q, s in val_ds.groups() if q == user)
    ds_scores = dict(zip(val_ds.item_ids[sl], model.score_matrix(val_ds.features[sl]), strict=True))
    for item in out.items:
        assert item.raw_score == pytest.approx(ds_scores[item.item_id])
        assert sum(item.contributions.values()) == pytest.approx(item.raw_score)
