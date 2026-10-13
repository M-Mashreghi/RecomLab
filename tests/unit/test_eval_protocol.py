import pytest

from recomlab.datasets.labels import LabelPolicy
from recomlab.datasets.negative_sampling import TrainingNegativeSampler
from recomlab.evaluation.protocol import (
    SAMPLED_WARNING,
    EvaluationProtocol,
    MixedEvaluationModeError,
    assert_comparable,
    sample_universe,
)
from recomlab.experiments import ExperimentRegistry
from recomlab.experiments.runner import run_offline_experiment
from recomlab.models import PopularityRecommender

pytestmark = pytest.mark.metrics


def _run(tiny_ds, tiny_split, tiny_ref, protocol=None, registry=None):
    return run_offline_experiment(
        experiment_id="proto",
        model=PopularityRecommender(),
        dataset=tiny_ds,
        dataset_ref=tiny_ref,
        split=tiny_split,
        label_policy=LabelPolicy(),
        ks=(3,),
        protocol=protocol,
        registry=registry,
    )


def test_full_catalog_deterministic(tiny_ds, tiny_split, tiny_ref):
    m1, r1 = _run(tiny_ds, tiny_split, tiny_ref)
    m2, r2 = _run(tiny_ds, tiny_split, tiny_ref)
    assert r1.mode == "full" and r1.rankings == r2.rankings and r1.metrics == r2.metrics
    assert r1.warnings == [] and m1.evaluation["protocol"] == {"mode": "full", "policy_key": "full"}
    assert m1.evaluation["mean_universe_size"] > 3


def test_sampled_mode_labeled_and_seed_recorded(tiny_ds, tiny_split, tiny_ref):
    proto = EvaluationProtocol("sampled", n_negatives=2, seed=7)
    m, r = _run(tiny_ds, tiny_split, tiny_ref, proto)
    assert r.mode == "sampled" and r.warnings == [SAMPLED_WARNING]
    assert m.evaluation["protocol"]["seed"] == 7
    assert m.evaluation["protocol"]["n_negatives"] == 2
    assert m.evaluation["protocol"]["policy_key"] == "sampled:uniform:2:seed7"
    assert "sampled" in m.evaluation["universe"]
    assert m.evaluation["mean_universe_size"] <= 3  # 1 positive + 2 negatives per user
    _, again = _run(tiny_ds, tiny_split, tiny_ref, proto)
    assert again.rankings == r.rankings
    # Sampled metrics look easier than full-catalog ones on the same model.
    _, full = _run(tiny_ds, tiny_split, tiny_ref)
    assert r.metrics["hit_rate@3"] >= full.metrics["hit_rate@3"]


def test_sampled_and_full_cannot_be_mixed(tiny_ds, tiny_split, tiny_ref, tmp_path):
    reg = ExperimentRegistry(tmp_path / "r.db")
    full, _ = _run(tiny_ds, tiny_split, tiny_ref, registry=reg)
    sampled, _ = _run(
        tiny_ds, tiny_split, tiny_ref, EvaluationProtocol("sampled", 2, seed=1), registry=reg
    )
    other, _ = _run(tiny_ds, tiny_split, tiny_ref, EvaluationProtocol("sampled", 3, seed=1))
    assert assert_comparable([full.evaluation, full.evaluation]) == "full"
    with pytest.raises(MixedEvaluationModeError):
        assert_comparable([full.evaluation, sampled.evaluation])
    with pytest.raises(MixedEvaluationModeError):
        assert_comparable([sampled.evaluation, other.evaluation])


def test_universe_sampling_rules(tiny_ds):
    proto = EvaluationProtocol("sampled", n_negatives=3, seed=0)
    universe = sample_universe("u1", ["i01"], ["i01", "i02", "i03", "i04", "i05"], proto)
    assert universe[0] == "i01" and len(universe) == 4 and len(set(universe)) == 4
    assert sample_universe("u1", ["i01"], ["i01", "i02"], proto) == ["i01", "i02"]
    assert universe == sample_universe("u1", ["i01"], ["i05", "i04", "i03", "i02", "i01"], proto)
    negs = TrainingNegativeSampler().sample(tiny_ds.interactions, tiny_ds.item_ids, 2)
    with pytest.raises(TypeError):
        sample_universe("u1", negs, [], proto)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        EvaluationProtocol("approximate")
    with pytest.raises(ValueError):
        EvaluationProtocol("sampled", n_negatives=0)
