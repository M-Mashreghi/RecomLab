import dataclasses

import pytest

from recomlab.datasets.labels import LabelPolicy
from recomlab.evaluation.slices import SliceConfig, slice_report, user_slices
from recomlab.experiments import ExperimentRegistry
from recomlab.experiments.runner import run_offline_experiment
from recomlab.models import PopularityRecommender, RandomRecommender
from recomlab.registry.model_registry import ModelRegistry
from recomlab.registry.promotion import (
    PromotionInputError,
    PromotionPolicy,
    RegressionGuardrail,
    apply_decision,
    evaluate_promotion,
    replay_decision,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def env(tmp_path, tiny_ds, tiny_split, tiny_ref):
    exps = ExperimentRegistry(tmp_path / "runs.db")
    art = tmp_path / "artifacts"

    def run(model, created):
        m, res = run_offline_experiment(
            experiment_id="promo",
            model=model,
            dataset=tiny_ds,
            dataset_ref=tiny_ref,
            split=tiny_split,
            label_policy=LabelPolicy(),
            ks=(5,),
            partition="validation",
            artifacts_dir=art,
            created_at=created,
        )
        history = tiny_split.train_until("validation")
        cfg = SliceConfig()
        report = slice_report(
            res.per_user,
            user_slices(history, list(res.per_user), cfg),
            res.rankings,
            {},
            {},
            5,
            cfg,
        )
        m = dataclasses.replace(
            m, metrics={**m.metrics, "slices": report.to_dict()["user_dimensions"]}
        )
        exps.log_run(m)
        return m

    champion = run(RandomRecommender(seed=3), "2024-04-01T00:00:00+00:00")
    challenger = run(PopularityRecommender(), "2024-04-02T00:00:00+00:00")
    return exps, art, champion, challenger


LENIENT = PromotionPolicy("ndcg@5", min_improvement=0.0, require_significance=False)


def test_primary_rule_and_passing_guardrails(env):
    exps, art, champion, challenger = env
    policy = dataclasses.replace(
        LENIENT,
        guardrails=(
            RegressionGuardrail("catalog_coverage@5", max_regression=1.0),
            RegressionGuardrail("latency_ms.p50", higher_is_better=False, max_regression=1e6),
        ),
    )
    decision = evaluate_promotion(champion, challenger, art, policy, "bundle-x")
    delta = decision.comparison["delta"]
    assert decision.primary_passed == (delta >= 0)
    assert decision.promote == (delta >= 0)
    strict = dataclasses.replace(policy, min_improvement=delta + 0.01)
    assert not evaluate_promotion(champion, challenger, art, strict).promote


def test_failing_guardrail_blocks_promotion(env):
    exps, art, champion, challenger = env
    for guard in (
        RegressionGuardrail("catalog_coverage@5", max_regression=-1.0),  # impossible to satisfy
        RegressionGuardrail("slices.coldness.warm.ndcg@5", max_regression=-1.0),
        RegressionGuardrail("missing.metric"),
    ):
        decision = evaluate_promotion(
            champion, challenger, art, dataclasses.replace(LENIENT, guardrails=(guard,))
        )
        assert not decision.promote
        assert any("guardrail failed" in r for r in decision.reasons)


def test_decision_reproducible_from_stored_report(env, tmp_path):
    exps, art, champion, challenger = env
    policy = dataclasses.replace(
        LENIENT, guardrails=(RegressionGuardrail("catalog_coverage@5", max_regression=0.5),)
    )
    decision = evaluate_promotion(champion, challenger, art, policy, "bundle-x")
    path = decision.save(tmp_path / "decisions")
    assert replay_decision(path, ExperimentRegistry(exps.db_path), art) == decision


def test_input_validation(env):
    exps, art, champion, challenger = env
    other_split = dataclasses.replace(challenger, split={**challenger.split, "split_id": "other"})
    with pytest.raises(PromotionInputError):
        evaluate_promotion(champion, other_split, art, LENIENT)
    test_run = dataclasses.replace(challenger, split={**challenger.split, "partition": "test"})
    with pytest.raises(PromotionInputError):
        evaluate_promotion(champion, test_run, art, LENIENT)


def test_apply_and_rollback_to_previous_champion(env, tmp_path, bundle_factory, serving_context):
    exps, art, champion, challenger = env
    items, history, as_of = serving_context
    models = ModelRegistry(tmp_path / "registry")
    champ_bundle = bundle_factory(models, ranker="heuristic")
    chall_bundle = bundle_factory(models, ranker="collaborative")
    models.activate(champ_bundle.bundle_id, items, history, as_of)
    blocked = evaluate_promotion(
        champion,
        challenger,
        art,
        dataclasses.replace(
            LENIENT, guardrails=(RegressionGuardrail("catalog_coverage@5", max_regression=-1.0),)
        ),
        chall_bundle.bundle_id,
    )
    assert not apply_decision(blocked, models, items, history, as_of)
    assert models.active_bundle_id == champ_bundle.bundle_id
    forced = dataclasses.replace(blocked, promote=True)
    assert apply_decision(forced, models, items, history, as_of)
    assert models.active_bundle_id == chall_bundle.bundle_id
    models.rollback(items, history, as_of, "post-promotion regression")
    assert models.active_bundle_id == champ_bundle.bundle_id
    assert models.history()[-1]["reason"] == "post-promotion regression"
