import pytest

from recomlab.datasets.labels import LabelPolicy
from recomlab.evaluation.leaderboard import (
    Guardrail,
    SelectionError,
    SelectionPolicy,
    attach_test_report,
    build_leaderboard,
    pareto_front,
    select_champion,
)
from recomlab.evaluation.protocol import EvaluationProtocol, MixedEvaluationModeError
from recomlab.experiments import ExperimentRegistry
from recomlab.experiments.runner import run_offline_experiment
from recomlab.models import PopularityRecommender, RandomRecommender
from recomlab.models.als import ALSParams, ALSRecommender

POLICY = SelectionPolicy(
    primary_metric="ndcg@3",
    guardrails=(
        Guardrail("catalog_coverage@3", ">=", 0.1),
        Guardrail("n_users_evaluated", ">=", 1),
    ),
    pareto_metrics=("ndcg@3", "catalog_coverage@3"),
)


def _models():
    return [
        PopularityRecommender(),
        RandomRecommender(seed=1),
        ALSRecommender(ALSParams(factors=4, iterations=8)),
    ]


@pytest.fixture
def runs(tiny_ds, tiny_split, tiny_ref, tmp_path):
    reg = ExperimentRegistry(tmp_path / "runs.db")
    out = {"validation": [], "test": []}
    for partition in ("validation", "test"):
        for model in _models():
            m, _ = run_offline_experiment(
                experiment_id="lb",
                model=model,
                dataset=tiny_ds,
                dataset_ref=tiny_ref,
                split=tiny_split,
                label_policy=LabelPolicy(),
                ks=(3,),
                partition=partition,
                registry=reg,
                created_at=f"2024-02-01T00:00:0{len(out[partition])}+00:00",
            )
            out[partition].append(m)
    return reg, out


def test_test_set_cannot_choose(runs):
    _, r = runs
    with pytest.raises(SelectionError):
        select_champion(r["test"], POLICY)
    with pytest.raises(SelectionError):
        select_champion([*r["validation"], r["test"][0]], POLICY)


def test_champion_decision_records_rationale_and_metrics(runs):
    _, r = runs
    decision = select_champion(r["validation"], POLICY)
    assert decision.champion_run_id in {m.run_id for m in r["validation"]}
    assert "validation ndcg@3" in decision.rationale
    assert decision.policy["policy_id"] == POLICY.policy_id
    assert all("metrics" in row and "guardrails" in row for row in decision.rows)
    final = attach_test_report(decision, r["test"])
    assert final.test_report["note"].startswith("test metrics reported once")
    assert "ndcg@3" in final.test_report["metrics"]
    with pytest.raises(SelectionError):
        attach_test_report(decision, r["validation"])


def test_leaderboard_reproducible_from_manifests(runs):
    reg, r = runs
    ids = [m.run_id for m in r["validation"]]
    from_registry = [reg.get_run(i) for i in ids]
    assert select_champion(from_registry, POLICY) == select_champion(r["validation"], POLICY)
    assert select_champion(list(reversed(from_registry)), POLICY).champion_run_id == (
        select_champion(r["validation"], POLICY).champion_run_id
    )


def test_guardrail_failure_excludes_and_missing_evidence_fails_closed(runs):
    _, r = runs
    strict = SelectionPolicy("ndcg@3", guardrails=(Guardrail("catalog_coverage@3", ">=", 2.0),))
    decision = select_champion(r["validation"], strict)
    assert decision.champion_run_id is None and "no run passes" in decision.rationale
    missing = SelectionPolicy("ndcg@3", guardrails=(Guardrail("does_not_exist", ">=", 0),))
    assert select_champion(r["validation"], missing).champion_run_id is None


def test_pareto_front_and_mixed_protocol_refused(runs, tiny_ds, tiny_split, tiny_ref):
    front = pareto_front(
        [{"a": 1.0, "b": 0.1}, {"a": 0.5, "b": 0.9}, {"a": 0.4, "b": 0.05}],
        ["a", "b"],
        {"a": True, "b": True},
    )
    assert front == [True, True, False]
    _, r = runs
    rows = build_leaderboard(r["validation"], POLICY)
    assert rows[0].passes_guardrails and rows[0].protocol == "full"
    sampled, _ = run_offline_experiment(
        experiment_id="lb",
        model=PopularityRecommender(),
        dataset=tiny_ds,
        dataset_ref=tiny_ref,
        split=tiny_split,
        label_policy=LabelPolicy(),
        ks=(3,),
        partition="validation",
        protocol=EvaluationProtocol("sampled", 2),
    )
    with pytest.raises(MixedEvaluationModeError):
        build_leaderboard([*r["validation"], sampled], POLICY)
