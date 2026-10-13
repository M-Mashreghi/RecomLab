"""Deterministic end-to-end regression: data -> retrieval -> ranking -> evaluation -> serving.

No live external API is used; the explanation path runs with a mocked provider.
"""

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest

from recomlab.datasets import FixtureAdapter
from recomlab.datasets.labels import LabelPolicy
from recomlab.datasets.splits import SplitConfig, assert_no_leakage, make_split
from recomlab.evaluation.report import build_report, write_report
from recomlab.experiments import ExperimentRegistry
from recomlab.experiments.runner import run_offline_experiment
from recomlab.explanations.conditions import ExplanationRouter
from recomlab.explanations.evidence import build_evidence, validate_payload
from recomlab.explanations.fidelity import validate_explanation
from recomlab.explanations.llm import MockProvider, StructuredLLMExplainer
from recomlab.ingestion.cleaning import clean_interactions
from recomlab.ingestion.csv_io import load_items, load_users, read_rows
from recomlab.models import PopularityRecommender
from recomlab.models.als import ALSParams, ImplicitALS
from recomlab.registry.model_registry import ModelRegistry
from recomlab.services.pipeline import PipelineRecommender
from recomlab.services.recommendation import RecommendationRequest, RecommendationService

pytestmark = pytest.mark.e2e
ALS = {"factors": 4, "iterations": 8, "seed": 0}
SPEC = {
    "generators": ["popularity", "item_item", "collaborative"],
    "ranker": {"type": "heuristic"},
    "candidate_budget": 8,
    "artifacts": {"als": "als"},
}


def run_pipeline(tiny_dir: Path, work: Path) -> dict:
    # 1. ingest + clean
    users = load_users(tiny_dir / "users.csv")
    items = load_items(tiny_dir / "items.csv")
    cleaned = clean_interactions(list(read_rows(tiny_dir / "interactions.csv")), users, items)
    assert cleaned.report.reconciles()
    ds = FixtureAdapter().load(tiny_dir)
    # 2. split
    split = make_split(cleaned.interactions, SplitConfig())
    assert_no_leakage(split)
    ref = {"dataset_id": "tiny", "version": "1", "checksum": cleaned.checksum}
    # 3. offline evaluation of baseline and two-stage pipeline
    reg = ExperimentRegistry(work / "runs.db")
    art = work / "artifacts"
    results = {}
    for n, model in enumerate((PopularityRecommender(), PipelineRecommender(SPEC, ALS))):
        manifest, res = run_offline_experiment(
            experiment_id="e2e",
            model=model,
            dataset=ds,
            dataset_ref=ref,
            split=split,
            label_policy=LabelPolicy(),
            ks=(3,),
            partition="test",
            registry=reg,
            artifacts_dir=art,
            created_at=f"2024-05-01T00:00:0{n}+00:00",
        )
        results[model.name] = (manifest, res)
    report = build_report(
        reg,
        [m.run_id for m, _ in results.values()],
        art,
        primary_metric="ndcg@3",
        baseline_run_id=results["popularity"][0].run_id,
        n_resamples=200,
    )
    write_report(report, work / "reports")
    # 4. serving with a registered bundle trained on visible history
    history = split.train_until("test")
    as_of = datetime(2024, 1, 14, tzinfo=UTC)
    ImplicitALS(ALSParams(**ALS)).fit(history, ds.items).save(work / "als-src")
    models = ModelRegistry(work / "registry")
    bundle = models.register_bundle(SPEC, {"als": work / "als-src"}, {"catalog_size": 12})
    collab_bundle = models.register_bundle(
        {**SPEC, "generators": ["item_item"], "ranker": {"type": "collaborative"}},
        {"als": work / "als-src"},
        {"catalog_size": 12},
    )
    svc = RecommendationService(models, ds.items, history, as_of)
    svc.activate(bundle.bundle_id)
    warm = svc.recommend(RecommendationRequest("u1", 3, trace_id="warm"))
    cold = svc.recommend(RecommendationRequest("u7", 3, trace_id="cold"))
    svc.activate(collab_bundle.bundle_id)
    empty = svc.recommend(RecommendationRequest("u7", 3, trace_id="empty"))
    # 5. explanation with mocked provider + fidelity
    item = warm.items[0]
    payload = build_evidence(
        "u1",
        item.item_id,
        history=history,
        items=ds.items,
        sources=item.sources,
        contributions=item.contributions,
        as_of=as_of,
    )
    validate_payload(payload, history, ds.items)
    cite = sorted(payload.ids)[:1]
    provider = MockProvider(
        [
            json.dumps(
                {
                    "explanation": "Matches your recent activity.",
                    "cited_evidence_ids": cite,
                    "abstain": False,
                    "uncertainty": "low",
                }
            )
        ]
    )
    explanation = ExplanationRouter(StructuredLLMExplainer(provider)).explain(
        payload, "contextual_llm-v1"
    )
    fidelity = validate_explanation(explanation, payload, ds.items)
    return {
        "split": split,
        "history": history,
        "metrics": {
            k: {m: v for m, v in r[0].metrics.items() if m != "latency_ms"}
            for k, r in results.items()
        },
        "warm": warm,
        "cold": cold,
        "empty": empty,
        "explanation": explanation,
        "fidelity": fidelity,
        "report": report,
    }


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    tiny = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "tiny"
    first = run_pipeline(tiny, tmp_path_factory.mktemp("run1"))
    second_dir = tmp_path_factory.mktemp("run2")
    second = run_pipeline(tiny, second_dir)
    return first, second, second_dir


def test_metrics_reproducible_across_runs(e2e):
    first, second, _ = e2e
    assert first["metrics"] == second["metrics"]
    assert first["warm"].item_ids == second["warm"].item_ids


def test_no_test_leakage(e2e):
    first, _, _ = e2e
    split = first["split"]
    test_ids = {e.event_id for e in split.test}
    assert not test_ids & {e.event_id for e in first["history"]}
    train_counts = Counter(e.item_id for e in split.train_until("test"))
    full_counts = Counter(e.item_id for e in (*split.train, *split.validation, *split.test))
    assert train_counts != full_counts  # held-out events never reached serving popularity


def test_warm_cold_empty_and_fallback_paths(e2e):
    first, _, _ = e2e
    warm, cold, empty = first["warm"], first["cold"], first["empty"]
    assert len(warm.items) == 3 and any(i.sources for i in warm.items)
    assert len(cold.items) == 3  # cold user served via popularity candidates / fallback
    # item_item-only retrieval gives a cold user no candidates: explicit cold-start fallback
    assert empty.metadata["n_candidates"] == 0
    assert len(empty.items) == 3 and all(i.fallback == "popularity" for i in empty.items)


def test_explanation_path_with_mocked_provider(e2e):
    first, _, _ = e2e
    exp = first["explanation"]
    assert exp.generator == "llm:mock" and exp.prompt_version
    assert first["fidelity"].accepted


def test_regression_artifacts_bounded_and_deterministic(e2e):
    first, second, second_dir = e2e
    files = [p for p in second_dir.rglob("*") if p.is_file()]
    assert len(files) < 60
    assert sum(p.stat().st_size for p in files) < 5_000_000
    md = (second_dir / "reports" / "evaluation_report.md").read_text(encoding="utf-8")
    assert "SIMULATED" in md
    assert first["report"]["runs"][0]["metrics"] == second["report"]["runs"][0]["metrics"]
