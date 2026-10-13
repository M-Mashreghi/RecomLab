import pytest

from recomlab.datasets.labels import LabelPolicy
from recomlab.evaluation.protocol import EvaluationProtocol
from recomlab.evaluation.report import build_report, render_markdown, write_report
from recomlab.experiments import ExperimentRegistry
from recomlab.experiments.runner import run_offline_experiment
from recomlab.models import PopularityRecommender, RandomRecommender

pytestmark = pytest.mark.integration


@pytest.fixture
def env(tiny_ds, tiny_split, tiny_ref, tmp_path):
    reg = ExperimentRegistry(tmp_path / "runs.db")
    art = tmp_path / "artifacts"
    ids = []
    for n, model in enumerate((PopularityRecommender(), RandomRecommender(seed=2))):
        m, _ = run_offline_experiment(
            experiment_id="report",
            model=model,
            dataset=tiny_ds,
            dataset_ref=tiny_ref,
            split=tiny_split,
            label_policy=LabelPolicy(),
            ks=(3,),
            registry=reg,
            artifacts_dir=art,
            created_at=f"2024-03-01T00:00:0{n}+00:00",
        )
        ids.append(m.run_id)
    return reg, art, ids


def test_report_regenerates_from_stored_artifacts(env, tmp_path):
    reg, art, ids = env
    r1 = build_report(
        reg, ids, art, primary_metric="ndcg@3", baseline_run_id=ids[0], n_resamples=200
    )
    reopened = ExperimentRegistry(reg.db_path)
    r2 = build_report(
        reopened, ids, art, primary_metric="ndcg@3", baseline_run_id=ids[0], n_resamples=200
    )
    assert r1 == r2
    md1, js1 = write_report(r1, tmp_path / "out1")
    md2, js2 = write_report(r2, tmp_path / "out2")
    assert md1.read_text() == md2.read_text() and js1.read_text() == js2.read_text()


def test_report_contents_and_limitations(env):
    reg, art, ids = env
    report = build_report(
        reg, ids, art, primary_metric="ndcg@3", baseline_run_id=ids[0], n_resamples=200
    )
    md = render_markdown(report)
    for section in (
        "## Protocol",
        "## Ranking-Stage Results",
        "## Candidate-Stage (Retrieval) Diagnostics",
        "## Paired Comparisons",
        "## Slice Reports",
        "## Limitations",
    ):
        assert section in md
    assert "SIMULATED" in md
    assert "Exposure-aware (off-policy) correction unavailable" in md
    assert all(r["primary_ci"] for r in report["runs"])
    assert report["comparisons"][0]["baseline"] == ids[0]
    assert report["metric_definitions"]["ks"] == [3]
    assert report["runs"][0]["split_id"] and report["runs"][0]["label_version"]


def test_sampled_warning_in_report(tiny_ds, tiny_split, tiny_ref, tmp_path):
    reg = ExperimentRegistry(tmp_path / "r.db")
    m, _ = run_offline_experiment(
        experiment_id="s",
        model=PopularityRecommender(),
        dataset=tiny_ds,
        dataset_ref=tiny_ref,
        split=tiny_split,
        label_policy=LabelPolicy(),
        ks=(3,),
        registry=reg,
        protocol=EvaluationProtocol("sampled", 2),
    )
    report = build_report(reg, [m.run_id], tmp_path, primary_metric="ndcg@3")
    md = render_markdown(report)
    assert "SAMPLED EVALUATION" in md
    assert any("per-user metrics not stored" in c for c in report["caveats"])
