import pytest

from recomlab.datasets.registry import DatasetRegistry, verify_snapshot
from recomlab.domain.exposure import FeedbackEvent
from recomlab.experiments import ExperimentRegistry
from recomlab.experiments.retraining import RetrainingConfig, run_retraining
from recomlab.registry.model_registry import ModelRegistry
from recomlab.services.event_log import EventLog

pytestmark = pytest.mark.integration

SPEC = {
    "generators": ["popularity", "item_item", "collaborative"],
    "ranker": {"type": "heuristic"},
    "candidate_budget": 8,
    "artifacts": {"als": "als"},
}
ALS = {"factors": 4, "iterations": 6, "seed": 0}


@pytest.fixture
def stores(tmp_path):
    return {
        "processed_dir": tmp_path / "processed",
        "datasets": DatasetRegistry(tmp_path / "datasets"),
        "experiments": ExperimentRegistry(tmp_path / "runs.db"),
        "models": ModelRegistry(tmp_path / "registry"),
        "artifacts_dir": tmp_path / "artifacts",
    }


def _new_events(tmp_path):
    log = EventLog(tmp_path / "events.db")
    log.log_feedback(
        [
            FeedbackEvent.from_dict(
                {
                    "feedback_id": "f1",
                    "user_id": "u6",
                    "item_id": "i02",
                    "feedback_type": "click",
                    "event_time": "2024-01-20T10:00:00Z",
                }
            ),
            FeedbackEvent.from_dict(
                {
                    "feedback_id": "f2",
                    "user_id": "u6",
                    "item_id": "i03",
                    "feedback_type": "purchase",
                    "event_time": "2024-01-21T10:00:00Z",
                }
            ),
        ]
    )
    return log.as_interactions()


def test_end_to_end_on_demo_fixture_links_versions(tiny_ds, stores, tmp_path):
    cfg = RetrainingConfig("tiny", SPEC, ALS, ks=(3,))
    result = run_retraining(
        cfg, base=tiny_ds, new_events=_new_events(tmp_path), workflow_id="wf-test", **stores
    )
    assert result.status == "challenger_registered", result.error
    ds = result.dataset
    assert ds.stats["n_interactions"] == 35  # 33 base + 2 logged
    verify_snapshot(ds, stores["processed_dir"] / "tiny" / ds.version)
    assert stores["datasets"].resolve("tiny", ds.version) == ds
    run = stores["experiments"].get_run(result.validation_run.run_id)
    assert run.dataset["version"] == ds.version and run.split["partition"] == "validation"
    assert "wf-test" in run.notes
    bundle = result.bundle
    assert bundle.metadata["dataset"]["version"] == ds.version
    assert bundle.metadata["validation_run_id"] == run.run_id
    assert bundle.metadata["workflow_id"] == "wf-test"
    assert stores["models"].active_bundle_id is None  # never auto-promoted
    assert (stores["artifacts_dir"] / "runs" / run.artifacts["per_user_metrics"]).exists()


def test_snapshot_is_deterministic(tiny_ds, stores, tmp_path):
    cfg = RetrainingConfig("tiny", SPEC, ALS, ks=(3,))
    a = run_retraining(cfg, base=tiny_ds, new_events=[], workflow_id="wf-a", **stores)
    b = run_retraining(cfg, base=tiny_ds, new_events=[], workflow_id="wf-b", **stores)
    assert a.dataset.version == b.dataset.version and a.dataset.checksum == b.dataset.checksum


def test_failure_leaves_previous_active_model_untouched(
    tiny_ds, stores, tmp_path, bundle_factory, serving_context
):
    items, history, as_of = serving_context
    models = stores["models"]
    previous = bundle_factory(models)
    models.activate(previous.bundle_id, items, history, as_of)
    broken = {**SPEC, "ranker": {"type": "heuristic", "weights": {"nonexistent_feature": 1.0}}}
    result = run_retraining(
        RetrainingConfig("tiny", broken, ALS, ks=(3,)), base=tiny_ds, new_events=[], **stores
    )
    assert result.status == "failed" and result.bundle is None and result.error
    assert models.active_bundle_id == previous.bundle_id
    failed = [r for r in stores["experiments"].list_runs("retraining") if r.status == "failed"]
    assert failed and result.workflow_id in failed[0].notes


def test_cli_retrain_command(tmp_path, tiny_dir, monkeypatch, capsys):
    import json
    import shutil
    from pathlib import Path

    from recomlab.cli import main

    root = tmp_path / "proj"
    shutil.copytree(tiny_dir, root / "data" / "fixtures" / "tiny")
    monkeypatch.setenv("RECOMLAB_ROOT", str(root))
    monkeypatch.setenv("RECOMLAB_DB_URL", f"sqlite:///{tmp_path / 'runs.db'}")
    monkeypatch.setenv("RECOMLAB_EVAL_K_VALUES", "3")
    spec = Path(__file__).resolve().parents[2] / "configs" / "demo_bundle.json"
    assert main(["retrain", "--spec", str(spec)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "challenger_registered" and out["challenger_bundle"]
    assert ModelRegistry(root / "artifacts" / "registry").active_bundle_id is None
