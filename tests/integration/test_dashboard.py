import shutil
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from recomlab.dashboard import views  # noqa: E402
from recomlab.datasets.labels import LabelPolicy  # noqa: E402
from recomlab.experiments import ExperimentRegistry  # noqa: E402
from recomlab.experiments.runner import run_offline_experiment  # noqa: E402
from recomlab.models import PopularityRecommender  # noqa: E402
from recomlab.registry.model_registry import ModelRegistry  # noqa: E402
from recomlab.services.recommendation import RecommendationService  # noqa: E402

pytestmark = pytest.mark.integration
SCRIPT = str(Path(__file__).resolve().parents[2] / "src" / "recomlab" / "dashboard" / "app.py")


@pytest.fixture
def env(tmp_path, tiny_dir, tiny_ds, tiny_split, tiny_ref, bundle_factory, monkeypatch):
    root = tmp_path / "proj"
    shutil.copytree(tiny_dir, root / "data" / "fixtures" / "tiny")
    db = tmp_path / "runs.db"
    exps = ExperimentRegistry(db)
    run_offline_experiment(
        experiment_id="dash",
        model=PopularityRecommender(),
        dataset=tiny_ds,
        dataset_ref=tiny_ref,
        split=tiny_split,
        label_policy=LabelPolicy(),
        ks=(3,),
        registry=exps,
    )
    reg = ModelRegistry(root / "artifacts" / "registry")
    bundle = bundle_factory(reg)
    monkeypatch.setenv("RECOMLAB_ROOT", str(root))
    monkeypatch.setenv("RECOMLAB_DB_URL", f"sqlite:///{db}")
    return root, exps, reg, bundle


def _service(env):
    from recomlab.config import load_settings
    from recomlab.services.bootstrap import build_service

    return build_service(load_settings())


def test_views_use_registry_and_service_only(env):
    _, exps, reg, bundle = env
    svc = _service(env)
    svc.activate(bundle.bundle_id)
    rows = views.leaderboard_rows(exps)
    assert rows and "ndcg@3" in rows[0]
    detail = views.run_detail(exps, rows[0]["run_id"])
    assert set(detail) >= {"candidate_stage", "ranking_stage", "evaluation", "warnings"}
    users = views.demo_user_ids(svc)
    assert users and all(u.startswith("u") for u in users)
    rec = views.explorer(svc, users[0], 3)
    assert len(rec["items"]) == 3 and rec["bundle_id"] == bundle.bundle_id
    cold = views.cold_start_view(svc, 3)
    assert len(cold["cold_user_recommendations"]) == 3 and "i11" in cold["cold_items"]
    assert views.beyond_accuracy_view(exps)[0]["catalog_coverage@3"] > 0
    assert isinstance(svc, RecommendationService)


def test_no_pii_in_views(env):
    _, exps, _, bundle = env
    svc = _service(env)
    svc.activate(bundle.bundle_id)
    text = repr(views.explorer(svc, "u1", 5)) + repr(views.leaderboard_rows(exps))
    assert "signup_cohort" not in text and "2023Q4" not in text


@pytest.mark.parametrize(
    "page",
    ["Leaderboard", "Run details", "Recommendation explorer", "Cold start & beyond-accuracy"],
)
def test_critical_pages_load_with_fixture_data(env, page):
    _, _, reg, bundle = env
    from datetime import UTC, datetime

    from recomlab.config import load_settings
    from recomlab.services.bootstrap import load_dataset

    data = load_dataset(load_settings())
    reg.activate(bundle.bundle_id, data.items, data.interactions, datetime(2024, 2, 1, tzinfo=UTC))
    at = AppTest.from_file(SCRIPT, default_timeout=60).run()
    assert not at.exception
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception
    assert at.header[0].value


def test_dashboard_has_no_training_or_metric_logic():
    src = Path(SCRIPT).parent
    text = (src / "app.py").read_text(encoding="utf-8") + (src / "views.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (".fit(", "ImplicitALS", "train_ltr", "evaluate_recommender", "ndcg_at_k"):
        assert forbidden not in text, forbidden
