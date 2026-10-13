import shutil

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from recomlab.api.app import AppState, create_app  # noqa: E402
from recomlab.config.settings import SecretStr  # noqa: E402
from recomlab.datasets.labels import LabelPolicy  # noqa: E402
from recomlab.experiments import ExperimentRegistry  # noqa: E402
from recomlab.experiments.runner import run_offline_experiment  # noqa: E402
from recomlab.models import PopularityRecommender  # noqa: E402
from recomlab.registry.model_registry import ModelRegistry  # noqa: E402
from recomlab.services.recommendation import RecommendationService  # noqa: E402

pytestmark = pytest.mark.integration
TOKEN = "test-admin-token-not-a-secret"


@pytest.fixture
def ctx(tmp_path, bundle_factory, serving_context, tiny_ds, tiny_split, tiny_ref):
    items, history, as_of = serving_context
    reg = ModelRegistry(tmp_path / "registry")
    b1 = bundle_factory(reg)
    b2 = bundle_factory(reg, ranker="collaborative")
    svc = RecommendationService(reg, items, history, as_of)
    exps = ExperimentRegistry(tmp_path / "runs.db")
    manifest, _ = run_offline_experiment(
        experiment_id="api",
        model=PopularityRecommender(),
        dataset=tiny_ds,
        dataset_ref=tiny_ref,
        split=tiny_split,
        label_policy=LabelPolicy(),
        ks=(3,),
        registry=exps,
    )
    client = TestClient(create_app(AppState(svc, exps, SecretStr(TOKEN))))
    return client, svc, b1, b2, manifest


def test_health_and_readiness(ctx):
    client, svc, b1, _, _ = ctx
    assert client.get("/health").json()["status"] == "ok"
    not_ready = client.get("/ready")
    assert not_ready.status_code == 503 and not_ready.json()["error"]["code"] == "not_ready"
    svc.activate(b1.bundle_id)
    assert client.get("/ready").json() == {"status": "ready", "bundle_id": b1.bundle_id}


def test_recommendation_contract(ctx):
    client, svc, b1, _, _ = ctx
    r = client.post("/recommendations", json={"user_id": "u1", "k": 3})
    assert r.status_code == 503 and r.json()["error"]["code"] == "no_active_model"
    svc.activate(b1.bundle_id)
    r = client.post("/recommendations", json={"user_id": "u1", "k": 3, "trace_id": "abc"})
    assert r.status_code == 200
    body = r.json()
    assert body["trace_id"] == "abc" and body["bundle_id"] == b1.bundle_id
    assert len(body["items"]) == 3
    assert set(body["items"][0]) == {
        "item_id",
        "rank",
        "score",
        "fallback",
        "sources",
        "contributions",
    }
    for item in body["items"]:
        for src in item["sources"].values():
            assert set(src) == {"rank"}  # no evidence / user history leaked


@pytest.mark.parametrize(
    "payload",
    [
        {"k": 3},
        {"user_id": "u1", "k": 0},
        {"user_id": "bad id", "k": 1},
        {"user_id": "u1", "k": "x"},
    ],
)
def test_validation_errors_are_structured(ctx, payload):
    client, *_ = ctx
    r = client.post("/recommendations", json=payload)
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "validation_error" and err["details"][0]["field"]


def test_experiment_endpoints(ctx):
    client, _, _, _, manifest = ctx
    runs = client.get("/experiments/runs", params={"experiment_id": "api"}).json()
    assert [r["run_id"] for r in runs] == [manifest.run_id]
    detail = client.get(f"/experiments/runs/{manifest.run_id}").json()
    assert detail["metrics"]["n_users_evaluated"] > 0
    assert set(detail["software"]) == {"packages"}  # platform details withheld
    missing = client.get("/experiments/runs/nope")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"


def test_activation_requires_token_and_fails_closed(ctx):
    client, svc, b1, b2, _ = ctx
    svc.activate(b1.bundle_id)
    assert client.post("/models/activate", json={"bundle_id": b2.bundle_id}).status_code == 401
    bad = client.post(
        "/models/activate", json={"bundle_id": b2.bundle_id}, headers={"X-Admin-Token": "wrong"}
    )
    assert bad.status_code == 401
    ok = client.post(
        "/models/activate", json={"bundle_id": b2.bundle_id}, headers={"X-Admin-Token": TOKEN}
    )
    assert ok.status_code == 200 and ok.json()["active"] == b2.bundle_id
    missing = client.post(
        "/models/activate", json={"bundle_id": "bundle-nope"}, headers={"X-Admin-Token": TOKEN}
    )
    assert missing.status_code == 404 and svc.bundle_id == b2.bundle_id
    listing = client.get("/models/bundles").json()
    assert listing["active"] == b2.bundle_id and b1.bundle_id in listing["bundles"]
    assert client.get("/models/active").json()["bundle_id"] == b2.bundle_id


def test_mutation_disabled_without_configured_token(ctx):
    _, svc, b1, _, _ = ctx
    client = TestClient(create_app(AppState(svc, ExperimentRegistry(":memory:"), SecretStr(""))))
    r = client.post(
        "/models/activate", json={"bundle_id": b1.bundle_id}, headers={"X-Admin-Token": ""}
    )
    assert r.status_code == 403 and r.json()["error"]["code"] == "mutation_disabled"


def test_no_secrets_in_responses(ctx):
    client, svc, b1, _, manifest = ctx
    svc.activate(b1.bundle_id)
    texts = [
        client.get("/models/active").text,
        client.get(f"/experiments/runs/{manifest.run_id}").text,
        client.post("/recommendations", json={"user_id": "u2", "k": 2}).text,
    ]
    assert all(TOKEN not in t for t in texts)
    assert all("api_key" not in t.lower() and "admin_token" not in t.lower() for t in texts)


def test_build_app_from_settings(tmp_path, tiny_dir):
    from recomlab.api.main import build_app, sqlite_path
    from recomlab.config import load_settings

    root = tmp_path / "proj"
    (root / "data" / "fixtures").mkdir(parents=True)
    shutil.copytree(tiny_dir, root / "data" / "fixtures" / "tiny")
    settings = load_settings(
        environ={},
        overrides={"paths": {"root": str(root)}, "db_url": f"sqlite:///{tmp_path / 'r.db'}"},
    )
    client = TestClient(build_app(settings))
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503  # no bundle activated yet
    with pytest.raises(ValueError):
        sqlite_path("postgres://x")
