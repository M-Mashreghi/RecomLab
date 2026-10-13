"""Demo stack checks that run without Docker: image config + bootstrapped services."""

import json
import re
import shutil
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def test_dockerfile_is_slim_non_root_with_healthcheck_and_no_secrets():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert text.count("FROM python:3.12-slim") == 2  # multi-stage slim
    assert re.search(r"^USER recomlab$", text, re.M)
    assert "HEALTHCHECK" in text and "/health" in text
    assert "COPY .env" not in text and "COPY . " not in text
    assert not re.search(r"(API_KEY|TOKEN|SECRET)\s*=", text)
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").split()
    assert ".env" in ignore and "artifacts" in ignore and "data/raw" in ignore


def test_compose_mounts_data_and_has_healthchecks():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    assert set(services) == {"api", "dashboard"}
    for svc in services.values():
        assert "healthcheck" in svc
        assert "./artifacts:/app/artifacts" in svc["volumes"]
        assert all(
            not str(e).startswith("RECOMLAB_LLM_API_KEY=") for e in svc.get("environment", [])
        )
    assert services["dashboard"]["depends_on"]["api"]["condition"] == "service_healthy"


def test_bootstrapped_demo_serves_health_and_dashboard(tmp_path, tiny_dir, monkeypatch, capsys):
    pytest.importorskip("fastapi")
    pytest.importorskip("streamlit")
    from fastapi.testclient import TestClient
    from streamlit.testing.v1 import AppTest

    from recomlab.api.main import build_app
    from recomlab.cli import main

    root = tmp_path / "app"
    shutil.copytree(tiny_dir, root / "data" / "fixtures" / "tiny")
    monkeypatch.setenv("RECOMLAB_ROOT", str(root))
    monkeypatch.setenv("RECOMLAB_DB_URL", f"sqlite:///{root / 'artifacts' / 'recomlab.db'}")
    monkeypatch.setenv("RECOMLAB_EVAL_K_VALUES", "3")
    spec = str(ROOT / "configs" / "demo_bundle.json")
    assert main(["bootstrap-demo", "--spec", spec]) == 0
    active = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["active"]
    client = TestClient(build_app())
    assert client.get("/health").status_code == 200
    assert client.get("/ready").json()["bundle_id"] == active
    assert client.post("/recommendations", json={"user_id": "u1", "k": 3}).status_code == 200
    at = AppTest.from_file(
        str(ROOT / "src" / "recomlab" / "dashboard" / "app.py"), default_timeout=60
    ).run()
    assert not at.exception and at.title[0].value == "RecomLab"
    assert main(["bootstrap-demo", "--spec", spec]) == 0  # idempotent: keeps deployment
    assert "already deployed" in capsys.readouterr().out
