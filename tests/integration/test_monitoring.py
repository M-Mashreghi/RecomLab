import io
import json
import logging

import pytest

from recomlab.domain.schemas import Interaction, Item
from recomlab.monitoring.observability import (
    DriftThresholds,
    JsonFormatter,
    ServingMonitor,
    check_drift,
    profile,
    psi,
    scrub,
)

pytestmark = pytest.mark.integration


def _ev(n, user, item, etype="click"):
    return Interaction(
        f"e{n}", user, item, etype, "2024-01-01T00:00:00Z", 4 if etype == "rating" else None
    )  # type: ignore[arg-type]


def test_health_and_readiness_with_monitoring(tmp_path, bundle_factory, serving_context):
    fastapi = pytest.importorskip("fastapi")  # noqa: F841
    from fastapi.testclient import TestClient

    from recomlab.api.app import AppState, create_app
    from recomlab.config.settings import SecretStr
    from recomlab.experiments import ExperimentRegistry
    from recomlab.registry.model_registry import ModelRegistry
    from recomlab.services.recommendation import RecommendationService

    items, history, as_of = serving_context
    reg = ModelRegistry(tmp_path / "reg")
    svc = RecommendationService(reg, items, history, as_of)
    client = TestClient(create_app(AppState(svc, ExperimentRegistry(":memory:"), SecretStr(""))))
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    svc.activate(bundle_factory(reg).bundle_id)
    assert client.get("/ready").status_code == 200
    for user in ("u1", "u7"):
        client.post("/recommendations", json={"user_id": user, "k": 3})
    snap = client.get("/monitoring").json()
    assert snap["n_requests"] == 2 and snap["latency_ms"]["p50"] is not None
    assert snap["distinct_items_served"] >= 3


def test_structured_logs_have_ids_and_scrub_sensitive_fields():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    log = logging.getLogger("recomlab.test.monitoring")
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False
    log.info(
        "served for jane@example.com",
        extra={
            "request_id": "r1",
            "run_id": "run-1",
            "model_id": "m1",
            "api_key": "sk-x",
            "history": ["i1", "i2"],
            "n_items": 3,
        },
    )
    line = json.loads(stream.getvalue())
    assert (line["request_id"], line["run_id"], line["model_id"]) == ("r1", "run-1", "m1")
    assert "jane@example.com" not in line["message"]
    assert line["context"]["api_key"] == "[REDACTED]" and line["context"]["history"] == "[REDACTED]"
    assert line["context"]["n_items"] == 3
    assert scrub({"nested": {"token": "abc", "ok": "contact a@b.io"}}) == {
        "nested": {"token": "[REDACTED]", "ok": "contact [EMAIL]"}
    }


def test_synthetic_drift_triggers_warning():
    items = [Item(f"i{n}") for n in range(10)]
    train_events = [_ev(n, f"u{n % 5}", f"i{n % 10}") for n in range(100)]
    base = profile(train_events, items, [{"history_length": 10.0, "affinity": 0.5}] * 5)
    same = profile(train_events, items, [{"history_length": 10.0, "affinity": 0.5}] * 5)
    assert check_drift(base, same) == []
    drifted_events = [_ev(n, f"new{n}", "i0", "purchase") for n in range(300)]
    drifted = profile(
        drifted_events, items + [Item(f"x{n}") for n in range(5)], [{"history_length": 1.0}] * 5
    )
    checks = {w["check"] for w in check_drift(base, drifted)}
    assert {
        "catalog_size_change",
        "event_volume_change",
        "event_type_mix",
        "user_activity_psi",
        "item_popularity_psi",
        "feature_schema_skew",
        "feature_mean_skew:history_length",
    } <= checks
    mon = ServingMonitor()
    for n in range(10):
        mon.record(0 if n < 5 else 4, ["popularity"], ["i1"], 500.0)
    serving_checks = {
        w["check"] for w in check_drift(base, same, DriftThresholds(), mon.snapshot())
    }
    assert serving_checks == {"empty_candidate_rate", "latency_p95_ms"}
    assert psi({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}) == pytest.approx(0.0)
