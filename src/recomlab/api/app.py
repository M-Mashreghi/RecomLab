"""FastAPI application: thin controllers over services and registries.

Controllers validate input, call the recommendation service / registries, and
shape responses. They contain no training or metric logic. Responses never
include secrets, raw interaction histories, or per-candidate evidence.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from recomlab import __version__
from recomlab.config.settings import SecretStr
from recomlab.experiments.registry import ExperimentRegistry, RunNotFound
from recomlab.registry.model_registry import BundleNotFound, IncompatibleBundleError
from recomlab.services.recommendation import (
    NoActiveModelError,
    RecommendationRequest,
    RecommendationService,
)

ID_REGEX = r"^[A-Za-z0-9_.:\-]{1,64}$"


class RecommendationBody(BaseModel):
    user_id: str = Field(pattern=ID_REGEX)
    k: int = Field(default=10, ge=1, le=100)
    trace_id: str | None = Field(default=None, pattern=ID_REGEX)


class ActivationBody(BaseModel):
    bundle_id: str = Field(pattern=ID_REGEX)
    reason: str = Field(default="", max_length=200)


@dataclass
class AppState:
    service: RecommendationService
    experiments: ExperimentRegistry
    admin_token: SecretStr


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        self.status, self.code, self.message = status, code, message


def _error(status: int, code: str, message: str, details: Any = None) -> JSONResponse:
    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body)


def _public_manifest(data: dict[str, Any]) -> dict[str, Any]:
    """Run metadata safe to expose (no environment/platform details)."""
    software = data.get("software", {})
    return {
        **{k: v for k, v in data.items() if k != "software"},
        "software": {"packages": software.get("packages", {})},
    }


def create_app(state: AppState) -> FastAPI:
    app = FastAPI(title="RecomLab API", version=__version__)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"field": ".".join(str(p) for p in e["loc"][1:]), "issue": e["msg"]}
            for e in exc.errors()
        ]
        return _error(422, "validation_error", "request validation failed", details)

    @app.exception_handler(APIError)
    async def _api_error(_: Request, exc: APIError) -> JSONResponse:
        return _error(exc.status, exc.code, exc.message)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/ready")
    def ready() -> JSONResponse:
        if state.service.bundle_id is None:
            return _error(503, "not_ready", "no active model bundle")
        return JSONResponse({"status": "ready", "bundle_id": state.service.bundle_id})

    @app.post("/recommendations")
    def recommend(body: RecommendationBody) -> dict[str, Any]:
        try:
            resp = state.service.recommend(
                RecommendationRequest(body.user_id, body.k, body.trace_id)
            )
        except NoActiveModelError as err:
            raise APIError(503, "no_active_model", str(err)) from err
        data = resp.to_dict()
        for item in data["items"]:
            # Evidence may reference the user's own history; keep it server-side.
            item["sources"] = {s: {"rank": v.get("rank")} for s, v in item["sources"].items()}
        return data

    @app.get("/experiments/runs")
    def list_runs(experiment_id: str | None = Query(default=None, pattern=ID_REGEX)) -> Any:
        runs = state.experiments.list_runs(experiment_id)
        return [
            {
                "run_id": r.run_id,
                "experiment_id": r.experiment_id,
                "model": f"{r.model['name']}@{r.model['version']}",
                "partition": r.split.get("partition"),
                "evaluation_mode": r.evaluation.get("mode"),
                "created_at": r.created_at,
            }
            for r in runs
        ]

    @app.get("/experiments/runs/{run_id}")
    def get_run(run_id: str) -> Any:
        try:
            return _public_manifest(state.experiments.get_run(run_id).to_dict())
        except RunNotFound as err:
            raise APIError(404, "not_found", f"run {run_id} not found") from err

    @app.get("/models/bundles")
    def bundles() -> Any:
        reg = state.service.registry
        return {"bundles": reg.list_bundles(), "active": reg.active_bundle_id}

    @app.get("/models/active")
    def active() -> Any:
        reg = state.service.registry
        if reg.active_bundle_id is None:
            raise APIError(404, "not_found", "no active bundle")
        bundle = reg.get_bundle(reg.active_bundle_id)
        return {"bundle_id": bundle.bundle_id, "spec": bundle.spec, "history": reg.history()}

    @app.post("/models/activate")
    def activate(body: ActivationBody, x_admin_token: str | None = Header(default=None)) -> Any:
        expected = state.admin_token.get_secret_value()
        if not expected:
            raise APIError(403, "mutation_disabled", "model activation via API is disabled")
        if not x_admin_token or not hmac.compare_digest(x_admin_token, expected):
            raise APIError(401, "unauthorized", "missing or invalid admin token")
        try:
            state.service.activate(body.bundle_id, body.reason)
        except BundleNotFound as err:
            raise APIError(404, "not_found", f"bundle {body.bundle_id} not found") from err
        except IncompatibleBundleError as err:
            raise APIError(409, "incompatible_bundle", str(err)) from err
        return {"active": state.service.bundle_id}

    return app
