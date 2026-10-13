"""ASGI entry point: ``uvicorn recomlab.api.main:app``.

Builds the service from :class:`Settings`: the configured dataset (canonical CSV
directory under ``data/processed/<id>/<version>`` or the in-repo fixture), the
model registry under ``artifacts/registry`` and the experiment registry from
``RECOMLAB_DB_URL``.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from fastapi import FastAPI

from recomlab.api.app import AppState, create_app
from recomlab.config.settings import Settings, load_settings
from recomlab.datasets.adapters import FixtureAdapter
from recomlab.experiments.registry import ExperimentRegistry
from recomlab.registry.model_registry import ModelRegistry
from recomlab.services.recommendation import RecommendationService


def sqlite_path(db_url: str) -> str:
    prefix = "sqlite:///"
    if not db_url.startswith(prefix):
        raise ValueError("only sqlite:/// URLs are supported in v1")
    return db_url[len(prefix) :]


def dataset_dir(settings: Settings) -> Path:
    ds = settings.dataset
    processed = settings.paths.processed_dir / ds.dataset_id / (ds.version or "1")
    if (processed / "interactions.csv").exists():
        return processed
    return settings.paths.fixtures_dir / ds.dataset_id


def build_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    data = FixtureAdapter(settings.dataset.dataset_id).load(dataset_dir(settings))
    history = list(data.interactions)
    as_of = (max(e.event_time for e in history) if history else None) or None
    if as_of is None:
        raise ValueError("dataset has no interactions")
    registry = ModelRegistry(settings.paths.root / settings.paths.artifacts_dir / "registry")
    service = RecommendationService(registry, data.items, history, as_of + timedelta(seconds=1))
    experiments = ExperimentRegistry(sqlite_path(settings.db_url))
    return create_app(AppState(service, experiments, settings.api_admin_token))


def __getattr__(name: str) -> FastAPI:  # lazy module attribute for uvicorn
    if name == "app":
        return build_app()
    raise AttributeError(name)
