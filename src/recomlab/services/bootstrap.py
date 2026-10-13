"""Build services and registries from :class:`Settings` (shared by API and dashboard)."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from recomlab.config.settings import Settings
from recomlab.datasets.adapters import FixtureAdapter
from recomlab.datasets.dataset import CanonicalDataset
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


def registry_dir(settings: Settings) -> Path:
    return settings.paths.root / settings.paths.artifacts_dir / "registry"


def load_dataset(settings: Settings) -> CanonicalDataset:
    return FixtureAdapter(settings.dataset.dataset_id).load(dataset_dir(settings))


def build_service(settings: Settings) -> RecommendationService:
    """Serving context: full dataset history up to its last event."""
    data = load_dataset(settings)
    history = list(data.interactions)
    if not history:
        raise ValueError("dataset has no interactions")
    as_of = max(e.event_time for e in history) + timedelta(seconds=1)
    return RecommendationService(ModelRegistry(registry_dir(settings)), data.items, history, as_of)


def build_experiments(settings: Settings) -> ExperimentRegistry:
    return ExperimentRegistry(sqlite_path(settings.db_url))
