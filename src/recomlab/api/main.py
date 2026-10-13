"""ASGI entry point: ``uvicorn recomlab.api.main:app``.

Builds the service from :class:`Settings` (see ``recomlab.services.bootstrap``).
"""

from __future__ import annotations

from fastapi import FastAPI

from recomlab.api.app import AppState, create_app
from recomlab.config.settings import Settings, load_settings
from recomlab.services.bootstrap import build_experiments, build_service, sqlite_path

__all__ = ["build_app", "sqlite_path"]


def build_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    return create_app(
        AppState(build_service(settings), build_experiments(settings), settings.api_admin_token)
    )


def __getattr__(name: str) -> FastAPI:  # lazy module attribute for uvicorn
    if name == "app":
        return build_app()
    raise AttributeError(name)
