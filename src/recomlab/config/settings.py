"""Typed configuration loading.

Precedence (lowest to highest): built-in defaults, optional TOML file,
``RECOMLAB_*`` environment variables, explicit overrides. This module is the
only place in the package that reads environment variables; model, data and
evaluation code receive a :class:`Settings` object instead.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

ENVIRONMENTS = ("development", "test", "production")
EVAL_MODES = ("full", "sampled")
LLM_PROVIDERS = ("none", "mock", "openai", "anthropic")
ENV_PREFIX = "RECOMLAB_"


class ConfigError(ValueError):
    """Raised for missing mandatory settings or invalid values."""


class SecretStr:
    """String wrapper that never reveals its value in ``repr``/``str``."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value

    def __bool__(self) -> bool:
        return bool(self._value)

    def __repr__(self) -> str:
        return "SecretStr('***')" if self._value else "SecretStr('')"

    __str__ = __repr__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SecretStr) and other._value == self._value

    def __hash__(self) -> int:
        return hash(self._value)


@dataclass(frozen=True)
class PathsConfig:
    root: Path = Path(".")
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")

    @property
    def raw_dir(self) -> Path:
        return self.root / self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.root / self.data_dir / "processed"

    @property
    def fixtures_dir(self) -> Path:
        return self.root / self.data_dir / "fixtures"

    @property
    def models_dir(self) -> Path:
        return self.root / self.artifacts_dir / "models"

    @property
    def metrics_dir(self) -> Path:
        return self.root / self.artifacts_dir / "metrics"

    @property
    def reports_dir(self) -> Path:
        return self.root / self.artifacts_dir / "reports"


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str = "tiny"
    version: str | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    k_values: tuple[int, ...] = (5, 10)
    candidate_budget: int = 100
    mode: str = "full"
    sampled_negatives: int = 100
    seed: int = 42


@dataclass(frozen=True)
class ProviderConfig:
    llm_provider: str = "none"
    llm_model: str | None = None
    llm_api_key: SecretStr = field(default_factory=lambda: SecretStr(""))
    timeout_seconds: float = 20.0

    @property
    def enabled(self) -> bool:
        return self.llm_provider not in ("none",)


@dataclass(frozen=True)
class Settings:
    env: str = "development"
    db_url: str = "sqlite:///./data/recomlab.db"
    log_level: str = "INFO"
    api_admin_token: SecretStr = field(default_factory=lambda: SecretStr(""))
    paths: PathsConfig = field(default_factory=PathsConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    provider: ProviderConfig = field(default_factory=ProviderConfig)


# env var name -> (section or None, field name)
_ENV_MAP: dict[str, tuple[str | None, str]] = {
    "ENV": (None, "env"),
    "DB_URL": (None, "db_url"),
    "LOG_LEVEL": (None, "log_level"),
    "API_ADMIN_TOKEN": (None, "api_admin_token"),
    "ROOT": ("paths", "root"),
    "DATA_DIR": ("paths", "data_dir"),
    "ARTIFACTS_DIR": ("paths", "artifacts_dir"),
    "DATASET_ID": ("dataset", "dataset_id"),
    "DATASET_VERSION": ("dataset", "version"),
    "EVAL_K_VALUES": ("evaluation", "k_values"),
    "EVAL_CANDIDATE_BUDGET": ("evaluation", "candidate_budget"),
    "EVAL_MODE": ("evaluation", "mode"),
    "EVAL_SAMPLED_NEGATIVES": ("evaluation", "sampled_negatives"),
    "EVAL_SEED": ("evaluation", "seed"),
    "LLM_PROVIDER": ("provider", "llm_provider"),
    "LLM_MODEL": ("provider", "llm_model"),
    "LLM_API_KEY": ("provider", "llm_api_key"),
    "LLM_TIMEOUT_SECONDS": ("provider", "timeout_seconds"),
}

_SECTIONS: dict[str, type] = {
    "paths": PathsConfig,
    "dataset": DatasetConfig,
    "evaluation": EvaluationConfig,
    "provider": ProviderConfig,
}


def _coerce(section: str | None, name: str, value: Any) -> Any:
    try:
        if name in {"root", "data_dir", "artifacts_dir"}:
            return Path(value)
        if name == "k_values":
            if isinstance(value, str):
                value = [v for v in value.split(",") if v.strip()]
            return tuple(int(v) for v in value)
        if name in {"candidate_budget", "sampled_negatives", "seed"}:
            return int(value)
        if name == "timeout_seconds":
            return float(value)
        if name in {"llm_api_key", "api_admin_token"}:
            return value if isinstance(value, SecretStr) else SecretStr(str(value))
        if name in {"version", "llm_model"}:
            return value or None
        return str(value)
    except (TypeError, ValueError) as exc:
        where = f"{section}.{name}" if section else name
        raise ConfigError(f"invalid value for {where}: {value!r}") from exc


def _apply(settings: Settings, updates: Mapping[str, Any]) -> Settings:
    top: dict[str, Any] = {}
    nested: dict[str, dict[str, Any]] = {}
    top_fields = {f.name for f in fields(Settings)}
    for key, value in updates.items():
        if key in _SECTIONS:
            if not isinstance(value, Mapping):
                raise ConfigError(f"section {key!r} must be a mapping")
            allowed = {f.name for f in fields(_SECTIONS[key])}
            for sub, sub_val in value.items():
                if sub not in allowed:
                    raise ConfigError(f"unknown setting {key}.{sub}")
                nested.setdefault(key, {})[sub] = _coerce(key, sub, sub_val)
        elif key in top_fields:
            top[key] = _coerce(None, key, value)
        else:
            raise ConfigError(f"unknown setting {key!r}")
    for section, values in nested.items():
        top[section] = replace(getattr(settings, section), **values)
    return replace(settings, **top)


def _env_updates(environ: Mapping[str, str]) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    for suffix, (section, name) in _ENV_MAP.items():
        key = ENV_PREFIX + suffix
        if key in environ:
            if section is None:
                updates[name] = environ[key]
            else:
                updates.setdefault(section, {})[name] = environ[key]
    return updates


def validate_settings(settings: Settings) -> Settings:
    if settings.env not in ENVIRONMENTS:
        raise ConfigError(f"env must be one of {ENVIRONMENTS}, got {settings.env!r}")
    if not settings.db_url:
        raise ConfigError("db_url is mandatory")
    ev = settings.evaluation
    if not ev.k_values or any(k <= 0 for k in ev.k_values):
        raise ConfigError("evaluation.k_values must be positive integers")
    if ev.candidate_budget <= 0:
        raise ConfigError("evaluation.candidate_budget must be positive")
    if ev.mode not in EVAL_MODES:
        raise ConfigError(f"evaluation.mode must be one of {EVAL_MODES}")
    if ev.sampled_negatives <= 0:
        raise ConfigError("evaluation.sampled_negatives must be positive")
    pv = settings.provider
    if pv.llm_provider not in LLM_PROVIDERS:
        raise ConfigError(f"provider.llm_provider must be one of {LLM_PROVIDERS}")
    if pv.llm_provider in ("openai", "anthropic"):
        if not pv.llm_api_key:
            raise ConfigError(f"provider.llm_api_key is mandatory for {pv.llm_provider}")
        if not pv.llm_model:
            raise ConfigError(f"provider.llm_model is mandatory for {pv.llm_provider}")
    if pv.timeout_seconds <= 0:
        raise ConfigError("provider.timeout_seconds must be positive")
    if not settings.dataset.dataset_id:
        raise ConfigError("dataset.dataset_id is mandatory")
    return settings


def load_settings(
    *,
    config_file: Path | None = None,
    environ: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Settings:
    """Build validated :class:`Settings` from defaults, file, environment and overrides."""
    settings = Settings()
    if config_file is not None:
        with Path(config_file).open("rb") as fh:
            settings = _apply(settings, tomllib.load(fh))
    env = os.environ if environ is None else environ
    settings = _apply(settings, _env_updates(env))
    if overrides:
        settings = _apply(settings, overrides)
    return validate_settings(settings)
