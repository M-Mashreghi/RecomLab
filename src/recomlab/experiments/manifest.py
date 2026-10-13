"""Run manifests: everything needed to reproduce a reported result."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = "1.0"


class ManifestValidationError(ValueError):
    pass


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def git_commit(cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def collect_software_state(extra_packages: tuple[str, ...] = ()) -> dict[str, Any]:
    packages = {p: _version(p) for p in ("recomlab", "numpy", *extra_packages)}
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }


@dataclass(frozen=True)
class RunManifest:
    """Reproducibility manifest for one experiment run.

    ``config`` fields (dataset, split, features, labels, model, metrics, seeds,
    evaluation) define the run identity via :func:`config_hash`; ``metrics``,
    ``software``, ``git_commit`` and ``created_at`` describe the execution.
    """

    experiment_id: str
    dataset: dict[str, Any]
    split: dict[str, Any]
    model: dict[str, Any]
    metric_config: dict[str, Any]
    seeds: dict[str, int]
    features: dict[str, Any] = field(default_factory=dict)
    label_policy: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    software: dict[str, Any] = field(default_factory=dict)
    git_commit: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: str = "completed"
    notes: str = ""
    schema_version: str = MANIFEST_SCHEMA_VERSION

    def config(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "dataset": self.dataset,
            "split": self.split,
            "features": self.features,
            "label_policy": self.label_policy,
            "model": self.model,
            "metric_config": self.metric_config,
            "evaluation": self.evaluation,
            "seeds": self.seeds,
        }

    @property
    def config_hash(self) -> str:
        return config_hash(self.config())

    @property
    def run_id(self) -> str:
        stamp = hashlib.sha256(f"{self.config_hash}:{self.created_at}".encode()).hexdigest()
        return f"{self.experiment_id}-{self.config_hash[:8]}-{stamp[:8]}"

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["run_id"] = self.run_id
        out["config_hash"] = self.config_hash
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> RunManifest:
        validate_manifest(raw)
        data = {k: v for k, v in raw.items() if k not in ("run_id", "config_hash")}
        manifest = cls(**data)
        if "config_hash" in raw and raw["config_hash"] != manifest.config_hash:
            raise ManifestValidationError("config_hash does not match manifest content")
        return manifest


_REQUIRED: dict[str, type | tuple[type, ...]] = {
    "experiment_id": str,
    "dataset": dict,
    "split": dict,
    "model": dict,
    "metric_config": dict,
    "seeds": dict,
    "created_at": str,
    "schema_version": str,
}
_OPTIONAL: dict[str, type | tuple[type, ...]] = {
    "features": dict,
    "label_policy": dict,
    "evaluation": dict,
    "metrics": dict,
    "artifacts": dict,
    "software": dict,
    "git_commit": (str, type(None)),
    "status": str,
    "notes": str,
    "run_id": str,
    "config_hash": str,
}
_NESTED_REQUIRED = {
    "dataset": ("dataset_id", "version"),
    "split": ("split_id",),
    "model": ("name", "version"),
}
STATUSES = ("completed", "failed", "running")


def validate_manifest(raw: Mapping[str, Any]) -> None:
    """Validate a manifest dict against the run-manifest schema."""
    for key, typ in _REQUIRED.items():
        if key not in raw:
            raise ManifestValidationError(f"missing required field {key!r}")
        if not isinstance(raw[key], typ):
            raise ManifestValidationError(f"field {key!r} has wrong type")
    for key, value in raw.items():
        if key in _REQUIRED:
            continue
        if key not in _OPTIONAL:
            raise ManifestValidationError(f"unknown field {key!r}")
        if not isinstance(value, _OPTIONAL[key]):
            raise ManifestValidationError(f"field {key!r} has wrong type")
    for section, keys in _NESTED_REQUIRED.items():
        for k in keys:
            if k not in raw[section]:
                raise ManifestValidationError(f"{section}.{k} is required")
    if not all(isinstance(v, int) for v in raw["seeds"].values()):
        raise ManifestValidationError("seeds must be integers")
    if raw.get("status", "completed") not in STATUSES:
        raise ManifestValidationError(f"status must be one of {STATUSES}")
    if raw["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ManifestValidationError(f"unsupported schema_version {raw['schema_version']!r}")
