"""Model bundle registry with fail-closed activation and rollback.

A bundle is a directory ``<root>/bundles/<bundle_id>/`` holding ``bundle.json``
(spec, metadata, artifact checksums) and its artifact files. The active bundle
pointer and the activation history live in ``<root>/active.json``. Activation
validates compatibility first; any failure leaves the current active bundle
untouched.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recomlab.domain.schemas import Interaction, Item
from recomlab.models.als import ALSArtifact
from recomlab.services.pipeline import PipelineConfigError, RecommendationPipeline, validate_spec

BUNDLE_SCHEMA = "bundle-1"


class IncompatibleBundleError(RuntimeError):
    pass


class BundleNotFound(KeyError):
    pass


def _sha256_dir(path: Path) -> dict[str, str]:
    out = {}
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        out[f.relative_to(path).as_posix()] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")


@dataclass(frozen=True)
class ModelBundle:
    bundle_id: str
    spec: dict[str, Any]
    metadata: dict[str, Any]
    checksums: dict[str, str]
    directory: Path

    @property
    def artifacts_dir(self) -> Path:
        return self.directory / "artifacts"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BUNDLE_SCHEMA,
            "bundle_id": self.bundle_id,
            "spec": self.spec,
            "metadata": self.metadata,
            "checksums": self.checksums,
        }


class ModelRegistry:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        (self.root / "bundles").mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "active.json"

    # -- bundles -----------------------------------------------------------
    def register_bundle(
        self,
        spec: Mapping[str, Any],
        artifact_dirs: Mapping[str, Path],
        metadata: Mapping[str, Any] | None = None,
    ) -> ModelBundle:
        """Copy artifacts into a new immutable bundle directory."""
        try:
            validate_spec(spec)
        except PipelineConfigError as err:
            raise IncompatibleBundleError(str(err)) from err
        payload = json.dumps({"spec": spec, "metadata": metadata or {}}, sort_keys=True)
        digest = hashlib.sha256(payload.encode("utf-8"))
        for name, src in sorted(artifact_dirs.items()):
            digest.update(name.encode())
            for rel, h in _sha256_dir(Path(src)).items():
                digest.update(f"{rel}:{h}".encode())
        bundle_id = f"bundle-{digest.hexdigest()[:12]}"
        directory = self.root / "bundles" / bundle_id
        if directory.exists():
            return self.get_bundle(bundle_id)
        art = directory / "artifacts"
        art.mkdir(parents=True)
        for name, src in artifact_dirs.items():
            shutil.copytree(src, art / name)
        meta = {
            **dict(metadata or {}),
            "registered_at": datetime.now(UTC).isoformat(),
        }
        bundle = ModelBundle(bundle_id, dict(spec), meta, _sha256_dir(art), directory)
        _write_json(directory / "bundle.json", bundle.to_dict())
        return bundle

    def get_bundle(self, bundle_id: str) -> ModelBundle:
        path = self.root / "bundles" / bundle_id / "bundle.json"
        if not path.exists():
            raise BundleNotFound(bundle_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        return ModelBundle(
            data["bundle_id"], data["spec"], data["metadata"], data["checksums"], path.parent
        )

    def list_bundles(self) -> list[str]:
        return sorted(p.name for p in (self.root / "bundles").iterdir() if p.is_dir())

    # -- compatibility -----------------------------------------------------
    def validate(
        self,
        bundle: ModelBundle,
        items: Sequence[Item],
        history: Sequence[Interaction],
        as_of: datetime,
    ) -> RecommendationPipeline:
        """Fail closed unless artifacts are intact and the pipeline builds and serves."""
        if bundle.metadata.get("schema", BUNDLE_SCHEMA) != BUNDLE_SCHEMA:
            raise IncompatibleBundleError("unsupported bundle schema")
        current = _sha256_dir(bundle.artifacts_dir)
        if current != bundle.checksums:
            raise IncompatibleBundleError(f"{bundle.bundle_id}: artifact files changed or missing")
        expected_items = bundle.metadata.get("catalog_size")
        if expected_items is not None and expected_items != len(items):
            raise IncompatibleBundleError(
                f"catalog size {len(items)} differs from bundle catalog {expected_items}"
            )
        try:
            if "als" in bundle.spec.get("artifacts", {}):
                ALSArtifact.load(bundle.artifacts_dir / bundle.spec["artifacts"]["als"])
            pipeline = RecommendationPipeline(
                bundle.spec, history, items, as_of, artifacts_dir=bundle.artifacts_dir
            )
            pipeline.recommend("__compatibility_probe__", 1)
        except Exception as err:  # noqa: BLE001 - any failure must block activation
            raise IncompatibleBundleError(f"{bundle.bundle_id}: {err}") from err
        return pipeline

    # -- activation --------------------------------------------------------
    def _state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"active": None, "history": []}
        state: dict[str, Any] = json.loads(self.state_path.read_text(encoding="utf-8"))
        return state

    @property
    def active_bundle_id(self) -> str | None:
        active: str | None = self._state()["active"]
        return active

    def history(self) -> list[dict[str, Any]]:
        hist: list[dict[str, Any]] = self._state()["history"]
        return hist

    def activate(
        self,
        bundle_id: str,
        items: Sequence[Item],
        history: Sequence[Interaction],
        as_of: datetime,
        reason: str = "",
    ) -> RecommendationPipeline:
        bundle = self.get_bundle(bundle_id)
        pipeline = self.validate(bundle, items, history, as_of)  # raises before any state change
        state = self._state()
        state["history"].append(
            {
                "event": "activate",
                "bundle_id": bundle_id,
                "previous": state["active"],
                "reason": reason,
                "at": datetime.now(UTC).isoformat(),
            }
        )
        state["active"] = bundle_id
        _write_json(self.state_path, state)
        return pipeline

    def rollback(
        self,
        items: Sequence[Item],
        history: Sequence[Interaction],
        as_of: datetime,
        reason: str,
    ) -> RecommendationPipeline:
        """Reactivate the previously active bundle and record why."""
        state = self._state()
        previous = next(
            (
                h["previous"]
                for h in reversed(state["history"])
                if h["bundle_id"] == state["active"] and h["event"] == "activate"
            ),
            None,
        )
        if previous is None:
            raise IncompatibleBundleError("no previous bundle to roll back to")
        pipeline = self.validate(self.get_bundle(previous), items, history, as_of)
        state["history"].append(
            {
                "event": "rollback",
                "bundle_id": previous,
                "previous": state["active"],
                "reason": reason,
                "at": datetime.now(UTC).isoformat(),
            }
        )
        state["active"] = previous
        _write_json(self.state_path, state)
        return pipeline
