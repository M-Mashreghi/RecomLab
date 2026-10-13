"""Dataset registry, manifests, checksums and data cards.

Raw snapshots are immutable inputs identified by a checksum. Processed data
is written to ``<processed_dir>/<dataset_id>/<version>/`` together with a
machine-readable ``manifest.json`` and a human-readable ``DATA_CARD.md``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from recomlab.datasets.dataset import CanonicalDataset

MANIFEST_VERSION = "1.0"


class DatasetRegistryError(RuntimeError):
    pass


class DatasetChecksumMismatch(DatasetRegistryError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_checksums(directory: Path, files: list[str]) -> tuple[dict[str, str], str]:
    """Return per-file sha256 and a combined checksum over ``name:hash`` lines."""
    per_file = {name: sha256_file(Path(directory) / name) for name in sorted(files)}
    combined = hashlib.sha256(
        "\n".join(f"{n}:{h}" for n, h in per_file.items()).encode("utf-8")
    ).hexdigest()
    return per_file, combined


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    version: str
    source: str
    schema_version: str
    license_note: str
    retrieval_date: str
    files: dict[str, str]
    checksum: str
    synthetic: bool = False
    description: str = ""
    cleaning_policy: str | None = None
    label_policy: str | None = None
    stats: dict[str, int] = field(default_factory=dict)
    manifest_version: str = MANIFEST_VERSION

    def __post_init__(self) -> None:
        missing = [
            k
            for k in ("dataset_id", "version", "source", "schema_version", "license_note")
            if not getattr(self, k)
        ]
        if missing:
            raise DatasetRegistryError(f"manifest missing required fields: {missing}")
        if not self.files or not self.checksum:
            raise DatasetRegistryError("manifest requires file checksums")

    @property
    def key(self) -> str:
        return f"{self.dataset_id}@{self.version}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DatasetManifest:
        return cls(**raw)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def verify_snapshot(manifest: DatasetManifest, directory: Path) -> None:
    """Raise :class:`DatasetChecksumMismatch` if files differ from the manifest."""
    for name in manifest.files:
        if not (Path(directory) / name).exists():
            raise DatasetChecksumMismatch(f"{manifest.key}: missing file {name}")
    per_file, combined = compute_checksums(directory, list(manifest.files))
    if combined != manifest.checksum:
        changed = sorted(n for n, h in per_file.items() if manifest.files.get(n) != h)
        raise DatasetChecksumMismatch(f"{manifest.key}: checksum mismatch in {changed}")


class DatasetRegistry:
    """JSON-file backed registry of dataset manifests."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.index_path = self.root / "datasets.json"

    def _read(self) -> dict[str, dict[str, Any]]:
        if not self.index_path.exists():
            return {}
        data: dict[str, dict[str, Any]] = json.loads(self.index_path.read_text("utf-8"))
        return data

    def _write(self, data: dict[str, dict[str, Any]]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        _write_text(self.index_path, json.dumps(data, indent=2, sort_keys=True) + "\n")

    def register(self, manifest: DatasetManifest) -> DatasetManifest:
        data = self._read()
        existing = data.get(manifest.key)
        if existing is not None:
            if existing["checksum"] != manifest.checksum:
                raise DatasetChecksumMismatch(
                    f"{manifest.key} already registered with a different checksum; "
                    "bump the version instead of overwriting"
                )
            return DatasetManifest.from_dict(existing)
        data[manifest.key] = manifest.to_dict()
        self._write(data)
        return manifest

    def list(self, dataset_id: str | None = None) -> list[DatasetManifest]:
        out = [DatasetManifest.from_dict(v) for _, v in sorted(self._read().items())]
        return [m for m in out if dataset_id is None or m.dataset_id == dataset_id]

    def resolve(self, dataset_id: str, version: str | None = None) -> DatasetManifest:
        candidates = self.list(dataset_id)
        if version is not None:
            candidates = [m for m in candidates if m.version == version]
        if not candidates:
            raise DatasetRegistryError(f"dataset not registered: {dataset_id}@{version or '*'}")
        if version is None and len(candidates) > 1:
            # Deterministic: highest version (natural order); runs should pin versions.
            candidates.sort(key=lambda m: _version_key(m.version))
        return candidates[-1]


def _version_key(version: str) -> tuple[tuple[int, int | str], ...]:
    parts = re.split(r"[.\-_]", version)
    return tuple((0, int(p)) if p.isdigit() else (1, p) for p in parts)


def _write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def render_data_card(manifest: DatasetManifest) -> str:
    lines = [
        f"# Data Card: {manifest.dataset_id} (version {manifest.version})",
        "",
        manifest.description or "",
        "",
        f"- **Synthetic / simulated:** {'yes' if manifest.synthetic else 'no'}",
        f"- **Source:** {manifest.source}",
        f"- **Retrieval date:** {manifest.retrieval_date}",
        f"- **License / terms:** {manifest.license_note}",
        f"- **Schema version:** {manifest.schema_version}",
        f"- **Checksum (sha256):** `{manifest.checksum}`",
        f"- **Cleaning policy:** {manifest.cleaning_policy or 'not applied'}",
        f"- **Label policy:** {manifest.label_policy or 'not applied'}",
        "",
        "## Statistics",
        "",
    ]
    lines += [f"- {k}: {v}" for k, v in sorted(manifest.stats.items())] or ["- n/a"]
    lines += ["", "## Files", ""]
    lines += [f"- `{n}`: `{h}`" for n, h in sorted(manifest.files.items())]
    return "\n".join(lines).replace("\n\n\n", "\n\n") + "\n"


def materialize(
    dataset: CanonicalDataset,
    manifest_template: dict[str, Any],
    processed_dir: Path,
) -> DatasetManifest:
    """Write processed canonical CSVs, manifest and data card; return the manifest.

    ``manifest_template`` holds descriptive fields (id, version, source, ...);
    checksums and stats are computed from the written files.
    """
    out = Path(processed_dir) / manifest_template["dataset_id"] / manifest_template["version"]
    paths = dataset.write_csv(out)
    files, checksum = compute_checksums(out, [p.name for p in paths])
    manifest = DatasetManifest(
        **{**manifest_template, "files": files, "checksum": checksum, "stats": dataset.stats()}
    )
    _write_text(out / "manifest.json", manifest.to_json())
    _write_text(out / "DATA_CARD.md", render_data_card(manifest))
    return manifest
