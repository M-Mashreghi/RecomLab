"""Retraining workflow: snapshot -> validate -> train -> evaluate -> register challenger.

One ``workflow_id`` links every stage. The workflow never activates a model:
it registers a *challenger* bundle and returns it; promotion is a separate,
gated decision (``recomlab.registry.promotion``). Any failure records a
``failed`` run manifest and leaves the registry's active bundle untouched.
"""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recomlab.datasets.adapters import FixtureAdapter
from recomlab.datasets.dataset import CanonicalDataset
from recomlab.datasets.labels import LabelPolicy
from recomlab.datasets.registry import DatasetManifest, DatasetRegistry, materialize
from recomlab.datasets.splits import SplitConfig, make_split
from recomlab.domain.schemas import Interaction
from recomlab.experiments.manifest import RunManifest
from recomlab.experiments.registry import ExperimentRegistry
from recomlab.experiments.runner import run_offline_experiment
from recomlab.ingestion.cleaning import CleaningPolicy, clean_interactions
from recomlab.models.als import ALSParams, ImplicitALS
from recomlab.registry.model_registry import ModelBundle, ModelRegistry
from recomlab.services.pipeline import PipelineRecommender


class RetrainingError(RuntimeError):
    pass


@dataclass(frozen=True)
class RetrainingConfig:
    dataset_id: str
    spec: Mapping[str, Any]
    als_params: Mapping[str, Any] = field(default_factory=lambda: {"factors": 8, "iterations": 15})
    split: SplitConfig = field(default_factory=SplitConfig)
    label_policy: LabelPolicy = field(default_factory=LabelPolicy)
    ks: tuple[int, ...] = (5, 10)
    experiment_id: str = "retraining"


@dataclass(frozen=True)
class RetrainingResult:
    workflow_id: str
    status: str
    dataset: DatasetManifest | None
    validation_run: RunManifest | None
    bundle: ModelBundle | None
    error: str | None = None


def snapshot_dataset(
    base: CanonicalDataset,
    new_events: Sequence[Interaction],
    dataset_id: str,
    processed_dir: Path,
    datasets: DatasetRegistry,
) -> tuple[CanonicalDataset, DatasetManifest]:
    """Freeze base data + logged events into a cleaned, versioned snapshot."""
    rows = [e.to_dict() for e in (*base.interactions, *new_events)]
    cleaned = clean_interactions(rows, base.users, base.items, CleaningPolicy())
    snapshot = CanonicalDataset.build(base.users, base.items, cleaned.interactions)
    version = f"snap-{cleaned.checksum[:12]}"
    template = {
        "dataset_id": dataset_id,
        "version": version,
        "source": f"retraining snapshot of {dataset_id} + {len(new_events)} logged events",
        "schema_version": "1.0",
        "license_note": "Derived from the base dataset; same terms apply.",
        "retrieval_date": datetime.now(UTC).date().isoformat(),
        "synthetic": dataset_id == "tiny",
        "cleaning_policy": cleaned.report.policy_id,
        "description": f"cleaning report: {cleaned.report.to_dict()['reason_counts']}",
    }
    manifest = materialize(snapshot, template, processed_dir)
    datasets.register(manifest)
    return FixtureAdapter(dataset_id, version).load(processed_dir / dataset_id / version), manifest


def run_retraining(
    config: RetrainingConfig,
    *,
    base: CanonicalDataset,
    new_events: Sequence[Interaction],
    processed_dir: Path,
    datasets: DatasetRegistry,
    experiments: ExperimentRegistry,
    models: ModelRegistry,
    artifacts_dir: Path,
    workflow_id: str | None = None,
) -> RetrainingResult:
    started = datetime.now(UTC).isoformat()
    wf = workflow_id or "wf-" + hashlib.sha256(started.encode()).hexdigest()[:10]
    active_before = models.active_bundle_id
    dataset_manifest: DatasetManifest | None = None
    validation_run: RunManifest | None = None
    try:
        snapshot, dataset_manifest = snapshot_dataset(
            base, new_events, config.dataset_id, processed_dir, datasets
        )
        dataset_ref = {
            "dataset_id": dataset_manifest.dataset_id,
            "version": dataset_manifest.version,
            "checksum": dataset_manifest.checksum,
        }
        split = make_split(snapshot.interactions, config.split)
        validation_run, _ = run_offline_experiment(
            experiment_id=config.experiment_id,
            model=PipelineRecommender(config.spec, config.als_params),
            dataset=snapshot,
            dataset_ref=dataset_ref,
            split=split,
            label_policy=config.label_policy,
            ks=config.ks,
            partition="validation",
            seeds={"als": int(config.als_params.get("seed", 0))},
            registry=experiments,
            artifacts_dir=artifacts_dir / "runs",
            notes=f"workflow {wf}: validation of challenger spec",
        )
        # Final challenger artifacts are trained on the whole frozen snapshot.
        with tempfile.TemporaryDirectory(prefix="recomlab-retrain-") as tmp:
            art = Path(tmp) / "als"
            ImplicitALS(ALSParams(**config.als_params)).fit(
                snapshot.interactions, snapshot.items
            ).save(art)
            bundle = models.register_bundle(
                config.spec,
                {"als": art},
                {
                    "workflow_id": wf,
                    "dataset": dataset_ref,
                    "validation_run_id": validation_run.run_id,
                    "catalog_size": len(snapshot.items),
                    "als_params": dict(config.als_params),
                    "status": "challenger",
                },
            )
    except Exception as err:  # noqa: BLE001 - the workflow reports every failure
        failed = RunManifest(
            experiment_id=config.experiment_id,
            dataset={"dataset_id": config.dataset_id, "version": "n/a"},
            split={"split_id": "n/a"},
            model={"name": "pipeline", "version": "failed"},
            metric_config={},
            seeds={},
            status="failed",
            notes=f"workflow {wf} failed: {type(err).__name__}: {err}",
        )
        experiments.log_run(failed)
        if models.active_bundle_id != active_before:  # pragma: no cover - defensive
            raise RetrainingError("active bundle changed during a failed workflow") from err
        return RetrainingResult(wf, "failed", dataset_manifest, validation_run, None, str(err))
    if models.active_bundle_id != active_before:  # pragma: no cover - defensive
        raise RetrainingError("retraining must never change the active bundle")
    return RetrainingResult(wf, "challenger_registered", dataset_manifest, validation_run, bundle)
