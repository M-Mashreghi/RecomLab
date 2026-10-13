"""Run an offline experiment end to end and record it in the registry."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from recomlab.datasets.dataset import CanonicalDataset
from recomlab.datasets.labels import LabelPolicy
from recomlab.datasets.splits import Split
from recomlab.domain.eligibility import EligibilityPolicy
from recomlab.evaluation.metrics import MetricConfig
from recomlab.evaluation.offline import EvaluationResult, evaluate_recommender
from recomlab.experiments.manifest import RunManifest, collect_software_state, git_commit
from recomlab.experiments.registry import ExperimentRegistry
from recomlab.models.base import Recommender


def prediction_time(split: Split, partition: str) -> Any:
    held = split.validation if partition == "validation" else split.test
    return min((e.event_time for e in held), default=None)


def run_offline_experiment(
    *,
    experiment_id: str,
    model: Recommender,
    dataset: CanonicalDataset,
    dataset_ref: dict[str, Any],
    split: Split,
    label_policy: LabelPolicy,
    ks: Sequence[int] = (5, 10),
    partition: str = "test",
    seeds: dict[str, int] | None = None,
    eligibility_policy: EligibilityPolicy | None = None,
    registry: ExperimentRegistry | None = None,
    created_at: str | None = None,
    features: dict[str, Any] | None = None,
) -> tuple[RunManifest, EvaluationResult]:
    """Fit on history visible for ``partition``, evaluate, and log a manifest."""
    history = split.train_until(partition)
    as_of = None if split.is_diagnostic else prediction_time(split, partition)
    model.fit(history, dataset.items, as_of if split.config.protocol == "global_time" else None)
    result = evaluate_recommender(
        model,
        split,
        dataset.items,
        label_policy,
        ks,
        partition=partition,
        eligibility_policy=eligibility_policy,
    )
    kwargs: dict[str, Any] = {}
    if created_at is not None:
        kwargs["created_at"] = created_at
    manifest = RunManifest(
        experiment_id=experiment_id,
        dataset=dataset_ref,
        split={
            "split_id": split.split_id,
            "protocol": split.config.protocol,
            "partition": partition,
        },
        model={"name": model.name, "version": model.version, "params": model.params()},
        metric_config=MetricConfig(ks=tuple(ks)).to_dict(),
        seeds=seeds or {},
        label_policy=label_policy.to_dict(),
        features=features or {},
        evaluation={
            "mode": result.mode,
            "universe": result.universe,
            "partition": partition,
            "eligibility": (eligibility_policy or EligibilityPolicy()).to_dict(),
            "diagnostic_split": split.is_diagnostic,
        },
        metrics={**result.summary(), "latency_ms": result.latency_ms},
        software=collect_software_state(),
        git_commit=git_commit(),
        **kwargs,
    )
    if registry is not None:
        registry.log_run(manifest)
    return manifest, result
