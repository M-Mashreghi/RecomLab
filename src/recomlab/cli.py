"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from recomlab import __version__
from recomlab.logging_utils import configure_logging


def _cmd_smoke(_: argparse.Namespace) -> int:
    logger = configure_logging()
    logger.info("recomlab smoke check ok (version %s)", __version__)
    print(f"recomlab {__version__} ok")
    return 0


def _cmd_retrain(args: argparse.Namespace) -> int:
    from recomlab.config.settings import load_settings
    from recomlab.datasets.registry import DatasetRegistry
    from recomlab.experiments.retraining import RetrainingConfig, run_retraining
    from recomlab.registry.model_registry import ModelRegistry
    from recomlab.services.bootstrap import build_experiments, load_dataset, registry_dir
    from recomlab.services.event_log import EventLog

    settings = load_settings()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    events = EventLog(args.events).as_interactions() if args.events else []
    root = settings.paths.root
    result = run_retraining(
        RetrainingConfig(
            settings.dataset.dataset_id,
            spec["pipeline"],
            spec.get("als", {}),
            ks=settings.evaluation.k_values,
        ),
        base=load_dataset(settings),
        new_events=events,
        processed_dir=settings.paths.processed_dir,
        datasets=DatasetRegistry(root / settings.paths.artifacts_dir / "datasets"),
        experiments=build_experiments(settings),
        models=ModelRegistry(registry_dir(settings)),
        artifacts_dir=root / settings.paths.artifacts_dir,
    )
    print(
        json.dumps(
            {
                "workflow_id": result.workflow_id,
                "status": result.status,
                "dataset_version": result.dataset.version if result.dataset else None,
                "validation_run_id": (
                    result.validation_run.run_id if result.validation_run else None
                ),
                "challenger_bundle": result.bundle.bundle_id if result.bundle else None,
                "error": result.error,
            },
            indent=2,
        )
    )
    return 0 if result.status != "failed" else 1


def _cmd_bootstrap_demo(args: argparse.Namespace) -> int:
    """One-command demo: train a bundle on the configured dataset and deploy it.

    The very first deployment has no champion to compare against, so the
    challenger is activated explicitly here (recorded as ``initial demo
    deployment``). Later retraining runs never auto-promote.
    """
    from datetime import timedelta

    from recomlab.config.settings import load_settings
    from recomlab.registry.model_registry import ModelRegistry
    from recomlab.services.bootstrap import load_dataset, registry_dir

    settings = load_settings()
    models = ModelRegistry(registry_dir(settings))
    if models.active_bundle_id is not None and not args.force:
        print(json.dumps({"active": models.active_bundle_id, "note": "already deployed"}))
        return 0
    code = _cmd_retrain(args)
    if code != 0:
        return code
    data = load_dataset(settings)
    newest = max(
        (models.get_bundle(b) for b in models.list_bundles()),
        key=lambda b: b.metadata.get("registered_at", ""),
    )
    as_of = max(e.event_time for e in data.interactions) + timedelta(seconds=1)
    models.activate(
        newest.bundle_id, data.items, data.interactions, as_of, reason="initial demo deployment"
    )
    print(json.dumps({"active": newest.bundle_id}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recomlab", description="RecomLab CLI")
    parser.add_argument("--version", action="version", version=f"recomlab {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke", help="verify the installation")
    smoke.set_defaults(func=_cmd_smoke)
    retrain = sub.add_parser(
        "retrain", help="snapshot data, train and evaluate a challenger (never auto-promotes)"
    )
    retrain.add_argument("--spec", required=True, help="JSON file with 'pipeline' and 'als'")
    retrain.add_argument("--events", help="optional event-log SQLite file to include")
    retrain.set_defaults(func=_cmd_retrain)
    demo = sub.add_parser("bootstrap-demo", help="train and deploy the demo bundle")
    demo.add_argument("--spec", default="configs/demo_bundle.json")
    demo.add_argument("--events", help="optional event-log SQLite file to include")
    demo.add_argument("--force", action="store_true", help="redeploy even if a bundle is active")
    demo.set_defaults(func=_cmd_bootstrap_demo)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
