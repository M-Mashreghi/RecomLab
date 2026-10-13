"""Reproducible benchmark on a canonical dataset (default: the simulated ``tiny`` fixture).

Validation runs choose the champion (pre-registered selection policy); the test
partition is evaluated once for every model and reported with uncertainty.

Usage: python scripts/run_benchmark.py [--data DIR] [--dataset-id ID] [--out docs/reports]
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from recomlab.datasets import FixtureAdapter
from recomlab.datasets.labels import LabelPolicy
from recomlab.datasets.splits import SplitConfig, make_split
from recomlab.evaluation.leaderboard import (
    Guardrail,
    SelectionPolicy,
    attach_test_report,
    select_champion,
)
from recomlab.evaluation.report import build_report, render_markdown
from recomlab.experiments import ExperimentRegistry
from recomlab.experiments.runner import run_offline_experiment
from recomlab.models import PopularityRecommender, RandomRecommender
from recomlab.models.als import ALSParams, ALSRecommender
from recomlab.services.pipeline import PipelineRecommender

ROOT = Path(__file__).resolve().parents[1]
SPEC = {
    "generators": ["popularity", "item_item", "collaborative"],
    "ranker": {"type": "heuristic"},
    "candidate_budget": 20,
    "artifacts": {"als": "als"},
}
ALS = {"factors": 4, "iterations": 10, "seed": 0}
POLICY = SelectionPolicy(
    primary_metric="ndcg@5",
    guardrails=(Guardrail("catalog_coverage@5", ">=", 0.05),),
    pareto_metrics=("ndcg@5", "catalog_coverage@5"),
)


def models():
    return [
        RandomRecommender(seed=0),
        PopularityRecommender(),
        ALSRecommender(ALSParams(**ALS)),
        PipelineRecommender(SPEC, ALS),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "fixtures" / "tiny"))
    parser.add_argument("--dataset-id", default="tiny")
    parser.add_argument("--out", default=str(ROOT / "docs" / "reports"))
    args = parser.parse_args()
    ds = FixtureAdapter(args.dataset_id).load(Path(args.data))
    split = make_split(ds.interactions, SplitConfig())
    ref = {"dataset_id": args.dataset_id, "version": "1"}
    with tempfile.TemporaryDirectory() as tmp:
        reg = ExperimentRegistry(Path(tmp) / "runs.db")
        art = Path(tmp) / "artifacts"
        runs: dict[str, list] = {"validation": [], "test": []}
        for partition in ("validation", "test"):
            for n, model in enumerate(models()):
                manifest, _ = run_offline_experiment(
                    experiment_id="benchmark",
                    model=model,
                    dataset=ds,
                    dataset_ref=ref,
                    split=split,
                    label_policy=LabelPolicy(),
                    ks=(5, 10),
                    partition=partition,
                    seeds={"global": 0},
                    registry=reg,
                    artifacts_dir=art,
                    created_at=f"2024-06-01T00:00:{n:02d}+00:00",
                )
                runs[partition].append(manifest)
        decision = attach_test_report(select_champion(runs["validation"], POLICY), runs["test"])
        test_ids = [m.run_id for m in runs["test"]]
        report = build_report(
            reg,
            test_ids,
            art,
            primary_metric="ndcg@5",
            baseline_run_id=runs["test"][1].run_id,  # popularity baseline
            n_resamples=1000,
            title=f"RecomLab Benchmark ({args.dataset_id})",
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = render_markdown(report)
    md += (
        "\n## Model Selection (validation only)\n\n"
        f"- Policy `{decision.policy['policy_id']}`: primary `{POLICY.primary_metric}`, "
        f"guardrails {[f'{g.metric} {g.op} {g.threshold}' for g in POLICY.guardrails]}\n"
        f"- Decision: {decision.rationale}\n"
        + "".join(f"- Note: {n}\n" for n in decision.notes)
        + f"- Champion test result (reported once): {decision.test_report['model']}\n"
    )
    stem = f"benchmark_{args.dataset_id}"
    with (out / f"{stem}.md").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(md)
    with (out / f"{stem}.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(
            {"report": report, "selection": decision.to_dict()},
            fh,
            indent=2,
            sort_keys=True,
            default=str,
        )
        fh.write("\n")
    print(f"wrote {out / stem}.md")


if __name__ == "__main__":
    main()
