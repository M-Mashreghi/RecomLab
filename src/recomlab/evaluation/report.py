"""Evaluation report generated from the experiment registry and stored artifacts.

The report is a pure function of stored manifests and artifact files (no model
reruns, no notebook state): regenerating it yields identical Markdown/JSON.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from recomlab.evaluation.protocol import SAMPLED_WARNING, assert_comparable
from recomlab.evaluation.statistics import bootstrap_ci, paired_comparison
from recomlab.experiments.manifest import RunManifest
from recomlab.experiments.registry import ExperimentRegistry

STANDARD_LIMITATIONS = (
    "Offline metrics measure agreement with logged held-out interactions, which reflect "
    "what users were exposed to (exposure bias); they are not causal estimates of online impact.",
    "Uncertainty intervals resample users and do not cover dataset or split variability.",
    "Results on the simulated fixture are for software validation only.",
)


def model_label(run: RunManifest) -> str:
    """``name@version``; long versions are shortened to a stable 8-char hash."""
    version = str(run.model["version"])
    if len(version) > 24:
        version = "h" + hashlib.sha256(version.encode("utf-8")).hexdigest()[:8]
    return f"{run.model['name']}@{version}"


def _load_per_user(run: RunManifest, artifacts_root: Path) -> dict[str, dict[str, float]] | None:
    rel = run.artifacts.get("per_user_metrics")
    if not rel:
        return None
    data: dict[str, dict[str, float]] = json.loads(
        (artifacts_root / rel).read_text(encoding="utf-8")
    )
    return data


def build_report(
    registry: ExperimentRegistry,
    run_ids: Sequence[str],
    artifacts_root: Path,
    *,
    primary_metric: str,
    baseline_run_id: str | None = None,
    exposure_status: Mapping[str, Any] | None = None,
    n_resamples: int = 1000,
    seed: int = 0,
    title: str = "RecomLab Evaluation Report",
) -> dict[str, Any]:
    runs = [registry.get_run(r) for r in run_ids]
    if not runs:
        raise ValueError("no runs to report")
    protocol_key = assert_comparable([r.evaluation for r in runs])
    caveats: list[str] = []
    if any(r.evaluation.get("mode") == "sampled" for r in runs):
        caveats.append(SAMPLED_WARNING)
    if any(r.evaluation.get("diagnostic_split") for r in runs):
        caveats.append("A diagnostic (random) split was used; time order is not preserved.")
    if any(r.dataset.get("synthetic") or r.dataset.get("dataset_id") == "tiny" for r in runs):
        caveats.append(
            "Dataset is SIMULATED (synthetic fixture); results are not research evidence."
        )
    exposure = dict(exposure_status or {"availability": "unavailable"})
    if exposure.get("availability") != "propensity_available":
        caveats.append(
            "Exposure-aware (off-policy) correction unavailable: no valid logged propensities."
        )
    rows = []
    per_user_cache: dict[str, dict[str, dict[str, float]]] = {}
    for r in runs:
        per_user = _load_per_user(r, artifacts_root)
        ci = None
        if per_user:
            per_user_cache[r.run_id] = per_user
            values = {u: m[primary_metric] for u, m in per_user.items() if primary_metric in m}
            if values:
                ci = bootstrap_ci(values, n_resamples=n_resamples, seed=seed).to_dict()
        else:
            caveats.append(f"Run {r.run_id}: per-user metrics not stored; no interval reported.")
        rows.append(
            {
                "run_id": r.run_id,
                "model": model_label(r),
                "partition": r.split.get("partition"),
                "dataset": f"{r.dataset['dataset_id']}@{r.dataset['version']}",
                "split_id": r.split["split_id"],
                "label_version": r.label_policy.get("label_version"),
                "evaluation_universe": r.evaluation.get("universe"),
                "metrics": {k: v for k, v in sorted(r.metrics.items()) if not isinstance(v, dict)},
                "primary_ci": ci,
                "retrieval": r.metrics.get("retrieval"),
                "slice_report": r.artifacts.get("slice_report"),
                "git_commit": r.git_commit,
            }
        )
    comparisons = []
    if baseline_run_id is not None:
        base = per_user_cache.get(baseline_run_id)
        baseline_label = next(model_label(r) for r in runs if r.run_id == baseline_run_id)
        if base is None:
            raise ValueError("baseline run has no stored per-user metrics")
        for r in runs:
            if r.run_id == baseline_run_id or r.run_id not in per_user_cache:
                continue
            other = per_user_cache[r.run_id]
            a = {u: m[primary_metric] for u, m in base.items()}
            b = {u: m[primary_metric] for u, m in other.items()}
            comparisons.append(
                paired_comparison(
                    a,
                    b,
                    metric=primary_metric,
                    baseline=baseline_run_id,
                    challenger=r.run_id,
                    n_resamples=n_resamples,
                    seed=seed,
                ).to_dict()
                | {"challenger_model": model_label(r), "baseline_model": baseline_label}
            )
    metric_defs = runs[0].metric_config
    return {
        "title": title,
        "protocol": protocol_key,
        "primary_metric": primary_metric,
        "metric_definitions": metric_defs,
        "label_policy": runs[0].label_policy,
        "runs": rows,
        "comparisons": comparisons,
        "exposure": exposure,
        "caveats": sorted(set(caveats)),
        "limitations": list(STANDARD_LIMITATIONS),
        "uncertainty": {"n_resamples": n_resamples, "seed": seed},
    }


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return "n/a" if v is None else str(v)


def render_markdown(report: Mapping[str, Any]) -> str:
    pm = report["primary_metric"]
    lines = [f"# {report['title']}", ""]
    if report["caveats"]:
        lines += ["## Warnings & Caveats", ""] + [f"- **{c}**" for c in report["caveats"]] + [""]
    lines += [
        "## Protocol",
        "",
        f"- Evaluation protocol: `{report['protocol']}`",
        f"- Primary metric: `{pm}`",
        f"- Label policy: `{report['label_policy'].get('label_version')}`",
        f"- Metric definitions: `{json.dumps(report['metric_definitions'], sort_keys=True)}`",
        f"- Exposure correction: `{report['exposure'].get('availability')}`",
        "",
        "## Ranking-Stage Results",
        "",
        f"| Model | Partition | Dataset | Split | {pm} | 95% CI | Users |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in report["runs"]:
        ci = r["primary_ci"]
        ci_txt = f"[{ci['low']:.4f}, {ci['high']:.4f}]" if ci else "n/a"
        n_users = _fmt(r["metrics"].get("n_users_evaluated"))
        lines.append(
            f"| {r['model']} | {r['partition']} | {r['dataset']} | `{r['split_id']}` | "
            f"{_fmt(r['metrics'].get(pm))} | {ci_txt} | {n_users} |"
        )
    lines += ["", "## Candidate-Stage (Retrieval) Diagnostics", ""]
    retrieval = [r for r in report["runs"] if r["retrieval"]]
    if retrieval:
        for r in retrieval:
            diag = r["retrieval"]
            lines.append(
                f"- {r['model']}: "
                + ", ".join(
                    f"{k}={_fmt(v)}" for k, v in sorted(diag.items()) if not isinstance(v, dict)
                )
            )
    else:
        lines.append(
            "- Not recorded for these runs (full-catalog recommenders without a retrieval stage)."
        )
    lines += ["", "## Paired Comparisons", ""]
    if report["comparisons"]:
        lines += [
            "| Comparison | Δ | 95% CI | Wilcoxon p | Conclusion | Users |",
            "|---|---|---|---|---|---|",
        ]
        for c in report["comparisons"]:
            interval = f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]"
            lines.append(
                f"| {c['challenger_model']} vs {c['baseline_model']} | {c['delta']:+.4f} "
                f"| {interval} "
                f"| {c['wilcoxon_p']:.3g} | {c['conclusion']} | {c['n_users']} |"
            )
    else:
        lines.append("- No baseline comparison requested.")
    lines += ["", "## Slice Reports", ""]
    lines += [
        f"- {r['model']}: `{r['slice_report']}`" if r["slice_report"] else f"- {r['model']}: none"
        for r in report["runs"]
    ]
    lines += ["", "## Limitations", ""] + [f"- {x}" for x in report["limitations"]]
    lines += [
        "",
        f"_Uncertainty: user-level bootstrap, {report['uncertainty']['n_resamples']} resamples, "
        f"seed {report['uncertainty']['seed']}._",
        "",
    ]
    return "\n".join(lines)


def write_report(
    report: Mapping[str, Any], out_dir: Path, name: str = "evaluation_report"
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md, js = out_dir / f"{name}.md", out_dir / f"{name}.json"
    with md.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(render_markdown(report))
    with js.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return md, js
