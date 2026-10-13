"""Multi-metric leaderboard, Pareto comparison and champion selection.

Rules enforced here:

* The :class:`SelectionPolicy` (primary metric, direction, guardrails) is fixed
  before test evaluation; its ``policy_id`` is recorded with every decision.
* Champion selection uses **validation** runs only; test runs are refused for
  selection and are only attached to the final report of an already-chosen
  champion.
* Runs evaluated under different protocols (full vs sampled) are never ranked
  together (``assert_comparable``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from recomlab.evaluation.protocol import assert_comparable
from recomlab.experiments.manifest import RunManifest

OPS = (">=", "<=")


class SelectionError(RuntimeError):
    pass


def metric_value(run: RunManifest, name: str) -> float | None:
    """Look up ``name`` in run metrics; dotted names traverse nested dicts."""
    node: Any = run.metrics
    for part in name.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return float(node) if isinstance(node, int | float) else None


@dataclass(frozen=True)
class Guardrail:
    metric: str
    op: str
    threshold: float

    def __post_init__(self) -> None:
        if self.op not in OPS:
            raise ValueError(f"op must be one of {OPS}")

    def check(self, value: float | None) -> bool:
        if value is None:
            return False  # missing guardrail evidence fails closed
        return value >= self.threshold if self.op == ">=" else value <= self.threshold


@dataclass(frozen=True)
class SelectionPolicy:
    primary_metric: str
    higher_is_better: bool = True
    guardrails: tuple[Guardrail, ...] = ()
    pareto_metrics: tuple[str, ...] = ()

    @property
    def policy_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return f"policy-{hashlib.sha256(payload).hexdigest()[:12]}"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "policy_id": self.policy_id}


@dataclass(frozen=True)
class LeaderboardRow:
    run_id: str
    model: str
    partition: str
    protocol: str
    primary: float | None
    metrics: dict[str, float | None]
    guardrails: dict[str, bool]
    pareto_optimal: bool
    slice_report: str | None

    @property
    def passes_guardrails(self) -> bool:
        return all(self.guardrails.values())


@dataclass(frozen=True)
class ChampionDecision:
    policy: dict[str, Any]
    champion_run_id: str | None
    champion_config: str | None
    rationale: str
    rows: list[dict[str, Any]]
    test_report: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def model_config_key(run: RunManifest) -> str:
    """Model identity independent of retraining: name + hyperparameters.

    Refitting the same configuration on train+validation for the final test run
    changes artifact versions but not this key.
    """
    return json.dumps(
        {"name": run.model["name"], "params": run.model.get("params", {})}, sort_keys=True
    )


def _partition(run: RunManifest) -> str:
    return str(run.split.get("partition") or run.evaluation.get("partition") or "unknown")


def pareto_front(
    rows: Sequence[Mapping[str, float | None]],
    metrics: Sequence[str],
    higher_is_better: Mapping[str, bool],
) -> list[bool]:
    """``True`` for rows not dominated on ``metrics`` (missing values never dominate)."""

    def better_eq(a: float, b: float, m: str) -> bool:
        return a >= b if higher_is_better.get(m, True) else a <= b

    out = []
    for i, r in enumerate(rows):
        dominated = False
        for j, s in enumerate(rows):
            if i == j or any(r.get(m) is None or s.get(m) is None for m in metrics):
                continue
            all_ge = all(better_eq(s[m], r[m], m) for m in metrics)  # type: ignore[arg-type]
            any_gt = any(s[m] != r[m] for m in metrics)
            if all_ge and any_gt:
                dominated = True
                break
        out.append(not dominated)
    return out


def build_leaderboard(
    runs: Sequence[RunManifest],
    policy: SelectionPolicy,
    higher_is_better: Mapping[str, bool] | None = None,
) -> list[LeaderboardRow]:
    if not runs:
        return []
    protocol = assert_comparable([r.evaluation for r in runs])
    hib = {policy.primary_metric: policy.higher_is_better, **(higher_is_better or {})}
    names = sorted(
        {policy.primary_metric, *policy.pareto_metrics, *(g.metric for g in policy.guardrails)}
    )
    values = [{n: metric_value(r, n) for n in names} for r in runs]
    front = (
        pareto_front(values, policy.pareto_metrics, hib)
        if policy.pareto_metrics
        else [True] * len(runs)
    )
    rows = []
    for run, vals, opt in zip(runs, values, front, strict=True):
        rows.append(
            LeaderboardRow(
                run_id=run.run_id,
                model=f"{run.model['name']}@{run.model['version']}",
                partition=_partition(run),
                protocol=protocol,
                primary=vals[policy.primary_metric],
                metrics=vals,
                guardrails={
                    f"{g.metric} {g.op} {g.threshold}": g.check(vals[g.metric])
                    for g in policy.guardrails
                },
                pareto_optimal=opt,
                slice_report=run.artifacts.get("slice_report"),
            )
        )
    sign = -1.0 if policy.higher_is_better else 1.0
    rows.sort(
        key=lambda r: (
            not r.passes_guardrails,
            r.primary is None,
            sign * (r.primary or 0.0),
            r.run_id,
        )
    )
    return rows


def select_champion(
    validation_runs: Sequence[RunManifest],
    policy: SelectionPolicy,
    higher_is_better: Mapping[str, bool] | None = None,
) -> ChampionDecision:
    """Pick the best guardrail-passing run by the primary metric on validation data."""
    bad = [r.run_id for r in validation_runs if _partition(r) != "validation"]
    if bad:
        raise SelectionError(
            f"selection must use validation runs only; got non-validation runs {bad[:3]}"
        )
    rows = build_leaderboard(validation_runs, policy, higher_is_better)
    eligible = [r for r in rows if r.passes_guardrails and r.primary is not None]
    notes = []
    by_id = {r.run_id: r for r in validation_runs}
    config: str | None = None
    if not eligible:
        rationale = "no run passes every guardrail; no champion selected"
        champion = None
    else:
        best = eligible[0]
        champion = best.run_id
        config = model_config_key(by_id[champion])
        rationale = (
            f"{best.model} has the best validation {policy.primary_metric}={best.primary:.6g} "
            f"among {len(eligible)} guardrail-passing runs"
        )
        if not best.pareto_optimal:
            notes.append("champion is not Pareto-optimal on the configured trade-off metrics")
        if len([r for r in rows if r.pareto_optimal]) > 1:
            notes.append("multiple Pareto-optimal runs: no single model dominates")
    return ChampionDecision(
        policy=policy.to_dict(),
        champion_run_id=champion,
        champion_config=config,
        rationale=rationale,
        rows=[{**asdict(r), "passes_guardrails": r.passes_guardrails} for r in rows],
        notes=notes,
    )


def attach_test_report(
    decision: ChampionDecision, test_runs: Sequence[RunManifest]
) -> ChampionDecision:
    """Attach final test results for the already-selected champion (reporting only)."""
    if decision.champion_run_id is None:
        raise SelectionError("no champion selected; nothing to report on test")
    if any(_partition(r) != "test" for r in test_runs):
        raise SelectionError("final report accepts test-partition runs only")
    config = decision.champion_config
    matching = [r for r in test_runs if model_config_key(r) == config]
    if len(matching) != 1:
        raise SelectionError(f"expected exactly one test run for champion config {config}")
    test = matching[0]
    report = {
        "run_id": test.run_id,
        "model": f"{test.model['name']}@{test.model['version']}",
        "metrics": {k: v for k, v in test.metrics.items() if not isinstance(v, dict)},
        "note": "test metrics reported once after selection; not used for selection",
    }
    return ChampionDecision(**{**decision.__dict__, "test_report": report})
