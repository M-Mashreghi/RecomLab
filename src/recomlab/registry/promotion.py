"""Champion / challenger promotion gate.

The gate compares a challenger run against the current champion run under the
Phase 5 protocol (same split, same evaluation protocol, paired user-level
comparison) and applies predefined rules:

* primary metric: ``delta >= min_improvement`` and, when required, a paired
  bootstrap interval that excludes zero (``conclusion == "better"``);
* guardrails: each guardrail metric (dotted paths reach nested metrics such as
  ``latency_ms.p50`` or ``slices.coldness.cold.ndcg@10``) may not regress more
  than its tolerance relative to the champion; missing values fail closed.

Decisions are stored as JSON with every input id and recomputed values, and
:func:`replay_decision` re-derives the same decision from stored artifacts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from recomlab.domain.schemas import Interaction, Item
from recomlab.evaluation.leaderboard import metric_value
from recomlab.evaluation.protocol import assert_comparable
from recomlab.evaluation.statistics import paired_comparison
from recomlab.experiments.manifest import RunManifest
from recomlab.experiments.registry import ExperimentRegistry
from recomlab.registry.model_registry import ModelRegistry


class PromotionInputError(ValueError):
    pass


@dataclass(frozen=True)
class RegressionGuardrail:
    metric: str
    higher_is_better: bool = True
    max_regression: float = 0.0  # allowed worsening, in metric units (or ratio if relative)
    relative: bool = False

    def evaluate(self, champion: float | None, challenger: float | None) -> dict[str, Any]:
        if champion is None or challenger is None:
            return {"metric": self.metric, "passed": False, "reason": "missing metric"}
        worsening = (champion - challenger) if self.higher_is_better else (challenger - champion)
        allowed = self.max_regression * abs(champion) if self.relative else self.max_regression
        return {
            "metric": self.metric,
            "champion": champion,
            "challenger": challenger,
            "worsening": worsening,
            "allowed": allowed,
            "passed": worsening <= allowed + 1e-12,
        }


@dataclass(frozen=True)
class PromotionPolicy:
    primary_metric: str
    min_improvement: float = 0.0
    require_significance: bool = True
    guardrails: tuple[RegressionGuardrail, ...] = ()
    n_resamples: int = 1000
    seed: int = 0

    @property
    def policy_id(self) -> str:
        return (
            "promo-"
            + hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode("utf-8")).hexdigest()[
                :12
            ]
        )


@dataclass(frozen=True)
class PromotionDecision:
    policy: dict[str, Any]
    champion_run_id: str
    challenger_run_id: str
    challenger_bundle_id: str | None
    comparison: dict[str, Any]
    primary_passed: bool
    guardrails: list[dict[str, Any]]
    promote: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"decision-{self.challenger_run_id}.json"
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
            fh.write("\n")
        return path


def _per_user(run: RunManifest, artifacts_root: Path, metric: str) -> dict[str, float]:
    rel = run.artifacts.get("per_user_metrics")
    if not rel:
        raise PromotionInputError(f"run {run.run_id} has no stored per-user metrics")
    data = json.loads((artifacts_root / rel).read_text(encoding="utf-8"))
    return {u: float(m[metric]) for u, m in data.items()}


def evaluate_promotion(
    champion: RunManifest,
    challenger: RunManifest,
    artifacts_root: Path,
    policy: PromotionPolicy,
    challenger_bundle_id: str | None = None,
) -> PromotionDecision:
    if champion.split.get("split_id") != challenger.split.get("split_id"):
        raise PromotionInputError("champion and challenger must be evaluated on the same split")
    if challenger.split.get("partition") == "test" or champion.split.get("partition") == "test":
        raise PromotionInputError("promotion decisions use validation runs, not test runs")
    assert_comparable([champion.evaluation, challenger.evaluation])
    a = _per_user(champion, artifacts_root, policy.primary_metric)
    b = _per_user(challenger, artifacts_root, policy.primary_metric)
    comp = paired_comparison(
        a,
        b,
        metric=policy.primary_metric,
        baseline=champion.run_id,
        challenger=challenger.run_id,
        n_resamples=policy.n_resamples,
        seed=policy.seed,
    )
    reasons = []
    primary_ok = comp.delta >= policy.min_improvement
    if not primary_ok:
        reasons.append(
            f"primary delta {comp.delta:+.4g} below required {policy.min_improvement:+.4g}"
        )
    if policy.require_significance and comp.conclusion != "better":
        primary_ok = False
        reasons.append(f"improvement not significant (paired bootstrap: {comp.conclusion})")
    guard_results = [
        g.evaluate(metric_value(champion, g.metric), metric_value(challenger, g.metric))
        for g in policy.guardrails
    ]
    for g in guard_results:
        if not g["passed"]:
            reasons.append(f"guardrail failed: {g['metric']} ({g.get('reason', 'regression')})")
    promote = primary_ok and all(g["passed"] for g in guard_results)
    if promote:
        reasons.append("all promotion criteria passed")
    return PromotionDecision(
        policy={**asdict(policy), "policy_id": policy.policy_id},
        champion_run_id=champion.run_id,
        challenger_run_id=challenger.run_id,
        challenger_bundle_id=challenger_bundle_id,
        comparison=comp.to_dict(),
        primary_passed=primary_ok,
        guardrails=guard_results,
        promote=promote,
        reasons=reasons,
    )


def replay_decision(
    decision_file: Path, experiments: ExperimentRegistry, artifacts_root: Path
) -> PromotionDecision:
    """Recompute a stored decision from the registry and stored artifacts."""
    stored = json.loads(Path(decision_file).read_text(encoding="utf-8"))
    pol = dict(stored["policy"])
    pol.pop("policy_id")
    pol["guardrails"] = tuple(RegressionGuardrail(**g) for g in pol["guardrails"])
    policy = PromotionPolicy(**pol)
    return evaluate_promotion(
        experiments.get_run(stored["champion_run_id"]),
        experiments.get_run(stored["challenger_run_id"]),
        artifacts_root,
        policy,
        stored["challenger_bundle_id"],
    )


def apply_decision(
    decision: PromotionDecision,
    models: ModelRegistry,
    items: Sequence[Item],
    history: Sequence[Interaction],
    as_of: datetime,
) -> bool:
    """Activate the challenger bundle only when the decision says so."""
    if not decision.promote:
        return False
    if decision.challenger_bundle_id is None:
        raise PromotionInputError("decision has no challenger bundle to activate")
    models.activate(
        decision.challenger_bundle_id,
        items,
        history,
        as_of,
        reason=f"promotion {decision.policy['policy_id']}: {decision.challenger_run_id}",
    )
    return True
