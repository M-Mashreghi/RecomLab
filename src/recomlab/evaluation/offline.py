"""Full-catalog offline evaluation of a fitted recommender on a split partition."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from recomlab.datasets.labels import LabelPolicy, derive_labels, relevance_by_user
from recomlab.datasets.negative_sampling import TrainingNegatives
from recomlab.datasets.splits import Split
from recomlab.domain.eligibility import EligibilityFilter, EligibilityPolicy
from recomlab.domain.schemas import Item
from recomlab.evaluation.metrics import hit_rate_at_k, ndcg_at_k, recall_at_k
from recomlab.models.base import Recommender

MetricFn = Callable[[Sequence[str], Mapping[str, float], int], float]
DEFAULT_METRICS: dict[str, MetricFn] = {
    "recall": recall_at_k,
    "ndcg": ndcg_at_k,
    "hit_rate": hit_rate_at_k,
}


class EvaluationInputError(TypeError):
    pass


@dataclass
class EvaluationResult:
    partition: str
    mode: str
    universe: str
    ks: tuple[int, ...]
    metrics: dict[str, float]
    per_user: dict[str, dict[str, float]]
    rankings: dict[str, list[str]]
    n_users_evaluated: int
    users_without_positives: list[str]
    catalog_coverage: dict[int, float]
    latency_ms: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "mode": self.mode,
            "universe": self.universe,
            "n_users_evaluated": self.n_users_evaluated,
            "n_users_without_positives": len(self.users_without_positives),
            **self.metrics,
            **{f"catalog_coverage@{k}": v for k, v in self.catalog_coverage.items()},
        }


def holdout_truth(
    split: Split, partition: str, label_policy: LabelPolicy
) -> dict[str, dict[str, float]]:
    events = split.validation if partition == "validation" else split.test
    if partition not in ("validation", "test"):
        raise ValueError("partition must be 'validation' or 'test'")
    return relevance_by_user(derive_labels(events, label_policy))


def evaluate_recommender(
    model: Recommender,
    split: Split,
    items: Sequence[Item],
    label_policy: LabelPolicy,
    ks: Sequence[int] = (5, 10),
    *,
    partition: str = "test",
    eligibility_policy: EligibilityPolicy | None = None,
    metrics: Mapping[str, MetricFn] | None = None,
) -> EvaluationResult:
    """Rank the full eligible catalog for every held-out user with positives.

    The model must already be fitted on ``split.train_until(partition)``.
    """
    if isinstance(split, TrainingNegatives):  # pragma: no cover - defensive
        raise EvaluationInputError("training negatives are not evaluation truth")
    metric_fns = dict(metrics or DEFAULT_METRICS)
    history = split.train_until(partition)
    elig = EligibilityFilter(items, history, eligibility_policy)
    truth = holdout_truth(split, partition, label_policy)
    held_users = sorted(
        {e.user_id for e in (split.test if partition == "test" else split.validation)}
    )
    no_pos = [u for u in held_users if u not in truth]
    max_k = max(ks)
    per_user: dict[str, dict[str, float]] = {}
    rankings: dict[str, list[str]] = {}
    latencies: list[float] = []
    for user in sorted(truth):
        start = time.perf_counter()
        ranked = [s.item_id for s in model.recommend(user, max_k, elig)]
        latencies.append((time.perf_counter() - start) * 1000.0)
        rankings[user] = ranked
        per_user[user] = {
            f"{name}@{k}": fn(ranked, truth[user], k) for name, fn in metric_fns.items() for k in ks
        }
    names = sorted({m for row in per_user.values() for m in row})
    agg = (
        {m: statistics.fmean(row[m] for row in per_user.values()) for m in names}
        if per_user
        else {}
    )
    coverage = {
        k: (
            len({i for r in rankings.values() for i in r[:k]}) / len(elig.catalog)
            if elig.catalog
            else 0.0
        )
        for k in ks
    }
    lat = (
        {
            "mean": statistics.fmean(latencies),
            "p50": statistics.median(latencies),
            "max": max(latencies),
        }
        if latencies
        else {}
    )
    return EvaluationResult(
        partition=partition,
        mode="full",
        universe=f"full eligible catalog ({len(elig.catalog)} items, consumed excluded)",
        ks=tuple(ks),
        metrics=agg,
        per_user=per_user,
        rankings=rankings,
        n_users_evaluated=len(per_user),
        users_without_positives=no_pos,
        catalog_coverage=coverage,
        latency_ms=lat,
    )
