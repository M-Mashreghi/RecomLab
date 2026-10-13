"""Top-K ranking metrics.

Relevance semantics:

* ``relevant`` maps item id -> relevance grade (> 0). Items absent are irrelevant.
* Precision, Recall, HitRate, MAP and MRR are **binary** (grade > 0 counts as hit).
* NDCG is **graded**: gain ``rel`` (``gain="linear"``) or ``2**rel - 1``
  (``gain="exponential"``) with ``log2(rank + 1)`` discount; binary labels give
  standard binary NDCG.

Per-user metrics are undefined for users without relevant items; aggregation
excludes them explicitly and reports them, never scoring them as zero.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

GAINS = ("linear", "exponential")


class UndefinedMetricError(ValueError):
    pass


def _check(k: int, relevant: Mapping[str, float]) -> None:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        raise UndefinedMetricError("metric undefined for users without relevant items")


def hit_rate_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    _check(k, relevant)
    return 1.0 if any(i in relevant for i in ranked[:k]) else 0.0


def precision_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    """Hits in the top-k divided by k (short lists are not rewarded)."""
    _check(k, relevant)
    return sum(1 for i in ranked[:k] if i in relevant) / k


def recall_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    """Hits in the top-k divided by the number of relevant items."""
    _check(k, relevant)
    return sum(1 for i in ranked[:k] if i in relevant) / len(relevant)


def ndcg_at_k(
    ranked: Sequence[str], relevant: Mapping[str, float], k: int, gain: str = "linear"
) -> float:
    _check(k, relevant)
    if gain not in GAINS:
        raise ValueError(f"gain must be one of {GAINS}")

    def g(rel: float) -> float:
        return rel if gain == "linear" else 2.0**rel - 1.0

    dcg = sum(g(relevant.get(i, 0.0)) / math.log2(r + 2) for r, i in enumerate(ranked[:k]))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(g(rel) / math.log2(r + 2) for r, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    """AP@k = sum over hit positions of precision@position, / min(|relevant|, k)."""
    _check(k, relevant)
    hits = 0
    total = 0.0
    for r, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            hits += 1
            total += hits / r
    return total / min(len(relevant), k)


def reciprocal_rank_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    _check(k, relevant)
    for r, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            return 1.0 / r
    return 0.0


MetricFn = Callable[[Sequence[str], Mapping[str, float], int], float]
METRICS: dict[str, MetricFn] = {
    "precision": precision_at_k,
    "recall": recall_at_k,
    "ndcg": ndcg_at_k,
    "map": average_precision_at_k,
    "mrr": reciprocal_rank_at_k,
    "hit_rate": hit_rate_at_k,
}


@dataclass(frozen=True)
class MetricConfig:
    ks: tuple[int, ...] = (5, 10)
    metrics: tuple[str, ...] = ("precision", "recall", "ndcg", "map", "mrr", "hit_rate")
    gain: str = "linear"

    def __post_init__(self) -> None:
        unknown = set(self.metrics) - set(METRICS)
        if unknown:
            raise ValueError(f"unknown metrics {sorted(unknown)}")
        if not self.ks or any(k <= 0 for k in self.ks):
            raise ValueError("ks must be positive")
        if self.gain not in GAINS:
            raise ValueError(f"gain must be one of {GAINS}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ks": list(self.ks),
            "metrics": list(self.metrics),
            "gain": self.gain,
            "relevance": "binary except graded NDCG",
            "aggregation": "mean over users with >=1 relevant item",
        }

    def compute(self, ranked: Sequence[str], relevant: Mapping[str, float]) -> dict[str, float]:
        out = {}
        for name in self.metrics:
            for k in self.ks:
                if name == "ndcg":
                    out[f"ndcg@{k}"] = ndcg_at_k(ranked, relevant, k, self.gain)
                else:
                    out[f"{name}@{k}"] = METRICS[name](ranked, relevant, k)
        return out


@dataclass
class AggregatedMetrics:
    means: dict[str, float]
    per_user: dict[str, dict[str, float]]
    n_users: int
    users_without_positives: list[str] = field(default_factory=list)
    users_without_ranking: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.means,
            "n_users_evaluated": self.n_users,
            "n_users_without_positives": len(self.users_without_positives),
            "n_users_without_ranking": len(self.users_without_ranking),
        }


def aggregate(
    rankings: Mapping[str, Sequence[str]],
    truth: Mapping[str, Mapping[str, float]],
    config: MetricConfig,
    population: Sequence[str] | None = None,
) -> AggregatedMetrics:
    """Mean of per-user metrics over the evaluable population.

    ``population`` defaults to every user in ``truth``. Users without relevant
    items are excluded and listed; users with relevant items but no ranking are
    scored with an empty list (a recommender failure counts against it).
    """
    users = sorted(population if population is not None else truth)
    no_pos = [u for u in users if not truth.get(u)]
    evaluable = [u for u in users if truth.get(u)]
    no_rank = [u for u in evaluable if u not in rankings]
    per_user = {u: config.compute(list(rankings.get(u, [])), truth[u]) for u in evaluable}
    names = [f"{m}@{k}" for m in config.metrics for k in config.ks]
    means = (
        {n: statistics.fmean(per_user[u][n] for u in evaluable) for n in names} if evaluable else {}
    )
    return AggregatedMetrics(means, per_user, len(evaluable), no_pos, no_rank)
