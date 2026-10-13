"""Slice analysis for cold/warm, sparse/dense users and head/mid/tail items.

Buckets are deterministic functions of the *visible training history*:

* ``coldness``: ``cold`` (no history) or ``warm``.
* ``activity``: ``none`` (0 events), ``sparse`` (< ``sparse_max``), ``medium``,
  ``dense`` (>= ``dense_min``).
* item ``popularity``: items sorted by training interaction count (ties by id);
  the top ``head_fraction`` are ``head``, the bottom ``tail_fraction`` are
  ``tail``, the rest ``mid``; items never interacted with are ``cold``.
* custom user slices: any user attribute listed in ``user_attributes``.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from recomlab.domain.schemas import Interaction, User


@dataclass(frozen=True)
class SliceConfig:
    sparse_max: int = 3
    dense_min: int = 10
    head_fraction: float = 0.2
    tail_fraction: float = 0.5
    user_attributes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0 < self.sparse_max <= self.dense_min:
            raise ValueError("need 0 < sparse_max <= dense_min")
        if self.head_fraction + self.tail_fraction > 1:
            raise ValueError("head_fraction + tail_fraction must be <= 1")


def user_slices(
    history: Iterable[Interaction],
    users: Sequence[str],
    config: SliceConfig,
    user_records: Iterable[User] = (),
) -> dict[str, dict[str, str]]:
    counts = Counter(e.user_id for e in history)
    attrs = {u.user_id: u.attributes for u in user_records}
    out: dict[str, dict[str, str]] = {}
    for u in sorted(users):
        n = counts.get(u, 0)
        activity = (
            "none"
            if n == 0
            else "sparse"
            if n < config.sparse_max
            else "dense"
            if n >= config.dense_min
            else "medium"
        )
        row = {"coldness": "cold" if n == 0 else "warm", "activity": activity}
        for a in config.user_attributes:
            row[f"attr:{a}"] = str(attrs.get(u, {}).get(a, "unknown"))
        out[u] = row
    return out


def item_popularity_buckets(
    history: Iterable[Interaction], catalog: Sequence[str], config: SliceConfig
) -> dict[str, str]:
    counts = Counter(e.item_id for e in history)
    seen = sorted((i for i in catalog if counts.get(i, 0) > 0), key=lambda i: (-counts[i], i))
    n = len(seen)
    n_head = math.ceil(config.head_fraction * n) if n else 0
    n_tail = math.floor(config.tail_fraction * n)
    out = {i: "cold" for i in catalog if counts.get(i, 0) == 0}
    for rank, item in enumerate(seen):
        out[item] = "head" if rank < n_head else "tail" if rank >= n - n_tail else "mid"
    return out


@dataclass
class SliceReport:
    user_dimensions: dict[str, dict[str, dict[str, float]]]
    item_buckets: dict[str, dict[str, float]]
    population: int
    config: dict[str, Any] = field(default_factory=dict)

    def reconciles(self) -> bool:
        return all(
            sum(int(b["n_users"]) for b in dim.values()) == self.population
            for dim in self.user_dimensions.values()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "population": self.population,
            "user_dimensions": self.user_dimensions,
            "item_buckets": self.item_buckets,
            "config": self.config,
        }

    @property
    def report_id(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return f"slices-{hashlib.sha256(payload).hexdigest()[:12]}"

    def write_json(self, path: Path) -> str:
        data = {"report_id": self.report_id, **self.to_dict()}
        with Path(path).open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
        return self.report_id


def slice_report(
    per_user_metrics: Mapping[str, Mapping[str, float]],
    slices: Mapping[str, Mapping[str, str]],
    rankings: Mapping[str, Sequence[str]],
    truth: Mapping[str, Mapping[str, float]],
    item_buckets: Mapping[str, str],
    k: int,
    config: SliceConfig,
) -> SliceReport:
    """Aggregate per-user metrics by user slice and recall by item bucket."""
    population = sorted(per_user_metrics)
    missing = [u for u in population if u not in slices]
    if missing:
        raise ValueError(f"users without slice membership: {missing[:5]}")
    metric_names = sorted({m for row in per_user_metrics.values() for m in row})
    dims: dict[str, dict[str, dict[str, float]]] = {}
    for dim in sorted({d for u in population for d in slices[u]}):
        buckets: dict[str, list[str]] = {}
        for u in population:
            buckets.setdefault(slices[u][dim], []).append(u)
        dims[dim] = {
            b: {
                "n_users": float(len(us)),
                **{m: statistics.fmean(per_user_metrics[u][m] for u in us) for m in metric_names},
            }
            for b, us in sorted(buckets.items())
        }
    item_stats: dict[str, dict[str, float]] = {}
    for bucket in sorted(set(item_buckets.values()) | {"unknown"}):
        rel_total = 0
        hits = 0
        for u in population:
            rel = [i for i in truth.get(u, {}) if item_buckets.get(i, "unknown") == bucket]
            top = set(list(rankings.get(u, []))[:k])
            rel_total += len(rel)
            hits += sum(1 for i in rel if i in top)
        if rel_total:
            item_stats[bucket] = {
                "n_relevant": float(rel_total),
                f"recall@{k}": hits / rel_total,
            }
    return SliceReport(dims, item_stats, len(population), asdict(config))
