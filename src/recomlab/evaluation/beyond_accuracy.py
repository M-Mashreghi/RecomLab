"""Beyond-accuracy metrics: coverage, diversity, novelty, popularity bias, calibration.

Assumptions (documented in ``docs/EVALUATION_PROTOCOL.md``):

* Item representation for diversity is the item's category set; pairwise
  distance is ``1 - Jaccard``.
* Novelty is mean self-information ``-log2(p_i)`` with
  ``p_i = (users_i + 1) / (n_users + 1)`` from *training* interactions
  (add-one smoothing keeps never-seen items finite).
* Popularity concentration is the Gini coefficient of recommendation counts
  over the eligible catalog and the share of slots taken by head items (top
  ``head_fraction`` by training popularity).
* Calibration is mean ``KL(p_user || q_list)`` over categories with smoothing
  ``alpha``; it requires category metadata and a user category profile.

A metric that cannot be computed is reported as ``None`` with a reason in
``unavailable`` -- never as zero.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


def category_jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def intra_list_diversity(
    ranked: Sequence[str], categories: Mapping[str, Sequence[str]], k: int
) -> float | None:
    """Mean pairwise ``1 - Jaccard(categories)`` over the top-k; ``None`` if < 2 items."""
    top = list(ranked[:k])
    if len(top) < 2:
        return None
    pairs = list(itertools.combinations(top, 2))
    return sum(
        1.0 - category_jaccard(categories.get(a, ()), categories.get(b, ())) for a, b in pairs
    ) / len(pairs)


def gini(values: Sequence[float]) -> float | None:
    """Gini coefficient (0 = equal exposure, -> 1 = concentrated)."""
    xs = sorted(values)
    n = len(xs)
    total = sum(xs)
    if n == 0 or total == 0:
        return None
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * cum) / (n * total) - (n + 1) / n


def list_category_distribution(
    items: Sequence[str], categories: Mapping[str, Sequence[str]]
) -> dict[str, float]:
    counts: dict[str, float] = {}
    for i in items:
        cats = categories.get(i, ())
        for c in cats:
            counts[c] = counts.get(c, 0.0) + 1.0 / len(cats)
    total = sum(counts.values())
    return {c: v / total for c, v in counts.items()} if total else {}


def calibration_kl(
    profile: Mapping[str, float], dist: Mapping[str, float], alpha: float = 0.01
) -> float:
    kl = 0.0
    for c, p in profile.items():
        if p <= 0:
            continue
        q = (1 - alpha) * dist.get(c, 0.0) + alpha * p
        kl += p * math.log(p / q)
    return kl


@dataclass
class BeyondAccuracyReport:
    k: int
    values: dict[str, float | None]
    unavailable: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **{f"{n}@{self.k}": v for n, v in self.values.items()},
            "unavailable": dict(self.unavailable),
        }


def beyond_accuracy(
    rankings: Mapping[str, Sequence[str]],
    k: int,
    *,
    catalog: Sequence[str],
    train_item_users: Mapping[str, int],
    n_train_users: int,
    categories: Mapping[str, Sequence[str]] | None = None,
    profiles: Mapping[str, Mapping[str, float]] | None = None,
    population: Sequence[str] | None = None,
    head_fraction: float = 0.2,
    calibration_alpha: float = 0.01,
) -> BeyondAccuracyReport:
    users = sorted(population if population is not None else rankings)
    tops = {u: list(rankings.get(u, []))[:k] for u in users}
    values: dict[str, float | None] = {}
    unavailable: dict[str, str] = {}
    slots = [i for lst in tops.values() for i in lst]

    # Coverage
    if catalog:
        values["catalog_coverage"] = len(set(slots) & set(catalog)) / len(catalog)
    else:
        values["catalog_coverage"] = None
        unavailable["catalog_coverage"] = "empty eligible catalog"
    if users:
        values["user_coverage"] = sum(1 for u in users if tops[u]) / len(users)
    else:
        values["user_coverage"] = None
        unavailable["user_coverage"] = "no users"

    # Diversity
    if categories and any(categories.get(i) for i in slots):
        ild = [v for u in users if (v := intra_list_diversity(tops[u], categories, k)) is not None]
        values["ild"] = sum(ild) / len(ild) if ild else None
        if not ild:
            unavailable["ild"] = "no list with >= 2 items"
    else:
        values["ild"] = None
        unavailable["ild"] = "no category metadata"

    # Novelty
    if slots and n_train_users > 0:
        info = [-math.log2((train_item_users.get(i, 0) + 1) / (n_train_users + 1)) for i in slots]
        values["novelty"] = sum(info) / len(info)
    else:
        values["novelty"] = None
        unavailable["novelty"] = "no recommendations or no training users"

    # Popularity concentration
    exposure = Counter(slots)
    values["gini"] = gini([float(exposure.get(i, 0)) for i in catalog]) if catalog else None
    if values["gini"] is None:
        unavailable["gini"] = "no exposure"
    ranked_pop = sorted(catalog, key=lambda i: (-train_item_users.get(i, 0), i))
    head = (
        set(ranked_pop[: max(1, math.ceil(head_fraction * len(ranked_pop)))]) if catalog else set()
    )
    if slots:
        values["head_share"] = sum(1 for i in slots if i in head) / len(slots)
    else:
        values["head_share"] = None
        unavailable["head_share"] = "no recommendations"

    # Calibration
    if not categories:
        values["calibration_kl"] = None
        unavailable["calibration_kl"] = "no category metadata"
    else:
        kls = [
            calibration_kl(
                profiles[u], list_category_distribution(tops[u], categories), calibration_alpha
            )
            for u in users
            if profiles and profiles.get(u) and tops[u]
        ]
        values["calibration_kl"] = sum(kls) / len(kls) if kls else None
        if not kls:
            unavailable["calibration_kl"] = "no users with a category profile"
    return BeyondAccuracyReport(k, values, unavailable)
