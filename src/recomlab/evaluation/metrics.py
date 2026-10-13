"""Top-K ranking metrics (per-user functions)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")


def hit_rate_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    _check_k(k)
    return 1.0 if any(i in relevant for i in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    """Fraction of relevant items retrieved in the top-k (denominator: all relevant)."""
    _check_k(k)
    if not relevant:
        raise ValueError("recall undefined for users without relevant items")
    return sum(1 for i in ranked[:k] if i in relevant) / len(relevant)


def ndcg_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    """NDCG with linear gain ``rel`` and log2 discount; graded if relevances differ."""
    _check_k(k)
    if not relevant:
        raise ValueError("ndcg undefined for users without relevant items")
    dcg = sum(relevant.get(i, 0.0) / math.log2(r + 2) for r, i in enumerate(ranked[:k]))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(r + 2) for r, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0
