"""Beyond-accuracy list metrics."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence


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
