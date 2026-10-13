"""Random and global-popularity baselines (training data only)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from recomlab.domain.eligibility import EligibilityFilter
from recomlab.domain.schemas import Interaction, Item
from recomlab.experiments.seed import make_rng
from recomlab.models.base import NotFittedError, ScoredItem, assert_train_horizon, top_k


class RandomRecommender:
    """Seeded random ranking; per-user streams make output order-independent."""

    name = "random"
    version = "1"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._fitted = False

    def fit(
        self, train: Sequence[Interaction], items: Sequence[Item], as_of: datetime | None = None
    ) -> RandomRecommender:
        assert_train_horizon(train, as_of)
        self._fitted = True
        return self

    def recommend(self, user_id: str, k: int, eligibility: EligibilityFilter) -> list[ScoredItem]:
        if not self._fitted:
            raise NotFittedError(self.name)
        candidates = eligibility.eligible_items(user_id)
        draws = make_rng(self.seed, "random-rec", user_id).random(len(candidates))
        return top_k(dict(zip(candidates, draws.tolist(), strict=True)), k)

    def params(self) -> dict[str, Any]:
        return {"seed": self.seed}


class PopularityRecommender:
    """Global popularity counted from training events only."""

    name = "popularity"
    version = "1"

    def __init__(self, event_types: tuple[str, ...] | None = None) -> None:
        self.event_types = event_types
        self.counts: dict[str, int] | None = None

    def fit(
        self, train: Sequence[Interaction], items: Sequence[Item], as_of: datetime | None = None
    ) -> PopularityRecommender:
        assert_train_horizon(train, as_of)
        counts = Counter(
            e.item_id
            for e in train
            if self.event_types is None or str(e.event_type) in self.event_types
        )
        self.counts = dict(counts)
        return self

    def score(self, item_id: str) -> float:
        if self.counts is None:
            raise NotFittedError(self.name)
        return float(self.counts.get(item_id, 0))

    def recommend(self, user_id: str, k: int, eligibility: EligibilityFilter) -> list[ScoredItem]:
        if self.counts is None:
            raise NotFittedError(self.name)
        return top_k({i: self.score(i) for i in eligibility.eligible_items(user_id)}, k)

    def params(self) -> dict[str, Any]:
        return {"event_types": list(self.event_types) if self.event_types else None}
