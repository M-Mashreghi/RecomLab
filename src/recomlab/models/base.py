"""Full-catalog recommender interface used by baselines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from recomlab.domain.eligibility import EligibilityFilter
from recomlab.domain.schemas import Interaction, Item


@dataclass(frozen=True, slots=True)
class ScoredItem:
    item_id: str
    score: float


class NotFittedError(RuntimeError):
    pass


class Recommender(Protocol):
    name: str
    version: str

    def fit(
        self, train: Sequence[Interaction], items: Sequence[Item], as_of: datetime | None = None
    ) -> Recommender: ...

    def recommend(
        self, user_id: str, k: int, eligibility: EligibilityFilter
    ) -> list[ScoredItem]: ...

    def params(self) -> dict[str, Any]: ...


def top_k(scores: dict[str, float], k: int) -> list[ScoredItem]:
    """Deterministic top-k: descending score, ties broken by ascending item id."""
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [ScoredItem(i, s) for i, s in ranked[:k]]


def assert_train_horizon(train: Sequence[Interaction], as_of: datetime | None) -> None:
    """Fail closed if training data contain events at/after the prediction time."""
    if as_of is None:
        return
    late = [e.event_id for e in train if e.event_time >= as_of]
    if late:
        raise ValueError(f"training events at/after as_of: {late[:5]}")
