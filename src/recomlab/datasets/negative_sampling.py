"""Training negative sampling.

Sampled negatives are *synthetic training data*. They are returned inside a
:class:`TrainingNegatives` container marked ``purpose="training_only"`` and
evaluation code refuses that type (see ``recomlab.evaluation``), so they can
never be presented as evaluation truth.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np

from recomlab.domain.schemas import Interaction
from recomlab.experiments.seed import make_rng

STRATEGIES = ("uniform", "popularity")


@dataclass(frozen=True, slots=True)
class NegativeSample:
    user_id: str
    item_id: str
    synthetic: Literal[True] = True


@dataclass(frozen=True)
class TrainingNegatives:
    samples: tuple[NegativeSample, ...]
    strategy: str
    seed: int
    horizon: datetime | None
    exhausted_users: tuple[str, ...] = ()
    purpose: Literal["training_only"] = "training_only"
    metadata: dict[str, object] = field(default_factory=dict)

    def by_user(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for s in self.samples:
            out.setdefault(s.user_id, []).append(s.item_id)
        return out


class TrainingNegativeSampler:
    def __init__(self, strategy: str = "uniform", seed: int = 0, alpha: float = 0.75) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        self.strategy = strategy
        self.seed = seed
        self.alpha = alpha

    def sample(
        self,
        train_events: Iterable[Interaction],
        catalog: Sequence[str],
        n_per_user: int,
        *,
        horizon: datetime | None = None,
        users: Iterable[str] | None = None,
    ) -> TrainingNegatives:
        """Sample up to ``n_per_user`` negatives per user, without replacement.

        Only events at or before ``horizon`` are used, both to exclude known
        positives and to compute popularity. Every item the user interacted
        with inside the horizon is excluded (implicit positives included).
        """
        events = [e for e in train_events if horizon is None or e.event_time <= horizon]
        seen: dict[str, set[str]] = {}
        for e in events:
            seen.setdefault(e.user_id, set()).add(e.item_id)
        catalog_sorted = sorted(set(catalog))
        pop = Counter(e.item_id for e in events)
        weights = np.array([(pop[i] + 1.0) ** self.alpha for i in catalog_sorted])
        target_users = sorted(set(users) if users is not None else seen)
        samples: list[NegativeSample] = []
        exhausted: list[str] = []
        for user in target_users:
            known = seen.get(user, set())
            mask = np.array([i not in known for i in catalog_sorted], dtype=bool)
            eligible = np.flatnonzero(mask)
            if eligible.size == 0:
                exhausted.append(user)
                continue
            k = min(n_per_user, int(eligible.size))
            rng = make_rng(self.seed, "negatives", self.strategy, user)
            if self.strategy == "uniform":
                chosen = rng.choice(eligible, size=k, replace=False)
            else:
                p = weights[eligible] / weights[eligible].sum()
                chosen = rng.choice(eligible, size=k, replace=False, p=p)
            samples.extend(NegativeSample(user, catalog_sorted[j]) for j in sorted(chosen))
            if k < n_per_user:
                exhausted.append(user)
        return TrainingNegatives(
            samples=tuple(samples),
            strategy=self.strategy,
            seed=self.seed,
            horizon=horizon,
            exhausted_users=tuple(exhausted),
            metadata={"n_per_user": n_per_user, "catalog_size": len(catalog_sorted)},
        )
