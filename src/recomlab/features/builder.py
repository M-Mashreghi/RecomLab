"""Point-in-time user, item and user-item features.

Every aggregate is computed only from events strictly before ``as_of``. The
builder records the latest event time it used so leakage checks can verify
the point-in-time property; ``strict=True`` refuses inputs containing future
events instead of silently filtering them.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from recomlab.domain.schemas import EventType, Interaction, Item

FEATURE_VERSION = "pit-1.0"

USER_FEATURES = (
    "history_length",
    "distinct_items",
    "days_since_last_event",
    "days_active",
    "mean_rating",
)
ITEM_FEATURES = (
    "popularity",
    "unique_users",
    "recent_popularity",
    "days_since_last_event",
    "mean_rating",
)
PAIR_FEATURES = ("category_affinity", "seen_before")


class FutureDataError(ValueError):
    pass


@dataclass(frozen=True)
class FeatureManifest:
    version: str
    as_of: str
    n_events_used: int
    max_event_time_used: str | None
    user_features: tuple[str, ...] = USER_FEATURES
    item_features: tuple[str, ...] = ITEM_FEATURES
    pair_features: tuple[str, ...] = PAIR_FEATURES
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "as_of": self.as_of,
            "n_events_used": self.n_events_used,
            "max_event_time_used": self.max_event_time_used,
            "user_features": list(self.user_features),
            "item_features": list(self.item_features),
            "pair_features": list(self.pair_features),
            "params": self.params,
        }


def _days(delta_seconds: float) -> float:
    return delta_seconds / 86400.0


class PointInTimeFeatures:
    """Feature lookups valid at a single ``as_of`` instant."""

    def __init__(
        self,
        events: Iterable[Interaction],
        items: Iterable[Item],
        as_of: datetime,
        *,
        strict: bool = False,
        recency_half_life_days: float = 7.0,
    ) -> None:
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        all_events = list(events)
        future = [e for e in all_events if e.event_time >= as_of]
        if strict and future:
            raise FutureDataError(f"{len(future)} events at/after as_of {as_of.isoformat()}")
        self.as_of = as_of
        self.events = sorted(
            (e for e in all_events if e.event_time < as_of), key=lambda e: e.event_time
        )
        self.items = {i.item_id: i for i in items}
        self.half_life = recency_half_life_days
        self._build()

    def _build(self) -> None:
        self._user_events: dict[str, list[Interaction]] = defaultdict(list)
        self._item_events: dict[str, list[Interaction]] = defaultdict(list)
        for e in self.events:
            self._user_events[e.user_id].append(e)
            self._item_events[e.item_id].append(e)
        self._user_cat: dict[str, dict[str, float]] = {}
        for user, evs in self._user_events.items():
            counts: Counter[str] = Counter()
            for e in evs:
                for c in self.items.get(e.item_id, Item(e.item_id)).categories:
                    counts[c] += 1
            total = sum(counts.values())
            self._user_cat[user] = {c: n / total for c, n in counts.items()} if total else {}

    @property
    def manifest(self) -> FeatureManifest:
        last = self.events[-1].event_time.isoformat() if self.events else None
        return FeatureManifest(
            version=FEATURE_VERSION,
            as_of=self.as_of.isoformat(),
            n_events_used=len(self.events),
            max_event_time_used=last,
            params={"recency_half_life_days": self.half_life},
        )

    def _mean_rating(self, evs: list[Interaction]) -> float | None:
        ratings = [e.value for e in evs if e.event_type is EventType.RATING and e.value is not None]
        return sum(ratings) / len(ratings) if ratings else None

    def user_features(self, user_id: str) -> dict[str, float | None]:
        evs = self._user_events.get(user_id, [])
        if not evs:
            return {
                "history_length": 0.0,
                "distinct_items": 0.0,
                "days_since_last_event": None,
                "days_active": None,
                "mean_rating": None,
            }
        return {
            "history_length": float(len(evs)),
            "distinct_items": float(len({e.item_id for e in evs})),
            "days_since_last_event": _days((self.as_of - evs[-1].event_time).total_seconds()),
            "days_active": _days((evs[-1].event_time - evs[0].event_time).total_seconds()),
            "mean_rating": self._mean_rating(evs),
        }

    def item_features(self, item_id: str) -> dict[str, float | None]:
        evs = self._item_events.get(item_id, [])
        decay = math.log(2) / self.half_life
        recent = sum(
            math.exp(-decay * _days((self.as_of - e.event_time).total_seconds())) for e in evs
        )
        return {
            "popularity": float(len(evs)),
            "unique_users": float(len({e.user_id for e in evs})),
            "recent_popularity": recent,
            "days_since_last_event": (
                _days((self.as_of - evs[-1].event_time).total_seconds()) if evs else None
            ),
            "mean_rating": self._mean_rating(evs),
        }

    def user_category_profile(self, user_id: str) -> Mapping[str, float]:
        return self._user_cat.get(user_id, {})

    def user_history(self, user_id: str) -> list[Interaction]:
        return list(self._user_events.get(user_id, []))

    def pair_features(self, user_id: str, item_id: str) -> dict[str, float]:
        profile = self._user_cat.get(user_id, {})
        cats = self.items.get(item_id, Item(item_id)).categories
        affinity = sum(profile.get(c, 0.0) for c in cats) / len(cats) if cats else 0.0
        seen = any(e.item_id == item_id for e in self._user_events.get(user_id, []))
        return {"category_affinity": affinity, "seen_before": 1.0 if seen else 0.0}

    def item_popularity(self) -> dict[str, int]:
        return {i: len(evs) for i, evs in self._item_events.items()}


def assert_point_in_time(features: PointInTimeFeatures) -> None:
    """Raise :class:`FutureDataError` if any used event is at/after ``as_of``."""
    m = features.manifest
    if m.max_event_time_used is not None and m.max_event_time_used >= m.as_of:
        raise FutureDataError("feature builder used events at/after as_of")
    if any(e.event_time >= features.as_of for e in features.events):
        raise FutureDataError("feature builder used events at/after as_of")
