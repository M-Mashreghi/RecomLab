"""Interpretable weighted-feature ranker.

``score = sum_f w_f * x_f`` with features normalized to ``[0, 1]``:

* ``retrieval``: min-max of the fused retrieval score within the request
  (all equal -> 1.0);
* ``popularity``: ``log1p(count) / log1p(max count)`` over the catalog;
* ``recency``: exponentially decayed popularity divided by its catalog max;
* ``content``: user-to-item category affinity (point-in-time profile).

All item/user statistics come from a :class:`PointInTimeFeatures` snapshot,
so no post-event information is used. Per-feature contributions ``w_f * x_f``
sum exactly to the raw score and are exposed for explanation grounding.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from recomlab.features.builder import PointInTimeFeatures, assert_point_in_time
from recomlab.rankers.base import Ranker, RankRequest, Scores

FEATURES = ("retrieval", "popularity", "recency", "content")


@dataclass(frozen=True)
class HeuristicWeights:
    weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "retrieval": 0.4,
            "popularity": 0.2,
            "recency": 0.1,
            "content": 0.3,
        }
    )

    def __post_init__(self) -> None:
        unknown = set(self.weights) - set(FEATURES)
        if unknown:
            raise ValueError(f"unknown features {sorted(unknown)}")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("weights must be non-negative")

    def get(self, name: str) -> float:
        return float(self.weights.get(name, 0.0))

    def to_dict(self) -> dict[str, float]:
        return {f: self.get(f) for f in FEATURES}


class HeuristicRanker(Ranker):
    name = "heuristic"

    def __init__(self, features: PointInTimeFeatures, weights: HeuristicWeights | None = None):
        assert_point_in_time(features)
        self.features = features
        self.weights = weights or HeuristicWeights()
        pops = features.item_popularity()
        self._max_log_pop = math.log1p(max(pops.values(), default=0))
        recents = [features.item_features(i)["recent_popularity"] or 0.0 for i in pops]
        self._max_recent = max(recents, default=0.0)

    @property
    def version(self) -> str:
        w = ",".join(f"{k}={v}" for k, v in self.weights.to_dict().items())
        return f"heuristic-1[{w}]@{self.features.manifest.as_of}"

    def params(self) -> dict[str, Any]:
        return {"weights": self.weights.to_dict(), "features": self.features.manifest.to_dict()}

    def feature_vector(self, request: RankRequest) -> dict[str, dict[str, float]]:
        scores = [c.retrieval_score for c in request.candidates]
        lo, hi = (min(scores), max(scores)) if scores else (0.0, 0.0)
        out: dict[str, dict[str, float]] = {}
        for c in request.candidates:
            item = self.features.item_features(c.item_id)
            pop = item["popularity"] or 0.0
            recent = item["recent_popularity"] or 0.0
            out[c.item_id] = {
                "retrieval": 1.0 if hi == lo else (c.retrieval_score - lo) / (hi - lo),
                "popularity": math.log1p(pop) / self._max_log_pop if self._max_log_pop else 0.0,
                "recency": recent / self._max_recent if self._max_recent else 0.0,
                "content": self.features.pair_features(request.user_id, c.item_id)[
                    "category_affinity"
                ],
            }
        return out

    def _score(self, request: RankRequest) -> Scores:
        out: Scores = {}
        for item, x in self.feature_vector(request).items():
            contrib = {f: self.weights.get(f) * x[f] for f in FEATURES}
            out[item] = (sum(contrib.values()), contrib)
        return out
