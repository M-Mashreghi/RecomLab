"""Collaborative score ranker: ALS scores for fused candidates.

Fallbacks are explicit and deterministic:

* unknown user: every candidate is scored by training popularity
  (``metadata.fallback = "unknown_user"``);
* unknown item (known user): the item is placed below every known item, ordered
  by popularity; its contributions carry ``fallback_popularity``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from recomlab.models.als import ALSRecommender
from recomlab.rankers.base import RankedList, Ranker, RankRequest, Scores


class CollaborativeRanker(Ranker):
    name = "collaborative"

    def __init__(self, model: ALSRecommender, popularity: Mapping[str, int]) -> None:
        if model.artifact is None:
            raise ValueError("collaborative ranker needs a trained ALS artifact")
        self.model = model
        self.popularity = dict(popularity)
        self._max_pop = max(self.popularity.values(), default=0) or 1
        self._last_fallback: str | None = None

    @property
    def version(self) -> str:
        return f"collaborative-1[{self.model.version}]"

    def _pop_norm(self, item: str) -> float:
        return self.popularity.get(item, 0) / self._max_pop

    def _score(self, request: RankRequest) -> Scores:
        items = request.item_ids
        if not self.model.knows_user(request.user_id):
            self._last_fallback = "unknown_user"
            return {
                i: (self._pop_norm(i), {"fallback_popularity": self._pop_norm(i)}) for i in items
            }
        raw = self.model.score(request.user_id, items)
        known = {i: s for i, s in raw.items() if s is not None}
        floor = min(known.values(), default=0.0) - 1.0
        out: Scores = {i: (s, {"collaborative": s}) for i, s in known.items()}
        unknown = [i for i in items if i not in known]
        for i in unknown:
            # Strictly below every known item; popularity orders unknown items among themselves.
            score = floor - 0.5 * (1.0 - self._pop_norm(i))
            out[i] = (score, {"fallback_popularity": self._pop_norm(i)})
        self._last_fallback = f"unknown_items:{len(unknown)}" if unknown else None
        return out

    def rank(self, request: RankRequest) -> RankedList:
        ranked = super().rank(request)
        meta = {
            **ranked.metadata,
            "fallback": self._last_fallback,
            "model_version": self.model.version,
        }
        return RankedList(
            ranked.user_id,
            ranked.items,
            ranked.model_name,
            ranked.model_version,
            ranked.request_id,
            meta,
        )

    def rank_batch(self, requests: Sequence[RankRequest]) -> dict[str, RankedList]:
        """Offline batch scoring (one call per request; identical to :meth:`rank`)."""
        return {r.user_id: self.rank(r) for r in requests}
