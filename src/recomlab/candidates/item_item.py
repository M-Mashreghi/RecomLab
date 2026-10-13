"""Item-item similarity candidate generator.

Similarity = ``(1 - metadata_weight) * cooccurrence_cosine + metadata_weight *
category_jaccard``. Co-occurrence uses binary user-item incidence from the
training history only. A user's candidates aggregate neighbors of their most
recent ``history_window`` items with recency weights ``decay ** age_rank``.
Users without history get an empty set flagged ``fallback_required`` so the
caller applies an explicit cold-start source.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from recomlab.candidates.base import CandidateGenerator, CandidateRequest
from recomlab.domain.schemas import Interaction, Item, sort_key


class ItemItemCandidates(CandidateGenerator):
    name = "item_item"

    def __init__(
        self,
        metadata_weight: float = 0.0,
        history_window: int = 10,
        decay: float = 0.9,
        neighbors_per_item: int = 50,
    ) -> None:
        super().__init__()
        if not 0.0 <= metadata_weight <= 1.0:
            raise ValueError("metadata_weight must be in [0, 1]")
        self.metadata_weight = metadata_weight
        self.history_window = history_window
        self.decay = decay
        self.neighbors_per_item = neighbors_per_item

    @property
    def version(self) -> str:
        return f"item_item-1-mw{self.metadata_weight}-w{self.history_window}"

    def _fit(self, history: Sequence[Interaction], items: Sequence[Item]) -> None:
        users_of: dict[str, set[str]] = defaultdict(set)
        self.user_history: dict[str, list[str]] = defaultdict(list)
        for e in sorted(history, key=sort_key):
            users_of[e.item_id].add(e.user_id)
            seq = self.user_history[e.user_id]
            if e.item_id in seq:
                seq.remove(e.item_id)
            seq.append(e.item_id)  # most recent last, unique
        cats = {i.item_id: set(i.categories) for i in items}
        ids = sorted(set(users_of) | set(cats))
        sims: dict[str, list[tuple[str, float]]] = {}
        for a in ids:
            row = []
            for b in ids:
                if a == b:
                    continue
                s = self._similarity(users_of.get(a, set()), users_of.get(b, set()), cats, a, b)
                if s > 0:
                    row.append((b, s))
            row.sort(key=lambda t: (-t[1], t[0]))
            sims[a] = row[: self.neighbors_per_item]
        self.similar = sims

    def _similarity(
        self, ua: set[str], ub: set[str], cats: dict[str, set[str]], a: str, b: str
    ) -> float:
        co = len(ua & ub) / math.sqrt(len(ua) * len(ub)) if ua and ub else 0.0
        ca, cb = cats.get(a, set()), cats.get(b, set())
        jac = len(ca & cb) / len(ca | cb) if ca | cb else 0.0
        return (1.0 - self.metadata_weight) * co + self.metadata_weight * jac

    def neighbors(self, item_id: str) -> list[tuple[str, float]]:
        return list(self.similar.get(item_id, []))

    def _score(
        self, request: CandidateRequest
    ) -> tuple[dict[str, float], dict[str, dict[str, Any]], dict[str, Any]]:
        hist = self.user_history.get(request.user_id, [])
        if not hist:
            return {}, {}, {"fallback_required": True, "reason": "cold_user_no_history"}
        seeds = list(reversed(hist[-self.history_window :]))  # most recent first
        scores: dict[str, float] = defaultdict(float)
        evidence: dict[str, list[tuple[str, float]]] = defaultdict(list)
        consumed = set(hist)
        for age, seed in enumerate(seeds):
            w = self.decay**age
            for nb, sim in self.similar.get(seed, []):
                if nb in consumed:
                    continue
                scores[nb] += w * sim
                evidence[nb].append((seed, sim))
        ev = {
            i: {
                "seed_items": [
                    {"item_id": s, "similarity": round(v, 6)}
                    for s, v in sorted(seeds_, key=lambda t: (-t[1], t[0]))[:3]
                ]
            }
            for i, seeds_ in evidence.items()
        }
        return dict(scores), ev, {"fallback_required": False, "n_seed_items": len(seeds)}
