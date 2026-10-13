"""Re-ranking (diversity, calibration) and explicit cold-start fallback policy.

Re-rankers only reorder/select items already present in the ranked list (the
eligible candidates); they never add items. The cold-start policy is the only
component allowed to add items, and only from the eligible catalog, marking
each added item with its fallback source.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from recomlab.domain.eligibility import EligibilityFilter
from recomlab.evaluation.beyond_accuracy import category_jaccard, intra_list_diversity
from recomlab.evaluation.metrics import ndcg_at_k
from recomlab.features.builder import PointInTimeFeatures
from recomlab.rankers.base import RankedItem, RankedList


def _relevance_norm(items: Sequence[RankedItem]) -> dict[str, float]:
    scores = [i.final_score for i in items]
    lo, hi = (min(scores), max(scores)) if scores else (0.0, 0.0)
    return {i.item_id: 1.0 if hi == lo else (i.final_score - lo) / (hi - lo) for i in items}


def _rebuild(
    source: RankedList,
    order: Sequence[str],
    scores: Mapping[str, float],
    tag: str,
    extra: Mapping[str, Any],
) -> RankedList:
    by_id = {i.item_id: i for i in source.items}
    items = tuple(
        RankedItem(i, by_id[i].raw_score, scores[i], r + 1, by_id[i].contributions)
        for r, i in enumerate(order)
    )
    return RankedList(
        source.user_id,
        items,
        source.model_name,
        source.model_version,
        source.request_id,
        {**source.metadata, "reranker": tag, **extra},
    )


@dataclass(frozen=True)
class MMRDiversifier:
    """Maximal marginal relevance with category-Jaccard similarity.

    ``score = lam * rel_norm - (1 - lam) * max_sim_to_selected``; ``lam = 1`` keeps
    the original order.
    """

    categories: Mapping[str, Sequence[str]]
    lam: float = 0.7

    def __post_init__(self) -> None:
        if not 0.0 <= self.lam <= 1.0:
            raise ValueError("lam must be in [0, 1]")

    def rerank(self, ranked: RankedList, k: int | None = None) -> RankedList:
        k = k or len(ranked.items)
        rel = _relevance_norm(ranked.items)
        remaining = [i.item_id for i in ranked.items]
        chosen: list[str] = []
        mmr: dict[str, float] = {}
        while remaining and len(chosen) < k:

            def value(item: str) -> float:
                sim = max(
                    (
                        category_jaccard(self.categories.get(item, ()), self.categories.get(s, ()))
                        for s in chosen
                    ),
                    default=0.0,
                )
                return self.lam * rel[item] - (1.0 - self.lam) * sim

            best = max(remaining, key=lambda i: (value(i), -remaining.index(i)))
            mmr[best] = value(best)
            chosen.append(best)
            remaining.remove(best)
        return _rebuild(ranked, chosen, mmr, "mmr", {"mmr_lambda": self.lam})


@dataclass(frozen=True)
class CalibrationReranker:
    """Greedy calibrated re-ranking (Steck, 2018).

    Maximizes ``(1 - lam) * sum rel - lam * KL(p_user || q_list)`` where ``p_user``
    is the point-in-time category profile and ``q_list`` the list's category mix
    (smoothed with ``alpha``).
    """

    categories: Mapping[str, Sequence[str]]
    lam: float = 0.5
    alpha: float = 0.01

    def _kl(self, profile: Mapping[str, float], items: Sequence[str]) -> float:
        counts: dict[str, float] = {}
        for i in items:
            cats = self.categories.get(i, ())
            for c in cats:
                counts[c] = counts.get(c, 0.0) + 1.0 / len(cats)
        total = sum(counts.values()) or 1.0
        kl = 0.0
        for c, p in profile.items():
            q = counts.get(c, 0.0) / total
            q = (1 - self.alpha) * q + self.alpha * p
            if p > 0:
                kl += p * math.log(p / q)
        return kl

    def rerank(
        self, ranked: RankedList, profile: Mapping[str, float], k: int | None = None
    ) -> RankedList:
        k = k or len(ranked.items)
        if not profile:
            return _rebuild(
                ranked,
                [i.item_id for i in ranked.items][:k],
                {i.item_id: i.final_score for i in ranked.items},
                "calibration",
                {"calibration_skipped": "no_profile"},
            )
        rel = _relevance_norm(ranked.items)
        remaining = [i.item_id for i in ranked.items]
        chosen: list[str] = []
        obj: dict[str, float] = {}
        while remaining and len(chosen) < k:

            def value(item: str) -> float:
                cand = [*chosen, item]
                return (1 - self.lam) * sum(rel[i] for i in cand) - self.lam * self._kl(
                    profile, cand
                )

            best = max(remaining, key=lambda i: (value(i), -remaining.index(i)))
            obj[best] = value(best)
            chosen.append(best)
            remaining.remove(best)
        return _rebuild(ranked, chosen, obj, "calibration", {"calibration_lambda": self.lam})


@dataclass(frozen=True)
class ColdStartPolicy:
    """Explicit fallbacks for cold/sparse users and cold items.

    * Lists shorter than ``k`` (cold user, empty candidates) are filled with the most
      popular eligible items (``fallback = "popularity"``), then by content affinity
      for cold items when no popularity signal remains (``"content"``).
    * ``cold_item_slots`` reserves up to that many tail positions for eligible items
      with no training interactions, ordered by the user's content affinity.
    """

    cold_item_slots: int = 0

    def apply(
        self,
        ranked: RankedList,
        k: int,
        eligibility: EligibilityFilter,
        features: PointInTimeFeatures,
    ) -> RankedList:
        user = ranked.user_id
        pops = features.item_popularity()
        kept = [i for i in ranked.items if eligibility.is_eligible(user, i.item_id)][:k]
        present = {i.item_id for i in kept}
        eligible = [i for i in eligibility.eligible_items(user) if i not in present]
        cold_items = [i for i in eligible if pops.get(i, 0) == 0]

        def affinity(i: str) -> float:
            return features.pair_features(user, i)["category_affinity"]

        cold_pick = sorted(cold_items, key=lambda i: (-affinity(i), i))[
            : min(self.cold_item_slots, k)
        ]
        head_room = k - len(cold_pick)
        kept = kept[:head_room]
        added: list[tuple[str, str]] = []
        need = head_room - len(kept)
        warm_pool = sorted(
            (i for i in eligible if pops.get(i, 0) > 0 and i not in cold_pick),
            key=lambda i: (-pops[i], i),
        )
        for i in warm_pool[:need]:
            added.append((i, "popularity"))
        need = head_room - len(kept) - len(added)
        rest = sorted(
            (i for i in cold_items if i not in cold_pick), key=lambda i: (-affinity(i), i)
        )
        added += [(i, "content") for i in rest[:need]]
        added += [(i, "cold_item_content") for i in cold_pick]
        items = list(kept)
        fallbacks: dict[str, str] = {}
        base = min((i.final_score for i in kept), default=0.0)
        for n, (item, why) in enumerate(added):
            fallbacks[item] = why
            items.append(RankedItem(item, 0.0, base - 1.0 - n, 0, {}))
        items = [
            RankedItem(i.item_id, i.raw_score, i.final_score, r + 1, i.contributions)
            for r, i in enumerate(items)
        ]
        return RankedList(
            user,
            tuple(items),
            ranked.model_name,
            ranked.model_version,
            ranked.request_id,
            {**ranked.metadata, "fallbacks": fallbacks},
        )


def rerank_report(
    before: Mapping[str, Sequence[str]],
    after: Mapping[str, Sequence[str]],
    truth: Mapping[str, Mapping[str, float]],
    categories: Mapping[str, Sequence[str]],
    k: int,
) -> dict[str, dict[str, float | None]]:
    """Relevance and beyond-accuracy metrics before and after re-ranking."""

    def summarize(lists: Mapping[str, Sequence[str]]) -> dict[str, float | None]:
        users = [u for u in truth if truth[u] and u in lists]
        ndcg = [ndcg_at_k(list(lists[u]), truth[u], k) for u in users]
        ild = [v for u in lists if (v := intra_list_diversity(lists[u], categories, k)) is not None]
        covered = {i for lst in lists.values() for i in list(lst)[:k]}
        return {
            f"ndcg@{k}": sum(ndcg) / len(ndcg) if ndcg else None,
            f"ild@{k}": sum(ild) / len(ild) if ild else None,
            f"distinct_items@{k}": float(len(covered)),
        }

    b, a = summarize(before), summarize(after)
    delta = {
        m: (a[m] - b[m]) if a[m] is not None and b[m] is not None else None  # type: ignore[operator]
        for m in b
    }
    return {"before": b, "after": a, "delta": delta}
