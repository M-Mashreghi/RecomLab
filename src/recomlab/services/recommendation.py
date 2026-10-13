"""Application-layer recommendation service (no training or metric logic)."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from recomlab.domain.schemas import Interaction, Item
from recomlab.registry.model_registry import ModelRegistry
from recomlab.services.pipeline import RecommendationPipeline


class NoActiveModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class RecommendationRequest:
    user_id: str
    k: int = 10
    trace_id: str | None = None
    exclude_consumed: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.k <= 100:
            raise ValueError("k must be between 1 and 100")


@dataclass(frozen=True)
class RecommendedItem:
    item_id: str
    rank: int
    score: float
    sources: dict[str, dict[str, Any]]
    contributions: dict[str, float]
    fallback: str | None = None


@dataclass(frozen=True)
class RecommendationResponse:
    trace_id: str
    user_id: str
    bundle_id: str
    items: tuple[RecommendedItem, ...]
    component_versions: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(i.item_id for i in self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "user_id": self.user_id,
            "bundle_id": self.bundle_id,
            "component_versions": self.component_versions,
            "metadata": self.metadata,
            "items": [
                {
                    "item_id": i.item_id,
                    "rank": i.rank,
                    "score": i.score,
                    "fallback": i.fallback,
                    "sources": {
                        s: {k: v for k, v in d.items() if k in ("score", "rank", "evidence")}
                        for s, d in i.sources.items()
                    },
                    "contributions": i.contributions,
                }
                for i in self.items
            ],
        }


class RecommendationService:
    """Serves the registry's active bundle; swaps atomically on activation."""

    def __init__(
        self,
        registry: ModelRegistry,
        items: Sequence[Item],
        history: Sequence[Interaction],
        as_of: datetime,
    ) -> None:
        self.registry = registry
        self.items = list(items)
        self.history = list(history)
        self.as_of = as_of
        self._pipeline: RecommendationPipeline | None = None
        self._bundle_id: str | None = None
        if registry.active_bundle_id is not None:
            self.reload()

    @property
    def bundle_id(self) -> str | None:
        return self._bundle_id

    def reload(self) -> None:
        active = self.registry.active_bundle_id
        if active is None:
            raise NoActiveModelError("no active model bundle")
        pipeline = self.registry.validate(
            self.registry.get_bundle(active), self.items, self.history, self.as_of
        )
        self._pipeline, self._bundle_id = pipeline, active

    def activate(self, bundle_id: str, reason: str = "") -> None:
        pipeline = self.registry.activate(bundle_id, self.items, self.history, self.as_of, reason)
        self._pipeline, self._bundle_id = pipeline, bundle_id

    def rollback(self, reason: str) -> None:
        pipeline = self.registry.rollback(self.items, self.history, self.as_of, reason)
        self._pipeline, self._bundle_id = pipeline, self.registry.active_bundle_id

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        if self._pipeline is None or self._bundle_id is None:
            raise NoActiveModelError("no active model bundle")
        trace = request.trace_id or uuid.uuid4().hex
        out = self._pipeline.recommend(request.user_id, request.k, request_id=trace)
        provenance = {c.item_id: c.sources for c in out.fused.candidates}
        fallbacks = dict(out.ranked.metadata.get("fallbacks", {}))
        items = tuple(
            RecommendedItem(
                item_id=i.item_id,
                rank=i.rank,
                score=i.final_score,
                sources={s: dict(v) for s, v in provenance.get(i.item_id, {}).items()},
                contributions=dict(i.contributions),
                fallback=fallbacks.get(i.item_id),
            )
            for i in out.ranked.items
        )
        return RecommendationResponse(
            trace_id=trace,
            user_id=request.user_id,
            bundle_id=self._bundle_id,
            items=items,
            component_versions=self._pipeline.component_versions,
            metadata={
                "n_candidates": len(out.fused.candidates),
                "candidate_counts": dict(out.fused.counts),
                "ranker_fallback": out.ranked.metadata.get("fallback"),
            },
        )
