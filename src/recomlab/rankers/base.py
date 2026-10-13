"""Ranker interface and scoring contract.

A ranker receives a :class:`RankRequest` holding one user's candidate set and
returns a :class:`RankedList`. It may score only those candidates (never the
catalog), receives no held-out labels, and produces no presentation text.
Ordering is deterministic: descending final score, ties by ascending item id.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from recomlab.candidates.fusion import FusedCandidates


class RankerContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RankCandidate:
    item_id: str
    retrieval_score: float = 0.0
    sources: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class RankRequest:
    user_id: str
    candidates: tuple[RankCandidate, ...]
    k: int
    request_id: str | None = None

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise RankerContractError("k must be positive")
        ids = [c.item_id for c in self.candidates]
        if len(ids) != len(set(ids)):
            raise RankerContractError("duplicate candidates; fuse/deduplicate first")

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(c.item_id for c in self.candidates)

    @classmethod
    def from_fused(
        cls, fused: FusedCandidates, k: int, request_id: str | None = None
    ) -> RankRequest:
        return cls(
            user_id=fused.user_id,
            candidates=tuple(
                RankCandidate(c.item_id, c.score, c.sources) for c in fused.candidates
            ),
            k=k,
            request_id=request_id,
        )

    @classmethod
    def from_items(cls, user_id: str, item_ids: Sequence[str], k: int) -> RankRequest:
        return cls(user_id, tuple(RankCandidate(i) for i in item_ids), k)


@dataclass(frozen=True, slots=True)
class RankedItem:
    item_id: str
    raw_score: float
    final_score: float
    rank: int
    contributions: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RankedList:
    user_id: str
    items: tuple[RankedItem, ...]
    model_name: str
    model_version: str
    request_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(i.item_id for i in self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "request_id": self.request_id,
            "metadata": dict(self.metadata),
            "items": [{**asdict(i), "contributions": dict(i.contributions)} for i in self.items],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> RankedList:
        return cls(
            user_id=raw["user_id"],
            items=tuple(RankedItem(**i) for i in raw["items"]),
            model_name=raw["model_name"],
            model_version=raw["model_version"],
            request_id=raw.get("request_id"),
            metadata=raw.get("metadata", {}),
        )


Scores = dict[str, tuple[float, dict[str, float]]]


class Ranker(ABC):
    name: str = "ranker"

    @property
    def version(self) -> str:
        return f"{self.name}-1"

    @abstractmethod
    def _score(self, request: RankRequest) -> Scores:
        """Return ``{item_id: (raw_score, contributions)}`` for candidate items only."""

    def _finalize(self, raw: float) -> float:
        return raw

    def rank(self, request: RankRequest) -> RankedList:
        start = time.perf_counter()
        scores = self._score(request)
        extra = set(scores) - set(request.item_ids)
        if extra:
            raise RankerContractError(f"ranker scored non-candidate items: {sorted(extra)[:5]}")
        missing = set(request.item_ids) - set(scores)
        finals = {i: (self._finalize(raw), raw, contrib) for i, (raw, contrib) in scores.items()}
        ordered = sorted(finals.items(), key=lambda kv: (-kv[1][0], kv[0]))[: request.k]
        items = tuple(
            RankedItem(i, raw, final, r + 1, contrib)
            for r, (i, (final, raw, contrib)) in enumerate(ordered)
        )
        return RankedList(
            user_id=request.user_id,
            items=items,
            model_name=self.name,
            model_version=self.version,
            request_id=request.request_id,
            metadata={
                "n_candidates": len(request.candidates),
                "n_unscored": len(missing),
                "latency_ms": (time.perf_counter() - start) * 1000.0,
            },
        )


class RetrievalScoreRanker(Ranker):
    """Ranks by fused retrieval score (identity ranker; useful baseline/fallback)."""

    name = "retrieval_score"

    def _score(self, request: RankRequest) -> Scores:
        return {c.item_id: (c.retrieval_score, {}) for c in request.candidates}
