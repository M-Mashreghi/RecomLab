"""Candidate retrieval contract.

Retrieval optimizes recall under a candidate *budget*. Generators are fitted on
the visible training history only; ``generate`` receives a request (user,
budget, exclusions) and never any held-out labels. Every candidate carries its
source name, source score, source rank and optional evidence (provenance).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from recomlab.domain.schemas import Interaction, Item
from recomlab.models.base import NotFittedError, assert_train_horizon


class CandidateContractError(ValueError):
    pass


@dataclass(frozen=True)
class CandidateRequest:
    user_id: str
    budget: int
    exclude_items: frozenset[str] = frozenset()
    request_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.budget, int) or self.budget <= 0:
            raise CandidateContractError("candidate budget must be a positive integer")


@dataclass(frozen=True, slots=True)
class Candidate:
    item_id: str
    source: str
    score: float
    rank: int
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateSet:
    request: CandidateRequest
    source: str
    source_version: str
    candidates: tuple[Candidate, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(c.item_id for c in self.candidates)

    @property
    def is_empty(self) -> bool:
        return not self.candidates


class CandidateGenerator(ABC):
    """Base class enforcing the retrieval contract for every generator."""

    name: str = "generator"

    def __init__(self) -> None:
        self._fitted = False
        self.catalog: tuple[str, ...] = ()

    @property
    def version(self) -> str:
        return f"{self.name}-1"

    def fit(
        self, history: Sequence[Interaction], items: Sequence[Item], as_of: datetime | None = None
    ) -> CandidateGenerator:
        assert_train_horizon(history, as_of)
        self.catalog = tuple(sorted(i.item_id for i in items))
        self._fit(history, items)
        self._fitted = True
        return self

    @abstractmethod
    def _fit(self, history: Sequence[Interaction], items: Sequence[Item]) -> None: ...

    @abstractmethod
    def _score(
        self, request: CandidateRequest
    ) -> tuple[dict[str, float], dict[str, dict[str, Any]], dict[str, Any]]:
        """Return ``(scores, evidence_by_item, metadata)`` for the request."""

    def generate(self, request: CandidateRequest) -> CandidateSet:
        if not self._fitted:
            raise NotFittedError(self.name)
        start = time.perf_counter()
        scores, evidence, meta = self._score(request)
        return self._assemble(request, scores, evidence, meta, start)

    def _assemble(
        self,
        request: CandidateRequest,
        scores: Mapping[str, float],
        evidence: Mapping[str, Mapping[str, Any]],
        meta: Mapping[str, Any],
        start: float,
    ) -> CandidateSet:
        """Apply exclusions, deterministic ordering and the budget."""
        eligible = {i: s for i, s in scores.items() if i not in request.exclude_items}
        ranked = sorted(eligible.items(), key=lambda kv: (-kv[1], kv[0]))
        kept = ranked[: request.budget]
        candidates = tuple(
            Candidate(i, self.name, float(s), r + 1, evidence.get(i, {}))
            for r, (i, s) in enumerate(kept)
        )
        return CandidateSet(
            request=request,
            source=self.name,
            source_version=self.version,
            candidates=candidates,
            metadata={
                **meta,
                "n_scored": len(scores),
                "n_after_exclusions": len(eligible),
                "budget": request.budget,
                "truncated": len(eligible) > request.budget,
                "latency_ms": (time.perf_counter() - start) * 1000.0,
            },
        )

    def generate_batch(self, requests: Iterable[CandidateRequest]) -> dict[str, CandidateSet]:
        return {r.user_id: self.generate(r) for r in requests}


def validate_candidate_set(cs: CandidateSet) -> None:
    """Contract checks: budget, unique items, sequential ranks, single source."""
    ids = cs.item_ids
    if len(ids) > cs.request.budget:
        raise CandidateContractError("candidate budget exceeded")
    if len(ids) != len(set(ids)):
        raise CandidateContractError("duplicate items within one source")
    if [c.rank for c in cs.candidates] != list(range(1, len(ids) + 1)):
        raise CandidateContractError("ranks must be 1..n")
    if any(c.source != cs.source for c in cs.candidates):
        raise CandidateContractError("mixed sources in one candidate set")
    if set(ids) & cs.request.exclude_items:
        raise CandidateContractError("excluded item returned")


class PopularityCandidates(CandidateGenerator):
    """Training-history popularity; also the explicit cold-user fallback source."""

    name = "popularity"

    def _fit(self, history: Sequence[Interaction], items: Sequence[Item]) -> None:
        counts: dict[str, float] = {i: 0.0 for i in self.catalog}
        for e in history:
            counts[e.item_id] = counts.get(e.item_id, 0.0) + 1.0
        self.counts = counts

    def _score(
        self, request: CandidateRequest
    ) -> tuple[dict[str, float], dict[str, dict[str, Any]], dict[str, Any]]:
        evidence = {i: {"train_count": c} for i, c in self.counts.items()}
        return dict(self.counts), evidence, {}


def generate_all(
    generators: Sequence[CandidateGenerator], request: CandidateRequest
) -> dict[str, CandidateSet]:
    out: dict[str, CandidateSet] = {}
    for g in generators:
        if g.name in out:
            raise CandidateContractError(f"duplicate generator name {g.name!r}")
        out[g.name] = g.generate(request)
    return out
