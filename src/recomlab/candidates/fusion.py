"""Candidate fusion, deduplication and eligibility filtering.

Fusion methods:

* ``rrf``: reciprocal rank fusion, ``sum_s w_s / (rrf_k + rank_s)``.
* ``weighted_score``: ``sum_s w_s * minmax_s(score)`` with per-source min-max
  normalization (a constant-score source maps to 1.0).

Items are deduplicated by id while every contributing source's score and
rank are preserved as provenance. Eligibility is applied after deduplication,
so it can only remove items, never add or duplicate them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from recomlab.candidates.base import CandidateSet
from recomlab.domain.eligibility import EligibilityFilter

FUSION_METHODS = ("rrf", "weighted_score")


@dataclass(frozen=True)
class FusionConfig:
    method: str = "rrf"
    weights: Mapping[str, float] = field(default_factory=dict)  # default weight 1.0
    rrf_k: int = 60
    budget: int = 100

    def __post_init__(self) -> None:
        if self.method not in FUSION_METHODS:
            raise ValueError(f"method must be one of {FUSION_METHODS}")
        if self.budget <= 0:
            raise ValueError("budget must be positive")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("weights must be non-negative")

    def weight(self, source: str) -> float:
        return float(self.weights.get(source, 1.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "weights": dict(sorted(self.weights.items())),
            "rrf_k": self.rrf_k,
            "budget": self.budget,
        }


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    item_id: str
    score: float
    rank: int
    sources: Mapping[str, Mapping[str, Any]]  # source -> {score, rank, evidence}


@dataclass(frozen=True)
class FusedCandidates:
    user_id: str
    candidates: tuple[FusedCandidate, ...]
    counts: Mapping[str, int]
    removed: Mapping[str, int]
    source_versions: Mapping[str, str]
    config: FusionConfig

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(c.item_id for c in self.candidates)

    def items_from_source(self, source: str) -> set[str]:
        return {c.item_id for c in self.candidates if source in c.sources}


def _minmax(cs: CandidateSet) -> dict[str, float]:
    if not cs.candidates:
        return {}
    scores = [c.score for c in cs.candidates]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return {c.item_id: 1.0 for c in cs.candidates}
    return {c.item_id: (c.score - lo) / (hi - lo) for c in cs.candidates}


def fuse(
    user_id: str,
    candidate_sets: Mapping[str, CandidateSet],
    eligibility: EligibilityFilter,
    config: FusionConfig | None = None,
) -> FusedCandidates:
    config = config or FusionConfig()
    pre_dedup = sum(len(cs.candidates) for cs in candidate_sets.values())
    fused: dict[str, float] = {}
    provenance: dict[str, dict[str, dict[str, Any]]] = {}
    for source in sorted(candidate_sets):
        cs = candidate_sets[source]
        if cs.request.user_id != user_id:
            raise ValueError(f"candidate set from {source} is for another user")
        w = config.weight(source)
        norm = _minmax(cs) if config.method == "weighted_score" else {}
        for c in cs.candidates:
            contrib = w / (config.rrf_k + c.rank) if config.method == "rrf" else w * norm[c.item_id]
            fused[c.item_id] = fused.get(c.item_id, 0.0) + contrib
            provenance.setdefault(c.item_id, {})[source] = {
                "score": c.score,
                "rank": c.rank,
                "contribution": contrib,
                "evidence": dict(c.evidence),
            }
    post_dedup = len(fused)
    removed: dict[str, int] = {"not_eligible_catalog": 0, "consumed": 0}
    eligible: dict[str, float] = {}
    for item, score in fused.items():
        if eligibility.is_eligible(user_id, item):
            eligible[item] = score
        elif item not in eligibility.catalog:
            removed["not_eligible_catalog"] += 1
        else:
            removed["consumed"] += 1
    ranked = sorted(eligible.items(), key=lambda kv: (-kv[1], kv[0]))[: config.budget]
    candidates = tuple(
        FusedCandidate(i, s, r + 1, provenance[i]) for r, (i, s) in enumerate(ranked)
    )
    return FusedCandidates(
        user_id=user_id,
        candidates=candidates,
        counts={
            "pre_dedup": pre_dedup,
            "post_dedup": post_dedup,
            "post_filter": len(eligible),
            "final": len(candidates),
        },
        removed=removed,
        source_versions={s: cs.source_version for s, cs in sorted(candidate_sets.items())},
        config=config,
    )
