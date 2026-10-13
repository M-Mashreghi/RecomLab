"""Candidate-stage (retrieval) metrics and failure diagnostics.

Retrieval is measured before ranking: a relevant item missing from the
candidate set is a *retrieval failure*; a relevant item present in the
candidates but outside the final top-K is a *ranking failure*.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from recomlab.candidates.base import CandidateSet
from recomlab.domain.schemas import Interaction


class HasItems(Protocol):
    @property
    def item_ids(self) -> tuple[str, ...]: ...


def candidate_recall(candidates: HasItems, relevant: Mapping[str, float]) -> float:
    """Recall@CandidateN: share of the user's relevant items present in the candidate set."""
    if not relevant:
        raise ValueError("candidate recall undefined for users without relevant items")
    ids = set(candidates.item_ids)
    return sum(1 for i in relevant if i in ids) / len(relevant)


def candidate_hit(candidates: HasItems, relevant: Mapping[str, float]) -> float:
    ids = set(candidates.item_ids)
    return 1.0 if any(i in ids for i in relevant) else 0.0


def activity_segments(
    history: Iterable[Interaction], users: Iterable[str], sparse_max: int = 3
) -> dict[str, str]:
    """``cold`` (0 events), ``sparse`` (1..sparse_max-1), ``warm`` (>= sparse_max)."""
    counts: dict[str, int] = {}
    for e in history:
        counts[e.user_id] = counts.get(e.user_id, 0) + 1
    out = {}
    for u in users:
        n = counts.get(u, 0)
        out[u] = "cold" if n == 0 else "sparse" if n < sparse_max else "warm"
    return out


@dataclass(frozen=True)
class RetrievalDiagnostics:
    candidate_budget: int
    n_users: int
    recall: float
    hit_rate: float
    empty_candidate_rate: float
    mean_candidate_count: float
    candidate_count: dict[str, float]
    by_segment: dict[str, dict[str, float]]
    by_source: dict[str, dict[str, float]]
    per_user_recall: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_budget": self.candidate_budget,
            "n_users": self.n_users,
            f"recall@{self.candidate_budget}": self.recall,
            f"hit_rate@{self.candidate_budget}": self.hit_rate,
            "empty_candidate_rate": self.empty_candidate_rate,
            "mean_candidate_count": self.mean_candidate_count,
            "candidate_count": self.candidate_count,
            "by_segment": self.by_segment,
            "by_source": self.by_source,
        }

    def write_json(self, path: Path) -> None:
        with Path(path).open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
            fh.write("\n")


def _sources_of(cands: Any) -> dict[str, set[str]]:
    """Map source -> items for a fused set or a single-source CandidateSet."""
    if isinstance(cands, CandidateSet):
        return {cands.source: set(cands.item_ids)}
    out: dict[str, set[str]] = {}
    for c in cands.candidates:
        for s in c.sources:
            out.setdefault(s, set()).add(c.item_id)
    return out


def retrieval_diagnostics(
    candidates_by_user: Mapping[str, HasItems],
    truth: Mapping[str, Mapping[str, float]],
    segments: Mapping[str, str],
    candidate_budget: int,
) -> RetrievalDiagnostics:
    users = sorted(u for u in truth if truth[u])
    missing = [u for u in users if u not in candidates_by_user]
    if missing:
        raise ValueError(f"no candidates produced for users {missing[:5]}")
    per_user = {u: candidate_recall(candidates_by_user[u], truth[u]) for u in users}
    hits = {u: candidate_hit(candidates_by_user[u], truth[u]) for u in users}
    counts = [len(candidates_by_user[u].item_ids) for u in users]
    by_segment: dict[str, dict[str, float]] = {}
    for seg in sorted({segments.get(u, "unknown") for u in users}):
        members = [u for u in users if segments.get(u, "unknown") == seg]
        by_segment[seg] = {
            "n_users": float(len(members)),
            "recall": statistics.fmean(per_user[u] for u in members),
            "empty_candidate_rate": statistics.fmean(
                1.0 if not candidates_by_user[u].item_ids else 0.0 for u in members
            ),
        }
    source_items: dict[str, dict[str, set[str]]] = {
        u: _sources_of(candidates_by_user[u]) for u in users
    }
    all_sources = sorted({s for m in source_items.values() for s in m})
    by_source = {
        s: {
            "recall": statistics.fmean(
                sum(1 for i in truth[u] if i in source_items[u].get(s, set())) / len(truth[u])
                for u in users
            ),
            "mean_contributed": statistics.fmean(len(source_items[u].get(s, set())) for u in users),
        }
        for s in all_sources
    }
    n = len(users)
    return RetrievalDiagnostics(
        candidate_budget=candidate_budget,
        n_users=n,
        recall=statistics.fmean(per_user.values()) if n else 0.0,
        hit_rate=statistics.fmean(hits.values()) if n else 0.0,
        empty_candidate_rate=(sum(1 for c in counts if c == 0) / n) if n else 0.0,
        mean_candidate_count=statistics.fmean(counts) if n else 0.0,
        candidate_count=(
            {"min": float(min(counts)), "median": statistics.median(counts), "max": max(counts)}
            if n
            else {}
        ),
        by_segment=by_segment,
        by_source=by_source,
        per_user_recall=per_user,
    )


def failure_breakdown(
    candidates_by_user: Mapping[str, HasItems],
    rankings: Mapping[str, Sequence[str]],
    truth: Mapping[str, Mapping[str, float]],
    k: int,
) -> dict[str, int]:
    """Classify each held-out relevant item as hit, ranking failure, or retrieval failure."""
    out = {"hit": 0, "ranking_failure": 0, "retrieval_failure": 0}
    for user, rel in truth.items():
        cands = set(candidates_by_user[user].item_ids) if user in candidates_by_user else set()
        top = set(list(rankings.get(user, []))[:k])
        for item in rel:
            if item in top:
                out["hit"] += 1
            elif item in cands:
                out["ranking_failure"] += 1
            else:
                out["retrieval_failure"] += 1
    return out
