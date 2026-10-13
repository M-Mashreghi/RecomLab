"""Candidate-stage (retrieval) metrics, measured before any ranking."""

from __future__ import annotations

from collections.abc import Mapping

from recomlab.candidates.base import CandidateSet


def candidate_recall(candidates: CandidateSet, relevant: Mapping[str, float]) -> float:
    """Recall@CandidateN: share of the user's relevant items present in the candidate set."""
    if not relevant:
        raise ValueError("candidate recall undefined for users without relevant items")
    ids = set(candidates.item_ids)
    return sum(1 for i in relevant if i in ids) / len(relevant)


def candidate_hit(candidates: CandidateSet, relevant: Mapping[str, float]) -> float:
    ids = set(candidates.item_ids)
    return 1.0 if any(i in ids for i in relevant) else 0.0
