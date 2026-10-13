"""Learning-to-rank dataset builder.

One query group per user. For a group built at ``as_of``:

* features and candidates come only from ``history`` events strictly before
  ``as_of`` (point-in-time);
* labels come only from ``target`` events at/after ``as_of`` (the target window).

Training groups use an *inner* cutoff inside the training partition (both
history and target are training events); validation groups use train history
and validation targets; test groups are for final evaluation only. The builder
verifies the timestamp ordering and refuses targets drawn from a forbidden
partition via ``forbidden_event_ids``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from recomlab.candidates import CandidateRequest, PopularityCandidates
from recomlab.candidates.fusion import FusedCandidates, FusionConfig, fuse
from recomlab.candidates.item_item import ItemItemCandidates
from recomlab.datasets.labels import LabelPolicy, derive_labels, relevance_by_user
from recomlab.datasets.splits import LeakageError
from recomlab.domain.eligibility import EligibilityFilter
from recomlab.domain.schemas import Interaction, Item
from recomlab.features.builder import PointInTimeFeatures
from recomlab.rankers.base import RankRequest
from recomlab.rankers.heuristic import FEATURES as HEURISTIC_FEATURES
from recomlab.rankers.heuristic import HeuristicRanker, HeuristicWeights

LTR_FEATURE_SCHEMA = "ltr-features-1"
CandidateFn = Callable[[str], FusedCandidates]
CandidateFactory = Callable[[Sequence[Interaction], Sequence[Item]], CandidateFn]


@dataclass(frozen=True)
class LTRDataset:
    features: np.ndarray
    labels: np.ndarray
    group_sizes: tuple[int, ...]
    query_ids: tuple[str, ...]
    item_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    as_of: str
    label_version: str
    schema_version: str = LTR_FEATURE_SCHEMA
    skipped_users: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.features.shape[0]
        if sum(self.group_sizes) != n or len(self.labels) != n or len(self.item_ids) != n:
            raise ValueError("group sizes do not reconcile with rows")
        if len(self.query_ids) != len(self.group_sizes):
            raise ValueError("one query id per group required")

    def groups(self) -> Iterable[tuple[str, slice]]:
        start = 0
        for qid, size in zip(self.query_ids, self.group_sizes, strict=True):
            yield qid, slice(start, start + size)
            start += size

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feature_names": list(self.feature_names),
            "as_of": self.as_of,
            "label_version": self.label_version,
            "n_groups": len(self.group_sizes),
            "n_rows": int(self.features.shape[0]),
            "group_sizes": list(self.group_sizes),
            "skipped_users": dict(sorted(self.skipped_users.items())),
        }


def default_candidate_factory(budget: int = 20) -> CandidateFactory:
    """Popularity + item-item candidates fused by RRF, fitted on the given history."""

    def factory(history: Sequence[Interaction], items: Sequence[Item]) -> CandidateFn:
        gens = [
            PopularityCandidates().fit(history, items),
            ItemItemCandidates().fit(history, items),
        ]
        elig = EligibilityFilter(items, history)

        def candidates(user: str) -> FusedCandidates:
            sets = {g.name: g.generate(CandidateRequest(user, budget)) for g in gens}
            return fuse(user, sets, elig, FusionConfig(budget=budget))

        return candidates

    return factory


def feature_names(sources: Sequence[str]) -> tuple[str, ...]:
    return (
        *HEURISTIC_FEATURES,
        "user_history_length",
        "n_sources",
        *(f"src_{s}_rr" for s in sources),
    )


def build_ltr_dataset(
    history: Sequence[Interaction],
    target: Sequence[Interaction],
    items: Sequence[Item],
    as_of: datetime,
    label_policy: LabelPolicy,
    *,
    candidate_factory: CandidateFactory | None = None,
    sources: Sequence[str] = ("item_item", "popularity"),
    forbidden_event_ids: Iterable[str] = (),
    require_positive: bool = True,
) -> LTRDataset:
    late = [e.event_id for e in history if e.event_time >= as_of]
    early = [e.event_id for e in target if e.event_time < as_of]
    if late or early:
        raise LeakageError(f"feature/label time order violated: late={late[:3]} early={early[:3]}")
    forbidden = set(forbidden_event_ids)
    leaked = [e.event_id for e in target if e.event_id in forbidden]
    if leaked:
        raise LeakageError(f"target window contains forbidden (held-out) events: {leaked[:3]}")
    feats = PointInTimeFeatures(history, items, as_of)
    heuristic = HeuristicRanker(feats, HeuristicWeights({f: 1.0 for f in HEURISTIC_FEATURES}))
    candidates_for = (candidate_factory or default_candidate_factory())(history, items)
    truth = relevance_by_user(derive_labels(target, label_policy))
    names = feature_names(sources)
    rows: list[list[float]] = []
    labels: list[float] = []
    item_ids: list[str] = []
    sizes: list[int] = []
    qids: list[str] = []
    skipped: dict[str, str] = {}
    for user in sorted({e.user_id for e in target}):
        rel = truth.get(user, {})
        if require_positive and not rel:
            skipped[user] = "no_positive_target"
            continue
        fused = candidates_for(user)
        if not fused.candidates:
            skipped[user] = "no_candidates"
            continue
        req = RankRequest.from_fused(fused, k=len(fused.candidates))
        xs = heuristic.feature_vector(req)
        hist_len = feats.user_features(user)["history_length"] or 0.0
        for c in fused.candidates:
            src_rr = [1.0 / float(c.sources[s]["rank"]) if s in c.sources else 0.0 for s in sources]
            rows.append(
                [*(xs[c.item_id][f] for f in HEURISTIC_FEATURES), hist_len, len(c.sources), *src_rr]
            )
            labels.append(rel.get(c.item_id, 0.0))
            item_ids.append(c.item_id)
        sizes.append(len(fused.candidates))
        qids.append(user)
    return LTRDataset(
        features=np.array(rows, dtype=float).reshape(len(rows), len(names)),
        labels=np.array(labels, dtype=float),
        group_sizes=tuple(sizes),
        query_ids=tuple(qids),
        item_ids=tuple(item_ids),
        feature_names=names,
        as_of=as_of.isoformat(),
        label_version=label_policy.label_version,
        skipped_users=skipped,
    )


def inner_training_cutoff(train: Sequence[Interaction], quantile: float = 0.8) -> datetime:
    """Cutoff inside the training partition: features before it, labels after it."""
    times = sorted(e.event_time for e in train)
    if len(times) < 2:
        raise ValueError("not enough training events for an inner cutoff")
    return times[min(len(times) - 1, max(1, int(len(times) * quantile)))]
