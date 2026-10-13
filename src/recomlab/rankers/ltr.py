"""Pairwise learning-to-rank baseline (linear RankNet), pure NumPy.

For every pair (i, j) in a query group with ``label_i > label_j`` the model
minimizes ``log(1 + exp(-(s_i - s_j)))`` with ``s = w . z(x)``, where ``z``
standardizes features with training-set statistics. Optimization is full-batch
gradient descent with L2 regularization; it is deterministic for a fixed
config. Hyperparameters are tuned on a *validation* dataset only:
:func:`tune_on_validation` refuses datasets tagged as ``test``.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from recomlab.features.builder import PointInTimeFeatures
from recomlab.rankers.base import Ranker, RankRequest, Scores
from recomlab.rankers.ltr_data import LTRDataset, candidate_feature_rows, unit_heuristic


class HeldOutTuningError(RuntimeError):
    pass


@dataclass(frozen=True)
class LTRParams:
    learning_rate: float = 0.1
    l2: float = 0.01
    epochs: int = 200


@dataclass(frozen=True)
class LTRModel:
    params: LTRParams
    feature_names: tuple[str, ...]
    weights: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    schema_version: str
    label_version: str
    training_pairs: int
    validation: dict[str, Any] = field(default_factory=dict)

    def score_matrix(self, x: np.ndarray) -> np.ndarray:
        out: np.ndarray = ((x - self.mean) / self.std) @ self.weights
        return out

    def feature_importance(self) -> dict[str, float]:
        """Normalized |w| on standardized features (sums to 1)."""
        mag = np.abs(self.weights)
        total = float(mag.sum()) or 1.0
        return {n: float(m / total) for n, m in zip(self.feature_names, mag, strict=True)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": asdict(self.params),
            "feature_names": list(self.feature_names),
            "weights": self.weights.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "schema_version": self.schema_version,
            "label_version": self.label_version,
            "training_pairs": self.training_pairs,
            "feature_importance": self.feature_importance(),
            "validation": self.validation,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def _pairs(ds: LTRDataset) -> list[tuple[int, int]]:
    pairs = []
    for _, sl in ds.groups():
        idx = range(sl.start, sl.stop)
        for i, j in itertools.permutations(idx, 2):
            if ds.labels[i] > ds.labels[j]:
                pairs.append((i, j))
    return pairs


def train_ltr(ds: LTRDataset, params: LTRParams | None = None) -> LTRModel:
    if ds.partition == "test":
        raise HeldOutTuningError("refusing to train on a test-partition dataset")
    p = params or LTRParams()
    x = ds.features
    mean = x.mean(axis=0) if len(x) else np.zeros(x.shape[1])
    std = x.std(axis=0) if len(x) else np.ones(x.shape[1])
    std = np.where(std > 0, std, 1.0)
    z = (x - mean) / std
    pairs = _pairs(ds)
    w = np.zeros(x.shape[1])
    if pairs:
        hi = np.array([i for i, _ in pairs])
        lo = np.array([j for _, j in pairs])
        diff = z[hi] - z[lo]
        for _ in range(p.epochs):
            margin = diff @ w
            grad_coef = -1.0 / (1.0 + np.exp(margin))  # d/dm log(1+e^-m)
            grad = (diff * grad_coef[:, None]).mean(axis=0) + p.l2 * w
            w = w - p.learning_rate * grad
    return LTRModel(
        params=p,
        feature_names=ds.feature_names,
        weights=w,
        mean=mean,
        std=std,
        schema_version=ds.schema_version,
        label_version=ds.label_version,
        training_pairs=len(pairs),
    )


def group_ndcg(ds: LTRDataset, scores: np.ndarray, k: int) -> float:
    """Mean NDCG@k over groups with at least one positive (ties broken by item id)."""
    values = []
    for _, sl in ds.groups():
        labels = ds.labels[sl]
        if labels.max(initial=0.0) <= 0:
            continue
        items = ds.item_ids[sl]
        order = sorted(range(len(labels)), key=lambda n: (-float(scores[sl][n]), items[n]))
        dcg = sum(labels[n] / math.log2(r + 2) for r, n in enumerate(order[:k]))
        ideal = sorted(labels, reverse=True)[:k]
        idcg = sum(g / math.log2(r + 2) for r, g in enumerate(ideal))
        values.append(dcg / idcg if idcg else 0.0)
    return float(np.mean(values)) if values else 0.0


def tune_on_validation(
    train: LTRDataset,
    validation: LTRDataset,
    grid: Sequence[LTRParams],
    k: int = 5,
) -> tuple[LTRModel, list[dict[str, Any]]]:
    """Grid search: fit on ``train``, select by validation NDCG@k. Test data is refused."""
    for ds in (train, validation):
        if ds.partition == "test":
            raise HeldOutTuningError("test data cannot be used for tuning")
    if validation.feature_names != train.feature_names:
        raise ValueError("feature schema mismatch between train and validation")
    trials = []
    best: tuple[float, int, LTRModel] | None = None
    for n, params in enumerate(grid):
        model = train_ltr(train, params)
        score = group_ndcg(validation, model.score_matrix(validation.features), k)
        trials.append({"params": asdict(params), f"val_ndcg@{k}": score})
        if best is None or score > best[0]:
            best = (score, n, model)
    if best is None:
        raise ValueError("empty hyperparameter grid")
    score, _, model = best
    chosen = LTRModel(**{**model.__dict__, "validation": {f"ndcg@{k}": score, "trials": trials}})
    return chosen, trials


class LTRRanker(Ranker):
    """Serves an :class:`LTRModel` with the same feature construction as training."""

    name = "ltr"

    def __init__(
        self,
        model: LTRModel,
        features: PointInTimeFeatures,
        sources: Sequence[str] = ("item_item", "popularity"),
    ) -> None:
        self.model = model
        self.heuristic = unit_heuristic(features)
        self.sources = tuple(sources)

    @property
    def version(self) -> str:
        w = ",".join(f"{v:.6f}" for v in self.model.weights)
        return f"ltr-1[{self.model.schema_version}|{w}]"

    def params(self) -> Mapping[str, Any]:
        return self.model.to_dict()

    def _score(self, request: RankRequest) -> Scores:
        rows = candidate_feature_rows(self.heuristic, request, self.sources)
        if not rows:
            return {}
        items = list(rows)
        x = np.array([rows[i] for i in items])
        z = (x - self.model.mean) / self.model.std
        contrib = z * self.model.weights
        out: Scores = {}
        for n, item in enumerate(items):
            parts = {f: float(c) for f, c in zip(self.model.feature_names, contrib[n], strict=True)}
            out[item] = (float(contrib[n].sum()), parts)
        return out
