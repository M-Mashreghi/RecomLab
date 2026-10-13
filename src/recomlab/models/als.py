"""Implicit-feedback matrix factorization (weighted ALS) baseline.

Objective (Hu, Koren & Volinsky, 2008):

    min_{X,Y}  sum_{u,i} c_ui (p_ui - x_u . y_i)^2 + reg (|X|^2 + |Y|^2)

with preference ``p_ui = 1`` for observed (user, item) pairs and 0 otherwise,
and confidence ``c_ui = 1 + alpha * r_ui`` where ``r_ui`` is the interaction
strength. Feedback assumption: by default every training event is an implicit
positive signal of strength 1 (aggregated per pair); passing a
:class:`LabelPolicy` instead keeps only label-positive events with their
relevance as strength.

Training (:class:`ImplicitALS.fit`) produces an immutable
:class:`ALSArtifact`; serving (:class:`ALSRecommender`) only reads the
artifact, so training and scoring are separate code paths.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from recomlab.datasets.labels import LabelPolicy, derive_labels
from recomlab.domain.eligibility import EligibilityFilter
from recomlab.domain.schemas import Interaction, Item
from recomlab.experiments.seed import make_rng
from recomlab.models.base import NotFittedError, ScoredItem, assert_train_horizon, top_k


@dataclass(frozen=True)
class ALSParams:
    factors: int = 8
    regularization: float = 0.1
    alpha: float = 10.0
    iterations: int = 15
    seed: int = 0


@dataclass(frozen=True)
class ALSArtifact:
    params: ALSParams
    user_ids: tuple[str, ...]
    item_ids: tuple[str, ...]
    user_factors: np.ndarray
    item_factors: np.ndarray
    label_version: str | None = None

    @property
    def version(self) -> str:
        h = hashlib.sha256()
        h.update(json.dumps(asdict(self.params), sort_keys=True).encode())
        h.update("|".join(self.user_ids).encode())
        h.update("|".join(self.item_ids).encode())
        h.update(np.ascontiguousarray(self.item_factors.round(8)).tobytes())
        return f"als-{h.hexdigest()[:12]}"

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "factors.npz", users=self.user_factors, items=self.item_factors)
        meta = {
            "params": asdict(self.params),
            "user_ids": list(self.user_ids),
            "item_ids": list(self.item_ids),
            "label_version": self.label_version,
            "version": self.version,
        }
        with (directory / "meta.json").open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)
        return directory

    @classmethod
    def load(cls, directory: Path) -> ALSArtifact:
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        arrays = np.load(directory / "factors.npz")
        artifact = cls(
            params=ALSParams(**meta["params"]),
            user_ids=tuple(meta["user_ids"]),
            item_ids=tuple(meta["item_ids"]),
            user_factors=arrays["users"],
            item_factors=arrays["items"],
            label_version=meta["label_version"],
        )
        if artifact.version != meta["version"]:
            raise ValueError("ALS artifact version mismatch: files modified or incompatible")
        return artifact


class ImplicitALS:
    """Trainer: interactions -> :class:`ALSArtifact`."""

    def __init__(self, params: ALSParams | None = None, label_policy: LabelPolicy | None = None):
        self.params = params or ALSParams()
        self.label_policy = label_policy

    def _strengths(self, train: Sequence[Interaction]) -> dict[tuple[str, str], float]:
        out: dict[tuple[str, str], float] = {}
        if self.label_policy is None:
            for e in train:
                out[(e.user_id, e.item_id)] = out.get((e.user_id, e.item_id), 0.0) + 1.0
        else:
            for lab in derive_labels(train, self.label_policy):
                if lab.positive:
                    key = (lab.user_id, lab.item_id)
                    out[key] = out.get(key, 0.0) + lab.relevance
        return out

    def fit(self, train: Sequence[Interaction], items: Sequence[Item]) -> ALSArtifact:
        p = self.params
        strengths = self._strengths(train)
        user_ids = tuple(sorted({u for u, _ in strengths}))
        item_ids = tuple(sorted({i.item_id for i in items} | {i for _, i in strengths}))
        uidx = {u: n for n, u in enumerate(user_ids)}
        iidx = {i: n for n, i in enumerate(item_ids)}
        n_u, n_i, f = len(user_ids), len(item_ids), p.factors
        conf = np.zeros((n_u, n_i))
        for (u, i), r in strengths.items():
            conf[uidx[u], iidx[i]] = p.alpha * r
        pref = (conf > 0).astype(float)
        rng = make_rng(p.seed, "als-init")
        x = rng.normal(0, 0.01, (n_u, f))
        y = rng.normal(0, 0.01, (n_i, f))
        reg = p.regularization * np.eye(f)
        for _ in range(p.iterations):
            x = self._solve(y, conf, pref, reg)
            y = self._solve(x, conf.T, pref.T, reg)
        return ALSArtifact(
            params=p,
            user_ids=user_ids,
            item_ids=item_ids,
            user_factors=x,
            item_factors=y,
            label_version=self.label_policy.label_version if self.label_policy else None,
        )

    @staticmethod
    def _solve(
        fixed: np.ndarray, conf: np.ndarray, pref: np.ndarray, reg: np.ndarray
    ) -> np.ndarray:
        gram = fixed.T @ fixed
        out = np.zeros((conf.shape[0], fixed.shape[1]))
        for row in range(conf.shape[0]):
            c = conf[row]  # c_ui - 1
            a = gram + (fixed.T * c) @ fixed + reg
            b = (fixed.T * (1.0 + c)) @ pref[row]
            out[row] = np.linalg.solve(a, b)
        return out


class ALSRecommender:
    """Serving wrapper: scores from a trained artifact only."""

    name = "als"

    def __init__(self, params: ALSParams | None = None, label_policy: LabelPolicy | None = None):
        self.trainer = ImplicitALS(params, label_policy)
        self.artifact: ALSArtifact | None = None
        self._uidx: dict[str, int] = {}
        self._iidx: dict[str, int] = {}

    @property
    def version(self) -> str:
        return self.artifact.version if self.artifact else "unfitted"

    def fit(
        self, train: Sequence[Interaction], items: Sequence[Item], as_of: datetime | None = None
    ) -> ALSRecommender:
        assert_train_horizon(train, as_of)
        return self.load_artifact(self.trainer.fit(train, items))

    def load_artifact(self, artifact: ALSArtifact) -> ALSRecommender:
        self.artifact = artifact
        self._uidx = {u: n for n, u in enumerate(artifact.user_ids)}
        self._iidx = {i: n for n, i in enumerate(artifact.item_ids)}
        return self

    def knows_user(self, user_id: str) -> bool:
        return user_id in self._uidx

    def score(self, user_id: str, item_ids: Sequence[str]) -> dict[str, float | None]:
        """Score items; ``None`` for unknown users/items (callers apply a fallback)."""
        if self.artifact is None:
            raise NotFittedError(self.name)
        if user_id not in self._uidx:
            return {i: None for i in item_ids}
        u = self.artifact.user_factors[self._uidx[user_id]]
        out: dict[str, float | None] = {}
        for i in item_ids:
            j = self._iidx.get(i)
            out[i] = None if j is None else float(self.artifact.item_factors[j] @ u)
        return out

    def recommend(self, user_id: str, k: int, eligibility: EligibilityFilter) -> list[ScoredItem]:
        scores = self.score(user_id, eligibility.eligible_items(user_id))
        known = {i: s for i, s in scores.items() if s is not None}
        return top_k(known, k)

    def params(self) -> dict[str, Any]:
        return {
            **asdict(self.trainer.params),
            "feedback": "label_positive" if self.trainer.label_policy else "all_events_implicit",
        }
