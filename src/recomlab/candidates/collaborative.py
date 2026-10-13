"""Collaborative (ALS latent factor) candidate generator."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from recomlab.candidates.base import CandidateGenerator, CandidateRequest, CandidateSet
from recomlab.domain.schemas import Interaction, Item
from recomlab.models.als import ALSArtifact, ALSParams, ALSRecommender, ImplicitALS


class CollaborativeCandidates(CandidateGenerator):
    """Top-N items by ``x_u . y_i`` from an ALS artifact.

    The artifact is trained on the visible history in :meth:`fit` (or injected
    with :meth:`from_artifact`). ``exclude_consumed`` removes items the user
    interacted with in that history.
    """

    name = "collaborative"

    def __init__(self, params: ALSParams | None = None, exclude_consumed: bool = True) -> None:
        super().__init__()
        self.params = params or ALSParams()
        self.exclude_consumed = exclude_consumed
        self.model = ALSRecommender(self.params)
        self.consumed: dict[str, set[str]] = {}

    @property
    def version(self) -> str:
        return self.model.version

    @classmethod
    def from_artifact(
        cls,
        artifact: ALSArtifact,
        history: Sequence[Interaction],
        items: Sequence[Item],
        exclude_consumed: bool = True,
    ) -> CollaborativeCandidates:
        gen = cls(artifact.params, exclude_consumed)
        gen.catalog = tuple(sorted(i.item_id for i in items))
        gen.model.load_artifact(artifact)
        gen._index_history(history)
        gen._fitted = True
        return gen

    def _index_history(self, history: Sequence[Interaction]) -> None:
        self.consumed = {}
        for e in history:
            self.consumed.setdefault(e.user_id, set()).add(e.item_id)

    def _fit(self, history: Sequence[Interaction], items: Sequence[Item]) -> None:
        self.model.load_artifact(ImplicitALS(self.params).fit(history, items))
        self._index_history(history)

    def _score(
        self, request: CandidateRequest
    ) -> tuple[dict[str, float], dict[str, dict[str, Any]], dict[str, Any]]:
        if not self.model.knows_user(request.user_id):
            return {}, {}, {"fallback_required": True, "reason": "unknown_user"}
        raw = self.model.score(request.user_id, self.catalog)
        seen = self.consumed.get(request.user_id, set()) if self.exclude_consumed else set()
        scores = {i: s for i, s in raw.items() if s is not None and i not in seen}
        meta = {"fallback_required": False, "model_version": self.model.version}
        return scores, {}, meta

    def generate_batch(self, requests: Iterable[CandidateRequest]) -> dict[str, CandidateSet]:
        """Vectorized retrieval for evaluation; equals per-request :meth:`generate`."""
        reqs = list(requests)
        art = self.model.artifact
        if art is None:
            return {r.user_id: self.generate(r) for r in reqs}
        known = [r for r in reqs if self.model.knows_user(r.user_id)]
        out = {r.user_id: self.generate(r) for r in reqs if r not in known}
        if not known:
            return out
        uidx = {u: n for n, u in enumerate(art.user_ids)}
        iidx = {i: n for n, i in enumerate(art.item_ids)}
        cols = [i for i in self.catalog if i in iidx]
        users = art.user_factors[[uidx[r.user_id] for r in known]]
        items = art.item_factors[[iidx[i] for i in cols]]
        matrix = users @ items.T
        meta = {"fallback_required": False, "model_version": self.model.version}
        for row, req in zip(np.asarray(matrix), known, strict=True):
            start = time.perf_counter()
            seen = self.consumed.get(req.user_id, set()) if self.exclude_consumed else set()
            scores = {i: float(v) for i, v in zip(cols, row, strict=True) if i not in seen}
            out[req.user_id] = self._assemble(req, scores, {}, meta, start)
        return out
