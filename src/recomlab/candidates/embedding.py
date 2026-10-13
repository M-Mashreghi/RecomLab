"""Embedding retrieval with a pluggable nearest-neighbor index.

The mandatory index is :class:`ExactIndex` (brute force, deterministic). ANN
backends are optional adapters (e.g. :class:`HnswlibIndex`) and are never
needed by core tests. Every index records the embedding version it was built
from; querying with a different embedding version fails closed.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from recomlab.candidates.base import CandidateGenerator, CandidateRequest
from recomlab.domain.schemas import Interaction, Item
from recomlab.models.als import ALSArtifact

METRICS = ("cosine", "dot")


class IndexVersionMismatch(RuntimeError):
    pass


class OptionalDependencyError(ImportError):
    pass


@dataclass(frozen=True)
class EmbeddingTable:
    ids: tuple[str, ...]
    vectors: np.ndarray
    version: str

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2 or self.vectors.shape[0] != len(self.ids):
            raise ValueError("vectors must be (n_items, dim) aligned with ids")


class EmbeddingProvider(Protocol):
    @property
    def version(self) -> str: ...

    def item_table(self) -> EmbeddingTable: ...

    def user_vector(self, user_id: str) -> np.ndarray | None: ...


class ALSEmbeddingProvider:
    """Latent factors from a trained ALS artifact."""

    def __init__(self, artifact: ALSArtifact) -> None:
        self.artifact = artifact
        self._uidx = {u: n for n, u in enumerate(artifact.user_ids)}

    @property
    def version(self) -> str:
        return f"emb-{self.artifact.version}"

    def item_table(self) -> EmbeddingTable:
        return EmbeddingTable(self.artifact.item_ids, self.artifact.item_factors, self.version)

    def user_vector(self, user_id: str) -> np.ndarray | None:
        n = self._uidx.get(user_id)
        return None if n is None else self.artifact.user_factors[n]


class CategoryEmbeddingProvider:
    """Content embeddings: L2-normalized multi-hot item categories.

    A user's vector is the mean of the embeddings of items in their history.
    """

    def __init__(self, items: Sequence[Item], history: Sequence[Interaction]) -> None:
        self.categories = tuple(sorted({c for i in items for c in i.categories}))
        col = {c: n for n, c in enumerate(self.categories)}
        ids = tuple(sorted(i.item_id for i in items))
        by_id = {i.item_id: i for i in items}
        mat = np.zeros((len(ids), len(self.categories)))
        for r, iid in enumerate(ids):
            for c in by_id[iid].categories:
                mat[r, col[c]] = 1.0
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = np.divide(mat, norms, out=np.zeros_like(mat), where=norms > 0)
        self._table = EmbeddingTable(ids, mat, f"emb-cat-{len(self.categories)}-{len(ids)}")
        self._row = {iid: r for r, iid in enumerate(ids)}
        self.history: dict[str, list[str]] = {}
        for e in history:
            self.history.setdefault(e.user_id, []).append(e.item_id)

    @property
    def version(self) -> str:
        return self._table.version

    def item_table(self) -> EmbeddingTable:
        return self._table

    def user_vector(self, user_id: str) -> np.ndarray | None:
        rows = [self._row[i] for i in self.history.get(user_id, []) if i in self._row]
        if not rows:
            return None
        vec: np.ndarray = self._table.vectors[rows].mean(axis=0)
        return vec


@dataclass(frozen=True)
class IndexMetadata:
    index_type: str
    embedding_version: str
    metric: str
    n_items: int
    dim: int
    params: dict[str, Any]


class NearestNeighborIndex(Protocol):
    metadata: IndexMetadata

    def query(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return np.divide(m, norms, out=np.zeros_like(m, dtype=float), where=norms > 0)


class ExactIndex:
    """Brute-force exact search; the mandatory reference implementation."""

    def __init__(self, table: EmbeddingTable, metric: str = "cosine") -> None:
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        self.ids = table.ids
        self.metric = metric
        self.matrix = _normalize(table.vectors) if metric == "cosine" else table.vectors
        self.metadata = IndexMetadata(
            "exact", table.version, metric, len(table.ids), int(table.vectors.shape[1]), {}
        )

    def query(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        q = _normalize(vector) if self.metric == "cosine" else vector
        scores = self.matrix @ q
        order = sorted(range(len(self.ids)), key=lambda n: (-float(scores[n]), self.ids[n]))
        return [(self.ids[n], float(scores[n])) for n in order[:k]]


class HnswlibIndex:  # pragma: no cover - optional adapter, not exercised in mandatory CI
    """Optional ANN adapter (requires ``hnswlib``; see dependency policy)."""

    def __init__(self, table: EmbeddingTable, metric: str = "cosine", ef: int = 100) -> None:
        if importlib.util.find_spec("hnswlib") is None:
            raise OptionalDependencyError("hnswlib is not installed; use ExactIndex")
        import hnswlib

        space = "cosine" if metric == "cosine" else "ip"
        self.ids = table.ids
        self._index = hnswlib.Index(space=space, dim=table.vectors.shape[1])
        self._index.init_index(max_elements=len(table.ids), ef_construction=ef, M=16)
        self._index.add_items(table.vectors, np.arange(len(table.ids)))
        self._index.set_ef(ef)
        self.metadata = IndexMetadata(
            "hnswlib", table.version, metric, len(table.ids), table.vectors.shape[1], {"ef": ef}
        )

    def query(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        labels, dists = self._index.knn_query(vector, k=min(k, len(self.ids)))
        return [
            (self.ids[int(i)], 1.0 - float(d)) for i, d in zip(labels[0], dists[0], strict=True)
        ]


class EmbeddingCandidates(CandidateGenerator):
    """Retrieve items nearest to the user's embedding."""

    name = "embedding"

    def __init__(self, provider: EmbeddingProvider, index: NearestNeighborIndex) -> None:
        super().__init__()
        if index.metadata.embedding_version != provider.version:
            raise IndexVersionMismatch(
                f"index built from {index.metadata.embedding_version}, "
                f"provider is {provider.version}"
            )
        self.provider = provider
        self.index = index

    @property
    def version(self) -> str:
        return f"embedding-{self.index.metadata.index_type}-{self.provider.version}"

    def _fit(self, history: Sequence[Interaction], items: Sequence[Item]) -> None:
        self.consumed: dict[str, set[str]] = {}
        for e in history:
            self.consumed.setdefault(e.user_id, set()).add(e.item_id)

    def _score(
        self, request: CandidateRequest
    ) -> tuple[dict[str, float], dict[str, dict[str, Any]], dict[str, Any]]:
        if self.index.metadata.embedding_version != self.provider.version:
            raise IndexVersionMismatch("embedding version changed after index build")
        vec = self.provider.user_vector(request.user_id)
        if vec is None:
            return {}, {}, {"fallback_required": True, "reason": "no_user_embedding"}
        seen = self.consumed.get(request.user_id, set())
        k = request.budget + len(seen) + len(request.exclude_items)
        hits = [(i, s) for i, s in self.index.query(vec, k) if i not in seen]
        meta = {
            "fallback_required": False,
            "index_type": self.index.metadata.index_type,
            "embedding_version": self.provider.version,
        }
        return dict(hits), {}, meta
