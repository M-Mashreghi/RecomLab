import sys

import numpy as np
import pytest

from recomlab.candidates import CandidateRequest, validate_candidate_set
from recomlab.candidates.embedding import (
    ALSEmbeddingProvider,
    CategoryEmbeddingProvider,
    EmbeddingCandidates,
    EmbeddingTable,
    ExactIndex,
    IndexVersionMismatch,
)
from recomlab.models.als import ALSParams, ImplicitALS

TABLE = EmbeddingTable(
    ids=("a", "b", "c", "d"),
    vectors=np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]]),
    version="emb-test-1",
)


def test_exact_index_expected_neighbors():
    idx = ExactIndex(TABLE, "cosine")
    hits = idx.query(np.array([1.0, 0.0]), 3)
    assert [h[0] for h in hits] == ["a", "b", "c"]
    assert hits[0][1] == pytest.approx(1.0)
    assert hits[1][1] == pytest.approx(0.9 / np.hypot(0.9, 0.1))
    dot = ExactIndex(TABLE, "dot").query(np.array([0.0, 2.0]), 2)
    assert dot == [("c", 2.0), ("b", pytest.approx(0.2))]
    # deterministic tie-break by id
    tie = ExactIndex(EmbeddingTable(("y", "x"), np.ones((2, 2)), "v"), "dot")
    assert [i for i, _ in tie.query(np.ones(2), 2)] == ["x", "y"]
    assert idx.metadata.embedding_version == "emb-test-1" and idx.metadata.dim == 2


def test_category_embeddings_on_tiny_fixture(tiny_ds, tiny_split):
    provider = CategoryEmbeddingProvider(tiny_ds.items, tiny_split.train)
    gen = EmbeddingCandidates(provider, ExactIndex(provider.item_table())).fit(
        tiny_split.train, tiny_ds.items
    )
    cs = gen.generate(CandidateRequest("u5", 2))  # mostly-drama history (plus one action click)
    validate_candidate_set(cs)
    cats = {i.item_id: set(i.categories) for i in tiny_ds.items}
    assert all(cats[i] & {"drama", "romance"} for i in cs.item_ids)
    assert gen.generate(CandidateRequest("u7", 3)).metadata["fallback_required"]


def test_als_embedding_provider(tiny_ds, tiny_split):
    art = ImplicitALS(ALSParams(factors=4, iterations=5)).fit(tiny_split.train, tiny_ds.items)
    provider = ALSEmbeddingProvider(art)
    gen = EmbeddingCandidates(provider, ExactIndex(provider.item_table(), "dot")).fit(
        tiny_split.train, tiny_ds.items
    )
    cs = gen.generate(CandidateRequest("u1", 4))
    consumed = {e.item_id for e in tiny_split.train if e.user_id == "u1"}
    assert len(cs.item_ids) == 4 and not set(cs.item_ids) & consumed
    assert provider.version in gen.version


def test_index_model_version_mismatch_detected(tiny_ds, tiny_split):
    art1 = ImplicitALS(ALSParams(factors=4, iterations=3, seed=0)).fit(
        tiny_split.train, tiny_ds.items
    )
    art2 = ImplicitALS(ALSParams(factors=4, iterations=3, seed=1)).fit(
        tiny_split.train, tiny_ds.items
    )
    stale_index = ExactIndex(ALSEmbeddingProvider(art1).item_table())
    with pytest.raises(IndexVersionMismatch):
        EmbeddingCandidates(ALSEmbeddingProvider(art2), stale_index)


def test_core_needs_no_vector_infrastructure():
    for mod in ("faiss", "hnswlib", "annoy", "pinecone", "qdrant_client"):
        assert mod not in sys.modules


def test_invalid_table_and_metric():
    with pytest.raises(ValueError):
        EmbeddingTable(("a",), np.zeros((2, 2)), "v")
    with pytest.raises(ValueError):
        ExactIndex(TABLE, "l7")
