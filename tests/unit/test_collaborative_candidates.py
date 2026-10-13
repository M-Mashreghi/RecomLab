import pytest

from recomlab.candidates import CandidateRequest, validate_candidate_set
from recomlab.candidates.collaborative import CollaborativeCandidates
from recomlab.datasets.labels import LabelPolicy
from recomlab.evaluation.offline import holdout_truth
from recomlab.evaluation.retrieval import candidate_hit, candidate_recall
from recomlab.models.als import ALSParams, ImplicitALS

FAST = ALSParams(factors=4, iterations=8, seed=0)


@pytest.fixture
def gen(tiny_ds, tiny_split):
    return CollaborativeCandidates(FAST).fit(tiny_split.train, tiny_ds.items)


def test_consumed_items_filtered(gen, tiny_split, tiny_ds):
    consumed = {e.item_id for e in tiny_split.train if e.user_id == "u1"}
    cs = gen.generate(CandidateRequest("u1", 12))
    validate_candidate_set(cs)
    assert not set(cs.item_ids) & consumed
    keep = CollaborativeCandidates(FAST, exclude_consumed=False).fit(
        tiny_split.train, tiny_ds.items
    )
    assert consumed <= set(keep.generate(CandidateRequest("u1", 12)).item_ids)


def test_exposes_score_and_model_version(gen):
    cs = gen.generate(CandidateRequest("u2", 3))
    assert cs.source_version.startswith("als-")
    assert cs.metadata["model_version"] == cs.source_version
    scores = [c.score for c in cs.candidates]
    assert scores == sorted(scores, reverse=True)


def test_batch_and_single_agree(gen):
    reqs = [CandidateRequest(u, 4) for u in ("u1", "u2", "u3", "u4", "u5", "u7")]
    batch = gen.generate_batch(reqs)
    for r in reqs:
        single = gen.generate(r)
        assert batch[r.user_id].item_ids == single.item_ids
        for a, b in zip(batch[r.user_id].candidates, single.candidates, strict=True):
            assert a.score == pytest.approx(b.score)
    assert batch["u7"].is_empty and batch["u7"].metadata["fallback_required"]


def test_from_artifact_matches_fit(tiny_ds, tiny_split, gen):
    art = ImplicitALS(FAST).fit(tiny_split.train, tiny_ds.items)
    other = CollaborativeCandidates.from_artifact(art, tiny_split.train, tiny_ds.items)
    req = CandidateRequest("u3", 5)
    assert other.generate(req).item_ids == gen.generate(req).item_ids


def test_candidate_recall_evaluated_independently(gen, tiny_split):
    truth = holdout_truth(tiny_split, "test", LabelPolicy())
    recalls = {}
    for user, rel in truth.items():
        cs = gen.generate(CandidateRequest(user, 12))
        recalls[user] = candidate_recall(cs, rel)
        assert candidate_hit(cs, rel) == (1.0 if recalls[user] > 0 else 0.0)
    assert recalls and all(0.0 <= r <= 1.0 for r in recalls.values())
    # With a budget covering the whole eligible catalog, every non-consumed truth item is found.
    assert all(r == 1.0 for r in recalls.values())
    with pytest.raises(ValueError):
        candidate_recall(gen.generate(CandidateRequest("u1", 2)), {})
