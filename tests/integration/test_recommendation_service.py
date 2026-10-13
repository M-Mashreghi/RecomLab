import json

import pytest

from recomlab.candidates import CandidateRequest, PopularityCandidates
from recomlab.candidates.collaborative import CollaborativeCandidates
from recomlab.candidates.fusion import FusionConfig, fuse
from recomlab.candidates.item_item import ItemItemCandidates
from recomlab.domain.eligibility import EligibilityFilter
from recomlab.features import PointInTimeFeatures
from recomlab.models.als import ALSArtifact
from recomlab.rankers import RankRequest
from recomlab.rankers.heuristic import HeuristicRanker
from recomlab.registry.model_registry import IncompatibleBundleError, ModelRegistry
from recomlab.reranking import ColdStartPolicy
from recomlab.services.recommendation import (
    NoActiveModelError,
    RecommendationRequest,
    RecommendationService,
)

pytestmark = pytest.mark.integration


def test_active_model_swap(tmp_path, bundle_factory, serving_context):
    items, history, as_of = serving_context
    reg = ModelRegistry(tmp_path / "registry")
    b1 = bundle_factory(reg, ranker="heuristic")
    b2 = bundle_factory(reg, ranker="collaborative")
    svc = RecommendationService(reg, items, history, as_of)
    with pytest.raises(NoActiveModelError):
        svc.recommend(RecommendationRequest("u1", 3))
    svc.activate(b1.bundle_id, "initial")
    r1 = svc.recommend(RecommendationRequest("u1", 3, trace_id="t1"))
    assert r1.bundle_id == b1.bundle_id and r1.trace_id == "t1"
    svc.activate(b2.bundle_id, "swap")
    r2 = svc.recommend(RecommendationRequest("u1", 3, trace_id="t2"))
    assert r2.bundle_id == b2.bundle_id
    assert r2.component_versions["ranker"].startswith("collaborative")
    assert reg.active_bundle_id == b2.bundle_id
    assert [h["bundle_id"] for h in reg.history()] == [b1.bundle_id, b2.bundle_id]
    svc.rollback("regression")
    assert svc.bundle_id == b1.bundle_id and reg.history()[-1]["event"] == "rollback"
    restarted = RecommendationService(reg, items, history, as_of)
    assert restarted.bundle_id == b1.bundle_id


def test_incompatible_artifacts_fail_closed(tmp_path, bundle_factory, serving_context):
    items, history, as_of = serving_context
    reg = ModelRegistry(tmp_path / "registry")
    good = bundle_factory(reg)
    svc = RecommendationService(reg, items, history, as_of)
    svc.activate(good.bundle_id)
    bad = bundle_factory(reg, ranker="collaborative", seed=1)
    meta = bad.artifacts_dir / "als" / "meta.json"
    meta.write_text(meta.read_text().replace('"factors": 4', '"factors": 5'))
    with pytest.raises(IncompatibleBundleError):
        svc.activate(bad.bundle_id)
    assert reg.active_bundle_id == good.bundle_id and svc.bundle_id == good.bundle_id
    smaller_catalog = items[:-1]
    with pytest.raises(IncompatibleBundleError):
        reg.activate(good.bundle_id, smaller_catalog, history, as_of)
    with pytest.raises(IncompatibleBundleError):
        reg.register_bundle({"generators": ["magic"], "ranker": {"type": "heuristic"}}, {})
    with pytest.raises(IncompatibleBundleError):
        reg.rollback(items, history, as_of, "nothing before")


def test_service_matches_offline_ranker(tmp_path, bundle_factory, serving_context):
    items, history, as_of = serving_context
    reg = ModelRegistry(tmp_path / "registry")
    bundle = bundle_factory(reg)
    svc = RecommendationService(reg, items, history, as_of)
    svc.activate(bundle.bundle_id)
    art = ALSArtifact.load(bundle.artifacts_dir / "als")
    gens = [
        PopularityCandidates().fit(history, items),
        ItemItemCandidates().fit(history, items),
        CollaborativeCandidates.from_artifact(art, history, items),
    ]
    feats = PointInTimeFeatures(history, items, as_of)
    elig = EligibilityFilter(items, history)
    for user in ("u1", "u2", "u4", "u7"):
        sets = {g.name: g.generate(CandidateRequest(user, 8)) for g in gens}
        fused = fuse(user, sets, elig, FusionConfig(budget=8))
        ranked = HeuristicRanker(feats).rank(RankRequest.from_fused(fused, 4))
        offline = ColdStartPolicy().apply(ranked, 4, elig, feats)
        online = svc.recommend(RecommendationRequest(user, 4))
        assert online.item_ids == offline.item_ids, user
        assert [i.score for i in online.items] == pytest.approx(
            [i.final_score for i in offline.items]
        )


def test_response_provenance_is_serializable(tmp_path, bundle_factory, serving_context):
    items, history, as_of = serving_context
    reg = ModelRegistry(tmp_path / "registry")
    svc = RecommendationService(reg, items, history, as_of)
    svc.activate(bundle_factory(reg).bundle_id)
    resp = svc.recommend(RecommendationRequest("u2", 3))
    data = json.loads(json.dumps(resp.to_dict()))
    assert data["items"][0]["sources"] and data["items"][0]["contributions"]
    cold = svc.recommend(RecommendationRequest("u7", 3))
    assert len(cold.items) == 3
    with pytest.raises(ValueError):
        RecommendationRequest("u1", 0)
