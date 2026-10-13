"""Two-stage recommendation pipeline assembled from a bundle spec.

retrieval generators -> fusion + eligibility -> ranker -> cold-start policy.
The same pipeline object serves online requests and offline evaluation, so
serving and offline results agree on the same context.
"""

from __future__ import annotations

import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from recomlab.candidates import CandidateGenerator, CandidateRequest, PopularityCandidates
from recomlab.candidates.collaborative import CollaborativeCandidates
from recomlab.candidates.fusion import FusedCandidates, FusionConfig, fuse
from recomlab.candidates.item_item import ItemItemCandidates
from recomlab.domain.eligibility import EligibilityFilter, EligibilityPolicy
from recomlab.domain.schemas import Interaction, Item
from recomlab.features.builder import PointInTimeFeatures
from recomlab.models.als import ALSArtifact, ALSParams, ALSRecommender, ImplicitALS
from recomlab.models.base import ScoredItem
from recomlab.rankers.base import RankedList, Ranker, RankRequest, RetrievalScoreRanker
from recomlab.rankers.collaborative import CollaborativeRanker
from recomlab.rankers.heuristic import HeuristicRanker, HeuristicWeights
from recomlab.reranking.policies import ColdStartPolicy

GENERATORS = ("popularity", "item_item", "collaborative")
RANKERS = ("heuristic", "collaborative", "retrieval_score")


class PipelineConfigError(ValueError):
    pass


def validate_spec(spec: Mapping[str, Any]) -> None:
    gens = spec.get("generators")
    if not gens or not set(gens) <= set(GENERATORS):
        raise PipelineConfigError(f"generators must be a non-empty subset of {GENERATORS}")
    ranker = spec.get("ranker", {}).get("type")
    if ranker not in RANKERS:
        raise PipelineConfigError(f"ranker.type must be one of {RANKERS}")
    needs_als = "collaborative" in gens or ranker == "collaborative"
    if needs_als and "als" not in spec.get("artifacts", {}):
        raise PipelineConfigError("collaborative components need artifacts.als")
    if int(spec.get("candidate_budget", 0)) <= 0:
        raise PipelineConfigError("candidate_budget must be positive")


@dataclass(frozen=True)
class PipelineOutput:
    ranked: RankedList
    fused: FusedCandidates


class RecommendationPipeline:
    def __init__(
        self,
        spec: Mapping[str, Any],
        history: Sequence[Interaction],
        items: Sequence[Item],
        as_of: datetime,
        artifacts_dir: Path | None = None,
    ) -> None:
        validate_spec(spec)
        self.spec = dict(spec)
        self.items = list(items)
        self.history = list(history)
        self.features = PointInTimeFeatures(self.history, self.items, as_of)
        self.eligibility = EligibilityFilter(
            self.items, self.history, EligibilityPolicy(**spec.get("eligibility", {}))
        )
        self.budget = int(spec["candidate_budget"])
        self.fusion = FusionConfig(**{"budget": self.budget, **spec.get("fusion", {})})
        self.als: ALSRecommender | None = None
        if "als" in spec.get("artifacts", {}):
            if artifacts_dir is None:
                raise PipelineConfigError("artifacts_dir required for ALS artifacts")
            art = ALSArtifact.load(Path(artifacts_dir) / spec["artifacts"]["als"])
            self.als = ALSRecommender().load_artifact(art)
        self.generators = self._build_generators()
        self.ranker = self._build_ranker()
        self.cold_start = ColdStartPolicy(**spec.get("cold_start", {}))

    def _build_generators(self) -> list[CandidateGenerator]:
        gens: list[CandidateGenerator] = []
        for name in self.spec["generators"]:
            if name == "popularity":
                gens.append(PopularityCandidates().fit(self.history, self.items))
            elif name == "item_item":
                gens.append(
                    ItemItemCandidates(**self.spec.get("item_item", {})).fit(
                        self.history, self.items
                    )
                )
            else:
                assert self.als is not None and self.als.artifact is not None
                gens.append(
                    CollaborativeCandidates.from_artifact(
                        self.als.artifact, self.history, self.items
                    )
                )
        return gens

    def _build_ranker(self) -> Ranker:
        cfg = self.spec["ranker"]
        if cfg["type"] == "heuristic":
            weights = HeuristicWeights(cfg["weights"]) if "weights" in cfg else None
            return HeuristicRanker(self.features, weights)
        if cfg["type"] == "collaborative":
            assert self.als is not None
            return CollaborativeRanker(self.als, Counter(e.item_id for e in self.history))
        return RetrievalScoreRanker()

    @property
    def component_versions(self) -> dict[str, str]:
        return {
            **{f"generator:{g.name}": g.version for g in self.generators},
            "ranker": self.ranker.version,
        }

    def candidates(self, user_id: str, request_id: str | None = None) -> FusedCandidates:
        req = CandidateRequest(user_id, self.budget, request_id=request_id)
        sets = {g.name: g.generate(req) for g in self.generators}
        return fuse(user_id, sets, self.eligibility, self.fusion)

    def recommend(self, user_id: str, k: int, request_id: str | None = None) -> PipelineOutput:
        fused = self.candidates(user_id, request_id)
        ranked = self.ranker.rank(RankRequest.from_fused(fused, k, request_id))
        final = self.cold_start.apply(ranked, k, self.eligibility, self.features)
        return PipelineOutput(final, fused)


class PipelineRecommender:
    """Adapter so a bundle spec can be evaluated with the offline protocol.

    ``fit`` trains the ALS artifact (when the spec needs one) on the given
    history only, then builds the same pipeline that serving uses.
    """

    name = "pipeline"

    def __init__(self, spec: Mapping[str, Any], als_params: Mapping[str, Any] | None = None):
        validate_spec(spec)
        self.spec = dict(spec)
        self.als_params = dict(als_params or {})
        self.pipeline: RecommendationPipeline | None = None
        self.artifacts_dir: Path | None = None

    @property
    def version(self) -> str:
        if self.pipeline is None:
            return "unfitted"
        parts = sorted(self.pipeline.component_versions.items())
        return "pipeline[" + ";".join(f"{k}={v}" for k, v in parts) + "]"

    def fit(
        self,
        train: Sequence[Interaction],
        items: Sequence[Item],
        as_of: datetime | None = None,
        artifacts_dir: Path | None = None,
    ) -> PipelineRecommender:
        if as_of is None:
            if not train:
                raise ValueError("cannot infer as_of from empty training history")
            as_of = max(e.event_time for e in train) + timedelta(seconds=1)
        directory = artifacts_dir or Path(tempfile.mkdtemp(prefix="recomlab-pipeline-"))
        if "als" in self.spec.get("artifacts", {}):
            ImplicitALS(ALSParams(**self.als_params)).fit(train, items).save(
                directory / self.spec["artifacts"]["als"]
            )
        self.artifacts_dir = directory
        self.pipeline = RecommendationPipeline(self.spec, train, items, as_of, directory)
        return self

    def recommend(self, user_id: str, k: int, eligibility: EligibilityFilter) -> list[ScoredItem]:
        if self.pipeline is None:
            raise RuntimeError("pipeline not fitted")
        out = self.pipeline.recommend(user_id, k)
        return [
            ScoredItem(i.item_id, i.final_score)
            for i in out.ranked.items
            if eligibility.is_eligible(user_id, i.item_id)
        ]

    def params(self) -> dict[str, Any]:
        return {"spec": self.spec, "als": self.als_params}
