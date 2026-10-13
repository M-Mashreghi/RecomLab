"""Dashboard view models: plain data built from registries and the service.

The Streamlit script only renders these structures; no training or metric
computation happens in the dashboard. Only pseudonymous ids are exposed; user
attributes and raw histories are never shown.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from recomlab.experiments.manifest import RunManifest
from recomlab.experiments.registry import ExperimentRegistry
from recomlab.services.recommendation import RecommendationRequest, RecommendationService

RANKING_PREFIXES = ("precision@", "recall@", "ndcg@", "map@", "mrr@", "hit_rate@")
BEYOND_PREFIXES = (
    "catalog_coverage",
    "user_coverage",
    "ild",
    "novelty",
    "gini",
    "head_share",
    "calibration_kl",
)


def leaderboard_rows(experiments: ExperimentRegistry) -> list[dict[str, Any]]:
    rows = []
    for r in experiments.list_runs():
        ranking = {k: v for k, v in r.metrics.items() if k.startswith(RANKING_PREFIXES)}
        rows.append(
            {
                "run_id": r.run_id,
                "experiment": r.experiment_id,
                "model": f"{r.model['name']}@{r.model['version']}",
                "partition": r.split.get("partition"),
                "evaluation": r.evaluation.get("mode"),
                **ranking,
            }
        )
    return rows


def run_detail(experiments: ExperimentRegistry, run_id: str) -> dict[str, Any]:
    r: RunManifest = experiments.get_run(run_id)
    metrics = {k: v for k, v in r.metrics.items() if not isinstance(v, dict)}
    return {
        "identity": {
            "run_id": r.run_id,
            "dataset": f"{r.dataset['dataset_id']}@{r.dataset['version']}",
            "split_id": r.split.get("split_id"),
            "model": f"{r.model['name']}@{r.model['version']}",
            "label_version": r.label_policy.get("label_version"),
        },
        "evaluation": {k: v for k, v in r.evaluation.items() if k != "eligibility"},
        "warnings": list(r.evaluation.get("warnings", [])),
        "candidate_stage": r.metrics.get("retrieval") or {},
        "ranking_stage": {k: v for k, v in metrics.items() if k.startswith(RANKING_PREFIXES)},
        "beyond_accuracy": {k: v for k, v in metrics.items() if k.startswith(BEYOND_PREFIXES)},
        "slice_report": r.artifacts.get("slice_report"),
    }


def demo_user_ids(service: RecommendationService, limit: int = 50) -> list[str]:
    """Pseudonymous ids of users with history, most active first (no attributes)."""
    counts = Counter(e.user_id for e in service.history)
    return [u for u, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))][:limit]


def explorer(service: RecommendationService, user_id: str, k: int = 5) -> dict[str, Any]:
    resp = service.recommend(RecommendationRequest(user_id, k))
    return {
        "bundle_id": resp.bundle_id,
        "n_candidates": resp.metadata.get("n_candidates"),
        "candidate_counts": resp.metadata.get("candidate_counts"),
        "items": [
            {
                "rank": i.rank,
                "item_id": i.item_id,
                "score": round(i.score, 6),
                "sources": ", ".join(sorted(i.sources)) or "-",
                "fallback": i.fallback or "",
                "top_contributions": ", ".join(
                    f"{f}={v:.3f}"
                    for f, v in sorted(i.contributions.items(), key=lambda kv: -abs(kv[1]))[:3]
                ),
            }
            for i in resp.items
        ],
    }


def cold_start_view(service: RecommendationService, k: int = 5) -> dict[str, Any]:
    """Recommendations for a synthetic never-seen user plus catalog cold items."""
    seen_items = {e.item_id for e in service.history}
    cold_items = sorted(i.item_id for i in service.items if i.item_id not in seen_items)
    rec = explorer(service, "demo_cold_user", k)
    return {"cold_user_recommendations": rec["items"], "cold_items": cold_items}


def beyond_accuracy_view(experiments: ExperimentRegistry) -> list[dict[str, Any]]:
    rows = []
    for r in experiments.list_runs():
        vals = {k: v for k, v in r.metrics.items() if k.startswith(BEYOND_PREFIXES)}
        rows.append({"run_id": r.run_id, "model": r.model["name"], **vals})
    return rows
