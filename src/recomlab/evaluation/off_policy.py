"""Optional exposure-aware off-policy evaluation (IPS / SNIPS).

Gated on valid logged propensities: the estimator refuses impression logs
where any propensity is missing (``assess_exposure_log``) and never derives
propensities from clicks. Item-level estimator over logged impressions ``n``:

    w_n   = min(pi(item_n | ctx_n) / mu_n, clip)
    IPS   = sum_n w_n r_n / N
    SNIPS = sum_n w_n r_n / sum_n w_n

``mu_n`` is the logged propensity, ``pi`` the target policy's probability of
showing the item in that request, and ``r_n`` the linked reward (e.g. click).
Diagnostics: effective sample size ``(sum w)^2 / sum w^2``, max weight,
clipped count; extreme weights produce warnings.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from recomlab.domain.exposure import (
    FeedbackLink,
    Impression,
    LinkStatus,
    assess_exposure_log,
)


class ExposureDataUnavailable(RuntimeError):
    pass


TargetPolicy = Callable[[Impression], float]


@dataclass(frozen=True)
class OPEResult:
    ips: float
    snips: float | None
    n_impressions: int
    clip: float
    mean_weight: float
    max_weight: float
    n_clipped: int
    effective_sample_size: float
    ess_ratio: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rewards_from_links(
    links: Sequence[FeedbackLink], positive_types: Sequence[str] = ("click",)
) -> dict[str, float]:
    """Reward per impression id from explicitly linked feedback (inferred links excluded)."""
    out: dict[str, float] = {}
    for link in links:
        if link.status is LinkStatus.LINKED and link.impression is not None:
            r = 1.0 if str(link.feedback.feedback_type) in positive_types else 0.0
            key = link.impression.impression_id
            out[key] = max(out.get(key, 0.0), r)
    return out


def deterministic_topk_policy(rankings: Mapping[str, Sequence[str]], k: int) -> TargetPolicy:
    """pi = 1 if the target model ranks the item in the user's top-k, else 0."""

    def pi(imp: Impression) -> float:
        return 1.0 if imp.item_id in list(rankings.get(imp.user_id, []))[:k] else 0.0

    return pi


def off_policy_estimate(
    impressions: Sequence[Impression],
    rewards: Mapping[str, float],
    target_policy: TargetPolicy,
    *,
    clip: float = 10.0,
    max_weight_warning: float = 5.0,
    min_ess_ratio: float = 0.1,
) -> OPEResult:
    status = assess_exposure_log(impressions)
    if not status.exposure_aware_evaluation_allowed:
        raise ExposureDataUnavailable(
            f"exposure-aware evaluation unavailable: {status.availability} "
            f"({status.n_missing_propensity} impressions missing propensity); "
            "propensities are never inferred"
        )
    if clip <= 0:
        raise ValueError("clip must be positive")
    weights: list[float] = []
    clipped = 0
    weighted_reward = 0.0
    for imp in impressions:
        assert imp.propensity is not None
        raw = target_policy(imp) / imp.propensity
        w = min(raw, clip)
        clipped += raw > clip
        weights.append(w)
        weighted_reward += w * rewards.get(imp.impression_id, 0.0)
    n = len(weights)
    total = sum(weights)
    sq = sum(w * w for w in weights)
    ess = (total * total / sq) if sq > 0 else 0.0
    max_w = max(weights, default=0.0)
    warnings = []
    if max_w > max_weight_warning:
        warnings.append(f"extreme importance weight {max_w:.3g} > {max_weight_warning}")
    if n and ess / n < min_ess_ratio:
        warnings.append(f"low effective sample size ratio {ess / n:.3g} < {min_ess_ratio}")
    if clipped:
        warnings.append(f"{clipped} weights clipped at {clip} (estimate is biased)")
    return OPEResult(
        ips=weighted_reward / n if n else 0.0,
        snips=weighted_reward / total if total > 0 else None,
        n_impressions=n,
        clip=clip,
        mean_weight=total / n if n else 0.0,
        max_weight=max_w,
        n_clipped=clipped,
        effective_sample_size=ess,
        ess_ratio=ess / n if n else 0.0,
        warnings=warnings,
    )
