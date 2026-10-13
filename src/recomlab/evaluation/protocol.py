"""Evaluation protocol: full eligible catalog (reference) vs sampled (diagnostic).

Sampled evaluation ranks each user's held-out positives against ``n_negatives``
items drawn uniformly (seeded per user) from the user's eligible non-positive
items. These evaluation negatives are drawn fresh by this module; training
negatives (``TrainingNegatives``) are refused. Results from different modes or
sampling policies must never be compared directly (:func:`assert_comparable`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from recomlab.datasets.negative_sampling import TrainingNegatives
from recomlab.experiments.seed import make_rng

MODES = ("full", "sampled")
SAMPLED_WARNING = (
    "SAMPLED EVALUATION (diagnostic): metrics rank held-out positives against a small "
    "random negative sample and are not comparable to full-catalog results."
)


class MixedEvaluationModeError(ValueError):
    pass


@dataclass(frozen=True)
class EvaluationProtocol:
    mode: str = "full"
    n_negatives: int = 100
    seed: int = 0
    sampler: str = "uniform"

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.mode == "sampled" and self.n_negatives <= 0:
            raise ValueError("n_negatives must be positive in sampled mode")
        if self.sampler != "uniform":
            raise ValueError("only uniform evaluation sampling is supported")

    @property
    def is_sampled(self) -> bool:
        return self.mode == "sampled"

    @property
    def policy_key(self) -> str:
        if not self.is_sampled:
            return "full"
        return f"sampled:{self.sampler}:{self.n_negatives}:seed{self.seed}"

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["policy_key"] = self.policy_key
        if not self.is_sampled:
            out.pop("n_negatives")
            out.pop("seed")
            out.pop("sampler")
        else:
            out["warning"] = SAMPLED_WARNING
        return out


def sample_universe(
    user_id: str,
    positives: Iterable[str],
    eligible: Sequence[str],
    protocol: EvaluationProtocol,
) -> list[str]:
    """Positives plus seeded uniform negatives from eligible non-positive items."""
    if isinstance(positives, TrainingNegatives) or isinstance(eligible, TrainingNegatives):
        raise TypeError("training negatives cannot define an evaluation universe")
    pos = sorted(set(positives))
    pool = sorted(set(eligible) - set(pos))
    n = min(protocol.n_negatives, len(pool))
    rng = make_rng(protocol.seed, "eval-negatives", user_id)
    negs = [pool[i] for i in sorted(rng.choice(len(pool), size=n, replace=False))] if n else []
    return pos + negs


def evaluation_key(evaluation: Mapping[str, Any]) -> str:
    """Comparable-protocol key stored in a run manifest's ``evaluation`` section."""
    proto = evaluation.get("protocol")
    if isinstance(proto, Mapping) and "policy_key" in proto:
        return str(proto["policy_key"])
    return str(evaluation.get("mode", "unknown"))


def assert_comparable(evaluations: Sequence[Mapping[str, Any]]) -> str:
    """Raise if runs were evaluated under different modes / sampling policies."""
    keys = {evaluation_key(e) for e in evaluations}
    if len(keys) > 1:
        raise MixedEvaluationModeError(
            f"runs use different evaluation protocols {sorted(keys)}; never mix full-catalog "
            "and sampled results (or different sampling policies) in one comparison"
        )
    return keys.pop() if keys else "full"
