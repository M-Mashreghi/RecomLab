import pytest

from recomlab.domain.exposure import FeedbackEvent, Impression, link_feedback
from recomlab.evaluation.off_policy import (
    ExposureDataUnavailable,
    deterministic_topk_policy,
    off_policy_estimate,
    rewards_from_links,
)

pytestmark = pytest.mark.metrics


def imp(iid, user, item, prop, pos=1):
    return Impression(iid, f"r-{user}", user, item, pos, "logger-v1", "2024-01-01T00:00:00Z", prop)  # type: ignore[arg-type]


IMPS = [
    imp("m1", "u1", "a", 0.5),
    imp("m2", "u1", "b", 0.25),
    imp("m3", "u2", "a", 0.8),
    imp("m4", "u2", "c", 0.1),
]
FEEDBACK = [
    FeedbackEvent("f1", "u1", "a", "click", "2024-01-01T00:01:00Z", impression_id="m1"),  # type: ignore[arg-type]
    FeedbackEvent("f2", "u2", "c", "click", "2024-01-01T00:01:00Z", impression_id="m4"),  # type: ignore[arg-type]
    FeedbackEvent("f3", "u1", "b", "click", "2024-01-01T00:01:00Z"),  # unlinked -> ignored  # type: ignore[arg-type]
]


def test_hand_computed_ips_snips():
    rewards = rewards_from_links(link_feedback(IMPS, FEEDBACK))
    assert rewards == {"m1": 1.0, "m4": 1.0}
    target = deterministic_topk_policy({"u1": ["a"], "u2": ["c"]}, k=1)
    res = off_policy_estimate(IMPS, rewards, target, clip=100.0, max_weight_warning=100.0)
    # weights: m1 1/0.5=2, m2 0, m3 0, m4 1/0.1=10 ; rewards m1=1, m4=1
    assert res.ips == pytest.approx((2 + 10) / 4)
    assert res.snips == pytest.approx((2 + 10) / 12)
    assert res.effective_sample_size == pytest.approx(144 / (4 + 100))
    assert res.max_weight == 10.0 and res.n_clipped == 0


def test_diagnostics_flag_extreme_weights_and_clipping():
    rewards = {"m4": 1.0}
    target = deterministic_topk_policy({"u2": ["c"]}, k=1)
    res = off_policy_estimate(IMPS, rewards, target, clip=5.0, max_weight_warning=4.0)
    assert res.n_clipped == 1 and res.max_weight == 5.0
    assert res.ips == pytest.approx(5.0 / 4)
    assert any("extreme importance weight" in w for w in res.warnings)
    assert any("clipped" in w for w in res.warnings)
    calm = off_policy_estimate(
        IMPS, rewards, lambda i: i.propensity or 0.0, clip=10, max_weight_warning=5
    )
    assert calm.warnings == [] and calm.max_weight == 1.0


def test_refuses_incompatible_datasets():
    missing = [*IMPS, imp("m5", "u3", "a", None)]
    target = deterministic_topk_policy({}, 1)
    with pytest.raises(ExposureDataUnavailable, match="never inferred"):
        off_policy_estimate(missing, {}, target)
    with pytest.raises(ExposureDataUnavailable):
        off_policy_estimate([], {}, target)
