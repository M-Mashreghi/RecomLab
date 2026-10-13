import math

import numpy as np
import pytest

from recomlab.evaluation.statistics import (
    bootstrap_ci,
    paired_comparison,
    wilcoxon_signed_rank,
)

pytestmark = pytest.mark.metrics


def _users(values):
    return {f"u{n:03d}": float(v) for n, v in enumerate(values)}


def test_ci_contains_mean_and_same_seed_reproduces():
    rng = np.random.default_rng(1)
    per_user = _users(rng.random(200))
    ci = bootstrap_ci(per_user, n_resamples=500, seed=3)
    assert ci.low < ci.mean < ci.high
    assert bootstrap_ci(per_user, n_resamples=500, seed=3) == ci
    assert bootstrap_ci(per_user, n_resamples=500, seed=4) != ci
    assert ci.n_resamples == 500 and ci.seed == 3 and ci.n_users == 200
    assert bootstrap_ci({"u": 1.0}, n_resamples=10).low == 1.0
    with pytest.raises(ValueError):
        bootstrap_ci({})


def test_known_synthetic_comparison():
    rng = np.random.default_rng(0)
    base = rng.random(300)
    a = _users(base)
    clearly_better = _users(np.clip(base + 0.1 + rng.normal(0, 0.02, 300), 0, None))
    same = _users(base + rng.normal(0, 0.05, 300))
    res = paired_comparison(a, clearly_better, metric="ndcg@10", n_resamples=1000, seed=1)
    assert res.conclusion == "better" and res.significant
    assert res.ci_low > 0 and res.wilcoxon_p < 1e-6 and res.bootstrap_p < 0.01
    assert res.delta == pytest.approx(res.mean_b - res.mean_a)
    worse = paired_comparison(clearly_better, a, n_resamples=1000, seed=1)
    assert worse.conclusion == "worse"
    noise = paired_comparison(a, same, n_resamples=1000, seed=1)
    assert noise.conclusion == "uncertain" and not noise.significant
    assert noise.ci_low < 0 < noise.ci_high


def test_same_seed_reproduces_paired_intervals():
    rng = np.random.default_rng(2)
    a, b = _users(rng.random(50)), _users(rng.random(50))
    r1 = paired_comparison(a, b, seed=9, n_resamples=300)
    r2 = paired_comparison(a, b, seed=9, n_resamples=300)
    assert r1 == r2
    assert r1.to_dict()["seed"] == 9 and r1.to_dict()["n_resamples"] == 300


def test_wilcoxon_hand_computed():
    # diffs: 1, -2, 3, 4 (no ties) -> W+ = 1 + 3 + 4 = 8; n=4, mean=5, var=7.5
    w, p = wilcoxon_signed_rank(np.array([1.0, -2.0, 3.0, 4.0]))
    assert w == 8.0
    z = (8 - 5) / math.sqrt(7.5)
    assert p == pytest.approx(math.erfc(abs(z) / math.sqrt(2)))
    assert wilcoxon_signed_rank(np.zeros(5)) == (0.0, 1.0)
    w_tie, _ = wilcoxon_signed_rank(np.array([1.0, 1.0, -1.0]))
    assert w_tie == pytest.approx(4.0)  # ranks 2, 2, 2


def test_population_mismatch_rejected():
    with pytest.raises(ValueError):
        paired_comparison({"a": 1.0, "b": 1.0}, {"a": 1.0, "c": 1.0})
    with pytest.raises(ValueError):
        paired_comparison({"a": 1.0}, {"a": 2.0})
