"""User-level bootstrap confidence intervals and paired model comparison.

* :func:`bootstrap_ci` resamples *users* with replacement (percentile interval).
* :func:`paired_comparison` compares two recommenders on the same users:
  a paired bootstrap interval for the mean per-user difference (B - A), a
  two-sided bootstrap p-value, and a Wilcoxon signed-rank test (normal
  approximation with tie and zero handling, Pratt-free: zero differences are
  dropped). Effect sizes: mean delta, relative delta and paired Cohen's d.

A difference is called ``"better"`` / ``"worse"`` only when the bootstrap
interval excludes zero and at least ``min_users`` (default 10) users are
compared; otherwise the conclusion is ``"uncertain"``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from recomlab.experiments.seed import make_rng


@dataclass(frozen=True)
class ConfidenceInterval:
    mean: float
    low: float
    high: float
    n_users: int
    n_resamples: int
    confidence: float
    seed: int
    method: str = "user-level percentile bootstrap"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bootstrap_means(values: np.ndarray, n_resamples: int, seed: int, tag: str) -> np.ndarray:
    rng = make_rng(seed, "bootstrap", tag)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means: np.ndarray = values[idx].mean(axis=1)
    return means


def bootstrap_ci(
    per_user: Mapping[str, float],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    if not per_user:
        raise ValueError("no users to bootstrap")
    users = sorted(per_user)
    values = np.array([per_user[u] for u in users], dtype=float)
    means = _bootstrap_means(values, n_resamples, seed, "ci")
    a = (1 - confidence) / 2
    return ConfidenceInterval(
        mean=float(values.mean()),
        low=float(np.quantile(means, a)),
        high=float(np.quantile(means, 1 - a)),
        n_users=len(users),
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )


def wilcoxon_signed_rank(diffs: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test (normal approximation).

    Zero differences are dropped; tied absolute differences get average ranks
    with the standard variance correction. Returns ``(W+, p_value)``.
    """
    d = diffs[diffs != 0]
    n = len(d)
    if n == 0:
        return 0.0, 1.0
    abs_d = np.abs(d)
    order = np.argsort(abs_d, kind="mergesort")
    ranks = np.empty(n)
    sorted_abs = abs_d[order]
    i = 0
    tie_term = 0.0
    while i < n:
        j = i
        while j + 1 < n and sorted_abs[j + 1] == sorted_abs[i]:
            j += 1
        avg = (i + j + 2) / 2.0
        ranks[order[i : j + 1]] = avg
        t = j - i + 1
        tie_term += t**3 - t
        i = j + 1
    w_plus = float(ranks[d > 0].sum())
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var <= 0:
        return w_plus, 1.0
    z = (w_plus - mean) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2))
    return w_plus, min(1.0, p)


@dataclass(frozen=True)
class PairedComparison:
    metric: str
    baseline: str
    challenger: str
    n_users: int
    mean_a: float
    mean_b: float
    delta: float
    relative_delta: float | None
    ci_low: float
    ci_high: float
    bootstrap_p: float
    wilcoxon_w: float
    wilcoxon_p: float
    cohens_d: float | None
    n_resamples: int
    confidence: float
    seed: int
    conclusion: str

    @property
    def significant(self) -> bool:
        return self.conclusion != "uncertain"

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["significant"] = self.significant
        return out


def paired_comparison(
    a: Mapping[str, float],
    b: Mapping[str, float],
    *,
    metric: str = "metric",
    baseline: str = "A",
    challenger: str = "B",
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    min_users: int = 10,
) -> PairedComparison:
    """Paired comparison of challenger ``b`` against baseline ``a`` on common users."""
    users = sorted(set(a) & set(b))
    if len(users) < 2:
        raise ValueError("paired comparison needs at least 2 common users")
    if set(a) != set(b):
        raise ValueError("models were evaluated on different user populations")
    va = np.array([a[u] for u in users], dtype=float)
    vb = np.array([b[u] for u in users], dtype=float)
    diffs = vb - va
    boot = _bootstrap_means(diffs, n_resamples, seed, "paired")
    alpha = (1 - confidence) / 2
    low, high = float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))
    p_boot = min(1.0, 2 * min(float((boot <= 0).mean()), float((boot >= 0).mean())))
    w, p_w = wilcoxon_signed_rank(diffs)
    sd = float(diffs.std(ddof=1))
    delta = float(diffs.mean())
    conclusion = "better" if low > 0 else "worse" if high < 0 else "uncertain"
    if len(users) < min_users:
        conclusion = "uncertain"  # too few users for a reliable interval
    return PairedComparison(
        metric=metric,
        baseline=baseline,
        challenger=challenger,
        n_users=len(users),
        mean_a=float(va.mean()),
        mean_b=float(vb.mean()),
        delta=delta,
        relative_delta=delta / float(va.mean()) if va.mean() != 0 else None,
        ci_low=low,
        ci_high=high,
        bootstrap_p=p_boot,
        wilcoxon_w=w,
        wilcoxon_p=p_w,
        cohens_d=delta / sd if sd > 0 else None,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
        conclusion=conclusion,
    )
