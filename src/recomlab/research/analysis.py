"""Trust, acceptance, usefulness and satisfaction analysis by condition.

* Descriptive statistics per condition and outcome.
* Pre-specified comparisons (the study's hypotheses, optionally extended to
  every secondary outcome for the same contrast): Welch's t-test with exact
  Student-t p-values, a percentile bootstrap CI for the mean difference, and
  Cohen's d.
* Multiple-comparison policy: Holm-Bonferroni across all reported tests.

Only perception/behavior outcomes defined in the study protocol can be
analyzed; offline recommender metrics (NDCG, recall, ...) are refused so they
are never merged into perception scales. Reports carry the simulated/real flag
and the exclusion audit.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from recomlab.experiments.seed import make_rng
from recomlab.research.study import OUTCOMES, ResponseRecord, StudyConfig, StudyDataStore


class AnalysisError(ValueError):
    pass


# --- Student t distribution (regularized incomplete beta, Numerical Recipes) ----
def _betacf(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > 1e-300 else 1e-300
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


@dataclass(frozen=True)
class Comparison:
    hypothesis_id: str
    outcome: str
    treatment: str
    control: str
    n_treatment: int
    n_control: int
    mean_diff: float
    ci_low: float
    ci_high: float
    welch_t: float
    df: float
    p_value: float
    p_holm: float
    cohens_d: float | None
    pre_specified: bool


def _values(records: Sequence[ResponseRecord], condition: str, outcome: str) -> list[float]:
    return [float(r.outcomes[outcome]) for r in records if r.condition == condition]


def descriptive(
    records: Sequence[ResponseRecord], conditions: Sequence[str], outcomes: Sequence[str]
) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for c in conditions:
        out[c] = {}
        for o in outcomes:
            v = _values(records, c, o)
            out[c][o] = {
                "n": float(len(v)),
                "mean": statistics.fmean(v) if v else math.nan,
                "sd": statistics.stdev(v) if len(v) > 1 else math.nan,
                "median": statistics.median(v) if v else math.nan,
            }
    return out


def _compare(
    a: list[float], b: list[float], n_resamples: int, seed: int, tag: str
) -> tuple[float, float, float, float, float, float, float | None]:
    if len(a) < 2 or len(b) < 2:
        raise AnalysisError("each group needs at least 2 observations")
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    se2 = va / len(a) + vb / len(b)
    t = (ma - mb) / math.sqrt(se2) if se2 > 0 else 0.0
    df = (
        se2**2 / ((va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
        if se2 > 0
        else float(len(a) + len(b) - 2)
    )
    p = t_two_sided_p(t, df) if se2 > 0 else 1.0
    rng = make_rng(seed, "study-boot", tag)
    xa, xb = np.array(a), np.array(b)
    diffs = xa[rng.integers(0, len(a), (n_resamples, len(a)))].mean(axis=1) - xb[
        rng.integers(0, len(b), (n_resamples, len(b)))
    ].mean(axis=1)
    pooled = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    d = (ma - mb) / pooled if pooled > 0 else None
    return ma - mb, float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975)), t, df, p, d


def analyze(
    config: StudyConfig,
    store: StudyDataStore,
    *,
    include_secondary: bool = True,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    outcomes = (config.primary_outcome, *config.secondary_outcomes)
    bad = [o for o in outcomes if o not in OUTCOMES]
    if bad:
        raise AnalysisError(
            f"not perception outcomes: {bad}; offline metrics are analyzed elsewhere"
        )
    records = store.analyzable()
    excluded = store.excluded()
    tests: list[tuple[str, str, str, str, bool]] = []
    for h in config.hypotheses:
        tests.append((h.hypothesis_id, h.outcome, h.treatment, h.control, True))
        if include_secondary:
            for o in outcomes:
                if o != h.outcome:
                    tests.append((f"{h.hypothesis_id}:{o}", o, h.treatment, h.control, False))
    raw = []
    for hid, o, tr, ct, pre in tests:
        raw.append(
            (
                hid,
                o,
                tr,
                ct,
                pre,
                _compare(_values(records, tr, o), _values(records, ct, o), n_resamples, seed, hid),
            )
        )
    p_holm = holm_adjust([r[5][5] for r in raw])
    comparisons = [
        Comparison(
            hid,
            o,
            tr,
            ct,
            len(_values(records, tr, o)),
            len(_values(records, ct, o)),
            res[0],
            res[1],
            res[2],
            res[3],
            res[4],
            res[5],
            ph,
            res[6],
            pre,
        )
        for (hid, o, tr, ct, pre, res), ph in zip(raw, p_holm, strict=True)
    ]
    by_cond_excl: dict[str, dict[str, int]] = {}
    cond_of = {r.participant_id: r.condition for r in store.records}
    for pid, reason in excluded.items():
        slot = by_cond_excl.setdefault(cond_of[pid], {})
        slot[reason] = slot.get(reason, 0) + 1
    simulated = all(r.simulated for r in store.records)
    return {
        "study": config.config_version,
        "data_status": "SIMULATED (software validation only)" if simulated else "real",
        "n_enrolled": len(store.records),
        "n_analyzed": len(records),
        "exclusions": {
            "by_participant": dict(sorted(excluded.items())),
            "by_condition": by_cond_excl,
        },
        "conditions": list(config.conditions),
        "descriptive": descriptive(records, config.conditions, outcomes),
        "comparisons": [asdict(c) for c in comparisons],
        "multiple_comparison_policy": (
            f"Holm-Bonferroni across {len(comparisons)} tests "
            f"({sum(c.pre_specified for c in comparisons)} pre-specified primary)"
        ),
        "notes": "Perception outcomes only; offline recommender metrics are reported separately.",
    }
