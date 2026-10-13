import math

import pytest

from recomlab.research.analysis import (
    AnalysisError,
    analyze,
    holm_adjust,
    t_two_sided_p,
)
from recomlab.research.study import Hypothesis, StudyConfig, StudyDataStore, simulate_responses

CONDS = ("generic_llm-v1", "contextual_llm-v1", "template-v1")


def make(effect, n=300, seed=0):
    cfg = StudyConfig(
        study_id="analysis",
        version="1",
        research_questions=("Does context raise acceptance?",),
        hypotheses=(
            Hypothesis(
                "H1", "contextual > generic", "acceptance", "contextual_llm-v1", "generic_llm-v1"
            ),
        ),
        conditions=CONDS,
        primary_outcome="acceptance",
        secondary_outcomes=("trust", "satisfaction"),
        block_size=6,
    )
    store = StudyDataStore(cfg)
    for r in simulate_responses(
        cfg, n, {"contextual_llm-v1": {"acceptance": effect}}, seed=seed, exclusion_rate=0.1
    ):
        store.add(r)
    return cfg, store


def test_known_effect_detected_and_null_not():
    cfg, store = make(effect=1.0)
    rep = analyze(cfg, store, n_resamples=500)
    primary = rep["comparisons"][0]
    assert primary["pre_specified"] and primary["outcome"] == "acceptance"
    assert primary["mean_diff"] == pytest.approx(1.0, abs=0.35)
    assert primary["ci_low"] > 0 and primary["p_holm"] < 0.001
    trust = next(c for c in rep["comparisons"] if c["outcome"] == "trust")
    assert not trust["pre_specified"]
    cfg0, store0 = make(effect=0.0, seed=3)
    null = analyze(cfg0, store0, n_resamples=500)["comparisons"][0]
    assert null["ci_low"] < 0 < null["ci_high"]


def test_condition_labels_and_exclusions_auditable():
    cfg, store = make(effect=0.5)
    rep = analyze(cfg, store, n_resamples=200)
    assert rep["data_status"].startswith("SIMULATED")
    assert rep["n_enrolled"] == 300
    assert rep["n_analyzed"] == 300 - len(rep["exclusions"]["by_participant"])
    total_by_cond = sum(sum(v.values()) for v in rep["exclusions"]["by_condition"].values())
    assert total_by_cond == len(rep["exclusions"]["by_participant"])
    assert set(rep["descriptive"]) == set(CONDS)
    n_desc = sum(rep["descriptive"][c]["acceptance"]["n"] for c in CONDS)
    assert n_desc == rep["n_analyzed"]
    assert "Holm-Bonferroni" in rep["multiple_comparison_policy"]


def test_offline_metrics_not_merged_into_perception_scales():
    cfg, store = make(effect=0.0, n=30)
    hacked = cfg.__class__.__new__(cfg.__class__)
    object.__setattr__(hacked, "__dict__", {**cfg.__dict__, "secondary_outcomes": ("ndcg@10",)})
    with pytest.raises(AnalysisError):
        analyze(hacked, store)
    with pytest.raises(ValueError):
        StudyConfig(**{**cfg.__dict__, "secondary_outcomes": ("ndcg@10",)})


def test_statistics_helpers():
    assert t_two_sided_p(0.0, 10) == pytest.approx(1.0)
    assert t_two_sided_p(2.228, 10) == pytest.approx(0.05, abs=1e-3)  # t(.975, 10)
    assert t_two_sided_p(1.96, 1e6) == pytest.approx(0.05, abs=1e-3)
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert all(not math.isnan(p) for p in holm_adjust([0.5]))
