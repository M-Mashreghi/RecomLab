import dataclasses

import pytest

from recomlab.research.study import (
    EthicsRecord,
    Hypothesis,
    ResponseRecord,
    StudyConfig,
    StudyConfigError,
    StudyDataStore,
    StudyGovernanceError,
    assign_conditions,
    balance,
    sample_size_per_group,
    simulate_responses,
)

CONDS = ("none-v1", "template-v1", "generic_llm-v1", "contextual_llm-v1")


def config(**kw):
    base = dict(
        study_id="expl-study",
        version="1",
        research_questions=("Do contextual LLM explanations increase acceptance?",),
        hypotheses=(
            Hypothesis(
                "H1",
                "contextual > generic on acceptance",
                "acceptance",
                "contextual_llm-v1",
                "generic_llm-v1",
            ),
        ),
        conditions=CONDS,
        primary_outcome="acceptance",
        secondary_outcomes=("trust", "usefulness", "satisfaction"),
        seed=11,
    )
    base.update(kw)
    return StudyConfig(**base)


def test_assignment_deterministic_and_balanced():
    cfg = config()
    pids = [f"SIM-{n}" for n in range(203)]
    a = assign_conditions(cfg, pids)
    assert a == assign_conditions(cfg, pids)
    assert a != assign_conditions(config(seed=12), pids)
    counts = balance(a, CONDS)
    assert max(counts.values()) - min(counts.values()) <= cfg.block_size // len(CONDS)
    # every complete block is perfectly balanced
    first_block = [a[p] for p in pids[:8]]
    assert sorted(first_block) == sorted(CONDS * 2)
    with pytest.raises(StudyConfigError):
        assign_conditions(cfg, ["x", "x"])


def test_config_is_versioned_and_validated():
    assert config().config_version == config().config_version
    assert config().config_version != config(block_size=12).config_version
    with pytest.raises(StudyConfigError):
        config(conditions=("persuade-v1",))
    with pytest.raises(StudyConfigError):
        config(primary_outcome="happiness")
    with pytest.raises(StudyConfigError):
        config(block_size=6)


def test_real_mode_requires_ethics_approval():
    with pytest.raises(StudyGovernanceError):
        config(data_mode="real")
    with pytest.raises(StudyGovernanceError):
        config(data_mode="real", ethics=EthicsRecord("approved", "IRB-123"))
    ok = config(
        data_mode="real",
        ethics=EthicsRecord("approved", "IRB-123", consent_version="c1", data_retention_days=365),
    )
    with pytest.raises(StudyGovernanceError):
        simulate_responses(ok, 10, {})


def test_synthetic_cannot_be_confused_with_real():
    sim_cfg = config()
    records = simulate_responses(sim_cfg, 40, {"contextual_llm-v1": {"acceptance": 0.5}})
    assert all(r.simulated and r.participant_id.startswith("SIM-") for r in records)
    store = StudyDataStore(sim_cfg)
    for r in records:
        store.add(r)
    real_cfg = config(
        data_mode="real",
        ethics=EthicsRecord("approved", "IRB-1", consent_version="c1", data_retention_days=30),
    )
    real_store = StudyDataStore(real_cfg)
    with pytest.raises(StudyGovernanceError):
        real_store.add(dataclasses.replace(records[0], study_version=real_cfg.config_version))
    with pytest.raises(StudyGovernanceError):
        store.add(dataclasses.replace(records[0], participant_id="P-1"))
    with pytest.raises(StudyConfigError):
        store.add(dataclasses.replace(records[0], study_version="other"))


def test_exclusions_auditable():
    cfg = config()
    store = StudyDataStore(cfg)
    base = ResponseRecord(
        "SIM-1", "none-v1", {"acceptance": 4}, True, 120, True, cfg.config_version
    )
    store.add(base)
    store.add(dataclasses.replace(base, participant_id="SIM-2", attention_check_passed=False))
    store.add(dataclasses.replace(base, participant_id="SIM-3", completion_s=10))
    assert store.excluded() == {"SIM-2": "attention_check_failed", "SIM-3": "too_fast"}
    assert [r.participant_id for r in store.analyzable()] == ["SIM-1"]


def test_power_planning():
    assert sample_size_per_group(0.5) == 63  # classic d=0.5, alpha .05, power .8
    assert sample_size_per_group(0.5, n_comparisons=3) > 63
    with pytest.raises(ValueError):
        sample_size_per_group(0)
