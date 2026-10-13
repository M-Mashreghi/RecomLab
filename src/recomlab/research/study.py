"""Human-AI study protocol, randomization, power planning and ethics boundary.

The software supports study *design* and *simulated* data for testing. It does
not recruit participants. A study config in ``data_mode="real"`` is rejected
unless it records an approved ethics/IRB decision, a consent version and a
retention policy -- and even then this module only validates configuration;
running recruitment is outside the engineering workflow.

Simulated records always carry ``simulated=True`` and ``SIM-`` participant ids;
stores refuse to mix simulated and real records.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from statistics import NormalDist
from typing import Any

from recomlab.experiments.seed import make_rng
from recomlab.explanations.conditions import CONDITIONS

DATA_MODES = ("simulated", "real")
OUTCOMES = (
    "acceptance",
    "trust",
    "usefulness",
    "satisfaction",
    "understanding",
    "decision_time_s",
)


class StudyGovernanceError(RuntimeError):
    pass


class StudyConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    statement: str
    outcome: str
    treatment: str
    control: str
    direction: str = "greater"  # greater | less | two-sided


@dataclass(frozen=True)
class EthicsRecord:
    status: str = "not_obtained"  # not_obtained | approved
    reference: str | None = None
    consent_version: str | None = None
    data_retention_days: int | None = None


@dataclass(frozen=True)
class StudyConfig:
    study_id: str
    version: str
    research_questions: tuple[str, ...]
    hypotheses: tuple[Hypothesis, ...]
    conditions: tuple[str, ...]
    primary_outcome: str
    secondary_outcomes: tuple[str, ...] = ()
    exclusion_rules: Mapping[str, Any] = field(
        default_factory=lambda: {"attention_check_failed": True, "min_completion_s": 60}
    )
    block_size: int = 8
    seed: int = 0
    data_mode: str = "simulated"
    ethics: EthicsRecord = field(default_factory=EthicsRecord)

    def __post_init__(self) -> None:
        unknown = set(self.conditions) - set(CONDITIONS)
        if unknown:
            raise StudyConfigError(f"unknown explanation conditions {sorted(unknown)}")
        outcomes = {self.primary_outcome, *self.secondary_outcomes}
        if not outcomes <= set(OUTCOMES):
            raise StudyConfigError(f"unknown outcomes {sorted(outcomes - set(OUTCOMES))}")
        for h in self.hypotheses:
            if h.outcome not in outcomes or {h.treatment, h.control} - set(self.conditions):
                raise StudyConfigError(f"hypothesis {h.hypothesis_id} references undefined parts")
        if self.block_size % len(self.conditions):
            raise StudyConfigError("block_size must be a multiple of the number of conditions")
        if self.data_mode not in DATA_MODES:
            raise StudyConfigError(f"data_mode must be one of {DATA_MODES}")
        if self.data_mode == "real":
            e = self.ethics
            if e.status != "approved" or not e.reference:
                raise StudyGovernanceError(
                    "real-participant mode requires an approved ethics/IRB decision with a "
                    "reference; this repository does not authorize recruitment"
                )
            if not e.consent_version or not e.data_retention_days:
                raise StudyGovernanceError("real mode requires consent version and retention days")

    @property
    def config_version(self) -> str:
        digest = hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()
        return f"{self.study_id}@{self.version}#{digest[:10]}"


def assign_conditions(config: StudyConfig, participant_ids: Sequence[str]) -> dict[str, str]:
    """Permuted-block randomization in enrollment order, deterministic for ``config.seed``."""
    if len(set(participant_ids)) != len(participant_ids):
        raise StudyConfigError("duplicate participant ids")
    rng = make_rng(config.seed, "study", config.study_id, config.version)
    reps = config.block_size // len(config.conditions)
    out: dict[str, str] = {}
    block: list[str] = []
    for pid in participant_ids:
        if not block:
            block = [c for c in config.conditions for _ in range(reps)]
            block = [block[i] for i in rng.permutation(len(block))]
        out[pid] = block.pop()
    return out


def balance(assignments: Mapping[str, str], conditions: Sequence[str]) -> dict[str, int]:
    counts = Counter(assignments.values())
    return {c: counts.get(c, 0) for c in conditions}


def sample_size_per_group(
    effect_size_d: float, alpha: float = 0.05, power: float = 0.8, n_comparisons: int = 1
) -> int:
    """Two-sample, two-sided comparison of means (normal approximation, Bonferroni)."""
    if effect_size_d <= 0:
        raise ValueError("effect size must be positive")
    z = NormalDist()
    a = alpha / n_comparisons
    n = 2 * (z.inv_cdf(1 - a / 2) + z.inv_cdf(power)) ** 2 / effect_size_d**2
    return math.ceil(n)


@dataclass(frozen=True)
class ResponseRecord:
    participant_id: str
    condition: str
    outcomes: Mapping[str, float]
    attention_check_passed: bool
    completion_s: float
    simulated: bool
    study_version: str


class StudyDataStore:
    """In-memory store that never mixes simulated and real records."""

    def __init__(self, config: StudyConfig) -> None:
        self.config = config
        self.records: list[ResponseRecord] = []

    def add(self, record: ResponseRecord) -> None:
        simulated_mode = self.config.data_mode == "simulated"
        if record.simulated != simulated_mode:
            raise StudyGovernanceError("simulated and real participant records cannot be mixed")
        if record.simulated != record.participant_id.startswith("SIM-"):
            raise StudyGovernanceError("simulated records must use SIM- participant ids")
        if record.study_version != self.config.config_version:
            raise StudyConfigError("record belongs to another study config version")
        self.records.append(record)

    def excluded(self) -> dict[str, str]:
        rules = self.config.exclusion_rules
        out = {}
        for r in self.records:
            if rules.get("attention_check_failed") and not r.attention_check_passed:
                out[r.participant_id] = "attention_check_failed"
            elif r.completion_s < float(rules.get("min_completion_s", 0)):
                out[r.participant_id] = "too_fast"
        return out

    def analyzable(self) -> list[ResponseRecord]:
        ex = self.excluded()
        return [r for r in self.records if r.participant_id not in ex]


def simulate_responses(
    config: StudyConfig,
    n_participants: int,
    effects: Mapping[str, Mapping[str, float]],
    *,
    noise_sd: float = 1.0,
    baseline: float = 4.0,
    exclusion_rate: float = 0.05,
    seed: int = 0,
) -> list[ResponseRecord]:
    """Synthetic responses for software testing only (labelled simulated).

    ``effects[condition][outcome]`` shifts the outcome mean for that condition.
    """
    if config.data_mode != "simulated":
        raise StudyGovernanceError("simulation is only allowed for simulated-mode studies")
    pids = [f"SIM-{n:05d}" for n in range(n_participants)]
    assignment = assign_conditions(config, pids)
    rng = make_rng(seed, "simulate", config.study_id)
    outcomes = (config.primary_outcome, *config.secondary_outcomes)
    records = []
    for pid in pids:
        cond = assignment[pid]
        values = {
            o: float(baseline + effects.get(cond, {}).get(o, 0.0) + rng.normal(0, noise_sd))
            for o in outcomes
        }
        records.append(
            ResponseRecord(
                participant_id=pid,
                condition=cond,
                outcomes=values,
                attention_check_passed=bool(rng.random() >= exclusion_rate),
                completion_s=float(rng.uniform(90, 600)),
                simulated=True,
                study_version=config.config_version,
            )
        )
    return records
