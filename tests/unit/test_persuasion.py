from datetime import UTC, datetime

import pytest

from recomlab.experiments.seed import make_rng
from recomlab.explanations.conditions import ExplanationRouter, TemplateExplainer
from recomlab.explanations.evidence import build_evidence
from recomlab.research.analysis import analyze
from recomlab.research.persuasion import (
    ResearchMode,
    ResearchModeError,
    apply_wording,
    code_free_text,
    export_coding,
    research_report,
    sanitize_free_text,
    stress_test,
)
from recomlab.research.study import Hypothesis, StudyConfig, StudyDataStore, simulate_responses

AS_OF = datetime(2024, 1, 14, tzinfo=UTC)
STUDY = StudyConfig(
    study_id="wording",
    version="1",
    research_questions=("Does wording change acceptance independently of fidelity?",),
    hypotheses=(Hypothesis("H1", "template > none", "acceptance", "template-v1", "none-v1"),),
    conditions=("none-v1", "template-v1"),
    primary_outcome="acceptance",
    block_size=4,
)


@pytest.fixture
def payload(tiny_ds):
    return build_evidence(
        "u2",
        "i05",
        history=tiny_ds.interactions,
        items=tiny_ds.items,
        sources={"item_item": {"rank": 1, "evidence": {"seed_items": [{"item_id": "i01"}]}}},
        contributions={"content": 0.3},
        as_of=AS_OF,
    )


def test_research_mode_cannot_silently_modify_explanations(payload):
    base = TemplateExplainer().explain(payload)
    with pytest.raises(ResearchModeError):
        apply_wording(base, "enthusiastic-v1", None)
    with pytest.raises(ResearchModeError):
        ResearchMode(STUDY)  # must be enabled explicitly
    mode = ResearchMode(STUDY, enabled=True)
    worded = apply_wording(base, "enthusiastic-v1", mode)
    assert worded.text.startswith("You will love this!") and "wording:" in worded.generator
    assert worded.cited_evidence == base.cited_evidence
    assert base.text and not base.text.startswith("You will love")  # original untouched
    assert ExplanationRouter().explain(payload, "template-v1") == base  # production path
    mode.close()
    with pytest.raises(ResearchModeError):
        mode.apply_wording(base, "neutral-v1")


def test_report_distinguishes_persuasion_from_correctness(payload, tiny_ds):
    mode = ResearchMode(STUDY, enabled=True)
    base = TemplateExplainer().explain(payload)
    worded = {
        v: (mode.apply_wording(base, v), payload)
        for v in ("neutral-v1", "enthusiastic-v1", "authority-v1", "social_proof-v1")
    }
    rng = make_rng(0, "sim-wording")
    shift = {"neutral-v1": 0.0, "enthusiastic-v1": 0.8, "authority-v1": 0.4, "social_proof-v1": 0.6}
    responses = [
        {"variant": v, "acceptance": 4 + shift[v] + rng.normal(0, 1), "simulated": True}
        for v in shift
        for _ in range(150)
    ]
    result = stress_test(responses, worded, tiny_ds.items, n_resamples=300)
    assert result["simulated"]
    persuasive = {r["variant"]: r for r in result["persuasion"]}
    assert persuasive["enthusiastic-v1"]["ci_low"] > 0  # wording raises acceptance ...
    assert result["correctness"]["neutral-v1"]["accepted"]
    for v in ("enthusiastic-v1", "authority-v1", "social_proof-v1"):
        assert not result["correctness"][v]["accepted"]  # ... while being unfaithful
    store = StudyDataStore(STUDY)
    for r in simulate_responses(STUDY, 60, {"template-v1": {"acceptance": 0.5}}):
        store.add(r)
    report = research_report(
        STUDY, analyze(STUDY, store, n_resamples=200), result, {"accepted_rate": 0.25}
    )
    assert "SIMULATED DATA" in report
    for section in (
        "## Methods",
        "## Results",
        "## Explanation Fidelity",
        "## Persuasion vs Correctness",
        "## Limitations",
    ):
        assert section in report
    assert "(persuasion)" in report and "(correctness)" in report


def test_free_text_sanitized_before_coding(tmp_path):
    raw = (
        "Mail me at jane.doe@example.com or +1 (555) 123-4567, see https://x.io/p "
        "id 1234567. I did not trust it, felt pushy. " + "x" * 600
    )
    clean, flags = sanitize_free_text(raw)
    assert "@" not in clean and "https" not in clean and "1234567" not in clean
    assert set(flags) == {"email", "url", "phone", "digits", "truncated"}
    seen = []

    def spy_coder(text):
        seen.append(text)
        return ["llm-code"]

    rows = code_free_text(
        [{"participant_id": "SIM-1", "variant": "neutral-v1", "text": raw}], spy_coder
    )
    assert seen and "@" not in seen[0]
    default = code_free_text([{"participant_id": "SIM-2", "text": "felt pushy, can't trust"}])
    assert default[0]["codes"] == "manipulation;trust"
    path = export_coding(rows, tmp_path / "coding.csv")
    assert "jane.doe" not in path.read_text(encoding="utf-8")
