import ast
from datetime import UTC, datetime
from pathlib import Path

import pytest

from recomlab.explanations.conditions import (
    CONDITIONS,
    Explanation,
    ExplanationRouter,
    TemplateExplainer,
    get_condition,
    restrict_evidence,
)
from recomlab.explanations.evidence import build_evidence

AS_OF = datetime(2024, 1, 14, tzinfo=UTC)


@pytest.fixture
def payload(tiny_ds):
    return build_evidence(
        "u2",
        "i05",
        history=tiny_ds.interactions,
        items=tiny_ds.items,
        sources={"item_item": {"rank": 1, "evidence": {"seed_items": [{"item_id": "i01"}]}}},
        contributions={"content": 0.3, "popularity": 0.1, "retrieval": 0.4},
        as_of=AS_OF,
    )


class RecordingLLM:
    def __init__(self):
        self.seen = []

    def explain(self, payload, condition):
        self.seen.append((condition.condition_id, payload))
        return Explanation(condition.condition_id, "llm text", (), "mock-llm")


def test_condition_ids_are_stable_and_versioned():
    assert set(CONDITIONS) == {"none-v1", "template-v1", "generic_llm-v1", "contextual_llm-v1"}
    with pytest.raises(ValueError):
        get_condition("persuasive-v9")


def test_assignment_produces_requested_strategy(payload):
    llm = RecordingLLM()
    router = ExplanationRouter(llm)
    assert router.explain(payload, "none-v1") == Explanation("none-v1", None, (), "none")
    tmpl = router.explain(payload, "template-v1")
    assert tmpl.generator == "template-v1" and tmpl.text.startswith("Recommended because")
    assert router.explain(payload, "generic_llm-v1").generator == "mock-llm"
    assert router.explain(payload, "contextual_llm-v1").generator == "mock-llm"
    assert [c for c, _ in llm.seen] == ["generic_llm-v1", "contextual_llm-v1"]
    no_llm = ExplanationRouter(None).explain(payload, "contextual_llm-v1")
    assert no_llm.generator == "template-v1" and no_llm.fallback_from == "contextual_llm-v1"


def test_template_is_deterministic_cites_evidence_and_has_no_llm_dependency(payload):
    a = TemplateExplainer().explain(payload)
    assert a == TemplateExplainer().explain(payload)
    assert "Synthetic Film 01" in a.text and "action" in a.text
    assert set(a.cited_evidence) <= payload.ids and a.cited_evidence
    src = (
        Path(__file__).resolve().parents[2] / "src" / "recomlab" / "explanations" / "conditions.py"
    )
    imported = {
        n.module for n in ast.walk(ast.parse(src.read_text())) if isinstance(n, ast.ImportFrom)
    }
    assert not any(m and ("llm" in m or "anthropic" in m or "openai" in m) for m in imported)


def test_generic_vs_contextual_evidence_boundaries(payload):
    generic = restrict_evidence(payload, get_condition("generic_llm-v1"))
    contextual = restrict_evidence(payload, get_condition("contextual_llm-v1"))
    assert {e.kind for e in generic.evidence} == {"item_fact", "ranker_feature"}
    assert {e.data["feature"] for e in generic.by_kind("ranker_feature")} == {"popularity"}
    assert not generic.by_kind("history") and not generic.by_kind("retrieval")
    assert contextual.by_kind("history") and contextual.by_kind("retrieval")
    assert restrict_evidence(payload, get_condition("none-v1")).evidence == ()
    assert generic.metadata["condition"] == "generic_llm-v1"


def test_template_abstains_without_evidence(tiny_ds):
    empty = build_evidence(
        "u7", "i11", history=[], items=tiny_ds.items, sources={}, contributions={}, as_of=AS_OF
    )
    out = TemplateExplainer().explain(restrict_evidence(empty, get_condition("none-v1")))
    assert out.abstained and out.text is None
