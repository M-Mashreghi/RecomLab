import json
import logging
from datetime import UTC, datetime

import pytest

from recomlab.explanations.conditions import ExplanationRouter, get_condition
from recomlab.explanations.evidence import build_evidence
from recomlab.explanations.llm import (
    PROMPT_VERSION,
    MockProvider,
    ProviderError,
    StructuredLLMExplainer,
)

AS_OF = datetime(2024, 1, 14, tzinfo=UTC)
CONTEXTUAL = get_condition("contextual_llm-v1")


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


def ok(cited, text="Similar to a film you rated highly.", abstain=False):
    return json.dumps(
        {"explanation": text, "cited_evidence_ids": cited, "abstain": abstain, "uncertainty": "low"}
    )


def test_valid_output_used_with_versions(payload, caplog):
    provider = MockProvider([ok(["src:item_item", "feat:content"])])
    with caplog.at_level(logging.INFO, logger="recomlab.explanations.llm"):
        out = StructuredLLMExplainer(provider).explain(payload, CONTEXTUAL)
    assert out.generator == "llm:mock" and out.text.startswith("Similar")
    assert out.cited_evidence == ("src:item_item", "feat:content")
    assert out.prompt_version == PROMPT_VERSION and out.model == "mock-1"
    rec = [r for r in caplog.records if r.getMessage() == "llm explanation generated"][0]
    assert rec.prompt_version == PROMPT_VERSION and rec.model == "mock-1"
    assert "evidence" in provider.calls[0] and "u2" not in provider.calls[0].split("evidence")[0]


def test_malformed_then_valid_is_retried(payload):
    provider = MockProvider(["not json", ok(["feat:content"])])
    out = StructuredLLMExplainer(provider, max_attempts=3).explain(payload, CONTEXTUAL)
    assert out.generator == "llm:mock" and len(provider.calls) == 2


@pytest.mark.parametrize(
    "bad",
    [
        "{}",
        ok(["hist:does-not-exist"]),  # unsupported citation
        ok([]),  # claims without citations
        json.dumps(
            {
                "explanation": "x",
                "cited_evidence_ids": ["feat:content"],
                "abstain": False,
                "uncertainty": "certain",
            }
        ),
    ],
)
def test_unsupported_or_invalid_outputs_fall_back(payload, bad):
    provider = MockProvider([bad, bad])
    out = StructuredLLMExplainer(provider, max_attempts=2).explain(payload, CONTEXTUAL)
    assert out.generator == "template-v1" and out.fallback_from == "contextual_llm-v1"
    assert out.prompt_version == PROMPT_VERSION
    assert len(provider.calls) == 2  # bounded retries


def test_abstain_is_respected(payload):
    out = StructuredLLMExplainer(MockProvider([ok([], text="", abstain=True)])).explain(
        payload, CONTEXTUAL
    )
    assert out.abstained and out.text is None and out.generator == "llm:mock"


def test_provider_failures(payload):
    transient = MockProvider([ProviderError("429", True), ok(["feat:content"])])
    assert StructuredLLMExplainer(transient).explain(payload, CONTEXTUAL).generator == "llm:mock"
    permanent = MockProvider([ProviderError("401", False), ok(["feat:content"])])
    out = StructuredLLMExplainer(permanent).explain(payload, CONTEXTUAL)
    assert out.fallback_from and len(permanent.calls) == 1  # non-transient: no retry
    crash = MockProvider([RuntimeError("boom")])
    none_fb = StructuredLLMExplainer(crash, fallback="none").explain(payload, CONTEXTUAL)
    assert none_fb.text is None and none_fb.abstained


@pytest.mark.integration
def test_provider_failure_never_blocks_recommendations(
    tmp_path, bundle_factory, serving_context, tiny_ds
):
    from recomlab.registry.model_registry import ModelRegistry
    from recomlab.services.recommendation import RecommendationRequest, RecommendationService

    items, history, as_of = serving_context
    reg = ModelRegistry(tmp_path / "reg")
    svc = RecommendationService(reg, items, history, as_of)
    svc.activate(bundle_factory(reg).bundle_id)
    resp = svc.recommend(RecommendationRequest("u2", 3, trace_id="t"))
    router = ExplanationRouter(
        StructuredLLMExplainer(MockProvider([ProviderError("down", True)] * 9))
    )
    for item in resp.items:
        ev = build_evidence(
            "u2",
            item.item_id,
            history=history,
            items=items,
            sources=item.sources,
            contributions=item.contributions,
            as_of=as_of,
        )
        out = router.explain(ev, "contextual_llm-v1")
        assert out.fallback_from == "contextual_llm-v1"
    again = svc.recommend(RecommendationRequest("u2", 3, trace_id="t"))
    assert again.item_ids == resp.item_ids  # explanations never alter ranking
