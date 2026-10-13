import json
from datetime import UTC, datetime

import pytest

from recomlab.explanations.conditions import Explanation, TemplateExplainer
from recomlab.explanations.evidence import build_evidence
from recomlab.explanations.fidelity import (
    FidelityLog,
    feature_ablation,
    validate_explanation,
)
from recomlab.features import PointInTimeFeatures
from recomlab.rankers import RankRequest
from recomlab.rankers.heuristic import HeuristicRanker

AS_OF = datetime(2024, 1, 14, tzinfo=UTC)


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
        history_window=3,
    )


def exp(text, cited=("src:item_item",)):
    return Explanation("contextual_llm-v1", text, tuple(cited), "llm:mock")


def test_valid_grounded_explanations_pass(payload, tiny_ds):
    template = TemplateExplainer().explain(payload)
    assert validate_explanation(template, payload, tiny_ds.items).accepted
    grounded = exp("You watched Synthetic Film 01, an action film like this scifi pick.")
    res = validate_explanation(grounded, payload, tiny_ds.items)
    assert res.accepted and res.unsupported_rate == 0.0
    assert {c.kind for c in res.claims} >= {"citation", "item", "category", "history"}


@pytest.mark.parametrize(
    ("text", "cited", "kind"),
    [
        ("Because you loved Synthetic Film 09.", ("src:item_item",), "item"),
        ("A great romance for you.", ("src:item_item",), "category"),
        ("It is trending right now.", ("src:item_item",), "popularity"),
        ("Matches your taste.", ("hist:e999",), "citation"),
    ],
)
def test_synthetic_hallucinations_detected(payload, tiny_ds, text, cited, kind):
    res = validate_explanation(exp(text, cited), payload, tiny_ds.items)
    assert not res.accepted
    assert kind in {c.kind for c in res.unsupported}


def test_history_claim_without_history(tiny_ds):
    cold = build_evidence(
        "u7",
        "i11",
        history=tiny_ds.interactions,
        items=tiny_ds.items,
        sources={},
        contributions={},
        as_of=AS_OF,
    )
    res = validate_explanation(exp("You watched lots of drama.", ()), cold, tiny_ds.items)
    assert not res.accepted and "history" in {c.kind for c in res.unsupported}


def test_threshold_controls_rejection(payload, tiny_ds):
    mixed = exp("You watched Synthetic Film 01 and it is trending.")
    strict = validate_explanation(mixed, payload, tiny_ds.items, threshold=0.0)
    lenient = validate_explanation(mixed, payload, tiny_ds.items, threshold=0.5)
    assert not strict.accepted and lenient.accepted
    assert 0 < strict.unsupported_rate <= 0.5


def test_feature_ablation_on_heuristic_ranker(tiny_ds, tiny_split):
    feats = PointInTimeFeatures(tiny_split.train, tiny_ds.items, AS_OF)
    ranked = HeuristicRanker(feats).rank(
        RankRequest.from_items("u5", ["i08", "i10", "i11", "i01"], 4)
    )
    scores = {i.item_id: i.raw_score for i in ranked.items}
    contribs = {i.item_id: dict(i.contributions) for i in ranked.items}
    top = ranked.items[0].item_id
    abl = feature_ablation(scores, contribs, top, ["content", "retrieval"])
    assert abl["effects"]["content"]["score_delta"] == pytest.approx(-contribs[top]["content"])
    assert abl["effects"]["content"]["rank_after"] >= abl["effects"]["content"]["rank_before"]
    # retrieval feature is constant (all retrieval scores 0) -> normalized to 1.0, has effect;
    # a feature with zero contribution is flagged
    abl_zero = feature_ablation(scores, {top: {"recency": 0.0}}, top, ["recency"])
    assert abl_zero["features_without_effect"] == ["recency"]
    res = validate_explanation(
        Explanation("template-v1", None, (), "template-v1"),
        build_evidence(
            "u5", top, history=[], items=tiny_ds.items, sources={}, contributions={}, as_of=AS_OF
        ),
        tiny_ds.items,
        ablation=abl_zero,
    )
    assert not res.accepted


def test_fidelity_stored_separately(payload, tiny_ds, tmp_path):
    log = FidelityLog(tmp_path / "fidelity.jsonl")
    res = validate_explanation(exp("Because you loved Synthetic Film 09."), payload, tiny_ds.items)
    log.append(res)
    rows = log.read()
    assert rows[0]["record_type"] == "fidelity" and not rows[0]["accepted"]
    assert not {"trust", "acceptance", "satisfaction"} & set(rows[0])
    with (tmp_path / "fidelity.jsonl").open("a") as fh:
        fh.write(json.dumps({"record_type": "perception", "trust": 5}) + "\n")
    with pytest.raises(ValueError):
        log.read()
