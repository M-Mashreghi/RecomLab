import dataclasses
import json
from datetime import UTC, datetime

import pytest

from recomlab.domain.schemas import Item
from recomlab.explanations.evidence import (
    EVIDENCE_SCHEMA,
    EvidenceError,
    EvidenceItem,
    build_evidence,
    validate_payload,
)

AS_OF = datetime(2024, 1, 14, tzinfo=UTC)
SOURCES = {
    "item_item": {
        "rank": 1,
        "evidence": {"seed_items": [{"item_id": "i01", "similarity": 0.8}]},
    },
    "popularity": {"rank": 3, "evidence": {"train_count": 4}},
}
CONTRIB = {"content": 0.3, "popularity": 0.1}


@pytest.fixture
def payload(tiny_ds):
    return build_evidence(
        "u2",
        "i05",
        history=tiny_ds.interactions,
        items=tiny_ds.items,
        sources=SOURCES,
        contributions=CONTRIB,
        as_of=AS_OF,
        history_window=2,
    )


def test_payload_schema_and_roles(payload):
    data = json.loads(json.dumps(payload.to_dict()))
    assert data["schema_version"] == EVIDENCE_SCHEMA
    kinds = {e["kind"] for e in data["evidence"]}
    assert kinds == {"item_fact", "retrieval", "ranker_feature", "history"}
    assert {e.evidence_id for e in payload.ranking_evidence()} >= {
        "src:item_item",
        "src:popularity",
        "feat:content",
        "feat:popularity",
    }
    item_facts = payload.by_kind("item_fact")
    assert all(e.role == "presentation" for e in item_facts)


def test_traceable_to_stored_interactions_and_metadata(payload, tiny_ds):
    validate_payload(payload, tiny_ds.interactions, tiny_ds.items)
    hist = payload.by_kind("history")
    # window of 2 most recent + seed item i01 (older) kept because it was used for ranking
    assert len(hist) == 3
    seed = [e for e in hist if e.data["item_id"] == "i01"]
    assert seed and seed[0].role == "ranking"
    assert all(e.data["event_time"] < AS_OF.isoformat() for e in hist)


def test_unknown_facts_never_synthesized(tiny_ds):
    items = [*tiny_ds.items, Item("i99")]  # no title, no categories
    p = build_evidence(
        "u7",
        "i99",
        history=tiny_ds.interactions,
        items=items,
        sources={},
        contributions={},
        as_of=AS_OF,
    )
    reasons = {(u["kind"], u.get("field")) for u in p.unavailable}
    assert ("item_fact", "title") in reasons and ("item_fact", "categories") in reasons
    assert ("history", None) in reasons and ("retrieval", None) in reasons
    assert ("ranker_feature", None) in reasons
    assert p.evidence == ()
    with pytest.raises(EvidenceError):
        build_evidence(
            "u1", "nope", history=[], items=items, sources={}, contributions={}, as_of=AS_OF
        )


def test_fabricated_evidence_rejected(payload, tiny_ds):
    fake_hist = EvidenceItem("hist:e999", "history", "presentation", {"item_id": "i01"})
    with pytest.raises(EvidenceError):
        validate_payload(
            dataclasses.replace(payload, evidence=(*payload.evidence, fake_hist)),
            tiny_ds.interactions,
            tiny_ds.items,
        )
    other_user = EvidenceItem("hist:e001", "history", "presentation", {"item_id": "i01"})
    with pytest.raises(EvidenceError):  # e001 belongs to u1, not u2
        validate_payload(
            dataclasses.replace(payload, evidence=(other_user,)),
            tiny_ds.interactions,
            tiny_ds.items,
        )
    wrong_fact = EvidenceItem("item:i05:title", "item_fact", "presentation", {"title": "Oscar"})
    with pytest.raises(EvidenceError):
        validate_payload(
            dataclasses.replace(payload, evidence=(wrong_fact,)),
            tiny_ds.interactions,
            tiny_ds.items,
        )


def test_history_window_is_privacy_bounded(tiny_ds):
    p = build_evidence(
        "u1",
        "i09",
        history=tiny_ds.interactions,
        items=tiny_ds.items,
        sources={},
        contributions={},
        as_of=AS_OF,
        history_window=1,
    )
    assert len(p.by_kind("history")) == 1
    assert (
        build_evidence(
            "u1",
            "i09",
            history=tiny_ds.interactions,
            items=tiny_ds.items,
            sources={},
            contributions={},
            as_of=AS_OF,
            history_window=0,
        ).by_kind("history")
        == []
    )
    with pytest.raises(EvidenceError):
        EvidenceItem("x", "history", "secret", {})
