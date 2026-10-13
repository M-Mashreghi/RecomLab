"""Versioned explanation conditions and evidence boundaries.

* ``none-v1``: no explanation; sees nothing.
* ``template-v1``: deterministic template (no LLM); sees all payload evidence.
* ``generic_llm-v1``: LLM, item-level; sees item facts and non-personal ranker
  features only.
* ``contextual_llm-v1``: LLM, personalized; additionally sees bounded user
  history, retrieval provenance and all ranker features.

Conditions never change the recommendation or its ranking; they only choose
how (and whether) an already-ranked item is explained.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from recomlab.explanations.evidence import EvidenceItem, EvidencePayload

NON_PERSONAL_FEATURES = frozenset({"popularity", "recency"})


@dataclass(frozen=True)
class Explanation:
    condition_id: str
    text: str | None
    cited_evidence: tuple[str, ...]
    generator: str
    abstained: bool = False
    fallback_from: str | None = None
    prompt_version: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class ExplanationCondition:
    condition_id: str
    strategy: str  # none | template | llm
    evidence_scope: str  # none | all | item_level | contextual
    description: str

    def allows(self, ev: EvidenceItem) -> bool:
        if self.evidence_scope == "none":
            return False
        if self.evidence_scope in ("all", "contextual"):
            return True
        # item_level: catalog facts and non-personal ranker features only
        if ev.kind == "item_fact":
            return True
        return ev.kind == "ranker_feature" and ev.data.get("feature") in NON_PERSONAL_FEATURES


CONDITIONS: Mapping[str, ExplanationCondition] = {
    c.condition_id: c
    for c in (
        ExplanationCondition("none-v1", "none", "none", "Recommendation only"),
        ExplanationCondition("template-v1", "template", "all", "Deterministic template"),
        ExplanationCondition("generic_llm-v1", "llm", "item_level", "LLM with item-level evidence"),
        ExplanationCondition(
            "contextual_llm-v1", "llm", "contextual", "LLM with bounded user-history evidence"
        ),
    )
}


def get_condition(condition_id: str) -> ExplanationCondition:
    try:
        return CONDITIONS[condition_id]
    except KeyError as err:
        raise ValueError(f"unknown explanation condition {condition_id!r}") from err


def restrict_evidence(payload: EvidencePayload, condition: ExplanationCondition) -> EvidencePayload:
    """Payload visible to a condition (evidence outside its scope is removed)."""
    kept = tuple(e for e in payload.evidence if condition.allows(e))
    return dataclasses.replace(
        payload, evidence=kept, metadata={**payload.metadata, "condition": condition.condition_id}
    )


class TemplateExplainer:
    """Deterministic, LLM-free explanation from structured evidence."""

    generator = "template-v1"

    def explain(self, payload: EvidencePayload) -> Explanation:
        hist = {e.data["item_id"]: e for e in payload.by_kind("history")}
        cited: list[str] = []
        parts: list[str] = []
        seeds: list[str] = []
        for src in payload.by_kind("retrieval"):
            for item in src.data.get("seed_items", []):
                if item in hist and item not in seeds:
                    seeds.append(item)
                    cited.extend([src.evidence_id, hist[item].evidence_id])
        if seeds:
            names = [hist[i].data.get("title") or i for i in seeds[:2]]
            parts.append("it is similar to " + " and ".join(names) + ", which you interacted with")
        cats = [e for e in payload.by_kind("item_fact") if "categories" in e.data]
        if cats and cats[0].data["categories"]:
            parts.append("it is in " + ", ".join(cats[0].data["categories"]))
            cited.append(cats[0].evidence_id)
        pop = [
            e for e in payload.by_kind("ranker_feature") if e.data.get("feature") == "popularity"
        ]
        if not parts and pop and pop[0].data.get("value", 0) > 0:
            parts.append("it is popular with other users")
            cited.append(pop[0].evidence_id)
        if not parts:
            return Explanation("template-v1", None, (), self.generator, abstained=True)
        text = "Recommended because " + "; ".join(parts) + "."
        return Explanation("template-v1", text, tuple(dict.fromkeys(cited)), self.generator)


class LLMExplainerProtocol(Protocol):
    def explain(self, payload: EvidencePayload, condition: ExplanationCondition) -> Explanation: ...


class ExplanationRouter:
    """Dispatch an assigned condition to its strategy with the right evidence scope."""

    def __init__(self, llm: LLMExplainerProtocol | None = None) -> None:
        self.template = TemplateExplainer()
        self.llm = llm

    def explain(self, payload: EvidencePayload, condition_id: str) -> Explanation:
        condition = get_condition(condition_id)
        scoped = restrict_evidence(payload, condition)
        if condition.strategy == "none":
            return Explanation(condition_id, None, (), "none")
        if condition.strategy == "template":
            return self.template.explain(scoped)
        if self.llm is None:
            fallback = self.template.explain(payload)
            return dataclasses.replace(fallback, fallback_from=condition_id)
        return self.llm.explain(scoped, condition)
