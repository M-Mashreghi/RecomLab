"""Explanation fidelity & unsupported-claim validation.

Fidelity is a *system-side* property: does every claim in an explanation trace
to the recommender's evidence payload? It is stored separately from any user
perception outcome (trust, acceptance, ...).

Claim checks (rule-based, deterministic):

* ``citation``: every cited evidence id exists in the payload;
* ``item``: every catalog title mentioned is the recommended item or appears in
  the payload's history evidence;
* ``category``: every catalog category mentioned is a category of the
  recommended item or of a history-evidence item;
* ``history``: "you watched/rated/liked/bought/clicked ..." statements need
  history evidence;
* ``popularity``: popularity/trending/social-proof statements need popularity evidence;
* ``ungroundable``: predicted feelings ("you will love") and appeals to authority are
  never supported by recommender evidence.

A perturbation check (:func:`feature_ablation`) verifies, for rankers that
expose additive contributions, that each cited ranker feature actually moves
the item's score/rank when removed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from recomlab.domain.schemas import Item
from recomlab.explanations.conditions import Explanation
from recomlab.explanations.evidence import EvidencePayload

HISTORY_VERBS = re.compile(
    r"\byou(?:'ve| have)?\s+(?:watched|rated|liked|loved|bought|purchased|clicked|viewed)\b",
    re.IGNORECASE,
)
POPULARITY_WORDS = re.compile(
    r"\b(popular|trending|most[- ]watched|bestsell\w*|everyone|most people)\b", re.IGNORECASE
)
# Claims the recommender can never ground: predicted feelings and appeals to authority.
UNGROUNDABLE = re.compile(
    r"\b(you(?:'ll| will) (?:love|enjoy|adore)|experts?\b|guaranteed|you must)", re.IGNORECASE
)


@dataclass(frozen=True)
class Claim:
    kind: str
    value: str
    supported: bool
    reason: str = ""


@dataclass(frozen=True)
class FidelityResult:
    explanation_id: str
    condition_id: str
    claims: tuple[Claim, ...]
    unsupported_rate: float
    threshold: float
    accepted: bool
    ablation: Mapping[str, Any] = field(default_factory=dict)
    record_type: str = "fidelity"  # never mixed with perception outcomes

    @property
    def unsupported(self) -> list[Claim]:
        return [c for c in self.claims if not c.supported]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["claims"] = [asdict(c) for c in self.claims]
        out["ablation"] = dict(self.ablation)
        return out


def _mentions(text: str, phrase: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, re.IGNORECASE) is not None


def extract_claims(
    explanation: Explanation, payload: EvidencePayload, catalog: Sequence[Item]
) -> list[Claim]:
    text = explanation.text or ""
    claims: list[Claim] = []
    for cid in explanation.cited_evidence:
        ok = cid in payload.ids
        claims.append(Claim("citation", cid, ok, "" if ok else "cited id not in evidence payload"))
    history = payload.by_kind("history")
    history_items = {e.data["item_id"] for e in history}
    evidence_cats: set[str] = set()
    for e in history:
        evidence_cats.update(e.data.get("categories", []))
    rec_item = next((i for i in catalog if i.item_id == payload.item_id), None)
    rec_cats = set(rec_item.categories) if rec_item else set()
    evidence_cats |= {c for e in payload.by_kind("item_fact") for c in e.data.get("categories", [])}
    for item in catalog:
        if item.title and _mentions(text, item.title):
            ok = item.item_id == payload.item_id or item.item_id in history_items
            claims.append(
                Claim("item", item.title, ok, "" if ok else "item not in evidence payload")
            )
    vocabulary = sorted({c for i in catalog for c in i.categories})
    for cat in vocabulary:
        if _mentions(text, cat):
            ok = cat in evidence_cats or cat in rec_cats
            claims.append(Claim("category", cat, ok, "" if ok else "category not in evidence"))
    history_claim = HISTORY_VERBS.search(text)
    if history_claim is not None:
        ok = bool(history)
        reason = "" if ok else "history claim without history evidence"
        claims.append(Claim("history", history_claim.group(0), ok, reason))
    for m in UNGROUNDABLE.finditer(text):
        claims.append(Claim("ungroundable", m.group(0), False, "not derivable from evidence"))
    if POPULARITY_WORDS.search(text):
        ok = any(
            e.data.get("feature") == "popularity" for e in payload.by_kind("ranker_feature")
        ) or any(e.data.get("source") == "popularity" for e in payload.by_kind("retrieval"))
        claims.append(Claim("popularity", "popularity", ok, "" if ok else "no popularity evidence"))
    return claims


def validate_explanation(
    explanation: Explanation,
    payload: EvidencePayload,
    catalog: Sequence[Item],
    *,
    threshold: float = 0.0,
    explanation_id: str | None = None,
    ablation: Mapping[str, Any] | None = None,
) -> FidelityResult:
    """Score unsupported claims; reject when the unsupported rate exceeds ``threshold``."""
    claims = extract_claims(explanation, payload, catalog) if explanation.text else []
    unsupported = sum(1 for c in claims if not c.supported)
    rate = unsupported / len(claims) if claims else 0.0
    ablation_failed = bool(ablation and ablation.get("features_without_effect"))
    return FidelityResult(
        explanation_id=explanation_id
        or f"{payload.user_id}:{payload.item_id}:{explanation.condition_id}",
        condition_id=explanation.condition_id,
        claims=tuple(claims),
        unsupported_rate=rate,
        threshold=threshold,
        accepted=rate <= threshold and not ablation_failed,
        ablation=dict(ablation or {}),
    )


def feature_ablation(
    scores: Mapping[str, float],
    contributions: Mapping[str, Mapping[str, float]],
    item_id: str,
    cited_features: Sequence[str],
    epsilon: float = 1e-9,
) -> dict[str, Any]:
    """Counterfactual check for additive rankers: remove each cited feature's contribution.

    Returns the score delta and the item's rank before/after for every cited
    feature, plus the cited features that have no effect.
    """

    def rank_of(table: Mapping[str, float]) -> int:
        order = sorted(table, key=lambda i: (-table[i], i))
        return order.index(item_id) + 1

    base_rank = rank_of(scores)
    effects: dict[str, dict[str, float]] = {}
    no_effect: list[str] = []
    for feat in cited_features:
        contrib = contributions.get(item_id, {}).get(feat, 0.0)
        perturbed = dict(scores)
        perturbed[item_id] = scores[item_id] - contrib
        effects[feat] = {
            "score_delta": -contrib,
            "rank_before": float(base_rank),
            "rank_after": float(rank_of(perturbed)),
        }
        if abs(contrib) <= epsilon:
            no_effect.append(feat)
    return {"item_id": item_id, "effects": effects, "features_without_effect": no_effect}


class FidelityLog:
    """Append-only JSONL store for fidelity results (separate from perception data)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, result: FidelityResult) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = [json.loads(line) for line in self.path.read_text("utf-8").splitlines() if line]
        if any(r.get("record_type") != "fidelity" for r in rows):
            raise ValueError("fidelity log contains non-fidelity records")
        return rows
