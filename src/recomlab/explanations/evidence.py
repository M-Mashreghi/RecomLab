"""Recommendation grounding evidence (explanation input payload).

Every evidence entry is traceable to stored data: a user interaction
(``hist:<event_id>``), item metadata (``item:<item_id>:<field>``), a retrieval
source (``src:<source>``) or a ranker feature contribution
(``feat:<feature>``). Evidence is tagged with its ``role``:

* ``ranking``: information the recommender actually used to produce the
  recommendation (retrieval provenance, seed items, ranker contributions);
* ``presentation``: context shown to explain but not used for ranking
  (item title/categories, other recent history items).

Facts that are not available are listed in ``unavailable``; nothing is
synthesized. User history is limited to a privacy-safe window.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from recomlab.domain.schemas import Interaction, Item, sort_key

EVIDENCE_SCHEMA = "evidence-1"
ROLES = ("ranking", "presentation")


class EvidenceError(ValueError):
    pass


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    kind: str  # history | item_fact | retrieval | ranker_feature
    role: str
    data: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise EvidenceError(f"role must be one of {ROLES}")


@dataclass(frozen=True)
class EvidencePayload:
    user_id: str
    item_id: str
    evidence: tuple[EvidenceItem, ...]
    unavailable: tuple[Mapping[str, str], ...] = ()
    history_window: int = 5
    schema_version: str = EVIDENCE_SCHEMA
    as_of: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ids(self) -> set[str]:
        return {e.evidence_id for e in self.evidence}

    def by_kind(self, kind: str) -> list[EvidenceItem]:
        return [e for e in self.evidence if e.kind == kind]

    def ranking_evidence(self) -> list[EvidenceItem]:
        return [e for e in self.evidence if e.role == "ranking"]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["evidence"] = [{**asdict(e), "data": dict(e.data)} for e in self.evidence]
        return out


def build_evidence(
    user_id: str,
    item_id: str,
    *,
    history: Sequence[Interaction],
    items: Sequence[Item],
    sources: Mapping[str, Mapping[str, Any]],
    contributions: Mapping[str, float],
    as_of: datetime,
    history_window: int = 5,
) -> EvidencePayload:
    """Assemble bounded, traceable evidence for one recommended item."""
    if history_window < 0:
        raise EvidenceError("history_window must be >= 0")
    catalog = {i.item_id: i for i in items}
    evidence: list[EvidenceItem] = []
    unavailable: list[dict[str, str]] = []

    item = catalog.get(item_id)
    if item is None:
        raise EvidenceError(f"recommended item {item_id} is not in the catalog")
    if item.title:
        evidence.append(
            EvidenceItem(
                f"item:{item_id}:title", "item_fact", "presentation", {"title": item.title}
            )
        )
    else:
        unavailable.append({"kind": "item_fact", "field": "title", "reason": "not in catalog"})
    if item.categories:
        evidence.append(
            EvidenceItem(
                f"item:{item_id}:categories",
                "item_fact",
                "presentation",
                {"categories": list(item.categories)},
            )
        )
    else:
        unavailable.append({"kind": "item_fact", "field": "categories", "reason": "no metadata"})

    # Retrieval provenance (+ seed items actually used by item-item retrieval).
    seed_items: set[str] = set()
    for source, info in sorted(sources.items()):
        data = {"source": source, "rank": info.get("rank")}
        seeds = [s["item_id"] for s in info.get("evidence", {}).get("seed_items", [])]
        if seeds:
            data["seed_items"] = seeds
            seed_items.update(seeds)
        evidence.append(EvidenceItem(f"src:{source}", "retrieval", "ranking", data))
    if not sources:
        unavailable.append(
            {"kind": "retrieval", "reason": "fallback item without retrieval source"}
        )

    for feature, value in sorted(contributions.items()):
        evidence.append(
            EvidenceItem(
                f"feat:{feature}", "ranker_feature", "ranking", {"feature": feature, "value": value}
            )
        )
    if not contributions:
        unavailable.append({"kind": "ranker_feature", "reason": "ranker exposed no contributions"})

    # Bounded user history before as_of (most recent first).
    user_events = sorted(
        (e for e in history if e.user_id == user_id and e.event_time < as_of),
        key=sort_key,
        reverse=True,
    )
    window = user_events[:history_window]
    window_items = {e.item_id for e in window}
    # Seed items used for ranking are always included (they are ranking evidence).
    extra = [e for e in user_events[history_window:] if e.item_id in seed_items - window_items]
    for e in [*window, *extra]:
        hist_item = catalog.get(e.item_id)
        evidence.append(
            EvidenceItem(
                f"hist:{e.event_id}",
                "history",
                "ranking" if e.item_id in seed_items else "presentation",
                {
                    "item_id": e.item_id,
                    "event_type": str(e.event_type),
                    "event_time": e.event_time.isoformat(),
                    "title": hist_item.title if hist_item else None,
                    "categories": list(hist_item.categories) if hist_item else [],
                },
            )
        )
    if not user_events:
        unavailable.append({"kind": "history", "reason": "no interactions before as_of"})
    return EvidencePayload(
        user_id=user_id,
        item_id=item_id,
        evidence=tuple(evidence),
        unavailable=tuple(unavailable),
        history_window=history_window,
        as_of=as_of.isoformat(),
    )


def validate_payload(
    payload: EvidencePayload, history: Sequence[Interaction], items: Sequence[Item]
) -> None:
    """Raise :class:`EvidenceError` unless every evidence entry traces to stored data."""
    events = {e.event_id: e for e in history}
    catalog = {i.item_id: i for i in items}
    for ev in payload.evidence:
        if ev.kind == "history":
            event_id = ev.evidence_id.removeprefix("hist:")
            stored = events.get(event_id)
            if stored is None or stored.user_id != payload.user_id:
                raise EvidenceError(f"{ev.evidence_id}: no such interaction for this user")
            if stored.item_id != ev.data["item_id"]:
                raise EvidenceError(f"{ev.evidence_id}: item does not match stored interaction")
        elif ev.kind == "item_fact":
            item = catalog.get(payload.item_id)
            if item is None:
                raise EvidenceError("item not in catalog")
            if "title" in ev.data and ev.data["title"] != item.title:
                raise EvidenceError(f"{ev.evidence_id}: title differs from catalog")
            if "categories" in ev.data and list(ev.data["categories"]) != list(item.categories):
                raise EvidenceError(f"{ev.evidence_id}: categories differ from catalog")
        elif ev.kind not in ("retrieval", "ranker_feature"):
            raise EvidenceError(f"{ev.evidence_id}: unknown evidence kind {ev.kind}")
