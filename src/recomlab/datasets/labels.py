"""Explicit-to-implicit label policy and relevance semantics.

Raw feedback is never modified. A :class:`LabelPolicy` derives versioned
relevance labels per event; evaluation and training read the policy from the
run manifest (``RunManifest.label_policy``) so semantics are reproducible.

Relevance modes:

* ``binary``: relevance is 1.0 for positives and 0.0 otherwise.
* ``graded``: relevance is the event grade (rating above threshold, or the
  configured implicit event weight); 0.0 for non-positives.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from recomlab.domain.schemas import EventType, Interaction

LABEL_POLICY_SCHEMA = "1.0"
RELEVANCE_MODES = ("binary", "graded")


class LabelPolicyError(ValueError):
    pass


def _default_weights() -> dict[str, float]:
    return {"purchase": 3.0, "like": 2.0, "watch": 2.0, "add_to_cart": 1.0, "click": 1.0}


@dataclass(frozen=True)
class LabelPolicy:
    name: str = "default"
    relevance_mode: str = "binary"
    rating_positive_threshold: float = 4.0  # rating >= threshold is positive
    watch_positive_threshold: float = 0.8  # fraction watched >= threshold is positive
    positive_event_types: tuple[str, ...] = ("rating", "watch", "purchase", "like")
    implicit_weights: Mapping[str, float] = field(default_factory=_default_weights)

    def __post_init__(self) -> None:
        if self.relevance_mode not in RELEVANCE_MODES:
            raise LabelPolicyError(f"relevance_mode must be one of {RELEVANCE_MODES}")
        unknown = set(self.positive_event_types) - {e.value for e in EventType}
        if unknown:
            raise LabelPolicyError(f"unknown event types {sorted(unknown)}")
        if not 0.0 <= self.watch_positive_threshold <= 1.0:
            raise LabelPolicyError("watch_positive_threshold must be in [0, 1]")
        object.__setattr__(self, "positive_event_types", tuple(sorted(self.positive_event_types)))
        object.__setattr__(self, "implicit_weights", dict(sorted(self.implicit_weights.items())))

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["positive_event_types"] = list(self.positive_event_types)
        out["implicit_weights"] = dict(self.implicit_weights)
        out["schema"] = LABEL_POLICY_SCHEMA
        out["label_version"] = self.label_version
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> LabelPolicy:
        data = {k: v for k, v in raw.items() if k not in ("schema", "label_version")}
        if "positive_event_types" in data:
            data["positive_event_types"] = tuple(data["positive_event_types"])
        policy = cls(**data)
        if "label_version" in raw and raw["label_version"] != policy.label_version:
            raise LabelPolicyError("label_version does not match policy content")
        return policy

    @property
    def label_version(self) -> str:
        body = {k: v for k, v in asdict(self).items()}
        body["positive_event_types"] = list(self.positive_event_types)
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
        return f"labels-{self.name}-{digest[:10]}"


@dataclass(frozen=True, slots=True)
class Label:
    event_id: str
    user_id: str
    item_id: str
    event_time: datetime
    positive: bool
    relevance: float
    explicit_negative: bool = False


def _grade(event: Interaction, policy: LabelPolicy) -> tuple[bool, float, bool]:
    etype = str(event.event_type)
    if etype not in policy.positive_event_types:
        return False, 0.0, False
    if event.event_type is EventType.RATING:
        assert event.value is not None
        if event.value >= policy.rating_positive_threshold:
            return True, event.value - policy.rating_positive_threshold + 1.0, False
        return False, 0.0, True
    if event.event_type is EventType.WATCH and (
        event.value is None or event.value < policy.watch_positive_threshold
    ):
        return False, 0.0, False
    return True, float(policy.implicit_weights.get(etype, 1.0)), False


def derive_labels(events: Iterable[Interaction], policy: LabelPolicy) -> list[Label]:
    """Derive one label per raw event; inputs are not modified."""
    labels = []
    for ev in events:
        positive, grade, negative = _grade(ev, policy)
        relevance = (1.0 if positive else 0.0) if policy.relevance_mode == "binary" else grade
        labels.append(
            Label(ev.event_id, ev.user_id, ev.item_id, ev.event_time, positive, relevance, negative)
        )
    return labels


def relevance_by_user(labels: Iterable[Label]) -> dict[str, dict[str, float]]:
    """Aggregate positive labels to ``{user: {item: max relevance}}``."""
    out: dict[str, dict[str, float]] = {}
    for lab in labels:
        if lab.positive:
            user = out.setdefault(lab.user_id, {})
            user[lab.item_id] = max(user.get(lab.item_id, 0.0), lab.relevance)
    return out


def label_policy_from_manifest(manifest_label_policy: Mapping[str, Any]) -> LabelPolicy:
    """Rebuild the policy recorded in a run manifest; required for evaluation."""
    if not manifest_label_policy:
        raise LabelPolicyError("run manifest has no label policy; evaluation refuses to guess")
    return LabelPolicy.from_dict(manifest_label_policy)
