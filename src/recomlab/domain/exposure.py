"""Impression / exposure and feedback event contract (exposure schema v1).

Impressions record what a user was *shown*; feedback events record what the
user did. Propensities are optional and are never guessed: a missing value
stays ``None`` and makes exposure-aware evaluation unavailable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from recomlab.domain.schemas import (
    SchemaValidationError,
    parse_optional_float,
    parse_timestamp,
    validate_id,
)

EXPOSURE_SCHEMA_VERSION = "1.0"

# Fields that would carry raw free text or direct identifiers into analytics tables.
FORBIDDEN_FEEDBACK_FIELDS = frozenset(
    {"comment", "text", "free_text", "message", "review", "email", "name", "ip_address"}
)


class FeedbackType(StrEnum):
    CLICK = "click"
    LIKE = "like"
    DISLIKE = "dislike"
    WATCH = "watch"
    PURCHASE = "purchase"
    DISMISS = "dismiss"
    RATING = "rating"


@dataclass(frozen=True, slots=True)
class Impression:
    impression_id: str
    request_id: str
    user_id: str
    item_id: str
    position: int
    model_id: str
    shown_at: datetime
    propensity: float | None = None
    session_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("impression_id", "request_id", "user_id", "item_id", "model_id"):
            validate_id(getattr(self, name), name)
        try:
            position = int(self.position)
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError("invalid_value", "position", "integer required") from exc
        if position < 1:
            raise SchemaValidationError("value_out_of_range", "position", "1-based rank required")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "shown_at", parse_timestamp(self.shown_at, "shown_at"))
        prop = parse_optional_float(self.propensity, "propensity")
        if prop is not None and not 0.0 < prop <= 1.0:
            raise SchemaValidationError("value_out_of_range", "propensity", "must be in (0, 1]")
        object.__setattr__(self, "propensity", prop)
        if self.session_id == "":
            object.__setattr__(self, "session_id", None)
        elif self.session_id is not None:
            validate_id(self.session_id, "session_id")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Impression:
        return cls(
            impression_id=raw.get("impression_id"),  # type: ignore[arg-type]
            request_id=raw.get("request_id"),  # type: ignore[arg-type]
            user_id=raw.get("user_id"),  # type: ignore[arg-type]
            item_id=raw.get("item_id"),  # type: ignore[arg-type]
            position=raw.get("position"),  # type: ignore[arg-type]
            model_id=raw.get("model_id"),  # type: ignore[arg-type]
            shown_at=raw.get("shown_at"),  # type: ignore[arg-type]
            propensity=raw.get("propensity"),
            session_id=raw.get("session_id") or None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "impression_id": self.impression_id,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "item_id": self.item_id,
            "position": self.position,
            "model_id": self.model_id,
            "shown_at": self.shown_at.isoformat().replace("+00:00", "Z"),
            "propensity": self.propensity,
            "session_id": self.session_id,
        }


@dataclass(frozen=True, slots=True)
class FeedbackEvent:
    feedback_id: str
    user_id: str
    item_id: str
    feedback_type: FeedbackType
    event_time: datetime
    value: float | None = None
    impression_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("feedback_id", "user_id", "item_id"):
            validate_id(getattr(self, name), name)
        try:
            ftype = FeedbackType(self.feedback_type)
        except ValueError as exc:
            raise SchemaValidationError(
                "invalid_event_type", "feedback_type", f"unknown {self.feedback_type!r}"
            ) from exc
        object.__setattr__(self, "feedback_type", ftype)
        object.__setattr__(self, "event_time", parse_timestamp(self.event_time))
        object.__setattr__(self, "value", parse_optional_float(self.value, "value"))
        if self.impression_id == "":
            object.__setattr__(self, "impression_id", None)
        elif self.impression_id is not None:
            validate_id(self.impression_id, "impression_id")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FeedbackEvent:
        bad = sorted(k for k in raw if k.lower() in FORBIDDEN_FEEDBACK_FIELDS)
        if bad:
            raise SchemaValidationError(
                "free_text_not_allowed", bad[0], "free text / PII is not stored in analytics"
            )
        return cls(
            feedback_id=raw.get("feedback_id"),  # type: ignore[arg-type]
            user_id=raw.get("user_id"),  # type: ignore[arg-type]
            item_id=raw.get("item_id"),  # type: ignore[arg-type]
            feedback_type=raw.get("feedback_type"),  # type: ignore[arg-type]
            event_time=raw.get("event_time"),  # type: ignore[arg-type]
            value=raw.get("value"),
            impression_id=raw.get("impression_id") or None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "user_id": self.user_id,
            "item_id": self.item_id,
            "feedback_type": str(self.feedback_type),
            "event_time": self.event_time.isoformat().replace("+00:00", "Z"),
            "value": self.value,
            "impression_id": self.impression_id,
        }


class LinkStatus(StrEnum):
    LINKED = "linked"  # explicit impression_id that matches user/item/time
    INFERRED = "inferred"  # no impression_id; matched latest prior impression in window
    UNLINKED = "unlinked"  # no matching exposure
    INVALID = "invalid"  # impression_id given but inconsistent / unknown


@dataclass(frozen=True, slots=True)
class FeedbackLink:
    feedback: FeedbackEvent
    status: LinkStatus
    impression: Impression | None
    reason: str | None = None


def link_feedback(
    impressions: Iterable[Impression],
    feedback: Iterable[FeedbackEvent],
    *,
    infer_window: timedelta | None = None,
) -> list[FeedbackLink]:
    """Link feedback events back to the exposure that preceded them.

    Explicit ``impression_id`` references are verified (same user and item,
    feedback not before the impression). Inference for feedback without an id
    is off by default and only enabled with an explicit ``infer_window``.
    """
    imps = list(impressions)
    by_id = {imp.impression_id: imp for imp in imps}
    links: list[FeedbackLink] = []
    for fb in feedback:
        if fb.impression_id is not None:
            imp = by_id.get(fb.impression_id)
            if imp is None:
                links.append(FeedbackLink(fb, LinkStatus.INVALID, None, "unknown_impression"))
            elif (imp.user_id, imp.item_id) != (fb.user_id, fb.item_id):
                links.append(FeedbackLink(fb, LinkStatus.INVALID, imp, "user_item_mismatch"))
            elif fb.event_time < imp.shown_at:
                links.append(FeedbackLink(fb, LinkStatus.INVALID, imp, "feedback_before_exposure"))
            else:
                links.append(FeedbackLink(fb, LinkStatus.LINKED, imp))
            continue
        if infer_window is not None:
            prior = [
                imp
                for imp in imps
                if imp.user_id == fb.user_id
                and imp.item_id == fb.item_id
                and imp.shown_at <= fb.event_time <= imp.shown_at + infer_window
            ]
            if prior:
                best = max(prior, key=lambda i: (i.shown_at, i.impression_id))
                links.append(FeedbackLink(fb, LinkStatus.INFERRED, best))
                continue
        links.append(FeedbackLink(fb, LinkStatus.UNLINKED, None))
    return links


class ExposureAvailability(StrEnum):
    UNAVAILABLE = "unavailable"  # no impression log at all
    IMPRESSIONS_ONLY = "impressions_only"  # exposure known, propensities missing
    PROPENSITY_AVAILABLE = "propensity_available"  # every impression has a propensity


@dataclass(frozen=True, slots=True)
class ExposureStatus:
    availability: ExposureAvailability
    n_impressions: int
    n_missing_propensity: int

    @property
    def exposure_aware_evaluation_allowed(self) -> bool:
        return self.availability is ExposureAvailability.PROPENSITY_AVAILABLE


def assess_exposure_log(impressions: Iterable[Impression] | None) -> ExposureStatus:
    """Classify an impression log; missing propensities are counted, never filled in."""
    imps = list(impressions or [])
    missing = sum(1 for i in imps if i.propensity is None)
    if not imps:
        avail = ExposureAvailability.UNAVAILABLE
    elif missing:
        avail = ExposureAvailability.IMPRESSIONS_ONLY
    else:
        avail = ExposureAvailability.PROPENSITY_AVAILABLE
    return ExposureStatus(avail, len(imps), missing)
