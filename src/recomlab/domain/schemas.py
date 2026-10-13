"""Canonical user, item and interaction records (data contract v1).

These records are the only shapes model code may depend on. Validation is
strict and raises :class:`SchemaValidationError` with a stable ``code`` so
invalid inputs fail predictably.

Policies (see ``docs/DATA_CONTRACT.md``):

* Timestamps must be timezone-aware and are normalized to UTC.
* Explicit feedback (``rating``) requires a numeric value; implicit feedback
  events keep their own semantics and are never converted to ratings here.
* ``event_id`` is the duplicate key: identical re-deliveries are duplicates,
  conflicting records with the same ``event_id`` are contract violations.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

SCHEMA_VERSION = "1.0"

ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:\-]{1,64}$")

# Attribute keys that look like direct identifiers are rejected (PII minimization).
FORBIDDEN_ATTRIBUTE_KEYS = frozenset(
    {"name", "full_name", "email", "phone", "address", "ip", "ip_address", "device_id", "ssn"}
)


class SchemaValidationError(ValueError):
    """Raised when a record violates the data contract."""

    def __init__(self, code: str, field_name: str, message: str) -> None:
        super().__init__(f"[{code}] {field_name}: {message}")
        self.code = code
        self.field_name = field_name


class FeedbackKind(StrEnum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"


class EventType(StrEnum):
    RATING = "rating"
    CLICK = "click"
    VIEW = "view"
    WATCH = "watch"
    LIKE = "like"
    ADD_TO_CART = "add_to_cart"
    PURCHASE = "purchase"

    @property
    def kind(self) -> FeedbackKind:
        return FeedbackKind.EXPLICIT if self is EventType.RATING else FeedbackKind.IMPLICIT


# Events whose value carries meaning, and the allowed range.
_VALUE_RANGES: dict[EventType, tuple[float, float]] = {
    EventType.RATING: (0.0, 10.0),
    EventType.WATCH: (0.0, 1.0),  # fraction watched
    EventType.PURCHASE: (0.0, math.inf),  # quantity, non-negative
}


def validate_id(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SchemaValidationError("missing_id", field_name, "required non-empty string")
    if not ID_PATTERN.match(value):
        raise SchemaValidationError("invalid_id", field_name, f"invalid identifier {value!r}")
    return value


def parse_timestamp(value: Any, field_name: str = "event_time") -> datetime:
    """Parse an ISO-8601 string or datetime; require a timezone; return UTC."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SchemaValidationError("invalid_timestamp", field_name, str(exc)) from exc
    else:
        raise SchemaValidationError("missing_timestamp", field_name, "timestamp required")
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise SchemaValidationError("naive_timestamp", field_name, "timezone required")
    return dt.astimezone(UTC)


def _validate_attributes(attrs: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in attrs.items():
        if key.lower() in FORBIDDEN_ATTRIBUTE_KEYS:
            raise SchemaValidationError("pii_attribute", field_name, f"forbidden key {key!r}")
        out[str(key)] = val
    return out


def parse_optional_float(value: Any, field_name: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError("invalid_value", field_name, f"not numeric: {value!r}") from exc
    if not math.isfinite(out):
        raise SchemaValidationError("invalid_value", field_name, "must be finite")
    return out


@dataclass(frozen=True, slots=True)
class User:
    user_id: str
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_id(self.user_id, "user_id")
        object.__setattr__(self, "attributes", _validate_attributes(self.attributes, "attributes"))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> User:
        attrs = {k: v for k, v in raw.items() if k != "user_id" and v not in (None, "")}
        return cls(user_id=raw.get("user_id"), attributes=attrs)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class Item:
    item_id: str
    title: str | None = None
    categories: tuple[str, ...] = ()
    available: bool = True
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_id(self.item_id, "item_id")
        cats = tuple(c.strip() for c in self.categories if c and c.strip())
        object.__setattr__(self, "categories", cats)
        object.__setattr__(self, "attributes", _validate_attributes(self.attributes, "attributes"))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Item:
        cats_raw = raw.get("categories") or ()
        cats = tuple(cats_raw.split("|")) if isinstance(cats_raw, str) else tuple(cats_raw)
        avail = raw.get("available", True)
        if isinstance(avail, str):
            if avail.lower() not in {"true", "false", "1", "0", ""}:
                raise SchemaValidationError("invalid_value", "available", f"bad bool {avail!r}")
            avail = avail.lower() in {"true", "1", ""}
        known = {"item_id", "title", "categories", "available"}
        attrs = {k: v for k, v in raw.items() if k not in known and v not in (None, "")}
        return cls(
            item_id=raw.get("item_id"),  # type: ignore[arg-type]
            title=raw.get("title") or None,
            categories=cats,
            available=bool(avail),
            attributes=attrs,
        )


@dataclass(frozen=True, slots=True)
class Interaction:
    event_id: str
    user_id: str
    item_id: str
    event_type: EventType
    event_time: datetime
    value: float | None = None
    session_id: str | None = None

    def __post_init__(self) -> None:
        validate_id(self.event_id, "event_id")
        validate_id(self.user_id, "user_id")
        validate_id(self.item_id, "item_id")
        try:
            etype = EventType(self.event_type)
        except ValueError as exc:
            raise SchemaValidationError(
                "invalid_event_type", "event_type", f"unknown {self.event_type!r}"
            ) from exc
        object.__setattr__(self, "event_type", etype)
        object.__setattr__(self, "event_time", parse_timestamp(self.event_time))
        value = parse_optional_float(self.value, "value")
        if etype.kind is FeedbackKind.EXPLICIT and value is None:
            raise SchemaValidationError("missing_value", "value", "explicit feedback needs value")
        if value is not None and etype in _VALUE_RANGES:
            lo, hi = _VALUE_RANGES[etype]
            if not lo <= value <= hi:
                raise SchemaValidationError(
                    "value_out_of_range", "value", f"{value} not in [{lo}, {hi}] for {etype}"
                )
        object.__setattr__(self, "value", value)
        if self.session_id == "":
            object.__setattr__(self, "session_id", None)
        elif self.session_id is not None:
            validate_id(self.session_id, "session_id")

    @property
    def kind(self) -> FeedbackKind:
        return EventType(self.event_type).kind

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Interaction:
        return cls(
            event_id=raw.get("event_id"),  # type: ignore[arg-type]
            user_id=raw.get("user_id"),  # type: ignore[arg-type]
            item_id=raw.get("item_id"),  # type: ignore[arg-type]
            event_type=raw.get("event_type"),  # type: ignore[arg-type]
            event_time=raw.get("event_time"),  # type: ignore[arg-type]
            value=raw.get("value"),
            session_id=raw.get("session_id") or None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "user_id": self.user_id,
            "item_id": self.item_id,
            "event_type": str(self.event_type),
            "value": self.value,
            "event_time": self.event_time.isoformat().replace("+00:00", "Z"),
            "session_id": self.session_id,
        }


def sort_key(event: Interaction) -> tuple[datetime, str]:
    """Canonical deterministic event order: time, then event id."""
    return (event.event_time, event.event_id)


def find_duplicate_events(
    events: Iterable[Interaction],
) -> tuple[list[Interaction], list[Interaction]]:
    """Return ``(exact_duplicates, conflicting_duplicates)`` keyed by ``event_id``.

    The first occurrence is canonical; later identical records are exact
    duplicates; later records with the same id but different content conflict.
    """
    seen: dict[str, Interaction] = {}
    exact: list[Interaction] = []
    conflict: list[Interaction] = []
    for ev in events:
        prior = seen.get(ev.event_id)
        if prior is None:
            seen[ev.event_id] = ev
        elif prior == ev:
            exact.append(ev)
        else:
            conflict.append(ev)
    return exact, conflict
