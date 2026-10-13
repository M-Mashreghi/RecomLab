"""Deterministic interaction validation & cleaning with an auditable report.

Every removed record carries a reason code; the report reconciles input count
with output count plus removals per rule.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from recomlab.domain.schemas import (
    Interaction,
    Item,
    SchemaValidationError,
    User,
    parse_timestamp,
    sort_key,
)

CLEANING_POLICY_VERSION = "1.0"


class ReasonCode:
    SCHEMA_PREFIX = "schema:"
    DUPLICATE_EXACT = "duplicate_exact"
    DUPLICATE_CONFLICT = "duplicate_conflict"
    UNKNOWN_USER = "unknown_user"
    UNKNOWN_ITEM = "unknown_item"
    OUT_OF_TIME_WINDOW = "out_of_time_window"


@dataclass(frozen=True)
class CleaningPolicy:
    version: str = CLEANING_POLICY_VERSION
    require_known_users: bool = True
    require_known_items: bool = True
    min_time: str | None = None  # inclusive, ISO-8601 with timezone
    max_time: str | None = None  # exclusive

    @property
    def policy_id(self) -> str:
        payload = repr(sorted(asdict(self).items())).encode("utf-8")
        return f"clean-{self.version}-{hashlib.sha256(payload).hexdigest()[:8]}"


@dataclass(frozen=True)
class Rejection:
    row_index: int
    event_id: str | None
    reason: str


@dataclass(frozen=True)
class StepReport:
    rule: str
    n_before: int
    n_removed: int
    n_after: int


@dataclass
class CleaningReport:
    policy_id: str
    n_input: int
    n_output: int = 0
    steps: list[StepReport] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)

    @property
    def reason_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(r.reason for r in self.rejections).items()))

    def reconciles(self) -> bool:
        removed = sum(s.n_removed for s in self.steps)
        chained = all(
            a.n_after == b.n_before for a, b in zip(self.steps, self.steps[1:], strict=False)
        )
        return (
            self.n_input == self.n_output + removed
            and removed == len(self.rejections)
            and chained
            and (not self.steps or self.steps[-1].n_after == self.n_output)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "n_input": self.n_input,
            "n_output": self.n_output,
            "steps": [asdict(s) for s in self.steps],
            "reason_counts": self.reason_counts,
        }


@dataclass(frozen=True)
class CleaningResult:
    interactions: tuple[Interaction, ...]
    report: CleaningReport

    @property
    def checksum(self) -> str:
        h = hashlib.sha256()
        for e in self.interactions:
            h.update(repr(sorted(e.to_dict().items())).encode("utf-8"))
        return h.hexdigest()


def clean_interactions(
    raw_rows: Iterable[Mapping[str, Any]],
    users: Iterable[User],
    items: Iterable[Item],
    policy: CleaningPolicy | None = None,
) -> CleaningResult:
    policy = policy or CleaningPolicy()
    rows = list(raw_rows)
    report = CleaningReport(policy_id=policy.policy_id, n_input=len(rows))

    def step(rule: str, before: int, kept: list[tuple[int, Interaction]]) -> None:
        report.steps.append(StepReport(rule, before, before - len(kept), len(kept)))

    # 1. Schema validation.
    current: list[tuple[int, Interaction]] = []
    for idx, row in enumerate(rows):
        try:
            current.append((idx, Interaction.from_dict(row)))
        except SchemaValidationError as err:
            eid = row.get("event_id") or None
            report.rejections.append(Rejection(idx, eid, ReasonCode.SCHEMA_PREFIX + err.code))
    step("schema", len(rows), current)

    # 2. Duplicates by event_id: exact re-deliveries dropped, conflicting ids dropped entirely.
    groups: dict[str, list[tuple[int, Interaction]]] = {}
    for idx, ev in current:
        groups.setdefault(ev.event_id, []).append((idx, ev))
    kept: list[tuple[int, Interaction]] = []
    before = len(current)
    for idx, ev in current:
        group = groups[ev.event_id]
        if any(other != group[0][1] for _, other in group):
            report.rejections.append(Rejection(idx, ev.event_id, ReasonCode.DUPLICATE_CONFLICT))
        elif idx != group[0][0]:
            report.rejections.append(Rejection(idx, ev.event_id, ReasonCode.DUPLICATE_EXACT))
        else:
            kept.append((idx, ev))
    current = kept
    step("duplicates", before, current)

    # 3. Referential integrity.
    user_ids = {u.user_id for u in users}
    item_ids = {i.item_id for i in items}
    for rule, enabled, ids, attr, reason in (
        ("known_users", policy.require_known_users, user_ids, "user_id", ReasonCode.UNKNOWN_USER),
        ("known_items", policy.require_known_items, item_ids, "item_id", ReasonCode.UNKNOWN_ITEM),
    ):
        before = len(current)
        if enabled:
            kept = []
            for idx, ev in current:
                if getattr(ev, attr) in ids:
                    kept.append((idx, ev))
                else:
                    report.rejections.append(Rejection(idx, ev.event_id, reason))
            current = kept
        step(rule, before, current)

    # 4. Time window.
    lo: datetime | None = parse_timestamp(policy.min_time, "min_time") if policy.min_time else None
    hi: datetime | None = parse_timestamp(policy.max_time, "max_time") if policy.max_time else None
    before = len(current)
    kept = []
    for idx, ev in current:
        if (lo is not None and ev.event_time < lo) or (hi is not None and ev.event_time >= hi):
            report.rejections.append(Rejection(idx, ev.event_id, ReasonCode.OUT_OF_TIME_WINDOW))
        else:
            kept.append((idx, ev))
    current = kept
    step("time_window", before, current)

    report.rejections.sort(key=lambda r: r.row_index)
    report.n_output = len(current)
    return CleaningResult(tuple(sorted((ev for _, ev in current), key=sort_key)), report)
