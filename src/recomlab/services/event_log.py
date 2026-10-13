"""Impression & feedback logging (SQLite), idempotent by event id.

Stores only pseudonymous ids, positions, model ids, timestamps and optional
propensities (never fabricated). Records conform to the exposure contract
(``recomlab.domain.exposure``). Re-sending an identical event is a no-op;
re-sending an id with different content is rejected.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from recomlab.domain.exposure import (
    FeedbackEvent,
    FeedbackLink,
    Impression,
    link_feedback,
)
from recomlab.domain.schemas import Interaction
from recomlab.services.recommendation import RecommendationResponse

_SCHEMA = """
CREATE TABLE IF NOT EXISTS impressions (
    impression_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    model_id TEXT NOT NULL,
    shown_at TEXT NOT NULL,
    propensity REAL,
    session_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_imp_request ON impressions(request_id);
CREATE INDEX IF NOT EXISTS idx_imp_user_item ON impressions(user_id, item_id);
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    feedback_type TEXT NOT NULL,
    event_time TEXT NOT NULL,
    value REAL,
    impression_id TEXT
);
"""

_FEEDBACK_TO_EVENT = {
    "click": "click",
    "like": "like",
    "watch": "watch",
    "purchase": "purchase",
    "rating": "rating",
}


class ConflictingEventError(ValueError):
    pass


@dataclass(frozen=True)
class LogResult:
    inserted: int
    duplicates: int


class EventLog:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._memory = sqlite3.connect(":memory:") if self.db_path == ":memory:" else None
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        if self._memory is not None:
            yield self._memory
            self._memory.commit()
            return
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- writes ----------------------------------------------------------------
    def log_impressions(self, impressions: Sequence[Impression]) -> LogResult:
        inserted = dup = 0
        with self._conn() as c:
            for imp in impressions:
                row = imp.to_dict()
                existing = c.execute(
                    "SELECT * FROM impressions WHERE impression_id = ?", (imp.impression_id,)
                ).fetchone()
                if existing is not None:
                    if Impression.from_dict(_imp_row(existing)) != imp:
                        raise ConflictingEventError(f"impression {imp.impression_id} conflicts")
                    dup += 1
                    continue
                c.execute(
                    "INSERT INTO impressions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    tuple(row[k] for k in _IMP_COLS),
                )
                inserted += 1
        return LogResult(inserted, dup)

    def log_response(
        self,
        response: RecommendationResponse,
        shown_at: datetime,
        propensities: Sequence[float] | None = None,
    ) -> LogResult:
        """Log a served slate; impression ids derive from the trace id and position."""
        if propensities is not None and len(propensities) != len(response.items):
            raise ValueError("one propensity per shown item required")
        imps = [
            Impression(
                impression_id=f"{response.trace_id}:{item.rank}",
                request_id=response.trace_id,
                user_id=response.user_id,
                item_id=item.item_id,
                position=item.rank,
                model_id=response.bundle_id,
                shown_at=shown_at,
                propensity=None if propensities is None else propensities[n],
            )
            for n, item in enumerate(response.items)
        ]
        return self.log_impressions(imps)

    def log_feedback(self, events: Sequence[FeedbackEvent]) -> LogResult:
        inserted = dup = 0
        with self._conn() as c:
            for fb in events:
                existing = c.execute(
                    "SELECT * FROM feedback WHERE feedback_id = ?", (fb.feedback_id,)
                ).fetchone()
                if existing is not None:
                    if FeedbackEvent.from_dict(dict(zip(_FB_COLS, existing, strict=True))) != fb:
                        raise ConflictingEventError(f"feedback {fb.feedback_id} conflicts")
                    dup += 1
                    continue
                row = fb.to_dict()
                c.execute(
                    "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?)",
                    tuple(row[k] for k in _FB_COLS),
                )
                inserted += 1
        return LogResult(inserted, dup)

    # -- reads -----------------------------------------------------------------
    def impressions(self, request_id: str | None = None) -> list[Impression]:
        sql = "SELECT * FROM impressions"
        args: tuple[str, ...] = ()
        if request_id is not None:
            sql += " WHERE request_id = ?"
            args = (request_id,)
        with self._conn() as c:
            rows = c.execute(sql + " ORDER BY shown_at, impression_id", args).fetchall()
        return [Impression.from_dict(_imp_row(r)) for r in rows]

    def feedback(self) -> list[FeedbackEvent]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM feedback ORDER BY event_time, feedback_id").fetchall()
        return [FeedbackEvent.from_dict(dict(zip(_FB_COLS, r, strict=True))) for r in rows]

    def linked_feedback(self) -> list[FeedbackLink]:
        return link_feedback(self.impressions(), self.feedback())

    def feedback_for_request(self, request_id: str) -> list[FeedbackLink]:
        ids = {i.impression_id for i in self.impressions(request_id)}
        return [
            link
            for link in self.linked_feedback()
            if link.impression is not None and link.impression.impression_id in ids
        ]

    def as_interactions(self) -> list[Interaction]:
        """Feedback converted to canonical interactions for snapshotting/retraining."""
        out = []
        for fb in self.feedback():
            etype = _FEEDBACK_TO_EVENT.get(str(fb.feedback_type))
            if etype is None:
                continue  # dismiss/dislike are not positive interaction events
            out.append(
                Interaction(
                    event_id=f"fb:{fb.feedback_id}",
                    user_id=fb.user_id,
                    item_id=fb.item_id,
                    event_type=etype,  # type: ignore[arg-type]
                    event_time=fb.event_time,
                    value=fb.value,
                )
            )
        return out


_IMP_COLS = (
    "impression_id",
    "request_id",
    "user_id",
    "item_id",
    "position",
    "model_id",
    "shown_at",
    "propensity",
    "session_id",
)
_FB_COLS = (
    "feedback_id",
    "user_id",
    "item_id",
    "feedback_type",
    "event_time",
    "value",
    "impression_id",
)


def _imp_row(row: Sequence[object]) -> dict[str, object]:
    return dict(zip(_IMP_COLS, row, strict=True))
