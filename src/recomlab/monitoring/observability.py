"""Structured logging, serving metrics, and drift / train-serving skew checks."""

from __future__ import annotations

import json
import logging
import math
import re
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from recomlab.domain.schemas import Interaction, Item

SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "token",
        "password",
        "secret",
        "email",
        "text",
        "history",
        "free_text",
        "authorization",
        "x-admin-token",
    }
)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
CONTEXT_KEYS = ("request_id", "trace_id", "run_id", "model_id", "bundle_id", "workflow_id")


def scrub(value: Any, key: str | None = None) -> Any:
    """Recursively redact sensitive keys and email-like strings."""
    if key is not None and key.lower() in SENSITIVE_KEYS:
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {k: scrub(v, str(k)) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [scrub(v) for v in value]
    if isinstance(value, str):
        return _EMAIL.sub("[EMAIL]", value)
    return value


_STD_ATTRS = set(vars(logging.LogRecord("x", 0, "x", 0, "x", None, None))) | {"message"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line with context ids; sensitive fields are scrubbed."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": scrub(record.getMessage()),
        }
        extras = {k: v for k, v in vars(record).items() if k not in _STD_ATTRS}
        for k in CONTEXT_KEYS:
            if k in extras:
                payload[k] = extras.pop(k)
        if extras:
            payload["context"] = scrub(extras)
        return json.dumps(payload, sort_keys=True, default=str)


@dataclass
class ServingMonitor:
    """In-process serving metrics (requests, empty candidates, fallbacks, latency)."""

    n_requests: int = 0
    n_empty_candidates: int = 0
    fallbacks: Counter[str] = field(default_factory=Counter)
    latencies_ms: list[float] = field(default_factory=list)
    items_served: Counter[str] = field(default_factory=Counter)
    max_latency_samples: int = 10_000

    def record(
        self,
        n_candidates: int,
        fallbacks: Iterable[str | None],
        item_ids: Sequence[str],
        latency_ms: float,
    ) -> None:
        self.n_requests += 1
        self.n_empty_candidates += n_candidates == 0
        for f in fallbacks:
            if f:
                self.fallbacks[f] += 1
        self.items_served.update(item_ids)
        self.latencies_ms.append(latency_ms)
        if len(self.latencies_ms) > self.max_latency_samples:
            self.latencies_ms = self.latencies_ms[-self.max_latency_samples :]

    def snapshot(self) -> dict[str, Any]:
        lat = sorted(self.latencies_ms)

        def q(p: float) -> float | None:
            return lat[min(len(lat) - 1, int(p * len(lat)))] if lat else None

        return {
            "n_requests": self.n_requests,
            "empty_candidate_rate": self.n_empty_candidates / self.n_requests
            if self.n_requests
            else None,
            "fallbacks": dict(self.fallbacks),
            "latency_ms": {"p50": q(0.5), "p95": q(0.95), "max": lat[-1] if lat else None},
            "distinct_items_served": len(self.items_served),
        }


@dataclass(frozen=True)
class DataProfile:
    n_events: int
    n_users: int
    catalog_size: int
    event_type_mix: dict[str, float]
    user_activity_buckets: dict[str, float]
    item_popularity_buckets: dict[str, float]
    feature_schema: tuple[str, ...] = ()
    feature_means: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ACTIVITY_EDGES = (1, 3, 10, 30)


def _bucketize(counts: Iterable[int], edges: Sequence[int]) -> dict[str, float]:
    values = list(counts)
    labels = [f"<{edges[0]}"] + [f"{a}-{b - 1}" for a, b in zip(edges, edges[1:], strict=False)]
    labels.append(f">={edges[-1]}")
    out = dict.fromkeys(labels, 0.0)
    for v in values:
        idx = sum(v >= e for e in edges)
        out[labels[idx]] += 1
    total = len(values) or 1
    return {k: v / total for k, v in out.items()}


def profile(
    events: Sequence[Interaction],
    items: Sequence[Item],
    feature_rows: Sequence[Mapping[str, float | None]] = (),
) -> DataProfile:
    types = Counter(str(e.event_type) for e in events)
    total = sum(types.values()) or 1
    users = Counter(e.user_id for e in events)
    pops = Counter(e.item_id for e in events)
    schema = tuple(sorted({k for row in feature_rows for k in row}))
    means = {
        k: statistics.fmean(v for row in feature_rows if (v := row.get(k)) is not None)
        for k in schema
        if any(row.get(k) is not None for row in feature_rows)
    }
    return DataProfile(
        n_events=len(events),
        n_users=len(users),
        catalog_size=len(items),
        event_type_mix={k: v / total for k, v in sorted(types.items())},
        user_activity_buckets=_bucketize(users.values(), ACTIVITY_EDGES),
        item_popularity_buckets=_bucketize([pops.get(i.item_id, 0) for i in items], ACTIVITY_EDGES),
        feature_schema=schema,
        feature_means=means,
    )


def psi(expected: Mapping[str, float], actual: Mapping[str, float], eps: float = 1e-4) -> float:
    """Population stability index over shared bucket labels."""
    keys = set(expected) | set(actual)
    total = 0.0
    for k in keys:
        e = max(expected.get(k, 0.0), eps)
        a = max(actual.get(k, 0.0), eps)
        total += (a - e) * math.log(a / e)
    return total


@dataclass(frozen=True)
class DriftThresholds:
    catalog_change_ratio: float = 0.2
    event_mix_tvd: float = 0.15
    activity_psi: float = 0.25
    popularity_psi: float = 0.25
    feature_mean_rel_change: float = 0.5
    volume_change_ratio: float = 0.5
    empty_candidate_rate: float = 0.2
    latency_p95_ms: float = 200.0


def check_drift(
    train: DataProfile,
    serving: DataProfile,
    thresholds: DriftThresholds | None = None,
    serving_metrics: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return warnings (empty list = no major drift/skew)."""
    t = thresholds or DriftThresholds()
    warnings: list[dict[str, Any]] = []

    def warn(check: str, value: Any, limit: Any) -> None:
        warnings.append({"check": check, "value": value, "threshold": limit})

    if train.catalog_size:
        ratio = abs(serving.catalog_size - train.catalog_size) / train.catalog_size
        if ratio > t.catalog_change_ratio:
            warn("catalog_size_change", ratio, t.catalog_change_ratio)
    if train.n_events:
        vol = abs(serving.n_events - train.n_events) / train.n_events
        if vol > t.volume_change_ratio:
            warn("event_volume_change", vol, t.volume_change_ratio)
    keys = set(train.event_type_mix) | set(serving.event_type_mix)
    tvd = 0.5 * sum(
        abs(train.event_type_mix.get(k, 0) - serving.event_type_mix.get(k, 0)) for k in keys
    )
    if tvd > t.event_mix_tvd:
        warn("event_type_mix", tvd, t.event_mix_tvd)
    a_psi = psi(train.user_activity_buckets, serving.user_activity_buckets)
    if a_psi > t.activity_psi:
        warn("user_activity_psi", a_psi, t.activity_psi)
    p_psi = psi(train.item_popularity_buckets, serving.item_popularity_buckets)
    if p_psi > t.popularity_psi:
        warn("item_popularity_psi", p_psi, t.popularity_psi)
    if train.feature_schema and serving.feature_schema != train.feature_schema:
        warn(
            "feature_schema_skew",
            {
                "missing": sorted(set(train.feature_schema) - set(serving.feature_schema)),
                "extra": sorted(set(serving.feature_schema) - set(train.feature_schema)),
            },
            "identical schema",
        )
    for k, m in train.feature_means.items():
        s = serving.feature_means.get(k)
        if s is not None and m != 0 and abs(s - m) / abs(m) > t.feature_mean_rel_change:
            warn(f"feature_mean_skew:{k}", abs(s - m) / abs(m), t.feature_mean_rel_change)
    if serving_metrics:
        rate = serving_metrics.get("empty_candidate_rate")
        if rate is not None and rate > t.empty_candidate_rate:
            warn("empty_candidate_rate", rate, t.empty_candidate_rate)
        p95 = (serving_metrics.get("latency_ms") or {}).get("p95")
        if p95 is not None and p95 > t.latency_p95_ms:
            warn("latency_p95_ms", p95, t.latency_p95_ms)
    return warnings
