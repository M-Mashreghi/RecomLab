"""Leakage-safe train / validation / test split engine.

Primary protocols preserve time order:

* ``leave_last_n``: per user, the last ``n_test`` events are test and the
  preceding ``n_validation`` events are validation.
* ``global_time``: events before ``validation_cutoff`` are train, events in
  ``[validation_cutoff, test_cutoff)`` validation, and the rest test.

``random`` is supported only as an explicitly labeled diagnostic baseline.

Held-out events whose (user, item) pair already occurs earlier (in train, or in
validation for test) are dropped from the held-out set with reason
``repeat_interaction``; they are *not* moved into train, because that would put
future events into training data.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from recomlab.domain.schemas import Interaction, parse_timestamp, sort_key
from recomlab.experiments.seed import make_rng

PROTOCOLS = ("leave_last_n", "global_time", "random")
DIAGNOSTIC_PROTOCOLS = ("random",)


class SplitError(ValueError):
    pass


class LeakageError(AssertionError):
    pass


@dataclass(frozen=True)
class SplitConfig:
    protocol: str = "leave_last_n"
    n_test: int = 1
    n_validation: int = 1
    min_train_events: int = 1
    validation_cutoff: str | None = None
    test_cutoff: str | None = None
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 0
    drop_repeat_interactions: bool = True

    def __post_init__(self) -> None:
        if self.protocol not in PROTOCOLS:
            raise SplitError(f"protocol must be one of {PROTOCOLS}")
        if self.protocol == "global_time":
            if not (self.validation_cutoff and self.test_cutoff):
                raise SplitError("global_time requires validation_cutoff and test_cutoff")
            if parse_timestamp(self.validation_cutoff) > parse_timestamp(self.test_cutoff):
                raise SplitError("validation_cutoff must not be after test_cutoff")
        if self.n_test < 1 or self.n_validation < 0 or self.min_train_events < 1:
            raise SplitError("invalid n_test / n_validation / min_train_events")

    @property
    def is_diagnostic(self) -> bool:
        return self.protocol in DIAGNOSTIC_PROTOCOLS


@dataclass(frozen=True)
class Split:
    split_id: str
    config: SplitConfig
    train: tuple[Interaction, ...]
    validation: tuple[Interaction, ...]
    test: tuple[Interaction, ...]
    boundaries: dict[str, Any] = field(default_factory=dict)
    excluded_users: dict[str, str] = field(default_factory=dict)
    dropped_events: dict[str, str] = field(default_factory=dict)
    cold_test_items: tuple[str, ...] = ()

    @property
    def is_diagnostic(self) -> bool:
        return self.config.is_diagnostic

    @property
    def label(self) -> str:
        return f"{self.config.protocol}{' (DIAGNOSTIC)' if self.is_diagnostic else ''}"

    def train_until(self, partition: str) -> tuple[Interaction, ...]:
        """History visible when predicting ``partition`` (test sees train + validation)."""
        if partition == "validation":
            return self.train
        if partition == "test":
            return tuple(sorted((*self.train, *self.validation), key=sort_key))
        raise SplitError(f"unknown partition {partition!r}")

    def manifest(self) -> dict[str, Any]:
        return {
            "split_id": self.split_id,
            "protocol": self.config.protocol,
            "diagnostic": self.is_diagnostic,
            "config": asdict(self.config),
            "counts": {
                "train": len(self.train),
                "validation": len(self.validation),
                "test": len(self.test),
            },
            "boundaries": self.boundaries,
            "excluded_users": dict(sorted(self.excluded_users.items())),
            "dropped_events": dict(sorted(self.dropped_events.items())),
            "cold_test_items": list(self.cold_test_items),
        }

    def save(self, path: Path) -> None:
        data = self.manifest()
        data["event_ids"] = {
            "train": [e.event_id for e in self.train],
            "validation": [e.event_id for e in self.validation],
            "test": [e.event_id for e in self.test],
        }
        with Path(path).open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")


def _fingerprint(events: Iterable[Interaction], config: SplitConfig) -> str:
    h = hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode("utf-8"))
    for e in sorted(events, key=sort_key):
        h.update(f"{e.event_id}|{e.user_id}|{e.item_id}|{e.event_time.isoformat()}".encode())
    return f"split-{config.protocol}-{h.hexdigest()[:12]}"


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def make_split(events: Iterable[Interaction], config: SplitConfig) -> Split:
    ordered = sorted(events, key=sort_key)
    ids = [e.event_id for e in ordered]
    if len(ids) != len(set(ids)):
        raise SplitError("duplicate event_id in input; clean the dataset first")
    train: list[Interaction] = []
    val: list[Interaction] = []
    test: list[Interaction] = []
    excluded: dict[str, str] = {}
    boundaries: dict[str, Any] = {}

    if config.protocol == "leave_last_n":
        by_user: dict[str, list[Interaction]] = {}
        for e in ordered:
            by_user.setdefault(e.user_id, []).append(e)
        needed = config.min_train_events + config.n_validation + config.n_test
        per_user: dict[str, dict[str, str | None]] = {}
        for user, evs in sorted(by_user.items()):
            if len(evs) < needed:
                excluded[user] = "insufficient_history"
                train.extend(evs)
                continue
            cut_t = len(evs) - config.n_test
            cut_v = cut_t - config.n_validation
            train.extend(evs[:cut_v])
            val.extend(evs[cut_v:cut_t])
            test.extend(evs[cut_t:])
            per_user[user] = {
                "validation_start": _iso(evs[cut_v].event_time) if config.n_validation else None,
                "test_start": _iso(evs[cut_t].event_time),
            }
        boundaries["per_user"] = per_user
    elif config.protocol == "global_time":
        v_cut = parse_timestamp(config.validation_cutoff)
        t_cut = parse_timestamp(config.test_cutoff)
        for e in ordered:
            (train if e.event_time < v_cut else val if e.event_time < t_cut else test).append(e)
        boundaries = {"validation_cutoff": _iso(v_cut), "test_cutoff": _iso(t_cut)}
        train_users = {e.user_id for e in train}
        for e in (*val, *test):
            if e.user_id not in train_users:
                excluded.setdefault(e.user_id, "no_training_history")
    else:  # random diagnostic
        rng = make_rng(config.seed, "random_split")
        draws = rng.random(len(ordered))
        for e, r in zip(ordered, draws, strict=True):
            if r < config.test_fraction:
                test.append(e)
            elif r < config.test_fraction + config.validation_fraction:
                val.append(e)
            else:
                train.append(e)
        boundaries = {"diagnostic": "random split ignores time order"}

    dropped: dict[str, str] = {}
    if config.drop_repeat_interactions:
        seen = {(e.user_id, e.item_id) for e in train}
        kept_val = []
        for e in val:
            if (e.user_id, e.item_id) in seen:
                dropped[e.event_id] = "repeat_interaction"
            else:
                kept_val.append(e)
        seen |= {(e.user_id, e.item_id) for e in kept_val}
        seen_val = seen
        kept_test = []
        for e in test:
            if (e.user_id, e.item_id) in seen_val:
                dropped[e.event_id] = "repeat_interaction"
            else:
                kept_test.append(e)
        val, test = kept_val, kept_test

    train_items = {e.item_id for e in train}
    cold_items = tuple(sorted({e.item_id for e in test} - train_items))
    split = Split(
        split_id=_fingerprint(ordered, config),
        config=config,
        train=tuple(train),
        validation=tuple(sorted(val, key=sort_key)),
        test=tuple(sorted(test, key=sort_key)),
        boundaries=boundaries,
        excluded_users=excluded,
        dropped_events=dropped,
        cold_test_items=cold_items,
    )
    assert_no_leakage(split)
    return split


def assert_no_leakage(split: Split) -> None:
    """Check partition disjointness and (for temporal protocols) time order."""
    parts = {"train": split.train, "validation": split.validation, "test": split.test}
    ids = {name: {e.event_id for e in evs} for name, evs in parts.items()}
    if ids["test"] & (ids["train"] | ids["validation"]) or ids["train"] & ids["validation"]:
        raise LeakageError("event appears in more than one partition")
    if split.is_diagnostic:
        return
    if split.config.protocol == "global_time":
        last_train = max((e.event_time for e in split.train), default=None)
        first_val = min((e.event_time for e in split.validation), default=None)
        first_test = min((e.event_time for e in split.test), default=None)
        last_val = max((e.event_time for e in split.validation), default=None)
        if last_train and first_val and last_train >= first_val:
            raise LeakageError("train event at/after validation start")
        if last_train and first_test and last_train >= first_test:
            raise LeakageError("train event at/after test start")
        if last_val and first_test and last_val >= first_test:
            raise LeakageError("validation event at/after test start")
        return
    # leave_last_n: per-user order
    latest: dict[str, dict[str, Any]] = {}
    for name in ("train", "validation", "test"):
        for e in parts[name]:
            latest.setdefault(e.user_id, {}).setdefault(name, []).append(e)
    for user, p in latest.items():
        key = [sort_key(e) for e in p.get("train", [])]
        vkey = [sort_key(e) for e in p.get("validation", [])]
        tkey = [sort_key(e) for e in p.get("test", [])]
        if key and vkey and max(key) >= min(vkey):
            raise LeakageError(f"user {user}: train event after validation event")
        if (key or vkey) and tkey and max(key + vkey) >= min(tkey):
            raise LeakageError(f"user {user}: earlier partition event after test event")


def load_split_ids(path: Path) -> Mapping[str, list[str]]:
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["event_ids"]
