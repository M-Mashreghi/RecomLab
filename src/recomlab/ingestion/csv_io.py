"""CSV readers that turn raw rows into validated canonical records."""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterator
from pathlib import Path

from recomlab.domain.schemas import Interaction, Item, User


def read_rows(path: Path) -> Iterator[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        yield from csv.DictReader(fh)


def _load[T](path: Path, factory: Callable[[dict[str, str]], T]) -> list[T]:
    return [factory(row) for row in read_rows(path)]


def load_users(path: Path) -> list[User]:
    return _load(path, User.from_dict)


def load_items(path: Path) -> list[Item]:
    return _load(path, Item.from_dict)


def load_interactions(path: Path) -> list[Interaction]:
    return _load(path, Interaction.from_dict)
