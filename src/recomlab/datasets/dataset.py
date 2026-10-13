"""In-memory canonical dataset container."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from recomlab.domain.schemas import Interaction, Item, User, sort_key


@dataclass(frozen=True)
class CanonicalDataset:
    """Validated users, items and interactions in canonical event order."""

    users: tuple[User, ...]
    items: tuple[Item, ...]
    interactions: tuple[Interaction, ...]

    @classmethod
    def build(
        cls, users: Iterable[User], items: Iterable[Item], interactions: Iterable[Interaction]
    ) -> CanonicalDataset:
        return cls(
            users=tuple(sorted(users, key=lambda u: u.user_id)),
            items=tuple(sorted(items, key=lambda i: i.item_id)),
            interactions=tuple(sorted(interactions, key=sort_key)),
        )

    @property
    def user_ids(self) -> tuple[str, ...]:
        return tuple(u.user_id for u in self.users)

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(i.item_id for i in self.items)

    def item_map(self) -> dict[str, Item]:
        return {i.item_id: i for i in self.items}

    def stats(self) -> dict[str, int]:
        return {
            "n_users": len(self.users),
            "n_items": len(self.items),
            "n_interactions": len(self.interactions),
            "n_active_users": len({e.user_id for e in self.interactions}),
            "n_interacted_items": len({e.item_id for e in self.interactions}),
        }

    def write_csv(self, out_dir: Path) -> list[Path]:
        """Write canonical CSVs (users, items, interactions) deterministically."""
        import csv

        out_dir.mkdir(parents=True, exist_ok=True)
        attr_keys = sorted({k for u in self.users for k in u.attributes})
        paths = []
        p = out_dir / "users.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["user_id", *attr_keys])
            for u in self.users:
                w.writerow([u.user_id, *(u.attributes.get(k, "") for k in attr_keys)])
        paths.append(p)
        p = out_dir / "items.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["item_id", "title", "categories", "available"])
            for i in self.items:
                w.writerow(
                    [i.item_id, i.title or "", "|".join(i.categories), str(i.available).lower()]
                )
        paths.append(p)
        p = out_dir / "interactions.csv"
        cols = ["event_id", "user_id", "item_id", "event_type", "value", "event_time", "session_id"]
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(cols)
            for e in self.interactions:
                d = e.to_dict()
                w.writerow(["" if d[c] is None else _fmt(d[c]) for c in cols])
        paths.append(p)
        return paths


def _fmt(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
