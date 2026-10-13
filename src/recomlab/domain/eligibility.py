"""Item eligibility policy shared by recommenders, retrieval fusion and serving."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from recomlab.domain.schemas import Interaction, Item


@dataclass(frozen=True)
class EligibilityPolicy:
    require_available: bool = True
    exclude_consumed: bool = True
    excluded_items: frozenset[str] = field(default_factory=frozenset)

    def to_dict(self) -> dict[str, object]:
        return {
            "require_available": self.require_available,
            "exclude_consumed": self.exclude_consumed,
            "excluded_items": sorted(self.excluded_items),
        }


class EligibilityFilter:
    """Precomputes catalog-level eligibility; adds per-user consumed-item exclusion."""

    def __init__(
        self,
        items: Iterable[Item],
        history: Iterable[Interaction],
        policy: EligibilityPolicy | None = None,
    ) -> None:
        self.policy = policy or EligibilityPolicy()
        self.catalog: tuple[str, ...] = tuple(
            sorted(
                i.item_id
                for i in items
                if (i.available or not self.policy.require_available)
                and i.item_id not in self.policy.excluded_items
            )
        )
        self._catalog_set = frozenset(self.catalog)
        consumed: dict[str, set[str]] = {}
        for e in history:
            consumed.setdefault(e.user_id, set()).add(e.item_id)
        self.consumed: Mapping[str, frozenset[str]] = {u: frozenset(s) for u, s in consumed.items()}

    def is_eligible(self, user_id: str, item_id: str) -> bool:
        if item_id not in self._catalog_set:
            return False
        return not (self.policy.exclude_consumed and item_id in self.consumed.get(user_id, ()))

    def eligible_items(self, user_id: str) -> tuple[str, ...]:
        if not self.policy.exclude_consumed:
            return self.catalog
        seen = self.consumed.get(user_id, frozenset())
        return tuple(i for i in self.catalog if i not in seen)
