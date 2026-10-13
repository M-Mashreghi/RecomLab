"""Dataset adapters: raw snapshot directory -> canonical dataset + manifest fields."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from recomlab.datasets.dataset import CanonicalDataset
from recomlab.datasets.registry import compute_checksums
from recomlab.domain.schemas import SCHEMA_VERSION, Interaction, Item, User
from recomlab.ingestion.csv_io import load_interactions, load_items, load_users


class DatasetAdapter(Protocol):
    dataset_id: str
    raw_files: tuple[str, ...]

    def load(self, raw_dir: Path) -> CanonicalDataset: ...

    def describe(self, raw_dir: Path) -> dict[str, Any]: ...


class FixtureAdapter:
    """Canonical-format CSV directory (used for the in-repo simulated fixture)."""

    raw_files = ("users.csv", "items.csv", "interactions.csv")

    def __init__(self, dataset_id: str = "tiny", version: str = "1") -> None:
        self.dataset_id = dataset_id
        self.version = version

    def load(self, raw_dir: Path) -> CanonicalDataset:
        return CanonicalDataset.build(
            load_users(raw_dir / "users.csv"),
            load_items(raw_dir / "items.csv"),
            load_interactions(raw_dir / "interactions.csv"),
        )

    def describe(self, raw_dir: Path) -> dict[str, Any]:
        _, raw_checksum = compute_checksums(raw_dir, list(self.raw_files))
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "source": f"in-repo simulated fixture (raw sha256 {raw_checksum[:12]})",
            "schema_version": SCHEMA_VERSION,
            "license_note": "Synthetic data created for RecomLab tests; no external terms.",
            "retrieval_date": "2024-01-31",
            "synthetic": True,
            "description": "Hand-designed simulated dataset for deterministic CI tests.",
        }


class MovieLensAdapter:
    """MovieLens ``ml-latest-small`` / ``ml-*`` CSV layout (ratings.csv + movies.csv).

    Raw files are downloaded by the user (see ``scripts/fetch_movielens.py``) and
    never committed. IDs are prefixed to stay pseudonymous and namespace-safe.
    """

    raw_files = ("ratings.csv", "movies.csv")
    LICENSE = (
        "GroupLens MovieLens usage license: research use permitted, no redistribution, "
        "cite F. Maxwell Harper and Joseph A. Konstan (2015)."
    )

    def __init__(self, dataset_id: str = "movielens-latest-small", version: str = "1") -> None:
        self.dataset_id = dataset_id
        self.version = version

    def load(self, raw_dir: Path) -> CanonicalDataset:
        items = []
        with (raw_dir / "movies.csv").open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                genres = row.get("genres", "")
                cats = () if genres in ("", "(no genres listed)") else tuple(genres.split("|"))
                items.append(
                    Item(item_id=f"ml_i{row['movieId']}", title=row.get("title"), categories=cats)
                )
        events = []
        users: set[str] = set()
        with (raw_dir / "ratings.csv").open(newline="", encoding="utf-8") as fh:
            for n, row in enumerate(csv.DictReader(fh), 1):
                uid = f"ml_u{row['userId']}"
                users.add(uid)
                events.append(
                    Interaction(
                        event_id=f"ml_e{n}",
                        user_id=uid,
                        item_id=f"ml_i{row['movieId']}",
                        event_type="rating",  # type: ignore[arg-type]
                        value=float(row["rating"]),
                        event_time=datetime.fromtimestamp(int(row["timestamp"]), tz=UTC),
                    )
                )
        return CanonicalDataset.build((User(u) for u in users), items, events)

    def describe(self, raw_dir: Path) -> dict[str, Any]:
        _, raw_checksum = compute_checksums(raw_dir, list(self.raw_files))
        stamp = raw_dir / "RETRIEVED_AT"
        retrieved = stamp.read_text("utf-8").strip() if stamp.exists() else "unknown"
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "source": f"https://grouplens.org/datasets/movielens/ (raw sha256 {raw_checksum[:12]})",
            "schema_version": SCHEMA_VERSION,
            "license_note": self.LICENSE,
            "retrieval_date": retrieved,
            "synthetic": False,
            "description": "MovieLens explicit ratings converted to canonical rating events.",
        }
