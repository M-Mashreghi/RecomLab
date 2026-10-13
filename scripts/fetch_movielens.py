"""Download MovieLens ml-latest-small into data/raw/ (never committed).

Usage: python scripts/fetch_movielens.py [--dest data/raw/movielens-latest-small]
Read and accept the GroupLens usage license before running.
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="data/raw/movielens-latest-small")
    args = parser.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as resp:  # noqa: S310 - fixed https URL
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for name in ("ratings.csv", "movies.csv"):
            (dest / name).write_bytes(zf.read(f"ml-latest-small/{name}"))
    (dest / "RETRIEVED_AT").write_text(datetime.now(UTC).date().isoformat(), encoding="utf-8")
    print(f"MovieLens written to {dest}")


if __name__ == "__main__":
    main()
