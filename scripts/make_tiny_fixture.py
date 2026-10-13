"""Regenerate the simulated ``data/fixtures/tiny/interactions.csv`` fixture.

The fixture is fully synthetic and hand-designed: two taste clusters
(action/scifi vs drama/romance), one sparse user (u6), and one cold user (u7).
"""

from __future__ import annotations

from pathlib import Path

RATINGS = {
    "u1": [
        ("i01", 5, 1),
        ("i02", 4, 2),
        ("i03", 5, 4),
        ("i04", 3, 6),
        ("i05", 4, 9),
        ("i06", 2, 11),
    ],
    "u2": [("i01", 4, 1), ("i02", 5, 3), ("i04", 4, 5), ("i03", 4, 8), ("i05", 5, 10)],
    "u3": [("i02", 4, 2), ("i03", 5, 3), ("i01", 4, 5), ("i05", 4, 7), ("i04", 5, 12)],
    "u4": [
        ("i06", 5, 1),
        ("i07", 4, 3),
        ("i08", 5, 4),
        ("i09", 4, 7),
        ("i10", 5, 10),
        ("i01", 2, 12),
    ],
    "u5": [("i07", 5, 2), ("i06", 4, 4), ("i09", 5, 6), ("i08", 4, 9), ("i10", 4, 11)],
    "u6": [("i01", 4, 13)],
}
HOURS = {"u1": 9, "u2": 10, "u3": 11, "u4": 12, "u5": 13, "u6": 14}
IMPLICIT = [
    ("u1", "i07", "click", "", 7, 15),
    ("u2", "i06", "watch", "0.3", 6, 16),
    ("u4", "i08", "purchase", "1", 5, 17),
    ("u5", "i02", "click", "", 8, 18),
    ("u3", "i03", "watch", "0.95", 9, 19),
]


def build_rows() -> list[tuple[str, str, str, str, str]]:
    rows = []
    for user, events in RATINGS.items():
        for item, rating, day in events:
            rows.append(
                (user, item, "rating", str(rating), f"2024-01-{day:02d}T{HOURS[user]:02d}:00:00Z")
            )
    for user, item, etype, value, day, hour in IMPLICIT:
        rows.append((user, item, etype, value, f"2024-01-{day:02d}T{hour:02d}:00:00Z"))
    rows.sort(key=lambda r: (r[4], r[0]))
    return rows


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "tiny" / "interactions.csv"
    lines = ["event_id,user_id,item_id,event_type,value,event_time,session_id"]
    for n, (user, item, etype, value, ts) in enumerate(build_rows(), 1):
        lines.append(f"e{n:03d},{user},{item},{etype},{value},{ts},")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(lines) - 1} rows to {out}")


if __name__ == "__main__":
    main()
