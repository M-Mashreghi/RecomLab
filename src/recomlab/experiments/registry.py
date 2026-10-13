"""SQLite-backed experiment registry."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from recomlab.experiments.manifest import RunManifest, canonical_json, validate_manifest


class RunNotFound(KeyError):
    pass


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    manifest_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id, created_at);
"""


class ExperimentRegistry:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._memory_conn = sqlite3.connect(":memory:") if self.db_path == ":memory:" else None
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        if self._memory_conn is not None:
            yield self._memory_conn
            self._memory_conn.commit()
            return
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def log_run(self, manifest: RunManifest) -> str:
        data = manifest.to_dict()
        validate_manifest(data)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
                (
                    manifest.run_id,
                    manifest.experiment_id,
                    manifest.config_hash,
                    manifest.created_at,
                    manifest.status,
                    canonical_json(data),
                ),
            )
        return manifest.run_id

    def get_run(self, run_id: str) -> RunManifest:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT manifest_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise RunNotFound(run_id)
        return RunManifest.from_dict(json.loads(row[0]))

    def list_runs(self, experiment_id: str | None = None) -> list[RunManifest]:
        sql = "SELECT manifest_json FROM runs"
        args: tuple[Any, ...] = ()
        if experiment_id is not None:
            sql += " WHERE experiment_id = ?"
            args = (experiment_id,)
        sql += " ORDER BY created_at, run_id"
        with self._conn() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [RunManifest.from_dict(json.loads(r[0])) for r in rows]

    def find_by_config(self, config_hash: str) -> list[RunManifest]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT manifest_json FROM runs WHERE config_hash = ? ORDER BY created_at",
                (config_hash,),
            ).fetchall()
        return [RunManifest.from_dict(json.loads(r[0])) for r in rows]

    def compare_runs(
        self, run_ids: Sequence[str], metrics: Sequence[str] | None = None
    ) -> list[dict[str, Any]]:
        """Return one row per run with model identity and selected metric values."""
        rows = []
        for rid in run_ids:
            m = self.get_run(rid)
            names = metrics if metrics is not None else sorted(m.metrics)
            rows.append(
                {
                    "run_id": rid,
                    "model": f"{m.model['name']}@{m.model['version']}",
                    "dataset": f"{m.dataset['dataset_id']}@{m.dataset['version']}",
                    "split_id": m.split["split_id"],
                    "evaluation_mode": m.evaluation.get("mode"),
                    **{name: m.metrics.get(name) for name in names},
                }
            )
        return rows
