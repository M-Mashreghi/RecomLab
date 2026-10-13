import random
from collections import Counter

import numpy as np
import pytest

from recomlab.datasets import FixtureAdapter
from recomlab.experiments import (
    ExperimentRegistry,
    ManifestValidationError,
    RunManifest,
    RunNotFound,
    collect_software_state,
    derive_seed,
    make_rng,
    set_global_seed,
    validate_manifest,
)
from recomlab.experiments.manifest import git_commit


def _toy_baseline(tiny_dir, seed):
    """Deterministic toy result: popularity top-3 plus a seeded random pick."""
    ds = FixtureAdapter().load(tiny_dir)
    counts = Counter(e.item_id for e in ds.interactions)
    top = [i for i, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]]
    pick = make_rng(seed, "toy").choice(list(ds.item_ids))
    return {"top3": top, "random_pick": str(pick)}


def _manifest(tiny_dir, seed=42, created_at="2024-02-01T00:00:00+00:00"):
    return RunManifest(
        experiment_id="toy",
        dataset={"dataset_id": "tiny", "version": "1"},
        split={"split_id": "none"},
        model={"name": "toy", "version": "1", "params": {}},
        metric_config={"k": [3]},
        seeds={"global": seed},
        metrics=_toy_baseline(tiny_dir, seed),
        software=collect_software_state(),
        created_at=created_at,
    )


def test_seed_helpers_deterministic():
    assert derive_seed(1, "a") == derive_seed(1, "a") != derive_seed(1, "b")
    assert (
        make_rng(3, "x").integers(0, 1000, 5).tolist()
        == make_rng(3, "x").integers(0, 1000, 5).tolist()
    )
    set_global_seed(5)
    a = (random.random(), np.random.rand())
    set_global_seed(5)
    assert (random.random(), np.random.rand()) == a


def test_same_config_reproduces_result(tiny_dir):
    m1 = _manifest(tiny_dir)
    m2 = _manifest(tiny_dir, created_at="2024-03-01T00:00:00+00:00")
    assert m1.metrics == m2.metrics
    assert m1.config_hash == m2.config_hash
    assert m1.run_id != m2.run_id  # distinct executions of the same config
    assert _manifest(tiny_dir, seed=7).config_hash != m1.config_hash


def test_manifest_schema_validation(tiny_dir):
    data = _manifest(tiny_dir).to_dict()
    validate_manifest(data)
    assert RunManifest.from_dict(data) == _manifest(tiny_dir)
    for mutate in (
        lambda d: d.pop("dataset"),
        lambda d: d["model"].pop("version"),
        lambda d: d.update(seeds={"global": "42"}),
        lambda d: d.update(status="weird"),
        lambda d: d.update(unknown=1),
        lambda d: d.update(schema_version="0.1"),
        lambda d: d.update(config_hash="tampered"),
    ):
        bad = _manifest(tiny_dir).to_dict()
        mutate(bad)
        with pytest.raises(ManifestValidationError):
            RunManifest.from_dict(bad)


def test_registry_compares_runs_without_notebook_state(tiny_dir, tmp_path):
    reg = ExperimentRegistry(tmp_path / "runs.db")
    m1 = _manifest(tiny_dir)
    m2 = RunManifest(**{**m1.__dict__, "model": {"name": "toy", "version": "2"}})
    r1, r2 = reg.log_run(m1), reg.log_run(m2)
    reopened = ExperimentRegistry(tmp_path / "runs.db")
    assert reopened.get_run(r1) == m1
    assert [m.run_id for m in reopened.list_runs("toy")] == sorted([r1, r2])
    assert reopened.find_by_config(m1.config_hash) == [m1]
    rows = reopened.compare_runs([r1, r2], ["top3"])
    assert [r["model"] for r in rows] == ["toy@1", "toy@2"]
    assert rows[0]["top3"] == rows[1]["top3"]
    with pytest.raises(RunNotFound):
        reopened.get_run("missing")


def test_software_state_recorded():
    state = collect_software_state()
    assert state["packages"]["numpy"] == np.__version__
    assert state["python"]
    commit = git_commit()
    assert commit is None or len(commit) == 40


def test_in_memory_registry(tiny_dir):
    reg = ExperimentRegistry(":memory:")
    rid = reg.log_run(_manifest(tiny_dir))
    assert reg.get_run(rid).metrics["top3"]
