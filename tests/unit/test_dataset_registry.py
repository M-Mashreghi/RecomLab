import json
import shutil

import pytest

from recomlab.datasets import (
    DatasetChecksumMismatch,
    DatasetManifest,
    DatasetRegistry,
    FixtureAdapter,
    MovieLensAdapter,
    materialize,
    verify_snapshot,
)
from recomlab.datasets.registry import DatasetRegistryError


def _materialize_tiny(tiny_dir, tmp_path, version="1"):
    adapter = FixtureAdapter(version=version)
    ds = adapter.load(tiny_dir)
    return ds, materialize(ds, adapter.describe(tiny_dir), tmp_path / "processed")


def test_fixture_registers_and_resolves_deterministically(tiny_dir, tmp_path):
    _, m1 = _materialize_tiny(tiny_dir, tmp_path / "a")
    _, m2 = _materialize_tiny(tiny_dir, tmp_path / "b")
    assert m1.checksum == m2.checksum and m1.files == m2.files
    reg = DatasetRegistry(tmp_path / "registry")
    reg.register(m1)
    reg.register(m1)  # idempotent
    assert reg.resolve("tiny") == m1
    assert reg.resolve("tiny", "1") == m1
    assert m1.synthetic and m1.stats["n_interactions"] == 33
    out = tmp_path / "a" / "processed" / "tiny" / "1"
    assert json.loads((out / "manifest.json").read_text())["checksum"] == m1.checksum
    card = (out / "DATA_CARD.md").read_text()
    assert "Synthetic / simulated:** yes" in card and m1.checksum in card


def test_processed_output_kept_separate_from_raw(tiny_dir, tmp_path):
    raw = tmp_path / "raw"
    shutil.copytree(tiny_dir, raw)
    before = {p.name: p.read_bytes() for p in raw.iterdir()}
    adapter = FixtureAdapter()
    materialize(adapter.load(raw), adapter.describe(raw), tmp_path / "processed")
    assert {p.name: p.read_bytes() for p in raw.iterdir()} == before
    assert (tmp_path / "processed" / "tiny" / "1" / "interactions.csv").exists()


def test_checksum_change_detected(tiny_dir, tmp_path):
    _, manifest = _materialize_tiny(tiny_dir, tmp_path)
    out = tmp_path / "processed" / "tiny" / "1"
    verify_snapshot(manifest, out)
    with (out / "interactions.csv").open("a") as fh:
        fh.write("e999,u1,i02,click,,2024-02-01T00:00:00Z,\n")
    with pytest.raises(DatasetChecksumMismatch, match="interactions.csv"):
        verify_snapshot(manifest, out)


def test_same_version_with_different_checksum_rejected(tiny_dir, tmp_path):
    _, manifest = _materialize_tiny(tiny_dir, tmp_path)
    reg = DatasetRegistry(tmp_path / "registry")
    reg.register(manifest)
    tampered = DatasetManifest.from_dict({**manifest.to_dict(), "checksum": "0" * 64})
    with pytest.raises(DatasetChecksumMismatch):
        reg.register(tampered)
    for version in ("2", "10"):
        reg.register(DatasetManifest.from_dict({**manifest.to_dict(), "version": version}))
    assert reg.resolve("tiny").version == "10"
    with pytest.raises(DatasetRegistryError):
        reg.resolve("missing")


def test_manifest_requires_license_and_checksums():
    with pytest.raises(DatasetRegistryError):
        DatasetManifest(
            dataset_id="x",
            version="1",
            source="s",
            schema_version="1.0",
            license_note="",
            retrieval_date="d",
            files={"a": "b"},
            checksum="c",
        )


def test_movielens_adapter_on_synthetic_sample(tmp_path):
    raw = tmp_path / "ml"
    raw.mkdir()
    (raw / "movies.csv").write_text(
        'movieId,title,genres\n1,"Toy, A (1995)",Animation|Comedy\n2,B (1996),(no genres listed)\n',
        encoding="utf-8",
    )
    (raw / "ratings.csv").write_text(
        "userId,movieId,rating,timestamp\n1,1,4.0,964982703\n1,2,3.5,964982224\n2,1,5.0,964983000\n",
        encoding="utf-8",
    )
    adapter = MovieLensAdapter()
    ds = adapter.load(raw)
    assert ds.user_ids == ("ml_u1", "ml_u2")
    assert ds.item_map()["ml_i1"].categories == ("Animation", "Comedy")
    assert ds.item_map()["ml_i2"].categories == ()
    assert [e.event_id for e in ds.interactions] == ["ml_e2", "ml_e1", "ml_e3"]
    meta = adapter.describe(raw)
    assert not meta["synthetic"] and "GroupLens" in meta["license_note"]
    assert meta["retrieval_date"] == "unknown"
