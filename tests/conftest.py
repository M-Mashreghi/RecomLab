from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TINY_DIR = ROOT / "data" / "fixtures" / "tiny"


@pytest.fixture
def tiny_dir() -> Path:
    return TINY_DIR


@pytest.fixture
def tiny_ds(tiny_dir):
    from recomlab.datasets import FixtureAdapter

    return FixtureAdapter().load(tiny_dir)


@pytest.fixture
def tiny_split(tiny_ds):
    from recomlab.datasets.splits import SplitConfig, make_split

    return make_split(tiny_ds.interactions, SplitConfig(n_validation=1, n_test=1))


@pytest.fixture
def tiny_ref():
    return {"dataset_id": "tiny", "version": "1"}
