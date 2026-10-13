from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TINY_DIR = ROOT / "data" / "fixtures" / "tiny"


@pytest.fixture
def tiny_dir() -> Path:
    return TINY_DIR
