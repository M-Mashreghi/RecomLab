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


@pytest.fixture
def serving_context(tiny_ds, tiny_split):
    """History/items/as_of for serving tests (train + validation history)."""
    from datetime import UTC, datetime

    return tiny_ds.items, tiny_split.train_until("test"), datetime(2024, 1, 14, tzinfo=UTC)


@pytest.fixture
def bundle_factory(tmp_path, serving_context):
    """Train an ALS artifact on the serving history and register a bundle."""
    from recomlab.models.als import ALSParams, ImplicitALS

    items, history, _ = serving_context

    def make(
        registry,
        ranker="heuristic",
        generators=("popularity", "item_item", "collaborative"),
        seed=0,
        **spec_extra,
    ):
        art_dir = tmp_path / f"als-src-{seed}"
        if not art_dir.exists():
            ImplicitALS(ALSParams(factors=4, iterations=8, seed=seed)).fit(history, items).save(
                art_dir
            )
        spec = {
            "generators": list(generators),
            "ranker": {"type": ranker},
            "candidate_budget": 8,
            "artifacts": {"als": "als"},
            **spec_extra,
        }
        return registry.register_bundle(
            spec, {"als": art_dir}, {"catalog_size": len(items), "seed": seed}
        )

    return make
