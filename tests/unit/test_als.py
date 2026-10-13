import numpy as np
import pytest

from recomlab.datasets.labels import LabelPolicy
from recomlab.domain.eligibility import EligibilityFilter
from recomlab.domain.schemas import Interaction, Item
from recomlab.experiments import ExperimentRegistry
from recomlab.experiments.runner import run_offline_experiment
from recomlab.models import NotFittedError, PopularityRecommender
from recomlab.models.als import ALSArtifact, ALSParams, ALSRecommender, ImplicitALS

FAST = ALSParams(factors=4, iterations=10, seed=0)


def _ev(n, user, item):
    return Interaction(f"e{n}", user, item, "click", "2024-01-01T00:00:00Z")  # type: ignore[arg-type]


def _clusters():
    pairs = []
    for u in ("a1", "a2", "a3"):
        pairs += [(u, "x1"), (u, "x2"), (u, "x3")]
    for u in ("b1", "b2", "b3"):
        pairs += [(u, "y1"), (u, "y2"), (u, "y3")]
    pairs += [("a0", "x1"), ("a0", "x2"), ("b0", "y1"), ("b0", "y2")]
    # make y-items globally more popular so popularity would pick them for a0
    pairs += [(f"z{n}", "y3") for n in range(4)]
    events = [_ev(n, u, i) for n, (u, i) in enumerate(pairs)]
    items = [Item(i) for i in ("x1", "x2", "x3", "y1", "y2", "y3")]
    return events, items


def test_ranks_expected_items_on_tiny_fixture():
    events, items = _clusters()
    model = ALSRecommender(FAST).fit(events, items)
    elig = EligibilityFilter(items, events)
    assert model.recommend("a0", 1, elig)[0].item_id == "x3"
    assert model.recommend("b0", 1, elig)[0].item_id == "y3"
    pop = PopularityRecommender().fit(events, items)
    assert pop.recommend("a0", 1, elig)[0].item_id == "y3"  # CF beats a popularity strawman


def test_seeded_and_deterministic():
    events, items = _clusters()
    a = ImplicitALS(FAST).fit(events, items)
    b = ImplicitALS(FAST).fit(list(reversed(events)), items)
    assert np.allclose(a.item_factors, b.item_factors) and a.version == b.version
    c = ImplicitALS(ALSParams(factors=4, iterations=10, seed=1)).fit(events, items)
    assert c.version != a.version


def test_training_separate_from_serving(tmp_path):
    events, items = _clusters()
    artifact = ImplicitALS(FAST).fit(events, items)
    path = artifact.save(tmp_path / "als")
    served = ALSRecommender().load_artifact(ALSArtifact.load(path))
    scores = served.score("a0", ["x3", "y3", "unknown"])
    assert scores["x3"] > scores["y3"] and scores["unknown"] is None
    assert served.score("stranger", ["x3"]) == {"x3": None}
    meta = (path / "meta.json").read_text().replace(artifact.version, "als-tampered")
    (path / "meta.json").write_text(meta)
    with pytest.raises(ValueError):
        ALSArtifact.load(path)


def test_label_policy_feedback_and_params_recorded():
    events = [
        Interaction("r1", "u", "i1", "rating", "2024-01-01T00:00:00Z", 5),  # type: ignore[arg-type]
        Interaction("r2", "u", "i2", "rating", "2024-01-01T00:00:00Z", 1),  # type: ignore[arg-type]
    ]
    items = [Item("i1"), Item("i2")]
    model = ALSRecommender(FAST, LabelPolicy()).fit(events, items)
    assert model.artifact is not None
    assert model.artifact.label_version == LabelPolicy().label_version
    assert model.params()["feedback"] == "label_positive"
    assert model.params()["seed"] == 0
    with pytest.raises(NotFittedError):
        ALSRecommender().score("u", ["i1"])


@pytest.mark.integration
def test_evaluated_under_same_protocol_as_popularity(tiny_ds, tiny_split, tiny_ref, tmp_path):
    reg = ExperimentRegistry(tmp_path / "runs.db")
    run_ids = []
    for model in (PopularityRecommender(), ALSRecommender(FAST)):
        manifest, _ = run_offline_experiment(
            experiment_id="baselines",
            model=model,
            dataset=tiny_ds,
            dataset_ref=tiny_ref,
            split=tiny_split,
            label_policy=LabelPolicy(),
            ks=(3,),
            seeds={"model": 0},
            registry=reg,
        )
        run_ids.append(manifest.run_id)
    runs = [reg.get_run(r) for r in run_ids]
    assert runs[0].split == runs[1].split
    assert runs[0].label_policy == runs[1].label_policy
    assert runs[0].evaluation["universe"] == runs[1].evaluation["universe"]
    assert runs[1].model["name"] == "als" and runs[1].model["params"]["factors"] == 4
