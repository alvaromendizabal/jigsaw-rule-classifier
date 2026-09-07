import json
from pathlib import Path

import numpy as np
import pytest

from jigsaw_rules.embeddings import load_spec
from jigsaw_rules.pipeline import run_baseline
from jigsaw_rules.review import review_run
from jigsaw_rules.runtime import atomic_json, digest
from jigsaw_rules.semantic import classifier, features, reference_splits
from jigsaw_rules.semantic_pipeline import run_semantic
from tests.helpers import TestEncoder


def test_example_order_does_not_change_features():
    rng = np.random.default_rng(4)
    vectors = rng.normal(size=(10, 5, 16))
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
    changed = vectors[:, [0, 2, 1, 4, 3]]
    np.testing.assert_array_equal(features(vectors), features(changed))


def test_scaler_only_sees_training_rows():
    model = classifier({"classifier_c": 1, "seed": 2025})
    training = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    model.fit(training, [0, 1, 0, 1])
    before = model[0].mean_.copy()
    model.predict_proba([[1e6, -1e6]])
    np.testing.assert_array_equal(model[0].mean_, before)
    np.testing.assert_array_equal(before, training.mean(axis=0))


@pytest.fixture
def experiment(tmp_path, dataset):
    baseline = run_baseline(tmp_path)
    spec = load_spec(Path(__file__).resolve().parents[1])
    spec["baseline_run"] = baseline.name
    return tmp_path, dataset[0], spec


def test_saved_splits_reject_modified_training_data(experiment):
    root, train, spec = experiment
    with (root / "data/raw/train.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Training data differs"):
        reference_splits(root, train, spec["baseline_run"])


def test_saved_splits_reject_leakage_even_with_updated_checksum(experiment):
    root, train, spec = experiment
    directory = root / "runs" / spec["baseline_run"] / "seen_rule_rule_examples_0"
    path = directory / "split.json"
    record = json.loads(path.read_text())
    record["train_row_ids"].append(record["valid_row_ids"][0])
    atomic_json(path, record)
    marker = json.loads((directory / "complete.json").read_text())
    marker["files"]["split.json"] = digest(path)
    atomic_json(directory / "complete.json", marker)
    with pytest.raises(ValueError, match="overlapping"):
        reference_splits(root, train, spec["baseline_run"])


def test_semantic_workflow_resumes_without_encoding_or_fitting(experiment, monkeypatch):
    root, _, spec = experiment
    encoder = TestEncoder()
    first = run_semantic(root, spec, encoder=encoder)
    calls = encoder.calls
    assert review_run(root, first.name, allow_synthetic=True)["data_kind"] == "synthetic"

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed stages must not refit")

    monkeypatch.setattr("jigsaw_rules.semantic_pipeline.classifier", forbidden)
    assert run_semantic(root, spec, encoder=encoder) == first
    assert encoder.calls == calls
    metrics = json.loads((first / "review/results.json").read_text())
    assert len(metrics) == 4
    assert all(x["data_kind"] == "synthetic" for x in metrics)


def test_test_encoder_cannot_label_competition_results(experiment):
    root, _, spec = experiment
    (root / "data/raw/SYNTHETIC.txt").unlink()
    with pytest.raises(ValueError, match="cannot evaluate competition"):
        run_semantic(root, spec, encoder=TestEncoder())
