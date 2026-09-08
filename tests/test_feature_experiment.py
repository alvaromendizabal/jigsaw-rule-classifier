"""Contracts for task-informed, training-only feature ablations."""

import json

import numpy as np
import pandas as pd
import pytest

from jigsaw_rules.features import (
    FeatureClassifier,
    export_features,
    feature_evidence,
    run_features,
    structural_features,
)
from jigsaw_rules.pipeline import run_baseline
from jigsaw_rules.runtime import digest


@pytest.mark.parametrize(
    "variant,width",
    [("rule_text", 2), ("support_contrast", 16), ("structure", 8), ("combined", 26)],
)
def test_dense_groups_and_probability_contract(dataset, variant, width):
    train, test, _ = dataset
    model = FeatureClassifier(variant).fit(train)
    matrix, names = model.dense(test)
    assert matrix.shape == (len(test), width)
    assert len(set(names)) == width
    assert np.isfinite(matrix).all()
    prediction = model.predict(test)
    assert np.isfinite(prediction).all()
    assert ((0 <= prediction) & (prediction <= 1)).all()


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="Unknown feature"):
        FeatureClassifier("unregistered")


def test_support_order_is_invariant(dataset):
    train, test, _ = dataset
    model = FeatureClassifier("combined").fit(train)
    swapped = test.copy()
    for kind in ("positive", "negative"):
        swapped[f"{kind}_example_1"] = test[f"{kind}_example_2"]
        swapped[f"{kind}_example_2"] = test[f"{kind}_example_1"]
    np.testing.assert_allclose(model.dense(test)[0], model.dense(swapped)[0])
    np.testing.assert_allclose(model.predict(test), model.predict(swapped))


def test_validation_text_and_labels_never_fit_features(dataset):
    train, test, _ = dataset
    model = FeatureClassifier("combined").fit(train)
    held = test.copy()
    held["body"] = "uniquelyheldouttoken NEVER https://example.org WHY?"
    before = model.scaler.mean_.copy()
    original = model.dense(held)[0]
    held["rule_violation"] = 1
    np.testing.assert_array_equal(model.dense(held)[0], original)
    model.predict(held)
    np.testing.assert_array_equal(model.scaler.mean_, before)
    assert "uniquelyheldouttoken" not in model.word.vocabulary_


def test_structural_features_are_finite_for_empty_authored_input():
    values = structural_features(pd.DataFrame({"body": ["", "ABC?", "not a link"]}))
    assert values.shape == (3, 8)
    assert np.isfinite(values).all()


def test_complete_feature_experiment_reuses_folds_without_touching_reference(
    tmp_path, dataset, monkeypatch
):
    reference = run_baseline(tmp_path)
    original = {
        str(p.relative_to(reference)): digest(p) for p in reference.rglob("*") if p.is_file()
    }
    result = run_features(tmp_path, reference.name)
    records = json.loads((result / "review/results.json").read_text())
    assert len(records) == 8
    assert all(r["data_kind"] == "synthetic" for r in records)
    oof = pd.read_csv(result / "review/oof.csv")
    assert oof.groupby(["protocol", "model"]).size().eq(len(dataset[0])).all()
    assert not list(result.rglob("submission.csv"))
    assert original == {
        str(p.relative_to(reference)): digest(p) for p in reference.rglob("*") if p.is_file()
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("An intact completed feature fold must not refit")

    monkeypatch.setattr(FeatureClassifier, "fit", forbidden)
    assert run_features(tmp_path, reference.name) == result
    with pytest.raises(ValueError, match="matching competition"):
        export_features(tmp_path, result)


def test_absent_feature_evidence_is_not_fabricated(tmp_path):
    assert feature_evidence(tmp_path) is None


def test_research_figures_render_plotly_and_static_fallbacks():
    """Public figures need no private data and support both notebook renderers."""
    import nbformat

    from scripts.execute_notebooks import execute_inprocess, validate_execution

    source = "from pathlib import Path\nfrom scripts.build_research_report import display_figure\n"
    source += (
        "for name in ('ablation', 'screening', 'stability'):\n    display_figure(Path.cwd(), name)"
    )
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])
    execute_inprocess(nb, {})
    validate_execution(nb)
    outputs = [item for item in nb.cells[0].outputs if item.output_type == "display_data"]
    assert len(outputs) == 3
    for output in outputs:
        assert "image/svg+xml" in output.data
        assert "application/vnd.plotly.v1+json" in output.data
