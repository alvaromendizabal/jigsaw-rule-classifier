"""Representation control checks with authored data only."""

import numpy as np
import pytest
from scipy import sparse

from jigsaw_rules.diagnostics import nb_weights, support_scores
from tests.test_research import vectors


def test_nb_weights_equal_manual_class_conditional_counts():
    x = sparse.csr_matrix([[1.0, 0.0], [1.0, 2.0], [0.0, 3.0], [2.0, 1.0]])
    actual = nb_weights(x, np.array([0, 1, 0, 1]))
    positive, negative = np.array([4.0, 4.0]), np.array([2.0, 4.0])
    expected = np.log(positive / positive.sum()) - np.log(negative / negative.sum())
    np.testing.assert_allclose(actual, expected)
    with pytest.raises(ValueError, match="nonnegative"):
        nb_weights(-x, np.array([0, 1, 0, 1]))


def test_frozen_support_scores_are_order_invariant_and_bounded():
    x = vectors(20)
    scores = support_scores(x)
    swapped = support_scores(x[:, [0, 2, 1, 4, 3]])
    assert len(scores) == 7
    for name, values in scores.items():
        np.testing.assert_allclose(values, swapped[name])
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()


def test_latent_projection_and_classifier_do_not_fit_validation_inputs(tmp_path):
    import joblib
    import pandas as pd
    from scipy import sparse

    from jigsaw_rules.diagnostics import fit_lexical

    rng = np.random.default_rng(7)
    training = pd.DataFrame({"rule_violation": [0, 1] * 8})
    validation = pd.DataFrame({"row_id": [1, 2], "rule": ["a", "a"], "rule_violation": [0, 1]})
    x = sparse.csr_matrix(rng.random((16, 40)))
    v = sparse.csr_matrix(rng.random((2, 40)))
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    fit_lexical(a, "latent_word", {"word": (x, v)}, training, validation, "heldout_rule", 0)
    validation.rule_violation = 1 - validation.rule_violation
    fit_lexical(b, "latent_word", {"word": (x, v * 1000)}, training, validation, "heldout_rule", 0)
    model_a, projection_a = joblib.load(a / "model.joblib")
    model_b, projection_b = joblib.load(b / "model.joblib")
    np.testing.assert_array_equal(projection_a.components_, projection_b.components_)
    np.testing.assert_array_equal(model_a.coef_, model_b.coef_)
