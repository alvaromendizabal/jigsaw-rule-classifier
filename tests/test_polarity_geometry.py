import numpy as np
import pandas as pd
import pytest

from jigsaw_rules import polarity_geometry as pg


def frame():
    return pd.DataFrame(
        {
            "row_id": list(range(12)),
            "rule": ["a"] * 6 + ["b"] * 6,
            "rule_violation": [0, 1, 0, 1, 0, 1] * 2,
        }
    )


def arrays(seed=7):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(12, 3, 8))
    scores = rng.random(size=(12, 3, 3))
    row_ids = np.arange(12)
    return row_ids, scores, vectors


def test_alignment_reorders_by_row_id():
    f = frame().iloc[::-1].reset_index(drop=True)
    ids, _, vectors = arrays()
    out = pg.aligned_vectors(f, ids, vectors)
    assert np.allclose(out[0], vectors[11])


def test_alignment_rejects_missing_ids():
    f = frame()
    ids, _, vectors = arrays()
    with pytest.raises(ValueError, match="alignment"):
        pg.aligned_vectors(f, ids[:-1], vectors[:-1])


@pytest.mark.parametrize("mode", pg.MODES)
def test_polarity_scores_finite(mode):
    _, _, vectors = arrays()
    y = frame().rule_violation.to_numpy()
    pred = pg.polarity_score(vectors[:8, 2], y[:8], vectors[8:, 2], mode=mode)
    assert pred.shape == (4,)
    assert np.isfinite(pred).all()


def test_zero_norm_rejected():
    values = np.ones((4, 3))
    values[0] = 0
    with pytest.raises(ValueError, match="zero-norm"):
        pg.polarity_score(
            values,
            np.array([0, 1, 0, 1]),
            np.ones((2, 3)),
            mode="centroid",
        )


def test_whole_rule_folds_are_isolated():
    f = frame()
    folds = pg.whole_rule_folds(f)
    assert len(folds) == 2
    for train_idx, val_idx in folds:
        assert set(f.iloc[train_idx].rule).isdisjoint(set(f.iloc[val_idx].rule))


def test_end_to_end_has_no_compute_authorization():
    f = frame()
    ids, scores, vectors = arrays()
    for i, label in enumerate(f.rule_violation):
        vectors[i, :, 0] += 3 * label
        scores[i, 2, 0] = 0.5
    out = pg.evaluate(f, ids, scores, vectors)
    assert out["query_targets_read"] is False
    assert out["saved_prediction_arrays_read"] is False
    assert out["model_calls"] == 0
    assert out["gpu"] is False
    assert len(out["results"]["frozen_joint"]["folds"]) == 2
