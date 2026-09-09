"""Authored probability experiments test calibration math, not project performance."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit
from sklearn.metrics import log_loss

from jigsaw_rules.calibration import MonotoneSigmoid, loss_interval


def spec():
    return json.loads(
        (Path(__file__).resolve().parents[1] / "configs/model_validation.json").read_text()
    )["sigmoid"]


def test_sigmoid_improves_separate_authored_data_and_preserves_order():
    rng = np.random.default_rng(101)
    signal = rng.normal(size=4000)
    y = rng.binomial(1, expit(signal))
    raw = expit(3 * signal + 1)
    model = MonotoneSigmoid.fit(raw[:2000], y[:2000], spec())
    calibrated = model.predict(raw[2000:])
    assert log_loss(y[2000:], calibrated) < log_loss(y[2000:], raw[2000:])
    assert model.slope > 0
    assert (np.diff(model.predict(np.linspace(0, 1, 101))) >= 0).all()
    interval = loss_interval(
        y[2000:],
        raw[2000:],
        calibrated,
        np.arange(2000) // 2,
        draws=100,
        seed=2025,
        quantiles=[0.0125, 0.9875],
    )
    assert interval["lower"] > 0


def test_identity_and_extreme_probabilities_are_finite_nonmutating():
    raw = np.array([0.0, 0.2, 0.8, 1.0])
    np.testing.assert_array_equal(MonotoneSigmoid().predict(raw), raw)
    assert np.isfinite(MonotoneSigmoid(0.5, 1, fitted=True).predict(raw)).all()
    np.testing.assert_array_equal(raw, [0, 0.2, 0.8, 1])


@pytest.mark.parametrize("bad", [[0.2, np.nan], [-0.1, 0.3], [[0.5]], []])
def test_bad_probability_inputs_are_rejected(bad):
    with pytest.raises(ValueError):
        MonotoneSigmoid().predict(bad)


def test_label_alignment_and_positive_slope_are_required():
    with pytest.raises(ValueError, match="aligned"):
        MonotoneSigmoid.fit([0.2, 0.8], [1, 1], spec())
    with pytest.raises(ValueError, match="monotone"):
        MonotoneSigmoid(-1, fitted=True).predict([0.2, 0.8])
