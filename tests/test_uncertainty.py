import numpy as np
import pytest

from jigsaw_rules.uncertainty import paired_auc_interval


def test_identical_predictions_have_zero_paired_difference():
    y = np.tile([0, 1], 20)
    p = np.linspace(0, 1, len(y))
    result = paired_auc_interval(y, p, p, ["rule"] * len(y), np.arange(len(y)), draws=20)
    assert result["observed_delta"] == result["ci_lower"] == result["ci_upper"] == 0


def test_bootstrap_is_reproducible_and_preserves_grouped_pairs():
    y = np.tile([0, 1], 20)
    args = (y, 1 - y, y, ["rule"] * len(y), np.repeat(np.arange(20), 2))
    first = paired_auc_interval(*args, draws=20)
    assert first == paired_auc_interval(*args, draws=20)
    assert first["observed_delta"] == first["ci_lower"] == first["ci_upper"] == 1


def test_bootstrap_rejects_misaligned_predictions():
    with pytest.raises(ValueError, match="aligned"):
        paired_auc_interval([0, 1], [0.2], [0.1, 0.9], ["rule"] * 2, [0, 1])
