"""Resolution geometry must preserve the full control and never mutate cached vectors."""

import numpy as np
import pytest

from jigsaw_rules.diagnostics import support_scores
from jigsaw_rules.resolution import prefix_scores


def vectors():
    values = np.random.default_rng(2025).normal(size=(12, 5, 1024)).astype(np.float32)
    return values / np.linalg.norm(values, axis=2, keepdims=True)


def test_prefix_geometry_preserves_reference_cache_and_batch_independence():
    values = vectors()
    original = values.copy()
    result = prefix_scores(values, [32, 64, 128, 256, 512, 1024])
    np.testing.assert_array_equal(result["centroid_1024"], support_scores(values)["qwen_centroid"])
    prefix = values[:, :, :64].copy()
    prefix /= np.linalg.norm(prefix, axis=2, keepdims=True)
    np.testing.assert_array_equal(result["centroid_64"], support_scores(prefix)["qwen_centroid"])
    np.testing.assert_array_equal(values, original)
    for name, part in prefix_scores(values[:3], [32, 1024]).items():
        np.testing.assert_array_equal(part, result[name][:3])


@pytest.mark.parametrize("fault", ["nonfinite", "nonunit", "wrong_width", "zero_prefix"])
def test_invalid_geometry_is_rejected(fault):
    values = vectors()
    if fault == "nonfinite":
        values[0, 0, 0] = np.nan
    elif fault == "nonunit":
        values *= 3
    elif fault == "wrong_width":
        values = values[:, :, :64]
    else:
        values[:, :, :32] = 0
        values /= np.linalg.norm(values, axis=2, keepdims=True)
    with pytest.raises(ValueError):
        prefix_scores(values, [32, 1024])
