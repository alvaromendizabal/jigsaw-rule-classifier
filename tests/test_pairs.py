"""Joint-text feature contracts and inference reuse without a model download."""

import numpy as np
import pytest

from jigsaw_rules import pairs
from jigsaw_rules.runtime import Progress
from tests.test_research import frame


def test_pair_inputs_ignore_targets_and_keep_both_rule_hypotheses():
    data = frame(3)
    original = pairs.pair_inputs(data)
    data.rule_violation = 1 - data.rule_violation
    assert pairs.pair_inputs(data) == original
    assert len(original) == 18
    assert "violates" in original[0][1]
    assert "follows" in original[1][1]


def test_pair_geometry_is_finite_order_invariant_and_has_exact_counts():
    p = np.random.default_rng(4).dirichlet([1, 1, 1], size=(20, 6))
    x, names = pairs.pair_candidates(p)
    swapped, changed_names = pairs.pair_candidates(p[:, [0, 1, 3, 2, 5, 4]])
    assert names == changed_names
    np.testing.assert_allclose(x, swapped)
    assert x.shape == (20, 99)
    assert len(set(names)) == 99
    assert sum(name.startswith("rule/") for name in names) == 27
    assert sum(name.startswith("support/") for name in names) == 72
    with pytest.raises(ValueError, match="six aligned"):
        pairs.pair_candidates(p * 2)


def test_pair_cache_reuses_verified_inference_after_reordering(tmp_path, monkeypatch):
    calls = []

    def encode(self, inputs, log):
        calls.append(len(inputs))
        return np.tile([0.2, 0.3, 0.5], (len(inputs), 1))

    monkeypatch.setattr(pairs.NLIEncoder, "encode", encode)
    data = frame(20)
    with Progress(tmp_path / "logs/test.jsonl", "authored_cache_test") as log:
        first, _ = pairs.cached_pair_probabilities(tmp_path, data, {}, log)
        completed = len(calls)
        second, _ = pairs.cached_pair_probabilities(tmp_path, data.iloc[::-1], {}, log)
    assert len(calls) == completed
    np.testing.assert_array_equal(first[::-1], second)
