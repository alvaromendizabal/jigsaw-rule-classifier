"""Adversarial tests for target leakage, fold references and tied AUC comparisons."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from jigsaw_rules.context import ContextEncoder, ReferenceRanks
from jigsaw_rules.data import EXAMPLES, normalize
from jigsaw_rules.uncertainty import paired_auc_comparisons
from tests.test_research import frame


def test_target_encoding_excludes_own_labels_and_all_duplicate_comment_labels():
    training = frame(90)
    training.loc[1, "body"] = training.loc[0, "body"]
    before, names = ContextEncoder().fit_transform(training)
    training.loc[[0, 1], "rule_violation"] = 1 - training.loc[[0, 1], "rule_violation"]
    after, _ = ContextEncoder().fit_transform(training)
    np.testing.assert_array_equal(before[:2], after[:2])
    assert len(names) == 9


def test_inner_target_encoding_purges_support_copies_of_validation_comments():
    training = frame(90)
    groups = training.body.map(normalize)
    fit, valid = next(GroupKFold(3).split(training, groups=groups))
    training.loc[fit[0], EXAMPLES[0]] = training.loc[valid[0], "body"]
    encoder = ContextEncoder()
    first, _ = encoder.fit_transform(training)
    training.loc[fit[0], "rule_violation"] = 1 - training.loc[fit[0], "rule_violation"]
    second, _ = ContextEncoder().fit_transform(training)
    np.testing.assert_array_equal(first[valid], second[valid])
    assert encoder.inner_audit_[0]["purged_rows"] >= 1


def test_unknown_metadata_uses_training_prior_and_never_reads_validation_targets():
    training = frame(30)
    encoder = ContextEncoder().fit(training)
    validation = frame(4).assign(subreddit="unseen", rule="new rule")
    actual, names = encoder.transform(validation.drop(columns="rule_violation"))
    np.testing.assert_array_equal(actual[:, [0, 3, 6]], np.full((4, 3), 0.5))
    np.testing.assert_array_equal(actual[:, [1, 4, 7]], np.zeros((4, 3)))
    np.testing.assert_array_equal(actual[:, [2, 5, 8]], np.ones((4, 3)))
    assert len(set(names)) == len(names)


def test_rank_reference_is_fixed_under_validation_extremes_and_ties():
    ranker = ReferenceRanks().fit(np.array([[1, 2], [1, 4], [3, 6]], dtype=float))
    before = ranker.sorted_.copy()
    result = ranker.transform(np.array([[1, 3], [-100, 100]], dtype=float))
    np.testing.assert_allclose(result, [[1 / 3, 1 / 3], [0, 1]])
    np.testing.assert_array_equal(before, ranker.sorted_)
    with pytest.raises(ValueError, match="schema"):
        ranker.transform(np.zeros((3, 1)))


def test_joint_bootstrap_matches_sklearn_with_ties_and_shared_groups():
    y = np.tile([0, 1, 0, 1, 1, 0], 20)
    rules = np.repeat(["first", "second"], 60)
    groups = np.arange(120) // 2
    rng = np.random.default_rng(7)
    a, b = rng.integers(0, 5, 120) / 5, rng.integers(0, 5, 120) / 5
    comparisons = [("a-b", "a", "b"), ("b-a", "b", "a")]
    result = paired_auc_comparisons(y, {"a": a, "b": b}, rules, groups, comparisons, draws=100)
    expected = np.mean(
        [
            roc_auc_score(y[rules == r], a[rules == r])
            - roc_auc_score(y[rules == r], b[rules == r])
            for r in set(rules)
        ]
    )
    assert result[0]["observed_delta"] == pytest.approx(expected)
    assert result[1]["observed_delta"] == pytest.approx(-expected)
    assert result == paired_auc_comparisons(
        y, {"a": a, "b": b}, rules, groups, comparisons, draws=100
    )


def test_joint_bootstrap_rejects_misaligned_or_invalid_scores():
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="aligned"):
        paired_auc_comparisons(y, {"a": y[:2]}, y, y, [("x", "a", "a")])
    with pytest.raises(ValueError, match="both classes"):
        paired_auc_comparisons(y, {"a": y}, y, np.arange(4), [("x", "a", "a")])
