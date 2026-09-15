from __future__ import annotations

import math

import numpy as np

from scripts import selection_stage1 as s


def test_rmse_is_sqrt_brier():
    metric = s._metric(np.array([0, 1, 1, 0]), np.array([-2.0, 2.0, 1.0, -1.0]))
    assert abs(metric["probability_rmse"] - math.sqrt(metric["brier"])) < 1e-12


def test_macro_summary_rmse_is_sqrt_macro_brier():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    policy = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    score = np.array([-2.0, 2.0, -1.0, 1.0, -1.5, 1.5, -0.5, 0.5])
    metric = s._summary(y, score, policy)
    assert abs(metric["probability_rmse"] - math.sqrt(metric["brier"])) < 1e-12


def test_linear_shap_formula():
    class Model:
        coef_ = np.array([[2.0, -3.0]])

    train = np.array([[0.0, 1.0], [2.0, 3.0]])
    valid = np.array([[2.0, 2.0]])
    expected = np.abs((valid - train.mean(0)) * Model.coef_[0]).mean(0)
    np.testing.assert_allclose(s._linear_shap_abs(Model(), train, valid), expected)


def test_count_rule_uses_auc_one_se_and_brier_guard():
    rows = [
        {"k": 1, "mean_auc": 0.700, "se_auc": 0.010, "mean_brier": 0.260, "se_brier": 0.002},
        {"k": 2, "mean_auc": 0.715, "se_auc": 0.010, "mean_brier": 0.220, "se_brier": 0.003},
        {"k": 5, "mean_auc": 0.720, "se_auc": 0.020, "mean_brier": 0.218, "se_brier": 0.004},
    ]
    chosen, auc_threshold, brier_threshold = s._count_choice(rows)
    assert abs(auc_threshold - 0.700) < 1e-12
    assert abs(brier_threshold - 0.222) < 1e-12
    assert chosen == 2


def test_correlation_cluster_removes_redundant_channel():
    matrix = np.array([[0, 0, 1], [1, 1, 0], [2, 2, 1], [3, 3, 0]], float)
    representatives = s._cluster_representatives(matrix, [0, 1, 2], 0.98)
    assert 0 in representatives and 1 not in representatives and 2 in representatives


def test_global_group_split_no_overlap_and_all_policy_classes():
    rng = np.random.default_rng(1)
    rows = []
    for policy in (0, 1):
        for label in (0, 1):
            for group in range(30):
                rows.append((policy, label, f"g-{policy}-{label}-{group}"))
    policy = np.array([row[0] for row in rows])
    y = np.array([row[1] for row in rows])
    groups = np.array([row[2] for row in rows], dtype=object)
    X = rng.normal(size=(len(rows), 5))
    strata = policy * 2 + y
    for train, valid in s._splitter(5, 7).split(X, strata, groups):
        s._assert_split(y, policy, groups, train, valid)


def test_inner_selection_on_authored_data():
    rng = np.random.default_rng(5)
    rows = []
    for policy in (0, 1):
        for label in (0, 1):
            for group in range(40):
                rows.append((policy, label, f"g-{policy}-{label}-{group}"))
    policy = np.array([row[0] for row in rows])
    y = np.array([row[1] for row in rows])
    groups = np.array([row[2] for row in rows], dtype=object)
    X = rng.normal(size=(len(rows), 8))
    X[:, 0] += y
    X[:, 1] += 0.4 * y
    config = {
        "inner_splits": 4,
        "permutation_repeats": 1,
        "correlation_threshold": 0.98,
        "k_grid": [1, 2, 3, 5, 8],
    }
    result = s._inner_rank(X, y, policy, groups, [f"f{i}" for i in range(8)], config, 12)
    assert result["chosen_k"] in {1, 2, 3, 5, 8}
    assert np.isfinite(result["shap"]).all()
    assert np.isfinite(result["permutation"]).all()
