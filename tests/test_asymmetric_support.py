from __future__ import annotations

import copy

import numpy as np

from jigsaw_rules.asymmetric_support import METHOD, _pick, select_batch, summarize_fold
from jigsaw_rules.support_selection import prepare_selector


def fold_fixture():
    rule = "No advertising"
    return {
        "rule": rule,
        "audit": {},
        "training": [
            {"body": "positive near", "rule": rule, "rule_violation": 1, "repeat": 2},
            {"body": "positive far", "rule": rule, "rule_violation": 1, "repeat": 2},
            {"body": "negative near", "rule": rule, "rule_violation": 0, "repeat": 2},
            {"body": "negative far", "rule": rule, "rule_violation": 0, "repeat": 2},
        ],
        "queries": [{"row_id": 10, "body": "query body", "rule": rule}],
    }


def test_pick_supports_min_and_max_with_stable_first_tie():
    scores = np.array([0.4, 0.9, -0.8, -0.8])
    positions = np.array([0, 1, 2, 3])
    assert _pick(scores, positions, maximize=True) == 1
    assert _pick(scores, positions, maximize=False) == 2


def test_asymmetric_selector_nearest_positive_and_farthest_negative():
    fold = fold_fixture()
    train = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.8, 0.2],
            [-0.9, 0.1],
        ]
    )
    query = np.array([[1.0, 0.0]])
    state = prepare_selector(fold, train, query)
    rows = select_batch(fold, state, 0, 1)
    row = rows[0]
    assert row["asymmetric"]["positive"]["training_index"] == 0
    assert row["asymmetric"]["negative"]["training_index"] == 3
    assert row["semantic"]["negative"]["training_index"] == 2
    assert row["asymmetric_changed_vs_semantic"] is True


def test_selector_preserves_training_vector_mapping_under_joint_reorder():
    fold = fold_fixture()
    train = np.array([[1.0, 0.0], [-1.0, 0.0], [0.8, 0.2], [-0.9, 0.1]])
    query = np.array([[1.0, 0.0]])
    first = select_batch(fold, prepare_selector(fold, train, query), 0, 1)[0]["asymmetric"]
    order = [3, 1, 0, 2]
    reordered = copy.deepcopy(fold)
    reordered["training"] = [fold["training"][i] for i in order]
    second = select_batch(
        reordered, prepare_selector(reordered, train[order], query), 0, 1
    )[0]["asymmetric"]
    for sign in ("positive", "negative"):
        assert first[sign]["support_id"] == second[sign]["support_id"]


def test_zero_vector_support_is_never_selected():
    fold = fold_fixture()
    train = np.array([[1.0, 0.0], [0.0, 0.0], [0.8, 0.2], [-0.9, 0.1]])
    query = np.array([[1.0, 0.0]])
    row = select_batch(fold, prepare_selector(fold, train, query), 0, 1)[0]
    assert row["asymmetric"]["positive"]["training_index"] == 0


def test_summary_is_target_free_and_reports_diversity():
    fold = fold_fixture()
    train = np.array([[1.0, 0.0], [-1.0, 0.0], [0.8, 0.2], [-0.9, 0.1]])
    query = np.array([[1.0, 0.0]])
    state = prepare_selector(fold, train, query)
    rows = select_batch(fold, state, 0, 1)
    summary = summarize_fold(
        0,
        fold,
        rows,
        {
            "zero_training_rows": state["zero_training_rows"],
            "zero_same_rule_supports": state["zero_same_rule_supports"],
        },
    )
    assert summary["method"] == METHOD
    assert summary["query_targets_read"] is False
    assert summary["prediction_arrays_read"] is False
    assert summary["new_model_calls"] == 0
    assert summary["classes"]["positive"]["distinct_supports"] == 1
    assert summary["classes"]["negative"]["distinct_supports"] == 1


def test_empty_summary_is_rejected():
    try:
        summarize_fold(0, fold_fixture(), [], {})
    except ValueError as exc:
        assert "At least one query row" in str(exc)
    else:
        raise AssertionError("Expected empty summary to fail")
