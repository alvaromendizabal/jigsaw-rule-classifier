import numpy as np
import pandas as pd
import pytest

from jigsaw_rules import conflict_supervision as cs


def sample_frame(rules=("r1", "r2", "r3", "r4", "r5")):
    rows = []
    row_id = 0
    for rule in rules:
        for i in range(10):
            rows.append(
                {
                    "row_id": row_id,
                    "body": f"{rule} body {i}",
                    "rule": rule,
                    "rule_violation": i % 2,
                }
            )
            row_id += 1
    rows.extend(
        [
            {"row_id": row_id, "body": "duplicate conflict", "rule": rules[0], "rule_violation": 0},
            {"row_id": row_id + 1, "body": "duplicate conflict", "rule": rules[0], "rule_violation": 1},
            {"row_id": row_id + 2, "body": "majority conflict", "rule": rules[-1], "rule_violation": 1},
            {"row_id": row_id + 3, "body": "majority conflict", "rule": rules[-1], "rule_violation": 1},
            {"row_id": row_id + 4, "body": "majority conflict", "rule": rules[-1], "rule_violation": 0},
        ]
    )
    return pd.DataFrame(rows)


def test_soft_keeps_empirical_target_and_weight():
    out = cs.collapse_training(sample_frame(), "soft")
    tie = out[(out.rule == "r1") & (out.body == "duplicate conflict")].iloc[0]
    majority = out[(out.rule == "r5") & (out.body == "majority conflict")].iloc[0]
    assert tie.target == pytest.approx(0.5)
    assert tie.sample_weight == 2
    assert bool(tie.is_conflict)
    assert majority.target == pytest.approx(2 / 3)
    assert majority.sample_weight == 3


def test_majority_drops_tie_and_hardens_nontie():
    out = cs.collapse_training(sample_frame(), "majority")
    assert not ((out.rule == "r1") & (out.body == "duplicate conflict")).any()
    majority = out[(out.rule == "r5") & (out.body == "majority conflict")].iloc[0]
    assert majority.target == 1.0
    assert majority.sample_weight == 3


def test_drop_conflicts_removes_all_conflicting_pairs():
    out = cs.collapse_training(sample_frame(), "drop_conflicts")
    assert not out.is_conflict.any()
    assert "duplicate conflict" not in set(out.body)
    assert "majority conflict" not in set(out.body)


def test_unknown_regime_rejected():
    with pytest.raises(ValueError, match="unknown regime"):
        cs.collapse_training(sample_frame(), "other")


def test_grouped_rule_folds_have_zero_rule_overlap():
    frame = sample_frame()
    for train_idx, val_idx in cs.grouped_rule_folds(frame, n_splits=5):
        train_rules = set(frame.iloc[train_idx].rule)
        val_rules = set(frame.iloc[val_idx].rule)
        assert train_rules.isdisjoint(val_rules)


def test_grouped_rule_folds_are_deterministic():
    frame = sample_frame()
    a = cs.grouped_rule_folds(frame, n_splits=5)
    b = cs.grouped_rule_folds(frame, n_splits=5)
    assert len(a) == len(b)
    for (atr, ava), (btr, bva) in zip(a, b, strict=True):
        np.testing.assert_array_equal(atr, btr)
        np.testing.assert_array_equal(ava, bva)


def test_two_rule_gate_requires_both_folds_not_impossible_three_wins():
    frame = sample_frame(("r1", "r2"))
    result = cs.evaluate_regimes(frame, n_splits=5)
    assert len(result["fold_metadata"]) == 2
    assert result["gate"]["valid_auc_folds"] == 2
    assert result["gate"]["required_fold_wins"] == 2
    assert cs.required_fold_wins(2) == 2


def test_five_fold_gate_requires_sixty_percent():
    assert cs.required_fold_wins(5) == 3
    assert cs.required_fold_wins(4) == 3
    assert cs.required_fold_wins(3) == 2


def test_missing_columns_stop():
    with pytest.raises(ValueError, match="missing columns"):
        cs.evaluate_regimes(pd.DataFrame({"rule": ["a", "b"]}))


def test_end_to_end_returns_all_regimes_and_no_gpu_authorization():
    result = cs.evaluate_regimes(sample_frame(), n_splits=5)
    assert set(result["regimes"]) == set(cs.REGIMES)
    assert result["gate"]["gpu_training_authorized_by_this_module"] is False
    assert result["query_targets_read"] is False
    assert result["prediction_arrays_read"] is False
    assert result["model_calls"] == 0
    assert result["gpu"] is False
    for meta in result["fold_metadata"]:
        assert meta["train_rules"] + meta["validation_rules"] == 5
