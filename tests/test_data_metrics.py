import numpy as np
import pytest

from jigsaw_rules.data import normalize, validate_frame, validate_submission
from jigsaw_rules.metrics import evaluate


def test_macro_auc_is_not_pooled_auc():
    result = evaluate([0, 1, 0, 1], [0.7, 0.8, 0.1, 0.2], ["a", "a", "b", "b"])
    assert result["rule_macro_auc"] == 1.0
    assert result["pooled_auc"] == 0.75


def test_macro_weights_rules_equally():
    result = evaluate([0, 1, 0, 1, 0, 1], [0.1, 0.9, 0.1, 0.9, 0.9, 0.1], ["a"] * 4 + ["b"] * 2)
    assert result["rule_macro_auc"] == 0.5


@pytest.mark.parametrize("p", [[0.2, np.nan], [0.1, 1.2], [0.1]])
def test_metrics_reject_invalid_predictions(p):
    with pytest.raises(ValueError):
        evaluate([0, 1], p, ["r", "r"])


def test_single_class_rule_is_explicit_error():
    with pytest.raises(ValueError, match="single-class"):
        evaluate([0, 1], [0.2, 0.7], ["a", "b"])


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "label", "blank"])
def test_schema_rejects_bad_input(dataset, mutation):
    train, _, _ = dataset
    if mutation == "duplicate":
        train.loc[1, "row_id"] = train.loc[0, "row_id"]
    elif mutation == "missing":
        train = train.drop(columns="rule")
    elif mutation == "label":
        train.loc[0, "rule_violation"] = 2
    else:
        train.loc[0, "body"] = "  "
    with pytest.raises(ValueError):
        validate_frame(train, train=True)


def test_submission_requires_exact_alignment(dataset):
    _, _, sample = dataset
    validate_submission(sample, sample)
    with pytest.raises(ValueError, match="order"):
        validate_submission(sample.iloc[::-1].reset_index(drop=True), sample)
    sample.loc[0, "rule_violation"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        validate_submission(sample, sample)


def test_unicode_normalization():
    assert normalize("  Ｈello\nWORLD  ") == "hello world"
