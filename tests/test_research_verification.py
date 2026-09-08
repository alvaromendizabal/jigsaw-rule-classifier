import pandas as pd
import pytest

from jigsaw_rules.metrics import evaluate
from scripts.verify_research import compare_metrics, verify_predictions


def test_metric_comparison_detects_tampering():
    with pytest.raises(ValueError, match="differs"):
        compare_metrics({"per_rule_auc": {"rule": 0.7}}, {"per_rule_auc": {"rule": 0.8}})


def test_oof_verification_aligns_ids_and_rejects_wrong_labels():
    train = pd.DataFrame(
        {"row_id": [1, 2, 3, 4], "rule": ["a"] * 4, "rule_violation": [0, 1, 0, 1]}
    )
    oof = train.copy()
    oof["probability"] = [0.1, 0.8, 0.3, 0.6]
    oof["model"], oof["protocol"] = "candidate", "heldout_rule"
    records = [
        {
            "model": "candidate",
            "protocol": "heldout_rule",
            "metrics": evaluate(train.rule_violation, oof.probability, train.rule),
        }
    ]
    assert verify_predictions(train, oof.iloc[::-1], records) == 1
    oof.loc[0, "rule_violation"] = 1
    with pytest.raises(ValueError, match="identity differs"):
        verify_predictions(train, oof, records)
