"""Public confirmation must reconcile arithmetic and preserve negative decisions."""

import copy

import pytest

from jigsaw_rules.confirmation_evaluation import score_predictions
from jigsaw_rules.confirmation_report import audit_result
from tests.test_confirmation_evaluation import comparison


def evidence():
    table, labels, plan = comparison()
    plan.update(
        eligible_rows=len(table), familiar_rule_ids=["advertising"], unseen_rule_ids=["financial"]
    )
    return score_predictions(table, labels, plan), plan


def test_independent_aggregate_audit_reconciles_fixed_decision():
    result, plan = evidence()
    audit = audit_result(result, plan)
    assert audit["passed_checks"] == audit["acceptance_checks"] == 12
    assert audit["maximum_aggregate_difference"] < 1e-12


@pytest.mark.parametrize(
    "failure", ["mean", "weighted_loss", "decision", "interval", "count", "policy", "nan"]
)
def test_publication_refuses_inconsistent_aggregate_evidence(failure):
    original, plan = evidence()
    result = copy.deepcopy(original)
    if failure == "mean":
        result["scopes"]["all"]["candidate"]["rule_macro_auc"] -= 0.1
    elif failure == "weighted_loss":
        result["scopes"]["familiar"]["candidate"]["log_loss"] += 0.1
    elif failure == "decision":
        result["checks"]["positive_unseen_gain"] = False
    elif failure == "interval":
        result["primary_interval"]["observed_delta"] -= 0.1
    elif failure == "policy":
        result["scopes"]["all"]["candidate"]["per_rule_auc"]["financial"] -= 0.1
    elif failure == "nan":
        result["primary_interval"]["ci_lower"] = float("nan")
    else:
        result["class_counts"][0]["positive"] += 1
    with pytest.raises(ValueError):
        audit_result(result, plan)


def test_independent_audit_preserves_a_rejected_candidate():
    table, labels, plan = comparison()
    plan.update(
        eligible_rows=len(table), familiar_rule_ids=["advertising"], unseen_rule_ids=["financial"]
    )
    table["candidate"] = 1 - table["candidate"]
    result = score_predictions(table, labels, plan)
    assert result["status"] == "rejected"
    audit = audit_result(result, plan)
    assert audit["decision_reconciled"] and audit["passed_checks"] < 12
