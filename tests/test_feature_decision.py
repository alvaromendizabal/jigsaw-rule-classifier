"""A better average cannot bypass policy, uncertainty or probability-quality gates."""

import copy
from pathlib import Path

import numpy as np
import pytest

from jigsaw_rules.feature_decision import choose, fixed_mean, load_plan


def fixture():
    plan = load_plan(Path(__file__).resolve().parents[1])
    reference = {
        "rule_macro_auc": 0.7,
        "per_rule_auc": {"A": 0.7, "B": 0.7},
        "log_loss": 0.62,
        "brier": 0.22,
    }
    metrics = {name: copy.deepcopy(reference) for name in [plan["reference"], *plan["candidates"]]}
    intervals = [
        {"candidate": name, "reference": plan["reference"], "simultaneous_lower": -0.1}
        for name in plan["candidates"]
    ]
    return plan, metrics, intervals


def improve(metrics, intervals, name, auc=0.74):
    metrics[name].update(rule_macro_auc=auc, per_rule_auc={"A": auc, "B": auc})
    next(r for r in intervals if r["candidate"] == name)["simultaneous_lower"] = 0.01


def test_reference_is_retained_when_no_new_candidate_qualifies():
    plan, metrics, intervals = fixture()
    result = choose(metrics, intervals, plan)
    assert result["selected_candidate"] == "qwen_centroid"
    assert result["eligible_candidates"] == 0
    assert not result["final_training_authorized_by_evidence"]


@pytest.mark.parametrize("fault", ["gain", "interval", "policy", "log_loss", "brier"])
def test_each_independent_requirement_can_reject_a_candidate(fault):
    plan, metrics, intervals = fixture()
    name = plan["candidates"][0]
    improve(metrics, intervals, name)
    if fault == "gain":
        metrics[name].update(rule_macro_auc=0.704, per_rule_auc={"A": 0.704, "B": 0.704})
    elif fault == "interval":
        intervals[0]["simultaneous_lower"] = 0
    elif fault == "policy":
        metrics[name]["per_rule_auc"] = {"A": 0.81, "B": 0.67}
    else:
        metrics[name][fault] += 0.03
    result = choose(metrics, intervals, plan)
    assert result["selected_candidate"] == "qwen_centroid"
    assert not result["candidate_eligibility"][0]["eligible"]


def test_preregistered_simplicity_breaks_near_ties():
    plan, metrics, intervals = fixture()
    improve(metrics, intervals, "asymmetric_centroid", 0.74)
    improve(metrics, intervals, "semantic_intent", 0.741)
    assert choose(metrics, intervals, plan)["selected_candidate"] == "asymmetric_centroid"


def test_missing_policy_or_duplicate_interval_is_rejected():
    plan, metrics, intervals = fixture()
    metrics[plan["candidates"][0]]["per_rule_auc"].pop("B")
    with pytest.raises(ValueError, match="policy coverage"):
        choose(metrics, intervals, plan)
    plan, metrics, intervals = fixture()
    with pytest.raises(ValueError, match="once"):
        choose(metrics, [*intervals, intervals[0]], plan)


def test_fixed_average_is_row_local_symmetric_and_nonmutating():
    first, second = np.array([0.1, 0.9, 0.4]), np.array([0.3, 0.7, 0.8])
    before = first.copy()
    result = fixed_mean(first, second)
    np.testing.assert_allclose(result, [0.2, 0.8, 0.6])
    np.testing.assert_array_equal(first, before)
    np.testing.assert_array_equal(fixed_mean(second, first), result)
    np.testing.assert_array_equal(fixed_mean(first[[2, 0]], second[[2, 0]]), result[[2, 0]])
    with pytest.raises(ValueError):
        fixed_mean(first, [-1, 0, 0])
    with pytest.raises(ValueError):
        fixed_mean(first, [0.2])
