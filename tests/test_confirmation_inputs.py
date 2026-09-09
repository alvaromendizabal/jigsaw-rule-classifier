"""Confirmation preparation must work with target-free text alone."""

import pytest

from jigsaw_rules.confirmation_inputs import eligibility
from tests.test_expanded import cohort


def test_target_blind_eligibility_rejects_targets_and_exact_development_exposure():
    training = cohort().iloc[:30].drop(columns="rule_violation")
    reserved = cohort().iloc[30:40].drop(columns="rule_violation").copy()
    labeled = reserved.assign(rule_violation=1)
    with pytest.raises(ValueError, match="target labels"):
        eligibility(training, labeled)
    reserved.iloc[0, reserved.columns.get_loc("body")] = training.iloc[0].body
    with pytest.raises(ValueError, match="exposed"):
        eligibility(training, reserved)


def test_near_copy_and_self_support_flags_are_distinct():
    training = cohort().iloc[:30].drop(columns="rule_violation").copy()
    reserved = cohort().iloc[30:32].drop(columns="rule_violation").copy()
    text = "An extended unique sentence about the village festival and its recent announcement " * 4
    training.iloc[0, training.columns.get_loc("body")] = text
    reserved.iloc[0, reserved.columns.get_loc("body")] = text + "."
    reserved.iloc[1, reserved.columns.get_loc("body")] = "A totally different comment"
    reserved.iloc[1, reserved.columns.get_loc("positive_example_1")] = "A totally different comment"
    result = eligibility(training, reserved)
    assert result.near_development_copy.tolist() == [True, False]
    assert result.eligible.tolist() == [False, True]
    assert result.self_support_match.tolist() == [False, True]
