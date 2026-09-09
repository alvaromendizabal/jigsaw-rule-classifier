"""Confirmatory checks must reject bad candidates and prevent unfrozen target access."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jigsaw_rules import confirmation_evaluation as evaluation
from jigsaw_rules.confirmation import load_protocol


def comparison():
    plan = load_protocol(Path(__file__).resolve().parents[1])
    plan = copy.deepcopy(plan)
    plan["bootstrap"]["draws"] = 100
    labels = np.tile([0, 1], 60)
    table = pd.DataFrame(
        {
            "row_id": np.arange(len(labels)),
            "rule_id": np.repeat(["advertising", "financial"], 60),
            "familiar": np.repeat([True, False], 60),
            "body_sha256": ["body_" + str(i % 60) for i in range(len(labels))],
            "self_support_match": False,
            "candidate": np.where(labels, 0.8, 0.2),
            "reference": 0.5,
            "uncalibrated_route": np.where(labels, 0.99, 0.01),
            "centroid_everywhere": 0.5,
        }
    )
    return table, labels, plan


def test_fixed_candidate_passes_all_guards_and_shared_group_interval():
    table, y, plan = comparison()
    result = evaluation.score_predictions(table, y, plan)
    assert result["status"] == "accepted"
    assert all(result["checks"].values())
    assert result["primary_interval"]["observed_delta"] == pytest.approx(0.5)
    assert result["primary_interval"]["ci_lower"] == pytest.approx(0.5)
    assert len(result["confidence_coverage"]) == 18
    assert result["self_support_sensitivity"]["status"] == "identical_cohort"
    json.dumps(result, allow_nan=False)


def test_descriptive_control_cannot_rescue_failed_familiar_route():
    table, y, plan = comparison()
    table.loc[table.familiar, "candidate"] = np.where(y[table.familiar], 0.2, 0.8)
    result = evaluation.score_predictions(table, y, plan)
    assert result["status"] == "rejected"
    assert not result["checks"]["positive_familiar_gain"]
    assert not result["checks"]["maximum_policy_auc_drop"]
    assert result["scopes"]["all"]["uncalibrated_route"]["rule_macro_auc"] == 1


def test_probability_quality_guard_rejects_worse_log_loss_despite_better_auc():
    table, y, plan = comparison()
    # Most positives rank correctly, but three overconfident errors per policy harm proper scores.
    table["candidate"] = np.where(y, 0.999999999, 0.000000001)
    table.loc[[1, 3, 5, 61, 63, 65], "candidate"] = 0.000000000001
    table["reference"] = np.where(y, 0.51, 0.49)
    # Make the reference ranking imperfect but retain its moderate probabilities.
    table.loc[table.index % 4 == 1, "reference"] = 0.48
    result = evaluation.score_predictions(table, y, plan)
    assert (
        result["scopes"]["all"]["candidate"]["rule_macro_auc"]
        > result["scopes"]["all"]["reference"]["rule_macro_auc"]
    )
    assert not result["checks"]["all_log_loss_guard"]
    assert result["status"] == "rejected"


def test_single_class_policy_is_explicit_rejection_not_an_auc_crash():
    table, y, plan = comparison()
    y[table.familiar] = 0
    result = evaluation.score_predictions(table, y, plan)
    assert result["status"] == "rejected"
    assert result["reason"] == "AUC undefined for a single-class policy"


def test_small_class_count_blocks_acceptance_even_with_perfect_ranking():
    table, y, plan = comparison()
    plan["acceptance"]["minimum_each_policy_class_count"] = 31
    result = evaluation.score_predictions(table, y, plan)
    assert result["status"] == "rejected"
    assert not result["checks"]["sufficient_policy_class_counts"]


def test_stream_only_interprets_eligible_targets_and_preserves_input_order(tmp_path):
    path = tmp_path / "solution.csv"
    path.write_text(
        "row_id,Usage,rule_violation,rule\n8,Private,0,financial\n9,Public,NOT_A_LABEL,unselected\n7,Private,1,financial\n"
    )
    assignments = pd.DataFrame({"row_id": [7, 8], "Usage": "Private", "rule_id": "financial"})
    np.testing.assert_array_equal(evaluation.read_eligible_targets(path, assignments), [1, 0])


@pytest.mark.parametrize(
    "rows,message",
    [
        ("7,Private,1,financial\n7,Private,0,financial\n", "Duplicate"),
        ("7,Public,1,financial\n", "Usage"),
        ("7,Private,1,advertising\n", "policy"),
        ("7,Private,2,financial\n", "binary"),
        ("8,Private,1,financial\n", "coverage"),
    ],
)
def test_target_reader_rejects_mismatched_frozen_assignments(tmp_path, rows, message):
    path = tmp_path / "solution.csv"
    path.write_text("row_id,Usage,rule_violation,rule\n" + rows)
    assignments = pd.DataFrame({"row_id": [7], "Usage": ["Private"], "rule_id": ["financial"]})
    with pytest.raises(ValueError, match=message):
        evaluation.read_eligible_targets(path, assignments)


def test_uncommitted_prediction_freeze_blocks_before_solution_access(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(evaluation, "load_protocol", lambda root: {})
    monkeypatch.setattr(evaluation, "prediction_freeze", lambda root, directory: {})
    monkeypatch.setattr(
        evaluation, "read_eligible_targets", lambda *a: events.append("target_read")
    )
    with pytest.raises(ValueError, match="Git commit"):
        evaluation.evaluate_confirmation(tmp_path, tmp_path / "run", "uncommitted")
    assert events == []


def test_completed_score_reuses_receipt_and_blocks_a_changed_experiment(tmp_path, monkeypatch):
    from jigsaw_rules.runtime import atomic_bytes, digest

    table, y, plan = comparison()
    directory = tmp_path / "runs/confirmation/synthetic"
    assignments = table[["row_id", "rule_id"]].assign(Usage="Private")
    solution = tmp_path / "data/released/v1/solution.csv"
    records = assignments.assign(rule_violation=y).rename(columns={"rule_id": "rule"})
    atomic_bytes(
        solution,
        records[["row_id", "Usage", "rule_violation", "rule"]].to_csv(index=False).encode(),
    )
    plan["solution_sha256"] = digest(solution)
    atomic_bytes(directory / "predictions/predictions.csv", table.to_csv(index=False).encode())
    freeze = {"prediction_sha256": "synthetic_identity"}
    monkeypatch.setattr(evaluation, "load_protocol", lambda root: plan)
    monkeypatch.setattr(evaluation, "prediction_freeze", lambda root, path: freeze.copy())
    monkeypatch.setattr(evaluation, "require_committed_freeze", lambda *args: None)
    monkeypatch.setattr(evaluation, "read_inputs", lambda *args: (None, assignments))
    result = evaluation.evaluate_confirmation(tmp_path, directory, "a" * 40)
    assert json.loads((result / "results.json").read_text())["status"] == "accepted"
    marker, receipt = (
        digest(result / "complete.json"),
        digest(tmp_path / "runs/confirmation/target_access.json"),
    )

    def forbidden(*args):
        raise AssertionError("Completed comparison reopened targets")

    monkeypatch.setattr(evaluation, "read_eligible_targets", forbidden)
    assert evaluation.evaluate_confirmation(tmp_path, directory, "a" * 40) == result
    assert digest(result / "complete.json") == marker
    assert digest(tmp_path / "runs/confirmation/target_access.json") == receipt
    freeze["prediction_sha256"] = "changed_identity"
    with pytest.raises(ValueError, match="already opened"):
        evaluation.evaluate_confirmation(tmp_path, directory, "a" * 40)
