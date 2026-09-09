"""Nested validation isolation and final artifact contracts, using authored data."""

import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from jigsaw_rules import model_validation
from jigsaw_rules.data import EXAMPLES, normalize
from jigsaw_rules.embeddings import content_key
from jigsaw_rules.expanded import design
from tests.test_expanded import cohort

ROOT = Path(__file__).resolve().parents[1]


def test_committed_model_protocol_rejects_changed_thresholds(tmp_path):
    plan = model_validation.load_plan(ROOT)
    (tmp_path / "configs").mkdir()
    plan["calibration_selection"]["minimum_log_loss_improvement"] = 0
    (tmp_path / "configs/model_validation.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="pre-score"):
        model_validation.load_plan(tmp_path)


def test_nested_splits_cover_only_outer_training_and_ignore_outer_targets():
    frame = cohort()
    plan = model_validation.load_plan(ROOT)
    outer = design(frame, {"seen_rule_folds": 3, "seed": 2025})["seen_rule"][0]
    records = model_validation.nested_splits(frame, outer, plan)
    modified = frame.copy()
    modified["rule_violation"] = modified.rule_violation.astype(object)
    modified.loc[outer["valid"], "rule_violation"] = "UNOPENED"
    assert model_validation.nested_splits(modified, outer, plan) == records
    covered = [i for r in records for i in r["valid"]]
    assert sorted(covered) == sorted(outer["train"])
    for record in records:
        assert not (set(record["train"]) | set(record["valid"])) & set(outer["valid"])
        forbidden = set(frame.iloc[record["valid"]].body.map(normalize))
        for column in ["body", *EXAMPLES]:
            assert not set(frame.iloc[record["train"]][column].map(normalize)) & forbidden


@pytest.mark.parametrize("column", ["body", *EXAMPLES])
def test_outer_exposure_is_rejected_before_calibration(column):
    frame = cohort()
    plan = model_validation.load_plan(ROOT)
    outer = design(frame, {"seen_rule_folds": 3, "seed": 2025})["seen_rule"][0]
    frame.loc[outer["train"][0], column] = frame.loc[outer["valid"][0], "body"]
    with pytest.raises(ValueError, match="Outer validation"):
        model_validation.nested_splits(frame, outer, plan)


def test_final_fit_persists_selected_families_and_identity_fallback(tmp_path):
    frame = cohort()
    rng = np.random.default_rng(100)
    vectors = rng.normal(size=(len(frame), 5, 8))
    vectors /= np.linalg.norm(vectors, axis=2, keepdims=True)
    plan = model_validation.load_plan(ROOT)
    decisions = {p: {"retain_calibration": False} for p in ["seen_rule", "heldout_rule"]}
    model_validation.final_fit(
        tmp_path, frame, vectors, np.full(len(frame), 0.5), decisions, plan, None
    )
    artifact = json.loads((tmp_path / "artifact.json").read_text())
    model = joblib.load(tmp_path / "candidate.joblib")
    assert not model.familiar_calibrator.fitted and not model.unseen_calibrator.fitted
    assert artifact["families"] == list(model.familiar.screens_) == plan["families"]
    assert artifact["training_ids_sha256"] == content_key(frame.row_id.tolist())
    assert artifact["selected_features"] == sum(len(x) for x in model.familiar.selected_.values())
    assert not artifact["confirmation_targets_accessed"]
    for family in plan["families"]:
        catalog = pd.read_csv(tmp_path / f"catalog_{family}.csv.gz")
        assert len(catalog) == model.familiar.audit_[family]["candidates"]


def test_calibration_decision_requires_all_predeclared_checks():
    from jigsaw_rules.calibration import calibration_decision

    frame = cohort()
    raw = np.where(frame.rule_violation, 0.8, 0.2)
    plan = copy.deepcopy(model_validation.load_plan(ROOT)["calibration_selection"])
    plan["bootstrap_draws"] = 100
    decision = calibration_decision(frame, raw, raw, plan)
    assert not decision["retain_calibration"]
    assert not decision["checks"]["practical_log_loss_gain"]
    assert not decision["checks"]["paired_loss_interval"]
