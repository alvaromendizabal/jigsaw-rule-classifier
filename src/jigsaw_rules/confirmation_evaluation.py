"""One protected comparison, gated by a committed prediction freeze and access receipt."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from jigsaw_rules.confirmation import (
    MODELS,
    PROTOCOL_COMMIT,
    PROTOCOL_SHA256,
    load_protocol,
    read_inputs,
    strict_stage,
)
from jigsaw_rules.embeddings import QwenEncoder, content_key, load_spec
from jigsaw_rules.final_model import verify_stage
from jigsaw_rules.metrics import calibration_table, evaluate
from jigsaw_rules.runtime import Progress, atomic_json, digest
from jigsaw_rules.uncertainty import paired_auc_comparisons


def prediction_freeze(root: Path, directory: Path) -> dict:
    """Validate row, source, model and encoder lineage before publishing prediction hashes."""
    plan = load_protocol(root)
    _, assignments = read_inputs(root, directory)
    path = directory / "predictions"
    verify_stage(path)
    provenance = json.loads((path / "provenance.json").read_text())
    audit = json.loads((path / "audit.json").read_text())
    for key, expected in {
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "inputs_stage_sha256": digest(directory / "inputs/complete.json"),
        "candidate_sha256": plan["candidate_sha256"],
        "reference_sha256": plan["reference_sha256"],
        "targets_accessed": False,
    }.items():
        if provenance.get(key) != expected:
            raise ValueError("Prediction provenance differs: " + key)
    if not provenance.get("sources"):
        raise ValueError("Missing inference source lineage")
    if provenance["encoder"] != QwenEncoder(root, load_spec(root)).contract:
        raise ValueError("Protected encoder contract differs")
    for name, value in provenance["sources"].items():
        source = root / name
        if not source.resolve().is_relative_to(root) or digest(source) != value:
            raise ValueError("Inference source differs: " + name)
    table = pd.read_csv(path / "predictions.csv", float_precision="round_trip")
    validate_predictions(table, assignments, plan)
    if (
        audit["prediction_sha256"] != digest(path / "predictions.csv")
        or audit["eligible_ids_sha256"] != plan["eligible_ids_sha256"]
        or audit["rows"] != plan["eligible_rows"]
        or audit["targets_accessed"] is not False
    ):
        raise ValueError("Prediction audit differs")
    return {
        "schema": 1,
        "run_id": directory.name,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "prediction_sha256": digest(path / "predictions.csv"),
        "prediction_stage_sha256": digest(path / "complete.json"),
        "prediction_provenance_sha256": digest(path / "provenance.json"),
        "input_stage_sha256": digest(directory / "inputs/complete.json"),
        "eligible_ids_sha256": plan["eligible_ids_sha256"],
        "rows": len(table),
        "candidate_sha256": plan["candidate_sha256"],
        "reference_sha256": plan["reference_sha256"],
        "encoder": provenance["encoder"],
        "familiar_rows": int(table.familiar.sum()),
        "unseen_rows": int((~table.familiar).sum()),
        "targets_accessed": False,
    }


def validate_predictions(table: pd.DataFrame, assignments: pd.DataFrame, plan: dict) -> None:
    expected_columns = [
        "row_id",
        "rule_id",
        "Usage",
        "body_sha256",
        "self_support_match",
        "familiar",
        *MODELS,
    ]
    if list(table.columns) != expected_columns:
        raise ValueError("Protected prediction schema differs")
    if (
        len(table) != plan["eligible_rows"]
        or content_key(table.row_id.tolist()) != plan["eligible_ids_sha256"]
    ):
        raise ValueError("Protected prediction identities differ")
    for name in ["row_id", "rule_id", "Usage", "body_sha256", "self_support_match"]:
        if not np.array_equal(table[name], assignments[name]):
            raise ValueError("Protected prediction assignment differs: " + name)
    if table.familiar.dtype != bool or not np.array_equal(
        table.familiar, assignments.rule_id.isin(plan["familiar_rule_ids"])
    ):
        raise ValueError("Protected route membership differs")
    if not np.isfinite(table[list(MODELS)].to_numpy()).all() or any(
        not table[name].between(0, 1).all() for name in MODELS
    ):
        raise ValueError("Protected scores must be finite probabilities")


def require_committed_freeze(root: Path, commit: str, expected: dict) -> None:
    """Require Git evidence of the exact prediction freeze, descending from preregistration."""
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("An exact prediction-freeze Git commit is required")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, commit],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ValueError("Prediction commit does not descend from preregistration")
    blob = subprocess.check_output(
        ["git", "show", f"{commit}:reports/confirmation/prediction_freeze.json"], cwd=root
    )
    if json.loads(blob) != expected:
        raise ValueError("Committed prediction freeze differs from current artifacts")
    config = subprocess.check_output(
        ["git", "show", f"{commit}:configs/confirmation.json"], cwd=root
    )
    if hashlib.sha256(config).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Committed acceptance protocol differs")


def read_eligible_targets(path: Path, assignments: pd.DataFrame) -> np.ndarray:
    """Only eligible labels are interpreted; all other target values remain unused strings."""
    if assignments.row_id.duplicated().any():
        raise ValueError("Duplicate eligible target identities")
    expected = assignments.set_index("row_id")
    labels = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["row_id", "Usage", "rule_violation", "rule"]:
            raise ValueError("Solution schema differs")
        for row in reader:
            row_id = int(row["row_id"])
            if row_id not in expected.index:
                continue
            if row_id in labels:
                raise ValueError("Duplicate eligible target")
            metadata = expected.loc[row_id]
            if row["rule"] != metadata.rule_id or row["Usage"] != metadata.Usage:
                raise ValueError("Eligible target policy/Usage differs from frozen assignment")
            if row["rule_violation"] not in {"0", "1"}:
                raise ValueError("Eligible target is not binary")
            labels[row_id] = int(row["rule_violation"])
    if set(labels) != set(assignments.row_id):
        raise ValueError("Eligible target coverage differs")
    return np.asarray([labels[row_id] for row_id in assignments.row_id], dtype=np.int64)


def score_predictions(table: pd.DataFrame, labels: np.ndarray, plan: dict) -> dict:
    """Evaluate predeclared models; descriptive controls never change the candidate decision."""
    if labels.shape != (len(table),) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Aligned binary confirmation labels required")
    masks = {
        "all": np.ones(len(table), dtype=bool),
        "familiar": table.familiar.to_numpy(),
        "unseen": ~table.familiar.to_numpy(),
    }
    scopes, counts = {}, []
    for policy in sorted(table.rule_id.unique()):
        mask = table.rule_id.eq(policy).to_numpy()
        counts.append(
            {
                "policy": policy,
                "rows": int(mask.sum()),
                "positive": int(labels[mask].sum()),
                "negative": int((1 - labels[mask]).sum()),
            }
        )
    sufficient = all(
        min(x["positive"], x["negative"]) >= plan["acceptance"]["minimum_each_policy_class_count"]
        for x in counts
    )
    if any(min(x["positive"], x["negative"]) == 0 for x in counts):
        return {
            "status": "rejected",
            "checks": {"sufficient_policy_class_counts": False},
            "class_counts": counts,
            "reason": "AUC undefined for a single-class policy",
        }
    for scope, mask in masks.items():
        if not mask.any():
            raise ValueError("Both frozen routes need eligible observations")
        scopes[scope] = {
            name: evaluate(labels[mask], table.loc[mask, name], table.loc[mask, "rule_id"])
            for name in MODELS
        }
    interval = paired_auc_comparisons(
        labels,
        {name: table[name].to_numpy() for name in ["candidate", "reference"]},
        table.rule_id.to_numpy(),
        table.body_sha256.to_numpy(),
        [("protected_candidate_minus_reference", "candidate", "reference")],
        draws=plan["bootstrap"]["draws"],
        seed=plan["bootstrap"]["seed"],
    )[0]
    interval["scope"] = (
        "Fixed protected predictions and observed six policies; "
        "normalized-body clusters shared across policies"
    )
    c, r, guard = scopes["all"]["candidate"], scopes["all"]["reference"], plan["acceptance"]
    checks = {
        "sufficient_policy_class_counts": sufficient,
        "minimum_macro_auc_gain": c["rule_macro_auc"] - r["rule_macro_auc"]
        >= guard["minimum_macro_auc_gain"],
        "positive_primary_interval": interval["ci_lower"]
        > guard["minimum_primary_interval_lower_exclusive"],
        "maximum_policy_auc_drop": max(
            r["per_rule_auc"][p] - value for p, value in c["per_rule_auc"].items()
        )
        <= guard["maximum_policy_auc_drop"],
    }
    for scope in ["familiar", "unseen"]:
        checks[f"positive_{scope}_gain"] = (
            scopes[scope]["candidate"]["rule_macro_auc"]
            - scopes[scope]["reference"]["rule_macro_auc"]
            > guard["minimum_route_macro_auc_gain_exclusive"]
        )
    for metric in ["log_loss", "brier"]:
        for scope in masks:
            checks[f"{scope}_{metric}_guard"] = (
                scopes[scope]["candidate"][metric] - scopes[scope]["reference"][metric]
                <= guard[f"maximum_{metric}_increase_overall_and_each_route"]
            )
    by_policy = {
        policy: {
            name: evaluate(
                labels[table.rule_id.eq(policy)],
                table.loc[table.rule_id.eq(policy), name],
                table.loc[table.rule_id.eq(policy), "rule_id"],
            )
            for name in MODELS
        }
        for policy in sorted(table.rule_id.unique())
    }
    coverage = []
    for scope, mask in masks.items():
        p, y = table.loc[mask, "candidate"].to_numpy(), labels[mask]
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            keep = np.maximum(p, 1 - p) >= threshold
            coverage.append(
                {
                    "scope": scope,
                    "minimum_confidence": threshold,
                    "rows": int(keep.sum()),
                    "coverage": float(keep.mean()),
                    "error_rate": float(np.mean((p[keep] >= 0.5) != y[keep]))
                    if keep.any()
                    else None,
                }
            )
    self_support = table.self_support_match.to_numpy()
    sensitivity = {"excluded_rows": int(self_support.sum()), "status": "identical_cohort"}
    if self_support.any():
        reduced = table.loc[~self_support]
        sensitivity = {
            "excluded_rows": int(self_support.sum()),
            "status": "descriptive_only",
            "models": {
                name: evaluate(labels[~self_support], reduced[name], reduced.rule_id)
                for name in MODELS
            },
        }
    return {
        "status": "accepted" if all(checks.values()) else "rejected",
        "checks": checks,
        "class_counts": counts,
        "scopes": scopes,
        "per_policy": by_policy,
        "primary_interval": interval,
        "confidence_coverage": coverage,
        "reliability": {
            name: calibration_table(labels, table[name]).to_dict("records") for name in MODELS
        },
        "self_support_sensitivity": sensitivity,
        "control_selection": "Descriptive controls cannot replace the preregistered candidate",
        "scope": plan["scope"],
    }


def evaluate_confirmation(root: Path, directory: Path, prediction_commit: str) -> Path:
    plan = load_protocol(root)
    freeze = prediction_freeze(root, directory)
    require_committed_freeze(root, prediction_commit, freeze)
    _, assignments = read_inputs(root, directory)
    solution = root / "data/released/v1/solution.csv"
    if digest(solution) != plan["solution_sha256"]:
        raise ValueError("Pinned solution bytes differ")
    identity = {
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "prediction_commit": prediction_commit,
        "prediction_freeze": freeze,
        "solution_sha256": plan["solution_sha256"],
        "evaluation_source_sha256": digest(Path(__file__)),
    }
    receipt = root / "runs/confirmation/target_access.json"
    if receipt.exists():
        prior = json.loads(receipt.read_text())
        if prior["identity"] != identity:
            raise ValueError("Reserve already opened under a different frozen experiment")
    if (directory / "evaluation").exists():
        verify_stage(directory / "evaluation")
        if not receipt.exists():
            raise ValueError("Completed confirmation lacks its target-access receipt")
        if json.loads((directory / "evaluation/identity.json").read_text()) != identity:
            raise ValueError("Completed confirmation belongs to a different experiment")
        return directory / "evaluation"

    def action(path):
        if not receipt.exists():
            # Create exclusively before target interpretation; never replace an access receipt.
            receipt.parent.mkdir(parents=True, exist_ok=True)
            with receipt.open("x") as stream:
                json.dump(
                    {"first_access_at": datetime.now(UTC).isoformat(), "identity": identity},
                    stream,
                    indent=2,
                )
                stream.flush()
                os.fsync(stream.fileno())
        with Progress(directory / "evaluation_events.jsonl", "protected_confirmation"):
            labels = read_eligible_targets(solution, assignments)
            table = pd.read_csv(
                directory / "predictions/predictions.csv", float_precision="round_trip"
            )
            result = score_predictions(table, labels, plan)
        atomic_json(path / "results.json", result)
        atomic_json(path / "identity.json", identity)
        atomic_json(path / "target_access.json", json.loads(receipt.read_text()))

    return strict_stage(directory, "evaluation", action)
