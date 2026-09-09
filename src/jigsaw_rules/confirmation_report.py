"""Audited public aggregates for the single protected comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jigsaw_rules.confirmation import MODELS, PROTOCOL_COMMIT, PROTOCOL_SHA256, load_protocol
from jigsaw_rules.confirmation_evaluation import prediction_freeze
from jigsaw_rules.final_model import verify_stage
from jigsaw_rules.runtime import atomic_bytes, atomic_json, digest

PUBLIC_FILES = {"results.json", "audit.json"}


def audit_result(result: dict, plan: dict) -> dict:
    """Independently reconcile aggregate arithmetic and the fixed promotion guards."""
    counts = {x["policy"]: x for x in result["class_counts"]}
    policies = set(plan["familiar_rule_ids"] + plan["unseen_rule_ids"])
    if (
        len(counts) != len(result["class_counts"])
        or set(counts) != policies
        or sum(x["rows"] for x in counts.values()) != plan["eligible_rows"]
    ):
        raise ValueError("Protected aggregate policy coverage differs")
    if any(x["positive"] + x["negative"] != x["rows"] for x in counts.values()):
        raise ValueError("Protected class-count arithmetic differs")
    guard = plan["acceptance"]
    sufficient = all(
        min(x["positive"], x["negative"]) >= guard["minimum_each_policy_class_count"]
        for x in counts.values()
    )
    if any(min(x["positive"], x["negative"]) == 0 for x in counts.values()):
        if result["status"] != "rejected" or result["checks"] != {
            "sufficient_policy_class_counts": False
        }:
            raise ValueError("A single-class policy cannot pass confirmation")
        return {"decision_reconciled": True, "metrics_available": False}
    scopes = result["scopes"]
    groups = {
        "all": sorted(policies),
        "familiar": plan["familiar_rule_ids"],
        "unseen": plan["unseen_rule_ids"],
    }
    if set(scopes) != set(groups) or set(result["per_policy"]) != policies:
        raise ValueError("Protected metric scope coverage differs")
    differences = []
    for scope, selected in groups.items():
        if set(scopes[scope]) != set(MODELS):
            raise ValueError("Protected descriptive model set differs")
        weights = np.asarray([counts[p]["rows"] for p in selected])
        for model in MODELS:
            values = scopes[scope][model]
            per_policy = [result["per_policy"][p][model] for p in selected]
            if set(values["per_rule_auc"]) != set(selected) or any(
                values["per_rule_auc"][p] != result["per_policy"][p][model]["rule_macro_auc"]
                for p in selected
            ):
                raise ValueError("Protected per-policy AUC differs")
            expected = {
                "rule_macro_auc": float(np.mean([x["rule_macro_auc"] for x in per_policy])),
                **{
                    metric: float(np.average([x[metric] for x in per_policy], weights=weights))
                    for metric in ["log_loss", "brier"]
                },
            }
            for metric, value in expected.items():
                difference = abs(values[metric] - value)
                differences.append(difference)
                if not np.isfinite(difference) or difference > 1e-12:
                    raise ValueError("Protected aggregate metric arithmetic differs")
    candidate, reference = scopes["all"]["candidate"], scopes["all"]["reference"]
    gain = candidate["rule_macro_auc"] - reference["rule_macro_auc"]
    interval = result["primary_interval"]
    if (
        not np.isfinite([interval[k] for k in ["observed_delta", "ci_lower", "ci_upper"]]).all()
        or abs(interval["observed_delta"] - gain) > 1e-12
        or interval["ci_lower"] > interval["ci_upper"]
        or interval["draws"] != plan["bootstrap"]["draws"]
        or interval["seed"] != plan["bootstrap"]["seed"]
        or interval["comparisons"] != 1
    ):
        raise ValueError("Primary protected interval identity differs")
    checks = {
        "sufficient_policy_class_counts": sufficient,
        "minimum_macro_auc_gain": gain >= guard["minimum_macro_auc_gain"],
        "positive_primary_interval": interval["ci_lower"]
        > guard["minimum_primary_interval_lower_exclusive"],
        "maximum_policy_auc_drop": max(
            reference["per_rule_auc"][p] - candidate["per_rule_auc"][p] for p in policies
        )
        <= guard["maximum_policy_auc_drop"],
    }
    for scope in ["familiar", "unseen"]:
        checks[f"positive_{scope}_gain"] = (
            scopes[scope]["candidate"]["rule_macro_auc"]
            - scopes[scope]["reference"]["rule_macro_auc"]
        ) > guard["minimum_route_macro_auc_gain_exclusive"]
    for scope in groups:
        for metric in ["log_loss", "brier"]:
            checks[f"{scope}_{metric}_guard"] = (
                scopes[scope]["candidate"][metric] - scopes[scope]["reference"][metric]
            ) <= guard[f"maximum_{metric}_increase_overall_and_each_route"]
    decision = "accepted" if all(checks.values()) else "rejected"
    if result["checks"] != checks or result["status"] != decision:
        raise ValueError("Protected acceptance decision differs from frozen guards")
    return {
        "decision_reconciled": True,
        "metrics_available": True,
        "maximum_aggregate_difference": max(differences),
        "acceptance_checks": len(checks),
        "passed_checks": sum(checks.values()),
        "primary_macro_auc_gain": gain,
        "scope": (
            "Aggregate arithmetic and fixed guard audit; no new target access or model selection"
        ),
    }


def export_report(root: Path, directory: Path) -> None:
    plan = load_protocol(root)
    path = directory / "evaluation"
    verify_stage(path)
    identity = json.loads((path / "identity.json").read_text())
    receipt = json.loads((path / "target_access.json").read_text())
    if (
        identity["prediction_freeze"] != prediction_freeze(root, directory)
        or json.loads((root / "reports/confirmation/prediction_freeze.json").read_text())
        != identity["prediction_freeze"]
        or receipt["identity"] != identity
        or identity["evaluation_source_sha256"]
        != digest(root / "src/jigsaw_rules/confirmation_evaluation.py")
        or identity["protocol_sha256"] != PROTOCOL_SHA256
    ):
        raise ValueError("Protected result lineage differs")
    result = json.loads((path / "results.json").read_text())
    audit = audit_result(result, plan)
    folder = root / "reports/confirmation"
    atomic_bytes(folder / "results.json", (path / "results.json").read_bytes())
    atomic_json(folder / "audit.json", audit)
    atomic_json(
        folder / "metadata.json",
        {
            "schema": 1,
            "run_id": directory.name,
            "protocol_commit": PROTOCOL_COMMIT,
            "protocol_sha256": PROTOCOL_SHA256,
            "prediction_commit": identity["prediction_commit"],
            "prediction_freeze_sha256": digest(folder / "prediction_freeze.json"),
            "candidate_sha256": plan["candidate_sha256"],
            "reference_sha256": plan["reference_sha256"],
            "solution_sha256": plan["solution_sha256"],
            "private_checkpoint_sha256": digest(path / "complete.json"),
            "first_target_access_at": receipt["first_access_at"],
            "evaluation_source_sha256": identity["evaluation_source_sha256"],
            "report_source_sha256": digest(Path(__file__)),
            "files": {name: digest(folder / name) for name in sorted(PUBLIC_FILES)},
            "scope": plan["scope"],
        },
    )


def confirmation_evidence(root: Path) -> dict | None:
    folder = root / "reports/confirmation"
    if not (folder / "metadata.json").exists():
        return None
    plan = load_protocol(root)
    metadata = json.loads((folder / "metadata.json").read_text())
    freeze = json.loads((folder / "prediction_freeze.json").read_text())
    if (
        metadata["run_id"] != freeze["run_id"]
        or freeze["rows"] != plan["eligible_rows"]
        or freeze["eligible_ids_sha256"] != plan["eligible_ids_sha256"]
        or freeze["targets_accessed"] is not False
        or metadata["protocol_commit"] != PROTOCOL_COMMIT
        or metadata["protocol_sha256"] != PROTOCOL_SHA256
        or metadata["candidate_sha256"] != plan["candidate_sha256"]
        or metadata["reference_sha256"] != plan["reference_sha256"]
        or metadata["solution_sha256"] != plan["solution_sha256"]
        or metadata["prediction_freeze_sha256"] != digest(folder / "prediction_freeze.json")
        or metadata["evaluation_source_sha256"]
        != digest(root / "src/jigsaw_rules/confirmation_evaluation.py")
        or metadata["report_source_sha256"] != digest(Path(__file__))
        or set(metadata["files"]) != PUBLIC_FILES
    ):
        raise ValueError("Stale protected confirmation evidence")
    for name, value in metadata["files"].items():
        if digest(folder / name) != value:
            raise ValueError("Protected public evidence checksum differs")
    result = json.loads((folder / "results.json").read_text())
    audit = audit_result(result, plan)
    if json.loads((folder / "audit.json").read_text()) != audit:
        raise ValueError("Protected public audit differs")
    return {"metadata": metadata, "results": result, "audit": audit}
