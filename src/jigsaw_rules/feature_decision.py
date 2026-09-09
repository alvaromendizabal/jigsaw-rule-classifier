"""A preregistered feature stopping decision, distinct from confirmation or promotion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jigsaw_rules.data import normalize
from jigsaw_rules.embeddings import content_key
from jigsaw_rules.expanded import aligned_predictions, expanded_evidence, load_development
from jigsaw_rules.formatting import formatting_evidence, verified_stage
from jigsaw_rules.metrics import evaluate
from jigsaw_rules.resolution import resolution_evidence
from jigsaw_rules.retrieval import retrieval_evidence
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest, stage
from jigsaw_rules.uncertainty import paired_auc_comparisons

PROTOCOL_COMMIT = "ceaea9f512507606d12532be40b5a6378f04242a"
PLAN_SHA256 = "75e03eaa263e31566b6dbb112a58983b26ed1f2af8f6cd57a70776235aa8eab0"
FUSION_COMMIT = "af33a31837ea7d3e98b58783fa07ba77b46ac521"
FUSION_SHA256 = "0310b68f06b4d245bb71eb082ad9a05acc63bd0d66d9d04650cd12ddb16d1d26"
PUBLIC_FILES = ("decision.json", "uncertainty.json", "verification.json", "fusion.json")
STUDIES = {
    "expanded": expanded_evidence,
    "retrieval": retrieval_evidence,
    "resolution": resolution_evidence,
    "formatting": formatting_evidence,
}


def load_plan(root):
    path = root / "configs/feature_decision.json"
    if digest(path) != PLAN_SHA256:
        raise ValueError("Feature decision thresholds differ from their pre-score protocol")
    return json.loads(path.read_text())


def choose(metrics: dict, uncertainty: list[dict], plan: dict) -> dict:
    reference = metrics[plan["reference"]]
    intervals = {r["candidate"]: r for r in uncertainty}
    if set(intervals) != set(plan["candidates"]) or len(intervals) != len(uncertainty):
        raise ValueError("Selection uncertainty does not cover every frozen candidate once")
    candidates = []
    for name in plan["candidates"]:
        result = metrics[name]
        if set(result["per_rule_auc"]) != set(reference["per_rule_auc"]):
            raise ValueError("Candidate policy coverage differs")
        gain = result["rule_macro_auc"] - reference["rule_macro_auc"]
        drops = {
            r: reference["per_rule_auc"][r] - result["per_rule_auc"][r]
            for r in reference["per_rule_auc"]
        }
        loss_change = result["log_loss"] - reference["log_loss"]
        brier_change = result["brier"] - reference["brier"]
        lower = intervals[name]["simultaneous_lower"]
        values = [gain, *drops.values(), loss_change, brier_change, lower]
        if not np.isfinite(values).all() or intervals[name]["reference"] != plan["reference"]:
            raise ValueError("Invalid candidate selection statistics")
        conditions = {
            "practical_auc_gain": gain >= plan["minimum_macro_auc_gain"],
            "positive_simultaneous_interval": lower > 0,
            "per_policy_stability": max(drops.values()) <= plan["maximum_per_policy_auc_drop"],
            "log_loss_tolerance": loss_change <= plan["maximum_log_loss_increase"],
            "brier_tolerance": brier_change <= plan["maximum_brier_increase"],
        }
        candidates.append(
            {
                "candidate": name,
                "macro_auc": result["rule_macro_auc"],
                "auc_gain": gain,
                "simultaneous_lower": lower,
                "policy_auc_drops": drops,
                "log_loss_change": loss_change,
                "brier_change": brier_change,
                "eligible": all(conditions.values()),
                "conditions": conditions,
                "failed_conditions": [name for name, passed in conditions.items() if not passed],
            }
        )
    eligible = [r for r in candidates if r["eligible"]]
    selected = plan["reference"]
    if eligible:
        best = max(r["macro_auc"] for r in eligible)
        tied = {r["candidate"] for r in eligible if best - r["macro_auc"] <= plan["tie_tolerance"]}
        selected = next(name for name in plan["simplicity_order"] if name in tied)
    return {
        "selected_candidate": selected,
        "reference": plan["reference"],
        "candidate_eligibility": candidates,
        "eligible_candidates": len(eligible),
        "selection_uses_development_only": True,
        "confirmation_targets_accessed": False,
        "final_training_authorized_by_evidence": False,
        "next_gate": "Review marginal feature value, then freeze the confirmation protocol",
    }


def fixed_mean(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first.ndim != 1 or first.shape != second.shape or not len(first):
        raise ValueError("Expected aligned nonempty probability vectors")
    combined = np.stack([first, second])
    if not np.isfinite(combined).all() or ((combined < 0) | (combined > 1)).any():
        raise ValueError("Expected finite probabilities in [0, 1]")
    return combined.mean(axis=0)


def dependencies(root: Path) -> dict:
    studies = {name: reader(root) for name, reader in STUDIES.items()}
    if any(value is None for value in studies.values()):
        raise ValueError("Every required feature study must be verified before stopping")
    return {
        "studies": {name: value["metadata"]["run_id"] for name, value in studies.items()},
        "metadata_sha256": {
            name: digest(root / "reports" / name / "metadata.json") for name in STUDIES
        },
        "coverage_sha256": digest(root / "docs/FEATURE_COVERAGE.md"),
        "verifier_sha256": digest(root / "scripts/verify_formatting.py"),
        "runner_sha256": digest(root / "scripts/run_feature_decision.py"),
    }


def run_decision(root: Path, verification: dict) -> Path:
    """The canonical runner obtains verification by replaying private artifacts first."""
    plan, inputs = load_plan(root), dependencies(root)
    fusion_path = root / "configs/fusion.json"
    if digest(fusion_path) != FUSION_SHA256:
        raise ValueError("Fixed complementarity control differs from its protocol")
    fusion = json.loads(fusion_path.read_text())
    if fusion["parent_formatting_run"] != inputs["studies"]["formatting"]:
        raise ValueError("Complementarity control requires its exact parent study")
    required = {
        "run_id": inputs["studies"]["formatting"],
        "metric_records": 20,
        "model_prediction_replays": 28,
        "feature_matrix_replays": 28,
        "frozen_score_replays": 8,
        "new_encoder_calls": 0,
        "new_model_fits": 0,
        "confirmation_targets_accessed": False,
    }
    if any(verification.get(k) != v for k, v in required.items()):
        raise ValueError("Complete private feature/model replay is required before stopping")
    source = root / "runs/formatting" / inputs["studies"]["formatting"] / "review"
    verified_stage(source)
    frame = load_development(root)
    oof = pd.read_csv(source / "oof.csv")
    predictions = {
        name: aligned_predictions(frame, part)
        for name, part in oof[oof.protocol == "heldout_rule"].groupby("model")
    }
    evidence = formatting_evidence(root)
    metrics = {
        r["model"]: r["metrics"] for r in evidence["results"] if r["protocol"] == "heldout_rule"
    }
    identity = {
        "schema": 1,
        "protocol_commit": PROTOCOL_COMMIT,
        "plan_sha256": digest(root / "configs/feature_decision.json"),
        "fusion_protocol_commit": FUSION_COMMIT,
        "fusion_plan_sha256": FUSION_SHA256,
        "source_sha256": digest(Path(__file__)),
        "inputs": inputs,
        "private_oof_sha256": digest(source / "oof.csv"),
        "verification_sha256": content_key(verification),
        "confirmation_targets_accessed": False,
    }
    directory = root / "runs/feature_decision" / content_key(identity)[:20]
    atomic_json(directory / "provenance.json", identity)
    with Progress(root / "logs/feature_decision.jsonl", "feature_stopping_decision"):

        def review(path):
            contrasts = [(name, name, plan["reference"]) for name in plan["candidates"]]
            uncertainty = paired_auc_comparisons(
                frame.rule_violation,
                predictions,
                frame.rule,
                frame.body.map(normalize),
                contrasts,
                draws=plan["bootstrap_draws"],
                seed=plan["seed"],
            )
            decision = choose(metrics, uncertainty, plan)
            fused = fixed_mean(*(predictions[name] for name in fusion["components"]))
            fusion_metrics = evaluate(frame.rule_violation, fused, frame.rule)
            fusion_uncertainty = paired_auc_comparisons(
                frame.rule_violation,
                {fusion["candidate"]: fused, fusion["reference"]: predictions[fusion["reference"]]},
                frame.rule,
                frame.body.map(normalize),
                [(fusion["candidate"], fusion["candidate"], fusion["reference"])],
                draws=fusion["bootstrap_draws"],
                seed=fusion["seed"],
            )
            fusion_decision = choose(
                {
                    fusion["candidate"]: fusion_metrics,
                    fusion["reference"]: metrics[fusion["reference"]],
                },
                fusion_uncertainty,
                {
                    **plan,
                    "candidates": [fusion["candidate"]],
                    "simplicity_order": [fusion["candidate"]],
                },
            )
            closed = (
                not decision["eligible_candidates"] and not fusion_decision["eligible_candidates"]
            )
            decision.update(
                {
                    "feature_research_status": "complete_for_declared_scope" if closed else "open",
                    "notebook_02_scientific_gate": "passed" if closed else "follow-up required",
                    "final_training_authorized_by_evidence": closed,
                    "production_promotion_authorized_by_evidence": False,
                    "fusion_eligible": bool(fusion_decision["eligible_candidates"]),
                    "stopping_rationale": (
                        "The major applicable families, targeted semantic hypotheses and one fixed "
                        "complementarity control are complete. None of the latest alternatives "
                        "meets the predeclared requirements to replace the simpler centroid. "
                        "The source-bound coverage ledger explains exclusions and remaining model "
                        "research. Further undirected feature expansion has lower priority than "
                        "locked confirmation. This does not prove universal feature exhaustion."
                        if closed
                        else "A candidate meets the declared requirements. Preserve it and inspect "
                        "the resulting research opportunity before claiming diminishing returns."
                    ),
                }
            )
            saved = frame[["row_id", "rule", "rule_violation"]].copy()
            saved["probability"] = fused
            atomic_bytes(path / "fusion_predictions.csv", saved.to_csv(index=False).encode())
            atomic_json(path / "decision.json", decision)
            atomic_json(path / "uncertainty.json", uncertainty)
            atomic_json(path / "verification.json", verification)
            atomic_json(
                path / "fusion.json",
                {
                    "metrics": fusion_metrics,
                    "uncertainty": fusion_uncertainty,
                    "decision": fusion_decision,
                    "new_encoder_calls": 0,
                    "new_model_fits": 0,
                    "adaptive_development_control": True,
                },
            )

        review_path = stage(directory, "review", review)
        saved = pd.read_csv(review_path / "fusion_predictions.csv")
        actual = aligned_predictions(frame, saved)
        np.testing.assert_allclose(
            actual, fixed_mean(*(predictions[n] for n in fusion["components"])), rtol=0, atol=1e-12
        )
        recomputed = evaluate(frame.rule_violation, actual, frame.rule)
        recorded = json.loads((review_path / "fusion.json").read_text())["metrics"]
        for metric in ("rule_macro_auc", "log_loss", "brier"):
            if abs(recomputed[metric] - recorded[metric]) > 1e-12:
                raise ValueError("Fixed fusion metric differs from saved predictions")
        public = root / "reports/feature_decision"
        for name in PUBLIC_FILES:
            atomic_bytes(public / name, (review_path / name).read_bytes())
        atomic_json(
            public / "metadata.json",
            {
                **identity,
                "run_id": directory.name,
                "private_checkpoint_sha256": digest(review_path / "complete.json"),
                "files": {name: digest(public / name) for name in PUBLIC_FILES},
            },
        )
    return directory


def decision_evidence(root: Path) -> dict | None:
    path = root / "reports/feature_decision/metadata.json"
    if not path.exists():
        return None
    metadata = json.loads(path.read_text())
    if (
        metadata.get("protocol_commit") != PROTOCOL_COMMIT
        or metadata.get("fusion_protocol_commit") != FUSION_COMMIT
        or metadata.get("fusion_plan_sha256") != digest(root / "configs/fusion.json")
        or metadata.get("plan_sha256") != digest(root / "configs/feature_decision.json")
        or metadata.get("source_sha256") != digest(Path(__file__))
        or metadata.get("inputs") != dependencies(root)
        or set(metadata.get("files", {})) != set(PUBLIC_FILES)
    ):
        raise ValueError("Feature stopping evidence is stale")
    identity = {
        k: v
        for k, v in metadata.items()
        if k not in {"run_id", "files", "private_checkpoint_sha256"}
    }
    if content_key(identity)[:20] != metadata["run_id"]:
        raise ValueError("Feature stopping identity differs")
    result = {"metadata": metadata}
    for name, sha in metadata["files"].items():
        if digest(path.parent / name) != sha:
            raise ValueError("Feature stopping public checksum differs")
        result[name.removesuffix(".json")] = json.loads((path.parent / name).read_text())
    return result
