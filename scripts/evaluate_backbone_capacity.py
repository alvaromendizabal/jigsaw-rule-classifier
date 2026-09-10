"""Fixed two-candidate backbone comparison on the existing development cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from jigsaw_rules.data import normalize
from jigsaw_rules.metrics import evaluate
from jigsaw_rules.runtime import atomic_json, digest, stage
from jigsaw_rules.uncertainty import paired_auc_comparisons
from scripts.competition_features import content_hash
from scripts.complementary_worker import checked_fold, verified_plan
from scripts.evaluate_support_adaptation import rank_scores


def blend_scores(qwen4b, qwen8b, weights):
    qwen4b, qwen8b = np.asarray(qwen4b), np.asarray(qwen8b)
    if qwen4b.ndim != 1 or qwen4b.shape != qwen8b.shape or not len(qwen4b):
        raise ValueError("Rank blend needs aligned nonempty score arrays")
    if not np.isfinite(qwen4b).all() or not np.isfinite(qwen8b).all():
        raise ValueError("Rank blend scores must be finite")
    if set(weights) != {"qwen4b", "qwen8b"} or any(v < 0 for v in weights.values()):
        raise ValueError("Invalid rank blend weights")
    if not np.isclose(sum(weights.values()), 1):
        raise ValueError("Rank blend weights must sum to one")
    return weights["qwen4b"] * rank_scores(qwen4b) + weights["qwen8b"] * rank_scores(qwen8b)


def promotion_decision(results, uncertainty, spec, selection):
    """Select only from two frozen candidates with simultaneous positive-gain evidence."""
    reference = results["qwen4b"]
    decisions = {}
    for name in selection["candidates"]:
        candidate = results[name]
        contrast = next(x for x in uncertainty if x["candidate"] == name)
        deltas = {
            rule: value - reference["per_rule_auc"][rule]
            for rule, value in candidate["per_rule_auc"].items()
        }
        checks = {
            "positive_macro_gain": candidate["rule_macro_auc"] > reference["rule_macro_auc"],
            "simultaneous_interval_positive": contrast["simultaneous_lower"]
            > spec["simultaneous_ci_lower_minimum"],
            "per_policy_regression_within_limit": min(deltas.values())
            >= -spec["maximum_per_policy_regression"],
        }
        decisions[name] = {
            "eligible": all(checks.values()),
            "checks": checks,
            "macro_delta": candidate["rule_macro_auc"] - reference["rule_macro_auc"],
            "per_rule_delta": deltas,
        }
    eligible = [name for name in selection["priority"] if decisions[name]["eligible"]]
    chosen = max(eligible, key=lambda name: results[name]["rule_macro_auc"]) if eligible else None
    return {
        "eligible_for_kaggle_candidate": chosen is not None,
        "selected": chosen,
        "candidates": decisions,
        "selection": selection,
        "meaning": "Development eligibility only; offline and hidden-score gates remain",
    }


def run(root, cloud, baseline, plan_path, training_path, output):
    spec = json.loads((root / "configs/backbone_capacity.json").read_text())
    plan = verified_plan(plan_path, spec)
    if digest(training_path) != spec["original_training_sha256"]:
        raise ValueError("Original training data checksum differs")
    frozen = spec["baseline"]
    for name, expected in (
        ("contract.json", frozen["contract_sha256"]),
        ("complete.json", frozen["receipt_sha256"]),
    ):
        if digest(baseline / name) != expected:
            raise ValueError("Pinned Qwen baseline identity differs")
    receipt = json.loads((cloud / "complete.json").read_text())
    cloud_contract = json.loads((cloud / "contract.json").read_text())
    if content_hash(cloud_contract)[:20] != receipt["run_id"]:
        raise ValueError("Completed cloud contract differs")
    if cloud_contract["config"] != spec or cloud_contract["model_spec"] != json.loads(
        (root / "configs/qwen3_8b.json").read_text()
    ):
        raise ValueError("Executed configuration differs from preregistered study")
    for name, expected in cloud_contract["source"].items():
        path = root / name
        if root.resolve() not in path.resolve().parents or digest(path) != expected:
            raise ValueError("Executed source checksum differs")
    if len(receipt["folds"]) != len(plan["folds"]):
        raise ValueError("Incomplete study folds")
    hashes = []
    for index, fold in enumerate(plan["folds"]):
        record = checked_fold(cloud / f"fold_{index}", index, fold)
        if record != receipt["folds"][index]:
            raise ValueError("Fold receipt differs from study receipt")
        actual = digest(baseline / f"fold_{index}/representations.npz")
        if actual != frozen["fold_sha256"][index]:
            raise ValueError("Pinned Qwen predictions checksum differs")
        hashes.append({"qwen4b": actual, "qwen8b": record["sha256"]})
    contract = {
        "spec": spec,
        "cloud_run": receipt["run_id"],
        "fold_hashes": hashes,
        "source": digest(Path(__file__)),
        "core_source": {
            p.name: digest(p) for p in sorted((root / "src/jigsaw_rules").glob("*.py"))
        },
    }

    def execute(directory):
        predictions = {name: [] for name in ("qwen4b", "qwen8b_frozen", "qwen8b", "blend")}
        ids, correlations = [], []
        for index, fold in enumerate(plan["folds"]):
            query_ids = [r["row_id"] for r in fold["queries"]]
            ids.extend(query_ids)
            with (
                np.load(baseline / f"fold_{index}/representations.npz", allow_pickle=False) as q,
                np.load(cloud / f"fold_{index}/representations.npz", allow_pickle=False) as p,
            ):
                if not np.array_equal(q["query_row_ids"], query_ids):
                    raise ValueError("Qwen query IDs/order differ")
                qwen4b = q["query_adapted_scores"][:, 1].astype(float)
                qwen8b = p["query_adapted_scores"][:, 1].astype(float)
                blend = blend_scores(qwen4b, qwen8b, spec["blend_weights"])
                for name, values in (
                    ("qwen4b", rank_scores(qwen4b)),
                    ("qwen8b", rank_scores(qwen8b)),
                    ("qwen8b_frozen", rank_scores(p["query_frozen_scores"][:, 1])),
                    ("blend", blend),
                ):
                    predictions[name].extend(values)
                correlation = None
                if len(np.unique(qwen4b)) > 1 and len(np.unique(qwen8b)) > 1:
                    correlation = float(spearmanr(qwen4b, qwen8b).statistic)
                correlations.append({"rule": fold["rule"], "spearman": correlation})
        # All four prediction arrays are fixed before any query labels are loaded.
        original = pd.read_csv(training_path).set_index("row_id")
        query = original.loc[ids]
        expected_rules = [r["rule"] for f in plan["folds"] for r in f["queries"]]
        if query.rule.tolist() != expected_rules:
            raise ValueError("Evaluation query policies differ")
        results = {}
        for name, values in predictions.items():
            metrics = evaluate(query.rule_violation, values, query.rule)
            results[name] = {
                key: metrics[key] for key in ("rule_macro_auc", "per_rule_auc", "pooled_auc")
            }
        uncertainty = paired_auc_comparisons(
            query.rule_violation,
            predictions,
            query.rule,
            query.body.map(normalize),
            [
                ("larger backbone contribution", "qwen8b", "qwen4b"),
                ("fixed 4B/8B blend contribution", "blend", "qwen4b"),
            ],
            draws=spec["bootstrap_replicates"],
            seed=spec["training"]["seed"],
        )
        decision = promotion_decision(results, uncertainty, spec["promotion"], spec["selection"])
        atomic_json(directory / "results.json", results)
        atomic_json(directory / "uncertainty.json", uncertainty)
        atomic_json(directory / "decision.json", decision)
        atomic_json(directory / "correlations.json", correlations)
        atomic_json(directory / "provenance.json", contract)
        atomic_json(directory / "inference.json", receipt)
        atomic_json(
            directory / "protocol.json",
            {
                "rows": len(ids),
                "policies": len(plan["folds"]),
                "primary_metric": "Equal-weight policy-macro ROC AUC",
                "candidate": "Adapted Qwen3-8B or fixed 50/50 4B/8B rank blend; no fitted weights",
                "secondary_control": "Frozen 8B is descriptive, excluded from candidate selection",
                "scope": "Previously examined original-data policies; not a new holdout",
                "target_source": "Original train.csv only, loaded after predictions were fixed",
                "architecture_changes": "8B, different vintage, non-thinking chat, micro-batch 2",
                "causal_limit": "Size is not isolated from model vintage and native formatting",
                "limits": "Two policies; conditional uncertainty does not establish transfer",
            },
        )
        np.savez_compressed(directory / "predictions.npz", row_ids=ids, **predictions)

    return stage(output, content_hash(contract)[:20], execute)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("cloud", "baseline", "plan", "training", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(
        run(
            Path(__file__).resolve().parents[1],
            args.cloud,
            args.baseline,
            args.plan,
            args.training,
            args.output,
        )
    )
