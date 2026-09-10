"""Paired, preregistered rank-blend evaluation on the existing development cohort."""

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


def blend_scores(qwen, phi, weights):
    qwen, phi = np.asarray(qwen), np.asarray(phi)
    if qwen.ndim != 1 or qwen.shape != phi.shape or not len(qwen):
        raise ValueError("Rank blend needs aligned nonempty score arrays")
    if not np.isfinite(qwen).all() or not np.isfinite(phi).all():
        raise ValueError("Rank blend scores must be finite")
    if set(weights) != {"qwen", "phi"} or any(v < 0 for v in weights.values()):
        raise ValueError("Invalid rank blend weights")
    if not np.isclose(sum(weights.values()), 1):
        raise ValueError("Rank blend weights must sum to one")
    return weights["qwen"] * rank_scores(qwen) + weights["phi"] * rank_scores(phi)


def promotion_decision(results, uncertainty, spec):
    reference = results["qwen"]["rule_macro_auc"]
    candidate = results["blend"]["rule_macro_auc"]
    delta = candidate - reference
    contrast = next(x for x in uncertainty if x["candidate"] == "blend")
    by_rule = {
        rule: value - results["qwen"]["per_rule_auc"][rule]
        for rule, value in results["blend"]["per_rule_auc"].items()
    }
    checks = {
        "positive_macro_gain": delta > 0,
        "simultaneous_interval_positive": (
            contrast["simultaneous_lower"] > spec["simultaneous_ci_lower_minimum"]
        ),
        "per_policy_regression_within_limit": (
            min(by_rule.values()) >= -spec["maximum_per_policy_regression"]
        ),
    }
    return {
        "eligible_for_kaggle_candidate": all(checks.values()),
        "checks": checks,
        "macro_delta": delta,
        "per_rule_delta": by_rule,
        "meaning": "Development gate only; never evidence that the leaderboard target was met.",
    }


def run(root, cloud, baseline, plan_path, training_path, output):
    spec = json.loads((root / "configs/complementarity.json").read_text())
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
        (root / "configs/complementary_model.json").read_text()
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
        hashes.append({"qwen": actual, "phi": record["sha256"]})
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
        predictions = {name: [] for name in ("qwen", "phi_frozen", "phi", "blend")}
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
                qwen = q["query_adapted_scores"][:, 1].astype(float)
                phi = p["query_adapted_scores"][:, 1].astype(float)
                blend = blend_scores(qwen, phi, spec["blend_weights"])
                for name, values in (
                    ("qwen", rank_scores(qwen)),
                    ("phi", rank_scores(phi)),
                    ("phi_frozen", rank_scores(p["query_frozen_scores"][:, 1])),
                    ("blend", blend),
                ):
                    predictions[name].extend(values)
                correlation = None
                if len(np.unique(qwen)) > 1 and len(np.unique(phi)) > 1:
                    correlation = float(spearmanr(qwen, phi).statistic)
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
                ("fixed blend contribution", "blend", "qwen"),
                ("Phi support adaptation", "phi", "phi_frozen"),
            ],
            draws=spec["bootstrap_replicates"],
            seed=spec["training"]["seed"],
        )
        decision = promotion_decision(results, uncertainty, spec["promotion"])
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
                "candidate": "Fixed 50/50 within-policy rank blend; no fitted weights",
                "scope": "Previously examined original-data policies; not a new holdout",
                "target_source": "Original train.csv only, loaded after predictions were fixed",
                "architecture_changes": "Native Phi template, tokenizer and fused LoRA projections",
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
