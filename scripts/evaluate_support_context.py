"""Evaluate one frozen prompt candidate; preserve private rows and public aggregates."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from jigsaw_rules.data import normalize
from jigsaw_rules.metrics import evaluate
from jigsaw_rules.runtime import atomic_json, digest, stage
from jigsaw_rules.uncertainty import paired_auc_comparisons
from scripts.competition_features import content_hash
from scripts.evaluate_support_adaptation import rank_scores
from scripts.support_adaptation import validate_study
from scripts.support_context import SOURCES


def run(root, cloud, plan_path, baseline_archive, training, output):
    spec = json.loads((root / "configs/support_context.json").read_text())
    if (
        digest(plan_path) != spec["plan_sha256"]
        or digest(training) != spec["original_training_sha256"]
    ):
        raise ValueError("Original plan/training identity differs")
    if digest(baseline_archive) != spec["cpu_baseline_archive_sha256"]:
        raise ValueError("Saved baseline evaluation archive differs")
    plan = json.loads(plan_path.read_text())
    validate_study(plan)
    receipt = json.loads((cloud / "complete.json").read_text())
    executed = json.loads((cloud / "contract.json").read_text())
    if content_hash(executed)[:20] != receipt["run_id"] or executed["config"] != spec:
        raise ValueError("Executed candidate contract differs")
    if executed["model_spec"] != json.loads(
        (root / "configs/competition_features.json").read_text()
    ):
        raise ValueError("Executed model identity differs")
    if executed["source"] != {n: digest(root / n) for n in SOURCES}:
        raise ValueError("Executed source differs")
    if receipt["training_performed"] or len(receipt["folds"]) != len(plan["folds"]):
        raise ValueError("Unexpected training or incomplete folds")
    contract = {
        "run_id": receipt["run_id"],
        "receipt_sha256": digest(cloud / "complete.json"),
        "spec": spec,
        "evaluation_source": digest(Path(__file__)),
        "metric_source": digest(root / "src/jigsaw_rules/metrics.py"),
        "uncertainty_source": digest(root / "src/jigsaw_rules/uncertainty.py"),
    }

    def execute(directory):
        with tarfile.open(baseline_archive) as archive:
            # Read only the pre-existing original-data prediction array.
            payload = archive.extractfile("predictions.npz").read()
        with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
            saved_ids, baseline = saved["row_ids"], saved["qwen4b"].astype(float)
        ids, candidate = [], []
        for index, fold in enumerate(plan["folds"]):
            record = json.loads((cloud / f"fold_{index}/complete.json").read_text())
            path = cloud / f"fold_{index}/predictions.npz"
            if record != receipt["folds"][index] or digest(path) != record["sha256"]:
                raise ValueError("Completed fold integrity differs")
            if record["optimizer_steps"] or not record["batch_replay_verified"]:
                raise ValueError("Frozen-inference/replay contract failed")
            if record["parity_max_abs_margin"] > spec["parity_max_abs_margin"]:
                raise ValueError("Retained-model parity gate failed")
            order = [r["row_id"] for r in fold["queries"]]
            with np.load(path, allow_pickle=False) as values:
                if (
                    not np.array_equal(values["row_ids"], order)
                    or not np.isfinite(values["margins"]).all()
                ):
                    raise ValueError("Candidate query order or finite-score check failed")
                candidate.extend(rank_scores(values["margins"]))
            ids.extend(order)
        if not np.array_equal(ids, saved_ids):
            raise ValueError("Candidate/baseline row identities differ")
        # Query labels enter only after the candidate and reference are frozen.
        query = pd.read_csv(training).set_index("row_id").loc[ids]
        expected = [r["rule"] for f in plan["folds"] for r in f["queries"]]
        if query.rule.tolist() != expected:
            raise ValueError("Original query policies differ")
        predictions = {"qwen4b": baseline, "support_context": candidate}
        results = {}
        for name, scores in predictions.items():
            metrics = evaluate(query.rule_violation, scores, query.rule)
            results[name] = {
                k: metrics[k] for k in ("rule_macro_auc", "per_rule_auc", "pooled_auc")
            }
        uncertainty = paired_auc_comparisons(
            query.rule_violation,
            predictions,
            query.rule,
            query.body.map(normalize),
            [("support-context contribution", "support_context", "qwen4b")],
            draws=spec["bootstrap_replicates"],
            seed=spec["seed"],
        )
        baseline_metrics, context_metrics = results["qwen4b"], results["support_context"]
        delta = context_metrics["rule_macro_auc"] - baseline_metrics["rule_macro_auc"]
        per_policy = {
            k: v - baseline_metrics["per_rule_auc"][k]
            for k, v in context_metrics["per_rule_auc"].items()
        }
        checks = {
            "positive_macro_gain": delta > 0,
            "positive_interval": uncertainty[0]["simultaneous_lower"] > 0,
            "no_policy_regression": min(per_policy.values()) >= 0,
        }
        decision = {
            "eligible_for_offline_candidate": all(checks.values()),
            "checks": checks,
            "macro_delta": delta,
            "per_rule_delta": per_policy,
            "meaning": "Development evidence only; no automatic submission or follow-on run",
        }
        for name, value in {
            "results": results,
            "uncertainty": uncertainty,
            "decision": decision,
            "provenance": contract,
            "inference": receipt,
        }.items():
            atomic_json(directory / f"{name}.json", value)
        np.savez_compressed(directory / "predictions.npz", row_ids=ids, **predictions)

    return stage(output, content_hash(contract)[:20], execute)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cloud", "plan", "baseline-archive", "training", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(
        run(
            Path(__file__).resolve().parents[1],
            args.cloud,
            args.plan,
            args.baseline_archive,
            args.training,
            args.output,
        )
    )
