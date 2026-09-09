"""Score novel comments after fixed support learning; export aggregate evidence only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata

from jigsaw_rules.data import normalize
from jigsaw_rules.metrics import evaluate
from jigsaw_rules.runtime import atomic_json, digest, stage
from jigsaw_rules.uncertainty import paired_auc_comparisons
from scripts.competition_features import content_hash
from scripts.evaluate_competition_features import fit_bank
from scripts.support_adaptation import prototype_features, validate_study


def rank_scores(values):
    return (rankdata(values, method="average") - 0.5) / len(values)


def run(root: Path, cloud: Path, plan_path: Path, output: Path):
    spec = json.loads((root / "configs/support_adaptation.json").read_text())
    if digest(plan_path) != spec["plan_sha256"]:
        raise ValueError("Support study plan differs")
    plan = json.loads(plan_path.read_text())
    validate_study(plan)
    receipt = json.loads((cloud / "complete.json").read_text())
    cloud_contract = json.loads((cloud / "contract.json").read_text())
    if content_hash(cloud_contract)[:20] != receipt["run_id"]:
        raise ValueError("Cloud contract differs from completed job")
    if cloud_contract["config"] != spec:
        raise ValueError("Training and evaluation configurations differ")
    model_spec = json.loads((root / "configs/competition_features.json").read_text())
    if cloud_contract["model_spec"] != model_spec:
        raise ValueError("Training and evaluation model revisions differ")
    for name, expected in cloud_contract["source"].items():
        if Path(name).name != name or digest(root / "scripts" / name) != expected:
            raise ValueError("Adaptation source differs from the executed cloud contract")
    if len(receipt["folds"]) != len(plan["folds"]):
        raise ValueError("Completed cloud fold count differs")
    fold_hashes = {}
    for index, record in enumerate(receipt["folds"]):
        name = f"fold_{index}/representations.npz"
        fold_hashes[name] = digest(cloud / name)
        if fold_hashes[name] != record["sha256"]:
            raise ValueError("Adaptation fold checksum differs")
    original = pd.read_csv(root / "data/raw/train.csv").set_index("row_id")
    contract = {
        "config": spec,
        "cloud_run": receipt["run_id"],
        "fold_hashes": fold_hashes,
        "source": digest(Path(__file__)),
        "geometry_source": digest(root / "scripts/support_adaptation.py"),
        "screen_source": digest(root / "scripts/evaluate_competition_features.py"),
        "training_sha256": digest(root / "data/raw/train.csv"),
        "core_source": {
            path.name: digest(path) for path in sorted((root / "src/jigsaw_rules").glob("*.py"))
        },
    }

    def execute(directory):
        predictions, probabilities, screening = {}, {}, []
        row_ids = []
        for index, fold in enumerate(plan["folds"]):
            artifact = cloud / f"fold_{index}/representations.npz"
            record = receipt["folds"][index]
            if digest(artifact) != record["sha256"]:
                raise ValueError("Adaptation fold checksum differs")
            ids = [row["row_id"] for row in fold["queries"]]
            row_ids.extend(ids)
            training = pd.DataFrame(fold["training"])
            selected = training.rule.map(normalize).eq(normalize(fold["rule"])).to_numpy()
            labels = training.rule_violation.to_numpy()[selected]
            with np.load(artifact, allow_pickle=False) as data:
                if not np.array_equal(data["query_row_ids"], ids):
                    raise ValueError("Query prediction order differs")
                for model in ("frozen", "adapted"):
                    train = data["train_" + model + "_vectors"][selected]
                    query = data["query_" + model + "_vectors"]
                    margin = data["query_" + model + "_scores"][:, 1].astype(float)
                    scores = {"raw": rank_scores(margin)}
                    probabilities.setdefault(model + "_raw", []).extend(expit(margin))
                    names = [f"decision/coordinate_{j:04d}" for j in range(train.shape[1])]
                    # Query targets are not passed to any fitted readout or screen.
                    combined = np.concatenate([train, query])
                    fitting_labels = np.r_[labels, np.zeros(len(query), dtype=int)]
                    fitted, audit = fit_bank(
                        combined,
                        names,
                        fitting_labels,
                        np.arange(len(train)),
                        np.arange(len(train), len(combined)),
                        {**spec, "seed": spec["training"]["seed"]},
                    )
                    scores["screened_embedding"] = rank_scores(fitted)
                    screening.append({"fold": index, "model": model, **audit})
                    geometry, geometry_names = prototype_features(query, train, labels)
                    for name in ("centroid", "nearest", "top5"):
                        scores[name] = rank_scores(
                            geometry[:, geometry_names.index("margin/" + name)]
                        )
                    weights = spec["blend_weights"]
                    scores["geometry_blend"] = (
                        weights["answer_rank"] * scores["raw"]
                        + weights["centroid_rank"] * scores["centroid"]
                        + weights["nearest_rank"] * scores["nearest"]
                    )
                    for name, values in scores.items():
                        predictions.setdefault(model + "_" + name, []).extend(values)
        # Labels are read only after all prediction arrays have been produced.
        query = original.loc[row_ids]
        target, rules = query.rule_violation.to_numpy(), query.rule.to_numpy()
        results = []
        for name, prediction in predictions.items():
            metrics = evaluate(target, prediction, rules)
            result = {
                "model": name,
                "primary_readout": name in spec["readouts"],
                "metrics": {
                    key: metrics[key] for key in ("rule_macro_auc", "per_rule_auc", "pooled_auc")
                },
                "score_kind": "within-policy rank score; not a calibrated probability",
            }
            if name in probabilities:
                calibration = evaluate(target, probabilities[name], rules)
                result["raw_probability_diagnostics"] = {
                    key: calibration[key]
                    for key in ("log_loss", "brier", "ece_10_equal_width_bins")
                }
            results.append(result)
        contrasts = [
            ("support adaptation at fixed backbone", "adapted_raw", "frozen_raw"),
            (
                "adapted coordinates at fixed readout",
                "adapted_screened_embedding",
                "frozen_screened_embedding",
            ),
            ("prototype blend contribution", "adapted_geometry_blend", "adapted_raw"),
            ("adapted class geometry", "adapted_centroid", "frozen_centroid"),
            ("screened representation versus answer", "adapted_screened_embedding", "adapted_raw"),
        ]
        uncertainty = paired_auc_comparisons(
            target,
            predictions,
            rules,
            query.body.map(normalize).to_numpy(),
            contrasts,
            draws=spec["bootstrap_replicates"],
            seed=spec["training"]["seed"],
        )
        atomic_json(directory / "results.json", results)
        atomic_json(directory / "screening.json", screening)
        atomic_json(directory / "uncertainty.json", uncertainty)
        atomic_json(directory / "provenance.json", contract)
        atomic_json(directory / "inference.json", receipt)
        atomic_json(
            directory / "protocol.json",
            {
                "protocol": plan["protocol"],
                "rows": len(query),
                "policies": len(plan["folds"]),
                "primary_metric": "equal-weight policy-macro ROC AUC",
                "unique_candidate_columns": 2 * (len(names) + len(geometry_names)),
                "learned_readout_fits": len(screening),
                "folds": [{"rule": f["rule"], **f["audit"]} for f in plan["folds"]],
                "limitations": [
                    "Previously examined original policies; not an untouched test.",
                    "Only novel comments absent from the supplied support pool are evaluated.",
                    "Support adaptation differs from zero-shot held-out-rule CV.",
                    "One LoRA configuration and two rules cannot establish 0.92 hidden-policy AUC.",
                    "Feature contrasts hold the backbone and adaptation inputs fixed.",
                    "Rank scores preserve log-odds order; rank blends are not calibrated.",
                ],
            },
        )
        np.savez_compressed(directory / "predictions.npz", row_ids=row_ids, **predictions)

    return stage(output, content_hash(contract)[:20], execute)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path("runs/competition/adaptation/plan.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("runs/competition/adaptation/evaluation")
    )
    args = parser.parse_args()
    print(run(Path(__file__).resolve().parents[1], args.cloud, args.plan, args.output))
