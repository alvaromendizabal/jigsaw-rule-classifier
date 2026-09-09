"""Fixed feature-family ablations on original competition development rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from jigsaw_rules.data import EXAMPLES, normalize, validate_frame
from jigsaw_rules.metrics import evaluate
from jigsaw_rules.model import LexicalClassifier
from jigsaw_rules.representations import TrainingScreen
from jigsaw_rules.runtime import Progress, atomic_json, digest, stage
from jigsaw_rules.splits import make_splits
from jigsaw_rules.uncertainty import paired_auc_comparisons
from scripts.competition_features import PROMPTS, content_hash, feature_banks, support_pairs


def fit_bank(matrix, names, target, train_indices, valid_indices, spec):
    """Validation statistics and targets cannot enter the screen or scaler."""
    screen = TrainingScreen(spec["screen_maximum"])
    train = screen.fit(matrix[train_indices], target[train_indices], names).transform(
        matrix[train_indices], names
    )
    valid = screen.transform(matrix[valid_indices], names)
    if not train.shape[1]:
        raise ValueError("Screen retained no usable representation features")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=spec["classifier_c"], solver="liblinear", max_iter=2000, random_state=spec["seed"]
        ),
    )
    model.fit(train, target[train_indices])
    selected = [names[i] for i in screen.indices_]
    return model.predict_proba(valid)[:, 1], {
        **screen.audit_,
        "selected_features": selected,
        "selected_families": pd.Series([n.split("/")[0] for n in selected])
        .value_counts()
        .to_dict(),
    }


def run(root: Path, representations: Path, output: Path):
    spec = json.loads((root / "configs/competition_features.json").read_text())
    frame = pd.read_csv(root / "data/raw/train.csv")
    validate_frame(frame, train=True)
    inference = json.loads(representations.with_name("complete.json").read_text())
    if digest(representations) != inference["representations_sha256"]:
        raise ValueError("Representation checksum differs from completed inference")
    target_free_hash = hashlib.sha256(
        frame.drop(columns="rule_violation").to_csv(index=False).encode()
    ).hexdigest()
    if target_free_hash != spec["input_sha256"] or target_free_hash != inference["input_sha256"]:
        raise ValueError("Inference inputs and development inputs differ")
    with np.load(representations, allow_pickle=False) as stored:
        if not np.array_equal(stored["row_ids"], frame.row_id.to_numpy()):
            raise ValueError("Representation row order differs from development data")
        scores, vectors = stored["scores"], stored["vectors"]
    # The float32 extraction probability saturates at 1 around log odds 17.
    # Reconstruct in float64 from the saved margins; never discard their ranking.
    scores = scores.astype(float)
    scores[:, :, 0] = expit(scores[:, :, 1])
    contract = {
        "training_sha256": digest(root / "data/raw/train.csv"),
        "representations_sha256": digest(representations),
        "config_sha256": digest(root / "configs/competition_features.json"),
        "evaluation_source_sha256": digest(Path(__file__)),
        "feature_source_sha256": digest(Path(__file__).with_name("competition_features.py")),
        "inference_run_id": inference["run_id"],
        "core_source": {
            path.name: digest(path) for path in sorted((root / "src/jigsaw_rules").glob("*.py"))
        },
    }
    key = content_hash(contract)[:20]

    def execute(directory):
        banks = feature_banks(scores, vectors)
        full_names = list(banks)
        # all_likelihood duplicates the three scalar banks: include it only once.
        full_names = [n for n in full_names if n not in [p + "_likelihood" for p in PROMPTS]]
        banks["all_features"] = (
            np.concatenate([banks[n][0] for n in full_names], axis=1),
            [name for n in full_names for name in banks[n][1]],
        )
        results, screening, uncertainty, splits, augmentation = [], [], [], [], []
        target, rules = frame.rule_violation.to_numpy(), frame.rule.to_numpy()
        groups = frame.body.map(normalize).to_numpy()
        self_overlap = np.column_stack(
            [frame[column].map(normalize).to_numpy() == groups for column in EXAMPLES]
        ).any(axis=1)
        with Progress(directory / "events.jsonl", "competition_feature_ablation") as log:
            for protocol in ("seen_rule", "heldout_rule"):
                predictions = {"lexical_reference": np.full(len(frame), np.nan)}
                for i, prompt in enumerate(PROMPTS):
                    predictions[prompt + "_raw"] = scores[:, i, 0].astype(float)
                for name in banks:
                    predictions[name] = np.full(len(frame), np.nan)
                for fold, (ti, vi, purged) in enumerate(make_splits(frame, protocol, 3, 2025)):
                    split_record = {
                        "protocol": protocol,
                        "fold": fold,
                        "training_rows": len(ti),
                        "validation_rows": len(vi),
                        "purged_rows": purged,
                        "split_sha256": content_hash({"train": ti.tolist(), "valid": vi.tolist()}),
                    }
                    splits.append(split_record)
                    reference = LexicalClassifier().fit(frame.iloc[ti])
                    predictions["lexical_reference"][vi] = reference.predict(frame.iloc[vi])
                    _, audit = support_pairs(
                        frame.iloc[ti],
                        frame.iloc[vi].drop(columns="rule_violation"),
                        forbidden=frame.iloc[vi].body,
                    )
                    augmentation.append({**split_record, **audit})
                    for name, (matrix, names) in banks.items():
                        prediction, audit = fit_bank(matrix, names, target, ti, vi, spec)
                        predictions[name][vi] = prediction
                        screening.append(
                            {"protocol": protocol, "fold": fold, "bank": name, **audit}
                        )
                        log.emit(
                            "bank_complete",
                            protocol=protocol,
                            fold=fold,
                            bank=name,
                            retained=audit["retained"],
                        )
                if any(not np.isfinite(p).all() for p in predictions.values()):
                    raise ValueError("OOF predictions are incomplete")
                for name, probability in predictions.items():
                    results.append(
                        {
                            "protocol": protocol,
                            "model": name,
                            "metrics": evaluate(target, probability, rules),
                            "without_self_support_metrics": evaluate(
                                target[~self_overlap],
                                probability[~self_overlap],
                                rules[~self_overlap],
                            ),
                        }
                    )
                contrasts = [
                    ("4B rule score versus lexical", "rule_raw", "lexical_reference"),
                    ("supplied context at fixed model", "rule_support_raw", "rule_raw"),
                    (
                        "joint versus rule representation",
                        "rule_support_embedding",
                        "rule_embedding",
                    ),
                    ("interaction bank contribution", "all_features", "rule_support_embedding"),
                    (
                        "learned representation versus direct score",
                        "all_features",
                        "rule_support_raw",
                    ),
                ]
                uncertainty.extend(
                    {"protocol": protocol, **item}
                    for item in paired_auc_comparisons(
                        target,
                        predictions,
                        rules,
                        groups,
                        contrasts,
                        draws=spec["bootstrap_replicates"],
                        seed=spec["seed"],
                    )
                )
                np.savez_compressed(
                    directory / (protocol + "_predictions.npz"),
                    row_ids=frame.row_id.to_numpy(),
                    **predictions,
                )
            unique_names = {name for _, names in banks.values() for name in names}
            atomic_json(directory / "results.json", results)
            atomic_json(directory / "screening.json", screening)
            atomic_json(directory / "uncertainty.json", uncertainty)
            atomic_json(
                directory / "protocol.json",
                {
                    "scope": "Original-data development; not a new holdout or Kaggle score",
                    "primary_metric": "equal-weight policy-macro ROC AUC",
                    "secondary_metric": "pooled ROC AUC and probability diagnostics",
                    "unique_candidates": len(unique_names),
                    "fixed_classifier_fits": len(screening),
                    "splits": splits,
                    "self_support_overlap_rows": int(self_overlap.sum()),
                    "limitations": [
                        "Two development policies only; four hidden new policies untested here.",
                        "Purging reduces training size; absolute CV is not an LB forecast.",
                        "Old 0.6B comparisons change the model, prompt, and field budgets.",
                        "Feature contrasts hold the 4B model and classifier settings fixed.",
                        "No tuning on the already-consumed post-competition protected evaluation.",
                    ],
                },
            )
            atomic_json(directory / "augmentation.json", augmentation)
            atomic_json(directory / "provenance.json", contract)
            atomic_json(directory / "inference.json", inference)

    return stage(output, key, execute)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/competition/evaluation"))
    args = parser.parse_args()
    result = run(Path(__file__).resolve().parents[1], args.representations, args.output)
    print("Verified feature study:", result)


if __name__ == "__main__":
    main()
