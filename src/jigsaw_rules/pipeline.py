"""Resumable evaluation and independent full-training submission generation."""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from filelock import FileLock

from jigsaw_rules.cloud import backup
from jigsaw_rules.data import audit, load_data, validate_submission
from jigsaw_rules.metrics import evaluate
from jigsaw_rules.model import LexicalClassifier
from jigsaw_rules.report import build_report
from jigsaw_rules.runtime import atomic_bytes, atomic_json, fingerprint, stage
from jigsaw_rules.splits import make_splits


def csv_write(path: Path, frame: pd.DataFrame) -> None:
    atomic_bytes(path, frame.to_csv(index=False).encode())


def run_baseline(
    root: Path,
    *,
    data_dir: Path | None = None,
    folds: int = 3,
    seed: int = 2025,
    cloud: dict | None = None,
) -> Path:
    (root / "runs").mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "runs/pipeline.lock"), timeout=1):
        return _run_baseline(root, data_dir=data_dir, folds=folds, seed=seed, cloud=cloud)


def _run_baseline(root: Path, *, data_dir, folds, seed, cloud) -> Path:
    data_dir = data_dir or root / "data/raw"
    train, test, sample = load_data(data_dir)
    is_synthetic = (data_dir / "SYNTHETIC.txt").exists()
    config = {
        "seed": seed,
        "folds": folds,
        "synthetic": is_synthetic,
        "models": ["comment_only", "rule_examples"],
        "protocols": ["seen_rule", "heldout_rule"],
        "submission_model": "rule_examples",
    }
    run_id, provenance = fingerprint(data_dir, config)
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "provenance.json", provenance)

    def sync():
        if cloud:
            backup(root, cloud["bucket"], cloud["region"])

    stage(run_dir, "audit", lambda dest: atomic_json(dest / "audit.json", audit(train, test)))
    sync()
    results, all_oof = [], []
    for protocol in config["protocols"]:
        splits = make_splits(train, protocol, folds, seed)
        for name in config["models"]:
            chunks = []
            for fold, (train_idx, valid_idx, purged) in enumerate(splits):

                def fit_fold(
                    dest, ti=train_idx, vi=valid_idx, removed=purged, model_name=name, fold_id=fold
                ):
                    model = LexicalClassifier(context=model_name == "rule_examples", seed=seed)
                    started = time.monotonic()
                    model.fit(train.iloc[ti])
                    fit_seconds = time.monotonic() - started
                    started = time.monotonic()
                    probabilities = model.predict(train.iloc[vi])
                    predict_seconds = time.monotonic() - started
                    frame = train.iloc[vi][["row_id", "rule", "rule_violation"]].copy()
                    frame["probability"], frame["fold"] = probabilities, fold_id
                    csv_write(dest / "predictions.csv", frame)
                    atomic_json(
                        dest / "metrics.json",
                        {
                            **evaluate(frame.rule_violation, probabilities, frame.rule),
                            "fit_seconds": fit_seconds,
                            "predict_seconds": predict_seconds,
                        },
                    )
                    atomic_json(
                        dest / "split.json",
                        {
                            "train_row_ids": train.iloc[ti].row_id.tolist(),
                            "valid_row_ids": train.iloc[vi].row_id.tolist(),
                            "purged_training_rows": removed,
                            "train_rules": sorted(train.iloc[ti].rule.unique().tolist()),
                            "valid_rules": sorted(train.iloc[vi].rule.unique().tolist()),
                        },
                    )

                dest = stage(run_dir, f"{protocol}_{name}_{fold}", fit_fold)
                sync()
                chunks.append(pd.read_csv(dest / "predictions.csv"))
            oof = pd.concat(chunks, ignore_index=True)
            if (
                len(oof) != len(train)
                or set(oof.row_id) != set(train.row_id)
                or oof.row_id.duplicated().any()
            ):
                raise ValueError("Every training row must have exactly one OOF prediction")
            metric = evaluate(oof.rule_violation, oof.probability, oof.rule)
            results.append(
                {
                    "run_id": run_id,
                    "data_kind": "synthetic" if is_synthetic else "competition",
                    "training_rows": len(train),
                    "model": name,
                    "protocol": protocol,
                    "metrics": metric,
                }
            )
            oof["model"], oof["protocol"] = name, protocol
            all_oof.append(oof)
    combined = pd.concat(all_oof, ignore_index=True)

    def full_fit(dest):
        model = LexicalClassifier(context=True, seed=seed).fit(train)
        predictions = pd.DataFrame({"row_id": test.row_id, "rule_violation": model.predict(test)})
        validate_submission(predictions, sample)
        csv_write(dest / "submission.csv", predictions)
        csv_write(dest / "coefficients.csv", model.coefficients())
        joblib.dump(model, dest / "model.joblib")
        atomic_json(
            dest / "submission_metadata.json",
            {
                "synthetic": is_synthetic,
                "rows": len(test),
                "model": "rule_examples",
                "selection": "Predeclared reference model; not selected using hidden test labels.",
                "status": "Preview only. Kaggle notebook regenerates predictions on hidden test.",
            },
        )

    stage(run_dir, "full_training", full_fit)
    sync()

    def report(dest):
        atomic_json(dest / "results.json", results)
        csv_write(dest / "oof.csv", combined)
        build_report(dest, results, combined, synthetic=is_synthetic)
        errors = combined.assign(absolute_error=lambda x: np.abs(x.rule_violation - x.probability))
        csv_write(dest / "error_review.csv", errors.sort_values("absolute_error", ascending=False))

    stage(run_dir, "review", report)
    atomic_json(run_dir / "status.json", {"status": "completed", "synthetic": is_synthetic})
    sync()
    print(f"RUN_COMPLETED {run_dir}", flush=True)
    print(f"REPORT {run_dir / 'review/report.html'}", flush=True)
    return run_dir
