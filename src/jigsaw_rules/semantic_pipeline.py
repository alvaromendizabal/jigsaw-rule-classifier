"""Resumable frozen-encoder experiments against preserved baseline split assignments."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from filelock import FileLock
from scipy.special import expit

from jigsaw_rules.cloud import backup
from jigsaw_rules.data import audit, load_data, normalize, validate_submission
from jigsaw_rules.embeddings import QwenEncoder, encode_cached
from jigsaw_rules.metrics import evaluate
from jigsaw_rules.pipeline import csv_write
from jigsaw_rules.report import build_report
from jigsaw_rules.runtime import Progress, atomic_json, fingerprint, stage
from jigsaw_rules.semantic import FEATURE_NAMES, classifier, features, input_texts, reference_splits
from jigsaw_rules.uncertainty import paired_auc_interval


def run_semantic(root: Path, spec: dict, *, cloud: dict | None = None, encoder=None) -> Path:
    (root / "runs").mkdir(exist_ok=True, parents=True)
    with FileLock(str(root / "runs/semantic.lock"), timeout=1):
        with Progress(root / "logs/semantic.jsonl", "semantic_experiment") as log:
            return _run(root, spec, cloud=cloud, encoder=encoder, log=log)


def _run(root, spec, *, cloud, encoder, log):
    started = time.monotonic()
    train, test, sample = load_data(root / "data/raw")
    splits = reference_splits(root, train, spec["baseline_run"])
    synthetic = (root / "data/raw/SYNTHETIC.txt").exists()
    encoder = encoder or QwenEncoder(root, spec)
    if encoder.contract.get("software_test") and not synthetic:
        raise ValueError("A test encoder cannot evaluate competition data")
    config = {
        "experiment": "semantic",
        "synthetic": synthetic,
        "spec": spec,
        "encoder": encoder.contract,
        "splits": splits,
        "models": ["semantic_margin", "semantic_classifier"],
        "protocols": ["seen_rule", "heldout_rule"],
    }
    run_id, provenance = fingerprint(root / "data/raw", config)
    run_dir = root / "runs" / run_id
    run_dir.mkdir(exist_ok=True, parents=True)
    atomic_json(run_dir / "provenance.json", provenance)

    def sync():
        if cloud:
            backup(root, cloud["bucket"], cloud["region"])

    stage(run_dir, "audit", lambda dest: atomic_json(dest / "audit.json", audit(train, test)))
    stage(run_dir, "validation", lambda dest: atomic_json(dest / "splits.json", splits))
    all_texts = input_texts(pd.concat([train, test], ignore_index=True), spec)
    vectors, statistics = encode_cached(
        root, all_texts, encoder, shard_size=spec["cache_batch"], checkpoint=sync if cloud else None
    )
    matrix = features(vectors.reshape(len(train) + len(test), 5, -1))
    train_x, test_x = matrix[: len(train)], matrix[len(train) :]
    stage(run_dir, "embeddings", lambda dest: atomic_json(dest / "statistics.json", statistics))
    results, frames = [], []
    for protocol, assignments in splits.items():
        for name in config["models"]:
            chunks = []
            for fold, assignment in enumerate(assignments):

                def fit_fold(destination, assignment=assignment, name=name, fold=fold):
                    ti, vi = assignment["train"], assignment["valid"]
                    begin = time.monotonic()
                    fitted = (
                        classifier(spec).fit(train_x[ti], train.iloc[ti].rule_violation)
                        if name == "semantic_classifier"
                        else None
                    )
                    fit_seconds = time.monotonic() - begin
                    begin = time.monotonic()
                    p = (
                        fitted.predict_proba(train_x[vi])[:, 1]
                        if fitted
                        else expit(train_x[vi, 4] / spec["margin_temperature"])
                    )
                    predict_seconds = time.monotonic() - begin
                    frame = train.iloc[vi][["row_id", "rule", "rule_violation"]].copy()
                    frame["probability"], frame["fold"] = p, fold
                    csv_write(destination / "predictions.csv", frame)
                    atomic_json(
                        destination / "timing.json",
                        {"fit_seconds": fit_seconds, "predict_seconds": predict_seconds},
                    )
                    if fitted is not None:
                        # Store portable numeric state; resume never unpickles a model.
                        atomic_json(
                            destination / "classifier.json",
                            {
                                "features": FEATURE_NAMES,
                                "mean": fitted[0].mean_.tolist(),
                                "scale": fitted[0].scale_.tolist(),
                                "coef": fitted[1].coef_.tolist(),
                                "intercept": fitted[1].intercept_.tolist(),
                            },
                        )

                destination = stage(run_dir, f"{protocol}_{name}_{fold}", fit_fold)
                chunks.append(pd.read_csv(destination / "predictions.csv"))
                sync()
            frame = pd.concat(chunks, ignore_index=True)
            if len(frame) != len(train) or frame.row_id.duplicated().any():
                raise ValueError("Incomplete semantic OOF coverage")
            metric = evaluate(frame.rule_violation, frame.probability, frame.rule)
            results.append(
                {
                    "model": name,
                    "protocol": protocol,
                    "metrics": metric,
                    "run_id": run_id,
                    "data_kind": "synthetic" if synthetic else "competition",
                    "training_rows": len(train),
                    "probability_note": "Fixed temperature, not calibrated"
                    if name == "semantic_margin"
                    else "Fold-fitted logistic probabilities; not separately calibrated",
                }
            )
            frame["model"], frame["protocol"] = name, protocol
            frames.append(frame)
    oof = pd.concat(frames, ignore_index=True)

    def full_fit(destination):
        fitted = classifier(spec).fit(train_x, train.rule_violation)
        submission = pd.DataFrame(
            {"row_id": test.row_id, "rule_violation": fitted.predict_proba(test_x)[:, 1]}
        )
        validate_submission(submission, sample)
        csv_write(destination / "submission.csv", submission)
        atomic_json(
            destination / "classifier.json",
            {
                "features": FEATURE_NAMES,
                "mean": fitted[0].mean_.tolist(),
                "scale": fitted[0].scale_.tolist(),
                "coef": fitted[1].coef_.tolist(),
                "intercept": fitted[1].intercept_.tolist(),
            },
        )
        atomic_json(
            destination / "submission_metadata.json",
            {
                "synthetic": synthetic,
                "rows": len(test),
                "status": "Preview only; not a Kaggle submission or score.",
                "model": "semantic_classifier",
            },
        )

    stage(run_dir, "full_training", full_fit)

    def review(destination):
        atomic_json(destination / "results.json", results)
        csv_write(destination / "oof.csv", oof)
        build_report(destination, results, oof, synthetic=synthetic)
        baseline = json.loads(
            (root / "runs" / spec["baseline_run"] / "review/results.json").read_text()
        )
        base_oof = pd.read_csv(root / "runs" / spec["baseline_run"] / "review/oof.csv")
        intervals = []
        for record in results:
            current = (
                oof[(oof.model == record["model"]) & (oof.protocol == record["protocol"])]
                .set_index("row_id")
                .loc[train.row_id]
            )
            reference = (
                base_oof[
                    (base_oof.model == "rule_examples") & (base_oof.protocol == record["protocol"])
                ]
                .set_index("row_id")
                .loc[train.row_id]
            )
            intervals.append(
                {
                    "model": record["model"],
                    "protocol": record["protocol"],
                    **paired_auc_interval(
                        train.rule_violation,
                        reference.probability,
                        current.probability,
                        train.rule,
                        train.body.map(normalize),
                        draws=20 if synthetic else 500,
                        seed=spec["seed"],
                    ),
                }
            )
        atomic_json(destination / "uncertainty.json", intervals)
        atomic_json(
            destination / "comparison.json",
            {
                "baseline_run": spec["baseline_run"],
                "baseline": baseline,
                "semantic": results,
                "encoder": encoder.contract,
                "statistics": statistics,
                "wall_seconds_before_report": time.monotonic() - started,
            },
        )
        csv_write(
            destination / "error_review.csv",
            oof.assign(absolute_error=lambda f: abs(f.rule_violation - f.probability)).sort_values(
                "absolute_error", ascending=False
            ),
        )

    stage(run_dir, "review", review)
    stage(
        run_dir,
        "performance",
        lambda dest: atomic_json(
            dest / "timing.json",
            {
                "first_completion_wall_seconds": time.monotonic() - started,
                "encoder": statistics,
                "note": (
                    "First successful invocation, including downloads if needed. "
                    "Earlier failed invocations are in logs."
                ),
            },
        ),
    )
    atomic_json(
        run_dir / "status.json",
        {"status": "completed", "synthetic": synthetic, "experiment": "semantic"},
    )
    sync()
    log.emit(
        "SEMANTIC_COMPLETED",
        run_id=run_id,
        data_kind="synthetic" if synthetic else "competition",
        wall_seconds=round(time.monotonic() - started, 3),
    )
    print(f"REPORT {run_dir / 'review/report.html'}", flush=True)
    return run_dir
