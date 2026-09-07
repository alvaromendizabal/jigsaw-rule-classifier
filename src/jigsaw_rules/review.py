"""Read verified completed experiments without training or changing their artifacts."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path, PurePosixPath

import pandas as pd

from jigsaw_rules.metrics import evaluate
from jigsaw_rules.report import build_report
from jigsaw_rules.runtime import Progress, atomic_json, digest


def verified_stage(run_dir: Path, name: str) -> dict:
    directory = run_dir / name
    marker = json.loads((directory / "complete.json").read_text())
    if not marker.get("files"):
        raise ValueError(f"Empty checkpoint: {name}")
    for filename, expected in marker["files"].items():
        relative = PurePosixPath(filename)
        path = directory / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in filename
            or not path.resolve().is_relative_to(directory.resolve())
        ):
            raise ValueError("Unsafe checkpoint path")
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {name}/{filename}")
    return marker


def select_run(root: Path, run_id: str | None, *, allow_synthetic: bool = False) -> Path:
    if run_id is not None and not re.fullmatch(r"[0-9a-f]{20}", run_id):
        raise ValueError("Run ID must be the 20-character experiment fingerprint")
    candidates = [root / "runs" / run_id] if run_id else sorted((root / "runs").glob("*"))
    available = []
    for path in candidates:
        if not (path / "status.json").is_file():
            continue
        status = json.loads((path / "status.json").read_text())
        if status.get("status") != "completed":
            continue
        if status.get("synthetic") is not False and not allow_synthetic:
            continue
        marker = verified_stage(path, "review")
        available.append((marker["finished_at"], path))
    if not available:
        kind = "competition-data" if not allow_synthetic else "requested"
        raise FileNotFoundError(f"No completed {kind} run found. Restore saved runs first.")
    return max(available)[1]


def _same_metrics(expected, actual) -> bool:
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(
            _same_metrics(value, actual[key]) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(expected) == len(actual) and all(
            _same_metrics(a, b) for a, b in zip(expected, actual, strict=True)
        )
    if isinstance(expected, float):
        return math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-12)
    return expected == actual


def review_run(root: Path, run_id: str | None = None, *, allow_synthetic: bool = False) -> dict:
    """Export clearly labeled evidence; original runs and checkpoints remain untouched."""
    with Progress(root / "logs/review.jsonl", "review_saved_run") as log:
        run_dir = select_run(root, run_id, allow_synthetic=allow_synthetic)
        provenance = json.loads((run_dir / "provenance.json").read_text())
        status = json.loads((run_dir / "status.json").read_text())
        synthetic = provenance["config"]["synthetic"]
        if type(synthetic) is not bool or status.get("synthetic") is not synthetic:
            raise ValueError("Run data identity is missing or inconsistent")
        if synthetic and not allow_synthetic:
            raise ValueError("Synthetic results require explicit --allow-synthetic")
        verified_stage(run_dir, "audit")
        audit = json.loads((run_dir / "audit/audit.json").read_text())
        results = json.loads((run_dir / "review/results.json").read_text())
        predictions = pd.read_csv(run_dir / "review/oof.csv")
        seen = set()
        for result in results:
            key = (result["model"], result["protocol"])
            if key in seen:
                raise ValueError("Duplicate experiment result")
            seen.add(key)
            frame = predictions[(predictions.model == key[0]) & (predictions.protocol == key[1])]
            if len(frame) != audit["train_rows"] or frame.row_id.duplicated().any():
                raise ValueError("OOF row coverage does not match the data audit")
            actual = evaluate(frame.rule_violation, frame.probability, frame.rule)
            if not _same_metrics(result["metrics"], actual):
                raise ValueError("Reported metrics do not match saved OOF predictions")
        expected = {
            (model, protocol)
            for model in provenance["config"]["models"]
            for protocol in provenance["config"]["protocols"]
        }
        if seen != expected or len(predictions) != len(expected) * audit["train_rows"]:
            raise ValueError("Experiment coverage does not match provenance")
        evidence = {
            "schema": 1,
            "run_id": run_dir.name,
            "data_kind": "synthetic" if synthetic else "competition",
            "evaluation": "local cross-validation; not a Kaggle leaderboard score",
            "training_rows": audit["train_rows"],
            "training_sha256": provenance["data"]["train.csv"],
            "original_results_sha256": digest(run_dir / "review/results.json"),
            "results": results,
        }
        output = root / "reports/private"
        atomic_json(output / "results.json", evidence)
        build_report(output, results, predictions, synthetic=synthetic)
        log.emit("REVIEW_VERIFIED", run_id=run_dir.name, data_kind=evidence["data_kind"])
        for result in results:
            print(
                f"{result['protocol']:12s} {result['model']:14s} "
                f"rule_macro_auc={result['metrics']['rule_macro_auc']:.6f}",
                flush=True,
            )
        print(f"REPORT {output / 'report.html'}", flush=True)
        print(f"RESULTS {output / 'results.json'}", flush=True)
        return evidence
