"""Verify private expanded-study artifacts and replay predictions without fitting."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from jigsaw_rules.expanded import (
    aligned_predictions,
    conflict_mask,
    design,
    expanded_evidence,
    load_development,
    load_plan,
    validate_oof,
    weighted_metrics,
)
from jigsaw_rules.metrics import evaluate
from jigsaw_rules.research import VARIANTS
from jigsaw_rules.retrieval import load_plan as retrieval_plan
from jigsaw_rules.retrieval import retrieval_evidence
from jigsaw_rules.runtime import Progress, digest
from scripts.verify_research import compare_metrics


def verify_stages(directory: Path) -> int:
    count = 0
    for marker in directory.glob("*/complete.json"):
        for name, expected in json.loads(marker.read_text())["files"].items():
            path = marker.parent / name
            if not path.resolve().is_relative_to(directory.resolve()) or digest(path) != expected:
                raise ValueError("Study checkpoint checksum mismatch")
            count += 1
    return count


def verify(root: Path) -> dict:
    evidence, plan, frame = expanded_evidence(root), load_plan(root), load_development(root)
    if evidence is None:
        raise ValueError("Complete expanded research before verification")
    directory = root / "runs/expanded" / evidence["metadata"]["run_id"]
    if (
        digest(directory / "review/complete.json")
        != evidence["metadata"]["private_checkpoint_sha256"]
    ):
        raise ValueError("Private expanded review differs")
    files = verify_stages(directory)
    splits = design(frame, plan)
    if json.loads((directory / "design/splits.json").read_text()) != splits:
        raise ValueError("Saved split design differs")
    if json.loads((directory / "design/row_ids.json").read_text()) != frame.row_id.tolist():
        raise ValueError("Saved row order differs")
    oof = pd.read_csv(directory / "review/oof.csv")
    validate_oof(frame, oof, plan, splits)
    expected = set(oof[["protocol", "model"]].itertuples(index=False, name=None))
    actual = [(r["protocol"], r["model"]) for r in evidence["results"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Expanded metric record coverage differs")
    conflicts = conflict_mask(frame)
    for record in evidence["results"]:
        p = aligned_predictions(
            frame, oof[(oof.protocol == record["protocol"]) & (oof.model == record["model"])]
        )
        compare_metrics(evaluate(frame.rule_violation, p, frame.rule), record["metrics"])
        compare_metrics(weighted_metrics(frame, p), record["equal_body_policy_weight"])
        compare_metrics(
            evaluate(
                frame.loc[~conflicts, "rule_violation"],
                p[~conflicts],
                frame.loc[~conflicts, "rule"],
            ),
            record["excluding_conflicts"],
        )
    replays = 0
    # These exercise a compact and a full family composition against saved matrices.
    for protocol, records in splits.items():
        for fold, assignment in enumerate(records):
            for name in ("word_semantic_scalar", "all_transfer"):
                prefix = f"{protocol}_{fold}"
                model_path = directory / f"{prefix}_model_{name}"
                model = joblib.load(model_path / "model.joblib")
                matrix = sparse.hstack(
                    [
                        sparse.load_npz(directory / f"{prefix}_{family}/valid.npz")
                        for family in VARIANTS[name]
                    ],
                    format="csr",
                )
                part = pd.read_csv(model_path / "predictions.csv")
                predicted = model.predict_proba(matrix)[:, 1]
                if not np.allclose(predicted, part.probability, rtol=0, atol=1e-12):
                    raise ValueError(
                        "Saved model does not reproduce predictions from its feature bank"
                    )
                if part.row_id.tolist() != frame.iloc[assignment["valid"]].row_id.tolist():
                    raise ValueError("Replayed matrix row order differs")
                replays += 1
    for name, expected_sha in evidence["metadata"]["files"].items():
        if digest(directory / "review" / name) != expected_sha:
            raise ValueError("Private/public expanded report differs")
    retrieval = retrieval_evidence(root)
    retrieval_records = 0
    if retrieval is not None:
        folder = root / "runs/retrieval" / retrieval["metadata"]["run_id"]
        if (
            digest(folder / "review/complete.json")
            != retrieval["metadata"]["private_checkpoint_sha256"]
        ):
            raise ValueError("Private retrieval review differs")
        files += verify_stages(folder)
        predictions = pd.read_csv(folder / "review/oof.csv")
        expected = {
            (protocol, model) for protocol in splits for model in retrieval_plan(root)["variants"]
        }
        actual = [(r["protocol"], r["model"]) for r in retrieval["results"]]
        if set(actual) != expected or len(actual) != len(expected):
            raise ValueError("Retrieval metric record coverage differs")
        for record in retrieval["results"]:
            p = aligned_predictions(
                frame,
                predictions[
                    (predictions.protocol == record["protocol"])
                    & (predictions.model == record["model"])
                ],
            )
            compare_metrics(evaluate(frame.rule_violation, p, frame.rule), record["metrics"])
            retrieval_records += 1
        for name, expected_sha in retrieval["metadata"]["files"].items():
            if digest(folder / "review" / name) != expected_sha:
                raise ValueError("Private/public retrieval report differs")
    return {
        "expanded_run": directory.name,
        "files_verified": files,
        "expanded_metric_records": len(evidence["results"]),
        "retrieval_metric_records": retrieval_records,
        "model_prediction_replays": replays,
        "new_model_fits": 0,
        "confirmation_targets_accessed": False,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    with Progress(
        root / "logs/expanded_verification.jsonl", "private_expanded_verification"
    ) as log:
        log.emit("EXPANDED_EVIDENCE_VERIFIED", **verify(root))


if __name__ == "__main__":
    main()
