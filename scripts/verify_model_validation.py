"""Replay saved nested predictions and calibration without fitting classifiers."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from jigsaw_rules.calibration import MonotoneSigmoid, calibration_decision
from jigsaw_rules.diagnostics import support_scores
from jigsaw_rules.expanded import aligned_predictions, load_development
from jigsaw_rules.final_model import FeaturePipeline, RoutedClassifier, verify_stage
from jigsaw_rules.model_validation import load_plan, parent_checkpoints, validation_evidence
from jigsaw_rules.research import CORE, cached_vectors
from jigsaw_rules.runtime import Progress, atomic_json, digest

ROOT = Path(__file__).resolve().parents[1]


def verify(root: Path = ROOT) -> dict:
    evidence, plan = validation_evidence(root), load_plan(root)
    if evidence is None:
        raise ValueError("Model validation report is required")
    metadata = evidence["metadata"]
    directory = root / "runs/model_validation" / metadata["run_id"]
    if parent_checkpoints(root, plan) != metadata["parent_checkpoints"]:
        raise ValueError("Saved research parents differ from model-validation lineage")
    for stage_name, field in [
        ("review", "private_checkpoint_sha256"),
        ("final", "final_checkpoint_sha256"),
    ]:
        if digest(directory / stage_name / "complete.json") != metadata[field]:
            raise ValueError("Saved validation checkpoint differs from the public evidence")
        verify_stage(directory / stage_name)
    verify_stage(directory / "design")
    design = json.loads((directory / "design/splits.json").read_text())
    frame = load_development(root)
    with Progress(root / "logs/model_verification.jsonl", "saved_model_verification") as log:
        vectors, cache = cached_vectors(root, frame)
        if cache != metadata["embedding_cache"]:
            raise ValueError("Embedding input lineage differs")
        replays = []
        for protocol, folds in design["outer"].items():
            oof = pd.read_csv(directory / "review" / f"{protocol}_oof.csv").set_index("row_id")
            for fold, assignment in enumerate(folds):
                ti, vi = assignment["train"], assignment["valid"]
                training, validation = frame.iloc[ti], frame.iloc[vi]
                scores = []
                if protocol == "seen_rule":
                    for number, inner in enumerate(design["inner"][str(fold)]):
                        checkpoint = directory / f"outer_{fold}_inner_{number}"
                        verify_stage(checkpoint)
                        if (
                            json.loads((checkpoint / "training_ids.json").read_text())
                            != frame.iloc[inner["train"]].row_id.tolist()
                        ):
                            raise ValueError("Saved inner model training identities differ")
                        model = joblib.load(checkpoint / "pipeline.joblib")
                        valid = frame.iloc[inner["valid"]]
                        actual = model.predict(
                            valid.drop(columns="rule_violation"), vectors[inner["valid"]]
                        )
                        saved = pd.read_csv(checkpoint / "predictions.csv")
                        error = float(np.max(np.abs(actual - aligned_predictions(valid, saved))))
                        if error > 1e-12:
                            raise ValueError("Saved inner feature/model prediction differs")
                        scores.append(saved)
                        replays.append(
                            {"stage": checkpoint.name, "rows": len(valid), "maximum_error": error}
                        )
                    calibration_scores = aligned_predictions(
                        training, pd.concat(scores, ignore_index=True)
                    )
                else:
                    calibration_scores = support_scores(vectors[ti])["qwen_centroid"]
                parent = FeaturePipeline.from_legacy(
                    root / "runs/expanded" / plan["expanded_run"],
                    f"{protocol}_{fold}",
                    training,
                    vectors.shape[2],
                )
                raw = RoutedClassifier(parent).predict(
                    validation.drop(columns="rule_violation"), vectors[vi]
                )
                calibrator = MonotoneSigmoid.fit(
                    calibration_scores, training.rule_violation, plan["sigmoid"]
                )
                saved = oof.loc[validation.row_id]
                error = max(
                    float(np.max(np.abs(raw - saved.raw))),
                    float(np.max(np.abs(calibrator.predict(raw) - saved.calibrated))),
                )
                if error > 1e-12:
                    raise ValueError("Nested outer calibration replay differs")
                replays.append(
                    {"stage": f"{protocol}_{fold}", "rows": len(vi), "maximum_error": error}
                )
                log.emit("outer_verified", protocol=protocol, fold=fold)
            oof = oof.loc[frame.row_id]
            decision = calibration_decision(
                frame, oof.raw.to_numpy(), oof.calibrated.to_numpy(), plan["calibration_selection"]
            )
            expected = evidence["calibration"]["decisions"][protocol]
            if (
                decision["checks"] != expected["checks"]
                or decision["retain_calibration"] != expected["retain_calibration"]
            ):
                raise ValueError("Saved calibration acceptance decision differs")
        final = directory / "final"
        candidate = joblib.load(final / "candidate.joblib")
        audit = evidence["audit"]["final_artifact"]
        if (
            list(candidate.familiar.screens_) != list(CORE)
            or candidate.familiar.classifier_.n_features_in_ != audit["selected_features"]
            or asdict(candidate.familiar_calibrator) != audit["calibrators"]["seen_rule"]
            or asdict(candidate.unseen_calibrator) != audit["calibrators"]["heldout_rule"]
            or digest(final / "candidate.joblib") != audit["candidate_sha256"]
            or digest(final / "reference.joblib") != audit["reference_sha256"]
        ):
            raise ValueError("Final candidate or calibrator contract differs")
        indices = np.arange(0, len(frame), 97)
        inputs, v = frame.iloc[indices].drop(columns="rule_violation"), vectors[indices]
        batched = candidate.predict(inputs, v)
        reordered = candidate.predict(inputs.iloc[::-1], v[::-1])[::-1]
        if not np.allclose(batched, reordered, atol=1e-12, rtol=0):
            raise ValueError("Final candidate depends on input order")
        result = {
            "run_id": directory.name,
            "classifier_refits": 0,
            "inner_model_replays": 9,
            "outer_route_replays": 7,
            "replays": replays,
            "final_selected_features": audit["selected_features"],
            "final_batch_rows": len(indices),
            "batch_order_parity": True,
            "confirmation_targets_accessed": False,
        }
        atomic_json(root / "logs/model_verification.json", result)
        return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
