"""Nested calibration of the fixed route, then development-only final fitting."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from filelock import FileLock

from jigsaw_rules.calibration import MonotoneSigmoid, calibration_decision
from jigsaw_rules.data import EXAMPLES, normalize
from jigsaw_rules.diagnostics import support_scores
from jigsaw_rules.embeddings import content_key
from jigsaw_rules.expanded import SOURCES as EXPANDED_SOURCES
from jigsaw_rules.expanded import aligned_predictions, design, load_development, validate_splits
from jigsaw_rules.expanded import load_plan as expanded_plan
from jigsaw_rules.feature_decision import decision_evidence
from jigsaw_rules.final_model import FeaturePipeline, RoutedClassifier, familiar_mask, verify_stage
from jigsaw_rules.gate import require_feature_completion
from jigsaw_rules.metrics import calibration_table, evaluate
from jigsaw_rules.model import LexicalClassifier
from jigsaw_rules.research import CORE, cached_vectors
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest, environment, stage
from jigsaw_rules.splits import make_splits

PROTOCOL_COMMIT = "47ba1336fc1d467fba518a00c37aba16fc30efbc"
PLAN_SHA256 = "680ba0077538809e3772db4034439e2d5d675e8483ec171147b3c95e26cd5997"
PUBLIC_FILES = ("results.json", "calibration.json", "audit.json")
SOURCES = tuple(
    sorted(
        set(EXPANDED_SOURCES)
        | {
            "calibration.py",
            "final_model.py",
            "model_validation.py",
        }
    )
)


def load_plan(root: Path) -> dict:
    path = root / "configs/model_validation.json"
    if digest(path) != PLAN_SHA256:
        raise ValueError("Model validation differs from its committed pre-score protocol")
    plan = json.loads(path.read_text())
    if plan["families"] != list(CORE) or plan["inner_folds"] != 3:
        raise ValueError("Model implementation differs from the fixed representation")
    return plan


def sources(root: Path) -> dict:
    return {name: digest(root / "src/jigsaw_rules" / name) for name in SOURCES}


def parent_checkpoints(root: Path, plan: dict) -> dict:
    parent = root / "runs/expanded" / plan["expanded_run"]
    names = ["design"] + [
        f"{protocol}_{fold}_{suffix}"
        for protocol, folds in [("seen_rule", 3), ("heldout_rule", 4)]
        for fold in range(folds)
        for suffix in ["vocabulary", "model_all_transfer", "model_rule_examples", *CORE]
    ]
    return {name: digest(parent / name / "complete.json") for name in names}


def nested_splits(frame: pd.DataFrame, outer: dict, plan: dict) -> list[dict]:
    """Inner positions map explicitly to development IDs; outer labels stay excluded."""
    ti, vi = np.asarray(outer["train"]), np.asarray(outer["valid"])
    training = frame.iloc[ti]
    records = [
        {"train": t.tolist(), "valid": v.tolist(), "purged": purged}
        for t, v, purged in make_splits(training, "seen_rule", plan["inner_folds"], plan["seed"])
    ]
    validate_splits(training, {"seen_rule": records})
    forbidden = set(frame.iloc[vi].body.map(normalize))
    if set(ti) & set(vi) or any(
        set(training[column].map(normalize)) & forbidden for column in ["body", *EXAMPLES]
    ):
        raise ValueError("Outer validation is exposed to an inner fit or calibrator")
    return [
        {"train": ti[r["train"]].tolist(), "valid": ti[r["valid"]].tolist(), "purged": r["purged"]}
        for r in records
    ]


def prediction_frame(frame: pd.DataFrame, probability: np.ndarray) -> pd.DataFrame:
    part = frame[["row_id", "rule", "rule_violation"]].copy()
    part["probability"] = probability
    aligned_predictions(frame, part)
    return part


def save_pipeline(path: Path, model: FeaturePipeline, training: pd.DataFrame) -> None:
    joblib.dump(model, path / "pipeline.joblib", compress=3)
    atomic_json(path / "screening.json", model.audit_)
    atomic_json(path / "selected.json", model.selected_)
    atomic_json(path / "training_ids.json", training.row_id.tolist())
    # The full screen catalog includes rejection reason and training-only association.
    for family, (screen, _) in model.screens_.items():
        screen.catalog_.to_csv(path / f"catalog_{family}.csv.gz", index=False, compression="gzip")


def fit_inner(path, frame, vectors, record, plan, log):
    ti, vi = record["train"], record["valid"]
    training, validation = frame.iloc[ti], frame.iloc[vi]
    model = FeaturePipeline().fit(training, vectors[ti], plan["classifier"], log)
    p = model.predict(validation.drop(columns="rule_violation"), vectors[vi])
    save_pipeline(path, model, training)
    atomic_json(path / "split.json", record)
    atomic_bytes(
        path / "predictions.csv", prediction_frame(validation, p).to_csv(index=False).encode()
    )


def final_fit(path, frame, vectors, familiar_oof, decisions, plan, log):
    familiar = FeaturePipeline().fit(frame, vectors, plan["classifier"], log)
    save_pipeline(path, familiar, frame)
    calibrators = {}
    for protocol, values in [
        ("seen_rule", familiar_oof),
        ("heldout_rule", support_scores(vectors)["qwen_centroid"]),
    ]:
        calibrators[protocol] = (
            MonotoneSigmoid.fit(values, frame.rule_violation, plan["sigmoid"])
            if decisions[protocol]["retain_calibration"]
            else MonotoneSigmoid()
        )
    candidate = RoutedClassifier(familiar, calibrators["seen_rule"], calibrators["heldout_rule"])
    reference = LexicalClassifier(seed=plan["seed"]).fit(frame)
    joblib.dump(candidate, path / "candidate.joblib", compress=3)
    joblib.dump(reference, path / "reference.joblib", compress=3)
    # A serialization/batching replay is a software check, not a training-set metric.
    inputs = frame.iloc[:32].drop(columns="rule_violation")
    expected = candidate.predict(inputs, vectors[:32])
    restored = joblib.load(path / "candidate.joblib")
    if not np.array_equal(expected, restored.predict(inputs, vectors[:32])):
        raise ValueError("Serialized final candidate changed predictions")
    atomic_json(
        path / "artifact.json",
        {
            "training_rows": len(frame),
            "training_ids_sha256": content_key(frame.row_id.tolist()),
            "families": list(CORE),
            "selected_features": familiar.classifier_.n_features_in_,
            "normalized_training_policies": list(familiar.policies_),
            "embedding_dimension": familiar.width_,
            "calibrators": {name: asdict(value) for name, value in calibrators.items()},
            "candidate_sha256": digest(path / "candidate.joblib"),
            "reference_sha256": digest(path / "reference.joblib"),
            "serialized_prediction_parity": True,
            "confirmation_targets_accessed": False,
        },
    )


def execute(root, directory, frame, vectors, plan, log):
    parent = root / "runs/expanded" / plan["expanded_run"]
    verify_stage(parent / "design")
    outer = json.loads((parent / "design/splits.json").read_text())
    if (
        outer != design(frame, expanded_plan(root))
        or json.loads((parent / "design/row_ids.json").read_text()) != frame.row_id.tolist()
    ):
        raise ValueError("Original outer assignments or identities differ")
    inner = {
        str(i): nested_splits(frame, record, plan) for i, record in enumerate(outer["seen_rule"])
    }
    stage(
        directory,
        "design",
        lambda p: atomic_json(p / "splits.json", {"outer": outer, "inner": inner}),
    )
    parts, audits = {}, []
    for protocol, assignments in outer.items():
        rows = []
        for fold, assignment in enumerate(assignments):
            ti, vi = assignment["train"], assignment["valid"]
            training, validation = frame.iloc[ti], frame.iloc[vi]
            inputs = validation.drop(columns="rule_violation")
            parent_model = FeaturePipeline.from_legacy(
                parent, f"{protocol}_{fold}", training, vectors.shape[2]
            )
            known = familiar_mask(inputs, parent_model.policies_)
            if not (known.all() if protocol == "seen_rule" else (~known).all()):
                raise ValueError("Actual fitted policy set does not produce the expected route")
            if protocol == "seen_rule":
                inner_parts = []
                for number, record in enumerate(inner[str(fold)]):
                    checkpoint = stage(
                        directory,
                        f"outer_{fold}_inner_{number}",
                        lambda p, record=record: fit_inner(p, frame, vectors, record, plan, log),
                    )
                    verify_stage(checkpoint)
                    inner_parts.append(pd.read_csv(checkpoint / "predictions.csv"))
                calibration_scores = aligned_predictions(
                    training, pd.concat(inner_parts, ignore_index=True)
                )
                raw = RoutedClassifier(parent_model).predict(inputs, vectors[vi])
                saved = pd.read_csv(
                    parent / f"{protocol}_{fold}_model_all_transfer/predictions.csv"
                )
                delta = float(np.max(np.abs(raw - aligned_predictions(validation, saved))))
                if delta > 1e-12:
                    raise ValueError(
                        "Portable familiar parent differs from saved research predictions"
                    )
            else:
                calibration_scores = support_scores(vectors[ti])["qwen_centroid"]
                raw = RoutedClassifier(parent_model).predict(inputs, vectors[vi])
                delta = float(np.max(np.abs(raw - support_scores(vectors[vi])["qwen_centroid"])))
                if delta != 0:
                    raise ValueError("Unseen routing differs from the selected centroid")
            calibrator = MonotoneSigmoid.fit(
                calibration_scores, training.rule_violation, plan["sigmoid"]
            )
            calibrated = calibrator.predict(raw)
            reference_path = parent / f"{protocol}_{fold}_model_rule_examples"
            verify_stage(reference_path)
            reference = aligned_predictions(
                validation, pd.read_csv(reference_path / "predictions.csv")
            )
            part = prediction_frame(validation, raw).rename(columns={"probability": "raw"})
            part["calibrated"], part["reference"], part["fold"] = calibrated, reference, fold
            rows.append(part)
            audit = {
                "protocol": protocol,
                "fold": fold,
                "training_rows": len(ti),
                "validation_rows": len(vi),
                "training_ids_sha256": content_key(training.row_id.tolist()),
                "calibration_rows": len(calibration_scores),
                "parent_prediction_max_error": delta,
                "calibrator": asdict(calibrator),
                "inner_training_rows": [len(r["train"]) for r in inner[str(fold)]]
                if protocol == "seen_rule"
                else [],
                "raw_metrics": evaluate(validation.rule_violation, raw, validation.rule),
                "calibrated_metrics": evaluate(
                    validation.rule_violation, calibrated, validation.rule
                ),
            }
            audits.append(audit)

            def save_outer(p, part=part, audit=audit):
                atomic_bytes(p / "predictions.csv", part.to_csv(index=False).encode())
                atomic_json(p / "audit.json", audit)

            stage(directory, f"{protocol}_{fold}_evaluation", save_outer)
            log.emit(
                "outer_evaluated", protocol=protocol, fold=fold, parent_prediction_max_error=delta
            )
        combined = (
            pd.concat(rows, ignore_index=True).set_index("row_id").loc[frame.row_id].reset_index()
        )
        # Align and validate every score variant, labels, policy and row coverage.
        for column in ["raw", "calibrated", "reference"]:
            aligned_predictions(
                frame,
                combined[["row_id", "rule", "rule_violation", column]].rename(
                    columns={column: "probability"}
                ),
            )
        parts[protocol] = combined
    historical = json.loads((root / "reports/expanded/results.json").read_text())
    for protocol, part in parts.items():
        for name, old_name in [
            ("reference", "rule_examples"),
            ("raw", "all_transfer" if protocol == "seen_rule" else "qwen_centroid"),
        ]:
            expected = next(
                r["metrics"]
                for r in historical
                if r["protocol"] == protocol and r["model"] == old_name
            )
            actual = evaluate(frame.rule_violation, part[name].to_numpy(), frame.rule)
            if any(
                abs(actual[k] - expected[k]) > 1e-12
                for k in ["rule_macro_auc", "pooled_auc", "log_loss", "brier"]
            ):
                raise ValueError("Replayed route differs from source-bound feature evidence")
    decisions = {
        name: calibration_decision(
            frame, part.raw.to_numpy(), part.calibrated.to_numpy(), plan["calibration_selection"]
        )
        for name, part in parts.items()
    }
    final = stage(
        directory,
        "final",
        lambda p: final_fit(
            p, frame, vectors, parts["seen_rule"].raw.to_numpy(), decisions, plan, log
        ),
    )
    verify_stage(final)

    def review(p):
        results, reliability = [], []
        for protocol, part in parts.items():
            for name in ["reference", "raw", "calibrated"]:
                values = part[name].to_numpy()
                results.append(
                    {
                        "protocol": protocol,
                        "model": name,
                        "metrics": evaluate(frame.rule_violation, values, frame.rule),
                    }
                )
                reliability.append(
                    {
                        "protocol": protocol,
                        "model": name,
                        "bins": calibration_table(frame.rule_violation, values).to_dict("records"),
                    }
                )
            atomic_bytes(p / f"{protocol}_oof.csv", part.to_csv(index=False).encode())
        artifact = json.loads((final / "artifact.json").read_text())
        atomic_json(p / "results.json", results)
        atomic_json(p / "calibration.json", {"decisions": decisions, "reliability": reliability})
        atomic_json(
            p / "audit.json",
            {
                "development_rows": len(frame),
                "policies": frame.rule.nunique(),
                "nested_classifier_fits": 9,
                "reused_outer_classifiers": 3,
                "unseen_routing_parents_verified": 4,
                "full_development_fits": 2,
                "outer_audits": audits,
                "final_artifact": artifact,
                "screening": json.loads((final / "screening.json").read_text()),
                "confirmation_targets_accessed": False,
                "production_promotion_authorized": False,
                "scope": (
                    "Nested development calibration, conditional on earlier adaptive feature "
                    "selection; confirmation remains unopened"
                ),
            },
        )

    stage(directory, "review", review)


def run_validation(root: Path) -> Path:
    root = root.resolve()
    require_feature_completion(root)
    plan, decision = load_plan(root), decision_evidence(root)
    if decision["metadata"]["run_id"] != plan["feature_decision_run"]:
        raise ValueError("Feature completion decision differs from the model protocol")
    frame = load_development(root)
    if len(frame) != plan["development_rows"]:
        raise ValueError("Unexpected development cohort")
    with Progress(root / "logs/model_validation.jsonl", "model_validation") as log:
        vectors, cache = cached_vectors(root, frame)
        provenance = {
            "schema": 1,
            "protocol_commit": PROTOCOL_COMMIT,
            "plan_sha256": PLAN_SHA256,
            "sources": sources(root),
            "development_sha256": content_key(frame.to_csv(index=False)),
            "feature_decision_metadata_sha256": digest(
                root / "reports/feature_decision/metadata.json"
            ),
            "expanded_metadata_sha256": digest(root / "reports/expanded/metadata.json"),
            "semantic_spec_sha256": digest(root / "configs/semantic.json"),
            "embedding_cache": cache,
            "parent_checkpoints": parent_checkpoints(root, plan),
            "environment": environment(),
        }
        directory = root / "runs/model_validation" / content_key(provenance)[:20]
        with FileLock(str(root / "runs/model_validation.lock"), timeout=1):
            atomic_json(directory / "provenance.json", provenance)
            execute(root, directory, frame, vectors, plan, log)
            export_validation(root, directory)
    return directory


def export_validation(root: Path, directory: Path) -> None:
    provenance = json.loads((directory / "provenance.json").read_text())
    if provenance["sources"] != sources(root) or directory.name != content_key(provenance)[:20]:
        raise ValueError("Model validation source or identity differs")
    review, final = directory / "review", directory / "final"
    verify_stage(review)
    verify_stage(final)
    frame = load_development(root)
    if provenance["development_sha256"] != content_key(frame.to_csv(index=False)):
        raise ValueError("Model validation development data differs")
    for result in json.loads((review / "results.json").read_text()):
        part = pd.read_csv(review / f"{result['protocol']}_oof.csv")
        part = part[["row_id", "rule", "rule_violation", result["model"]]].rename(
            columns={result["model"]: "probability"}
        )
        metrics = evaluate(frame.rule_violation, aligned_predictions(frame, part), frame.rule)
        for name in ["rule_macro_auc", "pooled_auc", "log_loss", "brier"]:
            if abs(metrics[name] - result["metrics"][name]) > 1e-12:
                raise ValueError("Model validation publication differs from saved predictions")
    public = root / "reports/model_validation"
    for name in PUBLIC_FILES:
        atomic_bytes(public / name, (review / name).read_bytes())
    atomic_json(
        public / "metadata.json",
        {
            **provenance,
            "run_id": directory.name,
            "files": {name: digest(review / name) for name in PUBLIC_FILES},
            "private_checkpoint_sha256": digest(review / "complete.json"),
            "final_checkpoint_sha256": digest(final / "complete.json"),
        },
    )


def validation_evidence(root: Path) -> dict | None:
    path = root / "reports/model_validation/metadata.json"
    if not path.exists():
        return None
    load_plan(root)
    require_feature_completion(root)
    metadata = json.loads(path.read_text())
    identity = {
        k: v
        for k, v in metadata.items()
        if k not in {"run_id", "files", "private_checkpoint_sha256", "final_checkpoint_sha256"}
    }
    if (
        metadata.get("sources") != sources(root)
        or metadata.get("plan_sha256") != PLAN_SHA256
        or metadata.get("protocol_commit") != PROTOCOL_COMMIT
        or metadata.get("run_id") != content_key(identity)[:20]
        or metadata.get("feature_decision_metadata_sha256")
        != digest(root / "reports/feature_decision/metadata.json")
        or metadata.get("expanded_metadata_sha256")
        != digest(root / "reports/expanded/metadata.json")
        or metadata.get("semantic_spec_sha256") != digest(root / "configs/semantic.json")
        or set(metadata.get("files", {})) != set(PUBLIC_FILES)
    ):
        raise ValueError("Stale model validation evidence")
    result = {"metadata": metadata}
    for name, sha in metadata["files"].items():
        if digest(path.parent / name) != sha:
            raise ValueError("Model validation evidence checksum mismatch")
        result[name.removesuffix(".json")] = json.loads((path.parent / name).read_text())
    return result
