"""Target-free preparation and frozen inference for protected confirmation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from jigsaw_rules.confirmation_inputs import reserve_frame
from jigsaw_rules.data import normalize, validate_frame
from jigsaw_rules.diagnostics import support_scores
from jigsaw_rules.embeddings import QwenEncoder, content_key, load_spec
from jigsaw_rules.expanded_embeddings import extend_embeddings
from jigsaw_rules.final_model import familiar_mask, verify_stage
from jigsaw_rules.released import load_plan as released_plan
from jigsaw_rules.research import cached_vectors
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest, environment, stage
from jigsaw_rules.semantic import input_texts

PROTOCOL_COMMIT = "5ca6bd50d54ceacc172786e6df0640810aab96d8"
PROTOCOL_SHA256 = "edc2b883bbd5bf08365a8c5c2251acb888a61bb7e002a783d45ad285d9e7b6ff"
MODELS = ("candidate", "reference", "uncalibrated_route", "centroid_everywhere")


def load_protocol(root: Path) -> dict:
    path = root / "configs/confirmation.json"
    if digest(path) != PROTOCOL_SHA256:
        raise ValueError("Protected confirmation protocol differs from preregistration")
    return json.loads(path.read_text())


def strict_stage(directory: Path, name: str, action) -> Path:
    target = directory / name
    if target.exists():
        verify_stage(target)
    return stage(directory, name, action)


def prepare_inputs(root: Path) -> Path:
    """Materialize exactly the previously eligible inputs; never open solution.csv."""
    plan = load_protocol(root)
    eligibility = root / "runs/confirmation_inputs" / plan["eligibility_run"] / "eligibility"
    if digest(eligibility / "complete.json") != plan["eligibility_stage_sha256"]:
        raise ValueError("Eligibility stage differs")
    verify_stage(eligibility)
    ids = json.loads((eligibility / "eligible_ids.json").read_text())
    if len(ids) != plan["eligible_rows"] or content_key(ids) != plan["eligible_ids_sha256"]:
        raise ValueError("Eligible identities differ")
    reserved, boundary = reserve_frame(root)
    frame = reserved.set_index("row_id").loc[ids].reset_index()
    assignment_path = root / "runs/released" / boundary["boundary_run"] / "boundary/assignments.csv"
    assignments = pd.read_csv(assignment_path).set_index("row_id").loc[ids].reset_index()
    if not assignments.role.eq("confirmation").all():
        raise ValueError("Nonreserved row in confirmation inputs")
    flags = pd.read_csv(eligibility / "eligibility.csv").set_index("row_id").loc[ids]
    assignments["self_support_match"] = flags.self_support_match.to_numpy()
    frame_bytes, assignment_bytes = (
        frame.to_csv(index=False).encode(),
        assignments.to_csv(index=False).encode(),
    )
    identity = {
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "eligible_ids_sha256": content_key(ids),
        "frame_content_key": content_key(frame.to_csv(index=False)),
        "assignment_content_key": content_key(assignments.to_csv(index=False)),
        "boundary": boundary,
        "targets_accessed": False,
    }
    directory = root / "runs/confirmation" / content_key(identity)[:20]

    def action(path):
        atomic_bytes(path / "frame.csv", frame_bytes)
        atomic_bytes(path / "assignments.csv", assignment_bytes)
        atomic_json(path / "identity.json", identity)

    strict_stage(directory, "inputs", action)
    return directory


def read_inputs(root: Path, directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    plan = load_protocol(root)
    inputs = directory / "inputs"
    verify_stage(inputs)
    identity = json.loads((inputs / "identity.json").read_text())
    if identity["protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("Input protocol differs")
    frame, assignments = pd.read_csv(inputs / "frame.csv"), pd.read_csv(inputs / "assignments.csv")
    validate_frame(frame, train=False)
    if (
        len(frame) != plan["eligible_rows"]
        or content_key(frame.row_id.tolist()) != plan["eligible_ids_sha256"]
        or not np.array_equal(frame.row_id, assignments.row_id)
        or not assignments.role.eq("confirmation").all()
        or not np.array_equal(
            frame.body.map(normalize).map(lambda t: hashlib.sha256(t.encode()).hexdigest()),
            assignments.body_sha256,
        )
    ):
        raise ValueError("Confirmation row identities or assignments differ")
    rules = released_plan(root)["rule_text"]
    if not np.array_equal(frame.rule, assignments.rule_id.map(rules)):
        raise ValueError("Confirmation policies differ")
    if (
        content_key(frame.to_csv(index=False)) != identity["frame_content_key"]
        or content_key(assignments.to_csv(index=False)) != identity["assignment_content_key"]
    ):
        raise ValueError("Confirmation input content differs")
    return frame, assignments


def predict_batch(candidate, reference, frame: pd.DataFrame, vectors: np.ndarray) -> pd.DataFrame:
    """No fitting or cohort statistics; preserve the original float32 centroid formula."""
    validate_frame(frame, train=False)
    known = familiar_mask(frame, candidate.familiar.policies_)
    centroid = support_scores(vectors)["qwen_centroid"]
    raw = centroid.astype(float).copy()
    if known.any():
        raw[known] = candidate.familiar.predict(frame.iloc[np.flatnonzero(known)], vectors[known])
    output = candidate.predict(frame, vectors)
    return pd.DataFrame(
        {
            "familiar": known,
            "candidate": output,
            "reference": reference.predict(frame),
            "uncalibrated_route": raw,
            "centroid_everywhere": centroid,
        }
    )


def run_predictions(root: Path, directory: Path, *, encode: bool = False) -> Path:
    plan = load_protocol(root)
    frame, assignments = read_inputs(root, directory)
    final = root / "runs/model_validation" / plan["model_run"] / "final"
    if digest(final / "complete.json") != plan["model_stage_sha256"]:
        raise ValueError("Fitted model stage differs")
    verify_stage(final)
    for name in ["candidate", "reference"]:
        if digest(final / f"{name}.joblib") != plan[f"{name}_sha256"]:
            raise ValueError("Serialized model differs")
    if (
        content_key(json.loads((final / "training_ids.json").read_text()))
        != plan["training_ids_sha256"]
    ):
        raise ValueError("Fitted training identities differ")
    spec = load_spec(root)
    contract = QwenEncoder(root, spec).contract
    provenance = {
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "inputs_stage_sha256": digest(directory / "inputs/complete.json"),
        "candidate_sha256": plan["candidate_sha256"],
        "reference_sha256": plan["reference_sha256"],
        "encoder": contract,
        "environment": environment(),
        "sources": {
            p.relative_to(root).as_posix(): digest(p)
            for p in sorted((root / "src/jigsaw_rules").glob("*.py"))
        },
        "requested_embedding_keys_sha256": content_key(
            [content_key(t) for t in input_texts(frame, spec)]
        ),
        "targets_accessed": False,
    }
    prior = directory / "predictions/provenance.json"
    if prior.exists() and json.loads(prior.read_text()) != provenance:
        raise ValueError("Frozen inference provenance differs")
    if (directory / "predictions").exists():
        verify_stage(directory / "predictions")
        return directory / "predictions"
    if encode:
        extend_embeddings(root, frame, workers=4, shard_size=64)
    vectors, cache = cached_vectors(root, frame)
    candidate, reference = (
        joblib.load(final / "candidate.joblib"),
        joblib.load(final / "reference.joblib"),
    )

    def action(path):
        blocks = []
        with Progress(directory / "prediction_events.jsonl", "protected_inference") as log:
            for start in range(0, len(frame), plan["prediction_batch_rows"]):
                end = start + plan["prediction_batch_rows"]
                blocks.append(
                    predict_batch(candidate, reference, frame.iloc[start:end], vectors[start:end])
                )
                log.emit("predicted", completed=min(end, len(frame)), total=len(frame))
        scores = pd.concat(blocks, ignore_index=True)
        expected = assignments.rule_id.isin(plan["familiar_rule_ids"]).to_numpy()
        if not np.array_equal(scores.familiar, expected):
            raise ValueError("Routing does not match frozen development membership")
        for name in MODELS:
            if not np.isfinite(scores[name]).all() or not scores[name].between(0, 1).all():
                raise ValueError("Invalid protected probabilities")
        order = np.arange(min(128, len(frame)))[::-1]
        parity = predict_batch(candidate, reference, frame.iloc[order], vectors[order])
        difference = float(
            np.max(
                np.abs(
                    parity[list(MODELS)].to_numpy() - scores.iloc[order][list(MODELS)].to_numpy()
                )
            )
        )
        if difference > 1e-12:
            raise ValueError("Inference order parity failed")
        table = pd.concat([assignments.drop(columns="role"), scores], axis=1)
        atomic_bytes(
            path / "predictions.csv", table.to_csv(index=False, float_format="%.17g").encode()
        )
        atomic_json(path / "provenance.json", provenance)
        atomic_json(
            path / "audit.json",
            {
                "rows": len(table),
                "models": list(MODELS),
                "eligible_ids_sha256": content_key(table.row_id.tolist()),
                "familiar_rows": int(table.familiar.sum()),
                "unseen_rows": int((~table.familiar).sum()),
                "prediction_sha256": digest(path / "predictions.csv"),
                "maximum_order_difference": difference,
                "cache": cache,
                "targets_accessed": False,
                "models_refitted": 0,
            },
        )

    return strict_stage(directory, "predictions", action)
