"""Preregistered Matryoshka support geometry; no fitted model or new inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jigsaw_rules.data import normalize
from jigsaw_rules.diagnostics import support_scores
from jigsaw_rules.embeddings import content_key
from jigsaw_rules.expanded import aligned_predictions, expanded_evidence, load_development
from jigsaw_rules.metrics import evaluate
from jigsaw_rules.research import cached_vectors
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest, environment, stage
from jigsaw_rules.uncertainty import paired_auc_comparisons

PROTOCOL_COMMIT = "ccd81fb0a589dd55cc763503fc7434b0c3e873ea"
PLAN_SHA256 = "edc25f949581742389b08717bac4513bd5f7406c2c0c98fd2886fdb4cd3ab07b"
PUBLIC_FILES = ("results.json", "uncertainty.json")


def prefix_scores(vectors: np.ndarray, dimensions: list[int]) -> dict[str, np.ndarray]:
    if (
        vectors.ndim != 3
        or vectors.shape[1] != 5
        or vectors.shape[2] != 1024
        or not np.isfinite(vectors).all()
        or not np.allclose(np.linalg.norm(vectors, axis=2), 1, atol=1e-4)
    ):
        raise ValueError("Expected five finite, normalized 1024-dimensional vectors per row")
    if not dimensions or any(d not in (32, 64, 128, 256, 512, 1024) for d in dimensions):
        raise ValueError("Unsupported preregistered prefix dimension")
    scores = {}
    for dimension in dimensions:
        prefix = vectors if dimension == 1024 else vectors[:, :, :dimension].copy()
        if dimension != 1024:
            norms = np.linalg.norm(prefix, axis=2, keepdims=True)
            if (norms <= 1e-12).any():
                raise ValueError("A zero prefix cannot be normalized")
            prefix /= norms
        scores[f"centroid_{dimension}"] = support_scores(prefix)["qwen_centroid"]
    return scores


def run_resolution(root: Path) -> Path:
    base = expanded_evidence(root)
    config = root / "configs/resolution.json"
    if base is None or digest(config) != PLAN_SHA256:
        raise ValueError("Resolution study requires expanded evidence and its frozen plan")
    plan, frame = json.loads(config.read_text()), load_development(root)
    vectors, cache = cached_vectors(root, frame)
    identity = {
        "schema": 1,
        "protocol_commit": PROTOCOL_COMMIT,
        "plan_sha256": digest(config),
        "source_sha256": digest(Path(__file__)),
        "expanded_metadata_sha256": digest(root / "reports/expanded/metadata.json"),
        "embedding_cache": cache,
        "environment": environment(),
        "model_fits": 0,
        "new_encoder_calls": 0,
        "confirmation_targets_accessed": False,
    }
    directory = root / "runs/resolution" / content_key(identity)[:20]
    with Progress(root / "logs/resolution.jsonl", "embedding_resolution"):
        atomic_json(directory / "provenance.json", identity)

        def review(path):
            scores = prefix_scores(vectors, plan["dimensions"])
            original = root / "runs/expanded" / base["metadata"]["run_id"] / "review"
            marker = json.loads((original / "complete.json").read_text())["files"]
            if digest(original / "oof.csv") != marker["oof.csv"]:
                raise ValueError("Expanded predictions checksum differs")
            oof = pd.read_csv(original / "oof.csv")
            reference = aligned_predictions(
                frame, oof[(oof.protocol == "heldout_rule") & (oof.model == "qwen_centroid")]
            )
            if not np.allclose(scores["centroid_1024"], reference, rtol=0, atol=1e-12):
                raise ValueError("Full-dimensional reference differs from expanded evidence")
            results = [
                {
                    "model": name,
                    "dimension": int(name.split("_")[1]),
                    "metrics": evaluate(frame.rule_violation, p, frame.rule),
                }
                for name, p in scores.items()
            ]
            contrasts = [
                (name, name, "centroid_1024") for name in scores if name != "centroid_1024"
            ]
            uncertainty = paired_auc_comparisons(
                frame.rule_violation,
                scores,
                frame.rule,
                frame.body.map(normalize),
                contrasts,
                draws=plan["bootstrap_draws"],
                seed=plan["seed"],
            )
            saved = frame[["row_id", "rule", "rule_violation"]].copy()
            for name, values in scores.items():
                saved[name] = values.astype(float)
            atomic_bytes(path / "predictions.csv", saved.to_csv(index=False).encode())
            atomic_json(path / "results.json", results)
            atomic_json(path / "uncertainty.json", uncertainty)

        review_path = stage(directory, "review", review)
        saved = pd.read_csv(review_path / "predictions.csv")
        if saved.row_id.tolist() != frame.row_id.tolist():
            raise ValueError("Resolution prediction row order differs")
        for item in json.loads((review_path / "results.json").read_text()):
            actual = evaluate(frame.rule_violation, saved[item["model"]], frame.rule)
            for metric in ("rule_macro_auc", "pooled_auc", "log_loss", "brier"):
                if abs(actual[metric] - item["metrics"][metric]) > 1e-12:
                    raise ValueError("Resolution metric differs from saved predictions")
        public = root / "reports/resolution"
        for name in PUBLIC_FILES:
            atomic_bytes(public / name, (review_path / name).read_bytes())
        atomic_json(
            public / "metadata.json",
            {
                **identity,
                "run_id": directory.name,
                "private_checkpoint_sha256": digest(review_path / "complete.json"),
                "files": {name: digest(public / name) for name in PUBLIC_FILES},
            },
        )
    return directory


def resolution_evidence(root: Path) -> dict | None:
    path = root / "reports/resolution/metadata.json"
    if not path.exists():
        return None
    metadata = json.loads(path.read_text())
    expanded_evidence(root)
    if (
        metadata.get("source_sha256") != digest(Path(__file__))
        or metadata.get("plan_sha256") != digest(root / "configs/resolution.json")
        or metadata.get("expanded_metadata_sha256")
        != digest(root / "reports/expanded/metadata.json")
        or metadata.get("protocol_commit") != PROTOCOL_COMMIT
        or set(metadata.get("files", {})) != set(PUBLIC_FILES)
    ):
        raise ValueError("Stale embedding resolution evidence")
    identity = {
        k: v
        for k, v in metadata.items()
        if k not in {"run_id", "files", "private_checkpoint_sha256"}
    }
    if content_key(identity)[:20] != metadata["run_id"]:
        raise ValueError("Resolution evidence identity differs")
    result = {"metadata": metadata}
    for name, sha in metadata["files"].items():
        if digest(path.parent / name) != sha:
            raise ValueError("Resolution public checksum differs")
        result[name.removesuffix(".json")] = json.loads((path.parent / name).read_text())
    return result
