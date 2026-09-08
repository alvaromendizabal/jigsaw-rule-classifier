"""Fixed lexical weighting controls and frozen support-score comparisons."""

from __future__ import annotations

import json
import sys
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from filelock import FileLock
from scipy import sparse
from scipy.special import expit
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize as normalize_matrix

from jigsaw_rules.data import EXAMPLES, load_data, normalize
from jigsaw_rules.embeddings import content_key
from jigsaw_rules.metrics import evaluate
from jigsaw_rules.research import cached_vectors, research_evidence, verify_research
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest, stage
from jigsaw_rules.semantic import reference_splits
from jigsaw_rules.uncertainty import paired_auc_comparisons

FIT_VARIANTS = (
    "word_full",
    "character_full",
    "word_character_full",
    "word_nb",
    "character_nb",
    "word_character_nb",
    "latent_word",
    "latent_character",
    "latent_joint_lexical",
    "latent_qwen_body",
    "latent_qwen_interaction",
)
PUBLIC_FILES = ("results.json", "uncertainty.json", "sensitivity.json", "audit.json")


def nb_weights(matrix, target) -> np.ndarray:
    """Add-one class-conditional token weights learned inside an outer training fold."""
    target = np.asarray(target)
    if matrix.shape[0] != len(target) or set(target.tolist()) != {0, 1}:
        raise ValueError("NB weights need aligned binary training labels")
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError("NB token values must be nonnegative and finite")
    positive = 1 + np.asarray(matrix[target == 1].sum(axis=0)).ravel()
    negative = 1 + np.asarray(matrix[target == 0].sum(axis=0)).ravel()
    return np.log(positive / positive.sum()) - np.log(negative / negative.sum())


def support_scores(vectors: np.ndarray) -> dict[str, np.ndarray]:
    if vectors.ndim != 3 or vectors.shape[1] != 5 or not np.isfinite(vectors).all():
        raise ValueError("Support scoring needs finite body and four example vectors")
    similarity = np.einsum("nd,nkd->nk", vectors[:, 0], vectors[:, 1:])
    positive, negative = similarity[:, :2], similarity[:, 2:]
    scores = {
        "qwen_maximum": expit(10 * (positive.max(1) - negative.max(1))),
        "qwen_mean": expit(10 * (positive.mean(1) - negative.mean(1))),
    }
    pc, nc = vectors[:, 1:3].mean(1), vectors[:, 3:5].mean(1)
    pcos = np.einsum("nd,nd->n", vectors[:, 0], pc) / np.linalg.norm(pc, axis=1).clip(1e-12)
    ncos = np.einsum("nd,nd->n", vectors[:, 0], nc) / np.linalg.norm(nc, axis=1).clip(1e-12)
    scores["qwen_centroid"] = expit(10 * (pcos - ncos))
    for temperature in (0.05, 0.1, 0.2, 0.5):
        kernels = np.exp((similarity - 1) / temperature)
        scores[f"qwen_kernel_{temperature}"] = kernels[:, :2].sum(1) / kernels.sum(1).clip(1e-12)
    return scores


def fit_lexical(path, variant, matrices, training, validation, protocol, fold):
    if variant.startswith("latent_"):
        groups = {
            "latent_word": ("word",),
            "latent_character": ("character",),
            "latent_joint_lexical": ("word", "character"),
            "latent_qwen_body": ("qwen_body",),
            "latent_qwen_interaction": ("qwen_interaction",),
        }[variant]
        x = sparse.hstack([matrices[group][0] for group in groups], format="csr")
        v = sparse.hstack([matrices[group][1] for group in groups], format="csr")
        count = min(128, x.shape[0] - 1, x.shape[1] - 1)
        projection = TruncatedSVD(n_components=count, random_state=2025)
        x = normalize_matrix(projection.fit_transform(x))
        v = normalize_matrix(projection.transform(v))
        if not np.isfinite(x).all() or not np.isfinite(v).all():
            raise ValueError("Nonfinite latent features")
        model = LogisticRegression(C=2, solver="liblinear", max_iter=2000, random_state=2025)
        model.fit(x, training.rule_violation)
        prediction = validation[["row_id", "rule", "rule_violation"]].copy()
        prediction["probability"] = model.predict_proba(v)[:, 1]
        prediction["model"], prediction["protocol"], prediction["fold"] = variant, protocol, fold
        atomic_bytes(path / "predictions.csv", prediction.to_csv(index=False).encode())
        joblib.dump((model, projection), path / "model.joblib", compress=3)
        atomic_json(
            path / "details.json",
            {
                "features": x.shape[1],
                "training_rows": len(training),
                "explained_variance_fraction": float(projection.explained_variance_ratio_.sum()),
                "scope": "SVD fitted on outer training rows only; row L2 normalization",
            },
        )
        return
    groups = (
        ("word", "character")
        if variant.startswith("word_character")
        else ("word" if variant.startswith("word") else "character",)
    )
    xs, vs, weights = [], [], {}
    for family in groups:
        x, v = matrices[family]
        if variant.endswith("nb"):
            weights[family] = nb_weights(x, training.rule_violation)
            x, v = x.multiply(weights[family]), v.multiply(weights[family])
        xs.append(x)
        vs.append(v)
    x, v = sparse.hstack(xs, format="csr"), sparse.hstack(vs, format="csr")
    model = LogisticRegression(C=2, solver="liblinear", max_iter=2000, random_state=2025)
    model.fit(x, training.rule_violation)
    prediction = validation[["row_id", "rule", "rule_violation"]].copy()
    prediction["probability"] = model.predict_proba(v)[:, 1]
    prediction["model"], prediction["protocol"], prediction["fold"] = variant, protocol, fold
    atomic_bytes(path / "predictions.csv", prediction.to_csv(index=False).encode())
    joblib.dump((model, weights), path / "model.joblib", compress=3)
    atomic_json(path / "details.json", {"features": x.shape[1], "training_rows": len(training)})


def run_diagnostics(root: Path) -> Path:
    root = root.resolve()
    evidence = research_evidence(root)
    if evidence is None:
        raise ValueError("Diagnostics require verified broad research")
    broad = root / "runs" / evidence["metadata"]["run_id"]
    verify_research(root, broad)
    baseline_run = evidence["metadata"]["baseline_run"]
    train, _, _ = load_data(root / "data/raw")
    splits = reference_splits(root, train, baseline_run)
    identity = {
        "source_sha256": digest(Path(__file__)),
        "broad_run": broad.name,
        "training_sha256": digest(root / "data/raw/train.csv"),
        "variants": FIT_VARIANTS,
        "python": sys.version.split()[0],
        "software": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib")},
    }
    directory = root / "runs" / content_key(identity)[:20]
    with (
        FileLock(str(root / "runs/diagnostics.lock"), timeout=1),
        Progress(root / "logs/diagnostics.jsonl", "feature_diagnostics") as log,
    ):
        atomic_json(directory / "provenance.json", identity)
        vectors, _ = cached_vectors(root, train)
        frozen = support_scores(vectors)
        dense = {
            "qwen_body": vectors[:, 0],
            "qwen_interaction": vectors[:, 0] * (vectors[:, 1:3].mean(1) - vectors[:, 3:5].mean(1)),
        }
        del vectors
        parts = []
        self_match = np.column_stack(
            [(train.body.map(normalize) == train[column].map(normalize)) for column in EXAMPLES]
        ).any(1)
        for protocol, assignments in splits.items():
            for fold, assignment in enumerate(assignments):
                training = train.iloc[assignment["train"]]
                validation = train.iloc[assignment["valid"]]
                word, character = joblib.load(
                    broad / f"{protocol}_{fold}_vocabulary/vocabulary.joblib"
                )
                matrices = {
                    name: (
                        vectorizer.transform(training.body),
                        vectorizer.transform(validation.body),
                    )
                    for name, vectorizer in (("word", word), ("character", character))
                }
                matrices.update(
                    {
                        name: (
                            sparse.csr_matrix(array[assignment["train"]]),
                            sparse.csr_matrix(array[assignment["valid"]]),
                        )
                        for name, array in dense.items()
                    }
                )
                for variant in FIT_VARIANTS:

                    def fit(
                        path,
                        variant=variant,
                        matrices=matrices,
                        training=training,
                        validation=validation,
                        protocol=protocol,
                        fold=fold,
                    ):
                        fit_lexical(path, variant, matrices, training, validation, protocol, fold)

                    path = stage(directory, f"{protocol}_{fold}_{variant}", fit)
                    parts.append(pd.read_csv(path / "predictions.csv"))
                for family, vectorizer in (("word", word), ("character", character)):
                    body = vectorizer.transform(validation.body)
                    similarities = np.column_stack(
                        [
                            np.asarray(
                                body.multiply(vectorizer.transform(validation[column])).sum(1)
                            ).ravel()
                            for column in EXAMPLES
                        ]
                    )
                    for aggregation in ("mean", "max"):
                        pos = getattr(similarities[:, :2], aggregation)(axis=1)
                        neg = getattr(similarities[:, 2:], aggregation)(axis=1)
                        part = validation[["row_id", "rule", "rule_violation"]].copy()
                        part["probability"] = expit(10 * (pos - neg))
                        part["model"] = f"{family}_support_{aggregation}"
                        part["protocol"], part["fold"] = protocol, fold
                        parts.append(part)
            for model, probability in frozen.items():
                part = train[["row_id", "rule", "rule_violation"]].copy()
                part["probability"], part["protocol"], part["model"] = probability, protocol, model
                parts.append(part)
            log.emit("DIAGNOSTICS_PROTOCOL_COMPLETED", protocol=protocol)
        oof = pd.concat(parts, ignore_index=True)
        original = pd.read_csv(root / f"runs/{baseline_run}/review/oof.csv")
        original = original[original.model == "rule_examples"].copy()
        original["model"] = "original_reference"
        broad_oof = pd.read_csv(broad / "review/oof.csv")
        all_predictions = pd.concat([oof, original, broad_oof], ignore_index=True)
        results, uncertainty, sensitivity = [], [], []
        for protocol in splits:
            bank = {}
            for model in sorted(all_predictions.model.unique()):
                part = all_predictions[
                    (all_predictions.protocol == protocol) & (all_predictions.model == model)
                ]
                if part.row_id.duplicated().any() or set(part.row_id) != set(train.row_id):
                    raise ValueError("Diagnostic OOF coverage differs")
                part = part.set_index("row_id").loc[train.row_id]
                bank[model] = part.probability.to_numpy()
                if model in set(oof.model):
                    results.append(
                        {
                            "model": model,
                            "protocol": protocol,
                            "metrics": evaluate(train.rule_violation, part.probability, train.rule),
                        }
                    )
                sensitivity.append(
                    {
                        "model": model,
                        "protocol": protocol,
                        "condition": "exclude_self_support",
                        "retained_rows": int((~self_match).sum()),
                        "metrics": evaluate(
                            train.rule_violation[~self_match],
                            part.probability[~self_match],
                            train.rule[~self_match],
                        ),
                    }
                )
            contrasts = [(model, model, "original_reference") for model in sorted(set(oof.model))]
            contrasts += [
                (f"reweight_{family}", f"{family}_nb", f"{family}_full")
                for family in ("word", "character", "word_character")
            ]
            uncertainty.extend(
                {"protocol": protocol, **item}
                for item in paired_auc_comparisons(
                    train.rule_violation, bank, train.rule, train.body.map(normalize), contrasts
                )
            )

        def review(path):
            atomic_json(path / "results.json", results)
            atomic_json(path / "uncertainty.json", uncertainty)
            atomic_json(path / "sensitivity.json", sensitivity)
            atomic_json(
                path / "audit.json",
                {
                    "self_support_rows": int(self_match.sum()),
                    "note": "Self-support exclusion changes scoring rows only; "
                    "it is a sensitivity diagnostic.",
                },
            )
            atomic_bytes(path / "oof.csv", oof.to_csv(index=False).encode())

        path = stage(directory, "review", review)
        for name in PUBLIC_FILES:
            atomic_bytes(root / "reports/sensitivity" / name, (path / name).read_bytes())
        atomic_json(
            root / "reports/sensitivity/metadata.json",
            {
                "schema": 1,
                "data_kind": "competition",
                "run_id": directory.name,
                **identity,
                "files": {name: digest(path / name) for name in PUBLIC_FILES},
            },
        )
        log.emit("FEATURE_DIAGNOSTICS_COMPLETED", run_id=directory.name, fitted_models=55)
        return directory


def diagnostic_evidence(root: Path) -> dict | None:
    path = root / "reports/sensitivity/metadata.json"
    if not path.exists():
        return None
    metadata = json.loads(path.read_text())
    broad = research_evidence(root)
    if (
        broad is None
        or metadata.get("source_sha256") != digest(Path(__file__))
        or metadata.get("broad_run") != broad["metadata"]["run_id"]
        or metadata.get("data_kind") != "competition"
        or set(metadata.get("files", {})) != set(PUBLIC_FILES)
    ):
        raise ValueError("Stale feature sensitivity evidence")
    result = {"metadata": metadata}
    for name, sha in metadata["files"].items():
        if digest(path.parent / name) != sha:
            raise ValueError("Feature sensitivity checksum mismatch")
        result[name.removesuffix(".json")] = json.loads((path.parent / name).read_text())
    return result
