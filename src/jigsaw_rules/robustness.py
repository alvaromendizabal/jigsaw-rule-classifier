"""Predeclared near-copy isolation stress test; no validation labels select removals."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from filelock import FileLock
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from jigsaw_rules.data import EXAMPLES, load_data, normalize
from jigsaw_rules.embeddings import content_key
from jigsaw_rules.metrics import evaluate
from jigsaw_rules.model import LexicalClassifier
from jigsaw_rules.research import research_evidence
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest, environment, stage
from jigsaw_rules.semantic import reference_splits
from jigsaw_rules.uncertainty import paired_auc_comparisons

POLICY = {"character_cosine": 0.95, "token_jaccard": 0.90, "minimum_characters": 40}
PUBLIC_FILES = ("results.json", "audit.json", "uncertainty.json")


def near_copy_edges(left, right, policy=POLICY) -> tuple[np.ndarray, np.ndarray]:
    """Conservative approximate copy detector with a label-free, fixed vocabulary.

    Hash collisions can create candidates; exact token-set verification removes them.
    This deliberately does not claim to detect every paraphrase or common origin.
    """
    left, right = [normalize(x) for x in left], [normalize(x) for x in right]
    hasher = HashingVectorizer(
        analyzer="char", ngram_range=(3, 5), n_features=2**18, alternate_sign=False
    )
    a, b = hasher.transform(left), hasher.transform(right)
    left_tokens, right_tokens = [set(x.split()) for x in left], [set(x.split()) for x in right]
    rows, columns = [], []
    for start in range(0, len(left), 128):
        similarities = (a[start : start + 128] @ b.T).tocoo()
        selected = similarities.data >= policy["character_cosine"]
        for i, j in zip(
            similarities.row[selected] + start, similarities.col[selected], strict=True
        ):
            if min(len(left[i]), len(right[j])) < policy["minimum_characters"]:
                continue
            union = left_tokens[i] | right_tokens[j]
            if (
                len(left_tokens[i] & right_tokens[j]) / max(1, len(union))
                >= policy["token_jaccard"]
            ):
                rows.append(i)
                columns.append(j)
    return np.asarray(rows, dtype=int), np.asarray(columns, dtype=int)


def purge_near(training: pd.DataFrame, validation: pd.DataFrame):
    corpus = [text for column in ["body", *EXAMPLES] for text in training[column]]
    rows, columns = near_copy_edges(corpus, validation.body)
    excluded = np.unique(rows % len(training))
    keep = np.ones(len(training), dtype=bool)
    keep[excluded] = False
    return training.iloc[np.flatnonzero(keep)], np.unique(columns)


def run_robustness(root: Path) -> Path:
    root = root.resolve()
    evidence = research_evidence(root)
    if evidence is None:
        raise ValueError("Robustness needs verified broad-research evidence")
    baseline_run = evidence["metadata"]["baseline_run"]
    train, _, _ = load_data(root / "data/raw")
    identity = {
        "source_sha256": digest(Path(__file__)),
        "broad_run": evidence["metadata"]["run_id"],
        "training_sha256": digest(root / "data/raw/train.csv"),
        "policy": POLICY,
        "environment": environment(),
    }
    # Platform and observation time are provenance, not experiment identity.
    contract = {key: value for key, value in identity.items() if key != "environment"}
    contract["packages"] = identity["environment"]["packages"]
    directory = root / "runs" / content_key(contract)[:20]
    with (
        FileLock(str(root / "runs/robustness.lock"), timeout=1),
        Progress(root / "logs/robustness.jsonl", "near_copy_stress_test") as log,
    ):
        atomic_json(directory / "provenance.json", identity)
        parts, audits = [], []
        for protocol, assignments in reference_splits(root, train, baseline_run).items():
            for fold, assignment in enumerate(assignments):
                training, validation = (
                    train.iloc[assignment["train"]],
                    train.iloc[assignment["valid"]],
                )

                def fit(
                    path, training=training, validation=validation, protocol=protocol, fold=fold
                ):
                    retained, affected = purge_near(training, validation)
                    if retained.rule_violation.nunique() != 2:
                        raise ValueError("Near-copy purge removes a training class")
                    audit = {
                        "protocol": protocol,
                        "fold": fold,
                        "before": len(training),
                        "after": len(retained),
                        "removed": len(training) - len(retained),
                        "validation_rows": len(validation),
                        "affected_validation_rows": len(affected),
                    }
                    result = []
                    for name in ("reference_near_purged", "character_near_purged"):
                        if name.startswith("reference"):
                            model = LexicalClassifier().fit(retained)
                            probability = model.predict(validation)
                        else:
                            vectorizer = TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                max_features=60000,
                                sublinear_tf=True,
                                dtype=np.float64,
                            )
                            corpus = [
                                text
                                for column in ["body", "rule", *EXAMPLES]
                                for text in retained[column]
                            ]
                            vectorizer.fit(corpus)
                            classifier = LogisticRegression(
                                C=2, solver="liblinear", max_iter=2000, random_state=2025
                            )
                            classifier.fit(
                                vectorizer.transform(retained.body), retained.rule_violation
                            )
                            probability = classifier.predict_proba(
                                vectorizer.transform(validation.body)
                            )[:, 1]
                            model = (vectorizer, classifier)
                        part = validation[["row_id", "rule", "rule_violation"]].copy()
                        part["probability"], part["model"] = probability, name
                        part["protocol"], part["fold"] = protocol, fold
                        result.append(part)
                        joblib.dump(model, path / f"{name}.joblib", compress=3)
                    atomic_bytes(
                        path / "predictions.csv", pd.concat(result).to_csv(index=False).encode()
                    )
                    atomic_json(path / "audit.json", audit)

                path = stage(directory, f"{protocol}_{fold}", fit)
                parts.append(pd.read_csv(path / "predictions.csv"))
                audits.append(json.loads((path / "audit.json").read_text()))
        oof = pd.concat(parts, ignore_index=True)
        baseline = pd.read_csv(root / f"runs/{baseline_run}/review/oof.csv")
        baseline = baseline[baseline.model == "rule_examples"].copy()
        baseline["model"] = "original_reference"
        predictions = pd.concat([oof, baseline], ignore_index=True)
        results, uncertainty = [], []
        for protocol in ("seen_rule", "heldout_rule"):
            bank = {}
            for model in predictions.model.unique():
                part = predictions[
                    (predictions.protocol == protocol) & (predictions.model == model)
                ]
                if part.row_id.duplicated().any() or set(part.row_id) != set(train.row_id):
                    raise ValueError("Near-copy OOF coverage differs")
                part = part.set_index("row_id").loc[train.row_id]
                bank[model] = part.probability.to_numpy()
                results.append(
                    {
                        "model": model,
                        "protocol": protocol,
                        "metrics": evaluate(train.rule_violation, part.probability, train.rule),
                    }
                )
            contrasts = [(name, name, "original_reference") for name in oof.model.unique()]
            uncertainty.extend(
                {"protocol": protocol, **item}
                for item in paired_auc_comparisons(
                    train.rule_violation, bank, train.rule, train.body.map(normalize), contrasts
                )
            )

        def review(path):
            atomic_json(path / "results.json", results)
            atomic_json(
                path / "audit.json",
                {
                    "policy": POLICY,
                    "folds": audits,
                    "interpretation": "Additional training-row removal only; "
                    "original validation rows retained. Fixed thresholds were not selected "
                    "with outcomes. This is not paraphrase isolation.",
                },
            )
            atomic_json(path / "uncertainty.json", uncertainty)
            atomic_bytes(path / "oof.csv", oof.to_csv(index=False).encode())

        path = stage(directory, "review", review)
        for name in PUBLIC_FILES:
            atomic_bytes(root / "reports/robustness" / name, (path / name).read_bytes())
        atomic_json(
            root / "reports/robustness/metadata.json",
            {
                "schema": 1,
                "data_kind": "competition",
                "run_id": directory.name,
                **identity,
                "files": {name: digest(path / name) for name in PUBLIC_FILES},
            },
        )
        log.emit("NEAR_COPY_STRESS_COMPLETED", run_id=directory.name, fitted_models=10)
    return directory


def robustness_evidence(root: Path) -> dict | None:
    path = root / "reports/robustness/metadata.json"
    if not path.exists():
        return None
    metadata = json.loads(path.read_text())
    broad = research_evidence(root)
    if (
        broad is None
        or metadata.get("source_sha256") != digest(Path(__file__))
        or metadata.get("broad_run") != broad["metadata"]["run_id"]
        or metadata.get("training_sha256") != broad["metadata"]["training_sha256"]
        or metadata.get("data_kind") != "competition"
        or set(metadata.get("files", {})) != set(PUBLIC_FILES)
    ):
        raise ValueError("Stale near-copy evidence")
    result = {"metadata": metadata}
    for name, sha in metadata["files"].items():
        if digest(path.parent / name) != sha:
            raise ValueError("Near-copy evidence checksum mismatch")
        result[name.removesuffix(".json")] = json.loads((path.parent / name).read_text())
    return result
