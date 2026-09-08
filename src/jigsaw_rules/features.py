"""Task-informed lexical interactions evaluated on unchanged, purged reference splits."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock
from scipy.sparse import csr_matrix, hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from jigsaw_rules.cloud import backup
from jigsaw_rules.data import EXAMPLES, load_data, normalize
from jigsaw_rules.metrics import evaluate
from jigsaw_rules.model import LexicalClassifier
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest, fingerprint, stage
from jigsaw_rules.semantic import reference_splits
from jigsaw_rules.uncertainty import paired_auc_interval

VARIANTS = ("rule_text", "support_contrast", "structure", "combined")
CONTRASTS = (
    "positive_min",
    "positive_max",
    "negative_min",
    "negative_max",
    "maximum_margin",
    "mean_margin",
    "positive_spread",
    "negative_spread",
)
STRUCTURE = (
    "log_characters",
    "log_words",
    "uppercase_fraction",
    "punctuation_fraction",
    "url",
    "question",
    "quotation",
    "negation",
)


def structural_features(frame: pd.DataFrame) -> np.ndarray:
    """Label-free language cues; hypotheses, not a claim to understand intent."""
    text = frame.body
    length = text.str.len().clip(lower=1)
    return np.column_stack(
        [
            np.log1p(length),
            np.log1p(text.str.split().str.len()),
            text.str.count(r"[A-Z]") / length,
            text.str.count(r"[^\w\s]") / length,
            text.str.contains(r"https?://|www\.", case=False, regex=True),
            text.str.contains("?", regex=False),
            text.str.contains(r'["“”]|^>', regex=True),
            text.str.contains(r"\b(?:no|not|never|without)\b|n't", case=False, regex=True),
        ]
    ).astype(np.float64)


class FeatureClassifier:
    """Same training-only word vocabulary across ablations; only dense groups vary."""

    def __init__(self, variant: str, seed: int = 2025):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown feature variant: {variant}")
        self.variant, self.seed = variant, seed
        self.word = LexicalClassifier(seed=seed).vectorizer
        self.character = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            max_features=30000,
            dtype=np.float64,
        )
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(
            C=2.0, solver="liblinear", max_iter=2000, random_state=seed
        )

    def dense(self, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        columns, names = [], []
        for label, vectorizer in (("word", self.word), ("character", self.character)):
            body = vectorizer.transform(frame.body)
            if self.variant in ("rule_text", "combined"):
                columns.append(
                    np.asarray(body.multiply(vectorizer.transform(frame.rule)).sum(axis=1)).ravel()
                )
                names.append(f"{label}_rule_similarity")
            if self.variant in ("support_contrast", "combined"):
                scores = np.column_stack(
                    [
                        np.asarray(
                            body.multiply(vectorizer.transform(frame[c])).sum(axis=1)
                        ).ravel()
                        for c in EXAMPLES
                    ]
                )
                pos, neg = np.sort(scores[:, :2], axis=1), np.sort(scores[:, 2:], axis=1)
                values = np.column_stack(
                    [
                        pos,
                        neg,
                        pos[:, 1] - neg[:, 1],
                        pos.mean(axis=1) - neg.mean(axis=1),
                        pos[:, 1] - pos[:, 0],
                        neg[:, 1] - neg[:, 0],
                    ]
                )
                columns.extend(values.T)
                names.extend(f"{label}_{c}" for c in CONTRASTS)
        if self.variant in ("structure", "combined"):
            columns.extend(structural_features(frame).T)
            names.extend(STRUCTURE)
        return np.column_stack(columns), names

    def fit(self, frame: pd.DataFrame) -> FeatureClassifier:
        # Identical corpus across variants; never fit vocabulary/scaling on held-out rows.
        corpus = [text for c in ["body", "rule", *EXAMPLES] for text in frame[c]]
        self.word.fit(corpus)
        self.character.fit(corpus)
        dense, self.feature_names_ = self.dense(frame)
        features = hstack(
            [self.word.transform(frame.body), csr_matrix(self.scaler.fit_transform(dense))],
            format="csr",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            self.classifier.fit(features, frame.rule_violation)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        dense, _ = self.dense(frame)
        features = hstack(
            [self.word.transform(frame.body), csr_matrix(self.scaler.transform(dense))],
            format="csr",
        )
        return self.classifier.predict_proba(features)[:, 1]


def run_features(root: Path, baseline_run: str, *, cloud: dict | None = None) -> Path:
    """Evaluate four predeclared ablations without refitting the reference or making a CSV."""
    root = Path(root).resolve()
    (root / "runs").mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "runs/features.lock"), timeout=1):
        with Progress(root / "logs/features.jsonl", "feature_experiment") as log:
            train, _, _ = load_data(root / "data/raw")
            splits = reference_splits(root, train, baseline_run)
            synthetic = (root / "data/raw/SYNTHETIC.txt").exists()
            reference_dir = root / "runs" / baseline_run / "review"
            reference_file = reference_dir / "oof.csv"
            marker = json.loads((reference_dir / "complete.json").read_text())
            if marker["files"].get("oof.csv") != digest(reference_file):
                raise ValueError("Reference OOF checksum mismatch")
            reference = pd.read_csv(reference_file)
            config = {
                "experiment": "features",
                "synthetic": synthetic,
                "baseline_run": baseline_run,
                "models": VARIANTS,
                "splits": splits,
                "seed": 2025,
                "reference_oof_sha256": digest(reference_file),
            }
            run_id, provenance = fingerprint(root / "data/raw", config)
            directory = root / "runs" / run_id
            atomic_json(directory / "provenance.json", provenance)
            log.emit("experiment_identified", run_id=run_id, variants=list(VARIANTS))

            def sync() -> None:
                if cloud:
                    backup(root, cloud["bucket"], cloud["region"])

            results, frames, coefficients = [], [], []
            for protocol, assignments in splits.items():
                for variant in VARIANTS:
                    chunks, timings = [], []
                    for fold, assignment in enumerate(assignments):

                        def fit(destination, assignment=assignment, fold=fold, variant=variant):
                            before = time.monotonic()
                            model = FeatureClassifier(variant).fit(train.iloc[assignment["train"]])
                            validation = train.iloc[assignment["valid"]]
                            part = validation[["row_id", "rule", "rule_violation"]].copy()
                            part["probability"], part["fold"] = model.predict(validation), fold
                            atomic_bytes(
                                destination / "predictions.csv", part.to_csv(index=False).encode()
                            )
                            atomic_json(
                                destination / "details.json",
                                {
                                    "seconds": time.monotonic() - before,
                                    "dense_features": model.feature_names_,
                                    "dense_coefficients": model.classifier.coef_[
                                        0, -len(model.feature_names_) :
                                    ].tolist(),
                                    "training_rows": len(assignment["train"]),
                                    "validation_rows": len(assignment["valid"]),
                                    "split_sha256": assignment["source_sha256"],
                                },
                            )

                        completed = stage(directory, f"{protocol}_{variant}_{fold}", fit)
                        chunks.append(pd.read_csv(completed / "predictions.csv"))
                        details = json.loads((completed / "details.json").read_text())
                        timings.append(details["seconds"])
                        coefficients.extend(
                            {
                                "protocol": protocol,
                                "model": variant,
                                "fold": fold,
                                "feature": feature,
                                "coefficient": coefficient,
                            }
                            for feature, coefficient in zip(
                                details["dense_features"],
                                details["dense_coefficients"],
                                strict=True,
                            )
                        )
                        sync()
                    oof = pd.concat(chunks, ignore_index=True)
                    if (
                        len(oof) != len(train)
                        or oof.row_id.duplicated().any()
                        or set(oof.row_id) != set(train.row_id)
                    ):
                        raise ValueError("Expected exactly one OOF prediction per training row")
                    metrics = evaluate(oof.rule_violation, oof.probability, oof.rule)
                    results.append(
                        {
                            "model": variant,
                            "protocol": protocol,
                            "metrics": metrics,
                            "run_id": run_id,
                            "training_rows": len(train),
                            "fit_seconds": sum(timings),
                            "data_kind": "synthetic" if synthetic else "competition",
                        }
                    )
                    oof["model"], oof["protocol"] = variant, protocol
                    frames.append(oof)
                    log.emit(
                        "variant_evaluated",
                        model=variant,
                        protocol=protocol,
                        rule_macro_auc=metrics["rule_macro_auc"],
                    )
            combined = pd.concat(frames, ignore_index=True)

            def review(destination):
                intervals = []
                for record in results:
                    protocol, variant = record["protocol"], record["model"]
                    candidate = (
                        combined[(combined.protocol == protocol) & (combined.model == variant)]
                        .set_index("row_id")
                        .loc[train.row_id]
                    )
                    ref = (
                        reference[
                            (reference.protocol == protocol) & (reference.model == "rule_examples")
                        ]
                        .set_index("row_id")
                        .loc[train.row_id]
                    )
                    if len(ref) != len(train) or (
                        not np.array_equal(ref.rule_violation, train.rule_violation)
                        or not np.array_equal(ref.rule, train.rule)
                    ):
                        raise ValueError("Reference OOF labels are not aligned")
                    intervals.append(
                        {
                            "model": variant,
                            "protocol": protocol,
                            **paired_auc_interval(
                                train.rule_violation,
                                ref.probability,
                                candidate.probability,
                                train.rule,
                                train.body.map(normalize),
                                draws=20 if synthetic else 500,
                            ),
                        }
                    )
                shape = pd.DataFrame(structural_features(train), columns=STRUCTURE)
                shape["rule"], shape["target"] = (
                    train.rule.to_numpy(),
                    train.rule_violation.to_numpy(),
                )
                grouped = shape.groupby(["rule", "target"])
                audit = grouped.mean()
                audit["rows"] = grouped.size()
                audit = audit.reset_index().to_dict("records")
                atomic_json(destination / "results.json", results)
                atomic_json(destination / "uncertainty.json", intervals)
                atomic_json(destination / "audit.json", audit)
                atomic_json(destination / "coefficients.json", coefficients)
                atomic_bytes(destination / "oof.csv", combined.to_csv(index=False).encode())

            stage(directory, "review", review)
            atomic_json(directory / "status.json", {"status": "completed", "synthetic": synthetic})
            sync()
            log.emit(
                "FEATURES_COMPLETED", run_id=run_id, models=len(VARIANTS), protocols=len(splits)
            )
            return directory


def export_features(root: Path, directory: Path) -> None:
    """Publish only fixed aggregate files after rejecting synthetic or mismatched evidence."""
    from jigsaw_rules.review import public_evidence

    provenance = json.loads((directory / "provenance.json").read_text())
    if provenance["config"]["synthetic"]:
        raise ValueError("Only matching competition evidence may be exported")
    baseline = public_evidence(root, "baseline")
    if (
        provenance["data"]["train.csv"] != baseline["training_sha256"]
        or provenance["config"]["baseline_run"] != baseline["run_id"]
    ):
        raise ValueError("Only matching competition evidence may be exported")
    names = ("results.json", "uncertainty.json", "audit.json", "coefficients.json")
    marker = json.loads((directory / "review/complete.json").read_text())
    for name in names:
        if marker["files"].get(name) != digest(directory / "review" / name):
            raise ValueError("Feature aggregate checksum mismatch")
    destination = root / "reports/features"
    for name in names:
        atomic_bytes(destination / name, (directory / "review" / name).read_bytes())
    atomic_json(
        destination / "metadata.json",
        {
            "schema": 1,
            "data_kind": "competition",
            "run_id": directory.name,
            "baseline_run": provenance["config"]["baseline_run"],
            "training_sha256": provenance["data"]["train.csv"],
            "source_sha256": provenance["code"],
            "files": {name: digest(destination / name) for name in names},
            "selection": "Exploratory four-candidate ablation; no automatic model promotion.",
        },
    )


def feature_evidence(root: Path) -> dict | None:
    """Read optional, checksummed aggregate evidence without fitting or private data."""
    path = root / "reports/features/metadata.json"
    if not path.exists():
        return None
    metadata = json.loads(path.read_text())
    if metadata.get("data_kind") != "competition" or set(metadata.get("files", {})) != {
        "results.json",
        "uncertainty.json",
        "audit.json",
        "coefficients.json",
    }:
        raise ValueError("Invalid public feature evidence")
    from jigsaw_rules.review import public_evidence

    if metadata["training_sha256"] != public_evidence(root, "baseline")["training_sha256"]:
        raise ValueError("Feature evidence uses different training data")
    result = {"metadata": metadata}
    for name, sha in metadata["files"].items():
        if digest(path.parent / name) != sha:
            raise ValueError("Feature aggregate checksum mismatch")
        result[name.removesuffix(".json")] = json.loads((path.parent / name).read_text())
    return result
