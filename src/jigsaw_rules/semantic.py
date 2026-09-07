"""Order-invariant example comparisons and strictly fold-fitted semantic classifiers."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from jigsaw_rules.data import EXAMPLES, normalize
from jigsaw_rules.embeddings import format_text
from jigsaw_rules.runtime import digest

FEATURE_NAMES = [
    "positive_min",
    "positive_max",
    "negative_min",
    "negative_max",
    "maximum_margin",
    "mean_margin",
    "positive_spread",
    "negative_spread",
]


def input_texts(frame: pd.DataFrame, spec: dict) -> list[str]:
    return [
        format_text(row.rule, getattr(row, column), spec)
        for row in frame.itertuples()
        for column in ["body", *EXAMPLES]
    ]


def features(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim != 3 or vectors.shape[1] != 5 or not np.isfinite(vectors).all():
        raise ValueError("Expected aligned body and four example vectors")
    similarity = np.einsum("nd,nkd->nk", vectors[:, 0], vectors[:, 1:])
    positive, negative = np.sort(similarity[:, :2], axis=1), np.sort(similarity[:, 2:], axis=1)
    return np.column_stack(
        [
            positive,
            negative,
            positive[:, 1] - negative[:, 1],
            positive.mean(axis=1) - negative.mean(axis=1),
            positive[:, 1] - positive[:, 0],
            negative[:, 1] - negative[:, 0],
        ]
    ).astype(np.float64)


def classifier(spec: dict):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=spec["classifier_c"], max_iter=2000, random_state=spec["seed"], solver="lbfgs"
        ),
    )


def reference_splits(root: Path, train: pd.DataFrame, baseline_run: str) -> dict:
    """Load exact saved assignments and recheck row coverage and text isolation."""
    if not re.fullmatch(r"[0-9a-f]{20}", baseline_run):
        raise ValueError("Invalid baseline run fingerprint")
    directory = root / "runs" / baseline_run
    provenance = json.loads((directory / "provenance.json").read_text())
    if provenance["data"]["train.csv"] != digest(root / "data/raw/train.csv"):
        raise ValueError("Training data differs from the reference experiment")
    lookup = {row_id: i for i, row_id in enumerate(train.row_id)}
    splits = {}
    for protocol in ["seen_rule", "heldout_rule"]:
        paths = sorted(directory.glob(f"{protocol}_rule_examples_*/split.json"))
        if not paths:
            raise FileNotFoundError("Restore the complete baseline split artifacts first")
        seen, body_fold, records = [], {}, []
        for fold, path in enumerate(paths):
            marker = json.loads((path.parent / "complete.json").read_text())
            if marker["files"].get("split.json") != digest(path):
                raise ValueError("Reference split checksum mismatch")
            record = json.loads(path.read_text())
            ti, vi = record["train_row_ids"], record["valid_row_ids"]
            if len(set(ti)) != len(ti) or len(set(vi)) != len(vi) or set(ti) & set(vi):
                raise ValueError("Reference split has duplicate or overlapping rows")
            if not set(ti + vi).issubset(lookup):
                raise ValueError("Reference split contains unknown row IDs")
            training, valid = (
                train.iloc[[lookup[x] for x in ti]],
                train.iloc[[lookup[x] for x in vi]],
            )
            forbidden = set(valid.body.map(normalize))
            if any(
                set(training[column].map(normalize)) & forbidden for column in ["body", *EXAMPLES]
            ):
                raise ValueError("Reference split leaks validation text into training")
            if protocol == "heldout_rule" and set(training.rule) & set(valid.rule):
                raise ValueError("Held-out rule appears in training")
            if protocol == "seen_rule":
                for body in forbidden:
                    if body in body_fold and body_fold[body] != fold:
                        raise ValueError("Duplicate body assigned to different folds")
                    body_fold[body] = fold
            seen.extend(vi)
            records.append(
                {
                    "train": [lookup[x] for x in ti],
                    "valid": [lookup[x] for x in vi],
                    "source_sha256": digest(path),
                    "source": str(path.relative_to(root)),
                }
            )
        if len(seen) != len(train) or set(seen) != set(train.row_id):
            raise ValueError("Reference folds do not cover every row exactly once")
        splits[protocol] = records
    return splits
