"""Competition schemas and deterministic synthetic data for software verification."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

EXAMPLES = ["positive_example_1", "positive_example_2", "negative_example_1", "negative_example_2"]
TEXT = ["body", "rule", "subreddit", *EXAMPLES]
FILES = ["train.csv", "test.csv", "sample_submission.csv"]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def validate_frame(df: pd.DataFrame, *, train: bool) -> None:
    required = ["row_id", *TEXT] + (["rule_violation"] if train else [])
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.empty or df.row_id.isna().any() or df.row_id.duplicated().any():
        raise ValueError("Rows and unique, non-null row_id values are required")
    for column in TEXT:
        if not df[column].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
            raise ValueError(f"Empty or non-text values in {column}")
    if train and (df.rule_violation.isna().any() or not df.rule_violation.isin([0, 1]).all()):
        raise ValueError("Targets must be binary 0/1 with no missing values")
    if not train and "rule_violation" in df:
        raise ValueError("Test data must not contain target labels")


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    if list(submission.columns) != ["row_id", "rule_violation"]:
        raise ValueError("Submission must have exactly row_id,rule_violation")
    if submission.empty or submission.row_id.isna().any() or submission.row_id.duplicated().any():
        raise ValueError("Submission IDs must be unique and non-null")
    if not submission.row_id.equals(sample.row_id):
        raise ValueError("Submission row IDs or order differ from sample")
    p = submission.rule_violation.to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Predictions must be finite probabilities in [0,1]")


def load_data(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for name in FILES:
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Missing {directory / name}; run jigsaw download first")
    train, test, sample = (pd.read_csv(directory / name) for name in FILES)
    validate_frame(train, train=True)
    validate_frame(test, train=False)
    if list(sample.columns) != ["row_id", "rule_violation"]:
        raise ValueError("Unexpected sample submission columns")
    if not test.row_id.equals(sample.row_id):
        raise ValueError("Test and sample submission IDs/order differ")
    validate_submission(sample, sample)
    if set(train.row_id) & set(test.row_id):
        raise ValueError("Train and test row IDs overlap")
    return train, test, sample


def audit(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    train_bodies = set(train.body.map(normalize))
    test_bodies = set(test.body.map(normalize))
    return {
        "train_rows": len(train),
        "preview_test_rows": len(test),
        "train_rules": sorted(train.rule.unique().tolist()),
        "preview_test_rules": sorted(test.rule.unique().tolist()),
        "duplicate_training_bodies": int(train.body.map(normalize).duplicated().sum()),
        "train_test_body_overlap": len(train_bodies & test_bodies),
        "label_prevalence": float(train.rule_violation.mean()),
        "by_rule": train.groupby("rule")
        .rule_violation.agg(["size", "mean"])
        .reset_index()
        .to_dict("records"),
        "test_note": "Downloaded test is a preview; hidden evaluation can replace it.",
    }


def synthetic(directory: Path) -> None:
    """Tiny authored examples; never use these metrics as competition performance."""
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / f).exists() for f in FILES):
        raise FileExistsError("Synthetic generation refuses to overwrite existing data")
    rows = []
    for r, rule in enumerate(["No advertisements", "No personal insults"]):
        for i in range(48):
            label = i % 2
            phrase = (
                ["buy discount offer", "you stupid fool"][r]
                if label
                else "thoughtful topic discussion"
            )
            rows.append(
                {
                    "row_id": r * 100 + i,
                    "body": f"{phrase} uniqueitem{r}x{i}",
                    "rule": rule,
                    "subreddit": f"community{i % 4}",
                    "positive_example_1": "buy discount coupon" if r == 0 else "you foolish idiot",
                    "positive_example_2": "sale offer now" if r == 0 else "stupid personal insult",
                    "negative_example_1": "thank you for this thoughtful discussion",
                    "negative_example_2": "interesting topic worth discussing",
                    "rule_violation": label,
                }
            )
    train = pd.DataFrame(rows)
    test = train.iloc[:8].drop(columns="rule_violation").copy()
    test["row_id"] = np.arange(1000, 1008)
    test["body"] = test.body + " unseen"
    sample = pd.DataFrame({"row_id": test.row_id, "rule_violation": 0.5})
    for frame, name in zip([train, test, sample], FILES, strict=True):
        frame.to_csv(directory / name, index=False)
    (directory / "SYNTHETIC.txt").write_text("SOFTWARE TEST DATA. NOT COMPETITION RESULTS.\n")
