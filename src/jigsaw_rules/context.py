"""Fold-local metadata, reference distributions, and target-derived diagnostics.

Target statistics cross-fit by normalized comment group and purge inner validation
comments from every reference body's and support example's text. No validation
target is accepted by transform. Unknown categories fall back to the fit prior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from jigsaw_rules.data import EXAMPLES, normalize


def category_keys(frame: pd.DataFrame) -> dict[str, pd.Series]:
    community = frame.subreddit.map(normalize)
    rule = frame.rule.map(normalize)
    # Tuples avoid collisions when a category contains a delimiter.
    pair = pd.Series(list(zip(rule, community, strict=True)), index=frame.index)
    return {"community": community, "rule": rule, "rule_community": pair}


class ContextEncoder:
    """Smoothed diagnostic encodings; never a claim of transportability to new rules."""

    def __init__(self, smoothing: float = 20.0):
        if not np.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("Smoothing must be positive and finite")
        self.smoothing = smoothing

    def fit(self, training: pd.DataFrame) -> ContextEncoder:
        if training.empty or not training.rule_violation.isin([0, 1]).all():
            raise ValueError("Encoding needs nonempty binary training labels")
        self.prior_ = float(training.rule_violation.mean())
        self.rows_ = len(training)
        self.tables_ = {}
        for family, keys in category_keys(training).items():
            table = pd.DataFrame({"key": keys, "target": training.rule_violation})
            self.tables_[family] = table.groupby("key").target.agg(["sum", "count"])
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        if not hasattr(self, "tables_"):
            raise ValueError("Fit context encoder before transforming")
        values, names = [], []
        for family, keys in category_keys(frame).items():
            table = self.tables_[family]
            count = keys.map(table["count"]).fillna(0).to_numpy(dtype=float)
            total = keys.map(table["sum"]).fillna(0).to_numpy(dtype=float)
            posterior = (total + self.smoothing * self.prior_) / (count + self.smoothing)
            values.extend([posterior, np.log1p(count) / np.log1p(self.rows_), count == 0])
            names.extend(f"{family}/{name}" for name in ("posterior", "frequency", "unseen"))
        return np.column_stack(values).astype(float), names

    def fit_transform(self, training: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        groups = training.body.map(normalize)
        if groups.nunique() < 3:
            raise ValueError("At least three comment groups are required for cross-fitting")
        output = np.empty((len(training), 9), dtype=float)
        audits = []
        splitter = GroupKFold(n_splits=3)
        for fit_indices, valid_indices in splitter.split(training, groups=groups):
            forbidden = set(groups.iloc[valid_indices])
            candidates = training.iloc[fit_indices]
            overlap = (
                candidates[["body", *EXAMPLES]]
                .apply(lambda column, forbidden=forbidden: column.map(normalize).isin(forbidden))
                .any(axis=1)
                .to_numpy()
            )
            retained = fit_indices[~overlap]
            if len(retained) == 0:
                raise ValueError("Inner purging left no target-encoding reference rows")
            encoder = ContextEncoder(self.smoothing).fit(training.iloc[retained])
            output[valid_indices], names = encoder.transform(
                training.iloc[valid_indices].drop(columns="rule_violation")
            )
            audits.append(
                {
                    "training_rows": len(retained),
                    "validation_rows": len(valid_indices),
                    "purged_rows": int(overlap.sum()),
                }
            )
        self.fit(training)
        self.inner_audit_ = audits
        return output, names


class ReferenceRanks:
    """Empirical CDFs fitted on training feature values only; ties use midranks."""

    def fit(self, values: np.ndarray) -> ReferenceRanks:
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Ranks need a finite nonempty training matrix")
        self.sorted_ = np.sort(values, axis=0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.sorted_.shape[1]:
            raise ValueError("Rank feature schema differs")
        if not np.isfinite(values).all():
            raise ValueError("Ranks need finite values")
        return np.column_stack(
            [
                (
                    np.searchsorted(self.sorted_[:, j], values[:, j], side="left")
                    + np.searchsorted(self.sorted_[:, j], values[:, j], side="right")
                )
                / (2 * len(self.sorted_))
                for j in range(values.shape[1])
            ]
        )
