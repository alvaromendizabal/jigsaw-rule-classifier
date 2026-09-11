from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion

REGIMES = ("drop_conflicts", "majority", "soft")


@dataclass(frozen=True)
class PromotionGate:
    minimum_mean_auc_delta: float = 0.002


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return " ".join(str(value).split())


def make_text(frame: pd.DataFrame) -> pd.Series:
    rule = frame["rule"].map(normalize_text)
    body = frame["body"].map(normalize_text)
    return "RULE: " + rule + "\nBODY: " + body


def collapse_training(frame: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Collapse duplicate (rule, body) labels under one supervision regime.

    The operation is target-local: it only uses labels from the supplied training frame.
    """
    if regime not in REGIMES:
        raise ValueError(f"unknown regime: {regime}")

    required = {"rule", "body", "rule_violation"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    work = frame[["rule", "body", "rule_violation"]].copy()
    work["rule"] = work["rule"].map(normalize_text)
    work["body"] = work["body"].map(normalize_text)
    work["rule_violation"] = work["rule_violation"].astype(float)

    grouped = (
        work.groupby(["rule", "body"], sort=True, dropna=False)["rule_violation"]
        .agg(["mean", "count", "min", "max"])
        .reset_index()
    )
    grouped["is_conflict"] = grouped["min"] != grouped["max"]
    grouped["sample_weight"] = grouped["count"].astype(float)

    if regime == "drop_conflicts":
        grouped = grouped.loc[~grouped["is_conflict"]].copy()
        grouped["target"] = grouped["mean"]
    elif regime == "majority":
        grouped = grouped.loc[grouped["mean"] != 0.5].copy()
        grouped["target"] = (grouped["mean"] > 0.5).astype(float)
    else:
        grouped["target"] = grouped["mean"].astype(float)

    return grouped.reset_index(drop=True)


def grouped_rule_folds(
    frame: pd.DataFrame, n_splits: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic GroupKFold indices with exact rule isolation."""
    groups = frame["rule"].map(normalize_text).to_numpy()
    actual_splits = min(n_splits, len(np.unique(groups)))
    if actual_splits < 2:
        raise ValueError("need at least two unique rules")
    splitter = GroupKFold(n_splits=actual_splits)
    dummy = np.zeros(len(frame), dtype=np.float64)
    return list(splitter.split(dummy, frame["rule_violation"].to_numpy(), groups=groups))


def build_vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=60_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=90_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def required_fold_wins(n_valid_folds: int) -> int:
    """Require both official-rule folds; otherwise require at least 60% of folds."""
    if n_valid_folds < 1:
        raise ValueError("need at least one valid AUC fold")
    if n_valid_folds <= 2:
        return n_valid_folds
    return max(2, math.ceil(0.60 * n_valid_folds))


def evaluate_regimes(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    gate: PromotionGate | None = None,
) -> dict:
    """Compare supervision regimes using identical features and unseen-rule folds."""
    if gate is None:
        gate = PromotionGate()

    required = {"row_id", "body", "rule", "rule_violation"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["rule_violation"] = frame["rule_violation"].astype(int)
    folds = grouped_rule_folds(frame, n_splits=n_splits)
    by_regime: dict[str, list[dict]] = {name: [] for name in REGIMES}
    fold_metadata: list[dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_raw = frame.iloc[train_idx].copy()
        val = frame.iloc[val_idx].copy()
        train_rules = set(train_raw["rule"].map(normalize_text))
        val_rules = set(val["rule"].map(normalize_text))
        if train_rules & val_rules:
            raise AssertionError("rule leakage detected across grouped fold")

        vectorizer = build_vectorizer()
        vectorizer.fit(make_text(train_raw))
        x_val = vectorizer.transform(make_text(val))
        y_val = val["rule_violation"].to_numpy(dtype=int)

        fold_metadata.append(
            {
                "fold": fold_idx,
                "train_rows": int(len(train_raw)),
                "validation_rows": int(len(val)),
                "train_rules": int(len(train_rules)),
                "validation_rules": int(len(val_rules)),
            }
        )

        for regime in REGIMES:
            collapsed = collapse_training(train_raw, regime)
            x_train = vectorizer.transform(make_text(collapsed))
            model = Ridge(alpha=1.0, solver="lsqr", fit_intercept=True)
            model.fit(
                x_train,
                collapsed["target"].to_numpy(dtype=float),
                sample_weight=collapsed["sample_weight"].to_numpy(dtype=float),
            )
            pred = model.predict(x_val)
            by_regime[regime].append(
                {
                    "fold": fold_idx,
                    "auc": safe_auc(y_val, pred),
                    "train_pairs": int(len(collapsed)),
                    "conflicting_pairs_retained": int(collapsed["is_conflict"].sum()),
                }
            )

    summary: dict[str, dict] = {}
    for regime, rows in by_regime.items():
        aucs = [row["auc"] for row in rows if row["auc"] is not None]
        if not aucs:
            raise ValueError(f"no valid AUC folds for regime {regime}")
        summary[regime] = {
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs, ddof=0)),
            "min_auc": float(np.min(aucs)),
            "max_auc": float(np.max(aucs)),
            "folds": rows,
        }

    baseline_rows = by_regime["drop_conflicts"]
    valid_auc_folds = sum(row["auc"] is not None for row in baseline_rows)
    minimum_fold_wins = required_fold_wins(valid_auc_folds)
    baseline = summary["drop_conflicts"]["mean_auc"]

    def candidate_gate(name: str) -> dict:
        delta = summary[name]["mean_auc"] - baseline
        fold_deltas = [
            float(candidate["auc"] - baseline_row["auc"])
            for candidate, baseline_row in zip(by_regime[name], baseline_rows, strict=True)
            if candidate["auc"] is not None and baseline_row["auc"] is not None
        ]
        fold_wins = sum(value > 0.0 for value in fold_deltas)
        return {
            "delta_auc": float(delta),
            "fold_deltas": fold_deltas,
            "fold_wins": int(fold_wins),
            "promote_to_single_policy_gpu_ablation": bool(
                delta >= gate.minimum_mean_auc_delta and fold_wins >= minimum_fold_wins
            ),
        }

    return {
        "schema": 1,
        "method": "grouped_unseen_rule_conflict_supervision_ablation",
        "train_rows": int(len(frame)),
        "unique_rules": int(frame["rule"].map(normalize_text).nunique()),
        "validation": (
            "GroupKFold by exact rule string; the official two-rule training file yields "
            "two whole-rule transfer folds."
        ),
        "regimes": summary,
        "fold_metadata": fold_metadata,
        "gate": {
            "minimum_mean_auc_delta": gate.minimum_mean_auc_delta,
            "valid_auc_folds": int(valid_auc_folds),
            "required_fold_wins": int(minimum_fold_wins),
            "soft": candidate_gate("soft"),
            "majority": candidate_gate("majority"),
            "gpu_training_authorized_by_this_module": False,
        },
        "query_targets_read": False,
        "prediction_arrays_read": False,
        "model_calls": 0,
        "gpu": False,
    }
