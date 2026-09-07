"""Competition-oriented per-rule macro AUC plus probability diagnostics."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def evaluate(y, probabilities, rules) -> dict:
    y = np.asarray(y)
    p = np.asarray(probabilities, dtype=float)
    rules = np.asarray(rules)
    if y.ndim != 1 or p.shape != y.shape or rules.shape != y.shape or len(y) == 0:
        raise ValueError("Labels, probabilities, and rules must be aligned 1-D arrays")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid labels or probabilities")
    scores = {}
    for rule in sorted(set(rules)):
        mask = rules == rule
        if len(set(y[mask])) != 2:
            raise ValueError(f"AUC undefined for single-class rule: {rule}")
        scores[str(rule)] = float(roc_auc_score(y[mask], p[mask]))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, p >= 0.5, average="binary", zero_division=0
    )
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(
        float(np.mean(bins == b) * abs(y[bins == b].mean() - p[bins == b].mean()))
        for b in range(10)
        if np.any(bins == b)
    )
    return {
        "rule_macro_auc": float(np.mean(list(scores.values()))),
        "per_rule_auc": scores,
        "pooled_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece_10_equal_width_bins": ece,
        "precision_at_0_5": float(precision),
        "recall_at_0_5": float(recall),
        "f1_at_0_5": float(f1),
        "confusion_matrix_at_0_5": confusion_matrix(y, p >= 0.5).tolist(),
        "threshold_note": "0.5 is a fixed diagnostic threshold, not tuned on validation labels.",
    }


def calibration_table(y, p) -> pd.DataFrame:
    frame = pd.DataFrame({"label": y, "probability": p})
    frame["bin"] = np.minimum((frame.probability * 10).astype(int), 9)
    return (
        frame.groupby("bin")
        .agg(
            mean_probability=("probability", "mean"),
            observed_rate=("label", "mean"),
            count=("label", "size"),
        )
        .reset_index()
    )
