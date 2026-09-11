from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

PROMPTS = ("rule", "support", "rule_support")
TRANSFORMS = ("rule", "support", "rule_support", "joint_minus_rule", "joint_minus_support")
MODES = ("centroid", "whitened", "hard_negative")


def _l2_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("expected finite 2D vectors")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= eps):
        raise ValueError("zero-norm vector")
    return values / norms


def aligned_vectors(frame: pd.DataFrame, row_ids: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim != 3 or vectors.shape[1] != len(PROMPTS):
        raise ValueError("expected row x 3 prompts x dimensions")
    if len(row_ids) != len(vectors):
        raise ValueError("row/vector length mismatch")
    lookup = {int(row_id): i for i, row_id in enumerate(np.asarray(row_ids).tolist())}
    wanted = frame["row_id"].astype(int).tolist()
    if len(lookup) != len(row_ids) or any(row_id not in lookup for row_id in wanted):
        raise ValueError("row-id alignment failed")
    out = vectors[[lookup[row_id] for row_id in wanted]]
    if not np.isfinite(out).all():
        raise ValueError("nonfinite cached vectors")
    return out.astype(np.float64, copy=False)


def transform_vectors(vectors: np.ndarray, name: str) -> np.ndarray:
    if name == "rule":
        return vectors[:, 0]
    if name == "support":
        return vectors[:, 1]
    if name == "rule_support":
        return vectors[:, 2]
    if name == "joint_minus_rule":
        return vectors[:, 2] - vectors[:, 0]
    if name == "joint_minus_support":
        return vectors[:, 2] - vectors[:, 1]
    raise ValueError(f"unknown transform: {name}")


def polarity_score(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    *,
    mode: str,
    variance_floor: float = 1e-4,
) -> np.ndarray:
    tx = _l2_rows(train_x)
    vx = _l2_rows(val_x)
    y = np.asarray(train_y, dtype=np.int64)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("training fold requires both classes")
    if mode == "centroid":
        direction = tx[y == 1].mean(axis=0) - tx[y == 0].mean(axis=0)
    elif mode == "whitened":
        mu1 = tx[y == 1].mean(axis=0)
        mu0 = tx[y == 0].mean(axis=0)
        pooled = np.var(tx[y == 1], axis=0) + np.var(tx[y == 0], axis=0)
        direction = (mu1 - mu0) / (pooled + variance_floor)
    elif mode == "hard_negative":
        sim = tx @ tx.T
        np.fill_diagonal(sim, -np.inf)
        weight = np.ones(len(y), dtype=np.float64)
        for cls in (0, 1):
            idx = np.flatnonzero(y == cls)
            opp = np.flatnonzero(y != cls)
            hardness = sim[np.ix_(idx, opp)].max(axis=1)
            weight[idx] = 1.0 + np.maximum(hardness, 0.0)
        mu1 = np.average(tx[y == 1], axis=0, weights=weight[y == 1])
        mu0 = np.average(tx[y == 0], axis=0, weights=weight[y == 0])
        direction = mu1 - mu0
    else:
        raise ValueError(f"unknown mode: {mode}")
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("degenerate polarity direction")
    scores = vx @ (direction / norm)
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite polarity score")
    return scores


def whole_rule_folds(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = frame["rule"].astype(str).to_numpy()
    n = len(np.unique(groups))
    if n < 2:
        raise ValueError("need at least two unique rules")
    return list(GroupKFold(n_splits=n).split(np.zeros(len(frame)), frame["rule_violation"], groups))


def evaluate(
    frame: pd.DataFrame,
    row_ids: np.ndarray,
    scores: np.ndarray,
    vectors: np.ndarray,
) -> dict:
    required = {"row_id", "rule", "rule_violation"}
    if required.difference(frame.columns):
        raise ValueError("missing required training columns")
    if scores.shape[:2] != (len(row_ids), len(PROMPTS)):
        raise ValueError("score schema mismatch")
    aligned = aligned_vectors(frame, row_ids, vectors)
    score_lookup = {int(row_id): i for i, row_id in enumerate(np.asarray(row_ids).tolist())}
    frozen_joint = np.asarray(
        [scores[score_lookup[int(r)], 2, 0] for r in frame.row_id],
        dtype=float,
    )
    y = frame["rule_violation"].to_numpy(dtype=int)
    folds = whole_rule_folds(frame)
    results: dict[str, list[dict]] = {"frozen_joint": []}
    for transform in TRANSFORMS:
        for mode in MODES:
            results[f"{transform}/{mode}"] = []
    for fold_idx, (tr, va) in enumerate(folds):
        train_rules = set(frame.iloc[tr]["rule"].astype(str))
        val_rules = set(frame.iloc[va]["rule"].astype(str))
        if train_rules & val_rules:
            raise AssertionError("rule leakage")
        baseline_auc = float(roc_auc_score(y[va], frozen_joint[va]))
        results["frozen_joint"].append(
            {
                "fold": fold_idx,
                "auc": baseline_auc,
                "validation_rules": sorted(val_rules),
            }
        )
        for transform in TRANSFORMS:
            x = transform_vectors(aligned, transform)
            for mode in MODES:
                pred = polarity_score(x[tr], y[tr], x[va], mode=mode)
                auc = float(roc_auc_score(y[va], pred))
                results[f"{transform}/{mode}"].append(
                    {
                        "fold": fold_idx,
                        "auc": auc,
                        "validation_rules": sorted(val_rules),
                    }
                )
    summary = {}
    baseline_rows = results["frozen_joint"]
    baseline = float(np.mean([r["auc"] for r in baseline_rows]))
    for name, rows in results.items():
        if len(rows) != len(baseline_rows):
            raise ValueError("candidate/baseline fold alignment mismatch")
        mean = float(np.mean([r["auc"] for r in rows]))
        deltas = [
            float(rows[index]["auc"] - baseline_rows[index]["auc"])
            for index in range(len(rows))
        ]
        summary[name] = {
            "mean_auc": mean,
            "delta_vs_frozen_joint": float(mean - baseline),
            "fold_deltas": deltas,
            "fold_wins": int(sum(value > 0 for value in deltas)),
            "folds": rows,
        }
    ranked = sorted(
        (values["mean_auc"], name) for name, values in summary.items() if name != "frozen_joint"
    )
    best_name = ranked[-1][1]
    best = summary[best_name]
    promote = bool(
        best["delta_vs_frozen_joint"] >= 0.005
        and best["fold_wins"] == len(folds)
        and min(best["fold_deltas"]) >= 0.0
    )
    return {
        "schema": 1,
        "method": "cached_frozen_rule_polarity_geometry",
        "rows": int(len(frame)),
        "rules": int(frame["rule"].nunique()),
        "baseline": "frozen_joint",
        "results": summary,
        "best_candidate": best_name,
        "promotion": {
            "minimum_mean_auc_delta": 0.005,
            "require_all_fold_wins": True,
            "no_policy_regression": True,
            "passed": promote,
        },
        "query_targets_read": False,
        "saved_prediction_arrays_read": False,
        "model_calls": 0,
        "gpu": False,
    }
