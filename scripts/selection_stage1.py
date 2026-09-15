"""Nested grouped selection over completed Jigsaw readouts; no new feature engineering."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import signal
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

CONFIG = "configs/selection_stage1.json"
COMBO_PUBLIC = "reports/feature_combinations"
COMBO_PRIVATE = "runs/feature_combinations"
PUBLIC = "reports/selection_stage1"
PRIVATE = "runs/selection_stage1"
NOTEBOOK = "notebooks/24_feature_selection_stability.ipynb"
CONTROLS = ("qwen_raw", "answer_only", "frozen_basic", "context_evidence", "uniform_all")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".stage1-", delete=False) as stream:
        temp = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_json(path: Path, value) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def _json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _metric(y: np.ndarray, score: np.ndarray) -> dict:
    probability = expit(score)
    brier = float(brier_score_loss(y, probability))
    return {
        "auc": float(roc_auc_score(y, score)),
        "brier": brier,
        "probability_rmse": float(math.sqrt(brier)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
    }


def _summary(y: np.ndarray, score: np.ndarray, policy: np.ndarray) -> dict:
    per = []
    for policy_id in sorted(np.unique(policy)):
        mask = policy == policy_id
        if len(np.unique(y[mask])) != 2:
            raise ValueError("Policy split lacks both classes")
        per.append(_metric(y[mask], score[mask]))
    pooled = _metric(y, score)
    macro_brier = float(np.mean([row["brier"] for row in per]))
    return {
        "macro_auc": float(np.mean([row["auc"] for row in per])),
        "pooled_auc": pooled["auc"],
        "brier": macro_brier,
        "probability_rmse": float(math.sqrt(macro_brier)),
        "log_loss": float(np.mean([row["log_loss"] for row in per])),
        "advertising_auc": per[0]["auc"],
        "legal_advice_auc": per[1]["auc"],
    }


def _splitter(n_splits: int, seed: int) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _assert_split(y, policy, groups, train, valid) -> None:
    if set(groups[train]) & set(groups[valid]):
        raise AssertionError("Normalized-body group leakage")
    for indices in (train, valid):
        for policy_id in sorted(np.unique(policy)):
            mask = indices[policy[indices] == policy_id]
            if len(mask) == 0 or len(np.unique(y[mask])) != 2:
                raise ValueError("Grouped split lost a policy/class stratum")


def _load(root: Path, config: dict):
    result_path = root / COMBO_PUBLIC / "results.json"
    review_path = root / COMBO_PUBLIC / "review.json"
    if not result_path.is_file() or not review_path.is_file():
        raise ValueError("Completed combination evidence missing")
    result = json.loads(result_path.read_text())
    review = json.loads(review_path.read_text())
    if (
        result.get("status") != config["combination_status"]
        or result.get("run_id") != config["combination_run_id"]
        or review.get("run_id") != config["combination_run_id"]
        or len(result.get("completed_variants", [])) != 49
    ):
        raise ValueError("Combination identity/status changed")
    if digest(root / "data/raw/train.csv") != config["raw_sha256"]:
        raise ValueError("Raw training hash changed")
    run_id = result["run_id"]
    inputs = root / COMBO_PRIVATE / run_id / "inputs"
    names = list(CONTROLS) + list(result["completed_variants"])
    matrices, labels, policies, groups, row_ids = [], [], [], [], []
    for fold in range(2):
        path = inputs / f"fold_{fold}.npz"
        if not path.is_file():
            raise ValueError("Prepared combination inputs missing")
        with np.load(path, allow_pickle=False) as data:
            y = data["query_y"].copy().astype(int)
            ids = data["row_ids"].copy()
            # These group IDs were constructed globally across both policy bundles.
            body_groups = data["bootstrap_groups"].copy()
            controls = {name: data["control_" + name].copy() for name in CONTROLS}
        columns = []
        for name in names:
            if name in controls:
                score = controls[name]
            else:
                candidate = (
                    root / COMBO_PRIVATE / run_id / f"fold_{fold}" / name / "predictions.npz"
                )
                if not candidate.is_file():
                    raise ValueError("Missing completed readout: " + name)
                with np.load(candidate, allow_pickle=False) as saved:
                    if not np.array_equal(saved["row_ids"], ids):
                        raise ValueError("Readout row order changed: " + name)
                    score = saved["score"].copy()
            if score.shape != y.shape or not np.isfinite(score).all():
                raise ValueError("Invalid readout: " + name)
            columns.append(score)
        matrices.append(np.column_stack(columns))
        labels.append(y)
        policies.append(np.full(len(y), fold, dtype=int))
        groups.extend([str(int(value)) for value in body_groups])
        row_ids.append(ids)
    X = np.vstack(matrices)
    y = np.concatenate(labels)
    policy = np.concatenate(policies)
    groups = np.asarray(groups, dtype=object)
    ids = np.concatenate(row_ids)
    if len(X) != 881 or X.shape[1] != len(names):
        raise ValueError("Expected fixed 881-row combination cohort")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("Query row IDs are not unique")
    keep, duplicates, seen = [], {}, {}
    for index, name in enumerate(names):
        values = np.ascontiguousarray(X[:, index], dtype=np.float64)
        fingerprint = hashlib.sha256(values.tobytes()).hexdigest()
        if fingerprint in seen and np.array_equal(values, X[:, seen[fingerprint]]):
            duplicates[name] = names[seen[fingerprint]]
        else:
            seen[fingerprint] = index
            keep.append(index)
    return (
        X[:, keep],
        y,
        policy,
        groups,
        ids,
        [names[index] for index in keep],
        duplicates,
        result,
        review,
    )


def _linear_shap_abs(model, x_train, x_valid):
    """Exact interventional linear SHAP magnitude for an independent background."""
    background = np.asarray(x_train).mean(axis=0)
    coefficients = model.coef_.ravel()
    return np.mean(np.abs((np.asarray(x_valid) - background) * coefficients), axis=0)


def _count_choice(rows):
    frame = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    best_auc = frame.loc[frame["mean_auc"].idxmax()]
    auc_threshold = float(best_auc["mean_auc"] - best_auc["se_auc"])
    auc_eligible = frame[frame["mean_auc"] >= auc_threshold].copy()
    best_brier = auc_eligible.loc[auc_eligible["mean_brier"].idxmin()]
    brier_threshold = float(best_brier["mean_brier"] + best_brier["se_brier"])
    guarded = auc_eligible[auc_eligible["mean_brier"] <= brier_threshold]
    if guarded.empty:
        guarded = auc_eligible
    chosen = int(guarded.sort_values("k").iloc[0]["k"])
    return chosen, auc_threshold, brier_threshold


def _cluster_representatives(X, order, threshold):
    if X.shape[1] <= 1:
        return [int(value) for value in order]
    correlation = np.asarray(spearmanr(X, axis=0).statistic, dtype=float)
    if correlation.ndim == 0:
        return [int(value) for value in order]
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    representatives = []
    for index in order:
        index = int(index)
        if not any(abs(correlation[index, prior]) >= threshold for prior in representatives):
            representatives.append(index)
    return representatives


def _inner_rank(X, y, policy, groups, names, config, seed):
    strata = policy * 2 + y
    splits = list(_splitter(config["inner_splits"], seed).split(X, strata, groups))
    shap_rows, permutation_rows = [], []
    for split_id, (train, valid) in enumerate(splits):
        _assert_split(y, policy, groups, train, valid)
        scaler = StandardScaler().fit(X[train])
        x_train = scaler.transform(X[train])
        x_valid = scaler.transform(X[valid])
        model = LogisticRegression(
            C=1.0,
            solver="liblinear",
            max_iter=2000,
            random_state=seed + split_id,
        ).fit(x_train, y[train])
        shap_rows.append(_linear_shap_abs(model, x_train, x_valid))
        baseline = _summary(y[valid], model.decision_function(x_valid), policy[valid])["macro_auc"]
        rng = np.random.default_rng(seed + split_id)
        drops = []
        for column in range(X.shape[1]):
            repeats = []
            for _ in range(config["permutation_repeats"]):
                permuted = x_valid.copy()
                order = rng.permutation(len(permuted))
                permuted[:, column] = permuted[order, column]
                value = _summary(y[valid], model.decision_function(permuted), policy[valid])[
                    "macro_auc"
                ]
                repeats.append(baseline - value)
            drops.append(float(np.mean(repeats)))
        permutation_rows.append(drops)
    shap_values = np.mean(shap_rows, axis=0)
    permutation_values = np.mean(permutation_rows, axis=0)
    shap_rank = pd.Series(shap_values).rank(method="average", ascending=False).to_numpy()
    permutation_rank = (
        pd.Series(permutation_values).rank(method="average", ascending=False).to_numpy()
    )
    raw_order = np.argsort(shap_rank + permutation_rank, kind="stable")
    representatives = _cluster_representatives(X, raw_order, config["correlation_threshold"])
    grid = sorted(
        {
            min(len(representatives), int(k))
            for k in config["k_grid"]
            if min(len(representatives), int(k)) > 0
        }
    )
    rows = []
    for k in grid:
        aucs, briers, log_losses = [], [], []
        selected = representatives[:k]
        for train, valid in splits:
            scaler = StandardScaler().fit(X[train][:, selected])
            x_train = scaler.transform(X[train][:, selected])
            x_valid = scaler.transform(X[valid][:, selected])
            model = LogisticRegression(
                C=1.0,
                solver="liblinear",
                max_iter=2000,
                random_state=seed,
            ).fit(x_train, y[train])
            metrics = _summary(y[valid], model.decision_function(x_valid), policy[valid])
            aucs.append(metrics["macro_auc"])
            briers.append(metrics["brier"])
            log_losses.append(metrics["log_loss"])
        n = len(aucs)
        rows.append(
            {
                "k": int(k),
                "mean_auc": float(np.mean(aucs)),
                "se_auc": float(np.std(aucs, ddof=1) / math.sqrt(n)),
                "mean_brier": float(np.mean(briers)),
                "se_brier": float(np.std(briers, ddof=1) / math.sqrt(n)),
                "mean_probability_rmse": float(math.sqrt(np.mean(briers))),
                "mean_log_loss": float(np.mean(log_losses)),
            }
        )
    chosen_k, auc_threshold, brier_threshold = _count_choice(rows)
    return {
        "representatives": representatives,
        "shap": shap_values,
        "permutation": permutation_values,
        "k_rows": rows,
        "chosen_k": chosen_k,
        "auc_one_se_threshold": auc_threshold,
        "brier_one_se_threshold": brier_threshold,
    }


def preflight_inputs(root: Path) -> dict:
    config = json.loads((root / CONFIG).read_text())
    X, y, policy, groups, ids, names, duplicates, result, review = _load(root, config)
    strata = policy * 2 + y
    outer = list(_splitter(config["outer_splits"], config["seed"]).split(X, strata, groups))
    for fold, (train, valid) in enumerate(outer):
        _assert_split(y, policy, groups, train, valid)
        inner_strata = policy[train] * 2 + y[train]
        inner = _splitter(config["inner_splits"], config["seed"] + fold)
        for inner_train, inner_valid in inner.split(X[train], inner_strata, groups[train]):
            _assert_split(y[train], policy[train], groups[train], inner_train, inner_valid)
    return {
        "status": "SELECTION_STAGE1_INPUTS_VERIFIED",
        "combination_run_id": result["run_id"],
        "rows": int(len(y)),
        "readouts_before_exact_dedup": int(len(names) + len(duplicates)),
        "readouts_after_exact_dedup": int(len(names)),
        "exact_duplicate_count": int(len(duplicates)),
        "global_body_groups": int(len(np.unique(groups))),
        "outer_splits": config["outer_splits"],
        "inner_splits": config["inner_splits"],
        "first_submission": review["first_submission"],
    }


def _fold_identity(run_id, fold, train, valid, names):
    return {
        "run_id": run_id,
        "fold": fold,
        "train_index_sha256": hashlib.sha256(np.asarray(train, np.int64).tobytes()).hexdigest(),
        "valid_index_sha256": hashlib.sha256(np.asarray(valid, np.int64).tobytes()).hexdigest(),
        "readout_names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
    }


def run_selection(root: Path):
    config = json.loads((root / CONFIG).read_text())
    X, y, policy, groups, ids, names, duplicates, result, review = _load(root, config)
    identity = {
        "config": config,
        "combination_run_id": result["run_id"],
        "combination_results_sha256": digest(root / COMBO_PUBLIC / "results.json"),
        "source_sha256": digest(Path(__file__)),
    }
    run_id = _json_hash(identity)[:20]
    public = root / PUBLIC
    private = root / PRIVATE / run_id
    finished = private / "finished.json"
    output = public / "results.json"
    if finished.is_file() and output.is_file():
        marker = json.loads(finished.read_text())
        if marker.get("identity") != identity or marker.get("results_sha256") != digest(output):
            raise ValueError("Completed selection identity changed")
        print("SELECTION_REPLAY: completed result reused; new outer folds = 0", flush=True)
        return json.loads(output.read_text())
    strata = policy * 2 + y
    outer = list(_splitter(config["outer_splits"], config["seed"]).split(X, strata, groups))
    rows, selected_sets, shap_all, permutation_all = [], [], [], []
    oof = np.full(len(y), np.nan, dtype=float)
    reused = 0
    private.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / PRIVATE / "selection.lock"), timeout=1):
        for fold, (train, valid) in enumerate(outer):
            _assert_split(y, policy, groups, train, valid)
            fold_dir = private / f"fold_{fold}"
            fold_json = fold_dir / "result.json"
            fold_npz = fold_dir / "oof.npz"
            fold_id = _fold_identity(run_id, fold, train, valid, names)
            if fold_json.is_file() and fold_npz.is_file():
                saved = json.loads(fold_json.read_text())
                if saved.get("identity") != fold_id or saved.get("oof_sha256") != digest(fold_npz):
                    raise ValueError("Saved outer-fold checkpoint identity differs")
                with np.load(fold_npz, allow_pickle=False) as data:
                    if not np.array_equal(data["valid_index"], valid):
                        raise ValueError("Saved outer-fold row order differs")
                    score = data["score"].copy()
                if not np.isfinite(score).all():
                    raise ValueError("Saved outer-fold score is invalid")
                reused += 1
                print(f"OUTER_FOLD_REUSED: {fold + 1}/{len(outer)}", flush=True)
            else:
                print(f"OUTER_FOLD_START: {fold + 1}/{len(outer)}", flush=True)
                rank = _inner_rank(
                    X[train],
                    y[train],
                    policy[train],
                    groups[train],
                    names,
                    config,
                    config["seed"] + 100 * fold,
                )
                chosen = rank["representatives"][: rank["chosen_k"]]
                scaler = StandardScaler().fit(X[train][:, chosen])
                x_train = scaler.transform(X[train][:, chosen])
                x_valid = scaler.transform(X[valid][:, chosen])
                model = LogisticRegression(
                    C=1.0,
                    solver="liblinear",
                    max_iter=2000,
                    random_state=config["seed"] + fold,
                ).fit(x_train, y[train])
                score = model.decision_function(x_valid)
                fold_metrics = _summary(y[valid], score, policy[valid])
                saved = {
                    "identity": fold_id,
                    "selected_count": int(len(chosen)),
                    "selected_readouts": [names[index] for index in chosen],
                    **fold_metrics,
                    "inner_k_curve": rank["k_rows"],
                    "auc_one_se_threshold": rank["auc_one_se_threshold"],
                    "brier_one_se_threshold": rank["brier_one_se_threshold"],
                    "shap": rank["shap"].tolist(),
                    "permutation": rank["permutation"].tolist(),
                }
                buffer = io.BytesIO()
                np.savez_compressed(buffer, valid_index=valid, score=score)
                atomic_bytes(fold_npz, buffer.getvalue())
                saved["oof_sha256"] = digest(fold_npz)
                atomic_json(fold_json, saved)
                print(
                    f"OUTER_FOLD_COMPLETE: {fold + 1}/{len(outer)} "
                    f"selected={len(chosen)} macro_auc={fold_metrics['macro_auc']:.6f}",
                    flush=True,
                )
            oof[valid] = score
            rows.append(
                {
                    key: value
                    for key, value in saved.items()
                    if key
                    not in {
                        "identity",
                        "oof_sha256",
                        "shap",
                        "permutation",
                    }
                }
                | {"outer_fold": fold}
            )
            selected_sets.append(set(saved["selected_readouts"]))
            shap_all.append(np.asarray(saved["shap"], dtype=float))
            permutation_all.append(np.asarray(saved["permutation"], dtype=float))
    if not np.isfinite(oof).all():
        raise ValueError("Outer-fold predictions are incomplete")
    frequency = {
        name: sum(name in selected for selected in selected_sets) / len(selected_sets)
        for name in names
    }
    shap_values = np.mean(shap_all, axis=0)
    permutation_values = np.mean(permutation_all, axis=0)
    importance = sorted(
        [
            {
                "readout": name,
                "selection_frequency": float(frequency[name]),
                "mean_abs_linear_shap": float(shap_values[index]),
                "mean_permutation_macro_auc_drop": float(permutation_values[index]),
            }
            for index, name in enumerate(names)
        ],
        key=lambda row: (
            -row["selection_frequency"],
            -row["mean_permutation_macro_auc_drop"],
            -row["mean_abs_linear_shap"],
            row["readout"],
        ),
    )
    median_count = max(1, int(round(float(np.median([row["selected_count"] for row in rows])))))
    shortlist = [
        row["readout"]
        for row in importance
        if row["selection_frequency"] >= config["minimum_selection_frequency"]
    ]
    for row in importance:
        if len(shortlist) >= median_count:
            break
        if row["readout"] not in shortlist:
            shortlist.append(row["readout"])
    shortlist = shortlist[:median_count]
    jaccard = []
    for left in range(len(selected_sets)):
        for right in range(left + 1, len(selected_sets)):
            union = selected_sets[left] | selected_sets[right]
            jaccard.append(
                len(selected_sets[left] & selected_sets[right]) / len(union) if union else 1.0
            )
    macro_brier = float(np.mean([row["brier"] for row in rows]))
    final = {
        "schema": 2,
        "status": "SELECTION_STAGE1_COMPLETE",
        "selection": "SHORTLIST_ONLY_NOT_PROMOTION",
        "run_id": run_id,
        "identity": identity,
        "rows": int(len(y)),
        "readouts_before_exact_dedup": int(len(names) + len(duplicates)),
        "readouts_after_exact_dedup": int(len(names)),
        "exact_duplicates": duplicates,
        "outer_folds": rows,
        "importance": importance,
        "shortlist": shortlist,
        "summary": {
            "macro_auc_mean": float(np.mean([row["macro_auc"] for row in rows])),
            "macro_auc_se": float(
                np.std([row["macro_auc"] for row in rows], ddof=1) / math.sqrt(len(rows))
            ),
            "pooled_auc_mean": float(np.mean([row["pooled_auc"] for row in rows])),
            "brier_mean": macro_brier,
            "probability_rmse": float(math.sqrt(macro_brier)),
            "log_loss_mean": float(np.mean([row["log_loss"] for row in rows])),
            "mean_selection_jaccard": float(np.mean(jaccard)),
            "median_selected_count": median_count,
            "outer_folds_reused_this_invocation": reused,
        },
        "first_submission": review["first_submission"],
        "limitations": [
            (
                "Nested resampling occurs on a repeatedly inspected development cohort, "
                "not a fresh holdout."
            ),
            "Historical feature invention already adapted to these policies.",
            (
                "Selection is over saved model/readout channels; raw token-column pruning "
                "belongs inside Stage 2 model pipelines."
            ),
            (
                "Interventional linear SHAP assumes an independent background; correlated "
                "channels are clustered and held-out macro-AUC permutation importance is "
                "also required."
            ),
            (
                "Probability RMSE equals sqrt(Brier) and is a secondary probability-quality "
                "guard, not an independent objective."
            ),
            "No feature, model, ensemble, or Kaggle submission is promoted automatically.",
        ],
    }
    public.mkdir(parents=True, exist_ok=True)
    atomic_json(output, final)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, row_ids=ids, oof_score=oof)
    atomic_bytes(private / "oof_predictions.npz", buffer.getvalue())
    atomic_json(
        finished,
        {
            "identity": identity,
            "results_sha256": digest(output),
            "oof_sha256": digest(private / "oof_predictions.npz"),
        },
    )
    return final


def figures(result):
    import plotly.graph_objects as go

    rows = pd.DataFrame(result["outer_folds"])
    importance = pd.DataFrame(result["importance"])
    charts = []
    figure = go.Figure()
    figure.add_bar(x=rows.outer_fold, y=rows.advertising_auc, name="Advertising")
    figure.add_bar(x=rows.outer_fold, y=rows.legal_advice_auc, name="Legal advice")
    figure.update_layout(title="01 | Nested outer-fold AUC by policy", barmode="group")
    charts.append(figure)
    curves = []
    for row in result["outer_folds"]:
        for point in row["inner_k_curve"]:
            curves.append({"fold": row["outer_fold"], **point})
    curve_frame = pd.DataFrame(curves)
    figure = go.Figure()
    for fold, part in curve_frame.groupby("fold"):
        figure.add_scatter(x=part.k, y=part.mean_auc, mode="lines+markers", name=f"Fold {fold}")
    figure.update_layout(title="02 | Inner CV count path; AUC one-SE + Brier guard")
    charts.append(figure)
    top = importance.head(20)
    figure = go.Figure(go.Bar(x=top.mean_abs_linear_shap, y=top.readout, orientation="h"))
    figure.update_layout(title="03 | Mean absolute interventional linear SHAP", height=650)
    charts.append(figure)
    figure = go.Figure(
        go.Bar(x=top.mean_permutation_macro_auc_drop, y=top.readout, orientation="h")
    )
    figure.update_layout(title="04 | Held-out permutation macro-AUC loss", height=650)
    charts.append(figure)
    figure = go.Figure(go.Bar(x=top.selection_frequency, y=top.readout, orientation="h"))
    figure.update_layout(title="05 | Outer-fold selection frequency", height=650)
    charts.append(figure)
    figure = go.Figure(go.Bar(x=rows.outer_fold, y=rows.selected_count))
    figure.update_layout(title="06 | Selected readout count by outer fold")
    charts.append(figure)
    figure = go.Figure()
    figure.add_bar(x=rows.outer_fold, y=rows.brier, name="Brier")
    figure.add_bar(x=rows.outer_fold, y=rows.probability_rmse, name="Probability RMSE")
    figure.add_bar(x=rows.outer_fold, y=rows.log_loss, name="Log loss")
    figure.update_layout(title="07 | Probability diagnostics; lower is better", barmode="group")
    charts.append(figure)
    values = [
        next(row for row in result["importance"] if row["readout"] == name)
        for name in result["shortlist"]
    ]
    figure = go.Figure(
        go.Table(
            header={"values": ["Shortlist", "Frequency", "SHAP", "Permutation Δmacro-AUC"]},
            cells={
                "values": [
                    [row["readout"] for row in values],
                    [row["selection_frequency"] for row in values],
                    [row["mean_abs_linear_shap"] for row in values],
                    [row["mean_permutation_macro_auc_drop"] for row in values],
                ]
            },
        )
    )
    figure.update_layout(title="08 | Stable shortlist for Stage 2 model selection")
    charts.append(figure)
    for figure in charts:
        figure.update_layout(template="plotly_white", margin=dict(l=100, r=40, t=80, b=80))
    return charts


def write_dashboard(root: Path, result):
    import plotly.io as pio

    parts = [
        "<!doctype html><meta charset='utf-8'><title>Jigsaw selection stage 1</title>",
        "<h1>Jigsaw — nested readout selection</h1>",
        "<p>Development selection only; not a Kaggle score or fresh holdout.</p>",
    ]
    for index, figure in enumerate(figures(result)):
        parts.append(pio.to_html(figure, full_html=False, include_plotlyjs=index == 0))
    path = root / PUBLIC / "dashboard.html"
    atomic_bytes(path, "\n".join(parts).encode())
    return path


def _run_with_alarm(root: Path):
    config = json.loads((root / CONFIG).read_text())
    budget = int(config["scientific_max_seconds"])
    if not 0 < budget <= 240 or not hasattr(signal, "setitimer"):
        raise ValueError("POSIX scientific budget <=240 seconds required")

    def expired(signum, frame):
        raise TimeoutError("Stage-1 scientific budget reached; completed outer folds preserved")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, budget)
    try:
        return run_selection(root)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.run:
        parser.error("select exactly one of --preflight or --run")
    root = args.root.resolve()
    if args.preflight:
        result = preflight_inputs(root)
        print("RESULT:", result["status"])
        print("ROWS:", result["rows"])
        print("READOUTS_AFTER_EXACT_DEDUP:", result["readouts_after_exact_dedup"])
        print("GLOBAL_BODY_GROUPS:", result["global_body_groups"])
        return
    result = _run_with_alarm(root)
    write_dashboard(root, result)
    print("RESULT:", result["status"])
    print("SHORTLIST:", ",".join(result["shortlist"]))
    print("MEDIAN_SELECTED_COUNT:", result["summary"]["median_selected_count"])
    print("OUTER_MACRO_AUC_MEAN:", round(result["summary"]["macro_auc_mean"], 6))
    print("OUTER_BRIER_MEAN:", round(result["summary"]["brier_mean"], 6))
    print("OUTER_PROBABILITY_RMSE:", round(result["summary"]["probability_rmse"], 6))


if __name__ == "__main__":
    main()
