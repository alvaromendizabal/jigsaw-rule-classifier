"Bounded original-data behavioral feature ablation; private checkpoints, public aggregates."

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import signal
import subprocess
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from filelock import FileLock
from scipy import sparse
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from jigsaw_rules.data import EXAMPLES, normalize, validate_frame
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest
from scripts.behavioral_features import (
    FAMILIES,
    basic_features,
    cross_fitted_support,
    screen_family,
)
from scripts.support_adaptation import build_study

BASE = "824e92bfae7414aa156f0bf3195ba677a1f55b21"
NOTEBOOK = "notebooks/06_behavioral_feature_investigation.ipynb"
CONFIG = "configs/behavioral_features.json"
SOURCE_PATHS = (
    "scripts/behavioral_features.py",
    "scripts/run_behavioral_features.py",
    CONFIG,
    "scripts/support_adaptation.py",
    "scripts/competition_features.py",
    "src/jigsaw_rules/data.py",
    "src/jigsaw_rules/runtime.py",
)
VARIANTS = {"lexical_control": ()}
VARIANTS.update({f"add_{f}": (f,) for f in FAMILIES})
VARIANTS["full"] = FAMILIES
VARIANTS.update({f"without_{f}": tuple(g for g in FAMILIES if g != f) for f in FAMILIES})
PUBLIC_DIR = "reports/behavioral_features"
PRIVATE_DIR = "runs/behavioral_features"


def environment():
    return {
        "python": platform.python_version(),
        "packages": {
            p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "plotly", "joblib")
        },
    }


def fingerprint(root: Path, frame_path: Path, config: dict):
    identity = {
        "train_sha256": digest(frame_path),
        "config": config,
        "source": {p: digest(root / p) for p in SOURCE_PATHS},
        "environment": environment(),
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    return key, identity


def check_time(started, budget):
    if time.monotonic() - started > budget:
        raise TimeoutError("Stage budget reached; completed candidate checkpoints are preserved")


def protocol(frame: pd.DataFrame, expected: list[int] | None = None):
    # Exact historical plan function. Its queries have no targets.
    plan = build_study(frame)
    counts = [len(f["queries"]) for f in plan["folds"]]
    if expected is not None and counts != expected:
        raise ValueError(f"Historical cohort differs: expected {expected}, observed {counts}")
    overview = []
    for fold in plan["folds"]:
        rule = fold["rule"]
        part = frame.loc[frame.rule == rule]
        local_known = {normalize(text) for c in EXAMPLES for text in part[c]}
        local = int((~part.body.map(normalize).isin(local_known)).sum())
        n = len(fold["queries"])
        overview.append(
            {
                "policy": rule,
                "readiness_same_rule_novel": local,
                "canonical_global_novel": n,
                "cross_rule_support_exclusions": local - n,
                "training_pairs": len(fold["training"]),
                "epoch_weight": sum(x["repeat"] for x in fold["training"]),
                "query_identity": hashlib.sha256(
                    json.dumps(fold["queries"], sort_keys=True).encode()
                ).hexdigest(),
                "query_text_in_training": 0,
            }
        )
    return plan, overview


def stage_write(folder: Path, files: dict[str, bytes], identity: dict):
    folder.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        atomic_bytes(folder / name, payload)
    atomic_json(
        folder / "complete.json",
        {"identity": identity, "files": {name: digest(folder / name) for name in files}},
    )
    stage_read(folder, identity)


def stage_read(folder: Path, identity: dict):
    marker = folder / "complete.json"
    if not marker.exists():
        return None
    record = json.loads(marker.read_text())
    if record["identity"] != identity or not record["files"]:
        raise ValueError("Candidate identity mismatch; refusing recomputation")
    for name, sha in record["files"].items():
        if Path(name).name != name or not (folder / name).is_file() or digest(folder / name) != sha:
            raise ValueError("Candidate checkpoint checksum mismatch; preserve and inspect")
    return record


def weighted_auc_samples(y: np.ndarray, score: np.ndarray, weights: np.ndarray):
    """Weighted tie-aware AUC for many paired bootstrap draws at once."""
    y, score = np.asarray(y), np.asarray(score)
    order = np.argsort(score, kind="stable")
    sy, ss, w = y[order], score[order], np.asarray(weights, float)[:, order]
    starts = np.r_[0, np.flatnonzero(ss[1:] != ss[:-1]) + 1]
    positive = np.add.reduceat(w * (sy == 1), starts, axis=1)
    negative = np.add.reduceat(w * (sy == 0), starts, axis=1)
    total = positive.sum(axis=1) * negative.sum(axis=1)
    numerator = (positive * (np.cumsum(negative, axis=1) - 0.5 * negative)).sum(axis=1)
    return np.divide(numerator, total, out=np.full(len(w), np.nan), where=total > 0)


def contrast_intervals(records, frame, replicates=500, seed=20260911):
    lookup = frame.set_index("row_id")
    # Canonical bodies define bootstrap units shared across every policy/candidate.
    all_ids = sorted({int(i) for record in records.values() for i in record["row_ids"]})
    bodies = lookup.loc[all_ids].body.map(normalize)
    codes, names = pd.factorize(bodies, sort=True)
    group_by_id = dict(zip(all_ids, codes, strict=True))
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(names), np.full(len(names), 1 / len(names)), size=replicates)
    aucs, point = {}, {}
    for name in VARIANTS:
        samples, points = [], []
        for fold in range(2):
            record = records[(fold, name)]
            idx = record["row_ids"].astype(int)
            y = lookup.loc[idx].rule_violation.to_numpy(dtype=int)
            weights = draws[:, [group_by_id[int(i)] for i in idx]]
            samples.append(weighted_auc_samples(y, record["probability"], weights))
            points.append(float(roc_auc_score(y, record["probability"])))
        aucs[name] = np.mean(samples, axis=0)
        point[name] = float(np.mean(points))
    comparisons = [
        (name, "lexical_control", name + " vs control")
        for name in VARIANTS
        if name != "lexical_control"
    ]
    comparisons += [("full", "without_" + f, "ablation: " + f) for f in FAMILIES]
    delta = np.array([point[a] - point[b] for a, b, _ in comparisons])
    boot = np.column_stack([aucs[a] - aucs[b] for a, b, _ in comparisons])
    valid = np.isfinite(boot).all(axis=1)
    if int(valid.sum()) < int(0.9 * replicates):
        raise ValueError("Too few valid paired group-bootstrap draws")
    band = float(np.quantile(np.max(np.abs(boot[valid] - delta), axis=1), 0.95))
    return [
        {
            "comparison": title,
            "delta_auc": float(d),
            "simultaneous_low": float(d - band),
            "simultaneous_high": float(d + band),
            "valid_draws": int(valid.sum()),
        }
        for (_, _, title), d in zip(comparisons, delta, strict=True)
    ]


def pooled_metrics(records, frame):
    "Same local cohort, pooled AUC before and after label-free per-policy ranks."
    lookup = frame.set_index("row_id")
    output = []
    for variant in VARIANTS:
        ids = np.concatenate([records[(f, variant)]["row_ids"] for f in range(2)])
        probability = np.concatenate([records[(f, variant)]["probability"] for f in range(2)])
        ranked = np.concatenate(
            [
                (rankdata(records[(f, variant)]["probability"], method="average") - 0.5)
                / len(records[(f, variant)]["probability"])
                for f in range(2)
            ]
        )
        target = lookup.loc[ids].rule_violation.to_numpy(dtype=int)
        output.append(
            {
                "variant": variant,
                "pooled_auc": float(roc_auc_score(target, probability)),
                "ranked_pooled_auc": float(roc_auc_score(target, ranked)),
                "rows": len(ids),
            }
        )
    return output


def metric_row(fold, variant, probability, labels, policy):
    return {
        "fold": fold,
        "policy": policy,
        "variant": variant,
        "auc": float(roc_auc_score(labels, probability)),
        "brier": float(brier_score_loss(labels, probability)),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "queries": len(labels),
    }


def run_study(
    root: Path,
    *,
    test_frame: pd.DataFrame | None = None,
    test_config: dict | None = None,
    max_new_fits: int | None = None,
):
    config = json.loads((root / CONFIG).read_text()) if test_config is None else test_config
    start = time.monotonic()
    raw = root / "data/raw/train.csv"
    if test_frame is None:
        if digest(raw) != config["train_sha256"]:
            raise ValueError("Original training SHA256 mismatch")
        frame = pd.read_csv(raw)
        expected = config["query_counts"]
    else:
        frame = test_frame.copy()
        expected = None
    validate_frame(frame, train=True)
    if frame.rule.nunique() != 2:
        raise ValueError("This registered experiment requires the two original policies")
    run_id, identity = fingerprint(root, raw, config)
    if test_frame is not None:
        identity["synthetic_frame_hash"] = hashlib.sha256(frame.to_json().encode()).hexdigest()
        run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    private = root / PRIVATE_DIR / run_id
    private.mkdir(parents=True, exist_ok=True)
    public = root / PUBLIC_DIR
    public.mkdir(parents=True, exist_ok=True)
    with (
        FileLock(str(root / PRIVATE_DIR / "study.lock"), timeout=1),
        threadpool_limits(limits=1),
        Progress(private / "events.jsonl", "behavioral_feature_round1") as log,
    ):
        finished = private / "finished.json"
        if finished.exists():
            receipt = json.loads(finished.read_text())
            if receipt["identity"] != identity:
                raise ValueError("Finished run identity differs")
            for name, sha in receipt["public_hashes"].items():
                if digest(public / name) != sha:
                    raise ValueError("Public result corruption; refusing silent repair")
            for fold in range(2):
                for name in VARIANTS:
                    stage_read(
                        private / f"fold_{fold}" / name,
                        {"run_id": run_id, "fold": fold, "variant": name},
                    )
            result = json.loads((public / "results.json").read_text())
            atomic_json(
                private / "last_invocation.json",
                {"new_fits": 0, "reused_fits": 20, "run_id": run_id},
            )
            log.emit("completed_run_reused", new_fits=0, reused_fits=20)
            return result
        plan, cohorts = protocol(frame, expected)
        atomic_json(private / "identity.json", identity)
        # Target-free query plan plus eligible training labels remain private.
        atomic_json(private / "plan.json", plan)
        for row in cohorts:
            log.emit(
                "cohort_verified",
                queries=row["canonical_global_novel"],
                training_pairs=row["training_pairs"],
            )
        metrics, selection, coeffs, records = [], [], [], {}
        new_fits, reused = 0, 0
        lookup = frame.set_index("row_id")
        for fold_idx, fold in enumerate(plan["folds"]):
            check_time(start, config["max_seconds"])
            train, query = pd.DataFrame(fold["training"]), pd.DataFrame(fold["queries"])
            y = train.rule_violation.to_numpy(dtype=int)
            dense_train, dense_query = (
                basic_features(train[["body", "rule"]]),
                basic_features(query[["body", "rule"]]),
            )
            support_train, support_query, inner_audit = cross_fitted_support(
                train, query, config["seed"]
            )
            dense_train = pd.concat([dense_train, support_train], axis=1)
            dense_query = pd.concat([dense_query, support_query], axis=1)
            atomic_json(private / f"fold_{fold_idx}" / "inner_crossfit.json", inner_audit)

            def text(df):
                return ("RULE: " + df.rule + "\nBODY: " + df.body).tolist()

            word = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=config["word_features"],
                sublinear_tf=True,
            )
            char = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=config["char_features"],
                sublinear_tf=True,
            )
            train_text, query_text = text(train), text(query)
            lx = sparse.hstack(
                [word.fit_transform(train_text), char.fit_transform(train_text)], format="csr"
            )
            lv = sparse.hstack(
                [word.transform(query_text), char.transform(query_text)], format="csr"
            )
            blocks, val_blocks, selected_names, scales = {}, {}, {}, {}
            for family in FAMILIES:
                columns = [c for c in dense_train if c.startswith(family + "/")]
                x, v = dense_train[columns].to_numpy(), dense_query[columns].to_numpy()
                indices, report = screen_family(x, y, columns, config["feature_budget_per_family"])
                selected_names[family] = report["names"]
                selection.append(
                    {"fold": fold_idx, "policy": fold["rule"], "family": family, **report}
                )
                if not len(indices):
                    blocks[family], val_blocks[family], scales[family] = (
                        np.empty((len(x), 0)),
                        np.empty((len(v), 0)),
                        None,
                    )
                else:
                    scaler = StandardScaler()
                    blocks[family] = scaler.fit_transform(x[:, indices])
                    val_blocks[family] = scaler.transform(v[:, indices])
                    scales[family] = {
                        "mean": scaler.mean_.tolist(),
                        "scale": scaler.scale_.tolist(),
                    }
            shared = private / f"fold_{fold_idx}" / "feature_manifest.json"
            atomic_json(
                shared,
                {
                    "names": selected_names,
                    "scalers": scales,
                    "word_vocabulary_size": len(word.vocabulary_),
                    "char_vocabulary_size": len(char.vocabulary_),
                    "inner_crossfit": inner_audit,
                },
            )
            # Shared fitted vocabulary/IDF saved once; never deserialized on replay.
            buf = io.BytesIO()
            joblib.dump({"word": word, "char": char}, buf, compress=3)
            atomic_bytes(private / f"fold_{fold_idx}" / "lexical_transform.joblib", buf.getvalue())
            for variant, families in VARIANTS.items():
                check_time(start, config["max_seconds"])
                folder = private / f"fold_{fold_idx}" / variant
                ident = {"run_id": run_id, "fold": fold_idx, "variant": variant}
                prior = stage_read(folder, ident)
                if prior is None:
                    if max_new_fits is not None and new_fits >= max_new_fits:
                        raise TimeoutError("Synthetic interruption for recovery testing")
                    x = sparse.hstack(
                        [lx] + [sparse.csr_matrix(blocks[f]) for f in families], format="csr"
                    )
                    xv = sparse.hstack(
                        [lv] + [sparse.csr_matrix(val_blocks[f]) for f in families], format="csr"
                    )
                    model = LogisticRegression(
                        C=config["classifier_c"],
                        solver="liblinear",
                        max_iter=1000,
                        random_state=config["seed"],
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", ConvergenceWarning)
                        model.fit(x, y, sample_weight=train.repeat.to_numpy(dtype=float))
                    probability = model.predict_proba(xv)[:, 1]
                    if not np.isfinite(probability).all():
                        raise ValueError("Nonfinite probabilities")
                    names = [name for f in families for name in selected_names[f]]
                    dense_coef = model.coef_[0, lx.shape[1] :]
                    coef_table = [
                        {
                            "fold": fold_idx,
                            "variant": variant,
                            "feature": n,
                            "coefficient": float(c),
                        }
                        for n, c in zip(names, dense_coef, strict=True)
                    ]
                    payload = io.BytesIO()
                    np.savez_compressed(
                        payload,
                        probability=probability,
                        row_ids=query.row_id.to_numpy(),
                        coefficients=model.coef_,
                        intercept=model.intercept_,
                    )
                    stage_write(
                        folder,
                        {
                            "predictions.npz": payload.getvalue(),
                            "coefficients.json": json.dumps(coef_table, sort_keys=True).encode(),
                        },
                        ident,
                    )
                    new_fits += 1
                else:
                    reused += 1
                with np.load(folder / "predictions.npz", allow_pickle=False) as a:
                    ids, probability = a["row_ids"], a["probability"]
                if not np.array_equal(ids, query.row_id.to_numpy()):
                    raise ValueError("Checkpoint query order differs")
                # Query labels are looked up only after the prediction exists.
                truth = lookup.loc[ids].rule_violation.to_numpy(dtype=int)
                metrics.append(metric_row(fold_idx, variant, probability, truth, fold["rule"]))
                records[(fold_idx, variant)] = {"row_ids": ids, "probability": probability}
                coeffs.extend(json.loads((folder / "coefficients.json").read_text()))
                log.emit(
                    "candidate_complete",
                    fold=fold_idx,
                    variant=variant,
                    completed=len(metrics),
                    total=20,
                    new_fits=new_fits,
                    reused_fits=reused,
                )
        pooled = pooled_metrics(records, frame)
        intervals = contrast_intervals(
            records, frame, config["bootstrap_replicates"], config["seed"]
        )
        primary = next(x for x in intervals if x["comparison"] == "full vs control")
        primary_folds = [
            next(r["auc"] for r in metrics if r["fold"] == fold and r["variant"] == "full")
            - next(
                r["auc"] for r in metrics if r["fold"] == fold and r["variant"] == "lexical_control"
            )
            for fold in range(2)
        ]
        pooled_delta = next(
            r["ranked_pooled_auc"] for r in pooled if r["variant"] == "full"
        ) - next(r["ranked_pooled_auc"] for r in pooled if r["variant"] == "lexical_control")
        passed = (
            pooled_delta >= 0
            and primary["delta_auc"] >= config["minimum_macro_delta"]
            and primary["simultaneous_low"] > 0
            and min(primary_folds) >= 0
        )
        stability = []
        for family in FAMILIES:
            sets = [set(x["names"]) for x in selection if x["family"] == family]
            stability.append(
                {
                    "family": family,
                    "selected_name_jaccard": len(sets[0] & sets[1])
                    / max(1, len(sets[0] | sets[1])),
                }
            )
        result = {
            "schema": 1,
            "run_id": run_id,
            "status": "FEATURE_ROUND_COMPLETE",
            "new_fits": new_fits,
            "reused_fits": reused,
            "fit_count": 20,
            "new_dense_candidates": 280,
            "primary_candidate": "full",
            "primary_comparison": primary,
            "primary_policy_deltas": primary_folds,
            "pooled_metrics": pooled,
            "primary_ranked_pooled_delta": float(pooled_delta),
            "decision": "ELIGIBLE_FOR_NEXT_VALIDATION_ONLY"
            if passed
            else "DO_NOT_PROMOTE_THIS_FEATURE_SET",
            "cohorts": cohorts,
            "metrics": metrics,
            "comparisons": intervals,
            "selection": selection,
            "stability": stability,
            "coefficients": coeffs,
            "identity": identity,
            "elapsed_seconds": round(time.monotonic() - start, 3),
            "new_neural_inference": 0,
            "gpu": False,
            "automatic_gpu_authorization": False,
            "kaggle_score": None,
            "limitations": [
                (
                    "The original-data development set has been inspected repeatedly; "
                    "it is not a fresh holdout."
                ),
                (
                    "Fixed lexical CPU readout tests feature value; it does not measur"
                    "e an improvement over the accepted Qwen model."
                ),
                (
                    "Heuristic speech-act, quotation and negation cues are fallible; t"
                    "heir firing never supplies a label."
                ),
                (
                    "Bootstrap intervals are conditional on these fitted predictions, "
                    "not independent policy or retraining uncertainty."
                ),
                (
                    "No external threads, timestamps, unseen labels or cross-rule nega"
                    "tive labels were manufactured."
                ),
                (
                    "A negative finding rejects this registered representation, not al"
                    "l feature engineering."
                ),
            ],
        }
        atomic_json(public / "results.json", result)
        atomic_json(
            public / "feature_catalog.json",
            {
                "candidate_names": list(dense_train.columns),
                "family_counts": {
                    f: sum(c.startswith(f + "/") for c in dense_train) for f in FAMILIES
                },
            },
        )
        atomic_json(
            private / "finished.json",
            {
                "identity": identity,
                "public_hashes": {
                    p: digest(public / p) for p in ("results.json", "feature_catalog.json")
                },
            },
        )
        atomic_json(
            private / "last_invocation.json",
            {"run_id": run_id, "new_fits": new_fits, "reused_fits": reused},
        )
        log.emit("results_saved", decision=result["decision"], macro_delta=primary["delta_auc"])
        return result


def bounded_compute(root: Path):
    "Notebook-facing synchronous call; process-tree timeout, no background work."
    args = [
        sys.executable,
        "-u",
        "-m",
        "scripts.run_behavioral_features",
        "--compute",
        "--root",
        str(root),
    ]
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    process = subprocess.Popen(args, cwd=root, env=env, start_new_session=True)
    try:
        code = process.wait(timeout=300)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    if code:
        raise RuntimeError("Feature worker stopped; preserve checkpoints and return the report")
    return json.loads((root / PUBLIC_DIR / "results.json").read_text())


def figures(result):
    import plotly.graph_objects as go

    charts = []
    metrics = pd.DataFrame(result["metrics"])
    fig = go.Figure()
    for policy, rows in metrics.groupby("policy", sort=True):
        fig.add_bar(
            x=rows.variant,
            y=rows.auc,
            name="Advertising" if "advert" in policy.lower() else "Legal advice",
        )
    fig.update_layout(
        title="01 | Local per-policy ROC AUC, not leaderboard performance",
        barmode="group",
        height=480,
        yaxis_title="ROC AUC — not a Kaggle score",
    )
    charts.append(fig)
    data = result["comparisons"]
    fig = go.Figure(
        go.Scatter(
            x=[r["delta_auc"] for r in data],
            y=[r["comparison"] for r in data],
            mode="markers",
            error_x={
                "array": [r["simultaneous_high"] - r["delta_auc"] for r in data],
                "arrayminus": [r["delta_auc"] - r["simultaneous_low"] for r in data],
            },
        )
    )
    fig.add_vline(x=0)
    fig.update_layout(
        title="02 | Matched contrasts: conditional simultaneous 95% intervals",
        height=600,
        xaxis_title="Change in policy-macro AUC",
    )
    charts.append(fig)
    ablations = [r for r in data if r["comparison"].startswith("ablation:")]
    fig = go.Figure(
        go.Bar(
            x=[r["comparison"].split(": ")[1] for r in ablations],
            y=[r["delta_auc"] for r in ablations],
        )
    )
    fig.update_layout(
        title="03 | What is lost when one family is removed?",
        yaxis_title="Full minus leave-family-out AUC",
        height=420,
    )
    charts.append(fig)
    selection = result["selection"]
    fig = go.Figure()
    for fold in range(2):
        rows = [r for r in selection if r["fold"] == fold]
        fig.add_bar(
            x=[r["family"] for r in rows], y=[r["retained"] for r in rows], name=f"Fold {fold}"
        )
    fig.update_layout(
        title="04 | Training-only feature screen retention",
        barmode="group",
        yaxis_title="Retained dense columns",
        height=420,
    )
    charts.append(fig)
    fig = go.Figure(
        go.Bar(
            x=[r["family"] for r in result["stability"]],
            y=[r["selected_name_jaccard"] for r in result["stability"]],
        )
    )
    fig.update_layout(
        title="05 | Feature-name stability across the two policy folds",
        yaxis_title="Jaccard overlap — not proof of utility",
        height=420,
        yaxis_range=[0, 1],
    )
    charts.append(fig)
    coefs = pd.DataFrame([r for r in result["coefficients"] if r["variant"] == "full"])
    pivot = coefs.pivot_table(index="feature", columns="fold", values="coefficient", fill_value=0)
    top = pivot.abs().max(axis=1).nlargest(20).index
    fig = go.Figure(
        go.Heatmap(
            z=pivot.loc[top].to_numpy(),
            x=[f"Fold {x}" for x in pivot.columns],
            y=top.tolist(),
            zmid=0,
        )
    )
    fig.update_layout(
        title="06 | Standardized dense coefficients: associations, not causation", height=670
    )
    charts.append(fig)
    fig = go.Figure()
    for field, title in (
        ("readiness_same_rule_novel", "Notebook 05: same-rule support"),
        ("canonical_global_novel", "Historical plan: global support"),
    ):
        fig.add_bar(
            x=[
                "Advertising" if "advert" in r["policy"].lower() else "Legal advice"
                for r in result["cohorts"]
            ],
            y=[r[field] for r in result["cohorts"]],
            name=title,
        )
    fig.update_layout(
        title="07 | Reconcile eligibility before comparing metrics",
        barmode="group",
        yaxis_title="Novel queries",
        height=420,
    )
    charts.append(fig)
    macro = metrics.groupby("variant", sort=False)[["brier", "log_loss"]].mean()
    fig = go.Figure()
    for field in macro:
        fig.add_bar(x=macro.index, y=macro[field], name=field)
    fig.update_layout(
        title="08 | Secondary probability diagnostics — lower is better",
        barmode="group",
        height=450,
    )
    charts.append(fig)
    for fig in charts:
        fig.update_layout(
            template="plotly_white",
            font={"family": "Arial", "size": 13},
            margin={"l": 120, "r": 30, "t": 70, "b": 100},
        )
        fig.update_yaxes(automargin=True)
    charts[1].update_layout(margin={"l": 250, "r": 40, "t": 70, "b": 70})
    charts[5].update_layout(margin={"l": 430, "r": 40, "t": 70, "b": 70})
    return charts


def write_dashboard(root: Path, result):
    import html

    charts = figures(result)
    parts = [
        (
            "<!doctype html><html><meta charset='utf-8'><title>Jigsaw behavior"
            "al feature investigation</title><body>"
        ),
        "<h1>Behavioral feature investigation · Round 1</h1>",
        (
            "<p>Exploratory original-data ablations. No new Kaggle score, neur"
            "al inference, or model promotion.</p>"
        ),
        "<h2>" + html.escape(result["decision"]) + "</h2>",
    ]
    for i, fig in enumerate(charts):
        parts.append(fig.to_html(full_html=False, include_plotlyjs=True if i == 0 else False))
    parts.append(
        "<h2>Limitations</h2>"
        + "".join("<p>" + html.escape(x) + "</p>" for x in result["limitations"])
        + "</body></html>"
    )
    target = root / PUBLIC_DIR / "dashboard.html"
    atomic_bytes(target, "\n".join(parts).encode())
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--compute", action="store_true")
    parser.add_argument("--execute-notebook", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.export_only:
        export_return(root)
        return
    if args.execute_notebook:
        first = execute_notebook(root)
        second = execute_notebook(root)
        run_study(root)  # Completed-stage reader: verifies all payloads without refitting.
        atomic_json(
            root / PRIVATE_DIR / "replay.json",
            {
                "first": first["status"],
                "second": second["status"],
                "cells_reexecuted": second["cells_reexecuted"],
            },
        )
        export_return(root)
        return
    if not args.compute:
        parser.error("Use --compute, --execute-notebook, or --export-only")
    spec = json.loads((root / CONFIG).read_text())
    budget = int(spec["max_seconds"])
    if not 0 < budget <= 240 or not hasattr(signal, "setitimer"):
        raise ValueError("Requires POSIX and a runtime budget no greater than 240 seconds")

    def timeout_handler(signum, frame):
        raise TimeoutError("CPU experiment hard limit reached; checkpoints preserved")

    old = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, budget)
    try:
        result = run_study(root)
        print("RESULT: " + result["status"], flush=True)
        print("DECISION: " + result["decision"], flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def notebook_source_hash(path: Path):
    import nbformat

    notebook = nbformat.read(path, as_version=4)
    content = [(cell.cell_type, cell.source) for cell in notebook.cells]
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def execute_notebook(root: Path):
    import nbformat
    from nbclient import NotebookClient

    target = root / NOTEBOOK
    work = root / PRIVATE_DIR
    work.mkdir(parents=True, exist_ok=True)
    receipt_path = work / "notebook_execution.json"
    identity = {
        "source_hash": notebook_source_hash(target),
        "sources": {p: digest(root / p) for p in SOURCE_PATHS},
        "environment": environment(),
    }
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["identity"] != identity or receipt["notebook_sha256"] != digest(target):
            raise ValueError("Executed notebook identity/hash changed; preserve edits and inspect")
        if receipt["results_sha256"] != digest(root / PUBLIC_DIR / "results.json"):
            raise ValueError("Executed notebook result hash changed")
        print("NOTEBOOK_REUSED: 0 cells reexecuted", flush=True)
        return {**receipt, "cells_reexecuted": 0, "status": "NOTEBOOK_REUSED"}
    notebook = nbformat.read(target, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    print("NOTEBOOK_EXECUTION_STARTED: eight code cells", flush=True)
    try:
        NotebookClient(
            notebook,
            timeout=330,
            kernel_name="jigsaw-rules",
            resources={"metadata": {"path": str(root)}},
        ).execute()
    except BaseException:
        nbformat.write(notebook, work / "interrupted_notebook.ipynb")
        raise
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    charts = sum(
        "application/vnd.plotly.v1+json" in o.get("data", {}) for c in code_cells for o in c.outputs
    )
    if any(c.execution_count is None for c in code_cells) or charts != 8:
        raise ValueError("Notebook did not produce all eight required Plotly charts")
    atomic_bytes(target, nbformat.writes(notebook).encode())
    receipt = {
        "status": "NOTEBOOK_EXECUTED",
        "identity": identity,
        "notebook_sha256": digest(target),
        "results_sha256": digest(root / PUBLIC_DIR / "results.json"),
        "code_cells": len(code_cells),
        "plotly_charts": charts,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def export_return(root: Path, destination: Path | None = None, failure: dict | None = None):
    "Allowlisted text/code/aggregate export. Never include data or private predictions."
    import zipfile

    destination = destination or Path.home() / "jigsaw_feature_round1_return.zip"
    names = [
        *SOURCE_PATHS[:3],
        "tests/test_behavioral_features.py",
        "docs/BEHAVIORAL_FEATURES.md",
        NOTEBOOK,
        PUBLIC_DIR + "/results.json",
        PUBLIC_DIR + "/feature_catalog.json",
        PRIVATE_DIR + "/notebook_execution.json",
        PRIVATE_DIR + "/tests.xml",
        PRIVATE_DIR + "/install.json",
        PRIVATE_DIR + "/launcher_report.json",
        PRIVATE_DIR + "/replay.json",
    ]
    names += [
        "scripts/notebook_readiness.py",
        "tests/test_notebook_readiness.py",
        "notebooks/05_data_readiness.ipynb",
        "reports/data_readiness/summary.json",
    ]
    payload = {name: (root / name).read_bytes() for name in names if (root / name).is_file()}
    # Only stable aggregate status; no environment variables, credentials, row outputs or logs
    # from arbitrary paths.
    payload["return_status.json"] = json.dumps(
        failure or {"status": "FEATURE_ROUND_COMPLETE", "github_updated": False}, sort_keys=True
    ).encode()
    payload["SHA256SUMS.json"] = json.dumps(
        {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()},
        sort_keys=True,
        indent=2,
    ).encode()
    with zipfile.ZipFile(
        destination.with_suffix(".zip.partial"), "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
    os.replace(destination.with_suffix(".zip.partial"), destination)
    print("RETURN_FILE:", destination, flush=True)
    return destination


if __name__ == "__main__":
    main()
