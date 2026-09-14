"Round 2: cached controls, training-only relational features, bounded ablations."

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock
from scipy import sparse
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from jigsaw_rules.data import normalize, validate_frame
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest
from scripts.behavioral_features import basic_features, screen_family
from scripts.relational_features import CATALOG, FAMILIES, relational_features
from scripts.run_behavioral_features import (
    SOURCE_PATHS as PRIOR_SOURCES,
)
from scripts.run_behavioral_features import (
    environment,
    metric_row,
    protocol,
    stage_read,
    stage_write,
    weighted_auc_samples,
)

CONFIG = "configs/relational_features.json"
NOTEBOOK = "notebooks/07_relational_feature_investigation.ipynb"
PUBLIC = "reports/relational_features"
PRIVATE = "runs/relational_features"
SOURCE_PATHS = (
    "scripts/relational_features.py",
    "scripts/run_relational_features.py",
    CONFIG,
    "tests/test_relational_features.py",
    *PRIOR_SOURCES,
)
CONTROLS = ("lexical_control", "add_behavior")
NEW = {f"add_{f}": (f,) for f in FAMILIES}
NEW["full_relations"] = FAMILIES
NEW.update({f"without_{f}": tuple(x for x in FAMILIES if x != f) for f in FAMILIES})
NEW["behavior_without_topic"] = ()
NEW["behavior_without_length"] = ()
VARIANTS = (*CONTROLS, *NEW)


def hashed_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def identity(root: Path, config: dict):
    return {
        "source": {p: digest(root / p) for p in SOURCE_PATHS},
        "config": config,
        "environment": environment(),
        "prior_results_sha256": digest(root / "reports/behavioral_features/results.json"),
        "train_sha256": digest(root / "data/raw/train.csv"),
    }


def verify_prior(root: Path, config: dict):
    "Validate old evidence and each reused prediction. Never rerun round 1."
    path = root / "reports/behavioral_features/results.json"
    if digest(path) != config["prior_results_sha256"]:
        raise ValueError("round-1 result hash differs; preserve existing work")
    prior = json.loads(path.read_text())
    if prior["run_id"] != config["prior_run_id"]:
        raise ValueError("round-1 run identity differs")
    if prior["identity"]["environment"] != environment():
        raise ValueError("environment changed since round 1; no silent comparison")
    for name, sha in prior["identity"]["source"].items():
        if digest(root / name) != sha:
            raise ValueError("round-1 source changed: " + name)
    old = root / "runs/behavioral_features" / prior["run_id"]
    complete = json.loads((old / "finished.json").read_text())
    if complete["identity"] != prior["identity"]:
        raise ValueError("round-1 completion identity differs")
    for name, sha in complete["public_hashes"].items():
        if digest(root / "reports/behavioral_features" / name) != sha:
            raise ValueError("round-1 public artifact changed")
    # All old fits are integrity-checked; only four controls are loaded.
    names = sorted({r["variant"] for r in prior["metrics"]})
    for fold in range(2):
        for name in names:
            found = stage_read(
                old / f"fold_{fold}" / name,
                {"run_id": prior["run_id"], "fold": fold, "variant": name},
            )
            if found is None:
                raise ValueError("completed round-1 checkpoint missing; do not refit it")
    return prior, old


def lexical_matrices(train: pd.DataFrame, query: pd.DataFrame, spec: dict):
    def text(df):
        return ("RULE: " + df.rule + "\nBODY: " + df.body).tolist()

    word = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=spec["word_features"], sublinear_tf=True
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=spec["char_features"],
        sublinear_tf=True,
    )
    x = sparse.hstack(
        [word.fit_transform(text(train)), char.fit_transform(text(train))], format="csr"
    )
    v = sparse.hstack([word.transform(text(query)), char.transform(text(query))], format="csr")
    return x, v


def transform_block(train: pd.DataFrame, query: pd.DataFrame, y, columns, budget):
    indices, report = screen_family(train[columns].to_numpy(), y, columns, budget)
    if not len(indices):
        return np.empty((len(train), 0)), np.empty((len(query), 0)), report
    scaler = StandardScaler()
    x = scaler.fit_transform(train[columns].to_numpy()[:, indices])
    v = scaler.transform(query[columns].to_numpy()[:, indices])
    report["scaler_mean"] = scaler.mean_.tolist()
    report["scaler_scale"] = scaler.scale_.tolist()
    return x, v, report


def load_reference(path: Path, ids, expected_design=None):
    with np.load(path, allow_pickle=False) as data:
        stored_ids = data["row_ids"]
        probability = data["probability"]
        if not np.array_equal(ids, stored_ids):
            raise ValueError("reference query order differs")
        if expected_design is not None:
            logits = expected_design @ data["coefficients"][0] + data["intercept"][0]
            actual = expit(np.asarray(logits).ravel())
            if not np.allclose(actual, probability, rtol=1e-10, atol=1e-10):
                raise ValueError("cached control design parity failed; no new fit permitted")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("invalid cached probabilities")
    return probability


def comparisons(records, frame, replicates, seed):
    ids = sorted({int(i) for r in records.values() for i in r["row_ids"]})
    lookup = frame.set_index("row_id")
    codes, labels = pd.factorize(lookup.loc[ids].body.map(normalize), sort=True)
    group_map = dict(zip(ids, codes, strict=True))
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(labels), np.full(len(labels), 1 / len(labels)), size=replicates)
    points, boot, pooled = {}, {}, []
    for name in VARIANTS:
        rows, draws, truths, probs, ranks = [], [], [], [], []
        for fold in range(2):
            rec = records[(fold, name)]
            y = lookup.loc[rec["row_ids"]].rule_violation.to_numpy(dtype=int)
            p = rec["probability"]
            w = weights[:, [group_map[int(i)] for i in rec["row_ids"]]]
            rows.append(float(roc_auc_score(y, p)))
            draws.append(weighted_auc_samples(y, p, w))
            truths.append(y)
            probs.append(p)
            ranks.append((rankdata(p, method="average") - 0.5) / len(p))
        points[name] = float(np.mean(rows))
        boot[name] = np.mean(draws, axis=0)
        pooled.append(
            {
                "variant": name,
                "macro_auc": points[name],
                "pooled_auc": float(roc_auc_score(np.concatenate(truths), np.concatenate(probs))),
                "ranked_pooled_auc": float(
                    roc_auc_score(np.concatenate(truths), np.concatenate(ranks))
                ),
            }
        )
    contrasts = [(name, "add_behavior", name + " vs behavior") for name in NEW]
    contrasts += [("full_relations", "without_" + f, "removal: " + f) for f in FAMILIES]
    contrasts += [("add_behavior", "lexical_control", "round-1 behavior vs lexical (reused)")]
    delta = np.array([points[a] - points[b] for a, b, _ in contrasts])
    draws = np.column_stack([boot[a] - boot[b] for a, b, _ in contrasts])
    valid = np.isfinite(draws).all(axis=1)
    if valid.sum() < 0.9 * replicates:
        raise ValueError("insufficient valid paired bootstrap draws")
    band = float(np.quantile(np.max(np.abs(draws[valid] - delta), axis=1), 0.95))
    rows = [
        {
            "comparison": title,
            "delta_auc": float(d),
            "simultaneous_low": float(d - band),
            "simultaneous_high": float(d + band),
            "valid_draws": int(valid.sum()),
        }
        for (_, _, title), d in zip(contrasts, delta, strict=True)
    ]
    return rows, pooled


def fit_candidate(x, y, v, weights, spec):
    "One unchanged LR specification. This function never receives query labels."
    model = LogisticRegression(
        C=spec["classifier_c"], solver="liblinear", max_iter=1000, random_state=spec["seed"]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y, sample_weight=weights)
    return model.predict_proba(v)[:, 1], model.coef_[0], model.intercept_


def run_study(root: Path, *, max_new_fits: int | None = None):
    started = time.monotonic()
    config = json.loads((root / CONFIG).read_text())
    if digest(root / "data/raw/train.csv") != config["train_sha256"]:
        raise ValueError("raw training hash mismatch")
    frame = pd.read_csv(root / "data/raw/train.csv")
    validate_frame(frame, train=True)
    prior, old = verify_prior(root, config)
    spec = prior["identity"]["config"]
    ident = identity(root, config)
    run_id = hashed_json(ident)[:20]
    work = root / PRIVATE / run_id
    public = root / PUBLIC
    work.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    with (
        FileLock(str(root / PRIVATE / "study.lock"), timeout=1),
        threadpool_limits(limits=1),
        Progress(work / "events.jsonl", "relational_round2", heartbeat_seconds=15) as log,
    ):
        finished = work / "finished.json"
        if finished.exists():
            receipt = json.loads(finished.read_text())
            if receipt["identity"] != ident:
                raise ValueError("round-2 completion identity differs")
            for name, sha in receipt["public_hashes"].items():
                if digest(public / name) != sha:
                    raise ValueError("completed result corruption; no silent repair")
            for fold in range(2):
                for name in NEW:
                    if (
                        stage_read(
                            work / f"fold_{fold}" / name,
                            {"run_id": run_id, "fold": fold, "variant": name},
                        )
                        is None
                    ):
                        raise ValueError("completed candidate checkpoint missing")
            atomic_json(
                work / "last_invocation.json",
                {"new_fits": 0, "reused_new_fits": 18, "reused_round1_controls": 4},
            )
            log.emit("completed_run_reused", new_fits=0, reused_new_fits=18)
            return json.loads((public / "results.json").read_text())
        plan, cohort = protocol(frame, config["query_counts"])
        if [c["query_identity"] for c in cohort] != [c["query_identity"] for c in prior["cohorts"]]:
            raise ValueError("query plan differs from round 1")
        atomic_json(work / "identity.json", ident)
        records, metrics, selections, parity = {}, [], [], []
        new_fits = reused = 0
        for fold_idx, fold in enumerate(plan["folds"]):
            if time.monotonic() - started > config["max_seconds"]:
                raise TimeoutError("round-2 budget exhausted")
            train, query = pd.DataFrame(fold["training"]), pd.DataFrame(fold["queries"])
            y = train.rule_violation.to_numpy(dtype=int)
            # Queries passed into feature construction contain no targets.
            basic_x = basic_features(train[["body", "rule"]])
            basic_v = basic_features(query[["body", "rule"]])
            relations_x = relational_features(train[["body"]])
            relations_v = relational_features(query[["body"]])
            lx, lv = lexical_matrices(train, query, spec)
            names = [c for c in basic_x if c.startswith("behavior/")]
            bx, bv, b_report = transform_block(
                basic_x, basic_v, y, names, spec["feature_budget_per_family"]
            )
            manifest = json.loads((old / f"fold_{fold_idx}/feature_manifest.json").read_text())
            if b_report["names"] != manifest["names"]["behavior"]:
                raise ValueError("behavior selection differs from verified round-1 selection")
            for name, design in (
                ("lexical_control", lv),
                ("add_behavior", sparse.hstack([lv, bv], format="csr")),
            ):
                p = load_reference(
                    old / f"fold_{fold_idx}" / name / "predictions.npz",
                    query.row_id.to_numpy(),
                    design,
                )
                records[(fold_idx, name)] = {"row_ids": query.row_id.to_numpy(), "probability": p}
                parity.append(
                    {"fold": fold_idx, "control": name, "design_parity": True, "new_fits": 0}
                )
            blocks = {}
            for family in FAMILIES:
                block = transform_block(
                    relations_x,
                    relations_v,
                    y,
                    CATALOG[family],
                    config["feature_budget_per_family"],
                )
                blocks[family] = block
                selections.append({"fold": fold_idx, "family": family, **block[2]})
            ablations = {
                "behavior_without_topic": [
                    n for n in names if not n.startswith("behavior/legal_topic_")
                ],
                "behavior_without_length": [
                    n for n in names if n not in ("behavior/log_words", "behavior/log_chars")
                ],
            }
            reduced = {
                key: transform_block(basic_x, basic_v, y, cols, spec["feature_budget_per_family"])
                for key, cols in ablations.items()
            }
            for key, value in reduced.items():
                selections.append({"fold": fold_idx, "family": key, **value[2]})
            for name, families in NEW.items():
                if time.monotonic() - started > config["max_seconds"]:
                    raise TimeoutError("round-2 budget exhausted; completed fits preserved")
                folder = work / f"fold_{fold_idx}" / name
                stage_id = {"run_id": run_id, "fold": fold_idx, "variant": name}
                receipt = stage_read(folder, stage_id)
                if receipt is None:
                    if max_new_fits is not None and new_fits >= max_new_fits:
                        raise TimeoutError("authored interruption for recovery test")
                    ax, av = reduced[name][:2] if name in reduced else (bx, bv)
                    x = sparse.hstack([lx, ax] + [blocks[f][0] for f in families], format="csr")
                    v = sparse.hstack([lv, av] + [blocks[f][1] for f in families], format="csr")
                    p, coef, intercept = fit_candidate(x, y, v, train.repeat.to_numpy(float), spec)
                    if not np.isfinite(p).all():
                        raise ValueError("nonfinite fitted probability")
                    out = io.BytesIO()
                    np.savez_compressed(
                        out,
                        row_ids=query.row_id.to_numpy(),
                        probability=p,
                        coefficients=coef,
                        intercept=intercept,
                    )
                    stage_write(folder, {"predictions.npz": out.getvalue()}, stage_id)
                    new_fits += 1
                else:
                    reused += 1
                records[(fold_idx, name)] = {
                    "row_ids": query.row_id.to_numpy(),
                    "probability": load_reference(
                        folder / "predictions.npz", query.row_id.to_numpy()
                    ),
                }
                log.emit(
                    "candidate_complete",
                    completed=new_fits + reused,
                    total=18,
                    fold=fold_idx,
                    variant=name,
                    new_fits=new_fits,
                    reused_fits=reused,
                )
        lookup = frame.set_index("row_id")
        for (fold_idx, name), rec in records.items():
            labels = lookup.loc[rec["row_ids"]].rule_violation.to_numpy(dtype=int)
            metrics.append(
                metric_row(
                    fold_idx, name, rec["probability"], labels, plan["folds"][fold_idx]["rule"]
                )
            )
        contrasts, pooled = comparisons(
            records, frame, config["bootstrap_replicates"], config["bootstrap_seed"]
        )
        primary_name = "add_act_roles"
        primary = next(c for c in contrasts if c["comparison"] == primary_name + " vs behavior")
        policy_deltas = []
        for fold in range(2):
            scores = {r["variant"]: r["auc"] for r in metrics if r["fold"] == fold}
            policy_deltas.append(scores[primary_name] - scores["add_behavior"])
        pmap = {r["variant"]: r["ranked_pooled_auc"] for r in pooled}
        rank_delta = pmap[primary_name] - pmap["add_behavior"]
        passed = (
            primary["delta_auc"] >= config["minimum_macro_delta"]
            and primary["simultaneous_low"] > 0
            and min(policy_deltas) >= 0
            and rank_delta >= 0
        )
        stability = []
        for family in FAMILIES:
            s = [set(r["names"]) for r in selections if r["family"] == family]
            stability.append(
                {
                    "family": family,
                    "selected_name_jaccard": len(s[0] & s[1]) / max(1, len(s[0] | s[1])),
                }
            )
        result = {
            "schema": 1,
            "status": "RELATIONAL_ROUND_COMPLETE",
            "run_id": run_id,
            "identity": ident,
            "prior_run_id": prior["run_id"],
            "new_fits": new_fits,
            "reused_new_fits": reused,
            "reused_round1_controls": 4,
            "candidate_fits": 18,
            "new_dense_candidates": 72,
            "control_parity": parity,
            "cohorts": cohort,
            "metrics": metrics,
            "pooled_metrics": pooled,
            "comparisons": contrasts,
            "selection": selections,
            "stability": stability,
            "primary_comparison": primary,
            "primary_policy_deltas": policy_deltas,
            "primary_ranked_pooled_delta": rank_delta,
            "decision": "ELIGIBLE_FOR_NEXT_VALIDATION_ONLY" if passed else "DO_NOT_PROMOTE_PRIMARY",
            "new_neural_inference": 0,
            "gpu": False,
            "automatic_gpu_authorization": False,
            "kaggle_score": None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "limitations": [
                ("Exploratory follow-up chosen after inspecting round 1; not a fresh holdout."),
                ("Intervals cover 13 current contrasts, not all adaptive project comparisons."),
                (
                    "Heuristic clause/role/negation extraction can be wrong; it never"
                    " assigns labels."
                ),
                ("The CPU behavior anchor is not the accepted Qwen model or a Kaggle score."),
                ("No result authorizes GPU work or automatic selection of a different winner."),
            ],
        }
        atomic_json(public / "results.json", result)
        atomic_json(public / "feature_catalog.json", CATALOG)
        atomic_json(
            work / "finished.json",
            {
                "identity": ident,
                "public_hashes": {
                    n: digest(public / n) for n in ("results.json", "feature_catalog.json")
                },
            },
        )
        atomic_json(
            work / "last_invocation.json",
            {"new_fits": new_fits, "reused_new_fits": reused, "reused_round1_controls": 4},
        )
        log.emit("results_saved", decision=result["decision"], new_fits=new_fits)
        return result


def bounded_compute(root: Path):
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "scripts.run_relational_features",
            "--compute",
            "--root",
            str(root),
        ],
        cwd=root,
        start_new_session=True,
    )
    try:
        code = proc.wait(timeout=300)
    except BaseException:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        raise
    if code:
        raise RuntimeError("relational worker stopped; preserve checkpoints and return ZIP")
    return json.loads((root / PUBLIC / "results.json").read_text())


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
        title=("01 | Per-policy AUC: cached anchors and new relationships"), barmode="group"
    )
    charts.append(fig)
    rows = result["comparisons"]
    fig = go.Figure(
        go.Scatter(
            x=[r["delta_auc"] for r in rows],
            y=[r["comparison"] for r in rows],
            mode="markers",
            error_x={"array": [r["simultaneous_high"] - r["delta_auc"] for r in rows]},
        )
    )
    fig.add_vline(x=0)
    fig.update_layout(
        title=("02 | Paired group-bootstrap differences, simultaneous intervals"), height=650
    )
    charts.append(fig)
    rem = [r for r in rows if r["comparison"].startswith("removal:")]
    fig = go.Figure(go.Bar(x=[r["comparison"] for r in rem], y=[r["delta_auc"] for r in rem]))
    fig.update_layout(title="03 | Full minus leave-one-family-out: not feature importance")
    charts.append(fig)
    sel = pd.DataFrame(result["selection"])
    fig = go.Figure()
    for fold, rows in sel.groupby("fold"):
        fig.add_bar(x=rows.family, y=rows.retained, name=f"Fold {fold}")
    fig.update_layout(title="04 | Training-only retained features", barmode="group")
    charts.append(fig)
    fig = go.Figure(
        go.Bar(
            x=[r["family"] for r in result["stability"]],
            y=[r["selected_name_jaccard"] for r in result["stability"]],
        )
    )
    fig.update_layout(title="05 | Selected-name Jaccard; stability is not proof of value")
    charts.append(fig)
    fig = go.Figure()
    for policy, rows in metrics.groupby("policy", sort=True):
        subset = rows[
            rows.variant.isin(["add_behavior", "behavior_without_topic", "behavior_without_length"])
        ]
        fig.add_bar(
            x=subset.variant,
            y=subset.auc,
            name="Advertising" if "advert" in policy.lower() else "Legal advice",
        )
    fig.update_layout(
        title=("06 | Is the behavior gain mostly legal-topic or length cues?"), barmode="group"
    )
    charts.append(fig)
    fig = go.Figure()
    p = result["pooled_metrics"]
    fig.add_bar(x=[r["variant"] for r in p], y=[r["macro_auc"] for r in p], name="Policy-macro")
    fig.add_bar(
        x=[r["variant"] for r in p], y=[r["ranked_pooled_auc"] for r in p], name="Ranked pooled"
    )
    fig.update_layout(
        title=("07 | Macro versus ranked pooled AUC: local cohort only"), barmode="group"
    )
    charts.append(fig)
    fig = go.Figure()
    for field in ("brier", "log_loss"):
        avg = metrics.groupby("variant")[field].mean()
        fig.add_bar(x=avg.index, y=avg.values, name=field)
    fig.update_layout(title="08 | Probability diagnostics: lower is better", barmode="group")
    charts.append(fig)
    for fig in charts:
        fig.update_layout(
            template="plotly_white",
            height=fig.layout.height or 500,
            margin=dict(l=100, r=40, t=90, b=150),
            font=dict(size=13),
            xaxis=dict(automargin=True),
            yaxis=dict(automargin=True),
        )
    return charts


def write_dashboard(root: Path, result):
    import html

    import plotly.io as pio

    title = "Jigsaw | Round 2: action relationships"
    parts = [
        "<!doctype html><html><meta charset='utf-8'><title>" + title + "</title>",
        "<body><h1>" + title + ("</h1><p>Exploratory CPU feature study. Not a Kaggle score.</p>"),
        "<p>Decision: " + html.escape(result["decision"]) + "</p>",
    ]
    for i, fig in enumerate(figures(result)):
        parts.append(pio.to_html(fig, full_html=False, include_plotlyjs=(i == 0)))
    parts.append("</body></html>")
    path = root / PUBLIC / "dashboard.html"
    atomic_bytes(path, "\n".join(parts).encode())
    return path


def notebook_source_hash(path: Path):
    import nbformat

    n = nbformat.read(path, as_version=4)
    return hashed_json([(c.cell_type, c.source) for c in n.cells])


def execute_notebook(root: Path):
    import nbformat
    from nbclient import NotebookClient

    target = root / NOTEBOOK
    work = root / PRIVATE
    work.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / CONFIG).read_text())
    marker = work / "notebook_execution.json"
    ident = {
        "scientific_identity": identity(root, config),
        "source_hash": notebook_source_hash(target),
    }
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["identity"] != ident or digest(target) != saved["notebook_sha256"]:
            raise ValueError("notebook identity changed; saved work preserved")
        for name, sha in saved["artifact_hashes"].items():
            if digest(root / PUBLIC / name) != sha:
                raise ValueError("notebook output checksum mismatch")
        return {**saved, "status": "NOTEBOOK_REUSED", "cells_reexecuted": 0}
    n = nbformat.read(target, as_version=4)
    for cell in n.cells:
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    try:
        NotebookClient(
            n, timeout=330, kernel_name="jigsaw-rules", resources={"metadata": {"path": str(root)}}
        ).execute()
    except BaseException:
        nbformat.write(n, work / "interrupted_notebook.ipynb")
        raise
    cells = [c for c in n.cells if c.cell_type == "code"]
    count = sum(
        "application/vnd.plotly.v1+json" in o.get("data", {}) for c in cells for o in c.outputs
    )
    if any(c.execution_count is None for c in cells) or count != 8:
        raise ValueError("notebook incomplete or Plotly output count differs")
    atomic_bytes(target, nbformat.writes(n).encode())
    record = {
        "status": "NOTEBOOK_EXECUTED",
        "identity": ident,
        "notebook_sha256": digest(target),
        "code_cells": len(cells),
        "plotly_charts": count,
        "artifact_hashes": {
            name: digest(root / PUBLIC / name) for name in ("results.json", "dashboard.html")
        },
    }
    atomic_json(marker, record)
    return record


def export_return(root: Path, destination: Path | None = None):
    destination = destination or Path.home() / "jigsaw_feature_round2_return.zip"
    names = [
        "scripts/relational_features.py",
        "scripts/run_relational_features.py",
        CONFIG,
        "tests/test_relational_features.py",
        "docs/RELATIONAL_FEATURES.md",
        NOTEBOOK,
        PUBLIC + "/results.json",
        PUBLIC + "/feature_catalog.json",
        PRIVATE + "/notebook_execution.json",
        PRIVATE + "/replay.json",
        PRIVATE + "/tests.xml",
        PRIVATE + "/install.json",
        PRIVATE + "/launcher_report.json",
        "reports/behavioral_features/results.json",
    ]
    payload = {n: (root / n).read_bytes() for n in names if (root / n).is_file()}
    payload["SHA256SUMS.json"] = json.dumps(
        {n: hashlib.sha256(b).hexdigest() for n, b in payload.items()}, indent=2, sort_keys=True
    ).encode()
    temp = destination.with_suffix(".zip.partial")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in payload.items():
            z.writestr(name, data)
    os.replace(temp, destination)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--compute", action="store_true")
    parser.add_argument("--execute-notebook", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.export_only:
        print("RETURN_FILE:", export_return(root))
        return
    if args.execute_notebook:
        first = execute_notebook(root)
        second = execute_notebook(root)
        atomic_json(
            root / PRIVATE / "replay.json",
            {
                "first": first["status"],
                "second": second["status"],
                "cells_reexecuted": second["cells_reexecuted"],
            },
        )
        result = bounded_compute(root)
        print("RESULT:", result["status"])
        print("DECISION:", result["decision"])
        return
    if not args.compute:
        parser.error("select --compute, --execute-notebook or --export-only")
    budget = json.loads((root / CONFIG).read_text())["max_seconds"]
    if not isinstance(budget, int) or not 0 < budget <= 240 or not hasattr(signal, "setitimer"):
        raise ValueError("POSIX and positive runtime budget <=240 seconds required")

    def expired(signum, frame):
        raise TimeoutError("hard runtime limit; successful checkpoints preserved")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, budget)
    try:
        result = run_study(root)
        print("RESULT:", result["status"], "DECISION:", result["decision"])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    main()
